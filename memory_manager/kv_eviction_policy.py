"""
KV Cache eviction policies — LRU, LFU, and hybrid strategies.

Implements the cache replacement algorithms referenced in MoonCake §4.2
(Table 1: "LRUCache performs best under this dataset's patterns, likely
due to the temporal proximity in request utilization") and the vLLM preemption
model (§4.5).

Policy catalog
--------------
``LRUEvictionPolicy``   — least-recently-used (default; best for most workloads)
``LFUEvictionPolicy``   — least-frequently-used (best when hot blocks are stable)
``TieredLRUPolicy``     — LRU per storage tier (GPU/CPU/SSD), used by Phase 3
``AgentAwarePolicy``    — LRU that protects pinned blocks and agent-active blocks

References
----------
- MoonCake §4.2, Table 1  (LRU vs LFU vs LengthAware cache hit rates)
- MoonCake §6.2            (KVCache hot-spot migration)
- vLLM §4.5                (preemption via swapping)
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Set

from memory_manager.kv_block import KVBlockState, StorageTier


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EvictionPolicy(ABC):
    """Abstract eviction policy for KV Cache blocks.

    Subclasses implement ``select_victims(n, tier)`` which returns the
    block IDs that should be evicted to make room for *n* new blocks.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._lock = threading.RLock()

    @abstractmethod
    def select_victims(self, n: int,
                       tier: StorageTier = StorageTier.GPU,
                       block_getter=None) -> List[int]:
        """Return up to *n* victim block IDs to evict from *tier*.

        Parameters
        ----------
        n : int
            Number of blocks to evict.
        tier : StorageTier
            Which storage tier to select victims from.
        block_getter : callable(id) → KVBlock | None
            Function that returns a ``KVBlock`` for a given ID.
            Policies use this to read state/ref_count/pin status.

        Returns
        -------
        list[int]
            Ordered list of victim block IDs (most-evictable first).
        """
        ...

    @abstractmethod
    def record_access(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        """Notify the policy that *block_id* was accessed."""
        ...

    @abstractmethod
    def record_insert(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        """Notify the policy that *block_id* was inserted into *tier*."""
        ...

    @abstractmethod
    def record_remove(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        """Notify the policy that *block_id* was removed from *tier*."""
        ...

    @abstractmethod
    def clear(self):
        """Reset all internal state."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of tracked blocks."""
        ...


# ---------------------------------------------------------------------------
# LRU — least-recently-used (MoonCake default)
# ---------------------------------------------------------------------------

class LRUEvictionPolicy(EvictionPolicy):
    """Least-Recently-Used eviction via ``OrderedDict``.

    ``record_access`` moves the block to the end (most-recent).
    ``select_victims`` iterates from the beginning (least-recent).

    This is the default policy in MoonCake §4.2 because real workloads
    exhibit strong temporal locality — blocks accessed recently are likely
    to be accessed again soon.
    """

    def __init__(self):
        super().__init__("LRU")
        self._order: OrderedDict = OrderedDict()  # block_id → timestamp

    def select_victims(self, n: int,
                       tier: StorageTier = StorageTier.GPU,
                       block_getter=None) -> List[int]:
        with self._lock:
            victims: List[int] = []
            # Iterate from least-recent (front of OrderedDict)
            candidates = list(self._order.keys())  # oldest (LRU) first

            for bid in candidates:
                if len(victims) >= n:
                    break

                # Skip blocks that shouldn't be evicted
                if block_getter is not None:
                    block = block_getter(bid)
                    if block is None:
                        continue
                    if block.state in (KVBlockState.PINNED,):
                        continue
                    if block.storage_tier != tier:
                        continue

                victims.append(bid)

            return victims

    def record_access(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._order.pop(block_id, None)
            self._order[block_id] = time.monotonic()

    def record_insert(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._order[block_id] = time.monotonic()

    def record_remove(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._order.pop(block_id, None)

    def clear(self):
        with self._lock:
            self._order.clear()

    @property
    def size(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return f"LRUEvictionPolicy({self.size} tracked)"


# ---------------------------------------------------------------------------
# LFU — least-frequently-used
# ---------------------------------------------------------------------------

class LFUEvictionPolicy(EvictionPolicy):
    """Least-Frequently-Used eviction with timestamp tiebreaker.

    MoonCake §4.2, Table 1 reports LFU underperforms LRU on their trace,
    but it can be superior when hot blocks are stable (e.g. system prompts
    accessed by every request).
    """

    def __init__(self):
        super().__init__("LFU")
        self._freq: Dict[int, int] = {}       # block_id → access count
        self._first_seen: Dict[int, float] = {}  # block_id → insertion time

    def select_victims(self, n: int,
                       tier: StorageTier = StorageTier.GPU,
                       block_getter=None) -> List[int]:
        with self._lock:
            # Sort by (frequency ASC, first_seen ASC) — least-used oldest first
            sorted_blocks = sorted(
                self._freq.keys(),
                key=lambda bid: (self._freq[bid], self._first_seen.get(bid, 0))
            )

            victims: List[int] = []
            for bid in sorted_blocks:
                if len(victims) >= n:
                    break
                if block_getter is not None:
                    block = block_getter(bid)
                    if block is None:
                        continue
                    if block.state in (KVBlockState.PINNED,):
                        continue
                    if block.storage_tier != tier:
                        continue
                victims.append(bid)
            return victims

    def record_access(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._freq[block_id] = self._freq.get(block_id, 0) + 1

    def record_insert(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._freq.setdefault(block_id, 0)
            self._first_seen.setdefault(block_id, time.monotonic())

    def record_remove(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._freq.pop(block_id, None)
            self._first_seen.pop(block_id, None)

    def clear(self):
        with self._lock:
            self._freq.clear()
            self._first_seen.clear()

    @property
    def size(self) -> int:
        return len(self._freq)

    def __repr__(self) -> str:
        return f"LFUEvictionPolicy({self.size} tracked)"


# ---------------------------------------------------------------------------
# Tiered LRU — per-tier eviction (Phase 3 hierarchical storage)
# ---------------------------------------------------------------------------

class TieredLRUPolicy(EvictionPolicy):
    """One LRU chain per storage tier.

    When the GPU tier is full, ``select_victims`` returns GPU blocks
    that should be demoted to CPU.  When CPU is full, blocks are demoted
    to SSD.  This is the policy backing Phase 3's ``HierarchicalKVStore``.
    """

    def __init__(self):
        super().__init__("TieredLRU")
        self._orders: Dict[StorageTier, OrderedDict] = {
            tier: OrderedDict() for tier in StorageTier
        }

    def select_victims(self, n: int,
                       tier: StorageTier = StorageTier.GPU,
                       block_getter=None) -> List[int]:
        order = self._orders[tier]
        with self._lock:
            victims: List[int] = []
            candidates = list(order.keys())  # oldest (LRU) first
            for bid in candidates:
                if len(victims) >= n:
                    break
                if block_getter is not None:
                    block = block_getter(bid)
                    if block is None:
                        continue
                    if block.state in (KVBlockState.PINNED,):
                        continue
                victims.append(bid)
            return victims

    def record_access(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._orders[tier].pop(block_id, None)
            self._orders[tier][block_id] = time.monotonic()

    def record_insert(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        self.record_access(block_id, tier)

    def record_remove(self, block_id: int, tier: StorageTier = StorageTier.GPU):
        with self._lock:
            self._orders[tier].pop(block_id, None)

    def clear(self):
        with self._lock:
            for order in self._orders.values():
                order.clear()

    @property
    def size(self) -> int:
        return sum(len(o) for o in self._orders.values())

    def size_for_tier(self, tier: StorageTier) -> int:
        return len(self._orders[tier])

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{t.value}={len(self._orders[t])}" for t in StorageTier
        )
        return f"TieredLRUPolicy({parts})"


# ---------------------------------------------------------------------------
# Agent-aware policy — protects agent-active blocks
# ---------------------------------------------------------------------------

class AgentAwarePolicy(LRUEvictionPolicy):
    """LRU policy that protects blocks belonging to active agent sessions.

    In addition to PINNED-block skipping (inherited from LRU), this policy
    never evicts blocks whose ``group_id`` matches a set of protected
    session IDs.  This prevents the evictor from stealing blocks from a
    session that is mid-turn (e.g. waiting for a tool result).

    Parameters
    ----------
    protected_groups : set[str] | None
        Session/request IDs whose blocks should never be evicted.
    """

    def __init__(self, protected_groups: Set[str] | None = None):
        super().__init__()
        self.name = "AgentAwareLRU"
        self._protected: Set[str] = set(protected_groups or ())

    def protect_group(self, group_id: str):
        """Add a session to the protected set."""
        self._protected.add(group_id)

    def unprotect_group(self, group_id: str):
        """Remove a session from the protected set."""
        self._protected.discard(group_id)

    def select_victims(self, n: int,
                       tier: StorageTier = StorageTier.GPU,
                       block_getter=None) -> List[int]:
        with self._lock:
            victims: List[int] = []
            candidates = list(self._order.keys())  # oldest (LRU) first

            for bid in candidates:
                if len(victims) >= n:
                    break

                if block_getter is not None:
                    block = block_getter(bid)
                    if block is None:
                        continue
                    if block.state in (KVBlockState.PINNED,):
                        continue
                    if block.storage_tier != tier:
                        continue
                    # Agent-aware: skip blocks belonging to protected sessions
                    if block.group_id and block.group_id in self._protected:
                        continue

                victims.append(bid)

            return victims

    @property
    def protected_count(self) -> int:
        return len(self._protected)

    def __repr__(self) -> str:
        return (
            f"AgentAwarePolicy({self.size} tracked, "
            f"{self.protected_count} protected groups)"
        )


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

_POLICY_REGISTRY: Dict[str, type] = {
    "lru": LRUEvictionPolicy,
    "lfu": LFUEvictionPolicy,
    "tiered_lru": TieredLRUPolicy,
    "agent_aware": AgentAwarePolicy,
}


def make_policy(kind: str = "lru", **kwargs) -> EvictionPolicy:
    """Factory: create an eviction policy by name.

    Parameters
    ----------
    kind : str
        One of ``"lru"``, ``"lfu"``, ``"tiered_lru"``, ``"agent_aware"``.
    **kwargs
        Passed to the policy constructor.
    """
    cls = _POLICY_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown eviction policy '{kind}'. "
            f"Available: {list(_POLICY_REGISTRY)}"
        )
    return cls(**kwargs)
