"""
KV Cache Block Allocator — physical block pool with reference counting.

Implements the fixed-size page-based memory allocation from vLLM's
PagedAttention (SOSP 2023, §4.2).  The allocator manages a pool of
``KVBlock`` instances, each representing one page of KV Cache memory
(default 16 tokens per block).

Allocation follows the OS virtual-memory model:

1. **allocate()** — reserve *N* free physical blocks, assign them to a
   request's block table, return their IDs.
2. **free()** — release all blocks owned by a request; physical blocks
   with ref_count=0 return to the free pool.
3. **free_block()** — release a single logical block (e.g. after COW clone).
4. **clone_block()** — copy-on-write: allocate a new block, memcpy data,
   decrement old block's ref_count.

Thread safety
-------------
All public methods acquire ``_lock`` (``threading.RLock``), making the
allocator safe for concurrent access from the agent's tool-execution
thread pool and the main conversation loop.

Reference
---------
- vLLM §4.2  (KV Cache Manager / Block Allocator)
- vLLM §4.3  (Decoding with PagedAttention, Figure 6)
- vLLM §4.4  (Copy-on-write for parallel sampling, Figure 8)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from memory_manager.kv_block import (
    KVBlock,
    KVBlockState,
    StorageTier,
)
from memory_manager.config import MemoryConfig


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OutOfMemoryError(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation request."""

    def __init__(self, requested: int, available: int, tier: StorageTier = StorageTier.GPU):
        self.requested = requested
        self.available = available
        self.tier = tier
        super().__init__(
            f"Out of memory: requested {requested} blocks, "
            f"only {available} free (tier={tier.value})"
        )


class BlockNotFoundError(KeyError):
    """Raised when a physical block ID is not found in the pool."""
    pass


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class KVBlockAllocator:
    """Fixed-size physical KV Cache block pool.

    Parameters
    ----------
    config : MemoryConfig
        Global memory configuration (block_size, capacities).
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._block_size = config.block_size
        self._lock = threading.RLock()

        # ── block pool ──
        max_blocks = config.max_gpu_blocks
        self._blocks: Dict[int, KVBlock] = {
            i: KVBlock(block_id=i) for i in range(max_blocks)
        }
        self._free_blocks: Set[int] = set(range(max_blocks))

        # ── monotonic clock ──
        self._clock = time.monotonic

        # ── allocation tracking (per-request) ──
        # request_id → set of physical_block_ids
        self._request_blocks: Dict[str, Set[int]] = {}

        # ── pinned system-prompt blocks ──
        # system_prompt_hash → set of physical_block_ids
        self._pinned_blocks: Dict[str, Set[int]] = {}

        # ── statistics ──
        self._total_allocations: int = 0
        self._total_frees: int = 0
        self._total_cow_clones: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, request_id: str, num_tokens: int,
                 group_id: str | None = None) -> List[int]:
        """Allocate enough blocks to hold *num_tokens* for *request_id*.

        Returns the list of physical block IDs assigned.  The caller is
        responsible for updating the request's block table.

        Raises ``OutOfMemoryError`` if there aren't enough free blocks.
        """
        blocks_needed = max(1, (num_tokens + self._block_size - 1) // self._block_size)

        with self._lock:
            if len(self._free_blocks) < blocks_needed:
                raise OutOfMemoryError(
                    requested=blocks_needed,
                    available=len(self._free_blocks),
                )

            allocated: List[int] = []
            ts = self._clock()

            for _ in range(blocks_needed):
                bid = self._free_blocks.pop()
                block = self._blocks[bid]
                block.mark_allocated(num_tokens=0, group_id=group_id)
                block.touch(ts)
                allocated.append(bid)

            # Track per-request ownership
            if request_id not in self._request_blocks:
                self._request_blocks[request_id] = set()
            self._request_blocks[request_id].update(allocated)

            self._total_allocations += blocks_needed
            return allocated

    def allocate_exact(self, request_id: str, num_blocks: int,
                       group_id: str | None = None) -> List[int]:
        """Allocate exactly *num_blocks* blocks (used for pre-allocated margins)."""
        with self._lock:
            if len(self._free_blocks) < num_blocks:
                raise OutOfMemoryError(
                    requested=num_blocks,
                    available=len(self._free_blocks),
                )

            allocated: List[int] = []
            ts = self._clock()

            for _ in range(num_blocks):
                bid = self._free_blocks.pop()
                block = self._blocks[bid]
                block.mark_allocated(group_id=group_id)
                block.touch(ts)
                allocated.append(bid)

            self._request_blocks.setdefault(request_id, set()).update(allocated)
            self._total_allocations += num_blocks
            return allocated

    def free(self, request_id: str) -> int:
        """Release all blocks owned by *request_id*.  Returns count freed."""
        with self._lock:
            if request_id not in self._request_blocks:
                return 0

            freed = 0
            for bid in list(self._request_blocks[request_id]):
                if self._release_block(bid):
                    freed += 1

            del self._request_blocks[request_id]
            self._total_frees += freed
            return freed

    def free_block(self, request_id: str, physical_id: int) -> bool:
        """Release a single physical block from *request_id*.

        Returns True if the block transitioned to FREE.
        """
        with self._lock:
            block = self._get_block(physical_id)
            if block.decrement_ref():
                self._add_to_free(physical_id)
                self._total_frees += 1
                # Remove from request tracking
                if request_id in self._request_blocks:
                    self._request_blocks[request_id].discard(physical_id)
                return True
            return False

    def clone_block(self, request_id: str,
                    old_physical_id: int) -> int:
        """Copy-on-write clone.

        Allocates a new physical block, copies the token count (and
        eventually tensor data — Phase 3), decrements the old block's
        ref_count, and returns the new physical block ID.

        This implements vLLM §4.3 / Figure 8: when a shared block needs
        to be written to by one of its owners, clone it first.

        Raises ``OutOfMemoryError`` if no free block is available.
        """
        with self._lock:
            old_block = self._get_block(old_physical_id)

            if old_block.ref_count <= 1:
                # No sharing — write in-place; no clone needed.
                return old_physical_id

            if not self._free_blocks:
                raise OutOfMemoryError(requested=1, available=0)

            # Allocate new block
            new_bid = self._free_blocks.pop()
            new_block = self._blocks[new_bid]
            ts = self._clock()

            # Clone metadata
            new_block.mark_allocated(
                num_tokens=old_block.num_tokens,
                group_id=old_block.group_id,
            )
            new_block.touch(ts)
            new_block.storage_tier = old_block.storage_tier

            # Release old block's ref
            old_block.decrement_ref()
            # Note: if old_block's ref_count hits 0 here, it won't be freed
            # because the parent request still owns it via its block table.
            # The caller is responsible for updating the block table.

            # Track
            self._request_blocks.setdefault(request_id, set()).add(new_bid)
            self._total_allocations += 1
            self._total_cow_clones += 1
            return new_bid

    def increment_ref(self, physical_id: int):
        """Add a reference to *physical_id* (used when sharing a block)."""
        with self._lock:
            self._get_block(physical_id).increment_ref()

    def touch_block(self, physical_id: int):
        """Update last-access timestamp of *physical_id*."""
        with self._lock:
            self._get_block(physical_id).touch(self._clock())

    # ------------------------------------------------------------------
    # Pinned / system-prompt blocks
    # ------------------------------------------------------------------

    def pin_blocks(self, group_key: str, block_ids: List[int]):
        """Pin a set of blocks so they are never evicted.

        Typically used for system-prompt prefix blocks.
        """
        with self._lock:
            for bid in block_ids:
                self._get_block(bid).state = KVBlockState.PINNED
            self._pinned_blocks[group_key] = set(block_ids)

    def unpin_blocks(self, group_key: str) -> Set[int]:
        """Unpin a previously pinned group; returns the block IDs."""
        with self._lock:
            block_ids = self._pinned_blocks.pop(group_key, set())
            for bid in block_ids:
                block = self._get_block(bid)
                # Return to appropriate state based on ref_count
                if block.ref_count > 1:
                    block.state = KVBlockState.SHARED
                elif block.ref_count == 1:
                    block.state = KVBlockState.ALLOCATED
            return block_ids

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_block(self, physical_id: int) -> KVBlock:
        """Read-only access to a block's metadata."""
        with self._lock:
            return self._get_block(physical_id)

    @property
    def total_blocks(self) -> int:
        return len(self._blocks)

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def shared_blocks(self) -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.is_shared)

    @property
    def pinned_blocks(self) -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.is_pinned)

    @property
    def usage_ratio(self) -> float:
        """Fraction of total blocks currently in use."""
        return self.used_blocks / max(self.total_blocks, 1)

    def get_request_blocks(self, request_id: str) -> Set[int]:
        """Return the set of physical block IDs owned by *request_id*."""
        with self._lock:
            return self._request_blocks.get(request_id, set()).copy()

    def get_free_block_ids(self) -> List[int]:
        """Return a snapshot of the free list (for debugging)."""
        with self._lock:
            return sorted(self._free_blocks)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return allocation statistics as a dict."""
        with self._lock:
            return {
                "total_blocks": self.total_blocks,
                "free_blocks": len(self._free_blocks),
                "used_blocks": self.total_blocks - len(self._free_blocks),
                "shared_blocks": self.shared_blocks,
                "pinned_blocks": self.pinned_blocks,
                "active_requests": len(self._request_blocks),
                "total_allocations": self._total_allocations,
                "total_frees": self._total_frees,
                "total_cow_clones": self._total_cow_clones,
                "usage_ratio": round(self.usage_ratio, 4),
                "block_size": self._block_size,
            }

    def reset_stats(self):
        """Zero out cumulative counters."""
        with self._lock:
            self._total_allocations = 0
            self._total_frees = 0
            self._total_cow_clones = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_block(self, physical_id: int) -> KVBlock:
        """Unchecked block lookup (caller MUST hold ``_lock``)."""
        if physical_id not in self._blocks:
            raise BlockNotFoundError(f"Block {physical_id} not found")
        return self._blocks[physical_id]

    def _add_to_free(self, physical_id: int):
        """Return a block to the free pool (caller MUST hold ``_lock``)."""
        self._blocks[physical_id].mark_free()
        self._free_blocks.add(physical_id)

    def _release_block(self, physical_id: int) -> bool:
        """Decrement ref; return True if block became FREE."""
        block = self._blocks[physical_id]
        if block.decrement_ref():
            self._add_to_free(physical_id)
            return True
        return False
