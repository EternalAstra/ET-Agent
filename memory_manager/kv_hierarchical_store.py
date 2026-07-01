"""
Hierarchical KV Cache Store — GPU → CPU → SSD tiered storage.

Implements the three-tier storage hierarchy from MoonCake (FAST 2025)
and vLLM's swap-out/swap-in model (§4.5).  Block data flows between
tiers based on access frequency and lifecycle phase:

    GPU VRAM (hot, ~80 GB)
      ↕  demote / promote
    CPU DRAM (warm, ~512 GB)
      ↕  archive / restore
    NVMe SSD (cold, ~2 TB)

Migration can be overlapped with computation (MoonCake §5.2 layer-wise
prefill), but this Python implementation models the *scheduling* aspect.
Actual GPU↔CPU memcpy is a passthrough for now and will be hooked into
the model backend in Phase 5.

Key operations
--------------
``demote_blocks()`` — move blocks from GPU → CPU or CPU → SSD
``promote_blocks()`` — move blocks from CPU → GPU or SSD → CPU
``evict_blocks()`` — free blocks from any tier entirely
``evict_cold_blocks()`` — bulk-evict LRU blocks to free GPU capacity

References
----------
- MoonCake §3, Figure 3  (KVCache pool in CPU memory)
- MoonCake §5.2         (layer-wise prefill: overlap transfer + compute)
- vLLM §4.5             (preemption via swapping to CPU)
- vLLM §4.6             (distributed KV cache manager)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from memory_manager.kv_block import KVBlock, KVBlockState, StorageTier
from memory_manager.config import MemoryConfig
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.kv_eviction_policy import EvictionPolicy, make_policy


# ---------------------------------------------------------------------------
# Migration queue entry
# ---------------------------------------------------------------------------

@dataclass
class MigrationTask:
    """A pending block migration between storage tiers.

    In production, this would submit GPU→CPU memcpy or disk I/O to a
    worker thread.  For now it models the scheduling decision.
    """

    block_ids: List[int]
    source_tier: StorageTier
    target_tier: StorageTier
    request_id: str
    created_at: float
    priority: int = 0   # higher = more urgent

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


# ---------------------------------------------------------------------------
# Hierarchical store
# ---------------------------------------------------------------------------

class HierarchicalKVStore:
    """Three-tier KV Cache storage with capacity tracking and migration.

    Parameters
    ----------
    config : MemoryConfig
        Global memory configuration (capacities per tier).
    allocator : KVBlockAllocator
        The physical block pool (for GPU blocks).
    eviction_policy : EvictionPolicy | None
        Policy for selecting victims when a tier is full.
    """

    def __init__(
        self,
        config: MemoryConfig,
        allocator: KVBlockAllocator,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self._config = config
        self._allocator = allocator
        self._policy = eviction_policy or make_policy("tiered_lru")
        self._lock = threading.RLock()

        # ── Per-tier usage (bytes) ──
        self._usage_bytes: Dict[StorageTier, int] = {
            StorageTier.GPU: 0,
            StorageTier.CPU: 0,
            StorageTier.SSD: 0,
        }

        # ── Per-tier block location map ──
        # block_id → current tier
        self._block_locations: Dict[int, StorageTier] = {}

        # ── Pending migrations ──
        self._migration_queue: List[MigrationTask] = []

        # ── Stats ──
        self._total_migrations: int = 0
        self._total_bytes_migrated: int = 0
        self._total_prefetches: int = 0

    # ------------------------------------------------------------------
    # Block registration (called after allocator.allocate())
    # ------------------------------------------------------------------

    def register_blocks(self, block_ids: List[int],
                        tier: StorageTier = StorageTier.GPU):
        """Record initial tier location for newly-allocated blocks.

        Must be called after ``allocator.allocate()`` so the store knows
        these blocks exist on GPU and can track usage correctly.
        """
        bs = self._config.block_size_bytes
        with self._lock:
            for bid in block_ids:
                old_tier = self._block_locations.get(bid)
                if old_tier is not None:
                    self._usage_bytes[old_tier] = max(0, self._usage_bytes[old_tier] - bs)
                self._block_locations[bid] = tier
                self._usage_bytes[tier] += bs
                self._policy.record_insert(bid, tier)

    def unregister_blocks(self, block_ids: List[int]):
        """Remove blocks from tier tracking (called before allocator.free())."""
        bs = self._config.block_size_bytes
        with self._lock:
            for bid in block_ids:
                tier = self._block_locations.pop(bid, None)
                if tier is not None:
                    self._usage_bytes[tier] = max(0, self._usage_bytes[tier] - bs)
                    self._policy.record_remove(bid, tier)

    # ------------------------------------------------------------------
    # Block location
    # ------------------------------------------------------------------

    def get_location(self, block_id: int) -> StorageTier:
        """Return the current storage tier for *block_id*."""
        with self._lock:
            return self._block_locations.get(block_id, StorageTier.GPU)

    def get_locations(self, block_ids: List[int]) -> Dict[int, StorageTier]:
        """Return tier locations for multiple blocks."""
        with self._lock:
            return {
                bid: self._block_locations.get(bid, StorageTier.GPU)
                for bid in block_ids
            }

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def demote_blocks(
        self,
        block_ids: List[int],
        target_tier: StorageTier,
        request_id: str = "",
    ) -> int:
        """Move blocks to a lower storage tier.

        GPU → CPU or CPU → SSD (data flows downward).
        Returns the number of blocks successfully demoted.

        Blocks that are PINNED or currently being migrated are skipped.
        """
        assert target_tier in (StorageTier.CPU, StorageTier.SSD), (
            f"demote target must be CPU or SSD, got {target_tier}"
        )

        with self._lock:
            demoted = 0
            for bid in block_ids:
                block = self._allocator.get_block(bid)
                if block is None:
                    continue
                if block.state == KVBlockState.PINNED:
                    continue
                if block.state == KVBlockState.EVICTING:
                    continue

                old_tier = self._block_locations.get(bid, StorageTier.GPU)

                # Only demote downward
                if old_tier == StorageTier.SSD:
                    continue
                if old_tier == StorageTier.CPU and target_tier != StorageTier.SSD:
                    continue
                if old_tier == target_tier:
                    continue

                # Check capacity
                needed = self._config.block_size_bytes
                if self._usage_bytes[target_tier] + needed > self._capacity(target_tier):
                    self._evict_from_tier(target_tier, needed)

                # Execute migration (metadata update; actual data copy in Phase 5)
                block.storage_tier = target_tier
                block.state = KVBlockState.EVICTING  # transient
                self._block_locations[bid] = target_tier
                self._usage_bytes[old_tier] -= needed
                self._usage_bytes[target_tier] += needed
                block.state = (KVBlockState.ALLOCATED
                               if block.ref_count <= 1
                               else KVBlockState.SHARED)

                self._total_migrations += 1
                self._total_bytes_migrated += needed
                demoted += 1

                self._policy.record_remove(bid, old_tier)
                self._policy.record_insert(bid, target_tier)

            return demoted

    def promote_blocks(
        self,
        block_ids: List[int],
        target_tier: StorageTier = StorageTier.GPU,
        request_id: str = "",
    ) -> int:
        """Move blocks to a higher storage tier (CPU → GPU, SSD → CPU).

        This is the prefetch path: before a waiting request resumes,
        its blocks are promoted back to GPU so they're hot for the
        next prefill/decoding step.

        Returns the number of blocks successfully promoted.
        """
        with self._lock:
            promoted = 0
            for bid in block_ids:
                block = self._allocator.get_block(bid)
                if block is None:
                    continue

                old_tier = self._block_locations.get(bid, StorageTier.GPU)
                if old_tier == target_tier:
                    continue

                # Promote one step at a time (SSD→CPU or CPU→GPU)
                if target_tier == StorageTier.GPU and old_tier == StorageTier.SSD:
                    # Two-step: SSD → CPU first (will need another call for CPU→GPU)
                    step_target = StorageTier.CPU
                else:
                    step_target = target_tier

                needed = self._config.block_size_bytes
                if self._usage_bytes[step_target] + needed > self._capacity(step_target):
                    if step_target == StorageTier.GPU:
                        self._evict_from_tier(StorageTier.GPU, needed)
                    else:
                        # CPU/SSD can exceed quota (soft limit)
                        pass

                # Metadata update
                block.storage_tier = step_target
                self._block_locations[bid] = step_target
                self._usage_bytes[old_tier] -= needed
                self._usage_bytes[step_target] += needed

                if step_target == StorageTier.GPU:
                    self._total_prefetches += 1

                self._total_migrations += 1
                self._total_bytes_migrated += needed
                promoted += 1

                self._policy.record_access(bid, step_target)

            return promoted

    def evict_blocks(self, block_ids: List[int]) -> int:
        """Permanently remove blocks from all tiers and return them to the
        free pool via the allocator.
        """
        with self._lock:
            evicted = 0
            for bid in list(block_ids):
                block = self._allocator.get_block(bid)
                if block is None:
                    continue
                if block.state == KVBlockState.PINNED:
                    continue

                tier = self._block_locations.get(bid, StorageTier.GPU)
                needed = self._config.block_size_bytes
                self._usage_bytes[tier] = max(0, self._usage_bytes[tier] - needed)
                self._block_locations.pop(bid, None)
                self._policy.record_remove(bid, tier)

                # Free the physical block
                self._allocator.free_block("__eviction__", bid)
                evicted += 1

            return evicted

    # ------------------------------------------------------------------
    # Bulk eviction
    # ------------------------------------------------------------------

    def evict_from_gpu(self, needed_bytes: int) -> int:
        """Evict enough blocks from GPU to free *needed_bytes*."""
        return self._evict_from_tier(StorageTier.GPU, needed_bytes)

    def _evict_from_tier(self, tier: StorageTier, needed_bytes: int) -> int:
        """Evict blocks from *tier* to free *needed_bytes*.  Returns count."""
        freed = 0
        bytes_freed = 0
        block_bytes = self._config.block_size_bytes

        victims = self._policy.select_victims(
            100, tier=tier, block_getter=self._allocator.get_block
        )

        for bid in victims:
            if bytes_freed >= needed_bytes:
                break
            self.evict_blocks([bid])
            bytes_freed += block_bytes
            freed += 1

        return freed

    def evict_cold_blocks(self, max_gpu_ratio: float = 0.8):
        """Evict the coldest GPU blocks until usage ≤ *max_gpu_ratio*.

        Called from ``LifecycleAwareKVManager.scan_and_migrate()``.
        """
        target_bytes = int(self._capacity(StorageTier.GPU) * max_gpu_ratio)
        excess = self._usage_bytes[StorageTier.GPU] - target_bytes
        if excess > 0:
            self._evict_from_tier(StorageTier.GPU, excess)

    # ------------------------------------------------------------------
    # Prefetch (Phase 5 hook)
    # ------------------------------------------------------------------

    def prefetch_for_resume(self, request_id: str,
                            block_ids: List[int]) -> int:
        """Prefetch blocks back to GPU ahead of a tool-call return.

        Called when the lifecycle tracker detects an impending phase
        transition from TOOL_CALL → PREFILL (resume).
        """
        return self.promote_blocks(block_ids, StorageTier.GPU, request_id)

    # ------------------------------------------------------------------
    # Stats & queries
    # ------------------------------------------------------------------

    def usage(self, tier: StorageTier) -> int:
        """Bytes used in *tier*."""
        with self._lock:
            return self._usage_bytes[tier]

    def usage_ratio(self, tier: StorageTier) -> float:
        """Fraction of *tier* capacity used."""
        cap = self._capacity(tier)
        return self._usage_bytes[tier] / max(cap, 1)

    def block_count(self, tier: StorageTier) -> int:
        """Number of blocks currently in *tier*."""
        with self._lock:
            return sum(
                1 for t in self._block_locations.values() if t == tier
            )

    def pending_migrations(self) -> int:
        return len(self._migration_queue)

    def stats(self) -> dict:
        with self._lock:
            return {
                "gpu_usage_bytes": self._usage_bytes[StorageTier.GPU],
                "cpu_usage_bytes": self._usage_bytes[StorageTier.CPU],
                "ssd_usage_bytes": self._usage_bytes[StorageTier.SSD],
                "gpu_usage_ratio": round(self.usage_ratio(StorageTier.GPU), 4),
                "cpu_usage_ratio": round(self.usage_ratio(StorageTier.CPU), 4),
                "ssd_usage_ratio": round(self.usage_ratio(StorageTier.SSD), 4),
                "gpu_blocks": self.block_count(StorageTier.GPU),
                "cpu_blocks": self.block_count(StorageTier.CPU),
                "ssd_blocks": self.block_count(StorageTier.SSD),
                "total_migrations": self._total_migrations,
                "total_bytes_migrated": self._total_bytes_migrated,
                "total_prefetches": self._total_prefetches,
                "pending_migrations": len(self._migration_queue),
            }

    def reset_stats(self):
        with self._lock:
            self._total_migrations = 0
            self._total_bytes_migrated = 0
            self._total_prefetches = 0

    def dump(self) -> str:
        """Human-readable tier usage dump."""
        with self._lock:
            lines = ["HierarchicalKVStore"]
            for tier in StorageTier:
                cap = self._capacity(tier)
                used = self._usage_bytes[tier]
                pct = 100 * used / max(cap, 1)
                blocks = self.block_count(tier)
                lines.append(
                    f"  {tier.value.upper():4s}  {used:>12_d} / {cap:>12_d} bytes "
                    f"({pct:5.1f}%)  {blocks:>6d} blocks"
                )
            lines.append(f"  migrations={self._total_migrations}")
            lines.append(f"  prefetches={self._total_prefetches}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capacity(self, tier: StorageTier) -> int:
        if tier == StorageTier.GPU:
            return self._config.gpu_capacity_bytes
        elif tier == StorageTier.CPU:
            return self._config.cpu_capacity_bytes
        else:
            return self._config.ssd_capacity_bytes

    def __repr__(self) -> str:
        return (
            f"HierarchicalKVStore(gpu={self.usage_ratio(StorageTier.GPU):.1%}, "
            f"cpu={self.usage_ratio(StorageTier.CPU):.1%}, "
            f"ssd={self.usage_ratio(StorageTier.SSD):.1%})"
        )
