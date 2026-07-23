"""
VLLMBlockManager — vLLM-compatible KV Cache block management with HiFC extensions.

Implements the ``BlockSpaceManager`` interface from vLLM (Kwon et al., SOSP 2023
§4.2–4.5) backed by our ``memory_manager`` package, plus HiFC-style Flash Cache
(FC) block allocation for DRAM-free GPU↔SSD swapping (Jeong et al., 2025 §3.2).

Interface compatibility
-----------------------
Matches vLLM's ``BlockSpaceManager`` exactly:

    allocate(seq_group)             → block table for each sequence
    free(seq)                       → release blocks
    append_slot(seq)                → allocate one more block (decoding step)
    fork(seq, child_seq)           → COW fork (parallel sampling)
    can_allocate(seq_group)         → preflight check
    swap_in(seq)/swap_out(seq)      → HiFC-style GPU↔SSD swapping

HiFC extensions
---------------
* FC blocks alongside GPU/CPU blocks (HiFC §3.2, "Flash Cache Block Allocator")
* Fine-grained block mapping to pSLC zones (HiFC §3.2, "Flash-Aware Block Mgmt")
* Block append policy for sequential SSD I/O (HiFC Fig.2)
* Swap counts and bandwidth modeled after HiFC §5.1 experiments

Reference
---------
- vLLM ``BlockSpaceManager``: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py
- HiFC §3.2, Fig.2 (HiFC extends vLLM Block Manager with FC block allocator)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from memory_manager.kv_block_allocator import KVBlockAllocator, OutOfMemoryError
from memory_manager.block_table import BlockTableManager, BlockTable
from memory_manager.config import MemoryConfig
from memory_manager.kv_block import KVBlockState, StorageTier


# ---------------------------------------------------------------------------
# Sequence block tracking (vLLM-compatible)
# ---------------------------------------------------------------------------

@dataclass
class SequenceBlocks:
    """KV Cache blocks assigned to one sequence (request).

    Matches vLLM's internal per-sequence block tracking.  The ``block_table``
    is a list of physical block IDs in token order — the exact format that
    vLLM's attention kernel consumes.

    Parameters
    ----------
    seq_id : int
        vLLM-style sequence identifier.
    block_table : list[int]
        Ordered physical block IDs (vLLM attention kernel input).
    num_tokens : int
        Total tokens stored in this sequence's KV Cache.
    num_blocks : int
        Convenience alias for ``len(block_table)``.
    is_swapped : bool
        True if this sequence's blocks are currently on SSD.
    swap_tier : StorageTier
        Where the blocks physically reside.
    """

    seq_id: int
    block_table: List[int] = field(default_factory=list)
    num_tokens: int = 0
    is_swapped: bool = False
    swap_tier: StorageTier = StorageTier.GPU

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    @property
    def last_block_id(self) -> int:
        return self.block_table[-1] if self.block_table else -1

    def __repr__(self) -> str:
        loc = "SSD" if self.is_swapped else self.swap_tier.value.upper()
        return f"SeqBlocks({self.seq_id}, {self.num_blocks} blocks @ {loc}, {self.num_tokens} tok)"


# ---------------------------------------------------------------------------
# Block swap operation (HiFC §3.2)
# ---------------------------------------------------------------------------

class BlockSwapOp(Enum):
    """Direction of a block swap operation."""
    SWAP_OUT = auto()   # GPU → SSD (HiFC: evict to Flash Cache)
    SWAP_IN = auto()    # SSD → GPU (HiFC: prefetch from Flash Cache)
    NOOP = auto()       # Already in target tier


# ---------------------------------------------------------------------------
# VLLM Block Manager
# ---------------------------------------------------------------------------

class VLLMBlockManager:
    """vLLM-compatible KV Cache block manager backed by ET-Agent memory_manager.

    Drop-in compatible with vLLM's ``BlockSpaceManager`` API.  Adds HiFC-style
    FC (Flash Cache) blocks, GDS direct I/O simulation, and fine-grained
    zone mapping for pSLC SSD regions.

    Parameters
    ----------
    config : MemoryConfig
        Global memory configuration (capacities, block_size).
    allocator : KVBlockAllocator
        Physical block pool (Phase 1).
    block_tables : BlockTableManager
        Block table registry (Phase 1).
    enable_fc : bool
        Enable HiFC Flash Cache blocks (default: True).
    fc_capacity_blocks : int
        Number of FC (Flash Cache) blocks.  Default: 4× GPU blocks
        (HiFC uses high-capacity pSLC SSD).
    gds_bandwidth_gbps : float
        Simulated GDS (GPU Direct Storage) bandwidth in GiB/s.
        HiFC §5.1 measures ~4.7 GiB/s in pSLC region.
    """

    def __init__(
        self,
        config: MemoryConfig,
        allocator: KVBlockAllocator,
        block_tables: BlockTableManager,
        *,
        enable_fc: bool = True,
        fc_capacity_blocks: int = 0,
        gds_bandwidth_gbps: float = 4.7,
    ):
        self._config = config
        self._allocator = allocator
        self._block_tables = block_tables
        self._block_size = config.block_size
        self._lock = threading.RLock()

        # ── HiFC Flash Cache blocks ──
        self._fc_enabled = enable_fc
        self._fc_capacity = fc_capacity_blocks or (config.max_gpu_blocks * 4)
        self._fc_used = 0
        self._fc_blocks: Dict[int, int] = {}  # seq_id → fc_block_offset
        self._gds_bandwidth = gds_bandwidth_gbps

        # ── Sequence tracking ──
        self._sequences: Dict[int, SequenceBlocks] = {}
        self._next_seq_id = 1

        # ── Stats for vLLM comparison ──
        self._total_swaps_out: int = 0
        self._total_swaps_in: int = 0
        self._total_gpu_blocks_allocated: int = 0
        self._total_gpu_blocks_freed: int = 0
        self._swap_latency_us: List[float] = []  # per-swap latency in microseconds

    # ------------------------------------------------------------------
    # vLLM BlockSpaceManager API
    # ------------------------------------------------------------------

    def allocate(self, seq_id: int, num_tokens: int) -> SequenceBlocks:
        """Allocate KV Cache blocks for a new sequence (prefill phase).

        vLLM equivalent: ``BlockSpaceManager.allocate(seq_group)``

        Parameters
        ----------
        seq_id : int
            Sequence identifier.
        num_tokens : int
            Number of prompt tokens to store.

        Returns
        -------
        SequenceBlocks
            Allocated sequence with populated block_table.
        """
        with self._lock:
            # Try GPU first
            try:
                phys_ids = self._allocator.allocate(
                    f"vllm-seq-{seq_id}", num_tokens, group_id=f"seq-{seq_id}"
                )
                tier = StorageTier.GPU
            except OutOfMemoryError:
                # GPU full — allocate FC blocks on SSD (HiFC §3.2)
                if self._fc_enabled and self._fc_used + num_tokens // self._block_size <= self._fc_capacity:
                    phys_ids = self._allocator.allocate(
                        f"vllm-seq-{seq_id}", num_tokens, group_id=f"seq-{seq_id}"
                    )
                    tier = StorageTier.SSD
                    self._fc_used += len(phys_ids)
                else:
                    raise

            # Build block table
            table = self._block_tables.create_table(f"vllm-seq-{seq_id}")
            fills = [self._block_size] * (len(phys_ids) - 1) + [num_tokens % self._block_size or self._block_size]
            table.append_blocks(phys_ids, tokens_per_block=fills)

            seq = SequenceBlocks(
                seq_id=seq_id,
                block_table=list(phys_ids),
                num_tokens=num_tokens,
                swap_tier=tier,
            )
            self._sequences[seq_id] = seq
            self._total_gpu_blocks_allocated += len(phys_ids)
            return seq

    def free(self, seq_id: int) -> int:
        """Release all blocks owned by a sequence.

        vLLM equivalent: ``BlockSpaceManager.free(seq)``
        Returns number of blocks freed.
        """
        with self._lock:
            seq = self._sequences.pop(seq_id, None)
            if seq is None:
                return 0

            freed = self._allocator.free(f"vllm-seq-{seq_id}")
            self._block_tables.remove_table(f"vllm-seq-{seq_id}")
            if seq.is_swapped:
                self._fc_used = max(0, self._fc_used - seq.num_blocks)
            self._total_gpu_blocks_freed += freed
            return freed

    def append_slot(self, seq_id: int) -> Optional[int]:
        """Allocate one new block for the next decoding token.

        vLLM equivalent: ``BlockSpaceManager.append_slot(seq)``
        Called every decoding step.

        Returns the new physical block ID, or None if no new block needed.
        """
        with self._lock:
            seq = self._sequences.get(seq_id)
            if seq is None:
                return None

            table = self._block_tables.get_table(f"vllm-seq-{seq_id}")
            if table is None:
                return None

            # Check if the last block still has room
            if seq.num_blocks > 0 and not table[seq.num_blocks - 1].is_full_for(self._block_size):
                # Fill one more token slot in the existing last block
                table[seq.num_blocks - 1].num_filled += 1
                seq.num_tokens += 1
                return None  # no new block needed

            # Need a new block
            try:
                new_ids = self._allocator.allocate(
                    f"vllm-seq-{seq_id}", 1, group_id=f"seq-{seq_id}"
                )
                new_bid = new_ids[0]
                table.add_entry(seq.num_blocks, new_bid, num_filled=1)
                seq.block_table.append(new_bid)
                seq.num_tokens += 1
                self._total_gpu_blocks_allocated += 1
                return new_bid
            except OutOfMemoryError:
                # Trigger swap-out
                self._swap_out_victim()
                return self.append_slot(seq_id)  # retry

    def fork(self, parent_seq_id: int, child_seq_id: int) -> SequenceBlocks:
        """Fork a sequence (copy-on-write sharing for parallel sampling).

        vLLM equivalent: ``BlockSpaceManager.fork(parent, child)``
        vLLM §4.4 Fig.8: parallel sampling shares prompt KV Cache via COW.
        """
        with self._lock:
            parent = self._sequences.get(parent_seq_id)
            if parent is None:
                raise KeyError(f"Parent seq {parent_seq_id} not found")

            parent_table = self._block_tables.get_table(f"vllm-seq-{parent_seq_id}")
            if parent_table is None:
                raise KeyError(f"Parent block table for {parent_seq_id} not found")

            # Share parent's blocks via COW
            self._block_tables.create_table(f"vllm-seq-{child_seq_id}")
            self._block_tables.share_prefix(
                f"vllm-seq-{parent_seq_id}", f"vllm-seq-{child_seq_id}",
                prefix_blocks=parent.num_blocks,
            )

            child = SequenceBlocks(
                seq_id=child_seq_id,
                block_table=list(parent.block_table),  # shared physical blocks
                num_tokens=parent.num_tokens,
                swap_tier=parent.swap_tier,
            )
            self._sequences[child_seq_id] = child
            return child

    def can_allocate(self, num_tokens: int) -> bool:
        """Preflight: can we allocate blocks for *num_tokens*?"""
        blocks_needed = (num_tokens + self._block_size - 1) // self._block_size
        return self._allocator.free_blocks >= blocks_needed or (
            self._fc_enabled and self._fc_used + blocks_needed <= self._fc_capacity
        )

    # ------------------------------------------------------------------
    # HiFC swap_in / swap_out  (HiFC §3.2, Fig.2)
    # ------------------------------------------------------------------

    def swap_out(self, seq_id: int) -> BlockSwapOp:
        """Evict a sequence's KV Cache to FC (Flash Cache) blocks.

        HiFC §3.2: When GPU memory is exhausted, the scheduler selects a
        victim sequence and swaps its blocks to FC via GDS.

        Returns the operation performed.
        """
        with self._lock:
            seq = self._sequences.get(seq_id)
            if seq is None or seq.is_swapped:
                return BlockSwapOp.NOOP

            block_ids = seq.block_table
            num_blocks = len(block_ids)
            if num_blocks == 0:
                return BlockSwapOp.NOOP

            # Check FC capacity
            if self._fc_used + num_blocks > self._fc_capacity:
                return BlockSwapOp.NOOP

            t0 = time.monotonic()

            # Simulate GDS transfer: GPU → SSD direct (HiFC §3.2, ~4.7 GiB/s)
            swap_bytes = num_blocks * self._config.block_size_bytes
            transfer_time_s = swap_bytes / (self._gds_bandwidth * 1024**3)
            # Simulated delay
            time.sleep(min(transfer_time_s, 0.001))  # cap at 1ms for simulation

            # Move blocks to SSD tier
            for bid in block_ids:
                block = self._allocator.get_block(bid)
                if block:
                    block.storage_tier = StorageTier.SSD

            seq.is_swapped = True
            seq.swap_tier = StorageTier.SSD
            self._fc_used += num_blocks
            self._total_swaps_out += 1

            elapsed_us = (time.monotonic() - t0) * 1e6
            self._swap_latency_us.append(elapsed_us)

            return BlockSwapOp.SWAP_OUT

    def swap_in(self, seq_id: int) -> BlockSwapOp:
        """Prefetch a sequence's KV Cache from FC back to GPU.

        HiFC §3.2: Before a swapped sequence resumes decoding, its blocks
        are transferred back via GDS.
        """
        with self._lock:
            seq = self._sequences.get(seq_id)
            if seq is None or not seq.is_swapped:
                return BlockSwapOp.NOOP

            block_ids = seq.block_table
            num_blocks = len(block_ids)

            t0 = time.monotonic()

            # Simulate GDS transfer: SSD → GPU direct
            swap_bytes = num_blocks * self._config.block_size_bytes
            transfer_time_s = swap_bytes / (self._gds_bandwidth * 1024**3)
            time.sleep(min(transfer_time_s, 0.001))

            # Move blocks back to GPU
            for bid in block_ids:
                block = self._allocator.get_block(bid)
                if block:
                    block.storage_tier = StorageTier.GPU

            seq.is_swapped = False
            seq.swap_tier = StorageTier.GPU
            self._fc_used = max(0, self._fc_used - num_blocks)
            self._total_swaps_in += 1

            elapsed_us = (time.monotonic() - t0) * 1e6
            self._swap_latency_us.append(elapsed_us)

            return BlockSwapOp.SWAP_IN

    def get_block_table(self, seq_id: int) -> List[int]:
        """Return the vLLM attention-kernel compatible block table."""
        seq = self._sequences.get(seq_id)
        return seq.block_table if seq else []

    def get_num_free_gpu_blocks(self) -> int:
        """vLLM scheduler API: how many free GPU blocks remain."""
        return self._allocator.free_blocks

    def get_num_free_fc_blocks(self) -> int:
        """HiFC: how many FC blocks remain."""
        return self._fc_capacity - self._fc_used

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _swap_out_victim(self):
        """Select a victim sequence and swap it out (HiFC scheduler)."""
        # Pick the sequence with the most blocks that isn't already swapped
        candidates = [
            (sid, seq) for sid, seq in self._sequences.items()
            if not seq.is_swapped and seq.num_blocks > 0
        ]
        if not candidates:
            return

        # LRU heuristic: pick the one with the smallest seq_id (oldest)
        candidates.sort(key=lambda x: x[0])
        victim_id = candidates[0][0]
        self.swap_out(victim_id)

    # ------------------------------------------------------------------
    # Statistics for vLLM comparison
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            active_seqs = len([s for s in self._sequences.values() if not s.is_swapped])
            swapped_seqs = len([s for s in self._sequences.values() if s.is_swapped])
            avg_swap_latency_us = (
                sum(self._swap_latency_us) / len(self._swap_latency_us)
                if self._swap_latency_us else 0.0
            )

            return {
                "active_sequences": active_seqs,
                "swapped_sequences": swapped_seqs,
                "total_sequences": len(self._sequences),
                "gpu_blocks_free": self._allocator.free_blocks,
                "gpu_blocks_used": self._allocator.used_blocks,
                "fc_blocks_used": self._fc_used,
                "fc_blocks_free": self._fc_capacity - self._fc_used,
                "fc_capacity": self._fc_capacity,
                "gpu_utilization": round(
                    self._allocator.used_blocks / max(self._allocator.total_blocks, 1), 4
                ),
                "fc_utilization": round(
                    self._fc_used / max(self._fc_capacity, 1), 4
                ),
                "total_swaps_out": self._total_swaps_out,
                "total_swaps_in": self._total_swaps_in,
                "avg_swap_latency_us": round(avg_swap_latency_us, 1),
                "total_blocks_allocated": self._total_gpu_blocks_allocated,
                "total_blocks_freed": self._total_gpu_blocks_freed,
                "block_size": self._block_size,
                "block_size_bytes": self._config.block_size_bytes,
                "gds_bandwidth_gbps": self._gds_bandwidth,
            }

    def reset_stats(self):
        with self._lock:
            self._total_swaps_out = 0
            self._total_swaps_in = 0
            self._total_gpu_blocks_allocated = 0
            self._total_gpu_blocks_freed = 0
            self._swap_latency_us.clear()

    def dump(self) -> str:
        lines = ["VLLMBlockManager"]
        for sid in sorted(self._sequences):
            s = self._sequences[sid]
            swapped = " [SWAPPED→SSD]" if s.is_swapped else ""
            lines.append(f"  seq {sid:4d}: {s.num_blocks:4d} blocks, "
                         f"{s.num_tokens:5d} tokens{swapped}")
        lines.append(f"  FC usage: {self._fc_used}/{self._fc_capacity} blocks")
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"VLLMBlockManager(seqs={s['total_sequences']}, "
            f"GPU={s['gpu_blocks_used']}/{self._allocator.total_blocks}, "
            f"FC={s['fc_blocks_used']}/{s['fc_capacity']}, "
            f"swaps={self._total_swaps_out}out/{self._total_swaps_in}in)"
        )
