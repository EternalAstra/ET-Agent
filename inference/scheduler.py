"""
KVCacheScheduler — HiFC-style KV cache scheduling.

Models the scheduling decisions from HiFC Fig.2:
1. Prefill scheduling: allocate KV blocks for new sequences
2. Decoding scheduling: append single blocks per step
3. Swap-out scheduling: select victim sequences when GPU is full
4. Swap-in scheduling: prefetch swapped sequences before resuming

Produces ``ScheduleDecision`` objects compatible with vLLM's scheduler output.

Reference: HiFC Fig.2 (workflow), HiFC §3.2 (scheduling algorithm)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from inference.vllm_block_manager import VLLMBlockManager, BlockSwapOp
from inference.swapping_engine import HiFCSwappingEngine, FlashZone


@dataclass
class ScheduleDecision:
    """One scheduling cycle output (vLLM-compatible format)."""
    # Sequences to run this step
    scheduled_seqs: List[int] = field(default_factory=list)
    # Sequences to swap out (GPU→SSD)
    swap_out_seqs: List[int] = field(default_factory=list)
    # Sequences to swap in (SSD→GPU)
    swap_in_seqs: List[int] = field(default_factory=list)
    # New blocks allocated this cycle
    blocks_allocated: int = 0
    # Blocks freed this cycle
    blocks_freed: int = 0
    # Whether GPU memory was exhausted (triggered swap-out)
    gpu_exhausted: bool = False


class KVCacheScheduler:
    """HiFC-style KV cache scheduler.

    Parameters
    ----------
    block_manager : VLLMBlockManager
        vLLM-compatible block manager.
    swapping_engine : HiFCSwappingEngine
        GDS-accelerated swapping engine.
    max_batch_size : int
        Maximum sequences to schedule per step.
    watermark_gpu_pct : float
        When GPU utilization exceeds this, trigger swap-out (default 85%).
    watermark_swap_back_pct : float
        When GPU utilization drops below this, trigger swap-in (default 50%).
    """

    def __init__(
        self,
        block_manager: VLLMBlockManager,
        swapping_engine: HiFCSwappingEngine,
        *,
        max_batch_size: int = 64,
        watermark_gpu_pct: float = 0.85,
        watermark_swap_back_pct: float = 0.50,
    ):
        self._bm = block_manager
        self._se = swapping_engine
        self._max_batch = max_batch_size
        self._watermark_high = watermark_gpu_pct
        self._watermark_low = watermark_swap_back_pct
        self._lock = threading.RLock()

        # Pending sequences
        self._waiting_seqs: List[int] = []
        # Running sequences (in decode phase)
        self._running_seqs: List[int] = []
        # Swapped sequences (on SSD, waiting to resume)
        self._swapped_seqs: List[int] = []

        self._step_count: int = 0
        self._total_swaps_out: int = 0
        self._total_swaps_in: int = 0

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def add_sequence(self, seq_id: int, num_tokens: int) -> bool:
        """Add a new sequence with *num_tokens* prompt tokens.

        Returns True if allocation succeeded, False if GPU full (queued).
        """
        with self._lock:
            if self._bm.can_allocate(num_tokens):
                self._bm.allocate(seq_id, num_tokens)
                self._running_seqs.append(seq_id)
                return True
            else:
                # Queue for later
                self._waiting_seqs.append(seq_id)
                return False

    def finish_sequence(self, seq_id: int):
        """Mark a sequence as complete and free its blocks."""
        with self._lock:
            self._bm.free(seq_id)
            if seq_id in self._running_seqs:
                self._running_seqs.remove(seq_id)
            if seq_id in self._swapped_seqs:
                self._swapped_seqs.remove(seq_id)

    def append_token(self, seq_id: int) -> Optional[int]:
        """Decoding step: add one more block if needed."""
        return self._bm.append_slot(seq_id)

    # ------------------------------------------------------------------
    # Scheduling cycle (HiFC Fig.2)
    # ------------------------------------------------------------------

    def schedule(self) -> ScheduleDecision:
        """One scheduling cycle.

        HiFC Fig.2 workflow:
        1. Check GPU memory
        2. If full → select victim → swap_out
        3. If free → swap_in a waiting swapped sequence
        4. Schedule running sequences for this step
        """
        with self._lock:
            self._step_count += 1
            decision = ScheduleDecision()

            gpu_free = self._bm.get_num_free_gpu_blocks()
            gpu_total = self._bm._allocator.total_blocks
            gpu_used_ratio = 1.0 - (gpu_free / max(gpu_total, 1))

            # ── Step 1: GPU full? Swap out ──
            if gpu_used_ratio >= self._watermark_high and self._running_seqs:
                # Select victim: the sequence with the most blocks
                victim = max(
                    self._running_seqs,
                    key=lambda sid: len(self._bm.get_block_table(sid)),
                )
                op = self._bm.swap_out(victim)
                if op == BlockSwapOp.SWAP_OUT:
                    decision.swap_out_seqs.append(victim)
                    self._running_seqs.remove(victim)
                    self._swapped_seqs.append(victim)
                    self._total_swaps_out += 1
                    decision.gpu_exhausted = True

            # ── Step 2: GPU has room? Swap in ──
            elif gpu_used_ratio <= self._watermark_low and self._swapped_seqs:
                candidate = self._swapped_seqs.pop(0)
                op = self._bm.swap_in(candidate)
                if op == BlockSwapOp.SWAP_IN:
                    decision.swap_in_seqs.append(candidate)
                    self._running_seqs.append(candidate)
                    self._total_swaps_in += 1

            # ── Step 3: Service waiting sequences ──
            while self._waiting_seqs and self._bm.get_num_free_gpu_blocks() > 0:
                sid = self._waiting_seqs.pop(0)
                # Estimate: allocate 1 block minimum
                if self._bm.can_allocate(self._bm._config.block_size):
                    self._bm.allocate(sid, self._bm._config.block_size)
                    self._running_seqs.append(sid)
                    decision.blocks_allocated += 1
                else:
                    self._waiting_seqs.insert(0, sid)  # put back
                    break

            # ── Step 4: Run active sequences ──
            active_count = min(len(self._running_seqs), self._max_batch)
            decision.scheduled_seqs = self._running_seqs[:active_count]
            # Rotate: move scheduled to end (round-robin)
            self._running_seqs = self._running_seqs[active_count:] + decision.scheduled_seqs

            return decision

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "step_count": self._step_count,
                "running_sequences": len(self._running_seqs),
                "waiting_sequences": len(self._waiting_seqs),
                "swapped_sequences": len(self._swapped_seqs),
                "total_swaps_out": self._total_swaps_out,
                "total_swaps_in": self._total_swaps_in,
                "gpu_free_blocks": self._bm.get_num_free_gpu_blocks(),
                "fc_free_blocks": self._bm.get_num_free_fc_blocks(),
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"KVCacheScheduler(running={s['running_sequences']}, "
            f"waiting={s['waiting_sequences']}, swapped={s['swapped_sequences']}, "
            f"step={s['step_count']})"
        )
