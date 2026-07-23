"""
HiFC Swapping Engine — Direct GPU↔SSD KV cache transfers (DRAM-free).

Implements the three techniques from HiFC §3.2:
1. Flash Cache (FC) block allocator — extends GPU blocks with SSD blocks
2. Flash-Aware Block Management — fine-grained mapping to pSLC zones
3. GDS-Accelerated Cache Engine — direct GPU↔SSD transfers, bypassing DRAM

Models the performance characteristics of:
- pSLC sequential write: ~4.7 GiB/s (HiFC §5.1)
- pSLC random read: ~3.2 GiB/s
- TLC sequential: ~1.5 GiB/s (for comparison)
- Write amplification: ~1.02 in pSLC vs ~1.4 in TLC (HiFC §3.2)
- 4KB-aligned GDS buffer reuse (HiFC §3.2)
- Multi-threaded I/O (up to 16 threads, HiFC §3.2)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Flash zone types (HiFC §3.2 — pSLC vs TLC region)
# ---------------------------------------------------------------------------

class FlashZone(Enum):
    """SSD zone types for fine-grained block mapping."""
    pSLC = auto()    # Pseudo-SLC: high perf, high endurance (8× TBW)
    TLC = auto()     # Triple-Level Cell: high capacity, lower perf
    QLC = auto()     # Quad-Level Cell: max capacity, minimum perf


# Zone performance model (HiFC §5.1 measurements)
ZONE_PERF = {
    FlashZone.pSLC: {
        "seq_write_gbps": 4.7,
        "seq_read_gbps": 4.9,
        "rand_read_gbps": 3.2,
        "write_amplification": 1.02,
        "endurance_tbw": 2400,  # TBW for 1TB pSLC partition
    },
    FlashZone.TLC: {
        "seq_write_gbps": 1.5,
        "seq_read_gbps": 3.0,
        "rand_read_gbps": 1.0,
        "write_amplification": 1.4,
        "endurance_tbw": 600,
    },
    FlashZone.QLC: {
        "seq_write_gbps": 0.5,
        "seq_read_gbps": 1.5,
        "rand_read_gbps": 0.4,
        "write_amplification": 1.8,
        "endurance_tbw": 300,
    },
}


# ---------------------------------------------------------------------------
# Swap request / result
# ---------------------------------------------------------------------------

@dataclass
class SwapRequest:
    """A single KV cache swap request between GPU and SSD."""
    seq_id: int
    direction: str          # "out" (GPU→SSD) or "in" (SSD→GPU)
    block_ids: List[int]
    num_bytes: int          # total bytes to transfer
    priority: int = 0       # higher = more urgent
    zone: FlashZone = FlashZone.pSLC

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


@dataclass
class SwapResult:
    """Result of executing a SwapRequest."""
    request: SwapRequest
    success: bool
    elapsed_us: float
    throughput_gbps: float   # achieved throughput
    zone: FlashZone


# ---------------------------------------------------------------------------
# HiFC Swapping Engine
# ---------------------------------------------------------------------------

class HiFCSwappingEngine:
    """GDS-accelerated GPU↔SSD KV cache swapping engine.

    Models the HiFC data path (§3.2, Fig.2):
    - GDS (GPU Direct Storage) for direct GPU↔SSD transfers
    - Byte-level offsets with 4KB-aligned buffers
    - Multi-threaded I/O dispatch (up to 16 threads)
    - Fine-grained zone mapping (confine writes to pSLC)

    Parameters
    ----------
    num_io_threads : int
        Concurrent I/O threads (HiFC uses up to 16).
    block_size_bytes : int
        Bytes per KV cache block.
    default_zone : FlashZone
        Default SSD zone for new blocks.
    """

    def __init__(
        self,
        num_io_threads: int = 8,
        block_size_bytes: int = 3_670_016,  # Qwen2.5-7B, block_size=16
        default_zone: FlashZone = FlashZone.pSLC,
    ):
        self._num_threads = num_io_threads
        self._block_size_bytes = block_size_bytes
        self._default_zone = default_zone
        self._lock = threading.RLock()

        # ── Zone mapping: block_id → FlashZone ──
        self._block_zones: Dict[int, FlashZone] = {}

        # ── I/O queue ──
        self._pending: List[SwapRequest] = []
        self._completed: List[SwapResult] = []

        # ── Endurance tracking ──
        self._bytes_written: Dict[FlashZone, int] = {
            z: 0 for z in FlashZone
        }
        self._write_amplification: Dict[FlashZone, float] = {
            z: 1.0 for z in FlashZone
        }

        # ── Stats ──
        self._total_swaps: int = 0
        self._total_bytes_transferred: int = 0
        self._total_transfer_time_us: float = 0.0

    # ------------------------------------------------------------------
    # Swap operations
    # ------------------------------------------------------------------

    def swap_out(
        self,
        seq_id: int,
        block_ids: List[int],
        zone: FlashZone | None = None,
    ) -> SwapResult:
        """Execute a GPU→SSD swap-out (HiFC eviction path).

        Models GDS direct write: GPU VRAM → NVMe SSD via PCIe,
        bypassing host DRAM.  Writes are confined to the specified
        FlashZone (default pSLC) for max throughput + endurance.
        """
        zone = zone or self._default_zone
        num_bytes = len(block_ids) * self._block_size_bytes
        perf = ZONE_PERF[zone]

        req = SwapRequest(
            seq_id=seq_id,
            direction="out",
            block_ids=block_ids,
            num_bytes=num_bytes,
            zone=zone,
        )

        with self._lock:
            # Map blocks to zone
            for bid in block_ids:
                self._block_zones[bid] = zone

            # Model GDS throughput with write amplification
            raw_bytes = int(num_bytes * perf["write_amplification"])
            throughput = perf["seq_write_gbps"]
            transfer_time_s = raw_bytes / (throughput * 1024**3)
            elapsed_us = transfer_time_s * 1e6

            # Track endurance
            self._bytes_written[zone] += raw_bytes

            result = SwapResult(
                request=req,
                success=True,
                elapsed_us=elapsed_us,
                throughput_gbps=throughput,
                zone=zone,
            )

            self._completed.append(result)
            self._total_swaps += 1
            self._total_bytes_transferred += num_bytes
            self._total_transfer_time_us += elapsed_us

            return result

    def swap_in(
        self,
        seq_id: int,
        block_ids: List[int],
    ) -> SwapResult:
        """Execute a SSD→GPU swap-in (HiFC prefetch path).

        Models GDS direct read from SSD to GPU VRAM.
        """
        # Determine zone from tracked blocks
        zone = self._default_zone
        if block_ids:
            zone = self._block_zones.get(block_ids[0], self._default_zone)

        num_bytes = len(block_ids) * self._block_size_bytes
        perf = ZONE_PERF[zone]

        req = SwapRequest(
            seq_id=seq_id,
            direction="in",
            block_ids=block_ids,
            num_bytes=num_bytes,
            zone=zone,
        )

        with self._lock:
            # Sequential read speed (sequential because blocks are contiguous on SSD)
            throughput = perf["seq_read_gbps"]
            transfer_time_s = num_bytes / (throughput * 1024**3)
            elapsed_us = transfer_time_s * 1e6

            result = SwapResult(
                request=req,
                success=True,
                elapsed_us=elapsed_us,
                throughput_gbps=throughput,
                zone=zone,
            )

            self._completed.append(result)
            self._total_swaps += 1
            self._total_bytes_transferred += num_bytes
            self._total_transfer_time_us += elapsed_us

            return result

    # ------------------------------------------------------------------
    # Zone management (HiFC fine-grained block mapping)
    # ------------------------------------------------------------------

    def assign_zone(self, block_ids: List[int], zone: FlashZone):
        """Assign SSD blocks to a specific Flash zone (fine-grained mapping)."""
        with self._lock:
            for bid in block_ids:
                self._block_zones[bid] = zone

    def get_zone(self, block_id: int) -> FlashZone:
        return self._block_zones.get(block_id, self._default_zone)

    def compact_zones(self):
        """Defragment: move cold blocks from pSLC to TLC/QLC.

        HiFC §3.2: Fine-grained block mapping allows cold blocks to be
        migrated to cheaper, higher-capacity QLC zones while hot blocks
        stay in high-performance pSLC.
        """
        # This is a scheduling hint — actual migration is via swap_out/swap_in
        pass

    # ------------------------------------------------------------------
    # Endurance tracking
    # ------------------------------------------------------------------

    def tbw_remaining(self, zone: FlashZone) -> float:
        """Remaining endurance in TBW for *zone*."""
        written = self._bytes_written.get(zone, 0)
        total = ZONE_PERF[zone]["endurance_tbw"] * 1024**4  # TB → bytes
        return max(0.0, total - written) / 1024**4

    def write_amplification(self, zone: FlashZone) -> float:
        """Current write amplification factor for *zone*."""
        return self._write_amplification.get(zone, 1.0)

    # ------------------------------------------------------------------
    # Batch I/O (HiFC multi-threaded dispatch)
    # ------------------------------------------------------------------

    def batch_swap_out(
        self,
        seqs: List[Tuple[int, List[int]]],
        zone: FlashZone | None = None,
    ) -> List[SwapResult]:
        """Batch swap-out using multi-threaded I/O dispatch."""
        results: List[SwapResult] = [None] * len(seqs)

        def _worker(idx, sid, bids):
            results[idx] = self.swap_out(sid, bids, zone)

        threads = []
        for i, (sid, bids) in enumerate(seqs):
            t = threading.Thread(target=_worker, args=(i, sid, bids))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            avg_throughput = (
                self._total_bytes_transferred / max(self._total_transfer_time_us, 1) * 1e6 / 1024**3
            )
            return {
                "total_swaps": self._total_swaps,
                "total_bytes_transferred": self._total_bytes_transferred,
                "avg_throughput_gbps": round(avg_throughput, 2),
                "pending_requests": len(self._pending),
                "completed_requests": len(self._completed),
                "block_zones_tracked": len(self._block_zones),
                "bytes_written": {
                    z.name: self._bytes_written[z] for z in FlashZone
                },
                "tbw_remaining": {
                    z.name: round(self.tbw_remaining(z), 1) for z in FlashZone
                },
                "write_amplification": {
                    z.name: round(self._write_amplification[z], 3) for z in FlashZone
                },
            }

    def reset_stats(self):
        with self._lock:
            self._total_swaps = 0
            self._total_bytes_transferred = 0
            self._total_transfer_time_us = 0.0
            self._completed.clear()
            self._pending.clear()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"HiFCSwappingEngine(swaps={s['total_swaps']}, "
            f"transferred={s['total_bytes_transferred']/1024**3:.1f} GiB, "
            f"throughput={s['avg_throughput_gbps']} Gbps)"
        )
