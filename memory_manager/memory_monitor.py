"""
Memory Monitor — real-time snapshot collection + JSON export for visualization.

Collects periodic snapshots from all memory subsystems and exports them
in a structured format consumed by the monitoring dashboard (web/monitor/).

Each snapshot captures:
- Block allocator state (free/used/shared/pinned per tier)
- Prefix cache statistics (hit rate, entries, hot blocks)
- Lifecycle tracker (phase distribution, migration counts)
- Hierarchical store (tier usage in bytes/ratio)
- Compressor/deduplicator (tokens saved, compression ratio)

The snapshot dict format is designed to be directly consumed by dashboard
charts without further transformation.

Usage
-----
    monitor = MemoryMonitor(memory_manager)
    monitor.start(interval_s=5.0)    # collect every 5 seconds
    # ... agent runs ...
    history = monitor.stop()         # returns list of snapshots
    monitor.export_json("benchmark_results.json")

References
----------
- EternCrypt/EternFlow dashboard visualization patterns
- ET-Agent Phase 1-5 memory_manager subsystems
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from memory_manager.kv_block import KVBlockState, StorageTier


# ---------------------------------------------------------------------------
# Snapshot data model
# ---------------------------------------------------------------------------

@dataclass
class BlockSnapshot:
    """One frame of block allocator state."""
    total: int = 0
    free: int = 0
    used: int = 0
    gpu_blocks: int = 0
    cpu_blocks: int = 0
    ssd_blocks: int = 0
    shared: int = 0
    pinned: int = 0
    active_requests: int = 0
    usage_ratio: float = 0.0

    def to_chart_data(self) -> dict:
        return {
            "labels": ["GPU", "CPU", "SSD"],
            "values": [self.gpu_blocks, self.cpu_blocks, self.ssd_blocks],
            "shared": self.shared,
            "pinned": self.pinned,
        }


@dataclass
class PrefixSnapshot:
    """One frame of prefix cache state."""
    total_entries: int = 0
    pinned_entries: int = 0
    hot_entries: int = 0
    hit_rate: float = 0.0
    total_lookups: int = 0
    hits: int = 0
    misses: int = 0
    blocks_reused: int = 0
    block_reuse_rate: float = 0.0


@dataclass
class TierSnapshot:
    """One frame of hierarchical store tier usage."""
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    ssd_bytes: int = 0
    gpu_ratio: float = 0.0
    cpu_ratio: float = 0.0
    ssd_ratio: float = 0.0
    total_migrations: int = 0
    total_prefetches: int = 0

    def to_chart_data(self) -> dict:
        return {
            "labels": ["GPU", "CPU", "SSD"],
            "bytes": [self.gpu_bytes, self.cpu_bytes, self.ssd_bytes],
            "ratios": [self.gpu_ratio, self.cpu_ratio, self.ssd_ratio],
        }


@dataclass
class LifecycleSnapshot:
    """One frame of agent lifecycle state."""
    total_requests: int = 0
    active_requests: int = 0
    waiting_requests: int = 0
    protected_sessions: int = 0
    prefill_count: int = 0
    decoding_count: int = 0
    tool_call_count: int = 0
    idle_count: int = 0
    completed_count: int = 0
    total_transitions: int = 0
    total_demotions: int = 0
    total_promotions: int = 0
    total_evictions: int = 0

    def to_chart_data(self) -> dict:
        return {
            "labels": ["Prefill", "Decoding", "ToolCall", "Idle", "Completed"],
            "values": [
                self.prefill_count, self.decoding_count,
                self.tool_call_count, self.idle_count, self.completed_count,
            ],
        }


@dataclass
class CompressionSnapshot:
    """One frame of compression state."""
    history_compressions: int = 0
    observation_compressions: int = 0
    total_tokens_saved: int = 0
    history_threshold: int = 0
    has_optimized_guideline: bool = False
    total_messages_dropped: int = 0
    dedup_tokens_saved: int = 0
    tools_tracked: int = 0
    tool_schema_tokens_saved: int = 0


@dataclass
class MemorySnapshot:
    """Complete snapshot of all memory subsystems at one point in time."""
    timestamp: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    blocks: BlockSnapshot = field(default_factory=BlockSnapshot)
    prefix: PrefixSnapshot = field(default_factory=PrefixSnapshot)
    tiers: TierSnapshot = field(default_factory=TierSnapshot)
    lifecycle: LifecycleSnapshot = field(default_factory=LifecycleSnapshot)
    compression: CompressionSnapshot = field(default_factory=CompressionSnapshot)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "elapsed_s": self.elapsed_s,
            "blocks": asdict(self.blocks),
            "prefix": asdict(self.prefix),
            "tiers": asdict(self.tiers),
            "lifecycle": asdict(self.lifecycle),
            "compression": asdict(self.compression),
        }

    def to_chart_json(self) -> str:
        """Export as chart-ready JSON (consumed by dashboard)."""
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Memory Monitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Periodic snapshot collector for memory subsystem visualization.

    Parameters
    ----------
    memory_manager : AgentMemoryManager
        The wired memory manager (from ``agent.memory_hooks``).
    """

    def __init__(self, memory_manager):
        self._mgr = memory_manager
        self._lock = threading.RLock()
        self._snapshots: List[MemorySnapshot] = []
        self._timer: Optional[threading.Timer] = None
        self._start_time: float = 0.0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, interval_s: float = 5.0):
        """Begin collecting snapshots every *interval_s* seconds."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._start_time = time.time()
            self._snapshots.clear()
            self._collect_snapshot()
            self._schedule_next(interval_s)

    def stop(self) -> List[MemorySnapshot]:
        """Stop collecting and return all snapshots."""
        with self._lock:
            self._running = False
            if self._timer:
                self._timer.cancel()
                self._timer = None
            return list(self._snapshots)

    def snapshot(self) -> MemorySnapshot:
        """Take a single snapshot now (can be called ad-hoc)."""
        return self._collect_snapshot()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str):
        """Export all snapshots to a JSON file for dashboard consumption."""
        data = {
            "metadata": {
                "total_snapshots": len(self._snapshots),
                "duration_s": self._snapshots[-1].elapsed_s if self._snapshots else 0,
                "model": self._mgr._model_name,
                "config": self._mgr._config.__dict__ if hasattr(self._mgr._config, '__dict__') else {},
            },
            "snapshots": [s.to_dict() for s in self._snapshots],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def export_chart_data(self) -> dict:
        """Export aggregated chart data for dashboard visualization.

        Returns a dict directly consumable by the dashboard's chart components.
        """
        if not self._snapshots:
            return {"error": "no data"}

        snapshots = self._snapshots

        # Time series data
        timestamps = [s.elapsed_s for s in snapshots]

        return {
            "timestamps": timestamps,
            "series": {
                # Block usage over time
                "gpu_blocks_used": [
                    s.blocks.gpu_blocks - s.blocks.free
                    for s in snapshots
                ],
                "cpu_blocks": [s.blocks.cpu_blocks for s in snapshots],
                "ssd_blocks": [s.blocks.ssd_blocks for s in snapshots],
                "shared_blocks": [s.blocks.shared for s in snapshots],

                # Prefix cache hit rate
                "prefix_hit_rate": [s.prefix.hit_rate for s in snapshots],

                # Tier usage ratios
                "gpu_usage_ratio": [s.tiers.gpu_ratio for s in snapshots],
                "cpu_usage_ratio": [s.tiers.cpu_ratio for s in snapshots],

                # Lifecycle — active vs waiting
                "active_requests": [s.lifecycle.active_requests for s in snapshots],
                "waiting_requests": [s.lifecycle.waiting_requests for s in snapshots],

                # Migration counts (cumulative)
                "total_migrations": [s.tiers.total_migrations for s in snapshots],
                "total_prefetches": [s.tiers.total_prefetches for s in snapshots],

                # Compression totals
                "tokens_saved": [s.compression.total_tokens_saved for s in snapshots],
            },
            "summary": {
                "peak_gpu_blocks": max(
                    s.blocks.gpu_blocks - s.blocks.free for s in snapshots
                ),
                "avg_prefix_hit_rate": sum(
                    s.prefix.hit_rate for s in snapshots
                ) / len(snapshots),
                "total_migrations": snapshots[-1].tiers.total_migrations,
                "total_tokens_saved": snapshots[-1].compression.total_tokens_saved,
                "max_active_sessions": max(
                    s.lifecycle.active_requests for s in snapshots
                ),
                "max_waiting_sessions": max(
                    s.lifecycle.waiting_requests for s in snapshots
                ),
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_snapshot(self) -> MemorySnapshot:
        """Capture one snapshot from all subsystems."""
        s = self._mgr.stats()

        a = s.get("allocator", {})
        p = s.get("prefix_cache", {})
        h = s.get("hierarchical_store", {})
        l = s.get("lifecycle", {})
        c = s.get("compressor", {})
        d = s.get("deduplicator", {})
        t = s.get("tool_compressor", {})

        phases = l.get("phases", {})

        snap = MemorySnapshot(
            elapsed_s=time.time() - self._start_time,
            blocks=BlockSnapshot(
                total=a.get("total_blocks", 0),
                free=a.get("free_blocks", 0),
                used=a.get("used_blocks", 0),
                gpu_blocks=h.get("gpu_blocks", 0),
                cpu_blocks=h.get("cpu_blocks", 0),
                ssd_blocks=h.get("ssd_blocks", 0),
                shared=a.get("shared_blocks", 0),
                pinned=a.get("pinned_blocks", 0),
                active_requests=a.get("active_requests", 0),
                usage_ratio=a.get("usage_ratio", 0.0),
            ),
            prefix=PrefixSnapshot(
                total_entries=p.get("total_entries", 0),
                pinned_entries=p.get("pinned_entries", 0),
                hot_entries=p.get("hot_entries", 0),
                hit_rate=p.get("hit_rate", 0.0),
                total_lookups=p.get("total_lookups", 0),
                hits=p.get("hits", 0),
                misses=p.get("misses", 0),
                blocks_reused=p.get("blocks_reused", 0),
                block_reuse_rate=p.get("block_reuse_rate", 0.0),
            ),
            tiers=TierSnapshot(
                gpu_bytes=h.get("gpu_usage_bytes", 0),
                cpu_bytes=h.get("cpu_usage_bytes", 0),
                ssd_bytes=h.get("ssd_usage_bytes", 0),
                gpu_ratio=h.get("gpu_usage_ratio", 0.0),
                cpu_ratio=h.get("cpu_usage_ratio", 0.0),
                ssd_ratio=h.get("ssd_usage_ratio", 0.0),
                total_migrations=h.get("total_migrations", 0),
                total_prefetches=h.get("total_prefetches", 0),
            ),
            lifecycle=LifecycleSnapshot(
                total_requests=l.get("total_requests", 0),
                active_requests=l.get("active_requests", 0),
                waiting_requests=l.get("waiting_requests", 0),
                protected_sessions=l.get("protected_sessions", 0),
                prefill_count=phases.get("PREFILL", 0),
                decoding_count=phases.get("DECODING", 0),
                tool_call_count=phases.get("TOOL_CALL", 0),
                idle_count=phases.get("IDLE", 0),
                completed_count=phases.get("COMPLETED", 0),
                total_transitions=l.get("total_transitions", 0),
                total_demotions=l.get("total_demotions", 0),
                total_promotions=l.get("total_promotions", 0),
                total_evictions=l.get("total_evictions", 0),
            ),
            compression=CompressionSnapshot(
                history_compressions=c.get("total_history_compressions", 0),
                observation_compressions=c.get("total_observation_compressions", 0),
                total_tokens_saved=int(c.get("total_tokens_saved", 0) + d.get("total_tokens_saved", 0) + t.get("total_tokens_saved", 0)),
                history_threshold=c.get("history_threshold", 0),
                has_optimized_guideline=c.get("has_optimized_guideline", False),
                total_messages_dropped=d.get("total_messages_dropped", 0),
                dedup_tokens_saved=d.get("total_tokens_saved", 0),
                tools_tracked=t.get("tracked_tools", 0),
                tool_schema_tokens_saved=t.get("total_tokens_saved", 0),
            ),
        )

        with self._lock:
            self._snapshots.append(snap)

        return snap

    def _schedule_next(self, interval_s: float):
        if not self._running:
            return
        self._timer = threading.Timer(interval_s, self._tick, args=(interval_s,))
        self._timer.daemon = True
        self._timer.start()

    def _tick(self, interval_s: float):
        self._collect_snapshot()
        self._schedule_next(interval_s)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        return (
            f"MemoryMonitor({self.snapshot_count} snapshots, "
            f"running={self._running})"
        )
