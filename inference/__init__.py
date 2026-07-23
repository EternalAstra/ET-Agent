"""
ET-Agent Inference Layer — vLLM-compatible KV Cache backend with HiFC extensions.

Bridges the ``memory_manager`` package (Phase 1–4) to the vLLM block manager API
so that our KV Cache management can be directly compared against vLLM's PagedAttention
and HiFC's Flash-based swapping.

Provides:
- ``VLLMBlockManager`` — drop-in compatible with vLLM's BlockSpaceManager interface
- ``HiFCSwappingEngine``  — DRAM-free GPU↔SSD KV cache swapping (HiFC §3.2)
- ``KVCacheScheduler``   — HiFC-style scheduler with fine-grained block mapping
- ``InferenceMetrics``   — throughput / memory-util / swap-count / TTFT / TBT

Architecture
------------
::

    vLLM Scheduler  ──block_table──►  VLLMBlockManager
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
               KVBlockAllocator   HierarchicalKVStore   HiFCSwappingEngine
               (Phase 1)          (Phase 3)             (GDS SSD direct)
                      │                │                │
                      └────────────────┼────────────────┘
                                       ▼
                               InferenceMetrics
                               (throughput, utilization, swap, TTFT, TBT)

References
----------
- vLLM (Kwon et al., SOSP 2023) — PagedAttention block manager
- HiFC (Jeong et al., 2025) — DRAM-free GPU↔SSD swapping
- MoonCake (Qin et al., FAST 2025) — KVCache-centric disaggregation
"""

from inference.vllm_block_manager import (
    VLLMBlockManager,
    SequenceBlocks,
    BlockSwapOp,
)
from inference.swapping_engine import (
    HiFCSwappingEngine,
    SwapRequest,
    SwapResult,
    FlashZone,
)
from inference.scheduler import (
    KVCacheScheduler,
    ScheduleDecision,
)
from inference.metrics import (
    InferenceMetrics,
    MetricsSnapshot,
    BenchmarkReport,
)

__all__ = [
    # Block manager
    "VLLMBlockManager",
    "SequenceBlocks",
    "BlockSwapOp",
    # Swapping engine
    "HiFCSwappingEngine",
    "SwapRequest",
    "SwapResult",
    "FlashZone",
    # Scheduler
    "KVCacheScheduler",
    "ScheduleDecision",
    # Metrics
    "InferenceMetrics",
    "MetricsSnapshot",
    "BenchmarkReport",
]
