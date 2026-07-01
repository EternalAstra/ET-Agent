"""
ET-Agent Memory Manager — Agent-aware KV Cache memory management.

Based on vLLM's PagedAttention (SOSP 2023), MoonCake's KVCache-centric
architecture (FAST 2025), and ACON context compression (ICML 2026).

Phase 1 — Block management
Phase 2 — Prefix caching
Phase 3 — Hierarchical storage
Phase 4 — Context compression (ACON-style)
"""

# ── Phase 1 ──
from memory_manager.kv_block import (
    KVBlock, KVBlockState, BlockTableEntry, StorageTier,
)
from memory_manager.block_table import BlockTable, BlockTableManager
from memory_manager.kv_block_allocator import (
    KVBlockAllocator, OutOfMemoryError, BlockNotFoundError,
)
from memory_manager.config import MemoryConfig, ModelKVProfile, KNOWN_PROFILES

# ── Phase 2 ──
from memory_manager.kv_eviction_policy import (
    EvictionPolicy, LRUEvictionPolicy, LFUEvictionPolicy,
    TieredLRUPolicy, AgentAwarePolicy, make_policy,
)
from memory_manager.kv_prefix_cache import (
    PrefixHashCache, PrefixCacheEntry,
    compute_prefix_hashes, compute_block_hash,
)
from memory_manager.agent_prefix_cache import (
    AgentPrefixCache, estimate_agent_savings,
)

# ── Phase 3 ──
from memory_manager.kv_lifecycle_tracker import (
    AgentPhase, RequestLifecycle, LifecycleAwareKVManager,
    LifecycleTiming, phase_requires_gpu, phase_latency_sensitive,
)
from memory_manager.kv_hierarchical_store import (
    HierarchicalKVStore, MigrationTask,
)

# ── Phase 4: Context compression (ACON, ICML 2026) ──
from memory_manager.context_compressor import (
    ContextCompressor, CompressionMode, CompressionThresholds,
    CompressionResult,
)
from memory_manager.prompt_deduplicator import (
    PromptDeduplicator, hash_content, DedupResult,
)
from memory_manager.tool_schema_compressor import (
    ToolSchemaCompressor, CompressionTier,
)

__all__ = [
    # Phase 1
    "KVBlockAllocator", "BlockTableManager", "BlockTable",
    "KVBlock", "KVBlockState", "BlockTableEntry", "StorageTier",
    "MemoryConfig", "ModelKVProfile", "KNOWN_PROFILES",
    "OutOfMemoryError", "BlockNotFoundError",
    # Phase 2
    "PrefixHashCache", "PrefixCacheEntry", "AgentPrefixCache",
    "compute_prefix_hashes", "compute_block_hash",
    "EvictionPolicy", "LRUEvictionPolicy", "LFUEvictionPolicy",
    "TieredLRUPolicy", "AgentAwarePolicy", "make_policy",
    # Phase 3
    "HierarchicalKVStore", "MigrationTask",
    "AgentPhase", "RequestLifecycle", "LifecycleAwareKVManager",
    "LifecycleTiming", "phase_requires_gpu", "phase_latency_sensitive",
    "estimate_agent_savings",
    # Phase 4
    "ContextCompressor", "CompressionMode", "CompressionThresholds",
    "CompressionResult",
    "PromptDeduplicator", "hash_content", "DedupResult",
    "ToolSchemaCompressor", "CompressionTier",
]
