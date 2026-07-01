"""
ET-Agent Memory Manager — Agent-aware KV Cache memory management.

Based on vLLM's PagedAttention (SOSP 2023) and MoonCake's KVCache-centric
architecture (FAST 2025).  Provides:

Phase 1 — Block management
    Fixed-size page-based KV Cache allocation, logical→physical block
    translation with copy-on-write sharing, and thread-safe block pool.

Phase 2 — Prefix caching
    MoonCake-style hash-chain prefix matching, agent-aware caching
    strategies (system prompts, tool schemas, session history), and
    eviction policies (LRU/LFU/AgentAware).

Phase 3 — Hierarchical storage  (coming)
    GPU→CPU→SSD tiered KV Cache with lifecycle-aware migration.

Components
----------
``KVBlockAllocator``       — fixed-size physical block pool with ref-counting
``BlockTableManager``      — per-request logical→physical mapping + COW
``PrefixHashCache``        — MoonCake hash-chain prefix matching
``AgentPrefixCache``       — agent lifecycle-aware cache strategies
``EvictionPolicy``         — LRU / LFU / AgentAware eviction policies
"""

# ── Phase 1: Block management ──
from memory_manager.kv_block import (
    KVBlock,
    KVBlockState,
    BlockTableEntry,
    StorageTier,
)
from memory_manager.block_table import (
    BlockTable,
    BlockTableManager,
)
from memory_manager.kv_block_allocator import (
    KVBlockAllocator,
    OutOfMemoryError,
    BlockNotFoundError,
)
from memory_manager.config import MemoryConfig, ModelKVProfile, KNOWN_PROFILES

# ── Phase 2: Prefix caching ──
from memory_manager.kv_eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    TieredLRUPolicy,
    AgentAwarePolicy,
    make_policy,
)
from memory_manager.kv_prefix_cache import (
    PrefixHashCache,
    PrefixCacheEntry,
    compute_prefix_hashes,
    compute_block_hash,
)
from memory_manager.agent_prefix_cache import (
    AgentPrefixCache,
    estimate_agent_savings,
)

__all__ = [
    # ── Block management ──
    "KVBlockAllocator",
    "BlockTableManager",
    "BlockTable",
    "KVBlock",
    "KVBlockState",
    "BlockTableEntry",
    "StorageTier",
    "MemoryConfig",
    "ModelKVProfile",
    "KNOWN_PROFILES",
    "OutOfMemoryError",
    "BlockNotFoundError",
    # ── Prefix caching ──
    "PrefixHashCache",
    "PrefixCacheEntry",
    "AgentPrefixCache",
    "compute_prefix_hashes",
    "compute_block_hash",
    # ── Eviction ──
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "TieredLRUPolicy",
    "AgentAwarePolicy",
    "make_policy",
    # ── Utilities ──
    "estimate_agent_savings",
]
