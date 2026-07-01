"""
ET-Agent Memory Manager — Phase 1: KV Cache Block Management.

Based on vLLM's PagedAttention (SOSP 2023) and MoonCake's KVCache-centric
architecture (FAST 2025).  Provides fixed-size page-based KV Cache allocation,
logical→physical block translation with copy-on-write sharing, and thread-safe
block pool management.

Core Components
---------------
- ``KVBlockAllocator``  — fixed-size physical block pool with ref-counting
- ``BlockTableManager`` — per-request logical→physical mapping + COW
- ``KVBlock``          — single physical KV Cache block metadata
- ``BlockTableEntry``  — one row in a request's logical→physical table
"""

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
from memory_manager.config import MemoryConfig

__all__ = [
    # Core classes
    "KVBlockAllocator",
    "BlockTableManager",
    "BlockTable",
    # Data classes
    "KVBlock",
    "KVBlockState",
    "BlockTableEntry",
    "StorageTier",
    # Config
    "MemoryConfig",
    # Errors
    "OutOfMemoryError",
    "BlockNotFoundError",
]
