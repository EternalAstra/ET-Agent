"""
KV Cache Block data types.

Implements the fixed-size KV Cache page abstraction from vLLM's
PagedAttention (SOSP 2023, §4).  Each block holds the key/value tensors
for one "page" of tokens (default 16) across all transformer layers.

Physical blocks are GPU/CPU resources managed by ``KVBlockAllocator``;
logical blocks are per-request indices resolved through a ``BlockTable``.

References
----------
- vLLM §4.1–4.3  (PagedAttention algorithm, block table, COW sharing)
- MoonCake §3, Figure 3  (KVCache pool, hash-based prefix dedup)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KVBlockState(Enum):
    """Lifecycle state of a physical KV Cache block."""
    FREE = auto()           # Available for allocation
    ALLOCATED = auto()      # In use by a single request
    SHARED = auto()         # In use by multiple requests (ref_count > 1)
    PINNED = auto()         # Permanently cached (e.g. system prompt prefix)
    EVICTING = auto()       # Being migrated to lower storage tier


class StorageTier(Enum):
    """Storage tier for a KV Cache block (Phase 3 hierarchical storage)."""
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"


# ---------------------------------------------------------------------------
# KVBlock — a single physical KV Cache page
# ---------------------------------------------------------------------------

@dataclass
class KVBlock:
    """A single physical KV Cache block (page).

    Maps to one page of *block_size* tokens.  Stores K/V tensors for all
    transformer layers.  When the block is shared across multiple requests,
    ``ref_count`` tracks how many logical blocks point here.

    Attributes
    ----------
    block_id : int
        Unique physical block identifier (index into the allocator pool).
    state : KVBlockState
        Current lifecycle state.
    num_tokens : int
        How many tokens are actually stored in this block (≤ block_size).
    ref_count : int
        Number of logical block-table entries referencing this block.
        Used for copy-on-write: when ref_count > 1 and a request wants to
        write, the block is cloned before modification.
    storage_tier : StorageTier
        Where the block data currently resides.
    last_access_ts : float
        Monotonic timestamp of last read/write (for LRU eviction).
    access_count : int
        Cumulative access counter (for LFU eviction heuristics).
    group_id : str | None
        Optional grouping key — e.g. session_id for same-session blocks
        that should be evicted together.
    """

    block_id: int
    state: KVBlockState = KVBlockState.FREE
    num_tokens: int = 0
    ref_count: int = 0
    storage_tier: StorageTier = StorageTier.GPU
    last_access_ts: float = 0.0
    access_count: int = 0
    group_id: Optional[str] = None

    # ── computed helpers ──

    @property
    def is_free(self) -> bool:
        return self.state == KVBlockState.FREE

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    @property
    def is_pinned(self) -> bool:
        return self.state == KVBlockState.PINNED

    @property
    def is_evictable(self) -> bool:
        """Can this block be evicted?  Pinned blocks are never evicted;
        shared blocks are eligible only when their ref_count drops to 0."""
        return self.state not in (KVBlockState.FREE, KVBlockState.PINNED)

    # ── mutation helpers ──

    def mark_allocated(self, num_tokens: int = 0, group_id: str | None = None):
        """Transition from FREE → ALLOCATED."""
        self.state = KVBlockState.ALLOCATED
        self.ref_count = 1
        self.num_tokens = num_tokens
        self.group_id = group_id

    def mark_free(self):
        """Transition to FREE, resetting all fields."""
        self.state = KVBlockState.FREE
        self.num_tokens = 0
        self.ref_count = 0
        self.storage_tier = StorageTier.GPU
        self.last_access_ts = 0.0
        self.access_count = 0
        self.group_id = None

    def increment_ref(self):
        """Add one reference (e.g. when another request shares this block)."""
        self.ref_count += 1
        if self.ref_count > 1:
            self.state = KVBlockState.SHARED

    def decrement_ref(self):
        """Release one reference.  Returns True if this was the last ref."""
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 1:
            self.state = KVBlockState.ALLOCATED
        elif self.ref_count == 0:
            self.mark_free()
            return True
        return False

    def touch(self, ts: float):
        """Record an access at monotonic timestamp *ts*."""
        self.last_access_ts = ts
        self.access_count += 1


# ---------------------------------------------------------------------------
# BlockTableEntry — one row in a request's logical→physical mapping
# ---------------------------------------------------------------------------

@dataclass
class BlockTableEntry:
    """A single row in a per-request Block Table.

    Maps one logical block index to a physical block, tracking how many
    token slots in that block are filled and whether this entry is under
    copy-on-write protection.

    Attributes
    ----------
    logical_idx : int
        Logical block position within the request's sequence.
    physical_id : int
        Physical block ID in the allocator pool.
    num_filled : int
        How many tokens in this block are valid (≤ block_size).
    is_cow : bool
        If True, the next write to this block must clone it first (vLLM §4.3).
    shared_from : str | None
        When this entry was created by prefix-sharing, the source request_id.
        Used for debugging and eviction-group tracking.
    """

    logical_idx: int
    physical_id: int
    num_filled: int = 0
    is_cow: bool = False
    shared_from: Optional[str] = None

    @property
    def is_full(self) -> bool:
        """True if every token slot in this block is occupied.

        Note: callers must provide *block_size*; we can't store it on the
        entry because it's a global constant.
        """
        # Block size comes from config; callers use `is_full_for(block_size)`.
        return False  # resolved at query time

    def is_full_for(self, block_size: int) -> bool:
        return self.num_filled >= block_size
