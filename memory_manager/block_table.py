"""
Block Table Manager — per-request logical→physical block mapping.

Implements the block table translation layer from vLLM's PagedAttention
(SOSP 2023, §4.2–4.4):

* **BlockTable** — one request's ordered mapping of logical block indices to
  physical block IDs, with per-entry fill counts and COW flags.
* **BlockTableManager** — registry of all active block tables, providing
  prefix-sharing lookups, COW-aware writes, and bulk table operations.

This is the analog of OS virtual memory page tables, but for KV Cache:
logical blocks are the request's view of its sequence; physical blocks are
the actual GPU memory pages managed by ``KVBlockAllocator``.

References
----------
- vLLM §4.2, Figure 6  (block table translation)
- vLLM §4.4, Figure 8  (copy-on-write for parallel sampling)
- vLLM §4.5           (scheduling and preemption via swapping)
- MoonCake §3          (prefix hash chains for sharing across requests)
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set, Tuple

from memory_manager.kv_block import (
    BlockTableEntry,
)
from memory_manager.kv_block_allocator import KVBlockAllocator


# ---------------------------------------------------------------------------
# BlockTable — a single request's logical→physical mapping
# ---------------------------------------------------------------------------

class BlockTable:
    """Ordered mapping of logical block indices to physical block IDs.

    Each request owns one ``BlockTable``.  Logical indices are contiguous
    starting from 0; physical IDs are arbitrary pool indices from the
    allocator.  Entries track how many token slots are filled in each block,
    enabling efficient appends during autoregressive decoding.

    Parameters
    ----------
    request_id : str
        Owning request identifier.
    block_size : int
        Global block size in tokens.
    """

    def __init__(self, request_id: str, block_size: int):
        self.request_id = request_id
        self.block_size = block_size
        self._entries: Dict[int, BlockTableEntry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, logical_idx: int) -> bool:
        return logical_idx in self._entries

    def __getitem__(self, logical_idx: int) -> BlockTableEntry:
        return self._entries[logical_idx]

    def get(self, logical_idx: int) -> Optional[BlockTableEntry]:
        return self._entries.get(logical_idx)

    @property
    def last_logical_idx(self) -> int:
        """Highest logical index, or -1 if empty."""
        return max(self._entries) if self._entries else -1

    @property
    def num_blocks(self) -> int:
        return len(self._entries)

    @property
    def total_tokens(self) -> int:
        """Total tokens stored in this block table."""
        return sum(e.num_filled for e in self._entries.values())

    # ------------------------------------------------------------------
    # Physical block enumeration
    # ------------------------------------------------------------------

    def get_physical_blocks(self) -> List[int]:
        """Physical block IDs in logical order (for the attention kernel)."""
        return [
            self._entries[i].physical_id
            for i in sorted(self._entries)
        ]

    def get_physical_block_ids(self) -> Set[int]:
        """Set of all physical block IDs (unordered)."""
        return {e.physical_id for e in self._entries.values()}

    def get_block_table(self) -> List[int]:
        """Alias for ``get_physical_blocks`` — matches vLLM nomenclature."""
        return self.get_physical_blocks()

    # ------------------------------------------------------------------
    # Mutation — add / remove entries
    # ------------------------------------------------------------------

    def add_entry(self, logical_idx: int, physical_id: int,
                  num_filled: int = 0, is_cow: bool = False,
                  shared_from: str | None = None):
        """Insert or overwrite a block-table entry."""
        with self._lock:
            self._entries[logical_idx] = BlockTableEntry(
                logical_idx=logical_idx,
                physical_id=physical_id,
                num_filled=num_filled,
                is_cow=is_cow,
                shared_from=shared_from,
            )

    def append_blocks(self, physical_ids: List[int],
                      tokens_per_block: List[int] | None = None,
                      is_cow: bool = False) -> int:
        """Append a sequence of physical blocks to the end of the table.

        Returns the starting logical index of the appended run.
        """
        with self._lock:
            start_idx = self.last_logical_idx + 1
            for i, pid in enumerate(physical_ids):
                n_filled = tokens_per_block[i] if tokens_per_block else 0
                self._entries[start_idx + i] = BlockTableEntry(
                    logical_idx=start_idx + i,
                    physical_id=pid,
                    num_filled=n_filled,
                    is_cow=is_cow,
                )
            return start_idx

    def remove_entry(self, logical_idx: int) -> Optional[BlockTableEntry]:
        """Remove and return a block-table entry."""
        with self._lock:
            return self._entries.pop(logical_idx, None)

    def trim_from(self, start_logical_idx: int) -> List[BlockTableEntry]:
        """Remove all entries with logical index ≥ *start_logical_idx*.

        Used when rewinding a partially-generated response.
        Returns the removed entries (caller should free them).
        """
        with self._lock:
            removed = []
            for idx in sorted(self._entries):
                if idx >= start_logical_idx:
                    removed.append(self._entries.pop(idx))
            return removed

    # ------------------------------------------------------------------
    # Token-level operations (for autoregressive decoding)
    # ------------------------------------------------------------------

    def tokens_to_blocks(self, num_tokens: int) -> int:
        """How many blocks are needed to hold *num_tokens*?"""
        return max(1, (num_tokens + self.block_size - 1) // self.block_size)

    def append_tokens(self, num_new_tokens: int,
                      new_block_ids: List[int]) -> Tuple[int, int]:
        """Fill existing block slots, then allocate new blocks as needed.

        This is the core autoregressive append operation: each decoding step
        produces 1 token which fills the last block's remaining slot.  When
        the last block is full, a new block is allocated.

        Returns ``(blocks_consumed, tokens_spilled)`` — the number of new
        blocks consumed, and any remaining tokens that couldn't be placed.
        """
        with self._lock:
            if not self._entries:
                # First allocation — prefill path
                return 0, num_new_tokens

            last_idx = self.last_logical_idx
            last_entry = self._entries[last_idx]
            available = self.block_size - last_entry.num_filled

            if num_new_tokens <= available:
                # Fits in the last block
                last_entry.num_filled += num_new_tokens
                return 0, 0

            # Fill the last block, spill to new blocks
            last_entry.num_filled = self.block_size
            remaining = num_new_tokens - available

            blocks_needed = self.tokens_to_blocks(remaining)
            consumed = min(blocks_needed, len(new_block_ids))

            for i, pid in enumerate(new_block_ids[:consumed]):
                idx = last_idx + 1 + i
                fill = min(self.block_size, remaining)
                self._entries[idx] = BlockTableEntry(
                    logical_idx=idx,
                    physical_id=pid,
                    num_filled=fill,
                )
                remaining -= fill

            return consumed, remaining

    def last_block_has_room(self) -> bool:
        """True if the last block can accept at least one more token."""
        if not self._entries:
            return False
        return self._entries[self.last_logical_idx].num_filled < self.block_size

    # ------------------------------------------------------------------
    # COW helpers
    # ------------------------------------------------------------------

    def get_cow_entries(self) -> List[int]:
        """Logical indices of entries flagged for copy-on-write."""
        return [
            idx for idx, e in self._entries.items() if e.is_cow
        ]

    def clear_cow(self, logical_idx: int):
        """Clear the COW flag on a block-table entry."""
        if logical_idx in self._entries:
            self._entries[logical_idx].is_cow = False

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def entries(self):
        """Yield entries in logical order."""
        for idx in sorted(self._entries):
            yield self._entries[idx]

    def __iter__(self):
        return self.entries()

    def __repr__(self) -> str:
        blocks = ",".join(
            f"{i}→P{e.physical_id}" for i, e in
            sorted(self._entries.items())
        )
        return f"BlockTable({self.request_id}, [{blocks}])"


# ---------------------------------------------------------------------------
# BlockTableManager — registry of all active block tables
# ---------------------------------------------------------------------------

class BlockTableManager:
    """Central registry of per-request ``BlockTable`` instances.

    Provides prefix-sharing lookups, COW-aware writes, and bulk operations
    across all active request block tables.

    Parameters
    ----------
    allocator : KVBlockAllocator
        The physical block pool.
    block_size : int
        Global block size in tokens.
    """

    def __init__(self, allocator: KVBlockAllocator, block_size: int):
        self._allocator = allocator
        self.block_size = block_size
        self._tables: Dict[str, BlockTable] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_table(self, request_id: str) -> BlockTable:
        """Create (or return existing) block table for *request_id*."""
        with self._lock:
            if request_id not in self._tables:
                self._tables[request_id] = BlockTable(request_id, self.block_size)
            return self._tables[request_id]

    def get_table(self, request_id: str) -> Optional[BlockTable]:
        """Return the block table for *request_id*, or None."""
        return self._tables.get(request_id)

    def remove_table(self, request_id: str) -> Optional[BlockTable]:
        """Remove and return the block table for *request_id*.

        Does NOT free physical blocks — the caller must call
        ``allocator.free(request_id)`` separately.
        """
        with self._lock:
            return self._tables.pop(request_id, None)

    def has_table(self, request_id: str) -> bool:
        return request_id in self._tables

    # ------------------------------------------------------------------
    # Prefix sharing (MoonCake §3, vLLM §4.4)
    # ------------------------------------------------------------------

    def find_shared_prefix(self, source_id: str,
                           target_id: str) -> int:
        """Find the longest common prefix (in tokens) between two requests.

        Walks both block tables in logical order comparing physical block
        IDs.  Returns the number of *tokens* shared (not blocks).

        This is O(min(N_src, N_tgt)) — fast enough for the common case
        where system prompts are shared across many requests.
        """
        src = self._tables.get(source_id)
        tgt = self._tables.get(target_id)
        if not src or not tgt:
            return 0

        shared_tokens = 0
        idx = 0
        while idx in src._entries and idx in tgt._entries:
            if src._entries[idx].physical_id != tgt._entries[idx].physical_id:
                break
            shared_tokens += src._entries[idx].num_filled
            idx += 1
        return shared_tokens

    def share_prefix(self, source_id: str, target_id: str,
                     prefix_blocks: int):
        """Share *prefix_blocks* blocks from *source_id* with *target_id*.

        The target's block-table entries for the first *prefix_blocks*
        logical indices are set to point to the same physical blocks as
        the source.  Reference counts on those physical blocks are
        incremented.  The last shared block is marked COW.

        This implements vLLM §4.4, Figure 8: parallel sampling share.
        """
        with self._lock:
            src = self._tables[source_id]
            tgt = self._tables.setdefault(
                target_id,
                BlockTable(target_id, self.block_size),
            )

            for i in range(prefix_blocks):
                src_entry = src._entries[i]
                phys_id = src_entry.physical_id

                # Increment ref count on the physical block
                self._allocator.increment_ref(phys_id)

                # Target entry
                is_cow = (i == prefix_blocks - 1)
                tgt._entries[i] = BlockTableEntry(
                    logical_idx=i,
                    physical_id=phys_id,
                    num_filled=src_entry.num_filled,
                    is_cow=is_cow,
                    shared_from=source_id,
                )

    # ------------------------------------------------------------------
    # COW-aware write (vLLM §4.3)
    # ------------------------------------------------------------------

    def ensure_writable(self, request_id: str,
                        logical_idx: int) -> int:
        """Ensure the block at *logical_idx* is safe to write.

        If the block is shared (COW flag set or ref_count > 1), clones it
        via the allocator.  Returns the (possibly new) physical block ID.

        The caller should use the returned physical_id for the write and
        update its block table entry.
        """
        table = self._tables.get(request_id)
        if not table:
            raise KeyError(f"No block table for request '{request_id}'")

        entry = table[logical_idx]
        old_phys = entry.physical_id

        # Check if COW is needed
        block = self._allocator.get_block(old_phys)
        if block.ref_count <= 1 and not entry.is_cow:
            return old_phys

        # Clone
        new_phys = self._allocator.clone_block(request_id, old_phys)

        # Update entry
        entry.physical_id = new_phys
        entry.is_cow = False
        entry.shared_from = None

        return new_phys

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def get_physical_blocks(self, request_id: str) -> List[int]:
        """Get ordered physical block IDs for a request (for kernel)."""
        table = self._tables.get(request_id)
        return table.get_physical_blocks() if table else []

    def get_total_tokens(self, request_id: str) -> int:
        """Total tokens stored for a request."""
        table = self._tables.get(request_id)
        return table.total_tokens if table else 0

    def active_requests(self) -> int:
        """Number of requests with active block tables."""
        return len(self._tables)

    # ------------------------------------------------------------------
    # Debug / inspection
    # ------------------------------------------------------------------

    def dump_table(self, request_id: str) -> str:
        """Human-readable block table dump."""
        table = self._tables.get(request_id)
        if not table:
            return f"BlockTable({request_id}): (empty)"

        lines = [f"BlockTable({request_id}): {table.num_blocks} entries"]
        for idx in sorted(table._entries):
            e = table._entries[idx]
            cow = " [COW]" if e.is_cow else ""
            shared = f" from={e.shared_from}" if e.shared_from else ""
            lines.append(
                f"  L{e.logical_idx:03d} → P{e.physical_id:05d}  "
                f"filled={e.num_filled}/{self.block_size}{cow}{shared}"
            )
        return "\n".join(lines)
