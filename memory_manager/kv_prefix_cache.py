"""
Prefix Hash Cache — MoonCake-style KVCache reuse via hash-chain matching.

Implements the token-block hashing scheme from MoonCake (FAST 2025, §3,
Figure 3) where each block's hash is derived from its own content AND
the hash of the preceding block, forming a deterministic hash-chain:

    hash₀ = H(tokens[0:block_size])
    hash₁ = H(hash₀ | tokens[block_size:2*block_size])
    hash₂ = H(hash₁ | tokens[2*block_size:3*block_size])
    ...

This chain property enables O(1) prefix lookups: given a new token sequence,
walk its hash chain until a mismatch is found; all preceding blocks are
guaranteed to be identical to the cached version and can be reused without
re-computation.

Cache tiers
-----------
- **Pinned** — system prompts, tool schemas (never evicted)
- **Hot** — blocks accessed above a configurable threshold
- **Cold** — remaining cached blocks (LRU evictable)

References
----------
- MoonCake §3, Figure 3   (KVCache pool with hash-based dedup)
- MoonCake §4.2, Table 1  (cache hit rates under LRU/LFU policies)
- MoonCake §6.2            (hot-spot replication to avoid transfer congestion)
- vLLM §4.4, Figure 10    (shared prefix for few-shot prompts)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Set, Tuple

from memory_manager.kv_eviction_policy import EvictionPolicy, make_policy


# ---------------------------------------------------------------------------
# Hash-chain computation
# ---------------------------------------------------------------------------

def compute_prefix_hashes(
    token_ids: List[int],
    block_size: int = 16,
) -> List[str]:
    """Compute the MoonCake-style hash-chain for a token sequence.

    Each block's hash = SHA-256(prev_hash || repr(tokens)).  The repr
    is deterministically repeatable: sorted, space-delimited integers.

    Returns a list of hex digests, one per block.  Only the first 16 hex
    characters are kept (64 bits of entropy — collision probability is
    ~1/2^63, negligible for cache purposes).

    Parameters
    ----------
    token_ids : list[int]
        The full token sequence to hash.
    block_size : int
        Tokens per block (must match the allocator config).

    Returns
    -------
    list[str]
        One 16-char hex hash per block, in order.
    """
    hashes: List[str] = []
    prev_hash = ""

    for i in range(0, len(token_ids), block_size):
        block_tokens = token_ids[i:i + block_size]
        # Deterministic representation: space-delimited integers
        block_repr = " ".join(str(t) for t in block_tokens)
        payload = f"{prev_hash}|{block_repr}"
        block_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        hashes.append(block_hash)
        prev_hash = block_hash

    return hashes


def compute_block_hash(
    token_block: List[int],
    prev_hash: str = "",
) -> str:
    """Compute a single block's prefix hash.

    Used when tokens arrive incrementally (e.g. autoregressive decoding).
    """
    block_repr = " ".join(str(t) for t in token_block)
    payload = f"{prev_hash}|{block_repr}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

class PrefixCacheEntry:
    """One entry in the prefix hash cache.

    Maps a block hash to one or more physical block IDs, tracking access
    metadata for eviction and hot-spot detection.

    Parameters
    ----------
    block_hash : str
        16-character hex digest from ``compute_prefix_hashes``.
    block_ids : list[int]
        Physical block IDs that hold this content.
    is_pinned : bool
        If True, never evict (system prompt, tool schema).
    """

    __slots__ = (
        "block_hash", "block_ids", "is_pinned",
        "first_seen", "last_access", "access_count",
    )

    def __init__(self, block_hash: str, block_ids: List[int],
                 is_pinned: bool = False):
        self.block_hash = block_hash
        self.block_ids = list(block_ids)
        self.is_pinned = is_pinned
        self.first_seen: float = time.monotonic()
        self.last_access: float = self.first_seen
        self.access_count: int = 0

    def touch(self):
        self.last_access = time.monotonic()
        self.access_count += 1

    def add_block_id(self, bid: int):
        if bid not in self.block_ids:
            self.block_ids.append(bid)

    def remove_block_id(self, bid: int):
        self.block_ids = [b for b in self.block_ids if b != bid]

    @property
    def is_hot(self) -> bool:
        """Hot block: accessed >= 10 times (MoonCake §4.2 threshold)."""
        return self.access_count >= 10

    @property
    def num_block_ids(self) -> int:
        return len(self.block_ids)

    def __repr__(self) -> str:
        pin = "🔒" if self.is_pinned else ""
        hot = "🔥" if self.is_hot else ""
        return (
            f"PrefixCacheEntry({self.block_hash}{pin}{hot}, "
            f"blocks={self.block_ids}, hits={self.access_count})"
        )


# ---------------------------------------------------------------------------
# Prefix Hash Cache
# ---------------------------------------------------------------------------

class PrefixHashCache:
    """MoonCake-style KV Cache prefix matching via hash-chains.

    Maintains a global mapping from block-hash → physical block IDs.
    New requests compute their hash-chain with ``compute_prefix_hashes``,
    then call ``find_longest_prefix`` to discover cache hits.

    Parameters
    ----------
    block_size : int
        Tokens per block (must match allocator config).
    max_entries : int
        Maximum number of hash→blocks entries.  When exceeded, the
        least-recently-accessed non-pinned entry is evicted.
    eviction_policy : EvictionPolicy | None
        Policy for selecting victims when the cache is full.
        Defaults to LRU.
    """

    def __init__(
        self,
        block_size: int = 16,
        max_entries: int = 100_000,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self.block_size = block_size
        self.max_entries = max_entries
        self._lock = threading.RLock()

        # hash → entry
        self._entries: Dict[str, PrefixCacheEntry] = {}
        # LRU order
        self._access_order: OrderedDict = OrderedDict()

        # Pinned hashes (never evict)
        self._pinned_hashes: Set[str] = set()

        # Eviction policy
        self._eviction = eviction_policy or make_policy("lru")

        # Stats
        self._total_lookups: int = 0
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_blocks_reused: int = 0

    # ------------------------------------------------------------------
    # Prefix matching (MoonCake §3, Figure 3)
    # ------------------------------------------------------------------

    def find_longest_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[int, List[str], List[int]]:
        """Find the longest prefix of *token_ids* that exists in the cache.

        Returns
        -------
        tuple[int, list[str], list[int]]
            ``(prefix_tokens, matched_hashes, matched_block_ids)``
            - *prefix_tokens* — number of tokens that can be reused
            - *matched_hashes* — hash-chain entries that matched
            - *matched_block_ids* — corresponding physical block IDs
        """
        hashes = compute_prefix_hashes(token_ids, self.block_size)
        matched_tokens = 0
        matched_hashes: List[str] = []
        matched_blocks: List[int] = []

        with self._lock:
            self._total_lookups += 1

            for i, h in enumerate(hashes):
                entry = self._entries.get(h)
                if entry is None:
                    break

                matched_tokens += self.block_size
                matched_hashes.append(h)
                # Use the first physical block ID for this hash
                if entry.block_ids:
                    matched_blocks.append(entry.block_ids[0])
                entry.touch()
                self._access_order.move_to_end(h)

            if matched_tokens > 0:
                self._total_hits += 1
                self._total_blocks_reused += len(matched_hashes)
            else:
                self._total_misses += 1

        return matched_tokens, matched_hashes, matched_blocks

    def find_prefix_block_count(
        self,
        token_ids: List[int],
    ) -> int:
        """Return how many blocks of the prefix can be reused.

        Faster than ``find_longest_prefix`` when only the block count is
        needed (skips the block-ID fetch).
        """
        hashes = compute_prefix_hashes(token_ids, self.block_size)
        count = 0

        with self._lock:
            for h in hashes:
                if h not in self._entries:
                    break
                count += 1

        return count

    # ------------------------------------------------------------------
    # Cache insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        block_hashes: List[str],
        block_ids: List[int],
        is_pinned: bool = False,
    ):
        """Insert newly-computed block hashes into the cache.

        Parameters
        ----------
        block_hashes : list[str]
            Hash-chain values (from ``compute_prefix_hashes``).
        block_ids : list[int]
            Corresponding physical block IDs.
        is_pinned : bool
            If True, these blocks will never be evicted.
        """
        assert len(block_hashes) == len(block_ids), (
            f"Mismatch: {len(block_hashes)} hashes vs {len(block_ids)} blocks"
        )

        with self._lock:
            # Evict if needed
            needed = len(block_hashes)
            current = len(self._entries)
            overflow = (current + needed) - self.max_entries
            if overflow > 0:
                self._evict_lru(overflow)

            for h, bid in zip(block_hashes, block_ids):
                if h in self._entries:
                    # Existing entry — add another physical block ID
                    self._entries[h].add_block_id(bid)
                    self._entries[h].touch()
                else:
                    entry = PrefixCacheEntry(h, [bid], is_pinned=is_pinned)
                    self._entries[h] = entry
                    if is_pinned:
                        self._pinned_hashes.add(h)

                self._access_order[h] = time.monotonic()
                self._access_order.move_to_end(h)

    def insert_range(
        self,
        token_ids: List[int],
        block_ids: List[int],
        is_pinned: bool = False,
    ) -> List[str]:
        """Compute hashes for *token_ids* and insert them.

        Convenience; calls ``compute_prefix_hashes`` then ``insert``.
        Returns the computed hash list.
        """
        hashes = compute_prefix_hashes(token_ids, self.block_size)
        self.insert(hashes, block_ids, is_pinned=is_pinned)
        return hashes

    # ------------------------------------------------------------------
    # Pinned blocks
    # ------------------------------------------------------------------

    def pin(self, block_hash: str):
        """Pin a block hash so it is never evicted."""
        with self._lock:
            self._pinned_hashes.add(block_hash)
            if block_hash in self._entries:
                self._entries[block_hash].is_pinned = True

    def unpin(self, block_hash: str):
        """Remove pin protection from a block hash."""
        with self._lock:
            self._pinned_hashes.discard(block_hash)
            if block_hash in self._entries:
                self._entries[block_hash].is_pinned = False

    def pin_group(self, group_hashes: List[str]):
        """Pin a group of block hashes (e.g. entire system prompt)."""
        for h in group_hashes:
            self.pin(h)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, block_hash: str):
        """Remove a block hash from the cache."""
        with self._lock:
            self._entries.pop(block_hash, None)
            self._access_order.pop(block_hash, None)
            self._pinned_hashes.discard(block_hash)

    def remove_block_id(self, physical_id: int):
        """Remove a physical block ID from all entries that reference it.

        Called when the allocator frees a block, to keep the prefix cache
        from dangling.
        """
        with self._lock:
            stale_hashes = []
            for h, entry in self._entries.items():
                entry.remove_block_id(physical_id)
                if not entry.block_ids:
                    stale_hashes.append(h)
            for h in stale_hashes:
                self.remove(h)

    # ------------------------------------------------------------------
    # Hot-spot detection (MoonCake §6.2)
    # ------------------------------------------------------------------

    def get_hot_blocks(self, threshold: int = 10) -> List[PrefixCacheEntry]:
        """Return entries whose access_count ≥ *threshold*.

        MoonCake §4.2 reports that >50% of cache blocks remain unused
        while certain blocks are accessed tens of thousands of times.
        These "hot" blocks should be replicated across nodes to avoid
        transfer congestion (MoonCake §6.2).
        """
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.access_count >= threshold
            ]

    def get_replication_candidates(
        self,
        top_n: int = 20,
    ) -> List[PrefixCacheEntry]:
        """Top-N most-accessed entries (for replication decisions)."""
        with self._lock:
            sorted_entries = sorted(
                self._entries.values(),
                key=lambda e: e.access_count,
                reverse=True,
            )
            return sorted_entries[:top_n]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, block_hash: str) -> Optional[PrefixCacheEntry]:
        """Return the cache entry for *block_hash*, or None."""
        with self._lock:
            entry = self._entries.get(block_hash)
            if entry is not None:
                entry.touch()
                self._access_order.move_to_end(block_hash)
            return entry

    def contains(self, block_hash: str) -> bool:
        return block_hash in self._entries

    def contains_prefix(self, token_ids: List[int]) -> bool:
        """True if at least one block of *token_ids* is cached."""
        return self.find_prefix_block_count(token_ids) > 0

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_lru(self, n: int):
        """Evict *n* least-recently-used non-pinned entries."""
        evicted = 0
        stale_keys: List[str] = []

        for key in list(self._access_order.keys()):
            if evicted >= n:
                break
            if key in self._pinned_hashes:
                continue
            stale_keys.append(key)
            evicted += 1

        for key in stale_keys:
            self._entries.pop(key, None)
            self._access_order.pop(key, None)

    def evict_to_capacity(self, target_capacity: int):
        """Evict until the cache has ≤ *target_capacity* entries."""
        with self._lock:
            excess = len(self._entries) - target_capacity
            if excess > 0:
                self._evict_lru(excess)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def pinned_count(self) -> int:
        return len(self._pinned_hashes)

    @property
    def hot_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_hot)

    @property
    def hit_rate(self) -> float:
        total = self._total_lookups
        return self._total_hits / max(total, 1)

    @property
    def block_reuse_rate(self) -> float:
        """Average blocks reused per hit lookup."""
        return self._total_blocks_reused / max(self._total_hits, 1)

    def stats(self) -> dict:
        """Return cache statistics as a dict."""
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "pinned_entries": len(self._pinned_hashes),
                "hot_entries": self.hot_count,
                "total_lookups": self._total_lookups,
                "hits": self._total_hits,
                "misses": self._total_misses,
                "hit_rate": round(self.hit_rate, 4),
                "blocks_reused": self._total_blocks_reused,
                "block_reuse_rate": round(self.block_reuse_rate, 2),
                "max_entries": self.max_entries,
                "block_size": self.block_size,
            }

    def reset_stats(self):
        """Zero out lookup counters (does not clear cache)."""
        with self._lock:
            self._total_lookups = 0
            self._total_hits = 0
            self._total_misses = 0
            self._total_blocks_reused = 0

    def clear(self):
        """Remove all entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._pinned_hashes.clear()
            self.reset_stats()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __contains__(self, block_hash: str) -> bool:
        return self.contains(block_hash)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def items(self):
        return self._entries.items()

    def __repr__(self) -> str:
        return (
            f"PrefixHashCache({self.size} entries, "
            f"{self.pinned_count} pinned, "
            f"hit_rate={self.hit_rate:.2%})"
        )
