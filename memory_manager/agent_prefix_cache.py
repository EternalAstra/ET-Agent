"""
Agent-aware prefix caching strategies.

Bridges the generic ``PrefixHashCache`` to the Hermes/ET-Agent conversation
lifecycle.  Provides caching primitives tailored to agent workloads:

* **System prompts** — cached once per session (or globally), reused every turn
* **Tool schemas** — cached per toolset, shared across sessions
* **Session prefixes** — multi-turn conversation history reuse
* **Cross-session** — similar prompts across different users/sessions

All strategies operate on token IDs and produce hash-chain entries that
can be handed to ``BlockTableManager.share_prefix()`` for zero-compute
KV Cache reuse.

Integration point
-----------------
Called from ``agent/memory_hooks.py`` (Phase 5) at session start and
before each LLM call.  The hooks:
1. Tokenize the system prompt → compute hashes → ``cache_system_prompt()``
2. Tokenize tool schemas → compute hashes → ``cache_tool_schemas()``
3. Before each turn → ``find_reusable_prefix()`` → ``share_prefix()``
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set, Tuple

from memory_manager.kv_prefix_cache import PrefixHashCache, compute_prefix_hashes


# ---------------------------------------------------------------------------
# Agent prefix cache
# ---------------------------------------------------------------------------

class AgentPrefixCache:
    """Agent lifecycle-aware cache wrapper around ``PrefixHashCache``.

    Maintains separate logical caches for system prompts, tool schemas,
    and per-session conversation prefixes.  All are backed by the same
    underlying hash→block store.

    Parameters
    ----------
    prefix_cache : PrefixHashCache
        The shared hash→block store.
    """

    def __init__(self, prefix_cache: PrefixHashCache):
        self._cache = prefix_cache
        self._lock = threading.RLock()

        # ── system prompts ──
        # key → (token_ids, block_hashes)
        self._system_prompts: Dict[str, Tuple[List[int], List[str]]] = {}

        # ── tool schemas ──
        # tool_name → (token_ids, block_hashes)
        self._tool_schemas: Dict[str, Tuple[List[int], List[str]]] = {}

        # ── session prefixes ──
        # session_id → accumulated token_ids
        self._session_tokens: Dict[str, List[int]] = {}

        # ── stats ──
        self._prefix_hits: int = 0
        self._prefix_misses: int = 0
        self._tokens_saved: int = 0

    # ------------------------------------------------------------------
    # System prompt caching
    # ------------------------------------------------------------------

    def cache_system_prompt(
        self,
        key: str,
        token_ids: List[int],
        block_ids: List[int],
        global_scope: bool = False,
    ):
        """Cache a system prompt's token→KV mapping.

        Parameters
        ----------
        key : str
            Identifier — typically ``"default"`` or a personality name.
        token_ids : list[int]
            Full tokenized system prompt.
        block_ids : list[int]
            Physical block IDs holding the KV Cache (from allocator).
        global_scope : bool
            If True, cache globally (shared across all profiles).
            If False, cache per-profile.
        """
        hashes = compute_prefix_hashes(
            token_ids, self._cache.block_size
        )

        with self._lock:
            existing = self._system_prompts.get(key)
            if existing is not None:
                # Remove old blocks from prefix cache
                for h in existing[1]:
                    self._cache.remove(h)

            self._system_prompts[key] = (list(token_ids), hashes)

            # Insert into hash cache as pinned (never evicted)
            self._cache.insert(hashes, block_ids, is_pinned=True)

    def get_system_prompt(
        self,
        key: str = "default",
    ) -> Optional[Tuple[List[int], List[str]]]:
        """Return cached system prompt hashes and tokens, or None."""
        with self._lock:
            return self._system_prompts.get(key)

    def all_system_prompts(self) -> List[str]:
        """List all cached system prompt keys."""
        with self._lock:
            return list(self._system_prompts)

    # ------------------------------------------------------------------
    # Tool schema caching
    # ------------------------------------------------------------------

    def cache_tool_schema(
        self,
        tool_name: str,
        tool_token_ids: List[int],
        block_ids: List[int],
    ):
        """Cache a tool definition's KV mapping.

        Tool schemas are pinned because they are reused across all
        sessions that enable that toolset.
        """
        hashes = compute_prefix_hashes(
            tool_token_ids, self._cache.block_size
        )

        with self._lock:
            existing = self._tool_schemas.get(tool_name)
            if existing is not None:
                for h in existing[1]:
                    self._cache.remove(h)

            self._tool_schemas[tool_name] = (list(tool_token_ids), hashes)
            self._cache.insert(hashes, block_ids, is_pinned=True)

    def get_tool_schema(
        self,
        tool_name: str,
    ) -> Optional[Tuple[List[int], List[str]]]:
        """Return cached tool schema hashes and tokens, or None."""
        with self._lock:
            return self._tool_schemas.get(tool_name)

    def cache_tool_schemas_batch(
        self,
        tools: Dict[str, Tuple[List[int], List[int]]],
    ):
        """Cache multiple tool schemas.

        Parameters
        ----------
        tools : dict[str, tuple[list[int], list[int]]]
            Mapping of tool_name → (token_ids, block_ids).
        """
        for name, (tokens, bids) in tools.items():
            self.cache_tool_schema(name, tokens, bids)

    def all_tool_schemas(self) -> List[str]:
        with self._lock:
            return list(self._tool_schemas)

    # ------------------------------------------------------------------
    # Session prefix tracking
    # ------------------------------------------------------------------

    def track_session(
        self,
        session_id: str,
        token_ids: Optional[List[int]] = None,
    ):
        """Register a session for prefix tracking.

        As the conversation progresses, call ``extend_session()`` after
        each turn to accumulate tokens.
        """
        with self._lock:
            self._session_tokens[session_id] = list(token_ids or [])

    def extend_session(self, session_id: str, new_token_ids: List[int]):
        """Append new tokens to a session's accumulated prefix."""
        with self._lock:
            if session_id in self._session_tokens:
                self._session_tokens[session_id].extend(new_token_ids)

    def get_session_prefix(
        self,
        session_id: str,
    ) -> Optional[List[int]]:
        """Return all accumulated tokens for a session."""
        with self._lock:
            return self._session_tokens.get(session_id)

    def end_session(self, session_id: str):
        """Remove a session's prefix tracking (keep the cached blocks)."""
        with self._lock:
            self._session_tokens.pop(session_id, None)

    # ------------------------------------------------------------------
    # Prefix reuse
    # ------------------------------------------------------------------

    def find_reusable_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[int, List[str], List[int]]:
        """Find the longest reusable prefix for a new message.

        Returns ``(tokens_reused, matched_hashes, matched_block_ids)``.

        The caller should then call
        ``BlockTableManager.share_prefix(..., blocks_reused)`` to wire up
        the physical block sharing and avoid recomputation.
        """
        prefix_tokens, hashes, bids = self._cache.find_longest_prefix(
            token_ids
        )

        with self._lock:
            self._tokens_saved += prefix_tokens
            if prefix_tokens > 0:
                self._prefix_hits += 1
            else:
                self._prefix_misses += 1

        return prefix_tokens, hashes, bids

    def estimate_reuse_ratio(self, token_ids: List[int]) -> float:
        """Fraction of *token_ids* that can be reused from cache."""
        prefix_tokens, _, _ = self._cache.find_longest_prefix(token_ids)
        return prefix_tokens / max(len(token_ids), 1)

    # ------------------------------------------------------------------
    # Session-aware prefix reuse
    # ------------------------------------------------------------------

    def find_session_prefix(
        self,
        session_id: str,
        new_message_tokens: List[int],
    ) -> Tuple[int, List[str], List[int]]:
        """Find reusable prefix combining session history + new message.

        This is the primary Agent-scenario lookup: the full prompt for
        a turn is [system_prompt ... history ... new_message].  The system
        prompt and history are already cached; we look up the combined
        prefix in the hash cache.
        """
        session_tokens = self._session_tokens.get(session_id, [])
        combined = session_tokens + list(new_message_tokens)
        return self.find_reusable_prefix(combined)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def prefix_hit_rate(self) -> float:
        total = self._prefix_hits + self._prefix_misses
        return self._prefix_hits / max(total, 1)

    @property
    def total_tokens_saved(self) -> int:
        return self._tokens_saved

    def stats(self) -> dict:
        """Return agent-cache statistics."""
        with self._lock:
            return {
                "system_prompts_cached": len(self._system_prompts),
                "tool_schemas_cached": len(self._tool_schemas),
                "active_sessions": len(self._session_tokens),
                "prefix_hits": self._prefix_hits,
                "prefix_misses": self._prefix_misses,
                "prefix_hit_rate": round(self.prefix_hit_rate, 4),
                "tokens_saved": self._tokens_saved,
                **self._cache.stats(),
            }

    def reset_stats(self):
        with self._lock:
            self._prefix_hits = 0
            self._prefix_misses = 0
            self._tokens_saved = 0
            self._cache.reset_stats()

    def clear(self):
        """Reset everything (cache + tracking)."""
        with self._lock:
            self._system_prompts.clear()
            self._tool_schemas.clear()
            self._session_tokens.clear()
            self._prefix_hits = 0
            self._prefix_misses = 0
            self._tokens_saved = 0
            self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"AgentPrefixCache(sp={len(self._system_prompts)}, "
            f"tools={len(self._tool_schemas)}, "
            f"sessions={len(self._session_tokens)}, "
            f"hit_rate={self.prefix_hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# Utility: estimate savings for a conversation template
# ---------------------------------------------------------------------------

def estimate_agent_savings(
    agent_cache: AgentPrefixCache,
    history_tokens: int,
    num_sessions: int,
) -> dict:
    """Estimate token savings from prefix caching in a multi-session scenario.

    Parameters
    ----------
    agent_cache : AgentPrefixCache
        Populated agent cache (system prompt + tool schemas).
    history_tokens : int
        Average conversation history length per turn.
    num_sessions : int
        Number of concurrent sessions.

    Returns
    -------
    dict
        Estimated savings breakdown.
    """
    sp_tokens = 0
    for key, (tokens, _) in agent_cache._system_prompts.items():
        sp_tokens += len(tokens)

    tool_tokens = 0
    for name, (tokens, _) in agent_cache._tool_schemas.items():
        tool_tokens += len(tokens)

    # Per session, per turn: only the new user message needs compute
    tokens_per_turn = sp_tokens + tool_tokens + history_tokens
    reusable_per_turn = sp_tokens + tool_tokens  # system + tools always cached

    return {
        "system_prompt_tokens": sp_tokens,
        "tool_schema_tokens": tool_tokens,
        "tokens_per_turn": tokens_per_turn,
        "reusable_per_turn": reusable_per_turn,
        "savings_per_turn_pct": round(
            100 * reusable_per_turn / max(tokens_per_turn, 1), 1
        ),
        "tokens_saved_across_sessions": reusable_per_turn * num_sessions,
    }
