"""
Prompt Deduplication Engine — static & dynamic context de-duplication.

Handles the "static redundancy" problem in agent inference:
every turn includes the full system prompt (often 5,000+ tokens) and
tool definitions (2,000+ tokens).  These are identical across turns
within a session, so they can be:

1. **Sent only once** — the first turn carries the full system prompt;
   subsequent turns rely on the KV Cache prefix.

2. **Inter-session deduplicated** — when two sessions share the same
   system prompt (e.g. default personality), only one copy of its KV
   Cache is stored.

3. **Tool schema de-duplicated** — shared across toolset configurations,
   with per-session overrides handled via COW.

This module works with ``PrefixHashCache`` (Phase 2) to detect which
tokens are already cached and can be safely elided from the API request.

Integration
-----------
Called from ``agent/memory_hooks.py`` (Phase 5) before each LLM call.
The hook:
1. Tokenize the complete API message list
2. For each message, check ``PrefixHashCache.contains_prefix()``
3. Strip messages whose tokens are fully cached
4. Build a minimal API request with only uncached content
"""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def hash_content(text: str) -> str:
    """Return a fast deterministic hash for text content dedup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Message categorizer
# ---------------------------------------------------------------------------

class MessageCategory:
    """Classification of a message for dedup decisions."""

    SYSTEM = "system"
    SYSTEM_REPEATED = "system_repeated"   # Same system prompt in later turn
    TOOL_DEF = "tool_definition"          # Tool schema message (not a result)
    TOOL_RESULT = "tool_result"           # Tool execution output
    USER = "user"
    ASSISTANT = "assistant"
    HISTORY = "history"                   # Old conversation turns (above threshold)


# ---------------------------------------------------------------------------
# Dedup result
# ---------------------------------------------------------------------------

class DedupResult:
    """Result of deduplicating a message list.

    Attributes
    ----------
    kept_messages : list[dict]
        Messages that need to be sent to the LLM.
    dropped_count : int
        Number of messages elided.
    dropped_tokens_est : int
        Estimated tokens saved by elision.
    cached_prefix_tokens : int
        Tokens that are already in the KV Cache (prefix reuse).
    """

    __slots__ = (
        "kept_messages", "dropped_count", "dropped_tokens_est",
        "cached_prefix_tokens",
    )

    def __init__(self, kept: List[Dict], dropped: int, tokens: int,
                 cached: int = 0):
        self.kept_messages = kept
        self.dropped_count = dropped
        self.dropped_tokens_est = tokens
        self.cached_prefix_tokens = cached

    @property
    def savings_ratio(self) -> float:
        total = len(self.kept_messages) + self.dropped_count
        return self.dropped_count / max(total, 1)

    def __repr__(self) -> str:
        return (
            f"DedupResult(kept={len(self.kept_messages)}, "
            f"dropped={self.dropped_count}, "
            f"saved={self.dropped_tokens_est}tok, "
            f"cached_prefix={self.cached_prefix_tokens}tok)"
        )


# ---------------------------------------------------------------------------
# Prompt Deduplicator
# ---------------------------------------------------------------------------

class PromptDeduplicator:
    """Static + dynamic context de-duplicator.

    Removes redundant messages from the prompt before each LLM call:
    - System prompts that haven't changed since the last turn
    - Tool definitions that match a previously cached set
    - Consecutive duplicate messages (e.g. retry loops)

    Parameters
    ----------
    prefix_cache : PrefixHashCache | None
        If provided, used to check which message prefixes are already
        cached in the KV store, enabling deeper elision.
    enable_tool_dedup : bool
        Whether to deduplicate tool definitions (default: True).
    """

    def __init__(
        self,
        prefix_cache=None,  # Optional[PrefixHashCache]
        enable_tool_dedup: bool = True,
    ):
        self._prefix_cache = prefix_cache
        self._enable_tool_dedup = enable_tool_dedup
        self._lock = threading.RLock()

        # Session-level state
        self._last_system_hash: Optional[str] = None
        self._last_tool_hashes: Set[str] = set()
        self._last_message_hashes: List[str] = []

        # Stats
        self._total_messages_dropped: int = 0
        self._total_tokens_saved: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        messages: List[Dict],
        session_id: str = "",
        force_full: bool = False,
    ) -> Tuple[List[Dict], DedupResult]:
        """Deduplicate a message list for an LLM call.

        Parameters
        ----------
        messages : list[dict]
            Full message list (system + history + user).
        session_id : str
            Session identifier (for session-scoped dedup).
        force_full : bool
            If True, send all messages regardless of cache state
            (e.g. when KV Cache was invalidated by compression).

        Returns
        -------
        tuple[list[dict], DedupResult]
            ``(deduped_messages, result_metadata)``
        """
        if force_full or not messages:
            return messages, DedupResult(messages, 0, 0, 0)

        original_count = len(messages)
        original_tokens = sum(len(str(m).split()) for m in messages)

        with self._lock:
            deduped = []
            dropped = 0
            cached_prefix = 0

            prev_hash = None

            for i, msg in enumerate(messages):
                role = msg.get("role", "")
                content = str(msg.get("content", ""))

                # ── System prompt dedup ──
                if role == "system":
                    content_hash = hash_content(content)
                    if content_hash == self._last_system_hash and i == 0:
                        # System prompt unchanged → elide (KV Cache has it)
                        dropped += 1
                        continue
                    self._last_system_hash = content_hash

                # ── Consecutive duplicate dedup ──
                if role == "assistant" and content:
                    content_hash = hash_content(content[:200])
                    if content_hash == prev_hash:
                        # Repeated assistant message → skip
                        dropped += 1
                        continue
                    prev_hash = content_hash
                else:
                    prev_hash = None

                # ── Tool definition dedup ──
                if role == "assistant" and "tool_calls" in msg:
                    tool_hash = hash_content(str(msg.get("tool_calls", "")))
                    if self._enable_tool_dedup and tool_hash in self._last_tool_hashes:
                        dropped += 1
                        continue
                    self._last_tool_hashes.add(tool_hash)

                # ── Prefix cache check ──
                if self._prefix_cache is not None and role in ("system", "assistant"):
                    # This is a no-op for now; in Phase 5 the tokenizer
                    # will check prefix match and compute savings
                    pass

                deduped.append(msg)

            dropped_tokens = original_tokens - sum(
                len(str(m).split()) for m in deduped
            )

            result = DedupResult(
                kept=deduped,
                dropped=original_count - len(deduped),
                tokens=dropped_tokens,
                cached=cached_prefix,
            )

            self._total_messages_dropped += result.dropped_count
            self._total_tokens_saved += result.dropped_tokens_est

            return deduped, result

    def deduplicate_tools(
        self,
        tool_definitions: List[Dict],
        toolset_name: str = "",
    ) -> Tuple[List[Dict], int]:
        """Deduplicate tool definitions.

        Tools that haven't changed since the last call are elided.
        Returns ``(kept_definitions, tokens_saved)``.
        """
        if not self._enable_tool_dedup:
            return tool_definitions, 0

        with self._lock:
            current_hash = hash_content(str(tool_definitions))
            if current_hash in self._last_tool_hashes:
                return [], sum(
                    len(str(t).split()) for t in tool_definitions
                )

            self._last_tool_hashes.add(current_hash)
            return tool_definitions, 0

    @staticmethod
    def compress_tool_results(
        messages: List[Dict],
        max_result_chars: int = 2000,
    ) -> List[Dict]:
        """Truncate excessively long tool results.

        Tool outputs can be very large (e.g. web page dumps).  This
        truncates them to a reasonable limit while preserving key
        signal at the beginning.
        """
        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                if len(content) > max_result_chars:
                    msg = dict(msg)
                    msg["content"] = (
                        content[:max_result_chars]
                        + f"\n... [truncated: {len(content) - max_result_chars} chars]"
                    )
            result.append(msg)
        return result

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_session_state(self):
        """Clear session-level dedup state (call on session reset)."""
        with self._lock:
            self._last_system_hash = None
            self._last_tool_hashes.clear()
            self._last_message_hashes.clear()

    def reset_stats(self):
        with self._lock:
            self._total_messages_dropped = 0
            self._total_tokens_saved = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_messages_dropped": self._total_messages_dropped,
                "total_tokens_saved": self._total_tokens_saved,
                "system_prompt_cached": self._last_system_hash is not None,
                "tool_sets_cached": len(self._last_tool_hashes),
            }

    # ------------------------------------------------------------------
    # Utility: estimate savings
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_session_savings(
        system_prompt_tokens: int,
        tool_tokens: int,
        turns_per_session: int,
    ) -> dict:
        """Estimate per-session savings from prompt deduplication.

        Parameters
        ----------
        system_prompt_tokens : int
            Token count of the system prompt.
        tool_tokens : int
            Token count of tool definitions.
        turns_per_session : int
            Average turns per session.

        Returns
        -------
        dict
            Savings breakdown.
        """
        static_per_turn = system_prompt_tokens + tool_tokens
        total_without_dedup = static_per_turn * turns_per_session
        total_with_dedup = static_per_turn  # Only sent once
        saved = total_without_dedup - total_with_dedup

        return {
            "static_tokens_per_turn": static_per_turn,
            "total_without_dedup": total_without_dedup,
            "total_with_dedup": total_with_dedup,
            "tokens_saved": saved,
            "savings_pct": round(100 * saved / max(total_without_dedup, 1), 1),
        }

    def __repr__(self) -> str:
        return (
            f"PromptDeduplicator(dropped={self._total_messages_dropped}, "
            f"saved={self._total_tokens_saved}tok)"
        )
