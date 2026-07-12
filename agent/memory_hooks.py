"""
Agent Memory Hooks — wire the memory manager into the Agent core loop.

This module bridges the ``memory_manager`` package (KV Block Allocator,
Prefix Cache, Hierarchical Store, Lifecycle Tracker, Context Compressor)
with the Hermes/ET-Agent conversation lifecycle.

It follows the existing hermes plugin hook convention: each hook is a
callable that receives a context dict and returns optional data.  The
hooks are registered on the ``AIAgent`` instance and invoked at key
points in ``conversation_loop.py``.

Integration points
------------------
1. **Session start** — init KV blocks, cache system prompt prefix
2. **Pre-LLM call** — check prefix cache, dedup, track PREFILL phase
3. **Post-LLM call** — track DECODING→TOOL_CALL transition
4. **Tool execution** — lifecycle notification, demote on wait
5. **Tool result** — pre-fetch promotion on resume
6. **Context compression** — fire ACON compressor when over threshold
7. **Session end** — free blocks, archive to SSD

Phases 1–4 wired together
--------------------------
::

    Session start
      ├── KVBlockAllocator: allocate system-prompt blocks
      ├── BlockTableManager: create session block table
      ├── PrefixHashCache: cache system prompt + tool schemas
      ├── HierarchicalKVStore: register blocks on GPU
      └── LifecycleAwareKVManager: register request (PREFILL)

    Pre-LLM call
      ├── PromptDeduplicator: elide cached system/tool messages
      ├── PrefixHashCache: find reusable prefix, share blocks
      ├── KVBlockAllocator: allocate blocks for new messages
      └── LifecycleAwareKVManager: on_phase_change → PREFILL

    Post-LLM call (assistant response)
      └── LifecycleAwareKVManager: on_phase_change → DECODING
         └── if tool_calls: on_phase_change → TOOL_CALL
            └── HierarchicalKVStore: demote_blocks (after 30s delay)

    Tool result arrives
      ├── HierarchicalKVStore: promote_blocks (prefetch back to GPU)
      └── LifecycleAwareKVManager: on_phase_change → PREFILL

    Context overflow (token count > threshold)
      └── ContextCompressor: compress_history() → structured summary
         ├── KVBlockAllocator: free compressed-out blocks
         └── BlockTableManager: update block table

    Session end
      ├── KVBlockAllocator: free all blocks
      ├── BlockTableManager: remove table
      ├── HierarchicalKVStore: archive to SSD
      └── LifecycleAwareKVManager: on_phase_change → COMPLETED
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager
from memory_manager.config import MemoryConfig
from memory_manager.kv_prefix_cache import PrefixHashCache
from memory_manager.agent_prefix_cache import AgentPrefixCache
from memory_manager.kv_eviction_policy import make_policy
from memory_manager.kv_lifecycle_tracker import (
    AgentPhase, LifecycleAwareKVManager, LifecycleTiming, RequestLifecycle,
)
from memory_manager.kv_hierarchical_store import HierarchicalKVStore
from memory_manager.context_compressor import (
    ContextCompressor, CompressionMode, CompressionThresholds, CompressionResult,
)
from memory_manager.prompt_deduplicator import PromptDeduplicator
from memory_manager.tool_schema_compressor import ToolSchemaCompressor
from memory_manager.kv_prefix_cache import compute_prefix_hashes as _compute_prefix_hashes
from memory_manager.kv_block import StorageTier as _StorageTier


# ---------------------------------------------------------------------------
# AgentMemoryManager — top-level facade
# ---------------------------------------------------------------------------

class AgentMemoryManager:
    """Top-level facade that owns all Phase 1–4 memory subsystems and wires
    them into the Agent's lifecycle hooks.

    This is the single entry point referenced from ``AIAgent.__init__``.
    It creates and connects:

    - ``KVBlockAllocator`` + ``BlockTableManager``  (Phase 1)
    - ``PrefixHashCache`` + ``AgentPrefixCache``     (Phase 2)
    - ``HierarchicalKVStore`` + ``LifecycleAwareKVManager`` (Phase 3)
    - ``ContextCompressor`` + ``PromptDeduplicator`` + ``ToolSchemaCompressor`` (Phase 4)

    Parameters
    ----------
    config : MemoryConfig | None
        Tuning parameters.  If None, defaults to a Qwen2.5-7B profile
        with 80 GB GPU, 512 GB CPU, 2 TB SSD.
    model_name : str
        Used to look up model-specific KV sizing.
    enable_stats : bool
        Whether to collect detailed per-turn statistics.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        model_name: str = "qwen2.5-7b",
        enable_stats: bool = True,
    ):
        self._config = config or MemoryConfig.for_model(model_name)
        self._model_name = model_name
        self._enable_stats = enable_stats
        self._lock = threading.RLock()

        # ── Phase 1: Block management ──
        self.allocator = KVBlockAllocator(self._config)
        self.block_tables = BlockTableManager(
            self.allocator, self._config.block_size
        )

        # ── Phase 2: Prefix caching ──
        self.prefix_cache = PrefixHashCache(
            block_size=self._config.block_size,
            max_entries=100_000,
            eviction_policy=make_policy("agent_aware"),
        )
        self.agent_cache = AgentPrefixCache(self.prefix_cache)

        # ── Phase 3: Hierarchical storage ──
        self.hierarchical_store = HierarchicalKVStore(
            self._config,
            self.allocator,
            eviction_policy=make_policy("tiered_lru"),
        )

        self.lifecycle = LifecycleAwareKVManager(
            timing=LifecycleTiming(
                tool_call_gpu_to_cpu_s=1.0,   # fast demote for demo
                tool_call_cpu_to_ssd_s=-1,    # disable SSD for local
            ),
            on_demote=self._on_demote,
            on_promote=self._on_promote,
            on_evict=self._on_evict,
        )

        # ── Phase 4: Context compression ──
        self.compressor = ContextCompressor(
            thresholds=CompressionThresholds.for_agent_scenario(),
            mode=CompressionMode.UT,
        )
        self.deduplicator = PromptDeduplicator(
            prefix_cache=self.prefix_cache,
        )
        self.tool_compressor = ToolSchemaCompressor(
            max_total_tokens=4000,
        )

        # ── Background scan timer ──
        self._scan_timer: Optional[threading.Timer] = None
        self._scan_interval_s: float = 5.0
        self._start_scan_timer()

        # ── Stats ──
        self._session_count: int = 0
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def on_session_start(
        self,
        session_id: str,
        system_prompt_tokens: List[int] | None = None,
        tool_definitions: List[Dict] | None = None,
    ):
        """Initialize memory resources for a new agent session.

        Called from ``AIAgent.__init__`` or ``conversation_loop.py``
        plugin hook ``on_session_start``.

        Parameters
        ----------
        session_id : str
            Unique session identifier.
        system_prompt_tokens : list[int] | None
            Tokenized system prompt (if available at session start).
            When provided, the prefix cache is seeded with it.
        tool_definitions : list[dict] | None
            Tool schema definitions to cache.
        """
        with self._lock:
            self._session_count += 1

            # Register the session in the lifecycle tracker
            self.lifecycle.register(
                request_id=session_id,
                session_id=session_id,
                phase=AgentPhase.PREFILL,
            )

            # Create a block table for this session
            self.block_tables.create_table(session_id)

            # Cache system prompt (if tokenized)
            if system_prompt_tokens:
                sp_blocks = self.allocator.allocate(
                    session_id, len(system_prompt_tokens), group_id=session_id,
                )
                self.hierarchical_store.register_blocks(sp_blocks, tier=_StorageTier.GPU)
                self.agent_cache.cache_system_prompt(
                    "default", system_prompt_tokens, sp_blocks,
                )
                self.lifecycle.link_blocks(session_id, sp_blocks)

                # Pin system-prompt blocks (never evict)
                self.allocator.pin_blocks(f"sp-{session_id}", sp_blocks)

            # Cache tool schemas (if provided)
            if tool_definitions:
                for td in tool_definitions:
                    name = td.get("function", {}).get("name", "")
                    # Tokenize the tool schema (simplified: use description as proxy)
                    desc = td.get("function", {}).get("description", "")
                    tool_tokens = [ord(c) for c in desc[:500]]  # placeholder tokenization
                    if tool_tokens:
                        tool_blocks = self.allocator.allocate(
                            f"tool-{name}", len(tool_tokens),
                        )
                        self.hierarchical_store.register_blocks(tool_blocks)
                        self.agent_cache.cache_tool_schema(name, tool_tokens, tool_blocks)
                        self.allocator.pin_blocks(f"tool-{name}", tool_blocks)

    def on_session_end(self, session_id: str):
        """Release all memory resources for a session.

        Called from plugin hook ``on_session_end`` or when the conversation
        loop terminates.
        """
        with self._lock:
            # Mark completed in lifecycle tracker
            self.lifecycle.on_phase_change(session_id, AgentPhase.COMPLETED)

            # Demote to SSD for archival
            blocks = self.allocator.get_request_blocks(session_id)
            if blocks:
                self.hierarchical_store.demote_blocks(
                    list(blocks), _StorageTier.SSD, session_id,
                )

            # Free from allocator
            self.allocator.free(session_id)

            # Remove block table
            self.block_tables.remove_table(session_id)

            # Clean up agent cache tracking
            self.agent_cache.end_session(session_id)

            # Unregister from lifecycle tracker
            self.lifecycle.unregister(session_id)

    # ------------------------------------------------------------------
    # Per-turn hooks
    # ------------------------------------------------------------------

    def pre_llm_call(
        self,
        session_id: str,
        messages: List[Dict],
    ) -> Dict[str, Any]:
        """Called before each LLM API call.

        1. Transition lifecycle to PREFILL (if new turn)
        2. Check prefix cache for reusable KV blocks
        3. Allocate new blocks for uncached messages
        4. Deduplicate system prompt / tool definitions
        5. Return hints for the conversation loop

        Returns a dict with:
          - ``"allocation"``: list of newly allocated block IDs
          - ``"prefix_hit"``: number of token blocks reused
          - ``"phase"``: current lifecycle phase
        """
        with self._lock:
            self._turn_count += 1

            # Phase transition
            lc = self.lifecycle.get(session_id)
            if lc and lc.is_waiting:
                # Tool result just arrived → promote blocks back to GPU
                self.lifecycle.on_phase_change(session_id, AgentPhase.PREFILL)
                blocks = self.allocator.get_request_blocks(session_id)
                if blocks:
                    self.hierarchical_store.prefetch_for_resume(
                        session_id, list(blocks),
                    )
            elif lc and lc.phase != AgentPhase.PREFILL:
                self.lifecycle.on_phase_change(session_id, AgentPhase.PREFILL)

            # Prefix cache lookup: combine all message text for prefix match
            combined_text = " ".join(
                str(m.get("content", ""))[:200] for m in messages
                if m.get("role") in ("system", "user")
            )
            token_ids = [ord(c) for c in combined_text[:1024]]
            prefix_tokens, matched_hashes, matched_blocks = (
                self.prefix_cache.find_longest_prefix(token_ids)
            )

            # Allocate new blocks for uncached content
            uncached_tokens = max(0, len(token_ids) - prefix_tokens)
            new_blocks: List[int] = []
            if uncached_tokens > 0:
                new_blocks = self.allocator.allocate(
                    session_id, uncached_tokens, group_id=session_id,
                )
                self.hierarchical_store.register_blocks(new_blocks)

            # Insert the new hashes into the prefix cache (Phase 2)
            if uncached_tokens > 0 and new_blocks:
                uncached_ids = token_ids[prefix_tokens:]
                hashes = _compute_prefix_hashes(
                    uncached_ids, self._config.block_size,
                )
                if hashes and new_blocks:
                    # Ensure matching counts: one hash per block
                    n = min(len(hashes), len(new_blocks))
                    if n > 0:
                        self.prefix_cache.insert(hashes[:n], new_blocks[:n])

            # Deduplicate messages
            deduped, dedup_result = self.deduplicator.deduplicate(
                messages, session_id=session_id,
            )

            return {
                "allocation": new_blocks,
                "prefix_hit": len(matched_blocks),
                "prefix_tokens_reused": prefix_tokens,
                "phase": lc.phase.value if lc else "prefill",
                "dedup_saved_tokens": dedup_result.dropped_tokens_est,
                "dedup_dropped": dedup_result.dropped_count,
            }

    def post_llm_call(
        self,
        session_id: str,
        assistant_message: Dict | None = None,
        has_tool_calls: bool = False,
    ) -> Dict[str, Any]:
        """Called after each LLM response.

        1. Transition lifecycle: PREFILL→DECODING, or DECODING→TOOL_CALL
        2. Record tool usage for schema compressor

        Returns lifecycle metadata.
        """
        with self._lock:
            lc = self.lifecycle.get(session_id)

            if has_tool_calls:
                # Agent is about to execute tools → TOOL_CALL
                if lc:
                    lc.record_turn()
                    lc.record_tool_call()
                self.lifecycle.on_phase_change(session_id, AgentPhase.TOOL_CALL)
            else:
                # Plain assistant response → completed this turn
                if lc:
                    lc.record_turn()
                    lc.record_activity()

            # Record tool usage for frequency-based schema compression
            if assistant_message and has_tool_calls:
                tool_calls = assistant_message.get("tool_calls", [])
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "")
                    if name:
                        self.tool_compressor.record_usage(name)

            return {
                "phase": "tool_call" if has_tool_calls else "decoding",
                "turn": lc.turn_count if lc else 1,
            }

    def on_tool_result(
        self,
        session_id: str,
        tool_name: str = "",
    ):
        """Called when a tool result arrives.

        Triggers promotion of blocks back to GPU so the agent can resume
        without cold-cache latency.
        """
        with self._lock:
            self.lifecycle.on_phase_change(session_id, AgentPhase.PREFILL)
            blocks = self.allocator.get_request_blocks(session_id)
            if blocks:
                self.hierarchical_store.prefetch_for_resume(
                    session_id, list(blocks),
                )

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    def maybe_compress(
        self,
        session_id: str,
        messages: List[Dict],
        current_tokens: int,
        max_tokens: int,
    ) -> Tuple[List[Dict], Optional[CompressionResult]]:
        """Check if context compression is needed and, if so, compress.

        Called from the conversation loop when token estimates approach
        the context window limit.

        Returns ``(maybe_compressed_messages, compression_result)``.
        If no compression was needed, returns the original messages and None.
        """
        with self._lock:
            ratio = current_tokens / max(max_tokens, 1)

            if ratio < 0.5:
                return messages, None  # well within limits

            # Choose mode based on urgency
            mode = CompressionMode.CO if ratio > 0.85 else CompressionMode.UT

            compressed_text, result = self.compressor.compress_history(
                messages, mode=mode,
            )

            if result.compression_ratio <= 0:
                return messages, None

            # Free blocks that were compressed out
            # (in production, this would compute which blocks are no longer
            #  referenced and free them)
            freed = 0
            blocks = self.allocator.get_request_blocks(session_id)
            if blocks and result.compression_ratio > 0.3:
                # Heuristic: free blocks proportional to compression ratio
                n_to_free = int(len(blocks) * result.compression_ratio * 0.7)
                for bid in sorted(blocks)[:n_to_free]:
                    # Only free non-pinned blocks
                    block = self.allocator.get_block(bid)
                    if block and not block.is_pinned:
                        self.allocator.free_block(session_id, bid)
                        freed += 1

            return messages, result

    def compress_tools(
        self,
        tool_definitions: List[Dict],
    ) -> Tuple[List[Dict], int]:
        """Compress tool definitions based on usage frequency.

        Called before building the LLM request.
        """
        compressed, _ = self.tool_compressor.compress(tool_definitions)
        before = sum(len(str(t)) for t in tool_definitions)
        after = sum(len(str(c)) for c in compressed)
        return compressed, before - after

    # ------------------------------------------------------------------
    # Demotion / promotion callbacks
    # ------------------------------------------------------------------

    def _on_demote(self, request_id: str, _: str, target: _StorageTier):
        """Callback: demote blocks to lower storage tier."""
        with self._lock:
            blocks = self.allocator.get_request_blocks(request_id)
            if blocks:
                self.hierarchical_store.demote_blocks(
                    list(blocks), target, request_id,
                )

    def _on_promote(self, request_id: str, _: str):
        """Callback: promote blocks back to GPU."""
        with self._lock:
            blocks = self.allocator.get_request_blocks(request_id)
            if blocks:
                self.hierarchical_store.promote_blocks(
                    list(blocks), _StorageTier.GPU, request_id,
                )

    def _on_evict(self, request_id: str, _: str):
        """Callback: permanently evict blocks."""
        with self._lock:
            blocks = self.allocator.get_request_blocks(request_id)
            if blocks:
                self.hierarchical_store.evict_blocks(list(blocks))
                self.lifecycle.unregister(request_id)

    # ------------------------------------------------------------------
    # Background scan
    # ------------------------------------------------------------------

    def _start_scan_timer(self):
        """Periodic background task that scans lifecycle states and triggers
        tier migrations for idle requests."""
        self._scan_timer = threading.Timer(self._scan_interval_s, self._scan_tick)
        self._scan_timer.daemon = True
        self._scan_timer.start()

    def _scan_tick(self):
        """One tick of the background scanner."""
        try:
            self.lifecycle.scan_and_migrate()
            self.hierarchical_store.evict_cold_blocks()
        except Exception:
            pass
        finally:
            if self._scan_timer is not None:
                self._start_scan_timer()

    def stop(self):
        """Stop background tasks (call before process exit)."""
        if self._scan_timer is not None:
            self._scan_timer.cancel()
            self._scan_timer = None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Aggregate statistics from all subsystems."""
        with self._lock:
            return {
                "allocator": self.allocator.stats(),
                "prefix_cache": self.prefix_cache.stats(),
                "agent_cache": self.agent_cache.stats(),
                "lifecycle": self.lifecycle.stats(),
                "hierarchical_store": self.hierarchical_store.stats(),
                "compressor": self.compressor.stats(),
                "deduplicator": self.deduplicator.stats(),
                "tool_compressor": self.tool_compressor.stats(),
                "sessions": self._session_count,
                "turns": self._turn_count,
                "model": self._model_name,
                "config": {
                    "block_size": self._config.block_size,
                    "gpu_blocks": self._config.max_gpu_blocks,
                    "block_size_bytes": self._config.block_size_bytes,
                },
            }

    def reset_stats(self):
        """Zero out cumulative counters across all subsystems."""
        with self._lock:
            self.allocator.reset_stats()
            self.prefix_cache.reset_stats()
            self.agent_cache.reset_stats()
            self.lifecycle.reset_stats()
            self.hierarchical_store.reset_stats()
            self.compressor.reset_stats()
            self.deduplicator.reset_stats()
            self.tool_compressor.reset_stats()
            self._session_count = 0
            self._turn_count = 0

    def dump(self) -> str:
        """Human-readable dump of all subsystem states."""
        lines = [
            "=" * 60,
            "AgentMemoryManager — Subsystem State",
            "=" * 60,
            "",
            "[Block Allocator]",
            self._format_dict(self.allocator.stats()),
            "",
            "[Prefix Cache]",
            self._format_dict(self.prefix_cache.stats()),
            "",
            "[Lifecycle Tracker]",
            self._format_dict(self.lifecycle.stats()),
            "",
            "[Hierarchical Store]",
            self._format_dict(self.hierarchical_store.stats()),
            "",
            "[Compressor]",
            self._format_dict(self.compressor.stats()),
            "",
            "[Deduplicator]",
            self._format_dict(self.deduplicator.stats()),
            "",
            "[Tool Schema Compressor]",
            self._format_dict(self.tool_compressor.stats()),
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_dict(d: dict, indent: int = 2) -> str:
        prefix = " " * indent
        return "\n".join(f"{prefix}{k}: {v}" for k, v in d.items())

    def __repr__(self) -> str:
        return (
            f"AgentMemoryManager(model={self._model_name}, "
            f"sessions={self._session_count}, turns={self._turn_count}, "
            f"blocks={self.allocator.used_blocks}/{self.allocator.total_blocks}, "
            f"cache_hit={self.prefix_cache.hit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

# ── Global reference for the monitor API server ──
# When create_agent_memory_manager() is called, it stores the instance here
# so scripts/monitor_api.py can find it without needing an Agent object.
_global_manager_instance = None


def get_global_memory_manager():
    """Return the most recently created AgentMemoryManager, or None."""
    return _global_manager_instance


def create_agent_memory_manager(
    model_name: str = "qwen2.5-7b",
    gpu_gb: int = 80,
    block_size: int = 16,
    enable_ssd: bool = True,
) -> "AgentMemoryManager":
    """Factory: create a pre-configured ``AgentMemoryManager``.

    Parameters
    ----------
    model_name : str
        Model to auto-detect KV sizing profile for.
    gpu_gb : int
        GPU VRAM available for KV Cache.
    block_size : int
        Tokens per KV Cache block (16 is the vLLM default).
    enable_ssd : bool
        Whether to enable the SSD tier (Phase 3).

    Returns
    -------
    AgentMemoryManager
        Fully wired memory subsystem ready for integration.
    """
    config = MemoryConfig.for_model(model_name, block_size=block_size, gpu_gb=gpu_gb)
    config.enable_ssd = enable_ssd
    mgr = AgentMemoryManager(config=config, model_name=model_name)
    global _global_manager_instance
    _global_manager_instance = mgr
    return mgr
