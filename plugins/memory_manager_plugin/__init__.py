"""
Memory Manager Plugin — bridges AgentMemoryManager into the Hermes hook system.

Registers lifecycle hooks (on_session_start, pre_llm_call, post_llm_call,
pre_tool_call, post_tool_call, on_session_end) that forward to
``agent.memory_hooks.AgentMemoryManager``.

This is the cleanest integration point — no changes to conversation_loop.py
or run_agent.py needed.  The plugin is discovered and loaded by the standard
Hermes plugin pipeline, and hooks are invoked exactly where they need to be.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global singleton — initialized on first hook call
_memory_manager: Optional[Any] = None  # AgentMemoryManager
_monitor: Optional[Any] = None         # MemoryMonitor


def _get_manager(plugin_ctx) -> Optional[Any]:
    """Lazy-init the memory manager on first hook call."""
    global _memory_manager, _monitor

    if _memory_manager is not None:
        return _memory_manager

    try:
        from agent.memory_hooks import create_agent_memory_manager
        from memory_manager.memory_monitor import MemoryMonitor

        # Detect model from config
        model_name = "qwen2.5-7b"
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
            mc = cfg.get("model", {})
            if isinstance(mc, dict):
                model_name = mc.get("default", model_name)
            elif isinstance(mc, str) and mc:
                model_name = mc
        except Exception:
            pass

        _memory_manager = create_agent_memory_manager(model_name, gpu_gb=80)
        _monitor = MemoryMonitor(_memory_manager)
        _monitor.start(interval_s=2.0)

        logger.info("ET-Agent Memory Manager initialized (model=%s)", model_name)
        return _memory_manager

    except Exception as exc:
        logger.warning("Memory Manager init failed: %s", exc)
        return None


def register(plugin_ctx):
    """Register lifecycle hooks with the Hermes plugin system."""

    # ── on_session_start ──
    def on_session_start(session_id: str = "", **kwargs):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return
        try:
            # Extract system prompt tokens if available
            system_content = kwargs.get("system_message", "") or ""
            sp_tokens = None
            if system_content:
                sp_tokens = [ord(c) for c in system_content[:4000]]

            mgr.on_session_start(
                session_id or "default",
                system_prompt_tokens=sp_tokens,
            )
        except Exception as exc:
            logger.debug("memory_manager on_session_start: %s", exc)

    plugin_ctx.register_hook("on_session_start", on_session_start)

    # ── on_session_end ──
    def on_session_end(session_id: str = "", **kwargs):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return
        try:
            mgr.on_session_end(session_id or "default")
        except Exception as exc:
            logger.debug("memory_manager on_session_end: %s", exc)

    plugin_ctx.register_hook("on_session_end", on_session_end)

    # ── pre_llm_call ──
    def pre_llm_call(
        session_id: str = "",
        messages: list = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return None
        try:
            info = mgr.pre_llm_call(
                session_id or "default",
                messages or [],
            )
            # Inject memory hints into the user message
            if info.get("prefix_tokens_reused", 0) > 0:
                hint = (
                    f"[Memory: {info['prefix_tokens_reused']} tokens reused "
                    f"from cache, {info.get('dedup_saved_tokens', 0)} dedup-saved]"
                )
                return {"target": "user_message", "content": hint}
        except Exception as exc:
            logger.debug("memory_manager pre_llm_call: %s", exc)
        return None

    plugin_ctx.register_hook("pre_llm_call", pre_llm_call)

    # ── post_llm_call ──
    def post_llm_call(
        session_id: str = "",
        assistant_message: dict = None,
        **kwargs,
    ):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return
        try:
            has_tools = bool(
                (assistant_message or {}).get("tool_calls")
            )
            mgr.post_llm_call(
                session_id or "default",
                assistant_message=assistant_message,
                has_tool_calls=has_tools,
            )
        except Exception as exc:
            logger.debug("memory_manager post_llm_call: %s", exc)

    plugin_ctx.register_hook("post_llm_call", post_llm_call)

    # ── pre_tool_call ──
    def pre_tool_call(
        session_id: str = "",
        tool_name: str = "",
        **kwargs,
    ):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return
        # Tool is about to execute → already in TOOL_CALL phase from post_llm_call
        # Tool compressor: record usage
        if tool_name:
            mgr.tool_compressor.record_usage(tool_name)
            logger.debug("memory_manager tool usage: %s", tool_name)

    plugin_ctx.register_hook("pre_tool_call", pre_tool_call)

    # ── post_tool_call ──
    def post_tool_call(
        session_id: str = "",
        tool_name: str = "",
        **kwargs,
    ):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return
        try:
            # Tool result arrived → promote KV Cache back to GPU
            mgr.on_tool_result(
                session_id or "default",
                tool_name=tool_name,
            )
        except Exception as exc:
            logger.debug("memory_manager post_tool_call: %s", exc)

    plugin_ctx.register_hook("post_tool_call", post_tool_call)

    # ── Register a /memory slash command for inspection ──
    def cmd_memory(args, **kwargs):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return "⚠️ Memory Manager not initialized."

        dump = mgr.dump()
        return f"```\n{dump}\n```"

    plugin_ctx.register_command("memory", cmd_memory, hidden=False)

    # ── Register a /memory-stats slash command ──
    def cmd_memory_stats(args, **kwargs):
        mgr = _get_manager(plugin_ctx)
        if mgr is None:
            return "⚠️ Memory Manager not initialized."

        import json
        stats = mgr.stats()
        return f"```json\n{json.dumps(stats, indent=2, default=str)}\n```"

    plugin_ctx.register_command("memory-stats", cmd_memory_stats, hidden=False)

    logger.info("ET-Agent Memory Manager plugin registered (6 hooks + 2 commands)")
