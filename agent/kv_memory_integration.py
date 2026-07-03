"""Phase 5 — wire AgentMemoryManager into the main Agent loop.

Separate from ``agent.memory_manager`` (Honcho/Mem0 plugins).  Controlled by
``memory.kv_manager`` in config.yaml.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _kv_config() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config
        return dict(cfg_get(load_config(), "memory", "kv_manager", default={}) or {})
    except Exception:
        return {}


def init_kv_memory_manager(agent: Any) -> None:
    """Create ``agent._kv_memory_manager`` if enabled."""
    agent._kv_memory_manager = None
    if getattr(agent, "parent_session_id", None):
        return

    kv_cfg = _kv_config()
    if kv_cfg.get("enabled", True) is False:
        return

    try:
        from agent.memory_hooks import create_agent_memory_manager

        model_name = str(getattr(agent, "model", "") or "qwen2.5-7b")
        bare = model_name.split("/")[-1] if "/" in model_name else model_name
        gpu_gb = int(kv_cfg.get("gpu_gb", 6))
        mgr = create_agent_memory_manager(bare, gpu_gb=gpu_gb)
        agent._kv_memory_manager = mgr
        if not getattr(agent, "quiet_mode", False):
            logger.info(
                "KV memory manager enabled (model=%s, gpu_gb=%d, blocks=%d)",
                bare, gpu_gb, mgr.allocator.total_blocks,
            )
    except Exception as exc:
        logger.warning("Failed to init KV memory manager: %s", exc)
        agent._kv_memory_manager = None


def get_kv_memory_manager(agent: Any):
    return getattr(agent, "_kv_memory_manager", None)


def _pseudo_tokens(text: str, limit: int = 8000) -> List[int]:
    text = text or ""
    return [ord(c) % 50000 for c in text[:limit]]


def _tool_definitions(agent: Any) -> List[Dict]:
    tools = getattr(agent, "tools", None) or []
    return [t for t in tools if isinstance(t, dict)]


def on_session_start_kv(agent: Any, *, system_prompt: str = "") -> None:
    mgr = get_kv_memory_manager(agent)
    sid = getattr(agent, "session_id", None)
    if not mgr or not sid:
        return
    try:
        sp = system_prompt or getattr(agent, "_cached_system_prompt", "") or ""
        mgr.on_session_start(
            sid,
            system_prompt_tokens=_pseudo_tokens(sp),
            tool_definitions=_tool_definitions(agent),
        )
    except Exception as exc:
        logger.debug("KV on_session_start failed: %s", exc)


def pre_llm_call_kv(agent: Any, messages: List[Dict]) -> None:
    mgr = get_kv_memory_manager(agent)
    sid = getattr(agent, "session_id", None)
    if not mgr or not sid:
        return
    try:
        mgr.pre_llm_call(sid, list(messages))
    except Exception as exc:
        logger.debug("KV pre_llm_call failed: %s", exc)


def post_llm_call_kv(agent: Any, assistant_message: Any, *, has_tool_calls: bool) -> None:
    mgr = get_kv_memory_manager(agent)
    sid = getattr(agent, "session_id", None)
    if not mgr or not sid:
        return
    try:
        am_dict: Optional[Dict] = None
        if assistant_message is not None:
            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            am_dict = {
                "role": "assistant",
                "content": getattr(assistant_message, "content", "") or "",
            }
            if tool_calls:
                am_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", ""),
                        }
                    }
                    for tc in tool_calls
                ]
        mgr.post_llm_call(sid, assistant_message=am_dict, has_tool_calls=has_tool_calls)
    except Exception as exc:
        logger.debug("KV post_llm_call failed: %s", exc)


def on_tool_results_kv(agent: Any, tool_names: Optional[List[str]] = None) -> None:
    mgr = get_kv_memory_manager(agent)
    sid = getattr(agent, "session_id", None)
    if not mgr or not sid:
        return
    names = tool_names or ["tool"]
    try:
        for name in names:
            mgr.on_tool_result(sid, tool_name=name or "tool")
    except Exception as exc:
        logger.debug("KV on_tool_result failed: %s", exc)


def on_session_end_kv(agent: Any) -> None:
    mgr = get_kv_memory_manager(agent)
    sid = getattr(agent, "session_id", None)
    if not mgr or not sid:
        return
    try:
        mgr.on_session_end(sid)
    except Exception as exc:
        logger.debug("KV on_session_end failed: %s", exc)


def kv_memory_stats(agent: Any) -> Optional[dict]:
    mgr = get_kv_memory_manager(agent)
    if not mgr:
        return None
    try:
        return mgr.stats()
    except Exception:
        return None
