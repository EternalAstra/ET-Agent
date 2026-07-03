"""Phase 5 — KV memory manager wired into AIAgent lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.kv_memory_integration import (
    get_kv_memory_manager,
    init_kv_memory_manager,
    kv_memory_stats,
    on_session_end_kv,
    on_session_start_kv,
    on_tool_results_kv,
    post_llm_call_kv,
    pre_llm_call_kv,
)


def _mock_agent(**overrides):
    agent = SimpleNamespace(
        session_id="test-session",
        model="qwen2.5:3b-instruct",
        parent_session_id=None,
        quiet_mode=True,
        tools=[],
        _cached_system_prompt="You are a helpful assistant.",
        _kv_memory_manager=None,
    )
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


class TestInit:
    def test_init_creates_manager_when_enabled(self):
        agent = _mock_agent()
        with patch("agent.kv_memory_integration._kv_config", return_value={"enabled": True, "gpu_gb": 6}):
            init_kv_memory_manager(agent)
        mgr = get_kv_memory_manager(agent)
        assert mgr is not None
        assert mgr.allocator.total_blocks > 0

    def test_init_skipped_when_disabled(self):
        agent = _mock_agent()
        with patch("agent.kv_memory_integration._kv_config", return_value={"enabled": False}):
            init_kv_memory_manager(agent)
        assert get_kv_memory_manager(agent) is None

    def test_init_skipped_for_subagent(self):
        agent = _mock_agent(parent_session_id="parent-1")
        init_kv_memory_manager(agent)
        assert get_kv_memory_manager(agent) is None


class TestLifecycleHooks:
    @pytest.fixture()
    def agent_with_mgr(self):
        agent = _mock_agent()
        with patch("agent.kv_memory_integration._kv_config", return_value={"gpu_gb": 6}):
            init_kv_memory_manager(agent)
        return agent

    def test_full_turn_flow(self, agent_with_mgr):
        agent = agent_with_mgr
        sid = agent.session_id

        on_session_start_kv(agent)
        mgr = get_kv_memory_manager(agent)
        assert mgr.lifecycle.get(sid) is not None

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        pre_llm_call_kv(agent, messages)
        assert mgr.allocator.used_blocks > 0

        assistant = SimpleNamespace(
            content="Hi there!",
            tool_calls=[],
        )
        post_llm_call_kv(agent, assistant, has_tool_calls=False)
        assert mgr.lifecycle.get(sid).turn_count == 1

        on_session_end_kv(agent)
        assert mgr.lifecycle.get(sid) is None

    def test_tool_call_flow(self, agent_with_mgr):
        agent = agent_with_mgr
        sid = agent.session_id

        on_session_start_kv(agent)
        pre_llm_call_kv(agent, [{"role": "user", "content": "search"}])

        tc = SimpleNamespace(
            function=SimpleNamespace(name="web_search", arguments='{"q":"test"}'),
        )
        assistant = SimpleNamespace(content=None, tool_calls=[tc])
        post_llm_call_kv(agent, assistant, has_tool_calls=True)

        mgr = get_kv_memory_manager(agent)
        assert mgr.lifecycle.get(sid).is_waiting

        on_tool_results_kv(agent, ["web_search"])
        assert not mgr.lifecycle.get(sid).is_waiting

    def test_hooks_noop_without_manager(self):
        agent = _mock_agent()
        on_session_start_kv(agent)
        pre_llm_call_kv(agent, [])
        post_llm_call_kv(agent, None, has_tool_calls=False)
        on_tool_results_kv(agent)
        on_session_end_kv(agent)
        assert kv_memory_stats(agent) is None

    def test_stats_after_turn(self, agent_with_mgr):
        agent = agent_with_mgr
        on_session_start_kv(agent)
        pre_llm_call_kv(agent, [{"role": "user", "content": "x"}])
        post_llm_call_kv(
            agent,
            SimpleNamespace(content="ok", tool_calls=[]),
            has_tool_calls=False,
        )
        stats = kv_memory_stats(agent)
        assert stats is not None
        assert stats["sessions"] == 1
        assert stats["turns"] == 1


class TestConversationLoopImports:
    """Ensure conversation_loop references the integration module."""

    def test_conversation_loop_has_kv_hooks(self):
        import agent.conversation_loop as cl

        source = open(cl.__file__, encoding="utf-8").read()
        assert "kv_memory_integration" in source
        assert "pre_llm_call_kv" in source
        assert "post_llm_call_kv" in source
        assert "on_tool_results_kv" in source

    def test_run_agent_has_session_end_kv(self):
        import run_agent

        source = open(run_agent.__file__, encoding="utf-8").read()
        assert "on_session_end_kv" in source

    def test_agent_init_has_kv_init(self):
        import agent.agent_init as ai

        source = open(ai.__file__, encoding="utf-8").read()
        assert "init_kv_memory_manager" in source
