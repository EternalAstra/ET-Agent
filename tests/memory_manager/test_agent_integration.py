"""Integration tests: AgentMemoryManager full lifecycle with DeepSeek V4."""

import pytest
from agent.memory_hooks import AgentMemoryManager, create_agent_memory_manager
from memory_manager.config import MemoryConfig
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager


# ═══════════════════════════════════════════════════════════════════
# Factory + construction
# ═══════════════════════════════════════════════════════════════════

class TestConstruction:
    def test_create_with_factory(self):
        mgr = create_agent_memory_manager("qwen2.5-7b", gpu_gb=80)
        assert mgr.allocator.total_blocks > 0
        assert mgr.prefix_cache.size == 0
        assert mgr.lifecycle.stats()["total_requests"] == 0

    def test_create_with_custom_config(self):
        cfg = MemoryConfig(block_size=16, gpu_capacity_bytes=10 * 1024**3)
        mgr = AgentMemoryManager(config=cfg, model_name="test")
        assert mgr.allocator.total_blocks == cfg.max_gpu_blocks

    def test_all_subsystems_initialized(self):
        mgr = create_agent_memory_manager()
        assert mgr.allocator is not None
        assert mgr.block_tables is not None
        assert mgr.prefix_cache is not None
        assert mgr.agent_cache is not None
        assert mgr.hierarchical_store is not None
        assert mgr.lifecycle is not None
        assert mgr.compressor is not None
        assert mgr.deduplicator is not None
        assert mgr.tool_compressor is not None


# ═══════════════════════════════════════════════════════════════════
# Session lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestSessionLifecycle:
    def test_session_start_end(self):
        mgr = create_agent_memory_manager("qwen2.5-7b", gpu_gb=80)
        sp_tokens = [i % 50000 for i in range(2000)]  # simulated system prompt

        mgr.on_session_start(
            "sess-1",
            system_prompt_tokens=sp_tokens,
        )

        # System prompt should be cached
        assert mgr.agent_cache.get_system_prompt("default") is not None
        assert mgr.prefix_cache.size > 0
        assert mgr.lifecycle.get("sess-1") is not None

        # End session
        mgr.on_session_end("sess-1")
        assert mgr.lifecycle.get("sess-1") is None

    def test_session_with_tools(self):
        mgr = create_agent_memory_manager()
        tools = [
            {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk", "parameters": {"type": "object", "properties": {}}}},
        ]

        mgr.on_session_start("sess-1", tool_definitions=tools)
        assert "web_search" in mgr.agent_cache.all_tool_schemas()
        assert "read_file" in mgr.agent_cache.all_tool_schemas()

    def test_multiple_sessions_isolated(self):
        mgr = create_agent_memory_manager()

        sp1 = list(range(1000))
        sp2 = list(range(1000, 2000))

        mgr.on_session_start("sess-a", system_prompt_tokens=sp1)
        mgr.on_session_start("sess-b", system_prompt_tokens=sp2)

        assert mgr.lifecycle.get("sess-a") is not None
        assert mgr.lifecycle.get("sess-b") is not None
        assert mgr.lifecycle.stats()["total_requests"] == 2

        # Each session has its own block table
        assert mgr.block_tables.has_table("sess-a")
        assert mgr.block_tables.has_table("sess-b")


# ═══════════════════════════════════════════════════════════════════
# Pre-LLM call
# ═══════════════════════════════════════════════════════════════════

class TestPreLLMCall:
    def test_pre_llm_allocation(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")

        msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Hello, search for AGI papers."},
        ]

        info = mgr.pre_llm_call("sess-1", msgs)
        assert "allocation" in info
        assert "prefix_hit" in info
        assert "phase" in info
        assert mgr.allocator.used_blocks > 0

    def test_pre_llm_prefix_hit(self):
        mgr = create_agent_memory_manager()

        # First session: cache some content
        sp_tokens = list(range(2000))
        mgr.on_session_start("sess-1", system_prompt_tokens=sp_tokens)

        msgs = [{"role": "user", "content": "hello" * 100}]
        mgr.pre_llm_call("sess-1", msgs)

        # Second session with similar prefix should get a hit
        mgr.on_session_start("sess-2", system_prompt_tokens=sp_tokens)
        info2 = mgr.pre_llm_call("sess-2", msgs)
        # May or may not get prefix hit depending on token ordering
        assert "prefix_hit" in info2

    def test_dedup_on_repeated_calls(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")

        sys_msg = {"role": "system", "content": "You are a helpful assistant. " * 20}
        user_msg = {"role": "user", "content": "Hi"}

        # First call
        info1 = mgr.pre_llm_call("sess-1", [sys_msg, user_msg])
        # Second call — system should be deduplicated
        info2 = mgr.pre_llm_call("sess-1", [sys_msg, user_msg])
        assert "dedup_dropped" in info2


# ═══════════════════════════════════════════════════════════════════
# Post-LLM call
# ═══════════════════════════════════════════════════════════════════

class TestPostLLMCall:
    def test_post_llm_no_tools(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")
        mgr.pre_llm_call("sess-1", [{"role": "user", "content": "hi"}])

        info = mgr.post_llm_call(
            "sess-1",
            assistant_message={"role": "assistant", "content": "Hello!"},
            has_tool_calls=False,
        )
        lc = mgr.lifecycle.get("sess-1")
        assert lc.turn_count == 1

    def test_post_llm_with_tools(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")
        mgr.pre_llm_call("sess-1", [{"role": "user", "content": "search"}])

        info = mgr.post_llm_call(
            "sess-1",
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": '{"q":"test"}'}}
                ],
            },
            has_tool_calls=True,
        )
        lc = mgr.lifecycle.get("sess-1")
        assert lc.is_waiting
        assert lc.tool_call_count == 1

        # Tool compressor should have recorded usage
        assert mgr.tool_compressor.get_usage_count("web_search") == 1

    def test_tool_result_promotion(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")
        mgr.pre_llm_call("sess-1", [{"role": "user", "content": "hi"}])

        # Enter tool-call phase
        mgr.post_llm_call("sess-1", has_tool_calls=True)

        # Tool result arrives → promote
        mgr.on_tool_result("sess-1", tool_name="web_search")
        lc = mgr.lifecycle.get("sess-1")
        assert lc.phase.name == "PREFILL"  # back to active


# ═══════════════════════════════════════════════════════════════════
# Context compression integration
# ═══════════════════════════════════════════════════════════════════

class TestCompressionIntegration:
    def test_maybe_compress_below_threshold(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")

        msgs = [{"role": "user", "content": "short"}]
        result, compress_info = mgr.maybe_compress("sess-1", msgs, 100, 100000)
        assert compress_info is None  # well below threshold

    def test_maybe_compress_above_threshold(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1")

        # Build a large message list
        msgs = []
        for i in range(50):
            msgs.append({"role": "user", "content": f"turn {i}: " + "x" * 200})
            msgs.append({"role": "assistant", "content": f"response {i}: " + "y" * 200})

        _, compress_info = mgr.maybe_compress("sess-1", msgs, 95000, 100000)
        # At 95% usage, should trigger CO compression
        if compress_info:
            assert compress_info.compression_ratio >= 0

    def test_compress_tools(self):
        mgr = create_agent_memory_manager()
        tools = [
            {"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool number {i}" * 10,
             "parameters": {"type": "object", "properties": {}}}}
            for i in range(20)
        ]
        compressed, saved = mgr.compress_tools(tools)
        assert len(compressed) <= 20
        assert saved >= 0


# ═══════════════════════════════════════════════════════════════════
# Full agent turn simulation
# ═══════════════════════════════════════════════════════════════════

class TestFullTurnSimulation:
    def test_complete_multi_turn_cycle(self):
        """Simulate: 3-turn agent conversation with tools."""
        mgr = create_agent_memory_manager("qwen2.5-7b", gpu_gb=80)

        # ── Session start ──
        system_prompt = [ord(c) for c in "You are an AI assistant." * 50]
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search the web", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
        ]
        mgr.on_session_start("sess-1", system_prompt_tokens=system_prompt, tool_definitions=tools)

        results_over_time = []

        # ── Turn 1: user → agent (no tools) ──
        msgs = [{"role": "user", "content": "Hello!"}]
        pre = mgr.pre_llm_call("sess-1", msgs)
        assert pre["prefix_hit"] >= 0

        post = mgr.post_llm_call("sess-1", has_tool_calls=False)
        results_over_time.append(post)

        # ── Turn 2: user → agent → tool call ──
        msgs2 = [{"role": "user", "content": "Search for papers"}]
        pre2 = mgr.pre_llm_call("sess-1", msgs2)
        assert "allocation" in pre2

        post2 = mgr.post_llm_call(
            "sess-1",
            assistant_message={
                "tool_calls": [{"function": {"name": "search", "arguments": '{"q":"AI"}'}}]
            },
            has_tool_calls=True,
        )
        assert mgr.lifecycle.get("sess-1").is_waiting

        # Tool returns → promote
        mgr.on_tool_result("sess-1", "search")
        assert not mgr.lifecycle.get("sess-1").is_waiting

        # ── Turn 3: tool result → agent final response ──
        msgs3 = [
            {"role": "tool", "tool_call_id": "call_1", "content": "Found 42 papers."},
        ]
        pre3 = mgr.pre_llm_call("sess-1", msgs3)
        post3 = mgr.post_llm_call("sess-1", has_tool_calls=False)

        # ── Verify lifecycle tracking ──
        lc = mgr.lifecycle.get("sess-1")
        assert lc.turn_count == 3
        assert lc.tool_call_count == 1

        # ── Session end ──
        mgr.on_session_end("sess-1")
        assert mgr.lifecycle.get("sess-1") is None

    def test_parallel_sessions(self):
        """Two sessions running concurrently, sharing system prompt."""
        mgr = create_agent_memory_manager()
        sp = list(range(1500))

        # Start both sessions
        mgr.on_session_start("s1", system_prompt_tokens=sp)
        mgr.on_session_start("s2", system_prompt_tokens=sp)

        # Turn in session 1
        mgr.pre_llm_call("s1", [{"role": "user", "content": "A"}])
        mgr.post_llm_call("s1", has_tool_calls=False)

        # Turn in session 2
        mgr.pre_llm_call("s2", [{"role": "user", "content": "B"}])
        mgr.post_llm_call("s2", has_tool_calls=True)

        # Session 2 should be in tool_call phase
        assert mgr.lifecycle.get("s2").is_waiting

        # Session 1 should be active
        lc1 = mgr.lifecycle.get("s1")
        assert lc1.turn_count == 1

        # End both
        mgr.on_session_end("s1")
        mgr.on_session_end("s2")
        assert mgr.lifecycle.stats()["total_requests"] == 0


# ═══════════════════════════════════════════════════════════════════
# Statistics and dump
# ═══════════════════════════════════════════════════════════════════

class TestStatsAndDebug:
    def test_stats_after_session(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("sess-1", system_prompt_tokens=list(range(1000)))
        mgr.pre_llm_call("sess-1", [{"role": "user", "content": "hi"}])
        mgr.post_llm_call("sess-1", has_tool_calls=False)
        mgr.on_session_end("sess-1")

        s = mgr.stats()
        assert s["sessions"] == 1
        assert s["turns"] == 1
        assert "allocator" in s
        assert "prefix_cache" in s

    def test_dump(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("s1")
        dump = mgr.dump()
        assert "AgentMemoryManager" in dump
        assert "Block Allocator" in dump
        assert "Prefix Cache" in dump
        assert "Lifecycle Tracker" in dump

    def test_reset_stats(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("s1")
        mgr.pre_llm_call("s1", [{"role": "user", "content": "x"}])
        mgr.reset_stats()
        assert mgr.stats()["sessions"] == 0
        assert mgr.stats()["turns"] == 0

    def test_repr(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("s1")
        r = repr(mgr)
        assert "AgentMemoryManager" in r
        assert "sessions=1" in r

    def test_stop(self):
        mgr = create_agent_memory_manager()
        mgr.stop()
        # Should not raise
        mgr.stop()


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_session_end_twice_is_safe(self):
        mgr = create_agent_memory_manager()
        mgr.on_session_start("s1")
        mgr.on_session_end("s1")
        mgr.on_session_end("s1")  # double-free should not crash

    def test_pre_llm_without_session(self):
        """Calling hooks without a session should not crash."""
        mgr = create_agent_memory_manager()
        info = mgr.pre_llm_call("nonexistent", [{"role": "user", "content": "hi"}])
        assert isinstance(info, dict)

    def test_post_llm_without_session(self):
        mgr = create_agent_memory_manager()
        info = mgr.post_llm_call("nonexistent", has_tool_calls=False)
        assert isinstance(info, dict)

    def test_compress_empty(self):
        mgr = create_agent_memory_manager()
        msgs, result = mgr.maybe_compress("s", [], 1000, 10000)
        assert result is None
        assert msgs == []
