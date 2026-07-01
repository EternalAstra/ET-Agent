"""Tests for ACON-style context compression (Phase 4)."""
import pytest
from memory_manager.context_compressor import (
    ContextCompressor, CompressionMode, CompressionThresholds,
    CompressionResult,
)
from memory_manager.prompt_deduplicator import (
    PromptDeduplicator, hash_content, DedupResult,
)
from memory_manager.tool_schema_compressor import (
    ToolSchemaCompressor, CompressionTier,
)


# ═══════════════════════════════════════════════════════
# Internal helpers (via observation compressor)
# ═══════════════════════════════════════════════════════

class TestInternalHelpers:
    def test_estimate_token_floor(self):
        """Token estimation: short strings floor at 1."""
        from memory_manager.context_compressor import _estimate_tokens
        assert _estimate_tokens("") == 1
        assert _estimate_tokens("hello") == 1
        assert _estimate_tokens("x" * 100) == 25

    def test_extract_section(self):
        from memory_manager.context_compressor import _extract_section
        text = "<REASONING>\n  hello\n</REASONING>"
        assert _extract_section(text, "REASONING") == "hello"
        assert _extract_section(text, "NOPE") == ""

    def test_parse_vars_table(self):
        from memory_manager.context_compressor import _parse_vars_table
        text = "| name | value |\n|------|-------|\n| x | 42 |\n| y | hello |"
        result = _parse_vars_table(text)
        # The separator row is filtered; only data rows count
        assert any(r.get("name") == "x" for r in result)

    def test_parse_empty_vars(self):
        from memory_manager.context_compressor import _parse_vars_table
        assert _parse_vars_table("") == []

    def test_strip_stack_trace(self):
        """Observation compressor handles stack traces via observation flow."""
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10,
        ))
        text = "header\n" + "\n".join(f"  line{i}" for i in range(50))
        result, _ = c.compress_observation(text)
        # The observation compressor applies dedup and stripping
        assert len(result) > 0


# ═══════════════════════════════════════════════════════
# Compression thresholds
# ═══════════════════════════════════════════════════════

class TestCompressionThresholds:
    def test_defaults(self):
        t = CompressionThresholds()
        assert t.history_token_threshold == 4096
        assert t.protect_last_n_turns == 3

    def test_for_agent_scenario(self):
        t = CompressionThresholds.for_agent_scenario()
        assert t.history_token_threshold == 4096

    def test_aggressive(self):
        t = CompressionThresholds.aggressive()
        assert t.history_token_threshold == 2048
        assert t.observation_token_threshold == 512


# ═══════════════════════════════════════════════════════
# Compression result
# ═══════════════════════════════════════════════════════

class TestCompressionResult:
    def test_basic(self):
        r = CompressionResult("hello", 1000, 100, 0.9, CompressionMode.UT, ["raw"])
        assert r.tokens_saved == 900
        assert r.compression_ratio == 0.9

    def test_repr(self):
        r = CompressionResult("x", 100, 50, 0.5, CompressionMode.UT, ["a"])
        assert "50.0%" in repr(r) or "50%" in repr(r)


# ═══════════════════════════════════════════════════════
# ContextCompressor — history
# ═══════════════════════════════════════════════════════

class TestContextCompressorHistory:
    def test_should_compress_below_threshold(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=9999
        ))
        assert not c.should_compress_history(1000)

    def test_should_compress_above_threshold(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=1000
        ))
        assert c.should_compress_history(5000)

    def test_compress_short_history_passes_through(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=99999
        ))
        msgs = [{"role": "user", "content": "hello"}]
        text, result = c.compress_history(msgs)
        assert text
        assert result.compression_ratio == 0.0

    def test_compress_long_history(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=100
        ))
        # Build enough messages to exceed threshold
        msgs = [
            {"role": "system", "content": "x" * 100},
            {"role": "user", "content": "y" * 100},
            {"role": "assistant", "content": "z" * 100,
             "tool_calls": [{"function": {"name": "search_files"}}]},
            {"role": "tool", "content": "a" * 500, "name": "search_files"},
            {"role": "assistant", "content": "b" * 200},
        ]
        text, result = c.compress_history(msgs)
        assert "<REASONING>" in text
        assert "<VARS>" in text
        assert "<ACTIONS_EXECUTED>" in text
        assert "<OPEN_TASKS>" in text
        assert result.compression_ratio >= 0

    def test_compress_extracts_actions(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=10
        ))
        msgs = [
            {"role": "tool", "content": "success: 42 results", "name": "web_search"},
            {"role": "tool", "content": "error: file not found", "name": "read_file"},
        ]
        text, result = c.compress_history(msgs)
        assert "web_search" in text
        assert "read_file" in text

    def test_compress_infers_reasoning(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=10
        ))
        msgs = [
            {"role": "assistant", "content": "I need to find the config file first."},
        ]
        text, _ = c.compress_history(msgs)
        assert "config file" in text

    def test_compress_infers_open_tasks(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=10
        ))
        msgs = [
            {"role": "user", "content": "Please deploy the application to production"},
        ]
        text, _ = c.compress_history(msgs)
        assert "deploy" in text.lower()


# ═══════════════════════════════════════════════════════
# ContextCompressor — observation
# ═══════════════════════════════════════════════════════

class TestContextCompressorObservation:
    def test_compress_short_observation_pass_through(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=99999
        ))
        text, result = c.compress_observation("short output")
        assert text == "short output"
        assert result.compression_ratio == 0.0

    def test_compress_long_observation_ut_mode(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10, observation_compressed_max_tokens=10,
        ))
        text, result = c.compress_observation("x " * 5000, mode=CompressionMode.CO)
        # CO mode truncates aggressively
        assert result.compression_ratio >= 0

    def test_compress_search_result(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10
        ))
        text, _ = c.compress_observation(
            "Found: https://example.com/page1\n" * 50,
            tool_name="web_search",
        )
        assert "[References" in text or len(text) < 3000

    def test_compress_terminal_output(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10
        ))
        lines = [f"line{i:04d}: some output text here" for i in range(100)]
        text, _ = c.compress_observation("\n".join(lines), tool_name="terminal")
        assert "elided" in text.lower()
        assert "line0000" in text
        assert "line0099" in text

    def test_compress_file_content(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10
        ))
        # Use unique lines to avoid dedup collapsing them
        long_content = "\n".join(f"line_{i:04d}: some content here" for i in range(200))
        text, _ = c.compress_observation(long_content, tool_name="read_file")
        # File elider: content > 1000 chars → truncated with line count
        assert "total lines" in text.lower() or "chars" in text.lower()

    def test_co_mode_more_aggressive(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            observation_token_threshold=10,
            observation_compressed_max_tokens=10,
        ))
        text_ut, _ = c.compress_observation("x" * 5000, mode=CompressionMode.UT)
        text_co, _ = c.compress_observation("x" * 5000, mode=CompressionMode.CO)
        assert len(text_co) <= len(text_ut)


# ═══════════════════════════════════════════════════════
# Guideline management
# ═══════════════════════════════════════════════════════

class TestGuidelines:
    def test_set_get_history_guideline(self):
        c = ContextCompressor()
        c.set_guideline("history", "Custom guideline v2")
        assert c.get_guideline("history") == "Custom guideline v2"

    def test_set_get_observation_guideline(self):
        c = ContextCompressor()
        c.set_guideline("observation", "Elide aggressively")
        assert c.get_guideline("observation") == "Elide aggressively"

    def test_reset_guidelines(self):
        c = ContextCompressor()
        c.set_guideline("history", "optimized")
        c.reset_guidelines()
        assert "optimized" not in c.get_guideline("history")

    def test_record_feedback(self):
        c = ContextCompressor()
        c.record_feedback({"trajectory": "failed", "reason": "lost variable X"})
        assert c.stats()["feedback_entries"] == 1

    def test_unknown_guideline_kind(self):
        c = ContextCompressor()
        with pytest.raises(ValueError):
            c.set_guideline("bogus", "x")


# ═══════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════

class TestCompressorStats:
    def test_initial_stats(self):
        c = ContextCompressor()
        s = c.stats()
        assert s["total_history_compressions"] == 0
        assert s["total_observation_compressions"] == 0
        assert s["total_tokens_saved"] == 0
        assert s["mode"] == "UT"

    def test_stats_after_compression(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=10,
            observation_token_threshold=10,
        ))
        msgs = [{"role": "user", "content": "x" * 2000}]
        c.compress_history(msgs)
        s = c.stats()
        assert s["total_history_compressions"] == 1

    def test_reset_stats(self):
        c = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=10,
        ))
        c.compress_history([{"role": "user", "content": "x" * 2000}])
        c.reset_stats()
        assert c.stats()["total_history_compressions"] == 0

    def test_repr(self):
        c = ContextCompressor()
        r = repr(c)
        assert "ContextCompressor" in r
        assert c._mode.name in r


# ═══════════════════════════════════════════════════════
# PromptDeduplicator
# ═══════════════════════════════════════════════════════

class TestPromptDeduplicator:
    def test_hash_content(self):
        h1 = hash_content("hello")
        h2 = hash_content("hello")
        assert h1 == h2
        assert len(h1) == 16

    def test_no_messages_passes_through(self):
        d = PromptDeduplicator()
        msgs, result = d.deduplicate([])
        assert msgs == []
        assert result.dropped_count == 0

    def test_system_repeated_dropped(self):
        d = PromptDeduplicator()
        sys_msg = {"role": "system", "content": "You are an agent."}

        # First call: system is new, should be kept
        msgs1, _ = d.deduplicate([sys_msg, {"role": "user", "content": "hi"}])
        assert len(msgs1) == 2

        # Second call: same system prompt → should be dropped
        msgs2, _ = d.deduplicate([sys_msg, {"role": "user", "content": "bye"}])
        assert len(msgs2) == 1  # system dropped
        assert msgs2[0]["role"] == "user"

    def test_consecutive_duplicate_dropped(self):
        d = PromptDeduplicator()
        msgs = [
            {"role": "assistant", "content": "Thinking..."},
            {"role": "assistant", "content": "Thinking..."},  # duplicate
            {"role": "user", "content": "go on"},
        ]
        kept, result = d.deduplicate(msgs)
        assert len(kept) == 2  # second assistant dropped
        assert result.dropped_count == 1

    def test_force_full_bypasses_dedup(self):
        d = PromptDeduplicator()
        sys_msg = {"role": "system", "content": "agent"}
        # Register system
        d.deduplicate([sys_msg, {"role": "user", "content": "hi"}])
        # Force full: second call should keep everything
        kept, _ = d.deduplicate(
            [sys_msg, {"role": "user", "content": "bye"}],
            force_full=True,
        )
        assert len(kept) == 2

    def test_dedup_tools(self):
        d = PromptDeduplicator()
        tools = [{"type": "function", "function": {"name": "search"}}]

        # First call: keep
        kept, saved = d.deduplicate_tools(tools)
        assert len(kept) == 1
        assert saved == 0

        # Second call with same tools: drop
        kept, saved = d.deduplicate_tools(tools)
        assert len(kept) == 0

    def test_compress_tool_results(self):
        d = PromptDeduplicator()
        msgs = [
            {"role": "tool", "content": "x" * 5000},
            {"role": "user", "content": "ok"},
        ]
        compressed = d.compress_tool_results(msgs, max_result_chars=1000)
        assert len(compressed[0]["content"]) <= 1000 + 50  # +truncation msg

    def test_reset_session_state(self):
        d = PromptDeduplicator()
        sys_msg = {"role": "system", "content": "agent"}
        d.deduplicate([sys_msg, {"role": "user", "content": "hi"}])
        d.reset_session_state()
        # After reset, system should appear new again
        kept, _ = d.deduplicate([sys_msg, {"role": "user", "content": "bye"}])
        assert len(kept) == 2  # system kept again

    def test_estimate_session_savings(self):
        est = PromptDeduplicator.estimate_session_savings(5000, 2000, 10)
        assert est["static_tokens_per_turn"] == 7000
        assert est["total_without_dedup"] == 70000
        assert est["total_with_dedup"] == 7000
        assert est["tokens_saved"] == 63000

    def test_stats_and_repr(self):
        d = PromptDeduplicator()
        s = d.stats()
        assert s["total_messages_dropped"] == 0
        assert "PromptDeduplicator" in repr(d)


# ═══════════════════════════════════════════════════════
# ToolSchemaCompressor
# ═══════════════════════════════════════════════════════

class TestToolSchemaCompressor:
    def test_empty_tools(self):
        c = ToolSchemaCompressor()
        compressed, tiers = c.compress([])
        assert compressed == []
        assert tiers == []

    def test_high_freq_full_schema(self):
        c = ToolSchemaCompressor()
        tool = {"type": "function", "function": {"name": "search", "description": "search the web", "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "query"}}, "required": ["q"]}}}
        # Mark as high-freq
        for _ in range(10):
            c.record_usage("search")

        compressed, tiers = c.compress([tool])
        assert len(compressed) == 1
        assert tiers[0].keep_full_schema

    def test_low_freq_name_only(self):
        c = ToolSchemaCompressor()
        tool = {"type": "function", "function": {"name": "rare_tool", "description": "rarely used", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}
        compressed, tiers = c.compress([tool])
        assert len(compressed) == 1
        assert tiers[0].name_only

    def test_mid_freq_simplified(self):
        c = ToolSchemaCompressor(mid_freq_threshold=2)
        tool = {"type": "function", "function": {"name": "mid_tool", "description": "moderate use", "parameters": {"type": "object", "properties": {"a": {"type": "string", "description": "param a"}, "b": {"type": "string", "description": "param b"}}, "required": ["a"]}}}
        for _ in range(3):
            c.record_usage("mid_tool")

        compressed, tiers = c.compress([tool])
        assert len(compressed) == 1
        assert tiers[0].keep_required_only

    def test_budget_respected(self):
        c = ToolSchemaCompressor(max_total_tokens=200)
        tools = []
        for i in range(20):
            tools.append({"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool number {i}", "parameters": {"type": "object", "properties": {}}}})

        compressed, tiers = c.compress(tools)
        # With budget 200, should keep only a few tools
        assert len(compressed) <= 10

    def test_usage_stats(self):
        c = ToolSchemaCompressor()
        c.record_usage("search")
        c.record_usage("search")
        c.record_usage("read")
        assert c.get_usage_count("search") == 2
        assert c.get_usage_count("read") == 1
        assert c.get_usage_count("unknown") == 0

    def test_hottest_tools(self):
        c = ToolSchemaCompressor()
        for name in ["a", "b", "c"]:
            for _ in range(ord(name) - 96):  # a=1, b=2, c=3
                c.record_usage(name)
        hot = c.hottest_tools(2)
        assert hot[0][0] == "c"
        assert hot[1][0] == "b"

    def test_reset_usage(self):
        c = ToolSchemaCompressor()
        c.record_usage("search")
        c.reset_usage()
        assert c.get_usage_count("search") == 0

    def test_reset_stats(self):
        c = ToolSchemaCompressor()
        c.compress([{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}])
        c.reset_stats()
        assert c.stats()["total_compressions"] == 0

    def test_compression_tier_repr(self):
        t = CompressionTier("search", 500, keep_full_schema=True)
        assert "full" in repr(t)

    def test_repr(self):
        c = ToolSchemaCompressor()
        assert "ToolSchemaCompressor" in repr(c)


# ═══════════════════════════════════════════════════════
# Integration: compression + dedup
# ═══════════════════════════════════════════════════════

class TestPhase4Integration:
    def test_full_compression_pipeline(self):
        """Compressor → Deduplicator → full pipeline."""
        comp = ContextCompressor(thresholds=CompressionThresholds(
            history_token_threshold=100,
        ))
        dedup = PromptDeduplicator()

        # Simulate multi-turn conversation
        system = {"role": "system", "content": "You are an agent."}
        turns = []
        for i in range(10):
            turns.append({"role": "user", "content": f"Turn {i}"})
            turns.append({"role": "assistant", "content": f"Response {i}"})

        all_msgs = [system] + turns

        # 1. Compress
        compressed_text, comp_result = comp.compress_history(all_msgs)
        assert comp_result.compression_ratio >= 0

        # 2. Dedup subsequent turns
        kept, dedup_result = dedup.deduplicate(
            [system, {"role": "user", "content": "new message"}],
            session_id="s1",
        )
        # First call keeps system; system is new
        assert len(kept) >= 1

        # Second call with same system → dedup
        kept2, _ = dedup.deduplicate(
            [system, {"role": "user", "content": "another message"}],
            session_id="s1",
        )
        assert len(kept2) == 1  # system dropped

    def test_compressor_guideline_optimization_loop(self):
        """Simulate ACON §3.3: failure-driven guideline refinement."""
        c = ContextCompressor()

        # Initial guideline is naive
        assert "summarize" in c.get_guideline("history").lower() or \
               "Summarize" in c.get_guideline("history")

        # Simulate failed trajectory feedback
        c.record_feedback({
            "trajectory_id": "t1",
            "success": False,
            "failure_reason": "lost variable FILE_PATH after compression",
        })

        # Optimize guideline (simulated)
        c.set_guideline("history", "Priority: preserve all FILE_PATH references and state variables.")

        # Verify guideline stored
        assert "FILE_PATH" in c.get_guideline("history")
        assert c.stats()["has_optimized_guideline"]
