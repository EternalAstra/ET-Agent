"""Tests for memory_manager.memory_monitor chart export."""

from agent.memory_hooks import create_agent_memory_manager
from memory_manager.memory_monitor import MemoryMonitor


def test_peak_gpu_blocks_non_negative():
    mgr = create_agent_memory_manager("qwen2.5:3b-instruct", gpu_gb=6)
    monitor = MemoryMonitor(mgr)

    mgr.on_session_start("s1", system_prompt_tokens=list(range(500)))
    mgr.pre_llm_call("s1", [{"role": "user", "content": "hello"}])
    monitor.snapshot()

    charts = monitor.export_chart_data()
    peak = charts["summary"]["peak_gpu_blocks"]
    assert peak >= 0, f"peak_gpu_blocks should be non-negative, got {peak}"


def test_gpu_blocks_used_matches_tier_not_free_pool():
    mgr = create_agent_memory_manager("qwen2.5:3b-instruct", gpu_gb=6)
    monitor = MemoryMonitor(mgr)

    mgr.on_session_start("s1", system_prompt_tokens=list(range(200)))
    snap = monitor.snapshot()

    charts = monitor.export_chart_data()
    assert charts["series"]["gpu_blocks_used"] == [snap.blocks.gpu_blocks]
    assert charts["series"]["gpu_blocks_used"][0] >= 0
