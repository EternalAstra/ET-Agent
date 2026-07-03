#!/usr/bin/env python
"""ET-Agent Memory Manager Benchmark Runner.

Simulates agent workloads and collects memory statistics for visualization.
Exports JSON data consumed by the monitoring dashboard (web/monitor/index.html).

Usage
-----
    python scripts/benchmark_memory.py                    # run all scenarios
    python scripts/benchmark_memory.py --scenario chat   # single scenario
    python scripts/benchmark_memory.py --export-only     # re-export dashboard
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.memory_hooks import create_agent_memory_manager
from memory_manager.memory_monitor import MemoryMonitor
from memory_manager.kv_block import StorageTier


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "chat": {
        "name": "Multi-turn Chat (20 turns)",
        "description": "Simulate 20-turn conversation with memory and search tools",
        "sessions": 2,
        "turns": 20,
        "tools": ["memory", "session_search"],
        "system_prompt_tokens": 3000,
        "avg_user_tokens": 500,
        "avg_tool_result_tokens": 1200,
        "tool_call_probability": 0.3,
    },
    "tool_chain": {
        "name": "Tool Chain (search→read→analyze→write)",
        "description": "Multi-step tool execution pipeline",
        "sessions": 2,
        "turns": 10,
        "tools": ["web_search", "read_file", "write_file", "terminal"],
        "system_prompt_tokens": 4000,
        "avg_user_tokens": 300,
        "avg_tool_result_tokens": 2500,
        "tool_call_probability": 0.8,
    },
    "long_context": {
        "name": "Long Context (128K tokens)",
        "description": "Agent with very long accumulation, triggers ACON compression",
        "sessions": 1,
        "turns": 50,
        "tools": ["read_file", "session_search", "web_search"],
        "system_prompt_tokens": 5000,
        "avg_user_tokens": 2000,
        "avg_tool_result_tokens": 4000,
        "tool_call_probability": 0.6,
    },
    "parallel": {
        "name": "Parallel Sessions (8 concurrent)",
        "description": "8 agent sessions running simultaneously, sharing system prompt",
        "sessions": 8,
        "turns": 15,
        "tools": ["memory", "web_search", "read_file", "terminal"],
        "system_prompt_tokens": 3500,
        "avg_user_tokens": 600,
        "avg_tool_result_tokens": 1500,
        "tool_call_probability": 0.5,
    },
    "lifecycle": {
        "name": "Lifecycle Stress (prefill→decode→tool→idle→complete)",
        "description": "Rapid phase transitions to stress hierarchical storage",
        "sessions": 4,
        "turns": 30,
        "tools": ["search_files", "read_file", "terminal"],
        "system_prompt_tokens": 3000,
        "avg_user_tokens": 400,
        "avg_tool_result_tokens": 2000,
        "tool_call_probability": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _token_ids(n: int) -> list:
    """Generate pseudo-token IDs for simulation."""
    import random
    n = max(1, abs(n))
    random.seed(n)
    return [random.randint(0, 50000) for _ in range(n)]


def _simulate_tools(tool_names: list) -> list:
    """Generate simulated tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Simulated {name} tool for benchmark.",
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}},
            },
        }
        for name in tool_names
    ]


def run_scenario(mgr, scenario: dict, scenario_key: str) -> dict:
    """Run one benchmark scenario against the memory manager."""
    monitor = MemoryMonitor(mgr)
    monitor.start(interval_s=1.0)

    sess_ids = [f"{scenario_key}-{i}" for i in range(scenario["sessions"])]
    tools = _simulate_tools(scenario["tools"])

    # Start sessions
    for sid in sess_ids:
        sp = _token_ids(scenario["system_prompt_tokens"])
        mgr.on_session_start(sid, system_prompt_tokens=sp, tool_definitions=tools)

    # Run turns
    for turn in range(scenario["turns"]):
        for sid in sess_ids:
            # Pre-LLM
            user_tokens = _token_ids(scenario["avg_user_tokens"])
            msgs = [
                {"role": "system", "content": "Agent system prompt (cached)"},
                {"role": "user", "content": f"Turn {turn}: " + " ".join(str(t) for t in user_tokens[:50])},
            ]
            mgr.pre_llm_call(sid, msgs)

            # Post-LLM: tool call or plain response
            # hash(sid) can be huge; mask to avoid MemoryError in _token_ids()
            _seed = (turn * 137 + (hash(sid) & 0xFFFF)) % 10000
            if _token_ids(_seed)[0] % 100 < scenario["tool_call_probability"] * 100:
                # Tool call
                mgr.post_llm_call(
                    sid,
                    assistant_message={
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": scenario["tools"][0], "arguments": "{}"}}],
                    },
                    has_tool_calls=True,
                )
                # Tool result arrives after simulated delay
                mgr.on_tool_result(sid, tool_name=scenario["tools"][0])
                # Allocate tool result blocks
                result_tokens = _token_ids(scenario["avg_tool_result_tokens"])
                blocks = mgr.allocator.allocate(sid, len(result_tokens), group_id=sid)
                mgr.hierarchical_store.register_blocks(blocks)
            else:
                mgr.post_llm_call(sid, has_tool_calls=False)

        # Periodic compression check (every 10 turns)
        if turn % 10 == 0 and turn > 0:
            for sid in sess_ids:
                msgs = [{"role": "user", "content": f"Turn {t}: content"} for t in range(turn)]
                total_tokens = sum(len(str(m)) for m in msgs)
                mgr.maybe_compress(sid, msgs, total_tokens, 100000)

    # End sessions
    for sid in sess_ids:
        mgr.on_session_end(sid)

    time.sleep(2)  # let final snapshots collect
    monitor.stop()

    return {
        "scenario": scenario["name"],
        "snapshots": [s.to_dict() for s in monitor._snapshots],
        "charts": monitor.export_chart_data(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ET-Agent Memory Benchmark")
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"],
                        default="all", help="Which scenario to run")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--dashboard-dir", default="web/monitor",
                        help="Dashboard directory to write chart data")
    parser.add_argument("--model", default="qwen2.5-7b", help="Model name")
    parser.add_argument("--gpu-gb", type=int, default=80,
                        help="GPU VRAM in GB")
    args = parser.parse_args()

    scenarios_to_run = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    all_results = {"metadata": {"model": args.model, "gpu_gb": args.gpu_gb}, "scenarios": {}}

    _ref = create_agent_memory_manager(args.model, gpu_gb=args.gpu_gb)
    print(f"Initializing memory manager for {args.model} ({args.gpu_gb}GB GPU)...")
    print(f"  Blocks: {_ref.allocator.total_blocks} total, "
          f"{_ref._config.block_size} tokens/block")
    del _ref

    for key in scenarios_to_run:
        sc = SCENARIOS[key]
        print(f"\n{'='*60}")
        print(f"Running: {sc['name']}")
        print(f"  {sc['description']}")
        print(f"  {sc['sessions']} sessions × {sc['turns']} turns")
        print(f"  Tools: {sc['tools']}")

        # Fresh manager per scenario — reset_stats() does not free blocks
        mgr = create_agent_memory_manager(args.model, gpu_gb=args.gpu_gb)
        result = run_scenario(mgr, sc, key)
        all_results["scenarios"][key] = result

        snaps = result["snapshots"]
        print(f"  Collected {len(snaps)} snapshots")
        if snaps:
            summary = result["charts"]["summary"]
            print(f"  Peak GPU blocks:    {summary['peak_gpu_blocks']}")
            print(f"  Avg prefix hit rate: {summary['avg_prefix_hit_rate']:.1%}")
            print(f"  Total migrations:    {summary['total_migrations']}")
            print(f"  Tokens saved:        {summary['total_tokens_saved']}")

    # Export
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results exported to {output_path}")

    # Write chart data for dashboard
    dash_dir = Path(args.dashboard_dir)
    dash_dir.mkdir(parents=True, exist_ok=True)
    chart_path = dash_dir / "benchmark_data.js"
    with open(chart_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by benchmark_memory.py\n")
        f.write("window.BENCHMARK_DATA = ")
        json.dump(all_results, f, indent=2, default=str)
        f.write(";\n")
    print(f"[OK] Dashboard data written to {chart_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'Sessions':>8} {'Snapshots':>10} {'HitRate':>8} {'Migrations':>11}")
    print("-" * 60)
    for key in scenarios_to_run:
        r = all_results["scenarios"][key]
        n = len(r["snapshots"])
        hr = r["charts"]["summary"]["avg_prefix_hit_rate"]
        mg = r["charts"]["summary"]["total_migrations"]
        print(f"{SCENARIOS[key]['name']:<20} {SCENARIOS[key]['sessions']:>8} {n:>10} {hr:>7.1%} {mg:>11}")


if __name__ == "__main__":
    main()
