#!/usr/bin/env python3
"""
ET-Agent vs vLLM / HiFC — Inference-level KV Cache Benchmark.

Simulates realistic LLM inference workloads (prefill + decoding steps)
using the VLLMBlockManager + HiFCSwappingEngine + KVCacheScheduler.
Produces a BenchmarkReport with vLLM-comparable metrics:

  - Throughput (tokens/s)
  - GPU memory utilization (%)
  - GPU waste rate (%)         ← key differentiator: vLLM ~60-80%, ET-Agent <5%
  - Swap count (GPU↔SSD)       ← HiFC comparable
  - TTFT / TBT (ms)
  - Memory expansion cost ($/3yr)

Usage
-----
    python scripts/benchmark_vllm.py                     # all scenarios
    python scripts/benchmark_vllm.py --scenario long     # long-context only
    python scripts/benchmark_vllm.py --compare-defaults  # print vLLM baseline
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_manager.config import MemoryConfig
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager

from inference.vllm_block_manager import VLLMBlockManager, BlockSwapOp
from inference.swapping_engine import HiFCSwappingEngine, FlashZone, ZONE_PERF
from inference.scheduler import KVCacheScheduler, ScheduleDecision
from inference.metrics import InferenceMetrics, BenchmarkReport


# ═══════════════════════════════════════════════════════════════════
# Benchmark scenarios (modeled after vLLM §6.1 and HiFC §5.1)
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = {
    "short": {
        "name": "Short Context (vLLM §6.1 ShareGPT)",
        "description": "Typical chatbot workload: ~1k input, ~200 output, batch 8",
        "num_sequences": 32,
        "input_tokens_range": (200, 2000),
        "output_tokens": 200,
        "batch_size": 8,
        "swap_threshold_gpu_pct": 0.85,
    },
    "long": {
        "name": "Long Context (HiFC §5.1 NarrativeQA)",
        "description": "Long-context QA: ~32k input, ~1k output, batch 4",
        "num_sequences": 16,
        "input_tokens_range": (8000, 64000),
        "output_tokens": 1024,
        "batch_size": 4,
        "swap_threshold_gpu_pct": 0.80,
    },
    "heavy_swap": {
        "name": "Heavy Swap Stress (HiFC Fig.3)",
        "description": "Small GPU, large batch → frequent swaps",
        "num_sequences": 64,
        "input_tokens_range": (4096, 16384),
        "output_tokens": 512,
        "batch_size": 16,
        "swap_threshold_gpu_pct": 0.60,  # trigger swaps early
    },
    "agent": {
        "name": "Agent Workload (ET-Agent scenario)",
        "description": "Multi-turn agent with tool calls: 4 sessions × 50 turns",
        "num_sequences": 4,   # 4 concurrent sessions
        "input_tokens_range": (500, 5000),  # per-turn user message
        "output_tokens": 300,
        "batch_size": 4,
        "swap_threshold_gpu_pct": 0.80,
        "agent_mode": True,
    },
}


# ═══════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════

def run_benchmark(
    scenario: dict,
    model_name: str = "qwen2.5-7b",
    gpu_gb: int = 6,
    gds_bandwidth_gbps: float = 4.7,
) -> BenchmarkReport:
    """Execute one benchmark scenario and return a report."""

    config = MemoryConfig.for_model(model_name, block_size=16, gpu_gb=gpu_gb)
    allocator = KVBlockAllocator(config)
    block_tables = BlockTableManager(allocator, config.block_size)
    swapping = HiFCSwappingEngine(
        num_io_threads=8,
        block_size_bytes=config.block_size_bytes,
        default_zone=FlashZone.pSLC,
    )

    vllm_bm = VLLMBlockManager(
        config, allocator, block_tables,
        enable_fc=True,
        gds_bandwidth_gbps=gds_bandwidth_gbps,
    )

    scheduler = KVCacheScheduler(
        vllm_bm, swapping,
        max_batch_size=scenario["batch_size"],
        watermark_gpu_pct=scenario["swap_threshold_gpu_pct"],
    )

    metrics = InferenceMetrics(
        vllm_bm, swapping,
        model_name=model_name,
        gpu_name=f"NVIDIA GPU {gpu_gb}GB",
    )

    print(f"\n{'='*60}")
    print(f"  {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  GPU blocks: {allocator.total_blocks} ({config.block_size} tok/block)")
    print(f"  FC blocks:  {vllm_bm._fc_capacity}")
    print(f"  GDS bandwidth: {gds_bandwidth_gbps} GiB/s (pSLC)")
    print(f"{'='*60}")

    # ── Create sequences ──
    num_seqs = scenario["num_sequences"]
    ir_min, ir_max = scenario["input_tokens_range"]
    output_len = scenario["output_tokens"]

    import random
    random.seed(42)

    seq_inputs = {}
    for i in range(num_seqs):
        input_tok = random.randint(ir_min, ir_max)
        seq_inputs[i] = input_tok

    allocated = 0
    for seq_id, input_tok in seq_inputs.items():
        if vllm_bm.can_allocate(input_tok):
            vllm_bm.allocate(seq_id, input_tok)
            allocated += 1
        else:
            # Queue via scheduler
            scheduler.add_sequence(seq_id, input_tok)

    print(f"  Allocated {allocated}/{num_seqs} sequences")
    print(f"  GPU utilization: {allocator.usage_ratio:.1%}")

    # ── Simulate decoding loop ──
    completed = set()
    max_steps = output_len
    total_swaps_out = 0

    for step in range(max_steps):
        # Snapshot every 10% of steps
        if step % max(1, max_steps // 20) == 0:
            metrics.snapshot()

        # Schedule
        decision = scheduler.schedule()
        total_swaps_out += len(decision.swap_out_seqs)

        # Simulate decoding for each scheduled sequence
        for seq_id in decision.scheduled_seqs:
            if seq_id in completed:
                continue

            # Append a token
            new_bid = vllm_bm.append_slot(seq_id)

            # Record metrics
            if step == 0:
                metrics.record_first_token()
            else:
                metrics.record_token()

            # Check if sequence is done
            current_blocks = len(vllm_bm.get_block_table(seq_id))
            est_tokens = current_blocks * config.block_size
            if est_tokens >= seq_inputs.get(seq_id, 0) + output_len:
                completed.add(seq_id)
                vllm_bm.free(seq_id)

        # Replenish with new sequences
        next_id = num_seqs + step
        if next_id < num_seqs + 20:  # add some ongoing demand
            input_tok = random.randint(ir_min, ir_max)
            if vllm_bm.can_allocate(input_tok):
                vllm_bm.allocate(next_id, input_tok)
                seq_inputs[next_id] = input_tok

    # Final snapshot
    metrics.snapshot()

    # ── Build report ──
    report = metrics.build_report()
    report.total_swaps_out = total_swaps_out
    report.total_swap_bytes = swapping._total_bytes_transferred

    return report


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ET-Agent vs vLLM / HiFC — KV Cache Benchmark"
    )
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"],
                        default="all", help="Which scenario to run")
    parser.add_argument("--model", default="qwen2.5-7b", help="Model name")
    parser.add_argument("--gpu-gb", type=int, default=6, help="GPU VRAM (GB)")
    parser.add_argument("--gds", type=float, default=4.7,
                        help="GDS bandwidth in GiB/s (HiFC default: 4.7)")
    parser.add_argument("--output", default="benchmark_vllm_report.json",
                        help="Output JSON path")
    parser.add_argument("--compare-defaults", action="store_true",
                        help="Print vLLM and HiFC baseline values for reference")
    args = parser.parse_args()

    if args.compare_defaults:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║           vLLM / HiFC / ET-Agent — Reference Values          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Metric              │ vLLM (baseline)  │ HiFC          │ ET ║
  ║─────────────────────────────────────────────────────────────╣
  ║  GPU waste rate      │ ~60-80%          │ ~3.7%         │ <5%║
  ║  GPU utilization     │ ~20-40%          │ ~96.3%        │ —  ║
  ║  Swap medium         │ DRAM             │ pSLC SSD      │ SSD║
  ║  Swap bandwidth      │ ~50 GiB/s        │ ~4.7 GiB/s    │ —  ║
  ║  Memory exp. cost/3yr│ ~$614            │ ~$136         │ —  ║
  ║  Write amplification │ N/A              │ ~1.02         │ —  ║
  ╚══════════════════════════════════════════════════════════════╝
  Reference: vLLM (Kwon et al., SOSP 2023), HiFC (Jeong et al., 2025)
        """)
        return

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    all_reports = {}
    for key in scenarios:
        sc = SCENARIOS[key]
        report = run_benchmark(sc, model_name=args.model, gpu_gb=args.gpu_gb,
                               gds_bandwidth_gbps=args.gds)
        all_reports[key] = report.to_dict()
        report.print_summary()

    # Save JSON
    output = {"metadata": {"model": args.model, "gpu_gb": args.gpu_gb,
                           "gds_gbps": args.gds},
              "scenarios": all_reports}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Report saved to {args.output}")

    # Save legacy format too (for monitor dashboard compatibility)
    dashboard_path = "benchmark_results.json"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[OK] Dashboard data saved to {dashboard_path}")


if __name__ == "__main__":
    main()
