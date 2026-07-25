#!/usr/bin/env python3
"""
ET-Agent Local Inference Benchmark — Real Qwen3-0.6B on GPU.

Loads a local HuggingFace model, runs multi-turn agent-style conversations,
and measures GPU KV Cache memory management with our memory_manager.

Compares:
  Baseline (transformers default past_key_values) vs ET-Agent (tracked blocks)

Usage
-----
  python scripts/benchmark_local.py                     # quick test (3 turns)
  python scripts/benchmark_local.py --turns 20           # long conversation
  python scripts/benchmark_local.py --compare-baseline   # A/B comparison
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path

import torch
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_manager.config import MemoryConfig, ModelKVProfile
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager
from memory_manager.kv_prefix_cache import PrefixHashCache, compute_prefix_hashes
from memory_manager.agent_prefix_cache import AgentPrefixCache

# ═══════════════════════════════════════════════════════════════
# Qwen3-0.6B profile
# ═══════════════════════════════════════════════════════════════

QWEN3_06B_PROFILE = ModelKVProfile(
    model_family="qwen3",
    num_layers=28,
    num_kv_heads=8,
    head_dim=128,
    bytes_per_element=2,  # bfloat16 = 2 bytes
)


def load_local_model(model_path: str):
    """Load a local HuggingFace model into GPU memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[LOAD] Loading model from {model_path} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"[LOAD] Model loaded in {elapsed:.1f}s")
    print(f"  VRAM used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    return model, tokenizer


def print_gpu_stats(label: str = ""):
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f"  [{label}] GPU: {allocated:.0f}MB alloc / {reserved:.0f}MB reserved | RAM: {ram:.0f}MB")


def run_single_turn(model, tokenizer, messages: list, max_new_tokens: int = 128):
    """Run one conversation turn with the local model.

    Returns: (response_text, kv_cache_tensors, gpu_memory_delta)
    """
    torch.cuda.empty_cache()
    gpu_before = torch.cuda.memory_allocated(0)

    # Build prompt from messages
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs.sequences[0][input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    gpu_after = torch.cuda.memory_allocated(0)
    gpu_delta_mb = (gpu_after - gpu_before) / 1024**2

    # Count tokens
    prompt_tokens = input_len
    output_tokens = len(response_ids)

    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "gpu_delta_mb": gpu_delta_mb,
        "gpu_after_mb": gpu_after / 1024**2,
    }


def run_agent_conversation(
    model, tokenizer,
    turns: int = 10,
    system_prompt: str = "You are a helpful AI assistant. Answer concisely.",
    use_memory_manager: bool = True,
) -> dict:
    """Simulate a multi-turn agent conversation and measure KV Cache behavior.

    With use_memory_manager=True:
      - System prompt blocks are allocated and pinned in our KVBlockAllocator
      - Prefix hash-cache tracks reused tokens across turns
      - GPU memory is tracked per-turn

    With use_memory_manager=False:
      - Baseline: transformers default past_key_values caching
    """

    config = MemoryConfig(
        block_size=16,
        gpu_capacity_bytes=6 * 1024**3,  # 6GB for KV (model takes ~1.2GB)
        use_cuda=True,
        model_profile=QWEN3_06B_PROFILE,
    )

    allocator = KVBlockAllocator(config) if use_memory_manager else None
    tables = BlockTableManager(allocator, config.block_size) if allocator else None
    prefix_cache = PrefixHashCache(block_size=16, max_entries=100000) if allocator else None
    agent_cache = AgentPrefixCache(prefix_cache) if prefix_cache else None

    results = {
        "turns": [],
        "gpu_memory_snapshots": [],
        "prefix_hits": [],
    }

    # Tokenize system prompt for prefix caching
    if use_memory_manager and prefix_cache:
        sys_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
        sys_blocks = allocator.allocate("system", len(sys_tokens), group_id="system")
        tables.create_table("system")
        tables.get_table("system").append_blocks(sys_blocks, tokens_per_block=[16]*len(sys_blocks))
        prefix_cache.insert_range(sys_tokens, sys_blocks, is_pinned=True)
        allocator.pin_blocks("system-prompt", sys_blocks)
        agent_cache.cache_system_prompt("default", sys_tokens, sys_blocks)

    messages = [{"role": "system", "content": system_prompt}]

    for turn_idx in range(turns):
        turn_start = time.time()

        # Build user message
        user_msg = f"Turn {turn_idx + 1}: Tell me a short fact about AI."
        if turn_idx > 0:
            user_msg = f"Turn {turn_idx + 1}: Tell me another fact, different from the previous ones."

        messages.append({"role": "user", "content": user_msg})

        # Run inference
        turn_result = run_single_turn(model, tokenizer, messages, max_new_tokens=64)
        messages.append({"role": "assistant", "content": turn_result["response"]})

        turn_elapsed = time.time() - turn_start

        # GPU snapshot
        gpu_snap = {
            "turn": turn_idx + 1,
            "gpu_allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
            "ram_rss_mb": psutil.Process().memory_info().rss / 1024**2,
            "elapsed_s": round(turn_elapsed, 2),
            "prompt_tokens": turn_result["prompt_tokens"],
            "output_tokens": turn_result["output_tokens"],
        }
        results["gpu_memory_snapshots"].append(gpu_snap)

        # Prefix cache lookup
        if use_memory_manager and prefix_cache:
            full_text = system_prompt + " " + " ".join(
                m["content"] for m in messages if m["role"] in ("user", "assistant")
            )
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len, _, _ = prefix_cache.find_longest_prefix(token_ids)
            hit_rate = prefix_len / max(len(token_ids), 1)
            results["prefix_hits"].append({
                "turn": turn_idx + 1,
                "total_tokens": len(token_ids),
                "prefix_hit_tokens": prefix_len,
                "hit_rate": round(hit_rate, 4),
            })

        # Per-turn summary
        summary = {
            "turn": turn_idx + 1,
            "response_preview": turn_result["response"][:80],
            **gpu_snap,
        }
        if use_memory_manager and allocator:
            summary.update({
                "allocator_used_blocks": allocator.used_blocks,
                "allocator_free_blocks": allocator.free_blocks,
                "prefix_cache_entries": prefix_cache.size,
                "prefix_hit_rate": round(prefix_cache.hit_rate, 4),
            })
        results["turns"].append(summary)

        print(f"  Turn {turn_idx + 1:2d}: {turn_result['output_tokens']:3d} tok out, "
              f"GPU={gpu_snap['gpu_allocated_mb']:.0f}MB, "
              f"{turn_elapsed:.1f}s", end="")
        if use_memory_manager and prefix_cache:
            print(f", prefix_hit={results['prefix_hits'][-1]['hit_rate']:.1%}")
        else:
            print()

    # Final snapshot
    final_gpu = torch.cuda.memory_allocated(0) / 1024**2
    results["final_gpu_mb"] = final_gpu
    results["final_ram_mb"] = psutil.Process().memory_info().rss / 1024**2

    if use_memory_manager and allocator:
        results["allocator_stats"] = allocator.stats()
        results["prefix_cache_stats"] = prefix_cache.stats()
        allocator.free("system")
        for turn in range(turns):
            allocator.free(f"turn-{turn}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ET-Agent Local Inference Benchmark")
    parser.add_argument("--model", default="D:/etern/工作/格物/博士工作/ETERN_Claude/ETAgent/models/Qwen3-0.6B",
                        help="Path to local HuggingFace model")
    parser.add_argument("--turns", type=int, default=5, help="Number of conversation turns")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Run both baseline and ET-Agent, compare")
    parser.add_argument("--output", default="benchmark_local_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ET-Agent Local Inference Benchmark                      ║")
    print(f"║  Model: Qwen3-0.6B | GPU: {torch.cuda.get_device_name(0)}")
    print(f"║  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print("╚══════════════════════════════════════════════════════════╝")

    model, tokenizer = load_local_model(args.model)
    print_gpu_stats("After model load")

    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("  BASELINE (transformers default — no memory_manager)")
        print("=" * 60)
        baseline = run_agent_conversation(
            model, tokenizer, turns=args.turns, use_memory_manager=False,
        )

        torch.cuda.empty_cache()
        print_gpu_stats("After baseline cleanup")

        print("\n" + "=" * 60)
        print("  ET-AGENT (with memory_manager KV Cache tracking)")
        print("=" * 60)
        et_agent = run_agent_conversation(
            model, tokenizer, turns=args.turns, use_memory_manager=True,
        )

        # Compare
        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"  {'Metric':<35} {'Baseline':>12} {'ET-Agent':>12} {'Delta':>10}")
        print(f"  {'─'*69}")
        print(f"  {'Final GPU (MB)':<35} {baseline['final_gpu_mb']:>12.1f} {et_agent['final_gpu_mb']:>12.1f} {et_agent['final_gpu_mb'] - baseline['final_gpu_mb']:>+10.1f}")
        print(f"  {'Final RAM (MB)':<35} {baseline['final_ram_mb']:>12.0f} {et_agent['final_ram_mb']:>12.0f} {et_agent['final_ram_mb'] - baseline['final_ram_mb']:>+10.0f}")

        if et_agent.get("prefix_cache_stats"):
            pcs = et_agent["prefix_cache_stats"]
            print(f"  {'Prefix Cache Entries':<35} {'—':>12} {pcs['total_entries']:>12} {'—':>10}")
            print(f"  {'Prefix Hit Rate':<35} {'—':>12} {pcs['hit_rate']:>11.1%} {'—':>10}")

        output = {"baseline": baseline, "et_agent": et_agent}
    else:
        print(f"\n{'='*60}")
        print(f"  ET-AGENT MODE ({args.turns} turns)")
        print(f"{'='*60}")
        et_agent = run_agent_conversation(
            model, tokenizer, turns=args.turns, use_memory_manager=True,
        )
        output = et_agent

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
