#!/usr/bin/env python3
"""
ET-Agent vs Original Hermes — Local Inference Benchmark.

Compares our paged KV Cache (via ETPagedCache + KVBlockAllocator)
against the original Hermes baseline (transformers default DynamicCache).

Usage
-----
  python scripts/benchmark_local.py --turns 20      # single test
  python scripts/benchmark_local.py --concurrent 4   # 4 concurrent sessions
  python scripts/benchmark_local.py --compare        # A/B comparison
"""

from __future__ import annotations

import json, os, sys, time, argparse, threading
from pathlib import Path

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_manager.config import MemoryConfig, ModelKVProfile
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager
from memory_manager.kv_prefix_cache import PrefixHashCache
from inference.et_cache import ETPagedCache

QWEN3_06B = ModelKVProfile("qwen3", 28, 8, 128, 2)
MODEL_PATH = "D:/etern/工作/格物/博士工作/ETERN_Claude/ETAgent/models/Qwen3-0.6B"


# ═══════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════

def load_model():
    print(f"[LOAD] Qwen3-0.6B from {MODEL_PATH} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[LOAD] {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    return model, tokenizer


# ═══════════════════════════════════════════════════════
# Inference runners — one per cache backend
# ═══════════════════════════════════════════════════════

def run_with_et_cache(model, tokenizer, messages: list,
                      allocator, block_tables,
                      max_new: int = 64) -> dict:
    """Run one turn with our ETPagedCache (PagedAttention backend)."""

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Create our paged KV cache
    cache = ETPagedCache(
        allocator=allocator,
        num_layers=QWEN3_06B.num_layers,
        block_size=16,
    )

    gpu_before = torch.cuda.memory_allocated(0)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            past_key_values=cache, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    gpu_after = torch.cuda.memory_allocated(0)

    response_ids = outputs[0][input_len:] if isinstance(outputs, torch.Tensor) else outputs.sequences[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "response": response[:120],
        "prompt_tokens": input_len,
        "output_tokens": len(response_ids),
        "gpu_before_mb": gpu_before / 1024**2,
        "gpu_after_mb": gpu_after / 1024**2,
        "gpu_delta_mb": (gpu_after - gpu_before) / 1024**2,
        "elapsed_s": round(elapsed, 2),
        "cache_blocks": cache.total_blocks_per_layer,
        "allocator_used": allocator.used_blocks,
        "allocator_vram_mb": allocator.cuda_bytes_allocated / 1024**2,
    }


def run_with_dynamic_cache(model, tokenizer, messages: list,
                           max_new: int = 64) -> dict:
    """Run one turn with transformers default DynamicCache (= original Hermes)."""

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    cache = DynamicCache()

    gpu_before = torch.cuda.memory_allocated(0)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            past_key_values=cache, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    gpu_after = torch.cuda.memory_allocated(0)

    response_ids = outputs[0][input_len:] if isinstance(outputs, torch.Tensor) else outputs.sequences[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "response": response[:120],
        "prompt_tokens": input_len,
        "output_tokens": len(response_ids),
        "gpu_before_mb": gpu_before / 1024**2,
        "gpu_after_mb": gpu_after / 1024**2,
        "gpu_delta_mb": (gpu_after - gpu_before) / 1024**2,
        "elapsed_s": round(elapsed, 2),
        "backend": "DynamicCache (original Hermes)",
    }


# ═══════════════════════════════════════════════════════
# Full session runner
# ═══════════════════════════════════════════════════════

def run_session(model, tokenizer, backend: str, turns: int,
                sys_prompt: str, allocator=None, block_tables=None) -> dict:
    """Run a full multi-turn conversation."""

    messages = [{"role": "system", "content": sys_prompt}]
    results = []

    for t in range(turns):
        user_msg = f"Turn {t+1}: Tell me a short fact about AI." if t == 0 else \
                   f"Turn {t+1}: Tell me another fact, different from before."
        messages.append({"role": "user", "content": user_msg})

        if backend == "et-agent":
            r = run_with_et_cache(model, tokenizer, messages,
                                  allocator, block_tables)
        else:
            r = run_with_dynamic_cache(model, tokenizer, messages)

        messages.append({"role": "assistant", "content": r["response"]})
        r["turn"] = t + 1
        results.append(r)

        ram_mb = psutil.Process().memory_info().rss / 1024**2
        print(f"  Turn {t+1:2d}: {r['output_tokens']:3d} tok, "
              f"GPU={r['gpu_after_mb']:.0f}MB, RAM={ram_mb:.0f}MB, {r['elapsed_s']:.1f}s",
              end="")
        if backend == "et-agent":
            print(f", alloc_vram={r['allocator_vram_mb']:.1f}MB")
        else:
            print()

    # Final stats
    final_gpu = torch.cuda.memory_allocated(0) / 1024**2
    final_ram = psutil.Process().memory_info().rss / 1024**2

    return {
        "backend": backend,
        "turns": results,
        "final_gpu_mb": final_gpu,
        "final_ram_mb": final_ram,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_elapsed_s": sum(r["elapsed_s"] for r in results),
    }


# ═══════════════════════════════════════════════════════
# Concurrent session runner
# ═══════════════════════════════════════════════════════

def run_concurrent(model, tokenizer, num_sessions: int, turns: int):
    """Run N concurrent sessions sharing a system prompt prefix."""

    config = MemoryConfig(block_size=16, gpu_capacity_bytes=6*1024**3,
                          use_cuda=True, model_profile=QWEN3_06B)
    allocator = KVBlockAllocator(config)
    block_tables = BlockTableManager(allocator, config.block_size)

    sys_prompt = "You are a helpful AI assistant. Be concise."
    sys_tokens = tokenizer.encode(sys_prompt, add_special_tokens=False)

    # Pre-cache system prompt in COW-shared blocks
    print(f"  System prompt: {len(sys_tokens)} tokens → "
          f"{(len(sys_tokens)+15)//16} blocks (shared by all sessions)")

    sys_ids = allocator.allocate("system-prompt", len(sys_tokens), group_id="system")
    block_tables.create_table("system-prompt")
    block_tables.get_table("system-prompt").append_blocks(
        sys_ids, tokens_per_block=[16]*len(sys_ids)
    )
    allocator.pin_blocks("system-prompt", sys_ids)

    # Shared prefix cache for hit-rate tracking
    pc = PrefixHashCache(block_size=16)

    results = []
    for sid in range(num_sessions):
        print(f"\n  Session {sid+1}/{num_sessions} ...")
        r = run_session(model, tokenizer, "et-agent", turns,
                        sys_prompt, allocator, block_tables)
        r["session_id"] = sid + 1
        results.append(r)

    allocator.free("system-prompt")
    for bid in range(allocator.total_gpu_blocks):
        if bid < allocator.total_gpu_blocks:
            gpu_blk = allocator.get_block(bid)
            if gpu_blk._tensor is not None:
                del gpu_blk._tensor
                gpu_blk._tensor = None
    torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ET-Agent vs Hermes Local Benchmark")
    parser.add_argument("--turns", type=int, default=5, help="Turns per session")
    parser.add_argument("--compare", action="store_true",
                        help="A/B compare: ET-Agent vs original Hermes")
    parser.add_argument("--concurrent", type=int, default=0,
                        help="Run N concurrent sessions (shared prefix)")
    parser.add_argument("--output", default="benchmark_local_results.json")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  ET-Agent vs Original Hermes — Local Benchmark        ║")
    print(f"║  GPU: {torch.cuda.get_device_name(0)}")
    print(f"║  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"║  Model: Qwen3-0.6B (28 layers, 8 KV heads, GQA)")
    print("╚══════════════════════════════════════════════════════╝")

    model, tokenizer = load_model()
    sys_prompt = "You are a helpful AI assistant. Be concise."

    output = {}

    if args.concurrent > 0:
        print(f"\n{'='*60}")
        print(f"  CONCURRENT SESSIONS ({args.concurrent} × {args.turns} turns)")
        print(f"{'='*60}")
        concurrent = run_concurrent(model, tokenizer, args.concurrent, args.turns)
        output["concurrent"] = concurrent

    if args.compare:
        config = MemoryConfig(block_size=16, gpu_capacity_bytes=6*1024**3,
                              use_cuda=True, model_profile=QWEN3_06B)
        allocator = KVBlockAllocator(config)
        block_tables = BlockTableManager(allocator, config.block_size)
        # Pin system prompt blocks for sharing
        sys_tokens = tokenizer.encode(sys_prompt, add_special_tokens=False)
        sys_blocks = allocator.allocate("sys", len(sys_tokens), group_id="sys")
        block_tables.create_table("sys")
        block_tables.get_table("sys").append_blocks(
            sys_blocks, tokens_per_block=[16]*len(sys_blocks))
        allocator.pin_blocks("sys-prompt", sys_blocks)

        print(f"\n{'='*60}")
        print(f"  BASELINE: Original Hermes (DynamicCache, {args.turns} turns)")
        print(f"{'='*60}")
        torch.cuda.empty_cache()
        baseline = run_session(model, tokenizer, "hermes", args.turns, sys_prompt)

        print(f"\n{'='*60}")
        print(f"  ET-AGENT: Paged KV Cache ({args.turns} turns)")
        print(f"{'='*60}")
        torch.cuda.empty_cache()
        et_agent = run_session(model, tokenizer, "et-agent", args.turns,
                               sys_prompt, allocator, block_tables)

        # Comparison summary
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Metric':<35} {'Hermes':>12} {'ET-Agent':>12} {'Delta':>10}")
        print(f"  {'─'*69}")
        print(f"  {'Final GPU (MB)':<35} {baseline['final_gpu_mb']:>12.1f} {et_agent['final_gpu_mb']:>12.1f} {et_agent['final_gpu_mb']-baseline['final_gpu_mb']:>+10.1f}")
        print(f"  {'Final RAM (MB)':<35} {baseline['final_ram_mb']:>12.0f} {et_agent['final_ram_mb']:>12.0f} {et_agent['final_ram_mb']-baseline['final_ram_mb']:>+10.0f}")
        print(f"  {'Total elapsed (s)':<35} {baseline['total_elapsed_s']:>12.1f} {et_agent['total_elapsed_s']:>12.1f} {et_agent['total_elapsed_s']-baseline['total_elapsed_s']:>+10.1f}")
        b_gpu = [t['gpu_after_mb'] for t in baseline['turns']]
        e_gpu = [t['gpu_after_mb'] for t in et_agent['turns']]
        print(f"  {'Peak GPU (MB)':<35} {max(b_gpu):>12.1f} {max(e_gpu):>12.1f} {max(e_gpu)-max(b_gpu):>+10.1f}")
        print(f"  {'Avg GPU (MB)':<35} {sum(b_gpu)/len(b_gpu):>12.1f} {sum(e_gpu)/len(e_gpu):>12.1f} {sum(e_gpu)/len(e_gpu)-sum(b_gpu)/len(b_gpu):>+10.1f}")
        print(f"  {'Tokens generated':<35} {baseline['total_output_tokens']:>12} {et_agent['total_output_tokens']:>12}")

        output["baseline_hermes"] = baseline
        output["et_agent"] = et_agent

    elif not args.concurrent:
        config = MemoryConfig(block_size=16, gpu_capacity_bytes=6*1024**3,
                              use_cuda=True, model_profile=QWEN3_06B)
        allocator = KVBlockAllocator(config)
        block_tables = BlockTableManager(allocator, config.block_size)

        print(f"\n{'='*60}")
        print(f"  ET-AGENT Paged KV Cache ({args.turns} turns)")
        print(f"{'='*60}")
        et_agent = run_session(model, tokenizer, "et-agent", args.turns,
                               sys_prompt, allocator, block_tables)
        output["et_agent"] = et_agent

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Saved to {args.output}")


if __name__ == "__main__":
    main()
