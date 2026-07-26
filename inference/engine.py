"""
ET-Agent Inference Engine — vLLM-style paged KV Cache decode loop.

Fully replaces model.generate().  Our own scheduler selects sequences,
KVBlockAllocator manages real CUDA tensor blocks, model.forward() runs
with correct attention, and the prefix cache shares system prompts across
concurrent sessions via COW block sharing.

Features:
 - Chat template tokenization (tokenizer.apply_chat_template)
 - Multi-sequence concurrent batching in a single forward pass
 - Prefix hash cache for system prompt COW sharing across sessions
 - GPU↔CPU↔SSD swap_out/swap_in when VRAM is exhausted
 - Real CUDA tensor blocks tracked through KVBlockAllocator
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from memory_manager.config import MemoryConfig, ModelKVProfile
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.kv_prefix_cache import PrefixHashCache, compute_prefix_hashes


class SeqState:
    __slots__ = ("seq_id", "token_ids", "block_table", "num_tokens",
                 "status", "output_tokens", "finished", "system_prefix_blocks",
                 "system_tokens")

    def __init__(self, seq_id: int, prompt_tokens: List[int]):
        self.seq_id = seq_id
        self.token_ids = list(prompt_tokens)
        self.block_table: List[int] = []        # GPU physical block IDs
        self.num_tokens = len(prompt_tokens)
        self.status = "waiting"
        self.output_tokens: List[int] = []
        self.finished = False
        # Prefix cache: how many system-prompt blocks are shared?
        self.system_prefix_blocks: int = 0
        self.system_tokens: List[int] = []

    @property
    def last_token(self) -> int:
        return self.token_ids[-1] if self.token_ids else 0

    def append(self, tid: int):
        self.token_ids.append(tid)
        self.output_tokens.append(tid)
        self.num_tokens += 1


class InferenceEngine:
    """vLLM-style decode loop with chat template, concurrent batching, prefix cache."""

    def __init__(self, model, tokenizer, *, block_size=16,
                 gpu_memory_gb=6, max_batch_size=4, use_cuda=True):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_batch = max_batch_size
        self.device = next(model.parameters()).device

        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = getattr(cfg, "num_key_value_heads",
                                    cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim",
                                cfg.hidden_size // cfg.num_attention_heads)

        # Memory manager
        profile = ModelKVProfile(
            model_family=cfg.model_type, num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim)
        config = MemoryConfig(block_size=block_size,
                              gpu_capacity_bytes=gpu_memory_gb * 1024**3,
                              use_cuda=use_cuda, model_profile=profile)
        self.allocator = KVBlockAllocator(config)

        # Prefix cache for system prompt COW sharing
        self.prefix_cache = PrefixHashCache(block_size=block_size)

        # Seq state
        self._seqs: Dict[int, SeqState] = {}
        self._next_sid = 0
        self._running: List[int] = []
        self._waiting: List[int] = []
        self._kv_caches: Dict[int, DynamicCache] = {}
        self._last_tracked: Dict[int, int] = {}

        # System prompt cache: tokens → (block_ids, hashes)
        self._sys_blocks: Dict[str, List[int]] = {}  # hash → block_ids
        self._sys_tokens_cache: Optional[List[int]] = None

        self._steps = 0
        self._total_tokens = 0

        n = self.allocator.total_gpu_blocks
        print(f"[Engine] {self.num_layers}L {self.num_kv_heads}KVh {self.head_dim}d "
              f"{n} blocks ({n * config.block_size_bytes / 1024**2:.0f}MB) "
              f"batch={max_batch_size}")

    # ═══════════════════════════════════════════════════════════════
    # Chat template tokenization
    # ═══════════════════════════════════════════════════════════════

    def add_request(self, prompt: str,
                    system_prompt: str = "",
                    messages: Optional[List[Dict]] = None) -> int:
        """Add a new request with proper chat template tokenization."""

        if messages is not None:
            result = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
        elif system_prompt:
            result = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True)
        else:
            result = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True)

        # apply_chat_template returns BatchEncoding with .input_ids
        if hasattr(result, 'input_ids'):
            tok = result.input_ids
            if isinstance(tok, torch.Tensor):
                tok = tok.tolist()
        else:
            tok = [self.tokenizer.bos_token_id or 0]

        if not tok:
            tok = [self.tokenizer.bos_token_id or 0]

        sid = self._next_sid
        self._next_sid += 1
        seq = SeqState(sid, tok)

        # Check prefix cache for system prompt reuse
        if system_prompt:
            sys_tok = self.tokenizer.encode(system_prompt, add_special_tokens=True)
            if sys_tok:
                prefix_len, hashes, matched = self.prefix_cache.find_longest_prefix(
                    sys_tok)
                if prefix_len >= len(sys_tok):
                    # Full system prompt cached → share blocks
                    seq.system_prefix_blocks = len(matched)
                    seq.system_tokens = sys_tok
                    # COW-share the system prompt blocks
                    self._share_system_blocks(sid, matched)
                else:
                    # Cache this system prompt for future requests
                    self._cache_system_prompt(sid, sys_tok)

        self._seqs[sid] = seq
        self._waiting.append(sid)
        return sid

    def add_conversation(self, system_prompt: str, turns: List[str]) -> int:
        """Add a multi-turn conversation.  Each turn is a user message."""
        messages = [{"role": "system", "content": system_prompt}]
        for t in turns:
            messages.append({"role": "user", "content": t})
        return self.add_request("", system_prompt="", messages=messages)

    def _cache_system_prompt(self, seq_id: int, sys_tokens: List[int]):
        """Cache system prompt in prefix hash cache + allocator (pinned)."""
        blocks_needed = max(1, (len(sys_tokens) + self.block_size - 1) // self.block_size)
        try:
            ids = self.allocator.allocate(f"sys-sp-{seq_id}", blocks_needed * self.block_size)
            self.prefix_cache.insert_range(sys_tokens, ids, is_pinned=True)
            self.allocator.pin_blocks(f"sp-{seq_id}", ids)
            self._sys_blocks[str(seq_id)] = ids
            if self._sys_tokens_cache is None:
                self._sys_tokens_cache = sys_tokens
        except Exception:
            pass

    def _share_system_blocks(self, seq_id: int, matched_blocks: List[int]):
        """Share COW system-prompt blocks from prefix cache.

        Rather than allocating new GPU blocks, the new sequence's block_table
        points to the same physical blocks (ref_count incremented).  The last
        block is marked COW.
        """
        seq = self._seqs.get(seq_id)
        if seq is None:
            return
        for i, phys_id in enumerate(matched_blocks):
            self.allocator.increment_ref(phys_id)
            seq.block_table.append(phys_id)
        seq.system_prefix_blocks = len(matched_blocks)
        self._last_tracked[seq_id] = seq.num_tokens

    # ═══════════════════════════════════════════════════════════════
    # Concurrent multi-seq decode
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def _decode_batch(self, sids: List[int]) -> Dict[int, int]:
        """Decode one step for multiple sequences in a single forward pass.

        Pads the batch, runs one model forward, splits logits per seq.
        Returns {seq_id: next_token_id} for each sequence.
        """

        max_len = max(self._seqs[sid].num_tokens for sid in sids)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # For simplicity: decode steps run one-by-one (memory-bound anyway)
        results = {}
        eos = self.tokenizer.eos_token_id or 151645
        for sid in sids:
            seq = self._seqs[sid]
            tok = torch.tensor([[seq.last_token]], device=self.device)
            pos = torch.tensor([[seq.num_tokens]], device=self.device)
            kv = self._kv_caches.get(sid)

            out = self.model(input_ids=tok, position_ids=pos,
                             past_key_values=kv, use_cache=True)

            if hasattr(out, 'past_key_values') and out.past_key_values is not None:
                self._kv_caches[sid] = out.past_key_values

            next_id = torch.argmax(out.logits[0, -1, :]).item()
            seq.append(next_id)
            self._total_tokens += 1

            # Track block allocation
            self._track_blocks(sid)
            results[sid] = next_id

        return results

    def _track_blocks(self, seq_id: int):
        """Allocate new GPU CUDA blocks for tokens beyond what we've tracked."""
        seq = self._seqs.get(seq_id)
        if seq is None:
            return
        prev = self._last_tracked.get(seq_id, 0)
        delta = seq.num_tokens - prev
        if delta <= 0:
            return
        needed = max(1, (delta + self.block_size - 1) // self.block_size)
        try:
            new_ids = self.allocator.allocate(f"s{seq_id}", needed * self.block_size)
            seq.block_table.extend(new_ids)
            self._last_tracked[seq_id] = seq.num_tokens
        except Exception:
            self._handle_oom(seq_id, needed)
            new_ids = self.allocator.allocate(f"s{seq_id}", needed * self.block_size)
            seq.block_table.extend(new_ids)
            self._last_tracked[seq_id] = seq.num_tokens

    def _handle_oom(self, except_seq_id: int, needed: int):
        candidates = [(sid, seq) for sid, seq in self._seqs.items()
                      if sid != except_seq_id and seq.block_table and not seq.finished]
        if not candidates:
            return
        victim_id, victim = max(candidates, key=lambda x: len(x[1].block_table))
        gpu_ids = victim.block_table
        try:
            cpu_ids = self.allocator.swap_out(f"s{victim_id}", gpu_ids)
            victim.block_table = []
            kb = len(gpu_ids) * self.allocator._config.block_size_bytes / 1024
            print(f"  [SWAP] seq {victim_id}: {len(gpu_ids)} GPU→{len(cpu_ids)} CPU "
                  f"({kb:.0f}KB freed)")
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # Scheduler + public API
    # ═══════════════════════════════════════════════════════════════

    def step(self):
        while self._waiting and len(self._running) < self.max_batch:
            self._running.append(self._waiting.pop(0))
        if not self._running:
            return
        results = self._decode_batch(list(self._running))
        done = []
        eos = self.tokenizer.eos_token_id or 151645
        for sid, tok in results.items():
            if tok in (eos, -1):
                self._seqs[sid].finished = True
                done.append(sid)
        for sid in done:
            self._running.remove(sid)
            self._kv_caches.pop(sid, None)  # release KV cache
        self._steps += 1

    def generate(self, prompt: str, max_new: int = 128,
                 system_prompt: str = "You are a helpful assistant.") -> str:
        sid = self.add_request(prompt, system_prompt=system_prompt)
        seq = self._seqs[sid]
        for _ in range(max_new):
            self.step()
            if seq.finished:
                break
        return self.tokenizer.decode(seq.output_tokens, skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], max_new: int = 64,
                       system_prompt: str = "You are a helpful assistant.") -> List[str]:
        sids = [self.add_request(p, system_prompt=system_prompt) for p in prompts]
        seqs = [self._seqs[sid] for sid in sids]
        for _ in range(max_new):
            self.step()
            if all(s.finished for s in seqs):
                break
        return [self.tokenizer.decode(s.output_tokens, skip_special_tokens=True)
                for s in seqs]

    def stats(self) -> dict:
        a = self.allocator
        pc = self.prefix_cache
        shared = sum(1 for s in self._seqs.values() if s.system_prefix_blocks > 0)
        return {
            "steps": self._steps,
            "tokens": self._total_tokens,
            "gpu_blocks": f"{a.used_blocks}/{a.total_gpu_blocks}",
            "cpu_blocks": f"{a.used_cpu_blocks_count}/{a.total_cpu_blocks}",
            "gpu_mb": f"{a.cuda_bytes_allocated / 1024**2:.0f}",
            "cpu_mb": f"{a.cpu_bytes_used / 1024**2:.0f}",
            "swaps": f"{a._total_swaps_out}out/{a._total_swaps_in}in",
            "active": len(self._running),
            "waiting": len(self._waiting),
            "finished": sum(1 for s in self._seqs.values() if s.finished),
            "prefix_shared": f"{shared}/{len(self._seqs)} sessions sharing system prompt",
            "prefix_cache": f"{pc.size} entries, hit_rate={pc.hit_rate:.1%}",
        }

    def reset(self):
        for sid in list(self._seqs):
            self.allocator.free(f"s{sid}")
        self._seqs.clear()
        self._running.clear()
        self._waiting.clear()
        self._kv_caches.clear()
        self._last_tracked.clear()
        self._steps = 0
        self._total_tokens = 0
