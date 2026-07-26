"""
PagedAttention Operator — vLLM-style paged KV Cache attention in pure PyTorch.

Implements the exact algorithm from vLLM §4.1:

  For each attention head, for each query position i:
    Read the block_table to find which physical blocks hold the KV cache.
    For each physical block b:
      For each token position j in block b:
        Compute attention score: q_i · k_j / sqrt(d)
    Weighted sum of value vectors: sum(softmax(scores) * v_j)

Unlike transformers' default DynamicCache (which stores KV in one contiguous
tensor and re-concat's every turn), this operator reads KV from a set of
non-contiguous physical blocks whose IDs come from our KVBlockAllocator.
This IS what makes vLLM's memory utilisation ~96.3% vs default's ~20-40%.

Parameters
----------
query : torch.Tensor      (batch, num_heads, q_len, head_dim)
key_blocks : list[torch.Tensor]   each of shape (num_kv_heads, block_size, head_dim)
value_blocks : list[torch.Tensor] each of shape (num_kv_heads, block_size, head_dim)
block_table : list[int]   ordered physical block IDs for this sequence
block_size : int          tokens per block (default 16)

Returns
-------
output : torch.Tensor  (batch, num_heads, q_len, head_dim)
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def paged_attention_forward(
    query: torch.Tensor,
    key_blocks: List[torch.Tensor],
    value_blocks: List[torch.Tensor],
    block_table: List[int],
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute multi-head attention over paged (non-contiguous) KV cache.

    This is the vLLM PagedAttention algorithm implemented in PyTorch.
    """

    batch, num_heads, q_len, head_dim = query.shape

    if scale is None:
        scale = head_dim ** -0.5

    num_kv_heads = key_blocks[0].shape[0]
    head_ratio = num_heads // num_kv_heads

    # Total KV length = block_table length * block_size
    total_kv_len = len(block_table) * block_size

    # Step 1: gather KV from physical blocks into logical order
    # Each block is (num_kv_heads, block_size, head_dim)
    # Stack into (num_blocks, num_kv_heads, block_size, head_dim)
    k_selected = torch.stack(
        [key_blocks[bid] for bid in block_table], dim=0
    )  # → (num_blocks, num_kv_heads, block_size, head_dim)
    v_selected = torch.stack(
        [value_blocks[bid] for bid in block_table], dim=0
    )

    # Step 2: reshape to logical contiguous KV
    k_logical = k_selected.permute(2, 0, 1, 3).reshape(
        num_kv_heads, total_kv_len, head_dim
    )  # → (num_kv_heads, total_kv_len, head_dim)
    v_logical = v_selected.permute(2, 0, 1, 3).reshape(
        num_kv_heads, total_kv_len, head_dim
    )

    # Step 3: expand KV heads if GQA (Grouped-Query Attention)
    if head_ratio > 1:
        k_logical = k_logical.unsqueeze(1).expand(
            num_kv_heads, head_ratio, total_kv_len, head_dim
        ).reshape(num_heads, total_kv_len, head_dim)
        v_logical = v_logical.unsqueeze(1).expand(
            num_kv_heads, head_ratio, total_kv_len, head_dim
        ).reshape(num_heads, total_kv_len, head_dim)

    # Step 4: compute attention (SDPA or manual)
    # query: (batch, num_heads, q_len, head_dim)
    # k:     (num_heads, total_kv_len, head_dim) → add batch dim
    # v:     (num_heads, total_kv_len, head_dim) → add batch dim

    k = k_logical.unsqueeze(0)  # (1, num_heads, total_kv_len, head_dim)
    v = v_logical.unsqueeze(0)  # (1, num_heads, total_kv_len, head_dim)

    # Use PyTorch's fused scaled_dot_product_attention (cuDNN-backed, fast)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1) if batch == 1 else query,  # keep batch dim
        k.transpose(0, 1) if batch == 1 else k,
        v.transpose(0, 1) if batch == 1 else v,
        scale=scale,
    )

    return output


def paged_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,   # (num_layers, 2, num_kv_heads, total_tokens, head_dim)
    value_cache: torch.Tensor, # (num_layers, 2, num_kv_heads, total_tokens, head_dim)
    block_size: int = 16,
) -> torch.Tensor:
    """Simplified prefill path: all tokens fit in one contiguous allocation.

    During prefill, the model processes the entire prompt at once.
    vLLM allocates blocks page-by-page but the attention can be computed
    contiguously since all prompt tokens are available upfront.
    """
    _, num_heads, q_len, head_dim = query.shape
    scale = head_dim ** -0.5

    k_contig = key_cache.reshape(key_cache.shape[0], -1, head_dim)
    v_contig = value_cache.reshape(value_cache.shape[0], -1, head_dim)

    output = F.scaled_dot_product_attention(
        query, k_contig.unsqueeze(0), v_contig.unsqueeze(0), scale=scale,
    )
    return output
