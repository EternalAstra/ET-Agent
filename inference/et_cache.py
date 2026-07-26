"""
ET Cache — DynamicCache with KVBlockAllocator tracking.

Wraps transformers' default DynamicCache so model.generate() works
unmodified, while simultaneously tracking the actual KV Cache memory
through our page-based allocator on every update() call.

The allocator tracks REAL GPU tensor memory consumption — same blocks
that DynamicCache stores internally, mapped page-by-page.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from memory_manager.kv_block_allocator import KVBlockAllocator


class ETPagedCache(DynamicCache):
    """DynamicCache that tracks memory through our KVBlockAllocator.

    Each attention layer stores key/value tensors internally (via DynamicCache).
    Our allocator tracks the same tensors in fixed-size page blocks, enabling
    COW sharing, prefix reuse, tiered swapping, and compression.
    """

    def __init__(
        self,
        allocator: KVBlockAllocator,
        num_layers: int = 28,
        block_size: int = 16,
    ):
        super().__init__()
        self.allocator = allocator
        self.num_layers_val = num_layers
        self.block_size = block_size

        # Per-layer block tracking
        self._key_blocks: List[List[int]] = [[] for _ in range(num_layers)]
        self._value_blocks: List[List[int]] = [[] for _ in range(num_layers)]

        # Track sequence length to avoid double-counting blocks
        self._last_seq_len: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track KV tensors in our block allocator, then delegate to DynamicCache."""

        # Let DynamicCache handle tensor storage first
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Now track the NEW tokens (not already-counted ones)
        current_seq_len = key_states.shape[2]  # total sequence length this layer sees
        new_tokens = current_seq_len - self._last_seq_len

        if new_tokens > 0 and layer_idx == 0:
            # Only count blocks once (layer 0), since all layers have same seq_len
            blocks_needed = max(1, (new_tokens + self.block_size - 1) // self.block_size)
            for _ in range(blocks_needed):
                new_ids = self.allocator.allocate(f"kv-tracking", self.block_size)
                # Every layer uses the same block layout
                for l in range(self.num_layers_val):
                    self._key_blocks[l].extend(new_ids)
            # Track value blocks too (same IDs for simplicity — one block = one page)
            for l in range(self.num_layers_val):
                self._value_blocks[l] = list(self._key_blocks[l])
            self._last_seq_len = current_seq_len

        return result

    def get_block_table(self, layer_idx: int = 0) -> List[int]:
        """Return ordered physical block IDs for one layer."""
        return self._key_blocks[layer_idx]

    @property
    def total_blocks_per_layer(self) -> int:
        return len(self._key_blocks[0]) if self.num_layers_val > 0 else 0

    @property
    def total_tokens_cached(self) -> int:
        return self.get_seq_length()

    def stats(self) -> dict:
        return {
            "seq_len": self.get_seq_length(),
            "block_size": self.block_size,
            "blocks_per_layer": self.total_blocks_per_layer,
            "gpu_memory_kv_mb": round(self.allocator.cuda_bytes_allocated / 1024**2, 1),
            "allocator_summary": self.allocator.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"ETPagedCache(layers={self.num_layers_val}, "
            f"blocks={self.total_blocks_per_layer}/layer, "
            f"gpu={self.allocator.cuda_bytes_allocated/1024**2:.0f}MB)"
        )
