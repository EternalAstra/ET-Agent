"""
KV Cache Block data types — with real CUDA tensor backing.

Each KVBlock now holds a real ``torch.Tensor`` on GPU VRAM of shape
``(num_layers, 2, num_kv_heads, block_size, head_dim)`` in float16.
This is the actual KV Cache page that the LLM attention kernel reads.

Copy-on-write uses ``torch.clone()`` — a real GPU memcpy, not metadata.

The ``_tensor`` field is None only when ``use_cuda=False`` (CPU testing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import torch


class KVBlockState(Enum):
    FREE = auto()
    ALLOCATED = auto()
    SHARED = auto()
    PINNED = auto()
    EVICTING = auto()


class StorageTier(Enum):
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"


@dataclass
class KVBlock:
    """A single physical KV Cache block — backed by a real CUDA tensor.

    Shape: ``(num_layers, 2, num_kv_heads, block_size, head_dim)``
    where dim=1 is 0=key, 1=value.  Stored in float16 on GPU.

    Parameters
    ----------
    block_id : int
        Unique physical block identifier (index into allocator pool).
    state : KVBlockState
        Current lifecycle state.
    num_tokens : int
        How many tokens are actually stored in this block (<= block_size).
    ref_count : int
        Number of logical block-table entries referencing this block.
    storage_tier : StorageTier
        Where the block data currently resides.
    group_id : str | None
        Optional grouping key (session_id for group eviction).
    _tensor : torch.Tensor | None
        The real CUDA tensor holding the KV Cache data.  None until
        ``allocate_tensor()`` is called (or when use_cuda=False).
    """

    block_id: int
    state: KVBlockState = KVBlockState.FREE
    num_tokens: int = 0
    ref_count: int = 0
    storage_tier: StorageTier = StorageTier.GPU
    last_access_ts: float = 0.0
    access_count: int = 0
    group_id: Optional[str] = None
    _tensor: Optional[torch.Tensor] = None
    _tensor_shape: Optional[Tuple[int, ...]] = None

    # ── computed helpers ──

    @property
    def is_free(self) -> bool:
        return self.state == KVBlockState.FREE

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    @property
    def is_pinned(self) -> bool:
        return self.state == KVBlockState.PINNED

    @property
    def is_evictable(self) -> bool:
        return self.state not in (KVBlockState.FREE, KVBlockState.PINNED)

    @property
    def tensor_bytes(self) -> int:
        """Bytes of GPU VRAM consumed by this block's tensor."""
        if self._tensor is None:
            return 0
        return self._tensor.element_size() * self._tensor.numel()

    # ── Tensor allocation (real cudaMalloc) ──

    def allocate_tensor(self, shape: Tuple[int, int, int, int, int],
                        device: str = "cuda:0",
                        dtype: torch.dtype = torch.float16):
        """Allocate real GPU memory for this block.

        Equivalent to cudaMalloc for the KV cache page.
        Only called on first use to avoid upfront VRAM consumption.
        """
        self._tensor = torch.zeros(shape, device=device, dtype=dtype)
        self._tensor_shape = shape

    def free_tensor(self):
        """Release GPU memory.  Equivalent to cudaFree."""
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._tensor_shape = None

    def clone_tensor_from(self, source: "KVBlock"):
        """Deep-copy GPU tensor from *source* block (torch.clone for COW).

        Returns True if a clone was performed.
        """
        if source._tensor is None:
            return False
        self._tensor = source._tensor.clone()
        self._tensor_shape = source._tensor_shape
        return True

    # ── mutation helpers ──

    def mark_allocated(self, num_tokens: int = 0, group_id: str | None = None):
        self.state = KVBlockState.ALLOCATED
        self.ref_count = 1
        self.num_tokens = num_tokens
        self.group_id = group_id

    def mark_free(self):
        self.state = KVBlockState.FREE
        self.num_tokens = 0
        self.ref_count = 0
        self.storage_tier = StorageTier.GPU
        self.last_access_ts = 0.0
        self.access_count = 0
        self.group_id = None
        self.free_tensor()

    def increment_ref(self):
        self.ref_count += 1
        if self.ref_count > 1:
            self.state = KVBlockState.SHARED

    def decrement_ref(self):
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 1:
            self.state = KVBlockState.ALLOCATED
        elif self.ref_count == 0:
            self.mark_free()
            return True
        return False

    def touch(self, ts: float):
        self.last_access_ts = ts
        self.access_count += 1

    def __repr__(self) -> str:
        tensor_info = f"{self.tensor_bytes/1024:.0f}KB" if self._tensor is not None else "no-tensor"
        return (
            f"KVBlock(id={self.block_id}, {self.state.name}, "
            f"tokens={self.num_tokens}, rc={self.ref_count}, "
            f"tier={self.storage_tier.value}, mem={tensor_info})"
        )


@dataclass
class BlockTableEntry:
    logical_idx: int
    physical_id: int
    num_filled: int = 0
    is_cow: bool = False
    shared_from: Optional[str] = None

    def is_full_for(self, block_size: int) -> bool:
        return self.num_filled >= block_size

    def __repr__(self) -> str:
        cow = " [COW]" if self.is_cow else ""
        return f"L{self.logical_idx:03d}→P{self.physical_id:05d} filled={self.num_filled}{cow}"
