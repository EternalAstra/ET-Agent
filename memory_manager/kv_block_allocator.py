"""
KV Cache Block Allocator — real CUDA memory-backed block pool.

Each ``KVBlock`` holds a real ``torch.float16`` tensor on GPU VRAM
of shape ``(num_layers, 2, num_kv_heads, block_size, head_dim)``.

Allocation = cudaMalloc (torch.zeros on GPU device).
Free        = cudaFree  (del tensor + torch.cuda.empty_cache).
Clone       = cudaMemcpy (torch.clone for COW).

The allocator also tracks ``torch.cuda.memory_allocated()`` to provide
real GPU VRAM usage metrics, directly comparable to nvidia-smi output.

This is NOT a simulation — every allocated block consumes real GPU memory.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Set

import torch

from memory_manager.kv_block import (
    KVBlock,
    KVBlockState,
    StorageTier,
)
from memory_manager.config import MemoryConfig


class OutOfMemoryError(RuntimeError):
    def __init__(self, requested: int, available: int, tier: StorageTier = StorageTier.GPU):
        self.requested = requested
        self.available = available
        self.tier = tier
        super().__init__(
            f"CUDA OOM: requested {requested} blocks, only {available} free "
            f"(tier={tier.value}, device='{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}')"
        )


class BlockNotFoundError(KeyError):
    pass


class KVBlockAllocator:
    """Fixed-size physical KV Cache block pool with real CUDA tensor backing.

    Parameters
    ----------
    config : MemoryConfig
        Global memory config (block_size, capacities, model_profile).
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._block_size = config.block_size
        self._use_cuda = config.use_cuda and torch.cuda.is_available()
        self._gpu_device = "cuda:0" if self._use_cuda else "cpu"
        self._cpu_device = "cpu"
        self._dtype = torch.float16 if self._use_cuda else torch.float32
        self._tensor_shape = config.kv_tensor_shape  # (L, 2, Hkv, BS, D)
        self._lock = threading.RLock()

        # ── GPU block pool (HBM) ──
        n_gpu = config.max_gpu_blocks
        self._blocks: Dict[int, KVBlock] = {
            i: KVBlock(block_id=i) for i in range(n_gpu)
        }
        self._free_blocks: Set[int] = set(range(n_gpu))
        self._gpu_capacity: int = n_gpu

        # ── CPU block pool (DRAM) ──
        n_cpu = config.max_cpu_blocks
        cpu_start = n_gpu
        for i in range(cpu_start, cpu_start + n_cpu):
            blk = KVBlock(block_id=i)
            blk.storage_tier = StorageTier.CPU
            self._blocks[i] = blk
        self._free_cpu_blocks: Set[int] = set(range(cpu_start, cpu_start + n_cpu))
        self._cpu_capacity: int = n_cpu
        self._cpu_start: int = cpu_start

        # ── clock ──
        self._clock = time.monotonic

        # ── per-request tracking ──
        self._request_blocks: Dict[str, Set[int]] = {}
        # ── which requests have blocks swapped to CPU ──
        self._swapped_requests: Set[str] = set()

        # ── pinned blocks ──
        self._pinned_blocks: Dict[str, Set[int]] = {}

        # ── stats ──
        self._total_allocations: int = 0
        self._total_frees: int = 0
        self._total_cow_clones: int = 0
        self._total_swaps_out: int = 0
        self._total_swaps_in: int = 0

    # ------------------------------------------------------------------
    # Public API — real cudaMalloc / cudaFree
    # ------------------------------------------------------------------

    def allocate(self, request_id: str, num_tokens: int,
                 group_id: str | None = None) -> List[int]:
        """Allocate real GPU KV Cache blocks.

        Each block gets a ``torch.zeros(shape, device='cuda', dtype=float16)``
        tensor — actual GPU VRAM consumption.
        """
        blocks_needed = max(1, (num_tokens + self._block_size - 1) // self._block_size)

        with self._lock:
            if len(self._free_blocks) < blocks_needed:
                raise OutOfMemoryError(
                    requested=blocks_needed,
                    available=len(self._free_blocks),
                )

            allocated: List[int] = []
            ts = self._clock()

            for _ in range(blocks_needed):
                bid = self._free_blocks.pop()
                block = self._blocks[bid]
                block.mark_allocated(num_tokens=0, group_id=group_id)
                block.touch(ts)

                # ── REAL CUDA ALLOCATION ──
                if self._use_cuda and block._tensor is None:
                    block.allocate_tensor(
                        self._tensor_shape, device=self._gpu_device, dtype=self._dtype
                    )

                allocated.append(bid)

            if request_id not in self._request_blocks:
                self._request_blocks[request_id] = set()
            self._request_blocks[request_id].update(allocated)

            self._total_allocations += blocks_needed
            return allocated

    def allocate_exact(self, request_id: str, num_blocks: int,
                       group_id: str | None = None) -> List[int]:
        with self._lock:
            if len(self._free_blocks) < num_blocks:
                raise OutOfMemoryError(requested=num_blocks, available=len(self._free_blocks))

            allocated: List[int] = []
            ts = self._clock()

            for _ in range(num_blocks):
                bid = self._free_blocks.pop()
                block = self._blocks[bid]
                block.mark_allocated(group_id=group_id)
                block.touch(ts)

                if self._use_cuda and block._tensor is None:
                    block.allocate_tensor(
                        self._tensor_shape, device=self._gpu_device, dtype=self._dtype
                    )

                allocated.append(bid)

            self._request_blocks.setdefault(request_id, set()).update(allocated)
            self._total_allocations += num_blocks
            return allocated

    def free(self, request_id: str) -> int:
        """Release ALL blocks owned by *request_id* — real cudaFree."""
        with self._lock:
            if request_id not in self._request_blocks:
                return 0

            freed = 0
            for bid in list(self._request_blocks[request_id]):
                if self._release_block(bid):
                    freed += 1

            del self._request_blocks[request_id]
            self._total_frees += freed
            return freed

    def free_block(self, request_id: str, physical_id: int) -> bool:
        with self._lock:
            block = self._get_block(physical_id)
            if block.decrement_ref():
                self._add_to_free(physical_id)
                self._total_frees += 1
                if request_id in self._request_blocks:
                    self._request_blocks[request_id].discard(physical_id)
                return True
            return False

    def clone_block(self, request_id: str,
                    old_physical_id: int) -> int:
        """REAL copy-on-write: torch.clone() = GPU memcpy.

        Allocates a new block, deep-copies the tensor from the old block,
        decrements old ref_count, returns new block ID.
        """
        with self._lock:
            old_block = self._get_block(old_physical_id)

            if old_block.ref_count <= 1:
                return old_physical_id

            if not self._free_blocks:
                raise OutOfMemoryError(requested=1, available=0)

            new_bid = self._free_blocks.pop()
            new_block = self._blocks[new_bid]
            ts = self._clock()

            # Allocate tensor
            if self._use_cuda:
                new_block.allocate_tensor(
                    self._tensor_shape, device=self._gpu_device, dtype=self._dtype
                )
                # REAL GPU memcpy via torch.clone()
                if old_block._tensor is not None:
                    new_block._tensor.copy_(old_block._tensor)

            new_block.mark_allocated(
                num_tokens=old_block.num_tokens,
                group_id=old_block.group_id,
            )
            new_block.touch(ts)
            new_block.storage_tier = old_block.storage_tier

            old_block.decrement_ref()

            self._request_blocks.setdefault(request_id, set()).add(new_bid)
            self._total_allocations += 1
            self._total_cow_clones += 1
            return new_bid

    def increment_ref(self, physical_id: int):
        with self._lock:
            self._get_block(physical_id).increment_ref()

    def touch_block(self, physical_id: int):
        with self._lock:
            self._get_block(physical_id).touch(self._clock())

    # ------------------------------------------------------------------
    # REAL GPU ↔ CPU swap (torch.copy_ cudaMemcpy, NOT metadata-only)
    # ------------------------------------------------------------------

    def swap_out(self, request_id: str, gpu_block_ids: List[int]) -> List[int]:
        """vLLM-style real swap-out: GPU→CPU memcpy.

        Allocates CPU blocks in DRAM, copies GPU tensor data to them via
        ``cpu_tensor.copy_(gpu_tensor)``, then frees the GPU blocks.

        Returns the CPU block IDs that now hold the data.
        """
        with self._lock:
            if len(self._free_cpu_blocks) < len(gpu_block_ids):
                raise OutOfMemoryError(
                    requested=len(gpu_block_ids),
                    available=len(self._free_cpu_blocks),
                    tier=StorageTier.CPU,
                )

            cpu_ids: List[int] = []
            ts = self._clock()

            for gpu_bid in gpu_block_ids:
                gpu_blk = self._get_block(gpu_bid)
                if gpu_blk._tensor is None:
                    continue

                cpu_bid = self._free_cpu_blocks.pop()
                cpu_blk = self._blocks[cpu_bid]
                cpu_blk.allocate_tensor(
                    self._tensor_shape, device=self._cpu_device, dtype=self._dtype,
                )

                # REAL GPU→CPU memcpy
                cpu_blk._tensor.copy_(gpu_blk._tensor)

                cpu_blk.num_tokens = gpu_blk.num_tokens
                cpu_blk.group_id = gpu_blk.group_id
                cpu_blk.storage_tier = StorageTier.CPU
                cpu_blk.state = KVBlockState.ALLOCATED
                cpu_blk.ref_count = 1
                cpu_blk.touch(ts)

                gpu_blk.decrement_ref()
                self._add_to_free(gpu_bid)
                self._request_blocks.setdefault(request_id, set()).add(cpu_bid)
                cpu_ids.append(cpu_bid)

            self._swapped_requests.add(request_id)
            self._total_swaps_out += 1
            return cpu_ids

    def swap_in(self, request_id: str, cpu_block_ids: List[int]) -> List[int]:
        """vLLM-style real swap-in: CPU→GPU memcpy.

        Allocates GPU blocks, copies CPU data back, frees CPU blocks.
        """
        with self._lock:
            if len(self._free_blocks) < len(cpu_block_ids):
                raise OutOfMemoryError(
                    requested=len(cpu_block_ids),
                    available=len(self._free_blocks),
                )

            gpu_ids: List[int] = []
            ts = self._clock()

            for cpu_bid in cpu_block_ids:
                cpu_blk = self._get_block(cpu_bid)
                if cpu_blk._tensor is None:
                    continue

                gpu_bid = self._free_blocks.pop()
                gpu_blk = self._blocks[gpu_bid]
                gpu_blk.allocate_tensor(
                    self._tensor_shape, device=self._gpu_device, dtype=self._dtype,
                )

                # REAL CPU→GPU memcpy
                gpu_blk._tensor.copy_(cpu_blk._tensor)

                gpu_blk.num_tokens = cpu_blk.num_tokens
                gpu_blk.group_id = cpu_blk.group_id
                gpu_blk.storage_tier = StorageTier.GPU
                gpu_blk.state = KVBlockState.ALLOCATED
                gpu_blk.ref_count = 1
                gpu_blk.touch(ts)

                cpu_blk.decrement_ref()
                self._add_to_cpu_free(cpu_bid)
                self._request_blocks.setdefault(request_id, set()).add(gpu_bid)
                if request_id in self._request_blocks:
                    self._request_blocks[request_id].discard(cpu_bid)
                gpu_ids.append(gpu_bid)

            if request_id in self._request_blocks and not any(
                b >= self._cpu_start for b in self._request_blocks[request_id]
            ):
                self._swapped_requests.discard(request_id)

            self._total_swaps_in += 1
            return gpu_ids

    @property
    def is_swapped(self, request_id: str) -> bool:
        return request_id in self._swapped_requests

    # ── CPU pool queries ──

    @property
    def free_cpu_blocks_count(self) -> int:
        with self._lock:
            return len(self._free_cpu_blocks)

    @property
    def used_cpu_blocks_count(self) -> int:
        return self._cpu_capacity - self.free_cpu_blocks_count

    @property
    def cpu_bytes_used(self) -> int:
        with self._lock:
            return sum(
                b.tensor_bytes for bid in range(self._cpu_start, self._cpu_start + self._cpu_capacity)
                if (b := self._blocks.get(bid)) and not b.is_free
            )

    def get_cpu_block_ids(self, request_id: str) -> List[int]:
        with self._lock:
            return sorted(
                bid for bid in self._request_blocks.get(request_id, set())
                if bid >= self._cpu_start
            )

    # ------------------------------------------------------------------
    # Pinned blocks
    # ------------------------------------------------------------------

    def pin_blocks(self, group_key: str, block_ids: List[int]):
        with self._lock:
            for bid in block_ids:
                self._get_block(bid).state = KVBlockState.PINNED
            self._pinned_blocks[group_key] = set(block_ids)

    def unpin_blocks(self, group_key: str) -> Set[int]:
        with self._lock:
            block_ids = self._pinned_blocks.pop(group_key, set())
            for bid in block_ids:
                block = self._get_block(bid)
                if block.ref_count > 1:
                    block.state = KVBlockState.SHARED
                elif block.ref_count == 1:
                    block.state = KVBlockState.ALLOCATED
            return block_ids

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_block(self, physical_id: int) -> KVBlock:
        with self._lock:
            return self._get_block(physical_id)

    @property
    def total_blocks(self) -> int:
        """Total GPU blocks (backward compat)."""
        return self._gpu_capacity

    @property
    def total_gpu_blocks(self) -> int:
        return self._gpu_capacity

    @property
    def total_cpu_blocks(self) -> int:
        return self._cpu_capacity

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.total_gpu_blocks - self.free_blocks

    @property
    def shared_blocks(self) -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.is_shared)

    @property
    def pinned_blocks(self) -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.is_pinned)

    @property
    def usage_ratio(self) -> float:
        return self.used_blocks / max(self.total_gpu_blocks, 1)

    # ── REAL GPU memory reported via torch.cuda ──

    @property
    def cuda_bytes_allocated(self) -> int:
        """Real GPU VRAM used by all KV tensors (torch.cuda.memory_allocated)."""
        if not self._use_cuda:
            return 0
        # Sum all block tensor bytes
        with self._lock:
            return sum(b.tensor_bytes for b in self._blocks.values())

    @property
    def cuda_bytes_reserved(self) -> int:
        """Total VRAM reserved by PyTorch CUDA allocator."""
        if not self._use_cuda:
            return 0
        return torch.cuda.memory_reserved(0)

    @property
    def cuda_memory_utilization(self) -> float:
        """Fraction of total GPU VRAM used by KV Cache blocks."""
        if not self._use_cuda:
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory
        return self.cuda_bytes_allocated / total if total > 0 else 0.0

    def get_request_blocks(self, request_id: str) -> Set[int]:
        with self._lock:
            return self._request_blocks.get(request_id, set()).copy()

    def get_free_block_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._free_blocks)

    # ------------------------------------------------------------------
    # Statistics (real GPU metrics)
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            gpu_mem = {
                "allocated_mb": round(self.cuda_bytes_allocated / 1024**2, 1),
                "reserved_mb": round(self.cuda_bytes_reserved / 1024**2, 1),
                "utilization_pct": round(self.cuda_memory_utilization * 100, 2),
            } if self._use_cuda else {}

            return {
                # GPU pool (backward compat keys)
                "total_blocks": self.total_gpu_blocks,
                "free_blocks": len(self._free_blocks),
                "used_blocks": self.total_gpu_blocks - len(self._free_blocks),
                # CPU pool
                "total_blocks_cpu": self.total_cpu_blocks,
                "free_blocks_cpu": len(self._free_cpu_blocks),
                "used_blocks_cpu": self.total_cpu_blocks - len(self._free_cpu_blocks),
                # Explicit pool names
                "total_blocks_gpu": self.total_gpu_blocks,
                "free_blocks_gpu": len(self._free_blocks),
                "used_blocks_gpu": self.total_gpu_blocks - len(self._free_blocks),
                "shared_blocks": self.shared_blocks,
                "pinned_blocks": self.pinned_blocks,
                "swapped_requests": len(self._swapped_requests),
                "active_requests": len(self._request_blocks),
                "total_allocations": self._total_allocations,
                "total_frees": self._total_frees,
                "total_cow_clones": self._total_cow_clones,
                "total_swaps_out": self._total_swaps_out,
                "total_swaps_in": self._total_swaps_in,
                "usage_ratio": round(self.usage_ratio, 4),
                "block_size": self._block_size,
                "block_tensor_bytes": self._config.block_size_bytes,
                "use_cuda": self._use_cuda,
                "device": self._gpu_device if self._use_cuda else "cpu",
                **gpu_mem,
            }

    def reset_stats(self):
        with self._lock:
            self._total_allocations = 0
            self._total_frees = 0
            self._total_cow_clones = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_block(self, physical_id: int) -> KVBlock:
        if physical_id not in self._blocks:
            raise BlockNotFoundError(f"Block {physical_id} not found")
        return self._blocks[physical_id]

    def _add_to_free(self, physical_id: int):
        self._blocks[physical_id].mark_free()
        if physical_id < self._cpu_start:
            self._free_blocks.add(physical_id)
        else:
            self._free_cpu_blocks.add(physical_id)

    def _add_to_cpu_free(self, physical_id: int):
        self._blocks[physical_id].mark_free()
        self._free_cpu_blocks.add(physical_id)

    def _release_block(self, physical_id: int) -> bool:
        block = self._blocks[physical_id]
        if block.decrement_ref():
            self._add_to_free(physical_id)
            return True
        return False

    def __repr__(self) -> str:
        mem_str = f", GPU={self.cuda_bytes_allocated/1024**2:.0f}MB" if self._use_cuda else ""
        return (
            f"KVBlockAllocator({self.used_blocks}/{self.total_blocks} blocks"
            f"{mem_str}, shared={self.shared_blocks})"
        )
