"""
Memory Manager configuration — now with real CUDA tensor sizing.

All tunable parameters for the KV Cache block allocator, block table manager,
and real GPU memory management via PyTorch CUDA.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelKVProfile:
    """Pre-computed KV Cache sizing for a given model architecture.

    Used to allocate real torch tensors on CUDA: each KVBlock holds one
    tensor of shape ``kv_tensor_shape`` in float16 (2 bytes per element).
    """

    model_family: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int = 2   # FP16 = 2 bytes

    @property
    def bytes_per_token(self) -> int:
        """Bytes for one token's KV Cache across all layers.

        K + V = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def bytes_per_block(self, block_size: int) -> int:
        return self.bytes_per_token * block_size

    @property
    def kv_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Shape of a single KV block's CUDA tensor.

        ``(num_layers, 2, num_kv_heads, block_size, head_dim)``
        where dim=1 indexes key (0) and value (1).
        This is the tensor that actually lives on GPU VRAM.
        """
        # block_size is NOT stored here — it's a runtime config.
        # Callers use: profile.kv_tensor_shape_for_block(block_size)
        return (-1, 2, self.num_kv_heads, -1, self.head_dim)

    def kv_tensor_shape_for_block(self, block_size: int) -> Tuple[int, int, int, int, int]:
        return (self.num_layers, 2, self.num_kv_heads, block_size, self.head_dim)

    @property
    def kv_elements_per_block(self) -> int:
        """Number of float16 elements in one full-shape KV block tensor."""
        return self.num_layers * 2 * self.num_kv_heads * self.num_kv_heads  # placeholder:

    def kv_elements_for_block(self, block_size: int) -> int:
        return self.num_layers * 2 * self.num_kv_heads * block_size * self.head_dim


# Known profiles
KNOWN_PROFILES: dict[str, ModelKVProfile] = {
    "llama-3.2-3b": ModelKVProfile(
        model_family="llama",
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
    ),
    "qwen2.5-7b": ModelKVProfile(
        model_family="qwen2",
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
    ),
    "qwen2.5-14b": ModelKVProfile(
        model_family="qwen2",
        num_layers=48,
        num_kv_heads=8,
        head_dim=128,
    ),
    "qwen2.5-3b": ModelKVProfile(
        model_family="qwen2",
        num_layers=36,
        num_kv_heads=4,
        head_dim=128,
    ),
    "deepseek-v4": ModelKVProfile(
        model_family="deepseek-v4",
        num_layers=60,
        num_kv_heads=1,
        head_dim=512,
    ),
    "minicpm3-4b": ModelKVProfile(
        model_family="minicpm",
        num_layers=32,
        num_kv_heads=4,
        head_dim=128,
    ),
    "deepseek-r1-distill-qwen-32b": ModelKVProfile(
        model_family="qwen2",
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
    ),
}


@dataclass
class MemoryConfig:
    """Global configuration for the memory manager.

    Attributes
    ----------
    block_size : int
        Number of tokens per KV Cache block (default 16).
    gpu_capacity_bytes : int
        Total GPU VRAM available for KV Cache blocks.
    cpu_capacity_bytes : int
        Total CPU DRAM available for swapped-out blocks.
    ssd_capacity_bytes : int
        Total NVMe SSD capacity for cold-storage blocks.
    enable_ssd : bool
        Whether to enable SSD tier.
    prefill_block_margin : int
        Extra blocks to pre-allocate for prefill.
    use_cuda : bool
        If True, allocate real CUDA tensors (torch.float16 on GPU).
        If False, fall back to metadata-only (for CPU-only testing).
    model_profile : ModelKVProfile | None
        Model-specific KV sizing (auto-detected from model_name).
    """

    block_size: int = 16
    gpu_capacity_bytes: int = 80 * 1024**3
    cpu_capacity_bytes: int = 512 * 1024**3
    ssd_capacity_bytes: int = 2 * 1024**4
    enable_ssd: bool = False
    prefill_block_margin: int = 8
    max_shared_blocks_pct: float = 0.95
    use_cuda: bool = True
    model_profile: ModelKVProfile | None = None

    def __post_init__(self):
        if self.model_profile is not None and isinstance(self.model_profile, dict):
            self.model_profile = ModelKVProfile(**self.model_profile)
        if self.use_cuda and self.model_profile is None:
            self.model_profile = KNOWN_PROFILES.get("qwen2.5-7b")

    @property
    def block_size_bytes(self) -> int:
        if self.model_profile is not None:
            return self.model_profile.bytes_per_block(self.block_size)
        return 2 * 28 * 4 * 128 * 2 * self.block_size  # Qwen2.5-7B fallback

    @property
    def kv_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Real CUDA tensor shape for one KV block."""
        if self.model_profile is not None:
            return self.model_profile.kv_tensor_shape_for_block(self.block_size)
        return (28, 2, 4, self.block_size, 128)

    @property
    def max_gpu_blocks(self) -> int:
        return self.gpu_capacity_bytes // max(self.block_size_bytes, 1)

    @property
    def max_cpu_blocks(self) -> int:
        return self.cpu_capacity_bytes // max(self.block_size_bytes, 1)

    @staticmethod
    def for_model(model_name: str, block_size: int = 16,
                  gpu_gb: int = 80, use_cuda: bool = True) -> "MemoryConfig":
        profile = None
        for key, prof in KNOWN_PROFILES.items():
            if key in model_name.lower() or model_name.lower() in key:
                profile = prof
                break
        return MemoryConfig(
            block_size=block_size,
            gpu_capacity_bytes=gpu_gb * 1024**3,
            model_profile=profile,
            use_cuda=use_cuda,
        )
