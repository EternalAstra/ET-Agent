"""
Memory Manager configuration.

All tunable parameters for the KV Cache block allocator, block table manager,
and future hierarchical storage / eviction systems.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model-specific presets (block size in tokens, bytes-per-block estimates)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelKVProfile:
    """Pre-computed KV Cache sizing for a given model architecture."""

    model_family: str       # "qwen2", "deepseek-v4", "llama3", etc.
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int = 2   # FP16 = 2 bytes

    @property
    def bytes_per_token(self) -> int:
        """Bytes needed for one token's KV Cache across all layers.

        K + V = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def bytes_per_block(self, block_size: int) -> int:
        """Bytes for one full block (block_size tokens)."""
        return self.bytes_per_token * block_size


# Known profiles (extend as needed)
KNOWN_PROFILES: dict[str, ModelKVProfile] = {
    "qwen2.5-7b": ModelKVProfile(
        model_family="qwen2",
        num_layers=28,
        num_kv_heads=4,     # GQA: 4 KV heads, 28 Q heads
        head_dim=128,
    ),
    "qwen2.5-14b": ModelKVProfile(
        model_family="qwen2",
        num_layers=48,
        num_kv_heads=8,
        head_dim=128,
    ),
    "deepseek-v4": ModelKVProfile(
        model_family="deepseek-v4",
        num_layers=60,          # estimated; DeepSeek V4 uses MLA (smaller KV)
        num_kv_heads=1,         # MLA compresses KV to single latent vector
        head_dim=512,           # compressed dimension
    ),
    "minicpm3-4b": ModelKVProfile(
        model_family="minicpm",
        num_layers=32,
        num_kv_heads=4,
        head_dim=128,
    ),
}


# ---------------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """Global configuration for the memory manager.

    Attributes
    ----------
    block_size : int
        Number of tokens per KV Cache block (default 16; vLLM paper finds
        16–256 works well, with smaller blocks giving better utilisation).
    gpu_capacity_bytes : int
        Total GPU VRAM available for KV Cache blocks.
    cpu_capacity_bytes : int
        Total CPU DRAM available for swapped-out KV Cache blocks.
    ssd_capacity_bytes : int
        Total NVMe SSD capacity for cold-storage KV Cache blocks.
    enable_ssd : bool
        Whether to enable SSD tier (Phase 3).
    prefill_block_margin : int
        Extra blocks to pre-allocate for prefill to avoid mid-prefill OOM.
    max_shared_blocks_pct : float
        Maximum percentage of total blocks that can be shared (safety cap).
    """

    block_size: int = 16
    gpu_capacity_bytes: int = 80 * 1024**3       # 80 GB
    cpu_capacity_bytes: int = 512 * 1024**3      # 512 GB
    ssd_capacity_bytes: int = 2 * 1024**4        # 2 TB
    enable_ssd: bool = False                      # off until Phase 3
    prefill_block_margin: int = 8
    max_shared_blocks_pct: float = 0.95

    # ── derived (computed after init via __post_init__) ──
    model_profile: ModelKVProfile | None = None

    def __post_init__(self):
        if self.model_profile is not None and isinstance(self.model_profile, dict):
            self.model_profile = ModelKVProfile(**self.model_profile)

    @property
    def block_size_bytes(self) -> int:
        """Bytes per block for the configured model profile."""
        if self.model_profile is not None:
            return self.model_profile.bytes_per_block(self.block_size)
        # Fallback: Qwen2.5-7B estimate
        return 2 * 28 * 4 * 128 * 2 * self.block_size  # ~3.7 MB for block_size=16

    @property
    def max_gpu_blocks(self) -> int:
        """Maximum number of blocks that fit in GPU VRAM."""
        return self.gpu_capacity_bytes // max(self.block_size_bytes, 1)

    @property
    def max_cpu_blocks(self) -> int:
        """Maximum number of blocks that fit in CPU DRAM."""
        return self.cpu_capacity_bytes // max(self.block_size_bytes, 1)

    @staticmethod
    def for_model(model_name: str, block_size: int = 16,
                  gpu_gb: int = 80) -> "MemoryConfig":
        """Factory: create config tuned for a specific model."""
        profile = None
        for key, prof in KNOWN_PROFILES.items():
            if key in model_name.lower():
                profile = prof
                break
        return MemoryConfig(
            block_size=block_size,
            gpu_capacity_bytes=gpu_gb * 1024**3,
            model_profile=profile,
        )
