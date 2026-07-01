"""Shared fixtures for memory_manager tests."""
import pytest
from memory_manager.config import MemoryConfig
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager, BlockTable


@pytest.fixture
def config():
    """Small test config: 256 blocks of 16 tokens each."""
    return MemoryConfig(
        block_size=16,
        gpu_capacity_bytes=256 * 3_670_016,  # ~256 blocks for Qwen2.5-7B
    )


@pytest.fixture
def large_config():
    """Larger config for stress tests."""
    return MemoryConfig(
        block_size=16,
        gpu_capacity_bytes=1024 * 3_670_016,  # ~1024 blocks
    )


@pytest.fixture
def allocator(config):
    """Fresh allocator for each test."""
    return KVBlockAllocator(config)


@pytest.fixture
def table_manager(allocator, config):
    """Fresh block table manager for each test."""
    return BlockTableManager(allocator, config.block_size)


@pytest.fixture
def block_table(config):
    """A single block table with request_id='test'."""
    return BlockTable("test", config.block_size)
