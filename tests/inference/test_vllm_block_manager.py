"""Unit tests for VLLMBlockManager — vLLM-compatible KV cache block management."""
import pytest
from memory_manager.config import MemoryConfig
from memory_manager.kv_block_allocator import KVBlockAllocator
from memory_manager.block_table import BlockTableManager
from inference.vllm_block_manager import (
    VLLMBlockManager, SequenceBlocks, BlockSwapOp,
)
from inference.swapping_engine import (
    HiFCSwappingEngine, FlashZone, SwapResult,
)


@pytest.fixture
def config():
    return MemoryConfig(block_size=16, gpu_capacity_bytes=256 * 3_670_016)


@pytest.fixture
def allocator(config):
    return KVBlockAllocator(config)


@pytest.fixture
def block_tables(allocator, config):
    return BlockTableManager(allocator, config.block_size)


@pytest.fixture
def vllm_bm(config, allocator, block_tables):
    return VLLMBlockManager(config, allocator, block_tables, enable_fc=True)


@pytest.fixture
def swapping():
    return HiFCSwappingEngine(block_size_bytes=3_670_016)


# ═══════════════════════════════════════════════════
# Basic vLLM API
# ═══════════════════════════════════════════════════

class TestVLLMBasic:
    def test_allocate_single_sequence(self, vllm_bm):
        seq = vllm_bm.allocate(1, num_tokens=100)
        assert seq.seq_id == 1
        assert seq.num_blocks > 0
        assert seq.num_tokens == 100
        assert not seq.is_swapped

    def test_allocate_multiple_sequences(self, vllm_bm):
        s1 = vllm_bm.allocate(1, 50)
        s2 = vllm_bm.allocate(2, 80)
        assert s1.block_table != s2.block_table
        assert all(b1 != b2 for b1, b2 in zip(s1.block_table, s2.block_table))

    def test_free_sequence(self, vllm_bm):
        vllm_bm.allocate(1, 100)
        before = vllm_bm.get_num_free_gpu_blocks()
        vllm_bm.free(1)
        assert vllm_bm.get_num_free_gpu_blocks() == vllm_bm._allocator.total_blocks

    def test_can_allocate(self, vllm_bm):
        assert vllm_bm.can_allocate(50)
        # FC (Flash Cache) extends capacity — check both GPU-only and FC+GPU
        max_gpu_tokens = vllm_bm._allocator.total_blocks * vllm_bm._config.block_size
        max_fc_tokens = (vllm_bm._allocator.total_blocks + vllm_bm._fc_capacity) * vllm_bm._config.block_size
        assert not vllm_bm.can_allocate(max_fc_tokens + 1)  # exceeds GPU+FC
        assert vllm_bm.can_allocate(max_gpu_tokens)  # FC covers this

    def test_get_block_table(self, vllm_bm):
        seq = vllm_bm.allocate(1, 50)
        bt = vllm_bm.get_block_table(1)
        assert bt == seq.block_table
        assert len(bt) > 0

    def test_append_slot(self, vllm_bm):
        vllm_bm.allocate(1, 15)  # 1 block, 1 slot free
        # Append 1 token → fills existing block
        bid = vllm_bm.append_slot(1)
        assert bid is None  # no new block, filled existing slot
        # Append another → needs new block (block_size=16, now has 17 tokens)
        bid2 = vllm_bm.append_slot(1)
        assert bid2 is not None  # new block allocated


# ═══════════════════════════════════════════════════
# COW / Fork (parallel sampling)
# ═══════════════════════════════════════════════════

class TestFork:
    def test_fork_shares_blocks(self, vllm_bm):
        parent = vllm_bm.allocate(10, num_tokens=200)
        child = vllm_bm.fork(10, 20)
        # Child shares parent's physical blocks
        assert child.block_table == parent.block_table
        assert child.num_tokens == parent.num_tokens

    def test_fork_unknown_parent_raises(self, vllm_bm):
        with pytest.raises(KeyError):
            vllm_bm.fork(99999, 1)


# ═══════════════════════════════════════════════════
# HiFC swap_in / swap_out
# ═══════════════════════════════════════════════════

class TestSwap:
    def test_swap_out_and_in(self, vllm_bm, swapping):
        seq = vllm_bm.allocate(1, num_tokens=100)
        assert not seq.is_swapped

        op = vllm_bm.swap_out(1)
        assert op == BlockSwapOp.SWAP_OUT
        assert seq.is_swapped

        op2 = vllm_bm.swap_in(1)
        assert op2 == BlockSwapOp.SWAP_IN
        assert not seq.is_swapped

    def test_swap_out_already_swapped_is_noop(self, vllm_bm):
        vllm_bm.allocate(1, 100)
        vllm_bm.swap_out(1)
        op = vllm_bm.swap_out(1)
        assert op == BlockSwapOp.NOOP

    def test_swap_updates_fc_counters(self, vllm_bm):
        vllm_bm.allocate(1, 100)
        fc_before = vllm_bm.get_num_free_fc_blocks()
        vllm_bm.swap_out(1)
        assert vllm_bm.get_num_free_fc_blocks() < fc_before


# ═══════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════

class TestStats:
    def test_stats_after_swap(self, vllm_bm):
        vllm_bm.allocate(1, 200)
        vllm_bm.swap_out(1)
        s = vllm_bm.stats()
        assert s["total_swaps_out"] == 1
        assert s["swapped_sequences"] == 1

    def test_dump(self, vllm_bm):
        vllm_bm.allocate(1, 50)
        d = vllm_bm.dump()
        assert "seq" in d and "blocks" in d

    def test_reset_stats(self, vllm_bm):
        vllm_bm.allocate(1, 100)
        vllm_bm.reset_stats()
        assert vllm_bm.stats()["total_swaps_out"] == 0


# ═══════════════════════════════════════════════════
# Swapping engine
# ═══════════════════════════════════════════════════

class TestSwappingEngine:
    def test_swap_out_modeled_performance(self, swapping):
        result = swapping.swap_out(1, [10, 11, 12, 13, 14])
        assert result.success
        assert result.throughput_gbps > 0
        assert result.elapsed_us > 0
        assert result.zone == FlashZone.pSLC

    def test_swap_in_modeled_performance(self, swapping):
        result = swapping.swap_in(2, [20, 21, 22])
        assert result.success
        assert result.throughput_gbps > 0

    def test_zone_performance_differs(self, swapping):
        r_pslc = swapping.swap_out(1, [10, 11], zone=FlashZone.pSLC)
        r_tlc = swapping.swap_out(2, [20, 21], zone=FlashZone.TLC)
        # pSLC is faster (lower latency)
        assert r_pslc.throughput_gbps > r_tlc.throughput_gbps

    def test_stats_after_swaps(self, swapping):
        swapping.swap_out(1, [1, 2, 3])
        swapping.swap_in(1, [1, 2, 3])
        s = swapping.stats()
        assert s["total_swaps"] == 2
        assert s["total_bytes_transferred"] > 0

    def test_reset_stats(self, swapping):
        swapping.swap_out(1, [1])
        swapping.reset_stats()
        assert swapping.stats()["total_swaps"] == 0

    def test_tbw_tracking(self, swapping):
        swapping.swap_out(1, [1]*10)  # 10 blocks
        remaining = swapping.tbw_remaining(FlashZone.pSLC)
        assert remaining < 2400  # less than max TBW
