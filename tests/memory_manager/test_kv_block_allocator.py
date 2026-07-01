"""Unit tests for KVBlockAllocator — PagedAttention block pool."""

import pytest
from memory_manager.kv_block import KVBlockState
from memory_manager.kv_block_allocator import (
    KVBlockAllocator,
    OutOfMemoryError,
    BlockNotFoundError,
)
from memory_manager.config import MemoryConfig


# ═══════════════════════════════════════════════════════════════════
# Basic allocation
# ═══════════════════════════════════════════════════════════════════

class TestBasicAllocation:
    """Test block allocation and deallocation."""

    def test_allocate_single_block(self, allocator):
        """Allocate 10 tokens → 1 block."""
        ids = allocator.allocate("req-1", num_tokens=10)
        assert len(ids) == 1
        assert allocator.free_blocks == allocator.total_blocks - 1

    def test_allocate_exact_block_boundary(self, allocator):
        """16 tokens exactly fills one block."""
        ids = allocator.allocate("req-1", num_tokens=16)
        assert len(ids) == 1

    def test_allocate_spans_two_blocks(self, allocator):
        """17 tokens → 2 blocks."""
        ids = allocator.allocate("req-1", num_tokens=17)
        assert len(ids) == 2

    def test_allocate_zero_tokens(self, allocator):
        """0 tokens → 1 block (minimum allocation)."""
        ids = allocator.allocate("req-1", num_tokens=0)
        assert len(ids) == 1

    def test_allocate_large_request(self, allocator):
        """Large token count — verify block count."""
        ids = allocator.allocate("req-1", num_tokens=1500)
        expected = (1500 + 15) // 16  # ceil division
        assert len(ids) == expected

    def test_blocks_are_unique(self, allocator):
        """Allocated block IDs should be unique within a request."""
        ids_a = allocator.allocate("req-a", num_tokens=100)
        ids_b = allocator.allocate("req-b", num_tokens=100)
        assert set(ids_a).isdisjoint(set(ids_b))

    def test_blocks_initialised_correctly(self, allocator):
        """Each allocated block should be ALLOCATED with ref_count=1."""
        ids = allocator.allocate("req-1", num_tokens=50)
        for bid in ids:
            block = allocator.get_block(bid)
            assert block.state == KVBlockState.ALLOCATED
            assert block.ref_count == 1
            assert not block.is_free


class TestDeallocation:
    """Test freeing blocks."""

    def test_free_releases_all_blocks(self, allocator):
        """Free should release every block owned by a request."""
        allocator.allocate("req-1", num_tokens=100)
        before = allocator.free_blocks

        freed = allocator.free("req-1")
        assert freed > 0
        assert allocator.free_blocks == allocator.total_blocks
        assert allocator.used_blocks == 0

    def test_free_unknown_request_returns_zero(self, allocator):
        """Freeing a non-existent request should return 0."""
        assert allocator.free("no-such-request") == 0

    def test_double_free_is_harmless(self, allocator):
        """Freeing the same request twice is safe."""
        allocator.allocate("req-1", num_tokens=50)
        allocator.free("req-1")
        assert allocator.free("req-1") == 0

    def test_free_block_single(self, allocator):
        """Free a single block from a request."""
        ids = allocator.allocate("req-1", num_tokens=50)
        blocks_before = allocator.used_blocks

        # Free just the first block
        result = allocator.free_block("req-1", ids[0])
        assert result is True
        assert allocator.used_blocks == blocks_before - 1
        assert allocator.get_block(ids[0]).is_free

    def test_free_block_then_remaining_still_owned(self, allocator):
        """After freeing one block, remaining blocks are still tracked."""
        ids = allocator.allocate("req-1", num_tokens=50)
        allocator.free_block("req-1", ids[0])

        # Remaining blocks should still be owned by req-1
        remaining = allocator.get_request_blocks("req-1")
        assert ids[0] not in remaining
        for bid in ids[1:]:
            assert bid in remaining


# ═══════════════════════════════════════════════════════════════════
# Reference counting & sharing
# ═══════════════════════════════════════════════════════════════════

class TestReferenceCounting:
    """Test ref_count mechanics (vLLM Fig.6 / Fig.8)."""

    def test_increment_ref(self, allocator):
        """Incrementing ref_count should switch to SHARED state."""
        ids = allocator.allocate("req-1", num_tokens=10)
        bid = ids[0]

        allocator.increment_ref(bid)
        block = allocator.get_block(bid)
        assert block.ref_count == 2
        assert block.state == KVBlockState.SHARED

    def test_decrement_ref_to_one(self, allocator):
        """Decrement from 2→1 should return to ALLOCATED state."""
        ids = allocator.allocate("req-1", num_tokens=10)
        bid = ids[0]
        allocator.increment_ref(bid)  # ref_count = 2

        block = allocator.get_block(bid)
        block.decrement_ref()
        assert block.ref_count == 1
        assert block.state == KVBlockState.ALLOCATED

    def test_decrement_ref_to_zero_frees(self, allocator):
        """Decrement from 1→0 should mark the block FREE."""
        ids = allocator.allocate("req-1", num_tokens=10)
        bid = ids[0]

        block = allocator.get_block(bid)
        freed = block.decrement_ref()
        assert freed is True
        assert block.is_free

    def test_clone_block_creates_independent_copy(self, allocator):
        """clone_block() should produce a new block with identical token count."""
        ids = allocator.allocate("req-1", num_tokens=10)
        bid = ids[0]

        # Simulate sharing
        allocator.increment_ref(bid)  # ref_count = 2

        new_bid = allocator.clone_block("req-2", bid)
        assert new_bid != bid

        old_block = allocator.get_block(bid)
        new_block = allocator.get_block(new_bid)

        assert old_block.ref_count == 1  # decremented from 2
        assert new_block.ref_count == 1  # fresh block
        assert new_block.num_tokens == old_block.num_tokens

    def test_clone_not_needed_for_single_ref(self, allocator):
        """clone_block with ref_count=1 returns the same block."""
        ids = allocator.allocate("req-1", num_tokens=10)
        bid = ids[0]  # ref_count = 1

        result = allocator.clone_block("req-1", bid)
        assert result == bid  # no clone needed


# ═══════════════════════════════════════════════════════════════════
# Pinned blocks
# ═══════════════════════════════════════════════════════════════════

class TestPinnedBlocks:
    """Test system-prompt pinning."""

    def test_pin_blocks(self, allocator):
        """Pinning should set state to PINNED."""
        ids = allocator.allocate("req-1", num_tokens=30)
        allocator.pin_blocks("sys-prompt-v1", ids)

        for bid in ids:
            assert allocator.get_block(bid).state == KVBlockState.PINNED

    def test_unpin_restores_state(self, allocator):
        """Unpinning should restore to ALLOCATED (ref_count=1)."""
        ids = allocator.allocate("req-1", num_tokens=30)
        allocator.pin_blocks("sys-prompt-v1", ids)
        allocator.unpin_blocks("sys-prompt-v1")

        for bid in ids:
            assert allocator.get_block(bid).state == KVBlockState.ALLOCATED

    def test_unpin_shared_block(self, allocator):
        """Unpinning a shared block should go to SHARED state."""
        ids = allocator.allocate("req-1", num_tokens=30)
        allocator.increment_ref(ids[0])   # ref_count = 2
        allocator.pin_blocks("sp", ids)
        allocator.unpin_blocks("sp")

        # Block 0: ref_count=2 → SHARED; others: ref_count=1 → ALLOCATED
        assert allocator.get_block(ids[0]).state == KVBlockState.SHARED
        for bid in ids[1:]:
            assert allocator.get_block(bid).state == KVBlockState.ALLOCATED

    def test_pinned_blocks_counted(self, allocator):
        ids = allocator.allocate("req-1", num_tokens=16)
        assert allocator.pinned_blocks == 0
        allocator.pin_blocks("sp", ids)
        assert allocator.pinned_blocks == 1


# ═══════════════════════════════════════════════════════════════════
# Error cases
# ═══════════════════════════════════════════════════════════════════

class TestErrors:
    """Test error conditions."""

    def test_out_of_memory(self, allocator):
        """Allocating more than total blocks raises OutOfMemoryError."""
        max_tokens = allocator.total_blocks * 16 + 1
        with pytest.raises(OutOfMemoryError):
            allocator.allocate("req-huge", num_tokens=max_tokens)

    def test_block_not_found(self, allocator):
        """Accessing a non-existent block raises BlockNotFoundError."""
        with pytest.raises(BlockNotFoundError):
            allocator.get_block(999999)

    def test_oom_on_clone(self, allocator):
        """Clone should raise OOM when pool is full."""
        # Fill up the pool
        max_tokens = allocator.total_blocks * 16
        ids = allocator.allocate("req-1", num_tokens=max_tokens)
        allocator.increment_ref(ids[0])  # ref_count = 2

        # Clone needs one free block — there are none
        with pytest.raises(OutOfMemoryError):
            allocator.clone_block("req-2", ids[0])


# ═══════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════

class TestStatistics:
    """Test allocation statistics."""

    def test_stats_after_allocation(self, allocator):
        allocator.allocate("req-1", num_tokens=100)
        s = allocator.stats()
        assert s["used_blocks"] > 0
        assert s["active_requests"] == 1
        assert s["total_allocations"] > 0

    def test_stats_after_free(self, allocator):
        allocator.allocate("req-1", num_tokens=100)
        allocator.free("req-1")
        s = allocator.stats()
        assert s["used_blocks"] == 0
        assert s["active_requests"] == 0

    def test_usage_ratio(self, allocator):
        assert allocator.usage_ratio == 0.0
        allocator.allocate("req-1", num_tokens=allocator.total_blocks * 16 // 2)
        assert 0.4 < allocator.usage_ratio < 0.6

    def test_reset_stats(self, allocator):
        allocator.allocate("req-1", num_tokens=100)
        allocator.reset_stats()
        s = allocator.stats()
        assert s["total_allocations"] == 0
        assert s["total_frees"] == 0


# ═══════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Test concurrent access safety."""

    def test_concurrent_allocations(self, allocator):
        """Multiple threads allocating simultaneously should not corrupt."""
        import threading

        errors = []
        results = []

        def allocate_n(n: int):
            try:
                ids = allocator.allocate(f"thread-{n}", num_tokens=100)
                results.append(ids)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_n, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All allocated IDs across threads should be unique
        all_ids = [bid for r in results for bid in r]
        assert len(all_ids) == len(set(all_ids)), "Duplicate block IDs!"

    def test_concurrent_alloc_and_free(self, allocator):
        """Interleaved allocate/free should stay consistent."""
        import threading
        import random

        def worker(seed):
            random.seed(seed)
            for _ in range(20):
                try:
                    ids = allocator.allocate(f"w-{seed}-{_}", num_tokens=random.randint(1, 80))
                    allocator.free(f"w-{seed}-{_}")
                except OutOfMemoryError:
                    pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all workers finish, pool should be in a consistent state
        assert allocator.free_blocks == allocator.total_blocks


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    """Test MemoryConfig."""

    def test_factory_for_known_model(self):
        cfg = MemoryConfig.for_model("qwen2.5-7b", block_size=16, gpu_gb=80)
        assert cfg.model_profile is not None
        assert cfg.model_profile.model_family == "qwen2"

    def test_block_size_bytes_fallback(self):
        cfg = MemoryConfig(block_size=16)
        assert cfg.block_size_bytes > 0

    def test_max_blocks(self):
        cfg = MemoryConfig(block_size=16, gpu_capacity_bytes=10 * 1024**3)
        assert cfg.max_gpu_blocks > 0
