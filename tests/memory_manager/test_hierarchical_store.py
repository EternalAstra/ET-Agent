"""Unit tests for HierarchicalKVStore — GPU→CPU→SSD tiered storage."""

import pytest
from memory_manager.kv_block import KVBlockState, StorageTier
from memory_manager.kv_hierarchical_store import HierarchicalKVStore


# ═══════════════════════════════════════════════════════════════════
# Basic operations
# ═══════════════════════════════════════════════════════════════════

class TestBasicStore:
    def test_initial_state(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        assert store.usage(StorageTier.GPU) == 0
        assert store.usage(StorageTier.CPU) == 0
        assert store.usage(StorageTier.SSD) == 0
        assert store.pending_migrations() == 0

    def test_get_location_default_gpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        assert store.get_location(999) == StorageTier.GPU  # unknown → default GPU

    def test_get_locations(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        locs = store.get_locations([1, 2, 3])
        assert locs[1] == StorageTier.GPU
        assert len(locs) == 3


# ═══════════════════════════════════════════════════════════════════
# Demotion
# ═══════════════════════════════════════════════════════════════════

class TestDemotion:
    def test_demote_gpu_to_cpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=32)

        n = store.demote_blocks(ids, StorageTier.CPU, "req-1")
        assert n == len(ids)
        for bid in ids:
            assert store.get_location(bid) == StorageTier.CPU

    def test_demote_gpu_to_ssd(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        n = store.demote_blocks(ids, StorageTier.SSD, "req-1")
        assert n == 1
        assert store.get_location(ids[0]) == StorageTier.SSD

    def test_demote_skips_pinned(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        allocator.pin_blocks("sp", ids)

        n = store.demote_blocks(ids, StorageTier.CPU, "req-1")
        assert n == 0  # pinned → skipped
        assert store.get_location(ids[0]) == StorageTier.GPU

    def test_demote_invalid_target(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        with pytest.raises(AssertionError):
            store.demote_blocks([1], StorageTier.GPU)

    def test_demote_updates_usage(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        gpu_before = store.usage(StorageTier.GPU)
        cpu_before = store.usage(StorageTier.CPU)

        store.demote_blocks(ids, StorageTier.CPU, "req-1")

        assert store.usage(StorageTier.GPU) < gpu_before
        assert store.usage(StorageTier.CPU) > cpu_before

    def test_demote_does_not_double_count(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        # First demotion: GPU→CPU
        store.demote_blocks(ids, StorageTier.CPU)
        cpu1 = store.usage(StorageTier.CPU)

        # Second demotion of same blocks: CPU→CPU is a no-op
        store.demote_blocks(ids, StorageTier.CPU)
        assert store.usage(StorageTier.CPU) == cpu1  # unchanged


# ═══════════════════════════════════════════════════════════════════
# Promotion
# ═══════════════════════════════════════════════════════════════════

class TestPromotion:
    def test_promote_cpu_to_gpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        store.demote_blocks(ids, StorageTier.CPU)
        assert store.get_location(ids[0]) == StorageTier.CPU

        n = store.promote_blocks(ids, StorageTier.GPU)
        assert n == 1
        assert store.get_location(ids[0]) == StorageTier.GPU

    def test_promote_ssd_to_cpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        store.demote_blocks(ids, StorageTier.SSD)
        assert store.get_location(ids[0]) == StorageTier.SSD

        # Promote SSD → GPU: two-step, first goes to CPU
        n = store.promote_blocks(ids, StorageTier.GPU)
        assert n == 1
        assert store.get_location(ids[0]) == StorageTier.CPU  # interim step

        # Second promotion: CPU → GPU
        n = store.promote_blocks(ids, StorageTier.GPU)
        assert n == 1
        assert store.get_location(ids[0]) == StorageTier.GPU

    def test_prefetch_for_resume(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        store.demote_blocks(ids, StorageTier.CPU)

        n = store.prefetch_for_resume("req-1", ids)
        assert n == 1
        assert store.get_location(ids[0]) == StorageTier.GPU

    def test_promote_already_on_gpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        n = store.promote_blocks(ids, StorageTier.GPU)
        assert n == 0  # already GPU → no-op


# ═══════════════════════════════════════════════════════════════════
# Eviction
# ═══════════════════════════════════════════════════════════════════

class TestEviction:
    def test_evict_blocks(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)

        n = store.evict_blocks(ids)
        assert n == 1
        # Block is now freed in the allocator
        assert allocator.get_block(ids[0]).is_free

    def test_evict_skips_pinned(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        allocator.pin_blocks("sp", ids)

        n = store.evict_blocks(ids)
        assert n == 0  # pinned → skipped
        assert not allocator.get_block(ids[0]).is_free

    def test_evict_from_gpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=100)
        store.register_blocks(ids)

        n = store.evict_from_gpu(10000)
        assert n > 0

    def test_evict_cold_blocks(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        allocator.allocate("req-1", num_tokens=200)
        before = allocator.used_blocks
        assert before > 0

        store.evict_cold_blocks(max_gpu_ratio=0.5)
        after = allocator.used_blocks
        assert after <= before  # some blocks freed


# ═══════════════════════════════════════════════════════════════════
# Usage ratios
# ═══════════════════════════════════════════════════════════════════

class TestUsageQueries:
    def test_usage_ratio_gpu(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        # GPU usage starts at 0
        assert store.usage_ratio(StorageTier.GPU) == 0.0

        ids = allocator.allocate("req-1", num_tokens=100)
        store.register_blocks(ids)
        # After allocation, GPU usage > 0
        assert store.usage_ratio(StorageTier.GPU) > 0

    def test_block_count_per_tier(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        assert store.block_count(StorageTier.GPU) == 1

        store.demote_blocks(ids, StorageTier.CPU)
        assert store.block_count(StorageTier.GPU) == 0
        assert store.block_count(StorageTier.CPU) == 1

        store.evict_blocks(ids)
        assert store.block_count(StorageTier.CPU) == 0

    def test_stats(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=32)
        store.demote_blocks(ids[:1], StorageTier.CPU)

        s = store.stats()
        assert "gpu_blocks" in s
        assert "cpu_blocks" in s
        assert s["total_migrations"] == 1

    def test_dump(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        ids = allocator.allocate("req-1", num_tokens=16)
        store.register_blocks(ids)
        store.demote_blocks(ids, StorageTier.CPU)

        dump = store.dump()
        assert "GPU" in dump
        assert "CPU" in dump
        assert "SSD" in dump


# ═══════════════════════════════════════════════════════════════════
# Store + Allocator integration
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_migration_cycle(self, config, allocator):
        """GPU → CPU → GPU → evict: full round-trip."""
        store = HierarchicalKVStore(config, allocator)

        # 1. Allocate
        ids = allocator.allocate("req-1", num_tokens=48)
        store.register_blocks(ids)
        assert store.block_count(StorageTier.GPU) == 3

        # 2. Demote to CPU
        store.demote_blocks(ids, StorageTier.CPU)
        assert store.block_count(StorageTier.GPU) == 0
        assert store.block_count(StorageTier.CPU) == 3

        # 3. Promote back
        store.promote_blocks(ids, StorageTier.GPU)
        assert store.block_count(StorageTier.GPU) == 3
        assert store.block_count(StorageTier.CPU) == 0

        # 4. Evict
        store.evict_blocks(ids)
        assert allocator.used_blocks == 0

    def test_multi_request_isolation(self, config, allocator):
        """Req-A's blocks on CPU don't affect Req-B's blocks on GPU."""
        store = HierarchicalKVStore(config, allocator)

        ids_a = allocator.allocate("req-a", num_tokens=32)
        ids_b = allocator.allocate("req-b", num_tokens=32)

        # Demote only A
        store.demote_blocks(ids_a, StorageTier.CPU)

        # B's blocks should still be on GPU
        for bid in ids_b:
            assert store.get_location(bid) == StorageTier.GPU
        for bid in ids_a:
            assert store.get_location(bid) == StorageTier.CPU

    def test_repr(self, config, allocator):
        store = HierarchicalKVStore(config, allocator)
        r = repr(store)
        assert "HierarchicalKVStore" in r
