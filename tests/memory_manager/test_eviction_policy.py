"""Unit tests for eviction policies — LRU, LFU, TieredLRU, AgentAware."""

import pytest
from memory_manager.kv_block import KVBlock, KVBlockState, StorageTier
from memory_manager.kv_eviction_policy import (
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    TieredLRUPolicy,
    AgentAwarePolicy,
    make_policy,
)


# ═══════════════════════════════════════════════════════════════════
# LRU
# ═══════════════════════════════════════════════════════════════════

class TestLRU:
    """Least-Recently-Used eviction."""

    def test_empty_returns_none(self):
        p = LRUEvictionPolicy()
        assert p.select_victims(5) == []

    def test_evicts_lru_first(self):
        p = LRUEvictionPolicy()
        for i in range(10):
            p.record_insert(i)

        # Access block 0 (most recent)
        p.record_access(0)

        # Evict 1 block — should be block 1 (now least recent because
        # 0 was re-accessed)
        victims = p.select_victims(1)
        assert victims == [1]

    def test_evicts_in_lru_order(self):
        p = LRUEvictionPolicy()
        for i in range(5):
            p.record_insert(i)

        # Access: 4, 2, 0 → reorder
        p.record_access(4)
        p.record_access(2)
        p.record_access(0)

        # LRU order should be: 1, 3, 4, 2, 0 (least→most recent)
        victims = p.select_victims(3)
        assert victims == [1, 3, 4]

    def test_access_moves_to_front(self):
        p = LRUEvictionPolicy()
        for i in range(3):
            p.record_insert(i)

        p.record_access(0)
        victims = p.select_victims(1)
        assert victims == [1]  # 1 is now LRU

    def test_remove(self):
        p = LRUEvictionPolicy()
        for i in range(5):
            p.record_insert(i)
        p.record_remove(2)
        assert p.size == 4
        victims = p.select_victims(5)
        assert 2 not in victims

    def test_clear(self):
        p = LRUEvictionPolicy()
        for i in range(5):
            p.record_insert(i)
        p.clear()
        assert p.size == 0
        assert p.select_victims(1) == []

    def test_skips_pinned_blocks(self):
        """Pinned blocks should never be returned as victims."""
        p = LRUEvictionPolicy()
        blocks = {}
        for i in range(5):
            p.record_insert(i)
            blocks[i] = KVBlock(block_id=i, state=KVBlockState.ALLOCATED)

        # Pin block 0
        blocks[0].state = KVBlockState.PINNED

        def getter(bid): return blocks.get(bid)

        victims = p.select_victims(2, block_getter=getter)
        assert 0 not in victims

    def test_skips_wrong_tier(self):
        p = LRUEvictionPolicy()
        blocks = {}
        for i in range(3):
            p.record_insert(i)
            blocks[i] = KVBlock(block_id=i)
        blocks[0].storage_tier = StorageTier.CPU

        def getter(bid): return blocks.get(bid)

        victims = p.select_victims(3, tier=StorageTier.GPU, block_getter=getter)
        assert 0 not in victims  # CPU block, not GPU


# ═══════════════════════════════════════════════════════════════════
# LFU
# ═══════════════════════════════════════════════════════════════════

class TestLFU:
    """Least-Frequently-Used eviction."""

    def test_evicts_lfu_first(self):
        p = LFUEvictionPolicy()
        for i in range(5):
            p.record_insert(i)

        # Access block 0 many times
        for _ in range(10):
            p.record_access(0)
        p.record_access(1)
        p.record_access(1)

        # Block 2 should be first victim (accessed 0 times)
        victims = p.select_victims(1)
        assert victims == [2]

    def test_freq_tiebreaker_uses_age(self):
        p = LFUEvictionPolicy()
        p.record_insert(10)
        p.record_insert(20)  # inserted later

        # Same frequency (0), so older (10) should go first
        victims = p.select_victims(1)
        assert victims == [10]

    def test_remove(self):
        p = LFUEvictionPolicy()
        for i in range(3):
            p.record_insert(i)
        p.record_remove(1)
        assert p.size == 2
        victims = p.select_victims(3)
        assert 1 not in victims

    def test_clear(self):
        p = LFUEvictionPolicy()
        for i in range(5):
            p.record_insert(i)
        p.clear()
        assert p.size == 0


# ═══════════════════════════════════════════════════════════════════
# TieredLRU
# ═══════════════════════════════════════════════════════════════════

class TestTieredLRU:
    """Per-tier LRU eviction."""

    def test_per_tier_isolation(self):
        p = TieredLRUPolicy()
        for i in range(5):
            p.record_insert(i, StorageTier.GPU)
        for i in range(100, 103):
            p.record_insert(i, StorageTier.CPU)

        assert p.size_for_tier(StorageTier.GPU) == 5
        assert p.size_for_tier(StorageTier.CPU) == 3

        # Evict from GPU only
        victims = p.select_victims(2, tier=StorageTier.GPU)
        assert all(v < 100 for v in victims)  # all from GPU range
        assert len(victims) == 2

    def test_moving_between_tiers(self):
        p = TieredLRUPolicy()
        p.record_insert(42, StorageTier.GPU)
        p.record_remove(42, StorageTier.GPU)
        p.record_insert(42, StorageTier.CPU)

        # GPU should be empty, CPU should have it
        assert p.size_for_tier(StorageTier.GPU) == 0
        assert p.size_for_tier(StorageTier.CPU) == 1

        victims = p.select_victims(1, tier=StorageTier.CPU)
        assert victims == [42]

    def test_clear(self):
        p = TieredLRUPolicy()
        for tier in StorageTier:
            for i in range(5):
                p.record_insert(i + 100 * list(StorageTier).index(tier), tier)
        p.clear()
        assert p.size == 0
        for tier in StorageTier:
            assert p.size_for_tier(tier) == 0


# ═══════════════════════════════════════════════════════════════════
# AgentAware
# ═══════════════════════════════════════════════════════════════════

class TestAgentAware:
    """Agent-aware LRU with group protection."""

    def test_protects_groups(self):
        p = AgentAwarePolicy(protected_groups={"active-session"})
        blocks = {}
        for i in range(5):
            p.record_insert(i)
            blocks[i] = KVBlock(block_id=i, group_id="active-session")

        def getter(bid): return blocks.get(bid)

        victims = p.select_victims(5, block_getter=getter)
        assert victims == []  # all protected

    def test_unprotected_evictable(self):
        p = AgentAwarePolicy(protected_groups={"safe"})
        blocks = {}
        for i in range(3):
            p.record_insert(i)
            blocks[i] = KVBlock(
                block_id=i,
                group_id="safe" if i == 0 else "evictable",
            )

        def getter(bid): return blocks.get(bid)

        victims = p.select_victims(2, block_getter=getter)
        assert 0 not in victims  # protected
        assert 1 in victims      # evictable

    def test_protect_unprotect(self):
        p = AgentAwarePolicy()
        p.protect_group("group-a")
        assert p.protected_count == 1
        p.unprotect_group("group-a")
        assert p.protected_count == 0

    def test_skips_pinned(self):
        p = AgentAwarePolicy(protected_groups={"a"})
        blocks = {}
        for i in range(3):
            p.record_insert(i)
            blocks[i] = KVBlock(block_id=i, group_id="a")
        blocks[1].state = KVBlockState.PINNED  # super-protected

        def getter(bid): return blocks.get(bid)

        # Both agent-protected AND pinned → both skipped
        victims = p.select_victims(3, block_getter=getter)
        assert victims == []


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

class TestFactory:
    def test_make_lru(self):
        p = make_policy("lru")
        assert isinstance(p, LRUEvictionPolicy)
        assert p.name == "LRU"

    def test_make_lfu(self):
        p = make_policy("lfu")
        assert isinstance(p, LFUEvictionPolicy)

    def test_make_tiered(self):
        p = make_policy("tiered_lru")
        assert isinstance(p, TieredLRUPolicy)

    def test_make_agent_aware_with_groups(self):
        p = make_policy("agent_aware", protected_groups={"a", "b"})
        assert isinstance(p, AgentAwarePolicy)
        assert p.protected_count == 2

    def test_make_invalid(self):
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            make_policy("nonexistent")
