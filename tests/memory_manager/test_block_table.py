"""Unit tests for BlockTable and BlockTableManager — logical→physical mapping."""

import pytest
from memory_manager.block_table import BlockTable, BlockTableManager


# ═══════════════════════════════════════════════════════════════════
# BlockTable — single request
# ═══════════════════════════════════════════════════════════════════

class TestBlockTable:
    """Test a single request's logical→physical mapping."""

    def test_empty_table(self, block_table):
        assert len(block_table) == 0
        assert block_table.last_logical_idx == -1
        assert block_table.get_physical_blocks() == []

    def test_add_and_lookup(self, block_table):
        block_table.add_entry(logical_idx=0, physical_id=42, num_filled=10)
        entry = block_table[0]
        assert entry.logical_idx == 0
        assert entry.physical_id == 42
        assert entry.num_filled == 10
        assert not entry.is_cow

    def test_append_blocks(self, block_table):
        start = block_table.append_blocks([100, 101, 102],
                                          tokens_per_block=[16, 16, 8])
        assert start == 0
        assert len(block_table) == 3
        assert block_table[0].physical_id == 100
        assert block_table[2].physical_id == 102
        assert block_table[2].num_filled == 8

    def test_append_blocks_after_existing(self, block_table):
        block_table.add_entry(0, physical_id=10, num_filled=16)
        start = block_table.append_blocks([20, 21],
                                          tokens_per_block=[16, 4])
        assert start == 1
        assert block_table[0].physical_id == 10
        assert block_table[1].physical_id == 20
        assert block_table[2].physical_id == 21

    def test_get_physical_blocks_ordered(self, block_table):
        """Physical blocks must be returned in logical order."""
        # Add out of logical order
        block_table.add_entry(2, physical_id=30)
        block_table.add_entry(0, physical_id=10)
        block_table.add_entry(1, physical_id=20)

        result = block_table.get_physical_blocks()
        assert result == [10, 20, 30]

    def test_total_tokens(self, block_table):
        block_table.append_blocks([1, 2, 3],
                                  tokens_per_block=[16, 16, 8])
        assert block_table.total_tokens == 40

    def test_remove_entry(self, block_table):
        block_table.append_blocks([10, 11], tokens_per_block=[16, 8])
        removed = block_table.remove_entry(0)
        assert removed is not None
        assert removed.physical_id == 10
        assert len(block_table) == 1
        assert block_table[1].physical_id == 11

    def test_trim_from(self, block_table):
        block_table.append_blocks([10, 11, 12, 13],
                                  tokens_per_block=[16, 16, 16, 8])
        removed = block_table.trim_from(2)
        assert len(removed) == 2
        assert len(block_table) == 2
        assert block_table[0].physical_id == 10
        assert block_table[1].physical_id == 11

    def test_cow_entries(self, block_table):
        block_table.append_blocks([10, 11], tokens_per_block=[16, 16])
        block_table._entries[0].is_cow = True
        cows = block_table.get_cow_entries()
        assert cows == [0]

        block_table.clear_cow(0)
        assert block_table.get_cow_entries() == []

    def test_last_block_has_room(self, block_table):
        """Initially empty: no room. After append with partial fill: has room."""
        assert not block_table.last_block_has_room()

        block_table.append_blocks([10], tokens_per_block=[8])
        assert block_table.last_block_has_room()

        # Make block full
        block_table._entries[0].num_filled = 16
        assert not block_table.last_block_has_room()


# ═══════════════════════════════════════════════════════════════════
# BlockTableManager
# ═══════════════════════════════════════════════════════════════════

class TestBlockTableManager:
    """Test the registry of per-request block tables."""

    def test_create_table(self, table_manager):
        t = table_manager.create_table("req-1")
        assert t.request_id == "req-1"
        assert table_manager.has_table("req-1")

    def test_create_table_idempotent(self, table_manager):
        t1 = table_manager.create_table("req-1")
        t2 = table_manager.create_table("req-1")
        assert t1 is t2  # same object

    def test_remove_table(self, table_manager):
        table_manager.create_table("req-1")
        table_manager.remove_table("req-1")
        assert not table_manager.has_table("req-1")

    def test_get_table_missing(self, table_manager):
        assert table_manager.get_table("no-such") is None

    def test_active_requests(self, table_manager):
        for i in range(5):
            table_manager.create_table(f"req-{i}")
        assert table_manager.active_requests() == 5

    def test_get_physical_blocks(self, table_manager):
        t = table_manager.create_table("req-1")
        t.append_blocks([100, 101, 102])
        assert table_manager.get_physical_blocks("req-1") == [100, 101, 102]


# ═══════════════════════════════════════════════════════════════════
# Prefix sharing (vLLM §4.4)
# ═══════════════════════════════════════════════════════════════════

class TestPrefixSharing:
    """Test vLLM-style prefix sharing between requests."""

    def test_find_shared_prefix_basic(self, allocator, config):
        """Two requests with identical first blocks should detect the shared prefix."""
        mgr = BlockTableManager(allocator, config.block_size)

        # Allocate blocks for req-A
        ids_a = allocator.allocate("req-a", num_tokens=32)
        ta = mgr.create_table("req-a")
        ta.append_blocks(ids_a, tokens_per_block=[16, 16])

        # Allocate blocks for req-B
        ids_b = allocator.allocate("req-b", num_tokens=32)
        tb = mgr.create_table("req-b")
        tb.append_blocks(ids_b, tokens_per_block=[16, 16])

        # No sharing yet — physical blocks are different
        shared = mgr.find_shared_prefix("req-a", "req-b")
        assert shared == 0  # different physical blocks

    def test_share_prefix_structure(self, allocator, config):
        """After sharing, target should point to source's physical blocks."""
        mgr = BlockTableManager(allocator, config.block_size)

        # Source request
        src_ids = allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_table.append_blocks(src_ids, tokens_per_block=[16, 16])

        # Target request — create empty table, then share prefix
        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        # Target's physical blocks should match source's
        tgt_blocks = mgr.get_physical_blocks("tgt")
        assert tgt_blocks == src_ids

    def test_share_prefix_last_block_cow(self, allocator, config):
        """The last shared block should be flagged COW (vLLM Fig.8)."""
        mgr = BlockTableManager(allocator, config.block_size)

        src_ids = allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_table.append_blocks(src_ids, tokens_per_block=[16, 16])

        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        tgt_table = mgr.get_table("tgt")
        # Last shared block (index 1) should be COW
        assert tgt_table[1].is_cow
        assert not tgt_table[0].is_cow

    def test_share_prefix_ref_counts(self, allocator, config):
        """After sharing, physical blocks' ref_counts should reflect sharing."""
        mgr = BlockTableManager(allocator, config.block_size)

        src_ids = allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_table.append_blocks(src_ids, tokens_per_block=[16, 16])

        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        # Both blocks shared → ref_count should be 2
        for bid in src_ids:
            assert allocator.get_block(bid).ref_count == 2
            assert allocator.get_block(bid).is_shared

    def test_find_shared_prefix_after_sharing(self, allocator, config):
        """After sharing, find_shared_prefix should return the full prefix length."""
        mgr = BlockTableManager(allocator, config.block_size)

        src_ids = allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_table.append_blocks(src_ids, tokens_per_block=[16, 16])

        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        shared_tokens = mgr.find_shared_prefix("src", "tgt")
        assert shared_tokens == 32  # 2 blocks × 16 tokens


# ═══════════════════════════════════════════════════════════════════
# COW-aware writes (vLLM §4.3)
# ═══════════════════════════════════════════════════════════════════

class TestCOW:
    """Test copy-on-write mechanics."""

    def test_ensure_writable_no_sharing(self, allocator, config):
        """Single-owner block: ensure_writable should return the same block."""
        mgr = BlockTableManager(allocator, config.block_size)

        ids = allocator.allocate("req", num_tokens=16)
        t = mgr.create_table("req")
        t.append_blocks(ids, tokens_per_block=[10])

        result = mgr.ensure_writable("req", 0)
        assert result == ids[0]

    def test_ensure_writable_with_cow(self, allocator, config):
        """COW-flagged block: ensure_writable should clone."""
        mgr = BlockTableManager(allocator, config.block_size)

        allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_ids = allocator.get_request_blocks("src")
        src_table.append_blocks(sorted(src_ids), tokens_per_block=[16, 16])

        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        # tgt[1] is COW — ensure_writable should clone it
        old_phys = mgr.get_table("tgt")[1].physical_id
        new_phys = mgr.ensure_writable("tgt", 1)
        assert new_phys != old_phys

        # Old block should have ref_count decremented
        assert allocator.get_block(old_phys).ref_count == 1  # only src still refs it

    def test_cow_clears_after_write(self, allocator, config):
        """After ensure_writable, the COW flag should be cleared."""
        mgr = BlockTableManager(allocator, config.block_size)

        allocator.allocate("src", num_tokens=32)
        src_table = mgr.create_table("src")
        src_ids = allocator.get_request_blocks("src")
        src_table.append_blocks(sorted(src_ids), tokens_per_block=[16, 16])

        mgr.create_table("tgt")
        mgr.share_prefix("src", "tgt", prefix_blocks=2)

        tgt_entry = mgr.get_table("tgt")[1]
        assert tgt_entry.is_cow  # was set by share_prefix

        mgr.ensure_writable("tgt", 1)
        assert not tgt_entry.is_cow  # cleared after COW clone
        assert tgt_entry.shared_from is None


# ═══════════════════════════════════════════════════════════════════
# Integration: allocator + block table
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end: allocate → map → write → free."""

    def test_full_lifecycle(self, allocator, config):
        """Simulate a complete request lifecycle."""
        mgr = BlockTableManager(allocator, config.block_size)

        # 1. Prefill: allocate 200 tokens
        ids = allocator.allocate("req-1", num_tokens=200)
        assert len(ids) == 13  # ceil(200/16)

        # 2. Build block table
        t = mgr.create_table("req-1")
        num_blocks = len(ids)
        fills = [16] * (num_blocks - 1) + [8]  # last block partial
        t.append_blocks(ids, tokens_per_block=fills)

        assert t.total_tokens == (num_blocks - 1) * 16 + 8

        # 3. Decoding: append tokens one at a time
        # Append 8 more tokens → fills last block's remaining 8 slots
        new_ids = allocator.allocate("req-1", num_tokens=16)
        consumed, remaining = t.append_tokens(8, new_ids)
        # 8 tokens exactly fill the last block (8→16 slots filled)
        # No new block consumed
        assert consumed == 0, "8 tokens fit in existing last block"
        assert remaining == 0, "no spillover"
        assert t.total_tokens == num_blocks * 16  # all blocks full
        assert not t.last_block_has_room()  # last block is now full (16/16)

        # 4. Free everything
        allocator.free("req-1")
        mgr.remove_table("req-1")
        assert allocator.usage_ratio == 0.0
        assert mgr.active_requests() == 0

    def test_two_requests_with_sharing(self, allocator, config):
        """Two requests sharing a common system prompt prefix."""
        mgr = BlockTableManager(allocator, config.block_size)

        # Request A: system prompt (80 tokens = 5 blocks)
        sys_ids = allocator.allocate("sys", num_tokens=80)
        sys_table = mgr.create_table("sys")
        sys_table.append_blocks(sys_ids, tokens_per_block=[16]*5)

        # Request B: same system prompt + user message
        # Share the 5 system-prompt blocks
        mgr.create_table("user-b")
        mgr.share_prefix("sys", "user-b", prefix_blocks=5)

        # Now user-b appends its own user-message blocks
        user_ids = allocator.allocate("user-b", num_tokens=48)
        user_table = mgr.get_table("user-b")
        user_table.append_blocks(user_ids, tokens_per_block=[16, 16, 16])

        # Verify: user-b's first 5 blocks point to sys blocks
        assert mgr.get_physical_blocks("user-b")[:5] == sys_ids
        # Verify: system blocks are shared
        for bid in sys_ids:
            assert allocator.get_block(bid).is_shared
        # Verify: user-specific blocks are NOT shared
        for bid in user_ids:
            assert not allocator.get_block(bid).is_shared


# ═══════════════════════════════════════════════════════════════════
# Debug helpers
# ═══════════════════════════════════════════════════════════════════

class TestDebug:
    """Test dump and debug utilities."""

    def test_dump_table(self, allocator, config):
        mgr = BlockTableManager(allocator, config.block_size)
        ids = allocator.allocate("req", num_tokens=32)
        t = mgr.create_table("req")
        t.append_blocks(ids, tokens_per_block=[16, 16])

        dump = mgr.dump_table("req")
        assert "req" in dump
        assert "P" in dump
        assert "filled=" in dump
        assert "2 entries" in dump

    def test_dump_empty_table(self, allocator, config):
        mgr = BlockTableManager(allocator, config.block_size)
        dump = mgr.dump_table("no-such")
        assert "empty" in dump

    def test_repr(self, block_table):
        block_table.append_blocks([1, 2], tokens_per_block=[16, 16])
        r = repr(block_table)
        assert "BlockTable" in r
        assert "test" in r
