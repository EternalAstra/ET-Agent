"""Unit tests for prefix hash cache + agent prefix cache."""

import pytest
from memory_manager.kv_prefix_cache import (
    PrefixHashCache,
    PrefixCacheEntry,
    compute_prefix_hashes,
    compute_block_hash,
)
from memory_manager.agent_prefix_cache import (
    AgentPrefixCache,
    estimate_agent_savings,
)


# ═══════════════════════════════════════════════════════════════════
# Hash-chain computation
# ═══════════════════════════════════════════════════════════════════

class TestHashComputation:
    """MoonCake §3, Figure 3: hash-chain computation."""

    def test_compute_empty_list(self):
        h = compute_prefix_hashes([], 16)
        assert h == []

    def test_compute_single_block(self):
        tokens = list(range(16))
        h = compute_prefix_hashes(tokens, 16)
        assert len(h) == 1
        assert len(h[0]) == 16  # hex digest length

    def test_compute_multiple_blocks(self):
        tokens = list(range(48))
        h = compute_prefix_hashes(tokens, 16)
        assert len(h) == 3

    def test_partial_last_block(self):
        tokens = list(range(20))
        h = compute_prefix_hashes(tokens, 16)
        assert len(h) == 2  # 1 full + 1 partial

    def test_same_tokens_produce_same_hash(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        h1 = compute_prefix_hashes(tokens, 16)
        h2 = compute_prefix_hashes(tokens, 16)
        assert h1 == h2

    def test_different_tokens_produce_different_hash(self):
        t1 = list(range(16))
        t2 = list(range(1, 17))
        h1 = compute_prefix_hashes(t1, 16)
        h2 = compute_prefix_hashes(t2, 16)
        assert h1 != h2

    def test_prefix_changes_chain(self):
        """Changing the first block changes all subsequent hashes."""
        tokens_a = list(range(32))
        tokens_b = list(range(32))
        tokens_b[0] = 999  # change first token

        ha = compute_prefix_hashes(tokens_a, 16)
        hb = compute_prefix_hashes(tokens_b, 16)

        # First blocks differ...
        assert ha[0] != hb[0]
        # ... and because it's a chain, all subsequent hashes differ too
        assert ha[1] != hb[1]

    def test_shared_prefix_same_hashes(self):
        """Two sequences sharing a prefix should have identical first hashes."""
        t1 = list(range(50))
        t2 = list(range(50))  # same first 50 tokens
        t2.extend([100, 101, 102])  # different suffix

        h1 = compute_prefix_hashes(t1, 16)
        h2 = compute_prefix_hashes(t2, 16)

        # First 3 blocks (48 tokens) identical → hashes match
        assert h1[:3] == h2[:3]
        # Block 4 differs because base changed
        assert h1[3:] != h2[3:]

    def test_hash_deterministic_across_calls(self):
        """Same input across two separate calls gives same hashes."""
        tokens = list(range(128))
        h1 = compute_prefix_hashes(tokens, 16)
        h2 = compute_prefix_hashes(tokens, 16)
        assert h1 == h2

    def test_compute_block_hash_single(self):
        h = compute_block_hash([1, 2, 3], prev_hash="abc")
        assert len(h) == 16

    def test_block_size_variants(self):
        for bs in [8, 16, 32, 64]:
            h = compute_prefix_hashes(list(range(100)), bs)
            expected_blocks = (100 + bs - 1) // bs
            assert len(h) == expected_blocks, f"block_size={bs}"


# ═══════════════════════════════════════════════════════════════════
# PrefixCacheEntry
# ═══════════════════════════════════════════════════════════════════

class TestCacheEntry:
    def test_create(self):
        e = PrefixCacheEntry("abcd", [1, 2, 3])
        assert e.block_hash == "abcd"
        assert e.block_ids == [1, 2, 3]
        assert not e.is_pinned
        assert not e.is_hot

    def test_pinned(self):
        e = PrefixCacheEntry("abcd", [1], is_pinned=True)
        assert e.is_pinned

    def test_touch(self):
        e = PrefixCacheEntry("x", [1])
        e.touch()
        assert e.access_count == 1
        e.touch()
        assert e.access_count == 2
        assert not e.is_hot  # needs 10
        for _ in range(8):
            e.touch()
        assert e.is_hot  # 10 accesses

    def test_add_remove_block_ids(self):
        e = PrefixCacheEntry("h", [1, 2])
        e.add_block_id(3)
        assert e.block_ids == [1, 2, 3]
        e.add_block_id(1)  # duplicate, skipped
        assert e.block_ids == [1, 2, 3]
        e.remove_block_id(2)
        assert e.block_ids == [1, 3]

    def test_repr(self):
        e = PrefixCacheEntry("deadbeef1234", [42], is_pinned=True)
        r = repr(e)
        assert "deadbeef1234" in r
        assert "P" in r or "🔒" in r


# ═══════════════════════════════════════════════════════════════════
# PrefixHashCache
# ═══════════════════════════════════════════════════════════════════

class TestPrefixHashCache:
    """MoonCake-style prefix hash cache."""

    def test_empty_cache_no_match(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(32))
        n, hashes, bids = c.find_longest_prefix(tokens)
        assert n == 0
        assert hashes == []
        assert bids == []

    def test_insert_and_match_full(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(32))
        hashes = compute_prefix_hashes(tokens, 16)
        c.insert(hashes, [10, 11])

        # Same tokens should fully match
        n, m_hashes, m_bids = c.find_longest_prefix(tokens)
        assert n == 32  # all tokens matched
        assert m_hashes == hashes
        assert m_bids == [10, 11]

    def test_insert_and_match_partial(self):
        c = PrefixHashCache(block_size=16)
        # Insert only first block
        tokens_partial = list(range(16))
        h_partial = compute_prefix_hashes(tokens_partial, 16)
        c.insert(h_partial, [10])

        # Query with 32 tokens — only first 16 should match
        tokens_full = list(range(32))
        n, m_hashes, m_bids = c.find_longest_prefix(tokens_full)
        assert n == 16
        assert len(m_hashes) == 1

    def test_hash_chain_integrity(self):
        """Inserting only the first block does NOT match the second block."""
        c = PrefixHashCache(block_size=16)
        # Insert block 0 of [0..31]
        t_half = list(range(16))
        h_half = compute_prefix_hashes(t_half, 16)
        c.insert(h_half, [100])

        # Query with different second block — should match only first
        t_diff = list(range(16)) + [999] * 16
        n, hashes, _ = c.find_longest_prefix(t_diff)
        assert n == 16  # only first block matches

    def test_insert_and_lookup(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(16))
        h = compute_prefix_hashes(tokens, 16)[0]
        c.insert([h], [42])

        entry = c.lookup(h)
        assert entry is not None
        assert entry.block_ids == [42]

    def test_lookup_nonexistent(self):
        c = PrefixHashCache(block_size=16)
        assert c.lookup("deadbeef00000000") is None

    def test_contains(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(16))
        h = compute_prefix_hashes(tokens, 16)[0]
        c.insert([h], [1])
        assert h in c
        assert "nope" not in c

    def test_contains_prefix(self):
        c = PrefixHashCache(block_size=16)
        assert not c.contains_prefix(list(range(32)))

        tokens = list(range(16))
        c.insert(compute_prefix_hashes(tokens, 16), [1])
        assert c.contains_prefix(list(range(32)))

    def test_insert_matching_blocks(self):
        """Insert must have same number of hashes and block IDs."""
        c = PrefixHashCache(block_size=16)
        with pytest.raises(AssertionError):
            c.insert(["a", "b"], [1])

    def test_find_prefix_block_count(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(48))
        hashes = compute_prefix_hashes(tokens, 16)
        c.insert(hashes[:2], [10, 11])  # only first 2 blocks

        count = c.find_prefix_block_count(list(range(64)))
        assert count == 2

    def test_pin_unpin(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(16))
        h = compute_prefix_hashes(tokens, 16)[0]
        c.insert([h], [1])
        c.pin(h)

        entry = c.lookup(h)
        assert entry.is_pinned
        assert c.pinned_count == 1

        c.unpin(h)
        entry = c.lookup(h)
        assert not entry.is_pinned

    def test_pin_group(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(48))
        hashes = compute_prefix_hashes(tokens, 16)
        c.insert(hashes, list(range(3)))
        c.pin_group(hashes)
        assert c.pinned_count == 3

    def test_remove(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(16))
        h = compute_prefix_hashes(tokens, 16)[0]
        c.insert([h], [1])
        assert c.size == 1
        c.remove(h)
        assert c.size == 0
        assert h not in c

    def test_remove_block_id(self):
        c = PrefixHashCache(block_size=16)
        h = compute_prefix_hashes(list(range(16)), 16)[0]
        c.insert([h], [1])   # one hash → one block
        # Manually add more block IDs to the entry (simulating multiple sessions
        # sharing the same prefix block)
        entry = c.lookup(h)
        entry.add_block_id(2)
        entry.add_block_id(3)
        assert entry.block_ids == [1, 2, 3]

        c.remove_block_id(2)
        entry = c.lookup(h)
        assert entry.block_ids == [1, 3]

        # Remove last block IDs → entry removed
        c.remove_block_id(1)
        c.remove_block_id(3)
        assert c.size == 0

    def test_hot_blocks_detection(self):
        c = PrefixHashCache(block_size=16)
        h1 = compute_prefix_hashes(list(range(16)), 16)[0]
        h2 = compute_prefix_hashes(list(range(16, 32)), 16)[0]
        c.insert([h1], [1])
        c.insert([h2], [2])

        # Access h2 10 times
        for _ in range(10):
            c.lookup(h2)

        hot = c.get_hot_blocks(threshold=10)
        assert len(hot) == 1
        assert hot[0].block_hash == h2

    def test_replication_candidates(self):
        c = PrefixHashCache(block_size=16)
        for i in range(5):
            tokens = list(range(i * 16, (i + 1) * 16))
            h = compute_prefix_hashes(tokens, 16)[0]
            c.insert([h], [i])

        # Access block 3 the most
        h3 = compute_prefix_hashes(list(range(48, 64)), 16)[0]
        for _ in range(10):
            c.lookup(h3)

        candidates = c.get_replication_candidates(top_n=2)
        assert len(candidates) == 2
        assert candidates[0].access_count >= 10  # most accessed

    def test_eviction(self):
        c = PrefixHashCache(block_size=16, max_entries=5)
        for i in range(10):
            tokens = list(range(i * 16, (i + 1) * 16))
            h = compute_prefix_hashes(tokens, 16)[0]
            c.insert([h], [i])

        assert c.size <= 5

    def test_pinned_blocks_not_evicted(self):
        c = PrefixHashCache(block_size=16, max_entries=3)
        for i in range(5):
            tokens = list(range(i * 16, (i + 1) * 16))
            h = compute_prefix_hashes(tokens, 16)[0]
            c.insert([h], [i], is_pinned=(i == 0))  # first block pinned

        # Pinned block should survive
        assert compute_prefix_hashes(list(range(16)), 16)[0] in c

    def test_hit_rate_stats(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(32))
        hashes = compute_prefix_hashes(tokens, 16)
        c.insert(hashes, [10, 11])

        # Hit
        c.find_longest_prefix(tokens)
        # Miss
        c.find_longest_prefix(list(range(100, 132)))

        s = c.stats()
        assert s["total_lookups"] == 2
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert 0.45 < s["hit_rate"] < 0.55

    def test_reset_stats(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(16))
        c.insert(compute_prefix_hashes(tokens, 16), [1])
        c.find_longest_prefix(tokens)
        c.reset_stats()
        assert c.stats()["total_lookups"] == 0

    def test_clear(self):
        c = PrefixHashCache(block_size=16)
        tokens = list(range(32))
        c.insert(compute_prefix_hashes(tokens, 16), [1, 2])
        c.clear()
        assert c.size == 0
        assert len(c) == 0

    def test_insert_range(self):
        c = PrefixHashCache(block_size=16)
        hashes = c.insert_range(list(range(48)), [100, 101, 102])
        assert len(hashes) == 3
        assert c.size == 3

    def test_repr(self):
        c = PrefixHashCache(block_size=16, max_entries=100)
        r = repr(c)
        assert "PrefixHashCache" in r
        assert "hit_rate" in r


# ═══════════════════════════════════════════════════════════════════
# AgentPrefixCache
# ═══════════════════════════════════════════════════════════════════

class TestAgentPrefixCache:
    """Agent lifecycle-aware prefix caching."""

    @pytest.fixture
    def agent_cache(self):
        pc = PrefixHashCache(block_size=16, max_entries=1000)
        return AgentPrefixCache(pc)

    def test_cache_system_prompt(self, agent_cache):
        sp_tokens = list(range(80))   # 5 blocks
        agent_cache.cache_system_prompt("default", sp_tokens, [10, 11, 12, 13, 14])

        cached = agent_cache.get_system_prompt("default")
        assert cached is not None
        assert len(cached[0]) == 80
        assert len(cached[1]) == 5  # 5 hashes

        # System prompt should be in the hash cache
        n, _, _ = agent_cache.find_reusable_prefix(sp_tokens)
        assert n == 80

    def test_cache_multiple_system_prompts(self, agent_cache):
        agent_cache.cache_system_prompt("v1", list(range(32)), [1, 2])
        agent_cache.cache_system_prompt("v2", list(range(50, 82)), [3, 4])
        assert set(agent_cache.all_system_prompts()) == {"v1", "v2"}

    def test_replace_system_prompt(self, agent_cache):
        agent_cache.cache_system_prompt("main", list(range(32)), [1, 2])
        # Replace
        agent_cache.cache_system_prompt("main", list(range(48)), [3, 4, 5])

        cached = agent_cache.get_system_prompt("main")
        assert len(cached[0]) == 48  # updated

    def test_cache_tool_schema(self, agent_cache):
        tool_tokens = list(range(32))
        agent_cache.cache_tool_schema("web_search", tool_tokens, [100, 101])

        cached = agent_cache.get_tool_schema("web_search")
        assert cached is not None
        assert len(cached[0]) == 32
        assert "web_search" in agent_cache.all_tool_schemas()

    def test_cache_multiple_tools(self, agent_cache):
        agent_cache.cache_tool_schemas_batch({
            "web_search": (list(range(16)), [1]),
            "read_file": (list(range(16, 32)), [2]),
            "terminal": (list(range(32, 48)), [3]),
        })
        assert len(agent_cache.all_tool_schemas()) == 3

    def test_session_tracking(self, agent_cache):
        agent_cache.track_session("sess-1", list(range(100)))
        assert agent_cache.get_session_prefix("sess-1") == list(range(100))

        agent_cache.extend_session("sess-1", list(range(100, 120)))
        assert len(agent_cache.get_session_prefix("sess-1")) == 120

        agent_cache.end_session("sess-1")
        assert agent_cache.get_session_prefix("sess-1") is None

    def test_find_reusable_prefix(self, agent_cache):
        # Cache some blocks
        tokens = list(range(48))
        agent_cache._cache.insert_range(tokens, [1, 2, 3])

        prefix_tokens, hashes, bids = agent_cache.find_reusable_prefix(
            list(range(64))  # query: 48 cached + 16 new
        )
        assert prefix_tokens == 48
        assert len(hashes) == 3
        assert bids == [1, 2, 3]

    def test_estimate_reuse_ratio(self, agent_cache):
        tokens = list(range(48))
        agent_cache._cache.insert_range(tokens, [1, 2, 3])

        ratio = agent_cache.estimate_reuse_ratio(list(range(64)))
        assert ratio == 48 / 64

    def test_find_session_prefix(self, agent_cache):
        # Cache session history tokens
        history = list(range(80))
        agent_cache._cache.insert_range(history, [10, 11, 12, 13, 14])
        agent_cache.track_session("sess-1", history)

        new_msg = list(range(80, 96))  # 16 new tokens
        n, _, _ = agent_cache.find_session_prefix("sess-1", new_msg)
        assert n == 80  # history is cached

    def test_stats(self, agent_cache):
        agent_cache.cache_system_prompt("default", list(range(32)), [1, 2])
        agent_cache.cache_tool_schema("search", list(range(16)), [3])

        s = agent_cache.stats()
        assert s["system_prompts_cached"] == 1
        assert s["tool_schemas_cached"] == 1
        assert s["active_sessions"] == 0

    def test_reset_stats(self, agent_cache):
        tokens = list(range(16))
        agent_cache._cache.insert_range(tokens, [1])
        agent_cache.find_reusable_prefix(tokens)
        agent_cache.reset_stats()
        assert agent_cache.stats()["prefix_hits"] == 0

    def test_clear(self, agent_cache):
        agent_cache.cache_system_prompt("default", list(range(16)), [1])
        agent_cache.cache_tool_schema("search", list(range(16)), [2])
        agent_cache.track_session("sess-1", list(range(16)))
        agent_cache.clear()
        assert len(agent_cache.all_system_prompts()) == 0
        assert len(agent_cache.all_tool_schemas()) == 0
        assert agent_cache.get_session_prefix("sess-1") is None

    def test_repr(self, agent_cache):
        r = repr(agent_cache)
        assert "AgentPrefixCache" in r


# ═══════════════════════════════════════════════════════════════════
# Savings estimator
# ═══════════════════════════════════════════════════════════════════

class TestSavingsEstimator:
    def test_estimate_basic(self):
        pc = PrefixHashCache(block_size=16)
        ac = AgentPrefixCache(pc)

        ac.cache_system_prompt("default", list(range(80)), list(range(5)))
        ac.cache_tool_schema("search", list(range(32)), [100, 101])

        est = estimate_agent_savings(ac, history_tokens=200, num_sessions=4)
        assert est["system_prompt_tokens"] == 80
        assert est["tool_schema_tokens"] == 32
        assert est["reusable_per_turn"] == 112  # 80 + 32
        assert est["savings_per_turn_pct"] > 30
        assert est["tokens_saved_across_sessions"] == 112 * 4

    def test_empty_cache(self):
        pc = PrefixHashCache(block_size=16)
        ac = AgentPrefixCache(pc)
        est = estimate_agent_savings(ac, history_tokens=100, num_sessions=1)
        assert est["reusable_per_turn"] == 0
        assert est["savings_per_turn_pct"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Integration: BlockTableManager + PrefixHashCache
# ═══════════════════════════════════════════════════════════════════

class TestPrefixCacheIntegration:
    """End-to-end: hash cache → block table sharing."""

    def test_prefix_match_to_block_sharing(self, allocator, config):
        """Hash-cache hit → share blocks via BlockTableManager."""
        from memory_manager.block_table import BlockTableManager

        mgr = BlockTableManager(allocator, config.block_size)
        pc = PrefixHashCache(block_size=config.block_size)

        # Source: allocate blocks for a system prompt
        sys_tokens = list(range(80))  # 5 blocks
        sys_ids = allocator.allocate("sys", num_tokens=80)
        sys_table = mgr.create_table("sys")
        sys_table.append_blocks(sys_ids, tokens_per_block=[16]*5)

        # Cache in prefix cache
        pc.insert_range(sys_tokens, sys_ids, is_pinned=True)

        # Target: new request with same system prompt prefix
        tgt_tokens = list(range(80)) + list(range(1000, 1050))
        prefix_tokens, hashes, matched_ids = pc.find_longest_prefix(tgt_tokens)

        assert prefix_tokens == 80
        assert matched_ids == sys_ids

        # Share the prefix via block table manager
        user_ids = allocator.allocate("user", num_tokens=50)
        mgr.create_table("user")
        mgr.share_prefix("sys", "user", prefix_blocks=len(sys_ids))

        # Verify: user's physical blocks for the prefix match sys's
        user_blocks = mgr.get_physical_blocks("user")[:len(sys_ids)]
        assert user_blocks == sys_ids
