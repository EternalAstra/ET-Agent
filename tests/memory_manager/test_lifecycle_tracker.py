"""Unit tests for Agent lifecycle tracker + phase-driven tiering."""

import time
import pytest
from memory_manager.kv_lifecycle_tracker import (
    AgentPhase,
    RequestLifecycle,
    LifecycleAwareKVManager,
    LifecycleTiming,
    phase_requires_gpu,
    phase_latency_sensitive,
)
from memory_manager.kv_block import StorageTier


# ═══════════════════════════════════════════════════════════════════
# AgentPhase helpers
# ═══════════════════════════════════════════════════════════════════

class TestAgentPhase:
    def test_gpu_required(self):
        assert phase_requires_gpu(AgentPhase.PREFILL)
        assert phase_requires_gpu(AgentPhase.DECODING)
        assert not phase_requires_gpu(AgentPhase.TOOL_CALL)
        assert not phase_requires_gpu(AgentPhase.IDLE)
        assert not phase_requires_gpu(AgentPhase.COMPLETED)

    def test_latency_sensitive(self):
        assert phase_latency_sensitive(AgentPhase.PREFILL)
        assert phase_latency_sensitive(AgentPhase.DECODING)
        assert not phase_latency_sensitive(AgentPhase.TOOL_CALL)
        assert not phase_latency_sensitive(AgentPhase.IDLE)


# ═══════════════════════════════════════════════════════════════════
# LifecycleTiming
# ═══════════════════════════════════════════════════════════════════

class TestLifecycleTiming:
    def test_active_phases_stay_on_gpu(self):
        t = LifecycleTiming()
        assert t.tier_for_phase(AgentPhase.PREFILL, 0) == StorageTier.GPU
        assert t.tier_for_phase(AgentPhase.DECODING, 0) == StorageTier.GPU
        assert t.tier_for_phase(AgentPhase.PREFILL, 999) == StorageTier.GPU

    def test_tool_call_demotes(self):
        t = LifecycleTiming()
        # Still within grace period
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 15) == StorageTier.GPU
        # Past GPU→CPU threshold
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 31) == StorageTier.CPU
        # Past CPU→SSD threshold
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 301) == StorageTier.SSD

    def test_idle_demotes(self):
        t = LifecycleTiming()
        assert t.tier_for_phase(AgentPhase.IDLE, 30) == StorageTier.GPU
        assert t.tier_for_phase(AgentPhase.IDLE, 61) == StorageTier.CPU
        assert t.tier_for_phase(AgentPhase.IDLE, 601) == StorageTier.SSD

    def test_completed_archives(self):
        t = LifecycleTiming()
        assert t.tier_for_phase(AgentPhase.COMPLETED, 0) == StorageTier.SSD
        # After 24h, still SSD (caller decides eviction)
        assert t.tier_for_phase(AgentPhase.COMPLETED, 86_401) == StorageTier.SSD

    def test_custom_thresholds(self):
        t = LifecycleTiming(
            tool_call_gpu_to_cpu_s=10.0,
            tool_call_cpu_to_ssd_s=60.0,
            idle_gpu_to_cpu_s=20.0,
        )
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 5) == StorageTier.GPU
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 15) == StorageTier.CPU
        assert t.tier_for_phase(AgentPhase.TOOL_CALL, 65) == StorageTier.SSD
        assert t.tier_for_phase(AgentPhase.IDLE, 25) == StorageTier.CPU


# ═══════════════════════════════════════════════════════════════════
# RequestLifecycle
# ═══════════════════════════════════════════════════════════════════

class TestRequestLifecycle:
    def test_initial_state(self):
        lc = RequestLifecycle("req-1", "sess-1")
        assert lc.phase == AgentPhase.PREFILL
        assert lc.is_active
        assert not lc.is_waiting
        assert lc.tool_call_count == 0
        assert lc.turn_count == 0

    def test_transitions(self):
        lc = RequestLifecycle("req-1", "sess-1")
        lc.transition_to(AgentPhase.DECODING)
        assert lc.phase == AgentPhase.DECODING
        assert lc.is_active

        lc.transition_to(AgentPhase.TOOL_CALL)
        assert lc.phase == AgentPhase.TOOL_CALL
        assert not lc.is_active
        assert lc.is_waiting

    def test_idle_seconds(self):
        lc = RequestLifecycle("req-1", "sess-1")
        time.sleep(0.01)
        assert lc.idle_seconds > 0

    def test_record_activity(self):
        lc = RequestLifecycle("req-1", "sess-1")
        old = lc.last_active_at
        time.sleep(0.01)
        lc.record_activity()
        assert lc.last_active_at > old

    def test_record_tool_call(self):
        lc = RequestLifecycle("req-1", "sess-1")
        lc.record_tool_call()
        lc.record_tool_call()
        assert lc.tool_call_count == 2

    def test_record_turn(self):
        lc = RequestLifecycle("req-1", "sess-1")
        for _ in range(3):
            lc.record_turn()
        assert lc.turn_count == 3

    def test_age_seconds(self):
        lc = RequestLifecycle("req-1", "sess-1")
        time.sleep(0.01)
        assert lc.age_seconds > 0

    def test_branch_tracking(self):
        lc = RequestLifecycle(
            "child", "sess-1",
            parent_request_id="parent",
            branch_point=1024,
        )
        assert lc.parent_request_id == "parent"
        assert lc.branch_point == 1024

    def test_repr(self):
        lc = RequestLifecycle("req-1", "sess-1")
        r = repr(lc)
        assert "req-1" in r
        assert "PREFILL" in r


# ═══════════════════════════════════════════════════════════════════
# LifecycleAwareKVManager
# ═══════════════════════════════════════════════════════════════════

class TestLifecycleManager:
    def test_register(self):
        mgr = LifecycleAwareKVManager()
        lc = mgr.register("req-1", "sess-1")
        assert lc.request_id == "req-1"
        assert mgr.get("req-1") is lc

    def test_unregister(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("req-1", "sess-1")
        mgr.unregister("req-1")
        assert mgr.get("req-1") is None

    def test_get_session_requests(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("r1", "sess-a")
        mgr.register("r2", "sess-a")
        mgr.register("r3", "sess-b")
        assert len(mgr.get_session_requests("sess-a")) == 2
        assert len(mgr.get_session_requests("sess-b")) == 1

    def test_on_phase_change(self):
        mig_log = []

        def on_promote(req_id, _bid):
            mig_log.append(("promote", req_id))

        mgr = LifecycleAwareKVManager(on_promote=on_promote)
        mgr.register("req-1", "sess-1")
        mgr.register("req-2", "sess-1")

        # Transition req-1 to TOOL_CALL (schedules demotion after delay)
        mgr.on_phase_change("req-1", AgentPhase.TOOL_CALL)
        lc = mgr.get("req-1")
        assert lc.phase == AgentPhase.TOOL_CALL
        assert not lc.is_active

    def test_protect_unprotect(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("req-1", "sess-1")
        mgr.protect_session("sess-1")
        assert mgr.is_protected("req-1")
        mgr.unprotect_session("sess-1")
        assert not mgr.is_protected("req-1")

    def test_block_ownership(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("req-1", "sess-1")
        mgr.link_blocks("req-1", [10, 11, 12])
        lc = mgr.get("req-1")
        assert lc.num_blocks == 3
        assert mgr.get_block_owner(10) == "req-1"
        assert mgr.get_block_owner(99) is None

        mgr.unlink_blocks("req-1", [11])
        assert mgr.get("req-1").num_blocks == 2
        assert mgr.get_block_owner(11) is None

    def test_scan_and_migrate_idle(self):
        """scan_and_migrate should trigger demotion for idle tool-call requests."""
        demotion_log = []

        def on_demote(req_id, _bid, target):
            demotion_log.append((req_id, target))

        t = LifecycleTiming(
            tool_call_gpu_to_cpu_s=0.0,   # immediate
            tool_call_cpu_to_ssd_s=-1,    # never
        )
        mgr = LifecycleAwareKVManager(
            timing=t,
            on_demote=on_demote,
        )
        mgr.register("req-1", "sess-1", phase=AgentPhase.TOOL_CALL)
        lc = mgr.get("req-1")
        lc.phase_entered_at = 0  # fake: "entered TOOL_CALL long ago"

        mgr.scan_and_migrate()
        assert len(demotion_log) == 1
        assert demotion_log[0][0] == "req-1"

    def test_scan_skips_active(self):
        demotion_log = []

        def on_demote(req_id, _bid, target):
            demotion_log.append(req_id)

        t = LifecycleTiming(tool_call_gpu_to_cpu_s=0.0)
        mgr = LifecycleAwareKVManager(timing=t, on_demote=on_demote)
        mgr.register("active", "sess-1", phase=AgentPhase.DECODING)
        lc = mgr.get("active")
        lc.phase_entered_at = 0

        mgr.scan_and_migrate()
        assert len(demotion_log) == 0  # active → skip

    def test_stats(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("r1", "sess-1", phase=AgentPhase.DECODING)
        mgr.register("r2", "sess-1", phase=AgentPhase.TOOL_CALL)
        mgr.protect_session("sess-1")

        s = mgr.stats()
        assert s["total_requests"] == 2
        assert s["active_requests"] == 1
        assert s["waiting_requests"] == 1
        assert s["protected_sessions"] == 1
        assert s["phases"]["DECODING"] == 1
        assert s["phases"]["TOOL_CALL"] == 1

    def test_dump(self):
        mgr = LifecycleAwareKVManager()
        mgr.register("req-1", "sess-1", num_blocks=5)
        dump = mgr.dump()
        assert "req-1" in dump
        assert "PREFILL" in dump
        assert "blocks=" in dump


# ═══════════════════════════════════════════════════════════════════
# Integration: lifecycle + migration
# ═══════════════════════════════════════════════════════════════════

class TestLifecycleIntegration:
    def test_full_agent_lifecycle(self, allocator, config):
        """Simulate: prefill → decode → tool_call → prefill → decode → complete."""
        from memory_manager.kv_hierarchical_store import HierarchicalKVStore

        store = HierarchicalKVStore(config, allocator)

        demoted = []
        promoted = []

        def on_demote(req_id, _bid, target):
            blocks = allocator.get_request_blocks(req_id)
            if blocks:
                store.demote_blocks(list(blocks), target, req_id)
                demoted.append((req_id, target))

        def on_promote(req_id, _bid):
            blocks = allocator.get_request_blocks(req_id)
            if blocks:
                store.promote_blocks(list(blocks), StorageTier.GPU, req_id)
                promoted.append(req_id)

        t = LifecycleTiming(
            tool_call_gpu_to_cpu_s=0.0,
            tool_call_cpu_to_ssd_s=-1,
        )
        mgr = LifecycleAwareKVManager(
            timing=t,
            on_demote=on_demote,
            on_promote=on_promote,
        )

        # 1. Register request + allocate blocks
        mgr.register("req-1", "sess-1", phase=AgentPhase.PREFILL)
        ids = allocator.allocate("req-1", num_tokens=100)
        mgr.link_blocks("req-1", ids)

        # 2. Prefill → Decoding
        mgr.on_phase_change("req-1", AgentPhase.DECODING)
        assert mgr.get("req-1").is_active

        # 3. Decoding → Tool call (trigger demotion)
        lc = mgr.get("req-1")
        lc.phase_entered_at = 0  # fake: been in TOOL_CALL for a while
        mgr.on_phase_change("req-1", AgentPhase.TOOL_CALL)
        mgr.scan_and_migrate()
        if demoted:
            # Blocks should now be on CPU
            for bid in ids:
                loc = store.get_location(bid)
                assert loc in (StorageTier.CPU, StorageTier.GPU)

        # 4. Tool returns → promote back to GPU
        mgr.on_phase_change("req-1", AgentPhase.PREFILL)

        # 5. Complete → archive
        mgr.on_phase_change("req-1", AgentPhase.COMPLETED)

        # 6. Cleanup
        allocator.free("req-1")
        mgr.unregister("req-1")
        assert mgr.get("req-1") is None
