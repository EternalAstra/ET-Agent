"""
Agent Lifecycle Tracker — phase-aware KV Cache tiering.

Tracks the lifecycle phase of each agent request and triggers automatic
KV Cache migrations between storage tiers (GPU ↔ CPU ↔ SSD) based on
heuristics tuned for agent workloads:

* **PREFILL** — arriving or new user input; keep on GPU
* **DECODING** — actively generating tokens; keep on GPU
* **TOOL_CALL** — waiting for tool results; demote to CPU after *N* seconds
* **IDLE** — session paused; demote to CPU then SSD after *M* seconds
* **COMPLETED** — session finished; archive to SSD, evict after *K* seconds

This is the agent analogue of paging/swapping in vLLM (§4.5), extended
with phase-driven policy. MoonCake (§5.2) demonstrates that layer-wise
loading can overlap compute with transfer; our lifecycle tracker feeds
the ``HierarchicalKVStore`` with migration commands so that data is
prefetchable before the next phase transition.

Lifecycle transitions
---------------------
::

    PREFILL → DECODING → [TOOL_CALL → PREFILL] × N → COMPLETED
                  │                           │
                  └─── IDLE ──────────────────┘

Each transition invokes ``on_phase_change()`` which schedules timed
migration callbacks via the ``HierarchicalKVStore``.

References
----------
- vLLM §4.5     (scheduling and preemption, swap-out/swap-in)
- MoonCake §5.2 (layer-wise prefill: overlap transfer with compute)
- MoonCake §7   (early rejection based on prediction, load fluctuation)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Agent lifecycle phases
# ---------------------------------------------------------------------------

class AgentPhase(Enum):
    """Phases in an agent request lifecycle."""
    PREFILL = auto()        # Processing new user input (prefill stage)
    DECODING = auto()       # Generating output tokens (decoding stage)
    TOOL_CALL = auto()      # Waiting for tool execution to complete
    IDLE = auto()           # Session is idle (user hasn't responded)
    COMPLETED = auto()      # Session finished / archived


# ── Phase properties ──

_PHASE_PROPERTIES = {
    AgentPhase.PREFILL: {
        "gpu_required": True,
        "latency_sensitive": True,
        "description": "Prefill — tokenising user input, computing initial KV",
    },
    AgentPhase.DECODING: {
        "gpu_required": True,
        "latency_sensitive": True,
        "description": "Decoding — autoregressive token generation",
    },
    AgentPhase.TOOL_CALL: {
        "gpu_required": False,
        "latency_sensitive": False,
        "description": "Tool call — waiting for external tool result",
    },
    AgentPhase.IDLE: {
        "gpu_required": False,
        "latency_sensitive": False,
        "description": "Idle — session paused, user away",
    },
    AgentPhase.COMPLETED: {
        "gpu_required": False,
        "latency_sensitive": False,
        "description": "Completed — session finished, archive candidate",
    },
}


def phase_requires_gpu(phase: AgentPhase) -> bool:
    return _PHASE_PROPERTIES[phase]["gpu_required"]

def phase_latency_sensitive(phase: AgentPhase) -> bool:
    return _PHASE_PROPERTIES[phase]["latency_sensitive"]


# ---------------------------------------------------------------------------
# Timing configuration
# ---------------------------------------------------------------------------

@dataclass
class LifecycleTiming:
    """Tier migration thresholds for each phase transition.

    Times are in seconds.  -1 means "never migrate".

    Defaults are tuned for a typical agent workload where tool calls
    take 5–60s, user think-time is 30–300s, and sessions persist for
    minutes to hours.
    """

    # Tool call: GPU → CPU after this many seconds idle
    tool_call_gpu_to_cpu_s: float = 30.0
    # Tool call: CPU → SSD after this many seconds idle
    tool_call_cpu_to_ssd_s: float = 300.0   # 5 minutes

    # Idle session: GPU → CPU
    idle_gpu_to_cpu_s: float = 60.0
    # Idle session: CPU → SSD
    idle_cpu_to_ssd_s: float = 600.0  # 10 minutes

    # Completed session: keep on SSD for this long, then evict
    completed_retain_s: float = 86_400.0   # 24 hours

    # Prefetch: how many seconds before expected resume to start loading
    prefetch_ahead_s: float = 5.0

    # Maximum fraction of GPU blocks that can be occupied by non-active sessions
    max_inactive_gpu_ratio: float = 0.3

    def tier_for_phase(self, phase: AgentPhase,
                       idle_seconds: float) -> "StorageTier":
        """Determine the target storage tier for a request in *phase*.

        Parameters
        ----------
        phase : AgentPhase
            Current lifecycle phase.
        idle_seconds : float
            How long the request has been in this phase.

        Returns
        -------
        StorageTier
            Recommended storage tier.
        """
        from memory_manager.kv_block import StorageTier

        if phase in (AgentPhase.PREFILL, AgentPhase.DECODING):
            return StorageTier.GPU

        if phase == AgentPhase.TOOL_CALL:
            if self.tool_call_cpu_to_ssd_s >= 0 and idle_seconds >= self.tool_call_cpu_to_ssd_s:
                return StorageTier.SSD
            if self.tool_call_gpu_to_cpu_s >= 0 and idle_seconds >= self.tool_call_gpu_to_cpu_s:
                return StorageTier.CPU
            return StorageTier.GPU

        if phase == AgentPhase.IDLE:
            if self.idle_cpu_to_ssd_s >= 0 and idle_seconds >= self.idle_cpu_to_ssd_s:
                return StorageTier.SSD
            if self.idle_gpu_to_cpu_s >= 0 and idle_seconds >= self.idle_gpu_to_cpu_s:
                return StorageTier.CPU
            return StorageTier.GPU

        if phase == AgentPhase.COMPLETED:
            if self.completed_retain_s >= 0 and idle_seconds >= self.completed_retain_s:
                # Eviction candidate — caller should free blocks
                return StorageTier.SSD
            return StorageTier.SSD  # archive immediately

        return StorageTier.GPU


# ---------------------------------------------------------------------------
# Request lifecycle record
# ---------------------------------------------------------------------------

@dataclass
class RequestLifecycle:
    """Tracks one agent request through its lifecycle phases.

    Parameters
    ----------
    request_id : str
        Unique request identifier (e.g. session_id + turn_id).
    session_id : str
        Owning session identifier (for group eviction).
    phase : AgentPhase
        Current lifecycle phase.
    created_at : float
        Monotonic timestamp when the request was created.
    phase_entered_at : float
        Monotonic timestamp when the current phase was entered.
    last_active_at : float
        Monotonic timestamp of the last token generation.
    total_tokens : int
        Cumulative tokens in this request's KV Cache.
    num_blocks : int
        Number of KV Cache blocks owned.
    tool_call_count : int
        How many tool calls have been made in this request.
    turn_count : int
        How many conversation turns this request spans.

    Branch tracking (for subagent delegation / parallel tool calls)
    ----------------------------------------------------------------
    parent_request_id : str | None
        If this request was forked from another, the parent's ID.
    branch_point : int | None
        Token index at which the fork occurred.
    """

    request_id: str
    session_id: str
    phase: AgentPhase = AgentPhase.PREFILL
    created_at: float = field(default_factory=time.monotonic)
    phase_entered_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    total_tokens: int = 0
    num_blocks: int = 0
    tool_call_count: int = 0
    turn_count: int = 0
    parent_request_id: Optional[str] = None
    branch_point: Optional[int] = None

    # ── computed properties ──

    @property
    def idle_seconds(self) -> float:
        """How long the request has been in its current phase."""
        return time.monotonic() - self.phase_entered_at

    @property
    def age_seconds(self) -> float:
        """Total age of the request."""
        return time.monotonic() - self.created_at

    @property
    def is_active(self) -> bool:
        """True if the request is actively computing (prefill or decoding)."""
        return self.phase in (AgentPhase.PREFILL, AgentPhase.DECODING)

    @property
    def is_waiting(self) -> bool:
        """True if the request is blocked on an external event."""
        return self.phase in (AgentPhase.TOOL_CALL,)

    # ── mutation ──

    def transition_to(self, new_phase: AgentPhase):
        """Move to a new lifecycle phase, recording the transition time."""
        now = time.monotonic()
        self.phase = new_phase
        self.phase_entered_at = now
        self.last_active_at = now

    def record_activity(self):
        """Mark the request as recently active."""
        self.last_active_at = time.monotonic()

    def record_tool_call(self):
        self.tool_call_count += 1

    def record_turn(self):
        self.turn_count += 1

    def __repr__(self) -> str:
        return (
            f"RequestLifecycle({self.request_id}, "
            f"phase={self.phase.name}, "
            f"idle={self.idle_seconds:.0f}s, "
            f"blocks={self.num_blocks})"
        )


# ---------------------------------------------------------------------------
# Lifecycle-aware KV manager
# ---------------------------------------------------------------------------

class LifecycleAwareKVManager:
    """Central tracker: maps requests to lifecycle phases and schedules
    tier migrations based on idle time.

    This is the agent-specific layer above vLLM's generic paging.  It
    watches phase transitions and issues migration commands to the
    ``HierarchicalKVStore`` (Phase 3) or directly to the ``KVBlockAllocator``.

    Parameters
    ----------
    timing : LifecycleTiming
        Migration thresholds.
    on_demote : callable(request_id, block_ids, target_tier) | None
        Callback invoked when blocks should be demoted from GPU.
    on_promote : callable(request_id, block_ids) | None
        Callback invoked when blocks should be promoted back to GPU.
    on_evict : callable(request_id, block_ids) | None
        Callback invoked when blocks should be freed entirely.
    """

    def __init__(
        self,
        timing: LifecycleTiming | None = None,
        on_demote: Callable | None = None,
        on_promote: Callable | None = None,
        on_evict: Callable | None = None,
    ):
        self._timing = timing or LifecycleTiming()
        self._on_demote = on_demote
        self._on_promote = on_promote
        self._on_evict = on_evict
        self._lock = threading.RLock()
        self._lifecycles: Dict[str, RequestLifecycle] = {}
        self._block_owners: Dict[int, str] = {}  # block_id → request_id

        # Protected sessions (mid-turn; never demote)
        self._protected_sessions: Set[str] = set()

        # Stats
        self._total_transitions: int = 0
        self._total_demotions: int = 0
        self._total_promotions: int = 0
        self._total_evictions: int = 0

    # ------------------------------------------------------------------
    # Registration & lifecycle tracking
    # ------------------------------------------------------------------

    def register(self, request_id: str, session_id: str,
                 num_tokens: int = 0, num_blocks: int = 0,
                 phase: AgentPhase = AgentPhase.PREFILL,
                 parent_request_id: str | None = None,
                 branch_point: int | None = None) -> RequestLifecycle:
        """Register a new agent request."""
        with self._lock:
            lc = RequestLifecycle(
                request_id=request_id,
                session_id=session_id,
                phase=phase,
                total_tokens=num_tokens,
                num_blocks=num_blocks,
                parent_request_id=parent_request_id,
                branch_point=branch_point,
            )
            self._lifecycles[request_id] = lc
            return lc

    def unregister(self, request_id: str):
        """Remove a completed request from tracking."""
        with self._lock:
            self._lifecycles.pop(request_id, None)

    def get(self, request_id: str) -> Optional[RequestLifecycle]:
        return self._lifecycles.get(request_id)

    def get_session_requests(self, session_id: str) -> List[RequestLifecycle]:
        """Return all requests belonging to a session."""
        with self._lock:
            return [
                lc for lc in self._lifecycles.values()
                if lc.session_id == session_id
            ]

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def on_phase_change(self, request_id: str, new_phase: AgentPhase):
        """Handle a lifecycle phase transition for *request_id*.

        Automatically determines the target storage tier and, if the
        tier differs from the current one, invokes the ``on_demote``
        or ``on_promote`` callbacks.
        """
        with self._lock:
            lc = self._lifecycles.get(request_id)
            if lc is None:
                return

            old_phase = lc.phase
            idle_before = lc.idle_seconds
            lc.transition_to(new_phase)
            self._total_transitions += 1

            # Determine target tier
            target_tier = self._timing.tier_for_phase(new_phase, 0)

            # GPU-required phases: promote if needed
            if phase_requires_gpu(new_phase):
                self._maybe_promote(request_id)

            # Non-GPU phases: schedule eventual demotion
            elif not phase_requires_gpu(new_phase) and phase_requires_gpu(old_phase):
                # Just left a GPU phase → schedule demotion
                self._schedule_demotion(request_id)

            return

    def _maybe_promote(self, request_id: str):
        """Promote blocks back to GPU if they were demoted."""
        if not self._on_promote:
            return
        lc = self._lifecycles.get(request_id)
        if lc is None or lc.num_blocks == 0:
            return
        self._total_promotions += 1
        # The callback is responsible for issuing the actual tier migration
        self._on_promote(request_id, request_id)

    def _schedule_demotion(self, request_id: str):
        """Defer demotion to a background timer (or invoke immediately)."""
        # In practice this would use threading.Timer or a scheduler.
        # For Phase 3 we invoke the demotion synchronously after the
        # configured idle threshold.
        if not self._on_demote:
            return

        # Simple: fire-and-forget timer
        delay = self._timing.tool_call_gpu_to_cpu_s
        if delay <= 0:
            return

        t = threading.Timer(delay, self._demote_step, args=(request_id,))
        t.daemon = True
        t.start()

    def _demote_step(self, request_id: str):
        """One step of the demotion cascade (GPU→CPU or CPU→SSD)."""
        with self._lock:
            lc = self._lifecycles.get(request_id)
            if lc is None:
                return

            # If the request became active again, skip
            if lc.is_active:
                return

            target_tier = self._timing.tier_for_phase(lc.phase, lc.idle_seconds)

        if target_tier.value not in ("cpu", "ssd"):
            return

        if self._on_demote:
            self._total_demotions += 1
            self._on_demote(request_id, request_id, target_tier)

    # ------------------------------------------------------------------
    # Session protection
    # ------------------------------------------------------------------

    def protect_session(self, session_id: str):
        """Mark a session as protected (never demote its blocks)."""
        with self._lock:
            self._protected_sessions.add(session_id)

    def unprotect_session(self, session_id: str):
        """Remove protection from a session."""
        with self._lock:
            self._protected_sessions.discard(session_id)

    def is_protected(self, request_id: str) -> bool:
        lc = self._lifecycles.get(request_id)
        return lc is not None and lc.session_id in self._protected_sessions

    # ------------------------------------------------------------------
    # Block ownership
    # ------------------------------------------------------------------

    def link_blocks(self, request_id: str, block_ids: List[int]):
        """Record that *block_ids* belong to *request_id*."""
        with self._lock:
            lc = self._lifecycles.get(request_id)
            if lc is not None:
                lc.num_blocks = len(block_ids)
            for bid in block_ids:
                self._block_owners[bid] = request_id

    def unlink_blocks(self, request_id: str, block_ids: List[int]):
        """Remove block-ownership links."""
        with self._lock:
            for bid in block_ids:
                if self._block_owners.get(bid) == request_id:
                    del self._block_owners[bid]
            lc = self._lifecycles.get(request_id)
            if lc is not None:
                lc.num_blocks = max(0, lc.num_blocks - len(block_ids))

    def get_block_owner(self, block_id: int) -> Optional[str]:
        return self._block_owners.get(block_id)

    # ------------------------------------------------------------------
    # Periodic scan (call from a background thread / tick)
    # ------------------------------------------------------------------

    def scan_and_migrate(self):
        """Scan all tracked requests and trigger migrations as needed.

        Should be called periodically (e.g. every 30s) from a background
        thread or the agent's main loop idle hook.
        """
        now = time.monotonic()
        with self._lock:
            for request_id, lc in list(self._lifecycles.items()):
                if lc.is_active or self.is_protected(request_id):
                    continue

                idle = now - lc.phase_entered_at
                target = self._timing.tier_for_phase(lc.phase, idle)

                if target == StorageTier.CPU and self._on_demote:
                    self._on_demote(request_id, request_id, target)
                elif target == StorageTier.SSD and self._on_evict:
                    self._total_evictions += 1
                    self._on_evict(request_id, request_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            phases = {}
            for lc in self._lifecycles.values():
                name = lc.phase.name
                phases[name] = phases.get(name, 0) + 1

            return {
                "total_requests": len(self._lifecycles),
                "active_requests": sum(1 for lc in self._lifecycles.values() if lc.is_active),
                "waiting_requests": sum(1 for lc in self._lifecycles.values() if lc.is_waiting),
                "protected_sessions": len(self._protected_sessions),
                "phases": phases,
                "total_transitions": self._total_transitions,
                "total_demotions": self._total_demotions,
                "total_promotions": self._total_promotions,
                "total_evictions": self._total_evictions,
            }

    def reset_stats(self):
        with self._lock:
            self._total_transitions = 0
            self._total_demotions = 0
            self._total_promotions = 0
            self._total_evictions = 0

    def dump(self) -> str:
        """Human-readable lifecycle dump."""
        with self._lock:
            lines = [f"LifecycleAwareKVManager ({len(self._lifecycles)} requests)"]
            for lc in sorted(self._lifecycles.values(),
                             key=lambda x: x.created_at):
                protected = "🔒" if lc.session_id in self._protected_sessions else ""
                lines.append(
                    f"  {lc.request_id:20s} {lc.phase.name:12s} "
                    f"idle={lc.idle_seconds:7.1f}s  "
                    f"blocks={lc.num_blocks:4d}  turns={lc.turn_count:3d} {protected}"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Re-export StorageTier for convenience
from memory_manager.kv_block import StorageTier  # noqa: E402
