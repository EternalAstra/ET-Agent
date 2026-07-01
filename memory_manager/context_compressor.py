"""
ACON-style Context Compressor — structured summarization for agent trajectories.

Implements the two-tier compression strategy from ACON (Kang et al., ICML 2026):

* **History Compression** (§3.2, Eq.3) — when the accumulated interaction history
  exceeds a threshold, compress it into a structured summary with REASONING,
  VARS, and ACTIONS sections.
* **Observation Compression** (§3.2, Eq.4) — per-step observation elision that
  retains only task-relevant details, stripping boilerplate and noise.

Key design decisions from ACON
-------------------------------
1. **Selective compression** — compressor only fires when context exceeds a
   predefined token threshold (T_hist, T_obs).  This avoids unnecessary
   overhead for short contexts.
2. **Structured output format** — compressed representation follows a strict
   template with named sections so the agent can reliably extract information.
3. **Natural-language guidelines** — compression is driven by prompts, not
   fine-tuned weights, making it compatible with any LLM (including
   proprietary API-based models).
4. **Two-mode operation**
   - *Utility mode* (UT) — prioritize accuracy, preserve all critical state
   - *Compression mode* (CO) — aggressive reduction for cost/latency

Integration point
-----------------
Called from ``agent/memory_hooks.py`` (Phase 5) when the token estimate
exceeds configurable thresholds.  Works alongside the existing hermes
``context_engine`` plugin.

References
----------
- Kang et al., "ACON: Optimizing Context Compression for Long-horizon
  LLM Agents", ICML 2026.
- ACON §3.2 (history & observation compression)
- ACON §3.3 (compression guideline optimization)
- ACON Table 1 (26–54% peak token reduction)
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Compression mode
# ---------------------------------------------------------------------------

class CompressionMode(Enum):
    """Compression aggressiveness (ACON §4.2)."""
    UT = auto()    # Utility Maximization — preserve accuracy
    CO = auto()    # Compression Maximization — aggressive reduction
    UTCO = auto()  # UT followed by CO (two-pass)


# ---------------------------------------------------------------------------
# Compression templates — ACON-style structured output
# ---------------------------------------------------------------------------

# Template for history compression (ACON Fig.3 shows the optimized guideline)
HISTORY_COMPRESSION_TEMPLATE = """You maintain a compact, state-preserving HISTORY_SUMMARY for a multi-step AI agent.

<HISTORY_SUMMARY>
1. REASONING
  • Key progress, decisions, outcomes, and their rationale.
  • Note how earlier steps influence later ones.

2. VARS
  | name | value | purpose |
  |------|-------|---------|
  Record every runtime value the agent must remember for subsequent steps.

3. ACTIONS_EXECUTED
  • List actions taken so far with their results (success/failure).
  • Omit action details that are no longer relevant.

4. OPEN_TASKS
  • Remaining tasks or unresolved questions.
  • Prioritize: what must be done next.
</HISTORY_SUMMARY>"""

# Template for observation compression
OBSERVATION_COMPRESSION_TEMPLATE = """You compress tool outputs and environment observations for an AI agent.
Preserve ONLY information that is:
- Task-relevant (directly answers the agent's current goal)
- Numeric or stateful (values, IDs, timestamps, status codes)
- Referenced by later steps (file paths, URLs, identifiers)

Strip:
- Boilerplate text, repeated headers, formatting cruft
- Stack traces beyond the first and last 3 lines
- Content already present in the history summary
- Verbose success/error messages (keep only the essential signal)

Output format:
<OBSERVATION>
[compressed observation — keep under 200 words]
</OBSERVATION>"""

# Template for the "naive prompting" baseline (used as initial guideline)
NAIVE_COMPRESSION_TEMPLATE = """Summarize the following interaction history concisely.
Focus on the key actions taken, their outcomes, and any important state changes.
Omit irrelevant details and redundant information."""


# ---------------------------------------------------------------------------
# Compression trigger configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionThresholds:
    """Token-count thresholds that trigger compression (ACON Fig.6).

    Compression only fires when context exceeds these limits, avoiding
    unnecessary overhead for short conversations.
    """
    # History compression: trigger when total history tokens > this
    history_token_threshold: int = 4096   # ACON default: 4096
    # Observation compression: trigger when a single observation > this
    observation_token_threshold: int = 1024  # ACON default: 1024
    # Maximum compressed history tokens (target after compression)
    history_compressed_max_tokens: int = 2048
    # Maximum compressed observation tokens
    observation_compressed_max_tokens: int = 512
    # Minimum turns before first compression (protect recent context)
    protect_last_n_turns: int = 3

    @staticmethod
    def for_agent_scenario() -> "CompressionThresholds":
        """Defaults tuned for ET-Agent tool-calling scenarios."""
        return CompressionThresholds(
            history_token_threshold=4096,
            observation_token_threshold=1024,
            history_compressed_max_tokens=2048,
            observation_compressed_max_tokens=512,
            protect_last_n_turns=3,
        )

    @staticmethod
    def aggressive() -> "CompressionThresholds":
        """Aggressive settings for very long contexts (128K+)."""
        return CompressionThresholds(
            history_token_threshold=2048,
            observation_token_threshold=512,
            history_compressed_max_tokens=1024,
            observation_compressed_max_tokens=256,
            protect_last_n_turns=1,
        )


# ---------------------------------------------------------------------------
# Compression result
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of a single compression pass."""
    compressed_text: str
    original_tokens: int         # Estimated token count before compression
    compressed_tokens: int       # Estimated token count after compression
    compression_ratio: float     # 1 - (compressed / original)
    mode: CompressionMode
    sections_retained: List[str]  # Which ACON sections are preserved

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    def __repr__(self) -> str:
        return (
            f"CompressionResult({self.original_tokens}→{self.compressed_tokens} "
            f"tokens, {self.compression_ratio:.1%} reduction, mode={self.mode.name})"
        )


# ---------------------------------------------------------------------------
# Structured extraction helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, tag: str) -> str:
    """Extract content between XML-style tags: <TAG>...</TAG>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_vars_table(vars_section: str) -> List[Dict[str, str]]:
    """Parse the VARS markdown table into a list of dicts."""
    if not vars_section:
        return []

    lines = vars_section.strip().split("\n")
    result = []
    in_table = False
    headers = []

    for line in lines:
        line = line.strip()
        if line.startswith("|") and "---" not in line.replace("-", "").strip():
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not in_table:
                headers = cells
                in_table = True
            elif len(cells) == len(headers):
                result.append(dict(zip(headers, cells)))

    return result


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token (GPT-family heuristic)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Context Compressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """ACON-style structured context compression for agent trajectories.

    This is the main entry point for Phase 4 context compression.
    It wraps two sub-compressors:

    * **History compressor** — compresses accumulated multi-turn interaction
      history into a structured summary (REASONING + VARS + ACTIONS).
    * **Observation compressor** — elides irrelevant details from single
      tool outputs or environment observations.

    The compressor can operate in three modes (ACON §4.2):
    - ``UT`` (Utility) — preserve accuracy, safe for information-seeking tasks
    - ``CO`` (Compression) — aggressive reduction for cost/latency
    - ``UTCO`` (Utility + Compression) — two-pass for maximum efficiency

    Parameters
    ----------
    thresholds : CompressionThresholds
        Token-count triggers for compression.
    mode : CompressionMode
        Default compression aggressiveness.
    """

    def __init__(
        self,
        thresholds: CompressionThresholds | None = None,
        mode: CompressionMode = CompressionMode.UT,
    ):
        self._thresholds = thresholds or CompressionThresholds.for_agent_scenario()
        self._mode = mode
        self._lock = threading.RLock()

        # ── Compression guidelines ──
        self._history_guideline: str = NAIVE_COMPRESSION_TEMPLATE
        self._observation_guideline: str = OBSERVATION_COMPRESSION_TEMPLATE

        # ── Optimized guidelines (updated via ACON optimization loop) ──
        self._history_guideline_optimized: Optional[str] = None
        self._observation_guideline_optimized: Optional[str] = None

        # ── Feedback log for guideline optimization (ACON §3.3) ──
        self._feedback_log: List[Dict] = []

        # ── Stats ──
        self._total_history_compressions: int = 0
        self._total_observation_compressions: int = 0
        self._total_tokens_saved: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compress_history(self, history_tokens: int) -> bool:
        """Check if history compression should fire (ACON Eq.3)."""
        return history_tokens > self._thresholds.history_token_threshold

    def should_compress_observation(self, observation_tokens: int) -> bool:
        """Check if observation compression should fire (ACON Eq.4)."""
        return observation_tokens > self._thresholds.observation_token_threshold

    def compress_history(
        self,
        messages: List[Dict],
        mode: CompressionMode | None = None,
    ) -> Tuple[str, CompressionResult]:
        """Compress accumulated interaction history into a structured summary.

        This is the ACON history compression path (Eq.3).  Messages below
        the threshold are passed through unchanged; longer histories are
        fed through the compression guideline template.

        Parameters
        ----------
        messages : list[dict]
            Agent conversation messages (role/content format).
        mode : CompressionMode | None
            Override the default compression mode.

        Returns
        -------
        tuple[str, CompressionResult]
            ``(compressed_summary, result_metadata)``
        """
        mode = mode or self._mode
        text = "\n".join(
            f"[{m.get('role', '?')}] {m.get('content', '')[:500]}"
            for m in messages
        )
        token_est = _estimate_tokens(text)

        if not self.should_compress_history(token_est):
            return text, CompressionResult(
                compressed_text=text,
                original_tokens=token_est,
                compressed_tokens=token_est,
                compression_ratio=0.0,
                mode=mode,
                sections_retained=["raw"],
            )

        # Build the compression prompt
        guideline = self._get_history_guideline()
        # In production, this would call the LLM compressor.
        # For now we produce the structured template with extracted data.
        sections = self._extract_structured_summary(messages, mode)

        compressed = guideline + "\n\n" + sections

        compressed_tokens = _estimate_tokens(compressed)
        ratio = 1.0 - (compressed_tokens / max(token_est, 1))

        result = CompressionResult(
            compressed_text=compressed,
            original_tokens=token_est,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            mode=mode,
            sections_retained=list(self._section_names(sections)),
        )

        with self._lock:
            self._total_history_compressions += 1
            self._total_tokens_saved += result.tokens_saved

        return compressed, result

    def compress_observation(
        self,
        observation_text: str,
        tool_name: str = "",
        mode: CompressionMode | None = None,
    ) -> Tuple[str, CompressionResult]:
        """Compress a single tool output / observation (ACON Eq.4).

        Strips boilerplate, retains task-relevant facts, numeric values,
        and identifiers.
        """
        mode = mode or self._mode
        token_est = _estimate_tokens(observation_text)

        if not self.should_compress_observation(token_est):
            return observation_text, CompressionResult(
                compressed_text=observation_text,
                original_tokens=token_est,
                compressed_tokens=token_est,
                compression_ratio=0.0,
                mode=mode,
                sections_retained=["raw"],
            )

        compressed = self._elide_observation(observation_text, tool_name, mode)

        compressed_tokens = _estimate_tokens(compressed)
        ratio = 1.0 - (compressed_tokens / max(token_est, 1))

        result = CompressionResult(
            compressed_text=compressed,
            original_tokens=token_est,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            mode=mode,
            sections_retained=["observation"],
        )

        with self._lock:
            self._total_observation_compressions += 1
            self._total_tokens_saved += result.tokens_saved

        return compressed, result

    # ------------------------------------------------------------------
    # Guideline management (ACON §3.3)
    # ------------------------------------------------------------------

    def set_guideline(self, kind: str, guideline: str):
        """Set an optimized compression guideline.

        Parameters
        ----------
        kind : str
            ``"history"`` or ``"observation"``.
        guideline : str
            The optimized natural-language guideline.
        """
        with self._lock:
            if kind == "history":
                self._history_guideline_optimized = guideline
            elif kind == "observation":
                self._observation_guideline_optimized = guideline
            else:
                raise ValueError(f"Unknown guideline kind: {kind}")

    def get_guideline(self, kind: str) -> str:
        """Return the current guideline (optimized if available)."""
        if kind == "history":
            return self._history_guideline_optimized or self._history_guideline
        elif kind == "observation":
            return self._observation_guideline_optimized or self._observation_guideline
        raise ValueError(f"Unknown guideline kind: {kind}")

    def record_feedback(self, feedback: Dict):
        """Record compression feedback for guideline optimization (ACON §3.3).

        Feedback is generated by contrasting successful trajectories
        (without compression) with failed ones (with compression).
        """
        with self._lock:
            self._feedback_log.append(feedback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_history_guideline(self) -> str:
        return self._history_guideline_optimized or self._history_guideline

    def _extract_structured_summary(
        self,
        messages: List[Dict],
        mode: CompressionMode,
    ) -> str:
        """Extract a structured summary from conversation messages.

        This mirrors ACON's structured output format (Fig.3):
        REASONING, VARS, ACTIONS_EXECUTED, OPEN_TASKS.
        """
        # Extract key actions executed
        actions = []
        for m in messages:
            if m.get("role") == "tool":
                name = m.get("name", m.get("tool_call_id", "?"))
                content = str(m.get("content", ""))[:100]
                status = "success" if "error" not in content.lower() else "failure"
                actions.append(f"  • {name}: {status} — {content[:80]}")

        # Extract state variables from tool outputs and assistant messages
        vars_found = []
        for m in messages:
            content = str(m.get("content", ""))
            # Look for structured value patterns
            for match in re.finditer(
                r"(?:^|\n)\s*(\w[\w_]*)\s*[:=]\s*(.+?)(?:\n|$)",
                content,
            ):
                name, value = match.groups()
                if len(name) < 30 and len(value) < 100:
                    vars_found.append({"name": name, "value": value.strip()[:80]})

        # Build structured output
        parts = []

        # REASONING
        reasoning = self._infer_reasoning(messages)
        parts.append("<REASONING>")
        parts.append(reasoning or "  • Task in progress")
        parts.append("</REASONING>")

        # VARS
        parts.append("\n<VARS>")
        if vars_found:
            parts.append("  | name | value |")
            parts.append("  |------|-------|")
            seen = set()
            for v in vars_found[:20]:
                key = v["name"]
                if key not in seen:
                    parts.append(f"  | {key} | {v['value']} |")
                    seen.add(key)
        else:
            parts.append("  (no state variables detected)")
        parts.append("</VARS>")

        # ACTIONS_EXECUTED
        parts.append("\n<ACTIONS_EXECUTED>")
        if actions:
            parts.extend(actions[-30:])  # keep last 30 actions
        else:
            parts.append("  (no actions executed)")
        parts.append("</ACTIONS_EXECUTED>")

        # OPEN_TASKS
        parts.append("\n<OPEN_TASKS>")
        open_tasks = self._infer_open_tasks(messages)
        parts.append(open_tasks or "  • Continue with the current task")
        parts.append("</OPEN_TASKS>")

        return "\n".join(parts)

    def _infer_reasoning(self, messages: List[Dict]) -> Optional[str]:
        """Infer reasoning from assistant messages."""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = str(m.get("content", ""))
                if content:
                    lines = content.strip().split("\n")[:3]
                    return "  • " + " ".join(l[:200] for l in lines)
        return None

    def _infer_open_tasks(self, messages: List[Dict]) -> Optional[str]:
        """Infer open tasks from the last user message."""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = str(m.get("content", ""))[:300]
                return f"  • Continue: {content}"
        return None

    def _elide_observation(
        self,
        text: str,
        tool_name: str,
        mode: CompressionMode,
    ) -> str:
        """Elide irrelevant content from an observation."""
        # 1. Strip stack traces beyond first+last 3 lines
        text = self._strip_stack_trace(text)

        # 2. Keep only the first occurrence of repeated patterns
        text = self._dedup_repeated_lines(text)

        # 3. For known tool types, apply specific elision rules
        if tool_name in ("web_search", "web_extract", "search_files"):
            text = self._elide_search_results(text)
        elif tool_name in ("terminal", "read_terminal", "process"):
            text = self._elide_terminal_output(text)
        elif tool_name in ("read_file",):
            text = self._elide_file_content(text)

        # 4. In CO mode, truncate more aggressively
        if mode in (CompressionMode.CO, CompressionMode.UTCO):
            max_chars = self._thresholds.observation_compressed_max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"

        return text.strip()

    @staticmethod
    def _strip_stack_trace(text: str) -> str:
        """Strip stack traces: keep first 3 and last 3 lines."""
        lines = text.split("\n")
        trace_start = None
        for i, line in enumerate(lines):
            if "Traceback" in line or "Exception" in line:
                trace_start = i
                break
        if trace_start is not None and trace_start + 10 < len(lines):
            return "\n".join(
                lines[:trace_start + 3] +
                ["  ... [stack trace elided] ..."] +
                lines[-3:]
            )
        return text

    @staticmethod
    def _dedup_repeated_lines(text: str) -> str:
        """Remove consecutive repeated lines."""
        lines = text.split("\n")
        result = []
        prev = None
        for line in lines:
            stripped = line.strip()
            if stripped != prev or not stripped:
                result.append(line)
            prev = stripped
        return "\n".join(result)

    @staticmethod
    def _elide_search_results(text: str) -> str:
        """Elide search result boilerplate."""
        # Keep only first 500 chars + key metadata
        if len(text) <= 500:
            return text
        # Extract URLs
        urls = re.findall(r'https?://[^\s<>"]+', text)
        summary = text[:500]
        if urls:
            summary += f"\n[References: {len(urls)} URLs]"
        return summary

    @staticmethod
    def _elide_terminal_output(text: str) -> str:
        """Elide terminal output: keep first and last 20 lines."""
        lines = text.split("\n")
        if len(lines) <= 40:
            return text
        return "\n".join(
            lines[:20] +
            [f"  ... [{len(lines) - 40} lines elided] ..."] +
            lines[-20:]
        )

    @staticmethod
    def _elide_file_content(text: str) -> str:
        """Elide file content: show first 1000 chars + line count."""
        if len(text) <= 1000:
            return text
        lines = text.split("\n")
        return (
            text[:1000] +
            f"\n... [{len(lines)} total lines, {len(text)} chars]"
        )

    @staticmethod
    def _section_names(sections_text: str) -> List[str]:
        """Extract section tag names from structured output."""
        return re.findall(r"<(\w+)>", sections_text)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_history_compressions": self._total_history_compressions,
                "total_observation_compressions": self._total_observation_compressions,
                "total_tokens_saved": self._total_tokens_saved,
                "history_threshold": self._thresholds.history_token_threshold,
                "observation_threshold": self._thresholds.observation_token_threshold,
                "mode": self._mode.name,
                "has_optimized_guideline": (
                    self._history_guideline_optimized is not None
                    or self._observation_guideline_optimized is not None
                ),
                "feedback_entries": len(self._feedback_log),
            }

    def reset_stats(self):
        with self._lock:
            self._total_history_compressions = 0
            self._total_observation_compressions = 0
            self._total_tokens_saved = 0

    def reset_guidelines(self):
        """Reset optimized guidelines to defaults."""
        with self._lock:
            self._history_guideline_optimized = None
            self._observation_guideline_optimized = None
            self._feedback_log.clear()

    def __repr__(self) -> str:
        return (
            f"ContextCompressor(mode={self._mode.name}, "
            f"thresholds=H{self._thresholds.history_token_threshold}/"
            f"O{self._thresholds.observation_token_threshold}, "
            f"saved={self._total_tokens_saved}tok)"
        )
