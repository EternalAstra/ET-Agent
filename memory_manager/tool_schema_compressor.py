"""
Tool Schema Compressor — progressive disclosure for agent tool definitions.

Agent tool schemas (JSON Schema function definitions) can be enormous:
a single tool's full definition may be 500+ tokens, and agents routinely
carry 20–80 tools, adding 10,000–40,000 tokens to every API call.

This module provides three compression tiers, inspired by ACON's
observation compression (§3.2) and extended with ET-Agent's existing
tool-search progressive disclosure:

Tier 1 — Full schema (high-frequency tools, ~500 tokens each)
    Complete JSON Schema with parameter descriptions.

Tier 2 — Simplified schema (medium-frequency tools, ~150 tokens each)
    Name + description + required parameters only.

Tier 3 — Name-only (low-frequency tools, ~30 tokens each)
    Just the function name and a one-line hint.

The tier for each tool is determined by usage frequency.  Tools that
are never used by a session can be deferred to tool-search entirely.

References
----------
- ACON §3.2 (observation compression — elide irrelevant details)
- Hermes tool_search (progressive disclosure when tool count > threshold)
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Compression tier
# ---------------------------------------------------------------------------

class CompressionTier:
    """Compression budget allocation for one tool.

    Parameters
    ----------
    name : str
        Tool name (e.g. "web_search").
    max_tokens : int
        Token budget for this tool's schema.
    keep_full_schema : bool
        If True, preserve the complete JSON Schema definition.
    keep_required_only : bool
        If True, keep only required-parameter definitions.
    name_only : bool
        If True, elide everything except the function name.
    """

    __slots__ = ("name", "max_tokens", "keep_full_schema",
                 "keep_required_only", "name_only")

    def __init__(self, name: str, max_tokens: int = 500,
                 keep_full_schema: bool = True,
                 keep_required_only: bool = False,
                 name_only: bool = False):
        self.name = name
        self.max_tokens = max_tokens
        self.keep_full_schema = keep_full_schema
        self.keep_required_only = keep_required_only
        self.name_only = name_only

    def __repr__(self) -> str:
        mode = ("full" if self.keep_full_schema
                else "required" if self.keep_required_only
                else "name_only")
        return f"Tier({self.name}, {mode}, {self.max_tokens}tok)"


# ---------------------------------------------------------------------------
# Tool Schema Compressor
# ---------------------------------------------------------------------------

class ToolSchemaCompressor:
    """Compress tool definition lists based on usage frequency.

    High-frequency tools get full schemas; low-frequency tools get
    progressively simpler definitions, until the total token budget
    is satisfied.

    Uses ACON's principle of selective detail: only preserve what the
    agent actually needs for accurate tool selection.

    Parameters
    ----------
    max_total_tokens : int
        Hard cap on total tool-definition tokens.
    high_freq_threshold : int
        Tools used ≥ this many times get full schemas.
    mid_freq_threshold : int
        Tools used ≥ this many times get simplified schemas.
        Below this threshold → name-only.
    """

    def __init__(
        self,
        max_total_tokens: int = 4000,
        high_freq_threshold: int = 10,
        mid_freq_threshold: int = 3,
    ):
        self._max_total_tokens = max_total_tokens
        self._high_freq_threshold = high_freq_threshold
        self._mid_freq_threshold = mid_freq_threshold
        self._lock = threading.RLock()

        # tool_name → usage_count (persistent across turns)
        self._usage_counts: Dict[str, int] = {}

        # Stats
        self._total_compressions: int = 0
        self._total_tokens_saved: int = 0

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_usage(self, tool_name: str):
        """Record that *tool_name* was called by the agent."""
        with self._lock:
            self._usage_counts[tool_name] = (
                self._usage_counts.get(tool_name, 0) + 1
            )

    def get_usage_count(self, tool_name: str) -> int:
        return self._usage_counts.get(tool_name, 0)

    @property
    def usage_stats(self) -> Dict[str, int]:
        """Return a copy of usage statistics."""
        with self._lock:
            return dict(self._usage_counts)

    def reset_usage(self):
        with self._lock:
            self._usage_counts.clear()

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(
        self,
        tool_definitions: List[Dict],
        budget_override: Optional[int] = None,
    ) -> Tuple[List[Dict], List[CompressionTier]]:
        """Compress a tool definition list to fit within the token budget.

        Parameters
        ----------
        tool_definitions : list[dict]
            Full OpenAI-format function definitions.
        budget_override : int | None
            If set, override ``max_total_tokens``.

        Returns
        -------
        tuple[list[dict], list[CompressionTier]]
            ``(compressed_definitions, tier_assignments)``
        """
        budget = budget_override or self._max_total_tokens

        with self._lock:
            # Sort by usage frequency (descending)
            def sort_key(td):
                name = td.get("function", {}).get("name", "")
                return -self._usage_counts.get(name, 0)

            sorted_tools = sorted(tool_definitions, key=sort_key)

            compressed: List[Dict] = []
            tiers: List[CompressionTier] = []
            remaining_budget = budget

            for td in sorted_tools:
                name = td.get("function", {}).get("name", "")
                usage = self._usage_counts.get(name, 0)

                if remaining_budget <= 0:
                    break

                if usage >= self._high_freq_threshold:
                    tier = self._compress_full(td, remaining_budget)
                elif usage >= self._mid_freq_threshold:
                    tier = self._compress_simplified(td, remaining_budget)
                else:
                    tier = self._compress_name_only(td, remaining_budget)

                compressed.append(td)
                tiers.append(tier)
                remaining_budget -= tier.max_tokens

            original_tokens = sum(
                len(str(td)) // 4 for td in tool_definitions
            )
            compressed_tokens = sum(
                len(str(c)) // 4 for c in compressed
            )
            saved = original_tokens - compressed_tokens

            self._total_compressions += 1
            self._total_tokens_saved += saved

            return compressed, tiers

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _compress_full(self, td: Dict, budget: int) -> CompressionTier:
        """Full schema — keep everything."""
        tier = CompressionTier(
            name=td["function"]["name"],
            max_tokens=min(500, budget),
            keep_full_schema=True,
        )
        return tier

    def _compress_simplified(self, td: Dict, budget: int) -> CompressionTier:
        """Simplified schema — name + description + required params only."""
        params = td.get("function", {}).get("parameters", {})
        if "properties" in params:
            required = set(params.get("required", []))
            params["properties"] = {
                k: {
                    "type": v.get("type", "string"),
                    "description": (v.get("description", "") or "")[:80],
                }
                for k, v in params["properties"].items()
                if k in required
            }
            # Remove optional-params metadata
            for key in list(params.keys()):
                if key not in ("type", "properties", "required"):
                    del params[key]

        tier = CompressionTier(
            name=td["function"]["name"],
            max_tokens=min(150, budget),
            keep_required_only=True,
        )
        return tier

    def _compress_name_only(self, td: Dict, budget: int) -> CompressionTier:
        """Name-only — just the function name and a one-line hint."""
        desc = td.get("function", {}).get("description", "")[:80]
        td["function"]["description"] = desc
        td["function"]["parameters"] = {
            "type": "object",
            "properties": {},
        }

        tier = CompressionTier(
            name=td["function"]["name"],
            max_tokens=min(50, budget),
            name_only=True,
        )
        return tier

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_tools": len(self._usage_counts),
                "total_compressions": self._total_compressions,
                "total_tokens_saved": self._total_tokens_saved,
                "max_budget": self._max_total_tokens,
                "high_freq_threshold": self._high_freq_threshold,
                "mid_freq_threshold": self._mid_freq_threshold,
                "usage_stats": dict(self._usage_counts),
            }

    def reset_stats(self):
        with self._lock:
            self._total_compressions = 0
            self._total_tokens_saved = 0

    def hottest_tools(self, n: int = 10) -> List[Tuple[str, int]]:
        """Top-N most-used tools."""
        with self._lock:
            return sorted(
                self._usage_counts.items(),
                key=lambda x: -x[1],
            )[:n]

    def __repr__(self) -> str:
        return (
            f"ToolSchemaCompressor({len(self._usage_counts)} tools, "
            f"{self._total_tokens_saved}tok saved)"
        )
