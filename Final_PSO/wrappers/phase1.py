"""Phase 1 — hierarchical, inference-free prompt filtering (Eq. 4).

``Phase1Filter`` composes Stage 1.1 (keyword/pattern block list) and the
pattern half of Stage 1.2 (escalation patterns over the history). The
drift half of Stage 1.2 needs D_s^(t) and is evaluated by the orchestrator
as soon as that score exists.

``KeywordHistoryWrapper`` is the "Keyword + History" baseline: Phase 1 static
filters only, no orchestrator.
"""

from __future__ import annotations

from typing import Dict, Optional

from wrappers.base import BaseWrapper, Conversation, WrapperDecision, WrapperResult
from wrappers.history_wrapper import HistoryWrapper
from wrappers.keyword_wrapper import KeywordWrapper


class Phase1Filter:
    """Stage 1.1 + Stage 1.2 pattern scan with early termination."""

    def __init__(self, cfg: Dict, use_keyword: bool = True, use_patterns: bool = True):
        p1 = cfg["pso"]["phase1"]
        self.keyword = KeywordWrapper(p1["keyword_filter"])
        self.history = HistoryWrapper(p1["escalation_patterns"], p1["ted"], cfg["pso"]["thresholds"]["tau_s"])
        self.use_keyword = use_keyword and p1["keyword_filter"]["enabled"]
        self.use_patterns = use_patterns and p1["escalation_patterns"]["enabled"]

    def check(self, prompt: str, conversation: Optional[Conversation]) -> Optional[WrapperResult]:
        """Return a BLOCK result if a Phase 1 filter fires, else None."""
        if self.use_keyword:
            res = self.keyword.evaluate(prompt)
            if res.decision == WrapperDecision.BLOCK:
                return res
        if self.use_patterns:
            res = self.history.evaluate(prompt, conversation)
            if res.decision == WrapperDecision.BLOCK:
                return res
        return None

    def flags(self, prompt: str, conversation: Optional[Conversation]) -> Dict[str, bool]:
        """Both Phase 1 flags without early termination (for ablation analysis)."""
        return {
            "keyword_match": bool(self.keyword.matches(prompt)),
            "pattern_match": bool(self.history.pattern_matches(prompt, conversation)),
        }


class KeywordHistoryWrapper(BaseWrapper):
    """Baseline: Phase 1 static filters only."""

    name = "keyword_history"

    def __init__(self, cfg: Dict):
        self.phase1 = Phase1Filter(cfg)

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        blocked = self.phase1.check(prompt, conversation)
        if conversation is not None:
            conversation.add(prompt, None)
        if blocked is not None:
            blocked.wrapper = self.name
            return blocked
        return WrapperResult(
            wrapper=self.name, decision=WrapperDecision.ALLOW, decided_by="",
            explanation="No Phase 1 filter fired.", risk_score=0.0,
        )
