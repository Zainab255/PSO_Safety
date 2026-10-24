"""Stage 1.2 — conversation-history filter (Section III-C-2).

Two threat classes:

* Pattern-based escalation: the history H plus the current prompt is scanned
  for instruction-override, persona-hijacking and context-poisoning patterns.
* Temporal Entropy Drift (Def. 3):

      Phi_W^(t) = D_s^(t) - D_s^(t-W)                                  (Eq. 2)
      EscalationAlert^(t) = 1[ Phi_W^(t) > delta  and  D_s^(t) > tau_s ]  (Eq. 3)

  computed on the normalised disagreement score D~_s, the scale on which
  tau_s is defined.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from wrappers.base import BaseWrapper, Conversation, WrapperDecision, WrapperResult

PATTERN_CLASSES = ("instruction_override", "persona_hijacking", "context_poisoning")


class HistoryWrapper(BaseWrapper):
    """Escalation-pattern scan and Temporal Entropy Drift monitor."""

    name = "history_filter"

    def __init__(self, escalation_cfg: Dict, ted_cfg: Dict, tau_s: float):
        self.patterns: List[Tuple[str, re.Pattern]] = [
            (cls, re.compile(p, re.IGNORECASE))
            for cls in PATTERN_CLASSES
            for p in escalation_cfg[cls]
        ]
        self.window: int = ted_cfg["window"]
        self.delta: float = ted_cfg["delta"]
        self.tau_s = tau_s

    def pattern_matches(self, prompt: str, conversation: Optional[Conversation]) -> List[str]:
        history = conversation.turns if conversation else []
        text = "\n".join([*history, prompt])
        return [f"{cls}:{p.pattern}" for cls, p in self.patterns if p.search(text)]

    def drift(self, conversation: Optional[Conversation], score: float) -> float:
        """Eq. (2) for the current turn t = len(history) + 1; NaN if D^(t-W) is unavailable."""
        past = conversation.scores if conversation else []
        t = len(past) + 1
        if t - self.window < 1:
            return math.nan
        previous = past[t - self.window - 1]
        return math.nan if previous is None else score - previous

    def escalation_alert(self, conversation: Optional[Conversation], score: float) -> Tuple[bool, float]:
        """Eq. (3)."""
        phi = self.drift(conversation, score)
        return (not math.isnan(phi)) and phi > self.delta and score > self.tau_s, phi

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        hits = self.pattern_matches(prompt, conversation)
        if hits:
            return WrapperResult(
                wrapper=self.name, decision=WrapperDecision.BLOCK, decided_by="history_pattern",
                explanation=f"Escalation patterns in history: {hits}", risk_score=1.0,
                metrics={"pattern_matches": hits},
            )
        return WrapperResult(
            wrapper=self.name, decision=WrapperDecision.ALLOW, explanation="No escalation pattern.",
            risk_score=0.0, metrics={"pattern_matches": []},
        )
