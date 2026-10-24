"""Base classes shared by all safety wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.decision import WrapperDecision

# Ordinal risk score used for ROC analysis of methods that only emit decisions.
DECISION_RISK = {
    WrapperDecision.ALLOW: 0.0,
    WrapperDecision.REQUERY: 0.5,
    WrapperDecision.BLOCK: 1.0,
}


@dataclass
class Conversation:
    """Multi-turn history H: prompts and their normalised disagreement scores."""

    turns: List[str] = field(default_factory=list)
    scores: List[Optional[float]] = field(default_factory=list)

    def add(self, prompt: str, score: Optional[float]) -> None:
        self.turns.append(prompt)
        self.scores.append(score)


@dataclass
class WrapperResult:
    """Structured result returned by every wrapper's ``evaluate`` method."""

    wrapper: str
    decision: WrapperDecision
    explanation: str = ""
    sanitized_prompt: Optional[str] = None
    decided_by: str = ""              # keyword | history_pattern | ted | orchestrator | judge | critique
    model_calls: int = 0              # backbone calls spent on the safety check
    risk_score: float = math.nan      # continuous score for ROC analysis
    response: Optional[str] = None    # text produced during the check (Self-Critique draft)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseWrapper:
    """Abstract base class: ``evaluate(prompt, conversation) -> WrapperResult``."""

    name: str = "base"

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        raise NotImplementedError
