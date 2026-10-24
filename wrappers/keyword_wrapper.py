"""Stage 1.1 — Keyword and Pattern Filter W_1 (Eq. 1).

Deterministic, case-insensitive substring and regular-expression matching
against the tunable block list K from the config. No model inference.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from config import resolve_path
from wrappers.base import BaseWrapper, Conversation, WrapperDecision, WrapperResult


class KeywordWrapper(BaseWrapper):
    """W_1(x) = BLOCK if some k in K occurs in x, else ALLOW."""

    name = "keyword_filter"

    def __init__(self, keyword_cfg: Dict):
        self.substrings: List[str] = [s.lower() for s in keyword_cfg["substrings"]]
        regexes: List[str] = list(keyword_cfg["regex"])
        if keyword_cfg.get("blocklist_file"):
            with open(resolve_path(keyword_cfg["blocklist_file"])) as f:
                for line in f:
                    entry = line.strip()
                    if not entry or entry.startswith("#"):
                        continue
                    if entry.startswith("re:"):
                        regexes.append(entry[3:])
                    else:
                        self.substrings.append(entry.lower())
        self.patterns = [re.compile(r, re.IGNORECASE) for r in regexes]

    def matches(self, prompt: str) -> List[str]:
        lower = prompt.lower()
        hits = [k for k in self.substrings if k in lower]
        hits += [p.pattern for p in self.patterns if p.search(prompt)]
        return hits

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        hits = self.matches(prompt)
        if hits:
            return WrapperResult(
                wrapper=self.name, decision=WrapperDecision.BLOCK, decided_by="keyword",
                explanation=f"Block-list match: {hits}", risk_score=1.0,
                metrics={"keyword_matches": hits},
            )
        return WrapperResult(
            wrapper=self.name, decision=WrapperDecision.ALLOW, explanation="No block-list match.",
            risk_score=0.0, metrics={"keyword_matches": []},
        )
