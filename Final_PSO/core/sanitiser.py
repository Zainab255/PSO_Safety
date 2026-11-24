"""Sanitisation template T applied on REQUERY (Section III-H-2)."""

from __future__ import annotations

import re
from typing import Dict, List

from core.templating import fill_template


class Sanitiser:
    """Strip persona-override / imperative constructions, then apply template T."""

    def __init__(self, cfg: Dict):
        self.template: str = cfg["template"]
        if "{prompt}" not in self.template:
            raise ValueError("pso.sanitiser.template must contain a {prompt} placeholder.")
        self.patterns: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in cfg["remove_patterns"]]

    def clean(self, prompt: str) -> str:
        """Remove the configured patterns, keeping the informational core."""
        text = prompt
        for pat in self.patterns:
            text = pat.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        # A request made only of override phrases has no core left to keep.
        return text or prompt.strip()

    def __call__(self, prompt: str) -> str:
        return fill_template(self.template, prompt=self.clean(prompt))
