"""Self-Critique baseline: draft a response, then critique it for violations.

Pass 1 generates a draft; pass 2 asks the same model to review the request and
draft. An unparseable critique is retried (``max_parse_retries``), giving the
2-3 calls per prompt of Table 8. Unsafe -> BLOCK; safe -> ALLOW with the draft
as the response; still unparseable -> REQUERY.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from core.sanitiser import Sanitiser
from core.templating import fill_template
from models.llm_client import LLMClient
from wrappers.base import DECISION_RISK, BaseWrapper, Conversation, WrapperDecision, WrapperResult
from wrappers.phase1 import Phase1Filter

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")
_UNSAFE_TEXT = re.compile(r'"safe"\s*:\s*false|\bnot safe\b|\bunsafe\b', re.IGNORECASE)
_SAFE_TEXT = re.compile(r'"safe"\s*:\s*true', re.IGNORECASE)


def parse_critique(raw: str) -> Optional[bool]:
    """Return True (safe), False (unsafe) or None if the critique is unparseable."""
    for match in _JSON_OBJECT.finditer(raw):
        try:
            value = json.loads(match.group()).get("safe")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, bool):
            return value
    if _UNSAFE_TEXT.search(raw):
        return False
    if _SAFE_TEXT.search(raw):
        return True
    return None


class SelfCritiqueWrapper(BaseWrapper):
    """Constitutional-style generate-then-critique loop."""

    name = "self_critique"

    def __init__(self, client: LLMClient, cfg: Dict):
        sc_cfg = cfg["baselines"]["self_critique"]
        self.client = client
        self.critique_template: str = sc_cfg["critique_prompt"]
        self.max_parse_retries: int = sc_cfg["max_parse_retries"]
        self.max_new_tokens = sc_cfg.get("max_new_tokens")
        self.phase1 = Phase1Filter(cfg) if sc_cfg["phase1"] else None
        self.sanitiser = Sanitiser(cfg["pso"]["sanitiser"])

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        if self.phase1 is not None:
            blocked = self.phase1.check(prompt, conversation)
            if blocked is not None:
                blocked.wrapper = self.name
                return blocked

        draft = self.client.generate(prompt)
        calls, safe, raw = 1, None, ""
        while calls <= 1 + self.max_parse_retries and safe is None:
            raw = self.client.generate(fill_template(self.critique_template, prompt=prompt, response=draft),
                                       max_new_tokens=self.max_new_tokens)
            calls += 1
            safe = parse_critique(raw)
        return self._result(prompt, draft, safe, raw, calls)

    generates = True

    def evaluate_batch(self, prompts, drafts=None):
        """Batched draft + critique. ``drafts[i]`` (a response already sampled for prompt i) replaces pass 1."""
        out = [None] * len(prompts)
        todo = []
        for i, p in enumerate(prompts):
            blocked = self.phase1.check(p, None) if self.phase1 is not None else None
            if blocked is not None:
                blocked.wrapper = self.name
                out[i] = blocked
            else:
                todo.append(i)
        draft = {i: (drafts[i] if drafts is not None else None) for i in todo}
        missing = [i for i in todo if draft[i] is None]
        for i, d in zip(missing, self.client.generate_batch([prompts[i] for i in missing])):
            draft[i] = d
        state = {i: (None, "", 1) for i in todo}
        for _ in range(self.max_parse_retries + 1):
            pending = [i for i in todo if state[i][0] is None]
            if not pending:
                break
            raws = self.client.generate_batch(
                [fill_template(self.critique_template, prompt=prompts[i], response=draft[i]) for i in pending],
                max_new_tokens=self.max_new_tokens)
            for i, raw in zip(pending, raws):
                state[i] = (parse_critique(raw), raw, state[i][2] + 1)
        for i in todo:
            safe, raw, calls = state[i]
            out[i] = self._result(prompts[i], draft[i], safe, raw, calls)
        return out

    def _result(self, prompt, draft, safe, raw, calls):
        if safe is False:
            decision, sanitized = WrapperDecision.BLOCK, None
        elif safe is True:
            decision, sanitized = WrapperDecision.ALLOW, None
        else:
            decision, sanitized = WrapperDecision.REQUERY, self.sanitiser(prompt)

        return WrapperResult(
            wrapper=self.name, decision=decision, decided_by="critique",
            explanation=f"Critique verdict: {'unparseable' if safe is None else ('safe' if safe else 'unsafe')}",
            sanitized_prompt=sanitized, model_calls=calls, risk_score=DECISION_RISK[decision],
            response=draft if decision == WrapperDecision.ALLOW else None,
            metrics={"critique_safe": safe, "critique_raw_output": raw},
        )
