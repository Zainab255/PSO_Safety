"""LLM-as-a-Judge baseline: one generation call returning a structured verdict.

If the verdict cannot be parsed the call is retried (``max_parse_retries``),
giving the 1-2 calls per prompt reported in Table 8. A verdict that is still
unparseable routes the prompt to REQUERY.
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


def parse_verdict(raw: str) -> str:
    """Extract SAFE / UNSAFE from the judge output (UNKNOWN if absent)."""
    for match in _JSON_OBJECT.finditer(raw):
        try:
            verdict = str(json.loads(match.group()).get("verdict", "")).upper()
        except (json.JSONDecodeError, AttributeError):
            continue
        if verdict in ("SAFE", "UNSAFE"):
            return verdict
    upper = raw.upper()
    if "UNSAFE" in upper:
        return "UNSAFE"
    if re.search(r"\bSAFE\b", upper):
        return "SAFE"
    return "UNKNOWN"


class LLMJudgeWrapper(BaseWrapper):
    """Text-generation safety classifier using the backbone itself."""

    name = "llm_judge"

    def __init__(self, client: LLMClient, cfg: Dict):
        judge_cfg = cfg["baselines"]["llm_judge"]
        self.client = client
        self.prompt_template: str = judge_cfg["prompt"]
        self.max_parse_retries: int = judge_cfg["max_parse_retries"]
        self.max_new_tokens = judge_cfg.get("max_new_tokens")
        self.phase1 = Phase1Filter(cfg) if judge_cfg["phase1"] else None
        self.sanitiser = Sanitiser(cfg["pso"]["sanitiser"])

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        if self.phase1 is not None:
            blocked = self.phase1.check(prompt, conversation)
            if blocked is not None:
                blocked.wrapper = self.name
                return blocked

        calls, verdict, raw = 0, "UNKNOWN", ""
        while calls <= self.max_parse_retries and verdict == "UNKNOWN":
            raw = self.client.generate(fill_template(self.prompt_template, prompt=prompt),
                                       max_new_tokens=self.max_new_tokens)
            calls += 1
            verdict = parse_verdict(raw)
        return self._result(prompt, verdict, raw, calls)

    generates = True

    def evaluate_batch(self, prompts, drafts=None):
        out = [None] * len(prompts)
        todo = []
        for i, p in enumerate(prompts):
            blocked = self.phase1.check(p, None) if self.phase1 is not None else None
            if blocked is not None:
                blocked.wrapper = self.name
                out[i] = blocked
            else:
                todo.append(i)
        state = {i: ("UNKNOWN", "", 0) for i in todo}
        for _ in range(self.max_parse_retries + 1):
            pending = [i for i in todo if state[i][0] == "UNKNOWN"]
            if not pending:
                break
            raws = self.client.generate_batch(
                [fill_template(self.prompt_template, prompt=prompts[i]) for i in pending],
                max_new_tokens=self.max_new_tokens)
            for i, raw in zip(pending, raws):
                state[i] = (parse_verdict(raw), raw, state[i][2] + 1)
        for i in todo:
            out[i] = self._result(prompts[i], *state[i])
        return out

    def _result(self, prompt, verdict, raw, calls):
        if verdict == "UNSAFE":
            decision, sanitized = WrapperDecision.BLOCK, None
        elif verdict == "SAFE":
            decision, sanitized = WrapperDecision.ALLOW, None
        else:
            decision, sanitized = WrapperDecision.REQUERY, self.sanitiser(prompt)

        return WrapperResult(
            wrapper=self.name, decision=decision, decided_by="judge",
            explanation=f"Judge verdict: {verdict}", sanitized_prompt=sanitized,
            model_calls=calls, risk_score=DECISION_RISK[decision],
            metrics={"judge_verdict": verdict, "judge_raw_output": raw},
        )
