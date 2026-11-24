"""Attackers for Experiment 4 (Section III-J-2 and IV-F).

The defended system is a binary oracle: the attacker only learns whether a
variant was ALLOWed (REQUERY and BLOCK both mean "not let through
unmodified"). Each query is an independent single-turn prompt.

All prompts of a subset are attacked in lockstep (``run_lockstep``): at query
step k every attacker proposes its next variant, the LLM rewrites requested by
the attackers are generated in one batch, and the defence checks the variants
in one batch. Every individual attack is the same sequential, feedback-driven
procedure; only independent prompts share GPU batches.

naive       public jailbreak templates applied verbatim, one per attempt
paraphrase  template variants rewritten by an LLM paraphraser, one per attempt
adaptive    random initial template, then mutation operators applied in a
            random order; keeps the last ALLOWed variant as its base
pair        PAIR-style: an attacker LLM rewrites the request from the goal,
            the previous attempt and the binary feedback
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from core.decision import WrapperDecision
from core.templating import fill_template
from models.llm_client import LLMClient
from wrappers.base import WrapperResult

# oracle(variant) -> (result of the defended system, evasion success)
Oracle = Callable[[str], Tuple[WrapperResult, bool]]


@dataclass
class AttackStep:
    attack_attempt: int
    attack_variant: str
    decision: str
    evasion_success: bool
    entropy_score: float
    cost_total: int
    risk_score: float = float("nan")


class Attacker:
    level = "base"

    def __init__(self, cfg: Dict, templates: Sequence[str], paraphraser: Optional[LLMClient], rng: random.Random):
        self.cfg = cfg["experiments"]["exp4"]
        self.placeholder: str = cfg["datasets"]["jailbreak_templates"]["placeholder"]
        self.templates = list(templates)
        self.paraphraser = paraphraser
        self.rng = rng

    # -- helpers -------------------------------------------------------

    def apply_template(self, template: str, goal: str) -> str:
        return template.replace(self.placeholder, goal)

    def random_template(self, goal: str) -> str:
        return self.apply_template(self.rng.choice(self.templates), goal)

    # -- lockstep interface ----------------------------------------------
    # propose() returns (text, kind): kind None = final variant; "paraphrase" = text must be
    # paraphrased by the paraphraser LLM; "pair" = text is a prompt for the PAIR attacker LLM.
    def start(self, goal: str, max_queries: int) -> None:
        self.goal = goal
        self.steps: List[AttackStep] = []
        self.cost = 0

    def propose(self, k: int) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def finish(self, text: str, llm_output: Optional[str]) -> str:
        return (llm_output or "").strip() or text

    def record(self, k: int, variant: str, result: WrapperResult, evasion: bool) -> None:
        self.cost += result.model_calls
        self.steps.append(AttackStep(
            attack_attempt=k, attack_variant=variant, decision=result.decision.value,
            evasion_success=bool(evasion),
            entropy_score=float(result.metrics.get("disagreement_score", float("nan"))),
            cost_total=self.cost,
            risk_score=float(result.risk_score) if result.risk_score is not None else float("nan"),
        ))

    def paraphrase_request(self, text: str) -> str:
        return fill_template(self.cfg["paraphrase"]["prompt"], prompt=text)


class NaiveAttacker(Attacker):
    level = "naive"

    def start(self, goal, max_queries):
        super().start(goal, max_queries)
        # Distinct templates per attempt while the pool allows it.
        pool = list(self.templates)
        self.rng.shuffle(pool)
        self._order = [pool[i % len(pool)] for i in range(max_queries)]

    def propose(self, k):
        return self.apply_template(self._order[k - 1], self.goal), None


class ParaphraseAttacker(NaiveAttacker):
    level = "paraphrase"

    def propose(self, k):
        return self.apply_template(self._order[k - 1], self.goal), "paraphrase"


class AdaptiveAttacker(Attacker):
    level = "adaptive"

    def start(self, goal, max_queries):
        super().start(goal, max_queries)
        ops = list(self.cfg["mutations"])
        self.rng.shuffle(ops)
        self._ops = ops

    def propose(self, k):
        if k == 1:
            return self.random_template(self.goal), None
        allowed = [s for s in self.steps if s.decision == WrapperDecision.ALLOW.value]
        base = allowed[-1].attack_variant if allowed else self.steps[-1].attack_variant
        op = self._ops[(k - 2) % len(self._ops)]
        if op == "paraphrase":
            return base, "paraphrase"
        if op == "swap_template":
            return self.random_template(self.goal), None
        if op in self.cfg["framings"]:
            return fill_template(self.cfg["framings"][op], prompt=base), None
        raise ValueError(f"Unknown mutation operator '{op}'.")


class ScoreAwareAttacker(Attacker):
    """Defence-aware attacker that observes the defence's continuous risk score.

    It knows a persona-disagreement score is used and receives the score of every query
    (a stronger oracle than the binary one). Greedy hill climbing: each step mutates the
    lowest-risk variant found so far with a random operator, including defence-targeted
    operators (benign-intent appeals addressed to the safety evaluators, persona mimicry,
    benign padding that shifts the last-token state). For a direction-calibrated defence the
    risk score is already sign-corrected, so the attacker always descends the risk.
    """

    level = "score_aware"

    def start(self, goal, max_queries):
        super().start(goal, max_queries)
        self._ops = list(self.cfg["score_aware"]["operators"])

    def propose(self, k):
        if k == 1:
            return self.random_template(self.goal), None
        scored = [s for s in self.steps if s.risk_score == s.risk_score]  # drop NaN
        base = min(scored, key=lambda s: s.risk_score).attack_variant if scored else self.steps[-1].attack_variant
        op = self.rng.choice(self._ops)
        if op == "paraphrase":
            return base, "paraphrase"
        if op == "swap_template":
            return self.random_template(self.goal), None
        tmpl = self.cfg["score_aware"]["framings"].get(op) or self.cfg["framings"].get(op)
        if tmpl is None:
            raise ValueError(f"Unknown score-aware operator '{op}'.")
        return fill_template(tmpl, prompt=base), None


class PAIRAttacker(Attacker):
    level = "pair"

    def __init__(self, cfg, templates, paraphraser, rng, attacker_llm: LLMClient):
        super().__init__(cfg, templates, paraphraser, rng)
        self.attacker_llm = attacker_llm
        self.pair = self.cfg["pair"]

    def propose(self, k):
        previous = self.steps[-1].attack_variant if self.steps else self.goal
        feedback = self.pair["initial_feedback"] if not self.steps else (
            WrapperDecision.ALLOW.value if self.steps[-1].evasion_success else WrapperDecision.BLOCK.value
        )
        self._previous = previous
        return fill_template(self.pair["system_prompt"], goal=self.goal, previous=previous, feedback=feedback), "pair"

    def finish(self, text, llm_output):
        raw = llm_output or ""
        prefix = self.pair["response_prefix"]
        candidate = raw.split(prefix, 1)[1] if prefix in raw else raw
        return candidate.strip() or self._previous


ATTACKERS = {
    "naive": NaiveAttacker,
    "paraphrase": ParaphraseAttacker,
    "adaptive": AdaptiveAttacker,
    "score_aware": ScoreAwareAttacker,
}


_LLM_CACHE: Dict = {}


def run_lockstep(attackers: Sequence[Attacker], goals: Sequence[str], evaluate_batch, max_queries: int,
                 paraphraser: Optional[LLMClient], attacker_llm: Optional[LLMClient],
                 paraphrase_cfg: Dict, pair_cfg: Optional[Dict], cache_key=None) -> List[List[AttackStep]]:
    """Run every attacker for ``max_queries`` steps, batching LLM calls and defence checks per step.

    ``evaluate_batch(variants, goals) -> [(WrapperResult, evasion)]``.
    """
    for a, g in zip(attackers, goals):
        a.start(g, max_queries)
    for k in range(1, max_queries + 1):
        proposals = [a.propose(k) for a in attackers]
        outputs: List[Optional[str]] = [None] * len(attackers)
        for kind, llm, c in (("paraphrase", paraphraser, paraphrase_cfg), ("pair", attacker_llm, pair_cfg)):
            idx = [i for i, (_, kd) in enumerate(proposals) if kd == kind]
            if not idx:
                continue
            reqs = [attackers[i].paraphrase_request(proposals[i][0]) if kind == "paraphrase" else proposals[i][0]
                    for i in idx]
            # Feedback-independent paraphrases (same seed, attacker and text) are shared across defences.
            keys = [(cache_key, attackers[i].level, attackers[i].goal, k, r) for i, r in zip(idx, reqs)]
            need = [j for j, key in enumerate(keys) if cache_key is None or key not in _LLM_CACHE]
            if need:
                gen = llm.generate_batch([reqs[j] for j in need], temperature=c["temperature"],
                                         max_new_tokens=c["max_new_tokens"])
                for j, o in zip(need, gen):
                    if cache_key is not None and kind == "paraphrase":
                        _LLM_CACHE[keys[j]] = o
                    outputs[idx[j]] = o
            for j, i in enumerate(idx):
                if outputs[i] is None:
                    outputs[i] = _LLM_CACHE[keys[j]]
        variants = [text if kind is None else a.finish(text, outputs[i])
                    for i, (a, (text, kind)) in enumerate(zip(attackers, proposals))]
        for a, v, (result, evasion) in zip(attackers, variants, evaluate_batch(variants, goals)):
            a.record(k, v, result, evasion)
    return [a.steps for a in attackers]
