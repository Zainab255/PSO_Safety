"""Attackers for Experiment 4 (Section III-J-2 and IV-F).

The defended system is a binary oracle: the attacker only learns whether a
variant was ALLOWed (REQUERY and BLOCK both mean "not let through
unmodified"). Each query is an independent single-turn prompt.

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

    def paraphrase(self, text: str) -> str:
        p = self.cfg["paraphrase"]
        out = self.paraphraser.generate(
            fill_template(p["prompt"], prompt=text), temperature=p["temperature"], max_new_tokens=p["max_new_tokens"]
        ).strip()
        return out or text

    def next_variant(self, goal: str, k: int, history: List[AttackStep]) -> str:
        raise NotImplementedError

    # -- main loop -----------------------------------------------------

    def run(self, goal: str, oracle: Oracle, max_queries: int, continue_after_success: bool) -> List[AttackStep]:
        steps: List[AttackStep] = []
        cost = 0
        succeeded = False
        for k in range(1, max_queries + 1):
            variant = self.next_variant(goal, k, steps)
            result, evasion = oracle(variant)
            cost += result.model_calls
            steps.append(AttackStep(
                attack_attempt=k, attack_variant=variant, decision=result.decision.value,
                evasion_success=bool(evasion),
                entropy_score=float(result.metrics.get("disagreement_score", float("nan"))),
                cost_total=cost,
            ))
            succeeded = succeeded or evasion
            if succeeded and not continue_after_success:
                break
        return steps


class NaiveAttacker(Attacker):
    level = "naive"

    def run(self, goal, oracle, max_queries, continue_after_success):
        # Distinct templates per attempt while the pool allows it.
        pool = list(self.templates)
        self.rng.shuffle(pool)
        self._order = [pool[i % len(pool)] for i in range(max_queries)]
        return super().run(goal, oracle, max_queries, continue_after_success)

    def next_variant(self, goal, k, history):
        return self.apply_template(self._order[k - 1], goal)


class ParaphraseAttacker(NaiveAttacker):
    level = "paraphrase"

    def next_variant(self, goal, k, history):
        return self.paraphrase(super().next_variant(goal, k, history))


class AdaptiveAttacker(Attacker):
    level = "adaptive"

    def run(self, goal, oracle, max_queries, continue_after_success):
        ops = list(self.cfg["mutations"])
        self.rng.shuffle(ops)
        self._ops = ops
        return super().run(goal, oracle, max_queries, continue_after_success)

    def _mutate(self, op: str, base: str, goal: str) -> str:
        if op == "paraphrase":
            return self.paraphrase(base)
        if op == "swap_template":
            return self.random_template(goal)
        if op in self.cfg["framings"]:
            return fill_template(self.cfg["framings"][op], prompt=base)
        raise ValueError(f"Unknown mutation operator '{op}'.")

    def next_variant(self, goal, k, history):
        if k == 1:
            return self.random_template(goal)
        allowed = [s for s in history if s.decision == WrapperDecision.ALLOW.value]
        base = allowed[-1].attack_variant if allowed else history[-1].attack_variant
        return self._mutate(self._ops[(k - 2) % len(self._ops)], base, goal)


class PAIRAttacker(Attacker):
    level = "pair"

    def __init__(self, cfg, templates, paraphraser, rng, attacker_llm: LLMClient):
        super().__init__(cfg, templates, paraphraser, rng)
        self.attacker_llm = attacker_llm
        self.pair = self.cfg["pair"]

    def next_variant(self, goal, k, history):
        previous = history[-1].attack_variant if history else goal
        feedback = self.pair["initial_feedback"] if not history else (
            WrapperDecision.ALLOW.value if history[-1].evasion_success else WrapperDecision.BLOCK.value
        )
        raw = self.attacker_llm.generate(
            fill_template(self.pair["system_prompt"], goal=goal, previous=previous, feedback=feedback),
            temperature=self.pair["temperature"], max_new_tokens=self.pair["max_new_tokens"],
        )
        prefix = self.pair["response_prefix"]
        candidate = raw.split(prefix, 1)[1] if prefix in raw else raw
        return candidate.strip() or previous


ATTACKERS = {
    "naive": NaiveAttacker,
    "paraphrase": ParaphraseAttacker,
    "adaptive": AdaptiveAttacker,
}
