"""Parallel Safety Orchestrator (Algorithm 1).

Phase 1  Hierarchical prompt filtering: keyword/pattern block list (Eq. 1)
         and history escalation patterns, with early termination (Eq. 4).
Phase 2  Persona guidance: x_k = concat(P_k, x) for k in {R, B, J} (Eq. 5),
         processed in ONE batched forward pass (or three sequential passes
         for the Sequential Orchestrator baseline).
Phase 3  Latent-space analysis: last-token pooling (Eq. 7), L2 normalisation
         (Eq. 8), MSDS D_s^(K) (Eq. 9), D~ = D/2, pairwise signals
         (Eqs. 10-12) and P(unsafe | D_s) from the fitted Predictive Safety
         Model (Eq. 13). The Temporal Entropy Drift alert (Eq. 3) needs
         D_s^(t), so it is evaluated here, before the decision; an alert
         blocks exactly as Eq. (4) prescribes.
Phase 4  Decision: geometric rule (Eq. 18) or probabilistic rule (Eq. 17).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from jinja2.exceptions import TemplateError

from core.decision import geometric_decision, probabilistic_decision
from core.msds import PERSONA_ORDER, score_embeddings
from core.predictive_model import PredictiveSafetyModel
from core.sanitiser import Sanitiser
from models.llm_client import LLMClient
from wrappers.base import BaseWrapper, Conversation, WrapperDecision, WrapperResult
from wrappers.phase1 import Phase1Filter


class ParallelSafetyOrchestrator(BaseWrapper):
    """Pre-generation safety layer driven by persona-conditioned hidden states."""

    name = "pso"

    def __init__(
        self,
        client: LLMClient,
        cfg: Dict,
        sequential: bool = False,
        personas: Optional[Dict[str, str]] = None,
        pooling: Optional[str] = None,
        use_keyword: bool = True,
        use_patterns: bool = True,
        use_ted: bool = True,
        predictive_model: Optional[PredictiveSafetyModel] = None,
        name: Optional[str] = None,
    ):
        pso = cfg["pso"]
        self.client = client
        self.sequential = sequential
        self.personas: Dict[str, str] = dict(personas or pso["personas"])
        self.framing: Dict = pso["framing"]
        self.layer: int = pso["layer"]
        self.pooling: str = pooling or pso["pooling"]
        self.normalisation: str = pso["normalisation"]
        self.entropy_temperature: float = pso["softmax_entropy_temperature"]
        self.tau_s: float = pso["thresholds"]["tau_s"]
        self.tau_b: float = pso["thresholds"]["tau_b"]
        self.decision_rule: str = pso["decision_rule"]
        self.rho_low: float = pso["probabilistic"]["rho_low"]
        self.rho_high: float = pso["probabilistic"]["rho_high"]
        self.phase1 = Phase1Filter(cfg, use_keyword=use_keyword, use_patterns=use_patterns)
        self.use_ted: bool = use_ted and pso["phase1"]["ted"]["enabled"]
        self.sanitiser = Sanitiser(pso["sanitiser"])
        self.predictive_model = predictive_model
        if name:
            self.name = name
        if self.decision_rule not in ("geometric", "probabilistic"):
            raise ValueError(f"Unknown pso.decision_rule '{self.decision_rule}'.")
        if self.decision_rule == "probabilistic" and predictive_model is None:
            raise ValueError(
                "decision_rule=probabilistic needs a fitted Predictive Safety Model "
                "(run experiments/exp6_predictive.py first)."
            )

    # ------------------------------------------------------------------
    # Phase 2: framed inputs (Eq. 5)
    # ------------------------------------------------------------------

    def frame(self, prompt: str) -> Tuple[List[str], List[Tuple[int, int]], bool]:
        """Return (framed texts in R/B/J order, persona char spans, add_special_tokens)."""
        sep = self.framing["separator"]
        texts, spans = [], []
        if self.framing["mode"] == "concat":
            for role in PERSONA_ORDER:
                persona = self.personas[role]
                texts.append(persona + sep + prompt)
                spans.append((0, len(persona)))
            return texts, spans, True
        if self.framing["mode"] == "chat_template":
            for role in PERSONA_ORDER:
                persona = self.personas[role]
                try:
                    text = self.client.apply_chat_template(
                        [{"role": "system", "content": persona}, {"role": "user", "content": prompt}]
                    )
                except TemplateError:
                    # Template without a system role (e.g. Mistral-Instruct): prepend to the user turn.
                    text = self.client.apply_chat_template([{"role": "user", "content": persona + sep + prompt}])
                start = text.find(persona)
                texts.append(text)
                spans.append((start, start + len(persona)))
            return texts, spans, False
        raise ValueError(f"Unknown pso.framing.mode '{self.framing['mode']}'.")

    def embed(self, prompt: str) -> Tuple[torch.Tensor, int]:
        """Persona embeddings [3, d] and the number of model forward calls used."""
        texts, spans, add_special = self.frame(prompt)
        if not self.sequential:
            emb = self.client.encode_hidden(texts, self.pooling, self.layer, add_special, spans)
            return emb, 1
        parts = [
            self.client.encode_hidden([t], self.pooling, self.layer, add_special, [s])
            for t, s in zip(texts, spans)
        ]
        return torch.cat(parts, dim=0), len(texts)

    # ------------------------------------------------------------------
    # Phase 3: latent-space analysis
    # ------------------------------------------------------------------

    def signals_from_embeddings(self, embeddings: torch.Tensor) -> Dict[str, float]:
        signals = score_embeddings(embeddings, self.normalisation, self.entropy_temperature)
        signals["p_unsafe"] = (
            self.predictive_model.predict_one(signals) if self.predictive_model is not None else math.nan
        )
        return signals

    def score(self, prompt: str) -> Dict[str, float]:
        """Phase 2-3 signals for a prompt regardless of Phase 1 (used for logging/analysis)."""
        embeddings, _ = self.embed(prompt)
        return self.signals_from_embeddings(embeddings)

    # ------------------------------------------------------------------
    # Phase 4: decision
    # ------------------------------------------------------------------

    def _decide(self, signals: Dict[str, float]) -> Tuple[WrapperDecision, float]:
        """Return (decision, risk score) from the Phase 3 signals."""
        if self.decision_rule == "probabilistic":
            return probabilistic_decision(signals["p_unsafe"], self.rho_low, self.rho_high), signals["p_unsafe"]
        score = signals["disagreement_score"]
        return geometric_decision(score, self.tau_s, self.tau_b), score

    # ------------------------------------------------------------------
    # Algorithm 1
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str, conversation: Optional[Conversation] = None) -> WrapperResult:
        conversation = conversation if conversation is not None else Conversation()

        blocked = self.phase1.check(prompt, conversation)
        if blocked is not None:
            conversation.add(prompt, None)
            blocked.wrapper = self.name
            return blocked

        embeddings, calls = self.embed(prompt)
        signals = self.signals_from_embeddings(embeddings)
        score = signals["disagreement_score"]

        alert, drift = (False, math.nan)
        if self.use_ted:
            alert, drift = self.phase1.history.escalation_alert(conversation, score)
        conversation.add(prompt, score)

        metrics = dict(signals, ted_drift=drift, mode="sequential" if self.sequential else "parallel")
        if alert:
            return WrapperResult(
                wrapper=self.name, decision=WrapperDecision.BLOCK, decided_by="ted",
                explanation=f"Escalation alert: drift {drift:.4f} > delta and D~ {score:.4f} > tau_s.",
                model_calls=calls, risk_score=score, metrics=metrics,
            )

        decision, risk = self._decide(signals)
        sanitized = self.sanitiser(prompt) if decision == WrapperDecision.REQUERY else None
        return WrapperResult(
            wrapper=self.name, decision=decision, decided_by="orchestrator",
            explanation=f"D~={score:.4f} (tau_s={self.tau_s}, tau_b={self.tau_b}).",
            sanitized_prompt=sanitized, model_calls=calls, risk_score=risk, metrics=metrics,
        )


class EmbeddingVarianceWrapper(ParallelSafetyOrchestrator):
    """Table 6 (‡): variance of the three persona embeddings, thresholded."""

    name = "embedding_variance"

    def __init__(self, client: LLMClient, cfg: Dict):
        use_phase1 = cfg["baselines"]["embedding_variance"]["phase1"]
        super().__init__(client, cfg, use_keyword=use_phase1, use_patterns=use_phase1, use_ted=False)
        self.threshold: float = cfg["baselines"]["embedding_variance"]["threshold"]

    def _decide(self, signals: Dict[str, float]) -> Tuple[WrapperDecision, float]:
        v = signals["embedding_variance"]
        return (WrapperDecision.BLOCK if v > self.threshold else WrapperDecision.ALLOW), v
