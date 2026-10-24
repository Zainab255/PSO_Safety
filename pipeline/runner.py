"""Evaluation engine shared by all experiments.

For one backbone it produces three kinds of per-prompt records:

analysis   Phase 1 flags and every Phase 2-3 signal (D~, D, S_RB, S_RJ, S_BJ,
           softmax entropy, embedding variance) for every prompt, including
           those Phase 1 would block — needed for Tables 5/16 and Figs. 2/4/10.
responses  per seed, the response each prompt would get if ALLOWed and if
           sanitised (REQUERY), whether that response is unsafe, and the
           utility degradation between the two (harmless prompts). Shared by
           all methods within a seed, so UOR differences reflect decisions.
evaluation per seed and method, the decision, wrapper latency (prompt receipt
           to decision, GPU-synchronised, downstream generation excluded),
           model calls, risk score and resulting unsafe-output flag.

Results are cached per (seed, method) so long runs can resume.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch.nn.functional as F
from transformers import set_seed

from benchmarks.loaders import PromptRecord
from core.decision import WrapperDecision
from core.predictive_model import PredictiveSafetyModel
from core.sanitiser import Sanitiser
from evaluation.output_judge import OutputJudge
from models.llm_client import LLMClient
from wrappers import (
    BaseWrapper,
    Conversation,
    EmbeddingVarianceWrapper,
    KeywordHistoryWrapper,
    LLMJudgeWrapper,
    ParallelSafetyOrchestrator,
    Phase1Filter,
    SelfCritiqueWrapper,
    WrapperResult,
)

logger = logging.getLogger(__name__)

METHOD_NAMES = ("keyword_history", "embedding_variance", "llm_judge", "self_critique", "seq_orch", "pso")


def build_method(
    name: str, client: LLMClient, cfg: Dict, predictive_model: Optional[PredictiveSafetyModel] = None
) -> BaseWrapper:
    if name == "keyword_history":
        return KeywordHistoryWrapper(cfg)
    if name == "embedding_variance":
        return EmbeddingVarianceWrapper(client, cfg)
    if name == "llm_judge":
        return LLMJudgeWrapper(client, cfg)
    if name == "self_critique":
        return SelfCritiqueWrapper(client, cfg)
    if name == "seq_orch":
        return ParallelSafetyOrchestrator(client, cfg, sequential=True, predictive_model=predictive_model,
                                          name="seq_orch")
    if name == "pso":
        return ParallelSafetyOrchestrator(client, cfg, predictive_model=predictive_model)
    raise KeyError(f"Unknown method '{name}'. Available: {METHOD_NAMES}")


def record_fields(r: PromptRecord) -> Dict:
    return {
        "prompt_id": r.prompt_id, "dataset": r.dataset, "prompt_category": r.category,
        "group": r.group, "ground_truth": r.ground_truth,
    }


class EvaluationRunner:
    def __init__(
        self,
        cfg: Dict,
        client: LLMClient,
        output_judge: Optional[OutputJudge] = None,
        predictive_model: Optional[PredictiveSafetyModel] = None,
    ):
        self.cfg = cfg
        self.client = client
        self.judge = output_judge
        self.predictive_model = predictive_model
        self.sanitiser = Sanitiser(cfg["pso"]["sanitiser"])
        self.phase1 = Phase1Filter(cfg)
        self.layer: int = cfg["pso"]["layer"]
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def scorer(self, **overrides) -> ParallelSafetyOrchestrator:
        """A PSO without Phase 1 that scores every prompt (optionally with other personas/pooling)."""
        return ParallelSafetyOrchestrator(
            self.client, self.cfg, use_keyword=False, use_patterns=False, use_ted=False,
            predictive_model=self.predictive_model, **overrides,
        )

    def warmup(self, prompt: str) -> None:
        """50 throwaway forward passes before any latency is recorded."""
        if self._warmed_up:
            return
        texts, _, _ = self.scorer().frame(prompt)
        self.client.warmup(texts, self.cfg["evaluation"]["warmup_passes"], self.layer)
        self._warmed_up = True

    def timed_evaluate(self, method: BaseWrapper, prompt: str, conversation: Optional[Conversation] = None):
        self.client.synchronize()
        t0 = time.perf_counter()
        result = method.evaluate(prompt, conversation)
        self.client.synchronize()
        return result, (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # Per-prompt signals
    # ------------------------------------------------------------------

    def analyse(self, records: Sequence[PromptRecord], scorer: Optional[ParallelSafetyOrchestrator] = None) -> pd.DataFrame:
        scorer = scorer or self.scorer()
        rows = []
        for i, r in enumerate(records):
            signals = scorer.score(r.text)
            rows.append({**record_fields(r), "prompt_text": r.text, **signals, **self.phase1.flags(r.text, None)})
            if (i + 1) % 100 == 0:
                logger.info("analysed %d/%d prompts", i + 1, len(records))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Counterfactual responses (UOR, utility degradation)
    # ------------------------------------------------------------------

    def utility_degradation(self, original: str, sanitised: str) -> float:
        """1 - cos between mean-pooled backbone embeddings of the two responses, clipped to [0, 1]."""
        if not original.strip() and not sanitised.strip():
            return 0.0
        if not original.strip() or not sanitised.strip():
            return 1.0
        emb = self.client.encode_hidden([original, sanitised], pooling="mean", layer=self.layer)
        return float(min(1.0, max(0.0, 1.0 - F.cosine_similarity(emb[0], emb[1], dim=0).item())))

    def responses(self, records: Sequence[PromptRecord], seed: int) -> pd.DataFrame:
        set_seed(seed)
        rows = []
        for i, r in enumerate(records):
            self.client.synchronize()
            t0 = time.perf_counter()
            original = self.client.generate(r.text)
            self.client.synchronize()
            gen_ms = (time.perf_counter() - t0) * 1000.0
            sanitised = self.client.generate(self.sanitiser(r.text))
            harmful = r.ground_truth == 1
            rows.append({
                **record_fields(r), "seed": seed, "generation_ms": gen_ms,
                "response_original": original, "response_sanitised": sanitised,
                "original_unsafe": float(self.judge.is_unsafe(r.text, original)) if harmful else math.nan,
                "sanitised_unsafe": float(self.judge.is_unsafe(r.text, sanitised)) if harmful else math.nan,
                "utility_degradation": math.nan if harmful else self.utility_degradation(original, sanitised),
            })
            if (i + 1) % 50 == 0:
                logger.info("seed %d: generated responses for %d/%d prompts", seed, i + 1, len(records))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Method evaluation
    # ------------------------------------------------------------------

    def _output_unsafe(self, r: PromptRecord, res: WrapperResult, resp: Optional[pd.DataFrame]) -> float:
        if r.ground_truth != 1:
            return math.nan
        if res.decision == WrapperDecision.BLOCK:
            return 0.0
        if res.response is not None:  # Self-Critique returns its own draft
            return float(self.judge.is_unsafe(r.text, res.response)) if self.judge else math.nan
        if resp is None:
            return math.nan
        col = "original_unsafe" if res.decision == WrapperDecision.ALLOW else "sanitised_unsafe"
        return float(resp.at[r.prompt_id, col])

    def run_method(
        self, name: str, method: BaseWrapper, records: Sequence[PromptRecord], seed: int,
        responses: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        set_seed(seed)
        self.warmup(records[0].text)
        resp = responses.set_index("prompt_id") if responses is not None else None
        rows = []
        for i, r in enumerate(records):
            res, ms = self.timed_evaluate(method, r.text)
            requery = res.decision == WrapperDecision.REQUERY
            rows.append({
                "backbone": self.client.key, "seed": seed, "method": name, **record_fields(r),
                "decision": res.decision.value, "decided_by": res.decided_by, "risk_score": res.risk_score,
                "latency_ms": ms, "model_calls": res.model_calls,
                "disagreement_score": res.metrics.get("disagreement_score", math.nan),
                "p_unsafe": res.metrics.get("p_unsafe", math.nan),
                "output_unsafe": self._output_unsafe(r, res, resp),
                "utility_degradation": (
                    float(resp.at[r.prompt_id, "utility_degradation"]) if requery and resp is not None else math.nan
                ),
            })
            if (i + 1) % 100 == 0:
                logger.info("[%s] seed %d: %d/%d prompts", name, seed, i + 1, len(records))
        return pd.DataFrame(rows)


def cached(path: Path, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load ``path`` if it exists, otherwise compute and save it."""
    if path.exists():
        logger.info("Using cached %s", path)
        return pd.read_csv(path, keep_default_na=True)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def evaluate_corpus(
    runner: EvaluationRunner,
    records: Sequence[PromptRecord],
    methods: Sequence[str],
    seeds: Sequence[int],
    run_dir: Path,
    generate_responses: bool,
) -> Dict[str, pd.DataFrame]:
    """Analysis + per-seed responses + per-seed/method evaluation, cached under ``run_dir``."""
    analysis = cached(run_dir / "analysis.csv", lambda: runner.analyse(records))
    responses: List[pd.DataFrame] = []
    evaluations: List[pd.DataFrame] = []
    instances = {m: build_method(m, runner.client, runner.cfg, runner.predictive_model) for m in methods}
    for seed in seeds:
        resp = None
        if generate_responses:
            resp = cached(run_dir / f"responses_seed{seed}.csv", lambda: runner.responses(records, seed))
            responses.append(resp)
        for m in methods:
            evaluations.append(cached(
                run_dir / f"eval_seed{seed}_{m}.csv",
                lambda: runner.run_method(m, instances[m], records, seed, resp),
            ))
    return {
        "analysis": analysis,
        "responses": pd.concat(responses, ignore_index=True) if responses else pd.DataFrame(),
        "evaluation": pd.concat(evaluations, ignore_index=True),
    }
