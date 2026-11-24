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
    if name == "pso_dircal":
        from core.calibration import direction_thresholds
        th = direction_thresholds(cfg, client.key)
        if th is None:
            raise RuntimeError("pso_dircal needs the main-corpus analysis of this backbone.")
        return ParallelSafetyOrchestrator(client, cfg, use_ted=False, name="pso_dircal",
                                          direction=th["direction"], thresholds=(th["tau_s"], th["tau_b"]))
    raise KeyError(f"Unknown method '{name}'. Available: {METHOD_NAMES}")


_CALIB_FRACTION = 0.3


def set_calibration_fraction(fraction: float) -> None:
    global _CALIB_FRACTION
    _CALIB_FRACTION = float(fraction)


def record_fields(r: PromptRecord) -> Dict:
    from benchmarks.loaders import split_of
    return {
        "prompt_id": r.prompt_id, "dataset": r.dataset, "prompt_category": r.category,
        "group": r.group, "ground_truth": r.ground_truth, "family": r.unit,
        "split": split_of(r.unit, _CALIB_FRACTION),
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
        """Phase 2-3 signals for every prompt; several prompts share one batched forward pass."""
        scorer = scorer or self.scorer()
        per_batch = int(self.cfg["backbones"].get("encode_batch_prompts", 1))
        rows = []
        for start in range(0, len(records), per_batch):
            chunk = records[start:start + per_batch]
            for r, signals in zip(chunk, scorer.score_many([r.text for r in chunk])):
                rows.append({**record_fields(r), "prompt_text": r.text, **signals, **self.phase1.flags(r.text, None)})
            if (start // per_batch) % 25 == 0:
                logger.info("analysed %d/%d prompts", min(start + per_batch, len(records)), len(records))
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

    def responses(self, records: Sequence[PromptRecord], seed: int, partial_path: Optional[Path] = None,
                  chunk: int = 128) -> pd.DataFrame:
        """Original and sanitised responses for every prompt (batched generation, checkpointed per chunk).

        ``generation_ms`` (unbatched, single-prompt generation time) is measured on the first
        ``evaluation.latency_subset`` prompts of a deterministic sample; it is NaN elsewhere.
        """
        set_seed(seed)
        done = pd.read_csv(partial_path) if partial_path is not None and partial_path.exists() else pd.DataFrame()
        done_ids = set(done["prompt_id"]) if len(done) else set()
        todo = [r for r in records if r.prompt_id not in done_ids]
        # Unbatched generation time is measured for the first seed only.
        timed = set(r.prompt_id for r in _latency_sample(records, self.cfg)) \
            if seed == self.cfg["evaluation"]["seeds"][0] else set()
        frames = [done] if len(done) else []
        for start in range(0, len(todo), chunk):
            part = todo[start:start + chunk]
            set_seed(seed + start)
            originals = self.client.generate_batch([r.text for r in part])
            sanitised = self.client.generate_batch([self.sanitiser(r.text) for r in part])
            rows = []
            for r, original, san in zip(part, originals, sanitised):
                gen_ms = math.nan
                if r.prompt_id in timed:
                    self.client.synchronize()
                    t0 = time.perf_counter()
                    self.client.generate(r.text)
                    self.client.synchronize()
                    gen_ms = (time.perf_counter() - t0) * 1000.0
                harmful = r.ground_truth == 1
                rows.append({
                    **record_fields(r), "seed": seed, "generation_ms": gen_ms,
                    "response_original": original, "response_sanitised": san,
                    "original_unsafe": float(self.judge.is_unsafe(r.text, original)) if harmful else math.nan,
                    "sanitised_unsafe": float(self.judge.is_unsafe(r.text, san)) if harmful else math.nan,
                    "utility_degradation": math.nan if harmful else self.utility_degradation(original, san),
                })
            frames.append(pd.DataFrame(rows))
            if partial_path is not None:
                pd.concat(frames, ignore_index=True).to_csv(partial_path, index=False)
            logger.info("seed %d: responses for %d/%d prompts", seed,
                        len(done_ids) + min(start + chunk, len(todo)), len(records))
        out = pd.concat(frames, ignore_index=True)
        order = {r.prompt_id: i for i, r in enumerate(records)}
        return out.sort_values("prompt_id", key=lambda c: c.map(order)).reset_index(drop=True)

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
        batched = getattr(method, "generates", False)
        results, latencies = {}, {}
        if batched:
            # Generation-based baselines: decisions for the whole corpus with batched generation
            # (Self-Critique reuses this seed's sampled response as its draft), wall-clock latency
            # timed unbatched on a fixed subset.
            drafts = None
            if resp is not None and isinstance(method, SelfCritiqueWrapper):
                drafts = [str(resp.at[r.prompt_id, "response_original"]) if r.prompt_id in resp.index else None
                          for r in records]
            for r, res in zip(records, method.evaluate_batch([r.text for r in records], drafts=drafts)):
                results[r.prompt_id] = res
            for r in _latency_sample(records, self.cfg):
                _, latencies[r.prompt_id] = self.timed_evaluate(method, r.text)
        rows = []
        for i, r in enumerate(records):
            if batched:
                res, ms = results[r.prompt_id], latencies.get(r.prompt_id, math.nan)
            else:
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


def _latency_sample(records: Sequence[PromptRecord], cfg: Dict) -> List[PromptRecord]:
    import random
    n = int(cfg["evaluation"].get("latency_subset") or len(records))
    if n >= len(records):
        return list(records)
    return random.Random(0).sample(list(records), n)


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
    if run_dir.name == "main":
        from core.calibration import ensure_thresholds
        ensure_thresholds(runner.cfg, runner.client.key, analysis)
    responses: List[pd.DataFrame] = []
    evaluations: List[pd.DataFrame] = []
    instances = {m: build_method(m, runner.client, runner.cfg, runner.predictive_model) for m in methods}
    for seed in seeds:
        resp = None
        if generate_responses:
            resp = cached(run_dir / f"responses_seed{seed}.csv",
                          lambda: runner.responses(records, seed, run_dir / f"responses_seed{seed}.partial.csv"))
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
