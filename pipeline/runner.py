"""Pipeline runner — evaluates prompts through one or more safety wrappers
and logs structured results to a JSONL file.

Supports all wrapper types including the SafetyOrchestrator (with Safety
Entropy metrics), LLM-as-a-Judge, and Self-Critique baselines.  Every
evaluation is timed end-to-end so latency comparisons can be made.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from models.llm_client import LLMClient
from wrappers.base import BaseWrapper, WrapperDecision, WrapperResult

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


class PipelineRunner:
    """Orchestrates prompt evaluation and result logging.

    Parameters
    ----------
    wrappers : Sequence[BaseWrapper]
        Ordered list of safety wrappers to apply.  The first wrapper whose
        decision is not ALLOW determines the final outcome.
    llm_client : LLMClient
        An already-loaded LLMClient (used for final generation when the
        prompt is allowed).
    config_path : str | None
        Path to config.json (for logging settings).
    pipeline_label : str
        Human-readable label stored in every record (e.g. "parallel",
        "sequential", "llm_judge_only").
    """

    def __init__(
        self,
        wrappers: Sequence[BaseWrapper],
        llm_client: LLMClient,
        config_path: Optional[str] = None,
        pipeline_label: str = "default",
    ):
        config_file = Path(config_path) if config_path else CONFIG_PATH
        with open(config_file) as f:
            self.config = json.load(f)

        self.wrappers = list(wrappers)
        self.llm_client = llm_client
        self.pipeline_label = pipeline_label

        log_cfg = self.config.get("logging", {})
        output_dir = Path(log_cfg.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / log_cfg.get("log_file", "experiment_001.jsonl")

    # ------------------------------------------------------------------
    # Single-prompt evaluation
    # ------------------------------------------------------------------

    def evaluate_prompt(self, prompt: str, run_id: int = 0, seed: int = 0) -> Dict:
        """Run a prompt through all wrappers and optionally generate a response.

        Returns a structured dict suitable for JSONL logging.
        """
        t_start = time.perf_counter()

        record: Dict = {
            "prompt": prompt,
            "timestamp": time.time(),
            "pipeline": self.pipeline_label,
            "run_id": run_id,
            "seed": seed,
            "wrapper_results": [],
            "final_decision": None,
            "generated_response": None,
            "total_latency_seconds": None,
            "total_model_calls": 0,
        }

        effective_prompt = prompt
        final_decision = WrapperDecision.ALLOW
        total_model_calls = 0

        for wrapper in self.wrappers:
            result: WrapperResult = wrapper.evaluate(effective_prompt)

            result_dict = {
                "wrapper": result.wrapper,
                "decision": result.decision.value,
                "explanation": result.explanation,
                "metrics": result.metrics,
            }
            record["wrapper_results"].append(result_dict)
            total_model_calls += result.metrics.get("model_calls", 0)

            if result.decision == WrapperDecision.BLOCK:
                final_decision = WrapperDecision.BLOCK
                break

            if result.decision == WrapperDecision.REQUERY:
                final_decision = WrapperDecision.REQUERY
                if result.sanitized_prompt:
                    effective_prompt = result.sanitized_prompt

        record["final_decision"] = final_decision.value

        # Generate only when the prompt passes (ALLOW or sanitized REQUERY)
        if final_decision != WrapperDecision.BLOCK:
            try:
                record["generated_response"] = self.llm_client.generate(
                    effective_prompt
                )
                total_model_calls += 1
            except Exception as exc:
                logger.error("Generation failed: %s", exc)
                record["generated_response"] = f"[ERROR] {exc}"

        record["total_latency_seconds"] = round(
            time.perf_counter() - t_start, 6
        )
        record["total_model_calls"] = total_model_calls
        return record

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def run_batch(
        self, prompts: List[str], run_id: int = 0, seed: int = 0
    ) -> List[Dict]:
        """Evaluate a list of prompts and return all records."""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(
                "[%s] run=%d  prompt %d/%d",
                self.pipeline_label, run_id, i + 1, len(prompts),
            )
            record = self.evaluate_prompt(prompt, run_id=run_id, seed=seed)
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_results(self, records: List[Dict]) -> Path:
        """Append records to the JSONL log file."""
        with open(self.log_path, "a") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        logger.info("Logged %d records to %s", len(records), self.log_path)
        return self.log_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run_and_log(
        self, prompts: List[str], run_id: int = 0, seed: int = 0
    ) -> List[Dict]:
        """Evaluate a batch and persist results in one call."""
        records = self.run_batch(prompts, run_id=run_id, seed=seed)
        self.log_results(records)
        return records
