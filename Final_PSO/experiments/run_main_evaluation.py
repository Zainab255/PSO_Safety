#!/usr/bin/env python3
"""Main evaluation on the full corpus (Experiments 1-3, 5, 6).

Runs every method in ``evaluation.methods`` over ``evaluation.corpus`` for
each seed in ``evaluation.seeds`` on one backbone, caching results under
``outputs/runs/<backbone>/main``.

    python -m experiments.run_main_evaluation [--config config/config.yaml] [--backbone llama2]
"""

from __future__ import annotations

from experiments.common import load_client, load_predictive_model, logger, run_dir, setup
from benchmarks.loaders import load_corpus
from evaluation.output_judge import OutputJudge
from pipeline.runner import EvaluationRunner, evaluate_corpus


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    ev = cfg["evaluation"]
    records = load_corpus(cfg, ev["corpus"])
    client = load_client(cfg, args.backbone)
    judge = OutputJudge(cfg).load() if ev["generate_responses"] else None
    runner = EvaluationRunner(cfg, client, judge, load_predictive_model(cfg))
    out = evaluate_corpus(
        runner, records, ev["methods"], ev["seeds"], run_dir(cfg, args.backbone, "main"), ev["generate_responses"]
    )
    logger.info("Main evaluation complete: %d evaluation rows.", len(out["evaluation"]))


if __name__ == "__main__":
    main()
