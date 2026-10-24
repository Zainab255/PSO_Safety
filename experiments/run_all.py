#!/usr/bin/env python3
"""Run the complete experimental pipeline in dependency order.

    python -m experiments.run_all [--config config/config.yaml] [--backbone llama2] [--skip exp4 exp7]

Every stage caches its intermediate results under ``outputs/runs`` so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import importlib

from experiments.common import setup

STAGES = [
    "run_main_evaluation",
    "exp6_predictive",      # fits the Predictive Safety Model used for p_unsafe
    "exp1_detection",
    "exp2_utility",
    "exp3_efficiency",
    "exp5_sensitivity",
    "exp4_robustness",
    "exp7_generalisation",
    "exp8_ambiguity",
    "exp9_multiturn",
    "export_logs",
    "summary",
]


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv, lambda p: p.add_argument(
        "--skip", nargs="*", default=[], choices=STAGES, help="Stages to skip."))
    passthrough = ["--backbone", args.backbone] + (["--config", args.config] if args.config else [])
    for stage in STAGES:
        if stage in args.skip:
            continue
        importlib.import_module(f"experiments.{stage}").main(passthrough)


if __name__ == "__main__":
    main()
