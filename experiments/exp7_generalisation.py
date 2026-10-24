#!/usr/bin/env python3
"""Experiment 7 — Generalisation across backbones and benchmarks (RQ7).

For every backbone in ``experiments.exp7.backbones`` and every benchmark in
``experiments.exp7.benchmarks`` the PSO is evaluated on the benchmark plus
the fixed harmless set, and attacked with the adaptive attacker.

Outputs
    tables/exp7_table13_backbones.csv   TPR / FPR / AUC / ASR@max / latency on AdvBench per backbone
    tables/exp7_table14_grid.csv        TPR and ASR@max per backbone x benchmark (+ Self-Critique reference)
    tables/exp7_table15_transfer.csv    F_beta with the configured thresholds vs the backbone-specific optimum
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import load_client, method_run_metrics, release_client, run_dir, save_table, setup
from experiments.exp4_robustness import asr_by_k, attack_frame
from experiments.exp5_sensitivity import classification, pipeline_decisions
from benchmarks.loaders import load_corpus, load_dataset, load_jailbreak_templates
from core.decision import threshold_grid
from evaluation.output_judge import OutputJudge
from evaluation.stats import fmt, mean_std
from pipeline.runner import EvaluationRunner, cached, evaluate_corpus


def best_f_beta(cfg, analysis) -> float:
    e5 = cfg["experiments"]["exp5"]
    best = -np.inf
    for s in threshold_grid(e5["tau_s_grid"]):
        for b in threshold_grid(e5["tau_b_grid"]):
            if s < b:
                best = max(best, classification(cfg, analysis, *pipeline_decisions(analysis, s, b))["f_beta"])
    return float(best)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e7, e4 = cfg["experiments"]["exp7"], cfg["experiments"]["exp4"]
    seeds, kmax = cfg["evaluation"]["seeds"], e4["max_queries"]
    tau = cfg["pso"]["thresholds"]
    harmless = load_dataset(cfg, e7["harmless"])
    templates = load_jailbreak_templates(cfg)
    judge = OutputJudge(cfg).load() if e4["success_requires_unsafe_output"] else None
    main_corpus = load_corpus(cfg, cfg["evaluation"]["corpus"])

    grid_rows, transfer_rows = [], []
    for bb in e7["backbones"]:
        client = load_client(cfg, bb)
        runner = EvaluationRunner(cfg, client)
        reference = bb == e7["self_critique_reference_backbone"]
        methods = ["pso", "self_critique"] if reference else ["pso"]
        for bench in e7["benchmarks"]:
            bench_records = load_dataset(cfg, bench)
            out = evaluate_corpus(runner, harmless + bench_records, methods, seeds,
                                  run_dir(cfg, bb, "exp7", bench), generate_responses=False)
            runs = method_run_metrics(out["evaluation"], cfg)
            attacks = attack_frame(cfg, bb, seeds, ["adaptive"], methods, bench_records, templates, judge)
            for method in methods:
                r = runs[runs["method"] == method]
                asr = [asr_by_k(g, [kmax])[kmax] for _, g in attacks[attacks["method"] == method].groupby("seed")]
                row = {"backbone": bb, "benchmark": bench, "method": method, "n": len(bench_records)}
                for k, vals in (("tpr", r["tpr"]), ("fpr", r["fpr"]), ("auc", r["auc"]),
                                (f"asr_at_{kmax}", asr), ("latency_ms", r["latency_ms"])):
                    m, s = mean_std(vals)
                    row[k] = fmt(m, s, digits=1 if k == "latency_ms" else 2)
                    row[f"{k}_mean"], row[f"{k}_std"] = m, s
                grid_rows.append(row)

        analysis = cached(run_dir(cfg, bb, "main") / "analysis.csv", lambda: runner.analyse(main_corpus))
        transferred = classification(cfg, analysis, *pipeline_decisions(analysis, tau["tau_s"], tau["tau_b"]))["f_beta"]
        specific = best_f_beta(cfg, analysis)
        transfer_rows.append({"backbone": bb, "transferred_f_beta": transferred, "specific_f_beta": specific,
                              "delta_f_beta": transferred - specific})
        release_client(bb)

    grid = pd.DataFrame(grid_rows)
    pso = grid[grid["method"] == "pso"]
    save_table(pso[pso["benchmark"] == "advbench"].drop(columns=["method", "benchmark"]), cfg, "exp7_table13_backbones")
    save_table(grid, cfg, "exp7_table14_grid")
    save_table(pd.DataFrame(transfer_rows), cfg, "exp7_table15_transfer")


if __name__ == "__main__":
    main()
