#!/usr/bin/env python3
"""Experiment 8 — Ambiguity control (RQ8).

Scores the harmless-but-ambiguous prompt set and compares it with the harmless
and adversarial categories of the main corpus.

Outputs
    tables/exp8_table16_ambiguity.csv   D~ (mean ± std over prompts) and ALLOW/REQUERY/BLOCK counts per category
    tables/exp8_significance.csv        one-sided Welch t-tests: each harmless/ambiguous vs each adversarial category
"""

from __future__ import annotations

import pandas as pd

from experiments.common import category_order, load_client, load_main, run_dir, save_table, setup
from experiments.exp5_sensitivity import pipeline_decisions
from benchmarks.loaders import load_dataset
from evaluation.stats import fmt, mean_std, welch
from pipeline.runner import EvaluationRunner, cached


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    tau = cfg["pso"]["thresholds"]
    main_analysis = load_main(cfg, args.backbone)["analysis"]
    reference = cfg["experiments"]["exp8"]["adversarial_reference"]
    records = load_dataset(cfg, "ambiguous")
    ambiguous = cached(run_dir(cfg, args.backbone, "exp8") / "analysis.csv",
                       lambda: EvaluationRunner(cfg, load_client(cfg, args.backbone)).analyse(records))
    # Keep ambiguous sub-categories distinct from same-named harmless ones (e.g. "creative").
    ambiguous = ambiguous.assign(prompt_category="ambiguous_" + ambiguous["prompt_category"].astype(str))

    frames = [main_analysis[main_analysis["group"] == "harmless"], ambiguous,
              main_analysis[(main_analysis["group"] == "adversarial") & (main_analysis["dataset"] == reference)]]
    df = pd.concat(frames, ignore_index=True)
    df["decision"] = pipeline_decisions(df, tau["tau_s"], tau["tau_b"])[0]
    order = [c for c in category_order(cfg, df)]

    rows = []
    for cat in order:
        g = df[df["prompt_category"] == cat]
        m, s = mean_std(g["disagreement_score"])
        rows.append({"category": cat, "group": g["group"].iloc[0], "n": len(g), "disagreement": fmt(m, s),
                     "disagreement_mean": m, "disagreement_std_over_prompts": s,
                     **{d: int((g["decision"] == d).sum()) for d in ("ALLOW", "REQUERY", "BLOCK")}})
    table = pd.DataFrame(rows)
    save_table(table, cfg, "exp8_table16_ambiguity")

    benign = [c for c in order if df.loc[df["prompt_category"] == c, "group"].iloc[0] != "adversarial"]
    adversarial = [c for c in order if c not in benign]
    tests = []
    for b in benign:
        for a in adversarial:
            xb = df.loc[df["prompt_category"] == b, "disagreement_score"]
            xa = df.loc[df["prompt_category"] == a, "disagreement_score"]
            tests.append({"benign_category": b, "adversarial_category": a, "gap": xa.mean() - xb.mean(),
                          "p_value_benign_lower": welch(xb, xa, "less")})
    save_table(pd.DataFrame(tests), cfg, "exp8_significance")


if __name__ == "__main__":
    main()
