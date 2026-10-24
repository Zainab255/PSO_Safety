#!/usr/bin/env python3
"""Experiment 6 — Predictive Safety Model (RQ6).

For each seed a calibration corpus of M prompts (stratified by ground truth)
is drawn from the main corpus; the base model sigma(a*D + b) (Eq. 13) and the
extended model with S_RJ, S_BJ (Eq. 14) are fitted by maximum likelihood and
evaluated on the remaining prompts. The first seed's fit is saved to
``paths.predictive_model`` for use by the PSO (p_unsafe, probabilistic rule).

Outputs
    tables/exp6_table12_calibration.csv  Brier / ECE / AUC-PR (mean ± std, paired t-tests vs base)
    tables/exp6_coefficients.csv         fitted coefficients and P(unsafe) at tau_s, tau_b per seed
    figures/fig10_calibration.png
    predictive_model.json, predictive_model_extended.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import figure_path, load_main, save_table, setup
from config import resolve_path
from benchmarks.loaders import stratified_indices_by_label
from core.predictive_model import PredictiveSafetyModel
from evaluation import metrics as M
from evaluation import plots
from evaluation.stats import fmt, mean_std, paired_one_tailed


def fit(df: pd.DataFrame, features) -> PredictiveSafetyModel:
    return PredictiveSafetyModel(features).fit({f: df[f] for f in features}, df["ground_truth"])


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e6 = cfg["experiments"]["exp6"]
    tau = cfg["pso"]["thresholds"]
    analysis = load_main(cfg, args.backbone)["analysis"].reset_index(drop=True)
    labels = analysis["ground_truth"].to_numpy()
    bins = cfg["evaluation"]["ece_bins"]
    seeds = cfg["evaluation"]["seeds"]

    rows, coefs, saved = [], [], {}
    for seed in seeds:
        idx = stratified_indices_by_label(labels, e6["calibration_size"], seed)
        cal = analysis.iloc[idx]
        held = analysis.drop(index=idx)
        y = held["ground_truth"].to_numpy()
        for name, feats in (("base", e6["features_base"]), ("extended", e6["features_extended"])):
            model = fit(cal, feats)
            p = model.predict_proba({f: held[f] for f in feats})
            rows.append({"model": name, "seed": seed, "brier": M.brier(y, p), "ece": M.ece(y, p, bins),
                         "auc_pr": M.auc_pr(y, p)})
            coef_row = {"model": name, "seed": seed, "intercept": model.intercept,
                        **{f"coef_{f}": c for f, c in zip(feats, model.coef)}}
            if feats == ["disagreement_score"]:
                coef_row["p_unsafe_at_tau_s"] = model.predict_one({"disagreement_score": tau["tau_s"]})
                coef_row["p_unsafe_at_tau_b"] = model.predict_one({"disagreement_score": tau["tau_b"]})
            coefs.append(coef_row)
            if seed == seeds[0]:
                saved[name] = (model, cal)
    rows = pd.DataFrame(rows)

    base = rows[rows["model"] == "base"].sort_values("seed")
    table = []
    for name, g in rows.groupby("model", sort=False):
        g = g.sort_values("seed")
        row = {"model": name}
        for metric, alt in (("brier", "less"), ("ece", "less"), ("auc_pr", "greater")):
            m, s = mean_std(g[metric])
            p = np.nan if name == "base" else paired_one_tailed(g[metric].to_numpy(), base[metric].to_numpy(), alt)
            row[metric] = fmt(m, s, p)
            row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_p_vs_base"] = m, s, p
        table.append(row)
    save_table(pd.DataFrame(table), cfg, "exp6_table12_calibration")
    save_table(pd.DataFrame(coefs), cfg, "exp6_coefficients")

    out = resolve_path(cfg["paths"]["predictive_model"])
    saved["base"][0].save(out)
    saved["extended"][0].save(out.with_name(out.stem + "_extended" + out.suffix))

    # Fig. 10 -------------------------------------------------------------
    model, cal = saved["base"]
    rng = np.random.default_rng(seeds[0])
    x = np.linspace(0.0, max(1.0, float(analysis["disagreement_score"].max())), 200)
    curves = []
    for _ in range(e6["bootstrap_samples"]):
        sample = cal.iloc[rng.integers(0, len(cal), len(cal))]
        if sample["ground_truth"].nunique() < 2:
            continue
        curves.append(fit(sample, ["disagreement_score"]).predict_proba({"disagreement_score": x}))
    band = np.percentile(np.vstack(curves), [2.5, 97.5], axis=0)
    plots.calibration(
        analysis["disagreement_score"].to_numpy(), labels.astype(float), analysis["group"].tolist(), x,
        model.predict_proba({"disagreement_score": x}), band, (model.coef[0], model.intercept),
        tau["tau_s"], tau["tau_b"], figure_path(cfg, "fig10_calibration"), seeds[0],
    )


if __name__ == "__main__":
    main()
