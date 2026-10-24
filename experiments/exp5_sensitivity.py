#!/usr/bin/env python3
"""Experiment 5 — Threshold sensitivity and ablation (RQ5).

Decisions of PSO variants are recomputed exactly from the logged per-prompt
signals (Phase 1 flags and D~), and UOR from the per-seed counterfactual
responses, so no variant needs a separate model run. The persona ablation
re-scores the corpus with the alternative personas.

Outputs
    tables/exp5_f_beta_grid.csv            F_beta over the (tau_s, tau_b) grid      -> figures/fig9_threshold_grid.png
    tables/exp5_roc_calibration.csv        Eq. (19) optimal thresholds per lambda
    tables/exp5_table10_phase_ablation.csv Orch. only / Kw.+Orch. / Hist.+Orch. / Full
    tables/exp5_table11_persona_ablation.csv (A) Red/Blue/Judge, (B) neutral, (C) random roles
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.common import figure_path, load_client, load_main, run_dir, save_table, setup
from benchmarks.loaders import load_corpus
from core.decision import decisions_for_thresholds, roc_threshold_calibration, threshold_grid
from core.templating import fill_template
from evaluation import metrics as M
from evaluation import plots
from evaluation.stats import fmt, mean_std, paired_one_tailed
from pipeline.runner import EvaluationRunner, cached

VARIANTS = {  # name -> (keyword filter, history pattern filter)
    "orch_only": (False, False),
    "kw_orch": (True, False),
    "hist_orch": (False, True),
    "full": (True, True),
}


def pipeline_decisions(analysis: pd.DataFrame, tau_s: float, tau_b: float,
                       use_kw: bool = True, use_pat: bool = True) -> Tuple[List[str], List[str]]:
    kw = analysis["keyword_match"].astype(bool).to_numpy() & use_kw
    pat = analysis["pattern_match"].astype(bool).to_numpy() & use_pat
    decisions = [d.value for d in decisions_for_thresholds(analysis["disagreement_score"].to_numpy(), kw | pat, tau_s, tau_b)]
    decided_by = ["keyword" if k else "history_pattern" if p else "orchestrator" for k, p in zip(kw, pat)]
    return decisions, decided_by


def classification(cfg, analysis, decisions, decided_by) -> Dict[str, float]:
    y = analysis["ground_truth"].to_numpy()
    pred = M.flagged(decisions, cfg["evaluation"]["positive_decisions"])
    scores = M.pipeline_scores(analysis["disagreement_score"].to_numpy(), decided_by)
    return {"tpr": M.tpr(y, pred), "fpr": M.fpr(y, pred), "auc": M.auc(y, scores),
            "f_beta": M.f_beta(y, pred, cfg["evaluation"]["f_beta"]),
            "delta_disagreement": float(analysis.loc[analysis["group"] == "adversarial", "disagreement_score"].mean()
                                        - analysis.loc[analysis["group"] == "harmless", "disagreement_score"].mean())}


def uor(analysis, decisions, responses_seed) -> float:
    r = responses_seed.set_index("prompt_id")
    out = []
    for pid, y, d in zip(analysis["prompt_id"], analysis["ground_truth"], decisions):
        if y != 1:
            continue
        out.append(0.0 if d == "BLOCK" else float(r.at[pid, "original_unsafe" if d == "ALLOW" else "sanitised_unsafe"]))
    return float(np.mean(out)) if out else np.nan


def summarise(rows: pd.DataFrame, reference: str, key: str, metrics: Dict[str, str]) -> pd.DataFrame:
    ref = rows[rows[key] == reference].sort_values("seed")
    out = []
    for name, g in rows.groupby(key, sort=False):
        g = g.sort_values("seed")
        row = {key: name}
        for metric, alt in metrics.items():
            m, s = mean_std(g[metric])
            p = np.nan if name == reference else paired_one_tailed(ref[metric].to_numpy(), g[metric].to_numpy(), alt)
            row[metric] = fmt(m, s, p)
            row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_p_vs_{reference}"] = m, s, p
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e5 = cfg["experiments"]["exp5"]
    seeds = cfg["evaluation"]["seeds"]
    tau = cfg["pso"]["thresholds"]
    data = load_main(cfg, args.backbone)
    analysis, responses = data["analysis"], data["responses"]

    # Threshold grid (Fig. 9) and Eq. (19) ----------------------------
    ts, tb = threshold_grid(e5["tau_s_grid"]), threshold_grid(e5["tau_b_grid"])
    grid = np.full((len(ts), len(tb)), np.nan)
    cells = []
    for i, s in enumerate(ts):
        for j, b in enumerate(tb):
            if s >= b:
                continue
            d, by = pipeline_decisions(analysis, s, b)
            c = classification(cfg, analysis, d, by)
            grid[i, j] = c["f_beta"]
            cells.append({"tau_s": s, "tau_b": b, "tpr": c["tpr"], "fpr": c["fpr"], "f_beta": c["f_beta"]})
    save_table(pd.DataFrame(cells), cfg, "exp5_f_beta_grid")
    plots.f_beta_heatmap(grid, ts, tb, (tau["tau_s"], tau["tau_b"]), cfg["evaluation"]["f_beta"],
                         figure_path(cfg, "fig9_threshold_grid"))

    def tpr_fpr(s, b):
        c = classification(cfg, analysis, *pipeline_decisions(analysis, s, b))
        return c["tpr"], c["fpr"]
    save_table(pd.DataFrame([
        dict(zip(("lambda", "tau_s", "tau_b", "objective"), (lam, *roc_threshold_calibration(tpr_fpr, ts, tb, lam))))
        for lam in e5["roc_lambdas"]
    ]), cfg, "exp5_roc_calibration")

    # Phase ablation (Table 10) ----------------------------------------
    rows = []
    for name, (kw, pat) in VARIANTS.items():
        d, by = pipeline_decisions(analysis, tau["tau_s"], tau["tau_b"], kw, pat)
        c = classification(cfg, analysis, d, by)
        for seed in seeds:
            r_seed = responses[responses["seed"] == seed] if not responses.empty else None
            rows.append({"variant": name, "seed": seed, **c,
                         "uor": uor(analysis, d, r_seed) if r_seed is not None and len(r_seed) else np.nan})
    save_table(summarise(pd.DataFrame(rows), "full", "variant",
                         {"tpr": "greater", "fpr": "less", "uor": "less", "f_beta": "greater"}),
               cfg, "exp5_table10_phase_ablation")

    # Persona ablation (Table 11) ----------------------------------------
    records = load_corpus(cfg, cfg["evaluation"]["corpus"])
    runner = None

    def scored(name: str, personas: Dict[str, str]) -> pd.DataFrame:
        def compute():
            nonlocal runner
            runner = runner or EvaluationRunner(cfg, load_client(cfg, args.backbone))
            return runner.analyse(records, runner.scorer(personas=personas))
        return cached(run_dir(cfg, args.backbone, "exp5") / f"analysis_{name}.csv", compute)

    pa = e5["persona_ablation"]
    neutral = scored("neutral", pa["neutral"])
    rows = []
    for seed in seeds:
        roles = random.Random(seed).sample(pa["random_roles"]["pool"], 3)
        rand_personas = {k: fill_template(pa["random_roles"]["template"], role=r)
                         for k, r in zip(("red", "blue", "judge"), roles)}
        configs = {"A_red_blue_judge": analysis, "B_neutral": neutral,
                   "C_random_roles": scored(f"random_roles_seed{seed}", rand_personas)}
        for name, df in configs.items():
            c = classification(cfg, df, *pipeline_decisions(df, tau["tau_s"], tau["tau_b"]))
            rows.append({"config": name, "seed": seed, **c})
    save_table(summarise(pd.DataFrame(rows), "A_red_blue_judge", "config",
                         {"tpr": "greater", "fpr": "less", "auc": "greater", "delta_disagreement": "greater"}),
               cfg, "exp5_table11_persona_ablation")


if __name__ == "__main__":
    main()
