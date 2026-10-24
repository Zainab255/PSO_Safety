#!/usr/bin/env python3
"""Experiment 2 — Utility preservation (RQ2).

Outputs
    tables/exp2_table7_utility.csv   FPR / PTR / UD per harmless sub-category (PSO, mean ± std over seeds)
    figures/fig5_pareto.png          safety-utility Pareto plot of all methods (95% CI over seeds)

UD is the mean utility degradation of harmless prompts routed to REQUERY:
1 - cosine similarity between the backbone embeddings of the response to the
original prompt and the response to the sanitised prompt (0 = unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import figure_path, load_main, method_run_metrics, save_table, setup
from evaluation import metrics as M
from evaluation import plots
from evaluation.stats import ci95, fmt, mean_std


def utility_rows(cfg, ev: pd.DataFrame, method: str) -> pd.DataFrame:
    pos = cfg["evaluation"]["positive_decisions"]
    harmless = ev[(ev["method"] == method) & (ev["group"] == "harmless")]
    subcats = list(dict.fromkeys(cfg["datasets"]["xstest"]["safe_subcategories"].values()))
    rows = []
    for name, part in [(c, harmless[harmless["prompt_category"] == c]) for c in subcats] + [("overall", harmless)]:
        if part.empty:
            continue
        per_seed = []
        for _, g in part.groupby("seed"):
            ud = g.loc[g["decision"] == "REQUERY", "utility_degradation"].dropna()
            per_seed.append({
                "fpr": float(M.flagged(g["decision"], pos).mean()),
                "ptr": float((g["decision"] == "ALLOW").mean()),
                "ud": float(ud.mean()) if len(ud) else np.nan,
            })
        s = pd.DataFrame(per_seed)
        row = {"sub_category": name, "n": int(part["prompt_id"].nunique())}
        for k in ("fpr", "ptr", "ud"):
            m, sd = mean_std(s[k])
            row[k] = fmt(m, sd)
            row[f"{k}_mean"], row[f"{k}_std"] = m, sd
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    ev = load_main(cfg, args.backbone)["evaluation"]
    save_table(utility_rows(cfg, ev, "pso"), cfg, "exp2_table7_utility")

    runs = method_run_metrics(ev, cfg)
    points = {}
    for method in cfg["evaluation"]["methods"]:
        r = runs[runs["method"] == method]
        points[method] = {"tpr": r["tpr"].mean(), "fpr": r["fpr"].mean(),
                          "tpr_ci": ci95(r["tpr"]), "fpr_ci": ci95(r["fpr"])}
    plots.pareto(points, figure_path(cfg, "fig5_pareto"))


if __name__ == "__main__":
    main()
