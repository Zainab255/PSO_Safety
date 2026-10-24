#!/usr/bin/env python3
"""Consolidated summary (Table 18) and performance radar (Fig. 11).

Reads the per-run outputs of Experiments 1, 3 and 4. A method without its own
adaptive-attack run inherits the PSO's ASR only if its decisions on the main
corpus are verified to be identical to the PSO's (marked ``asr_from_pso``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import figure_path, load_main, method_run_metrics, save_table, setup, table_path
from evaluation import plots


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    kmax = cfg["experiments"]["exp4"]["max_queries"]
    ev = load_main(cfg, args.backbone)["evaluation"]
    runs = method_run_metrics(ev, cfg)
    exp4 = table_path(cfg, "exp4_per_run")
    attacks = pd.read_csv(exp4) if exp4.exists() else pd.DataFrame(columns=["attacker", "method", f"asr_at_{kmax}"])
    adaptive = attacks[attacks["attacker"] == "adaptive"]

    def same_decisions_as_pso(method: str) -> bool:
        cols = ["seed", "prompt_id", "decision"]
        a = ev[ev["method"] == method][cols].sort_values(cols[:2]).reset_index(drop=True)
        b = ev[ev["method"] == "pso"][cols].sort_values(cols[:2]).reset_index(drop=True)
        return len(a) == len(b) and a.equals(b)

    rows = []
    for method in cfg["evaluation"]["methods"]:
        r = runs[runs["method"] == method]
        asr = adaptive.loc[adaptive["method"] == method, f"asr_at_{kmax}"]
        inherited = asr.empty and method != "pso" and same_decisions_as_pso(method)
        if inherited:
            asr = adaptive.loc[adaptive["method"] == "pso", f"asr_at_{kmax}"]
        lo, hi = int(r["calls_min"].min()), int(r["calls_max"].max())
        rows.append({
            "method": method, "tpr": r["tpr"].mean(), "fpr": r["fpr"].mean(), "uor": r["uor"].mean(),
            "auc": r["auc"].mean(), "wrapper_ms": r["latency_ms"].mean(),
            "calls": f"{lo}" if lo == hi else f"{lo}-{hi}",
            f"asr_at_{kmax}": asr.mean() if not asr.empty else np.nan, "asr_from_pso": inherited,
        })
    table = pd.DataFrame(rows)
    save_table(table, cfg, "summary_table18")

    pso_ms = float(table.loc[table["method"] == "pso", "wrapper_ms"].iloc[0])
    axes = ["Detection (TPR)", "Utility (1-FPR)", "Speed", f"Robustness (1-ASR@{kmax})", "Discrimination (AUC)"]
    values = {}
    for _, r in table.iterrows():
        v = {axes[0]: r["tpr"], axes[1]: 1 - r["fpr"], axes[2]: min(1.0, pso_ms / r["wrapper_ms"]),
             axes[3]: 1 - r[f"asr_at_{kmax}"], axes[4]: r["auc"]}
        if all(np.isfinite(list(v.values()))):
            values[r["method"]] = v
    plots.radar(values, axes, figure_path(cfg, "fig11_radar"))


if __name__ == "__main__":
    main()
