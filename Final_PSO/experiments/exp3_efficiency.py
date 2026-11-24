#!/usr/bin/env python3
"""Experiment 3 — Computational efficiency (RQ3).

Wrapper latency is measured from prompt receipt to control decision with
``torch.cuda.synchronize()`` before and after, after the configured warm-up
passes; downstream generation is excluded and reported separately.

Outputs
    tables/exp3_table8_latency.csv   latency mean ± std over seeds, multiplier vs PSO, model calls
    figures/fig6_latency.png         wrapper overhead stacked on mean downstream generation time
"""

from __future__ import annotations

import pandas as pd

from experiments.common import figure_path, load_main, method_run_metrics, save_table, setup
from evaluation import plots
from evaluation.stats import fmt, mean_std


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    data = load_main(cfg, args.backbone)
    runs = method_run_metrics(data["evaluation"], cfg)
    pso_ms = runs.loc[runs["method"] == "pso", "latency_ms"].mean()

    rows, per_seed = [], {}
    for method in cfg["evaluation"]["methods"]:
        r = runs[runs["method"] == method]
        m, s = mean_std(r["latency_ms"])
        lo, hi = int(r["calls_min"].min()), int(r["calls_max"].max())
        rows.append({
            "method": method, "latency_ms": fmt(m, s, digits=1), "latency_mean": m, "latency_std": s,
            "multiplier_vs_pso": m / pso_ms, "calls": f"{lo}" if lo == hi else f"{lo}-{hi}",
        })
        per_seed[method] = r["latency_ms"].tolist()
    save_table(pd.DataFrame(rows), cfg, "exp3_table8_latency")

    resp = data["responses"]
    if not resp.empty:
        downstream = float(resp["generation_ms"].mean())
        save_table(pd.DataFrame([{"downstream_generation_ms": downstream}]), cfg, "exp3_downstream_generation")
        plots.latency_bars(per_seed, downstream, figure_path(cfg, "fig6_latency"))


if __name__ == "__main__":
    main()
