#!/usr/bin/env python3
"""Experiment 1 — Detection effectiveness (RQ1).

Outputs
    tables/exp1_table5_category_scores.csv   Table 5 (mean ± std over prompts)
    tables/exp1_anova.csv, exp1_tukey.csv    one-way ANOVA + Tukey HSD on D~
    tables/exp1_table6_methods.csv           Table 6 (mean ± std over seeds, paired one-tailed t-tests vs PSO)
    tables/exp1_signal_separation.csv        MSDS vs full-matrix softmax entropy separation
    tables/exp1_pooling_ablation.csv         last-token vs mean vs persona pooling (Delta D~)
    figures/fig2_violin.png, fig3_roc.png, fig4_conflict_space.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.common import (
    category_order, figure_path, load_client, load_main, method_run_metrics, run_dir, save_table, setup,
)
from benchmarks.loaders import load_corpus
from evaluation import metrics as M
from evaluation import plots
from evaluation.stats import anova_tukey, fmt, mean_std, paired_one_tailed
from pipeline.runner import EvaluationRunner, cached

# One-tailed direction of "PSO is better" per metric.
BETTER = {"tpr": "greater", "auc": "greater", "fpr": "less", "uor": "less"}


def separation(analysis: pd.DataFrame, column: str) -> float:
    """Mean score on adversarial prompts minus mean score on harmless prompts."""
    return float(analysis.loc[analysis["group"] == "adversarial", column].mean()
                 - analysis.loc[analysis["group"] == "harmless", column].mean())


def table5(cfg, analysis):
    rows = []
    for cat in category_order(cfg, analysis):
        g = analysis[analysis["prompt_category"] == cat]
        row = {"category": cat, "group": g["group"].iloc[0], "n": len(g)}
        for col in ("disagreement_score", "s_rb", "s_rj", "s_bj"):
            m, s = mean_std(g[col])
            row[col] = fmt(m, s)
            row[f"{col}_mean"], row[f"{col}_std_over_prompts"] = m, s
        rows.append(row)
    return pd.DataFrame(rows)


def table6(cfg, ev):
    runs = method_run_metrics(ev, cfg)
    pso = runs[runs["method"] == "pso"].sort_values("seed")
    rows = []
    for method in cfg["evaluation"]["methods"]:
        r = runs[runs["method"] == method].sort_values("seed")
        row = {"method": method}
        for metric, alt in BETTER.items():
            m, s = mean_std(r[metric])
            p = np.nan if method == "pso" else paired_one_tailed(pso[metric].to_numpy(), r[metric].to_numpy(), alt)
            row[metric] = fmt(m, s, p)
            row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_p_vs_pso"] = m, s, p
        rows.append(row)
    return pd.DataFrame(rows), runs


def roc_figure(cfg, ev, runs):
    grid = np.linspace(0, 1, 201)
    curves, aucs = {}, {}
    for method in cfg["evaluation"]["methods"]:
        tprs = []
        for _, g in ev[ev["method"] == method].groupby("seed"):
            scores = M.pipeline_scores(g["risk_score"].to_numpy(), g["decided_by"].fillna(""))
            tprs.append(M.roc(g["ground_truth"].to_numpy(), scores, grid))
        curves[method] = np.vstack(tprs)
        aucs[method] = float(runs.loc[runs["method"] == method, "auc"].mean())
    pso = runs[runs["method"] == "pso"]
    op = (float(pso["fpr"].mean()), float(pso["tpr"].mean())) if len(pso) else None
    plots.roc_curves(curves, grid, aucs, op, figure_path(cfg, "fig3_roc"))


def conflict_space(cfg, analysis):
    x = analysis[["s_rb", "s_rj", "s_bj"]].to_numpy()
    pca = PCA(n_components=2).fit(x)
    e1 = cfg["experiments"]["exp1"]
    tsne = TSNE(n_components=2, perplexity=min(e1["tsne_perplexity"], len(x) - 1),
                max_iter=e1["tsne_iterations"], random_state=cfg["evaluation"]["seeds"][0], init="pca")
    plots.conflict_space(pca.transform(x), tsne.fit_transform(x), analysis["group"].tolist(),
                         analysis["prompt_category"].tolist(), pca.explained_variance_ratio_,
                         figure_path(cfg, "fig4_conflict_space"))


def pooling_ablation(cfg, backbone, analysis):
    """Re-score the corpus under each pooling strategy and compare Delta D~."""
    records = load_corpus(cfg, cfg["evaluation"]["corpus"])
    runner = None
    rows = []
    for pooling in cfg["experiments"]["exp1"]["pooling_ablation"]:
        if pooling == cfg["pso"]["pooling"]:
            df = analysis
        else:
            def compute(p=pooling):
                nonlocal runner
                runner = runner or EvaluationRunner(cfg, load_client(cfg, backbone))
                return runner.analyse(records, runner.scorer(pooling=p))
            df = cached(run_dir(cfg, backbone, "main") / f"analysis_pooling_{pooling}.csv", compute)
        rows.append({"pooling": pooling, "delta_disagreement": separation(df, "disagreement_score")})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv, lambda p: p.add_argument(
        "--skip-pooling-ablation", action="store_true", help="Do not re-score the corpus with other poolings."))
    data = load_main(cfg, args.backbone)
    analysis, ev = data["analysis"], data["evaluation"]
    tau = cfg["pso"]["thresholds"]

    save_table(table5(cfg, analysis), cfg, "exp1_table5_category_scores")

    groups = {c: analysis.loc[analysis["prompt_category"] == c, "disagreement_score"].to_numpy()
              for c in category_order(cfg, analysis)}
    anova, pairs = anova_tukey(groups)
    save_table(pd.DataFrame([anova]), cfg, "exp1_anova")
    save_table(pd.DataFrame(pairs), cfg, "exp1_tukey")
    group_of = analysis.groupby("prompt_category")["group"].first().to_dict()
    plots.violin_by_category(groups, group_of, tau["tau_s"], tau["tau_b"], figure_path(cfg, "fig2_violin"))

    t6, runs = table6(cfg, ev)
    save_table(t6, cfg, "exp1_table6_methods")
    save_table(runs, cfg, "exp1_per_run_metrics")
    roc_figure(cfg, ev, runs)
    conflict_space(cfg, analysis)

    save_table(pd.DataFrame([
        {"signal": "msds_normalised", "harmless_to_adversarial_separation": separation(analysis, "disagreement_score")},
        {"signal": "softmax_entropy_normalised", "harmless_to_adversarial_separation": separation(analysis, "softmax_entropy")},
    ]), cfg, "exp1_signal_separation")

    if not args.skip_pooling_ablation:
        save_table(pooling_ablation(cfg, args.backbone, analysis), cfg, "exp1_pooling_ablation")


if __name__ == "__main__":
    main()
