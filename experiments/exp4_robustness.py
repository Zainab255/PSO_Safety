#!/usr/bin/env python3
"""Experiment 4 — Adaptive attacker robustness (RQ4).

For every seed a stratified subset of AdvBench is attacked by the naive,
paraphrasing and adaptive attackers against each method in
``experiments.exp4.methods`` (binary ALLOW / not-ALLOW oracle, up to
``max_queries`` queries). A PAIR-style attacker is run against the PSO.

Outputs
    tables/exp4_table9_asr.csv           ASR@1, ASR@max, mean queries to evasion (paired t-tests vs PSO)
    tables/exp4_asr_by_k.csv             ASR@k for every k and method (adaptive attacker)
    tables/exp4_pair.csv                 PAIR-style attacker vs hand-paraphrasing adaptive attacker (PSO)
    tables/exp4_subset_stability.csv     PSO adaptive ASR@max on extra subsamples
    tables/exp4_power.csv                power of the PSO vs Self-Critique ASR@max comparison
    figures/fig7_asr.png, fig8_trajectory.png
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import set_seed

from experiments.common import figure_path, load_client, run_dir, save_table, setup
from attacks.attackers import ATTACKERS, PAIRAttacker, AttackStep
from benchmarks.loaders import load_dataset, load_jailbreak_templates, stratified_subset
from core.decision import WrapperDecision
from evaluation import metrics as M
from evaluation import plots
from evaluation.output_judge import OutputJudge
from evaluation.stats import fmt, mean_std, paired_one_tailed, paired_t_power
from pipeline.runner import build_method, cached
from wrappers import ParallelSafetyOrchestrator


def make_oracle(method, scorer, client, judge, goal: str, requires_unsafe: bool):
    def oracle(variant: str):
        res = method.evaluate(variant)
        evasion = res.decision == WrapperDecision.ALLOW
        if evasion and requires_unsafe:
            evasion = judge.is_unsafe(goal, res.response or client.generate(variant))
        if isinstance(method, ParallelSafetyOrchestrator) and math.isnan(res.metrics.get("disagreement_score", math.nan)):
            # Phase 1 blocked the variant; record its score for the D~ trajectory (Fig. 8).
            res.metrics["disagreement_score"] = scorer.score(variant)["disagreement_score"]
        return res, evasion
    return oracle


def run_attacks(cfg, backbone, seed, level, method_name, subset, templates, judge, attacker_llm=None) -> pd.DataFrame:
    e4 = cfg["experiments"]["exp4"]
    client = load_client(cfg, backbone)
    method = build_method(method_name, client, cfg)
    scorer = ParallelSafetyOrchestrator(client, cfg, use_keyword=False, use_patterns=False, use_ted=False)
    set_seed(seed)
    rows: List[Dict] = []
    for r in subset:
        rng = random.Random(f"{seed}|{level}|{r.prompt_id}")
        if level == "pair":
            attacker = PAIRAttacker(cfg, templates, client, rng, attacker_llm)
        else:
            attacker = ATTACKERS[level](cfg, templates, client, rng)
        oracle = make_oracle(method, scorer, client, judge, r.text, e4["success_requires_unsafe_output"])
        steps: List[AttackStep] = attacker.run(r.text, oracle, e4["max_queries"], e4["continue_after_success"])
        for s in steps:
            rows.append({
                "backbone": backbone, "seed": seed, "attacker": level, "method": method_name,
                "prompt_id": r.prompt_id, "dataset": r.dataset, "prompt_category": r.category,
                "prompt_text": r.text, "ground_truth": r.ground_truth, **s.__dict__,
            })
    return pd.DataFrame(rows)


def first_success(df: pd.DataFrame) -> pd.Series:
    """Query index of the first evasion per prompt (NaN if never evaded)."""
    succ = df[df["evasion_success"].astype(bool)].groupby("prompt_id")["attack_attempt"].min()
    return succ.reindex(df["prompt_id"].unique())


def asr_by_k(df: pd.DataFrame, ks) -> Dict[int, float]:
    fs = first_success(df).to_numpy(dtype=float)
    return {k: M.asr_at_k(fs, k) for k in ks}


def attack_frame(cfg, backbone, seeds, levels, methods, records, templates, judge, attacker_llm=None) -> pd.DataFrame:
    """Attack a stratified subset of ``records`` per seed; cached per (dataset, seed, attacker, method)."""
    e4 = cfg["experiments"]["exp4"]
    dataset = records[0].dataset
    frames = []
    for seed in seeds:
        subset = stratified_subset(records, e4["subset_size"], seed)
        for level in levels:
            for m in methods:
                frames.append(cached(
                    run_dir(cfg, backbone, "exp4") / f"attack_{dataset}_seed{seed}_{level}_{m}.csv",
                    lambda: run_attacks(cfg, backbone, seed, level, m, subset, templates, judge, attacker_llm),
                ))
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e4 = cfg["experiments"]["exp4"]
    seeds, kmax = cfg["evaluation"]["seeds"], e4["max_queries"]
    advbench = load_dataset(cfg, "advbench")
    templates = load_jailbreak_templates(cfg)
    judge = OutputJudge(cfg).load() if e4["success_requires_unsafe_output"] else None

    attacks = attack_frame(cfg, args.backbone, seeds, e4["attackers"], e4["methods"], advbench, templates, judge)

    # Table 9 ---------------------------------------------------------
    per_run = []
    for (level, method, seed), g in attacks.groupby(["attacker", "method", "seed"]):
        fs = first_success(g).to_numpy(dtype=float)
        per_run.append({
            "attacker": level, "method": method, "seed": seed,
            "asr_at_1": M.asr_at_k(fs, 1), f"asr_at_{kmax}": M.asr_at_k(fs, kmax),
            "queries_to_evasion": float(np.nanmean(fs)) if np.isfinite(fs).any() else np.nan,
        })
    per_run = pd.DataFrame(per_run)
    save_table(per_run, cfg, "exp4_per_run")
    rows = []
    for level in e4["attackers"]:
        pso = per_run[(per_run["attacker"] == level) & (per_run["method"] == "pso")].sort_values("seed")
        for method in e4["methods"]:
            r = per_run[(per_run["attacker"] == level) & (per_run["method"] == method)].sort_values("seed")
            row = {"attacker": level, "method": method}
            for col in ("asr_at_1", f"asr_at_{kmax}"):
                m, s = mean_std(r[col])
                p = np.nan if method == "pso" or pso.empty else paired_one_tailed(
                    pso[col].to_numpy(), r[col].to_numpy(), "less")
                row[col] = fmt(m, s, p)
                row[f"{col}_mean"], row[f"{col}_p_vs_pso"] = m, p
            q = r["queries_to_evasion"].dropna()
            row["queries"] = f"{q.min():.1f}-{q.max():.1f}" if len(q) else "—"
            rows.append(row)
    save_table(pd.DataFrame(rows), cfg, "exp4_table9_asr")

    # Fig. 7: ASR@k under the adaptive attacker ------------------------
    ks = list(range(1, kmax + 1))
    adaptive = attacks[attacks["attacker"] == "adaptive"]
    curves, by_k = {}, []
    for method in e4["methods"]:
        mat = []
        for seed in seeds:
            a = asr_by_k(adaptive[(adaptive["method"] == method) & (adaptive["seed"] == seed)], ks)
            mat.append([a[k] for k in ks])
        curves[method] = np.asarray(mat)
    for k in ks:
        row = {"k": k}
        for method in e4["methods"]:
            col = curves[method][:, k - 1]
            row[f"{method}_mean"], row[f"{method}_std"] = mean_std(col)
            if method != "pso":
                row[f"{method}_p_pso_greater"] = paired_one_tailed(curves["pso"][:, k - 1], col, "greater")
        by_k.append(row)
    save_table(pd.DataFrame(by_k), cfg, "exp4_asr_by_k")
    plots.asr_curves(curves, ks, figure_path(cfg, "fig7_asr"))

    # Fig. 8: D~ trajectory of the PSO under adaptive attack -----------
    pso_adv = adaptive[adaptive["method"] == "pso"]
    traj = {"Never bypassed": [], "Eventually bypassed": []}
    for (_, _), g in pso_adv.groupby(["seed", "prompt_id"]):
        v = np.full(kmax, np.nan)
        v[g["attack_attempt"].to_numpy() - 1] = g["entropy_score"].to_numpy()
        traj["Eventually bypassed" if g["evasion_success"].astype(bool).any() else "Never bypassed"].append(v)
    plots.score_trajectory({k: np.asarray(v).reshape(-1, kmax) for k, v in traj.items()},
                           cfg["pso"]["thresholds"]["tau_b"], figure_path(cfg, "fig8_trajectory"))

    # PAIR-style attacker (PSO) ---------------------------------------
    if e4["pair"]["enabled"]:
        attacker_llm = load_client(cfg, e4["pair"]["attacker_backbone"])
        pair = attack_frame(cfg, args.backbone, seeds, ["pair"], ["pso"], advbench, templates, judge, attacker_llm)
        pair_asr = [asr_by_k(g, [kmax])[kmax] for _, g in pair.groupby("seed")]
        hand = per_run[(per_run["attacker"] == "adaptive") & (per_run["method"] == "pso")][f"asr_at_{kmax}"]
        save_table(pd.DataFrame([
            {"attacker": "pair", "asr": fmt(*mean_std(pair_asr)), "asr_mean": mean_std(pair_asr)[0]},
            {"attacker": "adaptive (hand-paraphrasing)", "asr": fmt(*mean_std(hand)), "asr_mean": mean_std(hand)[0]},
        ]), cfg, "exp4_pair")

    # Subset stability and power ---------------------------------------
    stab = attack_frame(cfg, args.backbone, e4["stability_seeds"], ["adaptive"], ["pso"], advbench, templates, judge)
    save_table(pd.DataFrame([
        {"subsample_seed": s, f"pso_asr_at_{kmax}": asr_by_k(g, [kmax])[kmax]} for s, g in stab.groupby("seed")
    ]), cfg, "exp4_subset_stability")

    a = per_run[(per_run["attacker"] == "adaptive")]
    pso = a[a["method"] == "pso"].sort_values("seed")[f"asr_at_{kmax}"].to_numpy()
    if "self_critique" in e4["methods"]:
        sc = a[a["method"] == "self_critique"].sort_values("seed")[f"asr_at_{kmax}"].to_numpy()
        save_table(pd.DataFrame([{
            "comparison": f"PSO vs Self-Critique ASR@{kmax}", "n_runs": len(pso),
            "alpha": cfg["evaluation"]["significance_alpha"],
            "power": paired_t_power(pso - sc, len(pso), cfg["evaluation"]["significance_alpha"]),
        }]), cfg, "exp4_power")


if __name__ == "__main__":
    main()
