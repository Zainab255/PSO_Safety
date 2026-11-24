#!/usr/bin/env python3
"""Experiment 15 -- independent data and adaptive attacks for the template-aware probe
(pre-registered in PREREGISTRATION_exp15.md).

Part A (independent data): the Exp. 13 (untemplated) and Exp. 14 (template-aware) probes, fitted unchanged,
are applied to attack-generated jailbreaks (JailbreakBench PAIR / GCG artifacts), OR-Bench toxic prompts and
OR-Bench-Hard harmless prompts.
Part B (adaptive attacks; primary backbone, or any backbone with --attacks): template, adaptive and score-aware black-box attackers against the
template-aware probe, ten queries per goal, AdvBench goals from test-split families.

Outputs
    tables/exp15_independent_<bb>.csv      detection / flag rates and AUCs per set
    tables/exp15_attacks_<bb>.csv          ASR@1 / ASR@10 per attacker (lenient validity rule)
    runs/<bb>/exp15/                       per-prompt scores and per-query attack logs
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.attackers import AdaptiveAttacker, NaiveAttacker, ScoreAwareAttacker, run_lockstep
from benchmarks.loaders import load_jailbreak_templates, split_of
from core.decision import WrapperDecision
from evaluation import metrics as M
from experiments.common import load_client, release_client, run_dir, save_table, setup
from experiments.exp12_baselines import all_layer_states, fit_probe, youden_threshold
from experiments.exp13_template_control import build_frame, hidden_states
from experiments.exp14_template_aware_probe import fit_template_aware
from experiments.exp4_robustness import valid_evasions
from wrappers.base import WrapperResult

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def untemplated_probe(cfg, df, H):
    """The Exp. 13 probe (main-corpus calibration only), CV layer."""
    ev, e12 = cfg["evaluation"], cfg["experiments"]["exp12"]
    in_main = df["set"].isin(["advbench_framed", "xstest_safe", "xstest_unsafe"]).to_numpy()
    calib = in_main & (df["split"] == "calib").to_numpy()
    y = df["ground_truth"].to_numpy().astype(int); g = df["family"].astype(str).to_numpy()
    best, best_cv = None, -1.0
    for layer in range(H.shape[1]):
        _, _, cv = fit_probe(H[calib, layer].astype(np.float32), y[calib], g[calib], e12["probe_C"])
        if cv > best_cv:
            best, best_cv = layer, cv
    m, c, _ = fit_probe(H[calib, best].astype(np.float32), y[calib], g[calib], e12["probe_C"])
    t = youden_threshold(m.decision_function(H[calib, best].astype(np.float32)), y[calib], ev["calibration_lambda"])
    return m, best, t


def main(argv=None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attacks", action="store_true")
    known, rest = pre.parse_known_args(argv)
    cfg, args = setup(__doc__, rest)
    bb = args.backbone
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))
    out_dir = run_dir(cfg, bb, "exp15"); out_dir.mkdir(parents=True, exist_ok=True)

    df, texts = build_frame(cfg)
    H = hidden_states(cfg, bb, texts, run_dir(cfg, bb, "exp13"), bs)
    m14, L14, _, t14, _ = fit_template_aware(cfg, df, H)
    m13, L13, t13 = untemplated_probe(cfg, df, H)

    # ---------------------------------------------------------------- Part A: independent data
    ind = pd.read_csv(RAW / "independent_sets.csv")
    cache = out_dir / "independent_states.npy"
    client = None
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(ind):
        Hi = np.load(cache)
    else:
        client = load_client(cfg, bb)
        Hi = all_layer_states(client, ind["prompt"].astype(str).tolist(), bs)
        np.save(cache, Hi)
    xs_test = ((df["set"] == "xstest_safe") & (df["split"] == "test")).to_numpy()
    rows = []
    for name, m, L, t in (("template_aware", m14, L14, t14), ("untemplated", m13, L13, t13)):
        s_ind = m.decision_function(Hi[:, L].astype(np.float32))
        s_xs = m.decision_function(H[xs_test, L].astype(np.float32))
        ind[f"score_{name}"] = s_ind
        for st in ind["set"].unique():
            sel = (ind["set"] == st).to_numpy()
            rows.append({"backbone": bb, "probe": name, "kind": "flag_rate", "set": st, "n": int(sel.sum()),
                         "value": float((s_ind[sel] >= t).mean())})
        harm = (ind["label"] == 1).to_numpy(); hard = (ind["set"] == "orbench_hard").to_numpy()
        for hk in [s for s in ind["set"].unique() if s != "orbench_hard"]:
            hs_ = (ind["set"] == hk).to_numpy()
            for neg_name, neg in (("orbench_hard", s_ind[hard]), ("xstest_safe_test", s_xs)):
                yy = np.r_[np.ones(hs_.sum()), np.zeros(len(neg))]
                rows.append({"backbone": bb, "probe": name, "kind": "auc", "set": f"{hk} vs {neg_name}",
                             "n": int(len(yy)), "value": float(M.auc(yy, np.r_[s_ind[hs_], neg]))})
    ind.drop(columns=["prompt"]).to_csv(out_dir / "independent_scores.csv", index=False)
    tab = pd.DataFrame(rows)
    save_table(tab, cfg, f"exp15_independent_{bb}")
    print(tab.round(3).to_string())

    # ---------------------------------------------------------------- Part B: adaptive attacks
    # Attacks run on the primary backbone by default (or on any backbone with --attacks).
    if not (known.attacks or bb == cfg["backbones"]["primary"]):
        if client is not None:
            release_client(bb)
        return
    client = client or load_client(cfg, bb)
    e4 = cfg["experiments"]["exp4"]; kmax = e4["max_queries"]
    templates = load_jailbreak_templates(cfg)
    frac = cfg["evaluation"].get("calibration_fraction", 0.3)
    goals_df = pd.read_csv(RAW / "advbench_direct.csv")
    goals_df = goals_df[[split_of(f, frac) == "test" for f in goals_df["family"]]]

    def evaluate_batch(variants, goals):
        Hv = all_layer_states(client, list(variants), bs)[:, L14].astype(np.float32)
        sc = m14.decision_function(Hv)
        out = []
        for v, s in zip(variants, sc):
            dec = WrapperDecision.BLOCK if s >= t14 else WrapperDecision.ALLOW
            out.append((WrapperResult(wrapper="template_aware_probe", decision=dec, model_calls=1, risk_score=float(s), decided_by="probe"),
                        dec == WrapperDecision.ALLOW))
        return out

    logs = []
    for seed in (42, 43, 44):
        rng = random.Random(seed)
        sub = goals_df.sample(n=min(e4["subset_size"], len(goals_df)), random_state=seed)
        goals = sub["prompt"].astype(str).str.strip().str.rstrip(".").tolist()
        for cls in (NaiveAttacker, AdaptiveAttacker, ScoreAwareAttacker):
            path = out_dir / f"attack_seed{seed}_{cls.level}.csv"
            if path.exists():
                logs.append(pd.read_csv(path)); continue
            attackers = [cls(cfg, templates, client, random.Random(rng.random())) for _ in goals]
            steps = run_lockstep(attackers, goals, evaluate_batch, kmax, client, None, e4["paraphrase"], None,
                                 cache_key=(bb, seed))
            rec = [{"seed": seed, "attacker": cls.level, "prompt_id": f"{seed}-{i}", "prompt_text": g,
                    "attack_attempt": s.attack_attempt, "attack_variant": s.attack_variant,
                    "decision": s.decision, "evasion_success": s.evasion_success, "risk_score": s.risk_score}
                   for i, (g, st) in enumerate(zip(goals, steps)) for s in st]
            d = pd.DataFrame(rec); d.to_csv(path, index=False); logs.append(d)
    att = valid_evasions(pd.concat(logs, ignore_index=True), "lenient")
    res = []
    for (lev, seed), g in att.groupby(["attacker", "seed"]):
        first = g[g.evasion_success.astype(bool)].groupby("prompt_id").attack_attempt.min().reindex(g.prompt_id.unique())
        fs = first.to_numpy(dtype=float)
        res.append({"attacker": lev, "seed": seed, "asr_at_1": M.asr_at_k(fs, 1), "asr_at_10": M.asr_at_k(fs, kmax)})
    r = pd.DataFrame(res)
    summary = r.groupby("attacker")[["asr_at_1", "asr_at_10"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.insert(0, "backbone", bb)
    save_table(summary, cfg, f"exp15_attacks_{bb}")
    print(summary.round(3).to_string())
    release_client(bb)


if __name__ == "__main__":
    main()
