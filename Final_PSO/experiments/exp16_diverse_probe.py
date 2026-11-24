#!/usr/bin/env python3
"""Experiment 16 -- diverse-data probe (exploratory; pre-registered in PREREGISTRATION_exp16.md).

1. Leave-one-source-out (LOSO): the bare-prompt probe is trained on every source group but one and
   evaluated on the held-out group (detection for harmful groups, flag rate for harmless groups).
2. Operator augmentation + re-attack (primary backbone, or any backbone with --attacks): the probe is
   trained on the calibration families of every group plus attack-operator variants of calibration-family
   AdvBench goals and, equally, of calibration-family Alpaca instructions; then attacked as in Exp. 15.

Outputs
    tables/exp16_loso_<bb>.csv, tables/exp16_augmented_indist_<bb>.csv, tables/exp16_attacks_<bb>.csv
    runs/<bb>/exp16/  (extra hidden states, per-query attack logs)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.attackers import AdaptiveAttacker, NaiveAttacker, ScoreAwareAttacker, run_lockstep
from benchmarks.loaders import load_dataset, load_jailbreak_templates, split_of
from core.decision import WrapperDecision
from evaluation import metrics as M
from experiments.common import load_client, release_client, run_dir, save_table, setup
from experiments.exp12_baselines import all_layer_states, fit_probe, youden_threshold
from experiments.exp13_template_control import build_frame, hidden_states
from experiments.exp4_robustness import valid_evasions
from pipeline.runner import record_fields
from wrappers.base import WrapperResult

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GROUP = {
    "advbench_framed": "H1_advbench", "advbench_wild": "H1_advbench", "advbench_direct": "H1_advbench",
    "xstest_unsafe": "H2_xstest_unsafe", "harmbench": "H3_harmbench",
    "jailbreakbench": "H4_jbb", "jbb_harmful_framed": "H4_jbb", "indep_pair": "H4_jbb",
    "indep_gcg_llama2": "H4_jbb", "indep_gcg_vicuna": "H4_jbb", "orbench_toxic": "H5_orbench_toxic",
    "xstest_safe": "B1_xstest_safe", "benign_direct": "B2_alpaca", "benign_framed": "B2_alpaca",
    "benign_wild": "B2_alpaca", "orbench_hard": "B3_orbench_hard", "jbb_benign_framed": "B4_jbb_benign",
}


def extra_records(cfg):
    """HarmBench, JailbreakBench (direct) and the Exp. 15 independent sets."""
    rows, texts = [], []
    for name in ("harmbench", "jailbreakbench"):
        for r in load_dataset(cfg, name):
            f = record_fields(r)
            rows.append({"prompt_id": f["prompt_id"], "family": f["family"], "set": name, "ground_truth": 1})
            texts.append(r.text)
    ind = pd.read_csv(RAW / "independent_sets.csv")
    for _, r in ind.iterrows():
        rows.append({"prompt_id": r["id"], "family": r["family"], "set": r["set"], "ground_truth": int(r["label"])})
        texts.append(str(r["prompt"]))
    return pd.DataFrame(rows), texts


def augmentation(cfg, frac, rng):
    """Attack-operator variants of calibration-family AdvBench goals and Alpaca instructions (3 each)."""
    e4 = cfg["experiments"]["exp4"]
    templates = load_jailbreak_templates(cfg)
    ph = cfg["datasets"]["jailbreak_templates"]["placeholder"]
    frames = {**e4.get("framings", {}), **e4["score_aware"].get("framings", {})}
    ops = ["template"] + list(frames)
    rows, texts = [], []
    for src, label, name in (("advbench_direct.csv", 1, "aug_harm"), ("benign_direct.csv", 0, "aug_benign")):
        d = pd.read_csv(RAW / src)
        d = d[[split_of(f, frac) == "calib" for f in d["family"]]]
        for _, r in d.iterrows():
            goal = str(r["prompt"]).strip().rstrip(".")
            for j in range(3):
                op = rng.choice(ops)
                text = rng.choice(templates).replace(ph, goal) if op == "template" else frames[op].replace("{prompt}", goal)
                rows.append({"prompt_id": f"{name}-{r['id']}-{j}", "family": r["family"], "set": name, "ground_truth": label})
                texts.append(text)
    return pd.DataFrame(rows), texts


def main(argv=None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--attacks", action="store_true")
    known, rest = pre.parse_known_args(argv)
    cfg, args = setup(__doc__, rest)
    bb, ev, e12 = args.backbone, cfg["evaluation"], cfg["experiments"]["exp12"]
    frac = ev.get("calibration_fraction", 0.3)
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))
    out_dir = run_dir(cfg, bb, "exp16"); out_dir.mkdir(parents=True, exist_ok=True)

    df, texts = build_frame(cfg)
    H = hidden_states(cfg, bb, texts, run_dir(cfg, bb, "exp13"), bs)
    ex, ex_texts = extra_records(cfg)
    au, au_texts = augmentation(cfg, frac, random.Random(0))
    extra_texts = ex_texts + au_texts
    cache = out_dir / "extra_states.npy"
    client = None
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(extra_texts):
        He = np.load(cache)
    else:
        client = load_client(cfg, bb)
        He = all_layer_states(client, extra_texts, bs)
        np.save(cache, He)
    D = pd.concat([df[["prompt_id", "family", "set", "ground_truth"]], ex, au], ignore_index=True)
    HA = np.concatenate([H, He], axis=0)
    D["group"] = D["set"].map(GROUP)
    D.loc[D["set"].str.startswith("aug_"), "group"] = "AUG"
    keep = D["group"].notna()              # every set outside the groups (none at present) is dropped
    D, HA = D[keep].reset_index(drop=True), HA[keep.to_numpy()]
    D["fsplit"] = [split_of(str(f), frac) for f in D["family"]]
    y = D["ground_truth"].to_numpy().astype(int); fam = D["family"].astype(str).to_numpy()
    grp = D["group"].to_numpy()
    real = grp != "AUG"

    # Layer chosen once: grouped CV on the pooled calibration families of all real groups.
    cal = real & (D["fsplit"].to_numpy() == "calib")
    best, best_cv = None, -1.0
    for layer in range(HA.shape[1]):
        _, _, cv = fit_probe(HA[cal, layer].astype(np.float32), y[cal], fam[cal], e12["probe_C"])
        if cv > best_cv:
            best, best_cv = layer, cv
    X = HA[:, best].astype(np.float32)

    # 1) Leave-one-source-out ------------------------------------------------------------
    rows = []
    for g in sorted(set(grp[real])):
        tr = real & (grp != g); te = grp == g
        m, c, _ = fit_probe(X[tr], y[tr], fam[tr], e12["probe_C"])
        s = m.decision_function(X); t = youden_threshold(s[tr], y[tr], ev["calibration_lambda"])
        flag = s >= t
        rows.append({"backbone": bb, "layer": best, "held_out": g, "subset": "all", "n": int(te.sum()),
                     "label": int(y[te][0]), "flag_rate": float(flag[te].mean())})
        for st in sorted(D.loc[te, "set"].unique()):
            sel = te & (D["set"].to_numpy() == st)
            rows.append({"backbone": bb, "layer": best, "held_out": g, "subset": st, "n": int(sel.sum()),
                         "label": int(y[sel][0]), "flag_rate": float(flag[sel].mean())})
    loso = pd.DataFrame(rows)
    save_table(loso, cfg, f"exp16_loso_{bb}")
    print(loso.round(3).to_string())

    # 2) Augmented diverse probe: in-distribution check -----------------------------------
    trn = (real & (D["fsplit"].to_numpy() == "calib")) | (grp == "AUG")
    tst = real & (D["fsplit"].to_numpy() == "test")
    m_aug, c_aug, _ = fit_probe(X[trn], y[trn], fam[trn], e12["probe_C"])
    s = m_aug.decision_function(X); t_aug = youden_threshold(s[trn], y[trn], ev["calibration_lambda"])
    rows = []
    for st in sorted(D.loc[tst, "set"].unique()):
        sel = tst & (D["set"].to_numpy() == st)
        rows.append({"backbone": bb, "set": st, "n": int(sel.sum()), "label": int(y[sel][0]),
                     "flag_rate": float((s[sel] >= t_aug).mean())})
    rows.append({"backbone": bb, "set": "AUC all test families", "n": int(tst.sum()), "label": -1,
                 "flag_rate": float(M.auc(y[tst], s[tst]))})
    ind = pd.DataFrame(rows)
    save_table(ind, cfg, f"exp16_augmented_indist_{bb}")
    print(ind.round(3).to_string())

    # 3) Re-attack (primary backbone by default) ------------------------------------------
    if not (known.attacks or bb == cfg["backbones"]["primary"]):
        if client is not None:
            release_client(bb)
        return
    client = client or load_client(cfg, bb)
    e4 = cfg["experiments"]["exp4"]; kmax = e4["max_queries"]
    templates = load_jailbreak_templates(cfg)
    goals_df = pd.read_csv(RAW / "advbench_direct.csv")
    goals_df = goals_df[[split_of(f, frac) == "test" for f in goals_df["family"]]]

    def evaluate_batch(variants, goals):
        sc = m_aug.decision_function(all_layer_states(client, list(variants), bs)[:, best].astype(np.float32))
        return [(WrapperResult(wrapper="diverse_probe", decision=(WrapperDecision.BLOCK if v >= t_aug else WrapperDecision.ALLOW),
                               model_calls=1, risk_score=float(v), decided_by="probe"), bool(v < t_aug)) for v in sc]

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
                                 cache_key=(bb, seed, "exp16"))
            d = pd.DataFrame([{"seed": seed, "attacker": cls.level, "prompt_id": f"{seed}-{i}", "prompt_text": g,
                               "attack_attempt": s_.attack_attempt, "attack_variant": s_.attack_variant,
                               "decision": s_.decision, "evasion_success": s_.evasion_success, "risk_score": s_.risk_score}
                              for i, (g, st) in enumerate(zip(goals, steps)) for s_ in st])
            d.to_csv(path, index=False); logs.append(d)
    raw = pd.concat(logs, ignore_index=True)
    res = []
    for mode in ("lenient", "strict"):
        att = valid_evasions(raw, mode)
        for (lev, seed), g in att.groupby(["attacker", "seed"]):
            first = g[g.evasion_success.astype(bool)].groupby("prompt_id").attack_attempt.min().reindex(g.prompt_id.unique())
            fs = first.to_numpy(dtype=float)
            res.append({"mode": mode, "attacker": lev, "seed": seed, "asr_at_1": M.asr_at_k(fs, 1), "asr_at_10": M.asr_at_k(fs, kmax)})
    r = pd.DataFrame(res).groupby(["mode", "attacker"])[["asr_at_1", "asr_at_10"]].agg(["mean", "std"]).reset_index()
    r.columns = ["_".join(c).strip("_") for c in r.columns]
    r.insert(0, "backbone", bb)
    save_table(r, cfg, f"exp16_attacks_{bb}")
    print(r.round(3).to_string())
    release_client(bb)


if __name__ == "__main__":
    main()
