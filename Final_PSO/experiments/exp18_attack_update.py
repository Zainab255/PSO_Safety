#!/usr/bin/env python3
"""Experiment 18b -- updating the probe from observed evasions (pre-registered in PREREGISTRATION_exp18.md).

Round 0 probe: trained on the calibration families of all source groups (no operator augmentation).
Each round:
  1. harvest -- the adaptive and score-aware attackers attack calibration-family AdvBench goals; up to
     N_ADD strict evasions (goal text kept verbatim) are taken, one per goal, and each gets a harmless twin
     in which the goal is replaced by a calibration-family Alpaca instruction (same wrapper);
  2. update  -- the evasions and their twins are added to the training data and the probe is refitted;
  3. test    -- naive, adaptive and score-aware attackers attack test-family goals (seeds 42, 43), and flag
     rates on every test set are recorded.
Test goals never enter training; harvest and test goals come from disjoint families.

Outputs
    tables/exp18_update_attacks_<bb>.csv, tables/exp18_update_flags_<bb>.csv
    runs/<bb>/exp18/update_round<r>_*.csv
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.attackers import AdaptiveAttacker, NaiveAttacker, ScoreAwareAttacker, run_lockstep
from benchmarks.loaders import load_jailbreak_templates, split_of
from core.decision import WrapperDecision
from evaluation import metrics as M
from experiments.common import load_client, release_client, run_dir, save_table, setup
from experiments.exp12_baselines import fit_probe, youden_threshold
from experiments.exp18_common import diverse_data
from experiments.exp4_robustness import valid_evasions
from wrappers.base import WrapperResult

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ROUNDS = 3              # rounds 0, 1, 2
N_ADD = 20              # evasions added per round
N_HARVEST = 50          # calibration goals attacked per round
TEST_SEEDS = (42, 43)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    bb, ev, e12, e4 = args.backbone, cfg["evaluation"], cfg["experiments"]["exp12"], cfg["experiments"]["exp4"]
    frac, lam, kmax = ev.get("calibration_fraction", 0.3), ev["calibration_lambda"], e4["max_queries"]
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))
    out_dir = run_dir(cfg, bb, "exp18"); out_dir.mkdir(parents=True, exist_ok=True)

    D, X, layer, texts, enc = diverse_data(cfg, bb, bs)
    y = D["ground_truth"].to_numpy().astype(int); fam = D["family"].astype(str).to_numpy()
    grp, fs, sets = D["group"].to_numpy(), D["fsplit"].to_numpy(), D["set"].to_numpy()
    tr0 = (grp != "AUG") & (fs == "calib")
    tst = (grp != "AUG") & (fs == "test")

    templates = load_jailbreak_templates(cfg)
    adv = pd.read_csv(RAW / "advbench_direct.csv")
    adv_split = np.array([split_of(f, frac) for f in adv["family"]])
    test_goals = adv[adv_split == "test"]
    cal_goals = adv[adv_split == "calib"]
    alp = pd.read_csv(RAW / "benign_direct.csv")
    alp = alp[[split_of(f, frac) == "calib" for f in alp["family"]]]
    alp_goals = alp["prompt"].astype(str).str.strip().str.rstrip(".").tolist()
    norm = lambda s: str(s).strip().rstrip(".")

    enc.client = enc.client or load_client(cfg, bb)
    client = enc.client

    added_X, added_y, added_f, added_txt = [], [], [], []
    att_rows, flag_rows = [], []
    for r in range(ROUNDS):
        # ---- fit the round-r probe ----
        Xtr = np.concatenate([X[tr0]] + ([np.stack(added_X)] if added_X else []))
        ytr = np.concatenate([y[tr0], np.array(added_y, int)])
        ftr = np.concatenate([fam[tr0], np.array(added_f, str)])
        m, c, _ = fit_probe(Xtr, ytr, ftr, e12["probe_C"])
        t = youden_threshold(m.decision_function(Xtr), ytr, lam)
        s = m.decision_function(X)
        for st in sorted(set(sets[tst])):
            sel = tst & (sets == st)
            flag_rows.append({"backbone": bb, "round": r, "n_added_harmful": int(sum(added_y)), "set": st,
                              "n": int(sel.sum()), "label": int(y[sel][0]), "flag_rate": float((s[sel] >= t).mean())})
        flag_rows.append({"backbone": bb, "round": r, "n_added_harmful": int(sum(added_y)), "set": "AUC all test families",
                          "n": int(tst.sum()), "label": -1, "flag_rate": float(M.auc(y[tst], s[tst]))})

        def evaluate_batch(variants, goals, m=m, t=t):
            sc = m.decision_function(enc(list(variants)))
            return [(WrapperResult(wrapper="updated_probe", decision=(WrapperDecision.BLOCK if v >= t else WrapperDecision.ALLOW),
                                   model_calls=1, risk_score=float(v), decided_by="probe"), bool(v < t)) for v in sc]

        def attack(goals, seed, classes, tag):
            logs = []
            rng = random.Random(seed)
            for cls in classes:
                path = out_dir / f"update_round{r}_{tag}_seed{seed}_{cls.level}.csv"
                if path.exists():
                    logs.append(pd.read_csv(path)); continue
                attackers = [cls(cfg, templates, client, random.Random(rng.random())) for _ in goals]
                steps = run_lockstep(attackers, goals, evaluate_batch, kmax, client, None, e4["paraphrase"], None,
                                     cache_key=(bb, seed, tag))
                d = pd.DataFrame([{"round": r, "seed": seed, "attacker": cls.level, "prompt_id": f"{seed}-{i}", "prompt_text": g,
                                   "attack_attempt": s_.attack_attempt, "attack_variant": s_.attack_variant,
                                   "decision": s_.decision, "evasion_success": s_.evasion_success, "risk_score": s_.risk_score}
                                  for i, (g, st) in enumerate(zip(goals, steps)) for s_ in st])
                d.to_csv(path, index=False); logs.append(d)
            return pd.concat(logs, ignore_index=True)

        # ---- test attacks ----
        for seed in TEST_SEEDS:
            sub = test_goals.sample(n=min(e4["subset_size"], len(test_goals)), random_state=seed)
            raw = attack([norm(g) for g in sub["prompt"]], seed, (NaiveAttacker, AdaptiveAttacker, ScoreAwareAttacker), "test")
            for mode in ("lenient", "strict"):
                a = valid_evasions(raw, mode)
                for lev, g in a.groupby("attacker"):
                    first = g[g.evasion_success.astype(bool)].groupby("prompt_id").attack_attempt.min().reindex(g.prompt_id.unique())
                    fs_ = first.to_numpy(dtype=float)
                    att_rows.append({"backbone": bb, "round": r, "n_added_harmful": int(sum(added_y)), "mode": mode,
                                     "attacker": lev, "seed": seed, "asr_at_1": M.asr_at_k(fs_, 1), "asr_at_10": M.asr_at_k(fs_, kmax)})
        _df = pd.DataFrame(att_rows); print(_df[_df["round"] == r].groupby(["mode", "attacker"]).asr_at_10.mean().round(3).to_string(), flush=True)
        if r == ROUNDS - 1:
            break

        # ---- harvest evasions on calibration goals ----
        hseed = 1000 + r
        sub = cal_goals.sample(n=min(N_HARVEST, len(cal_goals)), random_state=hseed)
        fam_of = dict(zip(sub["prompt"].map(norm), sub["family"]))
        raw = attack(list(fam_of), hseed, (AdaptiveAttacker, ScoreAwareAttacker), "harvest")
        a = valid_evasions(raw, "strict")
        a = a[a.evasion_success.astype(bool)].sort_values(["attack_attempt"]).drop_duplicates("prompt_text")
        a = a.sample(n=min(N_ADD, len(a)), random_state=hseed)
        hrng = random.Random(hseed)
        new = []
        for _, row in a.iterrows():
            goal, var = row["prompt_text"], str(row["attack_variant"])
            twin = re.sub(re.escape(goal), lambda _m: hrng.choice(alp_goals), var, flags=re.I)
            new += [(var, 1, fam_of[goal]), (twin, 0, f"twin-{fam_of[goal]}")]
        if new:
            feats = enc([n[0] for n in new])
            for (txt, lab, f), v in zip(new, feats):
                added_X.append(v); added_y.append(lab); added_f.append(f); added_txt.append((r + 1, lab, f, txt))
        pd.DataFrame(added_txt, columns=["round_added", "label", "family", "prompt_text"]).to_csv(out_dir / "update_added_prompts.csv", index=False)
        print(f"[{bb}] round {r}: added {len(new) // 2} evasions + twins", flush=True)

    res = pd.DataFrame(att_rows)
    agg = res.groupby(["round", "n_added_harmful", "mode", "attacker"])[["asr_at_1", "asr_at_10"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.insert(0, "backbone", bb)
    save_table(agg, cfg, f"exp18_update_attacks_{bb}")
    save_table(pd.DataFrame(flag_rows), cfg, f"exp18_update_flags_{bb}")
    release_client(bb)
    print(f"[{bb}] exp18 update loop complete", flush=True)


if __name__ == "__main__":
    main()
