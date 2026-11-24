#!/usr/bin/env python3
"""Experiment 14 -- template-aware probe (EXPLORATORY; designed after the Exp. 13 result was known).

Exp. 13 showed that the bare-prompt probe ranks harmful above harmless requests inside identical templates,
but flags 31-43% of templated harmless requests because its harmless calibration prompts carry no template.
Here the probe is refitted with templated harmless prompts in its calibration data and tested on held-out
families. The criterion below was fixed before this experiment was run (PREREGISTRATION_exp14.md).

Calibration = main-corpus calibration split
            + the calibration families (30 %, same hash rule) of benign_direct, benign_framed, benign_wild.
Test        = main-corpus test split, the test families of the three Alpaca sets, advbench_wild and
              advbench_direct test families, and ALL JailbreakBench pairs (never used for fitting).
Layer and C are chosen by family-grouped CV on the calibration data; the threshold by the lambda rule.

Outputs
    tables/exp14_template_aware_probe_<bb>.csv
    runs/<bb>/exp14/scores.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from benchmarks.loaders import split_of
from evaluation import metrics as M
from experiments.common import run_dir, save_table, setup
from experiments.exp12_baselines import fit_probe, youden_threshold
from experiments.exp13_template_control import build_frame, hidden_states

logger = logging.getLogger("experiments.exp14")
ALPACA = ["benign_direct", "benign_framed", "benign_wild"]


def fit_template_aware(cfg, df, H):
    """Fit the template-aware probe exactly as in Exp. 14; returns (model, layer, C, threshold, calib mask)."""
    ev, e12 = cfg["evaluation"], cfg["experiments"]["exp12"]
    frac = ev.get("calibration_fraction", 0.3)
    st = df["set"]
    fam_split = df["family"].astype(str).map(lambda f: split_of(f, frac))
    in_main = st.isin(["advbench_framed", "xstest_safe", "xstest_unsafe"])
    calib = ((in_main & (df["split"] == "calib")) | (st.isin(ALPACA) & (fam_split == "calib"))).to_numpy()
    y = df["ground_truth"].to_numpy().astype(int)
    groups = df["family"].astype(str).to_numpy()
    best_layer, best_cv = None, -1.0
    for layer in range(H.shape[1]):
        _, _, cv = fit_probe(H[calib, layer].astype(np.float32), y[calib], groups[calib], e12["probe_C"])
        if cv > best_cv:
            best_layer, best_cv = layer, cv
    X = H[:, best_layer].astype(np.float32)
    m, c, cv = fit_probe(X[calib], y[calib], groups[calib], e12["probe_C"])
    t = youden_threshold(m.decision_function(X)[calib], y[calib], ev["calibration_lambda"])
    return m, best_layer, c, t, calib


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    bb, ev, e12 = args.backbone, cfg["evaluation"], cfg["experiments"]["exp12"]
    frac = ev.get("calibration_fraction", 0.3)
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))
    df, texts = build_frame(cfg)
    H = hidden_states(cfg, bb, texts, run_dir(cfg, bb, "exp13"), bs)   # same states as Exp. 13
    out_dir = run_dir(cfg, bb, "exp14")
    out_dir.mkdir(parents=True, exist_ok=True)

    st = df["set"]
    fam_split = df["family"].astype(str).map(lambda f: split_of(f, frac))
    in_main = st.isin(["advbench_framed", "xstest_safe", "xstest_unsafe"])
    y = df["ground_truth"].to_numpy().astype(int)
    m, best_layer, c, t, calib = fit_template_aware(cfg, df, H)
    s = m.decision_function(H[:, best_layer].astype(np.float32))
    flag = s >= t
    df["probe_template_aware"] = s

    test_fam = (fam_split == "test").to_numpy()
    sets = {
        "xstest_safe_test": (st == "xstest_safe").to_numpy() & (df["split"] == "test").to_numpy(),
        "xstest_unsafe_test": (st == "xstest_unsafe").to_numpy() & (df["split"] == "test").to_numpy(),
        "advbench_framed_test": (st == "advbench_framed").to_numpy() & (df["split"] == "test").to_numpy(),
        "advbench_wild_test": (st == "advbench_wild").to_numpy() & test_fam,
        "advbench_direct_test": (st == "advbench_direct").to_numpy() & test_fam,
        "benign_framed_test": (st == "benign_framed").to_numpy() & test_fam,
        "benign_wild_test": (st == "benign_wild").to_numpy() & test_fam,
        "benign_direct_test": (st == "benign_direct").to_numpy() & test_fam,
        "jbb_benign_framed": (st == "jbb_benign_framed").to_numpy(),
        "jbb_harmful_framed": (st == "jbb_harmful_framed").to_numpy(),
    }
    rows = [{"backbone": bb, "layer": best_layer, "C": c, "kind": "flag_rate", "set": k, "n": int(v.sum()),
             "value": float(flag[v].mean())} for k, v in sets.items()]
    main_test = in_main.to_numpy() & (df["split"] == "test").to_numpy()
    rows.append({"backbone": bb, "layer": best_layer, "C": c, "kind": "auc", "set": "main corpus test",
                 "n": int(main_test.sum()), "value": float(M.auc(y[main_test], s[main_test]))})
    for hk, bk in [("advbench_framed_test", "benign_framed_test"), ("advbench_wild_test", "benign_wild_test"),
                   ("advbench_direct_test", "benign_direct_test"), ("jbb_harmful_framed", "jbb_benign_framed")]:
        yy = np.r_[np.ones(sets[hk].sum()), np.zeros(sets[bk].sum())]
        rows.append({"backbone": bb, "layer": best_layer, "C": c, "kind": "auc", "set": f"{hk} vs {bk}",
                     "n": int(len(yy)), "value": float(M.auc(yy, np.r_[s[sets[hk]], s[sets[bk]]]))})
    out = pd.DataFrame(rows)
    df.drop(columns=["prompt_text"], errors="ignore").to_csv(out_dir / "scores.csv", index=False)
    save_table(out, cfg, f"exp14_template_aware_probe_{bb}")
    print(out.to_string())


if __name__ == "__main__":
    main()
