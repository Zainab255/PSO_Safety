#!/usr/bin/env python3
"""Experiment 13 -- template control for the single-view probe (pre-specified; see PREREGISTRATION_exp13.md).

Question: does the bare-prompt linear probe detect harmful requests, or only jailbreak templates?
The probe is fitted exactly as in Exp. 12 (main-corpus calibration split, layer and C by family-grouped CV)
and then applied, with its calibrated threshold, to harmless requests wrapped in the same templates:

    benign_direct      500 Alpaca instructions (harmless), unframed
    benign_framed      the same 500 in the 12 framing templates (same framing mix as AdvBench)
    benign_wild        the same 500 in the 42 in-the-wild JailbreakHub templates
    jbb_benign_framed  100 JailbreakBench benign behaviours (topic-matched to the harmful ones), framed
    jbb_harmful_framed the 100 harmful JailbreakBench behaviours with the identical template assignment

Outputs
    tables/exp13_template_control_<bb>.csv   flag rate per set, and AUC for each harmful/harmless pair
    runs/<bb>/exp13/scores.csv              per-prompt probe scores for every set
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.loaders import load_corpus, load_dataset
from evaluation import metrics as M
from experiments.common import load_client, run_dir, save_table, setup
from experiments.exp12_baselines import all_layer_states, fit_probe, youden_threshold
from pipeline.runner import record_fields

logger = logging.getLogger("experiments.exp13")

SETS = ["benign_direct", "benign_framed", "benign_wild", "jbb_benign_framed", "jbb_harmful_framed"]
# (harmful set, harmless set): the template-matched comparisons named in the pre-registration.
PAIRS = [("advbench_framed_test", "benign_framed"),      # primary
         ("advbench_wild", "benign_wild"),
         ("advbench_direct", "benign_direct"),
         ("jbb_harmful_framed", "jbb_benign_framed")]


def build_frame(cfg):
    """Main corpus + transfer jailbreak sets + template-control sets, in a fixed order."""
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    main_recs = load_corpus(cfg, cfg["evaluation"]["corpus"])
    df = pd.DataFrame([record_fields(r) for r in main_recs])
    df["set"] = np.where(df["dataset"] == "advbench", "advbench_framed", df["dataset"])
    texts = [r.text for r in main_recs]
    for name, label in (("advbench_wild", 1), ("advbench_direct", 1)):
        recs = load_dataset(cfg, name)
        d = pd.DataFrame([record_fields(r) for r in recs])
        d["set"], d["ground_truth"] = name, label
        df, texts = pd.concat([df, d], ignore_index=True), texts + [r.text for r in recs]
    for name in SETS:
        d = pd.read_csv(raw / f"{name}.csv")
        d = pd.DataFrame({"prompt_id": [f"{name}-{i}" for i in d["id"]], "family": d["family"],
                          "prompt_text": d["prompt"], "set": name,
                          "ground_truth": int(name.startswith("jbb_harmful")), "split": "eval"})
        df, texts = pd.concat([df, d], ignore_index=True), texts + d["prompt_text"].tolist()
    return df.reset_index(drop=True), texts


def hidden_states(cfg, bb, texts, out_dir, bs):
    """All-layer last-token states of the bare prompts (cached in out_dir)."""
    feats = out_dir / "single_view_states.npy"
    if feats.exists():
        try:
            if np.load(feats, mmap_mode="r").shape[0] == len(texts):
                return np.load(feats)
        except (ValueError, OSError) as exc:  # truncated or corrupt cache: recompute
            logger.warning("unreadable cache %s (%s); recomputing", feats, exc)
    client = load_client(cfg, bb)
    H = all_layer_states(client, texts, bs)
    np.save(feats, H)
    return H


def select(df, test):
    sel = {k: (df["set"] == k).to_numpy() for k in df["set"].unique()}
    sel["advbench_framed_test"] = sel["advbench_framed"] & test
    sel["xstest_safe_test"] = sel["xstest_safe"] & test
    return sel


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    bb, ev, e12 = args.backbone, cfg["evaluation"], cfg["experiments"]["exp12"]
    out_dir = run_dir(cfg, bb, "exp13")
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))

    df, texts = build_frame(cfg)
    H = hidden_states(cfg, bb, texts, out_dir, bs)

    in_main = df["set"].isin(["advbench_framed", "xstest_safe", "xstest_unsafe"]).to_numpy()
    calib = in_main & (df["split"] == "calib").to_numpy()
    test = in_main & (df["split"] == "test").to_numpy()
    y = df["ground_truth"].to_numpy().astype(int)
    groups = df["family"].astype(str).to_numpy()

    # Layer by family-grouped CV on the calibration split (as Exp. 12), plus the final layer.
    best_layer, best_cv = None, -1.0
    for layer in range(H.shape[1]):
        _, _, cv = fit_probe(H[calib, layer].astype(np.float32), y[calib], groups[calib], e12["probe_C"])
        if cv > best_cv:
            best_layer, best_cv = layer, cv
    rows = []
    sel = select(df, test)
    for name, layer in (("probe_cv_layer", best_layer), ("probe_final_layer", H.shape[1] - 1)):
        X = H[:, layer].astype(np.float32)
        m, c, cv = fit_probe(X[calib], y[calib], groups[calib], e12["probe_C"])
        s = m.decision_function(X)
        t = youden_threshold(s[calib], y[calib], ev["calibration_lambda"])
        df[name] = s
        flag = s >= t
        for k in ["xstest_safe_test", "advbench_framed_test", "advbench_wild", "advbench_direct"] + SETS:
            rows.append({"backbone": bb, "probe": name, "layer": layer, "C": c, "kind": "flag_rate", "set": k,
                         "n": int(sel[k].sum()), "value": float(flag[sel[k]].mean())})
        for hk, bk in PAIRS + [("advbench_framed_test", "xstest_safe_test")]:
            yy = np.r_[np.ones(sel[hk].sum()), np.zeros(sel[bk].sum())]
            rows.append({"backbone": bb, "probe": name, "layer": layer, "C": c, "kind": "auc",
                         "set": f"{hk} vs {bk}", "n": int(len(yy)),
                         "value": float(M.auc(yy, np.r_[s[sel[hk]], s[sel[bk]]]))})
        logger.info("%s (layer %d): test AUC %.3f; benign_framed flag rate %.3f", name, layer,
                    M.auc(y[test], s[test]), float(flag[sel["benign_framed"]].mean()))
    df.drop(columns=["prompt_text"], errors="ignore").to_csv(out_dir / "scores.csv", index=False)
    save_table(pd.DataFrame(rows), cfg, f"exp13_template_control_{bb}")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
