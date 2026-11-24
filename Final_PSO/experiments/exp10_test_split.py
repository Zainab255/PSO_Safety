#!/usr/bin/env python3
"""Exp. 10 on the test split only (audit correction).

The original Exp. 10 interaction used every framed jailbreak and direct AdvBench goal (calibration + test
families, n = 500 each) but only the 173 harmless test prompts, because the 77 harmless calibration prompts
form the reference set of the baseline-adjusted score. This script recomputes the final-layer interaction
of Eq. (2) from the cached per-prompt distances (no model needed) on (i) the original sample, to check the
recomputation, and (ii) test-split families only, with a family bootstrap.

Output: tables/exp10_interaction_testsplit.csv
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from benchmarks.loaders import load_dataset
from experiments.common import run_dir, save_table, setup
from experiments.exp10_prefix_controls import GROUP_LABEL
from pipeline.runner import record_fields

CONTRASTS = [("framed_jailbreak", "benign"), ("framed_jailbreak", "ambiguous"), ("framed_jailbreak", "direct_xstest"),
             ("direct_advbench", "benign"), ("ambiguous", "benign")]


def boot_delta(S, N, ga, gb, fams, rng, B):
    """Delta = (mean_a S - mean_b S) - (mean_a N - mean_b N); bootstrap resamples families within each group."""
    def stat(ia, ib):
        return (S[ia].mean() - S[ib].mean()) - (N[ia].mean() - N[ib].mean())
    def by_fam(idx):
        f = pd.Series(idx).groupby(fams[idx]).apply(list)
        return list(f.values)
    fa, fb = by_fam(ga), by_fam(gb)
    reps = []
    for _ in range(B):
        ia = np.concatenate([fa[i] for i in rng.integers(0, len(fa), len(fa))])
        ib = np.concatenate([fb[i] for i in rng.integers(0, len(fb), len(fb))])
        reps.append(stat(ia, ib))
    return stat(ga, gb), np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def main(argv=None):
    cfg, args = setup(__doc__, argv)
    e10 = cfg["experiments"]["exp10"]; bb = args.backbone
    recs = [r for g in e10["groups"] for r in load_dataset(cfg, g)]
    d = run_dir(cfg, bb, "exp10")
    dist = {c: np.load(d / f"distances_{c}.npy") for c in ("safety", "neutral", "paraphrase")}
    idp = d / "distances_safety_ids.json"
    ids = json.loads(idp.read_text()) if idp.exists() else [r.prompt_id for r in recs[:len(dist["safety"])]]
    pos = {r.prompt_id: r for r in recs}
    meta = pd.DataFrame([record_fields(pos[i]) for i in ids])
    meta["group10"] = meta["dataset"].map(GROUP_LABEL)
    final = dist["safety"].shape[1] - 1
    sc = {c: v[:, final, :].mean(axis=-1) / 2.0 for c, v in dist.items()}
    fams = meta["family"].astype(str).to_numpy()
    benign_calib = meta.index[(meta.group10 == "benign") & (meta.split == "calib")].to_numpy()
    rows, rng = [], np.random.default_rng(0)
    for sample in ("original", "test_split"):
        keep = np.setdiff1d(np.arange(len(meta)), benign_calib)
        if sample == "test_split":
            # Ambiguous prompts are never used for calibration and have no split; keep all of them.
            keep = keep[((meta.split.to_numpy()[keep] == "test") | (meta.group10.to_numpy()[keep] == "ambiguous"))]
        m = meta.loc[keep]
        for cond in ("safety", "paraphrase"):
            for a, b in CONTRASTS:
                ga = m.index[m.group10 == a].to_numpy(); gb = m.index[m.group10 == b].to_numpy()
                if len(ga) == 0 or len(gb) == 0:
                    continue
                dlt, lo, hi = boot_delta(sc[cond], sc["neutral"], ga, gb, fams, rng, e10.get("bootstrap", 1000))
                rows.append({"backbone": bb, "sample": sample, "conditioning": cond, "contrast": f"{a} - {b}",
                             "n_a": len(ga), "n_b": len(gb), "delta": dlt, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    save_table(out, cfg, f"exp10_interaction_testsplit_{bb}")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
