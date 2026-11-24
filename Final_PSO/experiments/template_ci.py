#!/usr/bin/env python3
"""Family-bootstrap 95% intervals for the template-control (Exp. 13) and template-aware (Exp. 14) results.
Reads runs/<bb>/exp13/scores.csv and runs/<bb>/exp14/scores.csv; thresholds are re-derived exactly as in the
experiments (lambda rule on the respective calibration data). CPU only.
Output: tables/template_ci.csv"""
import numpy as np, pandas as pd
from benchmarks.loaders import split_of
from evaluation import metrics as M
from experiments.common import run_dir, save_table, setup
from experiments.exp12_baselines import youden_threshold

def boot(fn, groups, rng, B=1000):
    keys = [np.unique(g) for g in groups]
    reps = []
    for _ in range(B):
        picks = [rng.choice(k, len(k)) for k in keys]
        reps.append(fn(picks))
    return np.percentile(reps, [2.5, 97.5])

def main(argv=None):
    cfg, args = setup(__doc__, argv); ev = cfg["evaluation"]; frac = ev.get("calibration_fraction", 0.3)
    rng = np.random.default_rng(0); rows = []
    for bb in ["llama2", "llama3", "mistral", "qwen"]:
        for exp, col in (("exp13", "probe_cv_layer"), ("exp14", "probe_template_aware")):
            p = run_dir(cfg, bb, exp) / "scores.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p); s = d[col].to_numpy(); y = d.ground_truth.to_numpy(); st = d.set.to_numpy()
            fam = d.family.astype(str).to_numpy(); fsplit = np.array([split_of(f, frac) for f in fam])
            main_ = np.isin(st, ["advbench_framed", "xstest_safe", "xstest_unsafe"])
            cal = main_ & (d.split.to_numpy() == "calib")
            if exp == "exp14":
                cal = cal | (np.isin(st, ["benign_direct", "benign_framed", "benign_wild"]) & (fsplit == "calib"))
            t = youden_threshold(s[cal], y[cal], ev["calibration_lambda"]); flag = s >= t
            harm = (st == "advbench_framed") & (d.split.to_numpy() == "test")
            harmless = (st == "benign_framed") & ((fsplit == "test") if exp == "exp14" else np.ones(len(d), bool))
            xs = (st == "xstest_safe") & (d.split.to_numpy() == "test")
            idx = lambda m: {f: np.where(m & (fam == f))[0] for f in np.unique(fam[m])}
            H, Bn, X = idx(harm), idx(harmless), idx(xs)
            def cat(dct, keys): return np.concatenate([dct[k] for k in keys])
            quant = {
                "tpr_framed_harmful": (lambda p_: flag[cat(H, p_[0])].mean(), [list(H)]),
                "fpr_framed_harmless": (lambda p_: flag[cat(Bn, p_[0])].mean(), [list(Bn)]),
                "fpr_xstest_safe": (lambda p_: flag[cat(X, p_[0])].mean(), [list(X)]),
                "auc_framed_vs_framed": (lambda p_: M.auc(np.r_[np.ones(len(cat(H, p_[0]))), np.zeros(len(cat(Bn, p_[1])))],
                                                          np.r_[s[cat(H, p_[0])], s[cat(Bn, p_[1])]]), [list(H), list(Bn)]),
            }
            for name, (fn, keys) in quant.items():
                point = fn([k for k in keys])
                lo, hi = boot(fn, [np.array(k) for k in keys], rng)
                rows.append({"backbone": bb, "experiment": exp, "quantity": name, "value": point, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows); save_table(out, cfg, "template_ci"); print(out.round(3).to_string())

if __name__ == "__main__":
    main()
