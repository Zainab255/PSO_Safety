#!/usr/bin/env python3
"""Probe OR Llama Guard (Experiment 17, offline part) with family-bootstrap intervals.

Reads the per-prompt scores of the diverse probe written by exp17_coverage.py and the Llama Guard 3-1B
prompt scores written by exp12_baselines.py / exp13_llama_guard.py / exp15_independent_and_adaptive.py.
The Llama Guard threshold is the one fixed on the main calibration split (exp12_baselines_llama2.csv);
nothing is refitted here.

Usage
    python tools/combination_ci.py <outputs_dir> <out_tables_dir> [--resamples 2000]
Writes
    <out_tables_dir>/exp17_combination_ci.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BACKBONES = ("llama2", "llama3", "mistral", "qwen")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--resamples", type=int, default=2000)
    a = ap.parse_args()
    runs = a.outputs / "runs"
    lg_files = [runs / "llama2" / "exp12" / "scores.csv",
                runs / "llama2" / "exp13" / "llama_guard_template_scores.csv",
                runs / "llama2" / "exp13" / "llama_guard_independent_scores.csv"]
    lg = pd.concat([pd.read_csv(f)[["prompt_id", "llama_guard"]] for f in lg_files if f.exists()])
    lg = lg.drop_duplicates("prompt_id")
    base = pd.read_csv(a.outputs / "tables" / "exp12_baselines_llama2.csv")
    tau = float(base.loc[base.method == "llama_guard", "threshold"].iloc[0])

    out = []
    for bb in BACKBONES:
        f = runs / bb / "exp17" / "scores_main.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        t = s[(s.fsplit == "test") & (s.group != "AUG")].merge(lg, on="prompt_id", how="left")
        t["lg"] = np.where(t.llama_guard.isna(), np.nan, (t.llama_guard >= tau).astype(float))
        t["both"] = np.maximum(t.aug_flag, t.lg)
        rng = np.random.default_rng(0)
        for st, g in t.groupby("set"):
            g = g.dropna(subset=["lg"])
            fam = g.groupby("family").agg(n=("aug_flag", "size"), p=("aug_flag", "sum"), b=("both", "sum"))
            idx = rng.integers(0, len(fam), size=(a.resamples, len(fam)))
            den = fam["n"].to_numpy()[idx].sum(1)

            def ci(col):
                return np.percentile(fam[col].to_numpy()[idx].sum(1) / den, [2.5, 97.5])

            r = {"backbone": bb, "set": st, "label": int(g.ground_truth.iloc[0]), "n": len(g),
                 "probe": g.aug_flag.mean(), "llama_guard": g.lg.mean(), "probe_or_lg": g.both.mean()}
            r["probe_lo"], r["probe_hi"] = ci("p")
            r["or_lo"], r["or_hi"] = ci("b")
            out.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(a.out / "exp17_combination_ci.csv", index=False)
    print(f"wrote {a.out / 'exp17_combination_ci.csv'} ({len(out)} rows, Llama Guard threshold {tau:.4f})")


if __name__ == "__main__":
    main()
