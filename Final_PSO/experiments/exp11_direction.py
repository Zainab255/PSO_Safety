#!/usr/bin/env python3
"""Experiment 11 — exploratory, direction-calibrated detector (CPU only, from cached signals).

The pre-specified PSO flags HIGH disagreement (Hypothesis 1). This exploratory analysis lets
the calibration split choose the direction of the score as well as the thresholds:

    direction = +1 if AUC_calib(D~) >= 0.5 else -1         (calibration split only)
    score'    = direction * D~                               (then Eq. 17 thresholds on score')

Everything is frozen on the calibration split and evaluated on the disjoint test split of the
main corpus and of each Exp. 7 benchmark. Because the direction is chosen after seeing that
the pre-specified direction fails, results are reported as exploratory, alongside the
pre-specified ones. No new model calls are made.

Outputs
    tables/exp11_direction_calibrated.csv   per backbone x corpus: pre-specified vs direction-calibrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import run_dir, save_table, setup
from experiments.exp5_sensitivity import classification, pipeline_decisions, uor
from core.calibration import calibrate, load_thresholds
from evaluation import metrics as M


def flipped(df: pd.DataFrame, direction: int) -> pd.DataFrame:
    out = df.copy()
    out["disagreement_score"] = direction * out["disagreement_score"]
    return out


def evaluate(cfg, df, tau_s, tau_b, responses):
    d, by = pipeline_decisions(df, tau_s, tau_b)
    c = classification(cfg, df, d, by)
    u = [uor(df, d, responses[responses["seed"] == s]) for s in sorted(responses["seed"].unique())] \
        if responses is not None and len(responses) else []
    c["uor"] = float(np.mean(u)) if u else np.nan
    return c


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    ev = cfg["evaluation"]
    backbones = list(dict.fromkeys([args.backbone, *cfg["experiments"]["exp7"]["backbones"]]))
    rows = []
    for bb in backbones:
        main_path = run_dir(cfg, bb, "main") / "analysis.csv"
        if not main_path.exists():
            continue
        main = pd.read_csv(main_path)
        cal = main[main["split"] == "calib"]
        auc_cal = M.auc(cal["ground_truth"].to_numpy(), cal["disagreement_score"].to_numpy())
        direction = 1 if auc_cal >= 0.5 else -1
        pre = load_thresholds(cfg, bb) or calibrate(main, ev["calibration_lambda"], ev["calibration_grid_size"])
        main_f = flipped(main, direction)
        th = calibrate(main_f, ev["calibration_lambda"], ev["calibration_grid_size"])

        corpora = {"main": (main, sorted(run_dir(cfg, bb, "main").glob("responses_seed*.csv")))}
        for bench_dir in sorted((run_dir(cfg, bb, "exp7")).glob("*")) if (run_dir(cfg, bb, "exp7")).exists() else []:
            if (bench_dir / "analysis.csv").exists():
                corpora[bench_dir.name] = (pd.read_csv(bench_dir / "analysis.csv"), [])
        for name, (df, resp_paths) in corpora.items():
            done = [p for p in resp_paths if "partial" not in p.name]
            resp = pd.concat([pd.read_csv(p) for p in done], ignore_index=True) if done else None
            a = evaluate(cfg, df, pre["tau_s"], pre["tau_b"], resp)
            b = evaluate(cfg, flipped(df, direction), th["tau_s"], th["tau_b"], resp)
            rows.append({
                "backbone": bb, "corpus": name, "calib_auc_of_D": auc_cal, "chosen_direction": direction,
                **{f"prespecified_{k}": v for k, v in a.items() if k != "delta_disagreement"},
                **{f"direction_calibrated_{k}": v for k, v in b.items() if k != "delta_disagreement"},
                "prespecified_tau_s": pre["tau_s"], "calibrated_tau_s": th["tau_s"], "calibrated_tau_b": th["tau_b"],
            })
    save_table(pd.DataFrame(rows), cfg, "exp11_direction_calibrated")


if __name__ == "__main__":
    main()
