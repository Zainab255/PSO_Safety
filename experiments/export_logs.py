#!/usr/bin/env python3
"""Write the per-prompt logs of Table 3.

    outputs/pso_<backbone>_<dataset>.csv, outputs/results.csv, outputs/aggregate_stats.csv

Standard rows come from every cached analysis (main corpus, Exp. 7
benchmarks, Exp. 8 ambiguous set) with the full-pipeline decision at the
configured thresholds and P(unsafe | D_s) from the fitted Predictive Safety
Model. Where the PSO itself was run, its logged decisions are checked against
these. RQ4 adaptive-attack rows fill the attack_* columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import load_predictive_model, logger, run_dir, setup
from experiments.exp5_sensitivity import pipeline_decisions
from config import output_dir
from evaluation.logs import write_logs


def standard_rows(cfg, backbone: str, analysis: pd.DataFrame, model) -> pd.DataFrame:
    tau = cfg["pso"]["thresholds"]
    out = analysis.copy()
    out["pso_decision"] = pipeline_decisions(out, tau["tau_s"], tau["tau_b"])[0]
    out["backbone_model"] = backbone
    out["p_unsafe"] = model.predict_proba({f: out[f] for f in model.features}) if model else np.nan
    return out


def check_against_runs(analysis: pd.DataFrame, directory) -> None:
    for path in sorted(directory.glob("eval_seed*_pso.csv")):
        ev = pd.read_csv(path).set_index("prompt_id")["decision"]
        logged = analysis.set_index("prompt_id")["pso_decision"].reindex(ev.index)
        mismatches = int((ev != logged).sum())
        if mismatches:
            logger.warning("%s: %d PSO decisions differ from the logged signals.", path, mismatches)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    model = load_predictive_model(cfg)
    backbones = list(dict.fromkeys([args.backbone, *cfg["experiments"]["exp7"]["backbones"]]))
    frames = []
    for bb in backbones:
        root = run_dir(cfg, bb)
        for path in sorted(root.glob("**/analysis.csv")):
            rows = standard_rows(cfg, bb, pd.read_csv(path), model)
            check_against_runs(rows, path.parent)
            frames.append(rows)
        for path in sorted((root / "exp4").glob("attack_*_adaptive_pso.csv")):
            a = pd.read_csv(path)
            a["backbone_model"] = bb
            a["pso_decision"] = a["decision"]
            a["disagreement_score"] = a["entropy_score"]
            if model and set(model.features) <= set(a.columns):
                a["p_unsafe"] = model.predict_proba({f: a[f] for f in model.features})
            frames.append(a)
    if not frames:
        raise FileNotFoundError("No experiment outputs found; run the experiments first.")
    combined = pd.concat(frames, ignore_index=True)
    # The same prompt can appear in several analyses (e.g. the harmless set of Exp. 7); keep one row each.
    standard = combined[combined["attack_attempt"].isna()] if "attack_attempt" in combined else combined
    attack = combined.drop(index=standard.index)
    standard = standard.drop_duplicates(subset=["backbone_model", "dataset", "prompt_id"])
    written = write_logs([standard, attack], output_dir(cfg))
    for name, path in written.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
