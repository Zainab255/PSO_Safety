"""Per-prompt output log schema (Section IV-B-7, Table 3).

Rows are written to ``outputs/pso_<backbone>_<dataset>.csv`` and aggregated
into ``outputs/results.csv`` and ``outputs/aggregate_stats.csv``. The five
``attack_*``-related columns are filled only for RQ4 adaptive-attacker rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

TABLE3_COLUMNS = [
    "prompt_id", "dataset", "backbone_model", "prompt_category", "prompt_text", "pso_decision",
    "disagreement_score", "p_unsafe", "s_rb", "s_rj", "s_bj", "ground_truth",
    "attack_attempt", "attack_variant", "evasion_success", "cost_total", "entropy_score",
]
ATTACK_COLUMNS = ["attack_attempt", "attack_variant", "evasion_success", "cost_total", "entropy_score"]


def to_table3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TABLE3_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[TABLE3_COLUMNS]


def write_logs(frames: List[pd.DataFrame], output_dir: Path) -> Dict[str, Path]:
    """Write per-(backbone, dataset) CSVs, results.csv and aggregate_stats.csv."""
    all_rows = to_table3(pd.concat(frames, ignore_index=True))
    written: Dict[str, Path] = {}
    for (backbone, dataset), part in all_rows.groupby(["backbone_model", "dataset"], sort=True):
        path = output_dir / f"pso_{backbone}_{dataset}.csv"
        part.to_csv(path, index=False)
        written[f"{backbone}/{dataset}"] = path
    results = output_dir / "results.csv"
    all_rows.to_csv(results, index=False)
    written["results"] = results

    standard = all_rows[all_rows["attack_attempt"].isna()]
    agg = (
        standard.groupby(["backbone_model", "dataset", "prompt_category"])
        .agg(
            n=("prompt_id", "size"),
            disagreement_mean=("disagreement_score", "mean"),
            disagreement_std=("disagreement_score", "std"),
            s_rb_mean=("s_rb", "mean"),
            s_rj_mean=("s_rj", "mean"),
            s_bj_mean=("s_bj", "mean"),
            p_unsafe_mean=("p_unsafe", "mean"),
            allow=("pso_decision", lambda s: int((s == "ALLOW").sum())),
            requery=("pso_decision", lambda s: int((s == "REQUERY").sum())),
            block=("pso_decision", lambda s: int((s == "BLOCK").sum())),
        )
        .reset_index()
    )
    stats_path = output_dir / "aggregate_stats.csv"
    agg.to_csv(stats_path, index=False)
    written["aggregate_stats"] = stats_path
    return written
