"""Threshold selection on the held-out calibration split (Eq. 17).

The fixed defaults in ``config.yaml`` (tau_s=0.35, tau_b=0.65) assume a
particular score scale, but the scale of D~ depends on the backbone, layer,
pooling and persona wording. When ``evaluation.calibrate_thresholds`` is true,
thresholds are therefore chosen on the calibration split only and frozen before
any test-split metric is computed:

    tau_s  = argmax_t  lam * TPR(t) - (1 - lam) * FPR(t)        (flagged = BLOCK or REQUERY)
    tau_b  = argmax_t  TPR_block(t)  s.t.  FPR_block(t) <= max_block_fpr,  t > tau_s
    EV     = argmax_t  lam * TPR(t) - (1 - lam) * FPR(t)        (embedding-variance baseline)

Phase 1 blocks are part of the pipeline in both rates. Candidate thresholds are
quantiles of the calibration-split scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import output_dir

logger = logging.getLogger(__name__)

MAX_BLOCK_FPR = 0.02


def _rates(y: np.ndarray, flagged: np.ndarray):
    pos, neg = y == 1, y == 0
    tpr = float(flagged[pos].mean()) if pos.any() else np.nan
    fpr = float(flagged[neg].mean()) if neg.any() else np.nan
    return tpr, fpr


def _candidates(scores: np.ndarray, n: int) -> np.ndarray:
    scores = scores[np.isfinite(scores)]
    q = np.quantile(scores, np.linspace(0.01, 0.99, n))
    return np.unique(np.round(q, 6))


def calibrate(analysis: pd.DataFrame, lam: float, grid_size: int) -> Dict[str, float]:
    """Return calibrated thresholds from the calibration rows of a main-corpus analysis frame."""
    cal = analysis[analysis["split"] == "calib"] if "split" in analysis else analysis
    y = cal["ground_truth"].to_numpy().astype(int)
    p1 = (cal["keyword_match"].astype(bool) | cal["pattern_match"].astype(bool)).to_numpy()
    d = cal["disagreement_score"].to_numpy(dtype=float)

    best_s, best_obj = np.nan, -np.inf
    for t in _candidates(d, grid_size):
        tpr, fpr = _rates(y, p1 | (d >= t))
        obj = lam * tpr - (1 - lam) * fpr
        if obj > best_obj:
            best_s, best_obj = float(t), obj

    above = d[d > best_s]
    best_b, best_tpr_b = float(np.nanmax(d)) + 1e-6, -1.0
    for t in (_candidates(above, grid_size) if len(above) > 1 else []):
        tpr_b, fpr_b = _rates(y, p1 | (d > t))
        if fpr_b <= MAX_BLOCK_FPR and tpr_b > best_tpr_b:
            best_b, best_tpr_b = float(t), tpr_b

    v = cal["embedding_variance"].to_numpy(dtype=float)
    best_v, best_v_obj = np.nan, -np.inf
    for t in _candidates(v, grid_size):
        tpr, fpr = _rates(y, p1 | (v > t))
        obj = lam * tpr - (1 - lam) * fpr
        if obj > best_v_obj:
            best_v, best_v_obj = float(t), obj

    # TED drift threshold: 95th percentile of |D~_i - D~_j| between independent harmless prompts,
    # i.e. the drift expected from benign topic changes alone.
    h = d[y == 0]
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(h), size=(5000, 2)) if len(h) > 1 else np.zeros((0, 2), int)
    ted_delta = float(np.quantile(np.abs(h[pairs[:, 0]] - h[pairs[:, 1]]), 0.95)) if len(pairs) else np.nan

    tpr_s, fpr_s = _rates(y, p1 | (d >= best_s))
    return {
        "ted_delta": ted_delta,
        "tau_s": best_s, "tau_b": best_b, "embedding_variance_threshold": best_v,
        "lambda": lam, "objective": float(best_obj), "calib_tpr": tpr_s, "calib_fpr": fpr_s,
        "n_calib": int(len(cal)), "max_block_fpr": MAX_BLOCK_FPR,
    }


def thresholds_path(cfg: Dict, backbone: str) -> Path:
    return output_dir(cfg, "runs", backbone) / "thresholds.json"


def apply_thresholds(cfg: Dict, th: Dict[str, float]) -> None:
    cfg["pso"]["thresholds"]["tau_s"] = float(th["tau_s"])
    cfg["pso"]["thresholds"]["tau_b"] = float(th["tau_b"])
    cfg["baselines"]["embedding_variance"]["threshold"] = float(th["embedding_variance_threshold"])
    if np.isfinite(th.get("ted_delta", np.nan)):
        cfg["pso"]["phase1"]["ted"]["delta"] = float(th["ted_delta"])


def load_thresholds(cfg: Dict, backbone: str) -> Optional[Dict[str, float]]:
    p = thresholds_path(cfg, backbone)
    return json.loads(p.read_text()) if p.exists() else None


def ensure_thresholds(cfg: Dict, backbone: str, analysis: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """Load (or, given the analysis frame, compute and save) calibrated thresholds and apply them to cfg."""
    if not cfg["evaluation"].get("calibrate_thresholds", False):
        return None
    th = load_thresholds(cfg, backbone)
    if th is None and analysis is not None:
        ev = cfg["evaluation"]
        th = calibrate(analysis, ev["calibration_lambda"], ev["calibration_grid_size"])
        thresholds_path(cfg, backbone).write_text(json.dumps(th, indent=2))
        logger.info("Calibrated thresholds for %s: %s", backbone, th)
    if th is not None:
        apply_thresholds(cfg, th)
    return th


def direction_thresholds(cfg: Dict, backbone: str, analysis: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """Exploratory direction-calibrated detector (Exp. 11): the calibration split chooses the sign of D~.

    direction = +1 if AUC_calib(D~) >= 0.5 else -1; thresholds are Eq. 17 on direction * D~.
    Saved as runs/<backbone>/thresholds_dircal.json.
    """
    p = output_dir(cfg, "runs", backbone) / "thresholds_dircal.json"
    if p.exists():
        return json.loads(p.read_text())
    if analysis is None:
        a = output_dir(cfg, "runs", backbone, "main") / "analysis.csv"
        if not a.exists():
            return None
        analysis = pd.read_csv(a)
    from evaluation import metrics as M
    cal = analysis[analysis["split"] == "calib"]
    auc = M.auc(cal["ground_truth"].to_numpy(), cal["disagreement_score"].to_numpy())
    direction = 1 if auc >= 0.5 else -1
    flipped = analysis.copy()
    flipped["disagreement_score"] = direction * flipped["disagreement_score"]
    ev = cfg["evaluation"]
    th = calibrate(flipped, ev["calibration_lambda"], ev["calibration_grid_size"])
    th.update(direction=direction, calib_auc_of_D=float(auc))
    p.write_text(json.dumps(th, indent=2))
    return th
