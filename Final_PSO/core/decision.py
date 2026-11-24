"""Decision rules of the PSO (Sections III-G-4 and III-H)."""

from __future__ import annotations

import enum
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


class WrapperDecision(enum.Enum):
    """Outcome of a safety evaluation (Def. 1)."""

    ALLOW = "ALLOW"
    REQUERY = "REQUERY"   # sanitise with template T and regenerate
    BLOCK = "BLOCK"


def geometric_decision(score: float, tau_s: float, tau_b: float) -> WrapperDecision:
    """Eq. (18): map the normalised disagreement score to a decision."""
    if score < tau_s:
        return WrapperDecision.ALLOW
    if score <= tau_b:
        return WrapperDecision.REQUERY
    return WrapperDecision.BLOCK


def probabilistic_decision(p_unsafe: float, rho_low: float, rho_high: float) -> WrapperDecision:
    """Eq. (17): decision from the Predictive Safety Model probability."""
    if p_unsafe > rho_high:
        return WrapperDecision.BLOCK
    if p_unsafe >= rho_low:
        return WrapperDecision.REQUERY
    return WrapperDecision.ALLOW


def threshold_grid(spec: Dict[str, float]) -> np.ndarray:
    """Inclusive arithmetic grid from a ``{start, stop, step}`` spec."""
    n = int(round((spec["stop"] - spec["start"]) / spec["step"])) + 1
    return np.round(spec["start"] + spec["step"] * np.arange(n), 10)


def decisions_for_thresholds(
    scores: np.ndarray, phase1_block: np.ndarray, tau_s: float, tau_b: float
) -> List[WrapperDecision]:
    """Full-pipeline decisions for a corpus at a given (tau_s, tau_b)."""
    out = []
    for s, p1 in zip(scores, phase1_block):
        out.append(WrapperDecision.BLOCK if p1 else geometric_decision(float(s), tau_s, tau_b))
    return out


def roc_threshold_calibration(
    evaluate: Callable[[float, float], Tuple[float, float]],
    tau_s_values: Iterable[float],
    tau_b_values: Sequence[float],
    lam: float,
) -> Tuple[float, float, float]:
    """Eq. (19): argmax over valid (tau_s, tau_b) of lam*TPR - (1-lam)*FPR.

    ``evaluate(tau_s, tau_b)`` must return (TPR, FPR) on the validation set.
    Returns (tau_s*, tau_b*, objective).
    """
    best = (float("nan"), float("nan"), -float("inf"))
    for ts in tau_s_values:
        for tb in tau_b_values:
            if ts >= tb:
                continue
            tpr, fpr = evaluate(ts, tb)
            obj = lam * tpr - (1.0 - lam) * fpr
            if obj > best[2]:
                best = (float(ts), float(tb), float(obj))
    return best
