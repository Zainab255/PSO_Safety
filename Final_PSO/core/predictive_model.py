"""Predictive Safety Model (Section III-G).

    P(unsafe | D_s) = sigma(alpha * D_s + beta)                          (Eq. 13)
    P(unsafe | s)   = sigma(a1 * D_s + a2 * S_RJ + a3 * S_BJ + beta)     (Eq. 14)

Coefficients are fitted by (unpenalised) maximum likelihood on a labelled
calibration corpus; nothing is preset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)


class PredictiveSafetyModel:
    """Logistic map from geometric signals to P(unsafe)."""

    def __init__(self, features: Sequence[str]):
        self.features: List[str] = list(features)
        self.coef: np.ndarray | None = None
        self.intercept: float | None = None
        self.n_calibration: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.coef is not None

    def _design(self, rows: Mapping[str, Sequence[float]]) -> np.ndarray:
        return np.column_stack([np.asarray(rows[f], dtype=float) for f in self.features])

    def fit(self, rows: Mapping[str, Sequence[float]], labels: Sequence[int]) -> "PredictiveSafetyModel":
        """Maximum-likelihood fit on the calibration corpus D_cal."""
        x = self._design(rows)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration corpus must contain both safe and unsafe examples.")
        # C = inf removes the regulariser: plain maximum likelihood.
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10_000)
        lr.fit(x, y)
        self.coef = lr.coef_[0].astype(float)
        self.intercept = float(lr.intercept_[0])
        self.n_calibration = int(len(y))
        return self

    def predict_proba(self, rows: Mapping[str, Sequence[float]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PredictiveSafetyModel is not fitted.")
        return sigmoid(self._design(rows) @ self.coef + self.intercept)

    def predict_one(self, signals: Mapping[str, float]) -> float:
        return float(self.predict_proba({f: [signals[f]] for f in self.features})[0])

    def as_dict(self) -> Dict:
        return {
            "features": self.features,
            "coef": None if self.coef is None else self.coef.tolist(),
            "intercept": self.intercept,
            "n_calibration": self.n_calibration,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PredictiveSafetyModel":
        with open(path) as f:
            d = json.load(f)
        model = cls(d["features"])
        model.coef = np.asarray(d["coef"], dtype=float)
        model.intercept = float(d["intercept"])
        model.n_calibration = int(d["n_calibration"])
        return model
