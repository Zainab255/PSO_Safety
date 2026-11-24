"""Statistical protocol (Section IV-B-5): mean ± std over runs, one-tailed
paired t-tests against the PSO, ANOVA + Tukey HSD across categories, and the
power calculation used to justify the adversarial subset size."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def ci95(values: Sequence[float]) -> float:
    """Half-width of the t-based 95% confidence interval of the mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return 0.0
    return float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))


def paired_one_tailed(a: Sequence[float], b: Sequence[float], alternative: str) -> float:
    """p-value of the paired t-test H1: mean(a - b) {'greater'|'less'} 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    if d.size < 2 or np.allclose(d, d[0]):
        # Zero variance of the differences: the t statistic is undefined.
        return float("nan")
    return float(stats.ttest_rel(a, b, alternative=alternative).pvalue)


def welch(a: Sequence[float], b: Sequence[float], alternative: str = "two-sided") -> float:
    return float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)


def stars(p: float) -> str:
    if p is None or math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt(mean: float, std: float, p: float = float("nan"), digits: int = 2) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.{digits}f}±{std:.{digits}f}{stars(p)}"


def anova_tukey(groups: Dict[str, Sequence[float]]):
    """One-way ANOVA across categories plus Tukey HSD pairwise comparisons."""
    names = list(groups)
    samples = [np.asarray(groups[n], dtype=float) for n in names]
    f, p = stats.f_oneway(*samples)
    df_between = len(samples) - 1
    df_within = sum(s.size for s in samples) - len(samples)
    tukey = stats.tukey_hsd(*samples)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({
                "group_a": names[i], "group_b": names[j],
                "mean_diff": float(samples[i].mean() - samples[j].mean()),
                "p_value": float(tukey.pvalue[i, j]),
            })
    return {"F": float(f), "p_value": float(p), "df_between": df_between, "df_within": df_within}, pairs


def paired_t_power(diffs: Sequence[float], n: int, alpha: float) -> float:
    """Power of a two-sided paired t-test with n pairs for the observed effect."""
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0 or n < 2:
        return float("nan")
    effect = abs(d.mean()) / sd
    df = n - 1
    nc = effect * math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc))
