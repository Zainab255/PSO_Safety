"""Figures 2-11 of the paper, drawn from the experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots (validated light-mode palette), assigned to entities in a fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_ORDER = ["pso", "self_critique", "llm_judge", "keyword_history", "embedding_variance", "seq_orch"]
METHOD_COLORS = {m: SLOTS[i] for i, m in enumerate(METHOD_ORDER)}
METHOD_LABELS = {
    "pso": "PSO", "seq_orch": "Seq. Orch.", "llm_judge": "LLM-Judge", "self_critique": "Self-Crit.",
    "keyword_history": "Kw.+Hist.", "embedding_variance": "Emb. Variance",
}
GROUP_COLORS = {"harmless": SLOTS[0], "explicit_harmful": SLOTS[1], "adversarial": SLOTS[2]}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def violin_by_category(data: Dict[str, np.ndarray], groups: Dict[str, str], tau_s: float, tau_b: float, path: Path):
    """Fig. 2: D~ distribution per category with the two thresholds."""
    names = list(data)
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(names)), 3.4))
    parts = ax.violinplot([data[n] for n in names], showmedians=True, showextrema=False)
    for body, n in zip(parts["bodies"], names):
        body.set_facecolor(GROUP_COLORS.get(groups[n], SLOTS[6]))
        body.set_edgecolor("white")
        body.set_alpha(0.75)
    parts["cmedians"].set_color(INK)
    ax.axhline(tau_s, color=INK_2, ls="--", lw=1)
    ax.axhline(tau_b, color=INK_2, ls=":", lw=1)
    ax.text(len(names) + 0.45, tau_s, r"$\tau_s$", va="center", color=INK_2)
    ax.text(len(names) + 0.45, tau_b, r"$\tau_b$", va="center", color=INK_2)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([f"{n}\n(n={len(data[n])})" for n in names], fontsize=7)
    ax.set_ylabel(r"Normalised disagreement $\tilde{D}_s^{(K)}$")
    _save(fig, path)


def roc_curves(curves: Dict[str, np.ndarray], grid: np.ndarray, aucs: Dict[str, float],
               op_point: Optional[Sequence[float]], path: Path):
    """Fig. 3: mean ROC ± 1 s.d. over runs per method."""
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for m, tprs in curves.items():
        mean, sd = tprs.mean(axis=0), tprs.std(axis=0)
        c = METHOD_COLORS.get(m, INK_2)
        ax.plot(grid, mean, color=c, ls="--" if m == "seq_orch" else "-", label=f"{label(m)} (AUC={aucs[m]:.2f})")
        ax.fill_between(grid, mean - sd, mean + sd, color=c, alpha=0.15, lw=0)
    ax.plot([0, 1], [0, 1], color=GRID, ls=":", lw=1)
    if op_point is not None:
        ax.plot(*op_point, marker="*", ms=12, color=METHOD_COLORS["pso"], mec="white", ls="none")
        ax.annotate(f"PSO op. point\n(FPR={op_point[0]:.2f}, TPR={op_point[1]:.2f})", op_point,
                    xytext=(0.3, 0.55), color=INK_2, fontsize=7, arrowprops={"arrowstyle": "-", "color": INK_2})
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, path)


def conflict_space(pca_xy: np.ndarray, tsne_xy: np.ndarray, groups: Sequence[str], categories: Sequence[str],
                   explained: Sequence[float], path: Path):
    """Fig. 4: PCA and t-SNE of (s_RB, s_RJ, s_BJ). Colour = group, marker = sub-category."""
    group_of = dict(zip(categories, groups))
    group_rank = {g: i for i, g in enumerate(GROUP_COLORS)}
    cats = sorted(group_of, key=lambda c: (group_rank.get(group_of[c], len(group_rank)), c))
    marker_of = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(cats)}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    categories = np.asarray(categories)
    for ax, xy, title in ((axes[0], pca_xy, "PCA"), (axes[1], tsne_xy, "t-SNE")):
        for c in cats:
            m = categories == c
            ax.scatter(xy[m, 0], xy[m, 1], s=10, marker=marker_of[c], color=GROUP_COLORS.get(group_of[c], SLOTS[6]),
                       alpha=0.7, linewidths=0.3, edgecolors="white", label=c)
        ax.set_title(title, color=INK)
    axes[0].set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% var.)")
    axes[0].set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% var.)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, markerscale=1.5)
    _save(fig, path)


def pareto(points: Dict[str, Dict[str, float]], path: Path):
    """Fig. 5: FPR-TPR plane with 95% CIs and the empirical Pareto frontier."""
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for m, p in points.items():
        c = METHOD_COLORS.get(m, INK_2)
        ax.errorbar(p["fpr"], p["tpr"], xerr=p["fpr_ci"], yerr=p["tpr_ci"], fmt="o", ms=7, color=c,
                    mec="white", capsize=2, label=label(m))
        ax.annotate(label(m), (p["fpr"], p["tpr"]), xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK_2)
    pts = sorted(points.values(), key=lambda p: (p["fpr"], -p["tpr"]))
    frontier, best = [], -1.0
    for p in pts:
        if p["tpr"] > best:
            frontier.append(p)
            best = p["tpr"]
    xs, ys = [], []
    for p in frontier:
        xs += [p["fpr"], p["fpr"]]
        ys += [ys[-1] if ys else 0, p["tpr"]]
    ax.plot(xs[1:] + [1], ys[1:] + [ys[-1]], color=INK_2, ls=":", lw=1, label="Pareto frontier")
    ax.set_xlabel("FPR (lower is better)")
    ax.set_ylabel("TPR (higher is better)")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def latency_bars(wrapper_ms: Dict[str, Sequence[float]], downstream_ms: float, path: Path):
    """Fig. 6: wrapper overhead stacked on the shared downstream generation cost."""
    methods = list(wrapper_ms)
    means = [float(np.mean(wrapper_ms[m])) for m in methods]
    sds = [float(np.std(wrapper_ms[m], ddof=1)) if len(wrapper_ms[m]) > 1 else 0.0 for m in methods]
    ref = means[methods.index("pso")] if "pso" in methods else None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(methods))
    ax.bar(x, [downstream_ms] * len(methods), color="#c3c2b7", width=0.6, label=f"Downstream gen. (~{downstream_ms:.0f} ms)")
    ax.bar(x, means, bottom=downstream_ms, yerr=sds, capsize=3, width=0.6, edgecolor="white", linewidth=2,
           color=[METHOD_COLORS.get(m, INK_2) for m in methods])
    for xi, mu in zip(x, means):
        ax.text(xi, downstream_ms + mu, f"{mu:.0f} ms", ha="center", va="bottom", fontsize=8, color=INK)
        if ref:
            ax.text(xi, downstream_ms / 2, f"{mu / ref:.2f}×", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods])
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, path)


def asr_curves(curves: Dict[str, np.ndarray], ks: Sequence[int], path: Path):
    """Fig. 7: ASR@k with 95% CI band over runs (rows = runs)."""
    from evaluation.stats import ci95
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for m, runs in curves.items():
        c = METHOD_COLORS.get(m, INK_2)
        mean = runs.mean(axis=0)
        half = np.array([ci95(runs[:, j]) for j in range(runs.shape[1])])
        ax.plot(ks, mean, color=c, label=label(m), lw=2.5 if m == "pso" else 1.5)
        ax.fill_between(ks, mean - half, mean + half, color=c, alpha=0.15, lw=0)
        ax.text(ks[-1] + 0.2, mean[-1], f"{mean[-1]:.2f}", color=INK_2, fontsize=7, va="center")
    ax.set_xlabel("Query count k")
    ax.set_ylabel("Attack success rate ASR@k")
    ax.legend(fontsize=7)
    _save(fig, path)


def score_trajectory(trajectories: Dict[str, np.ndarray], tau_b: float, path: Path):
    """Fig. 8: mean D~ per query step, split by attack outcome (rows = prompts)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, (name, traj) in enumerate(trajectories.items()):
        if traj.size == 0:
            continue
        mean, sd = np.nanmean(traj, axis=0), np.nanstd(traj, axis=0)
        k = np.arange(1, traj.shape[1] + 1)
        ax.plot(k, mean, marker="o", ms=4, color=SLOTS[i], label=f"{name} (n={traj.shape[0]})")
        ax.fill_between(k, mean - sd, mean + sd, color=SLOTS[i], alpha=0.15, lw=0)
    ax.axhline(tau_b, color=INK_2, ls="--", lw=1)
    ax.text(1, tau_b, r" $\tau_b$", va="bottom", color=INK_2)
    ax.set_xlabel("Adaptive attack query step k")
    ax.set_ylabel(r"Mean $\tilde{D}_s^{(K)}$")
    ax.legend(fontsize=7)
    _save(fig, path)


def f_beta_heatmap(grid: np.ndarray, tau_s: np.ndarray, tau_b: np.ndarray, default: Sequence[float],
                   beta: float, path: Path):
    """Fig. 9: F_beta over the (tau_s, tau_b) grid; NaN cells (tau_s >= tau_b) are left blank."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, origin="lower", cmap=SEQUENTIAL, aspect="auto")
    for i in range(len(tau_s)):
        for j in range(len(tau_b)):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                rel = (v - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-12)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if rel > 0.55 else INK)
    ax.set_xticks(range(len(tau_b)))
    ax.set_xticklabels([f"{t:.2f}" for t in tau_b], fontsize=7)
    ax.set_yticks(range(len(tau_s)))
    ax.set_yticklabels([f"{t:.2f}" for t in tau_s], fontsize=7)
    ax.set_xlabel(r"Block threshold $\tau_b$")
    ax.set_ylabel(r"Sanitise threshold $\tau_s$")
    ax.grid(False)
    di = int(np.argmin(np.abs(tau_s - default[0])))
    dj = int(np.argmin(np.abs(tau_b - default[1])))
    ax.plot(dj, di, marker="*", ms=14, mfc="none", mec=METHOD_COLORS["pso"], mew=1.5)
    fig.colorbar(im, ax=ax, label=f"$F_{{{beta:g}}}$ score")
    _save(fig, path)


def calibration(scores: np.ndarray, labels: np.ndarray, groups: Sequence[str], curve_x: np.ndarray,
                curve_p: np.ndarray, band: np.ndarray, coef: Sequence[float], tau_s: float, tau_b: float,
                path: Path, seed: int):
    """Fig. 10: fitted calibration curve (left) and decision regions (right)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    ax = axes[0]
    groups = np.asarray(groups)
    for g, c in GROUP_COLORS.items():
        m = groups == g
        if m.any():
            jitter = rng.uniform(-0.02, 0.02, m.sum())
            ax.scatter(scores[m], labels[m] + jitter, s=6, color=c, alpha=0.4, lw=0, label=f"{g} (n={m.sum()})")
    ax.fill_between(curve_x, band[0], band[1], color=METHOD_COLORS["pso"], alpha=0.2, lw=0)
    ax.plot(curve_x, curve_p, color=INK, label=rf"$\sigma({coef[0]:.2f}\,D_s{coef[1]:+.2f})$")
    for ax_ in axes:
        ax_.axvline(tau_s, color=INK_2, ls="--", lw=1)
        ax_.axvline(tau_b, color=INK_2, ls=":", lw=1)
        ax_.set_xlabel(r"Disagreement score $\tilde{D}_s^{(K)}$")
        ax_.set_ylabel(r"$P(\mathrm{unsafe}\mid D_s)$")
        ax_.set_ylim(-0.06, 1.06)
    ax.legend(fontsize=6, loc="center right")
    ax = axes[1]
    lo, hi = curve_x.min(), curve_x.max()
    for (a, b), name, c in (((lo, tau_s), "ALLOW", SLOTS[2]), ((tau_s, tau_b), "REQUERY", SLOTS[3]),
                            ((tau_b, hi), "BLOCK", SLOTS[7])):
        ax.axvspan(a, b, color=c, alpha=0.12, lw=0)
        ax.text((a + b) / 2, 0.05, name, ha="center", fontsize=8, color=INK)
    ax.plot(curve_x, curve_p, color=INK)
    for t in (tau_s, tau_b):
        p = float(np.interp(t, curve_x, curve_p))
        ax.axhline(p, color=INK_2, lw=0.6, ls=":")
        ax.text(lo, p, f" P={p:.2f}", va="bottom", fontsize=7, color=INK_2)
    _save(fig, path)


def radar(axes_values: Dict[str, Dict[str, float]], axis_names: List[str], path: Path):
    """Fig. 11: five-axis performance radar, axes normalised to [0, 1]."""
    angles = np.linspace(0, 2 * np.pi, len(axis_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"polar": True})
    for m, vals in axes_values.items():
        v = [vals[a] for a in axis_names]
        v += v[:1]
        c = METHOD_COLORS.get(m, INK_2)
        ax.plot(angles, v, color=c, lw=2.5 if m == "pso" else 1.2, ls="-" if m == "pso" else "--", label=label(m))
        if m == "pso":
            ax.fill(angles, v, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    _save(fig, path)
