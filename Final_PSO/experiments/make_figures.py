#!/usr/bin/env python3
"""Regenerate every result figure directly from the logged per-prompt data (CPU only).

No curve is smoothed or fitted for display: ROC curves are empirical step functions on the
held-out test split, ASR@k curves are empirical step functions per seed, and every point in
the scatter plots is one (method, seed) run. Each figure is written as PDF and PNG, and the
exact numbers plotted are written next to it as <figure>.csv.

    python -m experiments.make_figures [--config ...] [--backbone llama2]

Inputs (all produced by the experiment scripts)
    runs/<bb>/main/analysis.csv, eval_seed*_<method>.csv, responses_seed*.csv, thresholds*.json
    runs/<bb>/exp4/attack_*.csv, runs/<bb>/exp10/*, runs/<bb>/exp12/scores_with_probes.csv
    tables/exp10_*.csv, tables/exp12_*.csv
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from evaluation import metrics as M  # noqa: E402
from experiments.common import output_dir, run_dir, setup, test_rows  # noqa: E402

# Categorical palette in fixed order (validated: lightness band, chroma, CVD separation).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"
METHOD_LABEL = {"keyword_history": "Keyword+History", "embedding_variance": "Emb. variance",
                "llm_judge": "LLM-as-a-Judge", "self_critique": "Self-Critique", "seq_orch": "Seq. Orch.",
                "pso": "PSO (pre-specified)", "pso_dircal": "PSO (direction-calibrated)"}
METHOD_ORDER = ["pso", "pso_dircal", "seq_orch", "embedding_variance", "llm_judge", "self_critique", "keyword_history"]
GROUP_ORDER = ["benign", "ambiguous", "direct_xstest", "direct_advbench", "framed_jailbreak", "wild_jailbreak"]
GROUP_LABEL = {"benign": "Benign (XSTest safe)", "ambiguous": "Ambiguous (harmless)",
               "direct_xstest": "Direct harmful (XSTest)", "direct_advbench": "Direct harmful (AdvBench goals)",
               "framed_jailbreak": "Framed jailbreak", "wild_jailbreak": "In-the-wild jailbreak"}


def style():
    import matplotlib.font_manager as fm
    for _p in fm.findSystemFonts():   # register a Times-like serif if the system has one
        if "termes" in _p.lower() or "nimbusroman" in _p.lower().replace("-", ""):
            fm.fontManager.addfont(_p)
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "savefig.bbox": "tight", "font.family": "serif",
    })


def save(fig, cfg, name, data: pd.DataFrame):
    d = output_dir(cfg, "figures")
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=220)
    data.to_csv(d / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf (+png, csv)")


def color_of(method):
    return PALETTE[METHOD_ORDER.index(method) % len(PALETTE)] if method in METHOD_ORDER else MUTED


def load_evals(cfg, bb):
    d = run_dir(cfg, bb, "main")
    files = sorted(d.glob("eval_seed*_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ----------------------------------------------------------------------------- figures

def fig_distributions(cfg, bb, an, th):
    """D~ per category (all prompts of the main corpus), with the calibrated thresholds."""
    order = (an.groupby("prompt_category")["disagreement_score"].median().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    data = [an.loc[an["prompt_category"] == c, "disagreement_score"].to_numpy() for c in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for i, (b, c) in enumerate(zip(parts["bodies"], order)):
        grp = an.loc[an["prompt_category"] == c, "group"].iloc[0]
        col = {"harmless": PALETTE[0], "explicit_harmful": PALETTE[3], "adversarial": PALETTE[1]}.get(grp, MUTED)
        b.set_facecolor(col)
        b.set_alpha(0.35)
        b.set_edgecolor(col)
        q1, med, q3 = np.percentile(data[i], [25, 50, 75])
        ax.vlines(i + 1, q1, q3, color=col, lw=3)
        ax.scatter([i + 1], [med], color="white", edgecolor=col, zorder=3, s=14)
    for key, lab, ls in (("tau_s", r"$\tau_s$", "--"), ("tau_b", r"$\tau_b$", ":")):
        if th and np.isfinite(th.get(key, np.nan)):
            ax.axhline(th[key], color=MUTED, ls=ls, lw=1)
            ax.text(len(order) + 0.55, th[key], lab, va="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{c.replace('_', ' ')}\n(n={len(x)})" for c, x in zip(order, data)], rotation=25, ha="right")
    ax.set_ylabel(r"$\tilde D$ (final layer)")
    ax.set_title(f"Persona disagreement by prompt category — {bb}, all {len(an)} prompts")
    rows = [{"category": c, "n": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
             "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75))} for c, x in zip(order, data)]
    save(fig, cfg, f"fig2_distributions_{bb}", pd.DataFrame(rows))


def fig_roc(cfg, bb, ev, an, base):
    """Empirical ROC on the test split: pipeline methods (seed 42 decisions + risk) and score baselines."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    rows = []
    t = test_rows(ev)
    seed = sorted(t["seed"].unique())[0] if len(t) else None
    curves = []
    for m in METHOD_ORDER:
        g = t[(t["method"] == m) & (t["seed"] == seed)]
        if g.empty:
            continue
        s = M.pipeline_scores(g["risk_score"].to_numpy(), g["decided_by"].fillna(""))
        curves.append((METHOD_LABEL[m], g["ground_truth"].to_numpy(), s, color_of(m), "-"))
    if base is not None:
        bt = base[base["split"].eq("test") & base["in_main"].astype(bool)]
        for col, lab, c in (("msds_dircal", "D~ direction-calibrated (score only)", PALETTE[1]),
                            ("probe_last", "Linear probe (no personas)", PALETTE[5]),
                            ("hiddendetect", "HiddenDetect-style", PALETTE[6]),
                            ("log_ppl", "Perplexity", PALETTE[7])):
            if col in bt and bt[col].notna().any():
                curves.append((lab, bt["ground_truth"].to_numpy(), bt[col].to_numpy(), c, "--"))
    for lab, y, s, c, ls in curves:
        ok = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        auc = M.auc(y[ok], s[ok])
        ax.step(fpr, tpr, where="post", color=c, ls=ls, lw=1.3, label=f"{lab} ({auc:.2f})")
        rows += [{"method": lab, "fpr": a, "tpr": b, "auc": auc} for a, b in zip(fpr, tpr)]
    ax.plot([0, 1], [0, 1], color="#bdbcb6", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Empirical ROC, test split ({bb}; AUC in legend)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    save(fig, cfg, f"fig3_roc_{bb}", pd.DataFrame(rows))


def fig_pareto(cfg, bb, ev):
    """TPR vs FPR: one point per (method, seed) run on the test split; cross = mean."""
    from experiments.common import method_run_metrics
    r = method_run_metrics(ev, cfg)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for m in METHOD_ORDER:
        g = r[r["method"] == m]
        if g.empty:
            continue
        ax.scatter(g["fpr"], g["tpr"], s=12, color=color_of(m), alpha=0.6)
        ax.scatter(g["fpr"].mean(), g["tpr"].mean(), marker="P", s=50, color=color_of(m),
                   edgecolor="white", label=METHOD_LABEL[m])
    ax.set_xlabel("FPR on harmless prompts (lower is better)")
    ax.set_ylabel("TPR (higher is better)")
    ax.set_title(f"Safety–utility trade-off, test split ({bb})")
    ax.legend(fontsize=6, loc="best")
    save(fig, cfg, f"fig5_pareto_{bb}", r)


def fig_latency(cfg, bb, ev):
    """Wrapper-only latency (unbatched, timed subset) and like-for-like end-to-end latency."""
    d = run_dir(cfg, bb, "main")
    resp = [pd.read_csv(p) for p in sorted(d.glob("responses_seed*.csv")) if "partial" not in p.name]
    gen = pd.concat(resp)["generation_ms"].dropna() if resp else pd.Series(dtype=float)
    gen_ms = float(gen.median()) if len(gen) else np.nan
    rows = []
    for m in METHOD_ORDER:
        g = ev[(ev["method"] == m)]["latency_ms"].dropna()
        if g.empty:
            continue
        wrapper = float(g.median())
        # Self-Critique already produces the response (its draft); the others add one generation
        # for allowed prompts. End-to-end = time until a response (or a block) is returned.
        e2e = wrapper if m == "self_critique" else wrapper + gen_ms
        rows.append({"method": m, "wrapper_ms_median": wrapper, "wrapper_ms_q1": float(g.quantile(0.25)),
                     "wrapper_ms_q3": float(g.quantile(0.75)), "end_to_end_ms_allowed": e2e,
                     "generation_ms_median": gen_ms, "n_timed": len(g)})
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    y = np.arange(len(r))
    for ax, col, title in ((axes[0], "wrapper_ms_median", "Wrapper only (decision), median ms"),
                           (axes[1], "end_to_end_ms_allowed", "End-to-end for an allowed prompt, ms")):
        ax.barh(y, r[col], color=[color_of(m) for m in r["method"]], height=0.6)
        for yi, v in zip(y, r[col]):
            ax.text(v, yi, f" {v:,.0f}", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABEL[m] for m in r["method"]])
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    if len(r):
        axes[0].errorbar(r["wrapper_ms_median"], y, xerr=[r["wrapper_ms_median"] - r["wrapper_ms_q1"],
                                                          r["wrapper_ms_q3"] - r["wrapper_ms_median"]],
                         fmt="none", ecolor=MUTED, lw=0.8)
    save(fig, cfg, f"fig6_latency_{bb}", r)


def attacks(cfg, bb):
    d = run_dir(cfg, bb, "exp4")
    files = sorted(d.glob("attack_advbench_direct_seed*_*.csv")) if d.exists() else []
    stab = set(cfg["experiments"]["exp4"].get("stability_seeds", []))
    frames = [pd.read_csv(f) for f in files]
    a = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return a[~a["seed"].isin(stab)] if len(a) else a


def fig_asr(cfg, bb, a):
    """ASR@k per attacker and method: empirical step function, mean over seeds (thin = each seed)."""
    if a.empty:
        return
    kmax = int(a["attack_attempt"].max())
    levels = [l for l in ("adaptive", "score_aware") if l in set(a["attacker"])]
    if not levels:
        return
    fig, axes = plt.subplots(1, len(levels), figsize=(3.4 * len(levels), 2.8), sharey=True, squeeze=False)
    rows = []
    for ax, level in zip(axes[0], levels):
        al = a[a["attacker"] == level]
        for m in METHOD_ORDER:
            g = al[al["method"] == m]
            if g.empty:
                continue
            per_seed = []
            for s, gs in g.groupby("seed"):
                fs = gs[gs["evasion_success"].astype(bool)].groupby("prompt_id")["attack_attempt"].min() \
                    .reindex(gs["prompt_id"].unique())
                curve = [M.asr_at_k(fs.to_numpy(float), k) for k in range(1, kmax + 1)]
                per_seed.append(curve)
                ax.step(range(1, kmax + 1), curve, where="post", color=color_of(m), lw=0.6, alpha=0.35)
                rows += [{"attacker": level, "method": m, "seed": s, "k": k + 1, "asr": v} for k, v in enumerate(curve)]
            mean = np.mean(per_seed, axis=0)
            ax.step(range(1, kmax + 1), mean, where="post", color=color_of(m), label=METHOD_LABEL[m])
        ax.set_xlabel("query budget k")
        ax.set_xticks(range(1, kmax + 1))
        n_goals = int(al.groupby("seed")["prompt_id"].nunique().max())
        ax.set_title(f"{level.replace('_', '-')} attacker ({bb}, {n_goals} goals/seed)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("ASR@k (valid evasions)")
    # One legend for every method that appears in any panel.
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0][0].legend(handles.values(), handles.keys(), fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.84), framealpha=0.9)
    save(fig, cfg, f"fig7_asr_{bb}", pd.DataFrame(rows))


def fig_trajectory(cfg, bb, a, th_by_method):
    """Risk score of the defence per query step, aligned to the ALLOW boundary tau_s."""
    if a.empty:
        return
    levels = [l for l in ("adaptive", "score_aware") if l in set(a["attacker"])]
    methods = [m for m in ("pso", "pso_dircal") if m in set(a["method"])]
    if not levels or not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 2.8), squeeze=False)
    rows = []
    for ax, m in zip(axes[0], methods):
        g = a[(a["method"] == m) & (a["attacker"].isin(levels))]
        kmax = int(g["attack_attempt"].max())
        for li, level in enumerate(levels):
            gl = g[g["attacker"] == level]
            ev = gl.groupby(["seed", "prompt_id"])["evasion_success"].transform(lambda s: s.astype(bool).any())
            for outcome, mask, ls in (("evaded", ev, "-"), ("never evaded", ~ev, "--")):
                sub = gl[mask]
                if sub.empty:
                    continue
                # Phase-1 blocks carry risk 1.0; plot the persona score itself (sign-corrected for
                # the direction-calibrated defence), which is what the thresholds act on.
                sign = -1.0 if m == "pso_dircal" else 1.0
                mean = (sign * sub["entropy_score"]).groupby(sub["attack_attempt"]).mean().reindex(range(1, kmax + 1))
                n = sub.groupby(["seed", "prompt_id"]).ngroups
                ax.plot(mean.index, mean.values, ls=ls, color=PALETTE[li * 2], marker="o", ms=2.5,
                        label=f"{level.replace('_', '-')}: {outcome} (n={n})")
                rows += [{"method": m, "attacker": level, "outcome": outcome, "k": k, "mean_risk": v, "n": n}
                         for k, v in mean.items()]
        ts = th_by_method.get(m, {}).get("tau_s")
        if ts is not None:
            ax.axhline(ts, color=MUTED, ls=":", lw=1)
            ax.text(kmax, ts, r" $\tau_s$ (ALLOW below)", va="bottom", ha="right", fontsize=6.5, color=MUTED)
        ax.set_xlabel("query step k")
        ax.set_xticks(range(1, kmax + 1))
        ax.set_ylabel("persona score" + (r" ($-\tilde D$)" if m == "pso_dircal" else r" ($\tilde D$)"))
        ax.set_title(f"{METHOD_LABEL[m]} under attack ({bb})")
        ax.legend(fontsize=5.5)
    save(fig, cfg, f"fig8_trajectory_{bb}", pd.DataFrame(rows))


def fig_grid(cfg, bb, an, direction=1, tag=""):
    """F2 over the tau grid on the TEST split; thresholds of the grid are calibration-split quantiles."""
    from experiments.exp5_sensitivity import classification, pipeline_decisions, score_grid
    df = an.copy()
    df["disagreement_score"] = direction * df["disagreement_score"]
    ts, tb = score_grid(cfg, df)
    mat = np.full((len(ts), len(tb)), np.nan)
    rows = []
    for i, s in enumerate(ts):
        for j, b in enumerate(tb):
            if b <= s:
                continue
            c = classification(cfg, df, *pipeline_decisions(df, s, b))
            mat[i, j] = c["f_beta"]
            rows.append({"tau_s": s, "tau_b": b, "f2_test": c["f_beta"], "tpr": c["tpr"], "fpr": c["fpr"]})
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(mat, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ts)):
        for j in range(len(tb)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if mat[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(tb)))
    ax.set_xticklabels([f"{v:.3f}" for v in tb], rotation=60)
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels([f"{v:.3f}" for v in ts])
    ax.set_xlabel(r"$\tau_b$")
    ax.set_ylabel(r"$\tau_s$")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"$F_2$ (test split)")
    ax.set_title(f"Threshold sensitivity{tag} ({bb})")
    save(fig, cfg, f"fig9_grid{tag.replace(' ', '_')}_{bb}", pd.DataFrame(rows))


def fig_reliability(cfg, bb, an, direction):
    """Logistic P(harmful | D~) fitted on the calibration split, reliability on the test split."""
    from sklearn.linear_model import LogisticRegression
    cal, te = an[an["split"] == "calib"], an[an["split"] == "test"]
    x = lambda d: (direction * d["disagreement_score"].to_numpy()).reshape(-1, 1)  # noqa: E731
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x(cal), cal["ground_truth"])
    p = lr.predict_proba(x(te))[:, 1]
    y = te["ground_truth"].to_numpy()
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    rows = [{"bin_lo": bins[b], "bin_hi": bins[b + 1], "n": int((idx == b).sum()),
             "mean_pred": float(p[idx == b].mean()) if (idx == b).any() else np.nan,
             "frac_harmful": float(y[idx == b].mean()) if (idx == b).any() else np.nan} for b in range(10)]
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color="#bdbcb6", lw=0.8)
    ok = r["n"] > 0
    ax.plot(r.loc[ok, "mean_pred"], r.loc[ok, "frac_harmful"], marker="o", color=PALETTE[0])
    for _, row in r[ok].iterrows():
        ax.text(row["mean_pred"], row["frac_harmful"], f" {row['n']}", fontsize=5.5, color=MUTED)
    ax.set_xlabel("predicted P(harmful)")
    ax.set_ylabel("observed fraction harmful")
    ax.set_title(f"Reliability, test split ({bb}); Brier {M.brier(y, p):.3f}, "
                 f"ECE {M.ece(y, p, 10):.3f}\nfit on calib: a={lr.coef_[0][0]:.2f}, b={lr.intercept_[0]:.2f}"
                 f" (score = {'+' if direction > 0 else '-'}D~)", fontsize=7)
    r["coef"], r["intercept"], r["direction"] = lr.coef_[0][0], lr.intercept_[0], direction
    save(fig, cfg, f"fig10_reliability_{bb}", r)


def fig_layers(cfg, bb):
    """Layer-wise group means of D~ under safety vs neutral prefixes (exp10, family-bootstrap 95% CI)."""
    d = output_dir(cfg, "figures")
    primary = bb == cfg["backbones"]["primary"]
    p = d / ("fig12_layerwise.csv" if primary else f"fig12_layerwise_{bb}.csv")
    if not p.exists():
        return
    lm = pd.read_csv(p)
    lm = lm[lm["score"] == "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, c in zip(axes, ("safety", "neutral")):
        g = lm[lm["condition"] == c]
        for gi, grp in enumerate([x for x in GROUP_ORDER if x in set(g["group"])]):
            gg = g[g["group"] == grp]
            ax.plot(gg["layer"], gg["mean"], color=PALETTE[gi], label=GROUP_LABEL[grp], lw=1.2)
            ax.fill_between(gg["layer"], gg["ci_lo"], gg["ci_hi"], color=PALETTE[gi], alpha=0.15, lw=0)
        ax.set_title(f"{c} prefixes ({bb})")
        ax.set_xlabel("layer (0 = input embeddings)")
    axes[0].set_ylabel(r"mean $\tilde D$")
    axes[1].legend(fontsize=5.5)
    save(fig, cfg, f"fig12_layerwise_{bb}", lm)


def fig_interaction(cfg, bb):
    """Difference-in-differences Delta (safety minus neutral prefixes), final layer, with 95% CI."""
    t = output_dir(cfg, "tables") / ("exp10_interaction.csv" if bb == cfg["backbones"]["primary"]
                                     else f"exp10_interaction_{bb}.csv")
    if not t.exists():
        return
    r = pd.read_csv(t)
    r = r[r["conditioning"] == "safety"].reset_index(drop=True)
    bl = output_dir(cfg, "figures") / ("fig13_interaction_by_layer.csv" if bb == cfg["backbones"]["primary"]
                     else f"fig13_interaction_by_layer_{bb}.csv")
    if bl.exists():
        L = pd.read_csv(bl)
        L = L[L["contrast"] == "framed_jailbreak - benign"]
        fig2, ax2 = plt.subplots(figsize=(4.6, 2.6))
        for ci, (cond, lab) in enumerate([("safety", "safety personas"), ("paraphrase", "paraphrased personas")]):
            g = L[L["conditioning"] == cond].sort_values("layer")
            if g.empty:
                continue
            ax2.plot(g["layer"], g["delta"], color=PALETTE[ci], lw=1.2, label=lab)
            ax2.fill_between(g["layer"], g["ci_lo"], g["ci_hi"], color=PALETTE[ci], alpha=0.15, lw=0)
        ax2.axhline(0, color=MUTED, lw=0.8)
        ax2.set_xlabel("layer (0 = input embeddings)")
        ax2.set_ylabel(r"$\Delta^{(\ell)}$ (framed jailbreak $-$ benign)")
        ax2.set_title(f"Interaction by layer ({bb})")
        ax2.legend(fontsize=6)
        save(fig2, cfg, f"fig13b_interaction_layers_{bb}", L)
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(r) + 0.8))
    y = np.arange(len(r))
    ax.errorbar(r["delta"], y, xerr=[r["delta"] - r["ci_lo"], r["ci_hi"] - r["delta"]], fmt="o",
                color=PALETTE[0], ecolor=PALETTE[0], ms=3, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in r["contrast"]], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ = gap under safety personas $-$ gap under neutral prefixes")
    ax.set_title(f"Prompt–persona interaction, final layer ({bb})")
    save(fig, cfg, f"fig13_interaction_{bb}", r)


def fig_probe_layers(cfg, bb):
    p = output_dir(cfg, "tables") / f"exp12_probe_layers_{bb}.csv"
    q = output_dir(cfg, "tables") / ("exp10_layer_auc.csv" if bb == cfg["backbones"]["primary"]
                                     else f"exp10_layer_auc_{bb}.csv")
    if not p.exists():
        return
    pr = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(pr["layer"], pr["test_auc"], color=PALETTE[5], label="Linear probe, no personas (test split, harmful vs harmless)")
    if q.exists():
        la = pd.read_csv(q)
        ax.plot(la["layer"], la["auc_safety"], color=PALETTE[0], label=r"$\tilde D$, safety personas (framed jailbreak vs benign)")
        ax.plot(la["layer"], 1 - la["auc_safety"], color=PALETTE[0], ls="--", label=r"$-\tilde D$ (same prompts)")
    ax.axhline(0.5, color="#bdbcb6", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer (0 = input embeddings)")
    ax.set_ylabel("AUC")
    ax.set_title(f"Single-view probe vs persona disagreement ({bb})")
    ax.legend(fontsize=5.5)
    save(fig, cfg, f"fig14_probe_layers_{bb}", pr)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    style()
    backbones = list(dict.fromkeys([args.backbone, *cfg["experiments"]["exp7"]["backbones"]]))
    for bb in backbones:
        mdir = run_dir(cfg, bb, "main")
        if not (mdir / "analysis.csv").exists():
            continue
        an = pd.read_csv(mdir / "analysis.csv")
        th = json.loads((run_dir(cfg, bb) / "thresholds.json").read_text()) \
            if (run_dir(cfg, bb) / "thresholds.json").exists() else {}
        thd_p = run_dir(cfg, bb) / "thresholds_dircal.json"
        if thd_p.exists():
            thd = json.loads(thd_p.read_text())
        else:
            from core.calibration import direction_thresholds
            thd = direction_thresholds(cfg, bb, an) or {}
        direction = int(thd.get("direction", 1))
        fig_distributions(cfg, bb, an, th)
        fig_layers(cfg, bb)
        fig_interaction(cfg, bb)
        fig_probe_layers(cfg, bb)
        fig_grid(cfg, bb, an, 1, "")
        if direction != 1:
            fig_grid(cfg, bb, an, direction, " direction-calibrated")
        fig_reliability(cfg, bb, an, direction)
        ev = load_evals(cfg, bb)
        if ev.empty:
            continue
        bp = run_dir(cfg, bb, "exp12") / "scores_with_probes.csv"
        fig_roc(cfg, bb, ev, an, pd.read_csv(bp) if bp.exists() else None)
        fig_pareto(cfg, bb, ev)
        fig_latency(cfg, bb, ev)
        a = attacks(cfg, bb)
        if len(a):
            # Count only valid evasions (not paraphraser refusals), as in Table "attacks".
            from experiments.exp4_robustness import valid_evasions
            a = valid_evasions(a, "lenient")
        fig_asr(cfg, bb, a)
        fig_trajectory(cfg, bb, a, {"pso": th, "pso_dircal": thd})


if __name__ == "__main__":
    main()
