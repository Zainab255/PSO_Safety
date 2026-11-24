#!/usr/bin/env python3
"""Figures and LaTeX table rows for Experiments 16-18 and the two radar summaries.

Every number is read from the saved output tables; nothing is typed by hand.

Usage
    python tools/make_figs_exp16_18.py <tables_dir> <figure_dir> <latex_rows_dir>

<tables_dir> must contain the exp16_*, exp17_*, exp18_* tables, exp17_combination_ci.csv,
summary_table18.csv and exp4_asr_validity.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T, F, R = (Path(a) for a in sys.argv[1:4])
F.mkdir(parents=True, exist_ok=True)
R.mkdir(parents=True, exist_ok=True)

BB = [("llama2", "Llama-2-7B"), ("llama3", "Llama-3-8B"), ("mistral", "Mistral-7B"), ("qwen", "Qwen2.5-7B")]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BB_COL = {b: SLOTS[i] for i, (b, _) in enumerate(BB)}
BB_MARK = {"llama2": "o", "llama3": "s", "mistral": "^", "qwen": "D"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def cat(prefix):
    return pd.concat([pd.read_csv(T / f"{prefix}_{b}.csv") for b, _ in BB if (T / f"{prefix}_{b}.csv").exists()],
                     ignore_index=True)


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


# ------------------------------------------------------------------ Exp 17: few-shot coverage
few = cat("exp17_fewshot")
few_m = few.groupby(["backbone", "held_out", "subset", "k"]).flag_rate.agg(["mean", "std"]).reset_index()
PANELS = [("HarmBench", "H3_harmbench", ["harmbench"], 1), ("GCG (two target models)", "H4_jbb", ["indep_gcg_llama2", "indep_gcg_vicuna"], 1),
          ("PAIR", "H4_jbb", ["indep_pair"], 1), ("OR-Bench toxic", "H5_orbench_toxic", ["orbench_toxic"], 1),
          ("OR-Bench-Hard (harmless)", "B3_orbench_hard", ["orbench_hard"], 0), ("JBB benign, framed (harmless)", "B4_jbb_benign", ["jbb_benign_framed"], 0)]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
for ax, (title, grp, subs, lab) in zip(axes.ravel(), PANELS):
    for b, bl in BB:
        d = few_m[(few_m.backbone == b) & (few_m.held_out == grp) & (few_m.subset.isin(subs))]
        d = d.groupby("k")["mean"].mean()
        ax.plot(d.index, d.values, marker=BB_MARK[b], ms=4, color=BB_COL[b], label=bl)
    ax.axhline(0.80 if lab else 0.30, color=INK_2, ls="--", lw=0.9)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 5, 10, 25, 50])
    ax.set_ylabel("detection rate" if lab else "flag rate (lower is better)", fontsize=8)
fig.supxlabel("$k$ = number of calibration families of the held-out source added to training", fontsize=8.5)
axes[0, 0].legend(loc="lower right", fontsize=7)
save(fig, "fig16_coverage")

# Table rows: few-shot, Llama-2 per k, and k=25 range over the other backbones
SETS = [("HarmBench", "H3_harmbench", ["harmbench"]), ("GCG (Llama-2 target)", "H4_jbb", ["indep_gcg_llama2"]),
        ("GCG (Vicuna target)", "H4_jbb", ["indep_gcg_vicuna"]), ("PAIR", "H4_jbb", ["indep_pair"]),
        ("JBB behaviours (direct)", "H4_jbb", ["jailbreakbench"]), ("JBB behaviours (framed)", "H4_jbb", ["jbb_harmful_framed"]),
        ("OR-Bench toxic", "H5_orbench_toxic", ["orbench_toxic"]), (r"\midrule OR-Bench-Hard$^{\dagger}$", "B3_orbench_hard", ["orbench_hard"]),
        (r"JBB benign, framed$^{\dagger}$", "B4_jbb_benign", ["jbb_benign_framed"])]
rows = []
for lab, grp, subs in SETS:
    cells = [lab]
    d = few_m[(few_m.held_out == grp) & few_m.subset.isin(subs)]
    l2 = d[d.backbone == "llama2"].set_index("k")
    for k in (0, 5, 10, 25, 50):
        r = l2.loc[k]
        cells.append(f"{r['mean']:.2f}" + (f"\\,{{\\tiny$\\pm${r['std']:.2f}}}" if k and pd.notna(r['std']) else ""))
    oth = d[(d.backbone != "llama2")]
    for k in (0, 25):
        v = oth[oth.k == k]["mean"]
        cells.append(f"{v.min():.2f}--{v.max():.2f}")
    rows.append(" & ".join(cells) + r"\\")
(R / "tab_fewshot_rows.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ Exp 17: operators
ops = cat("exp17_operators")
OPS = [("ref_plain", "plain goal$^{\\ast}$"), ("ref_template", "JailbreakHub template$^{\\ast}$"), ("new_template", "new templates"),
       ("char_insertion", "character insertion"), ("word_split", "word splitting"), ("payload_split", "payload splitting"),
       ("base64", "base64"), ("rot13", "ROT13"), ("leetspeak", "leetspeak")]
rows = []
for k, lab in OPS:
    cells = [lab]
    for b, _ in BB:
        r = ops[(ops.backbone == b) & (ops.operator == k)].iloc[0]
        cells.append(f"{r.detection:.2f} & {r.benign_flag_rate:.2f} & {r.auc:.2f}")
    rows.append(" & ".join(cells) + r"\\")
(R / "tab_operators_rows.tex").write_text("\n".join(rows) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
ylab = [l.replace("$^{\\ast}$", " (ref.)") for _, l in OPS]
ypos = np.arange(len(OPS))[::-1]
for ax, col, title in zip(axes, ["detection", "benign_flag_rate", "auc"],
                          ["detection (harmful)", "flag rate (harmless)", "AUC"]):
    for j, (b, bl) in enumerate(BB):
        v = [ops[(ops.backbone == b) & (ops.operator == k)][col].iloc[0] for k, _ in OPS]
        ax.scatter(v, ypos + (j - 1.5) * 0.15, s=16, marker=BB_MARK[b], color=BB_COL[b], label=bl, zorder=3)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlim(-0.03, 1.03)
    if col == "detection":
        ax.axvline(0.80, color=INK_2, ls="--", lw=0.9)
    if col == "benign_flag_rate":
        ax.axvline(0.20, color=INK_2, ls="--", lw=0.9)
    if col == "auc":
        ax.axvline(0.50, color=INK_2, ls=":", lw=0.9)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(ylab, fontsize=7.5)
axes[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
save(fig, "fig17_operators")

# ------------------------------------------------------------------ Exp 16/17: in-distribution, LG, combination
ind = cat("exp17_indist_dedup")
comb = pd.read_csv(T / "exp17_combination_ci.csv")
SETS2 = [("advbench_framed", "AdvBench, framed"), ("advbench_wild", "AdvBench, in-the-wild"), ("advbench_direct", "AdvBench, direct"),
         ("xstest_unsafe", "XSTest unsafe"), ("harmbench", "HarmBench"), ("jailbreakbench", "JBB behaviours"),
         ("jbb_harmful_framed", "JBB behaviours, framed"), ("indep_gcg_llama2", "GCG (Llama-2 target)"),
         ("indep_gcg_vicuna", "GCG (Vicuna target)"), ("indep_pair", "PAIR"), ("orbench_toxic", "OR-Bench toxic"),
         ("xstest_safe", r"\midrule XSTest safe$^{\dagger}$"), ("benign_direct", r"Alpaca, direct$^{\dagger}$"),
         ("benign_framed", r"Alpaca, framed$^{\dagger}$"), ("benign_wild", r"Alpaca, in-the-wild$^{\dagger}$"),
         ("orbench_hard", r"OR-Bench-Hard$^{\dagger}$"), ("jbb_benign_framed", r"JBB benign, framed$^{\dagger}$")]
rows = []
for k, lab in SETS2:
    cells = [lab]
    for b, _ in BB:
        r = ind[(ind.backbone == b) & (ind.variant == "all") & (ind.set == k)]
        cells.append(f2(r.flag_rate.iloc[0]) if len(r) else "--")
    c = comb[(comb.backbone == "llama2") & (comb.set == k)].iloc[0]
    cells += [f"{c.llama_guard:.2f}", f"{c.probe_or_lg:.2f}\\,{{\\tiny[{c.or_lo:.2f}, {c.or_hi:.2f}]}}"]
    n = int(ind[(ind.backbone == "llama2") & (ind.variant == "all") & (ind.set == k)].n.iloc[0])
    rows.insert(len(rows), " & ".join([cells[0], str(n)] + cells[1:]) + r"\\")
a = ind[(ind.variant == "all") & (ind.set == "AUC all test families")].set_index("backbone").flag_rate
rows.append(r"\midrule AUC, all test families & " + str(int(ind[(ind.backbone == 'llama2') & (ind.set == 'AUC all test families')].n.iloc[0]))
            + " & " + " & ".join(f"{a[b]:.3f}" for b, _ in BB) + r" & -- & --\\")
(R / "tab_diverse_rows.tex").write_text("\n".join(rows) + "\n")

# combination on all backbones: FPR on XSTest safe, OR-Hard, JBB benign; minimum detection over independent sets
rows = []
for b, bl in BB:
    c = comb[comb.backbone == b].set_index("set")
    indep = ["indep_pair", "indep_gcg_llama2", "indep_gcg_vicuna", "orbench_toxic", "xstest_unsafe", "harmbench"]
    cells = [bl, f"{c.loc[indep, 'probe_or_lg'].min():.2f}"]
    for s in ("xstest_safe", "orbench_hard", "jbb_benign_framed"):
        cells.append(f"{c.loc[s, 'probe']:.2f} / {c.loc[s, 'llama_guard']:.2f} / {c.loc[s, 'probe_or_lg']:.2f}")
    ok = (c.loc[indep, "probe_or_lg"] >= c.loc[indep, "llama_guard"] - 1e-9).all() and \
        c.loc["xstest_safe", "probe_or_lg"] <= c.loc["xstest_safe", "llama_guard"] + 0.05 + 1e-9
    cells.append("pass" if ok else "fail")
    rows.append(" & ".join(cells) + r"\\")
(R / "tab_combination_rows.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ Exp 16 LOSO (Llama-2, all groups)
lo = pd.read_csv(T / "exp16_loso_llama2.csv")
GL = [("H1_advbench", "all", "H1 AdvBench (framed, in-the-wild, direct)"), ("H2_xstest_unsafe", "all", "H2 XSTest unsafe"),
      ("H3_harmbench", "all", "H3 HarmBench"), ("H4_jbb", "all", "H4 JailbreakBench, all"),
      ("H4_jbb", "indep_gcg_llama2", r"\quad GCG (Llama-2 target)"), ("H4_jbb", "indep_gcg_vicuna", r"\quad GCG (Vicuna target)"),
      ("H4_jbb", "indep_pair", r"\quad PAIR"), ("H5_orbench_toxic", "all", "H5 OR-Bench toxic"),
      ("B1_xstest_safe", "all", r"\midrule B1 XSTest safe$^{\dagger}$"), ("B2_alpaca", "all", r"B2 Alpaca (3 forms)$^{\dagger}$"),
      ("B3_orbench_hard", "all", r"B3 OR-Bench-Hard$^{\dagger}$"), ("B4_jbb_benign", "all", r"B4 JBB benign$^{\dagger}$")]
rows = []
for g, s, lab in GL:
    r = lo[(lo.held_out == g) & (lo.subset == s)].iloc[0]
    rows.append(f"{lab} & {int(r.n)} & {r.flag_rate:.2f}\\\\")
(R / "tab_loso_rows.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ Exp 18a: encodings
enc = cat("exp18_encodings")
COND = [("base", "base probe"), ("loeo", "LOEO"), ("all_encodings", "all encodings"),
        ("canonicaliser", "canonicaliser"), ("canonicaliser+loeo", "canon.\\ + LOEO")]
ENC = [("base64", "base64"), ("rot13", "ROT13"), ("leetspeak", "leetspeak"), ("hex", "hex")]
rows = []
for e, el in ENC:
    for i, (c, cl) in enumerate(COND):
        cells = [el if i == 0 else "", cl]
        for b, _ in BB:
            r = enc[(enc.backbone == b) & (enc.encoding == e) & (enc.condition == c)].iloc[0]
            cells.append(f"{r.detection:.2f} & {r.benign_flag_rate:.2f} & {r.auc:.2f}")
        rows.append(" & ".join(cells) + r"\\")
    rows.append(r"\midrule" if e != "hex" else "")
(R / "tab_encodings_rows.tex").write_text("\n".join(rows) + "\n")

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
x = np.arange(len(COND))
for ax, (e, el) in zip(axes, ENC):
    for j, (b, bl) in enumerate(BB):
        v = [enc[(enc.backbone == b) & (enc.encoding == e) & (enc.condition == c)].auc.iloc[0] for c, _ in COND]
        ax.scatter(x + (j - 1.5) * 0.12, v, s=16, marker=BB_MARK[b], color=BB_COL[b], label=bl, zorder=3)
    m = [enc[(enc.encoding == e) & (enc.condition == c)].auc.mean() for c, _ in COND]
    ax.bar(x, m, width=0.62, color="#cde2fb", edgecolor="white", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["base", "LOEO", "all enc.", "canon.", "canon.+LOEO"], fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_title(el, fontsize=8.5, color=INK)
    ax.axhline(0.5, color=INK_2, ls=":", lw=0.9)
    ax.set_ylim(0.3, 1.03)
axes[0].set_ylabel("AUC, encoded harmful vs. harmless")
axes[3].legend(loc="lower right", fontsize=6.5)
save(fig, "fig18_encodings")

se = cat("exp18_enc_side_effects")
SE = [("xstest_safe", "XSTest safe"), ("orbench_hard", "OR-Bench-Hard"), ("jbb_benign_framed", "JBB benign, framed"),
      ("benign_framed", "Alpaca, framed"), ("xstest_unsafe", "XSTest unsafe"), ("indep_pair", "PAIR"),
      ("orbench_toxic", "OR-Bench toxic"), ("harmbench", "HarmBench")]
rows = []
for s, sl in SE:
    cells = [sl]
    for b, _ in BB:
        d = se[(se.backbone == b) & (se.set == s)].set_index("condition").flag_rate
        cells.append(f"{d['base']:.2f} & {d['all_encodings']:.2f}")
    rows.append(" & ".join(cells) + r"\\")
(R / "tab_enc_side_rows.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ Exp 18b: updating from evasions
ua = pd.read_csv(T / "exp18_update_attacks_llama2.csv")
uf = pd.read_csv(T / "exp18_update_flags_llama2.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ATT = [("naive", "template (naive)"), ("adaptive", "adaptive"), ("score_aware", "score-aware")]
for j, (a, al) in enumerate(ATT):
    d = ua[(ua["mode"] == "strict") & (ua.attacker == a)].sort_values("round")
    axes[0].errorbar(d["round"] + (j - 1) * 0.05, d.asr_at_10_mean, yerr=d.asr_at_10_std, marker="o", ms=4,
                     color=SLOTS[j], capsize=2, label=al)
r0 = ua[(ua["mode"] == "strict") & (ua.attacker == "score_aware") & (ua["round"] == 0)].asr_at_10_mean.iloc[0]
axes[0].axhline(r0 / 2, color=INK_2, ls="--", lw=0.9)
axes[0].text(2.05, r0 / 2, "criterion", fontsize=7, color=INK_2, va="bottom", ha="right")
axes[0].set_xticks([0, 1, 2])
nadd = ua.groupby("round").n_added_harmful.first()
axes[0].set_xticklabels([f"round {r}\n(+{nadd[r]} evasions)" for r in (0, 1, 2)], fontsize=7)
axes[0].set_ylabel("strict ASR@10")
axes[0].set_ylim(0, None)
axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
HS = [("xstest_safe", "XSTest safe"), ("orbench_hard", "OR-Bench-Hard"), ("jbb_benign_framed", "JBB benign, framed"),
      ("benign_framed", "Alpaca, framed"), ("indep_pair", "PAIR (harmful)"), ("advbench_framed", "AdvBench framed (harmful)")]
for j, (s, sl) in enumerate(HS):
    d = uf[uf.set == s].sort_values("round")
    axes[1].plot(d["round"], d.flag_rate, marker="o", ms=4, color=(SLOTS[j + 3] if j < 4 else (SLOTS[7] if j == 4 else SLOTS[0])),
                 ls="-" if d.label.iloc[0] == 0 else "--", label=sl)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["round 0", "round 1", "round 2"], fontsize=7)
axes[1].set_ylabel("flag rate")
axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "fig19_update")

rows = []
for r in (0, 1, 2):
    cells = [str(r), str(int(nadd[r]))]
    for a, _ in ATT:
        d = ua[(ua["mode"] == "strict") & (ua.attacker == a) & (ua["round"] == r)].iloc[0]
        cells.append(f"{d.asr_at_10_mean:.3f}")
    d = ua[(ua["mode"] == "lenient") & (ua.attacker == "score_aware") & (ua["round"] == r)].iloc[0]
    cells.append(f"{d.asr_at_10_mean:.3f}")
    f = uf[uf["round"] == r].set_index("set").flag_rate
    cells += [f"{f['xstest_safe']:.3f}", f"{f['orbench_hard']:.3f}", f"{f['jbb_benign_framed']:.3f}",
              f"{f['indep_pair']:.3f}", f"{f['AUC all test families']:.3f}"]
    rows.append(" & ".join(cells) + r"\\")
(R / "tab_update_rows.tex").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------ radar helpers
def radar(ax, series, axes_names, styles):
    ang = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    ang += ang[:1]
    for name, vals in series.items():
        v = list(vals) + [vals[0]]
        st = styles[name]
        ax.plot(ang, v, color=st["c"], lw=st.get("lw", 1.4), ls=st.get("ls", "-"), label=name, zorder=3)
        if st.get("fill"):
            ax.fill(ang, v, color=st["c"], alpha=0.12, zorder=2)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes_names, fontsize=7.5, color=INK)
    ax.tick_params(axis="x", pad=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=6, color=INK_2)
    ax.set_rlabel_position(90 + 180 / len(axes_names))
    ax.spines["polar"].set_color(INK_2)
    ax.grid(color=GRID, lw=0.6)


# Radar 1: PSO and the generation-based baselines (Llama-2, main test split)
sm = pd.read_csv(T / "summary_table18.csv").set_index("method")
# robustness from valid evasions only (lenient rule, adaptive attacker), as in the attack table of the paper
_v = pd.read_csv(T / "exp4_asr_validity.csv")
_v = _v[_v.attacker == "adaptive"].set_index("method").lenient_asr_at_10
sm["asr_at_10"] = [_v.get(m, np.nan) for m in sm.index]
AX1 = ["Detection\n(TPR)", "Utility\n(1-FPR)", "Speed\n(rel. to PSO)", "Robustness\n(1-ASR@10)", "Discrimination\n(AUC)"]
pso_ms = sm.loc["pso", "wrapper_ms"]
names = [("pso", "PSO (pre-specified)"), ("keyword_history", "Keyword+History"), ("llm_judge", "LLM-as-a-Judge"),
         ("self_critique", "Self-Critique")]
series, styles = {}, {}
for i, (m, lab) in enumerate(names):
    r = sm.loc[m]
    series[lab] = [r.tpr, 1 - r.fpr, min(1.0, pso_ms / r.wrapper_ms), 1 - r.asr_at_10, r.auc]
    styles[lab] = {"c": SLOTS[i], "lw": 2.4 if m == "pso" else 1.3, "ls": "-" if m == "pso" else "--", "fill": m == "pso"}
fig = plt.figure(figsize=(3.6, 3.5))
ax = fig.add_subplot(111, polar=True)
radar(ax, series, AX1, styles)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7)
save(fig, "fig11_radar")

# Radar 2: deployable detectors on Llama-2 (held-out test families; higher is better on every axis)
c = comb[comb.backbone == "llama2"].set_index("set")
AX2 = ["PAIR", "GCG", "OR-Bench\ntoxic", "XSTest\nunsafe", "1 - FPR\nXSTest safe", "1 - FPR\nAlpaca framed",
       "1 - FPR\nOR-Bench-Hard", "1 - FPR\nJBB benign"]


def vec(get):
    g = lambda s: get(s)
    return [g("indep_pair"), (g("indep_gcg_llama2") + g("indep_gcg_vicuna")) / 2, g("orbench_toxic"), g("xstest_unsafe"),
            1 - g("xstest_safe"), 1 - g("benign_framed"), 1 - g("orbench_hard"), 1 - g("jbb_benign_framed")]


series = {"Llama Guard 3-1B": vec(lambda s: c.loc[s, "llama_guard"]),
          "Diverse probe (ours)": vec(lambda s: c.loc[s, "probe"]),
          "Diverse probe OR Llama Guard": vec(lambda s: c.loc[s, "probe_or_lg"])}
styles = {"Llama Guard 3-1B": {"c": SLOTS[1], "ls": "--"}, "Diverse probe (ours)": {"c": SLOTS[0], "lw": 2.2, "fill": True},
          "Diverse probe OR Llama Guard": {"c": SLOTS[2], "ls": "-."}}
fig = plt.figure(figsize=(3.8, 3.7))
ax = fig.add_subplot(111, polar=True)
radar(ax, series, AX2, styles)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1, fontsize=7)
save(fig, "fig20_radar_detectors")
pd.DataFrame(series, index=[a.replace("\n", " ") for a in AX2]).round(3).to_csv(F / "fig20_radar_detectors.csv")

# ------------------------------------------------------------------ console summary
print((R / "tab_combination_rows.tex").read_text())
print((R / "tab_update_rows.tex").read_text())
print(pd.DataFrame(series, index=[a.replace("\n", " ") for a in AX2]).round(3))
