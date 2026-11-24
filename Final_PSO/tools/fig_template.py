"""Figure: template control (Exp. 13) and template-aware probe (Exp. 14), from the result tables.
usage: python tools/fig_template.py <tables_dir> <out.pdf>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T, out = Path(sys.argv[1]), Path(sys.argv[2])
BB = [("llama2", "Llama-2"), ("llama3", "Llama-3"), ("mistral", "Mistral"), ("qwen", "Qwen2.5")]
def val(df, kind, s, probe=None):
    d = df if probe is None else df[df.probe == probe]
    return float(d[(d.kind == kind) & (d.set == s)].value.iloc[0])
e13 = {b: pd.read_csv(T / f"exp13_template_control_{b}.csv") for b, _ in BB}
e14 = {b: pd.read_csv(T / f"exp14_template_aware_probe_{b}.csv") for b, _ in BB}
lg = pd.read_csv(T / "exp13_llama_guard_template_control.csv")
fh13 = [val(e13[b], "flag_rate", "benign_framed", "probe_cv_layer") for b, _ in BB]
fh14 = [val(e14[b], "flag_rate", "benign_framed_test") for b, _ in BB]
tp13 = [val(e13[b], "flag_rate", "advbench_framed_test", "probe_cv_layer") for b, _ in BB]
tp14 = [val(e14[b], "flag_rate", "advbench_framed_test") for b, _ in BB]
lg_fh, lg_tp = val(lg, "flag_rate", "benign_framed"), val(lg, "flag_rate", "advbench_framed_test")

import matplotlib.font_manager as fm
for _p in fm.findSystemFonts():
    if "termes" in _p.lower() or "nimbusroman" in _p.lower().replace("-", ""):
        fm.fontManager.addfont(_p)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C13, C14, CLG = "#E69F00", "#0072B2", "#555555"
x, w = np.arange(len(BB)), 0.36
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, a, b, ref, title, ylab in [
        (axes[0], fh13, fh14, lg_fh, "Framed harmless requests flagged (lower is better)", "false-positive rate"),
        (axes[1], tp13, tp14, lg_tp, "Framed jailbreaks detected (higher is better)", "true-positive rate")]:
    r1 = ax.bar(x - w / 2 - 0.01, a, w, color=C13, label="probe, untemplated calibration (Exp. 13)")
    r2 = ax.bar(x + w / 2 + 0.01, b, w, color=C14, label="template-aware probe (Exp. 14)")
    ax.axhline(ref, color=CLG, ls="--", lw=1.2, label="Llama Guard 3-1B")
    for rr in (r1, r2):
        for rect in rr:
            ax.text(rect.get_x() + rect.get_width() / 2, max(rect.get_height(), 0.11) + 0.015, f"{rect.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9), zorder=5)
    ax.set_xticks(x, [n for _, n in BB]); ax.set_ylim(0, 1.08); ax.set_ylabel(ylab); ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
axes[0].axhline(0.10, color="#999999", ls=":", lw=1.0, label="criterion for framed harmless (0.10)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(out, bbox_inches="tight")
pd.DataFrame({"backbone": [n for _, n in BB], "fp_exp13": fh13, "fp_exp14": fh14, "tp_exp13": tp13,
              "tp_exp14": tp14, "fp_llama_guard": lg_fh, "tp_llama_guard": lg_tp}).to_csv(
    out.with_suffix(".csv"), index=False)
print("wrote", out)
