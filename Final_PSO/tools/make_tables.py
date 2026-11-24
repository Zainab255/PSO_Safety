"""Build LaTeX tables for the paper from merged output tables (no hand-typed numbers)."""
import sys, pandas as pd
from pathlib import Path
T = Path(sys.argv[1]); OUT = Path(sys.argv[2])
BB = [("llama2","Llama-2-7B"),("llama3","Llama-3-8B"),("mistral","Mistral-7B"),("qwen","Qwen2.5-7B")]
def tb(name, bb): 
    for p in ([T/f"{name}.csv", T/f"{name}_{bb}.csv"] if bb=="llama2" else [T/f"{name}_{bb}.csv"]):
        if p.exists(): return pd.read_csv(p)
    return None
f3 = lambda x: f"{x:.3f}".replace("-0.000","0.000")
def sgn(x, d=3): return ("$+" if x>=0 else "$-")+f"{abs(x):.{d}f}$"
def ci(r): return f"{sgn(r.delta)} [{sgn(r.ci_lo)}, {sgn(r.ci_hi)}]".replace("$ [$","$ [$")
rows=[]
for bb,lab in BB:
    it = tb("exp10_interaction", bb); lm = tb("exp10_layer_means_final", bb); ex = tb("exp12_baselines", bb)
    cells=[lab]
    if lm is not None:
        g = lm[(lm.condition=="safety")&(lm.score=="raw")].set_index("group")["mean"]
        cells += [f3(g["benign"]), f3(g["framed_jailbreak"]), f3(g["direct_advbench"])]
    else: cells += ["--"]*3
    if it is not None:
        s = it.set_index(["conditioning","contrast"])
        cells += [ci(s.loc[("safety","framed_jailbreak - benign")]), ci(s.loc[("paraphrase","framed_jailbreak - benign")]),
                  ci(s.loc[("safety","framed - direct (same goal, paired)")])]
    else: cells += ["--"]*3
    rows.append(" & ".join(cells)+r"\\")
(OUT/"tab_backbones_rows.tex").write_text("\n".join(rows)+"\n")
# detector table across backbones
names = [("msds","$\\tilde D$ (pre-specified)"),("msds_dircal","$-\\tilde D$ (exploratory)"),("probe_pairwise","Probe on $(s_{RB},s_{RJ},s_{BJ})$"),
         ("probe_red","Probe, Red view"),("probe_3views","Probe, three views"),("probe_final","Probe, bare prompt, final layer"),
         ("probe_last","Probe, bare prompt, CV layer"),("hiddendetect","HiddenDetect-style"),("log_ppl","Perplexity"),
         ("smoothllm","SmoothLLM"),("llama_guard","Llama Guard 3-1B")]
rows=[]
for key,lab in names:
    cells=[lab]; any_=False
    for bb,_ in BB:
        ex = tb("exp12_baselines", bb)
        if ex is not None and key in set(ex.method):
            r = ex[ex.method==key].iloc[0]; any_=True
            cells.append(f"{r.test_auc:.2f} & {r.test_tpr:.2f} & {r.test_fpr:.2f}")
        else: cells.append("-- & -- & --")
    if any_: rows.append(" & ".join(cells)+r"\\")
(OUT/"tab_detectors_rows.tex").write_text("\n".join(rows)+"\n")
# transfer table: AUC on transfer sets vs XSTest-safe test negatives
rows=[]
for key,lab in [("msds_dircal","$-\\tilde D$"),("probe_last","Probe, bare prompt"),("probe_3views","Probe, three views"),("hiddendetect","HiddenDetect-style"),("llama_guard","Llama Guard 3-1B")]:
    for bb,bl in BB:
        ex = tb("exp12_baselines", bb)
        if ex is None or key not in set(ex.method): continue
        r = ex[ex.method==key].iloc[0]
        cells=[lab, bl]+[f"{r[f'auc_{b}']:.2f} / {r[f'tpr_{b}']:.2f}" for b in ("advbench_direct","advbench_wild","harmbench","jailbreakbench")]
        rows.append(" & ".join(cells)+r"\\")
(OUT/"tab_transfer_rows.tex").write_text("\n".join(rows)+"\n")
print(open(OUT/"tab_backbones_rows.tex").read()); print(open(OUT/"tab_detectors_rows.tex").read()); print(open(OUT/"tab_transfer_rows.tex").read())

# ---- adaptive attacks (Llama-2): ASR@1 / ASR@10 per attacker x defence -------------------
_a = T / "exp4_asr_validity.csv"
if _a.exists():
    a = pd.read_csv(_a)
    meths = [("keyword_history", "Keyword"), ("llm_judge", "LLM-judge"), ("self_critique", "Self-Critique"),
             ("pso", "PSO"), ("pso_dircal", "PSO, $-\\tilde D$")]
    names = {"naive": "Template (naive)", "paraphrase": "Paraphrase", "adaptive": "Adaptive mutation",
             "score_aware": "Score-aware$^{\\dagger}$"}
    lines = []
    for lev, nm in names.items():
        cells = []
        for m, _ in meths:
            r = a[(a.attacker == lev) & (a.method == m)]
            cells.append("--" if r.empty else f"{r.lenient_asr_at_1.iloc[0]:.2f} / {r.lenient_asr_at_10.iloc[0]:.2f}")
        lines.append(nm + " & " + " & ".join(cells) + r"\\")
    (OUT / "tab_attacks_rows.tex").write_text("\n".join(lines) + "\n")
    print((OUT / "tab_attacks_rows.tex").read_text())

# ---- template control (Exp. 13): CV-layer bare-prompt probe, thresholds from Exp. 12 calibration -----
_rows = []
for bb, bl in BB:
    f = T / f"exp13_template_control_{bb}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d = d[d.probe == "probe_cv_layer"]
    v = lambda kind, s: d[(d.kind == kind) & (d.set == s)].value.iloc[0]  # noqa: E731
    cells = [bl, str(int(d.layer.iloc[0]))]
    cells += [f"{v('auc', s):.2f}" for s in ("advbench_framed_test vs benign_framed", "advbench_wild vs benign_wild",
                                            "advbench_direct vs benign_direct", "jbb_harmful_framed vs jbb_benign_framed")]
    cells += [f"{v('flag_rate', s):.2f}" for s in ("advbench_framed_test", "benign_framed", "benign_wild",
                                                  "benign_direct", "jbb_benign_framed", "xstest_safe_test")]
    _rows.append(" & ".join(cells) + r"\\")
if _rows:
    (OUT / "tab_template_rows.tex").write_text("\n".join(_rows) + "\n")
    print((OUT / "tab_template_rows.tex").read_text())
_lg = T / "exp13_llama_guard_template_control.csv"
if _lg.exists() and _rows:
    d = pd.read_csv(_lg)
    v = lambda kind, s: d[(d.kind == kind) & (d.set == s)].value.iloc[0]  # noqa: E731
    cells = ["\\midrule Llama Guard 3-1B$^{\\ast}$", "--"]
    cells += [f"{v('auc', s):.2f}" for s in ("advbench_framed_test vs benign_framed", "advbench_wild vs benign_wild",
                                            "advbench_direct vs benign_direct", "jbb_harmful_framed vs jbb_benign_framed")]
    cells += [f"{v('flag_rate', s):.2f}" for s in ("advbench_framed_test", "benign_framed", "benign_wild",
                                                  "benign_direct", "jbb_benign_framed", "xstest_safe_test")]
    _rows.append(" & ".join(cells) + r"\\")
    (OUT / "tab_template_rows.tex").write_text("\n".join(_rows) + "\n")
    print(_rows[-1])

# ---- template-aware probe (Exp. 14, exploratory) ------------------------------------------------
_r14 = []
for bb, bl in BB:
    f = T / f"exp14_template_aware_probe_{bb}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    v = lambda kind, s: d[(d.kind == kind) & (d.set == s)].value.iloc[0]  # noqa: E731
    ok = (v("flag_rate", "benign_framed_test") <= 0.10 and v("flag_rate", "advbench_framed_test") >= 0.90
          and v("flag_rate", "xstest_safe_test") <= 0.10)
    cells = [bl, str(int(d.layer.iloc[0]))]
    cells += [f"{v('auc', s):.3f}" for s in ("main corpus test", "advbench_framed_test vs benign_framed_test",
                                            "jbb_harmful_framed vs jbb_benign_framed")]
    cells += [f"{v('flag_rate', s):.2f}" for s in ("advbench_framed_test", "advbench_wild_test", "jbb_harmful_framed",
                                                  "xstest_safe_test", "benign_framed_test", "benign_wild_test",
                                                  "jbb_benign_framed")]
    cells.append("pass" if ok else "fail")
    _r14.append(" & ".join(cells) + r"\\")
if _r14:
    (OUT / "tab_template_aware_rows.tex").write_text("\n".join(_r14) + "\n")
    print((OUT / "tab_template_aware_rows.tex").read_text())

# ---- Exp. 15: independent data (all backbones) and adaptive attacks on the template-aware probe -------
_r15 = []
_lg15 = T / "exp15_llama_guard_independent.csv"
for bb, bl in BB:
    f = T / f"exp15_independent_{bb}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    for probe, pl in (("untemplated", "Probe, untemplated calib."), ("template_aware", "Probe, template-aware")):
        q = d[(d.probe == probe) & (d.kind == "flag_rate")].set_index("set").value
        gcg = (q["indep_gcg_llama2"] * 100 + q["indep_gcg_vicuna"] * 100) / 200
        _r15.append(" & ".join([pl, bl, f"{q['indep_pair']:.2f}", f"{gcg:.2f}", f"{q['orbench_toxic']:.2f}",
                                f"{q['orbench_hard']:.2f}"]) + r"\\")
if _lg15.exists() and _r15:
    q = pd.read_csv(_lg15); q = q[q.kind == "flag_rate"].set_index("set").value
    gcg = (q["indep_gcg_llama2"] + q["indep_gcg_vicuna"]) / 2
    _r15.append("\\midrule Llama Guard 3-1B & -- & " + " & ".join(f"{v:.2f}" for v in (q["indep_pair"], gcg, q["orbench_toxic"], q["orbench_hard"])) + r"\\")
if _r15:
    (OUT / "tab_independent_rows.tex").write_text("\n".join(_r15) + "\n")
    print((OUT / "tab_independent_rows.tex").read_text())
