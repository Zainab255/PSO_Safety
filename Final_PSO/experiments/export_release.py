#!/usr/bin/env python3
"""Assemble the data / results release (CPU only).

    python -m experiments.export_release --out release/

release/data/
    adversarial_framed.csv      500 framed AdvBench goals: goal, prompt, framing label, template id, split
    adversarial_wild.csv        the same goals inside in-the-wild jailbreak templates
    direct_harmful_advbench.csv the same goals, unframed (topic-matched direct-harmful control)
    framing_templates.csv       the 12 framing templates (4 per framing)
    xstest_assignment.csv       every XSTest prompt with its type, our sub-category and its split
    ambiguous_prompts.csv       the 150 ambiguity-control prompts (written for this study)
    multiturn_sequences.csv     the constructed escalation / injection / harmless sequences, per seed
    DATASHEET.md                construction, counts, licences, ethics note
release/results/
    per_prompt_<backbone>.csv   one row per main-corpus prompt: signals, split, PSO decisions (both
                                directions), every baseline score, generated responses' judgements
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.loaders import build_multiturn_sequences, load_dataset, split_of  # noqa: E402
from config import load_config, output_dir  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "release"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    frac = cfg["evaluation"].get("calibration_fraction", 0.3)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw"

    for src, dst in (("advbench_framed.csv", "adversarial_framed.csv"), ("advbench_wild.csv", "adversarial_wild.csv"),
                     ("advbench_direct.csv", "direct_harmful_advbench.csv"), ("ambiguous_prompts.csv", "ambiguous_prompts.csv"),
                     ("benign_direct.csv", "template_control/benign_direct.csv"),
                     ("benign_framed.csv", "template_control/benign_framed.csv"),
                     ("benign_wild.csv", "template_control/benign_wild.csv"),
                     ("jbb_benign_framed.csv", "template_control/jbb_benign_framed.csv"),
                     ("jbb_harmful_framed.csv", "template_control/jbb_harmful_framed.csv")):
        if (raw / src).exists():
            d = pd.read_csv(raw / src)
            if "family" in d:
                d["split"] = [split_of(f, frac) for f in d["family"]]
            (out / "data" / dst).parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(out / "data" / dst, index=False)

    sys.path.insert(0, str(ROOT / "data"))
    import prepare_data as P  # noqa: E402
    pd.DataFrame([{"framing": f, "template_id": f"{f}_{i}", "template": t}
                  for f, ts in P.FRAMINGS.items() for i, t in enumerate(ts)]).to_csv(
        out / "data" / "framing_templates.csv", index=False)

    x = pd.read_csv(raw / "xstest_prompts.csv")
    xc = cfg["datasets"]["xstest"]
    x["subcategory"] = x["type"].map(xc["safe_subcategories"]).where(x["label"] == "safe", xc["unsafe_category"])
    x["prompt_id"] = [f"xstest-{i}" for i in x["id"]]
    x["split"] = [split_of(p, frac) for p in x["prompt_id"]]
    x.to_csv(out / "data" / "xstest_assignment.csv", index=False)

    safe, adv = load_dataset(cfg, "xstest_safe"), load_dataset(cfg, "advbench")
    rows = []
    for seed in cfg["evaluation"]["seeds"]:
        for kind, seqs in build_multiturn_sequences(safe, adv, cfg, seed).items():
            for i, seq in enumerate(seqs):
                for t, r in enumerate(seq, 1):
                    rows.append({"seed": seed, "kind": kind, "sequence": i, "turn": t, "prompt_id": r.prompt_id,
                                 "harmful_turn": int(r.harmful), "text": r.text})
    pd.DataFrame(rows).to_csv(out / "data" / "multiturn_sequences.csv", index=False)

    # Per-prompt results -----------------------------------------------------------------
    runs = output_dir(cfg, "runs")
    for bbdir in sorted(p for p in runs.iterdir() if p.is_dir()):
        a = bbdir / "main" / "analysis.csv"
        if not a.exists():
            continue
        df = pd.read_csv(a)
        from experiments.exp5_sensitivity import pipeline_decisions
        from core.calibration import direction_thresholds, load_thresholds
        th = load_thresholds(cfg, bbdir.name)
        if th:
            df["pso_decision_prespecified"], df["pso_decided_by"] = pipeline_decisions(df, th["tau_s"], th["tau_b"])
        thd = direction_thresholds(cfg, bbdir.name, df)
        if thd:
            f = df.copy()
            f["disagreement_score"] = thd["direction"] * f["disagreement_score"]
            df["pso_decision_direction_calibrated"] = pipeline_decisions(f, thd["tau_s"], thd["tau_b"])[0]
        for e in sorted((bbdir / "main").glob("eval_seed*_*.csv")):
            ev = pd.read_csv(e)
            tag = e.stem.replace("eval_", "")
            df = df.merge(ev[["prompt_id", "decision", "output_unsafe"]].rename(
                columns={"decision": f"decision_{tag}", "output_unsafe": f"output_unsafe_{tag}"}), on="prompt_id", how="left")
        b = bbdir / "exp12" / "scores_with_probes.csv"
        if b.exists():
            bs = pd.read_csv(b)
            keep = [c for c in bs.columns if c in ("log_ppl", "hiddendetect", "smoothllm", "llama_guard")
                    or c.startswith("probe_")]
            df = df.merge(bs[["prompt_id"] + keep], on="prompt_id", how="left")
        df.to_csv(out / "results" / f"per_prompt_{bbdir.name}.csv", index=False)
        for j in ("thresholds.json", "thresholds_dircal.json"):
            if (bbdir / j).exists():
                shutil.copy(bbdir / j, out / "results" / f"{bbdir.name}_{j}")
    tables = output_dir(cfg, "tables")
    if tables.exists():
        shutil.copytree(tables, out / "results" / "tables", dirs_exist_ok=True)
    # Every query of every attack run (Exp. 4), and the datasheet.
    for bbdir in sorted(output_dir(cfg, "runs").glob("*")):
        e13 = bbdir / "exp13" / "scores.csv"
        if e13.exists():
            shutil.copy(e13, out / "results" / f"template_control_scores_{bbdir.name}.csv")
        e14 = bbdir / "exp14" / "scores.csv"
        if e14.exists():
            shutil.copy(e14, out / "results" / f"template_aware_probe_scores_{bbdir.name}.csv")
        lg = bbdir / "exp13" / "llama_guard_template_scores.csv"
        if lg.exists():
            shutil.copy(lg, out / "results" / "template_control_llama_guard_scores.csv")
        att = sorted((bbdir / "exp4").glob("attack_*.csv")) if (bbdir / "exp4").exists() else []
        if att:
            (out / "results" / "attacks").mkdir(parents=True, exist_ok=True)
            for p in att:
                shutil.copy(p, out / "results" / "attacks" / f"{bbdir.name}_{p.name}")
    # The 950-prompt main corpus (XSTest safe + XSTest unsafe + framed AdvBench) in one file, with labels.
    for bbdir in sorted(p for p in runs.iterdir() if p.is_dir()):
        a = bbdir / "main" / "analysis.csv"
        if a.exists():
            cols = ["prompt_id", "dataset", "prompt_category", "group", "ground_truth", "family", "split", "prompt_text"]
            mc = pd.read_csv(a)
            mc[[c for c in cols if c in mc]].to_csv(out / "data" / "main_corpus_950.csv", index=False)
            break
    ds = ROOT / "data" / "DATASHEET.md"
    if ds.exists():
        shutil.copy(ds, out / "data" / "DATASHEET.md")
    print(f"release written to {out}")


if __name__ == "__main__":
    main()
