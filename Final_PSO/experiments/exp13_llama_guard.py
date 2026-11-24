#!/usr/bin/env python3
"""Experiment 13b -- the template control applied to Llama Guard 3-1B as a prompt classifier.

Same scoring rule as Exp. 12 (P(unsafe) at the first verdict token, renormalised over safe/unsafe) and the
same threshold (lambda rule on the main-corpus calibration split, taken from the Exp. 12 scores of the
primary backbone). Runs on CPU (float32) or GPU. Resumable: scores are appended to a cache file.

Outputs
    runs/<bb>/exp13/llama_guard_template_scores.csv
    tables/exp13_llama_guard_template_control.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config
from evaluation import metrics as M
from experiments.common import run_dir, save_table
from experiments.exp12_baselines import find_local_model, youden_threshold

SETS = ["benign_direct", "benign_framed", "benign_wild", "jbb_benign_framed", "jbb_harmful_framed"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", default="llama2", help="backbone whose Exp. 12 scores hold the calibration")
    ap.add_argument("--model", default=None, help="local path of Llama Guard 3-1B (default: from the config)")
    ap.add_argument("--independent", action="store_true", help="score the Exp. 15 independent sets instead")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    ev = cfg["evaluation"]
    ref = pd.read_csv(run_dir(cfg, args.backbone, "exp12") / "scores_with_probes.csv")
    main_mask = ref["dataset"].isin(["advbench", "xstest_safe", "xstest_unsafe"])
    calib = (main_mask & (ref["split"] == "calib")).to_numpy()
    s_ref, y_ref = ref["llama_guard"].to_numpy(float), ref["ground_truth"].to_numpy(int)
    thr = youden_threshold(s_ref[calib], y_ref[calib], ev["calibration_lambda"])

    if args.independent:
        prompts = pd.read_csv(raw / "independent_sets.csv"); prompts["prompt_id"] = prompts["id"].astype(str)
    else:
        prompts = pd.concat([pd.read_csv(raw / f"{n}.csv").assign(set=n) for n in SETS], ignore_index=True)
        prompts["prompt_id"] = prompts["set"] + "-" + prompts["id"].astype(str)
    out_dir = run_dir(cfg, args.backbone, "exp13")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ("llama_guard_independent_scores.csv" if args.independent else "llama_guard_template_scores.csv")
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["prompt_id", "set", "llama_guard"])
    todo = prompts[~prompts["prompt_id"].isin(set(done["prompt_id"]))]

    if len(todo):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = args.model or find_local_model(cfg["experiments"]["exp12"]["llama_guard"]["name_or_path"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        safe_id = tok.encode("safe", add_special_tokens=False)[0]
        unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]
        buf = []
        for k, (_, r) in enumerate(todo.iterrows()):
            p = str(r["prompt"])
            try:
                text = tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                               tokenize=False, add_generation_prompt=True)
            except Exception:  # noqa: BLE001
                text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False, output_scores=True,
                                   return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
            new = g.sequences[0, enc["input_ids"].shape[1]:]
            step = next((i for i, t in enumerate(new.tolist()) if tok.decode([t]).strip()), 0)
            logits = g.scores[step][0].float()
            buf.append({"prompt_id": r["prompt_id"], "set": r["set"],
                        "llama_guard": float(torch.softmax(logits[[safe_id, unsafe_id]], dim=-1)[1])})
            if len(buf) >= 25 or k == len(todo) - 1:
                pd.concat([done, pd.DataFrame(buf)]).to_csv(cache, index=False)
                done, buf = pd.read_csv(cache), []
                print(f"{len(done)}/{len(prompts)} scored", flush=True)

    sc = pd.read_csv(cache)
    if args.independent:
        xs = s_ref[(main_mask & (ref["split"] == "test") & (ref["dataset"] == "xstest_safe")).to_numpy()]
        rows = []
        for st_, g in sc.groupby("set"):
            rows.append({"method": "llama_guard_3_1b", "kind": "flag_rate", "set": st_, "n": len(g),
                         "value": float((g["llama_guard"] >= thr).mean()), "threshold": thr})
        hard = sc.loc[sc["set"] == "orbench_hard", "llama_guard"].to_numpy(float)
        for st_, g in sc[sc["set"] != "orbench_hard"].groupby("set"):
            for neg_name, neg in (("orbench_hard", hard), ("xstest_safe_test", xs)):
                yy = np.r_[np.ones(len(g)), np.zeros(len(neg))]
                rows.append({"method": "llama_guard_3_1b", "kind": "auc", "set": f"{st_} vs {neg_name}", "n": len(yy),
                             "value": float(M.auc(yy, np.r_[g["llama_guard"].to_numpy(float), neg])), "threshold": thr})
        t = pd.DataFrame(rows); save_table(t, cfg, "exp15_llama_guard_independent"); print(t.to_string())
        return
    test = (main_mask & (ref["split"] == "test")).to_numpy()
    groups = {
        "advbench_framed_test": s_ref[test & (ref["dataset"] == "advbench").to_numpy()],
        "xstest_safe_test": s_ref[test & (ref["dataset"] == "xstest_safe").to_numpy()],
        "advbench_wild": s_ref[(ref["dataset"] == "advbench_wild").to_numpy()],
        "advbench_direct": s_ref[(ref["dataset"] == "advbench_direct").to_numpy()],
    }
    for n in SETS:
        groups[n] = sc.loc[sc["set"] == n, "llama_guard"].to_numpy(float)
    rows = [{"method": "llama_guard_3_1b", "kind": "flag_rate", "set": k, "n": len(v),
             "value": float((v >= thr).mean()), "threshold": thr} for k, v in groups.items()]
    for hk, bk in [("advbench_framed_test", "benign_framed"), ("advbench_wild", "benign_wild"),
                   ("advbench_direct", "benign_direct"), ("jbb_harmful_framed", "jbb_benign_framed"),
                   ("advbench_framed_test", "xstest_safe_test")]:
        yy = np.r_[np.ones(len(groups[hk])), np.zeros(len(groups[bk]))]
        rows.append({"method": "llama_guard_3_1b", "kind": "auc", "set": f"{hk} vs {bk}", "n": len(yy),
                     "value": float(M.auc(yy, np.r_[groups[hk], groups[bk]])), "threshold": thr})
    t = pd.DataFrame(rows)
    save_table(t, cfg, "exp13_llama_guard_template_control")
    print(t.to_string())


if __name__ == "__main__":
    main()
