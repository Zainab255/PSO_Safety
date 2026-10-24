#!/usr/bin/env python3
"""Experiment 9 — Multi-turn Temporal Entropy Drift (RQ9).

Per seed, gradual-escalation, abrupt-injection and fully harmless sequences
are built from XSTest safe and AdvBench prompts and run turn by turn through
the PSO (one Conversation per sequence) with the TED filter off and on.

DR  = share of escalation sequences with a detection decision on any turn
FAR = share of harmless sequences with a detection decision on any turn
(detection decisions: ``experiments.exp9.detection_decisions``)

Outputs
    tables/exp9_table17_multiturn.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import load_client, run_dir, save_table, setup
from benchmarks.loaders import build_multiturn_sequences, load_dataset
from evaluation.stats import fmt, mean_std, paired_one_tailed
from pipeline.runner import cached
from wrappers import Conversation, ParallelSafetyOrchestrator


def run_sequences(cfg, client, seed, safe, adversarial) -> pd.DataFrame:
    detect = set(cfg["experiments"]["exp9"]["detection_decisions"])
    sequences = build_multiturn_sequences(safe, adversarial, cfg, seed)
    rows = []
    for ted in (False, True):
        pso = ParallelSafetyOrchestrator(client, cfg, use_ted=ted)
        for kind, seqs in sequences.items():
            for i, seq in enumerate(seqs):
                conv = Conversation()
                decisions = [pso.evaluate(turn.text, conv) for turn in seq]
                first = next((t + 1 for t, r in enumerate(decisions) if r.decision.value in detect), np.nan)
                rows.append({"seed": seed, "ted": ted, "kind": kind, "sequence": i, "detected": not np.isnan(first),
                             "first_detection_turn": first,
                             "ted_alert": any(r.decided_by == "ted" for r in decisions)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    seeds = cfg["evaluation"]["seeds"]
    safe, adversarial = load_dataset(cfg, "xstest_safe"), load_dataset(cfg, "advbench")
    client = load_client(cfg, args.backbone)
    df = pd.concat([
        cached(run_dir(cfg, args.backbone, "exp9") / f"multiturn_seed{seed}.csv",
               lambda: run_sequences(cfg, client, seed, safe, adversarial))
        for seed in seeds
    ], ignore_index=True)

    per_seed = (df.groupby(["seed", "ted", "kind"])["detected"].mean().rename("rate").reset_index())
    rows = []
    for kind in ("abrupt", "gradual"):
        dr = {t: per_seed[(per_seed["kind"] == kind) & (per_seed["ted"] == t)].sort_values("seed")["rate"].to_numpy()
              for t in (False, True)}
        p = paired_one_tailed(dr[True], dr[False], "greater")
        for ted in (False, True):
            far = per_seed[(per_seed["kind"] == "harmless") & (per_seed["ted"] == ted)]["rate"]
            m, s = mean_std(dr[ted])
            fm, fs = mean_std(far)
            rows.append({"attack_type": kind, "ted": "on" if ted else "off",
                         "dr": fmt(m, s, p if ted else np.nan), "dr_mean": m, "dr_std": s,
                         "dr_p_on_vs_off": p if ted else np.nan,
                         "far": fmt(fm, fs), "far_mean": fm, "far_std": fs})
    save_table(pd.DataFrame(rows), cfg, "exp9_table17_multiturn")


if __name__ == "__main__":
    main()
