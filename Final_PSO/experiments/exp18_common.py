"""Shared data for Exp. 18: the diverse-probe corpus of Exp. 16/17 at the layer chosen by grouped CV."""
from __future__ import annotations

import gc
import random

import numpy as np
import pandas as pd

from benchmarks.loaders import split_of
from experiments.common import load_client, run_dir
from experiments.exp12_baselines import all_layer_states, fit_probe
from experiments.exp13_template_control import build_frame, hidden_states
from experiments.exp16_diverse_probe import GROUP, augmentation, extra_records


class Encoder:
    """Lazily loaded backbone that returns the last-token state at one layer."""

    def __init__(self, cfg, bb, bs):
        self.cfg, self.bb, self.bs, self.client, self.layer = cfg, bb, bs, None, None

    def __call__(self, texts):
        if not len(texts):
            return np.zeros((0, 1), np.float32)
        if self.client is None:
            self.client = load_client(self.cfg, self.bb)
        return all_layer_states(self.client, list(texts), self.bs)[:, self.layer].astype(np.float32)


def diverse_data(cfg, bb, bs):
    """Returns (D, X, layer, texts, encoder). D has group/fsplit columns; X are float32 states at `layer`."""
    frac = cfg["evaluation"].get("calibration_fraction", 0.3)
    e12 = cfg["experiments"]["exp12"]
    enc = Encoder(cfg, bb, bs)
    df, texts = build_frame(cfg)
    H = hidden_states(cfg, bb, texts, run_dir(cfg, bb, "exp13"), bs)
    ex, ex_texts = extra_records(cfg)
    au, au_texts = augmentation(cfg, frac, random.Random(0))
    extra_texts = ex_texts + au_texts
    e16 = run_dir(cfg, bb, "exp16"); e16.mkdir(parents=True, exist_ok=True)
    cache = e16 / "extra_states.npy"
    try:
        ok = cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(extra_texts)
    except (ValueError, OSError):
        ok = False
    if not ok:
        enc.client = enc.client or load_client(cfg, bb)
        np.save(cache, all_layer_states(enc.client, extra_texts, bs))
    He = np.load(cache, mmap_mode="r")

    D = pd.concat([df[["prompt_id", "family", "set", "ground_truth"]], ex, au], ignore_index=True)
    all_texts = list(texts) + extra_texts
    D["group"] = D["set"].map(GROUP)
    D.loc[D["set"].str.startswith("aug_"), "group"] = "AUG"
    keep = D["group"].notna().to_numpy()
    idx = np.where(keep)[0]
    D = D[keep].reset_index(drop=True)
    all_texts = [all_texts[i] for i in idx]
    D["fsplit"] = [split_of(str(f), frac) for f in D["family"]]

    def view(layer):
        return np.concatenate([H[:, layer], He[:, layer]], axis=0)[idx].astype(np.float32)

    y = D["ground_truth"].to_numpy().astype(int); fam = D["family"].astype(str).to_numpy()
    cal = (D["group"].to_numpy() != "AUG") & (D["fsplit"].to_numpy() == "calib")
    best, best_cv = None, -1.0
    for layer in range(H.shape[1]):
        _, _, cv = fit_probe(view(layer)[cal], y[cal], fam[cal], e12["probe_C"])
        if cv > best_cv:
            best, best_cv = layer, cv
    X = view(best)
    del H, He
    gc.collect()
    enc.layer = best
    print(f"[{bb}] layer {best} (grouped CV AUC {best_cv:.3f})", flush=True)
    return D, X, best, all_texts, enc
