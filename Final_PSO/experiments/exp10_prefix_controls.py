#!/usr/bin/env python3
"""Experiment 10 — Prefix controls and layer-wise analysis (Section III-D/E).

Every prompt of the groups in ``experiments.exp10.groups`` is encoded under four
prefix conditions, keeping prompts, model, framing, pooling and precision fixed:

    safety      the Red / Blue / Judge personas (operational condition)
    neutral     three distinct non-safety roles matched in structure and length
    paraphrase  reworded safety personas (wording sensitivity)
    identity    one neutral instruction repeated three times (implementation check: D~ ~ 0)

The last-token state is read at every layer (index 0 = input embeddings), and the
half-scale score D~ = D/2 and the pairwise distances are computed per layer.

Analyses
    identity check              max / mean D~ under the identity condition
    layer-wise means            group means with 95% family-bootstrap CIs per layer and condition
    baseline-adjusted score     C(l)(x) of Eq. (8), with b_ij estimated on a fixed benign reference
                                set A (calibration split of the harmless set; excluded from the rest)
    interaction                 Delta(l) = (mu_J,S - mu_B,S) - (mu_J,N - mu_B,N)   (Eq. 6)
                                for J = framed jailbreaks against B = benign, ambiguous, direct harmful,
                                and the matched contrast framed vs direct for the SAME AdvBench goal.
                                Inference uses the final layer (preselected); other layers are descriptive.
                                Uncertainty: bootstrap over prompt families (AdvBench goal = family).
    length control              OLS of final-layer D~ on group indicators and log token length
Outputs
    tables/exp10_identity.csv, exp10_layer_means.csv, exp10_interaction.csv,
    tables/exp10_interaction_by_layer.csv, exp10_adjusted_scores.csv, exp10_length_ols.csv,
    tables/exp10_layer_auc.csv
    figures/fig12_layerwise.png, fig13_interaction_by_layer.png
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch.nn.functional as F

from experiments.common import figure_path, load_client, run_dir, save_table, setup
from benchmarks.loaders import load_dataset
from core.msds import PERSONA_ORDER
from evaluation import metrics as M
from pipeline.runner import record_fields

PAIRS = [(0, 1), (0, 2), (1, 2)]  # (R,B), (R,J), (B,J)
GROUP_LABEL = {"xstest_safe": "benign", "ambiguous": "ambiguous", "xstest_unsafe": "direct_xstest",
               "advbench_direct": "direct_advbench", "advbench": "framed_jailbreak",
               "advbench_wild": "wild_jailbreak"}


def frame(prefixes: Dict[str, str], prompt: str, sep: str) -> List[str]:
    return [prefixes[r] + sep + prompt for r in PERSONA_ORDER]


def encode_condition(client, records, prefixes, sep, per_batch) -> np.ndarray:
    """Pairwise cosine distances per prompt and layer: [N, L+1, 3]."""
    import torch
    out, start, bs = [], 0, per_batch
    while start < len(records):
        chunk = records[start:start + bs]
        texts = [t for r in chunk for t in frame(prefixes, r.text, sep)]
        try:
            h = client.encode_all_layers(texts)                # [3n, L+1, d]
        except torch.cuda.OutOfMemoryError:                    # long prompts: halve the batch and retry
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            continue
        h = F.normalize(h, dim=-1).view(len(chunk), 3, h.shape[1], h.shape[2])
        d = [1.0 - (h[:, i] * h[:, j]).sum(-1) for i, j in PAIRS]   # each [n, L+1]
        out.append(np.stack([x.numpy() for x in d], axis=-1))
        start += len(chunk)
    return np.concatenate(out, axis=0)


def boot_ci(values_by_group: Dict[str, pd.DataFrame], stat, n_boot: int, seed: int):
    """Family-level bootstrap of ``stat(dict of frames)``; returns (estimate, lo, hi, p_le_0)."""
    rng = np.random.default_rng(seed)
    est = stat(values_by_group)
    fams = {g: df["family"].unique() for g, df in values_by_group.items()}
    idx = {g: {f: np.flatnonzero(df["family"].to_numpy() == f) for f in fams[g]} for g, df in values_by_group.items()}
    boots = []
    for _ in range(n_boot):
        sample = {}
        for g, df in values_by_group.items():
            pick = rng.choice(fams[g], size=len(fams[g]), replace=True)
            sample[g] = df.iloc[np.concatenate([idx[g][f] for f in pick])]
        boots.append(stat(sample))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return est, lo, hi, (boots <= 0).mean(axis=0)


def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e10 = cfg["experiments"]["exp10"]
    sep = cfg["pso"]["framing"]["separator"]
    per_batch = int(cfg["backbones"].get("encode_batch_prompts", 1))
    out_dir = run_dir(cfg, args.backbone, "exp10")
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = args.backbone == cfg["backbones"]["primary"]
    T = lambda name: name if primary else f"{name}_{args.backbone}"  # noqa: E731  per-backbone outputs

    records = [r for g in e10["groups"] for r in load_dataset(cfg, g)]
    meta = pd.DataFrame([{**record_fields(r), "prompt_text": r.text} for r in records])
    meta["group10"] = meta["dataset"].map(GROUP_LABEL)
    conditions = {
        "safety": cfg["pso"]["personas"],
        "neutral": e10["neutral_prefixes"],
        "paraphrase": e10["persona_paraphrases"],
        "identity": {r: e10["identity_prefix"] for r in PERSONA_ORDER},
    }

    client = None
    dist = {}
    for name, prefixes in conditions.items():
        path = out_dir / f"distances_{name}.npy"
        ids_path = out_dir / f"distances_{name}_ids.json"
        ids = [r.prompt_id for r in records]
        cached_d, cached_ids = None, None
        if path.exists():
            cached_d = np.load(path)
            cached_ids = json.loads(ids_path.read_text()) if ids_path.exists() else \
                [r.prompt_id for r in records[:len(cached_d)]]  # legacy cache: the first len() records
            if cached_ids == ids:
                dist[name] = cached_d
                continue
        # Encode only prompts missing from the cache (e.g. a group added later), then re-order.
        have = dict(zip(cached_ids, cached_d)) if cached_d is not None else {}
        todo = [r for r in records if r.prompt_id not in have]
        if todo:
            client = client or load_client(cfg, args.backbone)
            new = encode_condition(client, todo, prefixes, sep, per_batch)
            have.update(zip([r.prompt_id for r in todo], new))
        dist[name] = np.stack([have[i] for i in ids])
        np.save(path, dist[name])
        ids_path.write_text(json.dumps(ids))
    n_layers = dist["safety"].shape[1]
    final = n_layers - 1
    half = lambda d: d.mean(axis=-1) / 2.0                     # D~ per prompt and layer  [N, L+1]
    score = {c: half(d) for c, d in dist.items()}

    tok = None
    lengths_path = out_dir / "token_lengths.npy"
    if lengths_path.exists() and len(np.load(lengths_path)) == len(records):
        lengths = np.load(lengths_path)
    else:
        client = client or load_client(cfg, args.backbone)
        tok = client.tokenizer
        lengths = np.array([len(tok(r.text, add_special_tokens=False)["input_ids"]) for r in records])
        np.save(lengths_path, lengths)
    meta["tokens"] = lengths

    # Identity check ---------------------------------------------------------------
    ident = score["identity"]
    save_table(pd.DataFrame([{"layer": "final", "max_D": float(ident[:, final].max()),
                              "mean_D": float(ident[:, final].mean()),
                              "max_D_any_layer": float(ident.max())}]), cfg, T("exp10_identity"))

    # Baseline-adjusted score (Eq. 7-8) --------------------------------------------
    benign_calib = meta.index[(meta["group10"] == "benign") & (meta["split"] == "calib")].to_numpy()
    rng = np.random.default_rng(0)
    ref = rng.choice(benign_calib, size=min(e10["reference_size"], len(benign_calib)), replace=False)
    keep = np.setdiff1d(np.arange(len(meta)), ref)
    adjusted = {}
    for c in ("safety", "neutral", "paraphrase"):
        b = dist[c][ref].mean(axis=0)                          # [L+1, 3]
        adjusted[c] = (dist[c] - b[None]).mean(axis=-1) / 2.0   # C(l)(x), half scale
    m = meta.iloc[keep]

    # Layer-wise means with CIs -------------------------------------------------------
    rows = []
    for c in ("safety", "neutral", "paraphrase"):
        for kind, arr in (("raw", score[c]), ("adjusted", adjusted[c])):
            for g, gdf in m.groupby("group10"):
                vals = arr[gdf.index.to_numpy()]
                fam = gdf["family"].to_numpy()
                ufam = np.unique(fam)
                rs = np.random.default_rng(1)
                boots = []
                fam_means = np.stack([vals[fam == f].mean(axis=0) for f in ufam])
                for _ in range(min(500, e10["bootstrap"])):
                    boots.append(fam_means[rs.integers(0, len(ufam), len(ufam))].mean(axis=0))
                lo, hi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
                for layer in range(n_layers):
                    rows.append({"condition": c, "score": kind, "group": g, "layer": layer, "n": len(gdf),
                                 "mean": float(vals[:, layer].mean()), "ci_lo": float(lo[layer]),
                                 "ci_hi": float(hi[layer])})
    layer_means = pd.DataFrame(rows)
    save_table(layer_means[layer_means["layer"] == final], cfg, T("exp10_layer_means_final"))
    layer_means.to_csv(figure_path(cfg, T("fig12_layerwise")).with_suffix(".csv"), index=False)

    # AUC per layer (framed jailbreak vs benign) -------------------------------------------
    jb = m[m["group10"].isin(["framed_jailbreak", "benign"])]
    y = (jb["group10"] == "framed_jailbreak").astype(int).to_numpy()
    auc_rows = []
    for layer in range(n_layers):
        row = {"layer": layer}
        for c in ("safety", "neutral", "paraphrase"):
            row[f"auc_{c}"] = M.auc(y, score[c][jb.index.to_numpy(), layer])
            row[f"auc_{c}_adjusted"] = M.auc(y, adjusted[c][jb.index.to_numpy(), layer])
        auc_rows.append(row)
    save_table(pd.DataFrame(auc_rows), cfg, T("exp10_layer_auc"))

    # Interaction (difference-in-differences), final layer + per layer ---------------------
    frame_ = m.copy()
    contrasts = [("framed_jailbreak", "benign"), ("framed_jailbreak", "ambiguous"),
                 ("framed_jailbreak", "direct_xstest"), ("framed_jailbreak", "direct_advbench"),
                 ("wild_jailbreak", "benign"), ("wild_jailbreak", "direct_advbench"),
                 ("direct_advbench", "benign"), ("ambiguous", "benign")]
    inter_rows, by_layer = [], []
    for alt in ("neutral", "paraphrase_vs_neutral"):
        s_arr = score["paraphrase"] if alt == "paraphrase_vs_neutral" else score["safety"]
        diff = s_arr - score["neutral"]                         # per prompt, per layer: S - N
        for a, b in contrasts:
            ga = frame_[frame_["group10"] == a].assign(_i=lambda d: d.index)
            gb = frame_[frame_["group10"] == b].assign(_i=lambda d: d.index)
            if ga.empty or gb.empty:
                continue

            def stat(dct, layer=None):
                da = diff[dct["a"]["_i"].to_numpy()]
                db = diff[dct["b"]["_i"].to_numpy()]
                v = da.mean(axis=0) - db.mean(axis=0)
                return v if layer is None else v[layer]

            est, lo, hi, p = boot_ci({"a": ga, "b": gb}, stat, e10["bootstrap"], 0)
            gap_s = score["safety" if alt == "neutral" else "paraphrase"][ga["_i"], final].mean() \
                - score["safety" if alt == "neutral" else "paraphrase"][gb["_i"], final].mean()
            gap_n = score["neutral"][ga["_i"], final].mean() - score["neutral"][gb["_i"], final].mean()
            inter_rows.append({"conditioning": "safety" if alt == "neutral" else "paraphrase",
                               "contrast": f"{a} - {b}", "layer": int(final),
                               "gap_conditioned": float(gap_s), "gap_neutral": float(gap_n),
                               "delta": float(est[final]), "ci_lo": float(lo[final]), "ci_hi": float(hi[final]),
                               "p_one_sided": float(p[final]), "n_a": len(ga), "n_b": len(gb)})
            for layer in range(n_layers):
                by_layer.append({"conditioning": "safety" if alt == "neutral" else "paraphrase",
                                 "contrast": f"{a} - {b}", "layer": layer, "delta": float(est[layer]),
                                 "ci_lo": float(lo[layer]), "ci_hi": float(hi[layer])})

    # Matched contrast: framed vs direct for the same AdvBench goal (paired by family)
    for JG, JL in (("framed_jailbreak", "framed"), ("wild_jailbreak", "wild")):
        fr = frame_[frame_["group10"] == JG].set_index("family")
        di = frame_[frame_["group10"] == "direct_advbench"].set_index("family")
        common = fr.index.intersection(di.index)
        if len(common):
            diff = score["safety"] - score["neutral"]
            ia = np.array([meta.index[(meta["family"] == f) & (meta["group10"] == JG)][0] for f in common])
            ib = np.array([meta.index[(meta["family"] == f) & (meta["group10"] == "direct_advbench")][0] for f in common])
            per_fam = diff[ia] - diff[ib]                            # [F, L+1]
            rs = np.random.default_rng(0)
            boots = np.stack([per_fam[rs.integers(0, len(common), len(common))].mean(axis=0)
                              for _ in range(e10["bootstrap"])])
            est = per_fam.mean(axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            inter_rows.append({"conditioning": "safety", "contrast": f"{JL} - direct (same goal, paired)",
                               "layer": int(final),
                               "gap_conditioned": float((score["safety"][ia, final] - score["safety"][ib, final]).mean()),
                               "gap_neutral": float((score["neutral"][ia, final] - score["neutral"][ib, final]).mean()),
                               "delta": float(est[final]), "ci_lo": float(lo[final]), "ci_hi": float(hi[final]),
                               "p_one_sided": float((boots[:, final] <= 0).mean()), "n_a": len(common), "n_b": len(common)})
            for layer in range(n_layers):
                by_layer.append({"conditioning": "safety", "contrast": f"{JL} - direct (same goal, paired)",
                                 "layer": layer, "delta": float(est[layer]), "ci_lo": float(lo[layer]),
                                 "ci_hi": float(hi[layer])})
    save_table(pd.DataFrame(inter_rows), cfg, T("exp10_interaction"))
    by_layer = pd.DataFrame(by_layer)
    by_layer.to_csv(figure_path(cfg, T("fig13_interaction_by_layer")).with_suffix(".csv"), index=False)

    # Adjusted scores, final layer ---------------------------------------------------------
    adj_rows = []
    for g, gdf in m.groupby("group10"):
        i = gdf.index.to_numpy()
        adj_rows.append({"group": g, "n": len(i),
                         "raw_safety": float(score["safety"][i, final].mean()),
                         "raw_neutral": float(score["neutral"][i, final].mean()),
                         "adjusted_safety": float(adjusted["safety"][i, final].mean()),
                         "adjusted_neutral": float(adjusted["neutral"][i, final].mean()),
                         "s_minus_n": float((score["safety"][i, final] - score["neutral"][i, final]).mean())})
    save_table(pd.DataFrame(adj_rows), cfg, T("exp10_adjusted_scores"))

    # Length control (OLS, final layer, safety condition) -------------------------------------
    groups = sorted(m["group10"].unique())
    base_g = "benign"
    X = np.column_stack([np.ones(len(m))] + [(m["group10"] == g).to_numpy(float) for g in groups if g != base_g]
                        + [np.log(m["tokens"].to_numpy(float) + 1)])
    names = ["intercept"] + [f"group_{g}" for g in groups if g != base_g] + ["log_tokens"]
    ols_rows = []
    for c in ("safety", "neutral"):
        yv = score[c][m.index.to_numpy(), final]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        rs = np.random.default_rng(0)
        fam = m["family"].to_numpy()
        ufam = np.unique(fam)
        pos = {f: np.flatnonzero(fam == f) for f in ufam}
        boots = []
        for _ in range(min(1000, e10["bootstrap"])):
            ii = np.concatenate([pos[f] for f in rs.choice(ufam, len(ufam))])
            boots.append(np.linalg.lstsq(X[ii], yv[ii], rcond=None)[0])
        lo, hi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
        for n_, b_, l_, h_ in zip(names, beta, lo, hi):
            ols_rows.append({"condition": c, "term": n_, "coef": float(b_), "ci_lo": float(l_), "ci_hi": float(h_)})
    save_table(pd.DataFrame(ols_rows), cfg, T("exp10_length_ols"))

    # Figures --------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"benign": "#2a7ab9", "ambiguous": "#8fb43a", "direct_xstest": "#d9822b",
              "direct_advbench": "#b85c00", "framed_jailbreak": "#c0392b", "wild_jailbreak": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, c in zip(axes, ("safety", "neutral")):
        d = layer_means[(layer_means["condition"] == c) & (layer_means["score"] == "raw")]
        for g, gd in d.groupby("group"):
            ax.plot(gd["layer"], gd["mean"], label=g.replace("_", " "), color=colors.get(g))
            ax.fill_between(gd["layer"], gd["ci_lo"], gd["ci_hi"], alpha=0.2, color=colors.get(g))
        ax.set_title(f"{c} prefixes")
        ax.set_xlabel("layer (0 = embeddings)")
    axes[0].set_ylabel(r"mean $\tilde D$")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(cfg, T("fig12_layerwise")), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for contrast, gd in by_layer[by_layer["conditioning"] == "safety"].groupby("contrast"):
        ax.plot(gd["layer"], gd["delta"], label=contrast)
        ax.fill_between(gd["layer"], gd["ci_lo"], gd["ci_hi"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel(r"$\Delta^{(\ell)}$ (safety $-$ neutral)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path(cfg, T("fig13_interaction_by_layer")), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
