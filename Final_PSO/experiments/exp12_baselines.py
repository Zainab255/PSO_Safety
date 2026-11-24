#!/usr/bin/env python3
"""Experiment 12 — Single-view and published baselines on the PSO corpus and split.

Every method is fitted / thresholded on the calibration split only and evaluated on the
held-out test split of the main corpus (same family-level split as the PSO), then
transferred unchanged to HarmBench, JailbreakBench and the unframed AdvBench goals
(positives) against the XSTest-safe test split (negatives).

Single-view (no persona) baselines
    probe_last        logistic probe on the last-token hidden state of the raw prompt; layer and
                      L2 strength chosen by 5-fold family-grouped CV on the calibration split
    probe_final       the same probe at the final layer (the PSO readout layer)
    probe_layerwise   test AUC of a probe at every layer (diagnostic curve, tables/exp12_probe_layers.csv)
Persona-view baselines (are three personas needed?)
    probe_red         probe on the Red-persona-framed final-layer state (one persona view)
    probe_3views      probe on the concatenated R/B/J final-layer states
    probe_pairwise    logistic model on (s_RB, s_RJ, s_BJ)
    msds / msds_dircal  D~ with the pre-specified / calibration-chosen direction
Published baselines
    perplexity        prompt perplexity under the backbone (Jain et al., 2023)
    hiddendetect      HiddenDetect-style score (Jiang et al., 2025) adapted to text: cosine between
                      the logit-lens vocabulary projection of each layer and a refusal-token vector,
                      summed over the most discriminative layers (selected on the calibration split)
    smoothllm         SmoothLLM (Robey et al., 2023): N randomly perturbed copies, flagged when the
                      majority of responses refuse (refusal-keyword check); test split only
    llama_guard       Llama Guard 3 input classification, P(unsafe) (if the weights are available)

Outputs
    runs/<bb>/exp12/scores.csv               per-prompt scores of every method
    tables/exp12_baselines.csv               test-split AUC / TPR / FPR / F2 (+ transfer AUCs) per method
    tables/exp12_probe_layers.csv            layer-wise single-view probe AUC
    tables/exp12_latency.csv                 wrapper-only latency per method (unbatched, ms)
"""

from __future__ import annotations

import glob
import logging
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import set_seed

from benchmarks.loaders import load_corpus, load_dataset
from core.msds import PERSONA_ORDER
from evaluation import metrics as M
from experiments.common import load_client, release_client, run_dir, save_table, setup
from pipeline.runner import record_fields
from wrappers import ParallelSafetyOrchestrator

logger = logging.getLogger("experiments.exp12")


# ----------------------------------------------------------------------------- helpers

def youden_threshold(score: np.ndarray, y: np.ndarray, lam: float = 0.5, n: int = 40) -> float:
    s = score[np.isfinite(score)]
    best, best_obj = np.nan, -np.inf
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, n))):
        pred = score >= t
        obj = lam * pred[y == 1].mean() - (1 - lam) * pred[y == 0].mean()
        if obj > best_obj:
            best, best_obj = float(t), obj
    return best


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs, folds: int = 5):
    """Choose C by family-grouped CV AUC on the calibration rows, then refit on all of them."""
    best_c, best_auc = Cs[0], -1.0
    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    for c in Cs:
        aucs = []
        for tr, va in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            m.fit(X[tr], y[tr])
            aucs.append(M.auc(y[va], m.decision_function(X[va])))
        a = float(np.mean(aucs)) if aucs else -1.0
        if a > best_auc:
            best_c, best_auc = c, a
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000)).fit(X, y)
    return m, best_c, best_auc


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def oom_safe(fn, seq, bs):
    """Apply fn to consecutive chunks of seq; on CUDA OOM halve the chunk size and retry."""
    out, i = [], 0
    while i < len(seq):
        try:
            out.append(fn(seq[i:i + bs]))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            logger.warning("CUDA OOM; retrying with %d prompts per batch", bs)
    return out


def all_layer_states(client, texts: List[str], bs: int, add_special: bool = True) -> np.ndarray:
    out = oom_safe(lambda chunk: client.encode_all_layers(chunk, add_special_tokens=add_special).half().numpy(),
                   texts, bs)
    return np.concatenate(out, axis=0)


def lm_head_parts(client):
    model = client.model
    norm = getattr(model.base_model, "norm", None)
    return norm, model.lm_head


def refusal_profile(client, texts: List[str], refusal_ids: List[int], bs: int) -> np.ndarray:
    """Cosine(logit-lens projection of every layer's last token, refusal one-hot vector): [N, L+1]."""
    norm, head = lm_head_parts(client)
    vocab = head.weight.shape[0]
    r = torch.zeros(vocab)
    r[refusal_ids] = 1.0
    def one(chunk):
        client.tokenizer.padding_side = "right"
        enc = client.tokenizer(list(chunk), padding=True, truncation=True, max_length=client.max_input_tokens,
                               add_special_tokens=False, return_tensors="pt").to(client.device)
        with torch.no_grad():
            hs = client.model.base_model(**enc, output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1
            prof = []
            for li, h in enumerate(hs):
                v = h[torch.arange(h.size(0), device=h.device), last.to(h.device)]
                if norm is not None and li < len(hs) - 1:  # the last hidden state is already normalised
                    v = norm(v.to(next(norm.parameters()).device))
                logits = head(v.to(head.weight.device, head.weight.dtype)).float()
                prof.append(F.cosine_similarity(logits, r.to(logits.device).unsqueeze(0), dim=-1).cpu())
            return torch.stack(prof, dim=1).numpy()
    return np.concatenate(oom_safe(one, texts, bs), axis=0)


def log_perplexity(client, texts: List[str], bs: int) -> np.ndarray:
    def one(chunk):
        client.tokenizer.padding_side = "right"
        enc = client.tokenizer(list(chunk), padding=True, truncation=True, max_length=client.max_input_tokens,
                               add_special_tokens=True, return_tensors="pt").to(client.device)
        with torch.no_grad():
            logits = client.model(**enc).logits
        res = []
        for b in range(logits.size(0)):  # one sequence at a time: [T, V] in float32
            lp = torch.log_softmax(logits[b, :-1].float(), dim=-1)
            tgt = enc["input_ids"][b, 1:].to(lp.device)
            mask = enc["attention_mask"][b, 1:].to(lp.device).float()
            res.append(float(-(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum().clamp(min=1)))
        del logits
        return np.asarray(res)
    return np.concatenate(oom_safe(one, texts, max(1, bs // 2)))


def perturb(text: str, pct: float, rng: random.Random) -> str:
    """SmoothLLM random-swap perturbation: replace pct of the characters with random printable ones."""
    chars = list(text)
    k = max(1, int(len(chars) * pct))
    alphabet = [chr(i) for i in range(32, 127)]
    for i in rng.sample(range(len(chars)), min(k, len(chars))):
        chars[i] = rng.choice(alphabet)
    return "".join(chars)


def refused(resp: str, prefixes) -> bool:
    return any(p in resp for p in prefixes)


def find_local_model(name: str) -> str:
    """A locally attached copy of the weights (e.g. the Kaggle model metaresearch/llama-guard-3.2), else the hub id."""
    leaf = name.split("/")[-1]
    hits = sorted(glob.glob(f"/kaggle/input/**/{leaf}/config.json", recursive=True)) + \
        sorted(glob.glob("/kaggle/input/**/llama-guard*/**/config.json", recursive=True))
    return hits[0].rsplit("/", 1)[0] if hits else name


def llama_guard_scores(cfg, prompts: List[str]) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    name = find_local_model(cfg["experiments"]["exp12"]["llama_guard"]["name_or_path"])
    tok_ = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(name, token=tok_)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto", token=tok_).eval()
    safe_id = tok.encode("safe", add_special_tokens=False)[0]
    unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]
    assert safe_id != unsafe_id
    out = []
    for p in prompts:
        try:
            text = tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        # The verdict is the first generated token that is not whitespace ("safe" / "unsafe");
        # P(unsafe) is read at that step, renormalised over the two verdict tokens.
        new = g.sequences[0, enc["input_ids"].shape[1]:]
        step = next((i for i, t in enumerate(new.tolist()) if tok.decode([t]).strip()), 0)
        logits = g.scores[step][0].float()
        out.append(float(torch.softmax(logits[[safe_id, unsafe_id]], dim=-1)[1]))
    del model
    torch.cuda.empty_cache()
    return np.asarray(out)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    cfg, args = setup(__doc__, argv)
    e12, ev = cfg["experiments"]["exp12"], cfg["evaluation"]
    bb = args.backbone
    out_dir = run_dir(cfg, bb, "exp12")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.csv"
    bs = int(cfg["backbones"].get("encode_batch_prompts", 8))

    main_recs = load_corpus(cfg, ev["corpus"])
    extra = []
    for name in e12["transfer_benchmarks"]:
        try:
            extra += [r for r in load_dataset(cfg, name) if r.ground_truth == 1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("transfer benchmark %s unavailable: %s", name, exc)
    recs = main_recs + extra
    df = pd.DataFrame([{**record_fields(r), "in_main": i < len(main_recs)} for i, r in enumerate(recs)])
    texts = [r.text for r in recs]
    y = df["ground_truth"].to_numpy().astype(int)
    calib = (df["split"] == "calib").to_numpy() & df["in_main"].to_numpy()
    test = (df["split"] == "test").to_numpy() & df["in_main"].to_numpy()

    client = load_client(cfg, bb)
    # The main-corpus persona analysis of this backbone (also reused by Exp. 7); computed here if missing.
    an_path = run_dir(cfg, bb, "main") / "analysis.csv"
    if not an_path.exists():
        from core.calibration import ensure_thresholds
        from pipeline.runner import EvaluationRunner, cached
        an_path.parent.mkdir(parents=True, exist_ok=True)
        analysis = cached(an_path, lambda: EvaluationRunner(cfg, client).analyse(main_recs))
        ensure_thresholds(cfg, bb, analysis)
    feats_path = out_dir / "single_view_states.npy"
    if feats_path.exists():
        H = np.load(feats_path)
    else:
        H = all_layer_states(client, texts, bs)
        np.save(feats_path, H)
    n_layers = H.shape[1]
    logger.info("single-view states: %s", H.shape)

    # Persona views (final layer) for the one-view / three-view probes.
    pv_path = out_dir / "persona_views.npy"
    if pv_path.exists():
        PV = np.load(pv_path)
    else:
        scorer = ParallelSafetyOrchestrator(client, cfg, use_keyword=False, use_patterns=False, use_ted=False)
        def views(chunk):
            t_all, s_all, add = [], [], True
            for p in chunk:
                t, s_, add = scorer.frame(p)
                t_all += t
                s_all += s_
            e = client.encode_hidden(t_all, scorer.pooling, scorer.layer, add, s_all).numpy()
            return e.reshape(len(chunk), len(PERSONA_ORDER), -1).astype(np.float16)
        PV = np.concatenate(oom_safe(views, texts, bs), axis=0)
        np.save(pv_path, PV)

    # Perplexity, HiddenDetect-style profile ---------------------------------------
    if scores_path.exists():
        S = pd.read_csv(scores_path)
        df = df.merge(S[[c for c in S.columns if c not in df.columns or c == "prompt_id"]], on="prompt_id", how="left")
    if "log_ppl" not in df.columns:
        df["log_ppl"] = log_perplexity(client, texts, bs)
    prof_path = out_dir / "refusal_profile.npy"
    if prof_path.exists():
        prof = np.load(prof_path)
    else:
        rendered = [client.apply_chat_template([{"role": "user", "content": t}]) if client.has_chat_template else t
                    for t in texts]
        ids = sorted({client.tokenizer.encode(w, add_special_tokens=False)[0] for w in e12["refusal_tokens"]}
                     | {client.tokenizer.encode(" " + w, add_special_tokens=False)[-1] for w in e12["refusal_tokens"]})
        prof = refusal_profile(client, rendered, ids, bs)
        np.save(prof_path, prof)
    layer_auc = np.array([M.auc(y[calib], prof[calib, l]) for l in range(prof.shape[1])])
    top = np.argsort(-np.abs(layer_auc - 0.5))[: e12["hiddendetect_top_layers"]]
    sign = np.sign(layer_auc[top] - 0.5)
    df["hiddendetect"] = (prof[:, top] * sign).sum(1)

    # SmoothLLM (test split of the main corpus) --------------------------------------
    sm = e12["smoothllm"]
    prefixes = cfg["output_judge"]["refusal_prefixes"]
    primary = bb == cfg["backbones"]["primary"]
    if "smoothllm" not in df.columns and primary:
        rng = random.Random(0)
        set_seed(0)
        idx = np.where(test)[0]
        copies = [perturb(texts[i], sm["perturb_pct"], rng) for i in idx for _ in range(sm["copies"])]
        resp = client.generate_batch(copies, temperature=0.0, max_new_tokens=sm["max_new_tokens"])
        frac = np.array([np.mean([refused(r, prefixes) for r in resp[j * sm["copies"]:(j + 1) * sm["copies"]]])
                         for j in range(len(idx))])
        df["smoothllm"] = np.nan
        df.loc[idx, "smoothllm"] = frac

    # Latency (wrapper-only, unbatched) -----------------------------------------------
    lat_rows = []
    sample = random.Random(0).sample([int(v) for v in np.where(test)[0]], min(e12["latency_subset"], int(test.sum())))

    def timed(fn):
        client.synchronize()
        t0 = time.perf_counter()
        fn()
        client.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for i in sample[:3]:
        client.encode_all_layers([texts[i]])  # warm-up
    lat_rows.append({"method": "probe (single forward pass)", "latency_ms": np.mean(
        [timed(lambda i=i: client.encode_hidden([texts[i]], "last", -1)) for i in sample])})
    lat_rows.append({"method": "perplexity", "latency_ms": np.mean(
        [timed(lambda i=i: log_perplexity(client, [texts[i]], 1)) for i in sample])})
    if primary:
      lat_rows.append({"method": "smoothllm", "latency_ms": np.mean(
        [timed(lambda i=i: client.generate_batch([perturb(texts[i], sm["perturb_pct"], random.Random(i))
                                                  for _ in range(sm["copies"])], temperature=0.0,
                                                 max_new_tokens=sm["max_new_tokens"])) for i in sample[:10]])})
    release_client(bb)

    if e12["llama_guard"].get("enabled", False) and "llama_guard" not in df.columns and primary:
        try:
            df["llama_guard"] = np.nan
            mi = np.where(df["in_main"].to_numpy() | (y == 1))[0]
            df.loc[mi, "llama_guard"] = llama_guard_scores(cfg, [texts[i] for i in mi])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Llama Guard unavailable (%s); skipped.", exc)
            df = df.drop(columns=["llama_guard"])

    # PSO signals for every row from the same persona-view embeddings (Eq. 1; D~ = D/2) ----------
    U = PV.astype(np.float32)
    U = U / np.linalg.norm(U, axis=-1, keepdims=True)
    df["s_rb"] = (U[:, 0] * U[:, 1]).sum(-1)
    df["s_rj"] = (U[:, 0] * U[:, 2]).sum(-1)
    df["s_bj"] = (U[:, 1] * U[:, 2]).sum(-1)
    df["disagreement_score"] = (3 - df["s_rb"] - df["s_rj"] - df["s_bj"]) / 3 / 2
    an = pd.read_csv(run_dir(cfg, bb, "main") / "analysis.csv")[["prompt_id", "disagreement_score"]]
    chk = df.merge(an, on="prompt_id", suffixes=("", "_main"))
    logger.info("max |D~ (exp12) - D~ (main analysis)| = %.2e",
                float(np.abs(chk["disagreement_score"] - chk["disagreement_score_main"]).max()))
    df.drop(columns=[c for c in df.columns if c.startswith("probe_")], errors="ignore").to_csv(scores_path, index=False)

    # Probes ----------------------------------------------------------------------------
    groups = df["family"].to_numpy()
    Cs = e12["probe_C"]

    def probe(X, name):
        m, c, cv = fit_probe(X[calib], y[calib], groups[calib], Cs)
        df[name] = m.decision_function(X)
        return c, cv

    layer_rows = []
    best_layer, best_cv = None, -1
    for l in range(n_layers):
        m, c, cv = fit_probe(H[calib, l].astype(np.float32), y[calib], groups[calib], Cs[1:3])
        layer_rows.append({"layer": l, "cv_auc_calib": cv, "test_auc": M.auc(y[test], m.decision_function(
            H[test, l].astype(np.float32))), "C": c})
        if cv > best_cv:
            best_layer, best_cv = l, cv
    save_table(pd.DataFrame(layer_rows), cfg, f"exp12_probe_layers_{bb}")
    info = {}
    info["probe_last"] = probe(H[:, best_layer].astype(np.float32), "probe_last") + (best_layer,)
    info["probe_final"] = probe(H[:, -1].astype(np.float32), "probe_final") + (n_layers - 1,)
    info["probe_red"] = probe(PV[:, 0].astype(np.float32), "probe_red") + (n_layers - 1,)
    info["probe_3views"] = probe(PV.reshape(len(PV), -1).astype(np.float32), "probe_3views") + (n_layers - 1,)
    info["probe_pairwise"] = probe(df[["s_rb", "s_rj", "s_bj"]].to_numpy(dtype=float), "probe_pairwise") + (n_layers - 1,)

    # Evaluation ----------------------------------------------------------------------
    cal_auc = M.auc(y[calib], df.loc[calib, "disagreement_score"].to_numpy())
    df["msds"] = df["disagreement_score"]
    df["msds_dircal"] = (1 if cal_auc >= 0.5 else -1) * df["disagreement_score"]
    df.to_csv(out_dir / "scores_with_probes.csv", index=False)
    methods = ["msds", "msds_dircal", "probe_pairwise", "probe_red", "probe_3views", "probe_final", "probe_last",
               "log_ppl", "hiddendetect"] + [m for m in ("smoothllm", "llama_guard") if m in df.columns]
    harmless_test = test & (y == 0)
    rows = []
    for m in methods:
        s = df[m].to_numpy(dtype=float)
        has = np.isfinite(s)
        if m == "smoothllm":  # fixed SmoothLLM rule: flagged when the majority of copies refuse
            t = 0.5
        else:
            t = youden_threshold(s[calib & has], y[calib & has], ev["calibration_lambda"])
        tm = test & has
        pred = s[tm] >= t
        row = {"backbone": bb, "method": m, "threshold": t, "n_test": int(tm.sum()),
               "test_auc": M.auc(y[tm], s[tm]), "test_tpr": M.tpr(y[tm], pred), "test_fpr": M.fpr(y[tm], pred),
               "test_f2": M.f_beta(y[tm], pred, ev["f_beta"])}
        if m in info:
            row.update(probe_C=info[m][0], probe_cv_auc=info[m][1], layer=info[m][2])
        for name in e12["transfer_benchmarks"]:
            pos = (~df["in_main"].to_numpy()) & (df["dataset"] == name).to_numpy() & has
            neg = harmless_test & has
            if pos.any() and neg.any():
                yy = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
                ss = np.r_[s[pos], s[neg]]
                row[f"auc_{name}"] = M.auc(yy, ss)
                row[f"tpr_{name}"] = float((s[pos] >= t).mean())
        rows.append(row)
    save_table(pd.DataFrame(rows), cfg, f"exp12_baselines_{bb}")
    save_table(pd.DataFrame(lat_rows), cfg, f"exp12_latency_{bb}")
    feats_path.unlink(missing_ok=True)  # ~0.6 GB per backbone; the per-prompt probe scores are kept


if __name__ == "__main__":
    main()
