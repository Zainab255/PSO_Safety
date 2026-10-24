"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import backbone_config, load_config, output_dir, resolve_path  # noqa: E402
from core.predictive_model import PredictiveSafetyModel  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from models.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("experiments")

_CLIENTS: Dict[str, LLMClient] = {}


def setup(description: str, argv: Optional[Sequence[str]] = None,
          extra: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--backbone", default=None, help="Backbone key from config (default: backbones.primary)")
    if extra:
        extra(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_config(args.config)
    args.backbone = args.backbone or cfg["backbones"]["primary"]
    return cfg, args


def load_client(cfg: Dict, key: str) -> LLMClient:
    if key not in _CLIENTS:
        _CLIENTS[key] = LLMClient(backbone_config(cfg, key)).load()
    return _CLIENTS[key]


def release_client(key: str) -> None:
    import gc

    import torch
    _CLIENTS.pop(key, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_predictive_model(cfg: Dict) -> Optional[PredictiveSafetyModel]:
    path = resolve_path(cfg["paths"]["predictive_model"])
    if path.exists():
        return PredictiveSafetyModel.load(path)
    logger.info("No fitted Predictive Safety Model at %s; p_unsafe will be NaN.", path)
    return None


def run_dir(cfg: Dict, backbone: str, *parts: str) -> Path:
    return output_dir(cfg, "runs", backbone, *parts)


def load_main(cfg: Dict, backbone: str) -> Dict[str, pd.DataFrame]:
    """Concatenate the cached main-evaluation outputs of ``run_main_evaluation``."""
    d = run_dir(cfg, backbone, "main")
    analysis = d / "analysis.csv"
    if not analysis.exists():
        raise FileNotFoundError(
            f"{analysis} not found. Run `python -m experiments.run_main_evaluation --backbone {backbone}` first."
        )
    seeds = cfg["evaluation"]["seeds"]
    evals = [pd.read_csv(p) for s in seeds for p in sorted(d.glob(f"eval_seed{s}_*.csv"))]
    resps = [pd.read_csv(d / f"responses_seed{s}.csv") for s in seeds if (d / f"responses_seed{s}.csv").exists()]
    if not evals:
        raise FileNotFoundError(f"No evaluation files in {d}; run the main evaluation first.")
    return {
        "analysis": pd.read_csv(analysis),
        "evaluation": pd.concat(evals, ignore_index=True),
        "responses": pd.concat(resps, ignore_index=True) if resps else pd.DataFrame(),
    }


def table_path(cfg: Dict, name: str) -> Path:
    return output_dir(cfg, "tables") / f"{name}.csv"


def figure_path(cfg: Dict, name: str) -> Path:
    return output_dir(cfg, "figures") / f"{name}.png"


def save_table(df: pd.DataFrame, cfg: Dict, name: str) -> Path:
    path = table_path(cfg, name)
    df.to_csv(path, index=False)
    logger.info("Wrote %s\n%s", path, df.to_string(index=False))
    return path


def category_order(cfg: Dict, analysis: pd.DataFrame) -> List[str]:
    """Harmless sub-categories (config order), then explicit harmful, then adversarial ones."""
    harmless = list(dict.fromkeys(cfg["datasets"]["xstest"]["safe_subcategories"].values()))
    present = set(analysis["prompt_category"])
    order = [c for c in harmless if c in present]
    order += sorted(present - set(order) - {cfg["datasets"]["xstest"]["unsafe_category"]}
                    - set(analysis.loc[analysis["group"] == "adversarial", "prompt_category"]))
    if cfg["datasets"]["xstest"]["unsafe_category"] in present:
        order.append(cfg["datasets"]["xstest"]["unsafe_category"])
    order += sorted(set(analysis.loc[analysis["group"] == "adversarial", "prompt_category"]))
    return order


def method_run_metrics(ev: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """TPR, FPR, UOR, AUC, F-beta, PTR and mean latency per (method, seed)."""
    pos = cfg["evaluation"]["positive_decisions"]
    beta = cfg["evaluation"]["f_beta"]
    rows = []
    for (method, seed), g in ev.groupby(["method", "seed"], sort=False):
        y = g["ground_truth"].to_numpy()
        pred = M.flagged(g["decision"], pos)
        scores = M.pipeline_scores(g["risk_score"].to_numpy(), g["decided_by"].fillna(""))
        rows.append({
            "method": method, "seed": seed, "tpr": M.tpr(y, pred), "fpr": M.fpr(y, pred),
            "ptr": float(np.mean(g.loc[y == 0, "decision"] == "ALLOW")),
            "uor": M.unsafe_output_rate(y, g["output_unsafe"].to_numpy(dtype=float)),
            "auc": M.auc(y, scores), "f_beta": M.f_beta(y, pred, beta),
            "latency_ms": float(g["latency_ms"].mean()),
            "calls_min": int(g["model_calls"].min()), "calls_max": int(g["model_calls"].max()),
        })
    return pd.DataFrame(rows)
