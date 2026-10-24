"""Loaders for the public benchmarks used in the paper (Section IV-B-3).

No prompts ship with the code: every loader reads the file named in
``config.yaml`` and fails with an explicit message when it is missing.

Dataset names used across the code base:
    xstest_safe     XSTest safe prompts (harmless, n=250)
    xstest_unsafe   XSTest unsafe contrast prompts (explicit harmful, n=200)
    advbench        AdvBench harmful behaviours (adversarial)
    harmbench       HarmBench text behaviours (adversarial)
    jailbreakbench  JBB-Behaviors harmful behaviours (adversarial)
    ambiguous       harmless-but-ambiguous prompts (Exp. 8)
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config import resolve_path

logger = logging.getLogger(__name__)

HARMLESS = "harmless"
EXPLICIT_HARMFUL = "explicit_harmful"
ADVERSARIAL = "adversarial"
AMBIGUOUS = "ambiguous"

ADVERSARIAL_DATASETS = ("advbench", "harmbench", "jailbreakbench")
DATASET_NAMES = ("xstest_safe", "xstest_unsafe", *ADVERSARIAL_DATASETS, "ambiguous")


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    dataset: str
    text: str
    category: str        # sub-category used in the per-category tables
    group: str           # harmless | explicit_harmful | adversarial | ambiguous
    ground_truth: int    # 1 = harmful

    def as_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------

def _read_table(path_str: str, dataset: str) -> pd.DataFrame:
    path = resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found at {path}. Download it (see data/README.md) "
            f"and set datasets.{dataset}.path in config/config.yaml."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}' for dataset '{dataset}' ({path}).")


def _require_columns(df: pd.DataFrame, columns: Iterable[Optional[str]], dataset: str) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset '{dataset}' is missing column(s) {missing}; available: {list(df.columns)}. "
            f"Adjust the column names under datasets in config/config.yaml."
        )


def _ids(df: pd.DataFrame, id_column: Optional[str], prefix: str) -> List[str]:
    if id_column:
        return [f"{prefix}-{v}" for v in df[id_column]]
    return [f"{prefix}-{i}" for i in range(len(df))]


# ----------------------------------------------------------------------
# Individual datasets
# ----------------------------------------------------------------------

def _load_xstest(cfg: Dict, want_label: str) -> List[PromptRecord]:
    c = cfg["datasets"]["xstest"]
    df = _read_table(c["path"], "xstest")
    _require_columns(df, [c["id_column"], c["prompt_column"], c["label_column"], c["type_column"]], "xstest")
    df = df[df[c["label_column"]].astype(str).str.lower() == want_label.lower()].reset_index(drop=True)
    ids = _ids(df, c["id_column"], "xstest")

    if want_label == c["safe_label"]:
        mapping: Dict[str, str] = c["safe_subcategories"]
        unmapped = sorted(set(df[c["type_column"]]) - set(mapping))
        if unmapped:
            raise KeyError(f"XSTest safe types without a sub-category in config: {unmapped}")
        return [
            PromptRecord(pid, "xstest_safe", str(text), mapping[t], HARMLESS, 0)
            for pid, text, t in zip(ids, df[c["prompt_column"]], df[c["type_column"]])
        ]
    return [
        PromptRecord(pid, "xstest_unsafe", str(text), c["unsafe_category"], EXPLICIT_HARMFUL, 1)
        for pid, text in zip(ids, df[c["prompt_column"]])
    ]


def advbench_subcategory(text: str, heuristic: Dict) -> str:
    """Surface-form heuristic of Section IV-B-3: first matching marker class wins."""
    lower = text.lower()
    for cls in heuristic["order"]:
        for marker in heuristic["markers"][cls]:
            if re.search(r"\b" + re.escape(marker.lower()) + r"\b", lower):
                return cls
    return heuristic["default"]


def _load_advbench(cfg: Dict) -> List[PromptRecord]:
    c = cfg["datasets"]["advbench"]
    df = _read_table(c["path"], "advbench")
    _require_columns(df, [c["id_column"], c["prompt_column"], c["category_column"]], "advbench")
    ids = _ids(df, c["id_column"], "advbench")
    texts = [str(t) for t in df[c["prompt_column"]]]
    if c["category_column"]:
        cats = [str(v) for v in df[c["category_column"]]]
    else:
        cats = [advbench_subcategory(t, c["subcategory_heuristic"]) for t in texts]
    counts = pd.Series(cats).value_counts().to_dict()
    logger.info("AdvBench sub-categories: %s", counts)
    return [PromptRecord(pid, "advbench", t, cat, ADVERSARIAL, 1) for pid, t, cat in zip(ids, texts, cats)]


def _load_harmbench(cfg: Dict) -> List[PromptRecord]:
    c = cfg["datasets"]["harmbench"]
    df = _read_table(c["path"], "harmbench")
    _require_columns(
        df, [c["id_column"], c["prompt_column"], c["context_column"], c["category_column"],
             c["functional_category_column"]], "harmbench",
    )
    if c["keep_functional_categories"]:
        df = df[df[c["functional_category_column"]].isin(c["keep_functional_categories"])].reset_index(drop=True)
    ids = _ids(df, c["id_column"], "harmbench")
    records = []
    for pid, (_, row) in zip(ids, df.iterrows()):
        text = str(row[c["prompt_column"]])
        context = str(row[c["context_column"]]).strip() if c["context_column"] else ""
        if context:
            text = context + c["context_separator"] + text
        records.append(PromptRecord(pid, "harmbench", text, str(row[c["category_column"]]), ADVERSARIAL, 1))
    return records


def _load_jailbreakbench(cfg: Dict) -> List[PromptRecord]:
    c = cfg["datasets"]["jailbreakbench"]
    df = _read_table(c["path"], "jailbreakbench")
    _require_columns(df, [c["id_column"], c["prompt_column"], c["category_column"]], "jailbreakbench")
    ids = _ids(df, c["id_column"], "jbb")
    return [
        PromptRecord(pid, "jailbreakbench", str(t), str(cat), ADVERSARIAL, 1)
        for pid, t, cat in zip(ids, df[c["prompt_column"]], df[c["category_column"]])
    ]


def _load_ambiguous(cfg: Dict) -> List[PromptRecord]:
    c = cfg["datasets"]["ambiguous"]
    df = _read_table(c["path"], "ambiguous")
    _require_columns(df, [c["id_column"], c["prompt_column"], c["category_column"]], "ambiguous")
    ids = _ids(df, c["id_column"], "ambiguous")
    return [
        PromptRecord(pid, "ambiguous", str(t), str(cat), AMBIGUOUS, 0)
        for pid, t, cat in zip(ids, df[c["prompt_column"]], df[c["category_column"]])
    ]


_LOADERS = {
    "xstest_safe": lambda cfg: _load_xstest(cfg, cfg["datasets"]["xstest"]["safe_label"]),
    "xstest_unsafe": lambda cfg: _load_xstest(cfg, cfg["datasets"]["xstest"]["unsafe_label"]),
    "advbench": _load_advbench,
    "harmbench": _load_harmbench,
    "jailbreakbench": _load_jailbreakbench,
    "ambiguous": _load_ambiguous,
}


def load_dataset(cfg: Dict, name: str) -> List[PromptRecord]:
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_LOADERS)}")
    records = _LOADERS[name](cfg)
    if not records:
        raise ValueError(f"Dataset '{name}' loaded zero prompts; check the file and column settings.")
    logger.info("Loaded %d prompts from %s", len(records), name)
    return records


def load_corpus(cfg: Dict, names: Sequence[str]) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    for name in names:
        records.extend(load_dataset(cfg, name))
    ids = [r.prompt_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prompt ids across datasets; set distinct id columns.")
    return records


def load_jailbreak_templates(cfg: Dict) -> List[str]:
    c = cfg["datasets"]["jailbreak_templates"]
    df = _read_table(c["path"], "jailbreak_templates")
    _require_columns(df, [c["template_column"]], "jailbreak_templates")
    templates = [str(t) for t in df[c["template_column"]] if c["placeholder"] in str(t)]
    skipped = len(df) - len(templates)
    if skipped:
        logger.warning("Skipped %d jailbreak templates without placeholder %r", skipped, c["placeholder"])
    if not templates:
        raise ValueError(f"No jailbreak template contains the placeholder {c['placeholder']!r}.")
    return templates


# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------

def stratified_subset(records: Sequence[PromptRecord], n: int, seed: int) -> List[PromptRecord]:
    """Sample n records keeping category proportions (largest-remainder allocation)."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_cat: Dict[str, List[PromptRecord]] = defaultdict(list)
    for r in records:
        by_cat[r.category].append(r)
    total = len(records)
    quotas = {c: n * len(rs) / total for c, rs in by_cat.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    for c in sorted(quotas, key=lambda c: quotas[c] - alloc[c], reverse=True)[: n - sum(alloc.values())]:
        alloc[c] += 1
    out: List[PromptRecord] = []
    for c in sorted(by_cat):
        out.extend(rng.sample(by_cat[c], alloc[c]))
    return out


def stratified_indices_by_label(labels: Sequence[int], n: int, seed: int) -> List[int]:
    """Indices of n items keeping the ground-truth class ratio (calibration corpus)."""
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    n_pos = round(n * len(pos) / len(labels))
    return sorted(rng.sample(pos, n_pos) + rng.sample(neg, n - n_pos))


@dataclass(frozen=True)
class Turn:
    text: str
    harmful: bool
    prompt_id: str


def build_multiturn_sequences(
    safe: Sequence[PromptRecord], adversarial: Sequence[PromptRecord], cfg: Dict, seed: int
) -> Dict[str, List[List[Turn]]]:
    """Multi-turn corpus of Experiment 9.

    gradual:  turns 1..benign_turns from XSTest safe, remaining turns from AdvBench
    abrupt:   the same harmful turns presented from turn 1, without benign context
    harmless: all turns from XSTest safe (false-alarm control)
    """
    c = cfg["experiments"]["exp9"]
    rng = random.Random(seed)
    n, turns, benign = c["n_sequences"], c["turns"], c["benign_turns"]
    harmful_turns = turns - benign
    if harmful_turns < 1:
        raise ValueError("experiments.exp9.turns must exceed benign_turns.")

    def pick(pool: Sequence[PromptRecord], k: int, harmful: bool) -> List[Turn]:
        return [Turn(r.text, harmful, r.prompt_id) for r in rng.sample(list(pool), k)]

    gradual, abrupt, harmless = [], [], []
    for _ in range(n):
        harm = pick(adversarial, harmful_turns, True)
        gradual.append(pick(safe, benign, False) + harm)
        abrupt.append(list(harm))
        harmless.append(pick(safe, turns, False))
    return {"gradual": gradual, "abrupt": abrupt, "harmless": harmless}
