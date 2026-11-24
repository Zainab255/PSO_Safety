#!/usr/bin/env python3
"""Cohen's kappa between two annotators of the ambiguity set (see annotation/README.md)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

a, b = (pd.read_csv(p) for p in sys.argv[1:3])
key = pd.read_csv(Path(__file__).with_name("key_do_not_share_with_annotators.csv"))
cols = [c for c in a.columns if c.startswith("Q")]
norm = lambda s: s.astype(str).str.strip().str.lower()  # noqa: E731
for c in cols:
    x, y = norm(a[c]), norm(b[c])
    ok = (x != "nan") & (y != "nan")
    print(f"{c}: n={ok.sum()}  agreement={float((x[ok] == y[ok]).mean()):.3f}  kappa={cohen_kappa_score(x[ok], y[ok]):.3f}")
q3 = [c for c in cols if c.startswith("Q3")][0]
for name, d in (("A", a), ("B", b)):
    m = d.merge(key, on="item")
    print(f"annotator {name} vs intended category: agreement={float((norm(m[q3]) == m['subcategory']).mean()):.3f}")
q1 = [c for c in cols if c.startswith("Q1")][0]
flag = a[(norm(a[q1]) == "no") | (norm(b[q1]) == "no")]["item"].tolist()
print(f"items judged not harmless by at least one annotator: {flag}")
