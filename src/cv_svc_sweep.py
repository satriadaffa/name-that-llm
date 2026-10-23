"""Sweep LinearSVC C and view weights for the fork's n-gram member (train fold + public pool).
Per-view TF-IDF blocks are built once per fold and re-weighted per config."""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.argv = sys.argv[:1] + ["__import_only__"]
from cv_multiview import VIEWS, function_text, shape_text  # noqa: E402

# name: (C, weights for char, word, shape, function)
CONFIGS = {
    "fork (C0.5)": (0.5, (1.0, 0.7, 0.5, 0.5)),
    "C0.25": (0.25, (1.0, 0.7, 0.5, 0.5)),
    "C1.0": (1.0, (1.0, 0.7, 0.5, 0.5)),
    "equal views": (0.5, (1.0, 1.0, 1.0, 1.0)),
    "char-heavy": (0.5, (1.0, 0.5, 0.3, 0.3)),
}

train = pd.read_csv("data/train.csv")
pool = pd.read_parquet("data/external/pool/pool_v1.parquet")[["model", "text"]]
y = train.model.values
oof = {name: np.empty(len(train), dtype=object) for name in CONFIGS}

for k, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=5).split(train, y, train.prompt_id)):
    fit = pd.concat([train.iloc[tr_idx][["model", "text"]], pool], ignore_index=True)
    va_text = train.text.iloc[va_idx].tolist()
    blocks_tr, blocks_va = [], []
    for view, lo, hi, max_features, _ in VIEWS:
        prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "function" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        blocks_tr.append(vec.fit_transform([prep(t) for t in fit.text]))
        blocks_va.append(vec.transform([prep(t) for t in va_text]))
    for name, (C, w) in CONFIGS.items():
        Xtr = hstack([b * wi for b, wi in zip(blocks_tr, w)], format="csr")
        Xva = hstack([b * wi for b, wi in zip(blocks_va, w)], format="csr")
        svc = LinearSVC(C=C, max_iter=5000, class_weight="balanced", random_state=42).fit(Xtr, fit.model)
        oof[name][va_idx] = svc.predict(Xva)
    print(f"fold {k} done", flush=True)

for name, o in oof.items():
    print(f"{name:12s}: balanced acc {balanced_accuracy_score(y, o):.4f}", flush=True)
