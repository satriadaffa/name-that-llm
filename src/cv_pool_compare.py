"""Which public pool helps the fork's n-gram member most? Same SVC as the fork (C=0.5, fork view weights),
GroupKFold on host train, scored on host passages only."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.argv = sys.argv[:1] + ["__import_only__"]
from cv_multiview import featurize  # noqa: E402

v1 = pd.read_parquet("data/external/pool/pool_v1.parquet")[["model", "text"]]
v2 = pd.read_parquet("data/external/pool/pool_v2.parquet")
POOLS = {
    "pool_v1 (fork)": v1,
    "pool_v2 (all)": v2[["model", "text"]],
    "pool_v2 minus oleg": v2[v2.source != "oleg"][["model", "text"]],
}

train = pd.read_csv("data/train.csv")
y = train.model.values
for name, pool in POOLS.items():
    pool = pool.drop_duplicates("text")
    oof = np.empty(len(train), dtype=object)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(train, y, train.prompt_id):
        fit = pd.concat([train.iloc[tr_idx][["model", "text"]], pool], ignore_index=True)
        Xtr, (Xva,) = featurize(fit.text.tolist(), [train.text.iloc[va_idx].tolist()])
        svc = LinearSVC(C=0.5, max_iter=5000, class_weight="balanced", random_state=42).fit(Xtr, fit.model)
        oof[va_idx] = svc.predict(Xva)
    print(f"{name:20s} rows={len(pool):6d} | balanced acc {balanced_accuracy_score(y, oof):.4f}", flush=True)
