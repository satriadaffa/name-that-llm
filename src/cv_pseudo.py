"""Transductive pseudo-labelling test: does adding the most confident held-out predictions back into training
help? Each GroupKFold validation fold plays the role of the unseen-prompt test set.
Training data = host train fold + public pool (as in the fork's n-gram member)."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.argv = sys.argv[:1] + ["__import_only__"]
from cv_multiview import featurize  # noqa: E402

FRACS = [0.3, 0.5]  # share of the most confident val passages added back as pseudo-labels
PSEUDO_W = 1.0

train = pd.read_csv("data/train.csv")
pool = pd.read_parquet("data/external/pool/pool_v1.parquet")[["model", "text"]]
y = train.model.values
oof = {"base": np.empty(len(train), dtype=object), **{f: np.empty(len(train), dtype=object) for f in FRACS}}

for k, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=5).split(train, y, train.prompt_id)):
    fit = pd.concat([train.iloc[tr_idx][["model", "text"]], pool], ignore_index=True)
    va_text = train.text.iloc[va_idx].tolist()
    Xtr, (Xva,) = featurize(fit.text.tolist(), [va_text])
    svc = LinearSVC(C=0.5, max_iter=5000, class_weight="balanced", random_state=42).fit(Xtr, fit.model)
    dec = svc.decision_function(Xva)
    srt = np.sort(dec, axis=1)
    margin = srt[:, -1] - srt[:, -2]  # confidence = gap between best and runner-up class
    pred = svc.classes_[dec.argmax(1)]
    oof["base"][va_idx] = pred
    for f in FRACS:
        keep = margin >= np.quantile(margin, 1 - f)
        X2 = pd.concat([fit, pd.DataFrame({"model": pred[keep], "text": np.array(va_text)[keep]})],
                       ignore_index=True)
        w = np.r_[np.ones(len(fit)), np.full(keep.sum(), PSEUDO_W)]
        Xtr2, (Xva2,) = featurize(X2.text.tolist(), [va_text])
        svc2 = LinearSVC(C=0.5, max_iter=5000, class_weight="balanced", random_state=42)
        svc2.fit(Xtr2, X2.model, sample_weight=w)
        oof[f][va_idx] = svc2.predict(Xva2)
        print(f"fold {k} frac {f}: pseudo precision {(pred[keep] == y[va_idx][keep]).mean():.3f}", flush=True)
    print(f"fold {k} done", flush=True)

for name, o in oof.items():
    print(f"{str(name):5s}: balanced acc {balanced_accuracy_score(y, o):.4f}")
