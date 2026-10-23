"""Does generated data help? GroupKFold on host train; extra data is added to every training fold,
evaluation is on host passages only."""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

train = pd.read_csv("data/train.csv")
extra = pd.concat([pd.read_csv(f) for f in glob.glob("data/pilot_generated.csv") + glob.glob("data/gen_data_*.csv")])
extra = extra[~extra.text.astype(str).str.startswith("ERROR")]
extra["text"] = extra.text.str.replace(r"\s*\n+\s*", " ", regex=True).str.strip()
extra_w = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # sample weight for generated rows
print(f"extra rows: {len(extra)} | per model:", extra.model.value_counts().to_dict())


def featurize(fit_text, apply_texts):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False)
    char.fit(fit_text)
    word.fit(fit_text)
    return [hstack([char.transform(t), word.transform(t)]).tocsr() for t in apply_texts]


y = train.model.values
for use_extra in (False, True):
    oof = np.empty(len(train), dtype=object)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(train, y, train.prompt_id):
        tr = train.iloc[tr_idx]
        if use_extra:
            tr = pd.concat([tr.assign(w=1.0), extra.assign(w=extra_w)])
        else:
            tr = tr.assign(w=1.0)
        Xtr, Xva = featurize(tr.text, [tr.text, train.text.iloc[va_idx]])
        clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, tr.model, sample_weight=tr.w.values)
        oof[va_idx] = clf.predict(Xva)
    rec = pd.Series(oof == y).groupby(y).mean()
    print(f"\nextra={use_extra}: balanced acc {balanced_accuracy_score(y, oof):.4f}")
    if use_extra:
        print("recall on models WITH extra data:", rec[rec.index.isin(extra.model)].mean().round(3),
              "| WITHOUT:", rec[~rec.index.isin(extra.model)].mean().round(3))
    else:
        base = rec
print("\nper-model delta (extra - base):")
print((rec - base).round(2).sort_values().to_string())
