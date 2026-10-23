"""Baseline: TF-IDF (char + word n-grams) + linear model, validated with GroupKFold by prompt."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

N_SPLITS = 5

train = pd.read_csv("data/train.csv")
X_text, y, groups = train["text"], train["model"].values, train["prompt_id"].values


def featurize(fit_text, apply_texts):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False)
    char.fit(fit_text)
    word.fit(fit_text)
    return [hstack([char.transform(t), word.transform(t)]).tocsr() for t in apply_texts]


models = {
    "logreg": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    "linsvc": lambda: LinearSVC(C=0.5, class_weight="balanced"),
}

gkf = GroupKFold(n_splits=N_SPLITS)
for name, make in models.items():
    oof = np.empty(len(train), dtype=object)
    for tr_idx, va_idx in gkf.split(X_text, y, groups):
        Xtr, Xva = featurize(X_text.iloc[tr_idx], [X_text.iloc[tr_idx], X_text.iloc[va_idx]])
        clf = make().fit(Xtr, y[tr_idx])
        oof[va_idx] = clf.predict(Xva)
    family = lambda s: pd.Series(s).str.extract(r"^(gpt|gemini|claude|qwen|glm|kimi)")[0].values
    print(f"{name}: balanced acc = {balanced_accuracy_score(y, oof):.4f} | "
          f"family acc = {(family(y) == family(oof)).mean():.4f}")
    pd.DataFrame({"id": train["id"], "true": y, "pred": oof}).to_csv(f"oof_{name}.csv", index=False)
