"""Does exploiting 'one passage per model per prompt' help? Uses OOF probs from the TF-IDF baseline."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

train = pd.read_csv("data/train.csv")
X_text, y, groups = train["text"], train["model"].values, train["prompt_id"].values
classes = np.unique(y)


def featurize(fit_text, apply_texts):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False)
    char.fit(fit_text)
    word.fit(fit_text)
    return [hstack([char.transform(t), word.transform(t)]).tocsr() for t in apply_texts]


oof = np.zeros((len(train), len(classes)))
for tr_idx, va_idx in GroupKFold(n_splits=5).split(X_text, y, groups):
    Xtr, Xva = featurize(X_text.iloc[tr_idx], [X_text.iloc[tr_idx], X_text.iloc[va_idx]])
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr_idx])
    oof[va_idx] = clf.predict_proba(Xva)
np.save("oof_logreg_proba.npy", oof)


def assign(proba, group_ids):
    """Within each group, give each model at most one passage (Hungarian on -log p)."""
    pred = classes[proba.argmax(1)].copy()
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        cost = -np.log(proba[idx] + 1e-9)
        if len(idx) > len(classes):  # oversized cluster: allow each model twice
            cost = np.hstack([cost, cost])
        rows, cols = linear_sum_assignment(cost)
        pred[idx[rows]] = classes[cols % len(classes)]
    return pred


print(f"argmax:                    {balanced_accuracy_score(y, classes[oof.argmax(1)]):.4f}")
print(f"assign (true prompt_id):   {balanced_accuracy_score(y, assign(oof, groups)):.4f}")

# Realistic: test has no prompt_id -> recover prompts by clustering on topic words
topic = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(X_text)
n_prompts = len(np.unique(groups))
clu = AgglomerativeClustering(n_clusters=n_prompts, metric="cosine", linkage="average").fit_predict(topic.toarray())
purity = pd.crosstab(clu, groups).max(1).sum() / len(train)
print(f"cluster purity vs prompt:  {purity:.4f}")
print(f"assign (clustered):        {balanced_accuracy_score(y, assign(oof, clu)):.4f}")
