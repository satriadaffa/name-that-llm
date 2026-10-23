"""Multi-view n-gram LinearSVC (char / word / word-shape / function-word views), validated with GroupKFold
on host train. Extra data (public prvsiyan dataset, our own generations) is added to every training fold;
scoring is on host passages only.

usage: python src/cv_multiview.py [host|prv|prv+own]  (default: runs all three)"""
import glob
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("""a an the and or but if while although because as so than that which who whom whose this these
those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from
with without for of by over under between through into about is are was were be been being have has had do does did can
could will would shall should may might must not no nor all any some each every both either neither other another such
only also just very too more most less least here there when where why how what whether then now still yet even much many
few little""".split())
# (view, ngram lo, ngram hi, max_features, weight)
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5),
         ("function", 1, 3, 50000, 0.5)]


def shape_text(t):  # letters -> x / X: keeps punctuation, capitalisation and word-length rhythm
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):  # content words -> CONTENT: keeps syntax, drops topic
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def featurize(fit_texts, apply_texts):
    fit_mats, apply_mats = [], [[] for _ in apply_texts]
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "function" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        fit_mats.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        for k, texts in enumerate(apply_texts):
            apply_mats[k].append(vec.transform([prep(t) for t in texts]) * weight)
    return hstack(fit_mats, format="csr"), [hstack(m, format="csr") for m in apply_mats]


def load_extra(which):
    parts = []
    if "prv" in which:
        prv = pd.read_csv("data/external/prvsiyan/passages_clean.csv")
        prv = prv[prv.split != "train_regen"]  # answers to reconstructed *train* prompts -> would leak into CV
        parts.append(prv[["model", "text"]])
    if "own" in which:
        own = pd.concat([pd.read_csv(f) for f in ["data/pilot_generated.csv"] + glob.glob("data/gen_data_*.csv")])
        parts.append(own[~own.text.astype(str).str.startswith("ERROR")][["model", "text"]])
    if not parts:
        return pd.DataFrame(columns=["model", "text"])
    extra = pd.concat(parts)
    extra["text"] = extra.text.astype(str).str.replace(r"\s*\n+\s*", " ", regex=True).str.strip()
    return extra


train = pd.read_csv("data/train.csv")
y = train.model.values
variants = sys.argv[1:] or ["host", "prv", "prv+own"]
for which in [v for v in variants if v != "__import_only__"]:
    extra = load_extra(which)
    oof = np.empty(len(train), dtype=object)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(train, y, train.prompt_id):
        fit = pd.concat([train.iloc[tr_idx][["model", "text"]], extra])
        Xtr, (Xva,) = featurize(fit.text.tolist(), [train.text.iloc[va_idx].tolist()])
        svc = LinearSVC(C=0.5, max_iter=5000, class_weight="balanced", random_state=42).fit(Xtr, fit.model)
        oof[va_idx] = svc.predict(Xva)
    fam = lambda s: pd.Series(s).str.extract(r"^(gpt|gemini|claude|qwen|glm|kimi)")[0].values
    print(f"{which:8s} extra={len(extra):5d} | balanced acc {balanced_accuracy_score(y, oof):.4f} | "
          f"family acc {(fam(y) == fam(oof)).mean():.3f}", flush=True)
    pd.DataFrame({"id": train.id, "true": y, "pred": oof}).to_csv(f"oof_mv_{which}.csv", index=False)
