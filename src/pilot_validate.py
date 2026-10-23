"""Does the generated pilot data look like the host's data?
1) format stats vs host, 2) a classifier trained on host data scored on pilot passages."""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/train.csv")
gen = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/pilot_generated.csv")
gen = gen[~gen.text.str.startswith("ERROR")].copy()
host = train[train.model.isin(gen.model.unique())]


def stats(df):
    t = df.text
    return pd.Series({"words": t.str.split().str.len().mean(), "newline%": t.str.contains("\n").mean() * 100,
                      "markdown%": t.str.contains(r"\*\*|^#|^- ", regex=True).mean() * 100,
                      "emdash%": t.str.contains("—").mean() * 100,
                      "semicolon%": t.str.contains(";").mean() * 100})


print("FORMAT host vs pilot")
print(pd.DataFrame({"host": stats(host), "pilot": stats(gen)}).round(1).to_string())
gen["text"] = gen.text.str.replace(r"\s*\n+\s*", " ", regex=True).str.strip()
print("corr(target, actual words):", round(np.corrcoef(gen.target_words, gen.text.str.split().str.len())[0, 1], 2))

char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2).fit(train.text)
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                       token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False).fit(train.text)
feat = lambda t: hstack([char.transform(t), word.transform(t)]).tocsr()
clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(feat(train.text), train.model)
gen["pred"] = clf.predict(feat(gen.text))

# reference: same classifier's OOF accuracy on the host data for the same models
oof = pd.read_csv("oof_logreg.csv")
oof = oof[oof.true.isin(gen.model.unique())]
fam = lambda s: s.str.extract(r"^(gpt|gemini|claude)")[0]
print(f"\nhost-trained clf on pilot: exact {(gen.pred == gen.model).mean():.3f} | "
      f"family {(fam(gen.pred) == fam(gen.model)).mean():.3f}")
print(f"host OOF (same models):    exact {(oof.pred == oof.true).mean():.3f} | "
      f"family {(fam(oof.pred) == fam(oof.true)).mean():.3f}")
print(pd.DataFrame({"pilot": gen.assign(ok=gen.pred == gen.model).groupby("model").ok.mean(),
                    "host_oof": oof.assign(ok=oof.pred == oof.true).groupby("true").ok.mean()}).round(2).to_string())
