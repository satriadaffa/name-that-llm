"""How host-like is each pool source? Train the multi-view SVC on host train only, predict every pool passage,
and compare per-(source, model) accuracy with the host OOF recall of the same model (oof_mv_host.csv).
A source far below host recall is stylistically off-distribution (different settings, formatting, snapshots)."""
import sys

import pandas as pd
from sklearn.svm import LinearSVC

sys.argv = sys.argv[:1] + ["__import_only__"]
from cv_multiview import featurize  # noqa: E402  (module skips its CV loop for this variant name)

train = pd.read_csv("data/train.csv")
pool = pd.read_parquet("data/external/pool/pool_v1.parquet")
Xtr, (Xp,) = featurize(train.text.tolist(), [pool.text.tolist()])
svc = LinearSVC(C=0.5, max_iter=5000, class_weight="balanced", random_state=42).fit(Xtr, train.model)
pool["ok"] = svc.predict(Xp) == pool.model

oof = pd.read_csv("oof_mv_host.csv")
host_recall = (oof.true == oof.pred).groupby(oof.true).mean()
g = pool.groupby(["source", "model"]).ok.agg(["mean", "size"]).reset_index()
g["gap"] = g["mean"] - g.model.map(host_recall)
summary = g.groupby("source").apply(
    lambda d: pd.Series({"rows": d["size"].sum(), "models": len(d),
                         "acc": (d["mean"] * d["size"]).sum() / d["size"].sum(),
                         "gap_vs_host": (d.gap * d["size"]).sum() / d["size"].sum()}))
print(summary.sort_values("gap_vs_host").round(3).to_string())
summary.to_csv("pool_source_check.csv")
