"""Adversarial validation: which pair features distinguish test from the train universe?
Uses each record's top candidate by first-stage score p0 (independent of the matcher)."""
import sys
import numpy as np
import polars as pl
import lightgbm as lgb
from features import feats_path, KEY

N = 300_000
def best_pairs(split, country, seed):
    p = feats_path(split, False, country)
    k = pl.read_parquet(p, columns=KEY + ["p0"])
    b = k.sort("p0", descending=True).unique(subset=["src", "rid"], keep="first")
    b = b.filter(pl.col("p0") > 0.05)          # records that plausibly have a match
    b = b.sample(min(N, b.height), seed=seed).select(KEY)
    return pl.scan_parquet(p).join(b.lazy(), on=KEY, how="semi").collect()

for country in sys.argv[1:] or ["US", "India"]:
    a = best_pairs("train", country, 1).drop(["y", "fold"], strict=False)
    b = best_pairs("test", country, 2)
    cols = [c for c in a.columns if c not in KEY and c in b.columns]
    X = np.vstack([a.select(cols).to_numpy(), b.select(cols).to_numpy()]).astype(np.float32)
    y = np.r_[np.zeros(a.height), np.ones(b.height)]
    idx = np.random.default_rng(0).permutation(len(y)); tr, va = idx[: len(y) * 4 // 5], idx[len(y) * 4 // 5:]
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, num_threads=16, verbose=-1, metric="auc"),
                  lgb.Dataset(X[tr], y[tr], feature_name=cols), 150, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.log_evaluation(0)])
    from sklearn.metrics import roc_auc_score
    print(f"===== {country}: adversarial AUC = {roc_auc_score(y[va], m.predict(X[va])):.4f}")
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(g for _, g in imp)
    for c, g in imp[:12]:
        ma, mb = a[c].cast(pl.Float64), b[c].cast(pl.Float64)
        print(f"  {c:22s} gain {g/tot:.3f}  train mean {ma.mean():9.3f} med {ma.median():8.3f} | test mean {mb.mean():9.3f} med {mb.median():8.3f}")
