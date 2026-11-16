"""Step 1: why is test harder? (a) adversarial validation train-vs-test on plausible pairs, per country;
(b) examples of uncertain test records (v1 predictions) with raw text."""
import json
import lightgbm as lgb
import numpy as np
import polars as pl
from features import KEY, feats_path
from io_utils import load_source

DROP = set(KEY) | {"y", "fold", "country"}
rng = np.random.default_rng(0)
out = {}


def plausible(split, country, n=300_000):
    lf = pl.scan_parquet(feats_path(split, False, country))
    d = lf.filter(pl.col("p0") >= 0.3).collect()
    return d.sample(min(n, d.height), seed=1)


pairs = [("US", "US"), ("India", "India"), ("US", "France"), ("India", "France")]
for ctr, cte in pairs:
    a, b = plausible("train", ctr), plausible("test", cte)
    cols = [c for c in a.columns if c not in DROP and c in b.columns]
    X = np.vstack([a.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy(),
                   b.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()])
    y = np.r_[np.zeros(a.height), np.ones(b.height)]
    idx = rng.permutation(len(y)); tr, va = idx[: len(y) * 3 // 4], idx[len(y) * 3 // 4:]
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, num_threads=6, verbose=-1,
                       feature_fraction=0.8), lgb.Dataset(X[tr], y[tr], feature_name=cols), 150)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[va], m.predict(X[va]))
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp)
    top = [(c, round(g / tot, 3)) for c, g in imp[:12]]
    # mean of the top features on each side
    means = {c: (round(float(np.nanmean(a[c].cast(pl.Float32).to_numpy())), 4),
                 round(float(np.nanmean(b[c].cast(pl.Float32).to_numpy())), 4)) for c, _ in top[:8]}
    out[f"{ctr}_train_vs_{cte}_test"] = {"auc": round(auc, 4), "top": top, "means_train_test": means}
    print(f"{ctr} train vs {cte} test: AUC {auc:.4f}\n  top: {top}\n  means (train, test): {means}", flush=True)

json.dump(out, open("../reports/shift_adversarial.json", "w"), indent=1)

# (b) uncertain test examples, v1 predictions
te = pl.read_parquet("../work/pred_l2_test_v1.parquet")
best = te.sort("p2", descending=True).unique(subset=["src", "rid"], keep="first")
unc = best.filter(pl.col("p2").is_between(0.3, 0.7))
raw1 = load_source("test", 1).select(pl.col("idx").alias("s1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"))
rr = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"), "name", "addr")
                for s in (2, 3)])
with open("../reports/uncertain_test_examples.txt", "w") as f:
    for c in ("US", "India", "France"):
        x = unc.filter(pl.col("country") == c).sample(60, seed=2).join(raw1, on="s1").join(rr, on=["src", "rid"])
        f.write(f"===== {c} (record's best candidate p2 in 0.3-0.7)\n")
        for r in x.iter_rows(named=True):
            f.write(f"p1={r['p1']:.2f} p2={r['p2']:.2f} | {r['n1']} | {r['a1']}\n{'':18s}{r['name']} | {r['addr']}\n")
print("examples written")
