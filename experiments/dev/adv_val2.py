"""Adversarial validation on top pairs, excluding train records owned by dropped S1 (scattered distractors).
Run 1: all features. Run 2: without universe-scale features (counts, IDF, group sizes)."""
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import feats_path, KEY
from io_utils import load_gt
from splits import load_splits
drop_own = load_gt().join(load_splits().filter(pl.col("dropped")).select("s1"), on="s1", how="semi").select(["src", "rid"])
SCALE = ("cnt", "idf", "unsh", "g_n_", "s1_n", "rec_n", "_rk_", "c_n", "co_n", "g_p0_sum", "g_p0_conf", "r_rev", "r_comb", "r_name", "r_addr", "rank")
def best(split, c, seed, n=300_000):
    p = feats_path(split, False, c)
    k = pl.read_parquet(p, columns=KEY + ["p0"])
    if split == "train":
        k = k.join(drop_own, on=["src", "rid"], how="anti")
    b = k.sort("p0", descending=True).unique(subset=["src", "rid"], keep="first").filter(pl.col("p0") > 0.05)
    b = b.sample(min(n, b.height), seed=seed).select(KEY)
    return pl.scan_parquet(p).join(b.lazy(), on=KEY, how="semi").collect()
for c in ("US", "India"):
    a, b = best("train", c, 1), best("test", c, 2)
    allc = [x for x in a.columns if x in b.columns and x not in KEY + ["y", "fold"]]
    for tag, cols in (("all feats", allc), ("no scale feats", [x for x in allc if not any(s in x for s in SCALE)])):
        X = np.vstack([a.select(cols).to_numpy(), b.select(cols).to_numpy()]).astype(np.float32)
        y = np.r_[np.zeros(a.height), np.ones(b.height)]
        idx = np.random.default_rng(0).permutation(len(y)); tr, va = idx[: len(y) * 4 // 5], idx[len(y) * 4 // 5:]
        m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, num_threads=16, verbose=-1),
                      lgb.Dataset(X[tr], y[tr], feature_name=cols), 150)
        imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(g for _, g in imp)
        print(f"{c} [{tag}, {len(cols)} feats] AUC={roc_auc_score(y[va], m.predict(X[va])):.4f}  top: " +
              ", ".join(f"{n} {g/tot:.2f}" for n, g in imp[:8]), flush=True)
