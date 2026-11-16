"""Model-implied (expected) macro F0.5 under calibrated probabilities vs actual: holdout, then test (no labels).
Tells whether the model is overconfident on test (expected >> leaderboard) or knows test is harder."""
import json, pickle, sys
import numpy as np
import polars as pl
from decision import P_FLOOR, M_CAP, partition, shift
from io_utils import ROOT, load_source
from metrics import macro_f05
from splits import load_splits, universe_gt
from submit import decide

def best_value(ps):
    m = len(ps)
    if m == 0:
        return 1.0
    suf = [None] * (m + 1); suf[m] = [1.0]
    for k in range(m - 1, -1, -1):
        p, prev = ps[k], suf[k + 1]; nd = [0.0] * (len(prev) + 1)
        for i, v in enumerate(prev):
            nd[i] += v * (1 - p); nd[i + 1] += v * p
        suf[k] = nd
    best, pre = suf[0][0], [1.0]
    for k in range(1, m + 1):
        p = ps[k - 1]; nd = [0.0] * (len(pre) + 1)
        for i, v in enumerate(pre):
            nd[i] += v * (1 - p); nd[i + 1] += v * p
        pre = nd; e = 0.0
        for a in range(1, len(pre)):
            if pre[a] < 1e-9: continue
            for b, pb in enumerate(suf[k]):
                if pb >= 1e-9: e += pre[a] * pb * 1.25 * a / (0.25 * (a + b) + k)
        best = max(best, e)
    return best

def expected_macro(d, pcol, s1_all):
    x = d.filter(pl.col(pcol) >= P_FLOOR).sort(["s1", pcol], descending=[False, True])
    s1 = x["s1"].to_numpy(); p = x[pcol].to_numpy().astype(float)
    vals = {}
    if len(x):
        starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]]); ends = np.r_[starts[1:], len(x)]
        for a, b in zip(starts, ends):
            vals[int(s1[a])] = best_value(p[a:min(b, a + M_CAP)].tolist())
    ev = np.array([vals.get(int(s), 1.0) for s in s1_all])
    return ev

dec = json.loads((ROOT / "models/decision.json").read_text()); iso = pickle.load(open(ROOT / "models/iso.pkl", "rb"))
delta = dec["param"] if dec["method"] == "expected_f" else 0.0
def prep(d):
    d = partition(d, "p2")
    return d.with_columns(pl.Series("_p", shift(iso.predict(d["p2"].to_numpy()), delta).astype(np.float32)))

tr = pl.read_parquet("../work/pred_l2_train.parquet")
sp_ = load_splits().filter(~pl.col("dropped") & (pl.col("fold") < 0))
gt = universe_gt(False)
for c in ("India", "US"):
    univ = sp_.filter(pl.col("country") == c).select("s1")
    d = prep(tr.join(univ, on="s1", how="semi"))
    ev = expected_macro(d, "_p", univ["s1"].to_list())
    act = macro_f05(decide(d, "_p", "expected_f", 0.0), gt.join(univ, on="s1", how="semi"), univ)
    print(f"holdout {c:6s} expected {ev.mean():.5f}  actual {act:.5f}", flush=True)
te = pl.read_parquet("../work/pred_l2_test.parquet")
s1c = load_source("test", 1).select(pl.col("idx").alias("s1"), "country")
for c in ("France", "India", "US"):
    univ = s1c.filter(pl.col("country") == c).select("s1")
    d = prep(te.join(univ, on="s1", how="semi"))
    ev = expected_macro(d, "_p", univ["s1"].to_list())
    print(f"test    {c:6s} expected {ev.mean():.5f}", flush=True)
