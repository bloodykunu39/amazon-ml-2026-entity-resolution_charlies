"""Stage `decide`: partition -> calibration -> per-S1 expected-F0.5 subset (or global threshold) -> logit shift.

1. Partition: each S2/S3 record keeps only its best S1 (highest p).
2. Isotonic calibration fitted on OOF predictions (train folds 0..4).
3. Per S1: sort candidates by p; choose k maximizing E[F0.5] under independence
   (Poisson-binomial DP for TP among top-k and true matches among the rest); k=0 scores prod(1-p).
4. Global logit shift delta tuned on the test-like holdout.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from metrics import macro_f05

KEY = ["s1", "src", "rid"]
P_FLOOR = 0.003
M_CAP = 30


def partition(d: pl.DataFrame, pcol: str) -> pl.DataFrame:
    return d.sort(pcol, descending=True).unique(subset=["src", "rid"], keep="first")


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def shift(p: np.ndarray, delta: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + delta)))


def _pb(ps):
    """Poisson-binomial distribution as list (index = count)."""
    dist = [1.0]
    for p in ps:
        q = 1 - p
        nd = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nd[i] += v * q
            nd[i + 1] += v * p
        dist = nd
    return dist


def best_k(ps: list[float]) -> int:
    """ps sorted descending. Returns k maximizing expected F0.5."""
    m = len(ps)
    if m == 0:
        return 0
    # suffix distributions of true matches among ps[k:]
    suf = [None] * (m + 1)
    suf[m] = [1.0]
    for k in range(m - 1, -1, -1):
        p, q, prev = ps[k], 1 - ps[k], suf[k + 1]
        nd = [0.0] * (len(prev) + 1)
        for i, v in enumerate(prev):
            nd[i] += v * q
            nd[i + 1] += v * p
        suf[k] = nd
    best, bk = suf[0][0], 0          # k = 0: F=1 iff no true match
    pre = [1.0]
    for k in range(1, m + 1):
        p, q = ps[k - 1], 1 - ps[k - 1]
        nd = [0.0] * (len(pre) + 1)
        for i, v in enumerate(pre):
            nd[i] += v * q
            nd[i + 1] += v * p
        pre = nd
        rest = suf[k]
        e = 0.0
        for a in range(1, len(pre)):
            pa = pre[a]
            if pa < 1e-9:
                continue
            for b, pbv in enumerate(rest):
                if pbv < 1e-9:
                    continue
                e += pa * pbv * 1.25 * a / (0.25 * (a + b) + k)
        if e > best:
            best, bk = e, k
    return bk


def select_expected_f(d: pl.DataFrame, pcol: str) -> pl.DataFrame:
    """d partitioned; returns selected pairs (KEY)."""
    x = d.filter(pl.col(pcol) >= P_FLOOR).sort(["s1", pcol], descending=[False, True])
    s1 = x["s1"].to_numpy()
    p = x[pcol].to_numpy().astype(np.float64)
    keep = np.zeros(len(x), dtype=bool)
    if len(x) == 0:
        return x.select(KEY)
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    ends = np.r_[starts[1:], len(x)]
    for a, b in zip(starts, ends):
        ps = p[a:min(b, a + M_CAP)].tolist()
        k = best_k(ps)
        keep[a:a + k] = True
    return x.filter(pl.Series(keep)).select(KEY)


def select_threshold(d: pl.DataFrame, pcol: str, t: float) -> pl.DataFrame:
    return d.filter(pl.col(pcol) >= t).select(KEY)


def evaluate(d: pl.DataFrame, truth: pl.DataFrame, s1_univ: pl.DataFrame, pcol: str, iso=None,
             deltas=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0), thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """d: candidate predictions for the evaluated S1 (partitioned inside). Returns results dict."""
    d = partition(d, pcol)
    p = d[pcol].to_numpy()
    if iso is not None:
        p = iso.predict(p)
    d = d.with_columns(pl.Series("_pc", p.astype(np.float32)))
    res = {"threshold": {}, "expected_f": {}}
    for t in thresholds:
        res["threshold"][t] = macro_f05(select_threshold(d, "_pc", t), truth, s1_univ)
    for dl in deltas:
        dd = d.with_columns(pl.Series("_ps", shift(p, dl).astype(np.float32)))
        res["expected_f"][dl] = macro_f05(select_expected_f(dd, "_ps"), truth, s1_univ)
    return res
