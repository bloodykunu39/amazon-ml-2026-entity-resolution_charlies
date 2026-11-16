"""Recall report for a candidate set against the test-like universe GT."""
from __future__ import annotations

import sys

import polars as pl

from splits import load_splits, universe_gt

KEY = ["s1", "src", "rid"]


def recall_report(cands: pl.DataFrame, sample: bool, label: str = "") -> dict:
    gt = universe_gt(sample)
    sp_ = load_splits().select(["s1", "country"])
    gt = gt.join(sp_, on="s1")
    size = gt.group_by("s1").len().rename({"len": "n_true"})
    gt = gt.join(size, on="s1").with_columns(
        pl.when(pl.col("n_true") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_true").cast(pl.Utf8)).alias("size"))
    hit = gt.join(cands.select(KEY).unique(), on=KEY, how="semi").select(KEY).with_columns(pl.lit(1).alias("hit"))
    g = gt.join(hit, on=KEY, how="left").with_columns(pl.col("hit").fill_null(0))
    out = {"label": label, "pair_recall": g["hit"].mean(), "n_true": g.height, "n_cand": cands.height}
    for col in ("country", "size", "src"):
        out[f"by_{col}"] = {str(k): round(v, 5) for k, v in
                            g.group_by(col).agg(pl.col("hit").mean()).sort(col).iter_rows()}
    n_rec = cands.select(["src", "rid"]).n_unique()
    per_s1 = cands.group_by("s1").len()["len"]
    out["cands_per_rec"] = cands.height / max(n_rec, 1)
    out["cands_per_s1_mean"] = per_s1.mean()
    out["cands_per_s1_max"] = per_s1.max()
    return out


def channel_recall(cands: pl.DataFrame, sample: bool):
    gt = universe_gt(sample)
    c = cands.join(gt.with_columns(pl.lit(1).alias("y")), on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    pos = c.filter(pl.col("y") == 1)
    n_true = gt.height
    rows = []
    for ch in ["r_name", "r_addr", "r_addrc", "r_comb", "r_rev", "r_nwide"]:
        for k in (1, 3, 5, 10, 50):
            rows.append((ch, k, pos.filter(pl.col(ch).is_not_null() & (pl.col(ch) < k)).height / n_true))
    for ch in ["k_name", "k_addr"]:
        rows.append((ch, 0, pos.filter(pl.col(ch) == 1).height / n_true))
    for ch, k, r in rows:
        print(f"  {ch:7s} @{k:<2d} recall={r:.4f}")
    only = {}
    chans = ["r_name", "r_addr", "r_addrc", "r_comb", "r_rev", "r_nwide"]
    for ch in chans:
        others = [o for o in chans if o != ch]
        m = pl.col(ch).is_not_null()
        for o in others:
            m = m & pl.col(o).is_null()
        m = m & (pl.col("k_name") == 0) & (pl.col("k_addr") == 0)
        only[ch] = pos.filter(m).height / n_true
    print("  unique contribution:", {k: round(v, 5) for k, v in only.items()})


if __name__ == "__main__":
    split, sample = "train", "--sample" in sys.argv
    from blocking import cand_path
    cs = [pl.read_parquet(cand_path(split, sample, c)) for c in ("India", "US")]
    cands = pl.concat(cs)
    import json
    print(json.dumps(recall_report(cands, sample, "raw union"), indent=1, default=float))
    channel_recall(cands, sample)
