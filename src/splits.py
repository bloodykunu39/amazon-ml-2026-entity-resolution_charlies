"""Test-like validation universe + S1-grouped splits. Written ONCE to work/splits.parquet.

- dropped: 21% of train S1 removed from the universe (their GT rows vanish; their S2/S3
  records stay and become distractors) -> distractor share ~41-44% like test.
- holdout: 20% of the kept S1 (final holdout); fold 0..4 for the rest (grouped by S1).
- in_sample: 25% of S1 (iteration sample). Records: work/rec_sample.parquet flags matched
  records by their S1's sample flag and 25% of unmatched records at random.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from io_utils import CFG, WORK, load_gt, load_source

SPLITS = WORK / "splits.parquet"
REC_SAMPLE = WORK / "rec_sample.parquet"


def make_splits(force: bool = False):
    if SPLITS.exists() and not force:
        raise RuntimeError(f"{SPLITS} exists; never regenerate it (would invalidate OOF scores)")
    v = CFG["validation"]
    rng = np.random.default_rng(CFG["seed"])
    s1 = load_source("train", 1).select(["idx", "country"]).rename({"idx": "s1"})
    n = s1.height
    u_drop, u_hold, u_samp = rng.random(n), rng.random(n), rng.random(n)
    fold = rng.integers(0, v["n_folds"], n).astype(np.int8)
    dropped = u_drop < v["drop_frac"]
    holdout = (~dropped) & (u_hold < v["holdout_frac"])
    fold = np.where(dropped | holdout, -1, fold).astype(np.int8)
    sp = s1.with_columns(
        pl.Series("dropped", dropped), pl.Series("holdout", holdout),
        pl.Series("fold", fold), pl.Series("in_sample", u_samp < v["sample_frac"]),
    )
    sp.write_parquet(SPLITS)

    gt = load_gt()
    recs = pl.concat([
        load_source("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"))
        for s in (2, 3)
    ])
    rng2 = np.random.default_rng(CFG["seed"] + 1)
    recs = recs.with_columns(pl.Series("u", rng2.random(recs.height)))
    recs = recs.join(gt.join(sp.select(["s1", "in_sample"]), on="s1").select(["src", "rid", "in_sample"]),
                     on=["src", "rid"], how="left")
    recs = recs.with_columns(
        pl.when(pl.col("in_sample").is_null()).then(pl.col("u") < v["sample_frac"])
        .otherwise(pl.col("in_sample")).alias("in_sample")
    ).select(["src", "rid", "in_sample"])
    recs.write_parquet(REC_SAMPLE)
    return sp


def load_splits() -> pl.DataFrame:
    return pl.read_parquet(SPLITS)


def universe_gt(sample: bool = False) -> pl.DataFrame:
    """GT pairs restricted to the test-like universe (non-dropped S1)."""
    sp = load_splits().filter(~pl.col("dropped"))
    if sample:
        sp = sp.filter(pl.col("in_sample"))
    return load_gt().join(sp.select("s1"), on="s1", how="semi")


def distractor_share(sample: bool = False) -> float:
    gt = universe_gt(sample)
    recs = pl.read_parquet(REC_SAMPLE)
    if sample:
        recs = recs.filter(pl.col("in_sample"))
    matched = recs.join(gt.select(["src", "rid"]), on=["src", "rid"], how="semi").height
    return 1 - matched / recs.height


def excluded_records() -> pl.DataFrame:
    """Train records removed from the universe: those owned by dropped S1 (config universe.exclude_dropped_records)."""
    if not CFG.get("universe", {}).get("exclude_dropped_records", False):
        return pl.DataFrame({"src": pl.Series([], dtype=pl.Int8), "rid": pl.Series([], dtype=pl.Int32)})
    dropped = load_splits().filter(pl.col("dropped")).select("s1")
    return load_gt().join(dropped, on="s1", how="semi").select(["src", "rid"])


def generated_records() -> pl.DataFrame:
    """Train records that belong to no S1 at all (the data's own generated distractors)."""
    recs = pl.concat([load_source("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"))
                      for s in (2, 3)])
    return recs.join(load_gt().select(["src", "rid"]), on=["src", "rid"], how="anti")


def gen_weight() -> dict:
    return CFG.get("universe", {}).get("gen_weight", {}) or {}


if __name__ == "__main__":
    sp = make_splits()
    print(sp.group_by(["dropped", "holdout", "fold"]).len().sort(["dropped", "holdout", "fold"]))
    print("distractor share full:", distractor_share(False), "sample:", distractor_share(True))
