"""Evaluate predictions on a sibling-dense holdout: generated distractors (records with no owner at all)
are duplicated so their density per S1 matches test (US x1.87, India x1.63)."""
import sys, json
import numpy as np
import polars as pl
from decision import evaluate, fit_isotonic
from io_utils import load_gt
from splits import load_splits, universe_gt

RATE = {"US": 1.87, "India": 1.63}
KEY = ["s1", "src", "rid"]


def densify(d: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Add copies of orphan-distractor records (no GT owner) with prob RATE-1; copies get new rids."""
    owned = load_gt().select(["src", "rid"]).unique()
    recs = d.select(["src", "rid", "country"]).unique().join(owned, on=["src", "rid"], how="anti")
    rng = np.random.default_rng(seed)
    rate = recs["country"].replace_strict(RATE, default=1.0).to_numpy()
    pick = recs.filter(pl.Series(rng.random(recs.height) < rate - 1)).select(["src", "rid"])
    cp = d.join(pick, on=["src", "rid"], how="semi").with_columns((pl.col("rid") + 500_000_000).alias("rid"))
    return pl.concat([d, cp.select(d.columns)])


if __name__ == "__main__":
    path = sys.argv[1]
    pcol = "p2"
    d = pl.read_parquet(path).select(KEY + ["y", "fold", "country", pcol])
    sp_ = load_splits().filter(~pl.col("dropped"))
    gt = universe_gt(False)
    dd = densify(d)
    print(f"rows {d.height:,} -> densified {dd.height:,}")
    for tag, data in (("original", d), ("sibling-dense", dd)):
        oof = data.filter(pl.col("fold") >= 0)
        iso = fit_isotonic(oof[pcol].to_numpy(), oof["y"].to_numpy())
        for c in ("India", "US"):
            univ = sp_.filter((pl.col("fold") < 0) & (pl.col("country") == c)).select("s1")
            r = evaluate(data.join(univ, on="s1", how="semi"), gt.join(univ, on="s1", how="semi"), univ, pcol,
                         iso=iso, deltas=(-1.0, -0.5, -0.25, 0.0, 0.25), thresholds=(0.5, 0.7, 0.8, 0.9))
            print(tag, c, "expected_f", {k: round(v, 5) for k, v in r["expected_f"].items()},
                  "thr", {k: round(v, 5) for k, v in r["threshold"].items()}, flush=True)
