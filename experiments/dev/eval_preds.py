"""Evaluate a train prediction file (OOF + holdout macro F0.5, expected-F delta sweep) without writing decisions."""
import sys, json
import polars as pl
from decision import evaluate, fit_isotonic
from model import pred_path
from splits import load_splits, universe_gt
sample = "--sample" in sys.argv
pcol = "p2"
d = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".parquet") else pred_path(2, "train", sample))
sp_ = load_splits().filter(~pl.col("dropped"))
if sample:
    sp_ = sp_.filter(pl.col("in_sample"))
gt = universe_gt(sample)
oof = d.filter(pl.col("fold") >= 0)
iso = fit_isotonic(oof[pcol].to_numpy(), oof["y"].to_numpy())
for name, filt in (("oof", pl.col("fold") >= 0), ("holdout", pl.col("fold") < 0)):
    univ = sp_.filter(filt).select("s1")
    r = evaluate(d.join(univ, on="s1", how="semi"), gt.join(univ, on="s1", how="semi"), univ, pcol, iso=iso,
                 deltas=(-0.25, 0.0, 0.25), thresholds=(0.7,))
    print(name, json.dumps(r["expected_f"]), "thr0.7", round(r["threshold"][0.7], 5))
