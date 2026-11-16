"""Score a train prediction file on the v3 test-like metric: records of dropped S1 removed, generated
distractors densified to test density; isotonic fit on (densified) OOF; expected-F delta sweep."""
import sys, json
import polars as pl
from decision import evaluate, fit_isotonic
from splits import excluded_records, load_splits, universe_gt
from submit import densify

path = sys.argv[1]
sample = "--sample" in sys.argv
d = pl.read_parquet(path).select(["s1", "src", "rid", "y", "fold", "country", "p2"])
d = densify(d.join(excluded_records(), on=["src", "rid"], how="anti"))
sp_ = load_splits().filter(~pl.col("dropped"))
if sample:
    sp_ = sp_.filter(pl.col("in_sample"))
gt = universe_gt(sample)
oof = d.filter(pl.col("fold") >= 0)
iso = fit_isotonic(oof["p2"].to_numpy(), oof["y"].to_numpy())
for name, f in (("oof", pl.col("fold") >= 0), ("holdout", pl.col("fold") < 0)):
    for c in ("India", "US"):
        univ = sp_.filter(f & (pl.col("country") == c)).select("s1")
        r = evaluate(d.join(univ, on="s1", how="semi"), gt.join(univ, on="s1", how="semi"), univ, "p2", iso=iso,
                     deltas=(-0.5, -0.25, 0.0, 0.25), thresholds=(0.7,))
        print(f"{name:7s} {c:5s} expF " + " ".join(f"{k:+.2f}:{v:.5f}" for k, v in r["expected_f"].items())
              + f"  thr0.7:{r['threshold'][0.7]:.5f}", flush=True)
