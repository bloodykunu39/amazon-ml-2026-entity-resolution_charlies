"""Compare versions on the same footing: holdout S1 of the test-like universe, REAL records only (synthetic siblings,
rid >= 10M, removed), each version calibrated the same way (isotonic + logit shift delta fitted on folds 3-4,
real records only), then partition + expected-F0.5 on the holdout.

  python src/fair_eval.py NAME=PRED_PATH[:PCOL] ...
  e.g. python src/fair_eval.py v4=work/pred_l2_train_v4.parquet v4ce=work/pred_train_v4ce.parquet:p3
"""
import json
import sys

import polars as pl

from decision import evaluate, fit_isotonic
from splits import load_splits, universe_gt

KEY = ["s1", "src", "rid"]
DELTAS = (-0.5, -0.25, 0.0, 0.25, 0.5)

import os

import numpy as np

from splits import generated_records

# DENSE=1: test-like density of generated look-alikes (each generated record's candidate rows duplicated with
# probability rate-1; rates measured test/train, US 1.87, India 1.63); same seed for every version.
DENSE = os.environ.get("DENSE", "0") == "1"
RATES = {"US": 1.87, "India": 1.63}
GEN = generated_records().filter(pl.col("rid") < 10_000_000)


def densify(d):
    recs = d.select(["src", "rid", "country"]).unique().join(GEN, on=["src", "rid"], how="semi").sort(["src", "rid"])
    u = np.random.default_rng(0).random(recs.height)
    rate = recs["country"].replace_strict(RATES, default=1.0, return_dtype=pl.Float64).to_numpy()
    pick = recs.filter(pl.Series(u < rate - 1)).select(["src", "rid"])
    cp = d.join(pick, on=["src", "rid"], how="semi").with_columns((pl.col("rid") + 500_000_000).alias("rid"))
    return pl.concat([d, cp.select(d.columns)])


sp_ = load_splits().filter(~pl.col("dropped"))
gt = universe_gt(False)
rows = []
for arg in sys.argv[1:]:
    name, spec = arg.split("=", 1)
    path, pcol = (spec.split(":") + ["p2"])[:2]
    d = pl.read_parquet(path, columns=KEY + ["y", "fold", "country", pcol]).filter(pl.col("rid") < 10_000_000)
    if DENSE:
        d = densify(d)
    f34 = d.filter(pl.col("fold").is_in([3, 4]))
    iso = fit_isotonic(f34[pcol].to_numpy(), f34["y"].to_numpy())
    u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
    r34 = evaluate(d.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, pcol, iso=iso,
                   deltas=DELTAS, thresholds=())
    dl = max(r34["expected_f"].items(), key=lambda kv: kv[1])[0]
    row = {"version": name, "delta": dl, "folds34": round(r34["expected_f"][dl], 5)}
    for c in ("India", "US", None):
        uh = sp_.filter((pl.col("fold") < 0) & ((pl.col("country") == c) if c else pl.lit(True))).select("s1")
        r = evaluate(d.join(uh, on="s1", how="semi"), gt.join(uh, on="s1", how="semi"), uh, pcol, iso=iso,
                     deltas=(dl,), thresholds=())
        row[f"holdout_{c or 'all'}"] = round(r["expected_f"][dl], 5)
    rows.append(row)
    print(json.dumps(row), flush=True)
print(pl.DataFrame(rows))
