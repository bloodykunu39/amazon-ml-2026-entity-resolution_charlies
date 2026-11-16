"""Print FN (model misses) and FP examples with raw text on the holdout."""
import sys
import pickle
import numpy as np
import polars as pl
from decision import partition
from io_utils import load_source, ROOT
from model import pred_path
from splits import load_splits, universe_gt
from submit import decide
import json
sample = "--sample" in sys.argv
KEY = ["s1", "src", "rid"]
d = pl.read_parquet(pred_path(2, "train", sample))
sp_ = load_splits().filter(~pl.col("dropped") & (pl.col("fold") < 0))
if sample:
    sp_ = sp_.filter(pl.col("in_sample"))
d = d.join(sp_.select("s1"), on="s1", how="semi")
dec = json.loads((ROOT / "models/decision.json").read_text())
iso = pickle.load(open(ROOT / "models/iso.pkl", "rb"))
dp = partition(d, "p2")
dp = dp.with_columns(pl.Series("_p", iso.predict(dp["p2"].to_numpy()).astype(np.float32)))
sel = decide(dp, "_p", dec["method"], dec["param"]).with_columns(pl.lit(1).alias("pred"))
gt = universe_gt(sample).join(sp_.select("s1"), on="s1", how="semi").with_columns(pl.lit(1).alias("true"))
d = d.join(sel, on=KEY, how="left").join(gt, on=KEY, how="left").with_columns(pl.col("pred").fill_null(0), pl.col("true").fill_null(0))
raw1 = load_source("train", 1).select(pl.col("idx").alias("s1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"), "country")
rr = pl.concat([load_source("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"), "name", "addr") for s in (2, 3)])
fn = d.filter((pl.col("true") == 1) & (pl.col("pred") == 0))
fp = d.filter((pl.col("true") == 0) & (pl.col("pred") == 1))
print("FN in candidates:", fn.height, " p2 quantiles:", np.quantile(fn["p2"].to_numpy(), [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))
print("FP:", fp.height, " p2 quantiles:", np.quantile(fp["p2"].to_numpy(), [0.1, 0.5, 0.9]).round(3))
for lab, x in (("FN", fn), ("FP", fp)):
    x = x.join(raw1, on="s1").join(rr, on=["src", "rid"]).sample(min(25, x.height), seed=3)
    print(f"===== {lab}")
    for r in x.iter_rows(named=True):
        print(f"{r['country']:5s} p1={r['p1']:.3f} p2={r['p2']:.3f} | {r['n1']} | {r['a1']}\n{'':25s}{r['name']} | {r['addr']}")
