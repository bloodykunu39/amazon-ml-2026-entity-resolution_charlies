"""Sensitivity of the (uncapped) L1 model to the universe-size shift of rarity features.
Holdout pairs, US: rare-token rarity values (>= 13) shifted by log(n_test/n_train) = -0.478 (US test universe is smaller)."""
import json
import lightgbm as lgb
import numpy as np
import polars as pl
from decision import evaluate, fit_isotonic
from features import feats_path
from splits import load_splits, universe_gt
from model import MODELS
cols = json.loads((MODELS / "l1_meta.json").read_text())["cols"]
models = [lgb.Booster(model_file=str(MODELS / f"l1_fold{k}.txt")) for k in range(5)]
RARE = ["nt_unsh_l", "nt_unsh_r", "nt_idf_worst", "at_unsh_l", "at_unsh_r"]
c, shift = "US", float(np.log(4480137 / 7232483))
d = pl.read_parquet(feats_path("train", False, c))
oof = d.filter(pl.col("fold") >= 0).sample(3_000_000, seed=1)
ho = d.filter(pl.col("fold") < 0)
del d
def pred(x, shifted):
    if shifted:
        x = x.with_columns([pl.when(pl.col(f) >= 13).then(pl.col(f) + shift).otherwise(pl.col(f)).alias(f) for f in RARE])
    X = x.select([pl.col(f).cast(pl.Float32) for f in cols]).to_numpy()
    return np.mean([m.predict(X, num_threads=16) for m in models], axis=0)
po = pred(oof, False)
iso = fit_isotonic(po, oof["y"].to_numpy())
sp_ = load_splits().filter(~pl.col("dropped") & (pl.col("fold") < 0) & (pl.col("country") == c)).select("s1")
gt = universe_gt(False).join(sp_, on="s1", how="semi")
for tag, sh in (("original", False), ("rarity shifted like test", True)):
    h = ho.select(["s1", "src", "rid"]).with_columns(pl.Series("p1", pred(ho, sh)))
    r = evaluate(h, gt, sp_, "p1", iso=iso, deltas=(0.0,), thresholds=(0.7,))
    print(f"US holdout L1 [{tag}]: expected-F {r['expected_f'][0.0]:.5f}  thr0.7 {r['threshold'][0.7]:.5f}", flush=True)
