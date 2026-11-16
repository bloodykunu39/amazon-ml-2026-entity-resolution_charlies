"""A/B: level-2 with orphan-sibling features (sample)."""
import sys
import json
import polars as pl
import model as M
from l2extra import add_orphan, ORPHAN_COLS
from io_utils import Timer
sample = True
meta = M.load_meta("train", sample)
prev = pl.read_parquet(M.pred_path(1, "train", sample))
assert (prev["s1"].to_numpy() == meta["s1"].to_numpy()).all()
meta = meta.with_columns(prev["p1"])
cols = M.l1_cols("train", sample)
ctx, cc = M.l2_ctx(meta, "train")
with Timer("orphan"):
    ctx = add_orphan(ctx, "train")
cc = cc + ORPHAN_COLS
p2 = M.train_level("train", sample, meta, 2, "l2orph_sample", cols, ctx, cc, params=M.L2_PARAMS)
meta = meta.with_columns(pl.Series("p2", p2))
meta.write_parquet(M.pred_path(2, "train", sample))
