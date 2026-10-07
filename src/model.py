"""Stage `model`: level-1 / level-2 LightGBM, 5-fold GroupKFold by S1 (folds fixed in splits.parquet).

Memory-efficient: features stay on disk (work/feats_*.parquet); the training matrix is built from
positives + hard negatives (p0 >= EASY_P0) + a sample of easy negatives (weighted); predictions for all
rows are streamed in chunks. Rows are always in "canonical order" = feature files by country (sorted),
file row order — meta/context frames are aligned to it.

Level 1: all pair features -> OOF p1 (train folds) / mean of fold models (holdout, test).
Level 2: level-1 features + p1 context (record/S1 ranks, gaps, sums) + co-reference -> p2.
Outputs: work/pred_l2_{split}{_sample}.parquet (KEY, [y, fold], country, p1, p2); models/l{1,2}*_fold{k}.txt
"""
from __future__ import annotations

import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from features import KEY, feats_path
from io_utils import CFG, ROOT, WORK, Timer
from splits import gen_weight, generated_records

MODELS = ROOT / "models"
EASY_P0, EASY_KEEP = 0.02, float(os.environ.get("LGB_EASY_KEEP", 0.15))   # env overrides for experiments
NT = CFG["resources"]["n_threads"]
NF = CFG["validation"]["n_folds"]
PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("LGB_LR", 0.1)), num_leaves=int(os.environ.get("LGB_LEAVES", 127)), min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=255, num_threads=NT, verbose=-1,
              seed=int(os.environ.get("LGB_SEED", "2026")))
# optional run tag (e.g. a second seed): model files l1{sfx}{TAG}_fold*, predictions pred_l2_{split}{TAG}.parquet
TAG = os.environ.get("MODEL_TAG", "")
L2_PARAMS = dict(num_leaves=63, learning_rate=0.08)
MAX_ROUNDS = 3000
ES_ROUNDS = 50
USE_COREF = True
DROP = set(KEY) | {"y", "fold", "country"}
STEP = 2_000_000


def pred_path(level, split, sample):
    return WORK / f"pred_l{level}_{split}{'_sample' if sample else ''}{TAG}.parquet"


def countries(split, sample):
    return sorted({p.name.split("_")[-1].replace(".parquet", "")
                   for p in WORK.glob(f"feats_{split}{'_sample' if sample else ''}_*.parquet")})


def l1_cols(split, sample) -> list[str]:
    c0 = countries(split, sample)[0]
    return [c for c in pl.read_parquet_schema(feats_path(split, sample, c0)) if c not in DROP]


def load_meta(split, sample) -> pl.DataFrame:
    parts = []
    for c in countries(split, sample):
        p = feats_path(split, sample, c)
        cols = KEY + ["p0"] + (["y", "fold"] if split == "train" else [])
        parts.append(pl.read_parquet(p, columns=cols).with_columns(pl.lit(c).alias("country")))
    return pl.concat(parts, how="diagonal_relaxed")


def add_gen_weight(meta: pl.DataFrame) -> pl.DataFrame:
    """gen_w = per-country weight for records that are generated distractors (no owner), else 1."""
    gw = gen_weight()
    g = generated_records().with_columns(pl.lit(True).alias("_gen"))
    m = meta.join(g, on=["src", "rid"], how="left", maintain_order="left")
    w = pl.col("country").replace_strict(gw, default=1.0, return_dtype=pl.Float32) if gw else pl.lit(1.0)
    return m.with_columns(pl.when(pl.col("_gen").fill_null(False)).then(w).otherwise(1.0)
                          .cast(pl.Float32).alias("gen_w")).drop("_gen")


def iter_chunks(split, sample, cols):
    off = 0
    for c in countries(split, sample):
        p = feats_path(split, sample, c)
        have = set(pl.read_parquet_schema(p))
        lf = pl.scan_parquet(p)
        n = lf.select(pl.len()).collect().item()
        for s in range(0, n, STEP):
            ch = lf.slice(s, STEP).select([c for c in cols if c in have]).collect()
            miss = [c for c in cols if c not in have]
            if miss:
                ch = ch.with_columns([pl.lit(None, pl.Float32).alias(c) for c in miss])
            yield off + s, ch
        off += n


def to_f32(df: pl.DataFrame, cols) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def build_matrix(split, sample, cols, mask, ctx: pl.DataFrame | None = None, ctx_cols=()):
    """Float32 matrix for rows where mask is True (canonical order), feature cols + ctx cols."""
    idx_n = int(mask.sum())
    X = np.empty((idx_n, len(cols) + len(ctx_cols)), dtype=np.float32)
    pos = 0
    for off, ch in iter_chunks(split, sample, cols + ["s1", "rid"]):
        m = mask[off:off + ch.height]
        if not m.any():
            continue
        ms = pl.Series(m)
        a = to_f32(ch.filter(ms), cols)
        if ctx is not None:
            cc = ctx.slice(off, ch.height)
            assert (cc["s1"].to_numpy() == ch["s1"].to_numpy()).all() and \
                (cc["rid"].to_numpy() == ch["rid"].to_numpy()).all()
            a = np.hstack([a, to_f32(cc.filter(ms), ctx_cols)])
        X[pos:pos + len(a)] = a
        pos += len(a)
    assert pos == idx_n
    return X


def predict_stream(split, sample, cols, tag, fold: np.ndarray | None, ctx=None, ctx_cols=()):
    """Predict all rows: train fold-k rows -> fold-k model (OOF); holdout/test rows -> mean of fold models."""
    models = [lgb.Booster(model_file=str(MODELS / f"{tag}_fold{k}.txt")) for k in range(NF)]
    out = []
    for off, ch in iter_chunks(split, sample, cols + ["s1", "rid"]):
        X = to_f32(ch, cols)
        if ctx is not None:
            cc = ctx.slice(off, ch.height)
            assert (cc["s1"].to_numpy() == ch["s1"].to_numpy()).all()
            X = np.hstack([X, to_f32(cc, ctx_cols)])
        p = np.zeros(len(X), dtype=np.float32)
        if fold is not None:
            f = fold[off:off + len(X)]
            for k, m in enumerate(models):
                sel = f == k
                if sel.any():
                    p[sel] = m.predict(X[sel], num_threads=NT)
            ho = f < 0
            if ho.any():
                p[ho] = np.mean([m.predict(X[ho], num_threads=NT) for m in models], axis=0)
        else:
            p = np.mean([m.predict(X, num_threads=NT) for m in models], axis=0).astype(np.float32)
        out.append(p)
        del X
    return np.concatenate(out)


def train_level(split, sample, meta: pl.DataFrame, level: int, tag: str, cols, ctx=None, ctx_cols=(), params=None):
    params = dict(PARAMS, **(params or {}))
    y_all = meta["y"].to_numpy().astype(np.float32)
    fold_all = meta["fold"].to_numpy()
    p0 = meta["p0"].to_numpy()
    rng = np.random.default_rng(level)
    easy = (p0 < EASY_P0) & (y_all == 0)
    keep = (fold_all >= 0) & (~easy | (rng.random(len(p0)) < EASY_KEEP))
    # generated look-alike distractors: up-weight negatives to the test density (config universe.gen_weight)
    gw = meta["gen_w"].to_numpy().astype(np.float32) if "gen_w" in meta.columns else np.ones(len(p0), np.float32)
    with Timer(f"L{level} build matrix"):
        X = build_matrix(split, sample, list(cols), keep, ctx, list(ctx_cols))
    y = y_all[keep]
    w = (np.where(easy[keep], 1.0 / EASY_KEEP, 1.0) * np.where(y_all[keep] == 0, gw[keep], 1.0)).astype(np.float32)
    fold = fold_all[keep]
    names = list(cols) + list(ctx_cols)
    print(f"L{level}: train matrix {X.shape}, pos {int(y.sum()):,}", flush=True)
    best_its, scores = [], []
    for k in range(NF):
        with Timer(f"L{level} fold {k}"):
            tr, va = fold != k, fold == k
            dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=names, free_raw_data=True)
            dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr)
            m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(0)])
            best_its.append(m.best_iteration)
            scores.append(m.best_score["valid_0"]["binary_logloss"])
            m.save_model(str(MODELS / f"{tag}_fold{k}.txt"), num_iteration=m.best_iteration)
            print(f"   fold {k}: best_it={m.best_iteration} logloss={scores[-1]:.5f}", flush=True)
            if k == 0:
                imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])
                (MODELS / f"{tag}_importance.json").write_text(json.dumps([(a, float(b)) for a, b in imp]))
            del dtr, dva, m
            gc.collect()
    del X
    gc.collect()
    meta_j = {"cols": list(cols), "ctx_cols": list(ctx_cols), "best_its": best_its, "logloss": scores}
    (MODELS / f"{tag}_meta.json").write_text(json.dumps(meta_j))
    with Timer(f"L{level} predict all"):
        p = predict_stream(split, sample, list(cols), tag, fold_all, ctx, list(ctx_cols))
    return p


# ----------------------------------------------------------------------------- level-2 context
def add_l2_context(d: pl.DataFrame, pcol: str = "p1") -> pl.DataFrame:
    """d: KEY + pcol (+ anything); canonical order preserved."""
    p = pl.col(pcol)
    d = d.with_columns(
        p.rank("ordinal", descending=True).over(["src", "rid"]).cast(pl.Float32).alias("c_rk_rec"),
        p.max().over(["src", "rid"]).alias("_mx_r"),
        (p > 0.5).sum().over(["src", "rid"]).cast(pl.Float32).alias("c_n05_rec"),
        p.sum().over(["src", "rid"]).cast(pl.Float32).alias("c_sum_rec"),
        p.rank("ordinal", descending=True).over(["s1", "src"]).cast(pl.Float32).alias("c_rk_s1src"),
        p.sum().over("s1").cast(pl.Float32).alias("c_sum_s1"),
        (p > 0.5).sum().over("s1").cast(pl.Float32).alias("c_n05_s1"),
        (p > 0.9).sum().over("s1").cast(pl.Float32).alias("c_n09_s1"),
        p.max().over("s1").alias("_mx_1"),
    )
    sec = d.filter(pl.col("c_rk_rec") == 2).select(["src", "rid", p.alias("_s2_r")])
    d = d.join(sec, on=["src", "rid"], how="left", maintain_order="left").with_columns(
        (p - pl.when(pl.col("c_rk_rec") == 1).then(pl.col("_s2_r").fill_null(0)).otherwise(pl.col("_mx_r")))
        .cast(pl.Float32).alias("c_gap_rec"),
        (p - pl.col("_mx_1")).cast(pl.Float32).alias("c_gap_s1"),
        (pl.col("c_sum_s1") - p).cast(pl.Float32).alias("c_sum_s1_others"),
    ).drop(["_mx_r", "_mx_1", "_s2_r"])
    return d


def add_coref(d: pl.DataFrame, split: str, pcol: str = "p1", lo: float = 0.005, hi: float = 0.995,
              conf: float = 0.5) -> pl.DataFrame:
    """Similarity between an uncertain candidate record and the S1's other confident candidates."""
    from rapidfuzz import fuzz, process
    from normalize import load_norm
    band = d.filter(pl.col(pcol).is_between(lo, hi)).select(KEY)
    cf = d.filter(pl.col(pcol) >= conf).select(["s1", pl.col("src").alias("src2"), pl.col("rid").alias("rid2"),
                                                 pl.col(pcol).alias("pc")])
    x = band.join(cf, on="s1").filter(~((pl.col("src") == pl.col("src2")) & (pl.col("rid") == pl.col("rid2"))))
    cols = ["n_core", "a_tok", "a_house"]
    nr = pl.concat([load_norm(split, s, ["idx"] + cols).with_columns(pl.lit(s, pl.Int8).alias("src"))
                    for s in (2, 3)]).rename({"idx": "rid"})
    need = pl.concat([x.select(["src", "rid"]),
                      x.select([pl.col("src2").alias("src"), pl.col("rid2").alias("rid")])]).unique()
    nr = nr.join(need, on=["src", "rid"], how="semi")
    x = x.join(nr, on=["src", "rid"], how="left").join(
        nr.rename({"src": "src2", "rid": "rid2", **{c: c + "_2" for c in cols}}), on=["src2", "rid2"], how="left")
    del nr
    a = lambda c: x[c].fill_null("").to_list()
    sn = process.cpdist(a("n_core"), a("n_core_2"), scorer=fuzz.token_set_ratio, workers=NT, dtype=np.float32)
    sa = process.cpdist(a("a_tok"), a("a_tok_2"), scorer=fuzz.token_set_ratio, workers=NT, dtype=np.float32)
    x = x.with_columns(pl.Series("sn", sn), pl.Series("sa", sa),
                       ((pl.col("a_house") != "") & (pl.col("a_house") == pl.col("a_house_2")))
                       .cast(pl.Float32).alias("sh"),
                       (pl.col("src") == pl.col("src2")).cast(pl.Float32).alias("ss"))
    x = x.with_columns(pl.when(pl.col("a_tok") == "").then(None).otherwise(pl.col("sa")).alias("sa"))
    x = x.with_columns(((pl.col("sn") + pl.col("sa").fill_null(pl.col("sn"))) / 2).alias("sb"))
    g = x.group_by(KEY).agg(
        pl.col("sn").max().alias("co_name_max"), pl.col("sa").max().alias("co_addr_max"),
        pl.col("sb").max().alias("co_both_max"), pl.col("sh").max().alias("co_house_any"),
        pl.len().cast(pl.Float32).alias("co_n"), pl.col("pc").max().alias("co_pc_max"),
        pl.col("sb").min().alias("co_both_min"),
        pl.col("sb").filter(pl.col("ss") == 1).max().alias("co_both_same_src"),
    )
    del x
    return d.join(g, on=KEY, how="left", maintain_order="left")


CTX_BASE = ["c_rk_rec", "c_n05_rec", "c_sum_rec", "c_rk_s1src", "c_sum_s1", "c_n05_s1", "c_n09_s1", "c_gap_rec",
            "c_gap_s1", "c_sum_s1_others"]
COREF = ["co_name_max", "co_addr_max", "co_both_max", "co_house_any", "co_n", "co_pc_max", "co_both_min",
         "co_both_same_src"]


def l2_ctx(meta: pl.DataFrame, split: str) -> tuple[pl.DataFrame, list[str]]:
    ctx = add_l2_context(meta.select(KEY + ["p1"]), "p1")
    cc = ["p1"] + CTX_BASE
    if USE_COREF:
        with Timer("coref"):
            ctx = add_coref(ctx, split)
        cc += COREF
    return ctx.select(KEY + cc), cc


# ----------------------------------------------------------------------------- drivers
def run_train(sample: bool, l2_only: bool = False):
    MODELS.mkdir(exist_ok=True)
    sfx = "_sample" if sample else ""
    meta = add_gen_weight(load_meta("train", sample))
    cols = l1_cols("train", sample)
    print(f"train pairs {meta.height:,} pos {int(meta['y'].sum()):,}, {len(cols)} L1 features", flush=True)
    if l2_only:
        prev = pl.read_parquet(pred_path(1, "train", sample))
        assert (prev["s1"].to_numpy() == meta["s1"].to_numpy()).all()
        meta = meta.with_columns(prev["p1"])
    else:
        with Timer("L1"):
            p1 = train_level("train", sample, meta, 1, f"l1{sfx}{TAG}", cols)
        meta = meta.with_columns(pl.Series("p1", p1))
        meta.write_parquet(pred_path(1, "train", sample))
    ctx, cc = l2_ctx(meta, "train")
    with Timer("L2"):
        p2 = train_level("train", sample, meta, 2, f"l2{sfx}{TAG}", cols, ctx, cc, params=L2_PARAMS)
    meta = meta.with_columns(pl.Series("p2", p2))
    meta.write_parquet(pred_path(2, "train", sample))
    return meta


def run_predict(split: str, sample_models: bool = False):
    sfx = "_sample" if sample_models else ""
    meta = load_meta(split, False)
    m1 = json.loads((MODELS / f"l1{sfx}{TAG}_meta.json").read_text())
    with Timer("predict L1"):
        p1 = predict_stream(split, False, m1["cols"], f"l1{sfx}{TAG}", None)
    meta = meta.with_columns(pl.Series("p1", p1))
    ctx, cc = l2_ctx(meta, split)
    with Timer("predict L2"):
        p2 = predict_stream(split, False, m1["cols"], f"l2{sfx}{TAG}", None, ctx, cc)
    meta = meta.with_columns(pl.Series("p2", p2))
    meta.write_parquet(pred_path(2, split, False))
    return meta


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_train(sample="--sample" in sys.argv, l2_only="--l2-only" in sys.argv)
    else:
        run_predict(sys.argv[1], sample_models="--sample-models" in sys.argv)
