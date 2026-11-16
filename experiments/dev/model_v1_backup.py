"""Stage `model`: level-1 / level-2 LightGBM with 5-fold GroupKFold by S1 (folds fixed in splits.parquet).

Level 1: all pruned-candidate features -> OOF p1 (folds 0..4) + holdout/test p1 (mean of fold models).
Level 2: level-1 features + p1 + p1 context (record- and S1-level ranks/gaps/sums) -> p2.
Easy negatives (p0 < EASY_P0) are subsampled for training (weight compensates).
Outputs: work/pred_{level}_{split}{_sample}.parquet (s1, src, rid, p), models/l{level}_fold{k}.txt
"""
from __future__ import annotations

import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from features import KEY, feats_path
from io_utils import CFG, ROOT, WORK, Timer

MODELS = ROOT / "models"
EASY_P0, EASY_KEEP = 0.02, 0.25
NT = CFG["resources"]["n_threads"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=255, num_threads=NT, verbose=-1,
              seed=2026)
MAX_ROUNDS = 3000
ES_ROUNDS = 50
USE_COREF = True
DROP = set(KEY) | {"y", "fold", "country"}


def pred_path(level, split, sample):
    return WORK / f"pred_l{level}_{split}{'_sample' if sample else ''}.parquet"


def load_feats(split, sample, countries=None):
    if countries is None:
        countries = sorted({p.name.split("_")[-1].replace(".parquet", "")
                            for p in WORK.glob(f"feats_{split}{'_sample' if sample else ''}_*.parquet")})
    parts = []
    for c in countries:
        p = feats_path(split, sample, c)
        if p.exists():
            parts.append(pl.read_parquet(p).with_columns(pl.lit(c).alias("country")))
    return pl.concat(parts, how="diagonal_relaxed")


def add_l2_context(d: pl.DataFrame, pcol: str = "p1") -> pl.DataFrame:
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
    d = d.join(sec, on=["src", "rid"], how="left").with_columns(
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
    need = pl.concat([x.select(["src", "rid"]), x.select([pl.col("src2").alias("src"), pl.col("rid2").alias("rid")])]).unique()
    nr = nr.join(need, on=["src", "rid"], how="semi")
    x = x.join(nr, on=["src", "rid"], how="left").join(
        nr.rename({"src": "src2", "rid": "rid2", **{c: c + "_2" for c in cols}}), on=["src2", "rid2"], how="left")
    NTH = CFG["resources"]["n_threads"]
    a = lambda c: x[c].fill_null("").to_list()
    sn = process.cpdist(a("n_core"), a("n_core_2"), scorer=fuzz.token_set_ratio, workers=NTH, dtype=np.float32)
    sa = process.cpdist(a("a_tok"), a("a_tok_2"), scorer=fuzz.token_set_ratio, workers=NTH, dtype=np.float32)
    x = x.with_columns(pl.Series("sn", sn), pl.Series("sa", sa),
                       ((pl.col("a_house") != "") & (pl.col("a_house") == pl.col("a_house_2"))).cast(pl.Float32).alias("sh"),
                       (pl.col("src") == pl.col("src2")).cast(pl.Float32).alias("ss"))
    x = x.with_columns(pl.when(pl.col("a_tok") == "").then(None).otherwise(pl.col("sa")).alias("sa"),
                       ((pl.col("sn") + pl.col("sa").fill_null(pl.col("sn"))) / 2).alias("sb"))
    g = x.group_by(KEY).agg(
        pl.col("sn").max().alias("co_name_max"), pl.col("sa").max().alias("co_addr_max"),
        pl.col("sb").max().alias("co_both_max"), pl.col("sh").max().alias("co_house_any"),
        pl.len().cast(pl.Float32).alias("co_n"), pl.col("pc").max().alias("co_pc_max"),
        pl.col("sb").min().alias("co_both_min"),
        pl.col("sb").filter(pl.col("ss") == 1).max().alias("co_both_same_src"),
    )
    return d.join(g, on=KEY, how="left")


def feat_cols(d: pl.DataFrame) -> list[str]:
    return [c for c in d.columns if c not in DROP and not c.startswith("_")]


def train_level(d: pl.DataFrame, level: int, tag: str, params=None, max_rounds=MAX_ROUNDS):
    """d: train universe pairs with y and fold. Returns OOF preds for fold>=0 rows, holdout preds, models."""
    params = dict(PARAMS, **(params or {}))
    cols = feat_cols(d)
    rng = np.random.default_rng(level)
    if "p0" in d.columns:
        easy = (d["p0"].to_numpy() < EASY_P0) & (d["y"].to_numpy() == 0)
    else:
        easy = np.zeros(d.height, dtype=bool)
    keep = ~easy | (rng.random(d.height) < EASY_KEEP)
    w = np.where(easy, 1.0 / EASY_KEEP, 1.0).astype(np.float32)
    X = d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = d["y"].to_numpy().astype(np.float32)
    fold = d["fold"].to_numpy()
    oof = np.full(d.height, np.nan, dtype=np.float32)
    hold = np.zeros(d.height, dtype=np.float32)
    ho = fold < 0
    best_its, scores = [], []
    for k in range(CFG["validation"]["n_folds"]):
        with Timer(f"L{level} fold {k}"):
            tr = (fold >= 0) & (fold != k) & keep
            va = fold == k
            va_es = va & keep
            dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=cols, free_raw_data=True)
            dva = lgb.Dataset(X[va_es], y[va_es], weight=w[va_es], reference=dtr)
            m = lgb.train(params, dtr, max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(0)])
            best_its.append(m.best_iteration)
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration, num_threads=NT)
            hold[ho] += m.predict(X[ho], num_iteration=m.best_iteration, num_threads=NT) / CFG["validation"]["n_folds"]
            m.save_model(str(MODELS / f"{tag}_fold{k}.txt"), num_iteration=m.best_iteration)
            scores.append(m.best_score["valid_0"]["binary_logloss"])
            print(f"   fold {k}: best_it={m.best_iteration} logloss={scores[-1]:.5f}", flush=True)
            if k == 0:
                imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
                (MODELS / f"{tag}_importance.json").write_text(json.dumps([(a, float(b)) for a, b in imp]))
            del dtr, dva, m
            gc.collect()
    p = np.where(ho, hold, oof)
    meta = {"cols": cols, "best_its": best_its, "logloss": scores}
    (MODELS / f"{tag}_meta.json").write_text(json.dumps(meta))
    return p, meta


def predict_level(d: pl.DataFrame, tag: str) -> np.ndarray:
    meta = json.loads((MODELS / f"{tag}_meta.json").read_text())
    cols = meta["cols"]
    for c in cols:
        if c not in d.columns:
            d = d.with_columns(pl.lit(None, pl.Float32).alias(c))
    X = d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    p = np.zeros(d.height, dtype=np.float32)
    n = CFG["validation"]["n_folds"]
    for k in range(n):
        m = lgb.Booster(model_file=str(MODELS / f"{tag}_fold{k}.txt"))
        p += m.predict(X, num_threads=NT) / n
    return p


def run_train(sample: bool, l2_only: bool = False):
    MODELS.mkdir(exist_ok=True)
    sfx = "_sample" if sample else ""
    d = load_feats("train", sample)
    print(f"train pairs {d.height:,} pos {d['y'].sum():,}")
    if l2_only:
        d = d.join(pl.read_parquet(pred_path(1, "train", sample)).select(KEY + ["p1"]), on=KEY, how="left")
    else:
        with Timer("L1"):
            p1, meta1 = train_level(d, 1, f"l1{sfx}")
        d = d.with_columns(pl.Series("p1", p1))
        d.select(KEY + ["y", "fold", "country", "p1"]).write_parquet(pred_path(1, "train", sample))
    d = add_l2_context(d, "p1")
    if USE_COREF:
        with Timer("coref"):
            d = add_coref(d, "train")
    with Timer("L2"):
        p2, meta2 = train_level(d, 2, f"l2{sfx}", params=dict(num_leaves=63, learning_rate=0.08))
    d = d.with_columns(pl.Series("p2", p2))
    d.select(KEY + ["y", "fold", "country", "p1", "p2"]).write_parquet(pred_path(2, "train", sample))
    return d


def run_predict(split: str, sample_models: bool):
    sfx = "_sample" if sample_models else ""
    d = load_feats(split, False)
    d = d.with_columns(pl.Series("p1", predict_level(d, f"l1{sfx}")))
    d = add_l2_context(d, "p1")
    if USE_COREF:
        d = add_coref(d, split)
    d = d.with_columns(pl.Series("p2", predict_level(d, f"l2{sfx}")))
    d.select(KEY + ["country", "p1", "p2"]).write_parquet(pred_path(2, split, False))
    return d


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_train(sample="--sample" in sys.argv, l2_only="--l2-only" in sys.argv)
    else:
        run_predict(sys.argv[1], sample_models="--sample-models" in sys.argv)
