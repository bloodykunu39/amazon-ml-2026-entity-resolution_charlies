"""Stage `prune`: first-stage ranker over the raw blocking union -> final candidate set.

Cheap features: channel cosines/ranks/key hits + hashed exact-equality flags (house, street, city,
state, name key) + group context of cos_comb (per record and per S1/source).
LightGBM trained on train-universe pairs from folds 0-1 (never holdout); applied to every split.
Keep, per record, the top-K S1 whose score >= P_MIN; cap candidates per S1.
Output: work/cand_{split}{_sample}_{country}.parquet  (s1, src, rid, block features, p0)
"""
from __future__ import annotations

import gc
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import cand_path
from io_utils import CFG, WORK, Timer
from normalize import load_norm
from splits import excluded_records, load_splits, universe_gt

KEY = ["s1", "src", "rid"]
HASH_COLS = ["n_key", "a_house", "a_street", "a_city", "a_state", "a_tok"]
_P = CFG.get("prune", {}) or {}             # optional overrides (config `prune`), defaults = submitted version
TOP_K = int(_P.get("top_k", 50))
P_MIN = float(_P.get("p_min", 5e-5))
S1_CAP = int(_P.get("s1_cap", 100))
MODEL = WORK.parent / "models" / "prune_lgb.txt"


def pruned_path(split, sample, country):
    return WORK / f"cand_{split}{'_sample' if sample else ''}_{country}.parquet"


def hashes(split: str, src: int) -> pl.DataFrame:
    df = load_norm(split, src, ["idx"] + HASH_COLS)
    return df.select(
        [pl.col("idx")]
        + [pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c).hash(seed=7)).alias("h_" + c) for c in HASH_COLS]
    )


def hash_tables(split: str):
    h1 = hashes(split, 1).rename({"idx": "s1"}).rename({"h_" + k: "l_" + k for k in HASH_COLS})
    hr = pl.concat([hashes(split, s).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename(
        {"idx": "rid"}).rename({"h_" + k: "r_" + k for k in HASH_COLS})
    return h1, hr


def s1_aggs(c: pl.DataFrame):
    return (c.group_by(["s1", "src"]).agg(pl.col("cos_comb").max().alias("s1_max")),
            c.group_by("s1").agg(pl.len().cast(pl.Int16).alias("s1_n")))


def cheap_features(c: pl.DataFrame, split: str, h=None, aggs=None) -> pl.DataFrame:
    """c must contain complete record groups (all candidate S1 of each record)."""
    h1, hr = h if h is not None else hash_tables(split)
    s1max, s1n = aggs if aggs is not None else s1_aggs(c)
    c = c.join(h1, on="s1", how="left").join(hr, on=["src", "rid"], how="left")
    exprs = []
    for k in HASH_COLS:
        l, r = pl.col("l_" + k), pl.col("r_" + k)
        exprs.append(pl.when(l.is_null() | r.is_null()).then(-1).when(l == r).then(1).otherwise(0)
                     .cast(pl.Int8).alias("eq_" + k))
    c = c.with_columns(exprs).drop([p + k for k in HASH_COLS for p in ("l_", "r_")])
    c = c.join(s1max, on=["s1", "src"], how="left").join(s1n, on="s1", how="left")
    c = c.with_columns(
        pl.col("cos_comb").max().over(["src", "rid"]).alias("rec_max"),
        pl.col("cos_comb").rank("ordinal", descending=True).over(["src", "rid"]).cast(pl.Int16).alias("rec_rank"),
        pl.len().over(["src", "rid"]).cast(pl.Int16).alias("rec_n"),
        pl.col("cos_name").max().over(["src", "rid"]).alias("rec_max_name"),
        pl.col("cos_addr").max().over(["src", "rid"]).alias("rec_max_addr"),
    )
    second = (c.filter(pl.col("rec_rank") == 2).select(["src", "rid", pl.col("cos_comb").alias("rec_2nd")]))
    c = c.join(second, on=["src", "rid"], how="left").with_columns(
        (pl.col("cos_comb") - pl.when(pl.col("rec_rank") == 1).then(pl.col("rec_2nd").fill_null(0))
         .otherwise(pl.col("rec_max"))).alias("rec_gap"),
        (pl.col("cos_comb") - pl.col("s1_max")).alias("s1_gap"),
        (pl.col("cos_name") - pl.col("rec_max_name")).alias("rec_gap_name"),
        (pl.col("cos_addr") - pl.col("rec_max_addr")).alias("rec_gap_addr"),
    ).drop(["rec_2nd"])
    return c


NCH = 8


def chunk_cache(split, sample, country, k):
    return WORK / "tmp" / f"cheap_{split}{'_sample' if sample else ''}_{country}_{k}.parquet"


def chunks(c: pl.DataFrame, split: str, sample: bool = False, country: str | None = None):
    """Yield cheap-feature frames for record-disjoint chunks (rid % NCH); cached on disk per chunk."""
    paths = [chunk_cache(split, sample, country, k) for k in range(NCH)] if country else None
    if paths and all(p.exists() for p in paths):
        for p in paths:
            yield pl.read_parquet(p)
        return
    h = hash_tables(split)
    aggs = s1_aggs(c)
    for k in range(NCH):
        f = cheap_features(c.filter(pl.col("rid") % NCH == k), split, h, aggs)
        if paths:
            f.write_parquet(paths[k])
        yield f


def clear_chunk_cache():
    for p in (WORK / "tmp").glob("cheap_*.parquet"):
        p.unlink()


FEATS = ["cos_name", "cos_addr", "cos_comb", "r_name", "r_addr", "r_addrc", "r_comb", "r_rev", "r_nwide", "k_name", "k_addr", "src",
         "eq_n_key", "eq_a_house", "eq_a_street", "eq_a_city", "eq_a_state", "eq_a_tok",
         "rec_max", "rec_rank", "rec_n", "s1_max", "s1_n", "rec_gap", "s1_gap", "rec_gap_name",
         "rec_gap_addr"]


def to_np(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(f).cast(pl.Float32) for f in FEATS]).to_numpy()


def label(c: pl.DataFrame, sample: bool) -> pl.DataFrame:
    gt = universe_gt(sample).with_columns(pl.lit(1, pl.Int8).alias("y"))
    return c.join(gt, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))


def train_ranker(sample: bool, folds=(0, 1), path=None):
    path = path or MODEL
    sp_ = load_splits().filter(pl.col("fold").is_in(list(folds)))
    parts = []
    for country in ("India", "US"):
        c = pl.read_parquet(cand_path("train", sample, country)).join(excluded_records(), on=["src", "rid"], how="anti")
        for ch in chunks(c, "train", sample, country):
            ch = ch.join(sp_.select("s1"), on="s1", how="semi")
            if ch.height > 900_000:
                ch = ch.sample(900_000, seed=1)
            parts.append(label(ch, sample))
        del c
        gc.collect()
    d = pl.concat(parts)
    del parts
    X, y = to_np(d), d["y"].to_numpy()
    print(f"ranker train: {len(y):,} pairs, pos rate {y.mean():.4f}")
    n = len(y)
    idx = np.random.default_rng(0).permutation(n)
    va, tr = idx[: n // 10], idx[n // 10:]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=CFG["resources"]["n_threads"], verbose=-1)
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=FEATS, free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    MODEL.parent.mkdir(exist_ok=True)
    m.save_model(str(path))
    imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda x: -x[1])
    print("ranker importance:", [(f, round(g / 1e3)) for f, g in imp[:12]])
    return m


MODEL_B = WORK.parent / "models" / "prune_lgb_b.txt"


def train_rankers(sample: bool):
    """Cross-fitted rankers: A on folds 0-1, B on folds 2-3. p0 = B for folds 0-1, A everywhere else."""
    a = train_ranker(sample, (0, 1), MODEL)
    b = train_ranker(sample, (2, 3), MODEL_B)
    return a, b


def load_rankers():
    return lgb.Booster(model_file=str(MODEL)), lgb.Booster(model_file=str(MODEL_B))


def apply_prune(split: str, sample: bool, m):
    for country in sorted({p.name.split("_")[-1].replace(".parquet", "")
                           for p in WORK.glob(f"cand_raw_{split}{'_sample' if sample else ''}_*.parquet")}):
        with Timer(f"prune {split} {country}"):
            ma, mb = m if isinstance(m, tuple) else (m, None)
            fold = load_splits().select(["s1", "fold"]) if split == "train" else None
            raw = pl.read_parquet(cand_path(split, sample, country))
            if split == "train":
                raw = raw.join(excluded_records(), on=["src", "rid"], how="anti")
            kept = []
            for c in chunks(raw, split, sample, country):
                X = to_np(c)
                p = ma.predict(X, num_threads=CFG["resources"]["n_threads"]).astype(np.float32)
                if split == "train" and mb is not None:
                    f01 = c.join(fold, on="s1", how="left")["fold"].is_in([0, 1]).to_numpy()
                    if f01.any():
                        p[f01] = mb.predict(X[f01], num_threads=CFG["resources"]["n_threads"])
                c = c.with_columns(pl.Series("p0", p))
                c = c.with_columns(pl.col("p0").rank("ordinal", descending=True).over(["src", "rid"]).alias("_rk"))
                kept.append(c.filter((pl.col("_rk") <= TOP_K) & (pl.col("p0") >= P_MIN)).drop("_rk"))
                del X, c
            del raw
            c = pl.concat(kept)
            del kept
            c = c.with_columns(pl.col("p0").rank("ordinal", descending=True).over("s1").alias("_rk1"))
            c = c.filter(pl.col("_rk1") <= S1_CAP).drop(["_rk1"])
            c.write_parquet(pruned_path(split, sample, country))
            print(f"  kept {c.height:,} pairs")
            del c
            gc.collect()


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    sample = "--sample" in sys.argv
    clear_chunk_cache()
    if "--train" in sys.argv or not MODEL_B.exists():
        m = train_rankers(sample)
    else:
        m = load_rankers()
    apply_prune(split, sample, m)
    clear_chunk_cache()
