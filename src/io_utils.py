"""TSV I/O, integer id maps, row-count asserts, output writers.

Raw TSVs are read with quoting disabled and every field as a string, so literal
"null"/"NA" stay strings and stray quotes don't swallow rows. Each source is cached
once as parquet in work/ with an int32 `idx` (= row order in the file).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"


def setup_env():
    tmp = WORK / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for k in ("TMPDIR", "JOBLIB_TEMP_FOLDER", "POLARS_TEMP_DIR"):
        os.environ[k] = str(tmp)


setup_env()


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()
EXPECTED = CFG["expected_rows"]


def raw_path(split: str, name: str) -> Path:
    d = ROOT / CFG["paths"][f"data_{split}"]
    return d / f"{split}_{name}.tsv"


def read_tsv(path: Path) -> pl.DataFrame:
    """Read a challenge TSV: tab-separated, no quoting, all strings, no NA parsing."""
    assert "__MACOSX" not in str(path) and not Path(path).name.startswith("._")
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        null_values=None,
        has_header=True,
        encoding="utf8",
    )


SRC_NAMES = {1: "source1", 2: "source2", 3: "source3"}


def source_parquet(split: str, src: int) -> Path:
    return WORK / f"raw_{split}_s{src}.parquet"


def load_source(split: str, src: int) -> pl.DataFrame:
    """Columns: idx(int32), entity_id, name, addr, country.

    Train S2/S3 include the synthetic sibling records (src/synth.py, idx >= 10M) when config synth.enabled.
    """
    df = raw_source_only(split, src)
    if split == "train" and src in (2, 3) and CFG.get("synth", {}).get("enabled", False):
        p = WORK / f"synth_train_s{src}.parquet"
        if p.exists():
            df = pl.concat([df, pl.read_parquet(p).select(df.columns)])
    return df


def raw_source_only(split: str, src: int) -> pl.DataFrame:
    """The provided records only (cached as parquet; row counts asserted on first read)."""
    p = source_parquet(split, src)
    if p.exists():
        return pl.read_parquet(p)
    df = read_tsv(raw_path(split, SRC_NAMES[src]))
    assert df.columns == ["entity_id", "business_name", "business_address", "country"], df.columns
    n_exp = EXPECTED[split][f"s{src}"]
    assert df.height == n_exp, f"{split} s{src}: {df.height} rows != expected {n_exp}"
    assert df["entity_id"].str.starts_with(f"S{src}-").all()
    assert df["entity_id"].n_unique() == df.height
    df = df.rename({"business_name": "name", "business_address": "addr"}).with_columns(
        pl.int_range(0, pl.len(), dtype=pl.Int32).alias("idx")
    ).select(["idx", "entity_id", "name", "addr", "country"])
    df.write_parquet(p)
    return df


def load_gt() -> pl.DataFrame:
    """Train ground truth as pairs: s1(int32), src(int8: 2/3), rid(int32 row idx in that source).

    Also caches `gt_s1` with the full S1 list (singletons included) as `work/gt_raw.parquet`.
    """
    p = WORK / "gt_pairs.parquet"
    if p.exists():
        return pl.read_parquet(p)
    gt = read_tsv(raw_path("train", "ground_truth"))
    assert gt.columns == ["source1_entity_id", "matched_entity_ids"]
    assert gt.height == EXPECTED["train"]["gt"]
    s1 = load_source("train", 1).select(["idx", "entity_id"])
    s2 = load_source("train", 2).select(["idx", "entity_id"])
    s3 = load_source("train", 3).select(["idx", "entity_id"])
    ids = pl.concat([s2.with_columns(pl.lit(2, pl.Int8).alias("src")),
                     s3.with_columns(pl.lit(3, pl.Int8).alias("src"))])
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .join(s1.rename({"idx": "s1", "entity_id": "source1_entity_id"}), on="source1_entity_id", how="inner")
        .join(ids.rename({"idx": "rid", "entity_id": "matched_entity_ids"}), on="matched_entity_ids", how="inner")
        .select(["s1", "src", "rid"])
    )
    pairs.write_parquet(p)
    return pairs


def rec_key(src, rid):
    """Unified record key for S2/S3: src*2^28 + rid is not needed; we keep (src, rid) pairs."""
    return src, rid


def write_id_lists(path: Path, header: tuple[str, str], s1_ids: list[str], lists: dict) -> None:
    """Write one row per S1 in the given order; lists maps s1 entity_id -> list of ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s in s1_ids:
            ids = lists.get(s)
            f.write(s + "\t" + (",".join(ids) if ids else "") + "\n")
    os.replace(tmp, path)


def write_id_lists_df(path: Path, header: tuple[str, str], s1_order: pl.DataFrame, df: pl.DataFrame) -> None:
    """Vectorized writer.

    s1_order: DataFrame[entity_id] in the original test_source1 order.
    df: DataFrame[s1_entity_id, rec_entity_id] (long format; may be empty for some S1).
    """
    agg = (
        df.unique(subset=["s1_entity_id", "rec_entity_id"])
        .sort(["s1_entity_id", "rec_entity_id"])
        .group_by("s1_entity_id")
        .agg(pl.col("rec_entity_id").str.join(","))
    )
    out = (
        s1_order.select(pl.col("entity_id").alias("s1_entity_id"))
        .with_row_index("_o")
        .join(agg, on="s1_entity_id", how="left")
        .sort("_o")
        .select([pl.col("s1_entity_id").alias(header[0]),
                 pl.col("rec_entity_id").fill_null("").alias(header[1])])
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for a, b in out.iter_rows():
            f.write(a + "\t" + b + "\n")
    os.replace(tmp, path)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        print(f"[{self.name}] {time.time() - self.t:.1f}s rss={rss:.1f}GB", flush=True)
