"""Label-free diagnostics of a test prediction file (compare model versions on test without labels).

- uncertain share: records whose best p2 is in (0.1, 0.9), per country
- sibling clusters: >=2 name-similar records from both sources at a house != S1 house -> mean p2 and share > 0.5
- mean sum of p per S1 (partitioned, raw p2) and predicted matches per S1
  python src/test_diag.py work/pred_l2_test_v2.parquet work/pred_l2_test.parquet
"""
import sys
import polars as pl
from normalize import load_norm
from io_utils import load_source

KEY = ["s1", "src", "rid"]
n1 = load_norm("test", 1, ["idx", "n_core", "a_house"]).rename({"idx": "s1", "n_core": "l", "a_house": "h1"})
nr = pl.concat([load_norm("test", s, ["idx", "n_core", "a_house"]).with_columns(pl.lit(s, pl.Int8).alias("src"))
                for s in (2, 3)]).rename({"idx": "rid", "n_core": "r", "a_house": "hr"})
for path in sys.argv[1:]:
    d = pl.read_parquet(path, columns=KEY + ["p2", "country"])
    best = d.sort("p2", descending=True).unique(subset=["src", "rid"], keep="first")
    unc = best.group_by("country").agg(pl.col("p2").is_between(0.1, 0.9).mean().round(4).alias("uncertain"),
                                       (pl.col("p2") > 0.5).mean().round(4).alias("matched"))
    x = d.filter(pl.col("p2") > 0.001).join(n1, on="s1").join(nr, on=["src", "rid"])
    x = x.filter((pl.col("hr") != "") & (pl.col("h1") != "") & (pl.col("hr") != pl.col("h1")))
    # name similarity proxy: record core contains all S1 core tokens
    x = x.filter(pl.col("l").str.split(" ").list.set_difference(pl.col("r").str.split(" ")).list.len() == 0)
    cl = x.group_by(["s1", "hr"]).agg(pl.len().alias("k"), pl.col("src").n_unique().alias("ns"))
    sib = x.join(cl.filter((pl.col("k") >= 2) & (pl.col("ns") == 2)), on=["s1", "hr"], how="semi")
    sibs = sib.group_by("country").agg(pl.len().alias("sib_pairs"), pl.col("p2").mean().round(4).alias("sib_mean_p2"),
                                       (pl.col("p2") > 0.5).mean().round(4).alias("sib_share_matched"))
    print("==", path)
    print(unc.join(sibs, on="country", how="left").sort("country"))
