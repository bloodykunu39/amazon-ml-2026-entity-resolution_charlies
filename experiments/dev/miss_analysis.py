import sys
import polars as pl
from blocking import cand_path
from io_utils import load_source
from normalize import load_norm
from splits import universe_gt
sample = "--sample" in sys.argv
KEY = ["s1", "src", "rid"]
cands = pl.concat([pl.read_parquet(cand_path("train", sample, c)).select(KEY) for c in ("India", "US")])
gt = universe_gt(sample)
miss = gt.join(cands, on=KEY, how="anti")
print("missed", miss.height)
C = ["n_core", "n_nl", "n_dom", "a_tok", "a_empty", "a_house", "a_city", "country"]
s1n = load_norm("train", 1, ["idx"] + C).rename({"idx": "s1"}).rename({c: "l_" + c for c in C})
rn = pl.concat([load_norm("train", s, ["idx"] + C).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"}).rename({c: "r_" + c for c in C})
m = miss.join(s1n, on="s1").join(rn, on=["src", "rid"])
m = m.with_columns(
    (pl.col("r_a_empty") | pl.col("l_a_empty")).alias("addr_empty"),
    pl.col("r_n_nl").alias("nonlatin"),
    (pl.col("r_n_dom") != "").alias("dom"),
)
print(m.group_by(["l_country", "addr_empty", "nonlatin", "dom"]).len().sort("len", descending=True))
raw1 = load_source("train", 1).select(pl.col("idx").alias("s1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"))
rr = pl.concat([load_source("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"), "name", "addr") for s in (2, 3)])
x = m.join(raw1, on="s1").join(rr, on=["src", "rid"]).sample(40, seed=1)
for r in x.iter_rows(named=True):
    print(f"{r['l_country']:5s} {r['n1']} | {r['a1']}\n      {r['name']} | {r['addr']}")
