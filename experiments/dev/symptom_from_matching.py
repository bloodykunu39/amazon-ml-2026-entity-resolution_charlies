"""Test-side symptom check from submission files: near-certain pairs (country + name key + all address numbers + city
equal, key unique among S1) — share accepted, by the record's trailing generic word. Holdout reference ≈ 99%.
  python src/symptom_from_matching.py NAME=matching_results.tsv ...
"""
import sys

import polars as pl

from io_utils import load_source
from normalize import load_norm

PAT = (r"(?i)\b(limited|ltd|llp|pvt|private|inc|llc|group|co|corp|sarl|sas|sa|eurl)\b[\s\.\(\)\[\]]*"
       r"(center|centre|services?|partners|enterprises|group|groupe|cie|holdings|trading)[\)\]\s\.]*$")
KEYC = ["country", "n_key", "a_nums", "a_city"]

s1 = load_norm("test", 1, ["idx"] + KEYC).rename({"idx": "s1"}).filter(
    (pl.col("n_key") != "") & (pl.col("a_nums") != "") & (pl.col("a_city") != ""))
s1 = s1.filter(pl.len().over(KEYC) == 1)
r = pl.concat([load_norm("test", s, ["idx"] + KEYC).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"})
h = s1.join(r, on=KEYC).select(["s1", "src", "rid", "country"])
ids1 = load_source("test", 1).select(pl.col("idx").alias("s1"), pl.col("entity_id").alias("e1"))
idr = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                pl.col("entity_id").alias("e2"), "name") for s in (2, 3)])
h = h.join(ids1, on="s1").join(idr, on=["src", "rid"]).with_columns(
    pl.col("name").str.extract(PAT, 2).str.to_lowercase().fill_null("-none-").alias("word"))
out = None
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    m = m.rename({m.columns[0]: "e1", m.columns[1]: "ids"}).with_columns(pl.col("ids").str.split(",")).explode("ids") \
        .filter(pl.col("ids") != "").rename({"ids": "e2"}).with_columns(pl.lit(1).alias("acc"))
    x = h.join(m, on=["e1", "e2"], how="left").with_columns(pl.col("acc").fill_null(0))
    g = x.group_by(["country", "word"]).agg(pl.len().alias("n"), pl.col("acc").mean().round(4).alias(name))
    out = g if out is None else out.join(g.drop("n"), on=["country", "word"], how="left")
pl.Config.set_tbl_rows(40)
print(out.filter(pl.col("n") >= 300).sort(["country", "n"], descending=[False, True]))
