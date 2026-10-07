"""Compare two matching_results.tsv files per country (no labels needed).
  python src/compare_outputs.py output/v1/matching_results.tsv output/variants/v4/matching_results.tsv"""
import sys
import polars as pl
from io_utils import load_source

def read(p):
    d = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    return d.rename({d.columns[1]: "ids"}).with_columns(pl.col("ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")))

a, b = read(sys.argv[1]), read(sys.argv[2])
s1 = load_source("test", 1).select(pl.col("entity_id").alias("source1_entity_id"), "country")
x = a.join(b, on="source1_entity_id", suffix="_b").join(s1, on="source1_entity_id")
x = x.with_columns(pl.col("ids").list.len().alias("na"), pl.col("ids_b").list.len().alias("nb"),
                   pl.col("ids").list.set_intersection("ids_b").list.len().alias("both"))
print(x.group_by("country").agg(
    pl.len().alias("s1"),
    (pl.col("na") != pl.col("nb")).mean().round(4).alias("rows_diff_count"),
    ((pl.col("both") < pl.col("na")) | (pl.col("both") < pl.col("nb"))).mean().round(4).alias("rows_changed"),
    pl.col("na").mean().round(3).alias("matches_per_s1_A"), pl.col("nb").mean().round(3).alias("matches_per_s1_B"),
    ((pl.col("na") == 0) & (pl.col("nb") > 0)).sum().alias("empty_A_nonempty_B"),
    ((pl.col("na") > 0) & (pl.col("nb") == 0)).sum().alias("nonempty_A_empty_B"),
    (pl.col("na") - pl.col("both")).sum().alias("pairs_only_A"), (pl.col("nb") - pl.col("both")).sum().alias("pairs_only_B"),
).sort("country"))
