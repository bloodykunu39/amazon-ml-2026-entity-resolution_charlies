"""Blend versions with different candidate sets: outer join on (s1, src, rid); p2 = mean of the available p2's.
  python src/union_avg.py OUT.parquet IN1.parquet IN2.parquet ..."""
import sys
import polars as pl

KEY = ["s1", "src", "rid"]
out, ins = sys.argv[1], sys.argv[2:]
meta_cols = None
acc = None
for i, p in enumerate(ins):
    cols = pl.read_parquet_schema(p)
    mc = [c for c in ("y", "fold", "country") if c in cols]
    d = pl.read_parquet(p, columns=KEY + mc + ["p2"]).rename({"p2": f"p2_{i}", **{c: f"{c}_{i}" for c in mc}})
    acc = d if acc is None else acc.join(d, on=KEY, how="full", coalesce=True)
    meta_cols = mc
ps = [c for c in acc.columns if c.startswith("p2_")]
acc = acc.with_columns(pl.mean_horizontal(ps).cast(pl.Float32).alias("p2"),
                       *[pl.coalesce([f"{c}_{i}" for i in range(len(ins))]).alias(c) for c in meta_cols])
acc.select(KEY + meta_cols + ["p2"]).write_parquet(out)
print(f"wrote {out}: {acc.height:,} rows; in all versions: {acc.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in ps])).height:,}")
