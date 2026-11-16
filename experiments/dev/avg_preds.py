"""Average p2 (and p1) of several prediction files made from the same feature files (same row order).
  python src/avg_preds.py OUT.parquet IN1.parquet IN2.parquet [...]"""
import sys
import polars as pl

KEY = ["s1", "src", "rid"]
out, ins = sys.argv[1], sys.argv[2:]
base = pl.read_parquet(ins[0])
ps = [base["p2"]]
p1s = [base["p1"]] if "p1" in base.columns else []
for p in ins[1:]:
    d = pl.read_parquet(p)
    for k in KEY:
        assert (d[k] == base[k]).all(), f"row order differs: {p}"
    ps.append(d["p2"])
    if p1s:
        p1s.append(d["p1"])
res = base.with_columns((sum(ps) / len(ps)).cast(pl.Float32).alias("p2"))
if p1s:
    res = res.with_columns((sum(p1s) / len(p1s)).cast(pl.Float32).alias("p1"))
res.write_parquet(out)
print(f"wrote {out}: {res.height:,} rows, averaged {len(ins)} files")
