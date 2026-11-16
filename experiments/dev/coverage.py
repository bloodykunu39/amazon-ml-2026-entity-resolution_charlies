"""Phase 2 coverage report: agreement of parsed components on true train pairs + France parse coverage."""
import polars as pl
from io_utils import load_gt, load_source
from normalize import load_norm

C = ["a_state", "a_city", "a_stype", "a_house", "a_street", "n_core", "n_key", "n_legal"]
gt = load_gt()
s1 = load_norm("train", 1, ["idx", "country"] + C).rename({c: c + "_1" for c in C}).rename({"idx": "s1"})
r = pl.concat([load_norm("train", s, ["idx"] + C).with_columns(pl.lit(s, pl.Int8).alias("src")).rename({"idx": "rid"}) for s in (2, 3)])
j = gt.join(s1, on="s1").join(r, on=["src", "rid"])
lines = ["# Phase 2 coverage (true train pairs)", "", "| country | src | field | both present | agree when both present |", "|---|---|---|---|---|"]
for (country, src), g in sorted(j.group_by(["country", "src"]), key=lambda x: x[0]):
    for c in C:
        both = g.filter((pl.col(c + "_1") != "") & (pl.col(c) != ""))
        agree = (both[c + "_1"] == both[c]).mean() if both.height else float("nan")
        lines.append(f"| {country} | S{src} | {c} | {both.height / g.height:.3f} | {agree:.4f} |")
print("\n".join(lines))
# top disagreeing state / stype / city pairs
for c in ["a_state", "a_stype", "a_city"]:
    for country in ["US", "India"]:
        d = j.filter((pl.col("country") == country) & (pl.col(c + "_1") != "") & (pl.col(c) != "") & (pl.col(c + "_1") != pl.col(c)))
        print(f"\n## top {c} disagreements {country}")
        print(d.group_by([c + "_1", c]).len().sort("len", descending=True).head(15))
# France parse coverage
for split_src in [(1,), (2,), (3,)]:
    f = load_norm("test", split_src[0]).filter(pl.col("country") == "France")
    ne = f.filter(~pl.col("a_empty"))
    print(f"\nFrance test S{split_src[0]}: n={f.height} empty={1-ne.height/f.height:.3f} house={(ne['a_house']!='').mean():.3f} "
          f"stype={(ne['a_stype']!='').mean():.3f} city={(ne['a_city']!='').mean():.3f} state={(ne['a_state']!='').mean():.3f}")
    print("  top unparsed-stype street heads:", ne.filter(pl.col("a_stype")=="")["a_tok"].str.split(" ").list.slice(0,3).list.join(" ").value_counts().sort("count", descending=True).head(12).to_dicts())
    print("  top cities:", ne["a_city"].value_counts().sort("count", descending=True).head(25).to_dicts())
