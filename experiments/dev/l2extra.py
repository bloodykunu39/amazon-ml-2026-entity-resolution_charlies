"""Extra level-2 context: "orphan sibling" evidence per record name key.

For every S2/S3 record we know its best level-1 probability over its candidates (0 if it has none). For a
candidate pair (s1, r) with record name key K (same country):
  k_n_rec        records with key K
  k_n_orphan     records with key K whose best p1 < 0.2 (nobody claims them -> likely a missing same-name entity)
  k_n_conf       records with key K whose best p1 >= 0.5
  k_conf_this    records with key K whose best S1 is this s1 with p1 >= 0.5 (r excluded)
  k_conf_other   k_n_conf - k_conf_this - [r itself confident]
  k_orphan_addr  orphan records with key K that have a non-empty address
"""
from __future__ import annotations

import polars as pl

from normalize import load_norm

KEY = ["s1", "src", "rid"]
ORPHAN_COLS = ["k_n_rec", "k_n_orphan", "k_n_conf", "k_conf_this", "k_conf_other", "k_orphan_addr"]


def add_orphan(d: pl.DataFrame, split: str, pcol: str = "p1") -> pl.DataFrame:
    recs = pl.concat([load_norm(split, s, ["idx", "country", "n_key", "a_empty"]).with_columns(pl.lit(s, pl.Int8).alias("src"))
                      for s in (2, 3)]).rename({"idx": "rid"})
    best = (d.sort(pcol, descending=True).unique(subset=["src", "rid"], keep="first")
            .select(["src", "rid", pl.col("s1").alias("b_s1"), pl.col(pcol).alias("b_p")]))
    r = recs.join(best, on=["src", "rid"], how="left").with_columns(pl.col("b_p").fill_null(0.0))
    r = r.filter(pl.col("n_key") != "")
    kstats = r.group_by(["country", "n_key"]).agg(
        pl.len().cast(pl.Float32).alias("k_n_rec"),
        (pl.col("b_p") < 0.2).sum().cast(pl.Float32).alias("k_n_orphan"),
        (pl.col("b_p") >= 0.5).sum().cast(pl.Float32).alias("k_n_conf"),
        ((pl.col("b_p") < 0.2) & ~pl.col("a_empty")).sum().cast(pl.Float32).alias("k_orphan_addr"),
    )
    this = (r.filter(pl.col("b_p") >= 0.5).group_by(["country", "n_key", "b_s1"]).len()
            .rename({"b_s1": "s1", "len": "_cthis"}))
    rk = recs.select(["src", "rid", "country", "n_key"])
    x = d.select(KEY + [pcol]).join(rk, on=["src", "rid"], how="left", maintain_order="left")
    x = x.join(kstats, on=["country", "n_key"], how="left", maintain_order="left")
    x = x.join(this, on=["country", "n_key", "s1"], how="left", maintain_order="left")
    selfc = (pl.col(pcol) >= 0.5).cast(pl.Float32)
    x = x.with_columns(
        (pl.col("_cthis").fill_null(0).cast(pl.Float32) - selfc).clip(0, None).alias("k_conf_this"),
    ).with_columns(
        (pl.col("k_n_conf") - pl.col("k_conf_this") - selfc).clip(0, None).alias("k_conf_other"),
    )
    return pl.concat([d, x.select(ORPHAN_COLS)], how="horizontal")
