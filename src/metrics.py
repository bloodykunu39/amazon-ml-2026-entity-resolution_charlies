"""Exact macro F0.5 per S1 entity + breakdown reports.

Per entity: F0.5 = 1.25*tp / (0.25*n_true + n_pred), with empty/empty = 1.0
(this closed form equals 1.25PR/(0.25P+R) and gives 0 when exactly one side is empty).
Pairs are identified by (s1, src, rid) int columns.
"""
from __future__ import annotations

import polars as pl

KEY = ["s1", "src", "rid"]


def f05(tp: float, n_true: float, n_pred: float) -> float:
    if n_true == 0 and n_pred == 0:
        return 1.0
    return 1.25 * tp / (0.25 * n_true + n_pred)


def per_entity(pred: pl.DataFrame, truth: pl.DataFrame, s1_universe: pl.DataFrame) -> pl.DataFrame:
    """s1_universe: DataFrame[s1] (all evaluated S1, singletons included).

    Returns DataFrame[s1, n_true, n_pred, tp, f].
    """
    u = s1_universe.select(pl.col("s1").cast(pl.Int32)).unique()
    pred = pred.select(KEY).unique().join(u, on="s1", how="semi")
    truth = truth.select(KEY).unique().join(u, on="s1", how="semi")
    nt = truth.group_by("s1").agg(pl.len().alias("n_true"))
    npd = pred.group_by("s1").agg(pl.len().alias("n_pred"))
    tp = pred.join(truth, on=KEY, how="inner").group_by("s1").agg(pl.len().alias("tp"))
    df = (
        u.join(nt, on="s1", how="left").join(npd, on="s1", how="left").join(tp, on="s1", how="left")
        .with_columns([pl.col(c).fill_null(0).cast(pl.Int32) for c in ["n_true", "n_pred", "tp"]])
    )
    return df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
        .alias("f")
    )


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_universe: pl.DataFrame) -> float:
    return float(per_entity(pred, truth, s1_universe)["f"].mean())


def breakdown(pred: pl.DataFrame, truth: pl.DataFrame, s1_universe: pl.DataFrame,
              s1_country: pl.DataFrame | None = None, cands: pl.DataFrame | None = None) -> dict:
    """Score breakdowns.

    s1_country: DataFrame[s1, country]. cands: DataFrame[s1, src, rid] (scored candidate set).
    Returns dict of small tables (as lists of dicts) + error-type counts.
    """
    pe = per_entity(pred, truth, s1_universe)
    pe = pe.with_columns(
        pl.when(pl.col("n_true") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_true").cast(pl.Utf8)).alias("size")
    )
    out = {"macro_f05": float(pe["f"].mean()), "n_s1": pe.height}
    if s1_country is not None:
        pe = pe.join(s1_country.select(["s1", "country"]), on="s1", how="left")
        out["by_country"] = (pe.group_by("country").agg(pl.len().alias("n"), pl.col("f").mean())
                             .sort("country").to_dicts())
        out["by_country_size"] = (pe.group_by(["country", "size"]).agg(pl.len().alias("n"), pl.col("f").mean())
                                  .sort(["country", "size"]).to_dicts())
    out["by_size"] = pe.group_by("size").agg(pl.len().alias("n"), pl.col("f").mean()).sort("size").to_dicts()

    u = s1_universe.select(pl.col("s1").cast(pl.Int32)).unique()
    pr = pred.select(KEY).unique().join(u, on="s1", how="semi")
    tr_all = truth.select(KEY).unique()
    tr = tr_all.join(u, on="s1", how="semi")
    # pair-level P/R by source
    by_src = []
    for s in (2, 3):
        p_s, t_s = pr.filter(pl.col("src") == s), tr.filter(pl.col("src") == s)
        tp = p_s.join(t_s, on=KEY, how="inner").height
        by_src.append({"src": s, "n_pred": p_s.height, "n_true": t_s.height, "tp": tp,
                       "precision": tp / max(p_s.height, 1), "recall": tp / max(t_s.height, 1)})
    out["pair_by_src"] = by_src

    # error types
    fp = pr.join(tr, on=KEY, how="anti")
    # record's true owner in the whole truth table (any S1 in the universe)
    owner = tr_all.select(["src", "rid", pl.col("s1").alias("true_s1")])
    fp = fp.join(owner, on=["src", "rid"], how="left")
    fn = tr.join(pr, on=KEY, how="anti")
    err = {
        "fp_distractor": int(fp.filter(pl.col("true_s1").is_null()).height),
        "fp_wrong_s1": int(fp.filter(pl.col("true_s1").is_not_null()).height),
        "fn_total": int(fn.height),
    }
    if cands is not None:
        c = cands.select(KEY).unique()
        err["fn_blocking_miss"] = int(fn.join(c, on=KEY, how="anti").height)
        err["fn_model_miss"] = err["fn_total"] - err["fn_blocking_miss"]
        err["cand_pair_recall"] = 1 - err["fn_blocking_miss"] / max(tr.height, 1) if tr.height else None
    out["errors"] = err
    return out


def fmt_breakdown(b: dict) -> str:
    lines = [f"macro F0.5 = {b['macro_f05']:.5f} over {b['n_s1']} S1"]
    for k in ("by_country", "by_size", "by_country_size"):
        if k in b:
            lines.append(f"  {k}:")
            for r in b[k]:
                lab = ",".join(str(r[c]) for c in r if c not in ("n", "f"))
                lines.append(f"    {lab:<14} n={r['n']:>8}  f={r['f']:.5f}")
    lines.append("  pair_by_src:")
    for r in b["pair_by_src"]:
        lines.append(f"    S{r['src']}: P={r['precision']:.5f} R={r['recall']:.5f} tp={r['tp']} pred={r['n_pred']} true={r['n_true']}")
    lines.append(f"  errors: {b['errors']}")
    return "\n".join(lines)
