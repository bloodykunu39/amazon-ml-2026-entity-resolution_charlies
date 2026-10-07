"""Stage `decide`+`write`: tune the decision layer on the test-like holdout, apply it to test, write outputs.

  python src/submit.py tune  [--sample]     -> models/decision.json (pcol, calibration, delta, method)
  python src/submit.py write [--variant NAME --delta D]  -> output/matching_results.tsv + candidate_pairs.tsv
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import polars as pl

from decision import evaluate, fit_isotonic, partition, select_expected_f, select_threshold, shift
from io_utils import ROOT, WORK, Timer, load_source, write_id_lists_df
from metrics import breakdown, fmt_breakdown
from model import pred_path
from prune import pruned_path
from splits import gen_weight, generated_records, load_splits, universe_gt
from validate_outputs import check_submission, run_official

MODELS = ROOT / "models"
OUT = ROOT / "output"
KEY = ["s1", "src", "rid"]


def densify(d: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Test-like density of generated look-alike distractors: each generated record's candidate rows are
    copied (new rid) with probability gen_weight[country] - 1 (weights < 2)."""
    gw = gen_weight()
    if not gw:
        return d
    recs = d.select(["src", "rid", "country"]).unique().join(generated_records(), on=["src", "rid"], how="semi")
    rate = recs["country"].replace_strict(gw, default=1.0, return_dtype=pl.Float64).to_numpy()
    pick = recs.filter(pl.Series(np.random.default_rng(seed).random(recs.height) < rate - 1)).select(["src", "rid"])
    cp = d.join(pick, on=["src", "rid"], how="semi").with_columns((pl.col("rid") + 500_000_000).alias("rid"))
    print(f"densify: {pick.height:,} generated records duplicated ({cp.height:,} rows)")
    return pl.concat([d, cp.select(d.columns)])


def tune(sample: bool, pcol: str = "p2"):
    d = densify(pl.read_parquet(pred_path(2, "train", sample)))
    sp_ = load_splits().filter(~pl.col("dropped"))
    if sample:
        sp_ = sp_.filter(pl.col("in_sample"))
    gt = universe_gt(sample)
    oof = d.filter(pl.col("fold") >= 0)
    iso = fit_isotonic(oof[pcol].to_numpy(), oof["y"].to_numpy())
    res = {}
    for split_name, filt in (("oof", pl.col("fold") >= 0), ("holdout", pl.col("fold") < 0)):
        univ = sp_.filter(filt).select("s1")
        dd = d.join(univ, on="s1", how="semi")
        tr = gt.join(univ, on="s1", how="semi")
        for pc in ("p1", "p2"):
            with Timer(f"eval {split_name} {pc}"):
                r = evaluate(dd, tr, univ, pc, iso=fit_isotonic(oof[pc].to_numpy(), oof["y"].to_numpy()))
            res[f"{split_name}_{pc}"] = r
            print(split_name, pc, json.dumps(r, default=float))
    # choose method on OOF, report holdout
    best = max(((m, k, v) for m, kv in res[f"oof_{pcol}"].items() for k, v in kv.items()), key=lambda x: x[2])
    method, param, score = best
    print(f"best on OOF: {method} {param} -> {score:.5f}; holdout: {res[f'holdout_{pcol}'][method][param]:.5f}")
    dec = {"pcol": pcol, "method": method, "param": float(param), "oof": score,
           "holdout": res[f"holdout_{pcol}"][method][param], "all": res}
    (MODELS / "decision.json").write_text(json.dumps(dec, indent=1, default=float))
    with open(MODELS / "iso.pkl", "wb") as f:
        pickle.dump(iso, f)
    # breakdown on holdout with the chosen decision
    univ = sp_.filter(pl.col("fold") < 0).select("s1")
    dd = partition(d.join(univ, on="s1", how="semi"), pcol)
    pc = iso.predict(dd[pcol].to_numpy()).astype(np.float32)
    dd = dd.with_columns(pl.Series("_p", pc))
    sel = decide(dd, "_p", method, float(param))
    s1c = load_splits().select(["s1", "country"])
    cands = pl.concat([pl.read_parquet(pruned_path("train", sample, c), columns=KEY) for c in ("India", "US")])
    b = breakdown(sel, gt.join(univ, on="s1", how="semi"), univ, s1c, cands.join(univ, on="s1", how="semi"))
    txt = fmt_breakdown(b)
    print(txt)
    (ROOT / "reports" / f"holdout_breakdown{'_sample' if sample else ''}.txt").write_text(txt)
    return dec


def decide(d: pl.DataFrame, pcol: str, method: str, param: float, country_delta: dict | None = None,
           s1_country: pl.DataFrame | None = None) -> pl.DataFrame:
    """country_delta: extra logit shift per country (expected_f) or threshold override per country."""
    if country_delta:
        d = d.join(s1_country, on="s1", how="left")
        extra = np.zeros(d.height, dtype=np.float64)
        cc = d["country"].to_numpy()
        for c, v in country_delta.items():
            extra[cc == c] = v
    else:
        extra = 0.0
    if method == "threshold":
        return select_threshold(d, pcol, param) if not country_delta else \
            d.filter(pl.Series(d[pcol].to_numpy() >= param + extra)).select(KEY)
    x = d.with_columns(pl.Series("_ps", shift(d[pcol].to_numpy(), param + extra).astype(np.float32)))
    return select_expected_f(x, "_ps")


def ids_frame(split: str):
    s1 = load_source(split, 1).select(pl.col("idx").alias("s1"), pl.col("entity_id").alias("s1_entity_id"))
    rec = pl.concat([load_source(split, s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                   pl.col("entity_id").alias("rec_entity_id")) for s in (2, 3)])
    return s1, rec


def write(variant: str | None = None, delta: float | None = None, method: str | None = None,
          country_delta: dict | None = None):
    dec = json.loads((MODELS / "decision.json").read_text())
    with open(MODELS / "iso.pkl", "rb") as f:
        iso = pickle.load(f)
    pcol = dec["pcol"]
    method = method or dec["method"]
    param = dec["param"] if delta is None else delta
    d = pl.read_parquet(pred_path(2, "test", False))
    s1ids, recids = ids_frame("test")
    s1_order = load_source("test", 1).select("entity_id")
    out_dir = OUT if variant is None else OUT / "variants" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    with Timer("write candidates"):
        cands = pl.concat([pl.read_parquet(p, columns=KEY) for p in sorted(WORK.glob("cand_test_*.parquet"))])
        assert cands.height == d.height, (cands.height, d.height)
        cl = cands.join(s1ids, on="s1").join(recids, on=["src", "rid"]).select(["s1_entity_id", "rec_entity_id"])
        write_id_lists_df(out_dir / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), s1_order, cl)
    with Timer("decide"):
        dd = partition(d, pcol)
        dd = dd.with_columns(pl.Series("_p", iso.predict(dd[pcol].to_numpy()).astype(np.float32)))
        s1c_ = load_source("test", 1).select(pl.col("idx").alias("s1"), "country")
        sel = decide(dd, "_p", method, float(param), country_delta, s1c_)
        ml = sel.join(s1ids, on="s1").join(recids, on=["src", "rid"]).select(["s1_entity_id", "rec_entity_id"])
        write_id_lists_df(out_dir / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1_order, ml)
    s1c = load_source("test", 1).select(pl.col("idx").alias("s1"), "country")
    stats = (s1c.join(sel.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
             .group_by("country").agg(pl.col("len").mean().alias("matches_per_s1"),
                                      (pl.col("len") == 0).mean().alias("empty_share")).sort("country"))
    print(stats)
    errs = check_submission(out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv",
                            ROOT / "student_resource/dataset/test/test_source1.tsv")
    print("local checks:", "PASS" if not errs else errs)
    ok, txt = run_official(out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv",
                           ROOT / "student_resource/dataset/test")
    print(txt)
    return ok and not errs


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sample="--sample" in sys.argv)
    else:
        v = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else None
        dl = float(sys.argv[sys.argv.index("--delta") + 1]) if "--delta" in sys.argv else None
        cd = None
        if "--country-delta" in sys.argv:   # e.g. France=-0.5,India=0.25
            cd = {k: float(x) for k, x in (t.split("=") for t in sys.argv[sys.argv.index("--country-delta") + 1].split(","))}
        write(v, dl, country_delta=cd)
