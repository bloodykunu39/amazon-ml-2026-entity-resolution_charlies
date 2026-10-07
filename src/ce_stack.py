"""Combine the cross-encoder (src/crossencoder.py) with a version's level-2 probability.

  python src/ce_stack.py TAG TRAIN_PRED TEST_PRED BASE_OUTPUT_DIR
  e.g. python src/ce_stack.py v4 work/pred_l2_train.parquet work/pred_l2_test.parquet output/variants/v4

1. Uncertain band = pairs with LO <= p2 <= HI. Band pairs of folds 3-4 + holdout (train) and of test are scored by the
   cross-encoder (trained on folds 0-2 only) — scores cached in work/ce_scores_{train,test}.parquet.
2. Logistic combiner on folds 3-4 band pairs: [logit p2, ce_logit, product] -> p3 (p3 = p2 outside the band).
3. Fair comparison on the holdout: p2 and p3 each get isotonic calibration + delta fitted on folds 3-4, then
   partition + expected-F0.5 selection. Report per country.
4. Writes output/variants/{TAG}ce/ (matches from p3; candidate file = the base version's, the candidate set is
   unchanged) + reports/{TAG}ce.json, and runs the validators.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from decision import evaluate, fit_isotonic, partition, select_expected_f, shift
from io_utils import CFG, ROOT, WORK, load_source, write_id_lists_df
from splits import load_splits, universe_gt
from validate_outputs import check_submission, run_official

_BAND = (CFG.get("ce", {}) or {})
LO = float(os.environ.get("CE_LO", _BAND.get("lo", 0.005)))     # uncertain band scored by the transformer
HI = float(os.environ.get("CE_HI", _BAND.get("hi", 0.995)))
KEY = ["s1", "src", "rid"]
CE_PY = os.environ.get("CE_PY", sys.executable)   # python with torch + transformers (the pipeline env)
DELTAS = (-0.5, -0.25, 0.0, 0.25, 0.5)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


CE_NAME = os.environ.get("CE_NAME", (CFG.get("ce", {}) or {}).get("model", "e5small"))
SFX = "" if CE_NAME == "e5small" else f"_{CE_NAME}"


def ce_scores(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    cache = WORK / f"ce_scores_{split}{SFX}.parquet"
    have = pl.read_parquet(cache) if cache.exists() else pl.DataFrame(schema={"s1": pl.Int32, "src": pl.Int8, "rid": pl.Int32, "ce_logit": pl.Float32})
    todo = pairs.select(KEY).unique().join(have.select(KEY), on=KEY, how="anti")
    if todo.height:
        tp, op = WORK / f"ce_todo_{split}{SFX}.parquet", WORK / f"ce_new_{split}{SFX}.parquet"
        todo.write_parquet(tp)
        print(f"scoring {todo.height:,} {split} pairs with the cross-encoder", flush=True)
        subprocess.run([CE_PY, str(ROOT / "src/crossencoder.py"), "score", str(tp), str(op), split], check=True,
                       env={**os.environ, "CE_NAME": CE_NAME})
        have = pl.concat([have, pl.read_parquet(op).with_columns(pl.col("ce_logit").cast(pl.Float32))])
        have.write_parquet(cache)
    return pairs.join(have, on=KEY, how="left")


def combine(d: pl.DataFrame, clf) -> np.ndarray:
    p2 = d["p2"].to_numpy().astype(np.float64)
    ce = d["ce_logit"].to_numpy().astype(np.float64)
    band = (p2 >= LO) & (p2 <= HI) & ~np.isnan(ce)
    p3 = p2.copy()
    if band.any():
        lp = logit(p2[band])
        X = np.c_[lp, ce[band], lp * ce[band]]
        p3[band] = clf.predict_proba(X)[:, 1]
    return p3.astype(np.float32)


def best_delta(d, gt, univ, pcol, iso):
    r = evaluate(d, gt, univ, pcol, iso=iso, deltas=DELTAS, thresholds=())
    return max(r["expected_f"].items(), key=lambda kv: kv[1])


def clear_cache():
    """Scores depend on the trained model: call after (re)training it."""
    for split in ("train", "test"):
        (WORK / f"ce_scores_{split}{SFX}.parquet").unlink(missing_ok=True)


def main(tag, train_pred, test_pred=None, base_out=None, out_dir=None, sample=False):
    """test_pred=None: evaluation only (sample runs). out_dir: where to write the submission (default variants)."""
    tr = pl.read_parquet(train_pred, columns=KEY + ["y", "fold", "country", "p2"])
    band_tr = tr.filter(pl.col("fold").is_in([3, 4, -1]) & pl.col("p2").is_between(LO, HI))
    sc_tr = ce_scores(band_tr, "train")
    tr = tr.join(sc_tr.select(KEY + ["ce_logit"]), on=KEY, how="left", maintain_order="left")
    if test_pred is not None:
        te = pl.read_parquet(test_pred, columns=KEY + ["country", "p2"])
        sc_te = ce_scores(te.filter(pl.col("p2").is_between(LO, HI)), "test")
        te = te.join(sc_te.select(KEY + ["ce_logit"]), on=KEY, how="left", maintain_order="left")
    fit = tr.filter(pl.col("fold").is_in([3, 4]) & pl.col("ce_logit").is_not_null())
    lp = logit(fit["p2"].to_numpy().astype(np.float64))
    ce = fit["ce_logit"].to_numpy().astype(np.float64)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[lp, ce, lp * ce], fit["y"].to_numpy())
    print("combiner coef [logit p2, ce, product]:", clf.coef_.round(3), "intercept", clf.intercept_.round(3))
    tr = tr.with_columns(pl.Series("p3", combine(tr, clf)))
    if test_pred is not None:
        te = te.with_columns(pl.Series("p3", combine(te, clf)))

    sp_ = load_splits().filter(~pl.col("dropped"))
    if sample:
        sp_ = sp_.filter(pl.col("in_sample"))
    gt = universe_gt(sample)
    f34 = tr.filter(pl.col("fold").is_in([3, 4]))
    res = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "band": [LO, HI]}
    decision = {}
    for pc in ("p2", "p3"):
        iso = fit_isotonic(f34[pc].to_numpy(), f34["y"].to_numpy())
        u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
        dl, sc34 = best_delta(tr.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, pc, iso)
        decision[pc] = (iso, dl)
        res[pc] = {"delta": dl, "folds34": sc34}
        for c in ("India", "US", None):
            uh = sp_.filter((pl.col("fold") < 0) & ((pl.col("country") == c) if c else pl.lit(True))).select("s1")
            r = evaluate(tr.join(uh, on="s1", how="semi"), gt.join(uh, on="s1", how="semi"), uh, pc, iso=iso,
                         deltas=(dl,), thresholds=())
            res[pc][f"holdout_{c or 'all'}"] = r["expected_f"][dl]
        print(pc, json.dumps(res[pc]), flush=True)
    print(f"holdout gain p3 - p2: {res['p3']['holdout_all'] - res['p2']['holdout_all']:+.5f}")
    (ROOT / "reports" / f"{tag}ce{SFX}.json").write_text(json.dumps(res, indent=1, default=float))
    if test_pred is None:
        return res

    # write the variant from p3
    iso, dl = decision["p3"]
    d = partition(te, "p3")
    # optional per-country logit shift from config decision.country_delta (e.g. {France: 1.5}); default none
    cdel = (CFG.get("decision", {}) or {}).get("country_delta", {}) or {}
    extra = np.zeros(d.height)
    for c, v in cdel.items():
        extra[(d["country"] == c).to_numpy()] = float(v)
    if cdel:
        print("country logit shifts:", cdel)
    d = d.with_columns(pl.Series("_p", shift(iso.predict(d["p3"].to_numpy()), dl + extra).astype(np.float32)))
    sel = select_expected_f(d, "_p")
    out = Path(out_dir) if out_dir else ROOT / "output" / "variants" / f"{tag}ce{SFX}"
    out.mkdir(parents=True, exist_ok=True)
    s1 = load_source("test", 1).select(["idx", "entity_id"])
    ids = {s: load_source("test", s).select(["idx", "entity_id"]) for s in (2, 3)}
    rec = pl.concat([ids[s].with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid", "entity_id": "rec_entity_id"})
    m = sel.join(s1.rename({"idx": "s1", "entity_id": "s1_entity_id"}), on="s1").join(rec, on=["src", "rid"]).select(["s1_entity_id", "rec_entity_id"])
    write_id_lists_df(out / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1, m)
    if Path(base_out).resolve() != out.resolve():
        shutil.copy(Path(base_out) / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    te.select(KEY + ["country", "p2", "p3", "ce_logit"]).write_parquet(WORK / f"pred_test_{tag}ce{SFX}.parquet")
    tr.select(KEY + ["y", "fold", "country", "p2", "p3", "ce_logit"]).write_parquet(WORK / f"pred_train_{tag}ce{SFX}.parquet")
    test_dir = ROOT / "student_resource/dataset/test"
    errs = check_submission(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir / "test_source1.tsv")
    ok, log = run_official(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir)
    print("local checks:", errs or "PASS"); print(log.strip().splitlines()[-1])
    return res


if __name__ == "__main__":
    main(*sys.argv[1:5])
