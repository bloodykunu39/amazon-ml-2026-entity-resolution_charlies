"""Combiner with both cross-encoders (e5-small + e5-base scores, cached by ce_stack.py runs).
  python src/ce_stack2.py TAG TRAIN_PRED TEST_PRED BASE_OUTPUT_DIR
Same protocol as ce_stack.py (band 0.005-0.995, fit on folds 3-4, isotonic + delta on folds 3-4, holdout report),
writes output/variants/{TAG}ce2/ and work/pred_{train,test}_{TAG}ce2.parquet."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from ce_stack import DELTAS, HI, KEY, LO, logit
from decision import evaluate, fit_isotonic, partition, select_expected_f, shift
from io_utils import ROOT, WORK, load_source, write_id_lists_df
from splits import load_splits, universe_gt
from validate_outputs import check_submission, run_official

tag, trp, tep, base_out = sys.argv[1:5]


def attach(d, split):
    import os
    sa, sb = os.environ.get("CE2_SFX", ",_e5base").split(",")
    for name, sfx in (("ce_s", sa), ("ce_b", sb)):
        c = pl.read_parquet(WORK / f"ce_scores_{split}{sfx}.parquet").rename({"ce_logit": name})
        d = d.join(c, on=KEY, how="left", maintain_order="left")
    return d


def feats(d):
    lp = logit(d["p2"].to_numpy().astype(np.float64))
    a, b = d["ce_s"].to_numpy().astype(np.float64), d["ce_b"].to_numpy().astype(np.float64)
    return np.c_[lp, a, b, lp * a, lp * b]


def combine(d, clf):
    p2 = d["p2"].to_numpy().astype(np.float64)
    band = (p2 >= LO) & (p2 <= HI) & ~np.isnan(d["ce_s"].to_numpy()) & ~np.isnan(d["ce_b"].to_numpy())
    p3 = p2.copy()
    if band.any():
        p3[band] = clf.predict_proba(feats(d.filter(pl.Series(band))))[:, 1]
    return p3.astype(np.float32)


tr = attach(pl.read_parquet(trp, columns=KEY + ["y", "fold", "country", "p2"]), "train")
te = attach(pl.read_parquet(tep, columns=KEY + ["country", "p2"]), "test")
fit = tr.filter(pl.col("fold").is_in([3, 4]) & pl.col("p2").is_between(LO, HI) & pl.col("ce_s").is_not_null() & pl.col("ce_b").is_not_null())
clf = LogisticRegression(C=1.0, max_iter=2000).fit(feats(fit), fit["y"].to_numpy())
print("coef [lp, ce_small, ce_base, lp*ce_small, lp*ce_base]:", clf.coef_.round(3), clf.intercept_.round(3))
tr = tr.with_columns(pl.Series("p3", combine(tr, clf)))
te = te.with_columns(pl.Series("p3", combine(te, clf)))
sp_ = load_splits().filter(~pl.col("dropped"))
gt = universe_gt(False)
f34 = tr.filter(pl.col("fold").is_in([3, 4]))
iso = fit_isotonic(f34["p3"].to_numpy(), f34["y"].to_numpy())
u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
r34 = evaluate(tr.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, "p3", iso=iso, deltas=DELTAS, thresholds=())
dl = max(r34["expected_f"].items(), key=lambda kv: kv[1])[0]
res = {"delta": dl, "folds34": r34["expected_f"][dl]}
for c in ("India", "US", None):
    uh = sp_.filter((pl.col("fold") < 0) & ((pl.col("country") == c) if c else pl.lit(True))).select("s1")
    r = evaluate(tr.join(uh, on="s1", how="semi"), gt.join(uh, on="s1", how="semi"), uh, "p3", iso=iso, deltas=(dl,), thresholds=())
    res[f"holdout_{c or 'all'}"] = r["expected_f"][dl]
print("p3(both)", json.dumps(res))
(ROOT / "reports" / f"{tag}ce2.json").write_text(json.dumps(res, indent=1))
d = partition(te, "p3")
d = d.with_columns(pl.Series("_p", shift(iso.predict(d["p3"].to_numpy()), dl).astype(np.float32)))
sel = select_expected_f(d, "_p")
out = ROOT / "output" / "variants" / f"{tag}ce2"
out.mkdir(parents=True, exist_ok=True)
s1 = load_source("test", 1).select(["idx", "entity_id"])
rec = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                pl.col("entity_id").alias("rec_entity_id")) for s in (2, 3)])
m = sel.join(s1.rename({"idx": "s1", "entity_id": "s1_entity_id"}), on="s1").join(rec, on=["src", "rid"]).select(["s1_entity_id", "rec_entity_id"])
write_id_lists_df(out / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1, m)
shutil.copy(Path(base_out) / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
tr.select(KEY + ["y", "fold", "country", "p2", "p3"]).write_parquet(WORK / f"pred_train_{tag}ce2.parquet")
te.select(KEY + ["country", "p2", "p3"]).write_parquet(WORK / f"pred_test_{tag}ce2.parquet")
test_dir = ROOT / "student_resource/dataset/test"
errs = check_submission(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir / "test_source1.tsv")
ok, log = run_official(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir)
print("local checks:", errs or "PASS"); print(log.strip().splitlines()[-1])
