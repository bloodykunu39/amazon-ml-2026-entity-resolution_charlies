"""Leave-one-country-out: train L1+L2 on country A (its 5 folds), evaluate on all kept S1 of country B.

Shows how the decision layer (calibration + delta) transfers to an unseen country (proxy for France).
  python src/loco.py [--sample]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from decision import evaluate, fit_isotonic
from io_utils import ROOT, Timer
from model import KEY, add_coref, add_l2_context, load_feats, predict_level, train_level
from splits import load_splits, universe_gt


def run(sample: bool):
    sp_ = load_splits().filter(~pl.col("dropped"))
    if sample:
        sp_ = sp_.filter(pl.col("in_sample"))
    gt = universe_gt(sample)
    out = {}
    for a, b in (("US", "India"), ("India", "US")):
        with Timer(f"LOCO train {a} -> {b}"):
            da = load_feats("train", sample, (a,))
            db = load_feats("train", sample, (b,))
            tag = f"loco_{a}"
            p1a, _ = train_level(da, 1, f"{tag}_l1")
            da = da.with_columns(pl.Series("p1", p1a))
            db = db.with_columns(pl.Series("p1", predict_level(db, f"{tag}_l1")))
            da = add_coref(add_l2_context(da, "p1"), "train")
            db = add_coref(add_l2_context(db, "p1"), "train")
            p2a, _ = train_level(da, 2, f"{tag}_l2", params=dict(num_leaves=63, learning_rate=0.08))
            p2b = predict_level(db, f"{tag}_l2")
            db = db.with_columns(pl.Series("p2", p2b))
            oof_a = da.with_columns(pl.Series("p2", p2a)).filter(pl.col("fold") >= 0)
            iso = fit_isotonic(oof_a["p2"].to_numpy(), oof_a["y"].to_numpy())
            univ = sp_.filter(pl.col("country") == b).select("s1")
            r = evaluate(db.select(KEY + ["p2"]), gt.join(univ, on="s1", how="semi"), univ, "p2", iso=iso,
                         deltas=(-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0))
            out[f"{a}->{b}"] = r
            print(a, "->", b, json.dumps(r, default=float), flush=True)
            # calibration check on B: mean predicted vs observed rate by decile
            pc = iso.predict(db["p2"].to_numpy())
            y = db["y"].to_numpy()
            qs = np.quantile(pc, np.linspace(0, 1, 11))
            cal = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (pc >= lo) & (pc <= hi)
                if m.sum():
                    cal.append((round(float(pc[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum())))
            out[f"{a}->{b}_calibration"] = cal
            print("calibration (pred, obs, n):", cal, flush=True)
    (ROOT / "reports" / f"loco{'_sample' if sample else ''}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    run("--sample" in sys.argv)
