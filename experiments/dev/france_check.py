"""France sanity checks on test predictions (no labels): per-country prediction statistics.

- predicted matches per S1, share of empty predictions, candidates per S1
- distribution of the best p per S1 and of p for the chosen matches
- anchor agreement: pairs sharing a rare name key + house + city -> share whose street type / city / state agree
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import polars as pl

from decision import partition
from io_utils import ROOT, load_source
from model import pred_path
from normalize import load_norm
from submit import decide

KEY = ["s1", "src", "rid"]


def main():
    dec = json.loads((ROOT / "models/decision.json").read_text())
    iso = pickle.load(open(ROOT / "models/iso.pkl", "rb"))
    d = pl.read_parquet(pred_path(2, "test", False))
    s1c = load_source("test", 1).select(pl.col("idx").alias("s1"), "country")
    dp = partition(d, dec["pcol"])
    dp = dp.with_columns(pl.Series("_p", iso.predict(dp[dec["pcol"]].to_numpy()).astype(np.float32)))
    sel = decide(dp, "_p", dec["method"], dec["param"])
    per = (s1c.join(sel.group_by("s1").len().rename({"len": "n_pred"}), on="s1", how="left")
           .join(d.group_by("s1").len().rename({"len": "n_cand"}), on="s1", how="left")
           .join(dp.group_by("s1").agg(pl.col("_p").max().alias("pmax")), on="s1", how="left")
           .with_columns(pl.col("n_pred").fill_null(0), pl.col("n_cand").fill_null(0)))
    print(per.group_by("country").agg(
        pl.len().alias("n_s1"), pl.col("n_pred").mean().alias("pred_per_s1"),
        (pl.col("n_pred") == 0).mean().alias("empty_share"), pl.col("n_cand").mean().alias("cand_per_s1"),
        pl.col("pmax").median().alias("pmax_median"),
        (pl.col("pmax").is_between(0.2, 0.8)).mean().alias("pmax_uncertain_share"),
    ).sort("country"))
    rec_c = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                     "country") for s in (2, 3)])
    matched_share = (rec_c.join(sel.select(["src", "rid"]).with_columns(pl.lit(1).alias("m")), on=["src", "rid"],
                                how="left").group_by("country").agg(pl.col("m").fill_null(0).mean()
                                                                     .alias("rec_matched_share")))
    print(matched_share.sort("country"))
    # anchor agreement in France: S1-record pairs with identical rare name key + house + city
    for country in ("France", "US", "India"):
        cols = ["idx", "country", "n_key", "a_house", "a_city", "a_stype", "a_state", "a_street"]
        a1 = load_norm("test", 1, cols).filter(pl.col("country") == country)
        ar = pl.concat([load_norm("test", s, cols) for s in (2, 3)]).filter(pl.col("country") == country)
        k1 = a1.filter((pl.col("n_key") != "") & (pl.col("a_house") != ""))
        cnt = k1.group_by(["n_key", "a_house", "a_city"]).len().filter(pl.col("len") == 1)
        k1 = k1.join(cnt.select(["n_key", "a_house", "a_city"]), on=["n_key", "a_house", "a_city"])
        j = k1.join(ar, on=["n_key", "a_house", "a_city"], suffix="_r")
        both = lambda c: j.filter((pl.col(c) != "") & (pl.col(c + "_r") != ""))
        res = {c: round(float((both(c)[c] == both(c)[c + "_r"]).mean()), 4) for c in ("a_stype", "a_state", "a_street")}
        print(country, "anchor pairs", j.height, "agreement", res)


if __name__ == "__main__":
    main()
