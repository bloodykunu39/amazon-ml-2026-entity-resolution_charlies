"""Label-free check of the 'noise word' symptom: near-certain pairs (country + name key + all address numbers +
city equal, key unique among S1) — acceptance rate by the record's trailing generic word, holdout vs test.
v1/v2 India: '... Limited Center' accepted 98.9% (holdout) vs 20.4% (test).
  python src/check_symptom.py [train_pred.parquet] [test_pred.parquet]"""
import json, pickle, sys
import numpy as np, polars as pl
from decision import partition
from io_utils import ROOT, load_source
from normalize import load_norm
from splits import load_splits
from submit import decide
PAT = r"(?i)\b(limited|ltd|llp|pvt|private|inc|llc|group|co|corp|sarl|sas|sa|eurl)\b[\s\.\(\)\[\]]*(center|centre|services?|partners|enterprises|group|groupe|cie|holdings|trading)[\)\]\s\.]*$"
KEYC = ["country", "n_key", "a_nums", "a_city"]; KEY = ["s1", "src", "rid"]
dec = json.loads((ROOT / "models/decision.json").read_text()); iso = pickle.load(open(ROOT / "models/iso.pkl", "rb"))
tr_p = sys.argv[1] if len(sys.argv) > 1 else "work/pred_l2_train.parquet"
te_p = sys.argv[2] if len(sys.argv) > 2 else "work/pred_l2_test.parquet"

def check(split, pred, s1f=None):
    s1 = load_norm(split, 1, ["idx"] + KEYC).rename({"idx": "s1"}).filter(
        (pl.col("n_key") != "") & (pl.col("a_nums") != "") & (pl.col("a_city") != ""))
    s1 = s1.filter(pl.len().over(KEYC) == 1)
    if s1f is not None:
        s1 = s1.join(s1f, on="s1", how="semi")
    r = pl.concat([load_norm(split, s, ["idx"] + KEYC).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"})
    names = pl.concat([load_source(split, s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"), "name") for s in (2, 3)])
    d = partition(pred, "p2")
    d = d.with_columns(pl.Series("_p", iso.predict(d["p2"].to_numpy()).astype(np.float32)))
    sel = decide(d, "_p", dec["method"], dec["param"]).with_columns(pl.lit(1).alias("sel"))
    x = s1.join(r, on=KEYC).select(KEY + ["country"]).join(pred.select(KEY), on=KEY, how="semi").join(sel, on=KEY, how="left") \
        .join(names, on=["src", "rid"]).with_columns(pl.col("sel").fill_null(0),
                                                    pl.col("name").str.extract(PAT, 2).str.to_lowercase().fill_null("-none-").alias("word"))
    return x.group_by(["country", "word"]).agg(pl.len().alias("n"), pl.col("sel").mean().round(4).alias("accepted"))

ho = load_splits().filter(~pl.col("dropped") & (pl.col("fold") < 0)).select("s1")
a = check("train", pl.read_parquet(tr_p, columns=KEY + ["p2"]).join(ho, on="s1", how="semi"), ho).rename({"n": "n_holdout", "accepted": "acc_holdout"})
b = check("test", pl.read_parquet(te_p, columns=KEY + ["p2"])).rename({"n": "n_test", "accepted": "acc_test"})
pl.Config.set_tbl_rows(40)
print(b.join(a, on=["country", "word"], how="left").filter(pl.col("n_test") >= 200).sort(["country", "n_test"], descending=[False, True]))
