"""Level-3 combiner: LightGBM on [GBDT p2, transformer logit] + candidate context, replacing the logistic combiner.

The logistic combiner (ce_stack.py) sees only logit(p2) and the transformer logit of one pair. This model also sees
the record (empty address?), how the pair ranks among the record's other candidate S1s, and how many confident
records the S1 already has per source. Only uncertain-band pairs (transformer scored) are re-scored; p3 = p2 outside.
Cross-fitted on folds 3 and 4 (train on one, predict the other), so folds 3-4 stay out-of-fold for calibration;
holdout and test get the mean of both models.

  python src/l3_stack.py IN_TAG OUT_TAG [NORM_WORK_DIR]
reads work/pred_{train,test}_{IN_TAG}.parquet (columns s1, src, rid, [y, fold], country, p2, p3, ce_logit),
writes work/pred_{train,test}_{OUT_TAG}.parquet with p3 replaced (evaluate with fair_eval.py / write with country_shift.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from io_utils import CFG, ROOT, WORK

KEY = ["s1", "src", "rid"]
_CE = (CFG.get("ce", {}) or {})
LO = float(os.environ.get("CE_LO", _CE.get("lo", 0.005)))       # uncertain band scored by the transformer
HI = float(os.environ.get("CE_HI", _CE.get("hi", 0.995)))
FEATS = ["lp2", "ce", "lp2_ce", "a_empty", "rec_rank", "rec_n01", "rec_max_other", "s1_conf_src", "s1_conf_all",
         "s1_rank_src", "s1_sum_other", "ce_rank_s1src", "rec_max_other_ce", "n_s1_src"]
XSRC = os.environ.get("L3_XSRC", str((CFG.get("ce", {}) or {}).get("l3_xsource", 0))) in ("1", "True", "true")
XS_FEATS = ["xs_p_other", "xs_name_other", "xs_addr_other", "xs_p_same", "xs_name_same", "xs_addr_same"]
if XSRC:
    FEATS = FEATS + XS_FEATS
ACC = os.environ.get("L3_ACC", "0") == "1"          # experiment: accent features from the raw names
ACC_FEATS = ["acc_rec", "acc_s1", "acc_only_diff"]
if ACC:
    FEATS = FEATS + ACC_FEATS
PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("L3_LR", 0.05)), num_leaves=int(os.environ.get("L3_LEAVES", 63)), min_data_in_leaf=int(os.environ.get("L3_MINLEAF", 200)), feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=2026)
ROUNDS = int(os.environ.get("L3_ROUNDS", 500))


def context(d: pl.DataFrame, norm_dir: Path, split: str) -> pl.DataFrame:
    rec = pl.concat([pl.read_parquet(norm_dir / f"norm_{split}_s{s}.parquet", columns=["idx", "a_empty"])
                     .with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"})
    d = d.join(rec, on=["src", "rid"], how="left")
    conf = (pl.col("p2") > 0.9).cast(pl.Int32)
    d = d.with_columns(
        pl.col("p2").rank("ordinal", descending=True).over(["src", "rid"]).cast(pl.Float32).alias("rec_rank"),
        (pl.col("p2") > 0.1).sum().over(["src", "rid"]).cast(pl.Float32).alias("rec_n01"),
        pl.col("p2").max().over(["src", "rid"]).alias("_rmax"),
        pl.col("p2").sort(descending=True).slice(1, 1).first().over(["src", "rid"]).alias("_r2nd"),
        (conf.sum().over(["s1", "src"]) - conf).cast(pl.Float32).alias("s1_conf_src"),
        (conf.sum().over("s1") - conf).cast(pl.Float32).alias("s1_conf_all"),
        pl.col("p2").rank("ordinal", descending=True).over(["s1", "src"]).cast(pl.Float32).alias("s1_rank_src"),
        (pl.col("p2").sum().over("s1") - pl.col("p2")).alias("s1_sum_other"),
        pl.len().over(["s1", "src"]).cast(pl.Float32).alias("n_s1_src"),
        pl.col("ce_logit").rank("ordinal", descending=True).over(["s1", "src"]).cast(pl.Float32).alias("ce_rank_s1src"),
        pl.col("ce_logit").max().over(["src", "rid"]).alias("_cmax"),
        pl.col("ce_logit").sort(descending=True, nulls_last=True).slice(1, 1).first().over(["src", "rid"]).alias("_c2nd"),
    )
    lp = np.log(np.clip(d["p2"].to_numpy(), 1e-6, 1 - 1e-6) / (1 - np.clip(d["p2"].to_numpy(), 1e-6, 1 - 1e-6)))
    d = d.with_columns(pl.Series("lp2", lp.astype(np.float32)), pl.col("ce_logit").alias("ce"))
    d = d.with_columns(
        (pl.col("lp2") * pl.col("ce")).alias("lp2_ce"),
        pl.col("a_empty").cast(pl.Float32),
        pl.when(pl.col("rec_rank") == 1).then(pl.col("_r2nd")).otherwise(pl.col("_rmax")).fill_null(0).alias("rec_max_other"),
        pl.when(pl.col("ce_logit") == pl.col("_cmax")).then(pl.col("_c2nd")).otherwise(pl.col("_cmax")).alias("rec_max_other_ce"),
    ).drop(["_rmax", "_r2nd", "_cmax", "_c2nd"])
    return d


def xsource(d: pl.DataFrame, norm_dir: Path, split: str) -> pl.DataFrame:
    """Cross-source agreement for band pairs: similarity of the record to the S1's best candidate in the OTHER source and
    to its best other candidate in the SAME source (normalized name core / address tokens, token-set ratio)."""
    from rapidfuzz import fuzz, process
    band = d.filter(pl.col("p2").is_between(LO, HI) & pl.col("ce_logit").is_not_null()).select(KEY)
    top = (d.select(KEY + ["p2"]).sort("p2", descending=True).group_by(["s1", "src"], maintain_order=True)
           .agg(pl.col("rid").head(2).alias("rids"), pl.col("p2").head(2).alias("ps")))
    b = band.with_columns((5 - pl.col("src")).cast(pl.Int8).alias("osrc"))
    o = top.rename({"src": "osrc", "rids": "o_rids", "ps": "o_ps"})
    b = b.join(o, on=["s1", "osrc"], how="left").join(top, on=["s1", "src"], how="left")
    b = b.with_columns(pl.col("o_rids").list.first().alias("o_rid"), pl.col("o_ps").list.first().alias("xs_p_other"),
                       pl.when(pl.col("rids").list.first() == pl.col("rid")).then(pl.col("rids").list.get(1, null_on_oob=True))
                       .otherwise(pl.col("rids").list.first()).alias("s_rid"),
                       pl.when(pl.col("rids").list.first() == pl.col("rid")).then(pl.col("ps").list.get(1, null_on_oob=True))
                       .otherwise(pl.col("ps").list.first()).alias("xs_p_same"))
    txt = pl.concat([pl.read_parquet(norm_dir / f"norm_{split}_s{s}.parquet", columns=["idx", "n_core", "a_tok"])
                     .with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"})
    b = b.join(txt, on=["src", "rid"], how="left")
    b = b.join(txt.rename({"src": "osrc", "rid": "o_rid", "n_core": "o_n", "a_tok": "o_a"}), on=["osrc", "o_rid"], how="left")
    b = b.join(txt.rename({"rid": "s_rid", "n_core": "s_n", "a_tok": "s_a"}), on=["src", "s_rid"], how="left")
    def sim(x, y):
        xs, ys = b[x].fill_null("").to_list(), b[y].fill_null("").to_list()
        r = process.cpdist(xs, ys, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
        miss = (b[y].is_null() | (b[y] == "")).to_numpy()
        r[miss] = np.nan
        return r
    b = b.with_columns(pl.Series("xs_name_other", sim("n_core", "o_n")), pl.Series("xs_addr_other", sim("a_tok", "o_a")),
                       pl.Series("xs_name_same", sim("n_core", "s_n")), pl.Series("xs_addr_same", sim("a_tok", "s_a")))
    return d.join(b.select(KEY + XS_FEATS), on=KEY, how="left")


def accents(d: pl.DataFrame, norm_dir: Path, split: str) -> pl.DataFrame:
    """Accent features: accented characters in the record / S1 raw name, and names equal except for accents."""
    import unicodedata
    fold = lambda x: unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    nacc = lambda x: sum(1 for ch in (x or "") if ord(ch) > 127 and unicodedata.category(ch).startswith("L")
                         and unicodedata.normalize("NFKD", ch) != ch)
    s1 = pl.read_parquet(norm_dir / f"raw_{split}_s1.parquet", columns=["idx", "name"]).rename({"idx": "s1", "name": "n1"})
    rec = pl.concat([pl.read_parquet(norm_dir / f"raw_{split}_s{s}.parquet", columns=["idx", "name"])
                     .with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid", "name": "n2"})
    band = d.filter(pl.col("p2").is_between(LO, HI) & pl.col("ce_logit").is_not_null()).select(KEY)
    b = band.join(s1, on="s1", how="left").join(rec, on=["src", "rid"], how="left")
    n1, n2 = b["n1"].to_list(), b["n2"].to_list()
    b = b.with_columns(pl.Series("acc_rec", [float(nacc(x)) for x in n2], dtype=pl.Float32),
                       pl.Series("acc_s1", [float(nacc(x)) for x in n1], dtype=pl.Float32),
                       pl.Series("acc_only_diff", [float((x or "").lower() != (y or "").lower() and fold(x) == fold(y)) for x, y in zip(n1, n2)], dtype=pl.Float32))
    return d.join(b.select(KEY + ACC_FEATS), on=KEY, how="left")


def main(in_tag: str, out_tag: str, norm_dir: str | None = None):
    nd = Path(norm_dir) if norm_dir else WORK
    tr = pl.read_parquet(WORK / f"pred_train_{in_tag}.parquet")
    te = pl.read_parquet(WORK / f"pred_test_{in_tag}.parquet")
    tr, te = context(tr, nd, "train"), context(te, nd, "test")
    if XSRC:
        tr, te = xsource(tr, nd, "train"), xsource(te, nd, "test")
    if ACC:
        tr, te = accents(tr, nd, "train"), accents(te, nd, "test")
    band = lambda d: pl.col("p2").is_between(LO, HI) & pl.col("ce_logit").is_not_null()
    models = {}
    for f in (3, 4):
        fit = tr.filter(band(tr) & (pl.col("fold") == f))
        models[f] = lgb.train(PARAMS, lgb.Dataset(fit.select(FEATS).to_numpy(), fit["y"].to_numpy()), ROUNDS)
        print(f"fold {f}: trained on {fit.height:,} band pairs", flush=True)
    imp = dict(zip(FEATS, models[3].feature_importance("gain").round(0)))
    print("gain importance:", dict(sorted(imp.items(), key=lambda kv: -kv[1])))

    def predict(d: pl.DataFrame, which: str) -> np.ndarray:
        p3 = d["p2"].to_numpy().astype(np.float64).copy()
        m = d.select(band(d)).to_series().to_numpy()
        X = d.filter(band(d)).select(FEATS).to_numpy()
        if which == "cross":                 # train rows: fold 3 <- model 4, fold 4 <- model 3, others <- mean
            fold = d.filter(band(d))["fold"].to_numpy()
            pa, pb = models[3].predict(X), models[4].predict(X)
            p = np.where(fold == 3, pb, np.where(fold == 4, pa, (pa + pb) / 2))
        else:
            p = (models[3].predict(X) + models[4].predict(X)) / 2
        p3[m] = p
        return p3.astype(np.float32)

    keep_tr = [c for c in ["s1", "src", "rid", "y", "fold", "country", "p2", "ce_logit"] if c in tr.columns]
    tr.select(keep_tr).with_columns(pl.Series("p3", predict(tr, "cross"))).write_parquet(WORK / f"pred_train_{out_tag}.parquet")
    te.select(["s1", "src", "rid", "country", "p2", "ce_logit"]).with_columns(pl.Series("p3", predict(te, "mean"))) \
        .write_parquet(WORK / f"pred_test_{out_tag}.parquet")
    print("wrote", WORK / f"pred_train_{out_tag}.parquet", WORK / f"pred_test_{out_tag}.parquet")


def fr_sibling_accept(m: pl.DataFrame, min_p: float, s1: pl.DataFrame, rec: pl.DataFrame, words: set | None = None,
                      label: str = "") -> pl.DataFrame:
    """France rule: also accept same-address France pairs (not acronyms, record has an address) whose record name adds a
    word of textnorm.FR_NOISE_ACCEPT to the S1 name, if the France probability (logistic combiner of the first transformer,
    calibrated as in ce_stack, plus decision.country_delta) is >= min_p and the record is not matched yet."""
    from ce_stack import SFX, best_delta
    from decision import fit_isotonic, partition, shift
    from splits import load_splits, universe_gt
    from textnorm import FR_NOISE_ACCEPT
    tag = f"finalce{SFX}"
    tr = pl.read_parquet(WORK / f"pred_train_{tag}.parquet")
    f34 = tr.filter(pl.col("fold").is_in([3, 4]))
    iso = fit_isotonic(f34["p3"].to_numpy(), f34["y"].to_numpy())
    sp_ = load_splits().filter(~pl.col("dropped"))
    u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
    gt = universe_gt(False)
    dl, _ = best_delta(tr.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, "p3", iso)
    fx = float(((CFG.get("decision", {}) or {}).get("country_delta", {}) or {}).get("France", 0.0))
    d = partition(pl.read_parquet(WORK / f"pred_test_{tag}.parquet").filter(pl.col("country") == "France"), "p3")
    d = d.with_columns(pl.Series("p", shift(iso.predict(d["p3"].to_numpy()), dl + fx).astype(np.float32)))
    cols = ["idx", "n_core", "n_acr", "a_house", "a_street", "a_empty"]
    n1 = pl.read_parquet(WORK / "norm_test_s1.parquet", columns=cols).rename(lambda c: "s1" if c == "idx" else c + "_1")
    nr = pl.concat([pl.read_parquet(WORK / f"norm_test_s{s}.parquet", columns=cols).with_columns(pl.lit(s, pl.Int8).alias("src"))
                    for s in (2, 3)]).rename({"idx": "rid"})
    d = d.filter(pl.col("p") >= min_p).join(n1.drop("a_empty_1"), on="s1").join(nr, on=["src", "rid"])
    name_ex = pl.col("n_core") == pl.col("n_core_1")
    acr = ((pl.col("n_core") == pl.col("n_acr_1")) | (pl.col("n_core_1") == pl.col("n_acr"))) & ~name_ex
    same_addr = ((pl.col("a_house") == pl.col("a_house_1")) & (pl.col("a_street") == pl.col("a_street_1"))
                 & (pl.col("a_house") != "") & (pl.col("a_street") != ""))
    any_word = words == {"*"}                    # '*': any name change (additions, drops or swaps)
    words = pl.Series(sorted(words if (words is not None and not any_word) else FR_NOISE_ACCEPT))
    d = d.filter(~acr & ~pl.col("a_empty") & same_addr & ~name_ex)
    if not any_word:
        d = d.filter(pl.col("n_core").str.split(" ").list.set_difference(pl.col("n_core_1").str.split(" "))
                     .list.eval(pl.element().is_in(words)).list.any())
    add = d.select(KEY).join(s1.rename({"idx": "s1", "entity_id": "s1_entity_id"}), on="s1").join(rec, on=["src", "rid"]) \
        .select(["s1_entity_id", "rec_entity_id"]).join(m.select("rec_entity_id").unique(), on="rec_entity_id", how="anti")
    print(f"France rule fr_sibling_accept{label}(min_p={min_p}): +{add.height:,} pairs", flush=True)
    return pl.concat([m, add]).unique()


def finalize(tag: str, base_out: str, out_dir: str) -> bool:
    """Final decision from work/pred_{train,test}_{tag}.parquet: isotonic calibration + logit shift tuned on folds 3-4
    (as ce_stack), per-country shift from config decision.country_delta, per-S1 expected-F0.5 subset -> both TSVs."""
    import json
    import shutil
    from ce_stack import best_delta
    from decision import evaluate, fit_isotonic, partition, select_expected_f, shift
    from io_utils import load_source, write_id_lists_df
    from splits import load_splits, universe_gt
    from validate_outputs import check_submission, run_official
    tr = pl.read_parquet(WORK / f"pred_train_{tag}.parquet")
    te = pl.read_parquet(WORK / f"pred_test_{tag}.parquet")
    sp_ = load_splits().filter(~pl.col("dropped"))
    gt = universe_gt(False)
    f34 = tr.filter(pl.col("fold").is_in([3, 4]))
    iso = fit_isotonic(f34["p3"].to_numpy(), f34["y"].to_numpy())
    u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
    dl, sc34 = best_delta(tr.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, "p3", iso)
    rep = {"delta": dl, "folds34": sc34}
    for c in ("India", "US", None):
        uh = sp_.filter((pl.col("fold") < 0) & ((pl.col("country") == c) if c else pl.lit(True))).select("s1")
        rep[f"holdout_{c or 'all'}"] = evaluate(tr.join(uh, on="s1", how="semi"), gt.join(uh, on="s1", how="semi"), uh,
                                                "p3", iso=iso, deltas=(dl,), thresholds=())["expected_f"][dl]
    print("L3 final:", json.dumps(rep), flush=True)
    (ROOT / "reports" / f"{tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    l3c = (CFG.get("ce", {}) or {}).get("l3_countries")        # e.g. [India, US]: other countries keep the rows
    keep = None                                                  # already written by ce_stack (logistic combiner)
    if l3c:
        te = te.filter(pl.col("country").is_in(l3c))
        prev = pl.read_csv(Path(out_dir) / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
        c1 = load_source("test", 1).select(pl.col("entity_id").alias(prev.columns[0]), "country")
        keep = prev.join(c1, on=prev.columns[0]).filter(~pl.col("country").is_in(l3c))
    d = partition(te, "p3")
    cdel = (CFG.get("decision", {}) or {}).get("country_delta", {}) or {}
    extra = np.zeros(d.height)
    for c, v in cdel.items():
        extra[(d["country"] == c).to_numpy()] = float(v)
    d = d.with_columns(pl.Series("_p", shift(iso.predict(d["p3"].to_numpy()), dl + extra).astype(np.float32)))
    sel = select_expected_f(d, "_p")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s1 = load_source("test", 1).select(["idx", "entity_id"])
    rec = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                   pl.col("entity_id").alias("rec_entity_id")) for s in (2, 3)])
    m = sel.join(s1.rename({"idx": "s1", "entity_id": "s1_entity_id"}), on="s1").join(rec, on=["src", "rid"]) \
        .select(["s1_entity_id", "rec_entity_id"])
    if keep is not None:                                         # rows of non-L3 countries, unchanged
        k = keep.rename({keep.columns[0]: "s1_entity_id", keep.columns[1]: "ids"}).with_columns(
            pl.col("ids").fill_null("").str.split(",")).explode("ids").filter(pl.col("ids") != "") \
            .select(["s1_entity_id", pl.col("ids").alias("rec_entity_id")])
        m = pl.concat([m, k])
        print(f"L3 decisions for {l3c}; kept {keep.height:,} rows of other countries from the logistic combiner")
    rule = (CFG.get("decision", {}) or {}).get("fr_sibling_accept")
    if rule:
        m = fr_sibling_accept(m, float(rule["min_p"]), s1, rec)
        if rule.get("extra_words"):                  # further French noise words, with their own probability floor
            m = fr_sibling_accept(m, float(rule.get("extra_min_p", rule["min_p"])), s1, rec, set(rule["extra_words"]), " extra")
        if rule.get("any_min_p") is not None:        # any same-address name change in France
            m = fr_sibling_accept(m, float(rule["any_min_p"]), s1, rec, {"*"}, " any")
    write_id_lists_df(out / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1, m)
    if Path(base_out).resolve() != out.resolve():
        shutil.copy(Path(base_out) / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    print(f"country shifts {cdel}; accepted pairs by L3: {sel.height:,}")
    errs = check_submission(out / "matching_results.tsv", out / "candidate_pairs.tsv",
                            ROOT / "student_resource/dataset/test/test_source1.tsv")
    ok, txt = run_official(out / "matching_results.tsv", out / "candidate_pairs.tsv", ROOT / "student_resource/dataset/test")
    print("local checks:", "PASS" if not errs else errs)
    print(txt)
    return ok and not errs


if __name__ == "__main__":
    if sys.argv[1] == "finalize":               # l3_stack.py finalize TAG BASE_OUT OUT_DIR
        finalize(*sys.argv[2:5])
    else:
        main(*sys.argv[1:4])
