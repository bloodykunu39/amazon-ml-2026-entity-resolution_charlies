"""Stage `feats`: pairwise features for the pruned candidate set (country-agnostic, no country feature).

Output: work/feats_{split}{_sample}_{country}.parquet with s1, src, rid, [y, fold], features (float32).
"""
from __future__ import annotations

import gc
import math
import sys
from multiprocessing import get_context

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import load_universe
from io_utils import CFG, WORK, Timer
from normalize import load_norm
from prune import pruned_path
from splits import load_splits, universe_gt

KEY = ["s1", "src", "rid"]
TXT = ["n_fold", "n_norm", "n_core", "n_alt", "n_legal", "n_dom", "n_key", "n_phon", "n_acr", "n_nl", "n_alias",
       "a_tok", "a_house", "a_hsuf", "a_hrange", "a_nums", "a_street", "a_stype", "a_city", "a_state", "a_unit",
       "a_postal", "a_empty"]
CHUNK = 2_000_000
NW = CFG["resources"]["n_threads"]
IDF_CAP = float(CFG.get("features", {}).get("idf_cap", 99.0))
SYN_BASE = 10_000_000          # synthetic sibling records (src/synth.py) have idx >= SYN_BASE

IDF_N: dict = {}
IDF_A: dict = {}
DF_N: dict = {}
DF_A: dict = {}
DEF_N = DEF_A = 10.0
DF_CLIP = 100


NOISE_W: set = set()
SIB_W: set = set()


def _load_extra_words():
    global NOISE_W, SIB_W
    import json
    from textnorm import DICT_DIR
    p = DICT_DIR / "name_extra.json"
    if p.exists():
        d = json.loads(p.read_text())
        NOISE_W, SIB_W = set(d["noise"]), set(d["sibling"])
        if (CFG.get("features", {}) or {}).get("fr_words", False):      # French translations of both lists
            from textnorm import FR_NOISE, FR_SIBLING
            NOISE_W |= FR_NOISE
            SIB_W |= FR_SIBLING - NOISE_W


def _dfmin(df: dict, toks) -> float:
    """Doc frequency of the rarest token (clipped): scale-free for rare/typo tokens; -1 if no tokens."""
    return float(min((min(df.get(t, 0), DF_CLIP) for t in toks), default=-1))


def feats_path(split, sample, country):
    return WORK / f"feats_{split}{'_sample' if sample else ''}_{country}.parquet"


# ----------------------------------------------------------------------------- per-pair python features
def _name_tok(l: str, r: str):
    L, R = l.split(), r.split()
    if not L or not R:
        return (math.nan,) * 15
    sL, sR = set(L), set(R)
    sh = sL & sR
    g = IDF_N.get
    wL = sum(g(t, DEF_N) for t in sL)
    wR = sum(g(t, DEF_N) for t in sR)
    wS = sum(g(t, DEF_N) for t in sh)
    uL = max((g(t, DEF_N) for t in sL - sh), default=0.0)
    uR = max((g(t, DEF_N) for t in sR - sh), default=0.0)
    best = []
    for a in L:
        best.append(max(JaroWinkler.similarity(a, b) for b in R))
    bestr = [max(JaroWinkler.similarity(b, a) for a in L) for b in R]
    imin = int(np.argmin(best))
    n_low = sum(1 for b in best if b < 0.87) + sum(1 for b in bestr if b < 0.87)
    return (wS / (wL + wR - wS), wS / wL, wS / wR, uL, uR, sum(best) / len(best), min(best),
            sum(bestr) / len(bestr), float(n_low), g(L[imin], DEF_N), _dfmin(DF_N, sL - sh), _dfmin(DF_N, sR - sh),
            float(len((sR - sh) & NOISE_W)), float(len((sR - sh) & SIB_W)), float(len((sR - sh) - NOISE_W - SIB_W)))


NAME_TOK_COLS = ["nt_wjac", "nt_cont_l", "nt_cont_r", "nt_unsh_l", "nt_unsh_r", "nt_me_l", "nt_me_min",
                 "nt_me_r", "nt_nlow", "nt_idf_worst", "nt_dfmin_unsh_l", "nt_dfmin_unsh_r",
                 "nt_xr_noise", "nt_xr_sib", "nt_xr_other"]


def _addr_tok(l: str, r: str, ln: str, rn: str, lh: str):
    L, R = set(l.split()), set(r.split())
    out = []
    if L and R:
        sh = L & R
        g = IDF_A.get
        wL = sum(g(t, DEF_A) for t in L)
        wR = sum(g(t, DEF_A) for t in R)
        wS = sum(g(t, DEF_A) for t in sh)
        out += [wS / (wL + wR - wS), wS / wL, wS / wR,
                max((g(t, DEF_A) for t in L - sh), default=0.0), max((g(t, DEF_A) for t in R - sh), default=0.0),
                _dfmin(DF_A, R - sh)]
    else:
        out += [math.nan] * 6
    NL, NR = set(ln.split()), set(rn.split())
    if NL and NR:
        s = len(NL & NR)
        out += [s / len(NL | NR), s / len(NL), s / len(NR)]
    else:
        out += [math.nan] * 3
    out += [float(len(NL)), float(len(NR)), (1.0 if lh in NR else 0.0) if lh and NR else math.nan]
    return tuple(out)


ADDR_TOK_COLS = ["at_wjac", "at_cont_l", "at_cont_r", "at_unsh_l", "at_unsh_r", "at_dfmin_unsh_r", "num_jac", "num_cont_l",
                 "num_cont_r", "num_nl", "num_nr", "num_lhouse_in_r"]


def _py_chunk(args):
    lc, rc, ra, la, raa, ln, rn, lh = args
    a = [_name_tok(x, y) for x, y in zip(lc, rc)]
    b = [_name_tok(x, y) if y else (math.nan,) * 15 for x, y in zip(lc, ra)]
    c = [_addr_tok(*t) for t in zip(la, raa, ln, rn, lh)]
    return (np.array(a, dtype=np.float32).reshape(-1, 15), np.array(b, dtype=np.float32).reshape(-1, 15),
            np.array(c, dtype=np.float32).reshape(-1, 12))


def py_features(pool, d: pl.DataFrame) -> dict:
    cols = [d[c].to_list() for c in ("l_n_core", "r_n_core", "r_n_alt", "l_a_tok", "r_a_tok", "l_a_nums", "r_a_nums",
                                     "l_a_house")]
    n = d.height
    step = 50_000
    jobs = [tuple(c[s:s + step] for c in cols) for s in range(0, n, step)]
    A, B, C = [], [], []
    for a, b, c in pool.imap(_py_chunk, jobs):
        A.append(a), B.append(b), C.append(c)
    A, B, C = np.vstack(A), np.vstack(B), np.vstack(C)
    out = {k: A[:, i] for i, k in enumerate(NAME_TOK_COLS)}
    out.update({"alt_" + k: B[:, i] for i, k in enumerate(["nt_wjac", "nt_cont_l", "nt_cont_r"])})
    out.update({k: C[:, i] for i, k in enumerate(ADDR_TOK_COLS)})
    return out


# ----------------------------------------------------------------------------- vectorized string features
def cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=NW, dtype=np.float32, **kw)


def fuzzy_features(d: pl.DataFrame) -> dict:
    g = {c: d[c].to_list() for c in d.columns if c.startswith(("l_", "r_")) and d[c].dtype == pl.Utf8}
    f = {}
    lc, rc = g["l_n_core"], g["r_n_core"]
    f["fn_ratio"] = cp(lc, rc, fuzz.ratio)
    f["fn_pratio"] = cp(lc, rc, fuzz.partial_ratio)
    f["fn_tsort"] = cp(lc, rc, fuzz.token_sort_ratio)
    f["fn_tset"] = cp(lc, rc, fuzz.token_set_ratio)
    f["fn_jw"] = cp(lc, rc, JaroWinkler.normalized_similarity)
    f["fn_norm_tset"] = cp(g["l_n_norm"], g["r_n_norm"], fuzz.token_set_ratio)
    f["fn_fold_ratio"] = cp(g["l_n_fold"], g["r_n_fold"], fuzz.ratio)
    f["fn_phon_tset"] = cp(g["l_n_phon"], g["r_n_phon"], fuzz.token_set_ratio)
    f["fn_phon_ratio"] = cp(g["l_n_phon"], g["r_n_phon"], fuzz.ratio)
    lcc = [s.replace(" ", "") for s in lc]
    rcc = [s.replace(" ", "") for s in rc]
    f["fn_compact_ratio"] = cp(lcc, rcc, fuzz.ratio)
    rdom = g["r_n_dom"]
    f["fn_dom_pratio"] = cp(lcc, rdom, fuzz.partial_ratio)
    f["fn_dom_ratio"] = cp(lcc, rdom, fuzz.ratio)
    ralt = g["r_n_alt"]
    f["fn_alt_tset"] = cp(lc, ralt, fuzz.token_set_ratio)
    f["fn_alt_ratio"] = cp(lc, ralt, fuzz.ratio)
    f["fn_key_ratio"] = cp(g["l_n_key"], g["r_n_key"], fuzz.ratio)
    la, ra = g["l_a_tok"], g["r_a_tok"]
    f["fa_ratio"] = cp(la, ra, fuzz.ratio)
    f["fa_tset"] = cp(la, ra, fuzz.token_set_ratio)
    f["fa_tsort"] = cp(la, ra, fuzz.token_sort_ratio)
    f["fa_pratio"] = cp(la, ra, fuzz.partial_ratio)
    f["fs_ratio"] = cp(g["l_a_street"], g["r_a_street"], fuzz.ratio)
    f["fs_jw"] = cp(g["l_a_street"], g["r_a_street"], JaroWinkler.normalized_similarity)
    f["fc_ratio"] = cp(g["l_a_city"], g["r_a_city"], fuzz.ratio)
    f["fc_pratio"] = cp(g["l_a_city"], g["r_a_city"], fuzz.partial_ratio)
    f["fh_lev"] = cp(g["l_a_house"], g["r_a_house"], Levenshtein.distance)
    # masks: NaN when a side is empty
    empt = {c: np.array([len(s) == 0 for s in v]) for c, v in g.items()}
    def mask(keys, cols):
        m = np.zeros(d.height, dtype=bool)
        for c in cols:
            m |= empt[c]
        for k in keys:
            f[k][m] = np.nan
    mask([k for k in f if k.startswith("fn_") and not k.startswith(("fn_dom", "fn_alt", "fn_norm", "fn_fold",
                                                                    "fn_key", "fn_phon"))], ["l_n_core", "r_n_core"])
    mask(["fn_dom_pratio", "fn_dom_ratio"], ["r_n_dom"])
    mask(["fn_alt_tset", "fn_alt_ratio"], ["r_n_alt"])
    mask(["fn_key_ratio"], ["l_n_key", "r_n_key"])
    mask(["fn_phon_tset", "fn_phon_ratio"], ["l_n_phon", "r_n_phon"])
    mask(["fa_ratio", "fa_tset", "fa_tsort", "fa_pratio"], ["l_a_tok", "r_a_tok"])
    mask(["fs_ratio", "fs_jw"], ["l_a_street", "r_a_street"])
    mask(["fc_ratio", "fc_pratio"], ["l_a_city", "r_a_city"])
    mask(["fh_lev"], ["l_a_house", "r_a_house"])
    return f


def eq3(a: str, b: str) -> pl.Expr:
    """1 equal, 0 different, -1 one side missing."""
    return (pl.when((pl.col(a) == "") | (pl.col(b) == "")).then(-1).when(pl.col(a) == pl.col(b)).then(1)
            .otherwise(0).cast(pl.Int8))


_VOCAB = None


def vocab_series() -> pl.Series:
    global _VOCAB
    if _VOCAB is None:
        import json
        from textnorm import DICT_DIR
        _VOCAB = pl.Series("v", json.loads((DICT_DIR / "name_vocab.json").read_text()))
    return _VOCAB


def vocab_frac(col: str) -> pl.Expr:
    toks = pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return (toks.list.eval(pl.element().is_in(vocab_series().implode()).cast(pl.Float32)).list.mean()).cast(pl.Float32)


def struct_features(d: pl.DataFrame) -> pl.DataFrame:
    lh, rh = pl.col("l_a_house"), pl.col("r_a_house")
    lhi = lh.cast(pl.Int64, strict=False)
    rhi = rh.cast(pl.Int64, strict=False)
    lri = pl.col("l_a_hrange").cast(pl.Int64, strict=False)
    rri = pl.col("r_a_hrange").cast(pl.Int64, strict=False)
    both = (lh != "") & (rh != "")
    return d.select(
        eq3("l_n_legal", "r_n_legal").alias("legal_eq"),
        eq3("l_a_state", "r_a_state").alias("state_eq"),
        eq3("l_a_city", "r_a_city").alias("city_eq"),
        eq3("l_a_street", "r_a_street").alias("street_eq"),
        eq3("l_a_stype", "r_a_stype").alias("stype_eq"),
        eq3("l_a_unit", "r_a_unit").alias("unit_eq"),
        eq3("l_a_postal", "r_a_postal").alias("postal_eq"),
        eq3("l_a_hsuf", "r_a_hsuf").alias("hsuf_eq"),
        eq3("l_n_key", "r_n_key").alias("key_eq"),
        eq3("l_n_core", "r_n_core").alias("core_eq"),
        eq3("l_a_tok", "r_a_tok").alias("atok_eq"),
        pl.when(~both).then(-1).when(lh == rh).then(1).otherwise(0).cast(pl.Int8).alias("house_eq"),
        pl.when(~both).then(None).otherwise(
            (lh.str.ends_with(rh) | rh.str.ends_with(lh) | lh.str.starts_with(rh) | rh.str.starts_with(lh))
            .cast(pl.Int8)).alias("house_affix"),
        pl.when(~both).then(None).otherwise(
            (((lri.is_not_null()) & (rhi >= lhi) & (rhi <= lri)) | ((rri.is_not_null()) & (lhi >= rhi) & (lhi <= rri)))
            .cast(pl.Int8)).alias("house_range"),
        (lhi - rhi).abs().cast(pl.Float32).log1p().alias("house_logdiff"),
        pl.col("l_a_empty").cast(pl.Int8).alias("l_addr_empty"),
        pl.col("r_a_empty").cast(pl.Int8).alias("r_addr_empty"),
        pl.col("r_n_nl").cast(pl.Int8).alias("r_nonlatin"),
        pl.col("r_n_alias").cast(pl.Int8).alias("r_alias"),
        (pl.col("r_n_dom") != "").cast(pl.Int8).alias("r_has_dom"),
        ((pl.col("l_n_acr") != "") & ((pl.col("l_n_acr") == pl.col("r_n_core").str.replace_all(" ", ""))
                                      | (pl.col("r_n_acr") == pl.col("l_n_core").str.replace_all(" ", ""))))
        .cast(pl.Int8).alias("acr_match"),
        pl.col("l_n_core").str.count_matches(" ").add(1).cast(pl.Int8).alias("l_ntok"),
        pl.col("r_n_core").str.count_matches(" ").add(1).cast(pl.Int8).alias("r_ntok"),
        (pl.col("r_n_core").str.len_chars() / pl.col("l_n_core").str.len_chars().clip(1, None))
        .cast(pl.Float32).alias("name_len_ratio"),
        (pl.col("r_a_tok").str.len_chars() / pl.col("l_a_tok").str.len_chars().clip(1, None))
        .cast(pl.Float32).alias("addr_len_ratio"),
        (pl.col("l_n_legal") != "").cast(pl.Int8).alias("l_has_legal"),
        (pl.col("r_n_legal") != "").cast(pl.Int8).alias("r_has_legal"),
        vocab_frac("r_n_core").alias("r_vocab_frac"),
        vocab_frac("r_n_alt").alias("r_alt_vocab_frac"),
    )


# ----------------------------------------------------------------------------- frequencies (full universe)
def freq_tables(split: str, country: str):
    s1, recs = load_universe(split, sample=False)
    s1 = s1.filter(pl.col("country") == country)
    recs = recs.filter(pl.col("country") == country)
    full1 = load_norm(split, 1, ["idx", "a_city"]).rename({"idx": "s1"})
    s1 = s1.join(full1, on="s1", how="left")
    fk1 = s1.filter(pl.col("n_key") != "").group_by("n_key").len().rename({"len": "cnt_key_s1"})
    fkr = recs.filter(pl.col("n_key") != "").group_by("n_key").len().rename({"len": "cnt_key_rec"})
    fa1 = s1.filter(pl.col("a_tok") != "").group_by("a_tok").len().rename({"len": "cnt_atok_s1"})
    fh1 = (s1.filter((pl.col("a_house") != "")).group_by(["a_house", "a_street", "a_city"]).len()
           .rename({"len": "cnt_hsc_s1"}))
    fkc = s1.filter(pl.col("n_key") != "").group_by(["n_key", "a_city"]).len().rename({"len": "cnt_keycity_s1"})
    return fk1, fkr, fa1, fh1, fkc


def add_freq(d: pl.DataFrame, ft) -> pl.DataFrame:
    fk1, fkr, fa1, fh1, fkc = ft
    d = d.join(fk1.rename({"n_key": "l_n_key", "cnt_key_s1": "l_cnt_key_s1"}), on="l_n_key", how="left")
    d = d.join(fk1.rename({"n_key": "r_n_key", "cnt_key_s1": "r_cnt_key_s1"}), on="r_n_key", how="left")
    d = d.join(fkr.rename({"n_key": "r_n_key", "cnt_key_rec": "r_cnt_key_rec"}), on="r_n_key", how="left")
    d = d.join(fkr.rename({"n_key": "l_n_key", "cnt_key_rec": "l_cnt_key_rec"}), on="l_n_key", how="left")
    d = d.join(fa1.rename({"a_tok": "l_a_tok", "cnt_atok_s1": "l_cnt_atok_s1"}), on="l_a_tok", how="left")
    d = d.join(fa1.rename({"a_tok": "r_a_tok", "cnt_atok_s1": "r_cnt_atok_s1"}), on="r_a_tok", how="left")
    d = d.join(fh1.rename({"a_house": "l_a_house", "a_street": "l_a_street", "a_city": "l_a_city",
                           "cnt_hsc_s1": "l_cnt_hsc_s1"}), on=["l_a_house", "l_a_street", "l_a_city"], how="left")
    d = d.join(fkc.rename({"n_key": "l_n_key", "a_city": "l_a_city", "cnt_keycity_s1": "l_cnt_keycity_s1"}),
               on=["l_n_key", "l_a_city"], how="left")
    return d


FREQ_COLS = ["l_cnt_key_s1", "r_cnt_key_s1", "r_cnt_key_rec", "l_cnt_key_rec", "l_cnt_atok_s1", "r_cnt_atok_s1",
             "l_cnt_hsc_s1", "l_cnt_keycity_s1"]


# ----------------------------------------------------------------------------- idf
def global_idf_tables() -> list[pl.DataFrame]:
    """One rarity table per text view, shared by every split and country, so a word always gets the same value
    (the model can learn word identity once; e.g. the noise word 'center' is not re-valued per universe).
    Built from all provided source files (train + test, S1/S2/S3); unsupervised token counts only."""
    out = []
    for col in ("n_core", "a_tok"):
        p = WORK / f"idf_global_{col}.parquet"
        if not p.exists():
            parts = [load_norm(split, s, ["idx", col]).filter(pl.col("idx") < SYN_BASE).select(col)
                     for split in ("train", "test") for s in (1, 2, 3)]
            both = pl.concat(parts)
            n = both.height
            df = (both.select(pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
                  .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).group_by("t").len())
            df.with_columns(pl.lit(n).alias("n")).write_parquet(p)
        out.append(pl.read_parquet(p))
    return out


def build_idf(split: str, country: str):
    global IDF_N, IDF_A, DEF_N, DEF_A, DF_N, DF_A
    out = []
    for df in global_idf_tables():
        n = int(df["n"][0])
        idf_v = np.minimum(np.log((n + 1) / (df["len"].to_numpy() + 1)), IDF_CAP)
        toks = df["t"].to_list()
        out.append((dict(zip(toks, idf_v.tolist())), IDF_CAP, dict(zip(toks, df["len"].to_list()))))
    (IDF_N, DEF_N, DF_N), (IDF_A, DEF_A, DF_A) = out


# ----------------------------------------------------------------------------- group context
def group_context(d: pl.DataFrame) -> pl.DataFrame:
    """Rank/gap of raw scores within record and within (S1, source) groups; computed on a narrow frame
    in d's row order, then stacked horizontally (no wide joins)."""
    z = lambda c: pl.col(c).fill_nan(0).fill_null(0)
    n = d.select(KEY + [z("p0").alias("p0"), ((z("fn_tset") + z("fa_tset")) / 200).alias("sc_fz"),
                        z("fn_tset").alias("fn_tset"), z("fa_tset").alias("fa_tset")])
    new = [((z("fn_tset") + z("fa_tset")) / 200).cast(pl.Float32).alias("sc_fz")]
    out = pl.DataFrame({"sc_fz": d.select(new[0])["sc_fz"]})
    for sc in ("p0", "sc_fz", "fn_tset", "fa_tset"):
        t = n.select(KEY + [pl.col(sc).alias("v")]).with_columns(
            pl.col("v").rank("ordinal", descending=True).over(["src", "rid"]).alias("rk_r"),
            pl.col("v").max().over(["src", "rid"]).alias("mx_r"),
            pl.col("v").rank("ordinal", descending=True).over(["s1", "src"]).alias("rk_1"),
            pl.col("v").max().over(["s1", "src"]).alias("mx_1"),
        )
        sec_r = t.filter(pl.col("rk_r") == 2).select(["src", "rid", pl.col("v").alias("s2_r")])
        sec_1 = t.filter(pl.col("rk_1") == 2).select(["s1", "src", pl.col("v").alias("s2_1")])
        t = t.join(sec_r, on=["src", "rid"], how="left", maintain_order="left").join(
            sec_1, on=["s1", "src"], how="left", maintain_order="left")
        t = t.select(
            pl.col("rk_r").cast(pl.Float32).alias(f"g_{sc}_rk_rec"),
            (pl.col("v") - pl.when(pl.col("rk_r") == 1).then(pl.col("s2_r").fill_null(0))
             .otherwise(pl.col("mx_r"))).cast(pl.Float32).alias(f"g_{sc}_gap_rec"),
            pl.col("rk_1").cast(pl.Float32).alias(f"g_{sc}_rk_s1"),
            (pl.col("v") - pl.when(pl.col("rk_1") == 1).then(pl.col("s2_1").fill_null(0))
             .otherwise(pl.col("mx_1"))).cast(pl.Float32).alias(f"g_{sc}_gap_s1"),
        )
        out = pl.concat([out, t], how="horizontal")
        del t
    out = pl.concat([out, n.select(
        pl.len().over(["src", "rid"]).cast(pl.Float32).alias("g_n_rec"),
        pl.len().over(["s1", "src"]).cast(pl.Float32).alias("g_n_s1src"),
        pl.len().over("s1").cast(pl.Float32).alias("g_n_s1"),
        pl.col("p0").sum().over("s1").cast(pl.Float32).alias("g_p0_sum_s1"),
        (pl.col("p0") > 0.5).sum().over("s1").cast(pl.Float32).alias("g_p0_conf_s1"),
        (pl.col("p0") > 0.5).sum().over(["src", "rid"]).cast(pl.Float32).alias("g_p0_conf_rec"),
    )], how="horizontal")
    return pl.concat([d, out], how="horizontal")


# ----------------------------------------------------------------------------- driver
def run(split: str, sample: bool, force: bool = False, countries=None):
    countries = countries or sorted({p.name.split("_")[-1].replace(".parquet", "")
                                     for p in WORK.glob(f"cand_{split}{'_sample' if sample else ''}_*.parquet")})
    sp_ = load_splits() if split == "train" else None
    for country in countries:
        out_p = feats_path(split, sample, country)
        if out_p.exists() and not force:
            continue
        with Timer(f"feats {split} {country}"):
            c = pl.read_parquet(pruned_path(split, sample, country))
            n1 = load_norm(split, 1, ["idx"] + TXT).rename({"idx": "s1"}).rename({k: "l_" + k for k in TXT})
            nr = pl.concat([load_norm(split, s, ["idx"] + TXT).with_columns(pl.lit(s, pl.Int8).alias("src"))
                            for s in (2, 3)]).rename({"idx": "rid"}).rename({k: "r_" + k for k in TXT})
            n1 = n1.join(c.select("s1").unique(), on="s1", how="semi")
            nr = nr.join(c.select(["src", "rid"]).unique(), on=["src", "rid"], how="semi")
            build_idf(split, country)
            _load_extra_words()
            ft = freq_tables(split, country)
            parts = []
            ctx = get_context("fork")
            with ctx.Pool(min(NW, 20)) as pool:
                for s in range(0, c.height, CHUNK):
                    ch = c.slice(s, CHUNK).join(n1, on="s1", how="left").join(nr, on=["src", "rid"], how="left")
                    ch = add_freq(ch, ft)
                    f = fuzzy_features(ch)
                    f.update(py_features(pool, ch))
                    st = struct_features(ch)
                    blk = ch.select([col for col in c.columns] + FREQ_COLS)
                    part = pl.concat([blk, st, pl.DataFrame(f)], how="horizontal")
                    part = part.with_columns([pl.col(x).cast(pl.Float32) for x in part.columns if x not in KEY])
                    parts.append(part)
                    del part
                    del ch, f, st
                    gc.collect()
                    print(f"    chunk {s // CHUNK}: {parts[-1].height:,} rows", flush=True)
            d = pl.concat(parts)
            del parts
            d = group_context(d)
            if split == "train":
                gt = universe_gt(sample).with_columns(pl.lit(1, pl.Int8).alias("y"))
                lab = d.select(KEY).join(gt, on=KEY, how="left", maintain_order="left").join(
                    sp_.select(["s1", "fold"]), on="s1", how="left", maintain_order="left")
                d = d.with_columns(lab["y"].fill_null(0).alias("y"), lab["fold"].alias("fold"))
                del lab
            fcols = [x for x in d.columns if x not in KEY + ["y", "fold"]]
            d.write_parquet(out_p)
            print(f"  {country}: {d.height:,} pairs x {len(fcols)} features", flush=True)
            del d, c, n1, nr
            gc.collect()


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    run(split, sample="--sample" in sys.argv, force="--force" in sys.argv)
