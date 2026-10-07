"""Stage `block`: candidate generation (always within country).

Channels (union):
  name  : char 3-grams of the compact core name (spaces removed; domain if no core)   rec->S1 top-k
  addr  : address word tokens (canonical)                                              rec->S1 top-k
  comb  : [name, addr] / sqrt(2) (cosine = mean of both)                               rec->S1 and S1->rec top-k
  key_n : exact normalized name key (sorted non-generic core tokens), block cap
  key_a : exact (house number, street core), block cap
Every candidate pair then gets all view cosines + per-channel ranks (first-stage ranker input).
Output: work/cand_raw_{split}{_sample}_{country}.parquet
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from io_utils import CFG, WORK, Timer
from normalize import load_norm
from splits import REC_SAMPLE, excluded_records, load_splits

NT = CFG["resources"]["n_threads"]
_B = CFG.get("blocking", {}) or {}          # optional overrides (config `blocking`), defaults = submitted version
K_NAME, K_ADDR, K_COMB, K_REV, K_WIDE = (int(_B.get(k, v)) for k, v in
                                         (("k_name", 20), ("k_addr", 15), ("k_comb", 20), ("k_rev", 20), ("k_wide", 60)))
MAXDF_NAME, MAXDF_ADDR = 0.01, 0.02
K_EMPTY = int(_B.get("k_empty", 50))
KEY_CAP = int(_B.get("key_cap", 20))
FLOOR = 0.05
NAME_COLS = ["n_core", "n_dom", "n_key"]
ADDR_COLS = ["a_tok", "a_house", "a_street"]


def cand_path(split, sample, country):
    return WORK / f"cand_raw_{split}{'_sample' if sample else ''}_{country}.parquet"


def load_universe(split: str, sample: bool):
    cols = ["idx", "country"] + NAME_COLS + ADDR_COLS
    s1 = load_norm(split, 1, cols).rename({"idx": "s1"})
    if split == "train":
        sp_ = load_splits().filter(~pl.col("dropped"))
        if sample:
            sp_ = sp_.filter(pl.col("in_sample"))
        s1 = s1.join(sp_.select("s1"), on="s1", how="semi")
    recs = pl.concat([load_norm(split, s, cols).rename({"idx": "rid"}).with_columns(pl.lit(s, pl.Int8).alias("src"))
                      for s in (2, 3)])
    s1 = s1.join(id_tokens(split, 1).rename({"idx": "s1"}), on="s1", how="left")
    recs = recs.join(pl.concat([id_tokens(split, s).rename({"idx": "rid"}).with_columns(pl.lit(s, pl.Int8).alias("src"))
                                for s in (2, 3)]), on=["src", "rid"], how="left")
    if split == "train" and sample:
        rs = pl.read_parquet(REC_SAMPLE).filter(pl.col("in_sample"))
        recs = recs.join(rs.select(["src", "rid"]), on=["src", "rid"], how="semi")
    if split == "train":
        recs = recs.join(excluded_records(), on=["src", "rid"], how="anti")
    return s1, recs


def id_tokens(split: str, src: int) -> pl.DataFrame:
    """Glued alphanumeric identifiers from the raw address (B-126 -> #b126, C/76 -> #c76, 4514D -> #4514d)."""
    from io_utils import load_source
    a = load_source(split, src).select(["idx", pl.col("addr").str.to_lowercase().alias("a")])
    a = a.with_columns(pl.col("a").str.replace_all(r"([a-z])[\-/\.\s]?(\d)", "$1$2")
                       .str.replace_all(r"(\d)[\-/]?([a-z])", "$1$2"))
    a = a.with_columns(pl.col("a").str.extract_all(r"\b(?:[a-z]+\d[a-z0-9]*|\d+[a-z][a-z0-9]*)\b")
                       .list.eval(pl.lit("#") + pl.element()).list.join(" ").alias("a_ids"))
    return a.select(["idx", "a_ids"])


# ----------------------------------------------------------------------------- vectorizing
_HV_CHAR = HashingVectorizer(analyzer="char", ngram_range=(3, 3), n_features=2 ** 21, alternate_sign=False,
                             norm=None, dtype=np.float32, lowercase=False)
_HV_WORD = HashingVectorizer(analyzer=str.split, n_features=2 ** 21, alternate_sign=False, norm=None,
                             dtype=np.float32, lowercase=False)


def _hv(kind, docs):
    return (_HV_CHAR if kind == "char" else _HV_WORD).transform(docs)


def hash_docs(kind: str, docs: list[str], chunk: int = 200_000) -> sp.csr_matrix:
    parts = Parallel(n_jobs=min(NT, 16), backend="loky")(
        delayed(_hv)(kind, docs[i:i + chunk]) for i in range(0, len(docs), chunk))
    return sp.vstack(parts, format="csr")


def tfidf(mats: list[sp.csr_matrix], max_df: float = 0.05, min_df: int = 1, return_norms: bool = False):
    """Sublinear TF * IDF over the union of mats, drop very common features, L2 normalize."""
    n = sum(m.shape[0] for m in mats)
    df = np.zeros(mats[0].shape[1], dtype=np.int64)
    for m in mats:
        df += np.bincount(m.indices, minlength=m.shape[1])
    idf = np.log((n + 1) / (df + 1)).astype(np.float32) + 1.0
    idf[(df > max_df * n) | (df < min_df)] = 0.0
    out, norms = [], []
    for m in mats:
        m = m.copy()
        m.data = (1.0 + np.log(m.data)) * idf[m.indices]
        m.eliminate_zeros()
        nrm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        nrm[nrm == 0] = 1.0
        m = sp.diags(1.0 / nrm).astype(np.float32) @ m
        out.append(m.tocsr())
        norms.append(nrm)
    return (out, norms) if return_norms else out


def name_doc(core: str, dom: str) -> str:
    s = core.replace(" ", "") if core else dom
    return f" {s} " if s else ""


TBLOCK = 131_072   # target columns per block: per-thread dense accumulator stays in L2 cache


def topk(Q: sp.csr_matrix, T: sp.csr_matrix, k: int, chunk: int = 400_000):
    """For each row of Q, top-k rows of T by cosine (>= FLOOR). Returns (qi, ti, score, rank).

    T is split into column blocks of TBLOCK rows; per-block top-k lists are merged per query row.
    """
    nb = max(1, int(np.ceil(T.shape[0] / TBLOCK)))
    bounds = np.linspace(0, T.shape[0], nb + 1).astype(np.int64)
    TTs = [T[bounds[b]:bounds[b + 1]].T.tocsr() for b in range(nb)]
    qi, ti, sc, rk = [], [], [], []
    for s in range(0, Q.shape[0], chunk):
        Qc = Q[s:s + chunk]
        bq, bt, bs = [], [], []
        for b, TT in enumerate(TTs):
            C = sp_matmul_topn(Qc, TT, top_n=k, threshold=FLOOR, sort=False, n_threads=NT).tocsr()
            cnt = np.diff(C.indptr)
            bq.append(np.repeat(np.arange(C.shape[0], dtype=np.int32), cnt))
            bt.append((C.indices + bounds[b]).astype(np.int32))
            bs.append(C.data.astype(np.float32))
        q, t, v = np.concatenate(bq), np.concatenate(bt), np.concatenate(bs)
        order = np.lexsort((-v, q))
        q, t, v = q[order], t[order], v[order]
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        cnt = np.diff(np.r_[starts, len(q)])
        r = np.arange(len(q), dtype=np.int32) - np.repeat(starts, cnt).astype(np.int32)
        m = r < k
        qi.append(q[m] + s)
        ti.append(t[m])
        sc.append(v[m])
        rk.append(r[m].astype(np.int8))
    return (np.concatenate(qi), np.concatenate(ti), np.concatenate(sc), np.concatenate(rk))


def rowdot(A: sp.csr_matrix, ia: np.ndarray, B: sp.csr_matrix, ib: np.ndarray, chunk: int = 2_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = A[ia[s:s + chunk]]
        b = B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(1)).ravel()
    return out


# ----------------------------------------------------------------------------- per-country blocking
def block_country(s1: pl.DataFrame, rc: pl.DataFrame, country: str) -> pl.DataFrame:
    t0 = time.time()
    s1_ids = s1["s1"].to_numpy()
    r_src = rc["src"].to_numpy()
    r_rid = rc["rid"].to_numpy()
    n1, nr = s1.height, rc.height
    print(f"  [{country}] S1={n1:,} recs={nr:,}", flush=True)

    nd1 = [name_doc(c, d) for c, d in zip(s1["n_core"].to_list(), s1["n_dom"].to_list())]
    ndr = [name_doc(c, d) for c, d in zip(rc["n_core"].to_list(), rc["n_dom"].to_list())]
    N1, NR = tfidf([hash_docs("char", nd1), hash_docs("char", ndr)], max_df=MAXDF_NAME)
    del nd1, ndr
    ad1 = (s1["a_tok"] + " " + s1["a_ids"].fill_null("")).to_list()
    adr = (rc["a_tok"] + " " + rc["a_ids"].fill_null("")).to_list()
    (A1, AR), (nrm1, _) = tfidf([hash_docs("word", ad1), hash_docs("word", adr)], max_df=MAXDF_ADDR,
                                return_norms=True)
    del ad1, adr
    # containment-style S1 vectors: raw tf-idf / sqrt(norm) (less penalty for long S1 addresses)
    A1c = (sp.diags(np.sqrt(nrm1 / max(nrm1.mean(), 1e-6))).astype(np.float32) @ A1).tocsr()
    w = np.float32(1 / np.sqrt(2))
    C1 = (sp.hstack([N1, A1], format="csr") * w).astype(np.float32)
    CR = (sp.hstack([NR, AR], format="csr") * w).astype(np.float32)
    print(f"  [{country}] vectorized {time.time() - t0:.0f}s nnz name={NR.nnz:,} addr={AR.nnz:,}", flush=True)

    frames = []

    def add(qi_r, ti_1, score, rank, ch):
        frames.append(pl.DataFrame({"ri": qi_r, "i1": ti_1, f"r_{ch}": rank}))

    for ch, (Q, T, k) in {"name": (NR, N1, K_NAME), "addr": (AR, A1, K_ADDR), "addrc": (AR, A1c, K_ADDR)}.items():
        qi, ti, sc, rk = topk(Q, T, k)
        add(qi, ti, sc, rk, ch)
        print(f"  [{country}] {ch}: {len(qi):,} pairs {time.time() - t0:.0f}s", flush=True)
    # records without a usable address: wide name search
    ea = np.where(np.asarray(AR.getnnz(axis=1)) == 0)[0]
    if len(ea):
        qi, ti, sc, rk = topk(NR[ea], N1, K_EMPTY)
        frames.append(pl.DataFrame({"ri": ea[qi].astype(np.int32), "i1": ti, "r_nwide": rk}))
        print(f"  [{country}] empty-addr name top-{K_EMPTY}: {len(ea):,} recs {len(qi):,} pairs", flush=True)
    # comb: one wide forward pass (top K_WIDE S1 per record); the reverse channel (S1 -> top-k records
    # per source) is derived from it instead of a second, much slower product.
    qi, ti, sc, rk = topk(CR, C1, K_WIDE)
    cw = pl.DataFrame({"ri": qi, "i1": ti, "sc": sc, "r_comb": rk, "src": r_src[qi]})
    cw = cw.with_columns(pl.col("sc").rank("ordinal", descending=True).over(["i1", "src"])
                         .sub(1).clip(0, 127).cast(pl.Int8).alias("r_rev"))
    cw = cw.filter((pl.col("r_comb") < K_COMB) | (pl.col("r_rev") < K_REV))
    frames.append(cw.select(["ri", "i1", pl.when(pl.col("r_comb") < K_COMB).then(pl.col("r_comb")).alias("r_comb")])
                  .filter(pl.col("r_comb").is_not_null()))
    frames.append(cw.filter(pl.col("r_rev") < K_REV).select(["ri", "i1", "r_rev"]))
    print(f"  [{country}] comb+rev: {cw.height:,} pairs {time.time() - t0:.0f}s", flush=True)
    del cw

    # exact keys (block cap)
    s1l = s1.with_row_index("i1").with_columns(pl.col("i1").cast(pl.Int32))
    rcl = rc.with_row_index("ri").with_columns(pl.col("ri").cast(pl.Int32))
    kn1 = s1l.filter(pl.col("n_key") != "").select(["i1", "n_key"])
    kn1 = kn1.join(kn1.group_by("n_key").len().filter(pl.col("len") <= KEY_CAP).select("n_key"), on="n_key")
    kn = rcl.filter(pl.col("n_key") != "").select(["ri", "n_key"]).join(kn1, on="n_key").select(
        ["ri", "i1"]).with_columns(pl.lit(1, pl.Int8).alias("k_name"))
    ka1 = s1l.filter((pl.col("a_house") != "") & (pl.col("a_street") != "")).select(["i1", "a_house", "a_street"])
    ka1 = ka1.join(ka1.group_by(["a_house", "a_street"]).len().filter(pl.col("len") <= KEY_CAP)
                   .select(["a_house", "a_street"]), on=["a_house", "a_street"])
    ka = rcl.filter((pl.col("a_house") != "") & (pl.col("a_street") != "")).select(["ri", "a_house", "a_street"]) \
        .join(ka1, on=["a_house", "a_street"]).select(["ri", "i1"]).with_columns(pl.lit(1, pl.Int8).alias("k_addr"))
    print(f"  [{country}] keys: name {kn.height:,} addr {ka.height:,}", flush=True)

    # union via int64 keys (ri * n1 + i1): low-memory merge of all channels
    chans = []
    for f in frames:
        col = [c for c in f.columns if c.startswith("r_")][0]
        chans.append((col, f["ri"].to_numpy().astype(np.int64) * n1 + f["i1"].to_numpy(), f[col].to_numpy()))
    for col, kf in (("k_name", kn), ("k_addr", ka)):
        chans.append((col, kf["ri"].to_numpy().astype(np.int64) * n1 + kf["i1"].to_numpy(),
                      np.ones(kf.height, dtype=np.int8)))
    del frames, kn, ka
    allk = np.concatenate([c[1] for c in chans])
    uniq, inv = np.unique(allk, return_inverse=True)
    del allk
    cols = {}
    pos = 0
    for col, k, v in chans:
        n = len(k)
        if col.startswith("k_"):
            a = np.zeros(len(uniq), dtype=np.int8)
            a[inv[pos:pos + n]] = 1
            cols[col] = pl.Series(col, a)
        else:
            a = np.full(len(uniq), -1, dtype=np.int8)
            idx = inv[pos:pos + n]
            # keep the best (smallest) rank if a channel produced duplicates
            order = np.argsort(-v.astype(np.int16), kind="stable")
            a[idx[order]] = v[order]
            cols[col] = pl.Series(col, a).set(pl.Series(a == -1), None) if False else pl.Series(col, a)
        pos += n
    del chans, inv
    ri = (uniq // n1).astype(np.int32)
    i1 = (uniq % n1).astype(np.int32)
    del uniq
    u = pl.DataFrame(cols).with_columns([pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
                                         for c in cols if not c.startswith("k_")])
    u = u.with_columns(
        pl.Series("cos_name", rowdot(NR, ri, N1, i1)),
        pl.Series("cos_addr", rowdot(AR, ri, A1, i1)),
        pl.Series("s1", s1_ids[i1].astype(np.int32)),
        pl.Series("src", r_src[ri].astype(np.int8)),
        pl.Series("rid", r_rid[ri].astype(np.int32)),
    ).with_columns(((pl.col("cos_name") + pl.col("cos_addr")) / 2).alias("cos_comb"))
    print(f"  [{country}] union {u.height:,} pairs ({u.height / nr:.1f}/rec) {time.time() - t0:.0f}s", flush=True)
    del N1, NR, A1, AR, C1, CR, A1c
    gc.collect()
    return u


def run(split: str, sample: bool, force: bool = False):
    s1, recs = load_universe(split, sample)
    for country in sorted(s1["country"].unique().to_list()):
        p = cand_path(split, sample, country)
        if p.exists() and not force:
            continue
        with Timer(f"block {split} {country}"):
            rc = recs.filter(pl.col("country") == country)
            u = block_country(s1.filter(pl.col("country") == country), rc, country)
            u.write_parquet(p)
            del u
            gc.collect()
    unknown = recs.filter(~pl.col("country").is_in(s1["country"].unique().to_list()))
    if unknown.height:
        print(f"WARNING: {unknown.height} records with a country absent from S1 (no candidates)")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    run(split, sample="--sample" in sys.argv, force="--force" in sys.argv)
