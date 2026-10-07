"""Mine dictionaries from the provided files only (no external data).

Outputs (work/dicts/):
  name_vocab.json   S1 name tokens (train+test) with count >= 3 (OCR repair choice 1->l/i)
  translit.json     native-script name token -> Latin token (aligned train pairs)
  state.json        {country: {variant_key: canonical_state}} incl. codes and native script
  fr_region.json    French department/region variant -> canonical region (test-file co-occurrence)
  addr_syn.json     address token variant -> canonical S1 token (single-sided token substitutions)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import polars as pl

from io_utils import WORK, Timer, load_gt, load_source
from textnorm import ADDR_CANON, ADDR_TOK_RE, NONLATIN_RE, ZW_RE, fold, has_nonlatin

OUT = WORK / "dicts"
OUT.mkdir(parents=True, exist_ok=True)
PUNCT_RE = re.compile(r"[\s\.\,\-\(\)\[\]\|/&:;'\"।]+")


def comp_keys(addr: str) -> list[str]:
    out = []
    for c in addr.split(","):
        c0 = ZW_RE.sub("", c).strip()
        if not c0:
            continue
        if has_nonlatin(c0):
            out.append(c0)
        else:
            k = " ".join(ADDR_TOK_RE.findall(fold(c0)))
            if k:
                out.append(k)
    return out


def pairs_df(sample: int | None = None) -> pl.DataFrame:
    gt = load_gt()
    if sample:
        gt = gt.sample(sample, seed=7)
    s1 = load_source("train", 1).select(pl.col("idx").alias("s1"), pl.col("name").alias("n1"),
                                         pl.col("addr").alias("a1"), "country")
    r = pl.concat([load_source("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                  "name", "addr") for s in (2, 3)])
    return gt.join(s1, on="s1").join(r, on=["src", "rid"])


def mine_vocab():
    cnt = Counter()
    for split in ("train", "test"):
        for n in load_source(split, 1)["name"].to_list():
            cnt.update(re.sub(r"[^a-z0-9]+", " ", fold(n).replace(".", "").replace("'", "")).split())
    vocab = sorted(t for t, c in cnt.items() if c >= 3 and t.isalpha())
    (OUT / "name_vocab.json").write_text(json.dumps(vocab))
    print("vocab", len(vocab))


def mine_translit(P: pl.DataFrame):
    x = P.filter(pl.col("name").str.contains(r"[ऀ-෿]"))
    cnt = defaultdict(Counter)
    n_al = 0
    for n1, n2 in zip(x["n1"].to_list(), x["name"].to_list()):
        lt = [t for t in PUNCT_RE.split(fold(n1).replace(".", "")) if t]
        nt = [t for t in PUNCT_RE.split(ZW_RE.sub("", n2)) if t]
        if len(lt) == len(nt):
            n_al += 1
            for a, b in zip(nt, lt):
                if NONLATIN_RE.search(a):
                    cnt[a][b] += 1
    d = {}
    for a, c in cnt.items():
        b, k = c.most_common(1)[0]
        tot = sum(c.values())
        if k >= 2 and k / tot >= 0.6:
            d[a] = b
    (OUT / "translit.json").write_text(json.dumps(d, ensure_ascii=False))
    print(f"translit: {x.height} non-latin pairs, {n_al} aligned, {len(d)} tokens")
    top = sorted(((sum(c.values()), a, c.most_common(1)[0][0]) for a, c in cnt.items() if a in d), reverse=True)[:25]
    print("  top:", [(a, b, n) for n, a, b in top])
    return d


def s1_states():
    """Per country: S1 components that behave like a state (mostly last component)."""
    states = {}
    for country in ("US", "India"):
        last, anyc = Counter(), Counter()
        for split in ("train", "test"):
            df = load_source(split, 1).filter(pl.col("country") == country)
            for a in df["addr"].to_list():
                ks = comp_keys(a)
                if not ks:
                    continue
                last[ks[-1]] += 1
                anyc.update(set(ks))
        st = {k for k, v in last.items() if v >= 50 and v / anyc[k] >= 0.5 and not any(ch.isdigit() for ch in k)}
        states[country] = st
        print(country, "S1 states:", len(st), sorted(st)[:60])
    states["France"] = set(FR_REGIONS)       # France is not in train: hand-written regions (general knowledge)
    return states


def mine_states(P: pl.DataFrame, states: dict):
    forms = {c: {s: {s} for s in st} for c, st in states.items()}
    out = {c: {s: s for s in st} for c, st in states.items()}
    rows = list(zip(P["country"].to_list(), P["a1"].to_list(), P["addr"].to_list()))
    parsed = []
    for country, a1, a2 in rows:
        k1 = comp_keys(a1)
        st = [k for k in k1 if k in states.get(country, ())]
        if len(set(st)) != 1:
            continue
        k2 = comp_keys(a2)
        parsed.append((country, st[0], set(k1), k2))
    n_state = Counter((c, s) for c, s, _, _ in parsed)
    for it in (1, 2):
        cnt, tot, alone, edge = Counter(), Counter(), Counter(), Counter()
        known = {}
        for c in out:
            for v, s in out[c].items():
                known[(c, v)] = s
        for country, s, k1, k2 in parsed:
            fs = forms[country][s]
            for pos, v in enumerate(k2):
                if v in k1 or (country, v) in known:
                    continue
                cnt[(country, v, s)] += 1
                if pos == 0 or pos == len(k2) - 1:
                    edge[(country, v, s)] += 1
                tot[(country, v)] += 1
                if not any(w in fs for w in k2 if w != v):
                    alone[(country, v, s)] += 1
        added = 0
        for (country, v, s), k in cnt.items():
            if it == 1 and not (len(v) <= 3 and v.isalpha() or NONLATIN_RE.search(v)):
                continue
            alone_ok = (edge[(country, v, s)] / k >= 0.8) if it == 1 else alone[(country, v, s)] / k >= 0.9
            if (k >= 20 and k / tot[(country, v)] >= 0.95 and alone_ok
                    and k / n_state[(country, s)] >= (0.002 if it == 1 else 0.01)):
                out[country][v] = s
                forms[country][s].add(v)
                added += 1
        print(f"state mining iter {it}: +{added}")
    # mined merges: R state systematically differs from S1 state (e.g. Telangana written as Andhra Pradesh)
    conf, n_s = Counter(), Counter()
    for country, s, k1, k2 in parsed:
        rs = {out[country][v] for v in k2 if v in out[country]}
        if len(rs) == 1:
            r = rs.pop()
            n_s[(country, s)] += 1
            if r != s:
                conf[(country, s, r)] += 1
    for (country, s, r), k in conf.most_common():
        if k / n_s[(country, s)] >= 0.05 and k >= 1000:
            tgt = out[country][r]
            print(f"  merge state {s} -> {tgt} ({k}/{n_s[(country, s)]})")
            for v, cs in list(out[country].items()):
                if cs == s:
                    out[country][v] = tgt
    for c in out:
        print(c, "state variants:", len(out[c]))
        inv = defaultdict(list)
        for v, s in out[c].items():
            inv[s].append(v)
        for s in sorted(inv)[:80]:
            print("   ", s, "<-", sorted(inv[s], key=len)[:8])
    return out


# France is not in train, so its geography is hand-written general knowledge (not mined from test records):
# the three regions of the data and the departements that appear instead of them in S2/S3 addresses.
FR_REGIONS = ["hauts de france", "nouvelle aquitaine", "pays de la loire"]
FR_DEPT_REGION = {"nord": "hauts de france", "pas de calais": "hauts de france", "gironde": "nouvelle aquitaine",
                  "loire atlantique": "pays de la loire"}


def mine_fr_region(states: dict):
    out = {r: r for r in FR_REGIONS}
    out.update(FR_DEPT_REGION)
    print("fr_region (hand-written):", out)
    (OUT / "fr_region.json").write_text(json.dumps(out))
    return out


def mine_addr_syn(P: pl.DataFrame, state_map: dict):
    from anyascii import anyascii
    bad = set()
    for m in state_map.values():
        for v in m:
            bad.update(ADDR_TOK_RE.findall(fold(v)))
            bad.update(ADDR_TOK_RE.findall(anyascii(v).lower()))
    cnt, tot_b = Counter(), Counter()
    for a1, a2 in zip(P["a1"].to_list(), P["addr"].to_list()):
        t1 = {ADDR_CANON.get(t, t) for t in ADDR_TOK_RE.findall(fold(a1)) if not t.isdigit()}
        t2 = {ADDR_CANON.get(t, t) for t in ADDR_TOK_RE.findall(fold(a2)) if not t.isdigit()}
        A, B = t1 - t2, t2 - t1
        for b in B:
            tot_b[b] += 1
        if 1 <= len(A) <= 2 and 1 <= len(B) <= 2:
            for a in A:
                for b in B:
                    cnt[(b, a)] += 1
    syn = {}
    for (b, a), k in cnt.most_common():
        if k < 30:
            break
        if b in syn or len(b) < 2 or len(a) < 2 or a in bad or b in bad:
            continue
        if k / tot_b[b] >= 0.5:
            syn[b] = a
    print("addr_syn:", len(syn), list(syn.items())[:100])
    (OUT / "addr_syn.json").write_text(json.dumps(syn))
    return syn


def mine_name_extra(min_count: int = 300):
    """Words that appear only on the record side of a pair. Noise words (appended by the noise generator to
    true matches, e.g. center/services) vs sibling words (appended to generated look-alike distractors,
    e.g. ventures/enterprises). True pairs: ground truth. Look-alike pairs: generated distractors (no owner)
    joined to the S1 with the same name key and city. Train only."""
    gt = load_gt()
    s1 = load_source("train", 1).select(pl.col("idx").alias("s1"))
    from normalize import load_norm
    n1 = load_norm("train", 1, ["idx", "country", "n_core", "n_key", "a_city"]).rename({"idx": "s1"})
    nr = pl.concat([load_norm("train", s, ["idx", "country", "n_core", "n_key", "a_city"])
                    .filter(pl.col("idx") < 10_000_000)            # real records only (no synthetic siblings)
                    .with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)]).rename({"idx": "rid"})

    def extra(pairs):
        x = pairs.join(n1.select(["s1", pl.col("n_core").alias("l")]), on="s1").join(
            nr.select(["src", "rid", pl.col("n_core").alias("r")]), on=["src", "rid"])
        x = x.with_columns(pl.col("r").str.split(" ").list.set_difference(pl.col("l").str.split(" ")).alias("E"))
        return x.select("E").explode("E").filter(pl.col("E").is_not_null() & (pl.col("E") != "")).group_by("E").len()

    t = extra(gt.sample(min(3_000_000, gt.height), seed=1)).rename({"len": "n_true"})
    gen = nr.join(gt.select(["src", "rid"]), on=["src", "rid"], how="anti").filter(pl.col("n_key") != "")
    look = gen.join(n1.filter(pl.col("n_key") != ""), on=["country", "n_key", "a_city"]).select(["s1", "src", "rid"])
    f = extra(look).rename({"len": "n_false"})
    j = t.join(f, on="E", how="full", coalesce=True).fill_null(0)
    nt, nf = j["n_true"].sum(), j["n_false"].sum()
    j = j.with_columns(((pl.col("n_true") + 1) / nt / ((pl.col("n_false") + 1) / nf)).alias("ratio"))
    j = j.filter((pl.col("n_true") + pl.col("n_false")) >= min_count)
    noise = sorted(j.filter(pl.col("ratio") >= 2.0)["E"].to_list())
    sib = sorted(j.filter(pl.col("ratio") <= 0.5)["E"].to_list())
    (OUT / "name_extra.json").write_text(json.dumps({"noise": noise, "sibling": sib}))
    top = j.sort("n_true", descending=True).head(40)
    print("name_extra: noise", len(noise), "sibling", len(sib))
    print("  top by frequency:", [(e, a, b, round(r, 2)) for e, a, b, r in top.iter_rows()])


if __name__ == "__main__":
    import sys
    if "--skip-vocab" not in sys.argv:
        with Timer("vocab"):
            mine_vocab()
    with Timer("pairs"):
        P = pairs_df()
    with Timer("translit"):
        mine_translit(P)
    with Timer("states"):
        st = s1_states()
        state_map = mine_states(P.sample(3_000_000, seed=3), st)
        (OUT / "state.json").write_text(json.dumps(state_map, ensure_ascii=False))
    with Timer("fr_region"):
        mine_fr_region(st)
    with Timer("addr_syn"):
        mine_addr_syn(P.sample(1_500_000, seed=5), state_map)
