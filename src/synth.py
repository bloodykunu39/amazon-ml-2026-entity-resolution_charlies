"""Stage `synth`: synthetic sibling businesses for the train universe (v4).

Test contains hidden "sibling" businesses next to real S1 entities: the S1's name plus a sibling word
(Southside, Holdings, Industries, ...), the same street with the house number shifted by an offset from
{1,2,3,4,5,7,9,11,13,21}, and several records in both sources that share that shifted number. Train has the
same sibling words and offsets (mined from train's own no-owner records, see reports/phase5.md), but only as
isolated single records, so the model never learned to reject coherent sibling clusters.

For a random share of kept train S1 entities we copy their true S2/S3 records, apply ONE shared house
offset to all copies and an independent sibling word per copy (or none), and add the copies as
distractors (no owner). Output: work/synth_train_s{2,3}.parquet (idx >= SYN_BASE, entity_id S{k}-SYN-...)
and work/synth_map_s{2,3}.parquet (src, rid, parent_s1, offset).
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

from io_utils import CFG, WORK, Timer, load_gt, raw_source_only
from normalize import load_norm
from splits import load_splits

SYN_BASE = 10_000_000
OFFSETS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
# sibling words and relative frequencies, mined from train records that belong to no S1 (reports/phase5.md)
WORDS = {
    "US": {"Group": 6, "Holdings": 6, "East": 1, "West": 1, "Valley": 1, "Riverside": 1, "Uptown": 1, "Midtown": 1,
           "Southside": 1, "Westgate": 1, "Harbor": 1, "North": 1, "Coastal": 1, "Northside": 1, "Summit": 1,
           "Metro": 1, "Downtown": 1, "South": 1, "Greater": 1, "Central": 1, "Eastgate": 1, "Lakeside": 1,
           "Highland": 1},
    "India": {"Industries": 3, "Enterprises": 3, "Public": 3, "Exports": 3, "Overseas": 3, "Ventures": 3,
              "Infratech": 3, "Group": 3, "Holdings": 3, "Solutions": 1, "Technologies": 1, "Trading": 1,
              "Constructions": 1, "Foods": 1, "Engineering": 1, "Hotel": 1, "Garments": 1, "Traders": 1,
              "Steel": 1, "Hardware": 1, "Stores": 1, "Agencies": 1, "Motors": 1, "Furniture": 1},
}
P_NO_WORD = 0.3
P_COPY = 0.85
LEGAL_TAIL = re.compile(
    r"(?i)(\s*,?\s*(?:\(|\[)?\b(?:inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|lp|llp|pc|pllc|plc|"
    r"pvt|private|opc|sarl|sas|sasu|sa|eurl|sci|snc|ei)\b\.?(?:\)|\])?)+\s*$")
SKIP_NAME = re.compile(r"(?i)\|| dba | d/b/a | a/k/a | aka | t/a | trading as | formerly | fka |\.com\b|www\.|^@|"
                       r"[ऀ-෿]")


def insert_word(name: str, word: str) -> str:
    if name.isupper():
        word = word.upper()
    elif name.islower():
        word = word.lower()
    m = LEGAL_TAIL.search(name)
    if m and m.start() > 0:
        return name[:m.start()].rstrip(" ,") + " " + word + name[m.start():]
    return name.rstrip() + " " + word


def shift_house(addr: str, house: str, new: int) -> str | None:
    """Replace the first number token whose zero-stripped value is `house` by `new` (keeping zero padding)."""
    for m in re.finditer(r"\d+", addr):
        tok = m.group(0)
        if tok.lstrip("0") == house:
            rep = str(new).zfill(len(tok)) if tok.startswith("0") else str(new)
            return addr[:m.start()] + rep + addr[m.end():]
    return None


def run(force: bool = False):
    rates = CFG.get("synth", {}).get("rate", {})
    out_p = [WORK / f"synth_train_s{s}.parquet" for s in (2, 3)]
    if all(p.exists() for p in out_p) and not force:
        return
    rng = np.random.default_rng(CFG["seed"] + 7)
    sp_ = load_splits().filter(~pl.col("dropped")).select(["s1", "country"])
    h1 = load_norm("train", 1, ["idx", "a_house"]).rename({"idx": "s1", "a_house": "h1"})
    ent = sp_.join(h1, on="s1").filter(pl.col("h1").str.contains(r"^\d+$"))
    rate = ent["country"].replace_strict(rates, default=0.0, return_dtype=pl.Float64).to_numpy()
    ent = ent.filter(pl.Series(rng.random(ent.height) < rate))
    ent = ent.with_columns(pl.Series("offset", rng.choice(OFFSETS, ent.height)).cast(pl.Int64))
    recs = pl.concat([raw_source_only("train", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                        "name", "addr") for s in (2, 3)])
    cand = load_gt().join(ent, on="s1").join(recs, on=["src", "rid"])
    cand = cand.filter(pl.Series(rng.random(cand.height) < P_COPY))
    rows = []
    for r in cand.iter_rows(named=True):
        name, addr = r["name"], r["addr"]
        if SKIP_NAME.search(name):
            continue
        new_h = int(r["h1"]) + r["offset"]
        if addr.strip():
            a2 = shift_house(addr, r["h1"], new_h)
            if a2 is None:
                continue
        else:
            a2 = addr
        words = WORDS.get(r["country"])
        use_word = bool(words) and rng.random() >= P_NO_WORD
        if not addr.strip() and not use_word:
            continue                      # would be identical to a true name-only record: skip
        if use_word:
            w = rng.choice(list(words), p=np.array(list(words.values())) / sum(words.values()))
            name = insert_word(name, str(w))
        rows.append((r["src"], r["s1"], r["rid"], r["offset"], name, a2, r["country"]))
    df = pl.DataFrame(rows, schema=["src", "parent_s1", "orig_rid", "offset", "name", "addr", "country"], orient="row")
    df = df.with_columns(pl.col("src").cast(pl.Int8), pl.col("parent_s1").cast(pl.Int32))
    for s, p in zip((2, 3), out_p):
        x = df.filter(pl.col("src") == s).with_row_index("k")
        x = x.with_columns((pl.col("k") + SYN_BASE).cast(pl.Int32).alias("idx"),
                           (pl.lit(f"S{s}-SYN-") + pl.col("k").cast(pl.Utf8)).alias("entity_id"))
        x.select(["idx", "entity_id", "name", "addr", "country"]).write_parquet(p)
        x.select([pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"), "parent_s1", "offset"]).write_parquet(
            WORK / f"synth_map_s{s}.parquet")
    print(f"synth: {ent.height:,} sibling entities, {df.height:,} records "
          f"(S2 {df.filter(pl.col('src') == 2).height:,}, S3 {df.filter(pl.col('src') == 3).height:,})")
    print(ent.group_by("country").len().sort("country"))


if __name__ == "__main__":
    import sys
    with Timer("synth"):
        run(force="--force" in sys.argv)
