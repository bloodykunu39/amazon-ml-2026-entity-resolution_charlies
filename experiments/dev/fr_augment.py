"""French-style copies of US training pairs for the transformer judge (training data only, labels unchanged).

France is never in train, so the cross-encoder never sees French business words. This module rewrites US training
pairs with a hand-written English -> French dictionary (general language knowledge): name words (sibling words such
as Sons -> Fils, Group -> Groupe; noise words such as Foundation -> Fondation), legal forms (LLC -> SARL), "&"/"and"
-> "et", street types with French word order ("1445 River Cross Court" -> "1445 Cour River Cross") and US states ->
French departments/regions. The same mapping is applied to both sides of a pair, so the label stays valid.

  python src/fr_augment.py IN_PAIRS.parquet OUT_PAIRS.parquet [N_COPIES]
IN_PAIRS: s1, src, rid, y, ta, tb (from `crossencoder.py prep`). OUT = IN + N French copies of US pairs (shuffled).
"""
from __future__ import annotations

import re
import sys
import zlib

import polars as pl

SEED = 2026

NAME = {  # one-to-one translations only (ambiguous words stay English: an English noise word and an English sibling
          # word must never map onto the same French word, e.g. partners/associate vs associates)
    "sons": "Fils", "brothers": "Frères", "bros": "Frs", "associates": "Associés", "group": "Groupe",
    "agency": "Agence", "enterprises": "Entreprises", "enterprise": "Entreprise", "systems": "Systèmes",
    "trading": "Négoce", "traders": "Négociants", "management": "Gestion", "holdings": "Holding",
    "foundation": "Fondation", "federation": "Fédération", "authority": "Autorité", "technology": "Technologie",
    "labs": "Laboratoires", "council": "Comité", "board": "Bureau", "american": "Française",
    "society": "Société", "school": "École", "college": "Collège", "church": "Église", "friends": "Amis",
    "development": "Développement", "institute": "Institut", "academy": "Académie", "health": "Santé",
    "pharmacy": "Pharmacie", "clinic": "Clinique", "dental": "Dentaire", "medical": "Médical", "family": "Famille",
}
LEGAL = {  # legal forms (checked before NAME)
    "llc": "SARL", "l.l.c.": "S.A.R.L.", "l.l.c": "S.A.R.L", "inc": "SAS", "inc.": "S.A.S.", "incorporated": "SAS",
    "corp": "SA", "corp.": "S.A.", "corporation": "SA", "co": "Cie", "co.": "Cie", "company": "Compagnie",
    "ltd": "SARL", "ltd.": "SARL", "limited": "SARL", "lp": "SCS", "llp": "SELARL", "pllc": "SELARL", "pc": "SCP",
    "the": "", "and": "et", "&": "&",
}
STREET = {  # street type -> French (placed before the street name)
    "street": "Rue", "st": "R.", "st.": "R.", "avenue": "Avenue", "ave": "Av.", "ave.": "Av.", "av": "Av",
    "boulevard": "Boulevard", "blvd": "Bd", "road": "Route", "rd": "Rte", "drive": "Allée", "dr": "All.",
    "lane": "Chemin", "ln": "Ch.", "court": "Cour", "ct": "Cour", "place": "Place", "pl": "Pl.", "square": "Square",
    "way": "Voie", "parkway": "Promenade", "pkwy": "Prom.", "highway": "Route", "hwy": "Rte", "circle": "Rond-Point",
    "terrace": "Terrasse", "trail": "Sentier", "alley": "Passage", "loop": "Boucle",
}
UNIT = {"unit": "Appt", "apt": "Appt", "suite": "Bureau", "ste": "Bureau", "floor": "Étage", "fl": "Étage"}
STATES = {  # code: name
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
FR_AREAS = [("Hauts-de-France", "Nord"), ("Nouvelle-Aquitaine", "Gironde"), ("Pays de la Loire", "Loire-Atlantique"),
            ("Hauts-de-France", "Pas-de-Calais")]
STATE_OF = {**{c.lower(): c for c in STATES}, **{n.lower(): c for c, n in STATES.items()}}


def _case(src: str, fr: str) -> str:
    if src.isupper() and len(src) > 1:
        return fr.upper()
    if src.islower():
        return fr.lower()
    return fr


def _area(code: str, full: bool) -> str:
    region, dept = FR_AREAS[zlib.crc32(code.encode()) % len(FR_AREAS)]
    return dept if full else region          # full state name -> department, code -> its region (fr_region.json links them)


def fr_name(name: str) -> str:
    out = []
    for tok in re.findall(r"\[|\]|\(|\)|[^\s\[\]()]+", name):
        low = tok.lower().strip(",")
        tail = "," if tok.endswith(",") and len(tok) > 1 else ""
        if low in LEGAL:
            w = LEGAL[low]
            if w:
                out.append(_case(tok, w) + tail)
        elif low in NAME:
            out.append(_case(tok, NAME[low]) + tail)
        else:
            out.append(tok)
    return re.sub(r"\s+", " ", " ".join(out)).replace("( ", "(").replace(" )", ")").replace("[ ", "[").replace(" ]", "]").strip()


_STREET_RE = re.compile(r"^(\d+[A-Za-z]?)\s+(.+?)\s+(" + "|".join(re.escape(k) for k in sorted(STREET, key=len, reverse=True))
                        + r")\.?$", re.IGNORECASE)


def fr_addr(addr: str) -> str:
    parts = []
    for comp in [c.strip() for c in addr.split(",")]:
        low = comp.lower()
        m = _STREET_RE.match(comp)
        if m:
            num, body, st = m.groups()
            key = st.lower() if st.lower() in STREET else st.lower().rstrip(".")
            parts.append(f"{num} {_case(st, STREET[key])} {body}")
        elif low in STATE_OF:
            code = STATE_OF[low]
            parts.append(_case(comp, _area(code, full=len(low) > 2)) if len(low) > 2 else _area(code, full=False))
        else:
            toks = comp.split()
            if toks and toks[0].lower().rstrip(".") in UNIT:
                toks[0] = _case(toks[0], UNIT[toks[0].lower().rstrip(".")])
            parts.append(" ".join(toks))
    return ", ".join(p for p in parts if p)


def fr_text(t: str) -> str:
    """Pair text is 'name | address' (crossencoder.attach_text)."""
    name, _, addr = t.partition(" | ")
    return f"{fr_name(name)} | {fr_addr(addr)}"


def augment(pairs: pl.DataFrame, country: pl.DataFrame, n: int) -> pl.DataFrame:
    """n French copies of US pairs: pairs with a translatable name word first, then random US pairs."""
    us = pairs.join(country.filter(pl.col("country") == "US").select("s1"), on="s1", how="semi")
    words = "|".join(sorted(NAME))
    has = pl.col("tb").str.to_lowercase().str.contains(rf"\b({words})\b") | pl.col("ta").str.to_lowercase().str.contains(rf"\b({words})\b")
    a = us.filter(has).sample(fraction=1.0, shuffle=True, seed=SEED).head(n // 2)
    b = us.join(a.select(["s1", "src", "rid"]), on=["s1", "src", "rid"], how="anti").sample(n=n - a.height, seed=SEED)
    fr = pl.concat([a, b]).with_columns(pl.col("ta").map_elements(fr_text, return_dtype=pl.String),
                                        pl.col("tb").map_elements(fr_text, return_dtype=pl.String))
    return fr


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from splits import load_splits
    pairs = pl.read_parquet(src)
    fr = augment(pairs, load_splits().select(["s1", "country"]), n)
    out = pl.concat([pairs, fr.select(pairs.columns)]).sample(fraction=1.0, shuffle=True, seed=SEED)
    out.write_parquet(dst)
    print(f"pairs {pairs.height:,} + French copies {fr.height:,} (pos rate {fr['y'].mean():.3f}) -> {out.height:,}")
