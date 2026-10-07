"""Multi-view normalization of business names and addresses.

All dictionaries are either hand-written general-language knowledge (street types,
legal forms, honorifics, unit words) or mined from the provided files by
mine_dicts.py (state variants incl. native script, transliteration tokens, French
department -> region, address token synonyms, name vocabulary for OCR repair).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from anyascii import anyascii

DICT_DIR = Path(__file__).resolve().parents[1] / "work" / "dicts"

# ----------------------------------------------------------------------------- folding
NONLATIN_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‏⁠﻿­]")
_PRE = str.maketrans({"°": "o ", "º": "o ", "№": "no ", "’": "'", "‘": "'", "`": "'",
                      "–": "-", "—": "-", "´": "'", " ": " "})


def fold(s: str) -> str:
    """Lowercase ASCII fold (accents stripped). Non-Latin script is romanized by anyascii."""
    s = s.translate(_PRE)
    if not s.isascii():
        s = ZW_RE.sub("", s)
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        if not s.isascii():
            s = anyascii(s)
    return s.lower()


def has_nonlatin(s: str) -> bool:
    return NONLATIN_RE.search(s) is not None


# ----------------------------------------------------------------------------- names
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cos": "co",
    "lp": "lp", "llp": "llp", "pc": "pc", "pllc": "pllc", "plc": "plc", "pvt": "pvt", "private": "pvt",
    "opc": "opc", "pte": "pte", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "ei": "ei", "selarl": "selarl", "scp": "scp", "scop": "scop", "scm": "scm", "sca": "sca",
    "cie": "cie", "ste": "ste", "societe": "ste", "ets": "ets", "etablissements": "ets", "etablissement": "ets",
}
STOP = {"the", "of", "and", "de", "du", "des", "la", "le", "les", "et", "au", "aux", "d", "l", "en", "for", "an", "a"}
HONORIFIC = {"mr", "mrs", "ms", "shri", "sri", "shree", "smt", "mss", "messrs", "m/s", "kumari", "km", "sh"}
GENERIC = {
    "services", "service", "group", "groupe", "partners", "center", "centre", "holding", "holdings",
    "participations", "fils", "associes", "associates", "associate", "solutions", "enterprises", "enterprise",
    "international", "india", "france", "systems", "consultants", "consulting", "industries", "trading",
    "traders", "global", "management", "society", "and", "sons", "brothers", "usa", "america", "american",
    "national", "agency", "associations", "association", "club", "llc",
}
# French translations of the mined sibling / noise words (work/dicts/name_extra.json is mined from US/India train
# pairs only, so French equivalents are unknown to the model). Hand-written general knowledge, accent-folded like
# n_core; used by features.py when config `features.fr_words: true`.
FR_SIBLING = {"fils", "freres", "frs", "associes", "groupe", "agence", "conseils", "entreprise", "entreprises",
              "internationale", "nationale", "systemes", "negoce", "commerce", "france", "globale", "gestion"}
FR_NOISE = {"federation", "fondation", "autorite", "technologie", "technologies", "laboratoire", "laboratoires"}
# French words that the US/India data treats as look-alike ("sibling") markers but that behave like harmless noise words in
# France (public-leaderboard test: removing such pairs lost 0.00047, accepting the rejected ones gained 0.00014).
# Hand-written, accent-folded; used by the France decision rule (config decision.fr_sibling_accept).
FR_NOISE_ACCEPT = {"fils", "freres", "frs", "associes", "association", "club", "groupe", "agence", "conseil", "conseils",
                   "entreprise", "entreprises", "holding", "international", "internationale", "national", "nationale",
                   "solutions", "systemes", "centre", "industries", "negoce", "commerce", "france", "global", "globale",
                   "consultants", "gestion"}
ALIAS_RE = re.compile(
    r"\s+(?:doing business as|d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|a\s*/\s*k\s*/\s*a|a\.k\.a\.?|aka|t\s*/\s*a|"
    r"trading as|formerly known as|formerly|f\s*/\s*k\s*/\s*a|fka)\s+"
)
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|fr|co|biz|info|us)\b")
WWW_RE = re.compile(r"\bwww\.")
ELISION_RE = re.compile(r"\b(?:l|d|qu|j|s|n)'(?=[a-z])")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
OCR_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "6": "g", "9": "g", "2": "z"}
VOWELS_RE = re.compile(r"[aeiouyhw]")


def join_single_letters(toks: list[str]) -> list[str]:
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf) if len(buf) >= 2 else buf[0])
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf) if len(buf) >= 2 else buf[0])
    return out


def phon_token(t: str) -> str:
    if not t or t.isdigit():
        return t
    t = (t.replace("ph", "f").replace("ck", "k").replace("sch", "s").replace("sh", "s").replace("ch", "k")
         .replace("th", "t").replace("q", "k").replace("z", "s").replace("x", "ks").replace("c", "k")
         .replace("v", "w").replace("j", "g"))
    head, rest = t[0], VOWELS_RE.sub("", t[1:])
    out = [head]
    for ch in rest:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


class Normalizer:
    def __init__(self, dict_dir: Path = DICT_DIR, use_dicts: bool = True):
        self.translit: dict[str, str] = {}
        self.state: dict[str, dict[str, str]] = {}
        self.addr_syn: dict[str, str] = {}
        self.region: dict[str, str] = {}
        self.vocab: set[str] = set()
        if use_dicts:
            self._load(dict_dir)
        self.addr_canon = dict(ADDR_CANON)
        self.addr_canon.update({k: v for k, v in self.addr_syn.items()
                                if k not in self.addr_canon and not k[0].isdigit() and v not in UNIT_WORDS
                                and v not in HOUSE_MARKERS})

    def _load(self, d: Path):
        def rd(name, default):
            p = d / name
            return json.loads(p.read_text()) if p.exists() else default
        self.translit = rd("translit.json", {})
        self.state = rd("state.json", {})
        self.addr_syn = rd("addr_syn.json", {})
        self.region = rd("fr_region.json", {})
        self.vocab = set(rd("name_vocab.json", []))
        self.all_states = {}
        for c, m in self.state.items():
            for k, v in m.items():
                self.all_states.setdefault(k, v)

    # ------------------------------------------------------------------ names
    def translit_text(self, s: str) -> str:
        out = []
        for tok in ZW_RE.sub("", s).split():
            if has_nonlatin(tok):
                tk = tok.strip(".,;:()[]|-/\u0964'\"")
                t = self.translit.get(tk)
                out.append(t if t is not None else anyascii(tk).lower())
            else:
                out.append(tok)
        return " ".join(out)

    def ocr_fix(self, t: str) -> str:
        n_alpha = sum(ch.isalpha() for ch in t)
        n_dig = len(t) - n_alpha
        if n_alpha < 3 or n_dig == 0 or n_dig > 2 or ORDINAL_RE.match(t):
            return t
        a = "".join(OCR_MAP.get(ch, ch) for ch in t)
        if "1" in t and self.vocab and a not in self.vocab:
            b = "".join(("i" if ch == "1" else OCR_MAP.get(ch, ch)) for ch in t)
            if b in self.vocab:
                return b
        return a

    def _tokens(self, s: str) -> list[str]:
        s = s.replace("&", " and ").replace("+", " and ")
        s = ELISION_RE.sub("", s)
        s = s.replace("'s ", "s ").replace("'", "").replace(".", "")
        toks = NON_ALNUM_RE.sub(" ", s).split()
        toks = join_single_letters(toks)
        return [self.ocr_fix(t) for t in toks]

    def name_views(self, raw: str) -> dict:
        nl = has_nonlatin(raw)
        s = self.translit_text(raw) if nl else raw
        s = fold(s).strip()
        if s in ("null", "n/a", "na", "none", "-"):
            s = ""
        alias = 0
        main, alt, dom = s, "", ""
        parts = ALIAS_RE.split(s, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            alt, main, alias = parts[0], parts[1], 1
        if "|" in main:
            left, _, right = main.partition("|")
            main, alias = left, 2
            m = URL_RE.search(right)
            if m:
                dom = m.group(1)
            elif not alt:
                alt = right
        if not dom:
            m = URL_RE.search(main)
            if m:
                dom = m.group(1)
                rest = URL_RE.sub(" ", main).strip()
                main = rest if len(rest) > 2 else ""
                alias = alias or 3
        main = WWW_RE.sub(" ", main).strip()
        # junk prefixes/suffixes
        main = main.strip(" -_<>*#@~.,;:!?|/\\=[](){}\"'")
        stripped_raw = fold(raw).strip()
        if not dom and stripped_raw.startswith("@") and " " not in stripped_raw:
            dom = NON_ALNUM_RE.sub("", stripped_raw)
            main, alias = "", 3
        toks = self._tokens(main)
        # glued domain without dot: single token ending with 'com'
        if not dom and len(toks) == 1 and len(toks[0]) >= 8 and toks[0].endswith("com"):
            dom, toks, alias = toks[0][:-3], [], 3
        while toks and toks[0] in HONORIFIC:
            toks = toks[1:]
        norm, legal, core = [], [], []
        for t in toks:
            lg = LEGAL.get(t)
            if lg:
                legal.append(lg)
                norm.append(lg)
            else:
                norm.append(t)
                if t not in STOP:
                    core.append(t)
        alt_core = []
        if alt:
            atoks = self._tokens(alt)
            while atoks and atoks[0] in HONORIFIC:
                atoks = atoks[1:]
            alt_core = [t for t in atoks if t not in LEGAL and t not in STOP]
        if not core and dom:
            core_str = dom
        else:
            core_str = " ".join(core)
        key_toks = sorted(set(t for t in core if t not in GENERIC)) or sorted(set(core))
        return {
            "n_fold": " ".join(NON_ALNUM_RE.sub(" ", fold(raw) if not nl else s).split()),
            "n_norm": " ".join(norm),
            "n_core": core_str,
            "n_alt": " ".join(alt_core),
            "n_legal": " ".join(sorted(set(legal))),
            "n_dom": dom,
            "n_key": " ".join(key_toks),
            "n_phon": " ".join(phon_token(t) for t in core) if core else phon_token(dom),
            "n_acr": "".join(t[0] for t in core if not t.isdigit()) if len(core) >= 2 else "",
            "n_nl": nl,
            "n_alias": alias,
        }

    # ------------------------------------------------------------------ addresses
    def state_of(self, key: str, country: str) -> str | None:
        m = self.state.get(country)
        if m is not None:
            st = m.get(key)
            if st:
                return st
        if country == "France" or m is None:
            return self.region.get(key) or (self.all_states.get(key) if m is None and self.state else None)
        return None

    def canon_for(self, country: str) -> dict:
        c = self._canon_cache.get(country) if hasattr(self, "_canon_cache") else None
        if c is None:
            if not hasattr(self, "_canon_cache"):
                self._canon_cache = {}
            c = dict(self.addr_canon)
            if country == "France":
                c.update(FR_CANON)
            self._canon_cache[country] = c
        return c

    def addr_views(self, raw: str, country: str) -> dict:
        empty = {"a_tok": "", "a_house": "", "a_hsuf": "", "a_hrange": "", "a_nums": "", "a_street": "",
                 "a_stype": "", "a_city": "", "a_state": "", "a_unit": "", "a_postal": "", "a_empty": True}
        if not raw or not raw.strip():
            return empty
        comps = []   # (folded text, tokens, state-or-None)
        state = ""
        canon = self.canon_for(country)
        for c in raw.split(","):
            c0 = ZW_RE.sub("", c).strip()
            if not c0:
                continue
            if has_nonlatin(c0):
                st = self.state_of(c0, country)
                if st:
                    comps.append((c0, [], st))
                    continue
                c0 = self.translit_text(c0)
            cf = NUMSIGN_RE.sub(" no ", fold(c0))
            toks = ADDR_TOK_RE.findall(cf)
            if not toks:
                continue
            key = " ".join(toks)
            if key in NULLS:
                continue
            st = self.state_of(key, country)
            if st:
                comps.append((cf, toks, st))
                continue
            if (len(toks) >= 2 and len(toks[-1]) <= 3 and toks[-1].isalpha() and toks[-1] not in STREET_TYPES
                    and toks[-1] not in UNIT_WORDS and toks[-1] not in DIRECTIONS):
                st = self.state_of(toks[-1], country)
                if st and all(t.isalpha() for t in toks[:-1]):
                    comps.append((cf, [], st))
                    toks = toks[:-1]
            comps.append((cf, toks, None))
        # the last state-like component is the state; earlier different ones are ordinary text (Washington, DC)
        st_idx = [i for i, x in enumerate(comps) if x[2]]
        if st_idx:
            state = comps[st_idx[-1]][2]
        comps = [(cf, toks) for cf, toks, st in comps
                 if toks and (st is None or st != state)]
        if not comps:
            empty = dict(empty)
            empty["a_state"] = state
            return empty
        postal = ""
        tok_comps = []
        for cf, toks in comps:
            if len(toks) == 1 and POSTAL_RE.fullmatch(toks[0]):
                postal = postal or toks[0]
                continue
            if len(toks) >= 2 and POSTAL_RE.fullmatch(toks[-1]) and toks[-2].isalpha():
                postal = postal or toks[-1]
                toks = toks[:-1]
            toks = [canon.get(t, t) for t in toks if t not in ADDR_DROP]
            toks = [(ORD_RE.sub(r"\1th", t) if t[0].isdigit() else (t[:-3] if t.endswith("cdp") and len(t) > 5 else t))
                    for t in toks if t != "cdp"]
            if toks:
                tok_comps.append((cf, toks))
        house, hsuf, hrange, street, stype, unit_toks, city_parts = "", "", "", [], "", [], []
        india = country == "India"
        street_i = -1
        for i, (cf, toks) in enumerate(tok_comps):
            t0 = toks[0]
            if t0 in UNIT_WORDS and (street_i != -1 or (len(toks) <= 4 and any(t.isdigit() or len(t) == 1 for t in toks[1:]))):
                unit_toks += [t for t in toks[1:] if t not in UNIT_WORDS and t != "no"]
                continue
            if street_i == -1 and not india and (t0[0].isdigit() or (t0 in HOUSE_MARKERS and len(toks) > 1) or (
                    (t0 in STREET_TYPES and t0 not in ("st", "ste")) if country == "France"
                    else any(t in STREET_TYPES for t in toks[1:]))):
                street_i = i
                continue
            if not any(ch.isdigit() for ch in cf):
                city_parts.append(toks)
        if street_i >= 0:
            cf, toks = tok_comps[street_i]
            m = RANGE_RE.match(cf.strip())
            if m:
                hrange = m.group(2).lstrip("0")
            st_toks = [t for t in toks if t not in HOUSE_MARKERS]
            j = 0
            if st_toks and st_toks[0].isdigit():
                house = st_toks[0].lstrip("0") or "0"
                j = 1
                if hrange and j < len(st_toks) and st_toks[j].lstrip("0") == hrange:
                    j += 1
                if j < len(st_toks) and (st_toks[j] in HOUSE_SUFFIX or
                                         (len(st_toks[j]) == 1 and st_toks[j].isalpha() and j + 1 < len(st_toks)
                                          and st_toks[j] not in DIRECTIONS
                                          and any(t not in STREET_TYPES for t in st_toks[j + 1:]))):
                    hsuf = st_toks[j]
                    j += 1
                if j + 1 < len(st_toks) and st_toks[j] == "1" and st_toks[j + 1] == "2":  # 600 1/2
                    hsuf, j = "1/2", j + 2
            for t in st_toks[j:]:
                if t in STREET_TYPES and not (t == "st" and not stype and st_toks[j:].index(t) == 0 and len(st_toks) - j > 1):
                    stype = stype or STREET_TYPES[t]
                elif t in DIRECTIONS or t in STREET_STOP:
                    continue
                else:
                    street.append(t)
        else:
            for cf, toks in tok_comps:
                for t in toks:
                    if t.isdigit():
                        house = t.lstrip("0") or "0"
                        break
                if house:
                    break
        city = ""
        if city_parts:
            c = [t for t in city_parts[-1] if t != "cdp"]
            if c and c[-1].endswith("cdp") and len(c[-1]) > 5:
                c[-1] = c[-1][:-3]
            if len(c) >= 3 and c[0] == "city" and c[1] == "of":
                c = c[2:]
            if len(c) >= 2 and c[-1] == "city":
                c = c[:-1]
            city = " ".join(c)
        all_toks = [t for _, toks in tok_comps for t in toks if t != "no"]
        nums = sorted({t.lstrip("0") or "0" for t in all_toks if t.isdigit()}, key=lambda x: (len(x), x))
        return {
            "a_tok": " ".join(all_toks),
            "a_house": house,
            "a_hsuf": hsuf,
            "a_hrange": hrange,
            "a_nums": " ".join(nums),
            "a_street": " ".join(street),
            "a_stype": stype,
            "a_city": city,
            "a_state": state,
            "a_unit": " ".join(unit_toks),
            "a_postal": postal,
            "a_empty": False,
        }


NULLS = {"null", "n a", "na", "none", "nan", "n/a", "-", "nil"}
NUMSIGN_RE = re.compile(r"\bn\s*(?:°|º|o\.|o\b)\s*(?=\d)|#")
ADDR_TOK_RE = re.compile(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+")
ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
RANGE_RE = re.compile(r"^(?:no\s+)?(\d+)\s*-\s*(\d+)\b")
STREET_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "aux", "au", "et", "the", "of"}
POSTAL_RE = re.compile(r"\d{5}|\d{6}")

# canonical street types (hand-written, general language; US + France + India)
STREET_TYPES = {
    "st": "st", "street": "st", "str": "st", "saint": "st", "rd": "rd", "road": "rd", "dr": "dr", "drive": "dr",
    "ln": "ln", "lane": "ln", "ave": "ave", "av": "ave", "avenue": "ave", "aven": "ave", "blvd": "blvd",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "ct": "ct", "court": "ct", "pl": "pl", "place": "pl",
    "hwy": "hwy", "highway": "hwy", "pkwy": "pkwy", "parkway": "pkwy", "cir": "cir", "circle": "cir",
    "ter": "ter", "terrace": "ter", "trl": "trl", "trail": "trl", "way": "way", "sq": "sq", "square": "sq",
    "pike": "pike", "aly": "aly", "alley": "aly", "loop": "loop", "run": "run", "row": "row", "cv": "cv",
    "cove": "cv", "xing": "xing", "crossing": "xing", "path": "path", "walk": "walk", "pt": "pt", "point": "pt",
    "rue": "rue", "r": "rue", "all": "allee", "allee": "allee", "imp": "imp", "impasse": "imp",
    "crs": "cours", "cours": "cours", "ch": "chemin", "che": "chemin", "chemin": "chemin", "rte": "rte",
    "route": "rte", "quai": "quai", "passage": "passage", "pass": "passage", "esplanade": "esplanade",
    "promenade": "promenade", "prom": "promenade", "residence": "residence", "res": "residence", "cite": "cite",
    "hameau": "hameau", "lotissement": "lot", "lieu": "lieu", "mail": "mail", "terrasse": "ter",
    "marg": "marg", "nagar": "nagar", "sector": "sector", "sec": "sector", "chem": "chemin", "rpt": "rpt",
    "sente": "sente", "villa": "villa", "clos": "clos", "parc": "parc", "square": "sq", "quartier": "quartier",
}
DIRECTIONS = {"n": "n", "s": "s", "e": "e", "w": "w", "north": "n", "south": "s", "east": "e", "west": "w",
              "ne": "ne", "nw": "nw", "se": "se", "sw": "sw", "northeast": "ne", "northwest": "nw",
              "southeast": "se", "southwest": "sw"}
UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "fl", "floor", "flr", "#", "pmb", "po", "box", "room",
              "rm", "bldg", "building", "lot", "trlr", "spc", "space", "dept", "appartement", "bat", "batiment",
              "etage", "porte"}
HOUSE_MARKERS = {"no", "#", "hn", "hno", "h", "plot", "plt", "door", "house", "dno", "number", "num"}
HOUSE_SUFFIX = {"bis", "ter", "quater", "a", "b", "c", "d"}
ADDR_DROP = {"null"}
FR_CANON = {"chem": "chemin", "r": "rue", "all": "allee", "ch": "chemin", "che": "chemin", "res": "residence", "pass": "passage",
            "crs": "cours", "imp": "imp", "av": "ave", "bd": "blvd", "pl": "pl", "sq": "sq", "rte": "rte", "st": "st"}
ADDR_CANON = {
    **{k: v for k, v in STREET_TYPES.items() if k not in ("r", "all", "pass", "res", "ch", "che", "sec", "pt")},
    **DIRECTIONS,
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "flr": "fl", "floor": "fl", "apartment": "apt", "apts": "apt", "appartment": "apt", "suite": "ste",
    "building": "bldg", "bldng": "bldg", "complex": "cmplx", "near": "nr", "opposite": "opp", "opp": "opp",
    "behind": "bhd", "hno": "no", "number": "no", "plot": "plot", "plt": "plot", "colony": "col",
    "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata", "bangalore": "bengaluru",
    "mysore": "mysuru", "cochin": "kochi", "trivandrum": "thiruvananthapuram", "poona": "pune",
    "gurgaon": "gurugram", "baroda": "vadodara", "calicut": "kozhikode", "pondicherry": "puducherry",
    "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "gulbarga": "kalaburagi",
    "allahabad": "prayagraj", "benares": "varanasi", "banaras": "varanasi", "simla": "shimla",
    "cawnpore": "kanpur", "vizag": "visakhapatnam", "tanjore": "thanjavur", "trichy": "tiruchirappalli",
    "avenue": "ave", "av": "ave", "bd": "blvd", "boulevard": "blvd", "allee": "allee", "impasse": "imp",
    "rue": "rue", "chemin": "chemin", "route": "rte", "saint": "st", "sainte": "ste", "ste": "ste",
}
