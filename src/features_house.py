"""House-number relations (scalar reference implementation, used by tests and features)."""
from __future__ import annotations


def _lev1(a: str, b: str) -> bool:
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False


def house_rel(h1, s1, r1, h2, s2, r2) -> dict:
    """h: zero-stripped house digits, s: letter suffix, r: range end."""
    out = {"missing": 0, "equal": 0, "equal_base": 0, "suffix": 0, "range": 0, "lev1": 0, "diff": -1}
    if not h1 or not h2:
        out["missing"] = 1
        return out
    if h1 == h2:
        if r1 or r2:
            out["range"] = 1
        out["equal_base"] = 1
        if (s1 or "") == (s2 or ""):
            out["equal"] = 1
        else:
            out["equal"] = 1 if not s1 or not s2 else 0
        out["diff"] = 0
        return out
    try:
        a, b = int(h1), int(h2)
        out["diff"] = abs(a - b)
        for lo, hi, x in ((a, int(r1) if r1 else None, b), (b, int(r2) if r2 else None, a)):
            if hi is not None and lo <= x <= hi:
                out["range"] = 1
    except ValueError:
        pass
    if h1.endswith(h2) or h2.endswith(h1) or h1.startswith(h2) or h2.startswith(h1):
        out["suffix"] = 1
    if _lev1(h1, h2):
        out["lev1"] = 1
    return out
