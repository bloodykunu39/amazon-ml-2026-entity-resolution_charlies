"""Local submission checks + official validator.

check_file() returns a list of error strings (empty = OK). It checks header, one row
per required S1 (in order), no duplicate rows, no duplicate ids in a list, S2/S3
prefixes only, and (with a candidate file) matches ⊆ candidates.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_required(test_s1_path) -> list[str]:
    with open(test_s1_path, encoding="utf-8") as f:
        next(f)
        return [line.split("\t", 1)[0] for line in f if line.strip()]


def check_file(path, header, required: list[str], check_order: bool = True):
    errors = []
    mapping = {}
    with open(path, encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        if h != header:
            return [f"header mismatch: {h} != {header}"], None
        order = []
        for ln, line in enumerate(f, start=2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errors.append(f"line {ln}: no tab")
                continue
            if s1 in mapping:
                errors.append(f"duplicate row {s1}")
                continue
            ids = rest.split(",") if rest else []
            if len(ids) != len(set(ids)):
                errors.append(f"duplicate ids in list for {s1}")
            for i in ids:
                if i.startswith("S1-"):
                    errors.append(f"S1 id in list for {s1}: {i}")
                elif not (i.startswith("S2-") or i.startswith("S3-")):
                    errors.append(f"bad prefix in list for {s1}: {i!r}")
                if " " in i or '"' in i:
                    errors.append(f"space/quote in id for {s1}: {i!r}")
            mapping[s1] = set(ids)
            order.append(s1)
    req = set(required)
    missing = req - set(mapping)
    extra = set(mapping) - req
    if missing:
        errors.append(f"missing S1 rows: {len(missing)} e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"unknown S1 rows: {len(extra)} e.g. {sorted(extra)[:3]}")
    if check_order and not missing and not extra and order != list(required):
        errors.append("rows not in test_source1 order")
    return errors, mapping


def check_submission(matching, candidate, test_s1_path, max_err=20):
    required = read_required(test_s1_path)
    errs, m = check_file(matching, MATCH_HEADER, required)
    errs = [f"matching: {e}" for e in errs]
    if candidate is not None:
        e2, c = check_file(candidate, CAND_HEADER, required)
        errs += [f"candidate: {e}" for e in e2]
        if m is not None and c is not None:
            bad = [s for s, ids in m.items() if ids - c.get(s, set())]
            if bad:
                errs.append(f"matches not in candidates for {len(bad)} S1, e.g. {bad[:3]}")
    return errs[:max_err] if len(errs) > max_err else errs


def run_official(matching, candidate, test_dir) -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"),
           "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=str(ROOT / "output/matching_results.tsv"))
    ap.add_argument("--candidate", default=str(ROOT / "output/candidate_pairs.tsv"))
    ap.add_argument("--test-dir", default=str(ROOT / "student_resource/dataset/test"))
    a = ap.parse_args()
    errs = check_submission(a.matching, a.candidate, Path(a.test_dir) / "test_source1.tsv")
    print("local checks:", "PASS" if not errs else "FAIL")
    for e in errs:
        print("  ", e)
    ok, out = run_official(a.matching, a.candidate, a.test_dir)
    print(out)
    sys.exit(0 if (ok and not errs) else 1)
