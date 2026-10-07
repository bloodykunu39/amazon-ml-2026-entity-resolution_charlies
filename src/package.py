"""Build the final submission zip.

  python src/package.py --team TEAM [--outputs output]

Layout (as required by student_resource/README.md):
  <TEAM>_submission.zip
  ├── output/{matching_results.tsv, candidate_pairs.tsv}
  ├── code/business_entity_resolution/{src/, tests/, config.yaml, README.md, requirements.txt}
  └── Documentation_template.md
The chosen output files are validated first (local checks + official validator); packaging stops on failure.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_outputs import check_submission, run_official  # noqa: E402

CODE_FILES = ["config.yaml", "README.md", "requirements.txt"]
CODE_DIRS = {"src": (".py", ".sh"), "tests": (".py",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", default="output", help="folder holding the two final TSVs")
    a = ap.parse_args()
    out_dir = ROOT / a.outputs
    m, c = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    test_dir = ROOT / "student_resource/dataset/test"
    errs = check_submission(m, c, test_dir / "test_source1.tsv")
    ok, log = run_official(m, c, test_dir)
    print(log)
    if errs or not ok:
        print("VALIDATION FAILED — not packaging:", errs)
        sys.exit(1)
    doc = ROOT / "Documentation_template.md"
    if any(t in doc.read_text() for t in ("[FILLED AFTER FULL RUN]", "[TEST NUMBERS]", "[FINAL", "[TEAM]")):
        print("WARNING: Documentation_template.md still has placeholders")
    zpath = ROOT / f"{a.team}_submission.zip"
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(m, "output/matching_results.tsv")
        z.write(c, "output/candidate_pairs.tsv")
        for f in CODE_FILES:
            z.write(ROOT / f, f"{base}/{f}")
        for d, exts in CODE_DIRS.items():
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and p.suffix in exts and "__pycache__" not in p.parts:
                    z.write(p, f"{base}/{d}/{p.relative_to(ROOT / d)}")
        z.write(doc, "Documentation_template.md")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    for n in names:
        if not n.startswith(f"{base}/src/") and not n.startswith(f"{base}/tests/"):
            print("  ", n)
    print(f"   {base}/src/: {sum(n.startswith(base + '/src/') for n in names)} files, "
          f"{base}/tests/: {sum(n.startswith(base + '/tests/') for n in names)} files")


if __name__ == "__main__":
    main()
