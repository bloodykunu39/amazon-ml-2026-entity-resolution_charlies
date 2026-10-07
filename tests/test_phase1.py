import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from io_utils import load_gt, load_source  # noqa: E402
from metrics import f05, macro_f05  # noqa: E402
from splits import REC_SAMPLE, SPLITS, distractor_share, load_splits  # noqa: E402
from validate_outputs import CAND_HEADER, MATCH_HEADER, check_file, check_submission  # noqa: E402


def P(rows):
    return pl.DataFrame(rows, schema={"s1": pl.Int32, "src": pl.Int8, "rid": pl.Int32}, orient="row")


U = pl.DataFrame({"s1": [1]}, schema={"s1": pl.Int32})


def test_metric_example():
    pred = P([(1, 2, 1), (1, 2, 2), (1, 3, 3)])
    truth = P([(1, 2, 1), (1, 3, 3)])
    assert abs(macro_f05(pred, truth, U) - 0.7143) < 1e-3
    assert abs(f05(2, 2, 3) - 0.7143) < 1e-3


def test_metric_edge_cases():
    empty = P([])
    assert macro_f05(empty, empty, U) == 1.0
    assert macro_f05(P([(1, 2, 5)]), empty, U) == 0.0
    assert macro_f05(empty, P([(1, 2, 5)]), U) == 0.0
    t = P([(1, 2, 1), (1, 3, 3)])
    assert macro_f05(t, t, U) == 1.0
    # predictions for S1 outside the universe are ignored; singleton in universe counts
    u2 = pl.DataFrame({"s1": [1, 2]}, schema={"s1": pl.Int32})
    assert macro_f05(t, t, u2) == 1.0
    assert macro_f05(P([(1, 2, 1), (2, 2, 9)]), P([(1, 2, 1)]), u2) == 0.5


@pytest.fixture
def s1file(tmp_path):
    p = tmp_path / "s1.tsv"
    p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\ta\tb\tUS\nS1-2\ta\tb\tUS\n")
    return p


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_validator_detects(tmp_path, s1file):
    req = ["S1-1", "S1-2"]
    ok = _w(tmp_path, "ok.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\t\n")
    assert check_file(ok, MATCH_HEADER, req)[0] == []
    bad_header = _w(tmp_path, "h.tsv", "s1\tmatched\nS1-1\t\nS1-2\t\n")
    assert check_file(bad_header, MATCH_HEADER, req)[0]
    missing = _w(tmp_path, "m.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\t\n")
    assert any("missing" in e for e in check_file(missing, MATCH_HEADER, req)[0])
    dup_row = _w(tmp_path, "d.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-1\t\nS1-2\t\n")
    assert any("duplicate row" in e for e in check_file(dup_row, MATCH_HEADER, req)[0])
    dup_id = _w(tmp_path, "di.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\nS1-2\t\n")
    assert any("duplicate ids" in e for e in check_file(dup_id, MATCH_HEADER, req)[0])
    s1_in = _w(tmp_path, "s.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS1-2\nS1-2\t\n")
    assert any("S1 id" in e for e in check_file(s1_in, MATCH_HEADER, req)[0])
    cand = _w(tmp_path, "c.tsv", "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
    assert check_file(cand, CAND_HEADER, req)[0] == []
    errs = check_submission(ok, cand, s1file)
    assert any("not in candidates" in e for e in errs)


def test_gt_facts():
    gt = load_gt()
    assert gt.height == 7_638_365
    assert gt.select(pl.struct("src", "rid").n_unique()).item() == gt.height  # partition
    n_s1 = load_source("train", 1).height
    singles = n_s1 - gt["s1"].n_unique()
    assert abs(singles / n_s1 - 0.0558) < 0.001


def test_splits():
    assert SPLITS.exists() and REC_SAMPLE.exists()
    sp = load_splits()
    assert sp.height == 2_206_821
    assert abs(sp["dropped"].mean() - 0.21) < 0.003
    kept = sp.filter(~pl.col("dropped"))
    assert abs(kept["holdout"].mean() - 0.20) < 0.003
    assert set(kept.filter(~pl.col("holdout"))["fold"].unique().to_list()) == {0, 1, 2, 3, 4}
    assert (sp.filter(pl.col("dropped") | pl.col("holdout"))["fold"] == -1).all()
    assert sp["s1"].n_unique() == sp.height  # one fold per S1 => grouped
    assert 0.41 <= distractor_share(False) <= 0.44
