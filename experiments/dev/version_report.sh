#!/bin/bash
# Label-free + holdout report for a finished version:  bash src/version_report.sh TAG TRAIN_PRED TEST_PRED
# -> reports/TAG_check.md
cd "$(dirname "$0")/.."
T=$1; TR=$2; TE=$3
PY=/home/moskov/miniconda3/envs/amazon/bin/python
R=reports/${T}_check.md
{
echo "# $T check ($(date '+%Y-%m-%d %H:%M'))"
echo; echo "## Official validator"; echo '```'
$PY student_resource/utils/validate_submission.py --matching output/variants/$T/matching_results.tsv \
   --candidate output/variants/$T/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | tail -4
echo '```'
echo; echo "## Holdout (tune log)"; echo '```'
grep -E "best on OOF" logs/${T}_tune.log 2>/dev/null; sed -n '/macro F0.5/,/errors/p' logs/${T}_tune.log 2>/dev/null
echo '```'
echo; echo "## Symptom check: near-certain pairs accepted, by trailing word (holdout vs test)"; echo '```'
PYTHONPATH=src $PY src/check_symptom.py $TR $TE 2>&1 | grep -v Warning
echo '```'
echo; echo "## Test diagnostics: v1 vs $T"; echo '```'
PYTHONPATH=src $PY src/test_diag.py work/pred_l2_test_v1.parquet $TE 2>&1 | grep -v Warning
echo '```'
echo; echo "## Submission diff v1 -> $T (per country)"; echo '```'
PYTHONPATH=src $PY src/compare_outputs.py output/v1/matching_results.tsv output/variants/$T/matching_results.tsv 2>&1 | grep -vE "Warning|read_csv"
echo '```'
} > $R 2>&1
echo "wrote $R"
