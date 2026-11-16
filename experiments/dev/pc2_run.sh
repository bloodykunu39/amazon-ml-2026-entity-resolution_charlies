#!/bin/bash
# PC2: clean end-to-end run from raw data (reproducibility check of the final code).
#   mkdir -p ~/Documents/amazon && tar -xf pc2_bundle.tar -C ~/Documents/amazon && cd ~/Documents/amazon
#   bash src/pc2_run.sh
# Needs conda (miniconda). Creates env `amazon` (python 3.11) if missing. ~4-6 h on a 16-core / 64 GB machine.
set -e
cd "$(dirname "$0")/.."
if ! conda env list | grep -q "^amazon "; then conda create -n amazon -y python=3.11 pip; fi
PY="$(conda env list | awk '$1=="amazon"{print $NF}')/bin/python"
$PY -m pip install -q -r requirements.txt
mkdir -p logs work/tmp output
$PY -m pytest -q tests/ 2>&1 | tail -2 || true
if command -v systemd-run >/dev/null; then RUN="systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=4G"; else RUN=""; fi
( time $RUN $PY src/run.py --force ) > logs/pc2_run.log 2>&1
$PY student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids | tail -3
md5sum output/matching_results.tsv output/candidate_pairs.tsv
echo "Done. Send back output/matching_results.tsv (and logs/pc2_run.log, reports/run_report.json) for comparison."
