#!/bin/bash
# Full run after blocking: train side as soon as train blocking is done, test side after test blocking.
set -e
cd "$(dirname "$0")"
PY=/home/moskov/miniconda3/envs/amazon/bin/python
RUN="systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=4G $PY"
L=../logs
until [ -f ../work/cand_raw_train_US.parquet ] && ! pgrep -f "^$PY blocking.py train" >/dev/null; do sleep 20; done
echo "$(date) train blocking done"
$RUN prune.py train --train > $L/prune_train_full.log 2>&1
echo "$(date) prune train done"
$RUN features.py train > $L/feats_train_full.log 2>&1
echo "$(date) feats train done"
$RUN model.py train > $L/model_train_full.log 2>&1
echo "$(date) model train done"
$RUN submit.py tune > $L/tune_full.log 2>&1
echo "$(date) tune done"
until [ -f ../work/cand_raw_test_US.parquet ] && ! pgrep -f "^$PY blocking.py test" >/dev/null; do sleep 20; done
echo "$(date) test blocking done"
$RUN prune.py test > $L/prune_test_full.log 2>&1
$RUN features.py test > $L/feats_test_full.log 2>&1
$RUN model.py test > $L/model_test_full.log 2>&1
$RUN submit.py write > $L/write_full.log 2>&1
echo "$(date) write done"
