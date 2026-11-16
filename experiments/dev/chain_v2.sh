#!/bin/bash
# Iteration 2: larger blocking top-k; full rerun; writes output/variants/v2.
set -e
cd "$(dirname "$0")"
PY=/home/moskov/miniconda3/envs/amazon/bin/python
RUN="systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=4G $PY"
L=../logs
$RUN blocking.py train --force > $L/v2_block_train.log 2>&1; echo "$(date) block train"
$RUN blocking.py test --force > $L/v2_block_test.log 2>&1; echo "$(date) block test"
$RUN prune.py train --train > $L/v2_prune_train.log 2>&1; echo "$(date) prune train"
$RUN prune.py test > $L/v2_prune_test.log 2>&1; echo "$(date) prune test"
$RUN features.py train --force > $L/v2_feats_train.log 2>&1; echo "$(date) feats train"
$RUN features.py test --force > $L/v2_feats_test.log 2>&1; echo "$(date) feats test"
$RUN model.py train > $L/v2_model_train.log 2>&1; echo "$(date) model train"
$RUN submit.py tune > $L/v2_tune.log 2>&1; echo "$(date) tune"
$RUN model.py test > $L/v2_model_test.log 2>&1; echo "$(date) predict test"
$RUN submit.py write --variant v2 > $L/v2_write.log 2>&1; echo "$(date) write v2"
