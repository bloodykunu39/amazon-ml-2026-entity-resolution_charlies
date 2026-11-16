#!/bin/bash
# v4: synthetic sibling businesses + test-like universe (no records of dropped S1) + capped rarity features.
set -e
cd "$(dirname "$0")"
PY=/home/moskov/miniconda3/envs/amazon/bin/python
RUN="systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=4G $PY"
L=../logs
$RUN normalize.py --force > $L/v4_norm.log 2>&1; echo "$(date) norm"
$RUN blocking.py train --force > $L/v4_block_train.log 2>&1; echo "$(date) block train"
$RUN prune.py train --train > $L/v4_prune_train.log 2>&1; echo "$(date) prune train"
$RUN prune.py test > $L/v4_prune_test.log 2>&1; echo "$(date) prune test"
$RUN features.py train --force > $L/v4_feats_train.log 2>&1; echo "$(date) feats train"
$RUN features.py test --force > $L/v4_feats_test.log 2>&1; echo "$(date) feats test"
$RUN model.py train > $L/v4_model_train.log 2>&1; echo "$(date) model train"
$RUN submit.py tune > $L/v4_tune.log 2>&1; echo "$(date) tune"
$RUN model.py test > $L/v4_model_test.log 2>&1; echo "$(date) predict test"
$RUN submit.py write --variant v4 > $L/v4_write.log 2>&1; echo "$(date) write v4"
