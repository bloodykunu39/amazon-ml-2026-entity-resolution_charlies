#!/bin/bash
# Resume full run: model train -> tune -> (after test blocking) prune/features/predict test -> write.
set -e
cd "$(dirname "$0")"
PY=/home/moskov/miniconda3/envs/amazon/bin/python
RUN="systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=4G $PY"
L=../logs
$RUN model.py train > $L/model_train_full.log 2>&1
echo "$(date) model train done"
$RUN submit.py tune > $L/tune_full.log 2>&1
echo "$(date) tune done"
until [ -f ../work/cand_raw_test_US.parquet ] && ! pgrep -f "^$PY blocking.py test" >/dev/null; do sleep 20; done
echo "$(date) test blocking done"
$RUN prune.py test > $L/prune_test_full.log 2>&1
echo "$(date) prune test done"
$RUN features.py test > $L/feats_test_full.log 2>&1
echo "$(date) feats test done"
$RUN model.py test > $L/model_test_full.log 2>&1
echo "$(date) predict test done"
$RUN submit.py write > $L/write_full.log 2>&1
echo "$(date) write done"
