#!/bin/bash
set -e
cd "$(dirname "$0")"
PY="nice -n 10 /home/moskov/miniconda3/envs/amazon/bin/python"
L=../logs
$PY prune.py train --sample --train > $L/v3s_prune.log 2>&1; echo "$(date) prune"
$PY features.py train --sample --force > $L/v3s_feats.log 2>&1; echo "$(date) feats"
$PY model.py train --sample > $L/v3s_model.log 2>&1; echo "$(date) model"
$PY submit.py tune --sample > $L/v3s_tune.log 2>&1; echo "$(date) tune"
