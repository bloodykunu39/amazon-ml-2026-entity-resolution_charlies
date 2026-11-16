#!/bin/bash
# PC1: bundle code + raw data for a clean reproducibility run on the second PC.
#   bash src/pc2_bundle.sh            -> ~/pc2_bundle.tar (code, config, tests, raw TSVs; no caches, no outputs)
set -e
cd "$(dirname "$0")/.."
tar --exclude='__pycache__' --exclude='*/__MACOSX*' --exclude='._*' --exclude='.DS_Store' -cf ~/pc2_bundle.tar \
    src tests config.yaml README.md requirements.txt \
    student_resource/dataset student_resource/utils student_resource/README.md student_resource/Documentation_template.md
ls -lh ~/pc2_bundle.tar
echo "Copy ~/pc2_bundle.tar to the second PC (USB / scp), then run: bash src/pc2_run.sh (see its header)"
