#!/bin/bash
# kill python processes of the amazon env running the given script (anchored: never matches shells)
for p in $(pgrep -f "^/home/moskov/miniconda3/envs/amazon/bin/python $1"); do kill "$p" && echo "killed $p"; done
