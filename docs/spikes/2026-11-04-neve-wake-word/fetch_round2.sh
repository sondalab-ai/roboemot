#!/bin/bash
# Round 2 downloads: the first ~2.1 GB of the ACAV100M feature file (NC comparison only) and one
# MLS Italian train shard (CC BY 4.0).
set -e
S="${WAKEWORD_DIR:?set WAKEWORD_DIR to a scratch folder outside the repo}"
OREV=985bf1b47e7f19c07741af82bfe32d5a9dc56096
ROWS=700000
END=$((128 + ROWS * 3072 - 1))
D=$S/data/oww_features
[ -s $D/acav_head.raw ] || curl -sfL -r 0-$END -o $D/acav_head.raw "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/$OREV/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
ls -l $D/acav_head.raw
MLSREV=2e83e61823b4c47dcbcb1980bb88601274127609
[ -s $S/data/mls_it/train-00000-of-00008.parquet ] || curl -sfL -o $S/data/mls_it/train-00000-of-00008.parquet "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/$MLSREV/italian/train-00000-of-00008.parquet"
ls -l $S/data/mls_it/
echo done
