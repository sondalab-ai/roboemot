#!/bin/bash
set -e
S="${WAKEWORD_DIR:?set WAKEWORD_DIR to a scratch folder outside the repo}"
MLSREV=2e83e61823b4c47dcbcb1980bb88601274127609
D=$S/data/mls_it; mkdir -p $D
for f in 9_hours-00000-of-00001 test-00000-of-00001 dev-00000-of-00001; do
  [ -s $D/$f.parquet ] || curl -sfL -o $D/$f.parquet "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/$MLSREV/italian/$f.parquet"
done
D=$S/data/librispeech; mkdir -p $D
for f in dev-clean test-clean; do
  [ -s $D/$f.tar.gz ] || curl -sfL -o $D/$f.tar.gz "https://www.openslr.org/resources/12/$f.tar.gz"
  [ -d $D/LibriSpeech/$f ] || tar -xzf $D/$f.tar.gz -C $D
done
D=$S/data/oww_features; mkdir -p $D
OREV=985bf1b47e7f19c07741af82bfe32d5a9dc56096
[ -s $D/validation_set_features.npy ] || curl -sfL -o $D/validation_set_features.npy "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/$OREV/validation_set_features.npy"
du -sh $S/data/*
echo done
