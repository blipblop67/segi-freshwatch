#!/usr/bin/env bash
# Download the "raw" split of Project-AgML/fresh_rotten_fruit_classification (CC-BY-4.0, ~2.9 GB).
set -euo pipefail
OUT=${1:-data/raw}
mkdir -p "$OUT"
BASE=https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification/resolve/main/raw
for i in 0 1 2 3; do
  f="train-0000${i}-of-00004.parquet"
  dst="$OUT/raw-0000${i}-of-00004.parquet"
  [ -s "$dst" ] && { echo "have $dst"; continue; }
  curl -fL --retry 3 -o "$dst.part" "$BASE/$f" && mv "$dst.part" "$dst"
done
ls -lh "$OUT"
