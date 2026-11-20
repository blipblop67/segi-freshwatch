#!/usr/bin/env bash
# Build a TensorRT FP16 engine from the ONNX model ON THE TARGET JETSON.
# Engines are tied to the GPU + TensorRT version: rebuild after JetPack upgrades.
#
#   bash training/build_engine.sh models/freshness.onnx models/freshness_fp16.engine
set -euo pipefail

ONNX=${1:-models/freshness.onnx}
ENGINE=${2:-models/freshness_fp16.engine}
SIZE=${SIZE:-224}
MAX_BATCH=${MAX_BATCH:-16}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --minShapes=images:1x3x${SIZE}x${SIZE} \
  --optShapes=images:8x3x${SIZE}x${SIZE} \
  --maxShapes=images:${MAX_BATCH}x3x${SIZE}x${SIZE} \
  --memPoolSize=workspace:1024 \
  --skipInference \
  2>&1 | tail -n 5

echo "engine: $ENGINE"
