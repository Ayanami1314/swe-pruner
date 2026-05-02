#!/bin/bash
# Usage: ./label.sh <DATASET_NAME> <RESULT_DIR> [--model-name MODEL_PATH] ...
# Example: ./label.sh mydata ./out --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct
# Run from repo root so that python -m train.inference.build_label works.

set -e
DATASET_NAME="${1:?Usage: label.sh DATASET_NAME RESULT_DIR [options]}"
RESULT_DIR="${2:?Usage: label.sh DATASET_NAME RESULT_DIR [options]}"
shift 2 || true

python -m train.inference.build_label \
  --input-file "${RESULT_DIR}/${DATASET_NAME}_qs.jsonl" \
  --output-jsonl "${RESULT_DIR}/labeled_${DATASET_NAME}.jsonl" \
  "$@"
