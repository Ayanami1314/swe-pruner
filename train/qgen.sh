#!/bin/bash
# Usage: ./qgen.sh <DATASET_NAME> <RESULT_DIR> [--model MODEL_PATH]
# Example: ./qgen.sh mydata ./out --model Qwen/Qwen3-Coder-30B-A3B-Instruct
# Run from repo root so that python -m train.inference.qgen works (or set PYTHONPATH).

set -e
DATASET_NAME="${1:?Usage: qgen.sh DATASET_NAME RESULT_DIR [--model MODEL]}"
RESULT_DIR="${2:?Usage: qgen.sh DATASET_NAME RESULT_DIR [--model MODEL]}"
shift 2 || true

python -m train.inference.qgen \
  -i "${DATASET_NAME}.jsonl" \
  -o "${RESULT_DIR}/generated_queries-${DATASET_NAME}.jsonl" \
  "$@"
