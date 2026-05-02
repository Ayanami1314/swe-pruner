#!/bin/bash
# Usage: ./train_llm.sh <NUM_GPUS> <INPUT_FILE> [--model-name MODEL] [other typer options...]
# Example: ./train_llm.sh 4 /path/to/labeled_data.jsonl --model-name Qwen/Qwen3-Reranker-0.6B --epochs 5
# Run from repo root so that python -m train.train_llm.train works.

set -e
NUM_GPUS="${1:?Usage: train_llm.sh NUM_GPUS INPUT_FILE [--model-name MODEL] ...}"
INPUT_FILE="${2:?Usage: train_llm.sh NUM_GPUS INPUT_FILE [--model-name MODEL] ...}"
shift 2 || true

# Run from repo root so train package is found
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  train/train_llm/train.py \
  -i "$INPUT_FILE" \
  "$@"
