#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --partition=h100x
#SBATCH --job-name=longcodeqa_8x

# Ensure script is run with bash (not sh)
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh" >&2
    echo "Please run: bash $0" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1 # since the compute node is not connected to the internet
export PYTHONPATH=/path/to/your_python_path:$PYTHONPATH
export HF_ENDPOINT="" # disable HF hub access
echo "Starting LongCodeQA evaluation jobs with OnlineRerankPrunerModel..."
RERANK_API_PORT=8000
# Parameters
# MODEL_NAME="${1:-/path/to/model}"
# MODEL_NAME="${1:-ByteDance-Seed/Seed-Coder-8B-Instruct}"
MODEL_NAME="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BACKBONE_MODEL_PATH="${2:-/path/to/backbone_model}"
HF_MODEL_PATH="${3:-/path/to/provence-hf}"
export HF_MODEL_PATH=${HF_MODEL_PATH} # for online serving
RESULT_DIR="${4:-downstream_eval/longcodeqa/longcodeqa_8x}"
NUM_EXAMPLES="${5:-200}"
BATCH_SIZE="${6:-16}"
EMBED_MODEL_NAME="${7:-/path/to/unixcoder-base}"
TENSOR_PARALLEL_SIZE="${8:-1}"
TOP_K="${9:-8}"
DATASET_PATH="${10:-/path/to/longcodeqa_32k.jsonl}"

# Rerank API parameters
RERANK_THRESHOLD="${11:-0.4}"
RERANK_ALWAYS_KEEP_FIRST_FRAGS="${12:-False}"
RERANK_AGGREGATE_METHOD="${13:-line}"
RERANK_LANGUAGE="${14:-python}"

# Top-K parameters
RAG_TOP_K="${TOP_K}"
FILE_RAG_TOP_K="${TOP_K}"

# LongCodeZip parameters
# LONGCODEZIP_MODEL_NAME="${15:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LONGCODEZIP_MODEL_NAME="${15:-ByteDance-Seed/Seed-Coder-8B-Instruct}"
LONGCODEZIP_RATE="${16:-0.5}"
LONGCODEZIP_RANK_ONLY="${17:-False}"

# Reranker parameters (for rag_with_rerank, rag_with_pruner_rerank)
RERANKER_TYPE="${18:-qwen}"  # "bert", "bgev2m3", "qwen", or "online"
RERANKER_MODEL_NAME="${19:-${BACKBONE_MODEL_PATH}}"  # Default to BACKBONE_MODEL_PATH for qwen reranker

# LLMLingua-2 parameters
LLMLINGUA2_MODEL_NAME="${20:-/path/to/llmlingua-2-model}"
LLMLINGUA2_RATE="${21:-0.1}"

# LongLLMLingua parameters
LONGLLMLINGUA_RATE="${22:-0.1}"
LONGLLMLINGUA_CONDITION_IN_QUESTION="${23:-after_condition}"
LONGLLMLINGUA_REORDER_CONTEXT="${24:-sort}"
LONGLLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO="${25:-0.3}"
LONGLLMLINGUA_CONDITION_COMPARE="${26:-True}"
LONGLLMLINGUA_CONTEXT_BUDGET="${27:-100}"

# SelectiveContext parameters
SELECTIVE_CONTEXT_MODEL_TYPE="${28:-gpt2}"
SELECTIVE_CONTEXT_LANG="${29:-en}"
SELECTIVE_CONTEXT_REDUCE_RATIO="${30:-0.9}"

# SilverLabelPrunerModel parameters
PRUNER_MODEL="${31:-/path/to/pruner-model}"
PRUNER_TENSOR_PARALLEL_SIZE="${32:-2}"

PRUNER_START_GPU="${PRUNER_START_GPU:-1}"          # pruner 起始 GPU index
PRUNER_TENSOR_PARALLEL_SIZE="${PRUNER_TENSOR_PARALLEL_SIZE:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"  # 主模型自己用的 TP


PYTHON_BIN="/path/to/python"
WORK_DIR="${WORK_DIR:-hard_code_pruner/downstream_task/longcodeqa}"
RERANK_SERVER_SCRIPT="${RERANK_SERVER_SCRIPT:-hard_code_pruner/inference/online_serving.py}"
BASE_LOG_DIR="logs-longcodeqa-rerank"
# MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"   # 64k 时用 65536；
# MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # 128k 时用 131072；
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # 测试时用 32768；
mkdir -p ${BASE_LOG_DIR}
mkdir -p ${RESULT_DIR}

# Rerank API server configuration
RERANK_API_BASE="http://localhost:${RERANK_API_PORT}"
RERANK_PID_FILE="${BASE_LOG_DIR}/rerank_server.pid"

echo "Model: $MODEL_NAME"
echo "Backbone Model: $BACKBONE_MODEL_PATH"
echo "Result Dir: $RESULT_DIR"
echo "Num Examples: $NUM_EXAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "Embedding Model: $EMBED_MODEL_NAME"
echo "Main Model Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Top-K (unified): $TOP_K"
echo "Rerank API Port: $RERANK_API_PORT"
echo "Rerank Threshold: $RERANK_THRESHOLD"
echo "Rerank Aggregate Method: $RERANK_AGGREGATE_METHOD"
echo "LongCodeZip Model: $LONGCODEZIP_MODEL_NAME"
echo "LongCodeZip Rate: $LONGCODEZIP_RATE"
echo "LongCodeZip Rank Only: $LONGCODEZIP_RANK_ONLY"
echo "Reranker Type: $RERANKER_TYPE"
echo "Reranker Model Name: $RERANKER_MODEL_NAME"
echo "LLMLingua-2 Model: $LLMLINGUA2_MODEL_NAME"
echo "LLMLingua-2 Rate: $LLMLINGUA2_RATE"
echo "LongLLMLingua Rate: $LONGLLMLINGUA_RATE"
echo "LongLLMLingua Condition in Question: $LONGLLMLINGUA_CONDITION_IN_QUESTION"
echo "LongLLMLingua Reorder Context: $LONGLLMLINGUA_REORDER_CONTEXT"
echo "LongLLMLingua Dynamic Context Compression Ratio: $LONGLLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO"
echo "LongLLMLingua Condition Compare: $LONGLLMLINGUA_CONDITION_COMPARE"
echo "LongLLMLingua Context Budget: $LONGLLMLINGUA_CONTEXT_BUDGET"
echo "SelectiveContext Model Type: $SELECTIVE_CONTEXT_MODEL_TYPE"
echo "SelectiveContext Lang: $SELECTIVE_CONTEXT_LANG"
echo "SelectiveContext Reduce Ratio: $SELECTIVE_CONTEXT_REDUCE_RATIO"
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0 # port is in use
    else
        return 1 # port is free
    fi
}

# Function to wait for server to be ready
wait_for_server() {
    local url=$1
    local pid=$2
    local log_file=$3
    local attempt=0
    local max_log_lines=1000

    echo "Waiting for server at ${url} to be ready..."
    echo "Monitoring process PID: ${pid}"
    echo "Log file: ${log_file}"
    echo ""

    while true; do
        # Check if process is still running
        if ! kill -0 ${pid} 2>/dev/null; then
            echo ""
            echo "ERROR: Server process has exited unexpectedly!"
            echo "Last ${max_log_lines} lines from log:"
            tail -n ${max_log_lines} "${log_file}"
            return 1
        fi

        # Check if server is ready (health endpoint)
        if curl -s -f "${url}/health" >/dev/null 2>&1; then
            echo ""
            echo "Server at ${url} is ready!"
            return 0
        fi

        # Show periodic progress
        attempt=$((attempt + 1))
        if [ $((attempt % 30)) -eq 0 ]; then
            echo "[${attempt}s] Still waiting... Server may be loading large model."
        fi

        sleep 2
    done
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up servers..."

    if [ -f "${RERANK_PID_FILE}" ]; then
        RERANK_PID=$(cat "${RERANK_PID_FILE}")
        if kill -0 ${RERANK_PID} 2>/dev/null; then
            echo "Stopping rerank server (PID: ${RERANK_PID})..."
            kill ${RERANK_PID}
            wait ${RERANK_PID} 2>/dev/null
        fi
        rm -f "${RERANK_PID_FILE}"

        # Show last few lines of log if it exists
        RERANK_LOG_FILE="${BASE_LOG_DIR}/rerank_server.log"
        if [ -f "${RERANK_LOG_FILE}" ]; then
            echo ""
            echo "Last 30 lines from rerank server log:"
            tail -n 30 "${RERANK_LOG_FILE}"
        fi
    fi

    echo "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Methods to run (defined early for needs_rerank check)
METHODS=("full" "no_context" "rag" "llmlingua2" "longcodezip" "selective_context" "rag_with_pruner")
echo "METHODS: ${METHODS[@]}"

# Start rerank API server (only if needed by any method that uses OnlineRerankPrunerModel)
# Note: rag_with_rerank uses local reranker model s, not API
NEEDS_RERANK=false
for method in "${METHODS[@]}"; do
    if [[ "${method}" == "rag_with_pruner" || "${method}" == "file_rag_with_pruner" || "${method}" == "longcodezip_with_pruner" || "${method}" == "rag_with_pruner_rerank" || "${method}" == "rag_with_token_pruner" ]]; then
        NEEDS_RERANK=true
        break
    fi
done

if [ "${NEEDS_RERANK}" = true ]; then
    # Check if rerank port is available (only if we need to use rerank)
    if check_port ${RERANK_API_PORT}; then
        echo "WARNING: Port ${RERANK_API_PORT} is already in use. Please stop the existing server first."
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo "Starting rerank API server..."
    echo "Backbone Model: ${BACKBONE_MODEL_PATH}"
    echo "Port: ${RERANK_API_PORT}"
    echo "=========================================="

    # Start rerank server in background
    # Rerank server uses GPU 1+ (GPU 0 is for embedding model)
    # For now, we don't restrict CUDA_VISIBLE_DEVICES, but the server can use any available GPU
    RERANK_SERVER_LOG="${BASE_LOG_DIR}/rerank_server.log"
    ${PYTHON_BIN} "${RERANK_SERVER_SCRIPT}" >"${RERANK_SERVER_LOG}" 2>&1 &
    RERANK_SERVER_PID=$!
    echo ${RERANK_SERVER_PID} >"${RERANK_PID_FILE}"

    if ! wait_for_server "${RERANK_API_BASE}" "${RERANK_SERVER_PID}" "${RERANK_SERVER_LOG}"; then
        echo "Failed to start rerank server"
        exit 1
    fi

    # Wait a bit more to ensure server is fully ready
    sleep 3
fi

# Note: Evaluation jobs will manage their own GPU allocation via eval.py
# Rerank server uses GPU 1+ (GPU 0 is for embedding model)

echo ""
echo "=========================================="
echo "Starting evaluation jobs (serially)..."
echo "Note: Using OnlineRerankPrunerModel via API"
echo "=========================================="
echo ""

for method in "${METHODS[@]}"; do
    LOG_FILE="${BASE_LOG_DIR}/${MODEL_NAME//\//_slash_}_${method}.log"
    echo "=========================================="
    echo "Running method=${method} (log: ${LOG_FILE})"
    echo "=========================================="

    # Common args
    # ARGS=(
    #    --model_name "${MODEL_NAME}"
    #    --result_dir "${RESULT_DIR}"
    #    --embed_model_name "${EMBED_MODEL_NAME}"
    #    --dataset_path "${DATASET_PATH}"
    #    --num_examples ${NUM_EXAMPLES}
    #    --batch_size ${BATCH_SIZE}
    #    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}
    #    --max_model_len ${MAX_MODEL_LEN}
    #)

    RUN_DIR="${RESULT_DIR}/method_${method}"
    mkdir -p "${RUN_DIR}"

    ARGS=(
        --model_name "${MODEL_NAME}"
        --result_dir "${RUN_DIR}"
        --embed_model_name "${EMBED_MODEL_NAME}"
        --dataset_path "${DATASET_PATH}"
        --num_examples ${NUM_EXAMPLES}
        --batch_size ${BATCH_SIZE}
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}
        --max_model_len ${MAX_MODEL_LEN}
    )


    ARGS+=(
        --method ${method}
    )
    # Set method-specific args
    if [[ "${method}" == "rag" ]]; then
        ARGS+=(
            --rag_window_size 90
            --rag_overlap 15
            --rag_top_k 4
        )
    elif [[ "${method}" == "longcodezip" ]]; then
        ARGS+=(
            --longcodezip_model_name "${LONGCODEZIP_MODEL_NAME}"
            --longcodezip_rate 0.2
            --longcodezip_rank_only ${LONGCODEZIP_RANK_ONLY}
        )
    elif [[ "${method}" == "rag_with_pruner" ]]; then
        ARGS+=(
            --pruner_type "online_rerank"
            --rerank_api_base "${RERANK_API_BASE}"
            --rerank_threshold 0.4
            --rerank_always_keep_first_frags ${RERANK_ALWAYS_KEEP_FIRST_FRAGS}
            --rerank_aggregate_method "${RERANK_AGGREGATE_METHOD}"
            --rerank_language "${RERANK_LANGUAGE}"
            --rag_window_size 80
            --rag_overlap 40
            --rag_top_k 4
        )
    elif [[ "${method}" == "llmlingua2" ]]; then
        ARGS+=(
            --llmlingua2_model_name "${LLMLINGUA2_MODEL_NAME}"
            --llmlingua2_rate 0.15
        )
        ARGS+=(
            --longllmlingua_rate ${LONGLLMLINGUA_RATE}
            --longllmlingua_condition_in_question "${LONGLLMLINGUA_CONDITION_IN_QUESTION}"
            --longllmlingua_reorder_context "${LONGLLMLINGUA_REORDER_CONTEXT}"
            --longllmlingua_dynamic_context_compression_ratio ${LONGLLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO}
            --longllmlingua_condition_compare ${LONGLLMLINGUA_CONDITION_COMPARE}
            --longllmlingua_context_budget "${LONGLLMLINGUA_CONTEXT_BUDGET}"
        )
    elif [[ "${method}" == "selective_context" ]]; then
        ARGS+=(
            --selective_context_model_type "${SELECTIVE_CONTEXT_MODEL_TYPE}"
            --selective_context_lang "${SELECTIVE_CONTEXT_LANG}"
            --selective_context_reduce_ratio 0.86
        )
    fi

    # Run serially (not in background)
    ${PYTHON_BIN} ${WORK_DIR}/eval.py "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Method ${method} completed successfully"
    else
        echo "✗ Method ${method} failed with exit code ${PIPESTATUS[0]}"
    fi
    echo ""
done

echo "--- All LongCodeQA evaluation jobs completed ---"

