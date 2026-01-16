#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --partition=h100x
#SBATCH --job-name=lcc_8x_bytedance
# Ensure script is run with bash (not sh)
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh" >&2
    echo "Please run: bash $0" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1 # since the compute node is not connected to the internet
export PYTHONPATH=/path/to/your_python_path:$PYTHONPATH

echo "Starting LCC evaluation jobs with OnlineRerankPrunerModel..."
RERANK_API_PORT=8000
# Parameters
# MODEL_NAME="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MODEL_NAME="${1:-ByteDance-Seed/Seed-Coder-8B-Instruct}"
BACKBONE_MODEL_PATH="${2:-/path/to/backbone_model}"
HF_MODEL_PATH="${3:-/path/to/provence-hf}"
export HF_MODEL_PATH=${HF_MODEL_PATH} # for online serving
RESULT_DIR="${4:-downstream_eval/lcc/lcc_8x_bytedance}"
NUM_EXAMPLES="${5:-200}"
BATCH_SIZE="${6:-16}"
EMBED_MODEL_NAME="${7:-/path/to/unixcoder-base}"
TENSOR_PARALLEL_SIZE="${8:-1}"
TOP_K="${9:-3}"
DATASET_PATH="${10:-microsoft/LCC_python}"
DATASET_SPLIT="${11:-test}"

# Rerank API parameters
RERANK_THRESHOLD="${12:-0.5}"
RERANK_ALWAYS_KEEP_FIRST_FRAGS="${13:-True}"
RERANK_AGGREGATE_METHOD="${14:-line}"
RERANK_LANGUAGE="${15:-python}"


# Top-K parameters
RAG_TOP_K="${TOP_K}"
FUNCTION_RAG_TOP_K="${TOP_K}"

# LongCodeZip parameters
# LONGCODEZIP_MODEL_NAME="${16:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LONGCODEZIP_MODEL_NAME="${16:-ByteDance-Seed/Seed-Coder-8B-Instruct}"
LONGCODEZIP_RATE="${17:-0.5}"
LONGCODEZIP_RANK_ONLY="${18:-False}"

# LLMLingua2 parameters
LLMLINGUA2_MODEL_NAME="${19:-/path/to/llmlingua-2-model}"
LLMLINGUA2_RATE="${20:-0.33}"

# LongLLMLingua parameters
LONGLMLINGUA_RATE="${21:-0.55}"
LONGLMLINGUA_CONDITION_IN_QUESTION="${22:-after_condition}"
LONGLMLINGUA_REORDER_CONTEXT="${23:-sort}"
LONGLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO="${24:-0.3}"
LONGLMLINGUA_CONDITION_COMPARE="${25:-True}"
LONGLMLINGUA_CONTEXT_BUDGET="${26:-'+100'}"

# SilverLabelPrunerModel parameters
PRUNER_MODEL="${31:-/path/to/pruner-model}"
PRUNER_TENSOR_PARALLEL_SIZE="${32:-2}"

PRUNER_START_GPU="${PRUNER_START_GPU:-1}"          # pruner 起始 GPU index

SUMMARY_MODEL_NAME="${27:-/path/to/summary-model}"
SUMMARY_MAX_TOKENS="${28:-256}"
SUMMARY_TEMPERATURE="${29:-0.0}"

PYTHON_BIN="/path/to/python"
WORK_DIR="${WORK_DIR:-hard_code_pruner/downstream_task/LCC}"
RERANK_SERVER_SCRIPT="${RERANK_SERVER_SCRIPT:-hard_code_pruner/inference/online_serving.py}"
BASE_LOG_DIR="logs-lcc-rerank"

mkdir -p ${BASE_LOG_DIR}
mkdir -p ${RESULT_DIR}

# Rerank API server configuration
RERANK_API_BASE="http://localhost:${RERANK_API_PORT}"
RERANK_PID_FILE="${BASE_LOG_DIR}/rerank_server.pid"

echo "Model: $MODEL_NAME"
echo "Backbone Model: $BACKBONE_MODEL_PATH"
echo "HF Model Path: $HF_MODEL_PATH"
echo "Result Dir: $RESULT_DIR"
echo "Num Examples: $NUM_EXAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "Embedding Model: $EMBED_MODEL_NAME"
echo "Main Model Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Top-K (unified): $TOP_K"
echo "Dataset Path: $DATASET_PATH"
echo "Dataset Split: $DATASET_SPLIT"
echo "Rerank API Port: $RERANK_API_PORT"
echo "Rerank Threshold: $RERANK_THRESHOLD"
echo "Rerank Aggregate Method: $RERANK_AGGREGATE_METHOD"
echo "Rerank Language: $RERANK_LANGUAGE"
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
METHODS=("full" "no_context" "selective_context" "llmlingua2" "rag" "longcodezip" "rag_with_pruner")
# Start rerank API server (only if needed by any method)
NEEDS_RERANK=false
for method in "${METHODS[@]}"; do
    if [[ "${method}" == "rag_with_pruner" || "${method}" == "function_rag_with_pruner" || "${method}" == "longcodezip_with_pruner" || "${method}" == "rag_with_pruner_rerank" || "${method}" == "rag_with_silver_label_pruner" ]]; then
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
    export BACKBONE_MODEL_PATH="${BACKBONE_MODEL_PATH}"
    export DEVICE="cuda"
    
    # Start server from project root, redirect log to BASE_LOG_DIR
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

# Note: Evaluation jobs will manage their own GPU allocation via main.py
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
    RUN_DIR="${RESULT_DIR}/method_${method}"
    mkdir -p "${RUN_DIR}"

    # Common args
    ARGS=(
        --model_name "${MODEL_NAME}"
        --result_dir "${RUN_DIR}"
        --embed_model_name "${EMBED_MODEL_NAME}"
        --dataset_path "${DATASET_PATH}"
        --dataset_split "${DATASET_SPLIT}"
        --num_examples ${NUM_EXAMPLES}
        --batch_size ${BATCH_SIZE}
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}
        --syntax_check True
        --syntax_language "${RERANK_LANGUAGE}"
        --syntax_check_chunk True
        --method ${method}
    )

    # Set method-specific args
    if [[ "${method}" == "rag" ]]; then
        ARGS+=(
            --rag_window_size 40
            --rag_overlap 10
            --rag_top_k 2
        )
    elif [[ "${method}" == "rag_with_pruner" ]]; then
        ARGS+=(
            --pruner_type "online_rerank"
            --rerank_api_base "${RERANK_API_BASE}"
            --rerank_threshold 0.3
            --rerank_always_keep_first_frags ${RERANK_ALWAYS_KEEP_FIRST_FRAGS}
            --rerank_aggregate_method "${RERANK_AGGREGATE_METHOD}"
            --rerank_language "${RERANK_LANGUAGE}"
            --rag_window_size 40
            --rag_overlap 10
            --rag_top_k 2
        )
    elif [[ "${method}" == "llmlingua2" ]]; then
        ARGS+=(
            --llmlingua2_model_name "${LLMLINGUA2_MODEL_NAME}"
            --llmlingua2_rate 0.15
        )
    elif [[ "${method}" == "selective_context" ]]; then
        ARGS+=(
            --selective_context_model_type "gpt2"
            --selective_context_lang "en"
            --selective_context_reduce_ratio 0.85
        )
    elif [[ "${method}" == "longcodezip" ]]; then
        ARGS+=(
            --longcodezip_model_name "${LONGCODEZIP_MODEL_NAME}"
            --longcodezip_rate 0.15
            --longcodezip_rank_only "${LONGCODEZIP_RANK_ONLY}"
        )
    fi

    # Run serially (not in background)
    ${PYTHON_BIN} ${WORK_DIR}/main.py "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Method ${method} completed successfully"
    else
        echo "✗ Method ${method} failed with exit code ${PIPESTATUS[0]}"
    fi
    echo ""
done

echo "--- All LCC evaluation jobs completed ---"
