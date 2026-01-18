import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import fire

from vllm import LLM, SamplingParams
from loguru import logger
import gc
from typing import List, Optional, Protocol
import re
import sys
from pathlib import Path
import socket
import time
import numpy as np

# Import local CodeCompressor instead of longcodezip package
try:
    from code_compressor import CodeCompressor

    LONGCODEZIP_AVAILABLE = True
except ImportError:
    LONGCODEZIP_AVAILABLE = False
    logger.warning(
        "CodeCompressor not available. Make sure code_compressor.py is in the same directory."
    )


# Import LLMLingua PromptCompressor
try:
    from llmlingua import PromptCompressor

    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    logger.warning("LLMLingua not available. Install with: pip install llmlingua")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# HINT: this is fucking hack for running llmlingua on offline machine, need fucking hack the lib from `self.oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")` -> `self.oai_tokenizer = tiktoken.get_encoding('cl100k_base')` and download the tokenizer first
tiktoken_cache_dir = "./tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
import tiktoken

llmlingua_tokenizer = tiktoken.get_encoding("cl100k_base")


# Import SelectiveContext
try:
    from selective_context import SelectiveContext

    SELECTIVE_CONTEXT_AVAILABLE = True
except ImportError:
    SELECTIVE_CONTEXT_AVAILABLE = False
    logger.warning(
        "SelectiveContext not available. Install with: pip install selective-context"
    )

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_syntax_correctness import (
    check_syntax_correctness,
    reconstruct_code_from_kept_frags,
    prune_line_level_random,
    prune_random,
    prune_line_level_random_unconstrained,
    prune_random_unconstrained
)
from LCC.utils import load_data, compute_EM, compute_ES
from embedder import EmbedderAdapter, BertBasedEmbedder, QwenEmbedder, BGEM3Embedder
from reranker import (
    QwenReranker,
    RerankerAdapter,
    BertBasedReranker,
    BGEV2M3Reranker,
    JinaReranker,
    OnlineReranker,
)
from model import SilverLabelPrunerModel, OnlineRerankPrunerModel


class PrunerModel(Protocol):
    def prune(query: str, origin_code: str) -> str: ...
    def prune_batch(
        query: str,
        origin_codes: List[str],
        batch_size=16,
        sort: bool = False,
        return_raw_body: bool = False,
    ) -> List[str] | List[dict]: ...


# Helper function to check if a port is listening
def is_port_listening(port: int) -> bool:
    """Check if a port is listening on localhost."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


# Helper function for splitting code by functions (standalone version)
def split_code_by_functions_standalone(
    code: str, language: str = "python"
) -> List[str]:
    """
    Split code into chunks based on function and class definitions for various languages.
    Standalone version that doesn't require CodeCompressor instance.

    Args:
        code: The code to split
        language: Programming language of the code (python, cpp, java, typescript, rust, go)

    Returns:
        List of code chunks, each containing a function, class, or class method
    """
    # Define regex patterns for different languages
    patterns = {
        # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
        "python": r"(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*",
        # C++: Improved to better handle multi-line declarations
        "cpp": r"(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?",
        # Java: Improved for multi-line method declarations
        "java": r"(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?",
        # TypeScript: Enhanced to handle multi-line methods and arrow functions
        "typescript": r"(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?",
        # Rust: Improved for multi-line function declarations
        "rust": r"(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?",
        # Go: Improved for multi-line function declarations
        "go": r"(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?",
    }

    # Use default Python pattern if language not supported
    if language.lower() not in patterns:
        language = "python"

    function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
    matches = list(function_pattern.finditer(code))

    if not matches:
        return (
            [code] if code.strip() else []
        )  # No matches, return whole code if not empty

    result_chunks = []

    # Add code before first match if exists
    if matches[0].start() > 0:
        pre_code = code[: matches[0].start()].strip()
        if pre_code:
            result_chunks.append(pre_code)

    # Process each match
    for i, match in enumerate(matches):
        start = match.start()

        # End is either start of next match or end of code
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(code)

        chunk = code[start:end].strip()
        if chunk:
            result_chunks.append(chunk)

    return result_chunks


# Helper function for function-level RAG retrieval
def function_rag_retrieve(
    background_code: str,
    query_code: str,
    embedder: EmbedderAdapter,
    language: str,
    top_k: int,
    reranker: Optional[RerankerAdapter] = None,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
    return_chunks: bool = False,
) -> str | tuple[str, list[str]]:
    """Uses function-level chunking and retrieves top_k similar functions."""
    if not background_code.strip():
        return ""  # Return empty if no background context

    # Split code into function-based chunks
    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""  # Return empty if chunking results in nothing

    query_embedding = embedder.embed([query_code])[0]

    chunk_embeddings = []
    valid_chunks = [c for c in chunks if c.strip()]
    if not valid_chunks:
        return ""
    chunk_embeddings = embedder.embed(valid_chunks, batch_size=32)

    # Check if chunk_embeddings is empty (it's a tensor, so check shape)
    if isinstance(chunk_embeddings, torch.Tensor):
        if chunk_embeddings.numel() == 0 or chunk_embeddings.shape[0] == 0:
            return ""
    elif not chunk_embeddings:
        return ""
    # Compute cosine similarity
    similarities = torch.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings, dim=1
    )

    # Rank ALL chunks by similarity (do not pre-truncate by top_k)
    ranked_indices = torch.argsort(similarities, descending=True).tolist()
    ordered_chunks = [valid_chunks[i] for i in ranked_indices]

    # Optional reranking with batch processing
    if reranker is not None:
        # Batch rerank all retrieved chunks at once
        # reranker returns list[tuple[float, str]]: [(score, doc), ...]
        rerank_results = reranker.rerank(
            query_code, ordered_chunks[:top_k], batch_size=16
        )
        # Sort by rerank scores (descending) and extract documents
        rerank_results.sort(key=lambda x: x[0], reverse=True)
        ordered_chunks = [doc for score, doc in rerank_results]

    # Optional post-prune
    if pruner:
        ordered_chunks = pruner.prune_batch(
            query_code, ordered_chunks[:top_k], sort=pruner_sort
        )
    else:
        ordered_chunks = ordered_chunks[:top_k]
    combined_code = "\n\n".join(ordered_chunks)
    if return_chunks:
        return combined_code, ordered_chunks
    return combined_code


# Helper function for rerank-only retrieval (no embedder)
def rerank_only_retrieve(
    background_code: str,
    query_code: str,
    reranker: RerankerAdapter,
    window_size: int = 80,
    overlap: int = 40,
    top_k: int = 3,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
    return_chunks: bool = False,
) -> str | tuple[str, list[str]]:
    """
    Rerank-only retrieval: uses reranker directly without embedder.
    Chunks background code, reranks all chunks, and returns top_k.
    """
    if not background_code.strip():
        return ""

    chunks = chunk_sliding_window(background_code, window_size, overlap)
    if not chunks:
        return ""

    valid_chunks = [c for c in chunks if c.strip()]
    if not valid_chunks:
        return ""

    # Rerank all chunks directly (no embedding step)
    rerank_results = reranker.rerank(query_code, valid_chunks, batch_size=16)
    # Sort by rerank scores (descending) and extract documents
    rerank_results.sort(key=lambda x: x[0], reverse=True)
    ordered_chunks = [doc for score, doc in rerank_results]

    # Optional post-prune
    if pruner:
        ordered_chunks = pruner.prune_batch(
            query_code, ordered_chunks[:top_k], sort=pruner_sort
        )
    else:
        ordered_chunks = ordered_chunks[:top_k]

    combined_code = "\n\n".join(ordered_chunks)
    if return_chunks:
        return combined_code, ordered_chunks
    return combined_code


# Helper function for function-level rerank-only retrieval
def function_rerank_only_retrieve(
    background_code: str,
    query_code: str,
    reranker: RerankerAdapter,
    language: str,
    top_k: int,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
    return_chunks: bool = False,
) -> str | tuple[str, list[str]]:
    """
    Function-level rerank-only retrieval: uses reranker directly without embedder.
    Splits code by functions, reranks all functions, and returns top_k.
    """
    if not background_code.strip():
        return ""

    # Split code into function-based chunks
    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""

    valid_chunks = [c for c in chunks if c.strip()]
    if not valid_chunks:
        return ""

    # Rerank all chunks directly (no embedding step)
    rerank_results = reranker.rerank(query_code, valid_chunks, batch_size=16)
    # Sort by rerank scores (descending) and extract documents
    rerank_results.sort(key=lambda x: x[0], reverse=True)
    ordered_chunks = [doc for score, doc in rerank_results]

    # Optional post-prune
    if pruner:
        ordered_chunks = pruner.prune_batch(
            query_code, ordered_chunks[:top_k], sort=pruner_sort
        )
    else:
        ordered_chunks = ordered_chunks[:top_k]

    combined_code = "\n\n".join(ordered_chunks)
    if return_chunks:
        return combined_code, ordered_chunks
    return combined_code


def chunk_by_token_count(
    text: str, tokenizer, max_tokens: int = 500, overlap_tokens: int = 0
) -> list[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: Text to split
        tokenizer: Tokenizer to use for counting tokens
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Tokenize the entire text
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    stride = max_tokens - overlap_tokens

    if stride <= 0:
        stride = max_tokens

    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end_idx >= len(tokens):
            break

        start_idx += stride

    return chunks if chunks else [text]


# Helper function for sliding window chunking
def chunk_sliding_window(code: str, window_size: int, overlap: int) -> list[str]:
    """Splits code into overlapping chunks using a sliding window."""
    lines = code.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("Overlap size must be smaller than window size.")

    while True:
        end = min(start + window_size, len(lines))
        chunk_lines = lines[start:end]
        if not chunk_lines:  # Should not happen if lines is not empty, but safety check
            break
        chunks.append("\n".join(chunk_lines))
        if end == len(lines):
            break  # Exit loop if we reached the end
        next_start = start + stride
        # If the next window would go past the end, break
        if next_start >= len(lines):
            # Add the final overlapping chunk if needed
            final_start = max(0, len(lines) - window_size)
            if final_start > start:  # Ensure it's a new chunk not already added
                final_chunk_lines = lines[final_start:]
                chunks.append("\n".join(final_chunk_lines))
            break
        start = next_start

    # Handle case where code is shorter than window size
    if not chunks and lines:
        return ["\n".join(lines)]

    # Remove duplicates while preserving order (important for RAG)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks


def rag_retrieve(
    background_code: str,
    query_code: str,
    embedder: EmbedderAdapter,
    reranker: Optional[RerankerAdapter] = None,
    window_size: int = 80,
    overlap: int = 40,
    top_k: int = 3,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
    return_chunks: bool = False,
) -> str | tuple[str, list[str]]:
    if not background_code.strip():
        return ""

    chunks = chunk_sliding_window(background_code, window_size, overlap)
    if not chunks:
        return ""

    # Embed query and chunks using adapter
    query_embedding = embedder.embed([query_code])[0]  # Get first embedding

    # Embed all valid chunks
    valid_chunks = [c for c in chunks if c.strip()]

    # Batch embed all chunks
    chunk_embeddings = embedder.embed(valid_chunks, batch_size=32)

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings, dim=1
    )

    # Get top_k indices
    top_k_indices = torch.topk(
        similarities, k=min(top_k, len(valid_chunks)), dim=0
    ).indices.tolist()

    retrieved_chunks = [valid_chunks[i] for i in top_k_indices]

    # Optional reranking with batch processing
    if reranker is not None:
        # Batch rerank all retrieved chunks at once
        # reranker returns list[tuple[float, str]]: [(score, doc), ...]
        rerank_results = reranker.rerank(query_code, retrieved_chunks, batch_size=16)
        # Sort by rerank scores (descending) and extract documents
        rerank_results.sort(key=lambda x: x[0], reverse=True)
        retrieved_chunks = [doc for score, doc in rerank_results]

    if pruner:
        retrieved_chunks = pruner.prune_batch(
            query_code, retrieved_chunks, sort=pruner_sort
        )

    retrieved_chunks = retrieved_chunks[:top_k]
    combined_code = "\n\n".join(retrieved_chunks)
    if return_chunks:
        return combined_code, retrieved_chunks
    return combined_code


def save_json(data: dict, file_path: str):
    """Saves dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def generate_completions(llm, batch_prompts, max_new_tokens=128):
    # Generate completions for batch
    sampling_params = SamplingParams(
        temperature=0, top_p=0.95, max_tokens=max_new_tokens
    )
    batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

    return [x.outputs[0].text for x in batch_outputs]


def evaluate_completion(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    method: str = "full",
    result_dir: str = "results/completion_baselines",
    embed_model_name: str = "microsoft/unixcoder-base",
    embed_model_type: str = "bert",  # "bert", "qwen", "bgem3", "bgecode"
    reranker_model_name: Optional[str] = None,
    reranker_model_type: Optional[str] = None,  # "bert", "bgev2m3", "qwen", or None
    dataset_path: str = "microsoft/LCC_python",
    dataset_split: str = "test",
    num_examples: int = 200,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    # RAG params
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    # Function RAG params
    function_rag_language: str = "python",
    function_rag_top_k: int = 3,
    # vLLM params
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.9,
    filter_current_lines_max: int = 50,
    filter_background_tokens_min: int = 3000,
    # Pruner params for rag_with_pruner and function_rag_with_pruner
    pruner_type: str = "silver_label",  # "silver_label" or "online_rerank"
    pruner_model_name: str = None,
    pruner_tensor_parallel_size: int = 1,
    pruner_temperature: float = 0.3,
    pruner_max_tokens: int = 8192,  # HINT: pruner method is like cot, a higher max tokens needed.
    # Online mode params (for SilverLabelPrunerModel)
    pruner_online_mode: bool = False,
    pruner_api_base: str = None,
    pruner_api_key: str = None,
    pruner_online_model_name: str = "default-model",
    pruner_api_port: int = 8001,  # Default port for pruner API server detection
    pruner_start_gpu: int = 1,
    # OnlineRerankPrunerModel params
    rerank_api_base: str = "http://localhost:8000",
    rerank_threshold: float = 0.5,
    rerank_always_keep_first_frags: bool = False,
    rerank_aggregate_method: str = "line",
    rerank_language: str = "python",
    # LongCodeZip params
    longcodezip_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    longcodezip_rate: float = 0.5,
    longcodezip_rank_only: bool = False,
    # LLMLingua-2 params
    llmlingua2_model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    llmlingua2_rate: float = 0.33,
    llmlingua2_force_tokens: list = None,  # e.g., ['\n', '?']
    # LongLLMLingua params
    longllmlingua_rate: float = 0.55,
    longllmlingua_condition_in_question: str = "after_condition",
    longllmlingua_reorder_context: str = "sort",
    longllmlingua_dynamic_context_compression_ratio: float = 0.3,
    longllmlingua_condition_compare: bool = True,
    longllmlingua_context_budget: str = "+100",
    # SelectiveContext params
    selective_context_model_type: str = "gpt2",
    selective_context_lang: str = "en",
    selective_context_reduce_ratio: float = 0.5,
    syntax_check: bool = False,
    syntax_language: str = "python",
    syntax_check_chunk: bool = False,

    # Unified compression constraint: raw/compressed = x; 0 indicates off.
    target_compression_x: float = 0.0,

    # Qwen3 Summary params
    summary_model_name: str = "Qwen/Qwen3-0.6B",
    summary_max_tokens: int = 256,
    summary_temperature: float = 0.0,
):
    """Evaluates code completion baselines with a specified context preparation method."""
    # 1.Initialize the performance metric dictionary at the beginning of the function.
    performance_metrics = {
        "example_id": [],
        "context_preparation_time_seconds": [],
        "inference_time_per_example_seconds": [],
        "original_tokens": [],
        "processed_tokens": [],
        "compression_ratio": [],
    }

    syntax_correct = 0
    syntax_total = 0
    syntax_flags = []  # Optional: Boolean result for each line
    chunk_syntax_correct = 0
    chunk_syntax_total = 0
    chunk_syntax_random_correct = 0
    chunk_syntax_random_total = 0
    chunk_syntax_line_random_correct = 0
    chunk_syntax_line_random_total = 0

    # Modify result directory based on method and parameters
    method_suffix = f"method_{method}"
    # ... existing suffix logic ...
    if target_compression_x and target_compression_x > 1:
        method_suffix += f"_cx{target_compression_x}"

    # GPU allocation strategy:
    # - embedding model: cuda:0 (1 GPU)
    # - pruner: cuda:1 to cuda:1+pruner_tp-1 (pruner_tensor_parallel_size GPUs)
    # - main model: cuda:1+pruner_tp to cuda:1+pruner_tp+main_tp-1 (tensor_parallel_size GPUs)

    # Calculate GPU ranges BEFORE initializing CUDA
    # Strategy:
    # - GPU 0: embedding model
    # - GPU 1+: rerank/pruner server (external process)
    # - GPU 1+pruner_tp: main model

    main_tp = tensor_parallel_size
    embed_device = torch.device("cuda:0")

    # GPU Allocation Strategy:
    # - GPU 0: Embedding model
    # - GPU 1: ReRank server (external process, fixed in lcc_eval_online.sh)
    # - GPU 2+: Main model (LLM inference)
    #
    # Note: The rerank server is started with CUDA_VISIBLE_DEVICES="1" in lcc_eval_online.sh
    # to ensure it only uses GPU 1, preventing conflicts with the main model.
    pruner_tp = (
        pruner_tensor_parallel_size
        if "pruner" in method and pruner_type == "silver_label" and not pruner_online_mode
        else 0
    )
    pruner_end_gpu = pruner_start_gpu + pruner_tp - 1


    is_using_online_rerank = (
        method in ["rag_with_pruner", "function_rag_with_pruner"]
        and pruner_type == "online_rerank"
    )

    # Check if rerank server is running
    if is_using_online_rerank:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(rerank_api_base)
            rerank_port = parsed.port or 8000
        except Exception:
            rerank_port = 8000

        if is_port_listening(rerank_port):
            logger.info(f"Detected rerank server on port {rerank_port}")
            # Rerank server is running on GPU 1 (reserved by lcc_eval_online.sh)
            # Main model should start from GPU 2
            main_start_gpu = 2
        else:
            logger.warning(
                f"Rerank server on port {rerank_port} not detected. "
                "Starting main model from GPU 1 (assuming no GPU conflicts)"
            )
            main_start_gpu = 1
    else:
        # No rerank server, main model can start from GPU 1
        main_start_gpu = 1
    
    if pruner_tp > 0:
        main_start_gpu = max(main_start_gpu, pruner_end_gpu + 1)

    main_end_gpu = main_start_gpu + main_tp - 1

    # Now check CUDA availability (this will initialize CUDA)
    if not torch.cuda.is_available():
        logger.error("CUDA not available. GPU allocation requires CUDA.")
        return

    # Verify we have enough GPUs
    # Check BEFORE setting CUDA_VISIBLE_DEVICES, as we need to verify physical GPU existence
    num_gpus = torch.cuda.device_count()
    max_gpu_needed = main_end_gpu + 1  # +1 because GPU indices are 0-based
    if max_gpu_needed > num_gpus:
        logger.error(
            f"Not enough GPUs for allocation. "
            f"Need GPU 0-{max_gpu_needed - 1} ({max_gpu_needed} GPUs total), "
            f"but only {num_gpus} available. "
            f"Cannot proceed with GPU allocation."
        )
        return

    # Verify that the specific GPUs we need actually exist and are accessible
    for gpu_id in range(main_start_gpu, main_end_gpu + 1):
        try:
            torch.cuda.get_device_properties(gpu_id)
        except RuntimeError as e:
            logger.error(
                f"GPU {gpu_id} is not accessible: {e}. "
                f"Cannot proceed with GPU allocation."
            )
            return

    logger.info("GPU allocation strategy:")
    logger.info("  - Embedding model: cuda:0")
    if is_using_online_rerank:
        logger.info("  - Pruner model: OnlineRerankPrunerModel (API-based)")
        if is_port_listening(rerank_port):
            logger.info("  - Rerank server: GPU 1 (external process)")
        else:
            logger.info("  - Rerank server: NOT DETECTED (GPU 1 reserved but unused)")
    logger.info(
        f"  - Main model: cuda:{main_start_gpu} to cuda:{main_end_gpu} ({main_tp} GPUs)"
    )

    # --- 1. Load Data ---
    # Assuming python for now, might need modification if dataset has multiple languages
    # Note: Language info might be needed for CodeCompressor if not always python
    dataset, _ = load_data(
        path=dataset_path,
        split=dataset_split,
        num_examples=num_examples,
        filter_current_lines_max=filter_current_lines_max,
        filter_background_tokens_min=filter_background_tokens_min,
    )
    logger.info(
        f"Loaded {len(dataset)} examples from {dataset_path} ({dataset_split} split)"
    )

    # --- 2. Initialize Models ---
    embed_model = None
    if method in [
        "rag",
        "function_rag",
        "rag_with_pruner",
        "function_rag_with_pruner",
        "rag_with_silver_label_pruner",
        "function_rag_with_silver_label_pruner",
        "rag_with_pruner_rerank",
    ]:
        if embed_model_type == "bert":
            embed_model = BertBasedEmbedder(model_name=embed_model_name, max_length=512)
        elif embed_model_type == "qwen":
            embed_model = QwenEmbedder(model_name=embed_model_name, max_length=8192)
        elif embed_model_type == "bgem3":
            embed_model = BGEM3Embedder(model_name=embed_model_name, max_length=8192)

    if embed_model_name:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)  # Used to estimate the number of tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B"
        )  # Make a rough estimate

    reranker_model = None
    # Initialize reranker for rerank-only methods or methods that use reranker
    if method in ["rerank_only", "function_rerank_only"] or reranker_model_name:
        if not reranker_model_name:
            raise ValueError(
                f"{method} method requires reranker_model_name but it's not provided"
            )
        if reranker_model_type == "bert":
            reranker_model = BertBasedReranker(model_name=reranker_model_name)
        elif reranker_model_type == "bgev2m3":
            reranker_model = BGEV2M3Reranker(model_name=reranker_model_name)
        elif reranker_model_type == "qwen":
            reranker_model = QwenReranker(model_name=reranker_model_name)
        elif reranker_model_type == "online":
            reranker_model = OnlineReranker(
                api_base=rerank_api_base,
                aggregate_method=rerank_aggregate_method,
                language=rerank_language,
            )
        else:
            raise ValueError(
                f"Unknown reranker_model_type: {reranker_model_type}. Must be 'bert', 'bgev2m3', 'qwen', or 'online'"
            )

    compressor = None
    if method in ["longcodezip", "longcodezip_with_pruner"]:
        if not LONGCODEZIP_AVAILABLE:
            raise ImportError(
                "CodeCompressor is required for 'longcodezip' methods. Make sure code_compressor.py is available."
            )
        logger.info(f"Initializing CodeCompressor with model: {longcodezip_model_name}")
        compressor = CodeCompressor(model_name=longcodezip_model_name)
        logger.info("CodeCompressor initialized.")
    # Initialize LLMLingua PromptCompressor if needed
    llm_lingua_compressor = None
    if method == "llmlingua2":
        if not LLMLINGUA_AVAILABLE:
            raise ImportError(
                "LLMLingua is required for 'llmlingua2' method. Install with: pip install llmlingua"
            )
        logger.info(f"Initializing LLMLingua-2 with model: {llmlingua2_model_name}")
        llm_lingua_compressor = PromptCompressor(
            model_name=llmlingua2_model_name,
            use_llmlingua2=True,
        )
        logger.info("LLMLingua-2 initialized.")
    elif method == "longllmlingua":
        if not LLMLINGUA_AVAILABLE:
            raise ImportError(
                "LLMLingua is required for 'longllmlingua' method. Install with: pip install llmlingua"
            )
        logger.info("Initializing LongLLMLingua")
        llm_lingua_compressor = PromptCompressor(
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        logger.info("LongLLMLingua initialized.")

    # Initialize SelectiveContext if needed
    selective_context_compressor = None
    if method == "selective_context":
        if not SELECTIVE_CONTEXT_AVAILABLE:
            raise ImportError(
                "SelectiveContext is required for 'selective_context' method. Install with: pip install selective-context"
            )
        logger.info(
            f"Initializing SelectiveContext with model_type={selective_context_model_type}, lang={selective_context_lang}"
        )
        selective_context_compressor = SelectiveContext(
            model_type=selective_context_model_type,
            lang=selective_context_lang,
        )
        logger.info("SelectiveContext initialized.")
    

    summary_llm = None
    if method == "qwen3_summary":
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            summary_llm = LLM(
                model=summary_model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,
                tensor_parallel_size=1,
                max_model_len=32768,
            )
        finally:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info("Qwen3 Summary initialized.")

    # --- 3. Process the Specified Method ---
    logger.info(f"--- Processing Method: {method} ---")

    # Modify result directory based on method and parameters
    method_suffix = f"method_{method}"
    if method == "rag":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "function_rag":
        method_suffix += f"_lang{function_rag_language}_k{function_rag_top_k}"
    elif method == "rag_with_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
        if pruner_type == "online_rerank":
            th_str = f"{rerank_threshold:g}".replace(".", "p")
            method_suffix += f"_th{th_str}"
    elif method == "function_rag_with_pruner":
        method_suffix += f"_lang{function_rag_language}_k{function_rag_top_k}"
    elif method == "rag_with_embedder_reranker":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "rag_with_pruner_rerank":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "rerank_only":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_rerank{reranker_model_type}"
    elif method == "function_rerank_only":
        method_suffix += f"_lang{function_rag_language}_k{function_rag_top_k}_rerank{reranker_model_type}"
    elif method == "rag_with_silver_label_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "function_rag_with_silver_label_pruner":
        method_suffix += f"_lang{function_rag_language}_k{function_rag_top_k}"
    elif method == "llmlingua2":
        method_suffix += f"_rate{llmlingua2_rate}_model{llmlingua2_model_name.replace('/', '_slash_')}"
    elif method == "longllmlingua":
        method_suffix += f"_rate{longllmlingua_rate}_dyn{longllmlingua_dynamic_context_compression_ratio}"
    elif method == "selective_context":
        method_suffix += f"_model{selective_context_model_type}_lang{selective_context_lang}_ratio{selective_context_reduce_ratio}"
    elif method == "qwen3_summary":
        method_suffix += f"_model{summary_model_name}_max{summary_max_tokens}_temp{summary_temperature}"

    method_result_dir = os.path.join(result_dir, method_suffix)
    os.makedirs(method_result_dir, exist_ok=True)

    model_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}.jsonl",
    )
    score_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}-SCORES.json",
    )

    all_prompts = []
    original_data = []  # Store original data to merge with results

    # # Calculate GPU ranges for pruner (same as eval.py)
    # # SilverLabelPrunerModel in online mode doesn't need local GPU (uses external API)
    # pruner_tp = (
    #     pruner_tensor_parallel_size
    #     if "pruner" in method
    #     and pruner_type == "silver_label"
    #     and not pruner_online_mode
    #     else 0
    # )
    # pruner_start_gpu = 1
    # pruner_end_gpu = pruner_start_gpu + pruner_tp - 1

    # Initialize pruner if needed
    pruner = None
    if "pruner" in method:
        # Force silver_label pruner type and online mode for silver_label_pruner methods
        effective_pruner_type = pruner_type
        effective_pruner_online_mode = pruner_online_mode
        if method in [
            "rag_with_silver_label_pruner",
            "function_rag_with_silver_label_pruner",
        ]:
            if effective_pruner_type != "silver_label":
                logger.warning(
                    f"{method} method requires silver_label pruner type, but got {effective_pruner_type}. Forcing to silver_label."
                )
            effective_pruner_type = "silver_label"
            # # Force online mode (OpenAI API) for silver_label_pruner methods
            # if not effective_pruner_online_mode:
            #     logger.warning(
            #         f"{method} method requires online mode (OpenAI API), but online_mode is False. Forcing to online mode."
            #     )
            # effective_pruner_online_mode = True

        if effective_pruner_type == "online_rerank":
            pruner = OnlineRerankPrunerModel(
                api_base=rerank_api_base,
                threshold=rerank_threshold,
                always_keep_first_frags=rerank_always_keep_first_frags,
                aggregate_method=rerank_aggregate_method,
                language=rerank_language,
                batch_size=batch_size,
            )
        elif effective_pruner_type == "silver_label":
            if not effective_pruner_online_mode and not pruner_model_name:
                raise ValueError(
                    f"{method} method in offline mode requires pruner_model_name but it's not provided"
                )
            if effective_pruner_online_mode and not pruner_online_model_name:
                raise ValueError(
                    f"{method} method in online mode requires pruner_online_model_name but it's not provided"
                )

            original_cuda_visible = None
            if not effective_pruner_online_mode and pruner_tp > 0:
                original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                pruner_gpu_list = ",".join(
                    str(i) for i in range(pruner_start_gpu, pruner_end_gpu + 1)
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = pruner_gpu_list
                logger.info(
                    f"Setting CUDA_VISIBLE_DEVICES={pruner_gpu_list} for pruner initialization"
                )

            try:
                pruner = SilverLabelPrunerModel(
                    vllm_model_name=pruner_model_name
                    if not effective_pruner_online_mode
                    else None,
                    temperature=pruner_temperature,
                    max_tokens=pruner_max_tokens,
                    batch_size=batch_size,
                    tensor_parallel_size=pruner_tensor_parallel_size
                    if not effective_pruner_online_mode
                    else 1,
                    online_mode=effective_pruner_online_mode,
                    api_base=pruner_api_base,
                    api_key=pruner_api_key,
                    model_name=pruner_online_model_name
                    if effective_pruner_online_mode
                    else None,
                )
            finally:
                if original_cuda_visible is not None:
                    if original_cuda_visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            raise ValueError(
                f"Unknown pruner_type: {effective_pruner_type}. Must be 'silver_label' or 'online_rerank'"
            )

    # Prepare prompts based on method
    for i, example in enumerate(dataset):
        background_ctx = example["background_context"]
        current_func_ctx = example["current_function_context"]
        last_line = (
            current_func_ctx.strip().splitlines()[-1]
            if current_func_ctx.strip()
            else ""
        )
        query_code = (
            f"Find the lines most relevant to safely continue this function.\n"
            f"Keep definitions, control flow, and recent variable uses near the end.\n"
            f"Last line to complete: `{last_line}`\n"
            f"{current_func_ctx}"
        )
        ground_truth = example["gt"]  # This is the completion target
        # Determine language - assuming python for now based on dataset path
        language = (
            "python"  # IMPORTANT: Make dynamic if dataset contains multiple languages
        )

        kept_frags = example.get("kept_frags")
        base_code_for_syntax = (
            reconstruct_code_from_kept_frags(background_ctx, kept_frags)
            if kept_frags
            else background_ctx
        )


        

        retrieved_ctx = ""


        # AST analysis utils here
        chunk_syntax_list_random = None
        chunk_syntax_list_line_random = None
        if method in ["function_rag"]:
            raw_chunks = split_code_by_functions_standalone(
                base_code_for_syntax, language
            )
            
        elif method in [
            "function_rag_with_pruner",
            "function_rag_with_silver_label_pruner",
        ]:
            raw_chunks = split_code_by_functions_standalone(
                base_code_for_syntax, language
            )
            pruner: PrunerModel
            do_sort = True
            if method == "function_rag_with_silver_label_pruner":
                do_sort = False
            raw_responses = pruner.prune_batch(
                query=query_code,
                origin_codes=raw_chunks,
                batch_size=32,
                sort=do_sort,
                return_raw_body=True,
            )
            if syntax_check_chunk:
                chunk_syntax_list_random = []
                chunk_syntax_list_line_random = []
                for c, r in zip(raw_chunks, raw_responses):
                    if "kept_frags" not in r:
                        chunk_syntax_list_random.append(False)
                        chunk_syntax_list_line_random.append(False)
                        continue
                    origin_code = c
                    kept_frags = r["kept_frags"]
                    random_code = prune_random_unconstrained(origin_code)
                    ok_random = check_syntax_correctness(random_code, syntax_language)
                    chunk_syntax_list_random.append(ok_random)
                    line_random_code = prune_line_level_random_unconstrained(origin_code)
                    ok_line_random = check_syntax_correctness(
                        line_random_code, syntax_language
                    )
                    chunk_syntax_list_line_random.append(ok_line_random)

            kept_frags_list = [
                r["kept_frags"] for r in raw_responses if "kept_frags" in r
            ]
            raw_chunks = [
                reconstruct_code_from_kept_frags(c, k)
                for c, k in zip(raw_chunks, kept_frags_list)
            ]
        else:
            # HINT: we don't need rag's AST analysis anymore, only file rag
            raw_chunks = []
        
        
        
        # 2. Measure the time and number of tokens for context processing/compression.
        original_tokens = len(tokenizer.encode(background_ctx))
        start_context_prep_time = time.time()
        try:
            if method == "full":
                retrieved_ctx = background_ctx
                tokenized_context = tokenizer.encode(retrieved_ctx)
                if len(tokenized_context) > 32768 - 256:
                    logger.warning(
                        f"Context length exceeds 32768, truncating from the head. Original length: {len(tokenized_context)}, Truncated length: 32768"
                    )
                    retrieved_ctx = tokenizer.decode(
                        tokenized_context[-(32768 - 256) :]
                    )
            elif method == "rag":
                if not embed_model:
                    raise ValueError(
                        "RAG method selected but embedding model not initialized."
                    )
                retrieved_ctx = rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                )
            elif method == "function_rag":
                if not embed_model:
                    raise ValueError(
                        "Function RAG method selected but embedding model not initialized."
                    )
                retrieved_ctx = function_rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    language=function_rag_language,
                    top_k=function_rag_top_k,
                )
            elif method == "rag_with_pruner":
                if not embed_model:
                    raise ValueError(
                        "RAG with pruner method selected but embedding model not initialized."
                    )
                retrieved_ctx = rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )

            elif method == "function_rag_with_pruner":
                if not embed_model:
                    raise ValueError(
                        "Function RAG with pruner method selected but embedding model not initialized."
                    )
                retrieved_ctx = function_rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    language=function_rag_language,
                    top_k=function_rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "rag_with_embedder_reranker":
                if not embed_model:
                    raise ValueError(
                        "RAG with embedder and reranker method selected but embedding model not initialized."
                    )
                if not reranker_model:
                    raise ValueError(
                        "RAG with embedder and reranker method selected but reranker model not initialized."
                    )
                retrieved_ctx, raw_chunks = rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    reranker=reranker_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    return_chunks=True,
                )
            elif method == "rag_with_pruner_rerank":
                if not embed_model:
                    raise ValueError(
                        "RAG with pruner and reranker method selected but embedding model not initialized."
                    )
                if not pruner:
                    raise ValueError(
                        "RAG with pruner and reranker method selected but pruner not initialized."
                    )
                # Use adapter-based RAG with pruner (pruner itself acts as reranker)
                # Sorting is controlled by pruner_sort parameter
                retrieved_ctx, raw_chunks = rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=True,
                    return_chunks=True,
                )
            elif method == "rerank_only":
                if not reranker_model:
                    raise ValueError(
                        "Rerank-only method selected but reranker model not initialized."
                    )
                retrieved_ctx, raw_chunks = rerank_only_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    reranker=reranker_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    return_chunks=True,
                )
            elif method == "function_rerank_only":
                if not reranker_model:
                    raise ValueError(
                        "Function rerank-only method selected but reranker model not initialized."
                    )
                retrieved_ctx, raw_chunks = function_rerank_only_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    reranker=reranker_model,
                    language=function_rag_language,
                    top_k=function_rag_top_k,
                    return_chunks=True,
                )
            elif method == "rag_with_silver_label_pruner":
                if not embed_model:
                    raise ValueError(
                        "RAG with silver label pruner method selected but embedding model not initialized."
                    )
                if not pruner:
                    raise ValueError(
                        "RAG with silver label pruner method selected but pruner not initialized."
                    )
                # Use RAG + SilverLabelPrunerModel (vLLM mode)
                retrieved_ctx, raw_chunks = rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                    return_chunks=True,
                )
            elif method == "function_rag_with_silver_label_pruner":
                if not embed_model:
                    raise ValueError(
                        "Function RAG with silver label pruner method selected but embedding model not initialized."
                    )
                if not pruner:
                    raise ValueError(
                        "Function RAG with silver label pruner method selected but pruner not initialized."
                    )
                # Use function-level RAG + SilverLabelPrunerModel (vLLM mode)
                retrieved_ctx, raw_chunks = function_rag_retrieve(
                    background_code=background_ctx,
                    query_code=query_code,
                    embedder=embed_model,
                    language=function_rag_language,
                    top_k=function_rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                    return_chunks=True,
                )
            elif method == "no_context":
                retrieved_ctx = ""
            elif method == "longcodezip":
                if not compressor:
                    raise ValueError(
                        "LongCodeZip method selected but compressor not initialized."
                    )
                # Use CodeCompressor to compress the code
                code_to_compress = background_ctx
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        result = compressor.compress_code_file(
                            code=code_to_compress,
                            query=query_code,
                            instruction="",
                            rate=longcodezip_rate,
                            rank_only=longcodezip_rank_only,
                        )
                        raw_chunks = result.get("compressed_chunks", [])
                        retrieved_ctx = result.get("compressed_code", code_to_compress)
                    except Exception as e:
                        logger.warning(
                            f"Error compressing code with CodeCompressor for example {i}: {e}. Using original code."
                        )
                        retrieved_ctx = code_to_compress
            elif method == "longcodezip_with_pruner":
                if not compressor:
                    raise ValueError(
                        "LongCodeZip with pruner method selected but compressor not initialized."
                    )
                if not pruner:
                    raise ValueError(
                        "LongCodeZip with pruner method selected but pruner not initialized."
                    )
                # Use CodeCompressor with pruner to compress the code
                code_to_compress = background_ctx
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        result = compressor.compress_code_file(
                            code=code_to_compress,
                            query=query_code,
                            instruction="",
                            rate=longcodezip_rate,
                            rank_only=longcodezip_rank_only,
                            pruner=pruner,
                        )
                        retrieved_ctx = result.get("compressed_code", code_to_compress)
                        raw_chunks = result.get("compressed_chunks", [])
                    except Exception as e:
                        logger.warning(
                            f"Error compressing code with CodeCompressor+pruner for example {i}: {e}. Using original code."
                        )
                        retrieved_ctx = code_to_compress
            elif method == "llmlingua2":
                if not llm_lingua_compressor:
                    raise ValueError(
                        "LLMLingua-2 method selected but compressor not initialized."
                    )
                code_to_compress = background_ctx
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        chunks = chunk_by_token_count(
                            code_to_compress,
                            llmlingua_tokenizer,
                            max_tokens=512,
                            overlap_tokens=0,
                        )

                        # Compress each chunk separately
                        compressed_chunks = []
                        for chunk in chunks:
                            compress_kwargs = {
                                "context": [chunk],  # Must be a list!
                                "rate": llmlingua2_rate,
                            }
                            if llmlingua2_force_tokens:
                                compress_kwargs["force_tokens"] = (
                                    llmlingua2_force_tokens
                                )

                            compressed_result = llm_lingua_compressor.compress_prompt(
                                **compress_kwargs
                            )
                            # LLMLingua-2 always returns a dict
                            if isinstance(compressed_result, dict):
                                # Result has "compressed_prompt" or "compressed_prompt_list"
                                if "compressed_prompt" in compressed_result:
                                    compressed_chunks.append(
                                        compressed_result["compressed_prompt"]
                                    )
                                elif "compressed_prompt_list" in compressed_result:
                                    # If only list available, join them
                                    compressed_list = compressed_result.get(
                                        "compressed_prompt_list", []
                                    )
                                    if compressed_list:
                                        compressed_chunks.append(
                                            "\n\n".join(compressed_list)
                                        )
                        raw_chunks = compressed_chunks
                        # Combine all compressed chunks
                        retrieved_ctx = (
                            "\n\n".join(compressed_chunks)
                            if compressed_chunks
                            else code_to_compress
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error compressing code with LLMLingua-2 for example {i}: {e}. Using original code."
                        )
                        retrieved_ctx = code_to_compress
            elif method == "longllmlingua":
                if not llm_lingua_compressor:
                    raise ValueError(
                        "LongLLMLingua method selected but compressor not initialized."
                    )
                # Use LongLLMLingua to compress the prompt
                code_to_compress = background_ctx
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        chunks = chunk_by_token_count(
                            code_to_compress,
                            llmlingua_tokenizer,
                            max_tokens=512,
                            overlap_tokens=0,
                        )

                        # Note: LongLLMLingua can handle multiple contexts
                        compressed_result = llm_lingua_compressor.compress_prompt(
                            context=chunks,  # Pass all chunks as list
                            question=current_func_ctx,
                            rate=longllmlingua_rate,
                            condition_in_question=longllmlingua_condition_in_question,
                            reorder_context=longllmlingua_reorder_context,
                            dynamic_context_compression_ratio=longllmlingua_dynamic_context_compression_ratio,
                            condition_compare=longllmlingua_condition_compare,
                            context_budget=longllmlingua_context_budget,
                            rank_method="longllmlingua",
                        )

                        if isinstance(compressed_result, dict):
                            retrieved_ctx = compressed_result.get(
                                "compressed_prompt", code_to_compress
                            )
                        else:
                            # Unexpected return type
                            raise ValueError(
                                "Unexpected return type from LongLLMLingua"
                            )
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        logger.warning(
                            f"Error compressing code with LongLLMLingua for example {i}: {e}. Using original code."
                        )
                        retrieved_ctx = code_to_compress
            elif method == "selective_context":
                if not selective_context_compressor:
                    raise ValueError(
                        "SelectiveContext method selected but compressor not initialized."
                    )
                # Use SelectiveContext to compress the code
                code_to_compress = background_ctx
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    sc_tokenizer = selective_context_compressor.tokenizer
                    sc_model_max_len = 1024
                    if hasattr(sc_tokenizer, "model_max_length"):
                        sc_model_max_len = sc_tokenizer.model_max_length

                    def truncate_to_model(text):
                        ids = sc_tokenizer.encode(text, add_special_tokens=False)
                        ids = ids[
                            : max(sc_model_max_len - 24, 1)
                        ]  # 24 for selective context added prompt
                        return sc_tokenizer.decode(ids, skip_special_tokens=True)

                    try:
                        # SelectiveContext (especially with GPT2) has limitations on input length
                        # GPT2 max position is 1024, so we need to chunk long inputs
                        base_chunk_len = 1000
                        max_chunk_length = min(sc_model_max_len, base_chunk_len)

                        # Check if input needs chunking
                        input_tokens = len(sc_tokenizer.encode(code_to_compress))
                        if input_tokens <= max_chunk_length:
                            # Input is short enough, compress directly
                            safe_text = truncate_to_model(code_to_compress)
                            compressed_result, _ = selective_context_compressor(
                                safe_text,
                                reduce_ratio=selective_context_reduce_ratio,
                            )
                            retrieved_ctx = compressed_result
                            raw_chunks = [retrieved_ctx]
                        else:
                            # Input is too long, split into chunks and compress each chunk
                            logger.info(
                                f"Input too long ({input_tokens} tokens) for SelectiveContext "
                                f"with {selective_context_model_type} (max ~{max_chunk_length}). "
                                f"Splitting into chunks and compressing each chunk separately."
                            )

                            # Split code into chunks by token length (no overlap)
                            all_tokens = sc_tokenizer.encode(code_to_compress)
                            num_chunks = (
                                len(all_tokens) + max_chunk_length - 1
                            ) // max_chunk_length

                            compressed_chunks = []
                            for chunk_idx in range(num_chunks):
                                start_idx = chunk_idx * max_chunk_length
                                end_idx = min(
                                    start_idx + max_chunk_length, len(all_tokens)
                                )

                                # Extract chunk tokens and decode back to text
                                chunk_tokens = all_tokens[start_idx:end_idx]
                                chunk_text = sc_tokenizer.decode(
                                    chunk_tokens, skip_special_tokens=True
                                )
                                try:
                                    # Compress this chunk
                                    compressed_chunk, _ = selective_context_compressor(
                                        chunk_text,
                                        reduce_ratio=selective_context_reduce_ratio,
                                    )
                                    compressed_chunks.append(compressed_chunk)
                                    raw_chunks.append(compressed_chunk)
                                except Exception as chunk_error:
                                    logger.warning(
                                        f"Error compressing chunk {chunk_idx + 1}/{num_chunks}: {chunk_error}. "
                                        f"Using original chunk."
                                    )
                                    compressed_chunks.append(chunk_text)
                                    raw_chunks.append(chunk_text)

                            # Combine compressed chunks
                            if compressed_chunks:
                                retrieved_ctx = "\n\n".join(compressed_chunks)
                                logger.info(
                                    f"Compressed {num_chunks} chunks. "
                                    f"Original: {input_tokens} tokens, "
                                    f"Compressed: {len(tokenizer.encode(retrieved_ctx))} tokens"
                                )
                            else:
                                logger.warning(
                                    "All chunks failed to compress, using original code."
                                )
                                retrieved_ctx = code_to_compress
                    except Exception as e:
                        logger.warning(
                            f"Error compressing code with SelectiveContext for example {i}: {e}. Using original code."
                        )
                        retrieved_ctx = code_to_compress
            elif method == "qwen3_summary":
                if summary_llm is None:
                    raise ValueError("summary_llm not initialized")

                summary_prompt = (
                    "You will see code background and the current part to complete. Please extract only the key points or code snippets most relevant to solving the task, and keep it concise.\n\n"
                    "[QUESTION]\n"
                    f"{query_code}\n\n"
                    "[CODE BACKGROUND]\n"
                    f"{background_ctx}\n"
                )

                sampling_params = SamplingParams(
                    temperature=summary_temperature,
                    top_p=0.9,
                    max_tokens=summary_max_tokens,
                )
                summary_outputs = summary_llm.generate(
                    [summary_prompt], sampling_params, use_tqdm=False
                )
                retrieved_ctx = summary_outputs[0].outputs[0].text.strip()
                raw_chunks = [retrieved_ctx]
            else:
                raise ValueError(f"Unknown method: {method}")


            # Record context preparation time
            end_context_prep_time = time.time()
            performance_metrics["context_preparation_time_seconds"].append(
                end_context_prep_time - start_context_prep_time
            )

            # Record the number of tokens and compression ratio after processing.
            processed_tokens = len(tokenizer.encode(retrieved_ctx))
            performance_metrics["original_tokens"].append(original_tokens)
            performance_metrics["processed_tokens"].append(processed_tokens)
            ratio = processed_tokens / original_tokens if original_tokens > 0 else 0
            performance_metrics["compression_ratio"].append(ratio)
            performance_metrics["example_id"].append(example.get("id", i))

            prompt = retrieved_ctx + "\n\n" + current_func_ctx

            is_ok = None
            if syntax_check:
                is_ok = check_syntax_correctness(base_code_for_syntax, syntax_language)
                syntax_flags.append(is_ok)
                syntax_total += 1
                syntax_correct += int(is_ok)
            chunk_syntax_list = None
            if syntax_check_chunk:
                chunk_list = raw_chunks
                chunk_syntax_list = []
                for ch in chunk_list:
                    if not ch or not ch.strip():
                        # Empty blocks are considered passed
                        chunk_syntax_list.append(True)
                        chunk_syntax_total += 1
                        chunk_syntax_correct += 1
                        continue
                    ok = check_syntax_correctness(ch, syntax_language)
                    chunk_syntax_list.append(ok)
                    chunk_syntax_total += 1
                    chunk_syntax_correct += int(ok)

            if chunk_syntax_list_random:
                for ok in chunk_syntax_list_random:
                    chunk_syntax_random_total += 1
                    chunk_syntax_random_correct += int(ok)
            if chunk_syntax_list_line_random:
                for ok in chunk_syntax_list_line_random:
                    chunk_syntax_line_random_total += 1
                    chunk_syntax_line_random_correct += int(ok)

            if "prune" in method and pruner is not None:
                prompt = prompt
            
            prompt = prompt.rstrip()
            if not prompt.endswith("\n"):
                prompt += "\n"
            all_prompts.append(prompt)
            original_data.append(
                {
                    "id": example.get("id", i),
                    "gt": ground_truth,
                    "original_background_context": background_ctx,
                    "original_current_function_context": current_func_ctx,
                    "language": language,  # Store language if needed later
                    "retrieved_context": retrieved_ctx,
                    "syntax_correct": is_ok,
                    "syntax_correct_chunks": chunk_syntax_list
                    if chunk_syntax_list is not None
                    else None,
                    "syntax_correct_chunks_random": chunk_syntax_list_random
                    if chunk_syntax_list_random is not None
                    else None,
                    "syntax_correct_chunks_line_random": chunk_syntax_list_line_random
                    if chunk_syntax_list_line_random is not None
                    else None,
                    "base_code_for_syntax": base_code_for_syntax,
                }
            )
        except Exception as e:
            logger.warning(
                f"Error processing example {i} (ID: {example.get('id', 'N/A')}) for method {method}: {e}",
                exc_info=True,
            )
            # Record a placeholder even if an error occurs.
            performance_metrics["context_preparation_time_seconds"].append(-1)
            performance_metrics["original_tokens"].append(original_tokens)
            performance_metrics["processed_tokens"].append(-1)
            performance_metrics["compression_ratio"].append(-1)
            performance_metrics["example_id"].append(example.get("id", i))
            continue  # Skip this example

    # --- 4. Clean up Compression/Embedding Models ---
    logger.info("Freeing up GPU memory from compression/embedding models")
    if embed_model:
        del embed_model
    if compressor:
        del compressor
    if llm_lingua_compressor:
        del llm_lingua_compressor
    if selective_context_compressor:
        del selective_context_compressor
    if summary_llm:
        del summary_llm
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory freed")

    # --- 5. Initialize Generation LLM ---
    # Check if there are any prompts to process before initializing LLM
    if not all_prompts:
        logger.error(
            f"No valid prompts were prepared for method {method}. Skipping generation and scoring."
        )
        return

    # Use CUDA_VISIBLE_DEVICES to control GPU allocation
    # This avoids conflicts with other processes (like rerank server) on the same machine
    # IMPORTANT: Set CUDA_VISIBLE_DEVICES before initializing LLM, as vLLM will spawn
    # subprocesses that inherit this environment variable
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if tensor_parallel_size > 1:
        main_gpu_list = ",".join(
            str(i) for i in range(main_start_gpu, main_end_gpu + 1)
        )
    else:
        main_gpu_list = str(main_start_gpu)

    # Set CUDA_VISIBLE_DEVICES before LLM initialization
    # This ensures vLLM subprocesses see the correct GPU mapping
    os.environ["CUDA_VISIBLE_DEVICES"] = main_gpu_list
    logger.info(
        f"Setting CUDA_VISIBLE_DEVICES={main_gpu_list} for main model initialization"
    )
    logger.info(
        f"Note: vLLM will see GPU 0-{tensor_parallel_size - 1} mapped to physical GPUs {main_start_gpu}-{main_end_gpu}"
    )

    try:
        logger.info(f"Initializing generation LLM: {model_name}")
        # Clear CUDA cache to ensure clean state
        torch.cuda.empty_cache()
        llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            # hf_overrides={
            #     "rope_scaling": {
            #         "rope_type": "yarn",
            #         "factor": 2.0,
            #         "original_max_position_embeddings": 32768,
            #     }
            # },
            max_model_len=32768,
            enforce_eager=True,
        )
        logger.info(f"Generation LLM {model_name} initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise
    finally:
        # Restore original CUDA_VISIBLE_DEVICES after LLM initialization
        # Note: This won't affect already-spawned vLLM subprocesses
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # --- 6. Generate Completions ---
    all_outputs = []
    logger.info(f"Generating completions for {len(all_prompts)} prompts...")
    for i in tqdm(
        range(0, len(all_prompts), batch_size),
        desc=f"Generating completions for {method}",
    ):
        batch_prompts = all_prompts[i : i + batch_size]
        if not batch_prompts:
            continue

        try:
            # 3. Measurement Model Inference Time
            start_inference_time = time.time()

            batch_outputs = generate_completions(
                llm, batch_prompts, max_new_tokens=max_new_tokens
            )

            end_inference_time = time.time()

            # Calculate and record the average inference time for each sample
            duration = end_inference_time - start_inference_time
            time_per_prompt = duration / len(batch_prompts) if batch_prompts else 0
            performance_metrics["inference_time_per_example_seconds"].extend(
                [time_per_prompt] * len(batch_prompts)
            )

            all_outputs.extend(batch_outputs)
        except Exception as e:
            logger.error(
                f"Error during generation for batch starting at index {i}: {e}"
            )
            all_outputs.extend(["ERROR_GENERATING"] * len(batch_prompts))

    # --- 7. Evaluate and Save Results ---
    model_outputs_data = []
    total_es = 0
    total_em = 0
    valid_scores = 0

    if len(all_outputs) != len(original_data):
        logger.warning(
            f"Warning: Mismatch between generated outputs ({len(all_outputs)}) and original data ({len(original_data)}). Scores might be inaccurate."
        )
        min_len = min(len(all_outputs), len(original_data))
        all_outputs = all_outputs[:min_len]
        original_data = original_data[:min_len]
        all_prompts = all_prompts[:min_len]

    logger.info(
        f"Calculating scores and saving results for {len(all_outputs)} examples..."
    )
    # make sure that the path exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "w") as f_out:
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            # Ensure index is valid for original_data and all_prompts
            if i >= len(original_data) or i >= len(all_prompts):
                logger.error(
                    f"Index {i} out of bounds after potential mismatch alignment. Stopping result processing."
                )
                break
            orig_data = original_data[i]
            prompt = all_prompts[i]
            gt = orig_data["gt"]

            # Calculate compression ratio: prompt_length / (background + current) length
            original_background_length = len(
                orig_data.get("original_background_context", "")
            )
            original_current_length = len(
                orig_data.get("original_current_function_context", "")
            )
            original_total_length = original_background_length + original_current_length

            compression_ratio = 1.0
            if original_total_length > 0:
                compression_ratio = len(prompt) / original_total_length

            result = {
                **orig_data,
                "prompt": prompt,
                "output": output,
                "compression_ratio": compression_ratio,
            }

            if syntax_check and "syntax_correct" in orig_data:
                result["syntax_correct"] = orig_data["syntax_correct"]
            if syntax_check_chunk and "syntax_correct_chunks" in orig_data:
                result["syntax_correct_chunks"] = orig_data["syntax_correct_chunks"]

            es = 0
            em = 0

            if output != "ERROR_GENERATING" and gt is not None:
                try:
                    es = compute_ES(gt, output)
                    em = compute_EM(gt, output)
                    total_es += es
                    total_em += em
                    valid_scores += 1
                except Exception as e:
                    logger.error(f"Error scoring example {orig_data.get('id', i)}: {e}")

            result["es"] = es
            result["em"] = em
            model_outputs_data.append(result)
            f_out.write(json.dumps(result) + "\n")

    logger.info(f"Raw results saved to {model_output_path}")

    avg_es = (total_es / valid_scores) if valid_scores > 0 else 0
    avg_em = (total_em / valid_scores) if valid_scores > 0 else 0
    avg_context_ratio = (
        (
            sum(result.get("context_ratio", 0) for result in model_outputs_data)
            / valid_scores
        )
        if valid_scores > 0
        else 0
    )

    # Update the parameters dictionary in scores
    scores = {
        "model_name": model_name,
        "method": method,
        "num_examples_scored": valid_scores,
        "num_examples_total": len(
            original_data
        ),  # Use length of original_data before potential alignment issues
        "average_es": avg_es,
        "average_em": avg_em,
        "average_context_ratio": avg_context_ratio,
        "parameters": {
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "filter_current_lines_max": filter_current_lines_max,
            "filter_background_tokens_min": filter_background_tokens_min,
            "embed_model_name": embed_model_name
            if method
            in ["rag", "function_rag", "rag_with_pruner", "function_rag_with_pruner"]
            else None,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            # RAG specific params
            "rag_window_size": rag_window_size
            if method in ["rag", "rag_with_pruner"]
            else None,
            "rag_overlap": rag_overlap
            if method in ["rag", "rag_with_pruner"]
            else None,
            "rag_top_k": rag_top_k
            if method in ["rag", "rag_with_pruner", "rerank_only"]
            else None,
            # Function RAG params
            "function_rag_language": function_rag_language
            if method
            in ["function_rag", "function_rag_with_pruner", "function_rerank_only"]
            else None,
            "function_rag_top_k": function_rag_top_k
            if method
            in ["function_rag", "function_rag_with_pruner", "function_rerank_only"]
            else None,
            # Rerank-only params
            "reranker_model_name": reranker_model_name
            if method in ["rerank_only", "function_rerank_only"]
            else None,
            "reranker_model_type": reranker_model_type
            if method in ["rerank_only", "function_rerank_only"]
            else None,
            "rag_window_size_rerank_only": rag_window_size
            if method == "rerank_only"
            else None,
            "rag_overlap_rerank_only": rag_overlap if method == "rerank_only" else None,
            # Pruner params
            "pruner_model_name": pruner_model_name
            if method in ["rag_with_pruner", "function_rag_with_pruner"]
            else None,
            "pruner_online_model_name": pruner_online_model_name
            if method
            in ["rag_with_silver_label_pruner", "function_rag_with_silver_label_pruner"]
            else None,
            "pruner_tensor_parallel_size": pruner_tensor_parallel_size
            if method
            in [
                "rag_with_pruner",
                "function_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "function_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_temperature": pruner_temperature
            if method
            in [
                "rag_with_pruner",
                "function_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "function_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_max_tokens": pruner_max_tokens
            if method
            in [
                "rag_with_pruner",
                "function_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "function_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_type": "silver_label"
            if method
            in ["rag_with_silver_label_pruner", "function_rag_with_silver_label_pruner"]
            else None,
            "pruner_online_mode": True
            if method
            in ["rag_with_silver_label_pruner", "function_rag_with_silver_label_pruner"]
            else None,
            "llmlingua2_model_name": llmlingua2_model_name
            if method == "llmlingua2"
            else None,
            "llmlingua2_rate": llmlingua2_rate if method == "llmlingua2" else None,
            "llmlingua2_force_tokens": llmlingua2_force_tokens
            if method == "llmlingua2"
            else None,
            "longllmlingua_rate": longllmlingua_rate
            if method == "longllmlingua"
            else None,
            "longllmlingua_condition_in_question": longllmlingua_condition_in_question
            if method == "longllmlingua"
            else None,
            "longllmlingua_reorder_context": longllmlingua_reorder_context
            if method == "longllmlingua"
            else None,
            "longllmlingua_dynamic_context_compression_ratio": longllmlingua_dynamic_context_compression_ratio
            if method == "longllmlingua"
            else None,
            "longllmlingua_condition_compare": longllmlingua_condition_compare
            if method == "longllmlingua"
            else None,
            "longllmlingua_context_budget": longllmlingua_context_budget
            if method == "longllmlingua"
            else None,
            "selective_context_model_type": selective_context_model_type
            if method == "selective_context"
            else None,
            "selective_context_lang": selective_context_lang
            if method == "selective_context"
            else None,
            "selective_context_reduce_ratio": selective_context_reduce_ratio
            if method == "selective_context"
            else None,
            "compression_ratio": compression_ratio,
        },
    }

    logger.info(
        f"Method {method}: Avg ES = {avg_es:.2f}, Avg EM = {avg_em:.2f} ({valid_scores}/{len(original_data)} scored)"
    )
    if syntax_check:
        scores["syntax_correctness"] = {
            "accuracy": syntax_correct / syntax_total if syntax_total else 0.0,
            "correct": syntax_correct,
            "total": syntax_total,
        }
    if syntax_check_chunk:
        scores["syntax_correctness_chunks"] = {
            "accuracy": chunk_syntax_correct / chunk_syntax_total
            if chunk_syntax_total
            else 0.0,
            "correct": chunk_syntax_correct,
            "total": chunk_syntax_total,
        }
        if chunk_syntax_random_total > 0:
            scores["syntax_correctness_chunks_random"] = {
                "accuracy": chunk_syntax_random_correct / chunk_syntax_random_total
                if chunk_syntax_random_total
                else 0.0,
                "correct": chunk_syntax_random_correct,
                "total": chunk_syntax_random_total,
            }
        if chunk_syntax_line_random_total > 0:
            scores["syntax_correctness_chunks_line_random"] = {
                "accuracy": chunk_syntax_line_random_correct
                / chunk_syntax_line_random_total
                if chunk_syntax_line_random_total
                else 0.0,
                "correct": chunk_syntax_line_random_correct,
                "total": chunk_syntax_line_random_total,
            }
    save_json(scores, score_output_path)
    logger.info(f"Scores saved to {score_output_path}")

    # 4. Save the performance metrics file.
    performance_metrics_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}-PERFORMANCE.json",
    )
    save_json(performance_metrics, performance_metrics_output_path)
    logger.info(f"Performance metrics saved to {performance_metrics_output_path}")

    # (Optional) Print average value
    logger.info("\n--- Performance Metrics (Averages) ---")
    for key, values in performance_metrics.items():
        # Filter out ID and error value -1
        valid_values = [v for v in values if v != -1]
        if key != "example_id" and valid_values:
            avg = np.mean(valid_values)
            logger.info(f"Average {key}: {avg:.4f}")
    logger.info("--------------------------------------\n")

    logger.info("Evaluation complete.")
    # Clean up LLM explicitly
    if "llm" in locals() and llm is not None:
        del llm
        logger.info("Generation LLM deleted.")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    fire.Fire(evaluate_completion)
