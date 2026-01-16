import logging
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import fire
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from loguru import logger
import gc
from typing import Optional
import re
import sys
from pathlib import Path
from pydantic import BaseModel
import time
import numpy as np
from typing import List, Protocol
import socket

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
    logger.setLevel(logging.INFO)

# Import SelectiveContext
try:
    from selective_context import SelectiveContext

    SELECTIVE_CONTEXT_AVAILABLE = True
except ImportError:
    SELECTIVE_CONTEXT_AVAILABLE = False
    logger.warning(
        "SelectiveContext not available. Install with: pip install selective-context"
    )

# Add parent directory to path to import model and RAG functions
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RAG functions from LCC/main.py
from model import SilverLabelPrunerModel, OnlineRerankPrunerModel
from embedder import BertBasedEmbedder, QwenEmbedder, BGEM3Embedder, EmbedderAdapter
from reranker import (
    BertBasedReranker,
    BGEV2M3Reranker,
    QwenReranker,
    OnlineReranker,
    RerankerAdapter,
)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# HINT: this is fucking hack for running llmlingua on offline machine, need fucking hack the lib from `self.oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")` -> `self.oai_tokenizer = tiktoken.get_encoding('cl100k_base')` and download the tokenizer first
tiktoken_cache_dir = "./tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
import tiktoken

llmlingua_tokenizer = tiktoken.get_encoding("cl100k_base")


class PrunerModel(Protocol):
    def prune(query: str, origin_code: str) -> str: ...
    def prune_batch(
        query: str, origin_codes: List[str], batch_size=16, sort: bool = False
    ) -> List[str]: ...


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
) -> str:
    """
    RAG retrieval using EmbedderAdapter and optional RerankerAdapter.

    Args:
        background_code: Full code to retrieve from
        query_code: Query text
        embedder: EmbedderAdapter instance for embedding
        reranker: Optional RerankerAdapter instance for reranking
        window_size: Window size for chunking
        overlap: Overlap size for chunking
        top_k: Number of top chunks to retrieve
        pruner: Optional pruner for further filtering
        pre_prune: Whether to prune before retrieval
        compression_ratio: Compression ratio for pruner

    Returns:
        Combined relevant code
    """
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

    # Optional post-prune
    if pruner:
        retrieved_chunks = pruner.prune_batch(
            query_code, retrieved_chunks, sort=pruner_sort
        )

    retrieved_chunks = retrieved_chunks[:top_k]
    combined_code = "\n\n".join(retrieved_chunks)

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
) -> str:
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
    return combined_code


def _normalize_context_budget(val):
    if val is None:
        return "+0"
    if isinstance(val, (int, float)):
        return f"{int(val):+d}"
    # If it's already a string, return it directly (keeping the syntax +100/-100).
    return str(val)


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


# Helper function for function-level rerank-only retrieval
def function_rerank_only_retrieve(
    background_code: str,
    query_code: str,
    reranker: RerankerAdapter,
    language: str,
    top_k: int,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
) -> str:
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
    return combined_code


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


class LongCodeQADataItem(BaseModel):
    prompt: str
    correct_letter: str
    repo_text: Optional[str] = ""
    question: Optional[str] = ""
    repo: Optional[str] = ""
    prompt_goal: Optional[str] = ""
    is_hard: Optional[str] = ""

    class Config:
        extra = "allow"  # You only need these fields; you don't need to worry about the others.


class LongCodeQAAnswer(BaseModel):
    """Structured output format for model responses."""

    reason: str  # Brief reasoning (1-2 sentences)
    final_answer: str  # A, B, C, or D


class LongCodeQADataset:
    def __init__(self, data_file_jsonl: str):
        self.data_file_jsonl = data_file_jsonl
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.data_file_jsonl, "r") as f:
            text = f.read()
        if not text.strip():
            return data

        # If it's a JSON array, parse it all at once.
        stripped = text.lstrip()
        if stripped.startswith("["):
            arr = json.loads(text)
            for item_data in arr:
                data.append(LongCodeQADataItem.model_validate(item_data))
            return data

        # Otherwise, parse line by line using JSONL.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item_data = json.loads(line)
            data.append(LongCodeQADataItem.model_validate(item_data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_json(data: dict, file_path: str):
    """Saves dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def generate_completions(
    llm, batch_prompts, max_new_tokens=128, use_guided_decoding=True
):
    """Generate completions for batch with structured JSON output."""
    if use_guided_decoding:
        # Use guided decoding for structured JSON output
        json_schema = LongCodeQAAnswer.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            guided_decoding=guided_decoding_params,
            temperature=0,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=max_new_tokens
        )
    msgs = []
    for prompt in batch_prompts:
        msgs.append(
            [
                {
                    "role": "system",
                    "content": "You are a professional programming assistant helps in code repo query answering.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    batch_outputs = llm.chat(msgs, sampling_params, use_tqdm=False)
    return [x.outputs[0].text for x in batch_outputs]


def extract_json_from_output(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown code blocks."""
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)
    # Find outermost {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")
    return json.loads(match.group(0))


def extract_answer_and_reasoning(output: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract answer letter (A, B, C, or D) and reasoning from structured JSON output.
    Returns (answer_letter, reasoning)
    """
    try:
        # Try to parse as JSON first
        result = extract_json_from_output(output)
        reason = result.get("reason", "")
        final_answer = result.get("final_answer", "").upper().strip()

        # Validate answer is A, B, C, or D
        if final_answer in ["A", "B", "C", "D"]:
            return final_answer, reason
        else:
            logger.warning(f"Invalid final_answer in JSON: {final_answer}")
            return None, reason
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        # Fallback to regex extraction if JSON parsing fails
        logger.warning(f"Failed to parse JSON output, falling back to regex: {e}")
        # Try to find letter in various formats
        patterns = [
            r"\b([ABCD])\b",  # Standalone A/B/C/D
            r"(?:answer|Answer|选项|答案是)[:\s]*([ABCD])",  # "Answer: A" or "The answer is A"
            r"([ABCD])(?:\.|\)|）)",  # "A." or "A)" or "A）"
            r"选择[:\s]*([ABCD])",  # "Choose A"
        ]

        answer_letter = None
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in ["A", "B", "C", "D"]:
                    answer_letter = letter
                    break

        # If no pattern matches, try to find the first A/B/C/D in the output
        if not answer_letter:
            match = re.search(r"[ABCD]", output.upper())
            if match:
                answer_letter = match.group(0)

        # Extract reasoning - everything before the answer, or after if answer is early
        reasoning = None
        if answer_letter:
            # Try to find reasoning before the answer
            answer_pattern = rf"\b{answer_letter}\b"
            answer_match = re.search(answer_pattern, output, re.IGNORECASE)
            if answer_match:
                # Get text before answer as reasoning
                reasoning_text = output[: answer_match.start()].strip()
                if reasoning_text:
                    reasoning = reasoning_text
                # Also try to get text after answer (in case reasoning comes after)
                after_text = output[answer_match.end() :].strip()
                if after_text and not reasoning:
                    reasoning = after_text

        return answer_letter, reasoning


def extract_answer_letter(output: str) -> Optional[str]:
    """Extract only the answer letter for backward compatibility."""
    answer, _ = extract_answer_and_reasoning(output)
    return answer


def parse_repo_files(repo_text: str) -> dict[str, str]:
    """
    Parse repository text into individual files.
    Files are marked with [start of {filename}] and [end of {filename}]

    Returns:
        dict mapping filename to file content
    """
    files = {}
    # Pattern to match [start of filename] ... content ... [end of filename]
    pattern = r"\[start of ([^\]]+)\](.*?)\[end of \1\]"

    matches = re.finditer(pattern, repo_text, re.DOTALL)
    for match in matches:
        filename = match.group(1).strip()
        content = match.group(2).strip()
        files[filename] = content

    return files


def file_level_rag_retrieve(
    background_code: str,
    query_code: str,
    embedder: EmbedderAdapter,
    reranker: Optional[RerankerAdapter],
    window_size: int,
    overlap: int,
    top_k: int,
    pruner: Optional[PrunerModel] = None,
    pruner_sort: bool = False,
) -> str:
    """
    File-level RAG: First split by files, then use sliding window within each file.
    Chunks background, embeds chunks and query, retrieves top_k similar chunks.
    """
    if not background_code.strip():
        return ""

    # Parse files from repo_text
    files = parse_repo_files(background_code)

    if not files:
        # If no files found, fall back to regular sliding window
        return rag_retrieve(
            background_code=background_code,
            query_code=query_code,
            embedder=embedder,
            reranker=reranker,
            window_size=window_size,
            overlap=overlap,
            top_k=top_k,
            pruner=pruner,
        )

    # For each file, create chunks using sliding window
    all_chunks = []
    for filename, file_content in files.items():
        file_chunks = chunk_sliding_window(file_content, window_size, overlap)
        # Add filename prefix to each chunk for context
        for i, chunk in enumerate(file_chunks):
            all_chunks.append(f"# {filename}_chunk_{i}:\n{chunk}")

    if not all_chunks:
        return ""

    # Compute embeddings
    query_embedding = embedder.embed([query_code], batch_size=32)[0]
    chunk_embeddings = embedder.embed(all_chunks, batch_size=32)
    similarities = torch.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings, dim=1
    )

    # Retrieve top_k chunks
    retrieved_chunks = [all_chunks[i] for i in similarities.topk(top_k).indices]

    # Post-prune if needed
    if pruner:
        retrieved_chunks = pruner.prune_batch(
            query_code, retrieved_chunks[:top_k], sort=pruner_sort
        )
    else:
        retrieved_chunks = retrieved_chunks[:top_k]

    combined_code = "\n\n".join(retrieved_chunks)
    return combined_code


def evaluate_longcodeqa(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    method: str = "full",
    result_dir: str = "results/longcodeqa",
    embed_model_name: str = "microsoft/unixcoder-base",
    embedder_type: str = "auto",  # "auto" (transformer), "bert", "qwen", "bgem3"
    dataset_path: str = None,
    num_examples: int = 200,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    # RAG params
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    # Reranker params
    reranker_type: str = None,  # "bert", "bgev2m3", "qwen", or None
    reranker_model_name: str = None,  # Model name for reranker
    # vLLM params
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.9,
    # Pruner params for rag_with_pruner and function_rag_with_pruner
    pruner_type: str = "silver_label",  # "silver_label" or "online_rerank"
    pruner_model_name: str = None,
    pruner_tensor_parallel_size: int = 1,
    pruner_temperature: float = 0.3,
    pruner_max_tokens: int = 4096,
    # Online mode params (for SilverLabelPrunerModel)
    pruner_online_mode: bool = False,
    pruner_api_base: str = None,
    pruner_api_key: str = None,
    pruner_online_model_name: str = "default-model",
    pruner_api_port: int = 8001,
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
    pruner_use_token_pruned_code: bool = False,
    pruner_start_gpu: int = 1,
    max_model_len: int = 32768,
    # qwen3_summary params
    qwen_summary_model_name: str = "Qwen/Qwen3-0.6B",
    qwen_summary_max_tokens: int = 256,
    qwen_summary_temperature: float = 0.0,
    qwen_summary_max_input_tokens: int = 32768,
):
    """Evaluates LongCodeQA dataset with specified context preparation method."""
    performance_metrics = {
        "example_id": [],
        "context_preparation_time_seconds": [],
        "inference_time_per_example_seconds": [],
        "original_tokens": [],
        "processed_tokens": [],
        "compression_ratio": [],
    }

    if not torch.cuda.is_available():
        logger.error("CUDA not available. GPU allocation requires CUDA.")
        return
    print(model_name)
    # Calculate GPU ranges (same as LCC/main.py)
    # OnlineRerankPrunerModel and SilverLabelPrunerModel in online mode don't need local GPU
    pruner_tp = (
        pruner_tensor_parallel_size
        if method
        in [
            "rag_with_pruner",
            "file_rag_with_pruner",
            "longcodezip_with_pruner",
            "rag_with_silver_label_pruner",
            "file_rag_with_silver_label_pruner",
        ]
        and pruner_type == "silver_label"
        and not pruner_online_mode
        else 0
    )
    main_tp = tensor_parallel_size

    embed_device = torch.device("cuda:0")
    # pruner_start_gpu = 1
    pruner_end_gpu = pruner_start_gpu + pruner_tp - 1

    if pruner_online_mode and method in [
        "rag_with_pruner",
        "file_rag_with_pruner",
        "longcodezip_with_pruner",
    ]:
        main_start_gpu = 1 + pruner_tensor_parallel_size
    elif pruner_tp > 0:
        main_start_gpu = pruner_end_gpu + 1
    else:
        if is_port_listening(pruner_api_port) and pruner_tensor_parallel_size > 0:
            logger.info(
                f"Detected pruner server running on port {pruner_api_port}, avoiding its GPUs"
            )
            main_start_gpu = 1 + pruner_tensor_parallel_size
        else:
            main_start_gpu = 1
    main_end_gpu = main_start_gpu + main_tp - 1

    logger.info("GPU allocation strategy:")
    logger.info("  - Embedding model: cuda:0")
    if method in ["longcodezip", "longcodezip_with_pruner"]:
        logger.info(
            f"  - CodeCompressor: using model {longcodezip_model_name} (GPU allocation handled internally)"
        )
    if method in ["rag_with_pruner", "file_rag_with_pruner", "longcodezip_with_pruner"]:
        if pruner_online_mode:
            logger.info("  - Pruner model: online mode (API call, no local GPU)")
            logger.info(
                f"  - Pruner server (external): GPU 1 to GPU {1 + pruner_tensor_parallel_size - 1} ({pruner_tensor_parallel_size} GPUs)"
            )
        else:
            logger.info(
                f"  - Pruner model (SilverLabelPrunerModel, vLLM): cuda:{pruner_start_gpu} to cuda:{pruner_end_gpu} ({pruner_tp} GPUs)"
            )
    elif method in [
        "rag_with_silver_label_pruner",
        "file_rag_with_silver_label_pruner",
    ]:
        logger.info(
            "  - Pruner model (SilverLabelPrunerModel): online mode (OpenAI API, no local GPU)"
        )
        logger.info(
            f"  - Pruner server (external): GPU 1 to GPU {1 + pruner_tensor_parallel_size - 1} ({pruner_tensor_parallel_size} GPUs)"
        )
    elif is_port_listening(pruner_api_port) and pruner_tensor_parallel_size > 0:
        logger.info(f"  - External pruner server detected on port {pruner_api_port}")
        logger.info(
            f"  - Pruner server (external): GPU 1 to GPU {1 + pruner_tensor_parallel_size - 1} ({pruner_tensor_parallel_size} GPUs)"
        )
    logger.info(
        f"  - Main model: cuda:{main_start_gpu} to cuda:{main_end_gpu} ({main_tp} GPUs)"
    )

    # Verify we have enough GPUs
    num_gpus = torch.cuda.device_count()
    if pruner_online_mode and method in [
        "rag_with_pruner",
        "file_rag_with_pruner",
        "longcodezip_with_pruner",
    ]:
        required_gpus = 1 + pruner_tensor_parallel_size + main_tp
    elif is_port_listening(pruner_api_port) and pruner_tensor_parallel_size > 0:
        required_gpus = 1 + pruner_tensor_parallel_size + main_tp
    else:
        required_gpus = 1 + pruner_tp + main_tp
    if num_gpus < required_gpus:
        logger.error(
            f"Not enough GPUs. Required: {required_gpus}, Available: {num_gpus}"
        )
        return

    # --- 1. Load Data ---
    # Resolve dataset path relative to project root
    if dataset_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        dataset_path = project_root / "longcodeqa_32k.jsonl"
    else:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            dataset_path = project_root / dataset_path

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = LongCodeQADataset(str(dataset_path))
    if num_examples > 0 and num_examples < len(dataset):
        dataset.data = dataset.data[:num_examples]
    logger.info(f"Loaded {len(dataset)} examples")

    # --- 2. Initialize Models ---
    embed_model = None
    embed_tokenizer = None
    embedder = None
    reranker = None

    # Initialize embedder based on type
    rag_methods = [
        "rag",
        "file_rag",
        "rag_with_pruner",
        "file_rag_with_pruner",
        "rag_with_rerank",
        "rag_with_pruner_rerank",
        "rag_with_silver_label_pruner",
        "file_rag_with_silver_label_pruner",
        "rag_with_token_pruner",
        "no_context",
        "full",
    ]
    if method in rag_methods:
        logger.info(
            f"Initializing embedder: embedder_type={embedder_type}, model={embed_model_name}"
        )

        if embedder_type == "auto" or embedder_type == "transformer":
            # Fallback to old behavior for backward compatibility
            embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
            embed_model = AutoModel.from_pretrained(embed_model_name).to(embed_device)
            embed_model.eval()

            # Create an adapter to wrap embed_model and embed_tokenizer
            class TransformerEmbedderAdapter(EmbedderAdapter):
                def __init__(self, model, tokenizer, max_length=512):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def embed(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
                    all_embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        batch_dict = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        batch_dict = {
                            k: v.to(self.model.device) for k, v in batch_dict.items()
                        }
                        with torch.no_grad():
                            outputs = self.model(**batch_dict)
                        # Use mean pooling over token embeddings
                        attention_mask = batch_dict["attention_mask"]
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = (
                            attention_mask.unsqueeze(-1)
                            .expand(token_embeddings.size())
                            .float()
                        )
                        sum_embeddings = torch.sum(
                            token_embeddings * input_mask_expanded, 1
                        )
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = sum_embeddings / sum_mask
                        # Normalize embeddings
                        batch_embeddings = torch.nn.functional.normalize(
                            batch_embeddings, p=2, dim=1
                        )
                        all_embeddings.append(batch_embeddings)
                    return torch.cat(all_embeddings, dim=0)

            embedder = TransformerEmbedderAdapter(
                embed_model, embed_tokenizer, max_length=512
            )
            logger.info(f"Using transformer embedding model on {embed_device}.")
        elif embedder_type == "bert":
            embedder = BertBasedEmbedder(model_name=embed_model_name, max_length=512)
            # Move model to device for efficiency
            embedder.model = embedder.model.to(embed_device)
            logger.info(f"Using BertBasedEmbedder on {embed_device}.")
        elif embedder_type == "qwen":
            embedder = QwenEmbedder(model_name=embed_model_name, max_length=8192)
            logger.info("Using QwenEmbedder on cuda (automatically handled).")
        elif embedder_type == "bgem3":
            embedder = BGEM3Embedder(model_name=embed_model_name, max_length=8192)
            logger.info("Using BGEM3Embedder.")
        else:
            raise ValueError(f"Unknown embedder_type: {embedder_type}")

        # Initialize reranker if specified
        if reranker_type is not None:
            logger.info(f"Initializing reranker: reranker_type={reranker_type}")
            if reranker_type == "bert":
                reranker = BertBasedReranker(
                    model_name=reranker_model_name or "bert-base-uncased", device="cuda"
                )
            elif reranker_type == "bgev2m3":
                reranker = BGEV2M3Reranker(
                    model_name=reranker_model_name or "BAAI/bge-reranker-v2-m3",
                    device="cuda",
                )
            elif reranker_type == "qwen":
                reranker = QwenReranker(
                    model_name=reranker_model_name or "Qwen/Qwen3-Reranker-0.6B",
                    device="cuda",
                )
            elif reranker_type == "online":
                reranker = OnlineReranker(
                    api_base=rerank_api_base,
                    aggregate_method=rerank_aggregate_method,
                    language=rerank_language,
                )
            else:
                raise ValueError(f"Unknown reranker_type: {reranker_type}")
            logger.info(f"Reranker {reranker_type} initialized.")

    # Initialize reranker for rerank-only method
    if method == "rerank_only":
        if not reranker_type:
            raise ValueError(
                "rerank_only method requires reranker_type but it's not provided"
            )
        if not reranker_model_name:
            raise ValueError(
                "rerank_only method requires reranker_model_name but it's not provided"
            )
        logger.info(
            f"Initializing reranker for rerank-only: reranker_type={reranker_type}"
        )
        if reranker_type == "bert":
            reranker = BertBasedReranker(model_name=reranker_model_name, device="cuda")
        elif reranker_type == "bgev2m3":
            reranker = BGEV2M3Reranker(model_name=reranker_model_name, device="cuda")
        elif reranker_type == "qwen":
            reranker = QwenReranker(model_name=reranker_model_name, device="cuda")
        elif reranker_type == "online":
            reranker = OnlineReranker(
                api_base=rerank_api_base,
                aggregate_method=rerank_aggregate_method,
                language=rerank_language,
            )
        else:
            raise ValueError(f"Unknown reranker_type: {reranker_type}")
        logger.info(f"Reranker {reranker_type} initialized for rerank-only method.")

    # Initialize CodeCompressor if needed
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
        # Pre-initialize tiktoken to handle download issues
        # tiktoken will try to download encoding files even when model is local
        try:
            import tiktoken

            # Try to pre-load the encoding to trigger download if needed
            # This will cache the file locally for future use
            try:
                _ = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.debug("tiktoken encoding pre-loaded successfully")
            except Exception as tiktoken_error:
                logger.warning(
                    f"Failed to pre-load tiktoken encoding: {tiktoken_error}. "
                    "This may cause issues if the encoding file is not cached locally. "
                    "Consider downloading it manually or fixing network/proxy settings."
                )
        except ImportError:
            logger.warning("tiktoken not available, skipping pre-load")

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
        # Pre-initialize tiktoken to handle download issues
        try:
            import tiktoken

            try:
                _ = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.debug("tiktoken encoding pre-loaded successfully")
            except Exception as tiktoken_error:
                logger.warning(
                    f"Failed to pre-load tiktoken encoding: {tiktoken_error}. "
                    "This may cause issues if the encoding file is not cached locally."
                )
        except ImportError:
            logger.warning("tiktoken not available, skipping pre-load")

        llm_lingua_compressor = PromptCompressor(
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        logger.info("LongLLMLingua initialized.")
    
    qwen_summary_tokenizer = None
    qwen_summary_model = None
    if method == "qwen3_summary":
        logger.info(f"Initializing Qwen3-0.6B Summary Model: {qwen_summary_model_name}")
        qwen_summary_tokenizer = AutoTokenizer.from_pretrained(
            qwen_summary_model_name, trust_remote_code=True, local_files_only=True
        )
        qwen_summary_model = AutoModelForCausalLM.from_pretrained(
            qwen_summary_model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        qwen_summary_model.eval()

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # --- 3. Process the Specified Method ---
    logger.info(f"--- Processing Method: {method} ---")

    # Modify result directory based on method and parameters
    method_suffix = f"method_{method}"
    if method == "rag":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "file_rag":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "rag_with_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_threshold{rerank_threshold}"
    elif method == "file_rag_with_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_threshold{rerank_threshold}"
    elif method == "rag_with_rerank":
        method_suffix += (
            f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_rerank{reranker_type}"
        )
    elif method == "rag_with_pruner_rerank":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_rerank{reranker_type}_prune"
    elif method == "rerank_only":
        method_suffix += (
            f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}_rerank{reranker_type}"
        )
    elif method == "rag_with_silver_label_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "file_rag_with_silver_label_pruner":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "longcodezip":
        method_suffix += f"_rate{longcodezip_rate}_rankonly{longcodezip_rank_only}"
    elif method == "longcodezip_with_pruner":
        method_suffix += f"_rate{longcodezip_rate}_rankonly{longcodezip_rank_only}"
    elif method == "llmlingua2":
        method_suffix += f"_rate{llmlingua2_rate}_model{llmlingua2_model_name.replace('/', '_slash_')}"
    elif method == "longllmlingua":
        method_suffix += f"_rate{longllmlingua_rate}_dyn{longllmlingua_dynamic_context_compression_ratio}"
    elif method == "selective_context":
        method_suffix += f"_model{selective_context_model_type}_lang{selective_context_lang}_ratio{selective_context_reduce_ratio}"
    elif method == "qwen3_summary":
        method_suffix += f"_qwen3sum_max{qwen_summary_max_tokens}_temp{qwen_summary_temperature}_maxinput{qwen_summary_max_input_tokens}"

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
    original_data = []

    # Initialize pruner if needed
    pruner = None
    if method in [
        "rag_with_pruner",
        "rag_with_token_pruner",
        "file_rag_with_pruner",
        "longcodezip_with_pruner",
        "rag_with_pruner_rerank",
        "rag_with_silver_label_pruner",
        "file_rag_with_silver_label_pruner",
    ]:
        # Force silver_label pruner type and online mode for silver_label_pruner methods
        effective_pruner_type = pruner_type
        effective_pruner_online_mode = pruner_online_mode
        if method in [
            "rag_with_silver_label_pruner",
            "file_rag_with_silver_label_pruner",
        ]:
            if effective_pruner_type != "silver_label":
                logger.warning(
                    f"{method} method requires silver_label pruner type, but got {effective_pruner_type}. Forcing to silver_label."
                )
            effective_pruner_type = "silver_label"

        if effective_pruner_type == "online_rerank":
            pruner = OnlineRerankPrunerModel(
                api_base=rerank_api_base,
                threshold=rerank_threshold,
                always_keep_first_frags=rerank_always_keep_first_frags,
                aggregate_method=rerank_aggregate_method,
                language=rerank_language,
                batch_size=batch_size,
                use_token_pruned_code=pruner_use_token_pruned_code,
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
    # For LongCodeQA, prompt format is: {prompt_goal}\nRepository: {repo_text}\n{question}
    # We need to correctly parse and reconstruct the prompt
    for i, item in enumerate(dataset):
        prompt_text = item.prompt
        correct_letter = item.correct_letter
        repo_text = item.repo_text or ""
        question = item.question or ""
        prompt_goal = item.prompt_goal or ""

        # Use question as query for RAG
        query_code = question if question else prompt_text

        # 2. Measure the time and number of tokens for context processing/compression.
        original_context = repo_text if repo_text else prompt_text
        original_tokens = len(tokenizer.encode(original_context))

        start_context_prep_time = time.time()

        retrieved_ctx = ""
        try:
            if method == "full":
                retrieved_ctx = repo_text if repo_text else prompt_text
                tokenized_context = tokenizer.encode(retrieved_ctx)
                if len(tokenized_context) > max_model_len - 256:
                    logger.warning(
                        f"Context length exceeds {max_model_len}, truncating from the head. Original length: {len(tokenized_context)}, Truncated length: {max_model_len}"
                    )
                    retrieved_ctx = tokenizer.decode(
                        tokenized_context[-(max_model_len - 256) :]
                    )
            elif method == "rag":
                if embedder is None:
                    raise ValueError(
                        "RAG method selected but embedder not initialized."
                    )
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                )
            elif method == "file_rag":
                if embedder is None:
                    raise ValueError(
                        "File RAG method selected but embedder not initialized."
                    )
                retrieved_ctx = file_level_rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=None,
                )
            elif method == "rag_with_pruner":
                if embedder is None:
                    raise ValueError(
                        "RAG with pruner method selected but embedder not initialized."
                    )
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "file_rag_with_pruner":
                if embedder is None:
                    raise ValueError(
                        "File RAG with pruner method selected but embedder not initialized."
                    )
                retrieved_ctx = file_level_rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "rag_with_rerank":
                if embedder is None:
                    raise ValueError(
                        "RAG with rerank method selected but embedder not initialized."
                    )
                if reranker is None:
                    raise ValueError(
                        "RAG with rerank method selected but reranker not initialized."
                    )
                # Use adapter-based RAG with reranker
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=reranker,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                )
            elif method == "rag_with_pruner_rerank":
                if embedder is None:
                    raise ValueError(
                        "RAG with rerank and pruner method selected but embedder not initialized."
                    )
                if pruner is None:
                    raise ValueError(
                        "RAG with rerank and pruner method selected but pruner not initialized."
                    )
                # Use adapter-based RAG with pruner (pruner itself acts as reranker)
                # Sorting is controlled by pruner_sort parameter
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,  # No separate reranker, pruner handles reranking
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=True,  # Sort by pruner scores
                )
            elif method == "rerank_only":
                if reranker is None:
                    raise ValueError(
                        "Rerank-only method selected but reranker not initialized."
                    )
                retrieved_ctx = rerank_only_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    reranker=reranker,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                )
            elif method == "rag_with_silver_label_pruner":
                if embedder is None:
                    raise ValueError(
                        "RAG with silver label pruner method selected but embedder not initialized."
                    )
                if pruner is None:
                    raise ValueError(
                        "RAG with silver label pruner method selected but pruner not initialized."
                    )
                # Use RAG + SilverLabelPrunerModel (vLLM mode)
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "file_rag_with_silver_label_pruner":
                if embedder is None:
                    raise ValueError(
                        "File RAG with silver label pruner method selected but embedder not initialized."
                    )
                if pruner is None:
                    raise ValueError(
                        "File RAG with silver label pruner method selected but pruner not initialized."
                    )
                # Use file-level RAG + SilverLabelPrunerModel (vLLM mode)
                retrieved_ctx = file_level_rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "no_context":
                retrieved_ctx = ""
            elif method == "longcodezip":
                if not compressor:
                    raise ValueError(
                        "LongCodeZip method selected but compressor not initialized."
                    )
                # Use CodeCompressor to compress the code
                code_to_compress = repo_text if repo_text else prompt_text
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        result = compressor.compress_code_file(
                            code=code_to_compress,
                            query=query_code,
                            instruction=prompt_goal if prompt_goal else "",
                            rate=longcodezip_rate,
                            rank_only=longcodezip_rank_only,
                        )
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
                code_to_compress = repo_text if repo_text else prompt_text
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    try:
                        result = compressor.compress_code_file(
                            code=code_to_compress,
                            query=query_code,
                            instruction=prompt_goal if prompt_goal else "",
                            rate=longcodezip_rate,
                            rank_only=longcodezip_rank_only,
                            pruner=pruner,
                        )
                        retrieved_ctx = result.get("compressed_code", code_to_compress)
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
                # Use LLMLingua-2 to compress the prompt
                code_to_compress = repo_text if repo_text else prompt_text
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
                            else:
                                raise ValueError(
                                    "Unexpected return type from LLMLingua-2"
                                )

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
                code_to_compress = repo_text if repo_text else prompt_text
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
                            question=question if question else query_code,
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
                code_to_compress = repo_text if repo_text else prompt_text
                if not code_to_compress.strip():
                    retrieved_ctx = ""
                else:
                    #Safe truncation before SelectiveContext
                    sc_tokenizer = selective_context_compressor.tokenizer
                    sc_model_max_len = 1024
                    if hasattr(sc_tokenizer, "model_max_length"):
                        sc_model_max_len = sc_tokenizer.model_max_length

                    def truncate_to_model(text):
                        ids = sc_tokenizer.encode(text, add_special_tokens=False)
                        ids = ids[: max(sc_model_max_len - 24, 1)]
                        return sc_tokenizer.decode(ids, skip_special_tokens=True)

                    try:
                        max_chunk_length = sc_model_max_len - 24

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

                                if not chunk_text.strip():
                                    continue

                                try:
                                    # Compress this chunk
                                    compressed_chunk, _ = selective_context_compressor(
                                        chunk_text,
                                        reduce_ratio=selective_context_reduce_ratio,
                                    )
                                    if compressed_chunk.strip():
                                        compressed_chunks.append(compressed_chunk)
                                except Exception as chunk_error:
                                    logger.warning(
                                        f"Error compressing chunk {chunk_idx + 1}/{num_chunks}: {chunk_error}. "
                                        f"Using original chunk."
                                    )
                                    compressed_chunks.append(chunk_text)

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
            elif method == "rag_with_token_pruner":
                if embedder is None:
                    raise ValueError(
                        "RAG with token pruner method selected but embedder not initialized."
                    )
                if pruner is None:
                    raise ValueError(
                        "RAG with token pruner method selected but pruner not initialized."
                    )
                # OnlineRerankPrunerModel returns token_pruned_code
                retrieved_ctx = rag_retrieve(
                    background_code=repo_text if repo_text else prompt_text,
                    query_code=query_code,
                    embedder=embedder,
                    reranker=None,
                    window_size=rag_window_size,
                    overlap=rag_overlap,
                    top_k=rag_top_k,
                    pruner=pruner,
                    pruner_sort=False,
                )
            elif method == "qwen3_summary":
                code_to_summarize = repo_text if repo_text else prompt_text
                if not code_to_summarize.strip():
                    retrieved_ctx = ""
                else:
                    chunks = chunk_by_token_count(
                        code_to_summarize,
                        qwen_summary_tokenizer,
                        max_tokens=qwen_summary_max_input_tokens,
                        overlap_tokens=128,
                    )
                    summaries = []
                    summary_prompt_tpl = (
                        "You are a code reading assistant. Given a user question and a code snippet, "
                        "summarize only the information that is directly relevant to the question. "
                        "Do not answer the question or fabricate anything. "
                        "Output concise English bullet points (<=300 words), keeping key function/variable names.\n\n"
                        "Question: {question}\n\nCode snippet:\n{code}\n\nSummary:"
                    )
                    for chunk in chunks:
                        user_prompt = summary_prompt_tpl.format(
                            question=question or query_code, code=chunk
                        )
                        inputs = qwen_summary_tokenizer(
                            user_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=qwen_summary_max_input_tokens,
                        ).to(qwen_summary_model.device)
                        with torch.no_grad():
                            outputs = qwen_summary_model.generate(
                                **inputs,
                                max_new_tokens=qwen_summary_max_tokens,
                                temperature=qwen_summary_temperature,
                                top_p=0.9,
                            )
                        text = qwen_summary_tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                        if text:
                            summaries.append(text)
                    retrieved_ctx = "\n".join(summaries) if summaries else ""
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
            performance_metrics["example_id"].append(i)

            # Construct final prompt
            # Format: {prompt_goal}\nRepository: {retrieved_ctx or repo_text}\n{question}
            # Add instruction to output structured JSON
            json_instruction = '\n\nPlease provide your answer in the following JSON format (no markdown code blocks, just pure JSON):\n{\n  "reason": "<Brief 1-sentence reason explaining why you chose that answer, less than 50 words>",\n  "final_answer": "<A, B, C, or D>"\n}'

            if retrieved_ctx:
                # Use retrieved context
                if prompt_goal:
                    final_prompt = f"{prompt_goal}\nRepository: {retrieved_ctx}\n{question}{json_instruction}"
                else:
                    final_prompt = (
                        f"Repository: {retrieved_ctx}\n{question}{json_instruction}"
                    )
            else:
                # Use original prompt structure but add JSON instruction
                if prompt_goal:
                    final_prompt = f"{prompt_goal}\nRepository: {repo_text}\n{question}{json_instruction}"
                else:
                    final_prompt = f"{prompt_text}{json_instruction}"

            all_prompts.append(final_prompt.strip())
            original_data.append(
                {
                    "id": i,
                    "correct_letter": correct_letter,
                    "original_prompt": prompt_text,
                    "prompt": final_prompt,
                }
            )
        except Exception as e:
            logger.warning(
                f"Error processing example {i}: {e}",
                exc_info=True,
            )
            # Even if there's an error, record a placeholder to keep the list length consistent.
            performance_metrics["context_preparation_time_seconds"].append(-1)
            performance_metrics["original_tokens"].append(original_tokens)
            performance_metrics["processed_tokens"].append(-1)
            performance_metrics["compression_ratio"].append(-1)
            performance_metrics["example_id"].append(i)
            continue

    # --- 4. Clean up Compression/Embedding Models ---
    logger.info("Freeing up GPU memory from compression/embedding models")
    if embed_model:
        del embed_model
    if embed_tokenizer:
        del embed_tokenizer
    if compressor:
        del compressor
    if llm_lingua_compressor:
        del llm_lingua_compressor
    if selective_context_compressor:
        del selective_context_compressor
    if qwen_summary_model:
        del qwen_summary_model
    if qwen_summary_tokenizer:
        del qwen_summary_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory freed")

    # --- 5. Initialize Generation LLM ---
    if not all_prompts:
        logger.error(
            f"No valid prompts were prepared for method {method}. Skipping generation and scoring."
        )
        return

    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    main_gpu_list = ",".join(str(i) for i in range(main_start_gpu, main_end_gpu + 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = main_gpu_list
    logger.info(
        f"Setting CUDA_VISIBLE_DEVICES={main_gpu_list} for main model initialization"
    )

    try:
        logger.info(f"Initializing generation LLM: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
        logger.info(f"Generation LLM {model_name} initialized.")
    except Exception as e:
        # Catch engine startup errors (commonly from vLLM trying to initialize CUDA in worker)
        err_text = str(e)
        logger.error(f"Failed to initialize generation LLM: {err_text}")
        # Gather quick diagnostics to help the user
        try:
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count()
        except Exception:
            cuda_available = False
            cuda_count = 0

        logger.error("LLM init diagnostics:")
        logger.error(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.error(f"  torch.cuda.is_available()={cuda_available}")
        logger.error(f"  torch.cuda.device_count()={cuda_count}")

        # Restore env var before exiting
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Re-raise as RuntimeError with a shorter message so upstream callers stop cleanly
        raise RuntimeError(
            "Generation LLM initialization failed due to CUDA/driver issues. See log output for diagnostics."
        ) from e
    else:
        # Only restore CUDA_VISIBLE_DEVICES when initialization succeeded
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
            # 3. Measurement model inference time
            start_inference_time = time.time()
            batch_outputs = generate_completions(
                llm,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                use_guided_decoding=True,
            )
            end_inference_time = time.time()

            # Calculate and record the average inference time for each sample.
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
    correct_count = 0
    total_count = 0

    if len(all_outputs) != len(original_data):
        logger.warning(
            f"Warning: Mismatch between generated outputs ({len(all_outputs)}) and original data ({len(original_data)}). Scores might be inaccurate."
        )
        min_len = min(len(all_outputs), len(original_data))
        all_outputs = all_outputs[:min_len]
        original_data = original_data[:min_len]
        all_prompts = all_prompts[:min_len]

    logger.info(f"Evaluating results for {len(all_outputs)} examples...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "w") as f_out:
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            if i >= len(original_data) or i >= len(all_prompts):
                logger.error(
                    f"Index {i} out of bounds after potential mismatch alignment. Stopping result processing."
                )
                break
            orig_data = original_data[i]
            prompt = all_prompts[i]
            correct_letter = orig_data["correct_letter"]

            # Extract predicted letter and reasoning from output
            predicted_letter, reasoning = extract_answer_and_reasoning(output)
            is_correct = (
                predicted_letter == correct_letter.upper()
                if predicted_letter
                else False
            )

            if predicted_letter:
                total_count += 1
                if is_correct:
                    correct_count += 1

            # Calculate compression ratio using token count instead of character count
            original_prompt = orig_data.get("original_prompt", "")
            compression_ratio = 1.0
            if original_prompt:
                # Calculate token counts
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                original_prompt_tokens = len(
                    tokenizer.encode(original_prompt, add_special_tokens=False)
                )
                if original_prompt_tokens > 0:
                    compression_ratio = prompt_tokens / original_prompt_tokens

            result = {
                **orig_data,
                "pruned_prompt": prompt,
                "output": output,
                "predicted_letter": predicted_letter,
                "reasoning": reasoning,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
                "compression_ratio": compression_ratio,
            }

            model_outputs_data.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Raw results saved to {model_output_path}")

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    scores = {
        "model_name": model_name,
        "method": method,
        "num_examples_total": len(original_data),
        "num_examples_with_prediction": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "average_compression_ratio": sum(
            result.get("compression_ratio", 0) for result in model_outputs_data
        )
        / len(model_outputs_data)
        if model_outputs_data
        else 0,
        "parameters": {
            "dataset_path": str(dataset_path),
            "num_examples": num_examples,
            "embed_model_name": embed_model_name
            if method in ["rag", "file_rag", "rag_with_pruner", "file_rag_with_pruner"]
            else None,
            "rerank_model_name": reranker_model_name,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "rag_window_size": rag_window_size
            if method in ["rag", "rag_with_pruner", "rerank_only"]
            else None,
            "rag_overlap": rag_overlap
            if method in ["rag", "rag_with_pruner", "rerank_only"]
            else None,
            "rag_top_k": rag_top_k
            if method
            in [
                "rag",
                "rag_with_pruner",
                "file_rag",
                "file_rag_with_pruner",
                "rerank_only",
            ]
            else None,
            "reranker_type": reranker_type if method == "rerank_only" else None,
            "pruner_model_name": pruner_model_name
            if method in ["rag_with_pruner", "file_rag_with_pruner"]
            else None,
            "pruner_online_model_name": pruner_online_model_name
            if method
            in ["rag_with_silver_label_pruner", "file_rag_with_silver_label_pruner"]
            else None,
            "pruner_tensor_parallel_size": pruner_tensor_parallel_size
            if method
            in [
                "rag_with_pruner",
                "file_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "file_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_temperature": pruner_temperature
            if method
            in [
                "rag_with_pruner",
                "file_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "file_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_max_tokens": pruner_max_tokens
            if method
            in [
                "rag_with_pruner",
                "file_rag_with_pruner",
                "rag_with_silver_label_pruner",
                "file_rag_with_silver_label_pruner",
            ]
            else None,
            "pruner_type": "silver_label"
            if method
            in ["rag_with_silver_label_pruner", "file_rag_with_silver_label_pruner"]
            else None,
            "pruner_online_mode": True
            if method
            in ["rag_with_silver_label_pruner", "file_rag_with_silver_label_pruner"]
            else None,
            "longcodezip_model_name": longcodezip_model_name
            if method in ["longcodezip", "longcodezip_with_pruner"]
            else None,
            "longcodezip_rate": longcodezip_rate
            if method in ["longcodezip", "longcodezip_with_pruner"]
            else None,
            "longcodezip_rank_only": longcodezip_rank_only
            if method in ["longcodezip", "longcodezip_with_pruner"]
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
        },
    }

    logger.info(
        f"Method {method}: Accuracy = {accuracy:.2f}% ({correct_count}/{total_count} correct)"
    )
    save_json(scores, score_output_path)
    logger.info(f"Scores saved to {score_output_path}")

    # 4. Save the performance metrics file
    performance_metrics_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}-PERFORMANCE.json",
    )
    save_json(performance_metrics, performance_metrics_output_path)
    logger.info(f"Performance metrics saved to {performance_metrics_output_path}")

    logger.info("\n--- Performance Metrics (Averages) ---")
    for key, values in performance_metrics.items():
        # Filter out IDs and error values ​​-1
        valid_values = [v for v in values if v != -1]
        if key != "example_id" and valid_values:
            avg = np.mean(valid_values)
            logger.info(f"Average {key}: {avg:.4f}")
    logger.info("--------------------------------------\n")

    logger.info("Evaluation complete.")
    if "llm" in locals() and llm is not None:
        del llm
        logger.info("Generation LLM deleted.")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    fire.Fire(evaluate_longcodeqa)
