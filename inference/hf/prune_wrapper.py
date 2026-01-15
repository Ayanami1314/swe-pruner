import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoTokenizer

from .swepruner import SwePrunerForCodeCompression, SwePrunerOutput
from pydantic import BaseModel


class PruneRequest(BaseModel):
    query: str
    code: str
    threshold: float = 0.5
    always_keep_first_frags: bool = False
    chunk_overlap_tokens: int = 50


class PruneResponse(BaseModel):
    score: float
    pruned_code: str
    token_scores: List[List[str | float]]  # [[token_str, score], ...]
    kept_frags: List[int]
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    error_msg: Optional[str] = None


def format_instruction(instruction: Optional[str], query: str) -> str:
    """Format instruction and query (LLM style)."""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


def estimate_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """Estimate token count for a text."""
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return len(enc["input_ids"])


def split_code_into_chunks(
    code: str,
    tokenizer: AutoTokenizer,
    chunk_max_tokens: int,
    overlap_tokens: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    Split code into chunks with overlap, based on actual token counts.

    Args:
        code: Full code string
        tokenizer: Tokenizer for counting tokens
        chunk_max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if not code:
        return []

    # Tokenize full code with offsets to track character positions
    code_enc = tokenizer(
        code,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    total_tokens = len(code_enc["input_ids"])
    offsets = code_enc["offset_mapping"]

    # If code fits in one chunk, return as-is
    if total_tokens <= chunk_max_tokens:
        return [(code, 0, len(code))]

    # Sanity check: chunk_max_tokens should be large enough for meaningful chunking
    if chunk_max_tokens < overlap_tokens:
        overlap_tokens = 0

    chunks = []
    stride = chunk_max_tokens - overlap_tokens

    if stride <= 0:
        raise ValueError(
            f"Invalid configuration: stride={stride} "
            f"(chunk_max_tokens={chunk_max_tokens}, overlap_tokens={overlap_tokens})"
        )

    start_token_idx = 0

    while start_token_idx < total_tokens:
        end_token_idx = min(start_token_idx + chunk_max_tokens, total_tokens)

        # Get character positions for this chunk
        start_char = (
            offsets[start_token_idx][0] if start_token_idx < len(offsets) else 0
        )
        # For end, use the end of the last token, but ensure we don't go past code length
        if end_token_idx <= len(offsets):
            end_char = offsets[end_token_idx - 1][1]
        else:
            end_char = len(code)

        chunk_text = code[start_char:end_char]
        chunks.append((chunk_text, start_char, end_char))

        # Move to next chunk with overlap
        if end_token_idx >= total_tokens:
            break

        start_token_idx += stride

    return chunks


def build_input_for_llm(
    query: str,
    code: str,
    tokenizer: AutoTokenizer,
    max_length: int = 8192,
    instruction: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Build input tensors for LLM-style model inference."""
    # Format query
    formatted_query = format_instruction(instruction, query)

    # Tokenize query and code
    query_enc = tokenizer(
        formatted_query,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    code_enc = tokenizer(
        code,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    query_ids = query_enc["input_ids"]
    code_ids = code_enc["input_ids"]

    # LLM prefix and suffix (matching train.py)
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # Calculate available length
    available_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    query_len = len(query_ids)
    code_len = len(code_ids)

    # Truncate code if necessary (and corresponding offsets)
    if query_len + code_len > available_length:
        truncate_to = available_length - query_len
        code_ids = code_ids[:truncate_to]
        code_offsets = code_enc["offset_mapping"][:truncate_to]
        code_len = len(code_ids)
    else:
        # Use original offsets
        code_offsets = code_enc["offset_mapping"]

    # Build full sequence
    input_ids = prefix_tokens + query_ids + code_ids + suffix_tokens
    real_len = len(input_ids)

    # right padding for LLM
    pad_len = max_length - real_len
    # input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
    # attention_mask = [0] * pad_len + [1] * real_len
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * real_len + [0] * pad_len

    # Calculate code token positions
    # doc_start = pad_len + len(prefix_tokens) + query_len
    # doc_end = doc_start + code_len
    doc_start = len(prefix_tokens) + query_len
    doc_end = doc_start + code_len

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "doc_start": doc_start,
        "doc_end": doc_end,
        "code_offsets": code_offsets,
        "code_len": code_len,
    }


def aggregate_token_scores_to_lines(
    code: str,
    token_scores: List[Tuple[str, float]],
    token_offsets: List[Tuple[int, int]],
) -> Dict[int, float]:
    """Aggregate token scores to line-level scores based on character offsets and code lines.

    Args:
        code: Source code string
        token_scores: List of (token_str, score) tuples
        token_offsets: List of (start_char, end_char) tuples matching token_scores

    Returns:
        Dict mapping line number (1-indexed) to aggregated score.
        Only lines with token coverage are included in the result.
        Lines without tokens will not appear in the dict (treated as low relevance by pruner).
    """
    line_scores_dict: Dict[int, float] = {}

    # Build mapping from character position to score
    char_to_score: Dict[int, float] = {}
    for (token_str, score), (start, end) in zip(token_scores, token_offsets):
        score = max(0.0, min(1.0, float(score)))
        # Clamp to valid range
        start = max(0, min(start, len(code)))
        end = max(0, min(end, len(code)))
        for pos in range(start, end):
            char_to_score[pos] = score

    # Split code into lines and compute line scores
    lines = code.splitlines(keepends=False)
    current_pos = 0

    for line_num, line_text in enumerate(lines, start=1):
        line_start = current_pos
        line_end = current_pos + len(line_text)

        # Collect scores for all characters in this line
        line_scores = []
        for char_pos in range(line_start, line_end):
            if char_pos in char_to_score:
                line_scores.append(char_to_score[char_pos])

        # Only add to dict if this line has token coverage
        # Lines without tokens are treated as having no evidence (not included)
        if line_scores:
            line_scores_dict[line_num] = float(sum(line_scores) / len(line_scores))

        # Move to next line (account for newline character)
        current_pos = line_end + 1  # +1 for the newline character

    return line_scores_dict


def prune_code_lines(
    code: str,
    line_scores: Dict[int, float],
    threshold: float,
    always_keep_first_frags: bool = False,
) -> Tuple[str, List[int]]:
    """Prune code at line level, similar to model.py lines 249-272.

    Returns:
        Tuple of (pruned_code, kept_frags) where kept_frags is list of kept line numbers
    """
    lines = code.splitlines()
    kept_lines = []
    num_first_frags = 1 if always_keep_first_frags else 0

    # Determine which lines to keep
    for line_num in range(1, len(lines) + 1):
        should_keep = False
        if line_num <= num_first_frags:
            should_keep = True
        elif line_num in line_scores and line_scores[line_num] >= threshold:
            should_keep = True

        if should_keep:
            kept_lines.append(line_num)

    # Build pruned code with "..." for pruned lines
    kept_code_lines = []
    # 如果被裁掉的部分只有单行就别裁剪了
    extra_kept_lines = []
    for idx, k in enumerate(kept_lines):
        if idx > 0 and k - kept_lines[idx - 1] == 2:
            # Only one line gap, keep the skipped line as well
            extra_kept_lines.append(k - 1)
    kept_lines.extend(extra_kept_lines)
    kept_lines = sorted(kept_lines)

    filtered_lines_cnt = 0
    filtered_char_cnt = 0
    s_format = "(filtered {} lines)"
    for line in range(1, len(lines) + 1):
        if lines[line - 1].strip() == "":
            filtered_lines_cnt += 1
            continue  # Skip empty lines
        if line not in kept_lines:
            filtered_lines_cnt += 1
            filtered_char_cnt += len(lines[line - 1])
        else:
            if filtered_lines_cnt > 0:
                baseline_length = len(s_format.format(0))
                if filtered_char_cnt > baseline_length:
                    kept_code_lines.append(s_format.format(filtered_lines_cnt))
                else:
                    for j in range(filtered_lines_cnt, 0, -1):
                        kept_code_lines.append(lines[line - j - 1])
                filtered_lines_cnt = 0
                filtered_char_cnt = 0
            kept_code_lines.append(lines[line - 1])
    if filtered_lines_cnt > 0:
        kept_code_lines.append(s_format.format(filtered_lines_cnt))
    pruned_code = "\n".join(kept_code_lines)

    return pruned_code, kept_lines


def merge_token_scores_from_chunks(
    code: str,
    chunk_results: List[
        Tuple[List[Tuple[str, float]], List[Tuple[int, int]], int, int]
    ],
) -> Tuple[List[Tuple[str, float]], List[Tuple[int, int]]]:
    """
    Merge token scores from multiple chunks, averaging overlapping tokens by position.

    Strategy: Group tokens by their (start_pos, end_pos) regardless of token_str,
    since the same position should have the same semantics globally.

    Args:
        code: Full original code string
        chunk_results: List of (token_scores, offsets, start_char, end_char) for each chunk

    Returns:
        Tuple of merged token_scores and their character offsets for the full code
    """

    if not chunk_results:
        return [], []

    # Map from (start_pos, end_pos) to list of scores
    # This groups tokens by position, which is more robust than grouping by position+content
    position_to_scores: Dict[Tuple[int, int], List[Tuple[str, float]]] = defaultdict(
        list
    )

    for token_scores, offsets, start_char, _ in chunk_results:
        max_pairs = min(len(token_scores), len(offsets))
        for idx in range(max_pairs):
            token_str, score = token_scores[idx]
            tok_start, tok_end = offsets[idx]
            abs_start = start_char + tok_start
            abs_end = start_char + tok_end

            # Skip tokens outside code bounds
            if abs_start >= len(code) or abs_end > len(code) or abs_start < 0:
                continue

            position_to_scores[(abs_start, abs_end)].append((token_str, float(score)))

    # Sort by position and build result
    sorted_positions = sorted(position_to_scores.keys())

    merged_token_scores = []
    merged_offsets: List[Tuple[int, int]] = []

    for abs_start, abs_end in sorted_positions:
        token_score_pairs = position_to_scores[(abs_start, abs_end)]

        # Average scores for this position
        avg_score = sum(score for _, score in token_score_pairs) / len(
            token_score_pairs
        )

        # Use the first token_str we saw for this position (they should all be the same)
        token_str = token_score_pairs[0][0]

        merged_token_scores.append((token_str, avg_score))
        merged_offsets.append((abs_start, abs_end))

    return merged_token_scores, merged_offsets


class SwePrunerForCodePruning(SwePrunerForCodeCompression):
    """
    Wrapper around SwePrunerForCodeCompression that provides a simplified prune interface.

    This class can be loaded with trust_remote_code=True from HuggingFace Hub.
    It hides implementation details and always uses line-level aggregation.
    """

    def __init__(self, config):
        # Pass from_pretrained=True if we're being loaded (config._name_or_path will be set)
        # This prevents redundant weight initialization
        from_pretrained = (
            hasattr(config, "_name_or_path") and config._name_or_path is not None
        )
        super().__init__(config, from_pretrained=from_pretrained)
        self.eval()
        # Don't move to device here - let from_pretrained handle it after loading weights
        # Device will be determined when first used, or can be set explicitly
        self._device = None

        # Note: self.tokenizer is already set by parent class (SwePrunerForCodeCompression)
        # Set padding side based on model type (matching online_serving.py)
        # LLM models use left padding, BERT-style models use right padding
        if self.model.is_llm:
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"

        # Match online_serving.py process_single_chunk default instruction
        self.instruction = getattr(
            config,
            "instruction",
            "Given a query, judge if the document(code) is related to query.",
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model from pretrained with proper device handling."""
        # The key is to prevent HuggingFace from using device_map or other optimizations
        # that create meta tensors
        kwargs.pop("device_map", None)
        kwargs.pop("low_cpu_mem_usage", None)  # Also disable this optimization

        # Now call parent's from_pretrained without these optimizations
        # This will load weights directly into memory
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        # Determine target device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target_dtype = getattr(model.config, "torch_dtype", None) or getattr(
            model.config, "dtype", None
        )
        if isinstance(target_dtype, str):
            target_dtype = getattr(torch, target_dtype, None)

        # Move to device (and cast if dtype is specified)
        if target_dtype is not None:
            model = model.to(device=device, dtype=target_dtype)
        else:
            model = model.to(device)

        model._device = device
        model.eval()
        return model

    def _ensure_device(self):
        """Ensure model is on the correct device."""
        if self._device is None:
            # Get current device and set it
            try:
                device = next(self.parameters()).device
                self._device = device
            except StopIteration:
                # No parameters, set to cpu as default
                self._device = torch.device("cpu")

    def _process_single_chunk(
        self,
        query: str,
        code_chunk: str,
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        instruction: Optional[str] = None,
    ) -> Tuple[float, List[Tuple[str, float]], List[Tuple[int, int]]]:
        """Process a single code chunk and return score and token scores."""
        # Ensure model is on the correct device
        self._ensure_device()

        # Build input
        inputs = build_input_for_llm(
            query, code_chunk, tokenizer, max_length=max_length, instruction=instruction
        )

        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        doc_start = inputs["doc_start"]
        doc_end = inputs["doc_end"]
        code_offsets = inputs["code_offsets"]

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
            ):
                outputs: SwePrunerOutput = self(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            token_logits = outputs.token_logits.float()  # [1, L]
            score_logits = outputs.score_logits.float()  # [1]

        # Get score
        score_prob = score_logits.squeeze(0).cpu()
        # Check if this is LLM style model - use is_llm attribute from TokenScorer
        # LLM models output log_softmax (log probabilities), non-LLM output raw logits
        if self.model.is_llm:
            # LLM style: score_logits are log probabilities from log_softmax
            chunk_score = float(torch.exp(score_prob).item())
        else:
            # Non-LLM style: score_logits are raw logits
            chunk_score = float(torch.sigmoid(score_prob).item())

        # Extract code token scores
        token_logits_seq = token_logits.squeeze(0).cpu()  # [L]
        probs = torch.sigmoid(token_logits_seq)  # [L]

        # Get code token positions
        code_token_ids = input_ids[0][doc_start:doc_end].cpu().tolist()
        code_token_scores = []

        for idx, pos in enumerate(range(doc_start, doc_end)):
            token_id = code_token_ids[idx]
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            score = float(probs[pos].item())
            code_token_scores.append((token_str, score))

        return chunk_score, code_token_scores, code_offsets

    def prune(self, request: PruneRequest, max_length: int = 8192) -> PruneResponse:
        max_length = 8192
        # Check if we need to split into chunks
        formatted_query = format_instruction(None, request.query)

        # Estimate total tokens needed
        query_tokens = estimate_token_count(formatted_query, self.tokenizer)
        code_tokens = estimate_token_count(request.code, self.tokenizer)

        # Calculate available length for code (accounting for prefix/suffix)
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        available_length = max_length - len(prefix_tokens) - len(suffix_tokens)
        code_max_tokens = available_length - query_tokens

        # Minimum tokens required to meaningfully process code
        # Match online_serving.py
        MIN_CODE_TOKENS = 100

        # If query is too long and leaves insufficient space for code, skip pruning
        if code_max_tokens < MIN_CODE_TOKENS:
            # Return original code without pruning
            return PruneResponse(
                score=0.0,  # Unknown score since we can't process
                pruned_code=request.code,
                token_scores=[],
                kept_frags=list(
                    range(1, len(request.code.splitlines()) + 1)
                ),  # All lines kept
                origin_token_cnt=code_tokens,
                left_token_cnt=code_tokens,
                model_input_token_cnt=0,
                error_msg=(
                    f"Query too long, insufficient space for code processing. "
                    f"Available code tokens: {code_max_tokens}, "
                    f"minimum required: {MIN_CODE_TOKENS}."
                ),
            )

        # Check if splitting is needed
        if code_tokens > code_max_tokens:
            # Split code into chunks with overlap
            overlap_tokens = request.chunk_overlap_tokens
            chunks = split_code_into_chunks(
                request.code,
                self.tokenizer,
                chunk_max_tokens=code_max_tokens,
                overlap_tokens=overlap_tokens,
            )

            # Process each chunk
            chunk_scores = []
            chunk_results = []

            for chunk_text, start_char, end_char in chunks:
                chunk_score, token_scores, offsets = self._process_single_chunk(
                    request.query,
                    chunk_text,
                    self.tokenizer,
                    max_length=max_length,
                    instruction=self.instruction,
                )
                chunk_scores.append(chunk_score)
                chunk_results.append((token_scores, offsets, start_char, end_char))

            # Average scores across chunks
            # Use max score instead of average to avoid diluting the relevance score
            # when code is split across chunks (each chunk is independent evaluation)
            predicted_score = max(chunk_scores) if chunk_scores else 0.0

            # Merge token scores from chunks (averaging overlaps)
            code_token_scores, code_token_offsets = merge_token_scores_from_chunks(
                request.code, chunk_results
            )
        else:
            # Single chunk processing
            predicted_score, code_token_scores, code_token_offsets = (
                self._process_single_chunk(
                    request.query,
                    request.code,
                    self.tokenizer,
                    max_length=max_length,
                    instruction=self.instruction,
                )
            )

        line_scores = aggregate_token_scores_to_lines(
            request.code,
            code_token_scores,
            code_token_offsets,
        )
        pruned_code, kept_frags = prune_code_lines(
            request.code,
            line_scores,
            request.threshold,
            request.always_keep_first_frags,
        )
        # Format token_scores for response
        token_scores_response = [[token, score] for token, score in code_token_scores]

        return PruneResponse(
            score=predicted_score,
            pruned_code=pruned_code,
            token_scores=token_scores_response,
            kept_frags=kept_frags,
            origin_token_cnt=code_tokens,
            left_token_cnt=estimate_token_count(pruned_code, self.tokenizer),
            model_input_token_cnt=query_tokens
            + code_tokens
            + len(prefix_tokens)
            + len(suffix_tokens),
        )
