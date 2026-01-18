import torch
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
import re
import math
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import logging
import copy
import bisect
import json

# set up the logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CodeCompressor")


class CodeCompressor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device_map: str = "cuda",
        model_config: dict = {},
    ):
        """
        Initialize the CodeCompressor with a language model for compression.

        Args:
            model_name: The name of the model to load from HuggingFace
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        self.model_name = model_name
        self.device = device_map
        self.model_config = model_config
        self.load_model(model_name, device_map, model_config)

        # Add caching system for model outputs and token information
        self.cache = {
            "token_length": {},  # Cache for token length by text
            "encodings": {},  # Cache for tokenizer encodings
            "perplexity": {},  # Cache for perplexity calculations
            "conditional_ppl": {},  # Cache for conditional perplexity
            "context_rankings": {},  # Cache for context rankings
        }
        self.max_cache_size = 1000  # Limit cache size to prevent memory issues

        # set up the max position embeddings and cache bos num
        self.max_position_embeddings = getattr(
            self.model.config, "max_position_embeddings", 4096
        )
        self.cache_bos_num = 10
        self.prefix_bos_num = 100
        self.context_idxs = []

    def load_model(
        self, model_name: str, device_map: str = "cuda", model_config: dict = {}
    ):
        """
        Load the language model and tokenizer.

        Args:
            model_name: The name of the model to load
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        logger.debug(f"Loading model {model_name} on {device_map}")
        torch_dtype = (
            torch.float16
            if "torch_dtype" not in model_config
            else model_config["torch_dtype"]
        )
        model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype}

        for k, v in model_config.items():
            if k != "torch_dtype":
                model_kwargs[k] = v

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.tokenizer_is_gpt = "gpt" in model_name.lower()
        logger.debug("Model and tokenizer loaded successfully")

    def _manage_cache_size(self, cache_type):
        """
        Manage cache size by removing oldest entries when cache exceeds max size.

        Args:
            cache_type: The type of cache to manage
        """
        if len(self.cache[cache_type]) > self.max_cache_size:
            # Remove 20% of the oldest entries
            remove_count = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.cache[cache_type].keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[cache_type][key]

    def get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
    ):
        """
        Get the number of tokens in the given text.

        Args:
            text: The text to tokenize
            add_special_tokens: Whether to count special tokens

        Returns:
            The number of tokens
        """
        # Create a cache key based on text and parameters
        cache_key = f"{text}_{add_special_tokens}"

        # Check if result is in cache
        if cache_key in self.cache["token_length"]:
            return self.cache["token_length"][cache_key]

        # Calculate token length if not in cache
        token_length = len(
            self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

        # Store in cache
        self.cache["token_length"][cache_key] = token_length
        self._manage_cache_size("token_length")

        return token_length

    def get_ppl(
        self,
        text: str,
        granularity: str = "line",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        """
        Calculate perplexity for the given text at line level.

        Args:
            text: The text to calculate perplexity for
            granularity: The granularity of perplexity calculation (line, token, chunk)
            input_ids, attention_mask, past_key_values: Optional pre-processed inputs
            return_kv: Whether to return key-values
            end: End position for calculation
            condition_mode: Mode for conditional perplexity (none, prefix)
            condition_pos_id: Position ID for condition

        Returns:
            A dictionary with perplexity scores and processing information
        """
        # Create a cache key for this specific perplexity calculation
        cache_key = f"{text}_{granularity}_{condition_mode}_{condition_pos_id}"
        if (
            past_key_values is None
            and not return_kv
            and cache_key in self.cache["perplexity"]
        ):
            return self.cache["perplexity"][cache_key]

        # Initialize input processing
        if input_ids is None:
            encoding_key = text
            if encoding_key in self.cache["encodings"]:
                cached_encoding = self.cache["encodings"][encoding_key]
                input_ids = cached_encoding["input_ids"]
                attention_mask = cached_encoding["attention_mask"]
            else:
                encoding = self.tokenizer(text, return_tensors="pt", padding=True)
                input_ids = encoding["input_ids"].to(self.model.device)
                attention_mask = encoding["attention_mask"].to(self.model.device)

                # Cache the encoding
                self.cache["encodings"][encoding_key] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                self._manage_cache_size("encodings")

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0

        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
            )

        # Get logits and shift
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()

        # Flatten tokens for loss calculation
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)

        # Apply condition filtering if required
        if condition_mode == "prefix":
            loss = loss[condition_pos_id:]

        # Process based on granularity
        if granularity == "token":
            result_loss = loss
        else:
            result_loss = loss.mean()

        # Split text into lines for line-level granularity
        if granularity == "line" and text:
            segments = text.split("\n")
            segments = [seg for seg in segments if seg.strip()]
            lines_info = self.__get_lines_info(segments, input_ids[0], loss)
        else:
            segments = [text] if text else []
            lines_info = []

        # Calculate mean perplexity
        mean_loss = loss.mean() if len(loss) > 0 else torch.tensor(0.0)
        ppl = (
            torch.exp(mean_loss).item()
            if mean_loss.item() != float("inf")
            else float("inf")
        )

        result = {
            "loss": loss,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lines_info": lines_info,
            "segments": segments,
            "ppl": ppl,
        }

        if return_kv:
            result["past_key_values"] = outputs.past_key_values
        else:
            # Cache the result if we're not returning KV cache
            self.cache["perplexity"][cache_key] = result
            self._manage_cache_size("perplexity")

        return result

    def __get_lines_info(self, lines, input_ids, loss):
        """
        Get information about each line including start/end positions and importance.

        Args:
            lines: List of lines in the text
            input_ids: Token IDs for the entire text
            loss: Per-token loss values

        Returns:
            List of dictionaries with line information
        """
        line_info = []
        cumulative_tokens = 0

        input_ids_list = input_ids.cpu().tolist()

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Encode each line to find its token length
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            line_length = len(line_tokens)

            # Find position in the tokenized text
            start_pos = cumulative_tokens
            end_pos = start_pos + line_length

            # Calculate mean loss (importance) for this line
            # Loss might be shorter than the token IDs due to shifting
            if (
                isinstance(loss, torch.Tensor)
                and start_pos < len(loss)
                and end_pos <= len(loss)
            ):
                line_loss = loss[start_pos:end_pos].mean().item()
            else:
                # Handle edge cases
                line_loss = float("inf")

            line_info.append(
                {
                    "line": line,
                    "start": start_pos,
                    "end": end_pos,
                    "importance": line_loss,
                    "tokens": line_length,
                }
            )

            cumulative_tokens += line_length

        return line_info

    def get_prefix_length(self, prefix: str, text: str):
        """
        Calculate the length of a prefix in tokens when concatenated with a text.

        Args:
            prefix: The prefix text
            text: The main text

        Returns:
            Length of the prefix in tokens
        """
        possible_prefix_token = max(self.get_token_length(prefix, False) - 3, 1)
        full_input_ids = self.tokenizer(
            prefix + text[:100], add_special_tokens=False
        ).input_ids

        for i in range(possible_prefix_token, len(full_input_ids)):
            cur_prefix = self.tokenizer.decode(full_input_ids[:i])
            if cur_prefix == prefix:
                break

        return i

    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "line",
    ):
        """
        Calculate perplexity change of a question when given context text.
        A positive change means the context helps reduce question perplexity.

        Args:
            text: The context text
            question: The question to evaluate
            condition_in_question: Conditioning mode (none, prefix)
            granularity: Granularity for perplexity calculation

        Returns:
            Perplexity change for the question with/without context
        """
        # Create a cache key for this conditional perplexity calculation
        cache_key = f"{text}_{question}_{condition_in_question}_{granularity}"

        if cache_key in self.cache["conditional_ppl"]:
            return self.cache["conditional_ppl"][cache_key]

        if condition_in_question == "none":
            # Just return the perplexity of the text
            result = self.get_ppl(
                text=text, granularity=granularity, condition_mode="none"
            )
            ppl_value = result["ppl"]
        else:
            # First calculate question perplexity without context
            question_ppl_without_context = self.get_ppl(
                text=question, granularity=granularity
            )["ppl"]

            # Then calculate question perplexity with context
            question_ppl_with_context = self.get_ppl(
                text=text + "\n\n" + question,
                granularity=granularity,
                condition_mode="prefix",
                condition_pos_id=self.get_token_length(
                    text + "\n\n", add_special_tokens=True
                ),
            )["ppl"]

            # Calculate the change (positive means context helps)
            ppl_value = question_ppl_without_context - question_ppl_with_context

        # Cache the result
        self.cache["conditional_ppl"][cache_key] = ppl_value
        self._manage_cache_size("conditional_ppl")

        return ppl_value

    def get_estimate_threshold_base_distribution(
        self, ppl_values, ratio: float, condition_flag: bool = False
    ):
        """
        Estimate threshold value for compression based on perplexity distribution.

        Args:
            ppl_values: Perplexity values for tokens or lines
            ratio: Compression ratio (0.0-1.0)
            condition_flag: Whether values are conditional (affecting sorting direction)

        Returns:
            Threshold value for filtering
        """
        if ratio >= 1.0:
            return float("-inf")

        if isinstance(ppl_values, torch.Tensor):
            # Filter out extreme values that might skew the threshold
            valid_values = ppl_values[ppl_values != float("inf")]
            valid_values = valid_values[valid_values != -float("inf")]
            valid_values = valid_values[~torch.isnan(valid_values)]

            if len(valid_values) == 0:
                return 0.0

            # Calculate the target position for the percentile
            target_token = max(
                0, min(len(valid_values) - 1, int(len(valid_values) * ratio) - 1)
            )

            # Sort values based on condition_flag and get threshold
            sort_values = valid_values.sort(descending=not condition_flag).values
            if target_token < len(sort_values):
                return sort_values[target_token].item()
            return 0.0
        else:
            # Handle non-tensor inputs (lists, numpy arrays)
            valid_values = [
                v
                for v in ppl_values
                if v != float("inf") and v != -float("inf") and not math.isnan(v)
            ]

            if not valid_values:
                return 0.0

            # Calculate the target position for the percentile
            target_idx = max(
                0, min(len(valid_values) - 1, int(len(valid_values) * ratio) - 1)
            )

            # Sort values and get threshold
            sorted_values = sorted(valid_values, reverse=not condition_flag)
            if target_idx < len(sorted_values):
                return sorted_values[target_idx]
            return 0.0

    def get_dynamic_compression_ratio(
        self,
        context: list,
        target_token: float,
        iterative_size: int,
        dynamic_ratio: list,
        start: int,
    ):
        """
        Calculate dynamic compression ratios for iterative compression.

        Args:
            context: List of context strings
            target_token: Target number of tokens
            iterative_size: Size of each iteration
            dynamic_ratio: List of dynamic ratio adjustments
            start: Start position for processing

        Returns:
            List of ratios for each iteration chunk
        """

        def get_ratio(base: float, delta: float):
            return max(min(1, base + delta), 0)

        context_length = [self.get_token_length(ii, False) + 2 for ii in context]
        if start:
            context_length = context_length[1:]

        tau = target_token / (sum(context_length) + 1)
        res, idx, last, last_target = [], 0, 1, []

        while idx < len(context_length):
            if last + context_length[idx] >= iterative_size:
                last_target.append(
                    (iterative_size - last, get_ratio(tau, dynamic_ratio[idx]))
                )
                res.append(last_target)
                last = last + context_length[idx] - iterative_size

                if last > iterative_size:
                    k = last // iterative_size
                    res.extend(
                        [[(iterative_size, get_ratio(tau, dynamic_ratio[idx]))]] * k
                    )
                    last -= k * iterative_size

                last_target = (
                    [(last, get_ratio(tau, dynamic_ratio[idx]))] if last else []
                )
            else:
                last += context_length[idx]
                last_target.append(
                    (context_length[idx], get_ratio(tau, dynamic_ratio[idx]))
                )
            idx += 1

        if last_target:
            res.append(last_target)

        return res

    def iterative_compress_prompt(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_lines: bool = True,
        start: int = 0,
        dynamic_ratio: list = None,
        condition_compare: bool = False,
    ):
        """
        Iteratively compress text using a sliding window approach with KV caching.

        Args:
            context: List of text contexts to compress
            target_token: Target number of tokens after compression
            iterative_size: Size of each iteration window
            keep_lines: Whether to keep line structure
            start: Start position for processing
            dynamic_ratio: List of dynamic compression ratios
            condition_compare: Whether to use conditional comparison

        Returns:
            Compressed input IDs and attention mask
        """
        # Calculate dynamic compression ratios for each iteration
        iterative_ratios = self.get_dynamic_compression_ratio(
            context, target_token, iterative_size, dynamic_ratio, start
        )

        # Join contexts and tokenize
        context_joined = "\n\n".join(context)
        tokenized_text = self.tokenizer(
            context_joined, return_tensors="pt", add_special_tokens=False
        )
        input_ids = tokenized_text["input_ids"].to(self.model.device)
        attention_mask = tokenized_text["attention_mask"].to(self.model.device)

        # Initialize working variables
        compressed_input_ids, compressed_attention_mask = input_ids, attention_mask
        end = min(iterative_size + start, compressed_input_ids.shape[1])
        threshold, keep_flag = None, None

        if keep_lines:
            # Build a keep flag for important line tokens (e.g., indentation patterns)
            input_ids_numpy = input_ids.cpu().detach().numpy()[0]
            N = len(input_ids_numpy)
            # Identify line break patterns to preserve
            newline_ids = set(self.tokenizer.encode("\n", add_special_tokens=False))
            keep_flag = torch.zeros(N, dtype=torch.bool).to(self.model.device)

            # Mark tokens that represent indentation to be preserved
            for i in range(1, N):
                if input_ids_numpy[i - 1] in newline_ids:
                    # Check if this token is whitespace (indentation)
                    token = self.tokenizer.decode([input_ids_numpy[i]])
                    if token.isspace():
                        keep_flag[i] = True

        # Initialize processing state
        past_key_values, past_loss, ready_end = None, None, 0
        pop_compressed_input_ids = None
        idx = 0

        # Process text in chunks
        while end <= compressed_input_ids.shape[1]:
            # Handle KV-cache window sliding for long texts
            if end > self.max_position_embeddings and past_key_values is not None:
                # KV-Cache Compression
                e, s = (
                    end - self.max_position_embeddings,
                    min(self.cache_bos_num + start, self.max_position_embeddings),
                )
                if pop_compressed_input_ids is None:
                    pop_compressed_input_ids = compressed_input_ids[:, :e]
                else:
                    pop_compressed_input_ids = torch.cat(
                        [pop_compressed_input_ids, compressed_input_ids[:, :e]], dim=-1
                    )
                compressed_input_ids = compressed_input_ids[:, e:]
                compressed_attention_mask = compressed_attention_mask[:, e:]

                # Update KV cache - keep beginning tokens and skip processed tokens
                past_key_values = [
                    [
                        torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                        torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                    ]
                    for k, v in past_key_values
                ]

                if keep_flag is not None:
                    keep_flag = keep_flag[e:]

                end, ready_end = end - e, ready_end - e

            # Calculate perplexity for current window
            result = self.get_ppl(
                "",
                "token",
                compressed_input_ids,
                compressed_attention_mask,
                past_key_values=past_key_values,
                return_kv=True,
                end=end if idx else None,
            )

            loss, past_key_values = result["loss"], result["past_key_values"]

            if loss.shape[0] == 0:
                break

            # Merge with previous loss calculations
            if past_loss is not None:
                if end - 1 > len(past_loss):
                    past_loss = torch.cat(
                        [past_loss, torch.zeros_like(loss)[: end - 1 - len(past_loss)]]
                    )
                past_loss[ready_end : end - 1] = loss
                loss = past_loss
            else:
                past_loss = loss

            # Slide the KV cache window
            if idx:
                past_key_values = [
                    [k[:, :, : end - iterative_size], v[:, :, : end - iterative_size]]
                    for k, v in past_key_values
                ]
            else:
                past_key_values = None

            # Apply compression for each chunk in the current window
            for delta_end, ratio in iterative_ratios[idx]:
                loss = past_loss
                # Calculate threshold for token filtering
                threshold = self.get_estimate_threshold_base_distribution(
                    loss, ratio, False
                )

                # Filter tokens using the calculated threshold
                (
                    compressed_input_ids,
                    compressed_attention_mask,
                    keep_flag,
                    end,
                    past_loss,
                ) = self.get_compressed_input(
                    loss,
                    compressed_input_ids,
                    compressed_attention_mask,
                    end - iterative_size + delta_end,
                    iterative_size=delta_end,
                    threshold=threshold,
                    keep_flag=keep_flag,
                    start=start,
                )

                end += iterative_size

            ready_end = end - iterative_size if not (start and idx == 0) else 0
            idx += 1

        # Concatenate saved tokens with final compressed tokens
        if pop_compressed_input_ids is not None:
            compressed_input_ids = torch.cat(
                [pop_compressed_input_ids, compressed_input_ids], dim=-1
            )

        return compressed_input_ids[:, start:], compressed_attention_mask[:, start:]

    def iterative_compress_prompt_line(
        self,
        context: List[str],
        target_token: float,
        dynamic_ratio: list = None,
    ):
        """
        Compress text by evaluating and filtering entire lines based on importance.
        This is a line-level alternative to the token-level iterative_compress_prompt.

        Args:
            context: List of text contexts to compress
            target_token: Target number of tokens after compression
            dynamic_ratio: List of dynamic compression ratios for each context

        Returns:
            Compressed input IDs and attention mask
        """
        # Join contexts
        context_joined = "\n\n".join(context)

        # Split text into lines
        lines = context_joined.split("\n")

        # Get perplexity for the entire text at line level
        ppl_result = self.get_ppl(context_joined, granularity="line")
        lines_info = ppl_result["lines_info"]

        # Calculate token count for each line
        line_tokens = [
            (i, info["tokens"], info["importance"]) for i, info in enumerate(lines_info)
        ]

        # Apply dynamic ratio adjustments if provided
        if dynamic_ratio and len(dynamic_ratio) > 0:
            # Create dynamic ratios for each line based on context dynamic ratios
            # We'll infer which context each line belongs to
            line_contexts = []
            context_idx = 0
            line_count = 0

            # Map each line to its corresponding context
            for i, info in enumerate(lines_info):
                line_contexts.append(min(context_idx, len(dynamic_ratio) - 1))
                line_count += 1

                # Check if we've reached the end of a context
                if (
                    line_count >= lines.count("\n") + 1
                    and context_idx < len(context) - 1
                ):
                    context_idx += 1
                    line_count = 0

            # Apply dynamic ratio adjustments to line importance scores
            for i in range(len(line_tokens)):
                if i < len(line_contexts):
                    context_idx = line_contexts[i]
                    if context_idx < len(dynamic_ratio):
                        # Adjust importance using dynamic ratio
                        # Lower importance score means higher priority (will be kept)
                        adjustment = dynamic_ratio[context_idx]
                        line_tokens[i] = (
                            line_tokens[i][0],
                            line_tokens[i][1],
                            line_tokens[i][2]
                            - adjustment,  # Lower importance means keep
                        )

        # Sort lines by importance (lower score is more important)
        sorted_lines = sorted(line_tokens, key=lambda x: x[2])

        # Select lines to keep within token budget
        tokens_so_far = 0
        lines_to_keep = set()

        for line_idx, line_tokens, _ in sorted_lines:
            if tokens_so_far + line_tokens <= target_token:
                lines_to_keep.add(line_idx)
                tokens_so_far += line_tokens
            else:
                # Stop if we've reached our target
                break

        # Create compressed text with only the selected lines
        compressed_lines = [lines_info[i]["line"] for i in sorted(lines_to_keep)]
        compressed_text = "\n".join(compressed_lines)

        # Tokenize the compressed text
        tokenized_text = self.tokenizer(
            compressed_text, return_tensors="pt", add_special_tokens=False
        )
        compressed_input_ids = tokenized_text["input_ids"].to(self.model.device)
        compressed_attention_mask = tokenized_text["attention_mask"].to(
            self.model.device
        )

        return compressed_input_ids, compressed_attention_mask

    def get_compressed_input(
        self,
        loss,
        input_ids,
        attention_mask,
        end=200,
        iterative_size=200,
        threshold=0.5,
        keep_flag=None,
        start: int = 0,
    ):
        """
        Filter input tokens based on loss values and thresholds.

        Args:
            loss: Loss values for each token
            input_ids: Input token IDs
            attention_mask: Attention mask
            end: End position for processing
            iterative_size: Size of each iteration
            threshold: Threshold value for filtering
            keep_flag: Flags for tokens to always keep
            start: Start position for processing

        Returns:
            Compressed inputs and updated state
        """
        # Determine which tokens to keep based on loss values and threshold
        need_idx = torch.concat([loss > threshold, loss[:1] > 0])

        # Ensure we keep tokens at positions outside our current window
        need_idx[end:] = 1
        need_idx[: end - iterative_size] = 1

        # Get filtered loss
        loss = loss[need_idx[:-1]]

        # Ensure need_idx matches input_ids length
        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat(
                [
                    need_idx,
                    torch.ones(
                        input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool
                    ).to(need_idx.device),
                ]
            )
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[: input_ids.shape[1]]

        # Enforce keeping tokens marked in keep_flag
        if keep_flag is not None:
            need_idx[keep_flag] = 1

            # Optionally apply line break preservation logic
            # Find tokens representing newlines and always keep one of consecutive newlines
            tokens = input_ids[0]
            newline_ids = set(self.tokenizer.encode("\n", add_special_tokens=False))
            last_kept_newline = False

            for ii in range(max(0, end - iterative_size), end):
                if need_idx[ii] == 0:
                    continue

                token_id = tokens[ii].item()

                # Handle newline logic - avoid consecutive newlines unless marked important
                if token_id in newline_ids:
                    if last_kept_newline and keep_flag[ii].item() == 0:
                        need_idx[ii] = 0
                    else:
                        last_kept_newline = True
                else:
                    last_kept_newline = False

        # Apply the filtering to get compressed tokens
        compressed_input_ids = input_ids[attention_mask == 1][need_idx].unsqueeze(0)
        compressed_attention_mask = attention_mask[attention_mask == 1][
            need_idx
        ].unsqueeze(0)

        # Update the end position based on how many tokens we removed
        end -= (need_idx[:end] == 0).sum()

        return compressed_input_ids, compressed_attention_mask, keep_flag, end, loss

    def compress_code(
        self,
        code: str,
        query: str = "",
        instruction: str = "",
        rate: float = 0.5,
        target_token: int = -1,
        use_line_level_filter: bool = True,
        use_iterative_compression: bool = True,
        iterative_size: int = 200,
        dynamic_compression_ratio: float = 0.2,
    ):
        """
        Compress code by removing less important lines based on query relevance.

        Args:
            code: The code to compress
            query: Query to prioritize relevant lines
            instruction: Additional instruction to guide compression
            rate: Compression rate (0.0-1.0), where 1.0 means no compression
            target_token: Target number of tokens (alternative to rate)
            use_line_level_filter: Whether to use line-level filtering
            use_iterative_compression: Whether to use token-level iterative compression
            iterative_size: Size of each iteration for token-level compression
            dynamic_compression_ratio: Ratio for dynamic compression (0.0-1.0)

        Returns:
            Compressed code and statistics
        """
        logger.debug(
            f"Starting code compression with rate={rate}, target_token={target_token}"
        )
        start_time = time.time()

        # Calculate total tokens in the code
        total_tokens = self.get_token_length(code)
        logger.debug(f"Total tokens in code: {total_tokens}")

        # Determine target tokens
        if target_token <= 0:
            target_token = int(total_tokens * rate)
        logger.debug(f"Target tokens: {target_token}")

        if rate >= 1.0 or target_token >= total_tokens:
            # No compression needed
            return {
                "original_code": code,
                "compressed_code": code,
                "output": code,
                "original_tokens": total_tokens,
                "compressed_tokens": total_tokens,
                "final_compressed_tokens": total_tokens,
                "compression_ratio": 1.0,
                "kept_lines": list(range(len(code.split("\n")))),
            }

        # For very small code snippets, skip iterative compression
        if total_tokens < 100:
            use_iterative_compression = False

        if use_line_level_filter:
            # Split code into lines for line-level filtering
            lines = code.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            logger.debug(f"Split code into {len(non_empty_lines)} non-empty lines")

            # Get perplexity for entire code
            ppl_result = self.get_ppl(code, granularity="line")
            lines_info = ppl_result["lines_info"]

            # For query is provided, rank lines by relevance
            if query:
                logger.debug("Ranking lines by relevance to query")
                # Get conditional perplexity for each line
                line_importances = []
                for i, line_info in tqdm(
                    enumerate(lines_info),
                    total=len(lines_info),
                    desc="Calculating line importance",
                ):
                    # First calculate the perplexity of the query without the line
                    query_ppl_without_context = self.get_ppl(query, granularity="line")[
                        "ppl"
                    ]

                    # Then calculate the perplexity of the query with the line as context
                    query_ppl_with_context = self.get_ppl(
                        line_info["line"] + "\n\n" + query,
                        granularity="line",
                        condition_mode="prefix",
                        condition_pos_id=self.get_token_length(
                            line_info["line"] + "\n\n", add_special_tokens=True
                        ),
                    )["ppl"]

                    # Calculate the perplexity change (lower value means context is more helpful)
                    ppl_change = query_ppl_without_context - query_ppl_with_context

                    # Add length adjustment similar to before
                    line_importances.append(
                        (i, -ppl_change - line_info["tokens"] * 2 / 250 * 0)
                    )

                # Sort by importance (higher perplexity reduction = more relevant to query)
                sorted_lines = sorted(line_importances, key=lambda x: x[1])
            else:
                # Sort lines by importance (lower loss = more important)
                line_importances = [
                    (i, info["importance"]) for i, info in enumerate(lines_info)
                ]
                sorted_lines = sorted(line_importances, key=lambda x: x[1])

            # Apply dynamic compression ratio if specified
            if dynamic_compression_ratio > 0:
                N = len(sorted_lines)
                # This creates a gradient of compression rates from higher to lower importance
                dynamic_ratios = [
                    i * (dynamic_compression_ratio / (N - 1)) if N > 1 else 0
                    for i in range(-(N - 1), N, 2)
                ]

                # Assign dynamic ratios to lines based on their importance rank
                sorted_indices = [idx for idx, _ in sorted_lines]
                dynamic_ratio_map = {
                    idx: ratio for idx, ratio in zip(sorted_indices, dynamic_ratios)
                }
            else:
                dynamic_ratio_map = {i: 0 for i in range(len(lines_info))}

            # Determine which lines to keep based on token budget
            tokens_so_far = 0
            lines_to_keep = set()

            # First pass - keep most important lines within budget
            for line_idx, _ in sorted_lines:
                if line_idx >= len(lines_info):
                    continue

                line_info = lines_info[line_idx]
                line_tokens = line_info["tokens"]

                if tokens_so_far + line_tokens <= target_token:
                    lines_to_keep.add(line_idx)
                    tokens_so_far += line_tokens
                else:
                    # Stop if we've reached our target
                    break

            logger.debug(
                f"Selected {len(lines_to_keep)} lines to keep out of {len(lines_info)}"
            )

            # Construct code with only the selected lines
            preserved_code = "\n".join(
                [lines_info[i]["line"] for i in sorted(lines_to_keep)]
            )

            # If we need iterative token-level compression
            if use_iterative_compression:
                logger.debug("Applying iterative line-level compression")

                # Create dynamic ratios for iterative compression
                dynamic_ratios = [
                    dynamic_ratio_map.get(i, 0.0) for i in sorted(lines_to_keep)
                ]

                # Convert to list for iterative compression
                context = [preserved_code]

                # Apply line-level compression instead of token-level compression
                compressed_ids, compressed_mask = self.iterative_compress_prompt_line(
                    context,
                    target_token=target_token,
                    dynamic_ratio=dynamic_ratios,
                )

                # Convert back to text
                compressed_code = self.tokenizer.decode(compressed_ids[0])
            else:
                compressed_code = preserved_code
        else:
            # Without line-level filter, apply iterative compression directly
            if use_iterative_compression:
                logger.debug(
                    "Applying iterative line-level compression without line filtering"
                )

                # Apply line-level compression to the entire code
                compressed_ids, _ = self.iterative_compress_prompt_line(
                    [code],
                    target_token=target_token,
                    dynamic_ratio=[
                        0.0
                    ],  # No dynamic ratio adjustment for single context
                )

                # Convert back to text
                compressed_code = self.tokenizer.decode(compressed_ids[0])
            else:
                # Simple truncation
                logger.debug("No compression methods selected, using simple truncation")
                encoded = self.tokenizer.encode(code, add_special_tokens=False)
                truncated = encoded[:target_token]
                compressed_code = self.tokenizer.decode(truncated)

        # Construct final output with instruction and query
        output = ""
        if instruction:
            output += f"{instruction}\n\n"
        output += compressed_code
        if query:
            output += f"\n\n{query}"

        # Calculate compression statistics
        compressed_tokens = self.get_token_length(compressed_code)
        final_compressed_tokens = self.get_token_length(output)
        compression_ratio = (
            compressed_tokens / total_tokens if total_tokens > 0 else 1.0
        )

        end_time = time.time()
        logger.debug(
            f"Code compression completed in {end_time - start_time:.2f} seconds"
        )
        logger.debug(f"Compression ratio: {compression_ratio:.2f}")

        # For line-level filtering, include which lines were kept
        if use_line_level_filter:
            kept_lines = sorted(lines_to_keep)
        else:
            # Approximate which lines were kept based on content
            original_lines = code.split("\n")
            compressed_lines = compressed_code.split("\n")
            kept_lines = []
            for i, line in enumerate(original_lines):
                if line in compressed_lines:
                    kept_lines.append(i)

        return {
            "original_code": code,
            "compressed_code": compressed_code,
            "output": output,
            "original_tokens": total_tokens,
            "compressed_tokens": compressed_tokens,
            "final_compressed_tokens": final_compressed_tokens,
            "compression_ratio": compression_ratio,
            "kept_lines": kept_lines,
        }

    def control_context_budget(
        self,
        context_list: List[str],
        target_token: float,
        question: str = "",
        reorder_context: str = "original",
        condition_in_question: str = "none",
        force_context_ids: List[int] = None,
        force_context_number: int = None,
        context_budget: str = "+100",
        dynamic_context_compression_ratio: float = 0.0,
    ):
        """
        Control token budget for contexts based on relevance ranking, following LongLLMLingua.

        Args:
            context_list: List of contexts
            target_token: Target number of tokens
            question: Question for relevance ranking
            reorder_context: How to reorder contexts ("original", "importance", "two_stage")
            condition_in_question: Mode for conditional ranking
            force_context_ids: List of context IDs to always include
            force_context_number: Number of contexts to forcibly include
            context_budget: String expression to modify target token budget
            dynamic_context_compression_ratio: Ratio for dynamic compression (0.0-1.0)

        Returns:
            Selected contexts, their indices, and dynamic ratios
        """
        logger.debug(f"Controlling context budget with target_token={target_token}")
        start_time = time.time()

        if not context_list:
            return [], [], []

        # Get token counts for each context
        logger.debug("Calculating token lengths for contexts")
        context_tokens_length = [
            self.get_token_length(context) for context in context_list
        ]

        # If total tokens already fit within budget, return all contexts
        total_tokens = sum(context_tokens_length)
        if total_tokens <= target_token:
            logger.debug(
                f"All contexts fit within budget ({total_tokens} <= {target_token})"
            )
            end_time = time.time()
            logger.debug(
                f"Context budget control completed in {end_time - start_time:.2f} seconds"
            )
            return (
                context_list,
                list(range(len(context_list))),
                [0.0] * len(context_list),
            )

        # Rank contexts by relevance if question is provided
        logger.debug("Ranking contexts by relevance")
        if question:
            # Get perplexity change for each context with the question
            context_ppl_changes = []
            for d, dl in zip(context_list, context_tokens_length):
                # Calculate how much this context reduces question perplexity
                ppl_change = self.get_condition_ppl(
                    d,
                    question,
                    condition_in_question,
                )
                # Apply length adjustment factor similar to before
                context_ppl_changes.append(ppl_change - dl * 2 / 250 * 0)

            # Sort by perplexity change - higher is better (more reduction in question perplexity)
            demonstrations_sort = sorted(
                enumerate(context_ppl_changes), key=lambda x: -x[1]
            )
        else:
            # Without question, use default ordering
            demonstrations_sort = [(i, 0) for i in range(len(context_list))]

        # Extract ranking for later use
        self.context_idxs.append([x for idx, (x, _) in enumerate(demonstrations_sort)])

        # Calculate the target token budget with context_budget expression
        if target_token < 0:
            target_token = 100
        target_token = eval("target_token" + context_budget)

        # Initialize selected context tracking
        used = force_context_ids if force_context_ids is not None else []

        # Select contexts until we reach the token budget
        for idx, _ in demonstrations_sort:
            if idx >= len(context_tokens_length):
                continue
            target_token -= context_tokens_length[idx]
            if idx not in used:
                used.append(idx)
            if target_token < 0 or (
                force_context_number is not None and len(used) >= force_context_number
            ):
                break

        # Store original selection order
        original_used = used.copy()

        # Reorder contexts if requested
        if reorder_context == "original":
            used = sorted(used)
        elif reorder_context == "two_stage":
            l, r = (
                [_ for idx, _ in enumerate(used) if idx % 2 == 0],
                [_ for idx, _ in enumerate(used) if idx % 2 == 1],
            )
            used = l + r[::-1]

        # Calculate dynamic compression ratios if requested
        if dynamic_context_compression_ratio > 0:
            N = len(used)
            dynamic_ratio = [
                i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
                for i in range(-(N - 1), N, 2)
            ][::-1]
            dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
            dynamic_ratio = [dynamic_ratio_map[i] for i in used]
        else:
            dynamic_ratio = [0.0] * len(used)

        # Build list of selected contexts
        selected_contexts = [
            context_list[idx] for idx in used if idx < len(context_list)
        ]

        end_time = time.time()
        logger.debug(
            f"Selected {len(selected_contexts)} contexts out of {len(context_list)}"
        )
        logger.debug(
            f"Context budget control completed in {end_time - start_time:.2f} seconds"
        )

        return selected_contexts, used, dynamic_ratio, demonstrations_sort

    def compress_code_file(
        self,
        code: str,
        query: str = "",
        instruction: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        language: str = "python",
        use_iterative_compression: bool = True,
        iterative_size: int = 200,
        dynamic_compression_ratio: float = 0.2,
        context_budget: str = "+100",
        rank_only: bool = False,
        pruner=None,
    ):
        """
        Compress a code file by first splitting it into function-based chunks and then compressing.
        Functions are prioritized based on query relevance, similar to LongLLMLingua.

        Args:
            code: The code to compress
            query: Query to prioritize relevant functions
            instruction: Additional instruction to guide compression
            rate: Compression rate (0.0-1.0)
            target_token: Target number of tokens (alternative to rate)
            language: Programming language of the code
            use_iterative_compression: Whether to use iterative compression
            iterative_size: Size of each iteration for iterative compression
            dynamic_compression_ratio: Ratio for dynamic compression
            context_budget: String expression to modify token budget
            rank_only: If True, just rank and select contexts without fine-grained compression
            pruner: Optional pruner model to further prune the compressed chunks
            pre_prune: If True, prune chunks before compression; if False, prune after compression

        Returns:
            Compressed code and statistics, including compressed_chunks for further processing
        """
        logger.debug(
            f"Starting code file compression with rate={rate}, target_token={target_token}, language={language}"
        )
        start_time = time.time()

        # Split code into function-based chunks
        logger.debug("Splitting code into function-based chunks")
        code_chunks = self.split_code_by_functions(code, language=language)
        logger.debug(f"Split code into {len(code_chunks)} chunks")

        # Calculate total tokens
        logger.debug("Calculating total tokens")
        total_tokens = sum(self.get_token_length(chunk) for chunk in code_chunks)
        logger.debug(f"Total tokens: {total_tokens}")

        # If target token is not provided, use rate
        if target_token <= 0:
            target_token = int(total_tokens * rate)
        logger.debug(f"Target tokens: {target_token}")

        # Use context budget control to select important functions
        logger.debug("Selecting important functions using context budget control")
        selected_contexts, selected_indices, dynamic_ratios, demonstrations_sort = (
            self.control_context_budget(
                code_chunks,
                target_token=target_token,
                question=query,
                reorder_context="original",  # Keep original order to maintain code structure
                condition_in_question="prefix",
                context_budget=context_budget,
                dynamic_context_compression_ratio=dynamic_compression_ratio,
            )
        )

        # If rank_only is True, just use the selected contexts without further compression
        if rank_only:
            logger.debug(
                "Using rank-only mode: selecting top functions without fine-grained compression"
            )
            compressed_chunks = []
            compressed_tokens = 0
            function_compressions = {}

            # Just keep the selected contexts as is
            for i, chunk in enumerate(code_chunks):
                if i in selected_indices:
                    compressed_chunks.append(chunk)
                    chunk_tokens = self.get_token_length(chunk)
                    compressed_tokens += chunk_tokens

                    # Store compression info - no actual compression in this mode
                    function_compressions[i] = {
                        "original_tokens": chunk_tokens,
                        "compressed_tokens": chunk_tokens,
                        "compression_ratio": 1.0,
                    }
                else:
                    # Skip this function completely
                    comment_marker = (
                        "#"
                        if language.lower() in ["python", "typescript", "rust"]
                        else "//"
                    )
                    omission_text = f"{comment_marker} ... "
                    compressed_chunks.append(omission_text)
                    compressed_tokens += self.get_token_length(omission_text)

            # Apply pruner if provided
            if pruner:
                logger.debug("Pre-pruning compressed chunks before combination")
                compressed_chunks = pruner.prune_batch(query, compressed_chunks)
            # Combine compressed chunks
            compressed_code = "\n\n".join(compressed_chunks)
            output = f"{instruction}\n\n{compressed_code}\n\n{query}\n{instruction}"

            # Calculate actual compressed tokens
            final_compressed_tokens = self.get_token_length(output)

            end_time = time.time()
            logger.debug(
                f"Code file compression completed in {end_time - start_time:.2f} seconds"
            )
            logger.debug(
                f"Compression ratio: {compressed_tokens / total_tokens if total_tokens > 0 else 1.0:.2f}"
            )

            return {
                "original_code": code,
                "compressed_code": compressed_code,
                "compressed_prompt": output,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "final_compressed_tokens": final_compressed_tokens,
                "compression_ratio": compressed_tokens / total_tokens
                if total_tokens > 0
                else 1.0,
                "function_compressions": function_compressions,
                "selected_functions": selected_indices,
                "demonstrations_sort": demonstrations_sort,
                "compressed_chunks": compressed_chunks,  # Return chunks for further processing
            }

        # Compress each function according to its importance
        logger.debug("Compressing selected functions")
        compressed_chunks = []
        compressed_tokens = 0
        function_compressions = {}

        # Allocate tokens proportionally based on importance
        importance_scores = {}
        for i, idx in enumerate(selected_indices):
            # Higher importance for functions mentioned early in ranking
            importance_scores[idx] = len(selected_indices) - i

        # Calculate total importance
        total_importance = sum(importance_scores.values()) if importance_scores else 1

        # Allocate tokens based on importance
        token_allocation = {}
        for idx, importance in importance_scores.items():
            allocation = max(10, int(target_token * importance / total_importance))
            token_allocation[idx] = min(
                allocation, self.get_token_length(code_chunks[idx])
            )

        # Adjust allocations to fit target
        logger.debug("Adjusting token allocations to fit target")
        while sum(token_allocation.values()) > target_token:
            max_idx = max(token_allocation, key=token_allocation.get)
            token_allocation[max_idx] = max(0, token_allocation[max_idx] - 10)
        # Show the allocation
        logger.debug(f"Token allocation: {token_allocation}")

        # Process each chunk
        for i, chunk in tqdm(
            enumerate(code_chunks), total=len(code_chunks), desc="Compressing functions"
        ):
            if i in token_allocation and token_allocation[i] > 0:
                # Calculate compression rate for this chunk
                chunk_tokens = self.get_token_length(chunk)
                chunk_rate = token_allocation[i] / chunk_tokens

                # Apply dynamic compression ratio based on importance
                dynamic_ratio = (
                    dynamic_ratios[selected_indices.index(i)]
                    if i in selected_indices
                    else 0.0
                )

                # Compress the chunk using line-level compression if requested
                if use_iterative_compression and chunk_tokens > 50:
                    compressed_input_ids, _ = self.iterative_compress_prompt_line(
                        [chunk],
                        target_token=token_allocation[i],
                        dynamic_ratio=[dynamic_ratio],
                    )
                    compressed_chunk = self.tokenizer.decode(compressed_input_ids[0])
                else:
                    # Use simple line-level compression for smaller chunks
                    compress_result = self.compress_code(
                        code=chunk,
                        query=query,
                        rate=chunk_rate,
                        use_iterative_compression=False,
                    )
                    compressed_chunk = compress_result["compressed_code"]

                compressed_chunks.append(compressed_chunk)
                chunk_compressed_tokens = self.get_token_length(compressed_chunk)
                compressed_tokens += chunk_compressed_tokens

                # Store compression info for this function
                function_compressions[i] = {
                    "original_tokens": chunk_tokens,
                    "compressed_tokens": chunk_compressed_tokens,
                    "compression_ratio": chunk_compressed_tokens / chunk_tokens
                    if chunk_tokens > 0
                    else 1.0,
                }
            else:
                # Skip this function completely
                comment_marker = (
                    "#"
                    if language.lower() in ["python", "typescript", "rust"]
                    else "//"
                )
                # omission_text = f"{comment_marker} ... function omitted ..."
                omission_text = f"{comment_marker} ... "
                compressed_chunks.append(omission_text)
                compressed_tokens += self.get_token_length(omission_text)

        # Apply pruner if provided
        if pruner:
            logger.debug("Pre-pruning compressed chunks before combination")
            compressed_chunks = pruner.prune_batch(query, compressed_chunks)
        # Combine compressed chunks
        logger.debug("Combining compressed chunks")
        compressed_code = "\n\n".join(compressed_chunks)

        output = f"{instruction}\n\n{compressed_code}\n\n{query}\n{instruction}"

        # Calculate actual compressed tokens including instruction and query
        final_compressed_tokens = self.get_token_length(output)

        end_time = time.time()
        logger.debug(
            f"Code file compression completed in {end_time - start_time:.2f} seconds"
        )
        logger.debug(
            f"Compression ratio: {compressed_tokens / total_tokens if total_tokens > 0 else 1.0:.2f}"
        )

        return {
            "original_code": code,
            "compressed_code": compressed_code,
            "compressed_prompt": output,
            "original_tokens": total_tokens,
            "compressed_tokens": compressed_tokens,
            "final_compressed_tokens": final_compressed_tokens,
            "compression_ratio": compressed_tokens / total_tokens
            if total_tokens > 0
            else 1.0,
            "function_compressions": function_compressions,
            "selected_functions": selected_indices,
            "demonstrations_sort": demonstrations_sort,
            "compressed_chunks": compressed_chunks,  # Return chunks for further processing
        }

    def split_code_by_functions(self, code: str, language: str = "python") -> List[str]:
        """
        Split code into chunks based on function and class definitions for various languages.

        Args:
            code: The code to split
            language: Programming language of the code (python, cpp, java, typescript, rust, go)

        Returns:
            List of code chunks, each containing a function, class, or class method
        """
        logger.debug(
            f"Splitting code by functions and classes for language: {language}"
        )
        start_time = time.time()

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

        # Find all function and class definitions
        matches = list(function_pattern.finditer(code))
        logger.debug(f"Found {len(matches)} function and class definitions")

        # If no functions or classes found, return the whole code as one chunk
        if not matches:
            logger.debug(
                "No functions or classes found, returning entire code as one chunk"
            )
            end_time = time.time()
            logger.debug(
                f"Code splitting completed in {end_time - start_time:.2f} seconds"
            )
            return [code]

        # Extract chunks that include function and class definitions
        chunks = []

        # Add imports and other code before the first function or class
        if matches[0].start() > 0:
            chunks.append(code[: matches[0].start()])

        # Process each function or class match
        for i, match in enumerate(matches):
            # Get the current function or class
            start = match.start()

            # Determine end position (either the start of the next function/class or the end of the code)
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(code)

            # Extract the function/class and its body
            chunks.append(code[start:end])

        end_time = time.time()
        logger.debug(f"Code splitting completed in {end_time - start_time:.2f} seconds")
        logger.debug(f"Split code into {len(chunks)} chunks")

        return chunks


def load_examples(language: Optional[str] = None) -> List[Dict]:
    """Load examples from the results file, optionally filtered by language"""
    with open("../results/ntoken_16384/Qwen_slash_Qwen2.5-7B-Instruct.jsonl", "r") as f:
        # with open("../results/ntoken_16384/Qwen_slash_Qwen2.5-7B-Instruct-GPTQ-Int4.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    if language:
        data = [example for example in data if example["language"] == language]
        if not data:
            available_languages = set(ex["language"] for ex in data)
            raise ValueError(
                f"No examples found for language '{language}'. Available languages: {available_languages}"
            )

    return data


# Simple test code
if __name__ == "__main__":
    # Load real examples from the dataset
    examples = load_examples(language="python")
    example = examples[0]  # Use the first example
    sample_code = example["code_context"]
    query = example["description"]
    language = example["language"]

    print(f"Using example with language: {language}")
    print(f"Query: {query}")

    # Initialize compressor
    print("Initializing compressor...")
    compressor = CodeCompressor()

    # Test function-based code file compression with query
    print("\nTesting function-based code file compression with query...")

    start_time = time.time()
    file_result = compressor.compress_code_file(
        code=sample_code, query=query, rate=0.1, language=language
    )
    end_time = time.time()

    print(
        f"File compression with query completed in {end_time - start_time:.2f} seconds"
    )
    print(f"Original tokens: {file_result['original_tokens']}")
    print(f"Compressed tokens: {file_result['compressed_tokens']}")
    print(
        f"Final compressed tokens (with query): {file_result['final_compressed_tokens']}"
    )
    print(f"Compression ratio: {file_result['compression_ratio']:.2f}")
    print(f"Kept function IDs: {file_result['selected_functions']}")
    print(f"Demonstrations sort: {file_result['demonstrations_sort']}")

    chunk_ppl_scores = {idx: score for idx, score in file_result["demonstrations_sort"]}
    top_5_score = sorted(chunk_ppl_scores.values(), reverse=True)[5]
    # Split into chunks and show the chunks
    chunks = compressor.split_code_by_functions(sample_code, language=language)
    print(f"Split code into {len(chunks)} chunks")
    # show the chunk with corresponding ppl score
    for i, chunk in enumerate(chunks):
        print(
            f"==========Chunk {i + 1} with demonstration sort: {chunk_ppl_scores[i]}=========="
        )
        if chunk_ppl_scores[i] >= top_5_score:
            print(chunk)
            print("\n")
        else:
            # only show some lines and then use ... to indicate the rest
            print(chunk[:100])
            print("...")
            print(chunk[-100:])
            print("\n")

    print("\nCompressed Code File with Query:")
    print("-------------------")
    print(file_result["compressed_code"])
