from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Protocol, List
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import requests
import logging


class PrunerModel(Protocol):
    def prune(query: str, origin_code: str) -> str: ...
    def prune_batch(
        query: str,
        origin_codes: List[str],
        batch_size=16,
        sort: bool = False,
        return_raw_body: bool = False,
    ) -> List[str] | List[dict]: ...


# Prompt template for line-level code pruning
PRUNE_PROMPT_TEMPLATE = """
You are given:
- a natural-language query or code query. If query is in natual language, nothing special.Else, the query is for **code completion task**, You SHOULD KNOW that its `answer` will be the **next line** of the query.
- a code snippet split into numbered lines (1>, 2>, 3>, ...)

Question: {query}
Code Context:
{code}

Answer the Question, using ONLY information provided in the Code Context. If no useful information
is provided, you MUST output "No answer". If some parts of the Context are used to answer, you
MUST cite ALL the corresponding lines. 

Use the symbols [ ] to indicate when a fact comes from a line in the context, e.g [1] for a fact from line 1. 
- For multi-line context, use [line1-line2], e.g. [12-25]). 
- For multi context, use [line1,line2,...], e.g. [1,3,5-7].

You should only answer the given question and should not provide any additional information

HINT: 
- For code, context should be wider than `the line just answer the question`, for example, if the question is about a variable in a class method function, include the function definition, class definition and where it is used.
- When you try to cite something, its better to cite the structure of the code. 
e.g. if you want to cite B1 in the code structure below:
```
1> if cond:
2>    A1
3>    A2
4> else:
5>    B1
```
, best citation will be [1,4,5], which keeps the structure of the `if-else` block while removing the unrelated A1, A2.

Now give your answer with citations:
"""


def code_formatter(code: str, splitter: str = "line") -> str:
    """Format code with numbered lines/statements for LLM labeling.

    splitter is accepted for backward compatibility; only line-wise numbering is supported.
    """
    lines = code.splitlines()
    formatted_lines = []
    for idx, line in enumerate(lines, start=1):
        formatted_lines.append(f"{idx}> {line}")
    return "\n".join(formatted_lines)


class SilverLabelPrunerModel(PrunerModel):
    """Pruner model that uses vLLM for offline inference or OpenAI endpoint for online inference."""

    def __init__(
        self,
        model: ChatOpenAI = None,
        vllm_model_name: str = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        batch_size: int = 32,
        tensor_parallel_size: int = 8,
        online_mode: bool = False,
        api_base: str = None,
        api_key: str = None,
        model_name: str = None,
    ):
        """
        Initialize the pruner model.

        Args:
            model: ChatOpenAI model for single inference (optional)
            vllm_model_name: Path/name of the vLLM model (offline mode)
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            batch_size: Batch size for inference
            tensor_parallel_size: Tensor parallel size for vLLM (offline mode)
            online_mode: Whether to use online mode (OpenAI endpoint) or offline mode (vLLM)
            api_base: Base URL for OpenAI API (online mode)
            api_key: API key for OpenAI API (online mode)
            model_name: Model name for OpenAI API (online mode)
        """
        self.model = model
        self.prompt_template = PRUNE_PROMPT_TEMPLATE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.vllm_model_name = vllm_model_name
        self.online_mode = online_mode

        # Online mode configuration
        if online_mode:
            if not api_base:
                raise ValueError("api_base is required for online mode")
            if not model_name:
                raise ValueError("model_name is required for online mode")

            # Initialize OpenAI client for online mode
            self.api_base = api_base
            self.api_key = api_key or "dummy-key"  # Allow dummy key for local servers
            self.model_name = model_name

            # Create ChatOpenAI instance for online mode if not provided
            if not self.model:
                self.model = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=self.api_base,
                    api_key=self.api_key,
                )

            print(
                f"Initialized online mode with API base: {api_base}, model: {model_name}"
            )
        else:
            # Offline mode - initialize vLLM
            if vllm_model_name:
                vllm_model_name = vllm_model_name.removeprefix("'").removesuffix(
                    "'"
                )  # remove quotes if any
                print(f"Initializing offline vLLM with model: {vllm_model_name}")
                self.batch_model = LLM(
                    model=vllm_model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=8192,
                )

    def _kept_frags_to_code(self, kept_frags: List[int], origin_code: str) -> str:
        lines = origin_code.splitlines()
        kept_lines = sorted(list(set(kept_frags)))

        # Fill single line gaps
        extra_kept_lines = []
        for idx, k in enumerate(kept_lines):
            if idx > 0 and k - kept_lines[idx - 1] == 2:
                extra_kept_lines.append(k - 1)
        kept_lines.extend(extra_kept_lines)
        kept_lines = sorted(list(set(kept_lines)))

        kept_code_lines = []
        filtered_lines_cnt = 0
        filtered_char_cnt = 0
        s_format = "(filtered {} lines)"

        for line in range(1, len(lines) + 1):
            if lines[line - 1].strip() == "":
                filtered_lines_cnt += 1
                continue

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

        return "\n".join(kept_code_lines)

    def _fetch_kept_frags_from_output(self, out: str) -> List[int]:
        try:
            citations = []
            matches = re.findall(r"\[(?:\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]", out)
            no_answer = re.search(r"No answer", out, re.IGNORECASE)
            if no_answer:
                print(f"[DEBUG] Totally pruned.")
                return []
            if not matches:
                print(
                    f"[DEBUG] Invalid no citation matches found in output. Output preview: {out}"
                )
            for match in matches:
                nums = match.strip("[]").strip().split(",")
                for num in nums:
                    if "-" in num:  # range case
                        start, end = map(int, num.split("-"))
                        citations.extend(range(start, end + 1))
                    else:  # single number case
                        citations.append(int(num))

            citations = sorted(set(citations))  # remove duplicates and sort
            print(f"[DEBUG] Extracted {len(citations)} citations ({citations})")
        except Exception as e:
            print(f"Error parsing output: {e}")
            print(f"[DEBUG] Failed output preview: {out[:500]}")
            return []
        return citations

    def prune(self, query: str, origin_code: str) -> str:
        if self.model is None:
            raise ValueError("ChatOpenAI model not initialized for single inference")

        formatted_code = code_formatter(origin_code, splitter="line")

        p = ChatPromptTemplate.from_template(self.prompt_template)
        chain = p | self.model
        msg = chain.invoke({"query": query, "code": formatted_code})
        kept_frags = self._fetch_kept_frags_from_output(msg.content)
        return self._kept_frags_to_code(kept_frags, origin_code)

    def prune_batch(
        self,
        query: str,
        origin_codes: List[str],
        batch_size: int = -1,
        sort: bool = False,
        return_raw_body: bool = False,
    ) -> List[str] | List[dict]:
        assert not sort, "Sorting is not supported in llm silver label pruning"
        if batch_size == -1:
            batch_size = self.batch_size

        # Format all codes with line numbers
        formatted_codes = [
            code_formatter(code, splitter="line") for code in origin_codes
        ]

        # Build prompts
        prompts = []
        for formatted_code in formatted_codes:
            prompt = self.prompt_template.format(query=query, code=formatted_code)
            prompts.append(prompt)

        pruned_codes = []
        raw_bodies = []

        if self.online_mode:
            # Online mode: use OpenAI API
            system_prompt = "You are a code pruning expert. Identify which lines are relevant to the given query."

            # Process in batches
            for i in tqdm(
                range(0, len(prompts), batch_size), desc="Pruning code batch (online)"
            ):
                batch_prompts = prompts[i : i + batch_size]
                batch_codes = origin_codes[i : i + batch_size]
                batch_formatted = formatted_codes[i : i + batch_size]

                try:
                    # Process each prompt in the batch
                    for j, (prompt, orig_code, formatted_code) in enumerate(
                        zip(batch_prompts, batch_codes, batch_formatted)
                    ):
                        try:
                            # Create messages for OpenAI API
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt},
                            ]

                            # Invoke the model
                            response = self.model.invoke(messages)
                            kept_frags = self._fetch_kept_frags_from_output(
                                response.content
                            )
                            pruned_code = self._kept_frags_to_code(
                                kept_frags, orig_code
                            )
                            # Debug: Check if pruned code contains "..." and compression ratio
                            if j < 2:  # Only log first 2 items to avoid spam
                                orig_lines = len(orig_code.splitlines())
                                kept_lines = len(kept_frags)
                                pruned_lines = len(pruned_code.splitlines())
                                has_ellipsis = (
                                    "..." in pruned_code or "(filtered" in pruned_code
                                )
                                print(
                                    f"[DEBUG] Prune result [{i + j}]: orig_lines={orig_lines}, kept_frags={len(kept_frags)}, pruned_lines={pruned_lines}, has_ellipsis={has_ellipsis}"
                                )
                                if not has_ellipsis and kept_lines < orig_lines * 0.8:
                                    print(
                                        f"[DEBUG] WARNING: No ellipsis found but compression is significant! kept_frags sample: {kept_frags[:20]}"
                                    )
                            pruned_codes.append(pruned_code)
                            if return_raw_body:
                                raw_bodies.append(
                                    {
                                        "pruned_code": pruned_code,
                                        "kept_frags": kept_frags,
                                        "origin_code": orig_code,
                                    }
                                )

                        except Exception as e:
                            print(f"Error processing online API at index {i + j}: {e}")
                            pruned_codes.append("")
                            if return_raw_body:
                                raw_bodies.append({})

                except Exception as e:
                    print(
                        f"Error during online batch processing at batch starting index {i}: {e}"
                    )
                    pruned_codes.extend([""] * len(batch_prompts))
                    if return_raw_body:
                        raw_bodies.extend([{} for _ in range(len(batch_prompts))])

        else:
            # Offline mode: use vLLM
            if not hasattr(self, "batch_model"):
                raise ValueError("vLLM batch_model not initialized for offline mode")

            # Prepare messages for vLLM chat interface
            system_prompt = "You are a code pruning expert. Identify which lines are relevant to the given query."
            messages = []
            for prompt in prompts:
                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                messages.append(msg)

            # Run batch inference
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Process in batches
            for i in tqdm(
                range(0, len(messages), batch_size), desc="Pruning code batch (offline)"
            ):
                batch_messages = messages[i : i + batch_size]
                batch_codes = origin_codes[i : i + batch_size]
                batch_formatted = formatted_codes[i : i + batch_size]

                try:
                    batch_outputs = self.batch_model.chat(
                        messages=batch_messages,
                        sampling_params=sampling_params,
                    )

                    # Parse outputs and prune codes
                    for idx, (output, orig_code) in enumerate(
                        zip(batch_outputs, batch_codes)
                    ):
                        try:
                            output_text = output.outputs[0].text
                            kept_frags = self._fetch_kept_frags_from_output(output_text)
                            pruned_code = self._kept_frags_to_code(
                                kept_frags, orig_code
                            )
                            # Debug: Check if pruned code contains "..." and compression ratio
                            if idx < 2:  # Only log first 2 items to avoid spam
                                orig_lines = len(orig_code.splitlines())
                                kept_lines = len(kept_frags)
                                pruned_lines = len(pruned_code.splitlines())
                                has_ellipsis = (
                                    "..." in pruned_code or "(filtered" in pruned_code
                                )
                                print(
                                    f"[DEBUG] Prune result [{i + idx}]: orig_lines={orig_lines}, kept_frags={len(kept_frags)}, pruned_lines={pruned_lines}, has_ellipsis={has_ellipsis}"
                                )
                                if not has_ellipsis and kept_lines < orig_lines * 0.8:
                                    print(
                                        f"[DEBUG] WARNING: No ellipsis found but compression is significant! kept_frags sample: {kept_frags[:20]}"
                                    )
                            pruned_codes.append(pruned_code)
                            if return_raw_body:
                                raw_bodies.append(
                                    {
                                        "pruned_code": pruned_code,
                                        "kept_frags": kept_frags,
                                        "origin_code": orig_code,
                                    }
                                )
                        except Exception as e:
                            print(f"Error processing output at index {i + idx}: {e}")
                            pruned_codes.append("")
                            if return_raw_body:
                                raw_bodies.append({})
                except Exception as e:
                    print(
                        f"Error during vLLM batch inference at batch starting index {i}: {e}"
                    )
                    pruned_codes.extend([""] * len(batch_messages))
                    if return_raw_body:
                        raw_bodies.extend([{} for _ in range(len(batch_messages))])

        if return_raw_body:
            return raw_bodies
        return pruned_codes


class OnlineRerankPrunerModel(PrunerModel):
    """Pruner model that uses the online rerank API service (online_serving.py)."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        threshold: float = 0.5,
        always_keep_first_frags: bool = False,
        aggregate_method: str = "line",  # "line" or "statement"
        language: str = "python",
        batch_size: int = 32,
        timeout: float = 300.0,
        use_token_pruned_code: bool = False,
    ):
        """
        Initialize the online rerank pruner model.

        Args:
            api_base: Base URL of the online serving API (default: http://localhost:8000)
            threshold: Score threshold for pruning (default: 0.5)
            always_keep_first_frags: Whether to always keep the first fragment (default: False)
            aggregate_method: Aggregation method, "line" or "statement" (default: "line")
            language: Programming language for statement splitting (default: "python")
            batch_size: Batch size for batch processing (default: 32)
            timeout: Request timeout in seconds (default: 300.0)
        """
        self.api_base = api_base.rstrip("/")
        self.threshold = threshold
        self.always_keep_first_frags = always_keep_first_frags
        self.aggregate_method = aggregate_method
        self.language = language
        self.batch_size = batch_size
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._response_field = (
            "token_pruned_code" if use_token_pruned_code else "pruned_code"
        )

    def prune(self, query: str, origin_code: str) -> str:
        """Prune a single code snippet by calling the online API."""
        prune_url = f"{self.api_base}/prune"
        payload = {
            "query": query,
            "code": origin_code,
            "threshold": self.threshold,
            "always_keep_first_frags": self.always_keep_first_frags,
            "aggregate_method": self.aggregate_method,
            "language": self.language,
        }

        try:
            response = requests.post(prune_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            pruned = result.get(self._response_field, origin_code)
            return pruned
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling prune API: {e}")
            return origin_code

    def prune_batch(
        self,
        query: str,
        origin_codes: List[str],
        batch_size: int = -1,
        sort: bool = False,
        return_raw_body: bool = False,
    ) -> List[str] | List[dict]:
        """Prune a batch of code snippets by calling the online API."""
        if batch_size == -1:
            batch_size = self.batch_size

        pruned_codes = []
        code_scores = []
        prune_url = f"{self.api_base}/prune"
        raw_bodies = []
        for i in tqdm(
            range(0, len(origin_codes), batch_size),
            desc="Pruning code batch (online rerank)",
        ):
            batch_codes = origin_codes[i : i + batch_size]

            for origin_code in batch_codes:
                try:
                    payload = {
                        "query": query,
                        "code": origin_code,
                        "threshold": self.threshold,
                        "always_keep_first_frags": self.always_keep_first_frags,
                        "aggregate_method": self.aggregate_method,
                        "language": self.language,
                    }

                    response = requests.post(
                        prune_url, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    raw_bodies.append(result)
                    pruned = result.get(self._response_field, origin_code)
                    pruned_codes.append(pruned)
                    code_scores.append(result.get("score", 0))
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error processing online rerank API: {e}")
                    pruned_codes.append(origin_code)
                    code_scores.append(0)
        if return_raw_body:
            return raw_bodies
        if sort:
            pruned_codes = [
                code
                for _, code in sorted(
                    zip(code_scores, pruned_codes), key=lambda x: x[0], reverse=True
                )
            ]
        return pruned_codes
