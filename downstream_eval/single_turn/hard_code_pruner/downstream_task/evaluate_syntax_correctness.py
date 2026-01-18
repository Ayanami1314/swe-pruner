"""
Evaluate syntax correctness of different code pruning methods using tree-sitter.

This script compares the syntax correctness rate of different pruning methods:
- Origin (original code without pruning, as baseline)
- Our method (using kept_frags from dataset)
- LLMLingua
- Random pruning (token-level)
- Line-level random pruning

The script reads a jsonl dataset, reconstructs pruned code from kept_frags,
and checks syntax correctness using tree-sitter parsers.

Usage:
    python evaluate_syntax_correctness.py \\
        --dataset /path/to/dataset.jsonl \\
        --language python \\
        --methods ours random line_level_random llmlingua \\
        --output results.json

Dataset format (jsonl):
    Each line should be a JSON object with:
    - "code": str - Original code string
    - "kept_frags": List[int] - List of 1-based line numbers to keep
    - "query": str (optional) - Query string for context

Example:
    {"code": "def foo():\\n    return 1\\n\\ndef bar():\\n    return 2", "kept_frags": [1, 2], "query": "..."}
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import sys

from tqdm import tqdm

# Try to import rich for better output formatting
try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich import print as rprint

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print(
        "Warning: rich not available. Install with: pip install rich for better output formatting."
    )

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import tokenizer for token-level operations
try:
    from transformers import AutoTokenizer

    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print(
        "Warning: transformers not available. Token-level random pruning will use simple tokenization."
    )

from utils.code_stmt_splitter import LanguageEnum, LANGUAGE_PARSERS
from tree_sitter import Parser, Language
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo

# Try to import LLMLingua
try:
    from llmlingua import PromptCompressor

    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    print("Warning: LLMLingua not available. Install with: pip install llmlingua")


def reconstruct_code_from_kept_frags(code: str, kept_frags: List[int]) -> str:
    """
    Reconstruct pruned code from kept_frags (1-based line numbers).
    Simply concatenate the kept lines without placeholders.

    Args:
        code: Original code string
        kept_frags: List of 1-based line numbers to keep

    Returns:
        Pruned code string (only kept lines, no placeholders)
    """
    lines = code.splitlines()
    kept_code_lines = []

    for line in range(1, len(lines) + 1):
        if line in kept_frags:
            kept_code_lines.append(lines[line - 1])

    return "\n".join(kept_code_lines)


def check_syntax_correctness(code: str, language: str = "python") -> bool:
    """
    Check if code is syntactically correct using tree-sitter.

    Args:
        code: Code string to check
        language: Programming language (python, java, golang)

    Returns:
        True if syntax is correct, False otherwise
    """
    # If code is empty, consider it as syntax error
    if not code or not code.strip():
        return False

    # Map language string to LanguageEnum
    lang_map = {
        "python": LanguageEnum.py,
        "java": LanguageEnum.java,
        "golang": LanguageEnum.golang,
        "go": LanguageEnum.golang,
    }

    lang_enum = lang_map.get(language.lower(), LanguageEnum.py)

    if lang_enum not in LANGUAGE_PARSERS:
        # For unsupported languages, assume syntax is correct
        # (we can't verify without a parser)
        return True

    try:
        parser = Parser(LANGUAGE_PARSERS[lang_enum])
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        # Check if there are any syntax errors
        # tree-sitter marks errors with ERROR nodes
        def has_error(node):
            if node.type == "ERROR":
                return True
            for child in node.children:
                if has_error(child):
                    return True
            return False

        return not has_error(root_node)
    except Exception as e:
        # If parsing fails, assume syntax error
        return False


# Initialize random generator with fixed seed for unconstrained pruning
_rng = random.Random(42)
_tokenizer_cache = None


def _get_tokenizer():
    global _tokenizer_cache
    if _tokenizer_cache is None and TOKENIZER_AVAILABLE:
        try:
            _tokenizer_cache = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
        except Exception:
            pass
    return _tokenizer_cache


def tokenize_with_offsets(text: str) -> List[Tuple[int, int]]:
    """Tokenize text and return list of (start_pos, end_pos) tuples."""
    tokenizer = _get_tokenizer()
    if tokenizer:
        try:
            encoding = tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            offsets = encoding["offset_mapping"]
            return [
                (start, end)
                for start, end in offsets
                if start is not None and end is not None
            ]
        except Exception:
            pass

    # Fallback: simple whitespace-based tokenization
    import re

    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.start(), match.end()))
    return tokens


def prune_random(code: str, kept_frags: List[int]) -> str:
    """
    Random pruning: randomly select tokens at token-level.
    Keeps the same number of tokens as the kept_frags method.

    Args:
        code: Original code string
        kept_frags: List of kept line numbers (used to determine how many tokens to keep)

    Returns:
        Randomly pruned code at token level
    """
    if not code:
        return code

    # Calculate target token count from kept_frags
    kept_code = reconstruct_code_from_kept_frags(code, kept_frags)

    # Tokenize both original and kept code
    original_token_offsets = tokenize_with_offsets(code)
    kept_token_offsets = tokenize_with_offsets(kept_code)

    target_token_count = len(kept_token_offsets)

    if len(original_token_offsets) == 0:
        return code

    # Randomly select tokens
    target_token_count = min(target_token_count, len(original_token_offsets))
    selected_indices = set(
        random.sample(range(len(original_token_offsets)), target_token_count)
    )

    # Reconstruct code from selected tokens (preserve order)
    selected_offsets = [
        (start, end)
        for i, (start, end) in enumerate(original_token_offsets)
        if i in selected_indices
    ]
    selected_offsets.sort(key=lambda x: x[0])  # Sort by start position

    # Reconstruct code
    if not selected_offsets:
        return ""

    result_parts = []
    last_end = 0

    for start, end in selected_offsets:
        # Add whitespace gap before this token if needed
        if start > last_end:
            gap = code[last_end:start]
            # Preserve whitespace gaps
            if gap.strip() == "":
                result_parts.append(gap)

        result_parts.append(code[start:end])
        last_end = end

    return "".join(result_parts)


def prune_line_level_random(code: str, kept_frags: List[int]) -> str:
    """
    Line-level random pruning: randomly select the same number of lines as kept_frags.

    Args:
        code: Original code string
        kept_frags: List of kept line numbers (used to determine how many lines to keep)

    Returns:
        Line-level randomly pruned code
    """
    lines = code.splitlines()
    num_lines = len(lines)
    num_to_keep = len(kept_frags)

    if num_lines == 0:
        return code

    # Randomly select lines to keep
    # Ensure we don't try to sample more than available
    num_to_keep = min(num_to_keep, num_lines)
    random_kept_frags = sorted(random.sample(range(1, num_lines + 1), num_to_keep))

    return reconstruct_code_from_kept_frags(code, random_kept_frags)


def prune_random_unconstrained(code: str) -> str:
    """
    Random pruning: randomly select tokens at token-level with random count.
    Uses a fixed random seed (42) for reproducibility.

    Args:
        code: Original code string

    Returns:
        Randomly pruned code at token level
    """
    if not code:
        return code

    original_token_offsets = tokenize_with_offsets(code)
    if not original_token_offsets:
        return code

    # Randomly determine number of tokens to keep (1 to total)
    total_tokens = len(original_token_offsets)
    target_token_count = _rng.randint(1, total_tokens)

    selected_indices = set(_rng.sample(range(total_tokens), target_token_count))

    # Reconstruct code from selected tokens (preserve order)
    selected_offsets = [
        (start, end)
        for i, (start, end) in enumerate(original_token_offsets)
        if i in selected_indices
    ]
    selected_offsets.sort(key=lambda x: x[0])

    if not selected_offsets:
        return ""

    result_parts = []
    last_end = 0

    for start, end in selected_offsets:
        if start > last_end:
            gap = code[last_end:start]
            if gap.strip() == "":
                result_parts.append(gap)

        result_parts.append(code[start:end])
        last_end = end

    return "".join(result_parts)


def prune_line_level_random_unconstrained(code: str) -> str:
    """
    Line-level random pruning: randomly select a random number of lines.
    Uses a fixed random seed (42) for reproducibility.

    Args:
        code: Original code string

    Returns:
        Line-level randomly pruned code
    """
    lines = code.splitlines()
    num_lines = len(lines)

    if num_lines == 0:
        return code

    # Randomly determine how many lines to keep
    num_to_keep = _rng.randint(1, num_lines)

    random_kept_frags = sorted(_rng.sample(range(1, num_lines + 1), num_to_keep))

    return reconstruct_code_from_kept_frags(code, random_kept_frags)


def prune_llmlingua(
    code: str,
    query: str,
    compressor: Optional[PromptCompressor] = None,
) -> str:
    """
    Prune code using LLMLingua. Directly compress the code without using kept_frags.

    Args:
        code: Original code string
        query: Query string (for context)
        compressor: LLMLingua PromptCompressor instance

    Returns:
        LLMLingua pruned code
    """
    if not LLMLINGUA_AVAILABLE or compressor is None:
        raise NotImplementedError("LLMLingua compression not available")

    try:
        # Use LLMLingua to compress directly
        compressed_result = compressor.compress_prompt(
            code,
            question=query if query else "",
        )

        # Handle different return types
        if isinstance(compressed_result, str):
            return compressed_result
        elif isinstance(compressed_result, list) and len(compressed_result) > 0:
            if isinstance(compressed_result[0], str):
                return compressed_result[0]
            else:
                return compressed_result[0].get("compressed_prompt", code)
        elif isinstance(compressed_result, dict):
            return compressed_result.get("compressed_prompt", code)
        else:
            return code
    except Exception as e:
        print(f"Warning: LLMLingua compression failed: {e}. Using original code.")
        return code


def evaluate_dataset(
    dataset_path: str,
    language: str = "python",
    methods: List[str] = None,
    use_llmlingua: bool = True,
    seed: int = 42,
    num_examples: int = 0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    """
    Evaluate syntax correctness for different pruning methods on a dataset.

    Args:
        dataset_path: Path to jsonl dataset file
        language: Programming language of the code
        methods: List of methods to evaluate (default: all)
        use_llmlingua: Whether to use LLMLingua (requires installation)
        seed: Random seed for reproducibility
        num_examples: Number of examples to collect where line_level_random failed but ours passed

    Returns:
        Tuple of (results dictionary, examples list)
    """
    random.seed(seed)

    # Initialize LLMLingua compressor if needed
    llmlingua_compressor = None
    if use_llmlingua and LLMLINGUA_AVAILABLE:
        try:
            llmlingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
            print("LLMLingua compressor initialized.")
        except Exception as e:
            print(f"Warning: Failed to initialize LLMLingua: {e}")
            llmlingua_compressor = None

    # Default methods to evaluate
    if methods is None:
        methods = ["origin", "ours", "random", "line_level_random"]
        if llmlingua_compressor is not None:
            methods.append("llmlingua")

    # Statistics
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # Collect examples where random failed but ours passed
    examples = []

    # Read dataset
    print(f"Reading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            try:
                item = json.loads(line.strip())

                # Extract required fields
                code = item.get("code", "")
                kept_frags = item.get("kept_frags", [])
                query = item.get("query", "")

                if not code:
                    continue

                # Check which methods need kept_frags
                methods_need_frags = {"ours", "random", "line_level_random"}
                methods_to_eval = set(methods)

                # Skip if kept_frags is missing and any method requires it
                if not kept_frags and methods_to_eval & methods_need_frags:
                    continue

                # Store results for each method
                method_results = {}

                # Evaluate each method
                for method in methods:
                    if method == "origin":
                        # Use original code without any pruning
                        pruned_code = code
                    elif method == "ours":
                        if not kept_frags:
                            continue
                        pruned_code = reconstruct_code_from_kept_frags(code, kept_frags)
                    elif method == "random":
                        if not kept_frags:
                            continue
                        pruned_code = prune_random(code, kept_frags)
                    elif method == "line_level_random":
                        if not kept_frags:
                            continue
                        pruned_code = prune_line_level_random(code, kept_frags)
                    elif method == "llmlingua":
                        pruned_code = prune_llmlingua(code, query, llmlingua_compressor)
                    else:
                        continue

                    # Check syntax correctness
                    is_correct = check_syntax_correctness(pruned_code, language)

                    method_results[method] = {
                        "code": pruned_code,
                        "correct": is_correct,
                    }

                    stats[method]["total"] += 1
                    if is_correct:
                        stats[method]["correct"] += 1

                # Collect examples where line_level_random failed but ours passed
                if num_examples > 0 and len(examples) < num_examples:
                    if (
                        "line_level_random" in method_results
                        and "ours" in method_results
                    ):
                        line_level_random_correct = method_results["line_level_random"][
                            "correct"
                        ]
                        ours_correct = method_results["ours"]["correct"]

                        if not line_level_random_correct and ours_correct:
                            examples.append(
                                {
                                    "original_code": code,
                                    "ours_code": method_results["ours"]["code"],
                                    "line_level_random_code": method_results[
                                        "line_level_random"
                                    ]["code"],
                                    "kept_frags": kept_frags,
                                    "query": query,
                                }
                            )

                if line_num % 100 == 0:
                    print(f"Processed {line_num} examples...")

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Calculate accuracy rates
    results = {}
    for method, stat in stats.items():
        if stat["total"] > 0:
            accuracy = stat["correct"] / stat["total"]
            results[method] = {
                "accuracy": accuracy,
                "correct": stat["correct"],
                "total": stat["total"],
            }
        else:
            results[method] = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
            }

    return results, examples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate syntax correctness of different code pruning methods"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to jsonl dataset file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "java", "golang"],
        help="Programming language of the code",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to evaluate (ours, random, line_level_random, llmlingua)",
    )
    parser.add_argument(
        "--no-llmlingua",
        action="store_true",
        help="Disable LLMLingua evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="Number of examples to show where line_level_random failed but ours passed",
    )

    args = parser.parse_args()

    # Evaluate dataset
    print("=" * 60)
    print("Evaluating Syntax Correctness")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Language: {args.language}")
    print(f"Methods: {args.methods if args.methods else 'all'}")
    print("=" * 60)

    results, examples = evaluate_dataset(
        dataset_path=args.dataset,
        language=args.language,
        methods=args.methods,
        use_llmlingua=not args.no_llmlingua,
        seed=args.seed,
        num_examples=args.num_examples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for method, result in results.items():
        print(
            f"{method:20s}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})"
        )
    print("=" * 60)

    # Print examples if requested
    if examples:
        if RICH_AVAILABLE:
            rprint("\n" + "=" * 80)
            rprint(
                f"[bold cyan]Examples where line_level_random failed but ours passed ({len(examples)}):[/bold cyan]"
            )
            rprint("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(
                f"Examples where line_level_random failed but ours passed ({len(examples)}):"
            )
            print("=" * 80)

        # Map language to rich syntax language name
        syntax_lang_map = {
            "python": "python",
            "java": "java",
            "golang": "go",
            "go": "go",
        }
        syntax_lang = syntax_lang_map.get(args.language.lower(), "python")

        for i, example in enumerate(examples, 1):
            query = example["query"] if example["query"] else "N/A"

            if RICH_AVAILABLE:
                # Use rich for formatted output
                rprint(f"\n[bold yellow]Example {i}:[/bold yellow]")
                if query != "N/A":
                    rprint(
                        Panel(
                            f"[italic]{query}[/italic]",
                            title="Query",
                            border_style="blue",
                        )
                    )

                # Original code
                original_syntax = Syntax(
                    example["original_code"],
                    syntax_lang,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
                rprint(
                    Panel(
                        original_syntax,
                        title="[bold green]Original Code[/bold green]",
                        border_style="green",
                    )
                )

                # Ours code
                ours_syntax = Syntax(
                    example["ours_code"],
                    syntax_lang,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
                rprint(
                    Panel(
                        ours_syntax,
                        title="[bold cyan]Ours Code (✓ Syntax Correct)[/bold cyan]",
                        border_style="cyan",
                    )
                )

                # Line-level random code
                random_syntax = Syntax(
                    example["line_level_random_code"],
                    syntax_lang,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
                rprint(
                    Panel(
                        random_syntax,
                        title="[bold red]Line-level Random Code (✗ Syntax Error)[/bold red]",
                        border_style="red",
                    )
                )

                rprint("\n" + "-" * 80 + "\n")
            else:
                # Fallback to simple output
                print(f"\nExample {i}:")
                print(f"Query: {query}")
                print(f"\nOriginal code:\n{example['original_code']}")
                print(f"\nOurs code:\n{example['ours_code']}")
                print(f"\nLine-level random code:\n{example['line_level_random_code']}")
                print("-" * 80)

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {"results": results, "examples": examples if examples else []}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
