import datasets
import editdistance
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from modelscope.msdatasets import MsDataset
from pathlib import Path

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n([\s\S]*?)```", re.MULTILINE)

def _clean_prediction_text(prediction: str) -> str:
    if prediction is None:
        return ""
    pred = prediction.strip()

    # 1) If fenced code blocks are included, the content of the first block is retrieved first.
    m = _FENCE_RE.search(pred)
    if m:
        pred = m.group(1)

    # 2) Filter common non-code/format lines (to avoid occupying the "first len(target_lines) lines" window).
    cleaned_lines = []
    for line in pred.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("```"):
            continue
        if s.startswith("###"):
            continue
        # More templates/explanation prefixes can be added as needed for filtering.
        if s.lower().startswith("sure") or s.lower().startswith("answer"):
            continue
        cleaned_lines.append(s)

    return "\n".join(cleaned_lines)

def compute_ES(target, prediction):
    """Compute edit similarity score"""
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = "\n".join(target_lines)

    pred_clean = _clean_prediction_text(prediction)
    prediction_lines = [
        line.strip()
        for line in pred_clean.splitlines()
        if line.strip()
    ][: len(target_lines)]
    prediction_str = "\n".join(prediction_lines)

    return (
        1
        - (
            editdistance.eval(target_str, prediction_str)
            / max(len(target_str), len(prediction_str))
        )
    ) * 100


def compute_EM(target, prediction):
    """Compute exact match score"""
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    prediction_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ][: len(target_lines)]

    if len(target_lines) != len(prediction_lines):
        return 0
    return (int(target_lines == prediction_lines)) * 100


def load_data(
    path="microsoft/LCC_python",
    split="test",
    num_examples=500,
    filter_current_lines_max=50,
    filter_background_tokens_min=5000,
):
    """
    Loads the dataset, processes it to split contexts, filters it based on context lengths,
    and returns the filtered dataset along with the tokenizer used.
    """
    print(f"Loading initial {num_examples} examples from {path} ({split} split)...")
    
    project_root = Path(__file__).parent.parent.parent.parent
    dataset_disk_path = project_root / "lcc"
    dataset = datasets.load_from_disk(dataset_disk_path)
    # keep 5 times of num_examples for testing
    dataset = dataset.select(range(num_examples * 10))
    original_size = len(dataset)  # Size before filtering

    # Initialize tokenizer here for filtering and potential later use
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    print("Tokenizer Qwen/Qwen2.5-Coder-7B-Instruct initialized.")

    # Process dataset to add split contexts first
    print("Splitting context into background and current function...")

    def add_split_context(example):
        background, current_func = split_context_ast(example["context"])
        example["background_context"] = background
        example["current_function_context"] = current_func
        return example

    processed_dataset = dataset.map(
        add_split_context, num_proc=4
    )  # Use multiple processors if available

    # --- Filter the dataset ---
    filtered_dataset_list = []
    print(
        f"Filtering dataset: Keeping examples where current func lines <= {filter_current_lines_max} and background tokens >= {filter_background_tokens_min}."
    )

    for example in tqdm(processed_dataset):
        curr_ctx = example["current_function_context"]
        bg_ctx = example["background_context"]

        curr_line_count = len(curr_ctx.splitlines())

        # Check if background context is non-empty before tokenizing
        bg_token_count = 0
        if (
            bg_ctx and bg_ctx.strip()
        ):  # Check if bg_ctx is not None and not just whitespace
            # Use truncation=True and max_length to prevent overly long sequences if needed, though for filtering, just length is fine.
            bg_token_count = len(
                tokenizer.encode(bg_ctx, add_special_tokens=False)
            )  # Usually better to exclude special tokens for length calculation

        if (
            curr_line_count <= filter_current_lines_max
            and bg_token_count >= filter_background_tokens_min
        ):
            filtered_dataset_list.append(example)

    filtered_dataset = datasets.Dataset.from_list(filtered_dataset_list)
    if num_examples > len(filtered_dataset):
        selected_dataset = filtered_dataset
    else:
        selected_dataset = filtered_dataset.select(range(num_examples))

    print(
        f"Filtering complete. Original size: {original_size}, Filtered size: {len(filtered_dataset)}. Retaining {min(num_examples, len(filtered_dataset))} examples."
    )  # Adjusted print statement

    # Return both the filtered dataset and the tokenizer
    return selected_dataset, tokenizer


def find_last_func_or_class_start(code_string):
    """
    Finds the starting line of the last top-level function or class definition
    using line-based heuristics, robust to syntax errors.
    Accounts for decorators.
    Returns the 1-based line number or None if not found.
    """
    lines = code_string.splitlines()
    if not lines:
        return None
    last_def_line_index = -1

    # Iterate backwards to find the last line starting with def/async def/class
    # We use lstrip() to handle indentation
    for i in range(len(lines) - 1, -1, -1):
        stripped_line = lines[i].lstrip()
        # Using regex for potentially more robust matching (e.g., def func():)
        # Matches lines starting with 'def', 'async def', or 'class' followed by space
        if re.match(r"^(def|async\s+def|class)\s+", stripped_line):
            last_def_line_index = i
            break

    if last_def_line_index != -1:
        # Found a potential start, now check for decorators above it
        start_line_index = last_def_line_index
        for i in range(last_def_line_index - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            if stripped_line.startswith("@"):
                start_line_index = i
            elif stripped_line == "" or stripped_line.startswith(
                "#"
            ):  # Skip blank lines and comments
                continue
            else:
                # Found a non-decorator, non-empty, non-comment line, stop searching upwards
                break
        return start_line_index + 1  # Return 1-based line number
    else:
        # Heuristic failed, maybe return the start of the last non-empty block
        # or just None if no definitions found at all
        return None  # No function or class definition found


def split_context_ast(code_string):
    """
    Splits the code context into background and current function/class context using AST.
    """
    lines = code_string.splitlines()
    split_line_1_based = find_last_func_or_class_start(code_string)

    if split_line_1_based is not None and split_line_1_based > 0:
        # split_line_1_based is the start of the function/class
        # Background is lines *before* that line
        background_lines = lines[: split_line_1_based - 1]
        current_func_lines = lines[split_line_1_based - 1 :]
        return "\n".join(background_lines), "\n".join(current_func_lines)
    else:
        # If no function/class found or parse error, treat all as current
        return "", code_string


def analyze_dataset(dataset, tokenizer):  # Added tokenizer parameter
    """Analyzes and plots context length distributions, including function counts and token ratios."""
    # --- Analysis (Optional: Recalculate stats on the filtered dataset) ---
    background_lines = []
    current_func_lines = []
    background_tokens = []
    current_func_tokens = []
    background_func_counts = []  # Added list for function counts
    bg_curr_token_ratios = []  # Added list for token ratios

    # Ensure tokenizer is available - it's passed as an argument now
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct") # Removed: Use passed tokenizer

    print(f"\nAnalyzing {len(dataset)} examples...")  # Add count here
    for example in tqdm(dataset):  # Use tqdm here for progress
        bg_ctx = example.get("background_context", "")  # Use .get for safety
        curr_ctx = example.get("current_function_context", "")

        bg_token_count = 0
        curr_token_count = 0
        func_count = 0

        # Proceed only if contexts exist
        if bg_ctx:
            bg_lines = bg_ctx.splitlines()
            bg_line_count = len(bg_lines)
            background_lines.append(bg_line_count)
            # Use truncation for safety, exclude special tokens for consistency
            bg_token_count = len(tokenizer.encode(bg_ctx, add_special_tokens=False))
            background_tokens.append(bg_token_count)
            # Count functions in background context
            for line in bg_lines:
                if re.match(
                    r"^\s*def\s+", line
                ):  # Count lines starting with 'def ' after stripping leading whitespace
                    func_count += 1
            background_func_counts.append(func_count)

        if curr_ctx:
            curr_line_count = len(curr_ctx.splitlines())
            current_func_lines.append(curr_line_count)
            curr_token_count = len(tokenizer.encode(curr_ctx, add_special_tokens=False))
            current_func_tokens.append(curr_token_count)

        # Calculate ratio, handle division by zero
        if bg_token_count > 0 and curr_token_count > 0:
            # Add a small epsilon to avoid potential issues with very small token counts if needed, though direct ratio is fine here.
            bg_curr_token_ratios.append(bg_token_count / curr_token_count)
        elif bg_token_count > 0 and curr_token_count == 0:
            bg_curr_token_ratios.append(
                np.inf
            )  # Or some large number, or skip - np.inf might break histograms, let's use a large number
            # Alternatively, filter these out or handle them specifically during plotting/stats
            pass  # Let's skip infinity for plotting simplicity
        # else: ratio is 0 or undefined, skip

    # --- Plotting ---
    # Check if *any* data exists before proceeding
    if not any(
        [
            background_lines,
            current_func_lines,
            background_tokens,
            current_func_tokens,
            background_func_counts,
            bg_curr_token_ratios,
        ]
    ):
        print(
            "No data points found for analysis after filtering. Skipping plot generation."
        )
        return  # Exit if no data to plot

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))  # Changed to 3x2 grid
    # Use tokenizer name in titles dynamically if possible, or keep generic
    tokenizer_name = (
        tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "Tokenizer"
    )
    fig.suptitle(
        f"Context Analysis (Filtered LCC Python Dataset - {len(dataset)} examples, Tokenizer: {tokenizer_name})"
    )

    # Row 1: Background
    # Background Lines
    if background_lines:
        axs[0, 0].hist(background_lines, bins=50, color="skyblue", edgecolor="black")
        print(
            f"Background Lines: Min={np.min(background_lines)}, Max={np.max(background_lines)}, Avg={np.mean(background_lines):.2f}, Median={np.median(background_lines)}"
        )
    else:
        axs[0, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, 0].transAxes,
        )
    axs[0, 0].set_title("Background Context (Lines)")
    axs[0, 0].set_ylabel("Count")

    # Background Tokens
    if background_tokens:
        axs[0, 1].hist(background_tokens, bins=50, color="skyblue", edgecolor="black")
        print(
            f"Background Tokens: Min={np.min(background_tokens)}, Max={np.max(background_tokens)}, Avg={np.mean(background_tokens):.2f}, Median={np.median(background_tokens)}"
        )
    else:
        axs[0, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, 1].transAxes,
        )
    axs[0, 1].set_title("Background Context (Tokens)")
    axs[0, 1].set_ylabel("Count")

    # Row 2: Background Func Count & Ratio
    # Background Function Count
    if background_func_counts:
        # Use more bins if the range is small, decide based on max count?
        max_funcs = np.max(background_func_counts) if background_func_counts else 0
        bins = min(
            50, max(1, max_funcs + 1)
        )  # Adjust bins based on max count, ensure at least 1 bin
        axs[1, 0].hist(
            background_func_counts, bins=bins, color="lightgreen", edgecolor="black"
        )
        print(
            f"Background Func Count: Min={np.min(background_func_counts)}, Max={max_funcs}, Avg={np.mean(background_func_counts):.2f}, Median={np.median(background_func_counts)}"
        )
    else:
        axs[1, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[1, 0].transAxes,
        )
    axs[1, 0].set_title("Background Function Count")
    axs[1, 0].set_ylabel("Count")

    # Background/Current Token Ratio
    if bg_curr_token_ratios:
        # Ratios can have a large range, consider log scale or clipping?
        # Let's cap the ratio for visualization if it gets too extreme, e.g., at 50
        # ratios_to_plot = [min(r, 50) for r in bg_curr_token_ratios] # Cap ratio at 50 for plot
        ratios_to_plot = bg_curr_token_ratios
        axs[1, 1].hist(ratios_to_plot, bins=50, color="gold", edgecolor="black")
        # Calculate stats on original ratios before clipping for plot
        print(
            f"BG/Current Token Ratio: Min={np.min(bg_curr_token_ratios):.2f}, Max={np.max(bg_curr_token_ratios):.2f}, Avg={np.mean(bg_curr_token_ratios):.2f}, Median={np.median(bg_curr_token_ratios):.2f}"
        )
        axs[1, 1].set_title("BG/Current Token Ratio")

    else:
        axs[1, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[1, 1].transAxes,
        )
    axs[1, 1].set_ylabel("Count")

    # Row 3: Current Function
    # Current Function Lines
    if current_func_lines:
        axs[2, 0].hist(
            current_func_lines, bins=50, color="lightcoral", edgecolor="black"
        )
        print(
            f"Current Func Lines: Min={np.min(current_func_lines)}, Max={np.max(current_func_lines)}, Avg={np.mean(current_func_lines):.2f}, Median={np.median(current_func_lines)}"
        )
    else:
        axs[2, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[2, 0].transAxes,
        )
    axs[2, 0].set_title("Current Function Context (Lines)")
    axs[2, 0].set_xlabel("Number of Lines")
    axs[2, 0].set_ylabel("Count")

    # Current Function Tokens
    if current_func_tokens:
        axs[2, 1].hist(
            current_func_tokens, bins=50, color="lightcoral", edgecolor="black"
        )
        print(
            f"Current Func Tokens: Min={np.min(current_func_tokens)}, Max={np.max(current_func_tokens)}, Avg={np.mean(current_func_tokens):.2f}, Median={np.median(current_func_tokens)}"
        )
    else:
        axs[2, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[2, 1].transAxes,
        )
    axs[2, 1].set_title("Current Function Context (Tokens)")
    axs[2, 1].set_xlabel("Number of Tokens")
    axs[2, 1].set_ylabel("Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.savefig(
        "context_analysis_distributions_filtered.png"
    )  # Save with a new descriptive name


if __name__ == "__main__":
    # Load data, which now includes filtering and returns the tokenizer
    filtered_dataset, tokenizer = load_data(
        num_examples=2000,
        filter_current_lines_max=50,
        filter_background_tokens_min=5000,
    )
    analyze_dataset(filtered_dataset, tokenizer)  # Pass tokenizer to analyze_dataset
