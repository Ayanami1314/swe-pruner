"""
Line-level labeling only (LLM-style training). Produces labeled JSONL with kept_frags.
No binary .pt feature output; BERT-style feature build/load has been removed.
"""
import re
import json
from typing import List
from pathlib import Path

import typer
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, SamplingParams

from train.core.structure import CodeGroupItem
from train.core.prompts.llm_label import (
    llm_label_prompt_template_for_line,
    fetch_llm_label_from_output,
)
from train.utils.line_chunker import split_code_into_lines

app = typer.Typer(help="Build line-level labels (kept_frags) using vLLM; output is JSONL only")
console = Console()


def code_formatter(code_groups: List[CodeGroupItem]) -> str:
    """Format code with numbered lines for LLM labeling."""
    return "\n".join(f"{idx}> {group.text}" for idx, group in enumerate(code_groups, start=1))


def load_processed_codes(output_jsonl: Path) -> set:
    """Load already processed code snippets from output JSONL file."""
    processed_codes = set()
    if not output_jsonl.exists():
        return processed_codes
    try:
        with open(output_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed_codes.add(item.get("code", ""))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load processed codes: {e}[/yellow]")
    return processed_codes


@app.command()
def main(
    input_file: Path = typer.Option(
        "item_with_score.jsonl", help="Input JSONL file with query, code, score"
    ),
    output_jsonl: Path = typer.Option(
        "labeled_data.jsonl", help="Output JSONL with query, code, score, kept_frags"
    ),
    model_name: str = typer.Option(
        ..., help="vLLM model path (e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct)"
    ),
    tensor_parallel_size: int = typer.Option(8, help="Tensor parallel size for vLLM"),
    max_model_len: int = typer.Option(16384, help="Max model length for vLLM"),
    temperature: float = typer.Option(0.3, help="Sampling temperature"),
    max_tokens: int = typer.Option(1024, help="Max tokens for LLM output"),
    max_code_length: int = typer.Option(
        12000, help="Skip code longer than this (character count)"
    ),
):
    # 1. Load input data
    console.print(f"Loading data from {input_file}...")
    input_data = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception as e:
                console.print(f"[yellow]Skipping invalid line: {e}[/yellow]")
                continue
            if (
                isinstance(item, dict)
                and "code" in item
                and isinstance(item["code"], str)
            ):
                if len(item["code"]) > max_code_length:
                    continue
            if "query" in item and "code" in item and "score" in item:
                input_data.append(item)

    if not input_data:
        console.print("[red]No valid input data found![/red]")
        raise SystemExit(1)

    console.print(f"Loaded {len(input_data)} items")

    processed_codes = load_processed_codes(output_jsonl)
    console.print(f"Found {len(processed_codes)} already processed items in output file")

    filtered_input_data = [d for d in input_data if d["code"] not in processed_codes]
    console.print(f"Total items: {len(input_data)}, Remaining: {len(filtered_input_data)}")

    # 2. Prepare code groups (line-level only)
    items_with_groups = []
    for item in filtered_input_data:
        code = item["code"]
        chunks = split_code_into_lines(code)
        code_groups = [
            CodeGroupItem(start_byte=c.start_byte, end_byte=c.end_byte, text=c.text)
            for c in chunks
        ]
        items_with_groups.append({
            "query": item["query"],
            "code": code,
            "comment": item.get("comment", ""),
            "score": item["score"],
            "code_groups": code_groups,
        })

    err_cnt = 0
    empty_label_cnt = 0
    processed_count = 0

    # 3. Build prompts for vLLM
    console.print("Building prompts...")
    prompts = []
    for item in items_with_groups:
        formatted_code = code_formatter(item["code_groups"])
        prompt = llm_label_prompt_template_for_line.format(
            query=item["query"], code=formatted_code
        )
        prompts.append(prompt)

    # 4. Initialize vLLM
    console.print(f"Initializing vLLM with model {model_name}...")
    sampling_params = SamplingParams(
        max_tokens=max(16384, max_tokens),
        temperature=temperature,
    )
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
    )

    # 5. Run batch inference with streaming writes
    console.print(f"Running batch inference on {len(prompts)} prompts...")
    system_prompt = "You are a software development expert professional in code QA task."
    messages = []
    for p in prompts:
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ])
    batch_size = 128

    jsonl_file = open(output_jsonl, "a")
    try:
        for i in tqdm(range(0, len(messages), batch_size), desc="Labeling"):
            batch_outputs = llm.chat(
                messages=messages[i : i + batch_size],
                sampling_params=sampling_params,
            )
            batch_items = items_with_groups[i : i + batch_size]

            for idx, (item, output) in enumerate(zip(batch_items, batch_outputs)):
                text = output.outputs[0].text
                global_idx = i + idx
                try:
                    kept_indices = fetch_llm_label_from_output(text)
                    if not kept_indices:
                        empty_label_cnt += 1
                        kept_frags = []
                    else:
                        kept_frags = kept_indices

                    labeled_result = {
                        "query": item["query"],
                        "code": item["code"],
                        "score": item["score"],
                        "kept_frags": kept_frags,
                    }
                    jsonl_file.write(
                        json.dumps(labeled_result, ensure_ascii=False) + "\n"
                    )
                    processed_count += 1
                except Exception as e:
                    console.print(f"[red]Error processing item {global_idx}: {e}[/red]")
                    console.print(f"LLM output: {text[:500]}")
                    err_cnt += 1
    finally:
        jsonl_file.close()

    # 6. Summary
    console.print()
    console.rule("Processing complete")
    console.print(f"Total items: {len(input_data)}")
    console.print(f"Remaining items: {len(filtered_input_data)}")
    console.print(f"Successfully processed: {processed_count}")
    console.print(f"Errors: {err_cnt}")
    console.print(f"Empty labels: {empty_label_cnt}")
    if len(filtered_input_data) > 0:
        console.print(f"Error rate: {err_cnt / len(filtered_input_data):.2%}")
        console.print(f"Empty label rate: {empty_label_cnt / len(filtered_input_data):.2%}")
    console.print(f"Output: [bold]{output_jsonl}[/bold]")


if __name__ == "__main__":
    app()
