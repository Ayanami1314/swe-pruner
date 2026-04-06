import json

import typer
from rich.console import Console
from tqdm import tqdm

app = typer.Typer(help="Deduplicate JSONL by removing lines whose code appears in an eval set")
console = Console()


def build_code_set(filepath: str) -> set:
    code_set = set()
    console.print(f"Reading eval file: {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "code" in item and item["code"] is not None:
                        code_set.add(item["code"])
                except json.JSONDecodeError:
                    console.print(f"[yellow]Skipping unparseable line[/yellow]")
    except FileNotFoundError:
        console.print(f"[red]File not found: {filepath}[/red]")
        raise SystemExit(1)

    console.print(f"Eval file loaded: {len(code_set)} unique codes.")
    return code_set


def deduplicate_large_file(
    input_filepath: str, output_filepath: str, code_set_to_remove: set
) -> None:
    original_count = 0
    kept_count = 0
    removed_count = 0

    console.print(f"Processing: {input_filepath} -> {output_filepath}")

    with (
        open(input_filepath, "r", encoding="utf-8") as infile,
        open(output_filepath, "w", encoding="utf-8") as outfile,
        tqdm(desc="Dedup progress") as pbar,
    ):
        for line in infile:
            original_count += 1
            pbar.update(1)
            try:
                item = json.loads(line)
                if "code" in item and item["code"] is not None:
                    code = item["code"]
                    if code not in code_set_to_remove:
                        outfile.write(line)
                        kept_count += 1
                    else:
                        removed_count += 1
                else:
                    outfile.write(line)
                    kept_count += 1
            except json.JSONDecodeError:
                console.print(f"[yellow]Skipping unparseable line {original_count}[/yellow]")
                kept_count += 1

    console.rule("Done")
    console.print(f"Original rows: {original_count}")
    console.print(f"Eval code set size: {len(code_set_to_remove)}")
    console.print(f"Removed (duplicates): {removed_count}")
    console.print(f"Kept: {kept_count}")
    console.print(f"Output: {output_filepath}")


@app.command()
def main(
    final_dataset: str = typer.Option("final_dataset.jsonl", "--final-dataset", help="Input JSONL"),
    eval_dataset: str = typer.Option("eval_ds_enhanced.jsonl", "--eval-dataset", help="Eval JSONL (codes to remove)"),
    output: str = typer.Option("final_dataset_dedup.jsonl", "--output", help="Output JSONL"),
):
    eval_code_set = build_code_set(eval_dataset)
    deduplicate_large_file(final_dataset, output, eval_code_set)


if __name__ == "__main__":
    app()
