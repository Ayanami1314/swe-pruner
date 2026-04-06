#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream-download GitHub code from ModelScope, filter by language/size/lines,
and save to JSONL.
"""

import json
import unicodedata
from typing import Dict, Any, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

LANGUAGE_FILE_MAP = {
    "python": [".py"],
    "java": [".java"],
    "javascript": [".js", ".jsx", ".ts", ".tsx"],
    "c_cpp": [".c", ".cpp", ".h", ".hpp"],
    "markdown": [".md"],
    "go": [".go"],
    "yaml": [".yaml", ".yml"],
}

app = typer.Typer(help="Stream-download and filter GitHub code to JSONL")
console = Console()


def simple_chunker(content: str, max_lines: int, min_lines: int) -> List[str]:
    lines = [l for l in content.split("\n") if l.strip()]
    chunks = []
    while len(lines) > max_lines:
        chunks.append("\n".join(lines[:max_lines]))
        lines = lines[max_lines:]
    if len(lines) > min_lines:
        chunks.append("\n".join(lines))
    return chunks


def is_lang_file(path: str, lang: str) -> bool:
    assert lang in LANGUAGE_FILE_MAP, f"Unsupported lang: {lang}"
    for ext in LANGUAGE_FILE_MAP[lang]:
        if path.lower().endswith(ext):
            return True
    return False


def has_non_english_comment(content: str) -> bool:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") and not stripped.startswith("//"):
            continue
        text = stripped.lstrip("#").strip().lstrip("//").strip()
        if not text:
            continue
        for ch in text:
            if ord(ch) <= 127:
                continue
            cats = unicodedata.name(ch, "").split()
            if not cats:
                return True
            if cats[0] in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL",
                           "CYRILLIC", "ARABIC", "HEBREW", "THAI"):
                return True
    return False


def transform_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "code": row["content"],
        "repo": f"{row['repo_id']}/{row['file_path']}"
    }


@app.command()
def main(
    output_prefix: str = typer.Option("ghcode-filtered", "--output-prefix", help="Output file prefix (e.g. ghcode-filtered -> ghcode-filtered_python.jsonl)"),
    want_rows: int = typer.Option(200_000, "--want-rows", help="Target number of output rows"),
    max_size: int = typer.Option(32768, "--max-size", help="Max file size in bytes"),
    max_lines: int = typer.Option(150, "--max-lines", help="Max lines per chunk"),
    min_lines: int = typer.Option(20, "--min-lines", help="Min lines to keep a chunk"),
    lang: str = typer.Option("python", "--lang", help="Language filter"),
):
    from modelscope.msdatasets import MsDataset

    ds = MsDataset.load("nick007x/github-code-2025", subset_name="default", split="train", use_streaming=True)
    cnt_total = 0
    cnt_out = 0
    out_file = f"{output_prefix}_{lang}.jsonl"

    with open(out_file, "w", encoding="utf-8") as f_out:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)
            for sample in ds:
                cnt_total += 1
                if not is_lang_file(sample["file_path"], lang=lang):
                    continue
                if sample["size"] > max_size:
                    continue
                if has_non_english_comment(sample["content"]):
                    continue
                chunks = simple_chunker(sample["content"], max_lines, min_lines)
                for c in chunks:
                    f_out.write(json.dumps(transform_row({**sample, "content": c}), ensure_ascii=False) + "\n")
                cnt_out += len(chunks)
                if cnt_out % 100 == 0:
                    progress.update(task, description=f"Written {cnt_out:,} rows")
                if cnt_out >= want_rows:
                    break

    console.print(f"Done. Scanned {cnt_total:,}, wrote {cnt_out:,} -> {out_file}")


if __name__ == "__main__":
    app()
