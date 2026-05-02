# SPDX-License-Identifier: Apache-2.0

import re
import json
import random

from pydantic import BaseModel
import torch
import typer
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, SamplingParams

app = typer.Typer(help="Generate queries for code snippets using vLLM")
console = Console()


class Query(BaseModel):
    query: str


TASK_TYPES = [
    "code-summarize", "code-refactor", "find-relevant-part", "code-optimize",
    "code-locate", "code-explain", "code-debug", "feature-addition", "code-completion",
]
RELEVANCE_LEVELS = ["low", "medium", "high"]
QUERY_LEVELS = ["short", "medium"]


def get_prompt_template(
    task_type: str, code: str, relevance_level: str, query_level: str
) -> str:
    base_intro = (
        "Your task is to generate a realistic, developer-oriented query "
        "for a code agent, based on the provided code snippet. The query should be related to the topic: '{task_type}', "
        "and have a relevance level of '{relevance_level}' to the code."
    )
    relevance_hint = {
        "high": "Your query should cover most of the code and be tightly connected to its main logic or features.",
        "medium": "Your query should focus on a specific part or feature of the code, not the whole code.",
        "low": "Your query should only relate to a small portion or a minor aspect of the code.",
    }
    query_hint = {
        "short": "Your query should be concise and to the point. Never contains more than one idea.(<30 words)",
        "medium": "Your query can contains some context or background information. But more than one idea is not allowed.(20 ~ 50 words)",
        "long": "Your query can be more elaborate and cover multiple aspects of the code. However, it should still maintain a clear focus and not become overly verbose.(might >50 words)",
    }
    task_details = {
        "code-summarize": "Summarize the main purpose or functionality of the code, but do not explain every line. Frame your query as a developer seeking a summary for integration or review.",
        "code-refactor": "Suggest a refactoring or improvement for the code. Your query should be practical, such as asking to improve readability, modularity, or performance.",
        "find-relevant-part": "Ask to locate or identify the part of the code that implements a specific feature or logic. Your query should be about finding where something is handled in the code.",
        "code-optimize": "Request an optimization for the (core logic maybe) code, such as improving efficiency, reducing resource usage, or enhancing scalability.",
        "code-locate": "Ask to pinpoint the location of a bug, feature, or important logic within the code.",
        "code-explain": "Request an explanation for a particular logic, algorithm, or design choice in the code, but do not ask for a full code walkthrough.",
        "code-debug": "Ask for help debugging a specific issue, exception, or edge case in the code. Your query should be actionable and focused.",
        "feature-addition": "Request to add a new feature or capability to the code, specifying what should be added and how it should interact with existing logic.",
        "code-completion": "This is a special query format. In code completion, the query should be CODE instead of text, which means you should image yourself as a developer write other code snippet(query) that can used the code given for completion.The completion will be the next line for query, but you should keep it in your mind and never write the completion in query. QUERY like a PUZZLE.",
    }
    output_format = '\n\n### Output Format (JSON):\n{\n  "query": "..."\n}\n'
    template = (
        base_intro.format(task_type=task_type, relevance_level=relevance_level)
        + "\n\n## Task details:\n" + task_details.get(task_type, "No specific details for this task type.")
        + "\n\n## Relevance guidance:\n" + relevance_hint.get(relevance_level, "")
        + "\n## Query length guidance:\n" + query_hint.get(query_level, "")
        + output_format
        + "\n## Context:\n"
        + f"- Code Snippet:\n{code}\n"
        + "- Task Type: {task_type}\n"
        + f"- Relevance Level: {relevance_level}\n"
        + f"- Query Length Level: {query_level}\n"
        + "---\nNow generate the query JSON:"
    )
    return template


def extract_query(text: str):
    text = re.sub(r"```[a-zA-Z]*\s*|\s*```", "", text, flags=re.I)
    m = re.search(r'"query"\s*:\s*"(.*?)"', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = text.strip().splitlines()
    for line in reversed(lines):
        if line.strip().startswith('"query"'):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('" ,')
    return None


def load_processed_codes(output_file: str) -> set:
    processed_codes = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed_codes.add(item.get("code", ""))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return processed_codes


@app.command()
def main(
    input_file: str = typer.Option(..., "-i", "--input-file", help="Input JSONL (code items)"),
    output_file: str = typer.Option(..., "-o", "--output-file", help="Output JSONL (query, code, ...)"),
    model: str = typer.Option(..., "--model", help="vLLM model name or path"),
    tensor_parallel_size: int = typer.Option(8, "--tensor-parallel-size"),
    max_model_len: int = typer.Option(16384, "--max-model-len"),
):
    sampling_params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)
    tp = min(tensor_parallel_size, torch.cuda.device_count())
    console.print(f"Using TP={tp}")

    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        tensor_parallel_size=tp,
    )

    processed_codes = load_processed_codes(output_file)
    console.print(f"Found {len(processed_codes)} already processed in output")

    comment_code_data = []
    with open(input_file, "r") as f:
        for line in f:
            if len(line) > 12000:
                continue
            comment_code_data.append(json.loads(line))

    filtered = [d for d in comment_code_data if d["code"] not in processed_codes]
    console.print(f"Total: {len(comment_code_data)}, Remaining: {len(filtered)}")

    prompts = []
    for d in filtered:
        task_type = random.choice(TASK_TYPES)
        relevance_level = random.choice(RELEVANCE_LEVELS)
        query_level = random.choice(QUERY_LEVELS)
        prompt = get_prompt_template(
            task_type=task_type, code=d["code"],
            relevance_level=relevance_level, query_level=query_level,
        )
        prompts.append(prompt)

    system_prompt = "You are a software development expert professional in query generation task."
    messages = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]

    batch_size = 128
    with open(output_file, "a") as f:
        for i in tqdm(range(0, len(messages), batch_size), desc="Generating"):
            batch_outputs = llm.chat(
                messages=messages[i : i + batch_size],
                sampling_params=sampling_params,
            )
            queries = []
            err_cnt = 0
            for d, o in zip(filtered[i : i + batch_size], batch_outputs):
                text = o.outputs[0].text
                try:
                    query = extract_query(text)
                except Exception:
                    console.print(f"[yellow]Parse failed: {text[:200]}[/yellow]")
                    err_cnt += 1
                    continue
                if not query:
                    err_cnt += 1
                    continue
                queries.append({
                    "query": query,
                    "comment": d.get("comment"),
                    "code": d["code"],
                    "location": d.get("location"),
                })
            if len(filtered[i : i + batch_size]) > 0:
                console.print(f"Batch {i}: err rate {err_cnt / len(filtered[i : i + batch_size]):.2%}, generated {len(queries)}")
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
