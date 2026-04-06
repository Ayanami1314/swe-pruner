# SPDX-License-Identifier: Apache-2.0

import json
import math

import torch
import typer
from rich.console import Console
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

app = typer.Typer(help="Score (query, code) pairs with a reranker model")
console = Console()


def format_instruction(instruction: str, query: str, doc: str) -> list:
    return [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
    ]


def process_inputs(pairs, instruction, max_length, suffix_tokens, tokenizer):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages


def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        true_logit = final_logits[true_token].logprob if true_token in final_logits else -10
        false_logit = final_logits[false_token].logprob if false_token in final_logits else -10
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        scores.append(true_score / (true_score + false_score))
    return scores


@app.command()
def main(
    input_file: str = typer.Option(..., "-i", "--input-file", help="Input JSONL (query, code)"),
    output_file: str = typer.Option(..., "-o", "--output-file", help="Output JSONL with score"),
    model: str = typer.Option(..., "--model", help="Reranker model name or path (used for tokenizer and LLM)"),
    max_model_len: int = typer.Option(10000, "--max-model-len"),
    gpu_memory_utilization: float = typer.Option(0.8, "--gpu-memory-utilization"),
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    number_of_gpu = torch.cuda.device_count()
    llm = LLM(
        model=model,
        tensor_parallel_size=number_of_gpu,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length = 8192
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    instruction = "Given a query, judge if the document(code) is related to query."
    queries = []
    documents = []
    items = []

    with open(input_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                queries.append(data["query"])
                documents.append(data["code"])
                items.append(data)
            except Exception as e:
                console.print(f"[yellow]Skip line: {e}[/yellow]")
                continue

    pairs = list(zip(queries, documents))
    inputs = process_inputs(
        pairs, instruction, max_length - len(suffix_tokens), suffix_tokens, tokenizer
    )
    scores = compute_logits(llm, inputs, sampling_params, true_token, false_token)

    with open(output_file, "w") as f:
        for item, s in zip(items, scores):
            f.write(json.dumps({**item, "score": s}) + "\n")

    destroy_model_parallel()
    console.print(f"Wrote {output_file}")


if __name__ == "__main__":
    app()
