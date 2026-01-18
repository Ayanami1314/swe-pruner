import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
import time
from typing import List, Tuple, Dict, Optional
from contextlib import nullcontext
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from fastapi import FastAPI
import logging
from hf import (
    SwePrunerForCodeCompression,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Pruning Service")

# Global model and tokenizer
model: Optional[SwePrunerForCodeCompression] = None
tokenizer: Optional[AutoTokenizer] = None
device: Optional[torch.device] = None


def compute_ttft(model, input_length, tokenizer, device, num_runs=10):
    """Compute Time To First Token for a given input length, averaged over multiple runs"""
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, input_length))
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Warm up runs
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(5):
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if device == "cuda":
        torch.cuda.synchronize()

    # Measure TTFT multiple times
    ttft_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            with autocast_ctx:
                outputs = model(input_ids, attention_mask=attention_mask)
                # Generate one token
        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        ttft_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        torch.cuda.empty_cache()

    # Calculate statistics
    avg_ttft = sum(ttft_times) / len(ttft_times)
    min_ttft = min(ttft_times)
    max_ttft = max(ttft_times)

    return avg_ttft, min_ttft, max_ttft


def load_model(
    model_path: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code: bool = True,
):
    """Load Provence model from a HuggingFace-style directory or repository."""

    global model, tokenizer, device

    device = torch.device(device_str)

    logger.info(f"Loading Provence model from {model_path}")
    model = SwePrunerForCodeCompression.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    scorer = model.model if hasattr(model, "model") else model
    padding_side = "left" if getattr(scorer, "is_llm", True) else "right"
    tokenizer.padding_side = padding_side

    logger.info(
        "Model loaded successfully with padding_side=%s on device=%s",
        tokenizer.padding_side,
        device,
    )


if __name__ == "__main__":
    # Test different input lengths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ttft_runs", type=int, default=10, help="Number of TTFT runs for averaging"
    )
    parser.add_argument(
        "--model-path", type=str, default="./model", help="Dir path of the model"
    )
    args = parser.parse_args()
    load_model(args.model_path)
    input_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = defaultdict(list)
    ttft_results = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")

    # Compute TTFT for each input length
    print(
        f"\nComputing TTFT for different input lengths (averaging over {args.ttft_runs} runs)..."
    )
    for length in tqdm(input_lengths):
        print(f"  Computing TTFT for input_length={length}...")
        avg_ttft, min_ttft, max_ttft = compute_ttft(
            model, length, tokenizer, device, args.ttft_runs
        )
        ttft_results.append((avg_ttft, min_ttft, max_ttft))
        print(f"    Average TTFT: {avg_ttft:.2f} ms")
        print(f"    Min TTFT: {min_ttft:.2f} ms")
        print(f"    Max TTFT: {max_ttft:.2f} ms")
