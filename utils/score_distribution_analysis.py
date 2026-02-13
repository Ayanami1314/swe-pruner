"""
Score distribution analysis for SWE-Pruner.

Addresses the question of whether the CRF layer produces bimodal (polarized)
line-level scores, which would justify a fixed threshold τ=0.5.

This script:
1. Reads a validation JSONL dataset (query, code, score, kept_frags).
2. Calls the running inference service to obtain token_scores for each sample.
3. Computes line-level scores and plots a histogram to visualise the distribution.
4. Runs a threshold ablation at τ ∈ {0.3, 0.5, 0.7} (or custom values),
   reporting F1, IoU, precision, recall, and compression rate.
5. Optionally saves an enriched JSONL with token_scores for reuse by other
   utils (e.g. threshold_optimizer, token_vis).

Prerequisites:
    pip install requests numpy plotly
    # The inference service must already be running (see online_serving.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Re-use helpers from threshold_optimizer (same directory)
# ---------------------------------------------------------------------------
from threshold_optimizer import (
    POOLING_MEAN,
    POOLING_MAX,
    compute_line_scores,
    predict_kept_lines,
    evaluate_f1,
    evaluate_iou,
    evaluate_precision,
    evaluate_recall,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inference client
# ---------------------------------------------------------------------------


def call_prune_api(
    query: str,
    code: str,
    threshold: float = 0.5,
    base_url: str = "http://localhost:8000",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call the /prune endpoint and return the JSON response."""
    url = f"{base_url}/prune"
    payload = {
        "query": query,
        "code": code,
        "threshold": threshold,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def enrich_dataset(
    dataset: List[Dict[str, Any]],
    base_url: str = "http://localhost:8000",
    threshold: float = 0.5,
    timeout: int = 120,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Call inference API for each sample and attach token_scores + predicted score.

    Skips samples that already have a non-empty ``token_scores`` field.
    """
    total = len(dataset)
    enriched = []
    t0 = time.time()

    for idx, sample in enumerate(dataset):
        # Skip if already enriched
        if sample.get("token_scores"):
            enriched.append(sample)
            if verbose and (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{total}] skipped (already has token_scores)")
            continue

        query = sample.get("query", "")
        code = sample.get("code", "")
        if not query or not code:
            enriched.append(sample)
            continue

        try:
            resp = call_prune_api(
                query, code, threshold=threshold, base_url=base_url, timeout=timeout
            )
            sample_copy = dict(sample)
            sample_copy["token_scores"] = resp.get("token_scores", [])
            sample_copy["predicted_score"] = resp.get("score", None)
            sample_copy["origin_token_cnt"] = resp.get("origin_token_cnt", None)
            sample_copy["left_token_cnt"] = resp.get("left_token_cnt", None)
            sample_copy["model_input_token_cnt"] = resp.get(
                "model_input_token_cnt", None
            )
            enriched.append(sample_copy)
        except Exception as e:
            print(f"  [{idx + 1}/{total}] ERROR: {e}")
            enriched.append(sample)

        if verbose and (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1}/{total}] {elapsed:.1f}s elapsed, "
                f"{rate:.1f} samples/s, ETA {eta:.0f}s"
            )

    return enriched


# ---------------------------------------------------------------------------
# Histogram of line-level scores
# ---------------------------------------------------------------------------


def collect_line_scores(
    dataset: List[Dict[str, Any]],
    pooling: str = POOLING_MEAN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all line-level scores across the dataset.

    Returns:
        all_scores: 1-D array of every line score.
        gt_labels: 1-D array, 1 if line is in ground-truth kept_frags, 0 otherwise.
    """
    all_scores: List[float] = []
    gt_labels: List[int] = []

    for sample in dataset:
        code = sample.get("code", "")
        token_scores = sample.get("token_scores", [])
        kept_frags = set(sample.get("kept_frags", []))
        if not token_scores:
            continue

        line_scores = compute_line_scores(code, token_scores, pooling=pooling)
        for line_num, score in line_scores.items():
            all_scores.append(score)
            gt_labels.append(1 if line_num in kept_frags else 0)

    return np.array(all_scores), np.array(gt_labels)


def plot_score_histogram(
    all_scores: np.ndarray,
    gt_labels: np.ndarray,
    output_path: str = "score_histogram.html",
    pooling: str = POOLING_MEAN,
    num_bins: int = 50,
) -> None:
    """Plot histogram of line-level scores, split by ground-truth label."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")

    keep_scores = all_scores[gt_labels == 1]
    prune_scores = all_scores[gt_labels == 0]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Line-Level Score Distribution (All Lines)",
            "Score Distribution Split by Ground-Truth Label",
        ),
        vertical_spacing=0.12,
    )

    # --- Row 1: overall histogram ---
    fig.add_trace(
        go.Histogram(
            x=all_scores,
            nbinsx=num_bins,
            marker_color="steelblue",
            opacity=0.8,
            name="All lines",
        ),
        row=1,
        col=1,
    )

    # Add vertical lines at common thresholds
    for tau, color in [(0.3, "orange"), (0.5, "red"), (0.7, "purple")]:
        fig.add_vline(
            x=tau,
            line_dash="dash",
            line_color=color,
            annotation_text=f"τ={tau}",
            annotation_position="top",
            row=1,
            col=1,
        )

    # --- Row 2: split by label ---
    fig.add_trace(
        go.Histogram(
            x=keep_scores,
            nbinsx=num_bins,
            marker_color="green",
            opacity=0.6,
            name="GT: keep",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=prune_scores,
            nbinsx=num_bins,
            marker_color="red",
            opacity=0.6,
            name="GT: prune",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        barmode="overlay",
        row=2,
        col=1,
    )

    for tau, color in [(0.3, "orange"), (0.5, "red"), (0.7, "purple")]:
        fig.add_vline(
            x=tau,
            line_dash="dash",
            line_color=color,
            annotation_text=f"τ={tau}",
            annotation_position="top",
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"SWE-Pruner Line-Level Score Distribution (pooling={pooling})",
        height=900,
        width=1000,
        template="plotly_white",
        legend=dict(x=0.75, y=0.98),
    )
    fig.update_xaxes(title_text="Line Score", row=1, col=1)
    fig.update_xaxes(title_text="Line Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.write_html(output_path)
    print(f"Histogram saved to {output_path}")

    # Also print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Score distribution summary (pooling={pooling})")
    print(f"{'=' * 60}")
    print(f"  Total lines:   {len(all_scores)}")
    print(f"  GT-keep lines: {int(gt_labels.sum())}  ({gt_labels.mean():.1%})")
    print(
        f"  GT-prune lines:{int((1 - gt_labels).sum())}  ({1 - gt_labels.mean():.1%})"
    )
    print(f"  Mean score:    {all_scores.mean():.4f}")
    print(f"  Median score:  {np.median(all_scores):.4f}")
    print(f"  Std score:     {all_scores.std():.4f}")

    # Bimodality indicator: fraction of scores < 0.2 or > 0.8
    low = (all_scores < 0.2).mean()
    high = (all_scores > 0.8).mean()
    mid = ((all_scores >= 0.2) & (all_scores <= 0.8)).mean()
    print(f"\n  Bimodality check:")
    print(f"    score < 0.2:  {low:.1%}")
    print(f"    0.2 ≤ score ≤ 0.8: {mid:.1%}")
    print(f"    score > 0.8:  {high:.1%}")
    if low + high > 0.7:
        print("    → Strong bimodal tendency (>70% in tails)")
    elif low + high > 0.5:
        print("    → Moderate bimodal tendency (>50% in tails)")
    else:
        print("    → Weak bimodal tendency (<50% in tails)")


# ---------------------------------------------------------------------------
# Threshold ablation
# ---------------------------------------------------------------------------


def threshold_ablation(
    dataset: List[Dict[str, Any]],
    thresholds: List[float] = None,
    pooling: str = POOLING_MEAN,
) -> List[Dict[str, Any]]:
    """Evaluate F1 / IoU / precision / recall / compression at each threshold.

    Returns list of dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    results = []
    for tau in thresholds:
        f1_scores = []
        iou_scores = []
        prec_scores = []
        rec_scores = []
        comp_rates = []

        for sample in dataset:
            code = sample.get("code", "")
            token_scores = sample.get("token_scores", [])
            kept_frags = sample.get("kept_frags", [])
            if not token_scores:
                continue

            line_scores = compute_line_scores(code, token_scores, pooling=pooling)
            predicted = predict_kept_lines(line_scores, tau)

            f1_scores.append(evaluate_f1(predicted, kept_frags))
            iou_scores.append(evaluate_iou(predicted, kept_frags))
            prec_scores.append(evaluate_precision(predicted, kept_frags))
            rec_scores.append(evaluate_recall(predicted, kept_frags))

            total_lines = len(line_scores)
            if total_lines > 0:
                comp_rates.append((total_lines - len(predicted)) / total_lines)

        result = {
            "threshold": tau,
            "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            "precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
            "recall": float(np.mean(rec_scores)) if rec_scores else 0.0,
            "compression_rate": float(np.mean(comp_rates)) if comp_rates else 0.0,
            "num_samples": len(f1_scores),
        }
        results.append(result)

    return results


def print_ablation_table(results: List[Dict[str, Any]], pooling: str) -> None:
    """Pretty-print the ablation results as an ASCII table."""
    print(f"\n{'=' * 80}")
    print(f"Threshold Ablation (pooling={pooling})")
    print(f"{'=' * 80}")
    header = f"{'τ':>6s} | {'F1':>8s} | {'IoU':>8s} | {'Prec':>8s} | {'Recall':>8s} | {'Comp%':>8s} | {'N':>5s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['threshold']:6.2f} | "
            f"{r['f1']:8.4f} | "
            f"{r['iou']:8.4f} | "
            f"{r['precision']:8.4f} | "
            f"{r['recall']:8.4f} | "
            f"{r['compression_rate']:7.1%} | "
            f"{r['num_samples']:5d}"
        )
    print()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save a list of dicts as JSONL."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {file_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse SWE-Pruner score distribution and threshold sensitivity. "
            "Generates a histogram of line-level scores and a threshold ablation table."
        ),
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to validation JSONL (query, code, score, kept_frags).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running inference service (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Threshold values for ablation (default: 0.3 0.5 0.7).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "max", "both"],
        default="both",
        help="Line score pooling strategy (default: both).",
    )
    parser.add_argument(
        "--histogram-output",
        type=str,
        default="score_histogram.html",
        help="Output path for histogram HTML (default: score_histogram.html).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=50,
        help="Number of bins for the histogram (default: 50).",
    )
    parser.add_argument(
        "--save-enriched",
        type=str,
        default=None,
        help="If set, save enriched JSONL (with token_scores) to this path.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Skip inference API calls; assume token_scores already present "
            "in the JSONL (e.g. from a previous --save-enriched run)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout per request in seconds (default: 120).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample progress output.",
    )
    args = parser.parse_args()

    # 1. Load dataset
    print(f"Loading {args.jsonl_file} ...")
    dataset = load_jsonl(args.jsonl_file)
    print(f"  {len(dataset)} samples loaded.")

    # 2. Enrich with token_scores via inference API
    if not args.skip_inference:
        print(f"\nCalling inference service at {args.base_url} ...")
        dataset = enrich_dataset(
            dataset,
            base_url=args.base_url,
            timeout=args.timeout,
            verbose=not args.quiet,
        )

    # Check how many samples have token_scores
    n_with_scores = sum(1 for s in dataset if s.get("token_scores"))
    print(f"  {n_with_scores}/{len(dataset)} samples have token_scores.")
    if n_with_scores == 0:
        print("ERROR: No samples have token_scores. Is the inference service running?")
        sys.exit(1)

    # 3. Optionally save enriched JSONL
    if args.save_enriched:
        save_jsonl(dataset, args.save_enriched)

    # 4. Determine pooling strategies
    pooling_list = ["mean", "max"] if args.pooling == "both" else [args.pooling]

    for pooling in pooling_list:
        # 5. Histogram
        all_scores, gt_labels = collect_line_scores(dataset, pooling=pooling)
        if len(all_scores) == 0:
            print(f"WARNING: no line scores collected (pooling={pooling}). Skipping.")
            continue

        if PLOTLY_AVAILABLE:
            suffix = f"_{pooling}" if args.pooling == "both" else ""
            hist_path = args.histogram_output.replace(".html", f"{suffix}.html")
            plot_score_histogram(
                all_scores,
                gt_labels,
                output_path=hist_path,
                pooling=pooling,
                num_bins=args.num_bins,
            )
        else:
            print(
                "WARNING: plotly not installed — skipping histogram. "
                "Install with: pip install plotly"
            )
            # Still print summary stats
            print(f"\nScore distribution (pooling={pooling}):")
            print(
                f"  N={len(all_scores)}, mean={all_scores.mean():.4f}, "
                f"median={np.median(all_scores):.4f}, std={all_scores.std():.4f}"
            )
            low = (all_scores < 0.2).mean()
            high = (all_scores > 0.8).mean()
            print(f"  <0.2: {low:.1%}, >0.8: {high:.1%}, tails: {low + high:.1%}")

        # 6. Threshold ablation
        results = threshold_ablation(
            dataset, thresholds=args.thresholds, pooling=pooling
        )
        print_ablation_table(results, pooling)

    print("Done.")


if __name__ == "__main__":
    main()
