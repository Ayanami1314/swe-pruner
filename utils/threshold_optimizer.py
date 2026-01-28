"""
Threshold finder for code compression optimization.

Reads JSONL data with token scores and finds the optimal threshold that maximizes
a given evaluation metric (e.g., IoU or F1 with kept_frags). Supports Pareto
curve plotting (compression rate vs F1).
"""

import json
from typing import List, Dict, Any, Tuple, Callable

import numpy as np
from scipy.optimize import minimize_scalar

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# -----------------------------------------------------------------------------
# JSONL dataset format
# -----------------------------------------------------------------------------
#
# One JSON object per line with the following fields:
#
#   - code: str
#       Full original code (including newlines).
#   - token_scores: List[Tuple[str, float]]
#       Token-level scores; each item (token_str, score). token_str may contain
#       Ġ (space), Ċ (newline), ĉ (indent). score in [0, 1]; 1 = keep, 0 = remove.
#   - kept_frags: List[int]
#       Line numbers to keep (1-based), i.e. ground truth for evaluation.
#
# Example line:
#   {"code": "def f():\n  pass", "token_scores": [["Ġdef", 0.8], ["Ċ", 0.5], ...], "kept_frags": [1, 2]}
#


# -----------------------------------------------------------------------------
# Token markers (same as token_vis for consistency)
# -----------------------------------------------------------------------------

TOKEN_MARKER_SPACE = "Ġ"
TOKEN_MARKER_NEWLINE = "Ċ"
TOKEN_MARKER_INDENT = "ĉ"
REPLACEMENTS = (
    (TOKEN_MARKER_SPACE, " "),
    (TOKEN_MARKER_NEWLINE, "\n"),
    (TOKEN_MARKER_INDENT, "    "),
)

DEFAULT_LINE_SCORE = 0.5
POOLING_MEAN = "mean"
POOLING_MAX = "max"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file with eval token scores."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# -----------------------------------------------------------------------------
# Token string handling
# -----------------------------------------------------------------------------


def _token_to_content(token_str: str) -> str:
    """Convert tokenizer raw token to content string for checking non-whitespace."""
    s = token_str
    for old, new in REPLACEMENTS:
        s = s.replace(old, new)
    return s


def _token_has_newline(token_str: str) -> bool:
    """Return whether the token contains a newline marker."""
    return TOKEN_MARKER_NEWLINE in token_str


def _token_has_content(token_str: str) -> bool:
    """Return whether the token has non-whitespace content (for inclusion in line pooling)."""
    return bool(_token_to_content(token_str).strip())


def _clamp_score(score: float) -> float:
    """Clamp score to [0, 1]."""
    return max(0.0, min(1.0, float(score)))


# -----------------------------------------------------------------------------
# Line-level score computation
# -----------------------------------------------------------------------------


def compute_line_scores(
    code: str,
    token_scores: List[Tuple[str, float]],
    pooling: str = POOLING_MEAN,
) -> Dict[int, float]:
    """
    Compute token score per line based on newline tokens using pooling.

    Token scores should include "Ċ" (newline marker) tokens.
    Returns dict: {line_number (1-based) -> pooled_score}.

    Args:
        code: Source code string (used when token_scores is empty).
        token_scores: List of (token_str, score) tuples.
        pooling: 'mean' or 'max'.

    Returns:
        Dict mapping 1-based line number to pooled score.
    """
    if len(token_scores) == 0:
        lines = code.split("\n")
        return {i + 1: DEFAULT_LINE_SCORE for i in range(len(lines))}

    if pooling not in (POOLING_MEAN, POOLING_MAX):
        raise ValueError(f"pooling must be '{POOLING_MEAN}' or '{POOLING_MAX}', got '{pooling}'")

    line_scores: Dict[int, float] = {}
    current_line_num = 1
    current_line_scores: List[float] = []

    for token_str, score in token_scores:
        score = _clamp_score(score)
        has_newline = _token_has_newline(token_str)

        if _token_has_content(token_str):
            current_line_scores.append(score)

        if has_newline:
            if current_line_scores:
                if pooling == POOLING_MEAN:
                    line_scores[current_line_num] = float(np.mean(current_line_scores))
                else:
                    line_scores[current_line_num] = float(np.max(current_line_scores))
            else:
                line_scores[current_line_num] = DEFAULT_LINE_SCORE
            current_line_num += 1
            current_line_scores = []

    if current_line_scores:
        if pooling == POOLING_MEAN:
            line_scores[current_line_num] = float(np.mean(current_line_scores))
        else:
            line_scores[current_line_num] = float(np.max(current_line_scores))
    elif current_line_num == 1:
        line_scores[current_line_num] = DEFAULT_LINE_SCORE

    return line_scores


# -----------------------------------------------------------------------------
# Evaluation functions (predicted vs ground-truth line sets)
# -----------------------------------------------------------------------------


def evaluate_iou(predicted_lines: List[int], ground_truth_lines: List[int]) -> float:
    """
    Intersection over Union (IoU) between predicted and ground truth line sets.

    Returns IoU in [0, 1].
    """
    if not ground_truth_lines and not predicted_lines:
        return 1.0
    if not ground_truth_lines or not predicted_lines:
        return 0.0

    pred_set = set(predicted_lines)
    gt_set = set(ground_truth_lines)
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union if union else 1.0


def evaluate_f1(predicted_lines: List[int], ground_truth_lines: List[int]) -> float:
    """
    F1 score between predicted and ground truth line sets.

    Returns F1 in [0, 1].
    """
    if not ground_truth_lines and not predicted_lines:
        return 1.0

    pred_set = set(predicted_lines)
    gt_set = set(ground_truth_lines)
    intersection = len(pred_set & gt_set)

    if len(gt_set) == 0:
        precision = 1.0 if len(pred_set) == 0 else 0.0
        recall = 1.0
    else:
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = intersection / len(gt_set)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_precision(
    predicted_lines: List[int],
    ground_truth_lines: List[int],
) -> float:
    """Precision: intersection / len(predicted_lines)."""
    if not predicted_lines:
        return 1.0 if not ground_truth_lines else 0.0
    pred_set = set(predicted_lines)
    gt_set = set(ground_truth_lines)
    intersection = len(pred_set & gt_set)
    return intersection / len(pred_set)


def evaluate_recall(
    predicted_lines: List[int],
    ground_truth_lines: List[int],
) -> float:
    """Recall: intersection / len(ground_truth_lines)."""
    if not ground_truth_lines:
        return 1.0 if not predicted_lines else 0.0
    pred_set = set(predicted_lines)
    gt_set = set(ground_truth_lines)
    intersection = len(pred_set & gt_set)
    return intersection / len(gt_set)


# -----------------------------------------------------------------------------
# Threshold prediction and dataset-level metrics
# -----------------------------------------------------------------------------


def predict_kept_lines(line_scores: Dict[int, float], threshold: float) -> List[int]:
    """Predict which lines to keep: lines with score >= threshold."""
    return [line_num for line_num, score in line_scores.items() if score >= threshold]


def compute_compression_rate(
    dataset: List[Dict[str, Any]],
    threshold: float,
    pooling: str = POOLING_MEAN,
) -> float:
    """
    Average compression rate (deleted lines / total lines) over the dataset.

    Returns value in [0, 1]; higher = more lines deleted.
    """
    rates: List[float] = []
    for sample in dataset:
        code = sample.get("code", "")
        token_scores = sample.get("token_scores", [])
        if not token_scores:
            continue
        line_scores = compute_line_scores(code, token_scores, pooling=pooling)
        total_lines = len(line_scores)
        if total_lines == 0:
            continue
        predicted = predict_kept_lines(line_scores, threshold)
        deleted = total_lines - len(predicted)
        rates.append(deleted / total_lines)
    return float(np.mean(rates)) if rates else 0.0


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    threshold: float,
    eval_fn: Callable[[List[int], List[int]], float] = evaluate_iou,
    pooling: str = POOLING_MEAN,
) -> float:
    """
    Average evaluation score over the dataset at the given threshold.

    Each sample must have 'code', 'token_scores', and 'kept_frags'.
    """
    scores: List[float] = []
    for sample in dataset:
        code = sample.get("code", "")
        token_scores = sample.get("token_scores", [])
        kept_frags = sample.get("kept_frags", [])
        if not token_scores:
            continue
        line_scores = compute_line_scores(code, token_scores, pooling=pooling)
        predicted = predict_kept_lines(line_scores, threshold)
        scores.append(eval_fn(predicted, kept_frags))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------------------------------------------------------
# Pareto frontier and threshold scan
# -----------------------------------------------------------------------------


def scan_thresholds_for_pareto(
    dataset: List[Dict[str, Any]],
    thresholds: List[float],
    pooling: str = POOLING_MEAN,
) -> Tuple[List[float], List[float], List[float]]:
    """
    For each threshold, compute compression rate and F1 score.

    Returns (thresholds, compression_rates, f1_scores).
    """
    compression_rates: List[float] = []
    f1_scores: List[float] = []
    for threshold in thresholds:
        compression_rates.append(
            compute_compression_rate(dataset, threshold, pooling=pooling)
        )
        f1_scores.append(
            evaluate_dataset(dataset, threshold, evaluate_f1, pooling=pooling)
        )
    return thresholds, compression_rates, f1_scores


def compute_pareto_frontier(
    compression_rates: List[float],
    f1_scores: List[float],
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute Pareto frontier: for each F1 level, highest compression rate.

    A point is on the frontier if no other point has both same-or-higher F1
    and higher compression. Returns (pareto_comp, pareto_f1, pareto_indices).
    """
    points = list(zip(compression_rates, f1_scores, range(len(compression_rates))))
    pareto_points: List[Tuple[float, float, int]] = []

    for i, (comp_i, f1_i, idx_i) in enumerate(points):
        is_dominated = False
        for j, (comp_j, f1_j, _) in enumerate(points):
            if i == j:
                continue
            if f1_j >= f1_i and comp_j > comp_i:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((comp_i, f1_i, idx_i))

    pareto_points.sort(key=lambda x: x[0])
    return (
        [p[0] for p in pareto_points],
        [p[1] for p in pareto_points],
        [p[2] for p in pareto_points],
    )


# -----------------------------------------------------------------------------
# Optimal threshold search
# -----------------------------------------------------------------------------


def find_optimal_threshold(
    dataset: List[Dict[str, Any]],
    eval_fn: Callable[[List[int], List[int]], float] = evaluate_iou,
    method: str = "scipy",
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    grid_resolution: int = 100,
    pooling: str = POOLING_MEAN,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the evaluation metric.

    method: 'scipy' (bounded minimize_scalar) or 'grid' (grid search).
    Returns (optimal_threshold, optimal_score).
    """
    if method == "scipy":
        def objective(threshold: float) -> float:
            return -evaluate_dataset(dataset, threshold, eval_fn, pooling=pooling)

        result = minimize_scalar(
            objective,
            bounds=threshold_range,
            method="bounded",
            options={"xatol": 1e-4},
        )
        if result.success:
            return result.x, -result.fun
        method = "grid"

    if method == "grid":
        thresholds = np.linspace(
            threshold_range[0], threshold_range[1], grid_resolution
        )
        best_score = -np.inf
        best_threshold = threshold_range[0]
        for threshold in thresholds:
            score = evaluate_dataset(dataset, threshold, eval_fn, pooling=pooling)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold, best_score

    raise ValueError(f"Unknown method: {method}. Use 'scipy' or 'grid'.")


# -----------------------------------------------------------------------------
# Pareto curve plotting
# -----------------------------------------------------------------------------


def plot_pareto_curve(
    dataset: List[Dict[str, Any]],
    output_path: str = "pareto_curve.html",
    pooling: str = POOLING_MEAN,
    num_thresholds: int = 100,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
) -> None:
    """
    Plot Pareto curve: compression rate vs F1 score, with frontier highlighted.

    Saves an interactive HTML plot to output_path.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")

    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    _, compression_rates, f1_scores = scan_thresholds_for_pareto(
        dataset, thresholds.tolist(), pooling=pooling
    )
    pareto_comp, pareto_f1, pareto_indices = compute_pareto_frontier(
        compression_rates, f1_scores
    )
    pareto_thresholds = [float(thresholds[i]) for i in pareto_indices]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=compression_rates,
            y=f1_scores,
            mode="markers",
            name="All Thresholds",
            marker=dict(
                size=4,
                color=thresholds,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Threshold"),
                opacity=0.6,
            ),
            hovertemplate="<b>Compression Rate:</b> %{x:.3f}<br><b>F1 Score:</b> %{y:.3f}<br><extra></extra>",
        )
    )

    pareto_sorted = sorted(zip(pareto_comp, pareto_f1, pareto_thresholds))
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in pareto_sorted],
            y=[p[1] for p in pareto_sorted],
            mode="lines+markers",
            name="Pareto Frontier",
            line=dict(color="red", width=3),
            marker=dict(size=8, color="red", symbol="diamond"),
            hovertemplate="<b>Pareto</b><br>Compression: %{x:.3f}<br>F1: %{y:.3f}<br>Threshold: %{customdata:.3f}<br><extra></extra>",
            customdata=[p[2] for p in pareto_sorted],
        )
    )

    fig.update_layout(
        title=f"Pareto Curve: Compression Rate vs F1 (Pooling: {pooling.upper()})",
        xaxis_title="Compression Rate (Deleted / Total Lines)",
        yaxis_title="F1 Score",
        width=900,
        height=700,
        hovermode="closest",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.write_html(output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI: find optimal threshold and optionally plot Pareto curve."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find optimal threshold for code compression"
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to JSONL with code, token_scores, kept_frags",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        choices=["iou", "f1", "precision", "recall"],
        default="f1",
        help="Metric to maximize",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["scipy", "grid"],
        default="scipy",
        help="Optimization method",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=100,
        help="Grid size for grid search",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.0,
        help="Minimum threshold",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="Maximum threshold",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "max", "both"],
        default="both",
        help="Line score pooling: mean, max, or both",
    )
    parser.add_argument(
        "--plot-pareto",
        action="store_true",
        help="Save Pareto curve HTML",
    )
    parser.add_argument(
        "--pareto-output",
        type=str,
        default="pareto_curve.html",
        help="Output path for Pareto plot",
    )
    parser.add_argument(
        "--pareto-num-thresholds",
        type=int,
        default=100,
        help="Number of thresholds for Pareto curve",
    )
    args = parser.parse_args()

    dataset = load_jsonl(args.jsonl_file)
    eval_fns = {
        "iou": evaluate_iou,
        "f1": evaluate_f1,
        "precision": evaluate_precision,
        "recall": evaluate_recall,
    }
    eval_fn = eval_fns[args.eval_metric]
    pooling_list = ["mean", "max"] if args.pooling == "both" else [args.pooling]
    threshold_range = (args.threshold_min, args.threshold_max)

    results: Dict[str, Dict[str, float]] = {}
    for pooling in pooling_list:
        print(f"\n{'=' * 60}")
        print(f"Optimal threshold (method={args.method}, metric={args.eval_metric}, pooling={pooling})")
        opt_thresh, opt_score = find_optimal_threshold(
            dataset,
            eval_fn=eval_fn,
            method=args.method,
            threshold_range=threshold_range,
            grid_resolution=args.grid_resolution,
            pooling=pooling,
        )
        results[pooling] = {"threshold": opt_thresh, "score": opt_score}
        print(f"  Threshold: {opt_thresh:.6f}  Score: {opt_score:.6f}")

        print("  At selected thresholds:")
        for th in sorted({0.3, 0.4, 0.5, 0.6, 0.7, opt_thresh}):
            s = evaluate_dataset(dataset, th, eval_fn, pooling=pooling)
            mark = " <-- optimal" if abs(th - opt_thresh) < 1e-5 else ""
            print(f"    {th:.3f}: {s:.6f}{mark}")

    if len(results) == 2:
        print(f"\n{'=' * 60} COMPARISON")
        for p in ["mean", "max"]:
            r = results[p]
            print(f"  {p.upper()}: threshold={r['threshold']:.6f}, score={r['score']:.6f}")
        best = max(results.keys(), key=lambda p: results[p]["score"])
        print(f"  Best: {best.upper()}")

    if args.plot_pareto:
        if args.pooling == "both":
            for pooling in ["mean", "max"]:
                path = args.pareto_output.replace(".html", f"_{pooling}.html")
                plot_pareto_curve(
                    dataset,
                    output_path=path,
                    pooling=pooling,
                    num_thresholds=args.pareto_num_thresholds,
                    threshold_range=threshold_range,
                )
                print(f"Pareto curve saved: {path}")
        else:
            plot_pareto_curve(
                dataset,
                output_path=args.pareto_output,
                pooling=args.pooling,
                num_thresholds=args.pareto_num_thresholds,
                threshold_range=threshold_range,
            )
            print(f"Pareto curve saved: {args.pareto_output}")


if __name__ == "__main__":
    main()
