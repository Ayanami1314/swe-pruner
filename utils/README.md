This directory contains useful scripts.

## Visualization

### Token visualization (`token_vis.py`)

Requires `streamlit`.

1. Install: `pip install streamlit`
2. Run: `streamlit run utils/token_vis.py`
3. Upload a JSONL dataset (structure described at the top of `token_vis.py`)

You get a UI where:
- **Left** — Original code
- **Middle** — Pruned code by token score (greener = higher score)
- **Right** — Ground truth (kept lines)

Use the **Compression Settings** slider to see how the threshold affects pruning.

![token-vis](../images/token_vis.png)

---

## Threshold optimizer (`threshold_optimizer.py`)

Finds the **optimal score threshold** for code compression by maximizing a metric (IoU, F1, precision, or recall) against ground-truth kept lines. Optionally plots a **Pareto curve** (compression rate vs F1).

**Input:** A JSONL file where each line is a JSON object with:
- `code`: full source code string
- `token_scores`: list of `[token_str, score]` (score in [0, 1])
- `kept_frags`: list of 1-based line numbers to keep (ground truth)

**Dependencies:** `numpy`, `scipy`. For Pareto plotting: `plotly`.

### Basic usage

```bash
# Find optimal threshold (default: maximize F1, compare mean and max pooling)
python utils/threshold_optimizer.py path/to/eval_with_token_scores.jsonl

# Choose metric and pooling
python utils/threshold_optimizer.py data.jsonl --eval-metric iou --pooling mean

# Grid search instead of scipy (e.g. for reproducibility)
python utils/threshold_optimizer.py data.jsonl --method grid --grid-resolution 200
```

### Pareto curve (compression vs F1)

```bash
# Save interactive HTML: compression rate vs F1, with Pareto frontier
python utils/threshold_optimizer.py data.jsonl --plot-pareto --pareto-output pareto.html
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--eval-metric` | Metric to maximize: `iou`, `f1`, `precision`, `recall` | `f1` |
| `--method` | Search: `scipy` or `grid` | `scipy` |
| `--pooling` | Line score: `mean`, `max`, or `both` (compare both) | `both` |
| `--threshold-min`, `--threshold-max` | Search range | 0.0, 1.0 |
| `--grid-resolution` | Number of grid points when `--method grid` | 100 |
| `--plot-pareto` | Generate Pareto curve HTML | off |
| `--pareto-output` | Output path for Pareto plot | `pareto_curve.html` |
| `--pareto-num-thresholds` | Points for Pareto curve | 100 |

The script prints the optimal threshold and score for each pooling method, and (if `--pooling both`) which pooling is best. With `--plot-pareto`, it writes an interactive Plotly HTML file.
