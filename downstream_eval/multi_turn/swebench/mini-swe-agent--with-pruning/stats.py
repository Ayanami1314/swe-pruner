#!/usr/bin/env python3

"""Statistics script for mini-SWE-agent trajectory files.
Extracts and displays token usage statistics from .traj.json files.
"""

import json
import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(rich_markup_mode="rich")
console = Console()


def extract_token_stats(traj_path: Path) -> dict[str, Any]:
    """Extract token statistics from a trajectory file."""
    data = json.loads(traj_path.read_text())

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    api_calls = 0
    calls_with_usage = []
    pruned_token_totals: dict[str, int] = {}

    # Extract from messages
    for msg in data.get("messages", []):
        if msg.get("role") == "assistant" and "extra" in msg:
            extra = msg.get("extra", {})
            response = extra.get("response", {})
            usage = response.get("usage", {})

            if usage:
                prompt = usage.get("prompt_tokens", 0) or 0
                completion = usage.get("completion_tokens", 0) or 0
                total = usage.get("total_tokens", 0) or 0

                total_prompt_tokens += prompt
                total_completion_tokens += completion
                total_tokens += total
                api_calls += 1

                calls_with_usage.append(
                    {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": total,
                    }
                )

        pruned_stats = msg.get("pruned_stats")
        if isinstance(pruned_stats, dict):
            for key, value in pruned_stats.items():
                if key.endswith("_token_cnt") and isinstance(value, (int, float)):
                    token_value = int(value)
                    if token_value > 0:
                        pruned_token_totals[key] = pruned_token_totals.get(key, 0) + token_value

    # Get model stats from info if available
    model_stats = data.get("info", {}).get("model_stats", {})
    instance_cost = model_stats.get("instance_cost", 0.0)
    api_calls_from_info = model_stats.get("api_calls", 0)

    # Use api_calls from info if it's more accurate (sometimes messages might not have usage)
    if api_calls_from_info > api_calls:
        api_calls = api_calls_from_info

    return {
        "file": traj_path.name,
        "path": str(traj_path),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
        "instance_cost": instance_cost,
        "calls_with_usage": calls_with_usage,
        "pruned_token_totals": pruned_token_totals,
    }


def recur_find(d: dict[str, Any], key: str) -> Any:
    """Recursively find value for a given key in a nested dictionary."""
    if key in d:
        return d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            result = recur_find(v, key)
            if result is not None:
                return result
    return None


def extract_solve_stats(solve_path: Path) -> dict[str, Any]:
    data = json.loads(solve_path.read_text())
    return {"file": solve_path.name, "path": str(solve_path), **data}


def format_number(num: int | float) -> str:
    """Format number with thousand separators."""
    return f"{num:,}"


def format_pruner_column_name(key: str) -> str:
    base = key.removesuffix("_token_cnt").replace("_", " ").title()
    return f"{base} Tokens"


def find_traj_files(path: Path) -> list[Path]:
    """Find all .traj.json files in a directory or return the file if it's a file."""
    if path.is_file():
        return [path] if path.suffix == ".traj.json" else []
    elif path.is_dir():
        return sorted(path.rglob("*.traj.json"))
    else:
        return []


def find_solve_files(path: Path, reg_filter: str | None = None) -> list[Path]:
    """Find all report.json files in a directory or return the file if it's a file."""
    if path.is_file():
        return [path] if path.name == "report.json" else []
    elif path.is_dir():
        if reg_filter:
            pattern = re.compile(reg_filter)
            return [p for p in sorted(path.rglob("report.json")) if pattern.search(p.parent.name)]
        return sorted(path.rglob("report.json"))
    else:
        return []


def plot_token_distribution(stats: list[dict[str, Any]], output_path: Path):
    """Plot token usage distribution with modern scientific styling."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[red]Error: matplotlib is required for plotting. Please install it with 'pip install matplotlib'.[/red]"
        )
        return

    # Identify metrics to plot
    metrics = {
        "Prompt Tokens": [s["total_prompt_tokens"] for s in stats],
        "Completion Tokens": [s["total_completion_tokens"] for s in stats],
        "Total Tokens": [s["total_tokens"] for s in stats],
    }

    # Add pruner metrics
    pruner_keys = sorted({key for s in stats for key in s.get("pruned_token_totals", {})})

    for key in pruner_keys:
        display_name = format_pruner_column_name(key)
        metrics[display_name] = [s.get("pruned_token_totals", {}).get(key, 0) for s in stats]

    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_metrics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        if values:
            ax.hist(values, bins=min(20, len(values)), color="steelblue", edgecolor="black")
            ax.set_title(f"{name} Distribution", fontsize=14, fontweight="bold")
            ax.set_xlabel("Tokens", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)

            # Add mean/median lines
            if len(values) > 0:
                sorted_values = sorted(values)
                mean_val = sum(values) / len(values)
                median_val = sorted_values[len(values) // 2]
                p95_idx = min(int(len(values) * 0.95), len(values) - 1)
                p95_val = sorted_values[p95_idx]

                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"Mean: {mean_val:.0f}",
                )
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"Median: {median_val:.0f}",
                )
                ax.axvline(
                    p95_val,
                    color="orange",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"P95: {p95_val:.0f}",
                )
                ax.legend(fontsize=10)

    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    console.print(f"[green]Plot saved to {output_path}[/green]")


def plot_comparison_distribution(
    stats1: list[dict[str, Any]],
    stats2: list[dict[str, Any]],
    label1: str,
    label2: str,
    output_path: Path,
):
    """Plot comparison of token usage distributions using Plotly with modern styling."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        from scipy.stats import gaussian_kde
    except ImportError as e:
        console.print(f"[red]Error: plotly, numpy and scipy are required for plotting. {e}[/red]")
        return

    # Metrics to compare
    metrics_config = [
        ("total_prompt_tokens", "Prompt Tokens"),
        ("total_completion_tokens", "Completion Tokens"),
        ("total_tokens", "Total Tokens"),
        ("api_calls", "Agent Rounds"),
    ]

    # Add pruner metrics if present in both
    all_keys = set()
    for s in stats1 + stats2:
        all_keys.update(s.get("pruned_token_totals", {}).keys())

    for key in sorted(all_keys):
        metrics_config.append((f"pruned:{key}", format_pruner_column_name(key)))

    # Filter metrics where both datasets have data
    valid_metrics = []
    for key, name in metrics_config:
        data1 = []
        data2 = []
        if key.startswith("pruned:"):
            k = key.split(":", 1)[1]
            data1 = [s.get("pruned_token_totals", {}).get(k, 0) for s in stats1]
            data2 = [s.get("pruned_token_totals", {}).get(k, 0) for s in stats2]
        else:
            data1 = [s.get(key, 0) for s in stats1]
            data2 = [s.get(key, 0) for s in stats2]

        # Check if both have meaningful data (non-zero)
        if any(d > 0 for d in data1) and any(d > 0 for d in data2):
            valid_metrics.append((key, name))

    if not valid_metrics:
        console.print("[yellow]No common metrics found to plot.[/yellow]")
        return

    num_metrics = len(valid_metrics)
    cols = min(2, num_metrics)
    # cols = 1
    rows = (num_metrics + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    color1_fill = "rgba(31, 119, 180, 0.4)"  # Muted Blue
    color1_line = "rgb(31, 119, 180)"
    color2_fill = "rgba(255, 127, 14, 0.4)"  # Safety Orange
    color2_line = "rgb(255, 127, 14)"

    for i, (metric_key, metric_name) in enumerate(valid_metrics):
        row = i // cols + 1
        col = i % cols + 1

        data1 = []
        data2 = []

        if metric_key.startswith("pruned:"):
            key = metric_key.split(":", 1)[1]
            data1 = [s.get("pruned_token_totals", {}).get(key, 0) for s in stats1]
            data2 = [s.get("pruned_token_totals", {}).get(key, 0) for s in stats2]
        else:
            data1 = [s.get(metric_key, 0) for s in stats1]
            data2 = [s.get(metric_key, 0) for s in stats2]

        combined_data = data1 + data2
        if not combined_data:
            continue

        sorted_data = sorted(combined_data)
        p95_idx = int(len(sorted_data) * 0.95)
        x_max = sorted_data[p95_idx] if p95_idx < len(sorted_data) else max(combined_data)

        x_max = x_max * 1.1
        x_max = max(x_max, 20)

        bin_size = x_max / 20
        xbins = dict(start=0, end=x_max, size=bin_size)

        fig.add_trace(
            go.Histogram(
                x=data1,
                name=f"<b style='font-size:36px'>{label1}</b>",
                marker_color=color1_fill,
                opacity=0.6,
                xbins=xbins,
                autobinx=False,
                histnorm="probability density",
                showlegend=(i == 0),
                legendgroup="group1",
                marker=dict(line=dict(width=0)),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Histogram(
                x=data2,
                name=f"<b style='font-size:36px'>{label2}</b>",
                marker_color=color2_fill,
                opacity=0.6,
                xbins=xbins,
                autobinx=False,
                histnorm="probability density",
                showlegend=(i == 0),
                legendgroup="group2",
                marker=dict(line=dict(width=0)),
            ),
            row=row,
            col=col,
        )

        peak1 = None
        if len(data1) > 1 and np.std(data1) > 0:
            try:
                kde = gaussian_kde(data1)
                x_range = np.linspace(0, x_max, 500)
                y_vals = kde(x_range)
                peak_idx = np.argmax(y_vals)
                peak1 = (x_range[peak_idx], y_vals[peak_idx])

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_vals,
                        mode="lines",
                        line=dict(color=color1_line, width=7),
                        showlegend=False,
                        legendgroup="group1",
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            except Exception:
                pass

        peak2 = None
        if len(data2) > 1 and np.std(data2) > 0:
            try:
                kde = gaussian_kde(data2)
                x_range = np.linspace(0, x_max, 500)
                y_vals = kde(x_range)
                peak_idx = np.argmax(y_vals)
                peak2 = (x_range[peak_idx], y_vals[peak_idx])

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_vals,
                        mode="lines",
                        line=dict(color=color2_line, width=7),
                        showlegend=False,
                        legendgroup="group2",
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            except Exception:
                pass

        y_axis_range = None
        if peak1 and peak2:
            val1 = peak1[0]
            val2 = peak2[0]
            # Only show if there is a saving (val1 > val2) and it's significant
            if val1 > val2:
                saving = (val1 - val2) / val1 * 100

                # Determine axis names for this subplot
                axis_idx = i + 1
                xaxis = f"x{axis_idx}" if axis_idx > 1 else "x"
                yaxis = f"y{axis_idx}" if axis_idx > 1 else "y"

                # Position the arrow above the highest peak
                max_y = max(peak1[1], peak2[1])
                arrow_y = max_y * 1.2  # Slightly higher to avoid overlap
                y_axis_range = [
                    0,
                    max_y * 1.6,
                ]  # Ensure enough space for arrow and text (increased from 1.35)

                # Add vertical dashed lines for peaks extending up to the arrow
                fig.add_shape(
                    type="line",
                    x0=val1,
                    y0=0,
                    x1=val1,
                    y1=arrow_y,
                    xref=xaxis,
                    yref=yaxis,
                    opacity=0.8,
                    line=dict(color=color1_line, width=7, dash="dash"),
                )
                fig.add_shape(
                    type="line",
                    x0=val2,
                    y0=0,
                    x1=val2,
                    y1=arrow_y,
                    xref=xaxis,
                    yref=yaxis,
                    opacity=0.8,
                    line=dict(color=color2_line, width=7, dash="dash"),
                )

                # Add horizontal arrow from Baseline Peak X to Pruner Peak X
                fig.add_annotation(
                    x=val2,
                    y=arrow_y,  # Head (Pruner)
                    ax=val1,
                    ay=arrow_y,  # Tail (Baseline)
                    xref=xaxis,
                    yref=yaxis,
                    axref=xaxis,
                    ayref=yaxis,
                    text="",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.3,
                    arrowwidth=4,
                    arrowcolor="#d62728",
                )

                # Add percentage label above the arrow
                fig.add_annotation(
                    x=(val1 + val2) / 2,
                    y=arrow_y,
                    xref=xaxis,
                    yref=yaxis,
                    text=f"<b>{saving:.1f}%</b>",
                    showarrow=False,
                    yshift=36,
                    font=dict(size=36, color="#d62728", family="Arial"),
                )

        fig.update_xaxes(
            title_text=f"<b>{metric_name}</b>",
            range=[0, x_max],
            row=row,
            col=col,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickfont=dict(family="Arial", size=32, color="black"),
            title_font=dict(family="Arial", size=36, color="black"),
        )
        fig.update_yaxes(
            title_text="<b>Frequency</b>" if col == 1 else None,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            showticklabels=False,
            row=row,
            col=col,
            title_font=dict(family="Arial", size=36, color="black"),
            rangemode="tozero",
            range=y_axis_range,
        )

    fig.update_layout(
        font=dict(family="Arial", size=28, color="black"),
        template="simple_white",
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=24, family="Arial"),
        ),
        margin=dict(l=80, r=40, t=60, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.05,
        width=1200,
        height=1200,
    )

    html_path = output_path.with_suffix(".html")
    fig.write_html(html_path)
    console.print(f"[green]Interactive plot saved to {html_path}[/green]")

    try:
        fig.write_image(output_path, format="pdf", scale=2)
        console.print(f"[green]Static plot saved to {output_path}[/green]")
    except Exception:
        console.print("[yellow]Failed to save static image. Ensure kaleido is installed.[/yellow]")


@app.command()
def compare(
    path1: Path = typer.Argument(..., help="First directory of trajectory files"),
    path2: Path = typer.Argument(..., help="Second directory of trajectory files"),
    label1: str = typer.Option("Group 1", "--label1", "-l1", help="Label for first group"),
    label2: str = typer.Option("Group 2", "--label2", "-l2", help="Label for second group"),
    plot_output: Path = typer.Option("comparison.png", "--plot", "-p", help="Output path for comparison plot"),
):
    """Compare token usage between two sets of trajectory files."""

    def get_stats(path):
        stats_list = []
        if not path.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            return []

        files = find_traj_files(path)
        if not files:
            console.print(f"[yellow]Warning:[/yellow] No .traj.json files found in {path}")
            return []

        for f in files:
            try:
                stats_list.append(extract_token_stats(f))
            except Exception as e:
                console.print(f"[red]Error[/red] reading {f}: {e}")
        return stats_list

    stats1 = get_stats(path1)
    stats2 = get_stats(path2)

    if not stats1 or not stats2:
        console.print("[red]Could not load stats for comparison.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Comparing {label1} ({len(stats1)} files) vs {label2} ({len(stats2)} files)[/bold]")

    # Print simple comparison of means
    table = Table(title="Comparison of Means")
    table.add_column("Metric")
    table.add_column(f"{label1} (Mean)")
    table.add_column(f"{label2} (Mean)")
    table.add_column("Diff %")

    metrics = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "instance_cost",
        "api_calls",
    ]

    for m in metrics:
        val1 = sum(s[m] for s in stats1) / len(stats1)
        val2 = sum(s[m] for s in stats2) / len(stats2)
        diff = ((val2 - val1) / val1 * 100) if val1 > 0 else 0

        fmt = "${:.4f}" if "cost" in m else "{:,.0f}"
        table.add_row(
            m.replace("_", " ").title(),
            fmt.format(val1),
            fmt.format(val2),
            f"{diff:+.1f}%",
        )

    console.print(table)

    if plot_output:
        plot_comparison_distribution(stats1, stats2, label1, label2, plot_output)


@app.command()
def main(
    paths: list[Path] = typer.Argument(..., help="Trajectory file(s) or directory(ies) to analyze"),
    detailed: bool = typer.Option(False, "-d", "--detailed", help="Show detailed per-call statistics"),
    summary_only: bool = typer.Option(False, "-s", "--summary", help="Show only summary totals"),
    aggregate: bool = typer.Option(
        False,
        "-a",
        "--aggregate",
        help="Show aggregate statistics (mean, median, etc.) for multiple files",
    ),
    plot_output: Path = typer.Option(None, "--plot", "-p", help="Save token usage distribution plot to file"),
    stats_file: Path = typer.Option(None, "--stats-file", "-f", help="Save extracted stats to JSON file"),
    reg_filter: str = typer.Option(
        None,
        "--filter",
        "-F",
        help="Regex filter to apply to trajectory file names",
    ),
    resolve_log_dir: Path = typer.Option(
        None,
        "--resolve-log-dir",
        "-R",
        help="Directory containing solve log files to extract resolve statistics",
    ),
):
    """Display token usage statistics from mini-SWE-agent trajectory files.

    Can analyze individual files or entire directories (e.g., runs/pruner-v3-glm).
    """

    all_stats = []
    all_traj_files = []
    solve_cnt = 0
    slice_start = 0
    slice_end = 500
    # Collect all trajectory files
    for path in paths:
        if not path.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            continue

        traj_files = find_traj_files(path)
        origin_len = len(traj_files)
        if reg_filter:
            pattern = re.compile(reg_filter)
            traj_files = [f for f in traj_files if pattern.search(f.name)]
            print(f"Filtered instances: {origin_len} -> {len(traj_files)}")
        if not resolve_log_dir:
            if path.is_dir() and Path(path.parent / "logs").exists():
                resolve_log_dir = Path(path.parent / "logs")
        if resolve_log_dir:
            solve_files = find_solve_files(resolve_log_dir, reg_filter)
            print(f"Found {len(solve_files)} solve log files for resolve stats.")
            solve_cnt = 0
            for f in solve_files[slice_start:slice_end]:
                solve_cnt += int(recur_find(extract_solve_stats(f), "resolved") or 0)
            console.print(f"[yellow]Resolved:[/yellow] {solve_cnt} / {len(solve_files[slice_start:slice_end])}")

        if not traj_files:
            console.print(f"[yellow]Warning:[/yellow] No .traj.json files found in {path}")
            continue

        all_traj_files.extend(traj_files)

    # Process each trajectory file
    for traj_file in all_traj_files:
        try:
            stats = extract_token_stats(traj_file)
            all_stats.append(stats)
        except Exception as e:
            console.print(f"[red]Error[/red] reading {traj_file}: {e}")
            continue

    if not all_stats:
        console.print("[red]No valid trajectory files found.[/red]")
        raise typer.Exit(1)
    all_stats = all_stats[slice_start:slice_end]
    pruner_keys = sorted({key for stats in all_stats for key in stats.get("pruned_token_totals", {})})

    # Display summary table
    summary_table = Table(title="Token Usage Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("File", style="cyan", no_wrap=True)
    summary_table.add_column("API Calls", justify="right", style="green")
    summary_table.add_column("Prompt Tokens", justify="right", style="yellow")
    summary_table.add_column("Completion Tokens", justify="right", style="blue")
    summary_table.add_column("Total Tokens", justify="right", style="bold")
    summary_table.add_column("Cost ($)", justify="right", style="red")
    for key in pruner_keys:
        summary_table.add_column(format_pruner_column_name(key), justify="right", style="magenta")

    grand_total_prompt = 0
    grand_total_completion = 0
    grand_total_tokens = 0
    grand_total_calls = 0
    grand_total_cost = 0.0

    grand_pruner_totals = {key: 0 for key in pruner_keys}
    exceed_limit = 100
    exceed_limit_calls = 0
    for stats in all_stats:
        row = [
            stats["file"],
            format_number(stats["api_calls"]),
            format_number(stats["total_prompt_tokens"]),
            format_number(stats["total_completion_tokens"]),
            format_number(stats["total_tokens"]),
            f"${stats['instance_cost']:.4f}",
        ]
        for key in pruner_keys:
            value = stats.get("pruned_token_totals", {}).get(key, 0)
            row.append(format_number(value) if value else "-")
            grand_pruner_totals[key] += value
        summary_table.add_row(*row)
        # if stats["api_calls"] > exceed_limit:
        #    exceed_limit_calls += 1
        #    continue
        grand_total_prompt += stats["total_prompt_tokens"]
        grand_total_completion += stats["total_completion_tokens"]
        grand_total_tokens += stats["total_tokens"]
        grand_total_calls += stats["api_calls"]
        grand_total_cost += stats["instance_cost"]

    console.print(f"[yellow]Exceed Limit Calls:[/yellow] {exceed_limit_calls}")

    # Add totals row if multiple files
    if len(all_stats) > 1:
        total_row = [
            "[bold]TOTAL[/bold]",
            f"[bold]{format_number(grand_total_calls)}[/bold]",
            f"[bold]{format_number(grand_total_prompt)}[/bold]",
            f"[bold]{format_number(grand_total_completion)}[/bold]",
            f"[bold]{format_number(grand_total_tokens)}[/bold]",
            f"[bold]${grand_total_cost:.4f}[/bold]",
        ]
        for key in pruner_keys:
            value = grand_pruner_totals[key]
            total_row.append(f"[bold]{format_number(value)}[/bold]" if value else "-")
        summary_table.add_row(*total_row)

        # Add aggregate statistics row if requested
        if aggregate:
            num_files = len(all_stats)
            mean_row = [
                "[bold]MEAN[/bold]",
                f"[bold]{format_number(grand_total_calls // num_files)}[/bold]",
                f"[bold]{format_number(grand_total_prompt // num_files)}[/bold]",
                f"[bold]{format_number(grand_total_completion // num_files)}[/bold]",
                f"[bold]{format_number(grand_total_tokens // num_files)}[/bold]",
                f"[bold]${grand_total_cost / num_files:.4f}[/bold]",
            ]
            for key in pruner_keys:
                value = grand_pruner_totals[key]
                mean_row.append(f"[bold]{format_number(value // num_files)}[/bold]" if value else "-")
            summary_table.add_row(*mean_row)

    console.print(summary_table)

    # Show detailed per-call statistics if requested
    if detailed and not summary_only:
        for stats in all_stats:
            if not stats["calls_with_usage"]:
                continue

            console.print(f"\n[bold cyan]Detailed statistics for {stats['file']}:[/bold cyan]")
            detail_table = Table(show_header=True, header_style="bold")
            detail_table.add_column("Call #", justify="right", style="dim")
            detail_table.add_column("Prompt Tokens", justify="right")
            detail_table.add_column("Completion Tokens", justify="right")
            detail_table.add_column("Total Tokens", justify="right")

            for i, call in enumerate(stats["calls_with_usage"], 1):
                detail_table.add_row(
                    str(i),
                    format_number(call["prompt_tokens"]),
                    format_number(call["completion_tokens"]),
                    format_number(call["total_tokens"]),
                )

            console.print(detail_table)

    # Show aggregate statistics if multiple files
    if not summary_only and len(all_stats) > 1:
        num_files = len(all_stats)

        # Calculate per-instance statistics
        instance_stats = {
            "api_calls": [s["api_calls"] for s in all_stats],
            "prompt_tokens": [s["total_prompt_tokens"] for s in all_stats],
            "completion_tokens": [s["total_completion_tokens"] for s in all_stats],
            "total_tokens": [s["total_tokens"] for s in all_stats],
            "cost": [s["instance_cost"] for s in all_stats],
        }
        for key in pruner_keys:
            values = [s.get("pruned_token_totals", {}).get(key, 0) for s in all_stats]
            if any(values):
                instance_stats[key] = values

        console.print(f"\n[bold]Aggregate Statistics ({num_files} instances):[/bold]")

        for stat_name, values in instance_stats.items():
            filtered_values = [v for v in values if v > 0] if stat_name in pruner_keys else values
            if not filtered_values:
                continue

            values_sorted = sorted(filtered_values)
            mean_val = sum(filtered_values) / len(filtered_values)
            median_val = values_sorted[len(values_sorted) // 2]
            min_val = values_sorted[0]
            max_val = values_sorted[-1]

            if stat_name == "cost":
                display_name = "Cost"
                console.print(f"  {display_name}:")
                console.print(f"    Mean: ${mean_val:.4f}")
                console.print(f"    Median: ${median_val:.4f}")
                console.print(f"    Min: ${min_val:.4f}")
                console.print(f"    Max: ${max_val:.4f}")
            else:
                display_name = (
                    format_pruner_column_name(stat_name)
                    if stat_name in pruner_keys
                    else stat_name.replace("_", " ").title()
                )
                console.print(f"  {display_name}:")
                console.print(f"    Mean: {format_number(int(mean_val))}")
                console.print(f"    Median: {format_number(median_val)}")
                console.print(f"    Min: {format_number(min_val)}")
                console.print(f"    Max: {format_number(max_val)}")

        # Show averages per API call
        console.print("\n[bold]Averages per API call:[/bold]")
        if grand_total_calls > 0:
            console.print(f"  Prompt tokens: {format_number(grand_total_prompt // grand_total_calls)}")
            console.print(f"  Completion tokens: {format_number(grand_total_completion // grand_total_calls)}")
            console.print(f"  Total tokens: {format_number(grand_total_tokens // grand_total_calls)}")
            if grand_total_cost > 0:
                console.print(f"  Cost per call: ${grand_total_cost / grand_total_calls:.6f}")

        # Show per-instance averages
        console.print("\n[bold]Averages per instance:[/bold]")
        console.print(f"  API calls: {format_number(grand_total_calls / num_files)}")
        console.print(f"  Prompt tokens: {format_number(grand_total_prompt // num_files)}")
        console.print(f"  Completion tokens: {format_number(grand_total_completion // num_files)}")
        console.print(f"  Total tokens: {format_number(grand_total_tokens // num_files)}")
        console.print(f"  Cost: ${grand_total_cost / num_files:.4f}")
        if any(grand_pruner_totals.values()):
            for key in pruner_keys:
                value = grand_pruner_totals[key]
                if value:
                    console.print(f"  {format_pruner_column_name(key)}: {format_number(value // num_files)}")
    elif not summary_only and all_stats:
        # Single file case - show per-call averages
        console.print("\n[bold]Averages per API call:[/bold]")
        if grand_total_calls > 0:
            console.print(f"  Prompt tokens: {format_number(grand_total_prompt // grand_total_calls)}")
            console.print(f"  Completion tokens: {format_number(grand_total_completion // grand_total_calls)}")
            console.print(f"  Total tokens: {format_number(grand_total_tokens // grand_total_calls)}")
            if grand_total_cost > 0:
                console.print(f"  Cost per call: ${grand_total_cost / grand_total_calls:.6f}")
        if any(grand_pruner_totals.values()):
            console.print("\n[bold]Pruner token totals:[/bold]")
            for key in pruner_keys:
                value = grand_pruner_totals[key]
                if value:
                    console.print(f"  {format_pruner_column_name(key)}: {format_number(value)}")
    if stats_file:
        json.dump(all_stats, open(stats_file, "w"), indent=2)
    if plot_output:
        # save stats as json
        plot_token_distribution(all_stats, plot_output)


def extract_bash_commands(traj_path: Path) -> list[tuple[int, str, int]]:
    """Extract bash commands from assistant messages.

    Returns a list of tuples: (step_index, command, completion_tokens)
    where completion_tokens is the completion tokens for this step.
    """
    data = json.loads(traj_path.read_text())
    messages = data.get("messages", [])

    commands = []
    step_index = 0

    # Pattern to match ```bash\n[command]\n```
    bash_pattern = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            match = bash_pattern.search(content)

            if match:
                command = match.group(1).strip()
                # Get completion tokens from extra.response.usage
                completion_tokens = 0
                if "extra" in msg:
                    extra = msg.get("extra", {})
                    response = extra.get("response", {})
                    usage = response.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0) or 0

                commands.append((step_index, command, completion_tokens))
                step_index += 1

    return commands


def split_command(command: str) -> list[str]:
    """Split a command into individual commands by && and |.

    Handles commands connected with && or |, returning a list of individual commands.
    """
    # Split by && first, then by |
    parts = []
    for part in command.split("&&"):
        parts.extend(part.split("|"))

    # Clean up each part
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned


def classify_single_command(command: str) -> str:
    """Classify a command into one of: 'read', 'execute', 'edit', 'other'.

    Heuristic-based classification:
    - read: cat, nl, ls, grep, head, tail, find (without -exec), etc.
    - execute: curl, python, ping, pip, npm, node, make, etc.
    - edit: sed -i, vim, nano, etc.
    - other: everything else
    """
    command_lower = command.lower().strip()

    # Split command to get the base command (first word)
    parts = command_lower.split()
    if not parts:
        return "other"

    base_cmd = parts[0]

    # Reading operations (no editing or execution)
    read_commands = {
        "cat",
        "nl",
        "ls",
        "grep",
        "head",
        "tail",
        "find",
        "which",
        "whereis",
        "type",
        "file",
        "stat",
        "readlink",
        "dirname",
        "basename",
        "pwd",
        "echo",
        "printenv",
        "env",
        "export",
        "cd",
        "pushd",
        "popd",
    }

    # Execution operations
    execute_commands = {
        "curl",
        "wget",
        "python",
        "python3",
        "pip",
        "pip3",
        "npm",
        "node",
        "ping",
        "make",
        "cmake",
        "gcc",
        "g++",
        "javac",
        "java",
        "go",
        "rustc",
        "cargo",
        "docker",
        "kubectl",
        "git",
        "svn",
        "hg",
        "bash",
        "sh",
        "zsh",
        "perl",
        "ruby",
        "php",
        "dotnet",
        "dotnet",
        "mvn",
        "gradle",
        "yarn",
        "npx",
        "tsc",
        "eslint",
        "pytest",
        "unittest",
        "nosetests",
        "tox",
    }

    # Editing operations
    edit_commands = {"sed", "vim", "nano", "vi", "emacs", "ed"}

    # Check for edit patterns (like sed -i)
    if base_cmd in edit_commands:
        return "edit"

    # Check for sed -i specifically (in-place editing)
    if "sed" in command_lower and "-i" in command_lower:
        return "edit"

    # Check reading operations
    if base_cmd in read_commands:
        # Special case: find with -exec is execution, not reading
        if base_cmd == "find" and "-exec" in command_lower:
            return "execute"
        return "read"

    # Check execution operations
    if base_cmd in execute_commands:
        return "execute"

    # Default to other
    return "other"


def classify_command(command: str) -> str:
    """Classify a command (possibly containing multiple commands) into one category.

    If the command contains multiple commands (connected with && or |),
    classify each and return the highest priority one.
    Priority: edit > execute > read > other
    """
    # Split command into individual commands
    individual_commands = split_command(command)

    if not individual_commands:
        return "other"

    # Classify each command
    classifications = [classify_single_command(cmd) for cmd in individual_commands]

    # Return the highest priority classification
    priority_order = {"edit": 0, "execute": 1, "read": 2, "other": 3}
    return min(classifications, key=lambda x: priority_order[x])


def calculate_token_contributions(commands: list[tuple[int, str, int]], weighted: bool = True) -> dict[str, float]:
    """Calculate token contributions for each operation type.

    If weighted=True (O(n²) model):
        For a command at step i with completion_tokens t_i, its contribution is:
        t_i * (total_steps - i + 1)
        This accounts for the O(n²) complexity: each step's tokens are weighted
        by how many subsequent steps will include them (including the current step).

    If weighted=False:
        Each step's tokens contribute equally: t_i * 1
    """
    if not commands:
        return {"read": 0.0, "execute": 0.0, "edit": 0.0, "other": 0.0}

    total_steps = len(commands)
    contributions = {"read": 0.0, "execute": 0.0, "edit": 0.0, "other": 0.0}

    for step_index, command, completion_tokens in commands:
        op_type = classify_command(command)
        if weighted:
            # Weight: how many subsequent steps (including current) will include this step's tokens
            # Step i's tokens are used by steps i, i+1, ..., n-1, so weight = n - i
            # But we want to include the current step, so weight = n - i
            # Actually, if step_index goes from 0 to n-1, then:
            # - Step 0 is used by steps 0, 1, ..., n-1 (n steps), weight = n
            # - Step 1 is used by steps 1, 2, ..., n-1 (n-1 steps), weight = n-1
            # - Step i is used by steps i, i+1, ..., n-1 (n-i steps), weight = n - i
            weight = total_steps - step_index
        else:
            # No weighting, each step contributes equally
            weight = 1
        contributions[op_type] += completion_tokens * weight

    return contributions


def plot_operation_pie_chart(
    contributions: dict[str, float],
    output_path: Path,
    title: str = "Token Contribution by Operation Type",
    color_scheme: str = "scientific",  # 新增参数，可选择配色方案
):
    """Plot a pie chart showing token contribution proportions by operation type using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        console.print("[red]Error: plotly is required for plotting. Please install it with 'pip install plotly'.[/red]")
        return

    # Filter out zero contributions
    filtered_contributions = {k: v for k, v in contributions.items() if v > 0}

    if not filtered_contributions:
        console.print("[yellow]No token contributions found to plot.[/yellow]")
        return

    # Prepare data with modern scientific color schemes
    labels = []
    values = []

    # 定义多种配色方案
    colors = {
        "read": "#34db53",
        "execute": "#2e68cc",
        "edit": "#e67e22",
        "other": "#95a5a6",
    }
    pull_values = []
    color_list = []

    for op_type in ["read", "execute", "edit", "other"]:
        if op_type in filtered_contributions:
            labels.append(op_type.capitalize())
            values.append(filtered_contributions[op_type])
            color_list.append(colors[op_type])
            # Pull "read" section out to highlight it
            pull_values.append(0.15 if op_type == "read" else 0.0)

    # Calculate percentages and format labels
    total = sum(values)
    text_labels = []
    for i, (label, value) in enumerate(zip(labels, values)):
        percentage = (value / total * 100) if total > 0 else 0
        # Format large numbers with commas
        formatted_value = f"{value:,.0f}".replace(",", " ")
        text_labels.append(f"{label}<br>{percentage:.1f}%<br>({formatted_value} tokens)")

    # Create pie chart with Plotly
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,  # Donut chart style (more modern)
                pull=pull_values,  # Pull out "read" section
                marker=dict(
                    colors=color_list,
                    line=dict(color="#FFFFFF", width=2),  # 稍微减少边框宽度
                ),
                textinfo="text",
                text=text_labels,
                hovertemplate="<b>%{label}</b><br>"
                + "Value: %{value:,.0f}<br>"
                + "Percentage: %{percent}<br>"
                + "<extra></extra>",
            )
        ]
    )

    # Update layout with modern scientific publication style
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center title
            xanchor="center",
            font=dict(size=20, family="Arial, sans-serif", color="#333333"),  # 更现代的字体和颜色
            pad=dict(t=20, b=30),
        ),
        font=dict(size=14, family="Arial, sans-serif", color="#333333"),  # 更现代的字体和颜色
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,
            font=dict(size=14, family="Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCCCCC",  # 更柔和的边框颜色
            borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1000,
        height=800,
        margin=dict(l=50, r=200, t=100, b=50),
    )

    # Update text font size for better readability
    fig.update_traces(
        textfont=dict(size=14, family="Arial, sans-serif", color="white"),  # 白色文字在彩色背景上更易读
        textposition="inside",
    )

    # Save as HTML (interactive)
    html_path = output_path.with_suffix(".html")
    fig.write_html(html_path)
    console.print(f"[green]Interactive plot saved to {html_path}[/green]")

    # Try to save as static image
    try:
        fig.write_image(output_path, scale=3, width=1000, height=800)  # High resolution
        console.print(f"[green]Static plot saved to {output_path}[/green]")
    except Exception:
        console.print("[yellow]Could not save static image (requires kaleido). Please check the HTML file.[/yellow]")


def is_file_content_read_command(command: str) -> bool:
    """Check if a command reads file content (cat, nl, head, tail, less, more, sed -n).

    Only commands that actually read file content are considered, excluding
    metadata commands like ls, grep, find, etc.
    """
    if not isinstance(command, str):
        return False

    command_lower = command.lower().strip()
    parts = command_lower.split()
    if not parts:
        return False

    base_cmd = parts[0]

    # Commands that read file content
    content_read_commands = {"cat", "nl", "head", "tail", "less", "more"}

    # Special case: sed with -n flag (used for reading/printing)
    if base_cmd == "sed" and "-n" in parts:
        return True

    return base_cmd in content_read_commands


def is_edit_command(command: str) -> bool:
    """Check if a command edits files (sed -i, vim, nano, etc.)."""
    if not isinstance(command, str):
        return False

    command_lower = command.lower().strip()
    parts = command_lower.split()
    if not parts:
        return False

    base_cmd = parts[0]

    # Editing commands
    edit_commands = {"sed", "vim", "nano", "vi", "emacs", "ed"}

    if base_cmd in edit_commands:
        # sed without -i is not editing
        if base_cmd == "sed" and "-i" not in parts:
            return False
        return True

    return False


def extract_file_paths_from_bash_command(command: str) -> list[str]:
    """Extract file paths from a bash command.

    Only extracts paths that:
    1. Contain '/' (path separator)
    2. End with a file extension
    3. Are not example paths like /path/to/, /example/

    Returns sorted list of unique file paths.
    """
    if not isinstance(command, str):
        return []

    # Common file extensions
    extensions = [
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
        ".xml",
        ".sql",
        ".r",
        ".R",
        ".m",
        ".swift",
        ".kt",
        ".scala",
        ".vue",
        ".svelte",
        ".dart",
        ".lua",
        ".pl",
        ".pm",
        ".tcl",
        ".ini",
        ".conf",
        ".config",
        ".properties",
        ".env",
        ".lock",
        ".log",
        ".out",
        ".err",
    ]

    file_paths = set()

    # Simple word splitting (avoid shlex complexity)
    words = command.split()

    for word in words:
        # Skip flags and options
        if word.startswith("-"):
            continue

        # Skip redirect operators
        if word in (">", ">>", "<", "|", "&&", "||"):
            continue

        # Clean up quotes
        cleaned = word.strip("'\"")

        # Check if it's a file path: has '/' and ends with extension
        has_slash = "/" in cleaned
        has_extension = any(cleaned.lower().endswith(ext) for ext in extensions)

        if has_slash and has_extension:
            # Skip example paths
            if cleaned.startswith("/path/to/") or cleaned.startswith("/example/"):
                continue

            file_paths.add(cleaned)

    return sorted(list(file_paths))


def extract_read_operations_with_output(
    traj_path: Path,
) -> list[tuple[str, str, list[str]]]:
    """Extract file content read commands with their outputs from trajectory file.

    Returns list of (command, output, file_paths) tuples.
    Only extracts commands from bash code blocks, not from THOUGHT sections.
    """
    try:
        data = json.loads(traj_path.read_text())
    except Exception:
        return []

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return []

    results = []
    bash_pattern = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue

        # Extract content - handle both string and list
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if isinstance(item, str))
        elif not isinstance(content, str):
            content = str(content) if content else ""

        # Find bash command in code block
        match = bash_pattern.search(content)
        if not match:
            i += 1
            continue

        command = match.group(1).strip()

        # Check if this is a file content read command
        if not is_file_content_read_command(command):
            i += 1
            continue

        # Extract file paths from command
        file_paths = extract_file_paths_from_bash_command(command)
        if not file_paths:
            i += 1
            continue

        # Find output in next user message
        output = ""
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            if isinstance(next_msg, dict) and next_msg.get("role") == "user":
                next_content = next_msg.get("content", "")
                output = next_content if isinstance(next_content, str) else str(next_content)

        results.append((command, output, file_paths))
        i += 1

    return results


def extract_edit_operations(traj_path: Path) -> set[str]:
    """Extract file paths from edit commands in trajectory file.

    Returns set of file paths that were edited.
    Includes:
    - Traditional edit commands (sed -i, vim, etc.)
    - Redirection operations (command > file, command >> file)
    """
    try:
        data = json.loads(traj_path.read_text())
    except Exception:
        return set()

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return set()

    edit_files = set()
    bash_pattern = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        # Extract content
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if isinstance(item, str))
        elif not isinstance(content, str):
            content = str(content) if content else ""

        # Find bash command
        match = bash_pattern.search(content)
        if not match:
            continue

        command = match.group(1).strip()

        # Check if this is a traditional edit command
        if is_edit_command(command):
            file_paths = extract_file_paths_from_bash_command(command)
            edit_files.update(file_paths)

        # Also check for redirection operations (>, >>)
        # Pattern: command > file or command >> file
        redirection_pattern = re.compile(r">>?\s+([^\s&|]+)")
        redirection_matches = redirection_pattern.findall(command)
        for redir_file in redirection_matches:
            # Clean up the file path (remove quotes, etc.)
            redir_file = redir_file.strip("'\"")
            # Check if it looks like a file path (has extension or contains /)
            if "/" in redir_file or any(
                redir_file.endswith(ext)
                for ext in [
                    ".py",
                    ".md",
                    ".txt",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".cfg",
                    ".js",
                    ".ts",
                    ".jsx",
                    ".tsx",
                    ".html",
                    ".css",
                    ".scss",
                    ".sass",
                    ".java",
                    ".c",
                    ".cpp",
                    ".h",
                    ".hpp",
                    ".go",
                    ".rs",
                    ".rb",
                    ".php",
                    ".sh",
                    ".bash",
                    ".zsh",
                    ".fish",
                    ".ps1",
                    ".bat",
                    ".cmd",
                    ".xml",
                    ".sql",
                    ".r",
                    ".R",
                    ".m",
                    ".swift",
                    ".kt",
                    ".scala",
                    ".vue",
                    ".svelte",
                    ".dart",
                    ".lua",
                    ".pl",
                    ".pm",
                    ".tcl",
                    ".ini",
                    ".conf",
                    ".config",
                    ".properties",
                    ".env",
                ]
            ):
                # Skip example paths
                if not (redir_file.startswith("/path/to/") or redir_file.startswith("/example/")):
                    edit_files.add(redir_file)

    return edit_files


@app.command()
def analysis(
    paths: list[Path] = typer.Argument(..., help="Trajectory file(s) or directory(ies) to analyze"),
    plot_output: Path = typer.Option(None, "--plot", "-p", help="Output path for pie chart"),
    file_analysis: bool = typer.Option(
        False,
        "--file-analysis",
        "-f",
        help="Perform file-level analysis (read vs edit operations)",
    ),
):
    """Analyze token usage by operation type (read, execute, edit, other).

    This command extracts bash commands from assistant messages, classifies them,
    and calculates their token contributions using an O(n²) complexity model.
    Can analyze individual files or entire directories.
    """
    all_traj_files = []

    # Collect all trajectory files
    for path in paths:
        if not path.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            continue

        traj_files = find_traj_files(path)
        if not traj_files:
            console.print(f"[yellow]Warning:[/yellow] No .traj.json files found in {path}")
            continue

        all_traj_files.extend(traj_files)

    if not all_traj_files:
        console.print("[red]No valid trajectory files found.[/red]")
        raise typer.Exit(1)

    # Calculate contributions for each file separately, then aggregate
    weighted_contributions = {"read": 0.0, "execute": 0.0, "edit": 0.0, "other": 0.0}
    unweighted_contributions = {"read": 0.0, "execute": 0.0, "edit": 0.0, "other": 0.0}
    total_commands = 0

    for traj_file in all_traj_files:
        try:
            commands = extract_bash_commands(traj_file)
            if not commands:
                continue

            total_commands += len(commands)

            # Calculate contributions for this file
            file_weighted = calculate_token_contributions(commands, weighted=True)
            file_unweighted = calculate_token_contributions(commands, weighted=False)

            # Aggregate to total contributions
            for op_type in ["read", "execute", "edit", "other"]:
                weighted_contributions[op_type] += file_weighted[op_type]
                unweighted_contributions[op_type] += file_unweighted[op_type]

        except Exception as e:
            console.print(f"[red]Error[/red] reading {traj_file}: {e}")
            continue

    if total_commands == 0:
        console.print("[yellow]No bash commands found in trajectory files.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]Found {total_commands} commands from {len(all_traj_files)} file(s)[/green]")

    # Display weighted results
    console.print("\n[bold]Weighted Token Contribution (O(n²) model):[/bold]")
    weighted_table = Table(
        title="Weighted Token Contribution by Operation Type",
        show_header=True,
        header_style="bold magenta",
    )
    weighted_table.add_column("Operation Type", style="cyan")
    weighted_table.add_column("Weighted Token Contribution", justify="right", style="yellow")
    weighted_table.add_column("Percentage", justify="right", style="green")

    total_weighted = sum(weighted_contributions.values())

    for op_type in ["read", "execute", "edit", "other"]:
        value = weighted_contributions[op_type]
        percentage = (value / total_weighted * 100) if total_weighted > 0 else 0
        weighted_table.add_row(op_type.capitalize(), format_number(int(value)), f"{percentage:.2f}%")

    weighted_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{format_number(int(total_weighted))}[/bold]",
        "[bold]100.00%[/bold]",
    )

    console.print(weighted_table)

    # Display unweighted results
    console.print("\n[bold]Unweighted Token Contribution:[/bold]")
    unweighted_table = Table(
        title="Unweighted Token Contribution by Operation Type",
        show_header=True,
        header_style="bold magenta",
    )
    unweighted_table.add_column("Operation Type", style="cyan")
    unweighted_table.add_column("Token Contribution", justify="right", style="yellow")
    unweighted_table.add_column("Percentage", justify="right", style="green")

    total_unweighted = sum(unweighted_contributions.values())

    for op_type in ["read", "execute", "edit", "other"]:
        value = unweighted_contributions[op_type]
        percentage = (value / total_unweighted * 100) if total_unweighted > 0 else 0
        unweighted_table.add_row(op_type.capitalize(), format_number(int(value)), f"{percentage:.2f}%")

    unweighted_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{format_number(int(total_unweighted))}[/bold]",
        "[bold]100.00%[/bold]",
    )

    console.print(unweighted_table)

    # Show command breakdown
    console.print("\n[bold]Command Breakdown:[/bold]")
    breakdown = {"read": 0, "execute": 0, "edit": 0, "other": 0}
    # Re-traverse files to count commands by type
    for traj_file in all_traj_files:
        try:
            commands = extract_bash_commands(traj_file)
            for _, command, _ in commands:
                op_type = classify_command(command)
                breakdown[op_type] += 1
        except Exception:
            continue

    for op_type in ["read", "execute", "edit", "other"]:
        count = breakdown[op_type]
        console.print(f"  {op_type.capitalize()}: {count} commands")

    # Generate pie charts
    if plot_output:
        # Weighted chart
        weighted_output = plot_output.parent / f"{plot_output.stem}_weighted{plot_output.suffix}"
        plot_operation_pie_chart(
            weighted_contributions,
            weighted_output,
            title="Weighted Token Contribution by Operation Type (O(n²) model)",
        )

        # Unweighted chart
        unweighted_output = plot_output.parent / f"{plot_output.stem}_unweighted{plot_output.suffix}"
        plot_operation_pie_chart(
            unweighted_contributions,
            unweighted_output,
            title="Unweighted Token Contribution by Operation Type",
        )

    # File-level analysis
    if file_analysis:
        console.print("\n[bold]File-Level Analysis:[/bold]")

        # Collect read operations: file_path -> list of line counts
        read_files = {}  # file_path -> [line_count1, line_count2, ...]
        edit_files = set()

        for traj_file in all_traj_files:
            try:
                # Extract read operations with outputs
                read_ops = extract_read_operations_with_output(traj_file)
                for command, output, file_paths in read_ops:
                    # Count lines in output
                    if output and output.strip():
                        output_stripped = output.rstrip()
                        if output_stripped:
                            # Count lines: split by newline and count
                            line_count = len(output_stripped.split("\\n"))
                        else:
                            line_count = 0
                    else:
                        line_count = 0

                    for file_path in file_paths:
                        if file_path not in read_files:
                            read_files[file_path] = []
                        read_files[file_path].append(line_count)

                # Extract edit operations
                edit_ops_files = extract_edit_operations(traj_file)
                edit_files.update(edit_ops_files)
            except Exception as e:
                console.print(f"[yellow]Warning[/yellow] processing {traj_file} for file analysis: {e}")
                continue

        if not read_files:
            console.print("[yellow]No files found in read operations.[/yellow]")
        else:
            # Calculate statistics
            read_files_set = set(read_files.keys())
            edit_files_set = edit_files

            # Files that were both read and edited
            read_and_edited = read_files_set & edit_files_set
            read_only = read_files_set - edit_files_set

            # Calculate proportions
            total_read = len(read_files_set)
            edited_after_read = len(read_and_edited)
            read_only_count = len(read_only)

            edit_ratio = (edited_after_read / total_read * 100) if total_read > 0 else 0
            read_only_ratio = (read_only_count / total_read * 100) if total_read > 0 else 0

            # Calculate average line count for read files
            # Sum all reads for each file, then average across files
            total_lines = 0
            files_with_lines = 0
            for file_path, line_counts in read_files.items():
                if line_counts:
                    # Sum line count for this file across all reads
                    total_lines_for_file = sum(line_counts)
                    total_lines += total_lines_for_file
                    files_with_lines += 1

            avg_lines = (total_lines / files_with_lines) if files_with_lines > 0 else 0

            # Display results
            file_table = Table(
                title="File Operation Statistics",
                show_header=True,
                header_style="bold magenta",
            )
            file_table.add_column("Metric", style="cyan")
            file_table.add_column("Value", justify="right", style="yellow")

            file_table.add_row("Total files read", format_number(total_read))
            file_table.add_row("Files edited after read", format_number(edited_after_read))
            file_table.add_row("Files read but not edited", format_number(read_only_count))
            file_table.add_row("Edit ratio (read → edit)", f"{edit_ratio:.2f}%")
            file_table.add_row("Read-only ratio", f"{read_only_ratio:.2f}%")
            file_table.add_row("Average lines per read file", format_number(int(avg_lines)))
            file_table.add_row("Files with valid line counts", format_number(files_with_lines))

            console.print(file_table)

            # Additional details
            if read_and_edited:
                console.print(f"\n[bold]Files that were read and then edited ({len(read_and_edited)}):[/bold]")
                for file_path in sorted(read_and_edited)[:20]:
                    total_lines_for_file = sum(read_files[file_path]) if file_path in read_files else 0
                    console.print(f"  {file_path} (total {total_lines_for_file} lines)")
                if len(read_and_edited) > 20:
                    console.print(f"  ... and {len(read_and_edited) - 20} more")

            if read_only:
                console.print(f"\n[bold]Files that were read but not edited ({len(read_only)}):[/bold]")
                for file_path in sorted(read_only)[:20]:
                    total_lines_for_file = sum(read_files[file_path]) if file_path in read_files else 0
                    console.print(f"  {file_path} (total {total_lines_for_file} lines)")
                if len(read_only) > 20:
                    console.print(f"  ... and {len(read_only) - 20} more")


if __name__ == "__main__":
    app()
