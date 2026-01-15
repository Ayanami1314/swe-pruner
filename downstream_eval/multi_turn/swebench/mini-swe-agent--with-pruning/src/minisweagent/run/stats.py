#!/usr/bin/env python3

"""Statistics script for mini-SWE-agent trajectory files.
Extracts and displays token usage statistics from .traj.json files.
"""

import json
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
                
                calls_with_usage.append({
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                })

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


def format_number(num: int | float) -> str:
    """Format number with thousand separators."""
    return f"{num:,}"


def format_pruner_column_name(key: str) -> str:
    base = key.removesuffix("_token_cnt").replace("_", " ").title()
    return f"{base} Tokens"


def find_traj_files(path: Path) -> list[Path]:
    """Find all .traj.json files in a directory or return the file if it's a file."""
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.rglob("*.traj.json"))
    else:
        return []


@app.command()
def main(
    paths: list[Path] = typer.Argument(..., help="Trajectory file(s) or directory(ies) to analyze"),
    detailed: bool = typer.Option(False, "-d", "--detailed", help="Show detailed per-call statistics"),
    summary_only: bool = typer.Option(False, "-s", "--summary", help="Show only summary totals"),
    aggregate: bool = typer.Option(False, "-a", "--aggregate", help="Show aggregate statistics (mean, median, etc.) for multiple files"),
):
    """Display token usage statistics from mini-SWE-agent trajectory files.
    
    Can analyze individual files or entire directories (e.g., runs/pruner-v3-glm).
    """
    
    all_stats = []
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
        grand_total_prompt += stats["total_prompt_tokens"]
        grand_total_completion += stats["total_completion_tokens"]
        grand_total_tokens += stats["total_tokens"]
        grand_total_calls += stats["api_calls"]
        grand_total_cost += stats["instance_cost"]
    
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
                display_name = format_pruner_column_name(stat_name) if stat_name in pruner_keys else stat_name.replace("_", " ").title()
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
        console.print(f"  API calls: {format_number(grand_total_calls // num_files)}")
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


if __name__ == "__main__":
    app()
