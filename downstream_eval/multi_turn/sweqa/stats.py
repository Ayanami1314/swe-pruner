import json
import typer
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict

app = typer.Typer(
    help="Compare two scored JSONL files and compute answer token usage stats"
)


def load_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                question = record.get("question", "")
                if question:
                    records[question] = record
                else:
                    typer.echo(f"Warning: line {line_num} missing question field", err=True)
            except json.JSONDecodeError as e:
                typer.echo(f"Warning: line {line_num} JSON parse failed: {e}", err=True)
                continue
    return records


def has_score_one(score_dict: Dict[str, int]) -> bool:
    if not score_dict:
        return False
    return any(value == 1 for value in score_dict.values())


def filter_abnormal_records(
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    filtered = {}
    for question, record in records.items():
        score = record.get("score", {})
        if not has_score_one(score):
            filtered[question] = record
    return filtered


def calculate_total_score(score_dict: Dict[str, int]) -> int:
    return sum(score_dict.values()) if score_dict else 0


def compare_scores(
    file1_records: Dict[str, Dict[str, Any]], file2_records: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    common_questions = set(file1_records.keys()) & set(file2_records.keys())

    file1_filtered = {q: file1_records[q] for q in common_questions}
    file2_filtered = {q: file2_records[q] for q in common_questions}

    file1_filtered = filter_abnormal_records(file1_filtered)
    file2_filtered = filter_abnormal_records(file2_filtered)

    final_questions = set(file1_filtered.keys()) & set(file2_filtered.keys())

    stats = []
    stats.append(f"File1 total records: {len(file1_records)}")
    stats.append(f"File2 total records: {len(file2_records)}")
    stats.append(f"Common questions (before filter): {len(common_questions)}")
    stats.append(f"File1 after filter: {len(file1_filtered)}")
    stats.append(f"File2 after filter: {len(file2_filtered)}")
    stats.append(f"Final comparison records: {len(final_questions)}")

    comparison_results = {}
    file1_wins = 0
    file2_wins = 0
    ties = 0

    file1_total_scores = []
    file2_total_scores = []

    for question in final_questions:
        score1 = file1_filtered[question].get("score", {})
        score2 = file2_filtered[question].get("score", {})

        total1 = calculate_total_score(score1)
        total2 = calculate_total_score(score2)

        file1_total_scores.append(total1)
        file2_total_scores.append(total2)

        comparison_results[question] = {
            "file1_score": score1,
            "file2_score": score2,
            "file1_total": total1,
            "file2_total": total2,
            "difference": total1 - total2,
        }

        if total1 > total2:
            file1_wins += 1
        elif total2 > total1:
            file2_wins += 1
        else:
            ties += 1

    if file1_total_scores:
        stats.append(f"\nScore comparison:")
        stats.append(
            f"  File1 avg total: {sum(file1_total_scores) / len(file1_total_scores):.2f}"
        )
        stats.append(
            f"  File2 avg total: {sum(file2_total_scores) / len(file2_total_scores):.2f}"
        )
        stats.append(f"  File1 wins: {file1_wins}")
        stats.append(f"  File2 wins: {file2_wins}")
        stats.append(f"  Ties: {ties}")

        dimensions = [
            "correctness",
            "completeness",
            "clarity",
            "relevance",
            "reasoning",
        ]
        for dim in dimensions:
            file1_dim_scores = [
                file1_filtered[q].get("score", {}).get(dim, 0) for q in final_questions
            ]
            file2_dim_scores = [
                file2_filtered[q].get("score", {}).get(dim, 0) for q in final_questions
            ]
            if file1_dim_scores and file2_dim_scores:
                avg1 = sum(file1_dim_scores) / len(file1_dim_scores)
                avg2 = sum(file2_dim_scores) / len(file2_dim_scores)
                stats.append(
                    f"  {dim}: file1 avg={avg1:.2f}, file2 avg={avg2:.2f}, diff={avg1 - avg2:.2f}"
                )

    return stats, comparison_results


@app.command()
def compare(
    file1: str = typer.Option(..., "--file1", "-f1", help="First scored JSONL file path"),
    file2: str = typer.Option(..., "--file2", "-f2", help="Second scored JSONL file path"),
    output: str = typer.Option(
        None, "--output", "-o", help="Save detailed comparison to JSON file (optional)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed comparison results"),
) -> None:
    """Compare two scored JSONL files.

    Example:
    python stats.py compare -f1 score/file1.jsonl -f2 score/file2.jsonl
    """
    if not Path(file1).exists():
        typer.echo(f"Error: file not found: {file1}", err=True)
        raise typer.Exit(1)

    if not Path(file2).exists():
        typer.echo(f"Error: file not found: {file2}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Reading file1: {file1}")
    file1_records = load_jsonl(file1)

    typer.echo(f"Reading file2: {file2}")
    file2_records = load_jsonl(file2)

    typer.echo("\nComparing...")
    stats, comparison_results = compare_scores(file1_records, file2_records)

    typer.echo("\n" + "=" * 60)
    typer.echo("Statistics:")
    typer.echo("=" * 60)
    for stat in stats:
        typer.echo(stat)

    if output:
        output_data = {
            "file1": file1,
            "file2": file2,
            "statistics": stats,
            "comparisons": comparison_results,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\nDetailed comparison saved to: {output}")

    if verbose and comparison_results:
        typer.echo("\n" + "=" * 60)
        typer.echo("Detailed comparison (first 10):")
        typer.echo("=" * 60)
        for i, (question, result) in enumerate(list(comparison_results.items())[:10]):
            typer.echo(f"\nQuestion {i + 1}: {question[:80]}...")
            typer.echo(
                f"  File1 total: {result['file1_total']}, File2 total: {result['file2_total']}, diff: {result['difference']}"
            )
            typer.echo(f"  File1 score: {result['file1_score']}")
            typer.echo(f"  File2 score: {result['file2_score']}")


def load_trajectory_files(folder_path: str) -> List[Dict[str, Any]]:
    trajectories = []
    folder = Path(folder_path)

    if not folder.exists():
        return trajectories

    for traj_file in folder.glob("*.traj.json"):
        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                trajectories.append(data)
        except json.JSONDecodeError as e:
            typer.echo(f"Warning: {traj_file.name} JSON parse failed: {e}", err=True)
            continue
        except Exception as e:
            typer.echo(f"Warning: {traj_file.name} read failed: {e}", err=True)
            continue

    return trajectories


def calculate_round_count(events: List[Dict[str, Any]]) -> int:
    count = 0
    for event in events:
        source = event.get("source")
        if source in ["agent", "user"]:
            count += 1
    return count


def calculate_trajectory_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trajectories:
        return {
            "total_trajectories": 0,
            "total_rounds": 0,
            "avg_rounds": 0,
            "min_rounds": 0,
            "max_rounds": 0,
        }

    round_counts = []
    total_rounds = 0

    for traj in trajectories:
        events = traj.get("events", [])
        round_count = calculate_round_count(events)
        round_counts.append(round_count)
        total_rounds += round_count

    stats = {
        "total_trajectories": len(trajectories),
        "total_rounds": total_rounds,
        "avg_rounds": total_rounds / len(trajectories) if trajectories else 0,
        "min_rounds": min(round_counts) if round_counts else 0,
        "max_rounds": max(round_counts) if round_counts else 0,
    }

    return stats


def load_answer_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                typer.echo(f"Warning: line {line_num} JSON parse failed: {e}", err=True)
                continue
    return records


def calculate_token_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_records = len(records)

    total_token_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time_cost = 0

    valid_token_records = 0
    valid_time_records = 0

    token_costs = []
    prompt_tokens_list = []
    completion_tokens_list = []
    time_costs = []

    for record in records:
        token_cost = record.get("token_cost")
        if token_cost is not None:
            try:
                token_cost = float(token_cost)
                total_token_cost += token_cost
                token_costs.append(token_cost)
                valid_token_records += 1
            except (ValueError, TypeError):
                pass

        prompt_tokens = record.get("prompt_tokens")
        if prompt_tokens is not None:
            try:
                prompt_tokens = int(prompt_tokens)
                total_prompt_tokens += prompt_tokens
                prompt_tokens_list.append(prompt_tokens)
            except (ValueError, TypeError):
                pass

        completion_tokens = record.get("completion_tokens")
        if completion_tokens is not None:
            try:
                completion_tokens = int(completion_tokens)
                total_completion_tokens += completion_tokens
                completion_tokens_list.append(completion_tokens)
            except (ValueError, TypeError):
                pass

        time_cost = record.get("time_cost")
        if time_cost is not None:
            try:
                time_cost = float(time_cost)
                total_time_cost += time_cost
                time_costs.append(time_cost)
                valid_time_records += 1
            except (ValueError, TypeError):
                pass

    stats = {
        "total_records": total_records,
        "valid_token_records": valid_token_records,
        "valid_time_records": valid_time_records,
        "total_token_cost": total_token_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_time_cost": total_time_cost,
    }

    if valid_token_records > 0:
        stats["avg_token_cost"] = total_token_cost / valid_token_records
        stats["min_token_cost"] = min(token_costs) if token_costs else 0
        stats["max_token_cost"] = max(token_costs) if token_costs else 0
    else:
        stats["avg_token_cost"] = 0
        stats["min_token_cost"] = 0
        stats["max_token_cost"] = 0

    if prompt_tokens_list:
        stats["avg_prompt_tokens"] = total_prompt_tokens / len(prompt_tokens_list)
        stats["min_prompt_tokens"] = min(prompt_tokens_list)
        stats["max_prompt_tokens"] = max(prompt_tokens_list)
    else:
        stats["avg_prompt_tokens"] = 0
        stats["min_prompt_tokens"] = 0
        stats["max_prompt_tokens"] = 0

    if completion_tokens_list:
        stats["avg_completion_tokens"] = total_completion_tokens / len(
            completion_tokens_list
        )
        stats["min_completion_tokens"] = min(completion_tokens_list)
        stats["max_completion_tokens"] = max(completion_tokens_list)
    else:
        stats["avg_completion_tokens"] = 0
        stats["min_completion_tokens"] = 0
        stats["max_completion_tokens"] = 0

    if stats["total_tokens"] > 0:
        stats["avg_tokens"] = (
            stats["total_tokens"] / len(prompt_tokens_list) if prompt_tokens_list else 0
        )
    else:
        stats["avg_tokens"] = 0

    if valid_time_records > 0:
        stats["avg_time_cost"] = total_time_cost / valid_time_records
        stats["min_time_cost"] = min(time_costs) if time_costs else 0
        stats["max_time_cost"] = max(time_costs) if time_costs else 0
    else:
        stats["avg_time_cost"] = 0
        stats["min_time_cost"] = 0
        stats["max_time_cost"] = 0

    return stats


@app.command()
def traj_stats(
    folder: str = typer.Option(..., "--folder", "-fd", help="Trajectory folder path"),
    output: str = typer.Option(
        None, "--output", "-o", help="Save stats to JSON file (optional)"
    ),
) -> None:
    """Compute round-count statistics from a trajectory folder.

    Example:
    python stats.py traj-stats -fd trajectories/
    """
    if not Path(folder).exists():
        typer.echo(f"Error: folder not found: {folder}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Reading trajectory folder: {folder}")
    trajectories = load_trajectory_files(folder)

    if not trajectories:
        typer.echo("Error: no .traj.json files in folder", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(trajectories)} trajectory files")
    typer.echo("\nComputing trajectory stats...")

    stats = calculate_trajectory_stats(trajectories)

    typer.echo("\n" + "=" * 60)
    typer.echo("Trajectory stats:")
    typer.echo("=" * 60)
    typer.echo(f"\nBasic stats:")
    typer.echo(f"  Total trajectory files: {stats['total_trajectories']}")
    typer.echo(f"  Total rounds: {stats['total_rounds']}")
    typer.echo(f"  Avg rounds: {stats['avg_rounds']:.2f}")
    typer.echo(f"  Min rounds: {stats['min_rounds']}")
    typer.echo(f"  Max rounds: {stats['max_rounds']}")

    if output:
        output_data = {"folder": folder, "statistics": stats}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\nStats saved to: {output}")


@app.command()
def token_stats(
    file: str = typer.Option(..., "--file", "-f", help="Answer JSONL file path"),
    output: str = typer.Option(
        None, "--output", "-o", help="Save stats to JSON file (optional)"
    ),
) -> None:
    """Compute token usage statistics from an answer JSONL file.

    Example:
    python stats.py token-stats -f answer/reflex.jsonl
    """
    if not Path(file).exists():
        typer.echo(f"Error: file not found: {file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Reading file: {file}")
    records = load_answer_jsonl(file)

    if not records:
        typer.echo("Error: no valid records in file", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(records)} records")
    typer.echo("\nComputing token usage...")

    stats = calculate_token_stats(records)

    typer.echo("\n" + "=" * 60)
    typer.echo("Token Usage Stats:")
    typer.echo("=" * 60)
    typer.echo(f"\nRecord counts:")
    typer.echo(f"  Total records: {stats['total_records']}")
    typer.echo(f"  Valid token records: {stats['valid_token_records']}")
    typer.echo(f"  Valid time records: {stats['valid_time_records']}")

    typer.echo(f"\nToken cost:")
    typer.echo(f"  Total token cost: {stats['total_token_cost']:,.2f}")
    if stats["valid_token_records"] > 0:
        typer.echo(f"  Avg token cost: {stats['avg_token_cost']:,.2f}")
        typer.echo(f"  Min token cost: {stats['min_token_cost']:,.2f}")
        typer.echo(f"  Max token cost: {stats['max_token_cost']:,.2f}")

    typer.echo(f"\nToken counts:")
    typer.echo(f"  Total prompt tokens: {stats['total_prompt_tokens']:,}")
    typer.echo(f"  Total completion tokens: {stats['total_completion_tokens']:,}")
    typer.echo(f"  Total tokens: {stats['total_tokens']:,}")
    if stats["total_tokens"] > 0:
        typer.echo(f"  Avg prompt tokens: {stats['avg_prompt_tokens']:,.0f}")
        typer.echo(f"  Avg completion tokens: {stats['avg_completion_tokens']:,.0f}")
        typer.echo(f"  Avg tokens: {stats['avg_tokens']:,.0f}")
        typer.echo(f"  Min prompt tokens: {stats['min_prompt_tokens']:,}")
        typer.echo(f"  Max prompt tokens: {stats['max_prompt_tokens']:,}")
        typer.echo(f"  Min completion tokens: {stats['min_completion_tokens']:,}")
        typer.echo(f"  Max completion tokens: {stats['max_completion_tokens']:,}")

    typer.echo(f"\nTime stats:")
    typer.echo(f"  Total time cost: {stats['total_time_cost']:.2f} s")
    if stats["valid_time_records"] > 0:
        typer.echo(f"  Avg time cost: {stats['avg_time_cost']:.2f} s")
        typer.echo(f"  Min time cost: {stats['min_time_cost']:.2f} s")
        typer.echo(f"  Max time cost: {stats['max_time_cost']:.2f} s")
        typer.echo(f"  Total time: {stats['total_time_cost'] / 60:.2f} min")

    if output:
        output_data = {"file": file, "statistics": stats}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\nStats saved to: {output}")


if __name__ == "__main__":
    app()
