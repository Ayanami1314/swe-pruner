"""
统计和比较两个评分 JSONL 文件

该脚本读取两个包含评分结果的 JSONL 文件，取问题交集，
过滤掉分数中含有 1 的异常项，然后比较剩余项的分数。
同时支持统计单个 answer JSONL 文件的 token usage。
"""

import json
import typer
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict

app = typer.Typer(
    help="统计和比较两个评分 JSONL 文件，以及统计 answer 文件的 token usage"
)


def load_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 JSONL 文件并构建以问题为键的字典

    Args:
        file_path: JSONL 文件路径

    Returns:
        以问题为键的字典，值为包含完整记录的字典
    """
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
                    typer.echo(f"警告: 第 {line_num} 行缺少 question 字段", err=True)
            except json.JSONDecodeError as e:
                typer.echo(f"警告: 第 {line_num} 行 JSON 解析失败: {e}", err=True)
                continue
    return records


def has_score_one(score_dict: Dict[str, int]) -> bool:
    """
    检查分数字典中是否含有值为 1 的项

    Args:
        score_dict: 分数字典

    Returns:
        如果任何分数为 1 则返回 True，否则返回 False
    """
    if not score_dict:
        return False
    return any(value == 1 for value in score_dict.values())


def filter_abnormal_records(
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    过滤掉分数中含有 1 的异常项

    Args:
        records: 记录字典

    Returns:
        过滤后的记录字典
    """
    filtered = {}
    for question, record in records.items():
        score = record.get("score", {})
        if not has_score_one(score):
            filtered[question] = record
    return filtered


def calculate_total_score(score_dict: Dict[str, int]) -> int:
    """
    计算总分

    Args:
        score_dict: 分数字典

    Returns:
        所有分数的总和
    """
    return sum(score_dict.values()) if score_dict else 0


def compare_scores(
    file1_records: Dict[str, Dict[str, Any]], file2_records: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    比较两个文件的分数

    Args:
        file1_records: 第一个文件的记录
        file2_records: 第二个文件的记录

    Returns:
        (统计信息列表, 详细比较结果字典)
    """
    # 取问题交集
    common_questions = set(file1_records.keys()) & set(file2_records.keys())

    # 过滤异常项
    file1_filtered = {q: file1_records[q] for q in common_questions}
    file2_filtered = {q: file2_records[q] for q in common_questions}

    file1_filtered = filter_abnormal_records(file1_filtered)
    file2_filtered = filter_abnormal_records(file2_filtered)

    # 再次取交集（因为过滤后可能有些问题被移除了）
    final_questions = set(file1_filtered.keys()) & set(file2_filtered.keys())

    stats = []
    stats.append(f"文件1总记录数: {len(file1_records)}")
    stats.append(f"文件2总记录数: {len(file2_records)}")
    stats.append(f"问题交集数（过滤前）: {len(common_questions)}")
    stats.append(f"文件1过滤后记录数: {len(file1_filtered)}")
    stats.append(f"文件2过滤后记录数: {len(file2_filtered)}")
    stats.append(f"最终比较记录数: {len(final_questions)}")

    # 比较分数
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

    # 添加统计信息
    if file1_total_scores:
        stats.append(f"\n分数比较统计:")
        stats.append(
            f"  文件1平均总分: {sum(file1_total_scores) / len(file1_total_scores):.2f}"
        )
        stats.append(
            f"  文件2平均总分: {sum(file2_total_scores) / len(file2_total_scores):.2f}"
        )
        stats.append(f"  文件1获胜次数: {file1_wins}")
        stats.append(f"  文件2获胜次数: {file2_wins}")
        stats.append(f"  平局次数: {ties}")

        # 各维度统计
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
                    f"  {dim}: 文件1平均={avg1:.2f}, 文件2平均={avg2:.2f}, 差值={avg1 - avg2:.2f}"
                )

    return stats, comparison_results


@app.command()
def compare(
    file1: str = typer.Option(..., "--file1", "-f1", help="第一个评分 JSONL 文件路径"),
    file2: str = typer.Option(..., "--file2", "-f2", help="第二个评分 JSONL 文件路径"),
    output: str = typer.Option(
        None, "--output", "-o", help="输出详细比较结果到 JSON 文件（可选）"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细比较结果"),
) -> None:
    """
    比较两个评分 JSONL 文件

    示例:
    python stats.py compare -f1 score/file1.jsonl -f2 score/file2.jsonl
    """
    # 检查文件是否存在
    if not Path(file1).exists():
        typer.echo(f"错误: 文件不存在: {file1}", err=True)
        raise typer.Exit(1)

    if not Path(file2).exists():
        typer.echo(f"错误: 文件不存在: {file2}", err=True)
        raise typer.Exit(1)

    typer.echo(f"读取文件1: {file1}")
    file1_records = load_jsonl(file1)

    typer.echo(f"读取文件2: {file2}")
    file2_records = load_jsonl(file2)

    typer.echo("\n开始比较...")
    stats, comparison_results = compare_scores(file1_records, file2_records)

    # 显示统计信息
    typer.echo("\n" + "=" * 60)
    typer.echo("统计结果:")
    typer.echo("=" * 60)
    for stat in stats:
        typer.echo(stat)

    # 如果指定了输出文件，保存详细结果
    if output:
        output_data = {
            "file1": file1,
            "file2": file2,
            "statistics": stats,
            "comparisons": comparison_results,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\n详细比较结果已保存到: {output}")

    # 如果启用详细模式，显示部分比较结果
    if verbose and comparison_results:
        typer.echo("\n" + "=" * 60)
        typer.echo("详细比较结果（前10条）:")
        typer.echo("=" * 60)
        for i, (question, result) in enumerate(list(comparison_results.items())[:10]):
            typer.echo(f"\n问题 {i + 1}: {question[:80]}...")
            typer.echo(
                f"  文件1总分: {result['file1_total']}, 文件2总分: {result['file2_total']}, 差值: {result['difference']}"
            )
            typer.echo(f"  文件1分数: {result['file1_score']}")
            typer.echo(f"  文件2分数: {result['file2_score']}")


def load_trajectory_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    加载轨迹文件夹中的所有 .traj.json 文件

    Args:
        folder_path: 轨迹文件夹路径

    Returns:
        轨迹数据列表
    """
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
            typer.echo(f"警告: 文件 {traj_file.name} JSON 解析失败: {e}", err=True)
            continue
        except Exception as e:
            typer.echo(f"警告: 文件 {traj_file.name} 读取失败: {e}", err=True)
            continue

    return trajectories


def calculate_round_count(events: List[Dict[str, Any]]) -> int:
    """
    计算轮数（agent 或 user source 的 event 数量）

    Args:
        events: 事件列表

    Returns:
        轮数
    """
    count = 0
    for event in events:
        source = event.get("source")
        if source in ["agent", "user"]:
            count += 1
    return count


def calculate_trajectory_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算轨迹统计信息

    Args:
        trajectories: 轨迹数据列表

    Returns:
        统计信息字典
    """
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
    """
    加载 answer JSONL 文件

    Args:
        file_path: JSONL 文件路径

    Returns:
        记录列表
    """
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
                typer.echo(f"警告: 第 {line_num} 行 JSON 解析失败: {e}", err=True)
                continue
    return records


def calculate_token_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算 token usage 统计信息

    Args:
        records: 记录列表

    Returns:
        统计信息字典
    """
    total_records = len(records)

    # 初始化累计值
    total_token_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time_cost = 0

    # 统计有效记录数（有 token 信息的记录）
    valid_token_records = 0
    valid_time_records = 0

    # 用于计算最大值和最小值
    token_costs = []
    prompt_tokens_list = []
    completion_tokens_list = []
    time_costs = []

    for record in records:
        # 统计 token_cost
        token_cost = record.get("token_cost")
        if token_cost is not None:
            try:
                token_cost = float(token_cost)
                total_token_cost += token_cost
                token_costs.append(token_cost)
                valid_token_records += 1
            except (ValueError, TypeError):
                pass

        # 统计 prompt_tokens
        prompt_tokens = record.get("prompt_tokens")
        if prompt_tokens is not None:
            try:
                prompt_tokens = int(prompt_tokens)
                total_prompt_tokens += prompt_tokens
                prompt_tokens_list.append(prompt_tokens)
            except (ValueError, TypeError):
                pass

        # 统计 completion_tokens
        completion_tokens = record.get("completion_tokens")
        if completion_tokens is not None:
            try:
                completion_tokens = int(completion_tokens)
                total_completion_tokens += completion_tokens
                completion_tokens_list.append(completion_tokens)
            except (ValueError, TypeError):
                pass

        # 统计 time_cost
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

    # 计算平均值
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
    folder: str = typer.Option(..., "--folder", "-fd", help="轨迹文件夹路径"),
    output: str = typer.Option(
        None, "--output", "-o", help="输出统计结果到 JSON 文件（可选）"
    ),
) -> None:
    """
    统计轨迹文件夹中的轮数统计信息

    示例:
    python stats.py traj-stats -fd trajectories/
    """
    # 检查文件夹是否存在
    if not Path(folder).exists():
        typer.echo(f"错误: 文件夹不存在: {folder}", err=True)
        raise typer.Exit(1)

    typer.echo(f"读取轨迹文件夹: {folder}")
    trajectories = load_trajectory_files(folder)

    if not trajectories:
        typer.echo("错误: 文件夹中没有 .traj.json 文件", err=True)
        raise typer.Exit(1)

    typer.echo(f"共读取 {len(trajectories)} 个轨迹文件")
    typer.echo("\n开始统计轨迹数据...")

    stats = calculate_trajectory_stats(trajectories)

    # 显示统计信息
    typer.echo("\n" + "=" * 60)
    typer.echo("轨迹统计结果:")
    typer.echo("=" * 60)
    typer.echo(f"\n基础统计:")
    typer.echo(f"  总轨迹文件数: {stats['total_trajectories']}")
    typer.echo(f"  总轮数: {stats['total_rounds']}")
    typer.echo(f"  平均轮数: {stats['avg_rounds']:.2f}")
    typer.echo(f"  最小轮数: {stats['min_rounds']}")
    typer.echo(f"  最大轮数: {stats['max_rounds']}")

    # 如果指定了输出文件，保存统计结果
    if output:
        output_data = {"folder": folder, "statistics": stats}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\n统计结果已保存到: {output}")


@app.command()
def token_stats(
    file: str = typer.Option(..., "--file", "-f", help="answer JSONL 文件路径"),
    output: str = typer.Option(
        None, "--output", "-o", help="输出统计结果到 JSON 文件（可选）"
    ),
) -> None:
    """
    统计单个 answer JSONL 文件的 token usage

    示例:
    python stats.py token-stats -f answer/reflex.jsonl
    """
    # 检查文件是否存在
    if not Path(file).exists():
        typer.echo(f"错误: 文件不存在: {file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"读取文件: {file}")
    records = load_answer_jsonl(file)

    if not records:
        typer.echo("错误: 文件中没有有效记录", err=True)
        raise typer.Exit(1)

    typer.echo(f"共读取 {len(records)} 条记录")
    typer.echo("\n开始统计 token usage...")

    stats = calculate_token_stats(records)

    # 显示统计信息
    typer.echo("\n" + "=" * 60)
    typer.echo("Token Usage 统计结果:")
    typer.echo("=" * 60)
    typer.echo(f"\n记录统计:")
    typer.echo(f"  总记录数: {stats['total_records']}")
    typer.echo(f"  有效 token 记录数: {stats['valid_token_records']}")
    typer.echo(f"  有效时间记录数: {stats['valid_time_records']}")

    typer.echo(f"\nToken Cost 统计:")
    typer.echo(f"  总 token cost: {stats['total_token_cost']:,.2f}")
    if stats["valid_token_records"] > 0:
        typer.echo(f"  平均 token cost: {stats['avg_token_cost']:,.2f}")
        typer.echo(f"  最小 token cost: {stats['min_token_cost']:,.2f}")
        typer.echo(f"  最大 token cost: {stats['max_token_cost']:,.2f}")

    typer.echo(f"\nToken 数量统计:")
    typer.echo(f"  总 prompt tokens: {stats['total_prompt_tokens']:,}")
    typer.echo(f"  总 completion tokens: {stats['total_completion_tokens']:,}")
    typer.echo(f"  总 tokens: {stats['total_tokens']:,}")
    if stats["total_tokens"] > 0:
        typer.echo(f"  平均 prompt tokens: {stats['avg_prompt_tokens']:,.0f}")
        typer.echo(f"  平均 completion tokens: {stats['avg_completion_tokens']:,.0f}")
        typer.echo(f"  平均 tokens: {stats['avg_tokens']:,.0f}")
        typer.echo(f"  最小 prompt tokens: {stats['min_prompt_tokens']:,}")
        typer.echo(f"  最大 prompt tokens: {stats['max_prompt_tokens']:,}")
        typer.echo(f"  最小 completion tokens: {stats['min_completion_tokens']:,}")
        typer.echo(f"  最大 completion tokens: {stats['max_completion_tokens']:,}")

    typer.echo(f"\n时间统计:")
    typer.echo(f"  总时间成本: {stats['total_time_cost']:.2f} 秒")
    if stats["valid_time_records"] > 0:
        typer.echo(f"  平均时间成本: {stats['avg_time_cost']:.2f} 秒")
        typer.echo(f"  最小时间成本: {stats['min_time_cost']:.2f} 秒")
        typer.echo(f"  最大时间成本: {stats['max_time_cost']:.2f} 秒")
        typer.echo(f"  总时间: {stats['total_time_cost'] / 60:.2f} 分钟")

    # 如果指定了输出文件，保存统计结果
    if output:
        output_data = {"file": file, "statistics": stats}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\n统计结果已保存到: {output}")


if __name__ == "__main__":
    app()
