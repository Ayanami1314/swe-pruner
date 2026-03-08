"""
LLM-as-a-Judge: 使用 LLM 对问答答案进行评分

该脚本使用 GPT-5 Responses API 对候选答案进行评分，基于五个维度：
- Correctness (正确性)
- Completeness (完整性)
- Relevance (相关性)
- Clarity (清晰度)
- Reasoning (推理能力)
"""

import json
import os
import concurrent.futures
from typing import Dict, Any, Optional, List
from pathlib import Path

import typer
from openai import AzureOpenAI
from dotenv import load_dotenv
app = typer.Typer(help="LLM-as-a-Judge: 使用 LLM 对问答答案进行评分")

load_dotenv()
def get_eval_client() -> AzureOpenAI:
    print(os.getenv("EVAL_LLM_BASE_URL"))
    print(os.getenv("EVAL_LLM_API_VERSION"))
    print(os.getenv("EVAL_LLM_API_KEY"))
    return AzureOpenAI(
        azure_endpoint=os.getenv("EVAL_LLM_BASE_URL"),
        api_version=os.getenv("EVAL_LLM_API_VERSION"),
        api_key=os.getenv("EVAL_LLM_API_KEY"),
        default_headers={"X-TT-LOGID": "${your_logid}"},
    )


def score_answer(
    question: str,
    reference: str,
    candidate: str,
    eval_client: AzureOpenAI,
    model: str
) -> Optional[Dict[str, int]]:
    """
    对候选答案进行评分
    
    Args:
        question: 问题
        reference: 参考答案
        candidate: 候选答案
        eval_client: Azure OpenAI 客户端
        model: 模型名称
        
    Returns:
        包含五个维度评分的字典，如果评分失败则返回 None
    """
    prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
    Evaluation Criteria and Scoring Guidelines (each scored 1 to 10):
        1. Correctness:
            10 — Completely correct; core points and details are accurate with no ambiguity.
            8-9 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
            6-7 — Partially correct; some errors or omissions, but main points are generally accurate.
            4-5 — Several errors or ambiguities that affect understanding of the core information.
            2-3 — Many errors; misleading or fails to convey key information.
            1 — Serious errors; completely wrong or misleading.
        2. Completeness:
            10 — Covers all key points from the reference answer without omission.
            8-9 — Covers most key points; only minor non-critical information missing.
            6-7 — Missing several key points; content is somewhat incomplete.
            4-5 — Important information largely missing; content is one-sided.
            2-3 — Covers very little relevant information; seriously incomplete.
            1 — Covers almost no relevant information; completely incomplete.
        3. Relevance:
            10 — Content fully focused on the question topic; no irrelevant information.
            8-9 — Mostly focused; only minor irrelevant or peripheral information.
            6-7 — Generally on topic; some off-topic content but still relevant overall.
            4-5 — Topic not sufficiently focused; contains considerable off-topic content.
            2-3 — Content deviates from topic; includes excessive irrelevant information.
            1 — Majority of content irrelevant to the question.
        4. Clarity:
            10 — Fluent language; clear and precise expression; very easy to understand.
            8-9 — Mostly fluent; clear expression with minor unclear points.
            6-7 — Generally clear; some expressions slightly unclear or not concise.
            4-5 — Expression somewhat awkward; some ambiguity or lack of fluency.
            2-3 — Language obscure; sentences are not smooth; hinders understanding.
            1 — Expression confusing; very difficult to understand.
        5. Reasoning:
            10 — Reasoning is clear, logical, and well-structured; argumentation is excellent.
            8-9 — Reasoning is clear and logical; well-structured with solid argumentation.
            6-7 — Reasoning generally reasonable; mostly clear logic; minor jumps.
            4-5 — Reasoning is average; some logical jumps or organization issues.
            2-3 — Reasoning unclear; lacks logical order; difficult to follow.
            1 — No clear reasoning; logic is chaotic.

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,10], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-10>,
        "completeness": <1-10>,
        "relevance": <1-10>,
        "clarity": <1-10>,
        "reasoning": <1-10>
        }}

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON"""

    try:
        # Use GPT-5 Responses API for evaluation
        response = eval_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            extra_body={
                "thinking": {
                    "include_thoughts": False,
                    "budget_tokens": 1024
                }
            },
        )
        
        # Extract response text from Responses API format
        score_str = response.choices[0].message.content.strip()
        print(f"评分结果：{score_str}")
        
        try:
            # 清理可能的代码块标记
            if score_str.startswith("```json"):
                score_str = score_str[7:]  # 移除 ```json
            if score_str.endswith("```"):
                score_str = score_str[:-3]  # 移除 ```
            score_str = score_str.strip()
            
            # 解析JSON格式的小分
            scores = json.loads(score_str)
            # 验证所有维度都在1-10范围内
            required_keys = ["correctness", "completeness", "clarity", "relevance", "reasoning"]
            for key in required_keys:
                if key not in scores or not (1 <= scores[key] <= 10):
                    print(f"评分验证失败: {key} = {scores.get(key)}")
                    return None
            return scores
        except Exception as e:
            print(f"JSON解析失败: {e}")
            return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"评分出错: {e}")
        return None


def process_single_record(
    candidate_record: Dict[str, Any],
    reference_dict: Dict[str, str],
    eval_client: AzureOpenAI,
    model: str
) -> Optional[Dict[str, Any]]:
    """
    处理单个记录的函数，用于并行执行
    
    Args:
        candidate_record: 候选答案记录
        reference_dict: 参考答案字典
        eval_client: Azure OpenAI 客户端
        model: 模型名称
        
    Returns:
        包含评分结果的记录，如果处理失败则返回 None
    """
    try:
        question = candidate_record.get("question", "")
        candidate_answer = candidate_record.get("answer", "")
        
        # 从reference字典中获取对应问题的参考答案
        reference = reference_dict.get(question, "")
        
        if not reference:
            print(f"跳过记录: 缺少参考答案")
            return None
            
        if not candidate_answer or candidate_answer.strip() == "No answer found":
            print(f"跳过记录: 候选答案为空或'No answer found'")
            return None

        # 对候选答案进行评分
        scores = score_answer(question, reference, candidate_answer, eval_client, model)
        
        if scores is None:
            print(f"跳过记录: 评分失败")
            return None
        
        # 创建新的记录，格式类似于现有的评分文件
        result_record = {
            "question": question,
            "score": {
                "correctness": scores["correctness"],
                "completeness": scores["completeness"],
                "clarity": scores["clarity"],
                "relevance": scores["relevance"],
                "reasoning": scores["reasoning"]
            }
        }
        
        print(f"已评分问题: {question[:50]}... - 小分: {scores} - 总分: {sum(scores.values())}")
        return result_record
        
    except Exception as e:
        print(f"处理记录时出错: {e}")
        return None


def evaluate_jsonl_parallel(
    candidate_jsonl_path: str,
    reference_jsonl_path: str,
    output_jsonl_path: str,
    eval_client: AzureOpenAI,
    model: str,
    max_workers: int = 16
) -> None:
    """
    并行处理 JSONL 文件
    
    Args:
        candidate_jsonl_path: 候选答案 JSONL 文件路径
        reference_jsonl_path: 参考答案 JSONL 文件路径
        output_jsonl_path: 输出 JSONL 文件路径
        eval_client: Azure OpenAI 客户端
        model: 模型名称
        max_workers: 最大并行工作线程数
    """
    # 读取参考答案并构建字典
    reference_dict = {}
    with open(reference_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("answer", "")
                if question and answer:
                    reference_dict[question] = answer
            except Exception as e:
                print(f"[跳过] 无效参考答案JSON行: {e}")
                continue
    
    print(f"读取到 {len(reference_dict)} 条参考答案")
    
    # 读取候选答案记录
    candidate_records = []
    with open(candidate_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                candidate_records.append(record)
            except Exception as e:
                print(f"[跳过] 无效候选答案JSON行: {e}")
                continue
    
    print(f"总共读取到 {len(candidate_records)} 条候选答案记录，开始并行处理...")
    
    # 使用线程池并行处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_record = {
            executor.submit(process_single_record, record, reference_dict, eval_client, model): record
            for record in candidate_records
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"处理记录时出错: {e}")
    
    print(f"评分完成，共处理 {len(results)} 条记录，准备写入结果...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # 写入结果
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"结果已保存到: {output_jsonl_path}")


@app.command()
def evaluate(
    candidate_path: str = typer.Option(..., "--candidate", "-c", help="候选答案 JSONL 文件路径"),
    reference_path: str = typer.Option(..., "--reference", "-r", help="参考答案 JSONL 文件路径"),
    output_path: str = typer.Option(..., "--output", "-o", help="输出评分结果 JSONL 文件路径"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="评估模型名称（默认使用配置中的模型）"),
    max_workers: int = typer.Option(16, "--workers", "-w", help="最大并行工作线程数"),
) -> None:
    """
    评估单个候选答案文件
    """
    eval_client = get_eval_client()
    eval_model = model or os.getenv("EVAL_LLM_MODEL_NAME")
    
    if not os.path.exists(candidate_path):
        typer.echo(f"错误: 候选答案文件不存在: {candidate_path}", err=True)
        raise typer.Exit(1)
    
    if not os.path.exists(reference_path):
        typer.echo(f"错误: 参考答案文件不存在: {reference_path}", err=True)
        raise typer.Exit(1)
    
    evaluate_jsonl_parallel(
        candidate_path,
        reference_path,
        output_path,
        eval_client,
        eval_model,
        max_workers
    )


@app.command()
def batch(
    candidate_paths: str = typer.Option(..., "--candidates", "-c", help="候选答案 JSONL 文件路径，多个路径用逗号分隔"),
    reference_path: str = typer.Option(..., "--reference", "-r", help="参考答案 JSONL 文件路径"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="实验类型：pruner 或 baseline"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="输出目录（默认与候选答案文件同目录）"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="评估模型名称（默认使用配置中的模型）"),
    max_workers: int = typer.Option(48, "--workers", "-w", help="最大并行工作线程数"),
) -> None:
    """
    批量评估多个候选答案文件
    
    示例：
    python llm-as-judge.py batch -c "answer/full-glm/reflex.jsonl,answer/full-glm/streamlink.jsonl" -r answer/reference/reflex.jsonl -e pruner
    """
    eval_client = get_eval_client()
    eval_model = model or os.getenv("EVAL_LLM_MODEL_NAME")
    
    # 验证实验类型
    if experiment not in ["pruner", "baseline"]:
        typer.echo(f"错误: 实验类型必须是 'pruner' 或 'baseline'，当前为: {experiment}", err=True)
        raise typer.Exit(1)
    
    # 解析候选答案路径列表
    candidate_list = [p.strip() for p in candidate_paths.split(",") if p.strip()]
    
    if not candidate_list:
        typer.echo("错误: 至少需要提供一个候选答案文件路径", err=True)
        raise typer.Exit(1)
    
    # 检查参考答案文件
    if not os.path.exists(reference_path):
        typer.echo(f"错误: 参考答案文件不存在: {reference_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"使用模型: {eval_model}")
    typer.echo(f"实验类型: {experiment}")
    typer.echo(f"参考答案: {reference_path}")
    typer.echo(f"候选答案文件数: {len(candidate_list)}")
    
    # 处理每个候选答案文件
    for candidate_path in candidate_list:
        if not os.path.exists(candidate_path):
            typer.echo(f"跳过: 候选答案文件不存在: {candidate_path}", err=True)
            continue
        
        # 构建输出路径：在文件名中加上实验类型后缀
        # 例如：answer/full-glm/reflex.jsonl -> answer/full-glm/reflex_pruner_score.jsonl
        candidate_dir = os.path.dirname(candidate_path)
        candidate_basename = os.path.basename(candidate_path)
        # 移除 .jsonl 后缀
        if candidate_basename.endswith(".jsonl"):
            name_without_ext = candidate_basename[:-6]
        else:
            name_without_ext = candidate_basename
        
        # 构建输出文件名：{原文件名}_{experiment}_score.jsonl
        output_filename = f"{name_without_ext}_{experiment}_score.jsonl"
        
        if output_dir:
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = os.path.join(candidate_dir, output_filename)
        
        typer.echo(f"\n处理: {candidate_path}")
        typer.echo(f"输出: {output_path}")
        
        try:
            evaluate_jsonl_parallel(
                candidate_path,
                reference_path,
                output_path,
                eval_client,
                eval_model,
                max_workers
            )
            typer.echo(f"完成: {candidate_path}")
        except Exception as e:
            typer.echo(f"处理失败: {candidate_path} - {e}", err=True)
            continue


if __name__ == "__main__":
    app()
