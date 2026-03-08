#!/usr/bin/env python3
"""
修复脚本：从轨迹文件中提取 FinishAction 的 content 来填充空的 answer

使用方法:
    python fix_empty_answers.py [--traj-dir TRAJ_DIR] [--answer-dir ANSWER_DIR] [--dry-run]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def extract_finish_action_content(traj_file: str) -> Optional[str]:
    """从轨迹文件中提取 FinishAction 的 content"""
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        
        events = traj_data.get("events", [])
        answer = None
        # 从后往前查找 FinishAction
        for event in reversed(events):
            if event.get("type") == "ActionEvent" and event.get("action") and event["action"].get("kind") == "FinishAction":
                answer = event["action"]["message"]
        return answer
    except Exception as e:
        print(f"读取轨迹文件 {traj_file} 时出错: {e}")
        return None


def find_traj_file(traj_dir: str, repo_name: str, question: str) -> Optional[str]:
    """根据问题和仓库名查找对应的轨迹文件"""
    repo_traj_dir = os.path.join(traj_dir, repo_name)
    if not os.path.exists(repo_traj_dir):
        return None
    
    # 计算问题的哈希值
    import hashlib
    question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    
    # 尝试查找匹配的轨迹文件
    # 格式: {repo_name}_q{question_idx}.traj.json
    for filename in os.listdir(repo_traj_dir):
        if filename.endswith(".traj.json") and repo_name in filename:
            traj_file = os.path.join(repo_traj_dir, filename)
            try:
                with open(traj_file, "r", encoding="utf-8") as f:
                    traj_data = json.load(f)
                if traj_data.get("question") == question:
                    return traj_file
            except:
                continue
    
    return None


def load_answers_from_jsonl(answer_file: str) -> List[Dict]:
    """从 jsonl 文件加载答案"""
    answers = []
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        answers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return answers


def save_answers_to_jsonl(answer_file: str, answers: List[Dict]):
    """保存答案到 jsonl 文件"""
    with open(answer_file, "w", encoding="utf-8") as f:
        for answer in answers:
            json.dump(answer, f, ensure_ascii=False)
            f.write("\n")


def fix_empty_answers(
    traj_dir: str,
    answer_dir: str,
    dry_run: bool = False,
):
    """修复空的答案"""
    
    # 查找所有答案文件
    answer_files = []
    for filename in os.listdir(answer_dir):
        if filename.endswith("_answers.jsonl"):
            answer_files.append(os.path.join(answer_dir, filename))
    
    if not answer_files:
        print(f"在 {answer_dir} 中未找到答案文件")
        return
    
    total_fixed = 0
    total_checked = 0
    
    for answer_file in answer_files:
        print(f"\n处理文件: {answer_file}")
        
        # 从文件名提取仓库名
        repo_name = os.path.basename(answer_file).replace("_answers.jsonl", "")
        
        # 加载答案
        answers = load_answers_from_jsonl(answer_file)
        print(f"  加载了 {len(answers)} 个答案")
        
        fixed_count = 0
        for answer in answers:
            total_checked += 1
            question = answer.get("question", "")
            current_answer = answer.get("answer", "")
            
            # 如果答案为空或只有错误信息
            if not current_answer or current_answer.startswith("Error:") or current_answer.startswith("Timeout:"):
                # 查找对应的轨迹文件
                traj_file = find_traj_file(traj_dir, repo_name, question)
                
                if traj_file:
                    finish_content = extract_finish_action_content(traj_file)
                    if finish_content:
                        old_answer = current_answer
                        answer["answer"] = finish_content
                        fixed_count += 1
                        total_fixed += 1
                        
                        if not dry_run:
                            print(f"  ✓ 修复: {question[:50]}...")
                            print(f"    旧答案: {old_answer[:100] if old_answer else '(空)'}")
                            print(f"    新答案: {finish_content[:100]}...")
                        else:
                            print(f"  [DRY RUN] 将修复: {question[:50]}...")
                    else:
                        print(f"  ✗ 未找到 FinishAction: {question[:50]}...")
                else:
                    print(f"  ✗ 未找到轨迹文件: {question[:50]}...")
        
        # 保存修复后的答案
        if fixed_count > 0 and not dry_run:
            save_answers_to_jsonl(answer_file, answers)
            print(f"  ✓ 已保存 {fixed_count} 个修复的答案到 {answer_file}")
    
    print(f"\n{'='*60}")
    print(f"修复完成!")
    print(f"  检查总数: {total_checked}")
    print(f"  修复数量: {total_fixed}")
    if dry_run:
        print(f"  [DRY RUN 模式，未实际修改文件]")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="从轨迹文件中提取 FinishAction 的 content 来填充空的 answer"
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="./trajectories",
        help="轨迹文件目录 (默认: ./trajectories)",
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="./answer/openhands",
        help="答案文件目录 (默认: ./answer/openhands)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式，只显示将要修复的内容，不实际修改文件",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.traj_dir):
        print(f"错误: 轨迹目录不存在: {args.traj_dir}")
        return
    
    if not os.path.exists(args.answer_dir):
        print(f"错误: 答案目录不存在: {args.answer_dir}")
        return
    
    fix_empty_answers(args.traj_dir, args.answer_dir, args.dry_run)


if __name__ == "__main__":
    main()
