#!/usr/bin/env python3
"""Fix script: extract FinishAction content from trajectory files to fill empty answers."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def extract_finish_action_content(traj_file: str) -> Optional[str]:
    """Extract FinishAction content from a trajectory file."""
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            traj_data = json.load(f)

        events = traj_data.get("events", [])
        answer = None
        for event in reversed(events):
            if event.get("type") == "ActionEvent" and event.get("action") and event["action"].get("kind") == "FinishAction":
                answer = event["action"]["message"]
        return answer
    except Exception as e:
        print(f"Error reading trajectory file {traj_file}: {e}")
        return None


def find_traj_file(traj_dir: str, repo_name: str, question: str) -> Optional[str]:
    """Find the trajectory file corresponding to a given question and repo."""
    repo_traj_dir = os.path.join(traj_dir, repo_name)
    if not os.path.exists(repo_traj_dir):
        return None

    import hashlib
    question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]

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
    """Load answers from a jsonl file."""
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
    """Save answers to a jsonl file."""
    with open(answer_file, "w", encoding="utf-8") as f:
        for answer in answers:
            json.dump(answer, f, ensure_ascii=False)
            f.write("\n")


def fix_empty_answers(
    traj_dir: str,
    answer_dir: str,
    dry_run: bool = False,
):
    """Fix empty answers by extracting FinishAction content from trajectory files."""

    answer_files = []
    for filename in os.listdir(answer_dir):
        if filename.endswith("_answers.jsonl"):
            answer_files.append(os.path.join(answer_dir, filename))

    if not answer_files:
        print(f"No answer files found in {answer_dir}")
        return

    total_fixed = 0
    total_checked = 0

    for answer_file in answer_files:
        print(f"\nProcessing: {answer_file}")

        repo_name = os.path.basename(answer_file).replace("_answers.jsonl", "")

        answers = load_answers_from_jsonl(answer_file)
        print(f"  Loaded {len(answers)} answers")

        fixed_count = 0
        for answer in answers:
            total_checked += 1
            question = answer.get("question", "")
            current_answer = answer.get("answer", "")

            if not current_answer or current_answer.startswith("Error:") or current_answer.startswith("Timeout:"):
                traj_file = find_traj_file(traj_dir, repo_name, question)

                if traj_file:
                    finish_content = extract_finish_action_content(traj_file)
                    if finish_content:
                        old_answer = current_answer
                        answer["answer"] = finish_content
                        fixed_count += 1
                        total_fixed += 1

                        if not dry_run:
                            print(f"  Fixed: {question[:50]}...")
                            print(f"    Old answer: {old_answer[:100] if old_answer else '(empty)'}")
                            print(f"    New answer: {finish_content[:100]}...")
                        else:
                            print(f"  [DRY RUN] Would fix: {question[:50]}...")
                    else:
                        print(f"  No FinishAction found: {question[:50]}...")
                else:
                    print(f"  Trajectory file not found: {question[:50]}...")

        if fixed_count > 0 and not dry_run:
            save_answers_to_jsonl(answer_file, answers)
            print(f"  Saved {fixed_count} fixed answers to {answer_file}")

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Checked: {total_checked}")
    print(f"  Fixed: {total_fixed}")
    if dry_run:
        print(f"  [DRY RUN mode, no files modified]")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract FinishAction content from trajectory files to fill empty answers"
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="./trajectories",
        help="Trajectory files directory (default: ./trajectories)",
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="./answer/openhands",
        help="Answer files directory (default: ./answer/openhands)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: only show what would be fixed, do not modify files",
    )

    args = parser.parse_args()

    if not os.path.exists(args.traj_dir):
        print(f"Error: trajectory directory not found: {args.traj_dir}")
        return

    if not os.path.exists(args.answer_dir):
        print(f"Error: answer directory not found: {args.answer_dir}")
        return

    fix_empty_answers(args.traj_dir, args.answer_dir, args.dry_run)


if __name__ == "__main__":
    main()
