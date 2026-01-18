# dataset: Steefano/LCB

# prompt, repo, question, correct_letter(A,B,C,D), repo_text(context), prompt_goal, is_hard

import json
import os
import pandas as pd
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
# The LQA directory should be located in the project root directory.
project_root = script_dir.parent.parent
lqa_file = project_root / "LQA" / "32K.json"

# If 32K.json does not exist, try 32k.json.
if not lqa_file.exists():
    lqa_file = project_root / "LQA" / "32k.json"

if not lqa_file.exists():
    raise FileNotFoundError(f"找不到数据文件: {lqa_file}")

print(f"正在加载数据文件: {lqa_file}")

# Loading JSON data
with open(lqa_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始数据条数: {len(data)}")

# Validate data structure
required_keys = [
    "prompt",
    "repo",
    "question",
    "correct_letter",
    "repo_text",
    "prompt_goal",
    "is_hard",
]
valid_data = []

for i, item in enumerate(data):
    if not isinstance(item, dict):
        print(f"警告: 第 {i} 条数据不是字典格式，跳过")
        continue

    # Check required fields
    missing_keys = [key for key in required_keys if key not in item]
    if missing_keys:
        print(f"警告: 第 {i} 条数据缺少字段 {missing_keys}，跳过")
        continue

    # Verify if correct_letter is A, B, C, or D
    if item["correct_letter"] not in ["A", "B", "C", "D"]:
        print(
            f"警告: 第 {i} 条数据的correct_letter为 {item['correct_letter']}，不是有效的选项，跳过"
        )
        continue

    valid_data.append(item)

print(f"有效数据条数: {len(valid_data)}")

# Convert to DataFrame
df = pd.DataFrame(valid_data)

# Data cleaning
print("正在进行数据清洗...")

# Delete completely duplicate rows
before_dedup = len(df)
df = df.drop_duplicates()
print(f"删除重复数据: {before_dedup - len(df)} 条")

# Delete rows where the key field is empty.
before_dropna = len(df)
df = df.dropna(subset=required_keys)
print(f"删除包含空值的数据: {before_dropna - len(df)} 条")

# Reset Index
df = df.reset_index(drop=True)

print(f"最终数据条数: {len(df)}")

# Save as JSONL format (one JSON object per line).
output_file = project_root / "longcodeqa_32k.jsonl"
df.to_json(output_file, orient="records", lines=True, force_ascii=False)
print(f"数据已保存到: {output_file}")

# Print data statistics
print("\n数据统计:")
print(f"- 总记录数: {len(df)}")
