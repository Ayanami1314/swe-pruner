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
    raise FileNotFoundError(f"Data file not found: {lqa_file}")

print(f"Loading data file: {lqa_file}")

# Loading JSON data
with open(lqa_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Raw data count: {len(data)}")

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
        print(f"Warning: item {i} is not a dict, skipping")
        continue

    # Check required fields
    missing_keys = [key for key in required_keys if key not in item]
    if missing_keys:
        print(f"Warning: item {i} missing fields {missing_keys}, skipping")
        continue

    # Verify if correct_letter is A, B, C, or D
    if item["correct_letter"] not in ["A", "B", "C", "D"]:
        print(
            f"Warning: item {i} correct_letter is {item['correct_letter']}, not a valid option, skipping"
        )
        continue

    valid_data.append(item)

print(f"Valid data count: {len(valid_data)}")

# Convert to DataFrame
df = pd.DataFrame(valid_data)

# Data cleaning
print("Cleaning data...")

# Delete completely duplicate rows
before_dedup = len(df)
df = df.drop_duplicates()
print(f"Removed duplicates: {before_dedup - len(df)}")

# Delete rows where the key field is empty.
before_dropna = len(df)
df = df.dropna(subset=required_keys)
print(f"Removed rows with null values: {before_dropna - len(df)}")

# Reset Index
df = df.reset_index(drop=True)

print(f"Final data count: {len(df)}")

# Save as JSONL format (one JSON object per line).
output_file = project_root / "longcodeqa_32k.jsonl"
df.to_json(output_file, orient="records", lines=True, force_ascii=False)
print(f"Data saved to: {output_file}")

# Print data statistics
print(f"\nData statistics:")
print(f"- Total records: {len(df)}")
