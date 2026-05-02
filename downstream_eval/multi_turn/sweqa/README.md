# SWE-QA Downstream Evaluation (Multi-Turn)

This repository runs **SWE-QA** (Software Engineering Question Answering) as a downstream evaluation for code-context pruning. It uses the [OpenHands](https://github.com/OpenHandsHub/openhands) SDK to drive an agent that explores codebases and answers questions, optionally with a **pruner** that reduces tool output before feeding it to the model.

---

## Table of Contents

- [Install & Configure](#install--configure)
- [Benchmark Introduction](#benchmark-introduction)
- [Run Pipeline](#run-pipeline)
- [Scoring](#scoring)
- [Statistics](#statistics)
- [Reproduce](#reproduce)

---

## Install & Configure

### 1. Install dependencies

```bash
uv sync
```

(Or use your preferred Python 3.12+ environment and install from `pyproject.toml`.)

### 2. Repositories (codebases)

You need the SWE-QA repo clones on disk. In our experiments we use `./swe-repos`.

- Clone the repos and check out the commits listed in **`repos.txt`** (one line per repo: `URL COMMIT`).
- Example format in `repos.txt`:

  ```
  https://github.com/astropy/astropy 0a041d3
  https://github.com/django/django 14fc2e9
  ...
  ```

- Place them so that each repo lives at `./swe-repos/<repo_name>` (e.g. `./swe-repos/reflex`). Repo name is the last path segment of the URL (e.g. `reflex` from `https://github.com/reflex-dev/reflex`).

### 3. Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `API_TYPE` | `openai` or `azure` |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | For OpenAI-compatible API (or Azure) |
| `AZURE_ENDPOINT`, `AZURE_API_VERSION`, `AZURE_API_KEY` | For Azure OpenAI |
| `OPENHANDS_MODEL_NAME` | Model used by the agent (e.g. `openai/claude-sonnet-4-5`) |
| `OPENHANDS_REPOS` | Comma-separated repo names to run (e.g. `reflex,streamlink,conan`) |
| `BASE_REPO_PATH` | Root directory of cloned repos (default: `./swe-repos`) |
| `QUESTIONS_PATH` | Directory containing `{repo_name}.jsonl` question files (default: `./questions`) |
| `ANSWER_OUTPUT_PATH` | Directory for answer JSONL outputs (e.g. `./answer-claude`) |
| `TRAJ_OUTPUT_PATH` | Directory for trajectory outputs (e.g. `./traj-claude`) |
| `EXPERIMENT_TYPE` | `baseline` (no pruning) or `pruner` (use pruner for Bash tool output) |
| `PRUNER_URL`, `PRUNE_THRESHOLD` | Used only when `EXPERIMENT_TYPE=pruner` |
| `EVAL_*` | Used by `llm-as-judge.py` for scoring (Azure OpenAI by default) |

See **`.env.example`** for the full list and examples.

---

## Benchmark Introduction

**SWE-QA** is a benchmark for evaluating question answering over software repositories. The agent is given natural-language questions about a codebase and can use a terminal (Bash) to explore the repo and produce an answer.

- Paper: [https://arxiv.org/abs/2509.14635](https://arxiv.org/abs/2509.14635)
- Dataset: [https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark)

Questions are expected in JSONL files under `QUESTIONS_PATH`, one file per repo: `{repo_name}.jsonl`. Each line is a JSON object with at least a `"question"` field.

---

## Run Pipeline

### Step 1: Run the agent

```bash
# Ensure .env is set (ANSWER_OUTPUT_PATH, TRAJ_OUTPUT_PATH, OPENHANDS_REPOS, etc.)
python openhands-qa/main.py
```

- **Outputs:**
  - **Answer files:** `{ANSWER_OUTPUT_PATH}/openhands/{repo_name}_answers.jsonl` — one JSON object per line with `question`, `answer`, `timestamp`, `time_cost`, `prompt_tokens`, `completion_tokens`, etc.
  - **Trajectory files:** `{TRAJ_OUTPUT_PATH}/{repo_name}/{repo_name}_q{idx}.traj.json` — full conversation and tool events; for pruner runs, `*_prune_details.json` is also written.

The agent runs with a step limit per question (default 50 via `MAX_ITERATION_PER_RUN`) and a time limit (`MAX_TIME_PER_QUESTION`, default 1800s). Resumable: already-answered questions (by `question` text) are skipped.

### Step 2: Fix empty answers (recommended)

Some runs leave `answer` empty due to parsing/API edge cases. Two fix scripts are applied in order:

**2a. Fill from trajectory FinishAction**

```bash
python openhands-qa/fix_empty_answers.py --traj-dir "$TRAJ_OUTPUT_PATH" --answer-dir "$ANSWER_OUTPUT_PATH/openhands"
```

- Reads trajectory files and fills empty (or error/timeout) answers using the **FinishAction** content from the trajectory. No extra LLM calls.

**2b. Resummarize remaining empty answers**

```bash
python openhands-qa/fix_empty_answer_resummary.py --traj-dir "$TRAJ_OUTPUT_PATH" --answer-dir "$ANSWER_OUTPUT_PATH/openhands"
```

- For answers that are still empty or error/timeout: loads message history from the trajectory and calls the configured LLM once to synthesize an answer, then appends that answer (and token counts) into the same answer JSONL. Use this when the agent hit the step limit (e.g. 50) without emitting a finish (e.g. loops).

You can use `--dry-run` on both scripts to preview changes without writing files.

---

## Scoring

**LLM-as-Judge** scores each answer against a reference using an LLM (e.g. GPT) on five dimensions (1–10): **Correctness**, **Completeness**, **Relevance**, **Clarity**, **Reasoning**.

### Single candidate file

```bash
python llm-as-judge.py evaluate \
  --candidate path/to/candidate_answers.jsonl \
  --reference path/to/reference_answers.jsonl \
  --output path/to/scores.jsonl
```

### Batch (multiple candidate files, one reference)

```bash
python llm-as-judge.py batch \
  --candidates "answer-claude/openhands/reflex_answers.jsonl,answer-claude/openhands/streamlink_answers.jsonl" \
  --reference path/to/reference.jsonl \
  --experiment pruner \
  [--output-dir path/to/scores]
```

- Output files are named like `{original_basename}_{experiment}_score.jsonl` (e.g. `reflex_answers_pruner_score.jsonl`).
- Evaluation uses `EVAL_LLM_BASE_URL`, `EVAL_LLM_API_KEY`, `EVAL_LLM_API_VERSION`, `EVAL_LLM_MODEL_NAME` from `.env`.

---

## Statistics

### Compare scores and token usage

**`stats.py`** provides:

- **Compare two score JSONL files** (e.g. pruner vs baseline):

  ```bash
  python stats.py compare --file1 score/baseline.jsonl --file2 score/pruner.jsonl [--output comparison.json] [--verbose]
  ```

  Uses the intersection of questions, filters out records with any dimension score equal to 1, then reports per-dimension and total score stats.

- **Token usage** from answer JSONL (prompt/completion/total, time cost):

  ```bash
  python stats.py token-stats --file answer-claude/openhands/reflex_answers.jsonl [--output token_stats.json]
  ```

- **Trajectory stats** (e.g. round counts per run):

  ```bash
  python stats.py traj-stats --folder traj-claude/reflex [--output traj_stats.json]
  ```

### Prune statistics (type distribution of pruned content)

**`prune_stats.py`** analyzes what kind of content was **removed** by the pruner (e.g. comments, imports, docstrings, code), using the `*_prune_details.json` files produced by pruner runs.

- **`ast` subcommand (recommended):** AST-based classification (requires `tree-sitter` and `tree-sitter-python`):

  ```bash
  python prune_stats.py ast [--model claude] [--output ast_stats.json]
  ```

- **`basic`:** String-based heuristics (no extra deps).
- **`llm`:** Optional LLM-based classification (needs Azure OpenAI env or `--api-key` / `--endpoint`).
- **`origin`:** Distribution of **original** (pre-prune) code; use `--diff` to compare origin vs kept and show keep rates.
- **`deps`:** Checks preservation of long-range structures (e.g. try/except, if/else) after pruning.

Example:

```bash
uv add tree-sitter tree-sitter-python  # if not already in pyproject
python prune_stats.py ast -o prune_ast.json
```

---

## Reproduce

We provide pre-generated trajectories and answers for reference:

Trajectories and answers are available upon request.

- **Trajectories:** `traj-full-claude/`, `traj-full-claude-pruner/`, `traj-full-glm/`, `traj-full-glm-pruner/`
- **Answers:** `answer-claude/`, `answer-claude-pruner/`, `answer-glm/`, `answer-glm-pruner/`

Use the same **Run** and **Scoring** steps above with these paths to reproduce or extend the reported results. Configure `.env` to match the paths (e.g. `ANSWER_OUTPUT_PATH`, `TRAJ_OUTPUT_PATH`) if you run the pipeline yourself.

---

## File Layout Summary

| Path | Purpose |
|------|--------|
| `openhands-qa/main.py` | Agent runner: loads questions, runs OpenHands with Bash (optional pruner), writes answers + trajectories |
| `openhands-qa/tool_utils.py` | Bash tool implementations: baseline and pruner-wrapped |
| `openhands-qa/fix_empty_answers.py` | Fill empty answers from trajectory FinishAction |
| `openhands-qa/fix_empty_answer_resummary.py` | Resummarize remaining empty answers from trajectory via LLM |
| `llm-as-judge.py` | Score answer JSONL vs reference (single or batch) |
| `stats.py` | Compare score files; token and trajectory stats |
| `prune_stats.py` | Analyze type distribution of pruned lines (basic / ast / llm / origin / deps) |
| `repos.txt` | Repo URLs and commits for cloning |
| `.env.example` | Template for required environment variables |
