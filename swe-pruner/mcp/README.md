# SWE-Pruner MCP Server

A local MCP server that exposes a single `prune` tool.  Claude (or any MCP client) reads files with its own built-in tools, then sends the code here for neural pruning.

## Setup

### 1. Install

```bash
pip install swe-pruner          # installs both swe_pruner and swe_pruner_mcp
```

Or from source:

```bash
cd swe-pruner && pip install -e .
```

Flash-attn (GPU acceleration):

```bash
pip install flash-attn --no-build-isolation
```

### 2. Download the model (~1.3 GB)

```bash
huggingface-cli download ayanami-kitasan/code-pruner --local-dir ~/.swe-pruner/model
```

### 3. Verify

```bash
SWEPRUNER_MODEL_PATH=~/.swe-pruner/model python -m swe_pruner_mcp
```

The server starts on stdio and waits for MCP messages.  Press Ctrl-C to stop.

## Claude Code Configuration

Add to `~/.claude/settings.json` (global) or `.claude/settings.json` (per-project):

```json
{
  "mcpServers": {
    "swe-pruner": {
      "command": "python",
      "args": ["-m", "swe_pruner_mcp"],
      "env": {
        "SWEPRUNER_MODEL_PATH": "~/.swe-pruner/model"
      }
    }
  }
}
```

Claude Code will auto-discover the `prune` tool from this server.

## Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "swe-pruner": {
      "command": "python",
      "args": ["-m", "swe_pruner_mcp"],
      "env": {
        "SWEPRUNER_MODEL_PATH": "~/.swe-pruner/model"
      }
    }
  }
}
```

## Tool: `prune`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Task/question to focus pruning on |
| `code` | string | required | Source code to prune |
| `threshold` | float | 0.5 | Score threshold (0–1). Lower keeps more. |

**Returns** (JSON):

```json
{
  "pruned_code": "...",
  "score": 0.82,
  "origin_token_cnt": 1500,
  "left_token_cnt": 800,
  "tokens_saved": 700
}
```

## Programmatic Tool Calling (PTC)

With Claude's [Programmatic Tool Calling](https://www.anthropic.com/engineering/advanced-tool-use), you can make Claude read files and prune them **inside a code-execution sandbox** so that raw file contents never enter the LLM's context window.

### How it works

1. Mark the MCP prune tool with `allowed_callers: ["code_execution_20250825"]`
2. Claude writes Python in a code-execution block
3. Claude's own `Read` tool loads the file → content stays in sandbox
4. `prune()` filters the code → result stays in sandbox
5. Only the final `print()` enters Claude's context

```python
# Runs in Claude's code-execution sandbox
import json

# Step 1: Claude reads the file (content stays in sandbox, not in context)
content = Read("/src/services/auth.py")

# Step 2: Prune via MCP tool (result stays in sandbox)
result = json.loads(prune(
    query="How does JWT token validation work?",
    code=content,
    threshold=0.5
))

# Step 3: Only this output enters Claude's context window
print(result["pruned_code"])
```

### API configuration for PTC

```json
{
  "tools": [
    { "type": "code_execution_20250825", "name": "code_execution" },
    {
      "name": "mcp__swe-pruner__prune",
      "allowed_callers": ["code_execution_20250825"]
    }
  ]
}
```

This ensures pruning happens **before** LLM token consumption — no custom tool-pipeline abstraction needed.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | `export SWEPRUNER_MODEL_PATH=~/.swe-pruner/model` |
| Import error | `pip install swe-pruner` in the same Python env |
| CUDA OOM | Install `flash-attn` |
