# SWE-Pruner Integration Examples

This directory provides reference implementations demonstrating how to integrate SWE-Pruner into downstream coding agent frameworks. SWE-Pruner enables task-aware adaptive context pruning by dynamically selecting relevant code segments based on the agent's current focus, thereby reducing token consumption and latency while preserving critical implementation details.

## Integration Approach

SWE-Pruner operates as a service that prunes tool outputs (e.g., command outputs, file contents) based on a context focus question. The integration involves wrapping existing tools with pruning capabilities, where the agent can optionally provide a `context_focus_question` parameter to guide the pruning process. When provided, only code segments relevant to the question are retained; otherwise, the complete output is returned.

## Examples

### Claude Agent SDK Integration

The `claude_code_demo.py` demonstrates integration with the Claude Agent SDK using the Model Context Protocol (MCP). It shows how to:

1. Create pruned versions of standard tools (bash, cat) that accept an optional `context_focus_question` parameter
2. Configure an MCP server with pruned tools
3. Integrate with ClaudeSDKClient for agent interactions

The implementation automatically prunes tool outputs when the agent provides a context focus question, achieving significant token reduction without requiring changes to the agent's reasoning logic.

### OpenHands SDK Integration

The `openhands_demo.py` demonstrates integration with the OpenHands SDK framework. It shows how to:

1. Register pruned tool executors that support context-aware pruning
2. Switch between baseline and pruned tool implementations via environment configuration
3. Maintain compatibility with existing agent workflows

The OpenHands integration uses a synchronous pruning interface and provides detailed pruning metadata (token counts, scores) in tool observations.

## Prerequisites

Before running the examples, ensure:

1. The SWE-Pruner service is running (default: `http://localhost:8000/prune`)
2. Required API keys are set (e.g., `ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` for OpenHands)
3. Dependencies are installed via `uv` (see `pyproject.toml`)

## Usage

Each demo can be configured via environment variables:

- `PRUNER_URL`: Customize the pruner service endpoint (default: `http://localhost:8000/prune`)
- `EXPERIMENT_TYPE`: For OpenHands, set to `"pruner"` or `"baseline"` to toggle pruning

Refer to the individual demo files for detailed usage instructions and code comments.

