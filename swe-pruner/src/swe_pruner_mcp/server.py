"""
SWE-Pruner MCP Server

A lightweight MCP server that exposes a single ``prune`` tool.
Claude (or any MCP client) decides how to obtain the code — e.g. via its
own Read / Bash tools — and then sends the code here for pruning.

Usage
-----
    python -m swe_pruner_mcp

Environment Variables
---------------------
    SWEPRUNER_MODEL_PATH   Path to the model directory.  Default: ./model
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging (stderr — stdout is reserved for the MCP stdio transport)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("swe-pruner-mcp")

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "swe-pruner",
    description=(
        "SWE-Pruner: Task-aware neural code pruning for coding agents. "
        "Dynamically selects code lines relevant to a query, "
        "achieving 23-54% token reduction."
    ),
)

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Return the global SwePrunerForCodePruning instance (lazy-loaded)."""
    global _model
    if _model is not None:
        return _model

    from swe_pruner.prune_wrapper import SwePrunerForCodePruning

    model_path = os.environ.get("SWEPRUNER_MODEL_PATH", "~/.swe-pruner/model")
    model_dir = Path(model_path).expanduser()

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            "Set SWEPRUNER_MODEL_PATH or download the model. "
            "See: https://huggingface.co/ayanami-kitasan/code-pruner"
        )

    required = ["config.json", "model.safetensors"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_path}: {missing}. "
            "Ensure the model is fully downloaded."
        )

    logger.info("Loading SWE-Pruner model from %s ...", model_dir)
    _model = SwePrunerForCodePruning.from_pretrained(str(model_dir))
    logger.info("SWE-Pruner model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Tool: prune
# ---------------------------------------------------------------------------
@mcp.tool()
def prune(query: str, code: str, threshold: float = 0.5) -> str:
    """Prune source code to keep only lines relevant to a query.

    Uses a 0.6B neural skimmer to score each line against the query and
    filter out irrelevant code, achieving great token reductions(especially for long code and good query).

    When to use:
      - Reading large source files — prune first, then reason on the result.
      - Prefer reading files fully and pruning, over grep/find-based exploration.
      - Pruned output may omit details. Pass line-numbered code (e.g. from
        `cat -n` or `nl -ba`) so you can locate lines, then use sed/head/tail
        without pruning to inspect surrounding context when needed.

    Query guidelines:
      The query MUST be a complete, self-contained question or sentence — not
      bare keywords or fragments. It should describe what you are looking for
      semantically. Do NOT include file-level metadata (filenames)
      in the query; the model only sees the code text itself.

      Good queries:
        - "Find where the authentication middleware validates JWT tokens"
        - "How does the ConnectionPool handle timeout and retry logic?"
        - "Locate the data preprocessing pipeline for training input"
        - "Given that issue '...' reports a race condition in cache invalidation,
           find the locking mechanism in the cache module"

      Bad queries:
        - "load_raw function"            (too vague, just keywords)
        - "lines 50-100 of data_loader"  (file-level info, not semantic)
        - "fix the bug"                  (too vague to guide pruning)

    Args:
        query: A clear, specific question describing what to look for in the code.
        code: The source code text to prune (plain text, optionally with line numbers).
        threshold: Score threshold (0.0-1.0) for keeping lines.
                   Lower values keep more lines. Default: 0.5

    Returns:
        JSON with: pruned_code, score, origin_token_cnt, left_token_cnt, tokens_saved.
    """
    from swe_pruner.prune_wrapper import PruneRequest

    model = _get_model()
    request = PruneRequest(query=query, code=code, threshold=threshold)
    response = model.prune(request)

    return json.dumps(
        {
            "pruned_code": response.pruned_code,
            "score": round(response.score, 4),
            "origin_token_cnt": response.origin_token_cnt,
            "left_token_cnt": response.left_token_cnt,
            "tokens_saved": response.origin_token_cnt - response.left_token_cnt,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Run the SWE-Pruner MCP server (stdio transport)."""
    logger.info("Starting SWE-Pruner MCP server ...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
