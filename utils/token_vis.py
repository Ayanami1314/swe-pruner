"""
Streamlit visualization for token-level code compression scores.

Displays:
1. Left column: Original code
2. Middle column: Code with token scores as colors (adjustable threshold)
3. Right column: Code with only kept_frags lines
"""

import json
import os
import re
import tempfile
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# JSONL dataset format
# -----------------------------------------------------------------------------
#
# One JSON object per line with the following fields:
#
#   - query: str
#       Query / task description.
#   - code: str
#       Full original code (including newlines).
#   - kept_frags: List[int]
#       Line numbers to keep (1-based), i.e. lines to retain after compression (ground truth).
#   - score: float
#       Reference score (e.g. relevance-score).
#   - token_scores: List[Tuple[str, float]]
#       Use model-produced token_scores. Token-level scores, each item (token_str, score).
#       token_str is in tokenizer raw form and may contain:
#         - Ġ: space prefix → display as " "
#         - Ċ: newline marker → "\n"
#         - ĉ: indent marker → display as 4 spaces
#       score in [0, 1]; 1 = keep, 0 = remove.
#
# Example line:
#   {"query": "...", "code": "def f():\n  pass", "kept_frags": [1, 2], "score": 0.9, "token_scores": [["Ġdef", 0.8], ["Ċ", 0.5], ...]}
#

# Tokenizer markers to display characters (for decoding and content checks)
TOKEN_MARKER_SPACE = "Ġ"
TOKEN_MARKER_NEWLINE = "Ċ"
TOKEN_MARKER_INDENT = "ĉ"
REPLACEMENTS = (
    (TOKEN_MARKER_SPACE, " "),
    (TOKEN_MARKER_NEWLINE, "\n"),
    (TOKEN_MARKER_INDENT, "    "),
)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data with eval token scores from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# -----------------------------------------------------------------------------
# Score and color
# -----------------------------------------------------------------------------


def clamp_score(score: float) -> float:
    """Clamp score to [0, 1]."""
    return max(0.0, min(1.0, float(score)))


def score_to_color(score: float) -> str:
    """Convert score [0, 1] to RGB string: green=keep, red=remove."""
    r = int(255 * (1 - score))
    g = int(255 * score)
    b = 0
    return f"rgb({r}, {g}, {b})"


# -----------------------------------------------------------------------------
# Token string handling
# -----------------------------------------------------------------------------


def token_to_display(token_str: str) -> str:
    """Convert tokenizer raw token to display string (for content checks and display)."""
    s = token_str
    for old, new in REPLACEMENTS:
        s = s.replace(old, new)
    return s


def token_has_newline(token_str: str) -> bool:
    """Return whether the token contains a newline marker."""
    return TOKEN_MARKER_NEWLINE in token_str


def token_has_content(token_str: str) -> bool:
    """Return whether the token has non-whitespace content (for inclusion in line mean)."""
    return bool(token_to_display(token_str).strip())


# -----------------------------------------------------------------------------
# Line-level score computation
# -----------------------------------------------------------------------------


def compute_line_scores(
    code: str, token_scores: List[Tuple[str, float]]
) -> Dict[int, float]:
    """
    Compute mean token score per line based on actual newline tokens.

    token_scores must include "Ċ" newline markers.
    Returns: {line_number (1-based) -> mean score for that line}
    """
    if len(token_scores) == 0:
        lines = code.split("\n")
        return {i + 1: 0.5 for i in range(len(lines))}

    line_scores = {}
    current_line_num = 1
    current_line_scores = []

    for token_str, score in token_scores:
        score = clamp_score(score)
        has_newline = token_has_newline(token_str)

        if token_has_content(token_str):
            current_line_scores.append(score)

        if has_newline:
            if current_line_scores:
                line_scores[current_line_num] = float(np.mean(current_line_scores))
            else:
                line_scores[current_line_num] = 0.5
            current_line_num += 1
            current_line_scores = []

    if current_line_scores:
        line_scores[current_line_num] = float(np.mean(current_line_scores))
    elif current_line_num == 1:
        line_scores[current_line_num] = 0.5

    return line_scores


# -----------------------------------------------------------------------------
# HTML rendering: score-based coloring
# -----------------------------------------------------------------------------


def _escape_html(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _build_line_html(
    line_tokens_html: List[str], line_num: int, line_class: str = ""
) -> str:
    """Build a single line div from a list of token HTML fragments."""
    content = "".join(line_tokens_html) if line_tokens_html else "&nbsp;"
    cls = f" code-line {line_class}" if line_class.strip() else " code-line"
    return f'<div class="{cls.strip()}"><span class="line-num">{line_num:3d}</span>{content}</div>'


def render_code_with_scores(
    code: str, token_scores: List[Tuple[str, float]], threshold: float = 0.5
) -> str:
    """
    Render code as HTML with token-level coloring (text color only).

    Line mean token score is compared to threshold to decide if the line is dimmed (removed).
    Code is reconstructed from token order in token_scores.
    """
    line_scores_dict = compute_line_scores(code, token_scores)

    html_lines = []
    current_line_num = 1
    line_tokens_html = []

    for token_str, score in token_scores:
        score = clamp_score(score)
        has_newline = token_has_newline(token_str)
        clean_token = token_to_display(token_str)
        safe_token = _escape_html(clean_token)

        if clean_token.strip():
            text_color = score_to_color(score)
            token_html = f'<span class="token-scored" style="color: {text_color};" title="score: {score:.3f}">{safe_token}</span>'
        else:
            token_html = f'<span class="token-scored">{safe_token}</span>'
        line_tokens_html.append(token_html)

        if has_newline:
            line_mean_score = line_scores_dict.get(current_line_num, 0.5)
            line_class = "removed" if line_mean_score < threshold else ""
            html_line = _build_line_html(line_tokens_html, current_line_num, line_class)
            html_lines.append(html_line)
            line_tokens_html = []
            current_line_num += 1

    if line_tokens_html:
        line_mean_score = line_scores_dict.get(current_line_num, 0.5)
        line_class = "removed" if line_mean_score < threshold else ""
        html_line = _build_line_html(line_tokens_html, current_line_num, line_class)
        html_lines.append(html_line)

    return "\n".join(html_lines)


# -----------------------------------------------------------------------------
# Syntax highlighting
# -----------------------------------------------------------------------------

PYTHON_KEYWORDS = [
    "def",
    "class",
    "import",
    "from",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "as",
    "in",
    "and",
    "or",
    "not",
    "is",
    "None",
    "True",
    "False",
    "break",
    "continue",
    "pass",
    "raise",
    "yield",
    "lambda",
    "assert",
    "del",
    "global",
    "nonlocal",
]


def apply_simple_syntax_highlighting(code_line: str) -> str:
    """Apply simple syntax highlighting to a line of Python code."""
    result = code_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    result = re.sub(
        r'(["\'])(?:[^\\]|\\.)*?\1',
        lambda m: f'<span class="string">{m.group(0)}</span>',
        result,
    )
    result = re.sub(
        r"(#.*$)",
        lambda m: f'<span class="comment">{m.group(0)}</span>',
        result,
    )
    for kw in PYTHON_KEYWORDS:
        result = re.sub(
            rf"\b({kw})\b(?![^<]*>)",
            lambda m: f'<span class="keyword">{m.group(0)}</span>',
            result,
        )
    result = re.sub(
        r"\b(\d+\.?\d*)\b(?![^<]*>)",
        lambda m: f'<span class="number">{m.group(0)}</span>',
        result,
    )
    return result


# -----------------------------------------------------------------------------
# HTML rendering: original code / kept lines only
# -----------------------------------------------------------------------------


def render_original_code(code: str) -> str:
    """Render original code line-by-line as HTML with syntax highlighting."""
    lines = code.split("\n")
    html_lines = []
    for line_num, line in enumerate(lines, start=1):
        highlighted = apply_simple_syntax_highlighting(line)
        html_line = f'<div class="code-line"><span class="line-num">{line_num:3d}</span>{highlighted}</div>'
        html_lines.append(html_line)
    return "\n".join(html_lines)


def render_plain_code(code: str, kept_frags: List[int]) -> str:
    """Render only lines in kept_frags with syntax highlighting."""
    lines = code.split("\n")
    kept_set = set(kept_frags)
    html_lines = []
    for line_num, line in enumerate(lines, start=1):
        if line_num in kept_set:
            highlighted = apply_simple_syntax_highlighting(line)
            html_line = f'<div class="code-line kept"><span class="line-num">{line_num:3d}</span>{highlighted}</div>'
            html_lines.append(html_line)
    if not html_lines:
        return '<div class="code-line"><i>No lines kept</i></div>'
    return "\n".join(html_lines)


# -----------------------------------------------------------------------------
# Streamlit UI: styles and layout
# -----------------------------------------------------------------------------

STREAMLIT_CSS = """
<style>
.code-container {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.8;
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
    max-height: 700px;
    overflow-y: auto;
}
.code-line {
    display: block;
    white-space: pre;
    font-weight: 500;
    margin: 2px 0;
}
.code-line.removed {
    opacity: 0.25;
}
.code-line.kept {
    background-color: #e8f5e9;
    padding: 3px 5px;
    border-radius: 3px;
}
.line-num {
    color: #888;
    display: inline-block;
    width: 45px;
    text-align: right;
    margin-right: 15px;
    user-select: none;
    font-weight: 400;
}
.token-scored {
    display: inline;
    font-weight: 600;
    white-space: pre;
}
.keyword { color: #0000ff; font-weight: bold; }
.string { color: #a31515; }
.comment { color: #008000; font-style: italic; }
.function { color: #795e26; }
.number { color: #098658; }
.operator { color: #000000; }
</style>
"""


def _inject_css() -> None:
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)


def _load_uploaded_jsonl(uploaded_file) -> List[Dict[str, Any]]:
    """Load data from an uploaded JSONL file in Streamlit."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        return load_jsonl(tmp_path)
    finally:
        os.unlink(tmp_path)


def _render_sidebar(
    data: List[Dict[str, Any]],
) -> Tuple[int, float]:
    """Render sidebar: sample selection, metadata, threshold. Returns (sample_idx, threshold)."""
    st.sidebar.success(f"✅ Loaded {len(data)} samples")

    st.sidebar.header("🔍 Sample Selection")
    sample_idx = st.sidebar.slider(
        "Select sample index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )

    sample = data[sample_idx]
    query = sample.get("query", "")
    score = sample.get("score", 0.0)
    token_scores = sample.get("token_scores", [])
    kept_frags = sample.get("kept_frags", [])
    code = sample.get("code", "")

    st.sidebar.markdown("### 📊 Sample Info")
    st.sidebar.metric("Query", query[:50] + "..." if len(query) > 50 else query)
    st.sidebar.metric("Reference Score", f"{score:.4f}")
    st.sidebar.metric("# Code Lines", len(code.split("\n")))
    st.sidebar.metric("# Tokens", len(token_scores))
    st.sidebar.metric("Kept Lines", len(kept_frags))

    st.sidebar.header("⚙️ Compression Settings")
    threshold = st.sidebar.slider(
        "Token Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lines with mean token score below this threshold will be dimmed/removed",
    )
    return sample_idx, threshold


def _render_three_columns(
    code: str,
    token_scores: List[Tuple[str, float]],
    kept_frags: List[int],
    threshold: float,
) -> None:
    """Render main area: three columns (original code, score-colored code, kept lines only)."""
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.markdown("### 📄 Original Code")
        st.markdown(
            f'<div class="code-container">{render_original_code(code)}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 🎯 Token Scores (Threshold-adjusted)")
        st.caption(f"Threshold: {threshold:.2f} — Green: keep, Red: remove")
        st.markdown(
            f'<div class="code-container">{render_code_with_scores(code, token_scores, threshold)}</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("### ✅ Kept Lines (from kept_frags)")
        st.markdown(
            f'<div class="code-container">{render_plain_code(code, kept_frags)}</div>',
            unsafe_allow_html=True,
        )


def _render_footer(
    code: str,
    token_scores: List[Tuple[str, float]],
    kept_frags: List[int],
    threshold: float,
) -> None:
    """Render footer statistics."""
    st.divider()
    st.markdown("### 📈 Statistics")

    line_scores_dict = compute_line_scores(code, token_scores)
    mean_line_score = float(np.mean(list(line_scores_dict.values())))
    lines_above_threshold = sum(1 for s in line_scores_dict.values() if s >= threshold)
    compression_ratio = (
        lines_above_threshold / len(line_scores_dict) if line_scores_dict else 0
    )
    accuracy_match = len(kept_frags) == lines_above_threshold

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Mean Token Score", f"{mean_line_score:.3f}")
    with col_b:
        st.metric(
            "Lines Above Threshold",
            f"{lines_above_threshold}/{len(line_scores_dict)}",
        )
    with col_c:
        st.metric("Retention Ratio", f"{compression_ratio:.1%}")
    with col_d:
        st.metric(
            "Matches kept_frags?",
            "✅ Yes" if accuracy_match else "❌ No",
        )


def main() -> None:
    st.set_page_config(layout="wide", page_title="Token Score Visualization")
    st.title("🎨 Token-Level Code Compression Scores")
    _inject_css()

    st.sidebar.header("📁 Load Data")
    jsonl_file = st.sidebar.file_uploader(
        "Upload eval_with_token_scores.jsonl",
        type=["jsonl"],
        help="JSONL file containing query, code, kept_frags, and token_scores",
    )

    if jsonl_file is None:
        st.info("👈 Upload a JSONL file (eval_with_token_scores.jsonl) to get started.")
        return

    try:
        data = _load_uploaded_jsonl(jsonl_file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return

    if not data:
        st.error("❌ No data found in JSONL file.")
        return

    sample_idx, threshold = _render_sidebar(data)
    sample = data[sample_idx]
    query = sample.get("query", "")
    code = sample.get("code", "")
    kept_frags = sample.get("kept_frags", [])
    token_scores = sample.get("token_scores", [])

    st.markdown(f"**Sample #{sample_idx}** — Query: {query}")
    _render_three_columns(code, token_scores, kept_frags, threshold)
    _render_footer(code, token_scores, kept_frags, threshold)


if __name__ == "__main__":
    main()
