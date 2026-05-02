"""
Prune Statistics - analyze type distribution of pruned lines

Classifies lines removed from prune_details.json into:
comments, empty lines, imports, docstrings, decorators, code, etc.

Analysis modes:
  - basic:  simple string-matching classification
  - ast:    precise tree-sitter AST classification
  - llm:    semantic classification via Azure OpenAI LLM
  - origin: baseline distribution of original code (before pruning) for comparison
"""

import json
import re
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich import box

app = typer.Typer(help="Analyze pruned code line type distribution (basic / ast / llm / origin)")
console = Console()

MODELS = ["claude", "glm"]
REPOS = ["conan", "reflex", "streamlink"]
BASE_DIR = Path(__file__).resolve().parent

CATEGORY_LABELS = [
    "comment",
    "empty",
    "import",
    "docstring",
    "decorator",
    "code",
]


# ──────────────────────────── data loading ────────────────────────────


def discover_prune_files(model: str, repo: str) -> list[Path]:
    """Discover all prune_details.json for a given model+repo."""
    folder = BASE_DIR / f"traj-full-{model}-pruner" / repo
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{repo}_q*_prune_details.json"))


def load_prune_details(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────── removed-line extraction ─────────────────────────

FILTER_PATTERN = re.compile(r"^\(filtered (\d+) lines?\)$")


def extract_removed_lines(original: str, pruned: str) -> list[str]:
    """
    Compare original_output and pruned_output to figure out which lines
    were removed.  The pruned text keeps surviving lines in-order and
    replaces removed blocks with '(filtered N lines)'.
    """
    orig_lines = original.split("\n")
    pruned_lines = pruned.split("\n")

    removed: list[str] = []
    oi = 0  # pointer into orig_lines

    for pl in pruned_lines:
        m = FILTER_PATTERN.match(pl.strip())
        if m:
            n = int(m.group(1))
            removed.extend(orig_lines[oi : oi + n])
            oi += n
        else:
            # advance orig pointer past the matching line
            # (pruned line should equal orig_lines[oi])
            oi += 1

    return removed


def extract_removed_line_indices(original: str, pruned: str) -> set[int]:
    """Return 0-based line indices that were removed."""
    orig_lines = original.split("\n")
    pruned_lines = pruned.split("\n")

    removed: set[int] = set()
    oi = 0

    for pl in pruned_lines:
        m = FILTER_PATTERN.match(pl.strip())
        if m:
            n = int(m.group(1))
            for i in range(oi, oi + n):
                removed.add(i)
            oi += n
        else:
            oi += 1

    return removed


def collect_all_removed_lines(model: str, repo: str) -> list[dict[str, Any]]:
    """
    Return a flat list of dicts:
      { "model", "repo", "question_idx", "op_idx",
        "removed_lines": [str, ...], "original_output": str }
    """
    results = []
    for path in discover_prune_files(model, repo):
        data = load_prune_details(path)
        q_idx = data.get("question_idx", "?")
        for op_i, op in enumerate(data.get("prune_operations_detailed", [])):
            orig = op.get("original_output", "")
            prun = op.get("pruned_output", "")
            if not orig or not prun:
                continue
            removed = extract_removed_lines(orig, prun)
            results.append(
                {
                    "model": model,
                    "repo": repo,
                    "question_idx": q_idx,
                    "op_idx": op_i,
                    "removed_lines": removed,
                    "original_output": orig,
                }
            )
    return results


# ───────────────────── basic string classifier ────────────────────────


def classify_line_basic(line: str) -> str:
    """Classify a single line using simple pattern matching."""
    stripped = line.strip()

    if stripped == "":
        return "empty"

    if stripped.startswith("#") and not stripped.startswith("#!"):
        return "comment"

    if stripped.startswith(("import ", "from ")) and "import" in stripped:
        return "import"

    if stripped.startswith("@"):
        return "decorator"

    if (
        stripped in ('"""', "'''")
        or stripped.startswith('"""')
        or stripped.startswith("'''")
    ):
        return "docstring"

    return "code"


def classify_lines_basic(lines: list[str]) -> dict[str, int]:
    """Classify a list of lines and return category counts."""
    counts: dict[str, int] = defaultdict(int)
    in_docstring = False
    docstring_char: str | None = None

    for line in lines:
        stripped = line.strip()

        if in_docstring:
            counts["docstring"] += 1
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                docstring_char = None
            continue

        if stripped.startswith('"""') or stripped.startswith("'''"):
            doc_ch = stripped[:3]
            counts["docstring"] += 1
            if stripped.count(doc_ch) == 1:
                in_docstring = True
                docstring_char = doc_ch
            continue

        counts[classify_line_basic(line)] += 1

    return dict(counts)


# ──────────────────── tree-sitter AST classifier ─────────────────────


def _get_ts_parser():
    """Lazy-load tree-sitter Python parser."""
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser


def _map_line_to_node_type(source: str) -> dict[int, str]:
    """
    Parse *source* with tree-sitter and build a mapping
    line_number (0-based) -> coarsened AST node type.
    """
    parser = _get_ts_parser()
    tree = parser.parse(bytes(source, "utf-8"))

    line_type: dict[int, str] = {}

    def _coarsen(node_type: str) -> str:
        if node_type in ("comment",):
            return "comment"
        if node_type in ("import_statement", "import_from_statement"):
            return "import"
        if node_type in ("decorator",):
            return "decorator"
        if node_type in ("expression_statement",):
            return "expression_statement"
        if node_type in ("string", "concatenated_string"):
            return "string"
        return "code"

    def _walk(node):
        if node.child_count == 0:
            coarsened = _coarsen(node.type)
            for ln in range(node.start_point[0], node.end_point[0] + 1):
                if ln not in line_type:
                    line_type[ln] = coarsened
        else:
            ntype = node.type
            if ntype == "comment":
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    line_type[ln] = "comment"
                return
            if ntype in ("import_statement", "import_from_statement"):
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    line_type[ln] = "import"
                return
            if ntype == "decorator":
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    line_type[ln] = "decorator"
                return
            if ntype == "expression_statement" and node.child_count == 1:
                child = node.children[0]
                if child.type == "string":
                    for ln in range(node.start_point[0], node.end_point[0] + 1):
                        line_type[ln] = "docstring"
                    return
            for child in node.children:
                _walk(child)

    _walk(tree.root_node)
    return line_type


def classify_lines_ast(
    removed_lines: list[str], original_output: str
) -> dict[str, int]:
    """
    Use tree-sitter to classify removed lines.
    We parse the *original_output* to build a line->type map,
    then look up each removed line by matching it back to the original.
    """
    orig_lines = original_output.split("\n")
    line_type_map = _map_line_to_node_type(original_output)

    counts: dict[str, int] = defaultdict(int)

    removed_set = list(removed_lines)
    oi = 0
    ri = 0
    while ri < len(removed_set) and oi < len(orig_lines):
        if orig_lines[oi] == removed_set[ri]:
            node_type = line_type_map.get(oi, None)
            if node_type is None:
                if orig_lines[oi].strip() == "":
                    node_type = "empty"
                else:
                    node_type = "code"
            counts[node_type] += 1
            ri += 1
        oi += 1

    return dict(counts)


# ──────────────── AST scope (structural position) analysis ────────────

SCOPE_LABELS = [
    "module_level",
    "class_body",
    "function_body",
]

SCOPE_STYLES = {
    "module_level": "bright_cyan",
    "class_body": "bright_magenta",
    "function_body": "bright_green",
}


def _map_line_to_scope(source: str) -> dict[int, str]:
    """
    Parse *source* with tree-sitter and build a mapping
    line_number (0-based) -> enclosing scope label.

    Scope priority (innermost wins):
      function_body  — inside function_definition / method
      class_body     — inside class_definition (but not inside a method)
      module_level   — top-level (direct child of module)
    """
    parser = _get_ts_parser()
    tree = parser.parse(bytes(source, "utf-8"))
    total_lines = len(source.split("\n"))

    line_scope: dict[int, str] = {ln: "module_level" for ln in range(total_lines)}

    def _fill_scope(node, scope: str):
        for ln in range(node.start_point[0], node.end_point[0] + 1):
            line_scope[ln] = scope

    def _walk(node, current_scope: str):
        for child in node.children:
            if child.type in ("function_definition",):
                _fill_scope(child, "function_body")
                _walk(child, "function_body")
            elif child.type in ("class_definition",):
                _fill_scope(child, "class_body")
                _walk(child, "class_body")
            else:
                _walk(child, current_scope)

    _walk(tree.root_node, "module_level")
    return line_scope


def analyze_scope(
    removed_lines: list[str], original_output: str
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """
    Classify removed lines by their enclosing AST scope.

    Returns:
        (scope_counts, scope_by_category)
        - scope_counts:      { "module_level": N, "class_body": N, ... }
        - scope_by_category: { "module_level": { "comment": N, ... }, ... }
    """
    orig_lines = original_output.split("\n")
    line_type_map = _map_line_to_node_type(original_output)
    line_scope_map = _map_line_to_scope(original_output)

    scope_counts: dict[str, int] = defaultdict(int)
    scope_by_cat: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in SCOPE_LABELS
    }

    removed_set = list(removed_lines)
    oi = 0
    ri = 0
    while ri < len(removed_set) and oi < len(orig_lines):
        if orig_lines[oi] == removed_set[ri]:
            node_type = line_type_map.get(oi, None)
            if node_type is None:
                node_type = "empty" if orig_lines[oi].strip() == "" else "code"
            scope = line_scope_map.get(oi, "module_level")
            scope_counts[scope] += 1
            scope_by_cat[scope][node_type] += 1
            ri += 1
        oi += 1

    return dict(scope_counts), {k: dict(v) for k, v in scope_by_cat.items()}


# ────────────── long-range dependency analysis ────────────────────────


def find_dependency_structures(source: str) -> list[dict[str, Any]]:
    """
    Find compound statements that create long-range structural dependencies.

    Returns a list of structures, each containing:
      - type:       e.g. "try_statement", "if_statement"
      - components: list of { name, line } for each keyword clause
      - start_line, end_line
    """
    parser = _get_ts_parser()
    tree = parser.parse(bytes(source, "utf-8"))
    structures: list[dict[str, Any]] = []

    def _walk(node):
        if node.type == "try_statement":
            comps = [{"name": "try", "line": node.start_point[0]}]
            for child in node.children:
                if child.type == "except_clause":
                    comps.append({"name": "except", "line": child.start_point[0]})
                elif child.type == "finally_clause":
                    comps.append({"name": "finally", "line": child.start_point[0]})
                elif child.type == "else_clause":
                    comps.append({"name": "else", "line": child.start_point[0]})
            if len(comps) > 1:
                structures.append(
                    {
                        "type": "try_statement",
                        "components": comps,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                    }
                )

        elif node.type == "if_statement":
            comps = [{"name": "if", "line": node.start_point[0]}]
            for child in node.children:
                if child.type == "elif_clause":
                    comps.append({"name": "elif", "line": child.start_point[0]})
                elif child.type == "else_clause":
                    comps.append({"name": "else", "line": child.start_point[0]})
            if len(comps) > 1:
                structures.append(
                    {
                        "type": "if_statement",
                        "components": comps,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                    }
                )

        elif node.type in ("for_statement", "while_statement"):
            kw = node.type.replace("_statement", "")
            comps = [{"name": kw, "line": node.start_point[0]}]
            for child in node.children:
                if child.type == "else_clause":
                    comps.append({"name": "else", "line": child.start_point[0]})
            if len(comps) > 1:
                structures.append(
                    {
                        "type": node.type,
                        "components": comps,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                    }
                )

        elif node.type == "decorated_definition":
            decos: list[dict[str, Any]] = []
            defn: dict[str, Any] | None = None
            for child in node.children:
                if child.type == "decorator":
                    decos.append({"name": "@decorator", "line": child.start_point[0]})
                elif child.type in ("function_definition", "class_definition"):
                    defn = {
                        "name": child.type.replace("_definition", " def"),
                        "line": child.start_point[0],
                    }
            if decos and defn:
                structures.append(
                    {
                        "type": "decorated_definition",
                        "components": decos + [defn],
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                    }
                )

        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return structures


def check_dependency_preservation(original: str, pruned: str) -> list[dict[str, Any]]:
    """
    For each dependency structure in *original*, check whether the pruned
    output preserves it, fully removes it, or breaks it.

    Status values:
      preserved    – all keyword lines survived
      fully_removed – all keyword lines removed
      broken       – some keyword lines kept, others removed
    """
    removed_idx = extract_removed_line_indices(original, pruned)
    structures = find_dependency_structures(original)
    orig_lines = original.split("\n")

    results: list[dict[str, Any]] = []
    for struct in structures:
        kw_lines = [c["line"] for c in struct["components"]]
        kept = [ln for ln in kw_lines if ln not in removed_idx]
        removed = [ln for ln in kw_lines if ln in removed_idx]

        if len(removed) == 0:
            status = "preserved"
        elif len(kept) == 0:
            status = "fully_removed"
        else:
            status = "broken"

        start, end = struct["start_line"], struct["end_line"]
        results.append(
            {
                "type": struct["type"],
                "components": struct["components"],
                "status": status,
                "kept_keywords": kept,
                "removed_keywords": removed,
                "start_line": start,
                "end_line": end,
                "original_snippet": "\n".join(orig_lines[start : end + 1]),
            }
        )

    return results


# ─────────── dependency analysis pretty-printing ──────────────────────

STATUS_STYLES = {
    "preserved": "green",
    "fully_removed": "dim",
    "broken": "bold red",
}


def _annotate_snippet(
    orig_lines: list[str],
    start: int,
    end: int,
    removed_idx: set[int],
    keyword_lines: set[int],
) -> Text:
    """Build a rich Text object with colour-coded kept/removed lines."""
    text = Text()
    for i in range(start, end + 1):
        is_removed = i in removed_idx
        is_keyword = i in keyword_lines
        lineno = f"{i + 1:>4}  "

        if is_removed and is_keyword:
            text.append(lineno, style="bold red")
            text.append(orig_lines[i] + "\n", style="bold red strike")
        elif is_removed:
            text.append(lineno, style="dim red")
            text.append(orig_lines[i] + "\n", style="dim strike")
        elif is_keyword:
            text.append(lineno, style="bold green")
            text.append(orig_lines[i] + "\n", style="green")
        else:
            text.append(lineno, style="dim")
            text.append(orig_lines[i] + "\n", style="white")
    return text


def print_deps_results(
    model_results: dict[str, dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
    n_examples: int,
    title: str,
):
    """Print dependency analysis summary and examples."""
    active_models = [m for m in MODELS if m in model_results]

    # ── summary table ──
    summary = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold bright_white",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        pad_edge=True,
    )
    summary.add_column("Model", style="bold", min_width=10)
    summary.add_column("Preserved", justify="right", style="green", min_width=14)
    summary.add_column("Fully Removed", justify="right", style="dim", min_width=14)
    summary.add_column("Broken", justify="right", style="bold red", min_width=14)
    summary.add_column(
        "Total", justify="right", style="bold bright_yellow", min_width=10
    )
    summary.add_column(
        "Integrity %", justify="right", style="bold bright_cyan", min_width=12
    )

    for m in active_models:
        r = model_results[m]
        preserved = r.get("preserved", 0)
        removed = r.get("fully_removed", 0)
        broken = r.get("broken", 0)
        total = preserved + removed + broken
        relevant = preserved + broken  # structures that weren't entirely removed
        integrity = f"{preserved / relevant * 100:.1f}%" if relevant else "N/A"
        summary.add_row(
            f"[bold]{m}[/bold]",
            f"{preserved:,}  [dim]({_pct(preserved, total)})[/dim]",
            f"{removed:,}  [dim]({_pct(removed, total)})[/dim]",
            f"{broken:,}  [dim]({_pct(broken, total)})[/dim]",
            f"[bold]{total:,}[/bold]",
            integrity,
        )

    console.print()
    console.print(summary)

    # ── examples ──
    for m in active_models:
        model_examples = examples.get(m, [])

        preserved_ex = [e for e in model_examples if e["status"] == "preserved"]
        broken_ex = [e for e in model_examples if e["status"] == "broken"]

        if preserved_ex:
            console.print()
            console.print(
                f"[bold green]✓ Preserved examples[/bold green] [dim]({m})[/dim]"
            )
            for ex in preserved_ex[:n_examples]:
                _print_dep_example(ex, m)

        if broken_ex:
            console.print()
            console.print(f"[bold red]✗ Broken examples[/bold red] [dim]({m})[/dim]")
            for ex in broken_ex[:n_examples]:
                _print_dep_example(ex, m)

    console.print()


def _print_dep_example(ex: dict[str, Any], model: str):
    """Print a single dependency example as a rich Panel."""
    status = ex["status"]
    stype = ex["type"]
    comps = ", ".join(c["name"] for c in ex["components"])
    keyword_kept = set(ex.get("kept_keywords", []))
    keyword_removed = set(ex.get("removed_keywords", []))
    keyword_lines = keyword_kept | keyword_removed
    all_removed_idx: set[int] = ex.get("removed_idx", set())

    orig_lines = ex["original_snippet"].split("\n")
    start = ex["start_line"]

    text = Text()
    for i, line in enumerate(orig_lines):
        abs_line = start + i
        lineno = f"{abs_line + 1:>4}  "
        is_removed = abs_line in all_removed_idx
        is_keyword = abs_line in keyword_lines

        if is_keyword and is_removed:
            text.append(lineno, style="bold red")
            text.append(line + "\n", style="bold red strike")
        elif is_keyword:
            text.append(lineno, style="bold green")
            text.append(line + "\n", style="green")
        elif is_removed:
            text.append(lineno, style="dim red")
            text.append(line + "\n", style="dim strike")
        else:
            text.append(lineno, style="dim")
            text.append(line + "\n", style="white")

    style_border = STATUS_STYLES.get(status, "white")
    file_path = ex.get("file_path", "?")
    subtitle = (
        f"[dim]{file_path}  |  lines {start + 1}–{ex['end_line'] + 1}"
        f"  |  q={ex.get('question_idx', '?')}  op={ex.get('op_idx', '?')}[/dim]"
    )
    panel = Panel(
        text,
        title=f"[bold]{stype}[/bold]  ({comps})",
        subtitle=subtitle,
        border_style=style_border,
        padding=(0, 1),
    )
    console.print(panel)


# ──────────────────────── LLM classifier ──────────────────────────────

LLM_SYSTEM_PROMPT = """\
You are a code analyst. Given a list of code lines that were pruned from source files, \
classify EACH line into exactly one of these categories:
  comment, empty, import, docstring, decorator, code

Reply with a JSON array of objects: [{"line": "<the line>", "category": "<category>"}]
Do NOT include any other text outside the JSON array.\
"""


def classify_lines_llm(
    removed_lines: list[str],
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str = "2024-12-01-preview",
    batch_size: int = 80,
) -> dict[str, int]:
    """Send removed lines to Azure OpenAI for classification."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    counts: dict[str, int] = defaultdict(int)
    valid_categories = set(CATEGORY_LABELS)

    for start in range(0, len(removed_lines), batch_size):
        batch = removed_lines[start : start + batch_size]
        numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(batch))
        user_msg = f"Classify each of the following {len(batch)} lines:\n\n{numbered}"

        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content or "[]"
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            items = json.loads(content)
            for item in items:
                cat = item.get("category", "code").lower()
                if cat not in valid_categories:
                    cat = "code"
                counts[cat] += 1
        except Exception as e:
            console.print(f"  [yellow]LLM warning[/yellow] batch {start}: {e}")
            for ln in batch:
                counts[classify_line_basic(ln)] += 1

    return dict(counts)


# ──────────────────────── pretty printing ─────────────────────────────

CATEGORY_STYLES = {
    "comment": "green",
    "empty": "dim",
    "import": "cyan",
    "docstring": "yellow",
    "decorator": "magenta",
    "code": "bold white",
}


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def _bar(n: int, max_n: int, width: int = 30) -> str:
    """Build a rich-markup horizontal bar."""
    if max_n == 0:
        return ""
    filled = round(n / max_n * width)
    return (
        f"[bright_cyan]{'█' * filled}[/bright_cyan][dim]{'░' * (width - filled)}[/dim]"
    )


def print_category_table(
    stats: dict[str, dict[str, int]],
    title: str,
):
    """
    stats: { model: { category: count } }
    One row per model.
    """
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold bright_white",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        pad_edge=True,
    )
    table.add_column("Model", style="bold", min_width=10)
    for cat in CATEGORY_LABELS:
        table.add_column(
            cat.capitalize(),
            justify="right",
            style=CATEGORY_STYLES.get(cat, "white"),
            min_width=14,
        )
    table.add_column("Total", justify="right", style="bold bright_yellow", min_width=10)

    active_models = [m for m in MODELS if m in stats]
    for model in active_models:
        cats = stats[model]
        total = sum(cats.values())
        cells = [f"[bold]{model}[/bold]"]
        for cat in CATEGORY_LABELS:
            n = cats.get(cat, 0)
            cells.append(f"{n:,}  [dim]({_pct(n, total)})[/dim]")
        cells.append(f"[bold]{total:,}[/bold]")
        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print()


def print_scope_tables(
    scope_stats: dict[str, dict[str, int]],
    scope_by_cat_stats: dict[str, dict[str, dict[str, int]]],
    title: str,
):
    """
    scope_stats:        { model: { scope_label: count } }
    scope_by_cat_stats: { model: { scope_label: { category: count } } }
    """
    active_models = [m for m in MODELS if m in scope_stats]

    for model in active_models:
        scopes = scope_stats[model]
        max_cnt = max(scopes.values()) if scopes else 0
        total_all = sum(scopes.values())

        # ── 1) scope histogram ──
        hist_table = Table(
            title=f"{title}  —  Scope Distribution  —  [bold]{model}[/bold]",
            box=box.ROUNDED,
            show_lines=False,
            header_style="bold bright_white",
            title_style="bold bright_cyan",
            border_style="bright_blue",
        )
        hist_table.add_column("Scope", style="bold", min_width=16)
        hist_table.add_column("Count", justify="right", min_width=8)
        hist_table.add_column("Bar", min_width=32)
        hist_table.add_column("%", justify="right", min_width=6)

        for lbl in SCOPE_LABELS:
            n = scopes.get(lbl, 0)
            hist_table.add_row(
                f"[{SCOPE_STYLES.get(lbl, 'white')}]{lbl}[/{SCOPE_STYLES.get(lbl, 'white')}]",
                f"{n:,}",
                _bar(n, max_cnt),
                _pct(n, total_all),
            )

        console.print()
        console.print(hist_table)

        # ── 2) scope × category cross-table ──
        cross = scope_by_cat_stats.get(model, {})
        cross_table = Table(
            title=f"Scope × Category  —  [bold]{model}[/bold]",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            header_style="bold bright_white",
            title_style="bright_cyan",
            border_style="dim",
        )
        cross_table.add_column("Scope", style="bold", min_width=16)
        for cat in CATEGORY_LABELS:
            cross_table.add_column(
                cat.capitalize(),
                justify="right",
                style=CATEGORY_STYLES.get(cat, "white"),
                min_width=9,
            )
        cross_table.add_column(
            "Total", justify="right", style="bold bright_yellow", min_width=8
        )

        for lbl in SCOPE_LABELS:
            cats = cross.get(lbl, {})
            row_total = sum(cats.values())
            if row_total == 0:
                continue
            cells = [
                f"[{SCOPE_STYLES.get(lbl, 'white')}]{lbl}[/{SCOPE_STYLES.get(lbl, 'white')}]"
            ]
            for cat in CATEGORY_LABELS:
                n = cats.get(cat, 0)
                cells.append(f"{n:,}" if n else "[dim]·[/dim]")
            cells.append(f"[bold]{row_total:,}[/bold]")
            cross_table.add_row(*cells)

        console.print()
        console.print(cross_table)

    console.print()


# ──────────────────────── subcommands ─────────────────────────────────


@app.command()
def basic(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter to specific model (claude / glm), default all"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save JSON results to file"
    ),
):
    """String-matching based classification stats"""
    models = [model] if model else MODELS
    tasks = [(m, r) for m in models for r in REPOS]

    stats: dict[str, dict[str, int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Analyzing (basic) ...", total=len(tasks))
        for m, r in tasks:
            progress.update(ptask, description=f"[bold blue]basic [dim]|[/dim] {m}/{r}")
            stats.setdefault(m, defaultdict(int))
            for op in collect_all_removed_lines(m, r):
                for cat, cnt in classify_lines_basic(op["removed_lines"]).items():
                    stats[m][cat] += cnt
            progress.advance(ptask)

    stats = {m: dict(v) for m, v in stats.items()}
    print_category_table(stats, "Basic String Analysis")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def ast(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter to specific model (claude / glm), default all"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save JSON results to file"
    ),
):
    """Tree-sitter AST based classification stats (including scope analysis)"""
    models = [model] if model else MODELS
    tasks = [(m, r) for m in models for r in REPOS]

    stats: dict[str, dict[str, int]] = {}
    scope_stats: dict[str, dict[str, int]] = {}
    scope_by_cat_stats: dict[str, dict[str, dict[str, int]]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Analyzing (AST) ...", total=len(tasks))
        for m, r in tasks:
            progress.update(ptask, description=f"[bold blue]ast [dim]|[/dim] {m}/{r}")
            stats.setdefault(m, defaultdict(int))
            scope_stats.setdefault(m, defaultdict(int))
            scope_by_cat_stats.setdefault(
                m, {s: defaultdict(int) for s in SCOPE_LABELS}
            )

            for op in collect_all_removed_lines(m, r):
                try:
                    cats = classify_lines_ast(
                        op["removed_lines"], op["original_output"]
                    )
                    sc_counts, sc_cross = analyze_scope(
                        op["removed_lines"], op["original_output"]
                    )
                except Exception as e:
                    console.print(
                        f"  [yellow]AST fallback[/yellow] q{op['question_idx']} "
                        f"op{op['op_idx']}: {e}"
                    )
                    cats = classify_lines_basic(op["removed_lines"])
                    sc_counts, sc_cross = {}, {}

                for cat, cnt in cats.items():
                    stats[m][cat] += cnt
                for s, cnt in sc_counts.items():
                    scope_stats[m][s] += cnt
                for s, cat_dict in sc_cross.items():
                    for cat, cnt in cat_dict.items():
                        scope_by_cat_stats[m][s][cat] += cnt

            progress.advance(ptask)

    stats = {m: dict(v) for m, v in stats.items()}
    scope_stats = {m: dict(v) for m, v in scope_stats.items()}
    scope_by_cat_stats = {
        m: {s: dict(c) for s, c in v.items()} for m, v in scope_by_cat_stats.items()
    }

    print_category_table(stats, "Tree-Sitter AST Analysis")
    print_scope_tables(scope_stats, scope_by_cat_stats, "Tree-Sitter AST Analysis")

    if output:
        output_data = {
            "category_stats": stats,
            "scope_stats": scope_stats,
            "scope_by_category_stats": scope_by_cat_stats,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def llm(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter to specific model (claude / glm), default all"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save JSON results to file"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="AZURE_OPENAI_API_KEY", help="Azure OpenAI API Key"
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI Endpoint URL",
    ),
    deployment: str = typer.Option(
        "gpt-4o", "--deployment", "-d", help="Azure OpenAI deployment name"
    ),
    api_version: str = typer.Option(
        "2024-12-01-preview", "--api-version", help="Azure OpenAI API version"
    ),
    batch_size: int = typer.Option(
        80, "--batch-size", "-b", help="Lines per LLM batch"
    ),
):
    """Azure OpenAI LLM based semantic classification stats"""
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] Provide API key via --api-key or AZURE_OPENAI_API_KEY"
        )
        raise typer.Exit(1)
    if not endpoint:
        console.print(
            "[bold red]Error:[/bold red] Provide endpoint via --endpoint or AZURE_OPENAI_ENDPOINT"
        )
        raise typer.Exit(1)

    models = [model] if model else MODELS
    tasks = [(m, r) for m in models for r in REPOS]

    stats: dict[str, dict[str, int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Analyzing (LLM) ...", total=len(tasks))
        for m, r in tasks:
            progress.update(ptask, description=f"[bold blue]llm [dim]|[/dim] {m}/{r}")
            stats.setdefault(m, defaultdict(int))
            all_removed: list[str] = []
            for op in collect_all_removed_lines(m, r):
                all_removed.extend(op["removed_lines"])

            console.print(
                f"  [dim]{m}/{r}[/dim] — {len(all_removed):,} removed lines, sending to LLM ..."
            )
            cats = classify_lines_llm(
                all_removed,
                api_key=api_key,
                endpoint=endpoint,
                deployment=deployment,
                api_version=api_version,
                batch_size=batch_size,
            )
            for cat, cnt in cats.items():
                stats[m][cat] += cnt
            progress.advance(ptask)

    stats = {m: dict(v) for m, v in stats.items()}
    print_category_table(stats, "LLM (Azure OpenAI) Analysis")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def deps(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter to specific model (claude / glm), default all"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save JSON results to file"
    ),
    n_examples: int = typer.Option(
        3, "--examples", "-n", help="Number of preserved/broken examples to print"
    ),
):
    """Analyze structural preservation of long-range dependencies (try/except, if/else, etc.)"""
    models = [model] if model else MODELS
    tasks = [(m, r) for m in models for r in REPOS]

    model_counts: dict[str, dict[str, int]] = {}
    model_examples: dict[str, list[dict[str, Any]]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Analyzing (deps) ...", total=len(tasks))
        for m, r in tasks:
            progress.update(ptask, description=f"[bold blue]deps [dim]|[/dim] {m}/{r}")
            model_counts.setdefault(m, defaultdict(int))
            model_examples.setdefault(m, [])

            for path in discover_prune_files(m, r):
                data = load_prune_details(path)
                q_idx = data.get("question_idx", "?")
                for op_i, op in enumerate(data.get("prune_operations_detailed", [])):
                    orig = op.get("original_output", "")
                    prun = op.get("pruned_output", "")
                    if not orig or not prun:
                        continue
                    try:
                        results = check_dependency_preservation(orig, prun)
                    except Exception:
                        continue
                    removed_idx = extract_removed_line_indices(orig, prun)
                    rel_path = str(path.relative_to(BASE_DIR))
                    for res in results:
                        model_counts[m][res["status"]] += 1
                        res["question_idx"] = q_idx
                        res["op_idx"] = op_i
                        res["removed_idx"] = removed_idx
                        res["file_path"] = rel_path
                        model_examples[m].append(res)

            progress.advance(ptask)

    model_counts = {m: dict(v) for m, v in model_counts.items()}
    print_deps_results(
        model_counts, model_examples, n_examples, "Long-Range Dependency Analysis"
    )

    if output:
        serialisable = {
            m: {
                "counts": model_counts.get(m, {}),
                "broken_details": [
                    {
                        "type": e["type"],
                        "status": e["status"],
                        "components": e["components"],
                        "question_idx": e.get("question_idx"),
                        "file_path": e.get("file_path"),
                        "original_snippet": e["original_snippet"],
                    }
                    for e in model_examples.get(m, [])
                    if e["status"] == "broken"
                ],
            }
            for m in models
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


def _kept_pct(kept: int, total: int) -> str:
    """Kept rate: what percentage of original lines were kept (not pruned away)."""
    return f"{kept / total * 100:.1f}%" if total else "—"


def print_diff_category_table(
    orig: dict[str, dict[str, int]],
    kept: dict[str, dict[str, int]],
    title: str,
):
    """
    Show origin vs kept side-by-side with kept rate per category.
    orig: original counts; kept: counts from pruned_output.
    Display: orig → kept, label "kept X%"
    """
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold bright_white",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        pad_edge=True,
    )
    table.add_column("Model", style="bold", min_width=8)
    for cat in CATEGORY_LABELS:
        table.add_column(
            cat.capitalize(),
            justify="right",
            style=CATEGORY_STYLES.get(cat, "white"),
            min_width=22,
        )
    table.add_column("Total", justify="right", style="bold bright_yellow", min_width=22)

    active_models = [m for m in MODELS if m in orig]
    for model in active_models:
        o = orig[model]
        k = kept.get(model, {})
        o_total = sum(o.values())
        k_total = sum(k.values())
        cells = [f"[bold]{model}[/bold]"]
        for cat in CATEGORY_LABELS:
            o_n = o.get(cat, 0)
            k_n = k.get(cat, 0)
            rate = _kept_pct(k_n, o_n)
            cells.append(f"{o_n:,} → {k_n:,}\n[dim]kept {rate}[/dim]")
        rate_total = _kept_pct(k_total, o_total)
        cells.append(
            f"[bold]{o_total:,} → {k_total:,}[/bold]\n[dim]kept {rate_total}[/dim]"
        )
        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print()


def print_diff_scope_table(
    orig_scope: dict[str, dict[str, int]],
    kept_scope: dict[str, dict[str, int]],
    title: str,
):
    """
    Show origin vs kept scope distribution side-by-side.
    kept_scope: counts from pruned_output.
    Display: orig → kept, label "kept X%"
    """
    active_models = [m for m in MODELS if m in orig_scope]

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold bright_white",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        pad_edge=True,
    )
    table.add_column("Model", style="bold", min_width=8)
    for lbl in SCOPE_LABELS:
        table.add_column(
            lbl,
            justify="right",
            style=SCOPE_STYLES.get(lbl, "white"),
            min_width=22,
        )
    table.add_column("Total", justify="right", style="bold bright_yellow", min_width=22)

    for model in active_models:
        o = orig_scope[model]
        k = kept_scope.get(model, {})
        o_total = sum(o.values())
        k_total = sum(k.values())
        cells = [f"[bold]{model}[/bold]"]
        for lbl in SCOPE_LABELS:
            o_n = o.get(lbl, 0)
            k_n = k.get(lbl, 0)
            rate = _kept_pct(k_n, o_n)
            cells.append(f"{o_n:,} → {k_n:,}\n[dim]kept {rate}[/dim]")
        rate_total = _kept_pct(k_total, o_total)
        cells.append(
            f"[bold]{o_total:,} → {k_total:,}[/bold]\n[dim]kept {rate_total}[/dim]"
        )
        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print()


def print_diff_scope_by_cat_tables(
    orig_cross: dict[str, dict[str, dict[str, int]]],
    kept_cross: dict[str, dict[str, dict[str, int]]],
    title: str,
):
    """
    Per-model scope × category cross-table showing origin → kept + rate.
    kept_cross: counts from pruned_output.
    """
    active_models = [m for m in MODELS if m in orig_cross]

    for model in active_models:
        o_cross = orig_cross[model]
        k_cross = kept_cross.get(model, {})

        cross_table = Table(
            title=f"{title}  —  Scope × Category  —  [bold]{model}[/bold]",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold bright_white",
            title_style="bright_cyan",
            border_style="bright_blue",
            pad_edge=True,
        )
        cross_table.add_column("Scope", style="bold", min_width=16)
        for cat in CATEGORY_LABELS:
            cross_table.add_column(
                cat.capitalize(),
                justify="right",
                style=CATEGORY_STYLES.get(cat, "white"),
                min_width=16,
            )
        cross_table.add_column(
            "Total", justify="right", style="bold bright_yellow", min_width=16
        )

        for lbl in SCOPE_LABELS:
            o_cats = o_cross.get(lbl, {})
            k_cats = k_cross.get(lbl, {})
            o_row_total = sum(o_cats.values())
            k_row_total = sum(k_cats.values())
            if o_row_total == 0:
                continue
            cells = [
                f"[{SCOPE_STYLES.get(lbl, 'white')}]{lbl}[/{SCOPE_STYLES.get(lbl, 'white')}]"
            ]
            for cat in CATEGORY_LABELS:
                o_n = o_cats.get(cat, 0)
                k_n = k_cats.get(cat, 0)
                if o_n == 0 and k_n == 0:
                    cells.append("[dim]·[/dim]")
                else:
                    rate = _kept_pct(k_n, o_n)
                    cells.append(f"{o_n:,}→{k_n:,}\n[dim]kept {rate}[/dim]")
            rate_total = _kept_pct(k_row_total, o_row_total)
            cells.append(
                f"[bold]{o_row_total:,}→{k_row_total:,}[/bold]\n[dim]kept {rate_total}[/dim]"
            )
            cross_table.add_row(*cells)

        console.print()
        console.print(cross_table)

    console.print()


@app.command()
def origin(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter to specific model (claude / glm), default all"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save JSON results to file"
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        "-d",
        help="Also compute pruned stats and show origin → pruned comparison with retention rate",
    ),
):
    """Analyze baseline distribution of original code (before pruning) for comparison"""
    models = [model] if model else MODELS
    tasks = [(m, r) for m in models for r in REPOS]

    stats: dict[str, dict[str, int]] = {}
    scope_stats: dict[str, dict[str, int]] = {}
    scope_by_cat_stats: dict[str, dict[str, dict[str, int]]] = {}

    pruned_stats: dict[str, dict[str, int]] = {}
    pruned_scope_stats: dict[str, dict[str, int]] = {}
    pruned_scope_by_cat_stats: dict[str, dict[str, dict[str, int]]] = {}

    desc = "Analyzing (origin+diff) ..." if diff else "Analyzing (origin) ..."
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task(desc, total=len(tasks))
        for m, r in tasks:
            progress.update(
                ptask, description=f"[bold blue]origin [dim]|[/dim] {m}/{r}"
            )
            stats.setdefault(m, defaultdict(int))
            scope_stats.setdefault(m, defaultdict(int))
            scope_by_cat_stats.setdefault(
                m, {s: defaultdict(int) for s in SCOPE_LABELS}
            )
            if diff:
                pruned_stats.setdefault(m, defaultdict(int))
                pruned_scope_stats.setdefault(m, defaultdict(int))
                pruned_scope_by_cat_stats.setdefault(
                    m, {s: defaultdict(int) for s in SCOPE_LABELS}
                )

            for path in discover_prune_files(m, r):
                data = load_prune_details(path)
                for op in data.get("prune_operations_detailed", []):
                    orig = op.get("original_output", "")
                    if not orig:
                        continue
                    prun = op.get("pruned_output", "") if diff else ""
                    try:
                        orig_lines = orig.split("\n")
                        line_type_map = _map_line_to_node_type(orig)
                        line_scope_map = _map_line_to_scope(orig)

                        for ln_idx, line in enumerate(orig_lines):
                            node_type = line_type_map.get(ln_idx, None)
                            if node_type is None:
                                node_type = "empty" if line.strip() == "" else "code"
                            scope = line_scope_map.get(ln_idx, "module_level")

                            stats[m][node_type] += 1
                            scope_stats[m][scope] += 1
                            scope_by_cat_stats[m][scope][node_type] += 1

                        if diff and prun:
                            oi = 0
                            for pl in prun.split("\n"):
                                m_f = FILTER_PATTERN.match(pl.strip())
                                if m_f:
                                    oi += int(m_f.group(1))
                                else:
                                    node_type = line_type_map.get(oi, None)
                                    if node_type is None:
                                        node_type = (
                                            "empty"
                                            if orig_lines[oi].strip() == ""
                                            else "code"
                                        )
                                    scope = line_scope_map.get(oi, "module_level")
                                    pruned_stats[m][node_type] += 1
                                    pruned_scope_stats[m][scope] += 1
                                    pruned_scope_by_cat_stats[m][scope][node_type] += 1
                                    oi += 1
                    except Exception as e:
                        console.print(
                            f"  [yellow]AST fallback[/yellow] {path.name}: {e}"
                        )
                        for line in orig.split("\n"):
                            cat = classify_line_basic(line)
                            stats[m][cat] += 1

            progress.advance(ptask)

    stats = {m: dict(v) for m, v in stats.items()}
    scope_stats = {m: dict(v) for m, v in scope_stats.items()}
    scope_by_cat_stats = {
        m: {s: dict(c) for s, c in v.items()} for m, v in scope_by_cat_stats.items()
    }

    if diff:
        pruned_stats = {m: dict(v) for m, v in pruned_stats.items()}
        pruned_scope_stats = {m: dict(v) for m, v in pruned_scope_stats.items()}
        pruned_scope_by_cat_stats = {
            m: {s: dict(c) for s, c in v.items()}
            for m, v in pruned_scope_by_cat_stats.items()
        }

        print_diff_category_table(
            stats,
            pruned_stats,
            "Origin → Kept  —  Category Kept Rate",
        )
        print_diff_scope_table(
            scope_stats,
            pruned_scope_stats,
            "Origin → Kept  —  Scope Kept Rate",
        )
        print_diff_scope_by_cat_tables(
            scope_by_cat_stats,
            pruned_scope_by_cat_stats,
            "Origin → Kept",
        )
    else:
        print_category_table(
            stats, "Original Code (Before Pruning) — Category Distribution"
        )
        print_scope_tables(
            scope_stats,
            scope_by_cat_stats,
            "Original Code (Before Pruning)",
        )

    if output:
        output_data: dict[str, Any] = {
            "category_stats": stats,
            "scope_stats": scope_stats,
            "scope_by_category_stats": scope_by_cat_stats,
        }
        if diff:
            output_data["pruned_category_stats"] = pruned_stats
            output_data["pruned_scope_stats"] = pruned_scope_stats
            output_data["pruned_scope_by_category_stats"] = pruned_scope_by_cat_stats
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    app()
