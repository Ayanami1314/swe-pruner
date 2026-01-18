from enum import Enum
from typing import List
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
from tree_sitter import Parser, Language
import colorlog
from pydantic import BaseModel

logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.DEBUG)
logger.addHandler(colorlog.StreamHandler())


class LanguageEnum(Enum):
    py = "python"
    java = "java"
    golang = "golang"
    js = "javascript"
    ts = "typescript"


# Language mapping - wrap PyCapsule in Language object
LANGUAGE_PARSERS = {
    LanguageEnum.py: Language(tspython.language()),
    LanguageEnum.java: Language(tsjava.language()),
    LanguageEnum.golang: Language(tsgo.language()),
}

# Atomic statement node types for different languages (leaf-level statements)
ATOMIC_STATEMENT_TYPES = {
    LanguageEnum.py: {
        "expression_statement",
        "import_statement",
        "import_from_statement",
        "return_statement",
        "raise_statement",
        "pass_statement",
        "break_statement",
        "continue_statement",
        "assignment",
        "augmented_assignment",
        "assert_statement",
        "delete_statement",
        "global_statement",
        "nonlocal_statement",
    },
    LanguageEnum.java: {
        "expression_statement",
        "import_declaration",
        "return_statement",
        "throw_statement",
        "break_statement",
        "continue_statement",
        "local_variable_declaration",
        "assert_statement",
    },
    LanguageEnum.golang: {
        "expression_statement",
        "import_declaration",
        "return_statement",
        "break_statement",
        "continue_statement",
        "assignment_statement",
        "short_var_declaration",
        "inc_statement",
        "dec_statement",
        "go_statement",
        "defer_statement",
    },
}

# Compound statement types that contain other statements (should recurse into)
COMPOUND_STATEMENT_TYPES = {
    LanguageEnum.py: {
        "if_statement",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
        "function_definition",
        "class_definition",
        "decorated_definition",
        "elif_clause",
        "else_clause",
        "except_clause",
        "finally_clause",
    },
    LanguageEnum.java: {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "try_statement",
        "switch_statement",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "synchronized_statement",
        "catch_clause",
        "finally_clause",
    },
    LanguageEnum.golang: {
        "if_statement",
        "for_statement",
        "switch_statement",
        "select_statement",
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "var_declaration",
        "const_declaration",
    },
}


class Statement(BaseModel):
    start_byte: int
    end_byte: int
    text: str
    type: str
    node_type: str


def split_code_stmts(code: str, language: LanguageEnum) -> List[Statement]:
    """
    Use tree-sitter to split code into statements.

    Extracts atomic statements and fills gaps with compound statement structures.
    The result covers the entire file without overlap.

    @return: A list of code statements str, e.g. seperate by line in easiest case.
    """
    if language not in LANGUAGE_PARSERS:
        # Fallback: split by lines for unsupported languages
        return [line for line in code.split("\n") if line.strip()]

    # Get the parser for the language (new API: pass language to Parser constructor)
    parser = Parser(LANGUAGE_PARSERS[language])

    # Parse the code
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # Get statement types for this language
    atomic_stmt_types = ATOMIC_STATEMENT_TYPES.get(language, set())
    compound_stmt_types = COMPOUND_STATEMENT_TYPES.get(language, set())

    # Collect all statements
    all_statements = []

    def traverse(node):
        """
        Recursively traverse the tree.
        For atomic statements: extract the whole statement.
        For compound statements: extract the header line, then recurse into body.
        For blocks: extract gaps between statements.
        """
        if node.type in atomic_stmt_types:
            # This is an atomic statement - extract it
            all_statements.append(
                Statement(
                    text=code[node.start_byte : node.end_byte],
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    type="atomic",
                    node_type=node.type,
                )
            )
            # Don't recurse into atomic statements

        elif node.type in compound_stmt_types:
            # This is a compound statement
            # Find the body/block child
            body_child = None
            for child in node.children:
                if child.type in (
                    "block",
                    "body",
                    "consequence",
                    "alternative",
                    "suite",
                ):
                    body_child = child
                    break

            if body_child:
                # Extract the header (from node start to body start)
                header_text = code[node.start_byte : body_child.start_byte]
                if header_text.strip():
                    all_statements.append(
                        Statement(
                            text=header_text,
                            start_byte=node.start_byte,
                            end_byte=body_child.start_byte,
                            type="compound",
                            node_type=node.type,
                        )
                    )

                # Recurse into the body to find nested statements
                traverse(body_child)

                # Also check for any children after the body (like else clauses)
                body_index = node.children.index(body_child)
                for child in node.children[body_index + 1 :]:
                    if child.type in compound_stmt_types or child.type in (
                        "else_clause",
                        "elif_clause",
                        "except_clause",
                        "finally_clause",
                    ):
                        traverse(child)
            else:
                # No body found, just recurse into all children
                for child in node.children:
                    traverse(child)

        elif node.type in ("block", "body", "suite", "module"):
            # This is a block/body node - process children and extract gaps
            children = list(node.children)

            # Process each child
            for i, child in enumerate(children):
                traverse(child)

            # Extract gaps between children (whitespace, newlines, etc.)
            current_pos = node.start_byte
            for child in children:
                if current_pos < child.start_byte:
                    # Gap before this child
                    gap = code[current_pos : child.start_byte]
                    if gap:  # Include all gaps, even whitespace-only
                        all_statements.append(
                            Statement(
                                text=gap,
                                start_byte=current_pos,
                                end_byte=child.start_byte,
                                type="gap",
                                node_type="gap",
                            )
                        )
                current_pos = child.end_byte

            # Gap after last child
            if current_pos < node.end_byte:
                gap = code[current_pos : node.end_byte]
                if gap:
                    all_statements.append(
                        Statement(
                            text=gap,
                            start_byte=current_pos,
                            end_byte=node.end_byte,
                            type="gap",
                            node_type="gap",
                        )
                    )
        else:
            # Not a statement node - continue traversing
            for child in node.children:
                traverse(child)

    traverse(root_node)

    # Sort by position and extract text
    all_statements.sort(key=lambda x: x.start_byte)

    # merge gap stmt into non-gap stmt
    merged_statements = []
    for stmt in all_statements:
        if stmt.type == "gap":
            if merged_statements:
                merged_statements[-1].end_byte = stmt.end_byte
        else:
            merged_statements.append(stmt)
    return merged_statements


if __name__ == "__main__":
    example = __file__
    with open(example, "r") as f:
        code = f.read()
        print(code[:100])
        stmts = split_code_stmts(code, LanguageEnum.py)
        for st in stmts:
            logger.info(f"Found statement: {st.text}")
            logger.info("-" * 40)
        logger.info(f"Found {len(stmts)} statements")
