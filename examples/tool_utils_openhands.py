import logging
from typing import Any, List, Optional, Dict, Sequence
import subprocess
import os
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
from collections.abc import Sequence as ABCSequence

from openhands.sdk import Action, Observation, TextContent, ImageContent, ToolDefinition
from openhands.sdk.tool import ToolExecutor

load_dotenv()
pruner_url = os.getenv("PRUNER_URL", "http://localhost:8000/prune")


class PruneRequest(BaseModel):
    query: str
    code: str


class PruneResponse(BaseModel):
    score: float
    pruned_code: str
    token_scores: List[List[str | float]]  # [[token_str, score], ...]
    kept_frags: List[int]
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    error_msg: Optional[str] = None


# Global HTTP client for prune requests
_httpx_client: Optional[httpx.Client] = None


def _get_httpx_client() -> httpx.Client:
    """Get or create the global httpx client."""
    global _httpx_client
    if _httpx_client is None:
        # Disable proxy for prune service requests (local service should not use proxy)
        _httpx_client = httpx.Client(timeout=60.0, proxy=None)
    return _httpx_client


def prune_fn(context: str, query: str) -> PruneResponse:
    """Synchronous version of prune function using httpx."""
    logging.info(
        f"Calling prune_fn with query: {query[:100] if query else None}, context length: {len(context)}"
    )
    client = _get_httpx_client()
    try:
        response = client.post(
            pruner_url,
            json=PruneRequest(query=query, code=context).model_dump(),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            result = PruneResponse(**response.json())
            logging.info(
                f"Prune completed: score={result.score:.3f}, kept_frags={len(result.kept_frags)}, pruned_code_length={len(result.pruned_code)}"
            )
            return result
        else:
            raise Exception(f"Pruner request failed with status {response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Pruner request error: {str(e)}")
        raise Exception(f"Pruner request error: {str(e)}")


def _execute_bash_command(command: str) -> str:
    """Execute a bash command and return output as list of lines."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip() if result.stdout.strip() else "(No output)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


# --- Action / Observation ---


class OriginBashAction(Action):
    """Action for original bash commands without pruning."""

    command: str = Field(description="Bash command to execute")


class PrunerBashAction(Action):
    """Action for bash commands with context-aware pruning."""

    command: str = Field(description="Bash command to execute")
    context_focus_question: Optional[str] = Field(
        default=None,
        description="Optional question for context-aware pruning. Use this to filter large command outputs for relevant information. Must be a complete, self-contained question (not keywords/phrases).",
    )


class BashObservation(Observation):
    output: str = Field(default="")
    pruned: bool = Field(default=False)
    prune_info: Optional[Dict[str, Any]] = Field(default=None)
    original_output: Optional[str] = Field(
        default=None
    )  # Original output before pruning

    @property
    def to_llm_content(self) -> ABCSequence[TextContent | ImageContent]:
        if self.pruned and self.prune_info:
            if self.original_output == self.output:
                return [
                    TextContent(
                        text=f"All outputs are judged relevant. Raw output: \n{self.output}"
                    )
                ]
            return [TextContent(text=f"Pruned output: \n{self.output}")]
        output_text = (
            self.output
            if self.output
            else (self.original_output if self.original_output else "(No output)")
        )
        return [TextContent(text=f"Raw Output: \n{output_text}")]

    @property
    def content(self) -> str:
        return self.to_llm_content


# --- Executor ---


class OriginBashExecutor(ToolExecutor[OriginBashAction, BashObservation]):
    """Executor for original bash commands without pruning."""

    def __call__(self, action: OriginBashAction, conversation=None) -> BashObservation:  # noqa: ARG002
        output = _execute_bash_command(action.command)
        observation = BashObservation(
            output=output, pruned=False, original_output=output
        )
        return observation


class PrunerBashExecutor(ToolExecutor[PrunerBashAction, BashObservation]):
    """Executor for bash commands with context-aware pruning."""

    def __call__(self, action: PrunerBashAction, conversation=None) -> BashObservation:  # noqa: ARG002
        original_output = _execute_bash_command(action.command)

        if action.context_focus_question:
            try:
                prune_result = prune_fn(original_output, action.context_focus_question)
                prune_info = {
                    "score": prune_result.score,
                    "origin_token_cnt": prune_result.origin_token_cnt,
                    "left_token_cnt": prune_result.left_token_cnt,
                    "model_input_token_cnt": prune_result.model_input_token_cnt,
                    "context_focus_question": action.context_focus_question,
                }
                return BashObservation(
                    output=prune_result.pruned_code,
                    pruned=True,
                    prune_info=prune_info,
                    original_output=original_output,
                )
            except Exception as e:
                logging.error(f"Pruning failed: {e}")
                return BashObservation(
                    output=f"[Pruning failed: {str(e)}]\n{original_output}",
                    pruned=False,
                    original_output=original_output,
                )

        return BashObservation(
            output=original_output, pruned=False, original_output=original_output
        )


# --- Tool Definition ---

_BASH_DESCRIPTION = """Execute bash commands in the workspace.
* Use this tool to run shell commands, navigate directories, search files, etc.
* Commands are executed in the workspace directory.
* Output is returned as text.
"""

_BASH_DESCRIPTION_PRUNE = (
    _BASH_DESCRIPTION
    + """
The meaning of argument `context_focus_question` (Optional):

Use `context_focus_question` to filter large command outputs for relevant information.

**Requirements:**
- Must be a complete, self-contained question (not keywords/phrases)
- Be specific for effective filtering
- Don't include file-level info (filenames, line numbers) - use grep/sed instead

**Good examples:**
- Where is [some logic] implemented in [some class/function]?
- Given [background], what's the [problem]?
- How does the code implement [feature]?

**Bad examples:**
- load_raw function (too vague)
- lines 50-100 of data_loader.py (contains file info)
- fix bug in rwkv6.py (too vague)

**IMPORTANT:** With pruner enabled, prefer `cat -n` or `nl -ba` with context_focus_question to see line numbers. Then you can use `sed` without filtering for more detailed context since you have line number information.

**IMPORTANT:** If the command output is small and important like `ls`, just leave context_focus_question blank.
"""
)


class OriginBashTool(ToolDefinition[OriginBashAction, BashObservation]):
    """A bash tool for executing shell commands without pruning."""

    @classmethod
    def create(
        cls,
        conv_state,
        executor: Optional[ToolExecutor] = None,
    ) -> Sequence[ToolDefinition]:
        """Create OriginBashTool instance with an executor.

        Args:
            conv_state: Conversation state to get working directory from.
            executor: Optional executor to reuse. If not provided, a new one will be created.

        Returns:
            A sequence containing a single OriginBashTool instance.
        """
        if executor is None:
            executor = OriginBashExecutor()

        return [
            cls(
                description=_BASH_DESCRIPTION,
                action_type=OriginBashAction,
                observation_type=BashObservation,
                executor=executor,
            )
        ]


class PrunerBashTool(ToolDefinition[PrunerBashAction, BashObservation]):
    """A bash tool for executing shell commands with context-aware pruning."""

    @classmethod
    def create(
        cls,
        conv_state,
        executor: Optional[ToolExecutor] = None,
    ) -> Sequence[ToolDefinition]:
        """Create PrunerBashTool instance with an executor.

        Args:
            conv_state: Conversation state to get working directory from.
            executor: Optional executor to reuse. If not provided, a new one will be created.

        Returns:
            A sequence containing a single PrunerBashTool instance.
        """
        if executor is None:
            executor = PrunerBashExecutor()

        return [
            cls(
                description=_BASH_DESCRIPTION_PRUNE,
                action_type=PrunerBashAction,
                observation_type=BashObservation,
                executor=executor,
            )
        ]


# --- Factory functions for register_tool ---


def make_origin_bash_tool(conv_state) -> List[ToolDefinition]:
    """Factory function to create origin bash tool."""
    return list(OriginBashTool.create(conv_state))


def make_pruner_bash_tool(conv_state) -> List[ToolDefinition]:
    """Factory function to create pruner bash tool."""
    return list(PrunerBashTool.create(conv_state))


origin_bash = make_origin_bash_tool
pruner_bash = make_pruner_bash_tool
