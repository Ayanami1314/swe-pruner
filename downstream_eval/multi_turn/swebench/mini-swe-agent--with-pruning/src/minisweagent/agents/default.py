"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import re
import subprocess
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model

from typing import Any
from minisweagent.utils.pruner import PrunerClient, PrunerConfig, PruneResponse, PrunerRequest
def _resolve_env_placeholders(value: Any):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.getenv(key, "")
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value

from minisweagent.utils.log import logger

@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0
    pruner: dict[str, Any] | None = None


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.pruner_client: PrunerClient | None = None
        if self.config.pruner:
            print(f"Using Pruner Config: {self.config.pruner}")
            pruner_cfg = PrunerConfig(**{k: v for k, v in _resolve_env_placeholders(self.config.pruner).items()})
            print(f"Loaded Pruner Config: {pruner_cfg}")
            self.pruner_client = PrunerClient(pruner_cfg)

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        unparsed_err_cnt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except Exception as e:
                self.add_message("user", f"Error: {e}")
                unparsed_err_cnt += 1
                if unparsed_err_cnt >= 3:
                    return "Error", f"Unparsed error occurred {unparsed_err_cnt} times: {e}"

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        
        # Log raw model response for debugging
        logger.debug(f"Raw model response (step {self.model.n_calls}):\n{response.get('content', '')[:2000]}")
        
        # Parse action to extract context_focus_question before adding to messages
        action = self.parse_action(response)
        # Log parsed action for debugging
        logger.debug(f"Parsed action: {action.get('action', '')[:200]}")
        logger.debug(f"Context focus question: {action.get('context_focus_question', 'None')}")
        
        # Add context_focus_question to response extra if present
        if action.get("context_focus_question"):
            if "extra" not in response:
                response["extra"] = {}
            if "parsed_action" not in response["extra"]:
                response["extra"]["parsed_action"] = {}
            response["extra"]["parsed_action"]["context_focus_question"] = action["context_focus_question"]
            
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        message_kwargs: dict[str, Any] = {}
        if output.get("pruned_stats"):
            message_kwargs["pruned_stats"] = output["pruned_stats"]
        self.add_message("user", observation, **message_kwargs)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action with optional context_focus_question."""
        content = response["content"]
        
        # Extract bash command from code block first
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) != 1:
            raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
        
        action_text = actions[0].strip()
        
        # Try to extract context_focus_question from HTML comment after bash code block
        context_focus_pattern = r"```\s*(?:bash)?\s*\n.*?\n```\s*(?:<context_focus_question>\s*(.*?)\s*</context_focus_question>)?"
        context_focus_match = re.search(context_focus_pattern, content, re.DOTALL | re.IGNORECASE)
        context_focus_question = None
        if context_focus_match:
            # Try to extract from the match group
            if context_focus_match.lastindex and context_focus_match.group(1):
                context_focus_question = context_focus_match.group(1).strip()
        
        if context_focus_question:
            # Try to extract from the response extra if present
            if "extra" in response and "parsed_action" in response["extra"]:
                if "context_focus_question" in response["extra"]["parsed_action"]:
                    context_focus_question = response["extra"]["parsed_action"]["context_focus_question"]
        if context_focus_question == "":
            context_focus_question = None
        
        return {"action": action_text, "context_focus_question": context_focus_question, **response}

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        self._apply_pruner(action, output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

    def _apply_pruner(self, action: dict, output: dict[str, str]) -> None:
        if not self.pruner_client:
            return
        text = output.get("output")
        if not text:
            return
        
        # Priority 1: Use context_focus_question from action if provided
        context_focus_question = action.get("context_focus_question")
        if not context_focus_question:
            output["output"] = text
            return
        # Call pruner with context_focus_question
        req = PrunerRequest(
            code=text,
            query=context_focus_question,
            threshold=self.pruner_client.config.threshold,
            always_keep_first_frags=False,
            chunk_overlap_tokens=self.pruner_client.config.chunk_overlap_tokens,
        )
        pruned_result: PruneResponse = self.pruner_client.prune(req)
        if pruned_result.error_msg:
            output["output"] = f"[Pruner Error]: {pruned_result.error_msg}\n\nOriginal Output:\n{text}"
        else:
            if pruned_result.left_token_cnt == pruned_result.origin_token_cnt:
                output["output"] = "All outputs are judged as relevent! Output:\n" + text
            else:
                output["output"] = "Filtered some unrelevant parts judged by your context_focus_question, good try! Filtered Output:\n" + pruned_result.pruned_code
        
        # saves other stats
        output["pruned_stats"] = {
            "score": pruned_result.score,
            "origin_token_cnt": pruned_result.origin_token_cnt,
            "left_token_cnt": pruned_result.left_token_cnt,
            "model_input_token_cnt": pruned_result.model_input_token_cnt,
        }