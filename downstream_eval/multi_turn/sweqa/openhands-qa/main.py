import os

os.environ.setdefault("LITELLM_DROP_PARAMS", "true")

import json
import time
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
import openai

import httpx


def setup_http_hack_for_claude():
    """Intercept httpx requests to fix Claude API parameter conflicts."""
    try:
        if not hasattr(httpx.Client, "_original_request"):
            print(f"[HTTP HACK DEBUG] ========== saving original methods ==========")
            print(
                f"[HTTP HACK DEBUG] httpx.Client.request type: {type(httpx.Client.request)}"
            )
            print(
                f"[HTTP HACK DEBUG] httpx.Client.post type: {type(httpx.Client.post)}"
            )
            httpx.Client._original_request = httpx.Client.request
            httpx.Client._original_post = httpx.Client.post
            print(
                f"[HTTP HACK DEBUG] saved _original_request: {type(httpx.Client._original_request)}"
            )
            print(
                f"[HTTP HACK DEBUG] saved _original_post: {type(httpx.Client._original_post)}"
            )
            print(
                f"[HTTP HACK DEBUG] _original_request callable: {callable(httpx.Client._original_request)}"
            )
            print(
                f"[HTTP HACK DEBUG] _original_post callable: {callable(httpx.Client._original_post)}"
            )

        if not hasattr(httpx.AsyncClient, "_original_request"):
            httpx.AsyncClient._original_request = httpx.AsyncClient.request
            httpx.AsyncClient._original_post = httpx.AsyncClient.post

        def fix_json_body(json_data, url_str):
            """Fix parameter conflicts in JSON request body."""
            if not isinstance(json_data, dict):
                return False

            fixed = False
            if "max_tokens" in json_data and "max_completion_tokens" in json_data:
                json_data.pop("max_tokens", None)
                print(
                    f"[HTTP HACK] Removed conflicting max_tokens (URL: {url_str[:50]}...)"
                )
                fixed = True
            elif "max_tokens" in json_data:
                max_tokens_val = json_data.pop("max_tokens")
                if "max_completion_tokens" not in json_data:
                    json_data["max_completion_tokens"] = max_tokens_val
                    print(
                        f"[HTTP HACK] Converted max_tokens={max_tokens_val} to max_completion_tokens (URL: {url_str[:50]}...)"
                    )
                    fixed = True
            return fixed

        def patched_request_sync(self, method, url, **kwargs):
            """Intercept sync HTTP requests to fix Claude API parameter conflicts."""
            url_str = str(url)
            if "anthropic.com" in url_str.lower():
                if "json" in kwargs:
                    fix_json_body(kwargs["json"], url_str)
                elif "content" in kwargs and isinstance(
                    kwargs["content"], (str, bytes)
                ):
                    try:
                        if isinstance(kwargs["content"], bytes):
                            content_str = kwargs["content"].decode("utf-8")
                        else:
                            content_str = kwargs["content"]
                        json_data = json.loads(content_str)
                        if fix_json_body(json_data, url_str):
                            kwargs["content"] = json.dumps(json_data).encode("utf-8")
                    except Exception as e:
                        print(f"[HTTP HACK ERROR] Failed to parse content: {e}")
                        import traceback

                        traceback.print_exc()
                if hasattr(httpx.Client, "_original_request"):
                    response = httpx.Client._original_request(
                        self, method, url, **kwargs
                    )
                else:
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )
                if response is None:
                    raise RuntimeError(
                        f"HTTP request returned None for {method} {url_str}"
                    )
                return response
            else:
                if hasattr(httpx.Client, "_original_request"):
                    try:
                        if hasattr(httpx.Client._original_request, "__get__"):
                            bound_method = httpx.Client._original_request.__get__(
                                self, httpx.Client
                            )
                            response = bound_method(method, url, **kwargs)
                        else:
                            response = httpx.Client._original_request(
                                self, method, url, **kwargs
                            )

                        if response is None:
                            if hasattr(httpx.Client, "_original_send"):
                                import httpx as httpx_module
                                import json as json_module

                                send_kwargs = {}
                                request_kwargs = {}

                                for key, value in kwargs.items():
                                    if key in [
                                        "timeout",
                                        "follow_redirects",
                                        "extensions",
                                    ]:
                                        send_kwargs[key] = value
                                    elif key == "json":
                                        request_kwargs["content"] = json_module.dumps(
                                            value
                                        ).encode("utf-8")
                                        request_kwargs["headers"] = request_kwargs.get(
                                            "headers", {}
                                        )
                                        request_kwargs["headers"]["Content-Type"] = (
                                            "application/json"
                                        )
                                    elif key == "headers":
                                        request_kwargs["headers"] = {
                                            **request_kwargs.get("headers", {}),
                                            **value,
                                        }
                                    elif key == "content":
                                        request_kwargs["content"] = value
                                    else:
                                        request_kwargs[key] = value

                                if "headers" not in request_kwargs:
                                    request_kwargs["headers"] = {}

                                request = httpx_module.Request(
                                    method, url, **request_kwargs
                                )
                                response = httpx.Client._original_send(
                                    self, request, **send_kwargs
                                )
                            else:
                                print(f"[HTTP HACK ERROR] _original_send not found!")

                            if response is None:
                                print(
                                    f"[HTTP HACK ERROR] ========== patched_request_sync: non-Anthropic request returned None =========="
                                )
                                print(
                                    f"[HTTP HACK ERROR] Method: {method}, URL: {url_str}"
                                )
                                print(
                                    f"[HTTP HACK ERROR] kwargs keys: {list(kwargs.keys())}"
                                )
                                import traceback

                                print(f"[HTTP HACK ERROR] call stack:")
                                traceback.print_stack()
                                raise RuntimeError(
                                    f"HTTP request returned None for {method} {url_str}"
                                )

                        if response is None:
                            raise RuntimeError(
                                f"HTTP request returned None for {method} {url_str}"
                            )
                        return response
                    except Exception as e:
                        print(
                            f"[HTTP HACK ERROR] ========== patched_request_sync: exception calling _original_request =========="
                        )
                        print(f"[HTTP HACK ERROR] Exception type: {type(e)}")
                        print(f"[HTTP HACK ERROR] Exception message: {str(e)}")
                        print(f"[HTTP HACK ERROR] Method: {method}, URL: {url_str}")
                        import traceback

                        print(f"[HTTP HACK ERROR] Full stack:")
                        traceback.print_exc()
                        raise
                else:
                    print(f"[HTTP HACK ERROR] _original_request not found!")
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )

        def patched_post_sync(self, url, **kwargs):
            """Intercept sync POST requests."""
            url_str = str(url)
            if "anthropic.com" in url_str.lower():
                return patched_request_sync(self, "POST", url, **kwargs)
            else:
                if hasattr(httpx.Client, "_original_request"):
                    try:
                        if hasattr(httpx.Client._original_request, "__get__"):
                            bound_method = httpx.Client._original_request.__get__(
                                self, httpx.Client
                            )
                            response = bound_method("POST", url, **kwargs)
                        else:
                            response = httpx.Client._original_request(
                                self, "POST", url, **kwargs
                            )

                        if response is None:
                            if hasattr(httpx.Client, "_original_send"):
                                import httpx as httpx_module
                                import json as json_module

                                send_kwargs = {}
                                request_kwargs = {}

                                for key, value in kwargs.items():
                                    if key in [
                                        "timeout",
                                        "follow_redirects",
                                        "extensions",
                                    ]:
                                        send_kwargs[key] = value
                                    elif key == "json":
                                        request_kwargs["content"] = json_module.dumps(
                                            value
                                        ).encode("utf-8")
                                        request_kwargs["headers"] = request_kwargs.get(
                                            "headers", {}
                                        )
                                        request_kwargs["headers"]["Content-Type"] = (
                                            "application/json"
                                        )
                                    elif key == "headers":
                                        request_kwargs["headers"] = {
                                            **request_kwargs.get("headers", {}),
                                            **value,
                                        }
                                    elif key == "content":
                                        request_kwargs["content"] = value
                                    else:
                                        request_kwargs[key] = value

                                if "headers" not in request_kwargs:
                                    request_kwargs["headers"] = {}

                                request = httpx_module.Request(
                                    "POST", url, **request_kwargs
                                )
                                response = httpx.Client._original_send(
                                    self, request, **send_kwargs
                                )
                            else:
                                print(f"[HTTP HACK ERROR] _original_send not found!")

                            if response is None:
                                print(
                                    f"[HTTP HACK ERROR] ========== non-Anthropic POST returned None =========="
                                )
                                print(f"[HTTP HACK ERROR] URL: {url_str}")
                                print(
                                    f"[HTTP HACK ERROR] kwargs keys: {list(kwargs.keys())}"
                                )
                                import traceback

                                print(f"[HTTP HACK ERROR] call stack:")
                                traceback.print_stack()
                                raise RuntimeError(
                                    f"HTTP POST request returned None for {url_str}"
                                )

                        return response
                    except Exception as e:
                        print(
                            f"[HTTP HACK ERROR] ========== exception calling _original_request =========="
                        )
                        print(f"[HTTP HACK ERROR] Exception type: {type(e)}")
                        print(f"[HTTP HACK ERROR] Exception message: {str(e)}")
                        print(f"[HTTP HACK ERROR] URL: {url_str}")
                        import traceback

                        print(f"[HTTP HACK ERROR] Full stack:")
                        traceback.print_exc()
                        raise
                else:
                    print(f"[HTTP HACK ERROR] _original_request not found!")
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )

        async def patched_request_async(self, method, url, **kwargs):
            """Intercept async HTTP requests to fix Claude API parameter conflicts."""
            url_str = str(url)
            if "anthropic.com" in url_str.lower():
                if "json" in kwargs:
                    fix_json_body(kwargs["json"], url_str)
                elif "content" in kwargs and isinstance(
                    kwargs["content"], (str, bytes)
                ):
                    try:
                        if isinstance(kwargs["content"], bytes):
                            content_str = kwargs["content"].decode("utf-8")
                        else:
                            content_str = kwargs["content"]
                        json_data = json.loads(content_str)
                        if fix_json_body(json_data, url_str):
                            kwargs["content"] = json.dumps(json_data).encode("utf-8")
                    except Exception as e:
                        print(f"[HTTP HACK ERROR] Failed to parse content: {e}")
                        import traceback

                        traceback.print_exc()

            return await httpx.AsyncClient._original_request(
                self, method, url, **kwargs
            )

        async def patched_post_async(self, url, **kwargs):
            """Intercept async POST requests."""
            return await patched_request_async(self, "POST", url, **kwargs)

        httpx.Client.request = patched_request_sync
        httpx.Client.post = patched_post_sync
        httpx.AsyncClient.request = patched_request_async
        httpx.AsyncClient.post = patched_post_async

        try:
            if hasattr(httpx.Client, "send") and not hasattr(
                httpx.Client, "_original_send"
            ):
                httpx.Client._original_send = httpx.Client.send

                def patched_send_sync(self, request, **kwargs):
                    """Intercept send method to fix request body."""
                    url_str = str(request.url)
                    if "anthropic.com" in url_str.lower():
                        if hasattr(request, "content") and request.content:
                            try:
                                if isinstance(request.content, bytes):
                                    content_str = request.content.decode("utf-8")
                                else:
                                    content_str = str(request.content)

                                json_data = json.loads(content_str)

                                if fix_json_body(json_data, url_str):
                                    try:
                                        new_content = json.dumps(json_data).encode(
                                            "utf-8"
                                        )
                                        import httpx as httpx_module

                                        headers = dict(request.headers)
                                        headers_to_remove = []
                                        for key in headers.keys():
                                            if key.lower() == "content-length":
                                                headers_to_remove.append(key)
                                        for key in headers_to_remove:
                                            del headers[key]
                                        headers["content-length"] = str(
                                            len(new_content)
                                        )

                                        original_extensions = {}
                                        if (
                                            hasattr(request, "extensions")
                                            and request.extensions
                                        ):
                                            original_extensions = dict(
                                                request.extensions
                                            )

                                        new_request = httpx_module.Request(
                                            request.method,
                                            request.url,
                                            content=new_content,
                                            headers=headers,
                                        )

                                        if original_extensions:
                                            for (
                                                key,
                                                value,
                                            ) in original_extensions.items():
                                                if hasattr(new_request, "extensions"):
                                                    new_request.extensions[key] = value

                                        request = new_request
                                    except Exception as build_error:
                                        print(
                                            f"[HTTP HACK ERROR] Failed to rebuild request: {build_error}"
                                        )
                                        import traceback

                                        traceback.print_exc()
                            except Exception as e:
                                print(f"[HTTP HACK ERROR] send method fix failed: {e}")
                                import traceback

                                traceback.print_exc()

                        try:
                            response = httpx.Client._original_send(
                                self, request, **kwargs
                            )
                            return response
                        except Exception as send_error:
                            print(
                                f"[HTTP HACK ERROR] ========== send request failed =========="
                            )
                            print(f"[HTTP HACK ERROR] Error type: {type(send_error)}")
                            print(f"[HTTP HACK ERROR] Error message: {str(send_error)}")
                            print(f"[HTTP HACK ERROR] Request URL: {request.url}")
                            print(f"[HTTP HACK ERROR] Request Method: {request.method}")
                            print(
                                f"[HTTP HACK ERROR] Request Headers: {dict(request.headers)}"
                            )
                            if hasattr(request, "content") and request.content:
                                try:
                                    if isinstance(request.content, bytes):
                                        content_preview = request.content[:500].decode(
                                            "utf-8", errors="ignore"
                                        )
                                    else:
                                        content_preview = str(request.content)[:500]
                                    print(
                                        f"[HTTP HACK ERROR] Request body preview: {content_preview}"
                                    )
                                except:
                                    print(f"[HTTP HACK ERROR] Cannot preview request body")
                            import traceback

                            print(f"[HTTP HACK ERROR] Full error stack:")
                            traceback.print_exc()
                            raise

                httpx.Client.send = patched_send_sync
                print("[HTTP HACK] send method interception enabled")
        except Exception as e:
            print(f"[HTTP HACK DEBUG] Cannot intercept send method: {e}")

        print("[HTTP HACK] HTTP request layer interception enabled (sync+async)")
        return True
    except Exception as e:
        print(f"[WARNING] Cannot apply HTTP hack: {e}")
        import traceback

        traceback.print_exc()
        return False


_http_hack_enabled = setup_http_hack_for_claude()

from openhands.sdk import LLM, Conversation, Agent
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent
from openhands.sdk.tool import Tool, register_tool
from openhands.sdk import TextContent, ImageContent
import sys

sys.path.append(Path(__file__).parent)
from tool_utils import pruner_bash, origin_bash

load_dotenv()
experiment = os.getenv("EXPERIMENT_TYPE")
api_type = os.getenv("API_TYPE")
if api_type not in ["openai", "azure"]:
    print(f"Unsupported api type: {api_type}")
    exit(1)
MAX_WORKERS = 16
if experiment == "baseline":
    register_tool("Bash", origin_bash)
elif experiment == "pruner":
    register_tool("Bash", pruner_bash)
else:
    print(f"Wrong Experiment Type : {experiment}")
    exit(1)

tools = [Tool(name="Bash")]

# Load Azure OpenAI configuration from environment variables
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# Load OpenHands LLM configuration
# For Azure OpenAI with litellm, we need to prefix the model name with "azure/"
model_name = os.getenv("OPENHANDS_MODEL_NAME")
if api_type == "azure":
    LLM_CONFIG = {
        "model": model_name,
        "api_key": AZURE_API_KEY,
        "base_url": AZURE_ENDPOINT,
        "api_version": AZURE_API_VERSION,  # Required for Azure OpenAI
        "usage_id": "agent",
        "extra_headers": {"X-TT-LOGID": "${your_logid}"},
    }
elif api_type == "openai":
    LLM_CONFIG = {
        "model": model_name,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
    }
    if "claude" in model_name.lower():
        LLM_CONFIG["drop_params"] = True
    print(LLM_CONFIG)
else:
    raise NotImplementedError


# Load base paths from environment
BASE_REPO_PATH = os.getenv("BASE_REPO_PATH", "./swe-repos")
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "./questions")


def load_repos_config():
    """Load repository configuration from environment or use default"""
    repos_env = os.getenv("OPENHANDS_REPOS", "reflex,streamlink,conan")
    repos = [repo.strip() for repo in repos_env.split(",") if repo.strip()]

    repos_config = []
    for repo_name in repos:
        repos_config.append(
            {
                "name": repo_name,
                "workspace": os.path.join(BASE_REPO_PATH, repo_name),
                "input_file": os.path.join(QUESTIONS_PATH, f"{repo_name}.jsonl"),
            }
        )
    return repos_config


REPOS_CONFIG = load_repos_config()

# Output configuration
OUTPUT_DIR = os.getenv("ANSWER_OUTPUT_PATH", "./answer") + "/openhands"
TRAJ_DIR = os.getenv("TRAJ_OUTPUT_PATH", "./trajectories")
MAX_ITERATION_PER_RUN = int(os.getenv("MAX_ITERATION_PER_RUN", "50"))
MAX_TIME_PER_QUESTION = int(os.getenv("MAX_TIME_PER_QUESTION", "1800"))


def get_repo_name_from_path(file_path):
    """Extract repo name from file path."""
    basename = os.path.basename(file_path)
    if basename.endswith(".jsonl"):
        return basename[:-6]
    return basename


def load_questions_from_jsonl(file_path):
    """Load questions from a jsonl file."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get("question", "")
                if question:
                    questions.append(data)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    return questions


def load_answered_questions(output_file):
    """Load already-answered questions from the output file."""
    answered_questions = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get("question", "")
                        if question:
                            answered_questions.add(question)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading answered questions: {e}")
    return answered_questions


def get_message_history(state):
    """Extract full message history from conversation state."""
    messages = []
    events = list(state.events)

    print(f"[DEBUG] Total events: {len(events)}")

    for idx, event in enumerate(events):
        event_type = type(event).__name__

        if isinstance(event, MessageEvent):
            source = getattr(event, "source", "unknown")

            content = ""
            if hasattr(event, "extended_content") and event.extended_content:
                text_parts = []
                for item in event.extended_content:
                    if hasattr(item, "text"):
                        text_parts.append(str(item.text))
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    content = "\n".join(text_parts)

            if not content and hasattr(event, "llm_message") and event.llm_message:
                msg = event.llm_message
                if hasattr(msg, "content") and msg.content:
                    if isinstance(msg.content, list):
                        text_parts = []
                        for item in msg.content:
                            if hasattr(item, "text"):
                                text_parts.append(str(item.text))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            content = "\n".join(text_parts)
                    elif isinstance(msg.content, str):
                        content = msg.content

            if content:
                messages.append(
                    {
                        "role": source,  # 'user', 'agent', 'tool'
                        "content": content,
                        "timestamp": getattr(event, "timestamp", None),
                        "event_type": event_type,
                    }
                )
                print(
                    f"[DEBUG Event {idx}] MessageEvent, source={source}, content length={len(content)}"
                )

        elif isinstance(event, ActionEvent):
            action_info = ""
            if hasattr(event, "action") and event.action:
                action = event.action
                action_name = getattr(action, "name", "unknown")
                action_info = f"Action: {action_name}"

                if hasattr(action, "model_dump"):
                    try:
                        action_dict = action.model_dump()
                        action_info += f"\nParameters: {json.dumps(action_dict, ensure_ascii=False, indent=2)}"
                    except:
                        action_info += f"\nAction object: {str(action)}"

            if action_info:
                messages.append(
                    {
                        "role": "agent",
                        "content": action_info,
                        "timestamp": getattr(event, "timestamp", None),
                        "event_type": event_type,
                    }
                )
                print(
                    f"[DEBUG Event {idx}] ActionEvent, content length={len(action_info)}"
                )

        elif isinstance(event, ObservationEvent):
            observation_info = ""
            if hasattr(event, "observation") and event.observation:
                observation = event.observation
                if hasattr(observation, "message"):
                    observation_info = f"Observation: {observation.message}"
                elif hasattr(observation, "content"):
                    observation_info = f"Observation: {observation.content}"
                elif hasattr(observation, "model_dump"):
                    try:
                        obs_dict = observation.model_dump()
                        observation_info = f"Observation: {json.dumps(obs_dict, ensure_ascii=False, indent=2)}"
                    except:
                        observation_info = f"Observation: {str(observation)}"
                else:
                    observation_info = f"Observation: {str(observation)}"

            if observation_info:
                messages.append(
                    {
                        "role": "tool",
                        "content": observation_info,
                        "timestamp": getattr(event, "timestamp", None),
                        "event_type": event_type,
                    }
                )
                print(
                    f"[DEBUG Event {idx}] ObservationEvent, content length={len(observation_info)}"
                )
        else:
            print(f"[DEBUG Event {idx}] other event type: {event_type}")

    print(f"[DEBUG] Extracted {len(messages)} messages")
    return messages


def make_json_serializable(obj):
    """Recursively convert object to JSON-serializable format."""
    if isinstance(obj, TextContent):
        return {"type": "text", "text": obj.text}
    elif isinstance(obj, ImageContent):
        return {"type": "image", "image_url": getattr(obj, "image_url", None)}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        try:
            if hasattr(obj, "model_dump"):
                return make_json_serializable(obj.model_dump())
            else:
                return {
                    key: make_json_serializable(value)
                    for key, value in obj.__dict__.items()
                }
        except:
            return str(obj)
    else:
        return str(obj)


def serialize_event(event):
    """Serialize event object to dict."""
    event_dict = {
        "type": type(event).__name__,
        "id": getattr(event, "id", None),
        "timestamp": getattr(event, "timestamp", None),
        "source": getattr(event, "source", None),
    }

    if isinstance(event, MessageEvent):
        content = ""
        if hasattr(event, "extended_content") and event.extended_content:
            text_parts = []
            for item in event.extended_content:
                if isinstance(item, TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, ImageContent):
                    text_parts.append(f"[Image: {getattr(item, 'image_url', 'N/A')}]")
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                content = "\n".join(text_parts)

        if not content and hasattr(event, "llm_message") and event.llm_message:
            msg = event.llm_message
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, list):
                    text_parts = []
                    for item in msg.content:
                        if isinstance(item, TextContent):
                            text_parts.append(item.text)
                        elif isinstance(item, ImageContent):
                            text_parts.append(
                                f"[Image: {getattr(item, 'image_url', 'N/A')}]"
                            )
                        elif hasattr(item, "text"):
                            text_parts.append(str(item.text))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    if text_parts:
                        content = "\n".join(text_parts)
                elif isinstance(msg.content, str):
                    content = msg.content

        event_dict["content"] = content
        event_dict["source"] = getattr(event, "source", None)

    elif isinstance(event, ActionEvent):
        if hasattr(event, "action") and event.action:
            action = event.action
            action_dict = {
                "name": getattr(action, "name", None),
            }
            if hasattr(action, "model_dump"):
                try:
                    dumped = action.model_dump()
                    action_dict.update(make_json_serializable(dumped))
                except:
                    action_dict["str"] = str(action)
            event_dict["action"] = action_dict

    elif isinstance(event, ObservationEvent):
        if hasattr(event, "observation") and event.observation:
            observation = event.observation
            obs_dict = {}

            if hasattr(observation, "pruned") and observation.pruned:
                obs_dict["pruned"] = True
                obs_dict["output"] = getattr(observation, "output", "")
                obs_dict["original_output"] = getattr(
                    observation, "original_output", None
                )

                if hasattr(observation, "prune_info") and observation.prune_info:
                    prune_info = observation.prune_info
                    obs_dict["prune_info"] = {
                        "score": prune_info.get("score"),
                        "origin_token_cnt": prune_info.get("origin_token_cnt"),
                        "left_token_cnt": prune_info.get("left_token_cnt"),
                        "model_input_token_cnt": prune_info.get(
                            "model_input_token_cnt"
                        ),
                        "prune_threshold": prune_info.get("prune_threshold"),
                        "context_focus_question": prune_info.get(
                            "context_focus_question"
                        ),
                    }
                    if prune_info.get("origin_token_cnt") and prune_info.get(
                        "left_token_cnt"
                    ):
                        obs_dict["prune_stats"] = {
                            "compression_ratio": round(
                                prune_info["left_token_cnt"]
                                / prune_info["origin_token_cnt"],
                                4,
                            ),
                            "tokens_removed": prune_info["origin_token_cnt"]
                            - prune_info["left_token_cnt"],
                            "tokens_kept": prune_info["left_token_cnt"],
                            "tokens_original": prune_info["origin_token_cnt"],
                        }
            else:
                if hasattr(observation, "to_llm_content"):
                    try:
                        llm_content = observation.to_llm_content
                        if llm_content:
                            content_parts = []
                            for item in llm_content:
                                if isinstance(item, TextContent):
                                    content_parts.append(item.text)
                                elif isinstance(item, ImageContent):
                                    content_parts.append(
                                        f"[Image: {getattr(item, 'image_url', 'N/A')}]"
                                    )
                                else:
                                    content_parts.append(str(item))
                            if content_parts:
                                obs_dict["content"] = "\n".join(content_parts)
                    except:
                        pass

                if "content" not in obs_dict or not obs_dict["content"]:
                    if hasattr(observation, "message"):
                        obs_dict["message"] = observation.message
                    elif hasattr(observation, "content"):
                        obs_dict["content"] = observation.content
                    elif hasattr(observation, "output"):
                        obs_dict["output"] = observation.output
                    elif hasattr(observation, "model_dump"):
                        try:
                            dumped = observation.model_dump()
                            obs_dict.update(make_json_serializable(dumped))
                        except:
                            obs_dict["str"] = str(observation)
                    else:
                        obs_dict["str"] = str(observation)
                obs_dict["pruned"] = False

            event_dict["observation"] = obs_dict

    else:
        try:
            if hasattr(event, "model_dump"):
                dumped = event.model_dump()
                event_dict.update(make_json_serializable(dumped))
            else:
                event_dict["str"] = str(event)
        except:
            event_dict["str"] = str(event)

    return make_json_serializable(event_dict)


def extract_prune_statistics(events):
    """Extract prune statistics from events."""
    prune_stats = {
        "total_prune_operations": 0,
        "total_original_tokens": 0,
        "total_pruned_tokens": 0,
        "total_tokens_saved": 0,
        "average_compression_ratio": 0.0,
        "prune_operations": [],
    }

    for event in events:
        if isinstance(event, ObservationEvent):
            if hasattr(event, "observation") and event.observation:
                observation = event.observation
                if hasattr(observation, "pruned") and observation.pruned:
                    if hasattr(observation, "prune_info") and observation.prune_info:
                        prune_info = observation.prune_info
                        prune_stats["total_prune_operations"] += 1

                        origin_tokens = prune_info.get("origin_token_cnt", 0)
                        left_tokens = prune_info.get("left_token_cnt", 0)

                        prune_stats["total_original_tokens"] += origin_tokens
                        prune_stats["total_pruned_tokens"] += left_tokens
                        prune_stats["total_tokens_saved"] += origin_tokens - left_tokens

                        prune_op = {
                            "original_tokens": origin_tokens,
                            "pruned_tokens": left_tokens,
                            "tokens_saved": origin_tokens - left_tokens,
                            "compression_ratio": round(left_tokens / origin_tokens, 4)
                            if origin_tokens > 0
                            else 0.0,
                            "score": prune_info.get("score"),
                            "context_question": prune_info.get(
                                "context_focus_question"
                            ),
                        }
                        prune_stats["prune_operations"].append(prune_op)

    if prune_stats["total_prune_operations"] > 0:
        if prune_stats["total_original_tokens"] > 0:
            prune_stats["average_compression_ratio"] = round(
                prune_stats["total_pruned_tokens"]
                / prune_stats["total_original_tokens"],
                4,
            )

    return prune_stats


def save_trajectory(
    conversation, question, repo_name, question_idx, traj_dir, append=False
):
    """Save conversation trajectory to .traj.json file."""
    try:
        if conversation is None or conversation.state is None:
            return

        state = conversation.state
        events = list(state.events)

        serialized_events = [serialize_event(event) for event in events]

        prune_statistics = None
        if experiment == "pruner":
            prune_statistics = extract_prune_statistics(events)

        trajectory_data = {
            "question": question,
            "repo_name": repo_name,
            "question_idx": question_idx,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment,
            "events": serialized_events,
            "event_count": len(events),
        }

        if prune_statistics:
            trajectory_data["prune_statistics"] = prune_statistics

        traj_filename = f"{repo_name}_q{question_idx}.traj.json"
        traj_filepath = os.path.join(traj_dir, traj_filename)

        if append and os.path.exists(traj_filepath):
            try:
                with open(traj_filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_data["events"] = serialized_events
                existing_data["event_count"] = len(events)
                if prune_statistics:
                    existing_data["prune_statistics"] = prune_statistics
                trajectory_data = existing_data
            except:
                pass

        with open(traj_filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)

        if not append:
            print(f"[DEBUG] Trajectory saved to: {traj_filepath}")

        if (
            not append
            and experiment == "pruner"
            and prune_statistics
            and prune_statistics["total_prune_operations"] > 0
        ):
            prune_details_filename = f"{repo_name}_q{question_idx}_prune_details.json"
            prune_details_filepath = os.path.join(traj_dir, prune_details_filename)

            prune_details = {
                "question": question,
                "repo_name": repo_name,
                "question_idx": question_idx,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_prune_operations": prune_statistics[
                        "total_prune_operations"
                    ],
                    "total_original_tokens": prune_statistics["total_original_tokens"],
                    "total_pruned_tokens": prune_statistics["total_pruned_tokens"],
                    "total_tokens_saved": prune_statistics["total_tokens_saved"],
                    "average_compression_ratio": prune_statistics[
                        "average_compression_ratio"
                    ],
                },
                "prune_operations": prune_statistics["prune_operations"],
            }

            prune_operations_detailed = []
            for event in events:
                if isinstance(event, ObservationEvent):
                    if hasattr(event, "observation") and event.observation:
                        observation = event.observation
                        if hasattr(observation, "pruned") and observation.pruned:
                            if (
                                hasattr(observation, "prune_info")
                                and observation.prune_info
                            ):
                                prune_info = observation.prune_info
                                op_detail = {
                                    "original_output": getattr(
                                        observation, "original_output", ""
                                    ),
                                    "pruned_output": getattr(observation, "output", ""),
                                    "prune_info": {
                                        "score": prune_info.get("score"),
                                        "origin_token_cnt": prune_info.get(
                                            "origin_token_cnt"
                                        ),
                                        "left_token_cnt": prune_info.get(
                                            "left_token_cnt"
                                        ),
                                        "model_input_token_cnt": prune_info.get(
                                            "model_input_token_cnt"
                                        ),
                                        "prune_threshold": prune_info.get(
                                            "prune_threshold"
                                        ),
                                        "context_focus_question": prune_info.get(
                                            "context_focus_question"
                                        ),
                                    },
                                    "compression_stats": {
                                        "compression_ratio": round(
                                            prune_info.get("left_token_cnt", 0)
                                            / prune_info.get("origin_token_cnt", 1),
                                            4,
                                        ),
                                        "tokens_removed": prune_info.get(
                                            "origin_token_cnt", 0
                                        )
                                        - prune_info.get("left_token_cnt", 0),
                                        "tokens_kept": prune_info.get(
                                            "left_token_cnt", 0
                                        ),
                                        "tokens_original": prune_info.get(
                                            "origin_token_cnt", 0
                                        ),
                                    },
                                }
                                prune_operations_detailed.append(op_detail)

            prune_details["prune_operations_detailed"] = prune_operations_detailed

            with open(prune_details_filepath, "w", encoding="utf-8") as f:
                json.dump(prune_details, f, ensure_ascii=False, indent=2)

            print(f"[DEBUG] Prune details saved to: {prune_details_filepath}")

    except Exception as e:
        print(f"[DEBUG] Error saving trajectory: {e}")
        import traceback

        traceback.print_exc()


def save_experiment_config(config_dir, repo_name):
    """Save experiment config parameters."""
    try:
        config_data = {
            "experiment_type": experiment,
            "repo_name": repo_name,
            "timestamp": datetime.now().isoformat(),
            "llm_config": {
                "model": LLM_CONFIG["model"],
                "base_url": LLM_CONFIG["base_url"],
                "api_version": AZURE_API_VERSION,
                "usage_id": LLM_CONFIG["usage_id"],
            },
            "azure_config": {
                "endpoint": AZURE_ENDPOINT,
                "api_version": AZURE_API_VERSION,
            },
            "processing_config": {
                "max_iteration_per_run": MAX_ITERATION_PER_RUN,
                "max_time_per_question": MAX_TIME_PER_QUESTION,
                "base_repo_path": BASE_REPO_PATH,
                "questions_path": QUESTIONS_PATH,
            },
            "tools": [tool.name for tool in tools],
        }

        if experiment == "pruner":
            try:
                from tool_utils import pruner_url, prune_threshold

                config_data["prune_config"] = {
                    "pruner_url": pruner_url,
                    "prune_threshold": prune_threshold,
                }
            except:
                pass

        config_filename = f"{repo_name}_experiment_config.json"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        print(f"[DEBUG] Experiment config saved to: {config_filepath}")

    except Exception as e:
        print(f"[DEBUG] Error saving experiment config: {e}")


def save_prompts(prompt_dir, repo_name, question, enhanced_question):
    """Save prompt files."""
    try:
        prompt_data = {
            "repo_name": repo_name,
            "question": question,
            "enhanced_question": enhanced_question,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment,
        }

        question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        prompt_filename = f"{repo_name}_{question_hash}_prompt.json"
        prompt_filepath = os.path.join(prompt_dir, prompt_filename)

        with open(prompt_filepath, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)

        print(f"[DEBUG] Prompt saved to: {prompt_filepath}")

    except Exception as e:
        print(f"[DEBUG] Error saving prompt: {e}")


def generate_answer_from_history(llm_config, question, message_history):
    """Generate answer using LLM based on message history."""
    try:
        if api_type == "openai":
            client = openai.OpenAI(
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
            )
        elif api_type == "azure":
            client = openai.AzureOpenAI(
                azure_endpoint=llm_config["base_url"],
                api_key=llm_config["api_key"],
                api_version=AZURE_API_VERSION,
                default_headers={"X-TT-LOGID": "${your_logid}"},
            )

        conversation_text = []

        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if content:
                if role == "user":
                    conversation_text.append(f"User: {content}")
                elif role == "agent":
                    conversation_text.append(f"Assistant: {content}")
                elif role == "tool":
                    conversation_text.append(f"Tool Output: {content}")
                else:
                    conversation_text.append(f"{role}: {content}")

        full_conversation = "\n\n".join(conversation_text)

        system_prompt = """You are a code repository question answering assistant. 
Based on the conversation history provided, synthesize all the information gathered and provide a comprehensive answer to the user's question.
Even if the information is incomplete, provide the best answer you can based on what was discovered during the exploration."""

        user_prompt = f"""Original Question: {question}

Conversation History:
{full_conversation}

Based on the conversation history above, please provide a comprehensive answer to the original question."""

        formatted_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        print(f"[DEBUG] Calling LLM to generate answer, conversation history length: {len(full_conversation)} chars")

        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=formatted_messages,
            temperature=0.3,
            extra_headers={"X-TT-LOGID": "${your_logid}"},
            stream=False,
        )  # stream=False for gemini3 bug

        answer = response.choices[0].message.content.strip()

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        print(f"[DEBUG] LLM answer generated, length: {len(answer)} chars")
        print(
            f"[DEBUG] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
        )

        return answer, (prompt_tokens, completion_tokens)
    except Exception as e:
        print(f"[DEBUG] Error generating answer: {e}")
        import traceback

        traceback.print_exc()
        return None, (0, 0)


def safe_close_conversation(conversation):
    """Safely close conversation, ignoring agent-not-initialized errors."""
    if conversation is None:
        return
    try:
        conversation.close()
    except RuntimeError as e:
        if "not initialized" in str(e) or "Agent not initialized" in str(e):
            pass
        else:
            raise
    except Exception:
        pass


def process_single_question(
    qa_data,
    workspace,
    repo_name=None,
    question_idx=None,
    traj_dir=None,
    prompt_dir=None,
):
    """Process a single question."""
    question = qa_data.get("question", "")
    if not question:
        return None

    if "claude" in model_name.lower() and not _http_hack_enabled:
        try:
            import litellm

            if not hasattr(litellm, "_original_completion"):
                litellm._original_completion = litellm.completion

            def patched_completion(*args, **kwargs):
                model = kwargs.get("model", args[0] if args else None)
                if model and "claude" in str(model).lower():
                    if "max_tokens" in kwargs and "max_completion_tokens" in kwargs:
                        kwargs.pop("max_tokens", None)
                        print("[LITELLM PATCH] Removed conflicting max_tokens parameter")
                    elif "max_tokens" in kwargs:
                        max_tokens_val = kwargs.pop("max_tokens")
                        if "max_completion_tokens" not in kwargs:
                            kwargs["max_completion_tokens"] = max_tokens_val
                            print(
                                f"[LITELLM PATCH] Converted max_tokens={max_tokens_val} to max_completion_tokens"
                            )
                return litellm._original_completion(*args, **kwargs)

            litellm.completion = patched_completion
            print("[FALLBACK] LiteLLM function-level intercept patch enabled")
        except Exception as e2:
            print(f"[WARNING] Failed to apply LiteLLM patch: {e2}")

    llm = LLM(**LLM_CONFIG)

    print(LLM_CONFIG)
    agent = Agent(llm=llm, tools=tools)

    answer_data = {
        "question": question,
        "answer": "",
        "timestamp": datetime.now().isoformat(),
        "time_cost": 0.0,
        "token_cost": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    def on_event(event):
        nonlocal answer_data, conversation
        if isinstance(event, MessageEvent):
            if hasattr(event, "source") and event.source == "agent":
                if hasattr(event, "extended_content") and event.extended_content:
                    text_content = []
                    for item in event.extended_content:
                        if hasattr(item, "text"):
                            text_content.append(item.text)
                        elif isinstance(item, str):
                            text_content.append(item)
                    if text_content:
                        answer_data["answer"] = "\n".join(text_content)
                elif hasattr(event, "llm_message") and event.llm_message:
                    msg = event.llm_message
                    if hasattr(msg, "content") and msg.content:
                        if isinstance(msg.content, list):
                            text_content = []
                            for item in msg.content:
                                if hasattr(item, "text"):
                                    text_content.append(item.text)
                                elif isinstance(item, str):
                                    text_content.append(item)
                            if text_content:
                                answer_data["answer"] = "\n".join(text_content)
                        elif isinstance(msg.content, str):
                            answer_data["answer"] = msg.content

        elif isinstance(event, ActionEvent):
            if hasattr(event, "action") and event.action:
                action = event.action
                action_name = getattr(action, "name", None)
                if action_name and (
                    "finish" in action_name.lower() or "final" in action_name.lower()
                ):
                    if hasattr(action, "content"):
                        if isinstance(action.content, str):
                            answer_data["answer"] = action.content
                        elif isinstance(action.content, list):
                            text_content = []
                            for item in action.content:
                                if hasattr(item, "text"):
                                    text_content.append(item.text)
                                elif isinstance(item, str):
                                    text_content.append(item)
                            if text_content:
                                answer_data["answer"] = "\n".join(text_content)
                    elif hasattr(action, "model_dump"):
                        try:
                            action_dict = action.model_dump()
                            for key in ["content", "answer", "text", "message"]:
                                if key in action_dict and action_dict[key]:
                                    if isinstance(action_dict[key], str):
                                        answer_data["answer"] = action_dict[key]
                                        break
                                    elif isinstance(action_dict[key], list):
                                        text_content = []
                                        for item in action_dict[key]:
                                            if isinstance(item, str):
                                                text_content.append(item)
                                            elif hasattr(item, "text"):
                                                text_content.append(item.text)
                                        if text_content:
                                            answer_data["answer"] = "\n".join(
                                                text_content
                                            )
                                            break
                        except:
                            pass

        if traj_dir and repo_name and question_idx is not None and conversation:
            try:
                save_trajectory(
                    conversation,
                    question,
                    repo_name,
                    question_idx,
                    traj_dir,
                    append=True,
                )
            except Exception as e:
                pass

    conversation = None
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            callbacks=[on_event],
        )

        start_time = time.time()

        enhanced_question = f"""Please first explore the codebase structure to find the relevant files.
Use the terminal tool to search for files related to the question.
Then answer: {question}"""

        if prompt_dir and repo_name:
            save_prompts(prompt_dir, repo_name, question, enhanced_question)

        conversation.send_message(enhanced_question)

        timeout_occurred = False
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future = executor.submit(conversation.run)
                future.result(timeout=MAX_TIME_PER_QUESTION)
        except FutureTimeoutError:
            timeout_occurred = True
            print(f"[TIMEOUT] Question processing timed out (>{MAX_TIME_PER_QUESTION}s), aborting")
            answer_data["answer"] = (
                f"Timeout: Question processing exceeded {MAX_TIME_PER_QUESTION} seconds and was aborted."
            )
            answer_data["time_cost"] = MAX_TIME_PER_QUESTION
            return answer_data

        end_time = time.time()
        answer_data["time_cost"] = round(end_time - start_time, 2)

        state = conversation.state

        message_history = get_message_history(state)
        print(f"[DEBUG] Got {len(message_history)} message history entries")

        if hasattr(state, "stats") and state.stats:
            stats = state.stats
            if hasattr(stats, "usage_to_metrics"):
                total_prompt_tokens = 0
                total_completion_tokens = 0
                for usage_id, metrics in stats.usage_to_metrics.items():
                    if (
                        hasattr(metrics, "accumulated_token_usage")
                        and metrics.accumulated_token_usage
                    ):
                        token_usage = metrics.accumulated_token_usage
                        if hasattr(token_usage, "prompt_tokens") and hasattr(
                            token_usage, "completion_tokens"
                        ):
                            total_prompt_tokens += token_usage.prompt_tokens
                            total_completion_tokens += token_usage.completion_tokens
                    if hasattr(metrics, "token_usages") and metrics.token_usages:
                        for token_usage in metrics.token_usages:
                            if hasattr(token_usage, "prompt_tokens") and hasattr(
                                token_usage, "completion_tokens"
                            ):
                                total_prompt_tokens += token_usage.prompt_tokens
                                total_completion_tokens += token_usage.completion_tokens
                answer_data["prompt_tokens"] = total_prompt_tokens
                answer_data["completion_tokens"] = total_completion_tokens
                answer_data["token_cost"] = (
                    total_prompt_tokens + total_completion_tokens
                )

        if not answer_data["answer"]:
            events = list(state.events)
            for event in reversed(events):
                if (
                    isinstance(event, MessageEvent)
                    and hasattr(event, "source")
                    and event.source == "agent"
                ):
                    if hasattr(event, "extended_content") and event.extended_content:
                        text_content = []
                        for item in event.extended_content:
                            if hasattr(item, "text"):
                                text_content.append(item.text)
                            elif isinstance(item, str):
                                text_content.append(item)
                        if text_content:
                            answer_data["answer"] = "\n".join(text_content)
                            break
                    elif hasattr(event, "llm_message") and event.llm_message:
                        msg = event.llm_message
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, list):
                                text_content = []
                                for item in msg.content:
                                    if hasattr(item, "text"):
                                        text_content.append(item.text)
                                    elif isinstance(item, str):
                                        text_content.append(item)
                                if text_content:
                                    answer_data["answer"] = "\n".join(text_content)
                                    break
                            elif isinstance(msg.content, str):
                                answer_data["answer"] = msg.content
                                break

        if not answer_data["answer"]:
            events = list(state.events)
            action_count = sum(1 for event in events if isinstance(event, ActionEvent))

            print(
                f"[DEBUG] No answer found, ActionEvent count: {action_count}, max limit: {MAX_ITERATION_PER_RUN}"
            )

            if action_count >= MAX_ITERATION_PER_RUN:
                print(
                    f"[DEBUG] Reached max iteration limit ({MAX_ITERATION_PER_RUN}), generating answer from message_history..."
                )

                if message_history and len(message_history) > 0:
                    forced_answer, (forced_prompt_tokens, forced_completion_tokens) = (
                        generate_answer_from_history(
                            LLM_CONFIG, question, message_history
                        )
                    )

                    if forced_answer:
                        answer_data["answer"] = forced_answer
                        previous_prompt = answer_data.get("prompt_tokens", 0)
                        previous_completion = answer_data.get("completion_tokens", 0)
                        answer_data["prompt_tokens"] = (
                            previous_prompt + forced_prompt_tokens
                        )
                        answer_data["completion_tokens"] = (
                            previous_completion + forced_completion_tokens
                        )
                        answer_data["token_cost"] = (
                            answer_data["prompt_tokens"]
                            + answer_data["completion_tokens"]
                        )
                        print(
                            f"[DEBUG] Answer generated from message_history, length: {len(answer_data['answer'])} chars"
                        )
                        print(
                            f"[DEBUG] Token stats: prompt={previous_prompt}+{forced_prompt_tokens}={answer_data['prompt_tokens']}, completion={previous_completion}+{forced_completion_tokens}={answer_data['completion_tokens']}, total={answer_data['token_cost']}"
                        )
                    else:
                        print(f"[DEBUG] Warning: Failed to generate answer from message_history")
                        answer_data["answer"] = (
                            "Unable to generate answer based on conversation history."
                        )
                else:
                    print(f"[DEBUG] Warning: message_history is empty, cannot generate answer")
                    answer_data["answer"] = (
                        "No conversation history available to generate answer."
                    )

        if traj_dir and repo_name and question_idx is not None:
            save_trajectory(
                conversation, question, repo_name, question_idx, traj_dir, append=False
            )

        return answer_data

    except Exception as e:
        print(f"Failed to process question: {question[:50]}... error: {e}")
        answer_data["answer"] = f"Error: {str(e)}"
        if traj_dir and repo_name and question_idx is not None:
            try:
                save_trajectory(
                    conversation,
                    question,
                    repo_name,
                    question_idx,
                    traj_dir,
                    append=False,
                )
            except:
                pass
        return answer_data
    finally:
        safe_close_conversation(conversation)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(TRAJ_DIR, exist_ok=True)
    config_dir = os.path.join(TRAJ_DIR, "configs")
    prompt_dir = os.path.join(TRAJ_DIR, "prompts")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    all_processed_count = 0
    all_error_count = 0
    all_time_costs = []
    all_token_costs = []
    all_prompt_tokens_list = []
    all_completion_tokens_list = []

    for repo_idx, repo_config in enumerate(REPOS_CONFIG, 1):
        repo_name = repo_config["name"]
        workspace = repo_config["workspace"]
        input_file = repo_config["input_file"]

        print(f"\n{'=' * 60}")
        print(f"Processing repo {repo_idx}/{len(REPOS_CONFIG)}: {repo_name}")
        print(f"{'=' * 60}")
        print(f"Workspace: {workspace}")
        print(f"Questions file: {input_file}")

        if not os.path.exists(input_file):
            print(f"Warning: input file not found, skipping: {input_file}")
            continue

        output_file = os.path.join(OUTPUT_DIR, f"{repo_name}_answers.jsonl")

        repo_traj_dir = os.path.join(TRAJ_DIR, repo_name)
        os.makedirs(repo_traj_dir, exist_ok=True)

        save_experiment_config(config_dir, repo_name)

        answered_questions = load_answered_questions(output_file)
        if answered_questions:
            print(f"Found {len(answered_questions)} already-answered questions")

        print(f"Loading questions from {input_file}...")
        all_questions = load_questions_from_jsonl(input_file)
        print(f"Loaded {len(all_questions)} questions")

        questions = [
            qa_data
            for qa_data in all_questions
            if qa_data.get("question", "") not in answered_questions
        ]

        if len(questions) < len(all_questions):
            print(f"After filtering: {len(questions)} unanswered questions remaining")
        else:
            print(f"All questions unanswered, processing all {len(questions)} questions")

        if len(questions) == 0:
            print(f"Repo {repo_name} has no questions to process, skipping")
            continue

        processed_count = 0
        error_count = 0
        time_costs = []
        token_costs = []
        prompt_tokens_list = []
        completion_tokens_list = []

        for idx, qa_data in enumerate(questions, 1):
            try:
                result = process_single_question(
                    qa_data,
                    workspace,
                    repo_name=repo_name,
                    question_idx=idx,
                    traj_dir=repo_traj_dir,
                    prompt_dir=prompt_dir,
                )
                if result:
                    with open(output_file, "a", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")

                    time_cost = result.get("time_cost", 0)
                    token_cost = result.get("token_cost", 0)
                    prompt_tokens = result.get("prompt_tokens", 0)
                    completion_tokens = result.get("completion_tokens", 0)

                    if time_cost is not None:
                        time_costs.append(time_cost)
                        all_time_costs.append(time_cost)
                    if token_cost is not None:
                        token_costs.append(token_cost)
                        all_token_costs.append(token_cost)
                    if prompt_tokens is not None:
                        prompt_tokens_list.append(prompt_tokens)
                        all_prompt_tokens_list.append(prompt_tokens)
                    if completion_tokens is not None:
                        completion_tokens_list.append(completion_tokens)
                        all_completion_tokens_list.append(completion_tokens)

                    processed_count += 1
                    all_processed_count += 1
                    print(
                        f"[{repo_name}][{idx}/{len(questions)}] Done: {result['question'][:50]}..."
                    )
                else:
                    error_count += 1
                    all_error_count += 1
            except Exception as e:
                error_count += 1
                all_error_count += 1
                print(
                    f"[{repo_name}][{idx}/{len(questions)}] Failed: {qa_data.get('question', '')[:50]}... error: {e}"
                )

        avg_time_cost = sum(time_costs) / len(time_costs) if time_costs else 0
        avg_token_cost = sum(token_costs) / len(token_costs) if token_costs else 0
        total_time_cost = sum(time_costs)
        total_token_cost = sum(token_costs)

        avg_prompt_tokens = (
            sum(prompt_tokens_list) / len(prompt_tokens_list)
            if prompt_tokens_list
            else 0
        )
        avg_completion_tokens = (
            sum(completion_tokens_list) / len(completion_tokens_list)
            if completion_tokens_list
            else 0
        )
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)

        print(f"\nRepo {repo_name} done:")
        print(f"  Success: {processed_count}, Failed: {error_count}, Total: {len(questions)}")
        print(f"  Avg time_cost: {avg_time_cost:.2f}s")
        print(f"  Total time_cost: {total_time_cost:.2f}s")
        print(f"  Avg token_cost: {avg_token_cost:.0f} tokens")
        print(f"  Total token_cost: {total_token_cost:.0f} tokens")
        print(
            f"  Avg prompt_tokens: {avg_prompt_tokens:.0f}, completion_tokens: {avg_completion_tokens:.0f}"
        )
        print(
            f"  Total prompt_tokens: {total_prompt_tokens:.0f}, completion_tokens: {total_completion_tokens:.0f}"
        )
        print(f"  Results saved to: {output_file}")

    avg_time_cost = sum(all_time_costs) / len(all_time_costs) if all_time_costs else 0
    avg_token_cost = (
        sum(all_token_costs) / len(all_token_costs) if all_token_costs else 0
    )
    total_time_cost = sum(all_time_costs)
    total_token_cost = sum(all_token_costs)

    avg_prompt_tokens = (
        sum(all_prompt_tokens_list) / len(all_prompt_tokens_list)
        if all_prompt_tokens_list
        else 0
    )
    avg_completion_tokens = (
        sum(all_completion_tokens_list) / len(all_completion_tokens_list)
        if all_completion_tokens_list
        else 0
    )
    total_prompt_tokens = sum(all_prompt_tokens_list)
    total_completion_tokens = sum(all_completion_tokens_list)

    print(f"\n{'=' * 60}")
    print(f"All repos done!")
    print(f"{'=' * 60}")
    print(f"Success: {all_processed_count}, Failed: {all_error_count}")
    print(f"\nGlobal stats:")
    print(f"  Avg time_cost: {avg_time_cost:.2f}s")
    print(f"  Total time_cost: {total_time_cost:.2f}s")
    print(f"\nToken stats (total):")
    print(f"  Avg token_cost: {avg_token_cost:.0f} tokens")
    print(f"  Total token_cost: {total_token_cost:.0f} tokens")
    print(f"\nToken stats (split):")
    print(f"  Avg prompt_tokens (input): {avg_prompt_tokens:.0f} tokens")
    print(f"  Avg completion_tokens (output): {avg_completion_tokens:.0f} tokens")
    print(f"  Total prompt_tokens (input): {total_prompt_tokens:.0f} tokens")
    print(f"  Total completion_tokens (output): {total_completion_tokens:.0f} tokens")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
