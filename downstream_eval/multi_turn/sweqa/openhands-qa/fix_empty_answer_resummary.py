import os

os.environ.setdefault("LITELLM_DROP_PARAMS", "true")

import json
import time
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
import concurrent
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
import openai
from typing import Optional, List, Union, Any, Tuple, Dict
import argparse

import httpx


def setup_http_hack_for_claude():
    """Set up HTTP request layer interception to fix Claude API parameter conflicts."""
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
                    f"[HTTP HACK] Removed conflicting max_tokens parameter (URL: {url_str[:50]}...)"
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
            """Intercept synchronous HTTP requests to fix Claude API parameter conflicts."""
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

                                print(f"[HTTP HACK ERROR] Call stack:")
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
            """Intercept synchronous POST requests."""
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
                                    f"[HTTP HACK ERROR] ========== non-Anthropic POST request returned None =========="
                                )
                                print(f"[HTTP HACK ERROR] URL: {url_str}")
                                print(
                                    f"[HTTP HACK ERROR] kwargs keys: {list(kwargs.keys())}"
                                )
                                import traceback

                                print(f"[HTTP HACK ERROR] Call stack:")
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
            """Intercept asynchronous HTTP requests to fix Claude API parameter conflicts."""
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
            """Intercept asynchronous POST requests."""
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
                                            f"[HTTP HACK ERROR] Failed to build request: {build_error}"
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
                                f"[HTTP HACK ERROR] ========== Failed to send request =========="
                            )
                            print(f"[HTTP HACK ERROR] Error type: {type(send_error)}")
                            print(f"[HTTP HACK ERROR] Error message: {str(send_error)}")
                            print(f"[HTTP HACK ERROR] Request URL: {request.url}")
                            print(f"[HTTP HACK ERROR] Request method: {request.method}")
                            print(
                                f"[HTTP HACK ERROR] Request headers: {dict(request.headers)}"
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
        print(f"[WARNING] Failed to apply HTTP hack: {e}")
        import traceback

        traceback.print_exc()
        return False


_http_hack_enabled = setup_http_hack_for_claude()

from openhands.sdk import LLM, Conversation, Agent
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent
from openhands.sdk import TextContent, ImageContent
import sys

load_dotenv()
experiment = os.getenv("EXPERIMENT_TYPE")
api_type = os.getenv("API_TYPE")
if api_type not in ["openai", "azure"]:
    print(f"Unsupported api type: {api_type}")
    exit(1)
MAX_WORKERS = 8


AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENHANDS_MODEL_NAME")
if api_type == "azure":
    LLM_CONFIG = {
        "model": model_name,
        "api_key": AZURE_API_KEY,
        "base_url": AZURE_ENDPOINT,
        "api_version": AZURE_API_VERSION,
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


def extract_message_history_from_traj(traj_file: str) -> Optional[List[Dict]]:
    """Extract message history from a trajectory file."""
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            traj_data = json.load(f)

        events = traj_data.get("events", [])
        message_history = []

        for event in events:
            event_data = {}
            if event.get("type") == "MessageEvent":
                source = event.get("source", "")
                content_list = event.get("extended_content", [])

                if content_list:
                    text_parts = []
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "\n".join(text_parts)
                else:
                    content = ""

                if source == "user":
                    event_data["role"] = "user"
                    event_data["content"] = content
                elif source == "agent":
                    event_data["role"] = "agent"
                    event_data["content"] = content
                else:
                    continue

                if event_data and event_data.get("content"):
                    message_history.append(event_data)

            elif event.get("type") == "ObservationEvent":
                observation = event.get("observation", "")
                if observation:
                    message_history.append({"role": "tool", "content": observation})

        return message_history if message_history else None
    except Exception as e:
        print(f"Error reading trajectory file {traj_file}: {e}")
        return None


def generate_answer_from_history(
    llm_config, question: str, message_history: List[Dict]
) -> tuple:
    """Generate answer using LLM based on message history."""
    try:
        if api_type == "openai":
            client = openai.OpenAI(
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
            )
            model_name = llm_config["model"]
            if model_name.startswith("openai/"):
                model_name = model_name[7:]
        else:
            client = openai.AzureOpenAI(
                azure_endpoint=llm_config["base_url"],
                api_key=llm_config["api_key"],
                api_version=llm_config["api_version"],
            )
            model_name = llm_config["model"]
            if model_name.startswith("azure/"):
                model_name = model_name[6:]

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

        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0.3,
            top_p=None,
            extra_headers={"X-TT-LOGID": "${your_logid}"},
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return answer, (prompt_tokens, completion_tokens)
    except Exception as e:
        print(f"Error generating answer: {e}")
        import traceback

        traceback.print_exc()
        return None, (0, 0)


def find_traj_file(traj_dir: str, repo_name: str, question: str) -> Optional[str]:
    """Find the trajectory file corresponding to a given question and repo."""
    repo_traj_dir = os.path.join(traj_dir, repo_name)
    if not os.path.exists(repo_traj_dir):
        return None

    for filename in os.listdir(repo_traj_dir):
        if filename.endswith(".traj.json") and repo_name in filename:
            traj_file = os.path.join(repo_traj_dir, filename)
            try:
                with open(traj_file, "r", encoding="utf-8") as f:
                    traj_data = json.load(f)
                if traj_data.get("question") == question:
                    return traj_file
            except:
                continue

    return None


def load_answers_from_jsonl(answer_file: str) -> List[Dict]:
    """Load answers from a jsonl file."""
    answers = []
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        answers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return answers


def save_answers_to_jsonl(answer_file: str, answers: List[Dict]):
    """Save answers to a jsonl file."""
    with open(answer_file, "w", encoding="utf-8") as f:
        for answer in answers:
            json.dump(answer, f, ensure_ascii=False)
            f.write("\n")


def fix_empty_answer_resummary(
    traj_dir: str,
    answer_dir: str,
    dry_run: bool = False,
):
    """Re-summarize empty answers using trajectory message history."""

    answer_files = []
    for filename in os.listdir(answer_dir):
        if filename.endswith("_answers.jsonl"):
            answer_files.append(os.path.join(answer_dir, filename))

    if not answer_files:
        print(f"No answer files found in {answer_dir}")
        return

    total_fixed = 0
    total_checked = 0

    for answer_file in answer_files:
        print(f"ALL answer_files: {answer_files}")
        print(f"\nProcessing: {answer_file}")

        repo_name = os.path.basename(answer_file).replace("_answers.jsonl", "")

        answers = load_answers_from_jsonl(answer_file)
        print(f"  Loaded {len(answers)} answers")

        fixed_count = 0

        def process_answer(idx, answer, repo_name, traj_dir, LLM_CONFIG, dry_run):
            question = answer.get("question", "")
            current_answer = answer.get("answer", "")

            if (
                not current_answer
                or current_answer.startswith("Error:")
                or current_answer.startswith("Timeout:")
            ):
                traj_file = find_traj_file(traj_dir, repo_name, question)

                if traj_file:
                    message_history = extract_message_history_from_traj(traj_file)

                    if message_history:
                        if dry_run:
                            print(f"  [DRY RUN] Would regenerate answer: {question[:50]}...")
                            return False, answer
                        else:
                            print(f"  Generating answer ({idx + 1}): {question[:50]}...")
                            new_answer, (prompt_tokens, completion_tokens) = (
                                generate_answer_from_history(
                                    LLM_CONFIG, question, message_history
                                )
                            )

                            if new_answer:
                                old_answer = current_answer
                                answer["answer"] = new_answer
                                answer["prompt_tokens"] = (
                                    answer.get("prompt_tokens", 0) + prompt_tokens
                                )
                                answer["completion_tokens"] = (
                                    answer.get("completion_tokens", 0)
                                    + completion_tokens
                                )
                                answer["token_cost"] = (
                                    answer["prompt_tokens"]
                                    + answer["completion_tokens"]
                                )

                                print(f"    Answer generated successfully ({len(new_answer)} chars)")
                                print(
                                    f"    Old answer: {old_answer[:100] if old_answer else '(empty)'}"
                                )
                                print(f"    New answer: {new_answer[:100]}...")
                                return True, answer
                            else:
                                print(f"    Failed to generate answer")
                                return False, answer
                    else:
                        print(f"  No message history found: {question[:50]}...")
                        return False, answer
                else:
                    print(f"  Trajectory file not found: {question[:50]}...")
                    return False, answer

            return False, answer

        total_checked = 0
        fixed_count = 0
        total_fixed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            process_func = partial(
                process_answer,
                repo_name=repo_name,
                traj_dir=traj_dir,
                LLM_CONFIG=LLM_CONFIG,
                dry_run=dry_run,
            )

            futures = []
            for idx, answer in enumerate(answers):
                total_checked += 1
                futures.append(executor.submit(process_func, idx, answer))

            for future in concurrent.futures.as_completed(futures):
                is_fixed, updated_answer = future.result()
                if is_fixed:
                    fixed_count += 1
                    total_fixed += 1
                    for i, ans in enumerate(answers):
                        if ans.get("question") == updated_answer.get("question"):
                            answers[i] = updated_answer
                            break

        if fixed_count > 0 and not dry_run:
            save_answers_to_jsonl(answer_file, answers)
            print(f"  Saved {fixed_count} fixed answers to {answer_file}")
    print(f"\n{'=' * 60}")
    print(f"Regeneration done!")
    print(f"  Checked: {total_checked}")
    print(f"  Fixed: {total_fixed}")
    if dry_run:
        print(f"  [DRY RUN mode, no files modified]")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract message history from trajectory files to regenerate empty answers"
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="./trajectories",
        help="Trajectory files directory (default: ./trajectories)",
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="./answer/openhands",
        help="Answer files directory (default: ./answer/openhands)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: only show what would be fixed, do not modify files",
    )

    args = parser.parse_args()

    if not os.path.exists(args.traj_dir):
        print(f"Error: trajectory directory not found: {args.traj_dir}")
        return

    if not os.path.exists(args.answer_dir):
        print(f"Error: answer directory not found: {args.answer_dir}")
        return

    fix_empty_answer_resummary(args.traj_dir, args.answer_dir, args.dry_run)


if __name__ == "__main__":
    main()
