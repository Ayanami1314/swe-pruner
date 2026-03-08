import os

# 修复 Claude API 的 max_tokens 冲突问题：在导入之前设置环境变量
# LiteLLM 会将 max_output_tokens 同时映射为 max_tokens 和 max_completion_tokens
# 但 Anthropic API 不支持同时设置这两个参数
# 设置环境变量让 LiteLLM 正确处理参数
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

# 在导入 httpx 后立即设置 HTTP hack（必须在导入 OpenHands SDK 之前）
import httpx


# 全局 HTTP hack 设置（在导入 OpenHands SDK 之前执行）
def setup_http_hack_for_claude():
    """设置 HTTP 请求层拦截，修复 Claude API 的参数冲突"""
    try:
        # 保存原始的 httpx 请求方法（同步和异步）
        if not hasattr(httpx.Client, "_original_request"):
            print(f"[HTTP HACK DEBUG] ========== 保存原始方法 ==========")
            print(
                f"[HTTP HACK DEBUG] httpx.Client.request 类型: {type(httpx.Client.request)}"
            )
            print(
                f"[HTTP HACK DEBUG] httpx.Client.post 类型: {type(httpx.Client.post)}"
            )
            httpx.Client._original_request = httpx.Client.request
            httpx.Client._original_post = httpx.Client.post
            print(
                f"[HTTP HACK DEBUG] 已保存 _original_request: {type(httpx.Client._original_request)}"
            )
            print(
                f"[HTTP HACK DEBUG] 已保存 _original_post: {type(httpx.Client._original_post)}"
            )
            print(
                f"[HTTP HACK DEBUG] _original_request 是否为方法: {callable(httpx.Client._original_request)}"
            )
            print(
                f"[HTTP HACK DEBUG] _original_post 是否为方法: {callable(httpx.Client._original_post)}"
            )

        if not hasattr(httpx.AsyncClient, "_original_request"):
            httpx.AsyncClient._original_request = httpx.AsyncClient.request
            httpx.AsyncClient._original_post = httpx.AsyncClient.post

        def fix_json_body(json_data, url_str):
            """修复 JSON 请求体中的参数冲突"""
            if not isinstance(json_data, dict):
                return False

            fixed = False
            # 如果同时存在 max_tokens 和 max_completion_tokens，移除 max_tokens
            if "max_tokens" in json_data and "max_completion_tokens" in json_data:
                json_data.pop("max_tokens", None)
                print(
                    f"[HTTP HACK] 移除了冲突的 max_tokens 参数 (URL: {url_str[:50]}...)"
                )
                fixed = True
            # 如果只有 max_tokens，转换为 max_completion_tokens
            elif "max_tokens" in json_data:
                max_tokens_val = json_data.pop("max_tokens")
                if "max_completion_tokens" not in json_data:
                    json_data["max_completion_tokens"] = max_tokens_val
                    print(
                        f"[HTTP HACK] 将 max_tokens={max_tokens_val} 转换为 max_completion_tokens (URL: {url_str[:50]}...)"
                    )
                    fixed = True
            return fixed

        def patched_request_sync(self, method, url, **kwargs):
            """拦截同步 HTTP 请求，修复 Claude API 的参数冲突"""
            url_str = str(url)
            # 检查是否是 Anthropic API 请求
            if "anthropic.com" in url_str.lower():
                # 修改请求体
                if "json" in kwargs:
                    fix_json_body(kwargs["json"], url_str)
                # 也检查 content 参数（某些情况下可能使用 content）
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
                        print(f"[HTTP HACK ERROR] 解析 content 失败: {e}")
                        import traceback

                        traceback.print_exc()
                # 调用原始请求方法
                if hasattr(httpx.Client, "_original_request"):
                    response = httpx.Client._original_request(
                        self, method, url, **kwargs
                    )
                else:
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )
                # 确保返回了有效的 response
                if response is None:
                    raise RuntimeError(
                        f"HTTP request returned None for {method} {url_str}"
                    )
                return response
            else:
                # 对于非 Anthropic 请求，直接调用原始方法，避免不必要的拦截
                if hasattr(httpx.Client, "_original_request"):
                    try:
                        # 尝试使用描述符协议获取绑定方法
                        if hasattr(httpx.Client._original_request, "__get__"):
                            bound_method = httpx.Client._original_request.__get__(
                                self, httpx.Client
                            )
                            response = bound_method(method, url, **kwargs)
                        else:
                            # 直接调用（作为未绑定方法）
                            response = httpx.Client._original_request(
                                self, method, url, **kwargs
                            )

                        # 如果第一次尝试返回 None，使用 send 方法作为备选方案
                        if response is None:
                            # 备选方案：使用 send 方法
                            if hasattr(httpx.Client, "_original_send"):
                                import httpx as httpx_module
                                import json as json_module

                                # 构建 Request 对象，正确处理 json 参数
                                # send 方法不接受 json 参数，需要将其转换为 content
                                send_kwargs = {}
                                request_kwargs = {}

                                # 分离 send 接受的参数和 Request 构建参数
                                for key, value in kwargs.items():
                                    if key in [
                                        "timeout",
                                        "follow_redirects",
                                        "extensions",
                                    ]:
                                        send_kwargs[key] = value
                                    elif key == "json":
                                        # json 需要转换为 content
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

                                # 如果没有 content，使用默认的 headers
                                if "headers" not in request_kwargs:
                                    request_kwargs["headers"] = {}

                                request = httpx_module.Request(
                                    method, url, **request_kwargs
                                )
                                response = httpx.Client._original_send(
                                    self, request, **send_kwargs
                                )
                            else:
                                print(f"[HTTP HACK ERROR] _original_send 也不存在!")

                            if response is None:
                                print(
                                    f"[HTTP HACK ERROR] ========== patched_request_sync: 非 Anthropic 请求返回 None =========="
                                )
                                print(
                                    f"[HTTP HACK ERROR] Method: {method}, URL: {url_str}"
                                )
                                print(
                                    f"[HTTP HACK ERROR] kwargs keys: {list(kwargs.keys())}"
                                )
                                import traceback

                                print(f"[HTTP HACK ERROR] 调用堆栈:")
                                traceback.print_stack()
                                raise RuntimeError(
                                    f"HTTP request returned None for {method} {url_str}"
                                )

                        # 确保返回了有效的 response
                        if response is None:
                            raise RuntimeError(
                                f"HTTP request returned None for {method} {url_str}"
                            )
                        return response
                    except Exception as e:
                        print(
                            f"[HTTP HACK ERROR] ========== patched_request_sync: 调用 _original_request 时发生异常 =========="
                        )
                        print(f"[HTTP HACK ERROR] 异常类型: {type(e)}")
                        print(f"[HTTP HACK ERROR] 异常消息: {str(e)}")
                        print(f"[HTTP HACK ERROR] Method: {method}, URL: {url_str}")
                        import traceback

                        print(f"[HTTP HACK ERROR] 完整堆栈:")
                        traceback.print_exc()
                        raise
                else:
                    print(f"[HTTP HACK ERROR] _original_request 不存在!")
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )

        def patched_post_sync(self, url, **kwargs):
            """拦截同步 POST 请求"""
            url_str = str(url)
            # 检查是否是 Anthropic API 请求
            if "anthropic.com" in url_str.lower():
                # 对于 Anthropic 请求，使用 patched_request_sync 来处理参数修复
                return patched_request_sync(self, "POST", url, **kwargs)
            else:
                # 对于非 Anthropic 请求，直接调用原始 request 方法（POST 方法内部会调用 request）
                # 这样可以避免 _original_post 内部调用已经被替换的 request 方法
                # 尝试多种方式调用原始方法
                if hasattr(httpx.Client, "_original_request"):
                    try:
                        # 方法1：尝试使用描述符协议获取绑定方法
                        if hasattr(httpx.Client._original_request, "__get__"):
                            bound_method = httpx.Client._original_request.__get__(
                                self, httpx.Client
                            )
                            response = bound_method("POST", url, **kwargs)
                        else:
                            # 方法2：直接调用（作为未绑定方法）
                            response = httpx.Client._original_request(
                                self, "POST", url, **kwargs
                            )

                        # 如果第一次尝试返回 None，使用 send 方法作为备选方案
                        if response is None:
                            # 备选方案：使用 send 方法
                            if hasattr(httpx.Client, "_original_send"):
                                import httpx as httpx_module
                                import json as json_module

                                # 构建 Request 对象，正确处理 json 参数
                                # send 方法不接受 json 参数，需要将其转换为 content
                                send_kwargs = {}
                                request_kwargs = {}

                                # 分离 send 接受的参数和 Request 构建参数
                                for key, value in kwargs.items():
                                    if key in [
                                        "timeout",
                                        "follow_redirects",
                                        "extensions",
                                    ]:
                                        send_kwargs[key] = value
                                    elif key == "json":
                                        # json 需要转换为 content
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

                                # 如果没有 content，使用默认的 headers
                                if "headers" not in request_kwargs:
                                    request_kwargs["headers"] = {}

                                request = httpx_module.Request(
                                    "POST", url, **request_kwargs
                                )
                                response = httpx.Client._original_send(
                                    self, request, **send_kwargs
                                )
                            else:
                                print(f"[HTTP HACK ERROR] _original_send 也不存在!")

                            if response is None:
                                print(
                                    f"[HTTP HACK ERROR] ========== 非 Anthropic POST 请求返回 None =========="
                                )
                                print(f"[HTTP HACK ERROR] URL: {url_str}")
                                print(
                                    f"[HTTP HACK ERROR] kwargs keys: {list(kwargs.keys())}"
                                )
                                import traceback

                                print(f"[HTTP HACK ERROR] 调用堆栈:")
                                traceback.print_stack()
                                raise RuntimeError(
                                    f"HTTP POST request returned None for {url_str}"
                                )

                        return response
                    except Exception as e:
                        print(
                            f"[HTTP HACK ERROR] ========== 调用 _original_request 时发生异常 =========="
                        )
                        print(f"[HTTP HACK ERROR] 异常类型: {type(e)}")
                        print(f"[HTTP HACK ERROR] 异常消息: {str(e)}")
                        print(f"[HTTP HACK ERROR] URL: {url_str}")
                        import traceback

                        print(f"[HTTP HACK ERROR] 完整堆栈:")
                        traceback.print_exc()
                        raise
                else:
                    print(f"[HTTP HACK ERROR] _original_request 不存在!")
                    raise RuntimeError(
                        "HTTP hack: _original_request not found, patch may have failed"
                    )

        async def patched_request_async(self, method, url, **kwargs):
            """拦截异步 HTTP 请求，修复 Claude API 的参数冲突"""
            url_str = str(url)
            # 检查是否是 Anthropic API 请求
            if "anthropic.com" in url_str.lower():
                # 修改请求体
                if "json" in kwargs:
                    fix_json_body(kwargs["json"], url_str)
                # 也检查 content 参数
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
                        print(f"[HTTP HACK ERROR] 解析 content 失败: {e}")
                        import traceback

                        traceback.print_exc()

            # 调用原始请求方法
            return await httpx.AsyncClient._original_request(
                self, method, url, **kwargs
            )

        async def patched_post_async(self, url, **kwargs):
            """拦截异步 POST 请求"""
            return await patched_request_async(self, "POST", url, **kwargs)

        # 替换 httpx Client 的方法（同步和异步）
        httpx.Client.request = patched_request_sync
        httpx.Client.post = patched_post_sync
        httpx.AsyncClient.request = patched_request_async
        httpx.AsyncClient.post = patched_post_async

        # 尝试拦截更底层的方法：send（如果存在）
        try:
            if hasattr(httpx.Client, "send") and not hasattr(
                httpx.Client, "_original_send"
            ):
                httpx.Client._original_send = httpx.Client.send

                def patched_send_sync(self, request, **kwargs):
                    """拦截 send 方法，修复请求体"""
                    url_str = str(request.url)
                    if "anthropic.com" in url_str.lower():
                        # 尝试修改请求体
                        if hasattr(request, "content") and request.content:
                            try:
                                if isinstance(request.content, bytes):
                                    content_str = request.content.decode("utf-8")
                                else:
                                    content_str = str(request.content)

                                json_data = json.loads(content_str)

                                if fix_json_body(json_data, url_str):
                                    # 使用更安全的方式修改请求内容
                                    try:
                                        # 尝试直接设置 content
                                        new_content = json.dumps(json_data).encode(
                                            "utf-8"
                                        )
                                        # httpx Request 是不可变的，需要重新构建
                                        import httpx as httpx_module

                                        # 获取所有原始属性
                                        headers = dict(request.headers)
                                        # 删除旧的 Content-Length（大小写不敏感，需要删除所有变体）
                                        # httpx Headers 是大小写不敏感的，所以需要删除所有可能的变体
                                        headers_to_remove = []
                                        for key in headers.keys():
                                            if key.lower() == "content-length":
                                                headers_to_remove.append(key)
                                        for key in headers_to_remove:
                                            del headers[key]
                                        # 设置新的 Content-Length（使用小写，httpx 会自动处理）
                                        headers["content-length"] = str(
                                            len(new_content)
                                        )

                                        # 重新构建请求，保留所有原始属性
                                        # 获取原始 extensions（如果存在）
                                        original_extensions = {}
                                        if (
                                            hasattr(request, "extensions")
                                            and request.extensions
                                        ):
                                            original_extensions = dict(
                                                request.extensions
                                            )

                                        # 创建新请求
                                        new_request = httpx_module.Request(
                                            request.method,
                                            request.url,
                                            content=new_content,
                                            headers=headers,
                                        )

                                        # 复制 extensions 到新请求
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
                                            f"[HTTP HACK ERROR] 构建请求失败: {build_error}"
                                        )
                                        import traceback

                                        traceback.print_exc()
                                        # 如果构建失败，使用原始请求（至少参数已修复）
                            except Exception as e:
                                print(f"[HTTP HACK ERROR] send 方法修复失败: {e}")
                                import traceback

                                traceback.print_exc()

                        # 尝试发送请求并捕获错误
                        try:
                            response = httpx.Client._original_send(
                                self, request, **kwargs
                            )
                            return response
                        except Exception as send_error:
                            print(
                                f"[HTTP HACK ERROR] ========== 发送请求失败 =========="
                            )
                            print(f"[HTTP HACK ERROR] 错误类型: {type(send_error)}")
                            print(f"[HTTP HACK ERROR] 错误消息: {str(send_error)}")
                            print(f"[HTTP HACK ERROR] 请求 URL: {request.url}")
                            print(f"[HTTP HACK ERROR] 请求 Method: {request.method}")
                            print(
                                f"[HTTP HACK ERROR] 请求 Headers: {dict(request.headers)}"
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
                                        f"[HTTP HACK ERROR] 请求体预览: {content_preview}"
                                    )
                                except:
                                    print(f"[HTTP HACK ERROR] 无法预览请求体")
                            import traceback

                            print(f"[HTTP HACK ERROR] 完整错误堆栈:")
                            traceback.print_exc()
                            raise

                httpx.Client.send = patched_send_sync
                print("[HTTP HACK] 已启用 send 方法拦截")
        except Exception as e:
            print(f"[HTTP HACK DEBUG] 无法拦截 send 方法: {e}")

        print("[HTTP HACK] 已启用 HTTP 请求层拦截修复（同步+异步）")
        return True
    except Exception as e:
        print(f"[WARNING] 无法应用 HTTP hack: {e}")
        import traceback

        traceback.print_exc()
        return False


# 立即设置 HTTP hack（在导入 OpenHands SDK 之前）
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
        "usage_id": "agent",  # 使用 usage_id 替代已弃用的 service_id
        "extra_headers": {"X-TT-LOGID": "${your_logid}"},
    }
elif api_type == "openai":
    LLM_CONFIG = {
        "model": model_name,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
    }
    # 修复 Claude API 的 max_tokens 冲突问题
    # LiteLLM 会将 max_output_tokens 同时映射为 max_tokens 和 max_completion_tokens
    # 但 Anthropic API 不支持同时设置这两个参数
    #
    # 注意：OpenHands SDK 的 LLM 类不支持 additional_drop_params 参数
    # 只能通过 drop_params=True 让 LiteLLM 自动处理，或通过 monkey patch 修复
    if "claude" in model_name.lower():
        # 显式设置 drop_params=True，让 LiteLLM 自动丢弃不支持的参数
        # 但 drop_params 可能不够，还需要配合 monkey patch
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
TRAJ_DIR = os.getenv("TRAJ_OUTPUT_PATH", "./trajectories")  # 轨迹文件目录
MAX_ITERATION_PER_RUN = int(os.getenv("MAX_ITERATION_PER_RUN", "50"))
MAX_TIME_PER_QUESTION = int(
    os.getenv("MAX_TIME_PER_QUESTION", "1800")
)  # 每个问题的最大处理时间（秒）


# 从输入文件名提取仓库名
def get_repo_name_from_path(file_path):
    """从文件路径提取仓库名"""
    basename = os.path.basename(file_path)
    # 去掉 .jsonl 后缀
    if basename.endswith(".jsonl"):
        return basename[:-6]
    return basename


def load_questions_from_jsonl(file_path):
    """从 jsonl 文件加载问题"""
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
                print(f"跳过无效的 JSON 行: {e}")
    return questions


def load_answered_questions(output_file):
    """从输出文件中加载已经回答过的问题"""
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
            print(f"读取已回答问题时出错: {e}")
    return answered_questions


def get_message_history(state):
    """从 conversation state 中提取完整的 message history"""
    messages = []
    events = list(state.events)

    print(f"[DEBUG] 总共有 {len(events)} 个事件")

    for idx, event in enumerate(events):
        event_type = type(event).__name__

        # 处理 MessageEvent
        if isinstance(event, MessageEvent):
            source = getattr(event, "source", "unknown")

            # 提取消息内容
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
                    f"[DEBUG Event {idx}] MessageEvent, source={source}, content长度={len(content)}"
                )

        # 处理 ActionEvent (agent 执行的动作)
        elif isinstance(event, ActionEvent):
            action_info = ""
            if hasattr(event, "action") and event.action:
                action = event.action
                action_name = getattr(action, "name", "unknown")
                action_info = f"Action: {action_name}"

                # 尝试获取动作的参数
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
                    f"[DEBUG Event {idx}] ActionEvent, content长度={len(action_info)}"
                )

        # 处理 ObservationEvent (工具执行的结果)
        elif isinstance(event, ObservationEvent):
            observation_info = ""
            if hasattr(event, "observation") and event.observation:
                observation = event.observation
                # 尝试获取观察结果
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
                    f"[DEBUG Event {idx}] ObservationEvent, content长度={len(observation_info)}"
                )
        else:
            # 其他类型的事件，也尝试提取信息
            print(f"[DEBUG Event {idx}] 其他事件类型: {event_type}")

    print(f"[DEBUG] 提取到 {len(messages)} 条消息")
    return messages


def make_json_serializable(obj):
    """递归地将对象转换为 JSON 可序列化的格式"""
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
        # 尝试转换为字典
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
    """序列化事件对象为字典"""
    event_dict = {
        "type": type(event).__name__,
        "id": getattr(event, "id", None),
        "timestamp": getattr(event, "timestamp", None),
        "source": getattr(event, "source", None),
    }

    # 根据事件类型添加特定字段
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
                    # 确保所有内容都是可序列化的
                    action_dict.update(make_json_serializable(dumped))
                except:
                    action_dict["str"] = str(action)
            event_dict["action"] = action_dict

    elif isinstance(event, ObservationEvent):
        if hasattr(event, "observation") and event.observation:
            observation = event.observation
            obs_dict = {}

            # 检查是否是 BashObservation 并包含 prune 信息
            if hasattr(observation, "pruned") and observation.pruned:
                # 保存完整的 prune 信息
                obs_dict["pruned"] = True
                obs_dict["output"] = getattr(observation, "output", "")
                obs_dict["original_output"] = getattr(
                    observation, "original_output", None
                )

                # 保存 prune_info 中的所有信息
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
                    # 计算 prune 统计信息
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
                # 非 prune 的普通观察
                # 优先使用 to_llm_content 获取内容
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

                # 如果没有通过 to_llm_content 获取到内容，尝试其他方式
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
                            # 确保所有内容都是可序列化的
                            obs_dict.update(make_json_serializable(dumped))
                        except:
                            obs_dict["str"] = str(observation)
                    else:
                        obs_dict["str"] = str(observation)
                obs_dict["pruned"] = False

            event_dict["observation"] = obs_dict

    else:
        # 其他类型的事件，尝试序列化所有属性
        try:
            if hasattr(event, "model_dump"):
                dumped = event.model_dump()
                # 确保所有内容都是可序列化的
                event_dict.update(make_json_serializable(dumped))
            else:
                event_dict["str"] = str(event)
        except:
            event_dict["str"] = str(event)

    # 最后确保整个字典都是可序列化的
    return make_json_serializable(event_dict)


def extract_prune_statistics(events):
    """从事件中提取 prune 统计信息"""
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

                        # 保存每次 prune 操作的详细信息
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

    # 计算平均压缩比
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
    """保存对话轨迹到 .traj.json 文件

    Args:
        conversation: Conversation 对象
        question: 问题
        repo_name: 仓库名
        question_idx: 问题索引
        traj_dir: 轨迹目录
        append: 是否追加模式（实时保存时使用）
    """
    try:
        if conversation is None or conversation.state is None:
            return

        state = conversation.state
        events = list(state.events)

        # 序列化所有事件
        serialized_events = [serialize_event(event) for event in events]

        # 提取 prune 统计信息（如果是 pruner 实验）
        prune_statistics = None
        if experiment == "pruner":
            prune_statistics = extract_prune_statistics(events)

        # 构建轨迹数据
        trajectory_data = {
            "question": question,
            "repo_name": repo_name,
            "question_idx": question_idx,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment,
            "events": serialized_events,
            "event_count": len(events),
        }

        # 如果是 pruner 实验，添加 prune 统计信息
        if prune_statistics:
            trajectory_data["prune_statistics"] = prune_statistics

        # 保存到文件
        traj_filename = f"{repo_name}_q{question_idx}.traj.json"
        traj_filepath = os.path.join(traj_dir, traj_filename)

        # 如果是追加模式，读取现有数据并更新
        if append and os.path.exists(traj_filepath):
            try:
                with open(traj_filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                # 更新事件列表
                existing_data["events"] = serialized_events
                existing_data["event_count"] = len(events)
                # 更新 prune 统计（如果是 pruner 实验）
                if prune_statistics:
                    existing_data["prune_statistics"] = prune_statistics
                trajectory_data = existing_data
            except:
                # 如果读取失败，使用新数据
                pass

        with open(traj_filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)

        if not append:  # 只在非追加模式时打印，避免日志过多
            print(f"[DEBUG] 轨迹已保存到: {traj_filepath}")

        # 如果是 pruner 实验，额外保存 prune 详细信息（只在最终保存时，避免频繁IO）
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

            # 从事件中提取每次 prune 的完整信息（包括原始输出和 prune 后输出）
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

            print(f"[DEBUG] Prune 详细信息已保存到: {prune_details_filepath}")

    except Exception as e:
        print(f"[DEBUG] 保存轨迹时出错: {e}")
        import traceback

        traceback.print_exc()


def save_experiment_config(config_dir, repo_name):
    """保存实验配置参数"""
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

        # 如果是 pruner 实验，添加 prune 配置
        if experiment == "pruner":
            # 从 tool_utils 导入 prune 配置
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

        print(f"[DEBUG] 实验配置已保存到: {config_filepath}")

    except Exception as e:
        print(f"[DEBUG] 保存实验配置时出错: {e}")


def save_prompts(prompt_dir, repo_name, question, enhanced_question):
    """保存 prompt 文件"""
    try:
        prompt_data = {
            "repo_name": repo_name,
            "question": question,
            "enhanced_question": enhanced_question,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment,
        }

        # 使用问题的哈希值作为文件名的一部分，避免文件名过长
        question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        prompt_filename = f"{repo_name}_{question_hash}_prompt.json"
        prompt_filepath = os.path.join(prompt_dir, prompt_filename)

        with open(prompt_filepath, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)

        print(f"[DEBUG] Prompt 已保存到: {prompt_filepath}")

    except Exception as e:
        print(f"[DEBUG] 保存 Prompt 时出错: {e}")


def generate_answer_from_history(llm_config, question, message_history):
    """基于 message_history 使用 LLM 生成答案"""
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

        # 将 message_history 格式化为文本格式，避免 tool role 的问题
        # 将所有消息合并为一个文本，而不是使用 tool role
        conversation_text = []

        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if content:
                # 将不同角色的消息格式化为文本
                if role == "user":
                    conversation_text.append(f"User: {content}")
                elif role == "agent":
                    conversation_text.append(f"Assistant: {content}")
                elif role == "tool":
                    conversation_text.append(f"Tool Output: {content}")
                else:
                    conversation_text.append(f"{role}: {content}")

        # 构建完整的提示
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

        print(f"[DEBUG] 调用 LLM 生成答案，对话历史长度: {len(full_conversation)} 字符")

        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=formatted_messages,
            temperature=0.3,
            extra_headers={"X-TT-LOGID": "${your_logid}"},
            stream=False,
        )  # stream=False for gemini3 bug

        answer = response.choices[0].message.content.strip()

        # 返回答案和 token 使用量（分别返回 prompt 和 completion）
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        print(f"[DEBUG] LLM 生成答案成功，长度: {len(answer)} 字符")
        print(
            f"[DEBUG] Token使用: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
        )

        return answer, (prompt_tokens, completion_tokens)
    except Exception as e:
        print(f"[DEBUG] 生成答案时出错: {e}")
        import traceback

        traceback.print_exc()
        return None, (0, 0)  # 返回元组格式以匹配解包期望


def safe_close_conversation(conversation):
    """安全地关闭 conversation，避免 agent 未初始化的错误"""
    if conversation is None:
        return
    try:
        conversation.close()
    except RuntimeError as e:
        # 捕获 "Agent not initialized" 错误并忽略
        if "not initialized" in str(e) or "Agent not initialized" in str(e):
            pass
        else:
            # 其他 RuntimeError 重新抛出
            raise
    except Exception:
        # 其他所有错误都忽略，避免影响主流程
        pass


def process_single_question(
    qa_data,
    workspace,
    repo_name=None,
    question_idx=None,
    traj_dir=None,
    prompt_dir=None,
):
    """处理单个问题"""
    question = qa_data.get("question", "")
    if not question:
        return None

    # HTTP hack 已经在导入时设置，这里只需要确保 LiteLLM monkey patch 作为备用
    if "claude" in model_name.lower() and not _http_hack_enabled:
        # 如果 HTTP hack 失败，降级到 LiteLLM monkey patch
        try:
            import litellm

            if not hasattr(litellm, "_original_completion"):
                litellm._original_completion = litellm.completion

            def patched_completion(*args, **kwargs):
                model = kwargs.get("model", args[0] if args else None)
                if model and "claude" in str(model).lower():
                    if "max_tokens" in kwargs and "max_completion_tokens" in kwargs:
                        kwargs.pop("max_tokens", None)
                        print("[LITELLM PATCH] 移除了冲突的 max_tokens 参数")
                    elif "max_tokens" in kwargs:
                        max_tokens_val = kwargs.pop("max_tokens")
                        if "max_completion_tokens" not in kwargs:
                            kwargs["max_completion_tokens"] = max_tokens_val
                            print(
                                f"[LITELLM PATCH] 将 max_tokens={max_tokens_val} 转换为 max_completion_tokens"
                            )
                return litellm._original_completion(*args, **kwargs)

            litellm.completion = patched_completion
            print("[FALLBACK] 已启用 LiteLLM 函数层拦截修复")
        except Exception as e2:
            print(f"[WARNING] 无法应用 LiteLLM patch: {e2}")

    # 为每个任务创建独立的 agent 和 conversation
    llm = LLM(**LLM_CONFIG)

    print(LLM_CONFIG)
    agent = Agent(llm=llm, tools=tools)

    # 用于存储答案的变量
    answer_data = {
        "question": question,
        "answer": "",
        "timestamp": datetime.now().isoformat(),
        "time_cost": 0.0,
        "token_cost": 0,  # 总 token 数（向后兼容）
        "prompt_tokens": 0,  # input tokens
        "completion_tokens": 0,  # output tokens
    }

    # 回调函数来捕获答案并实时保存轨迹
    def on_event(event):
        nonlocal answer_data, conversation
        # 处理 MessageEvent
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

        # 处理 ActionEvent，检查是否是 FinishAction
        elif isinstance(event, ActionEvent):
            if hasattr(event, "action") and event.action:
                action = event.action
                action_name = getattr(action, "name", None)
                # 检查是否是 FinishAction（通常名称是 "finish" 或类似）
                if action_name and (
                    "finish" in action_name.lower() or "final" in action_name.lower()
                ):
                    # 尝试从 action 中提取内容
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
                    # 如果 action 有 model_dump，尝试从中提取
                    elif hasattr(action, "model_dump"):
                        try:
                            action_dict = action.model_dump()
                            # 查找可能的答案字段
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

        # 实时保存轨迹（每次事件后都保存）
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
                # 静默失败，避免影响主流程
                pass

    conversation = None
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            callbacks=[on_event],
        )

        # 记录开始时间
        start_time = time.time()

        # 添加探索提示
        enhanced_question = f"""Please first explore the codebase structure to find the relevant files.
Use the terminal tool to search for files related to the question.
Then answer: {question}"""

        # 保存 prompt
        if prompt_dir and repo_name:
            save_prompts(prompt_dir, repo_name, question, enhanced_question)

        conversation.send_message(enhanced_question)

        # 使用线程池执行 conversation.run() 并设置超时
        timeout_occurred = False
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future = executor.submit(conversation.run)
                future.result(timeout=MAX_TIME_PER_QUESTION)
        except FutureTimeoutError:
            timeout_occurred = True
            print(f"[TIMEOUT] 问题处理超时（>{MAX_TIME_PER_QUESTION}秒），放弃当前问题")
            answer_data["answer"] = (
                f"Timeout: Question processing exceeded {MAX_TIME_PER_QUESTION} seconds and was aborted."
            )
            answer_data["time_cost"] = MAX_TIME_PER_QUESTION
            return answer_data

        # 计算时间成本
        end_time = time.time()
        answer_data["time_cost"] = round(end_time - start_time, 2)

        # 从 conversation state 中获取最后的答案和 token 使用量
        state = conversation.state

        # 获取完整的 message history（运行10轮之后）
        message_history = get_message_history(state)
        print(f"[DEBUG] 获取到 {len(message_history)} 条消息历史")

        # 获取 token 使用量（分别统计 input 和 output）
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
                )  # 总 token 数（向后兼容）

        # 如果还没有获取到答案，从 events 中查找
        if not answer_data["answer"]:
            events = list(state.events)
            for event in reversed(events):
                # 检查 MessageEvent
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

        # 如果仍然没有答案，检查是否达到 MAX_ITERATION_PER_RUN 限制
        if not answer_data["answer"]:
            # 统计 ActionEvent 的数量来判断是否达到限制
            events = list(state.events)
            action_count = sum(1 for event in events if isinstance(event, ActionEvent))

            print(
                f"[DEBUG] 未找到答案，ActionEvent 数量: {action_count}, 最大限制: {MAX_ITERATION_PER_RUN}"
            )

            # 如果达到或超过限制，使用 message_history 生成答案
            if action_count >= MAX_ITERATION_PER_RUN:
                print(
                    f"[DEBUG] 达到最大迭代次数限制 ({MAX_ITERATION_PER_RUN})，使用 message_history 生成最终答案..."
                )

                if message_history and len(message_history) > 0:
                    forced_answer, (forced_prompt_tokens, forced_completion_tokens) = (
                        generate_answer_from_history(
                            LLM_CONFIG, question, message_history
                        )
                    )

                    if forced_answer:
                        answer_data["answer"] = forced_answer
                        # 更新 token 成本（累加强制生成的 tokens）
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
                            f"[DEBUG] 已使用 message_history 生成答案，答案长度: {len(answer_data['answer'])} 字符"
                        )
                        print(
                            f"[DEBUG] Token统计: prompt={previous_prompt}+{forced_prompt_tokens}={answer_data['prompt_tokens']}, completion={previous_completion}+{forced_completion_tokens}={answer_data['completion_tokens']}, total={answer_data['token_cost']}"
                        )
                    else:
                        print(f"[DEBUG] 警告: 使用 message_history 生成答案失败")
                        answer_data["answer"] = (
                            "Unable to generate answer based on conversation history."
                        )
                else:
                    print(f"[DEBUG] 警告: message_history 为空，无法生成答案")
                    answer_data["answer"] = (
                        "No conversation history available to generate answer."
                    )

        # 最终保存轨迹（确保所有事件都被保存，包括prune详细信息）
        if traj_dir and repo_name and question_idx is not None:
            save_trajectory(
                conversation, question, repo_name, question_idx, traj_dir, append=False
            )

        return answer_data

    except Exception as e:
        print(f"处理问题失败: {question[:50]}... 错误: {e}")
        answer_data["answer"] = f"Error: {str(e)}"
        # 即使出错也尝试保存轨迹
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
        # 确保无论什么情况都关闭 conversation，避免 __del__ 时的错误
        safe_close_conversation(conversation)


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建轨迹和配置目录
    os.makedirs(TRAJ_DIR, exist_ok=True)
    config_dir = os.path.join(TRAJ_DIR, "configs")
    prompt_dir = os.path.join(TRAJ_DIR, "prompts")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    # 全局统计
    all_processed_count = 0
    all_error_count = 0
    all_time_costs = []
    all_token_costs = []
    all_prompt_tokens_list = []
    all_completion_tokens_list = []

    # 依次处理每个仓库
    for repo_idx, repo_config in enumerate(REPOS_CONFIG, 1):
        repo_name = repo_config["name"]
        workspace = repo_config["workspace"]
        input_file = repo_config["input_file"]

        print(f"\n{'=' * 60}")
        print(f"开始处理仓库 {repo_idx}/{len(REPOS_CONFIG)}: {repo_name}")
        print(f"{'=' * 60}")
        print(f"工作空间: {workspace}")
        print(f"问题文件: {input_file}")

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在，跳过: {input_file}")
            continue

        output_file = os.path.join(OUTPUT_DIR, f"{repo_name}_answers.jsonl")

        # 为每个仓库创建轨迹子目录
        repo_traj_dir = os.path.join(TRAJ_DIR, repo_name)
        os.makedirs(repo_traj_dir, exist_ok=True)

        # 保存实验配置（每个仓库保存一次）
        save_experiment_config(config_dir, repo_name)

        # 加载已回答的问题
        answered_questions = load_answered_questions(output_file)
        if answered_questions:
            print(f"已找到 {len(answered_questions)} 个已回答的问题")

        # 加载所有问题
        print(f"从 {input_file} 加载问题...")
        all_questions = load_questions_from_jsonl(input_file)
        print(f"共加载 {len(all_questions)} 个问题")

        # 过滤掉已回答的问题
        questions = [
            qa_data
            for qa_data in all_questions
            if qa_data.get("question", "") not in answered_questions
        ]

        if len(questions) < len(all_questions):
            print(f"过滤后剩余 {len(questions)} 个未回答的问题")
        else:
            print(f"所有问题都未回答，将处理全部 {len(questions)} 个问题")

        if len(questions) == 0:
            print(f"仓库 {repo_name} 没有需要处理的问题，跳过")
            continue

        # 串行处理
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
                    # 写入文件
                    with open(output_file, "a", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")

                    # 收集统计数据
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
                        f"[{repo_name}][{idx}/{len(questions)}] 完成: {result['question'][:50]}..."
                    )
                else:
                    error_count += 1
                    all_error_count += 1
            except Exception as e:
                error_count += 1
                all_error_count += 1
                print(
                    f"[{repo_name}][{idx}/{len(questions)}] 处理失败: {qa_data.get('question', '')[:50]}... 错误: {e}"
                )

        # 计算当前仓库的统计信息
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

        print(f"\n仓库 {repo_name} 处理完成:")
        print(f"  成功: {processed_count}, 失败: {error_count}, 总计: {len(questions)}")
        print(f"  平均 time_cost: {avg_time_cost:.2f} 秒")
        print(f"  总 time_cost: {total_time_cost:.2f} 秒")
        print(f"  平均 token_cost: {avg_token_cost:.0f} tokens")
        print(f"  总 token_cost: {total_token_cost:.0f} tokens")
        print(
            f"  平均 prompt_tokens: {avg_prompt_tokens:.0f}, completion_tokens: {avg_completion_tokens:.0f}"
        )
        print(
            f"  总 prompt_tokens: {total_prompt_tokens:.0f}, completion_tokens: {total_completion_tokens:.0f}"
        )
        print(f"  结果已保存到: {output_file}")

    # 计算全局统计信息
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
    print(f"所有仓库处理完成!")
    print(f"{'=' * 60}")
    print(f"成功: {all_processed_count}, 失败: {all_error_count}")
    print(f"\n全局统计信息:")
    print(f"  平均 time_cost: {avg_time_cost:.2f} 秒")
    print(f"  总 time_cost: {total_time_cost:.2f} 秒")
    print(f"\nToken 统计 (总计):")
    print(f"  平均 token_cost: {avg_token_cost:.0f} tokens")
    print(f"  总 token_cost: {total_token_cost:.0f} tokens")
    print(f"\nToken 统计 (分开):")
    print(f"  平均 prompt_tokens (input): {avg_prompt_tokens:.0f} tokens")
    print(f"  平均 completion_tokens (output): {avg_completion_tokens:.0f} tokens")
    print(f"  总 prompt_tokens (input): {total_prompt_tokens:.0f} tokens")
    print(f"  总 completion_tokens (output): {total_completion_tokens:.0f} tokens")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
