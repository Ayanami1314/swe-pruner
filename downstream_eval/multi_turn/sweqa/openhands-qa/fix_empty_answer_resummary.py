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
import concurrent
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
import openai
from typing import Optional, List, Union, Any, Tuple, Dict
import argparse

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
from openhands.sdk import TextContent, ImageContent
import sys

load_dotenv()
experiment = os.getenv("EXPERIMENT_TYPE")
api_type = os.getenv("API_TYPE")
if api_type not in ["openai", "azure"]:
    print(f"Unsupported api type: {api_type}")
    exit(1)
MAX_WORKERS = 8


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


def extract_message_history_from_traj(traj_file: str) -> Optional[List[Dict]]:
    """从轨迹文件中提取 message history"""
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
        print(f"读取轨迹文件 {traj_file} 时出错: {e}")
        return None


def generate_answer_from_history(
    llm_config, question: str, message_history: List[Dict]
) -> tuple:
    """基于 message_history 使用 LLM 生成答案"""
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
        print(f"生成答案时出错: {e}")
        import traceback

        traceback.print_exc()
        return None, (0, 0)


def find_traj_file(traj_dir: str, repo_name: str, question: str) -> Optional[str]:
    """根据问题和仓库名查找对应的轨迹文件"""
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
    """从 jsonl 文件加载答案"""
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
    """保存答案到 jsonl 文件"""
    with open(answer_file, "w", encoding="utf-8") as f:
        for answer in answers:
            json.dump(answer, f, ensure_ascii=False)
            f.write("\n")


def fix_empty_answer_resummary(
    traj_dir: str,
    answer_dir: str,
    dry_run: bool = False,
):
    """重新总结空答案"""

    answer_files = []
    for filename in os.listdir(answer_dir):
        if filename.endswith("_answers.jsonl"):
            answer_files.append(os.path.join(answer_dir, filename))

    if not answer_files:
        print(f"在 {answer_dir} 中未找到答案文件")
        return

    total_fixed = 0
    total_checked = 0

    for answer_file in answer_files:
        print(f"ALL answer_files: {answer_files}")
        print(f"\n处理文件: {answer_file}")

        repo_name = os.path.basename(answer_file).replace("_answers.jsonl", "")

        answers = load_answers_from_jsonl(answer_file)
        print(f"  加载了 {len(answers)} 个答案")

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
                            print(f"  [DRY RUN] 将重新生成答案: {question[:50]}...")
                            return False, answer
                        else:
                            print(f"  生成答案 ({idx + 1}): {question[:50]}...")
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

                                print(f"    ✓ 成功生成答案 ({len(new_answer)} 字符)")
                                print(
                                    f"    旧答案: {old_answer[:100] if old_answer else '(空)'}"
                                )
                                print(f"    新答案: {new_answer[:100]}...")
                                return True, answer
                            else:
                                print(f"    ✗ 生成答案失败")
                                return False, answer
                    else:
                        print(f"  ✗ 未找到 message history: {question[:50]}...")
                        return False, answer
                else:
                    print(f"  ✗ 未找到轨迹文件: {question[:50]}...")
                    return False, answer

            return False, answer

        # 原始代码修改部分
        total_checked = 0
        fixed_count = 0
        total_fixed = 0

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 创建部分函数，固定部分参数
            process_func = partial(
                process_answer,
                repo_name=repo_name,
                traj_dir=traj_dir,
                LLM_CONFIG=LLM_CONFIG,
                dry_run=dry_run,
            )

            # 提交所有任务到线程池
            futures = []
            for idx, answer in enumerate(answers):
                total_checked += 1
                futures.append(executor.submit(process_func, idx, answer))

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(futures):
                is_fixed, updated_answer = future.result()
                if is_fixed:
                    fixed_count += 1
                    total_fixed += 1
                    # 更新原始answers列表中的对应项
                    for i, ans in enumerate(answers):
                        if ans.get("question") == updated_answer.get("question"):
                            answers[i] = updated_answer
                            break

        if fixed_count > 0 and not dry_run:
            save_answers_to_jsonl(answer_file, answers)
            print(f"  ✓ 已保存 {fixed_count} 个修复的答案到 {answer_file}")
    print(f"\n{'=' * 60}")
    print(f"重新生成完成!")
    print(f"  检查总数: {total_checked}")
    print(f"  修复数量: {total_fixed}")
    if dry_run:
        print(f"  [DRY RUN 模式，未实际修改文件]")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="从轨迹文件中提取 message history 重新生成空答案"
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="./trajectories",
        help="轨迹文件目录 (默认: ./trajectories)",
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="./answer/openhands",
        help="答案文件目录 (默认: ./answer/openhands)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式，只显示将要修复的内容，不实际修改文件",
    )

    args = parser.parse_args()

    if not os.path.exists(args.traj_dir):
        print(f"错误: 轨迹目录不存在: {args.traj_dir}")
        return

    if not os.path.exists(args.answer_dir):
        print(f"错误: 答案目录不存在: {args.answer_dir}")
        return

    fix_empty_answer_resummary(args.traj_dir, args.answer_dir, args.dry_run)


if __name__ == "__main__":
    main()
