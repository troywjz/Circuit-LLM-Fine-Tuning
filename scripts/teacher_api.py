"""使用 Python 标准库调用文本教师模型接口。"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class TeacherError(Exception):
    """对外安全的教师接口错误，不包含凭据或响应正文。"""


@dataclass
class TeacherConfig:
    """保存教师接口配置；密钥在 repr 中始终隐藏。"""

    provider: str
    base_url: str
    api_key: str = field(repr=False)
    model_id: str = ""
    openai_api: str = "responses"
    max_output_tokens: int = 8192
    timeout_seconds: float = 120
    max_retries: int = 2
    proxy_url: str = ""
    openai_max_tokens_field: str = "max_completion_tokens"
    extra_body: dict[str, Any] = field(default_factory=dict)


_PLACEHOLDERS = {"", "your_api_key", "your-api-key", "replace_me", "changeme", "none", "null"}
_PROTECTED_EXTRA = {
    "model", "messages", "input", "instructions", "system", "stream",
    "max_tokens", "max_completion_tokens", "max_output_tokens",
}


def _safe_url(value: str, name: str, *, allow_empty: bool = False) -> str:
    """校验 URL 结构，不把用户提供的 URL 写入错误信息。"""
    if not value and allow_empty:
        return ""
    try:
        parts = urlsplit(value)
        valid = (parts.scheme in {"http", "https"} and bool(parts.hostname)
                 and parts.username is None and parts.password is None
                 and not parts.query and not parts.fragment)
        if not valid:
            raise ValueError
        _ = parts.port
        return value.rstrip("/")
    except (ValueError, TypeError):
        raise TeacherError(f"{name} 必须是无凭据、查询参数和片段的 HTTP(S) URL。") from None


def _parse_env_file(path: Path) -> dict[str, str]:
    """仅解析简单 KEY=VALUE 行；不执行 shell，也不展开变量。"""
    result: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return result
    except (OSError, UnicodeError):
        raise TeacherError("无法读取指定的 UTF-8 配置文件。") from None
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*", line)
        if not match:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result


def load_config(env_file: Path, *, require_credentials: bool = True) -> TeacherConfig:
    """读取指定 .env，TEACHER_ 环境变量优先；不探测其他凭据文件。"""
    values = _parse_env_file(Path(env_file))
    names = (
        "PROVIDER", "OPENAI_API", "BASE_URL", "API_KEY", "MODEL_ID",
        "MAX_OUTPUT_TOKENS", "TIMEOUT_SECONDS", "MAX_RETRIES", "PROXY_URL",
        "OPENAI_MAX_TOKENS_FIELD", "EXTRA_BODY_JSON",
    )
    for name in names:
        env_name = "TEACHER_" + name
        if env_name in os.environ:
            values[env_name] = os.environ[env_name]

    def get(name: str, default: str = "") -> str:
        return values.get("TEACHER_" + name, default).strip()

    provider = get("PROVIDER", "openai").lower()
    api = get("OPENAI_API", "responses").lower()
    token_field = get("OPENAI_MAX_TOKENS_FIELD", "max_completion_tokens")
    if provider not in {"openai", "anthropic"}:
        raise TeacherError("TEACHER_PROVIDER 仅支持 openai 或 anthropic。")
    if api not in {"responses", "chat_completions"}:
        raise TeacherError("TEACHER_OPENAI_API 仅支持 responses 或 chat_completions。")
    if token_field not in {"max_completion_tokens", "max_tokens"}:
        raise TeacherError("TEACHER_OPENAI_MAX_TOKENS_FIELD 配置无效。")
    try:
        max_tokens = int(get("MAX_OUTPUT_TOKENS", "8192"))
        timeout = float(get("TIMEOUT_SECONDS", "120"))
        retries = int(get("MAX_RETRIES", "2"))
    except ValueError:
        raise TeacherError("token、timeout 或 retry 数值配置无效。") from None
    if (max_tokens < 1 or max_tokens > 1_000_000 or not math.isfinite(timeout)
            or timeout <= 0 or timeout > 3600 or retries < 0 or retries > 2):
        raise TeacherError("token、timeout 或 retry 数值超出允许范围。")
    base = _safe_url(get("BASE_URL"), "TEACHER_BASE_URL", allow_empty=not require_credentials)
    proxy = _safe_url(get("PROXY_URL"), "TEACHER_PROXY_URL", allow_empty=True)
    key, model = get("API_KEY"), get("MODEL_ID")
    if require_credentials:
        if not base or base == "http://placeholder" or base == "https://example.com":
            raise TeacherError("live 调用需要有效的 TEACHER_BASE_URL。")
        if key.lower() in _PLACEHOLDERS:
            raise TeacherError("live 调用需要有效的 TEACHER_API_KEY。")
        if model.lower() in _PLACEHOLDERS:
            raise TeacherError("live 调用需要有效的 TEACHER_MODEL_ID。")
    raw_extra = get("EXTRA_BODY_JSON", "{}") or "{}"
    try:
        extra = json.loads(raw_extra)
    except json.JSONDecodeError:
        raise TeacherError("TEACHER_EXTRA_BODY_JSON 必须是合法 JSON object。") from None
    if not isinstance(extra, dict):
        raise TeacherError("TEACHER_EXTRA_BODY_JSON 必须是 JSON object。")
    forbidden = _PROTECTED_EXTRA.intersection(extra)
    if forbidden:
        raise TeacherError("TEACHER_EXTRA_BODY_JSON 包含受保护的请求字段。")
    return TeacherConfig(provider, base, key, model, api, max_tokens, timeout, retries,
                         proxy, token_field, extra)


def endpoint_url(config: TeacherConfig) -> str:
    """为协议构造 endpoint；保留网关自定义路径前缀。"""
    base = _safe_url(config.base_url, "TEACHER_BASE_URL", allow_empty=False)
    path = urlsplit(base).path.rstrip("/")
    if not path:
        path = "/v1"
    if config.provider == "openai":
        suffix = "/responses" if config.openai_api == "responses" else "/chat/completions"
    elif config.provider == "anthropic":
        suffix = "/messages"
    else:
        raise TeacherError("未知教师服务协议。")
    if path.endswith(suffix):
        endpoint = path
    else:
        endpoint = path + suffix
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, endpoint, "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """拒绝所有重定向，防止认证头被转发到其他位置。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _extract(config: TeacherConfig, data: Any) -> dict[str, Any]:
    """仅提取最终文本和必要元数据，并拒绝不完整或工具调用结果。"""
    if not isinstance(data, dict):
        raise TeacherError("教师接口返回结构无效。")
    text = ""
    raw_usage = data.get("usage")
    usage: dict[str, int | float] | None = None
    if isinstance(raw_usage, dict):
        # 只保留白名单里的数值用量字段，避免透传服务端任意内容。
        allowed_usage = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens",
                         "completion_tokens", "cached_tokens", "reasoning_tokens"}
        filtered = {key: value for key, value in raw_usage.items()
                    if key in allowed_usage and isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(value)}
        usage = filtered or None
    raw_model = data.get("model")
    model = raw_model if isinstance(raw_model, str) and raw_model else config.model_id
    raw_id = data.get("id") or data.get("request_id")
    request_id = raw_id if isinstance(raw_id, str) else None
    if config.api_key and any(config.api_key in value for value in (model, request_id or "")):
        raise TeacherError("教师接口元数据包含配置密钥，已拒绝返回。")
    finish = ""
    if config.provider == "openai" and config.openai_api == "responses":
        status = data.get("status")
        if status != "completed":
            raise TeacherError("Responses 接口未返回 completed 结果。")
        output = data.get("output")
        if not isinstance(output, list):
            raise TeacherError("Responses 接口 output 结构无效。")
        for item in output:
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise TeacherError("Responses 接口 output 项结构无效。")
            item_type = item["type"]
            if item_type == "reasoning":
                continue
            if item_type != "message":
                raise TeacherError("Responses 接口返回工具或未知 output 项。")
            if item.get("role") != "assistant":
                raise TeacherError("Responses 接口 message 角色无效。")
            if "status" in item and item["status"] != "completed":
                raise TeacherError("Responses 接口 message 未完成。")
            content = item.get("content")
            if not isinstance(content, list):
                raise TeacherError("Responses 接口 message content 结构无效。")
            for block in content:
                if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                    raise TeacherError("Responses 接口 content 项结构无效。")
                if block["type"] == "output_text":
                    if not isinstance(block.get("text"), str):
                        raise TeacherError("Responses 接口文本结构无效。")
                    text += block["text"]
                elif block["type"] == "refusal":
                    raise TeacherError("教师模型拒绝了请求。")
                elif block["type"] in {"reasoning_text", "summary_text"}:
                    continue
                else:
                    raise TeacherError("Responses 接口返回未知 content 项。")
        finish = "completed"
    elif config.provider == "openai":
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TeacherError("Chat Completions 接口缺少 choices。")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TeacherError("Chat Completions 响应结构无效。")
        finish = choice.get("finish_reason")
        if finish != "stop":
            raise TeacherError("Chat Completions 停止原因缺失或非正常完成。")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise TeacherError("Chat Completions message 结构无效。")
        if message.get("refusal"):
            raise TeacherError("教师模型拒绝了请求。")
        if message.get("tool_calls") or message.get("function_call"):
            raise TeacherError("Chat Completions 返回工具调用，不能作为最终文本。")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    raise TeacherError("Chat Completions content 项结构无效。")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "refusal":
                    raise TeacherError("教师模型拒绝了请求。")
                else:
                    raise TeacherError("Chat Completions 返回未知 content 项。")
            text = "".join(parts)
        else:
            raise TeacherError("Chat Completions 未返回文本内容。")
    else:
        finish = data.get("stop_reason")
        if finish not in ("end_turn", "stop_sequence"):
            raise TeacherError("Anthropic Messages 停止原因缺失或非正常完成。")
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise TeacherError("Anthropic Messages 响应结构无效。")
        parts = []
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise TeacherError("Anthropic Messages content 项结构无效。")
            if block["type"] == "text":
                if not isinstance(block.get("text"), str):
                    raise TeacherError("Anthropic Messages 文本结构无效。")
                parts.append(block["text"])
            elif block["type"] in {"thinking", "redacted_thinking"}:
                continue
            else:
                raise TeacherError("Anthropic Messages 返回工具或未知内容项。")
        text = "".join(parts)
    if not text.strip():
        raise TeacherError("教师接口未返回最终文本。")
    return {"text": text, "usage": usage,
            "finish_reason": finish, "model": model,
            "request_id": str(request_id) if request_id is not None else None}


def call_teacher(config: TeacherConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """调用一次非 stream 文本接口；错误只包含状态类别，不包含正文。"""
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise TeacherError("system_prompt 和 user_prompt 必须是字符串。")
    # 直接构造配置也必须满足数值边界，避免 urllib 收到 NaN 或无穷超时。
    if (not isinstance(config.provider, str) or config.provider not in {"openai", "anthropic"}
            or not isinstance(config.api_key, str)
            or not isinstance(config.model_id, str)
            or not isinstance(config.timeout_seconds, (int, float))
            or isinstance(config.timeout_seconds, bool)
            or not math.isfinite(config.timeout_seconds)
            or config.timeout_seconds <= 0
            or not isinstance(config.max_retries, int)
            or isinstance(config.max_retries, bool)
            or config.max_retries < 0 or config.max_retries > 2):
        raise TeacherError("timeout 或 retry 数值配置无效。")
    # 再次校验运行时配置，防止调用方绕过 load_config 直接覆盖保护字段。
    if not isinstance(config.extra_body, dict) or _PROTECTED_EXTRA.intersection(config.extra_body):
        raise TeacherError("extra_body 必须是 object，且不能包含受保护的请求字段。")
    url = endpoint_url(config)
    if config.provider == "openai" and config.openai_api == "responses":
        body = {"model": config.model_id, "instructions": system_prompt,
                "input": user_prompt, "max_output_tokens": config.max_output_tokens,
                "store": False, "stream": False, **config.extra_body}
    elif config.provider == "openai":
        body = {"model": config.model_id,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                config.openai_max_tokens_field: config.max_output_tokens,
                "stream": False, **config.extra_body}
    elif config.provider == "anthropic":
        body = {"model": config.model_id, "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": config.max_output_tokens, **config.extra_body}
    else:
        raise TeacherError("未知教师服务协议。")
    try:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise TeacherError("教师接口请求配置无法编码为 JSON。") from None
    # 使用项目标识供网关识别请求，避免默认 Python-urllib 特征导致兼容拒绝。
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "CircuitLLM/0.1"}
    if config.provider == "anthropic":
        headers.update({"x-api-key": config.api_key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = "Bearer " + config.api_key
    request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
    handlers: list[Any] = [_NoRedirect()]
    # 本地 mock 始终直连，避免无论环境代理还是显式代理都将请求发送到远端代理。
    host = urlsplit(url).hostname or ""
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if host.lower() in local_hosts:
        handlers.append(urllib.request.ProxyHandler({}))
    elif config.proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": config.proxy_url, "https": config.proxy_url}))
    else:
        handlers.append(urllib.request.ProxyHandler())
    opener = urllib.request.build_opener(*handlers)
    for attempt in range(config.max_retries + 1):
        try:
            with opener.open(request, timeout=config.timeout_seconds) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            code = exc.code
            if code in {401, 403, 400}:
                raise TeacherError(f"教师接口请求失败（HTTP {code}）。") from None
            if code == 429 or 500 <= code <= 599:
                if attempt < config.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
            if 300 <= code < 400:
                raise TeacherError("教师接口重定向已拒绝。") from None
            raise TeacherError(f"教师接口请求失败（HTTP {code}）。") from None
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
                transient = False
            else:
                transient = True
            if transient and attempt < config.max_retries:
                time.sleep(min(2 ** attempt, 10))
                continue
            raise TeacherError("教师接口网络请求失败或超时。") from None
    try:
        response_text = raw.decode("utf-8")
        # 在解析成功响应前检查完整正文，避免密钥藏在文本或元数据中。
        if config.api_key and config.api_key in response_text:
            raise TeacherError("教师接口响应包含配置密钥，已拒绝处理。")
        data = json.loads(response_text)
    except TeacherError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise TeacherError("教师接口返回的 JSON 无效。") from None
    try:
        return _extract(config, data)
    except TeacherError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        raise TeacherError("教师接口响应结构无效。") from None
