"""按 .env 中的连接测试顺序发送一条极短的教师 API 连通性请求。"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from teacher_api import TeacherError, call_teacher, load_config, load_profile_order


# 这些状态表示 Anthropic 请求格式可能未被网关接受，可以按用户要求改试 OpenAI 协议。
_PROTOCOL_FALLBACK_STATUSES = {400, 401, 404, 405, 415, 422}


def _attempt(config, provider: str, openai_api: str):
    """单次请求；返回安全状态，不输出服务端响应正文。"""
    config.provider = provider
    config.openai_api = openai_api
    config.max_output_tokens = 24
    config.max_retries = 0
    return call_teacher(
        config,
        "你正在进行接口连通性检查。只按用户要求回复，不要解释。",
        "请只回复：连接正常。",
    )


def _save_backup_provider(env_file: Path, provider: str) -> None:
    """只在备用配置成功后记录实际可用协议；其他 .env 内容原样保留。"""
    path = env_file.resolve()
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    pattern = re.compile(r"^\s*TEACHER_BACKUP_PROVIDER\s*=")
    replaced = False
    for index, line in enumerate(lines):
        if pattern.match(line):
            if line.split("=", 1)[1].strip().strip("\"'").lower() == provider:
                return
            lines[index] = f"TEACHER_BACKUP_PROVIDER={provider}"
            replaced = True
            break
    if not replaced:
        lines.append(f"TEACHER_BACKUP_PROVIDER={provider}")
    payload = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as temp:
            temp.write(payload)
            temp_path = Path(temp.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _report_success(env_file: Path, profile: str, provider: str, openai_api: str,
                    attempted: list[dict[str, object]], response: dict[str, object]) -> int:
    if profile == "backup":
        try:
            _save_backup_provider(env_file, provider)
        except OSError:
            print(json.dumps({"ok": True, "profile": profile, "provider": provider,
                              "openai_api": openai_api if provider == "openai" else None,
                              "attempted": attempted, "reply_received": bool(response.get("text")),
                              "config_saved": False}, ensure_ascii=False))
            return 1
    print(json.dumps({"ok": True, "profile": profile, "provider": provider,
                      "openai_api": openai_api if provider == "openai" else None,
                      "attempted": attempted, "reply_received": bool(response.get("text")),
                      "config_saved": profile == "backup"}, ensure_ascii=False))
    return 0


def run(env_file: Path) -> int:
    try:
        order = load_profile_order(env_file, "connectivity")
    except TeacherError as exc:
        print(json.dumps({"ok": False, "stage": "route_config", "error": str(exc)}, ensure_ascii=False))
        return 2

    attempted: list[dict[str, object]] = []
    for profile in order:
        try:
            config = load_config(env_file, profile=profile)
        except TeacherError:
            attempted.append({"profile": profile, "result": "configuration_incomplete"})
            continue

        # 备用接口按当前任务明确要求先试 Anthropic；主接口保持本机已有协议设置。
        first_provider = "anthropic" if profile == "backup" else config.provider
        try:
            response = _attempt(config, first_provider, config.openai_api)
            attempted.append({"profile": profile, "provider": first_provider, "result": "ok"})
            return _report_success(env_file, profile, first_provider, config.openai_api,
                                   attempted, response)
        except TeacherError as exc:
            active_error = exc
            attempted.append({"profile": profile, "provider": first_provider,
                              "result": "rejected", "http_status": exc.http_status,
                              "category": exc.category})
            if (profile == "backup" and first_provider == "anthropic"
                    and exc.category != "quota_billing"
                    and exc.http_status in _PROTOCOL_FALLBACK_STATUSES):
                try:
                    # 仅在服务器明确拒绝 Anthropic 请求时改试一次 OpenAI 兼容 Responses。
                    response = _attempt(config, "openai", config.openai_api)
                    attempted.append({"profile": profile, "provider": "openai",
                                      "openai_api": config.openai_api, "result": "ok"})
                    return _report_success(env_file, profile, "openai", config.openai_api,
                                           attempted, response)
                except TeacherError as fallback_exc:
                    active_error = fallback_exc
                    attempted.append({"profile": profile, "provider": "openai",
                                      "openai_api": config.openai_api, "result": "rejected",
                                      "http_status": fallback_exc.http_status,
                                      "category": fallback_exc.category})

            # 网络超时、额度耗尽和服务明确拒绝权限时不换线路重发（结果可能不明）。
            # 以下两类可继续试下一顺位：503（服务端明确未受理）、网关明确报告上游连不上（请求没送达模型）。
            unreachable = getattr(active_error, "reason", None) == "upstream_unreachable"
            if (active_error.http_status is None
                    or (active_error.http_status >= 500
                        and active_error.http_status != 503 and not unreachable)
                    or active_error.category == "quota_billing" or active_error.http_status == 403):
                break

    print(json.dumps({"ok": False, "attempted": attempted,
                      "note": "没有收到完整模型回复；未输出接口地址、密钥或服务端原文。"},
                     ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    return run(args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
