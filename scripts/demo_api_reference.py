"""用已配置的文本 API 为冻结 cases 生成 Claude 强参考答案。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from demo_data_utils import DemoDataError


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_cases(path: Path, limit: int | None) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise DemoDataError("cases 文件无法以 UTF-8 读取") from None
    cases, seen = [], set()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DemoDataError(f"cases 第 {line_no} 行不是有效 JSON") from None
        cid = row.get("case_id") if isinstance(row, dict) else None
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise DemoDataError(f"cases 第 {line_no} 行 case_id 缺失或重复")
        if row.get("kind") not in {"domain", "general"} or row.get("split") not in {"dev", "test"}:
            raise DemoDataError(f"case_id={cid} 的 kind 或 split 无效")
        messages = row.get("input_messages")
        if not isinstance(messages, list) or not messages:
            raise DemoDataError(f"case_id={cid} 缺少 input_messages")
        clean = []
        for i, msg in enumerate(messages):
            if (not isinstance(msg, dict) or set(msg) != {"role", "content"}
                    or msg.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(msg.get("content"), str) or not msg["content"].strip()):
                raise DemoDataError(f"case_id={cid} 的 input_messages[{i}] 无效")
            clean.append(dict(msg))
        dialogue = clean[1:] if clean[0]["role"] == "system" else clean
        if (sum(m["role"] == "system" for m in clean) > 1 or not dialogue
                or dialogue[-1]["role"] != "user"):
            raise DemoDataError(f"case_id={cid} 的 input_messages 角色顺序无效")
        expected = "user"
        for msg in dialogue:
            if msg["role"] != expected:
                raise DemoDataError(f"case_id={cid} 的 input_messages 必须按 user/assistant 交替")
            expected = "assistant" if expected == "user" else "user"
        turns = row.get("user_turns", [])
        if not isinstance(turns, list) or any(not isinstance(x, str) or not x.strip() for x in turns):
            raise DemoDataError(f"case_id={cid} 的 user_turns 必须为非空字符串数组")
        if any("<think>" in m["content"].lower() or "</think>" in m["content"].lower() for m in clean + [{"content": x} for x in turns]):
            raise DemoDataError(f"case_id={cid} 含禁止的 think 标签")
        # 只留下模型输入所需字段，gold、rubric、参考答案和 source metadata 不会进入请求。
        cases.append({"case_id": cid, "kind": row["kind"], "split": row["split"],
                      "input_messages": clean, "user_turns": list(turns)})
        seen.add(cid)
        if limit is not None and len(cases) >= limit:
            break
    if not cases:
        raise DemoDataError("cases 文件没有有效记录")
    return cases


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _save_rows(path: Path, rows: list[dict]) -> None:
    _atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))


def _wallet(config) -> float | None:
    """仅向已配置服务的同一 origin 查询 /usage，凭据不跨 origin。"""
    from teacher_api import _NoRedirect
    parts = urlsplit(config.base_url)
    base_path = parts.path.rstrip("/")
    if base_path.endswith(("/responses", "/messages", "/chat/completions")):
        base_path = base_path.rsplit("/", 1)[0]
    usage_path = base_path + "/usage" if base_path.endswith("/v1") else (base_path + "/v1/usage" if base_path else "/v1/usage")
    url = urlunsplit((parts.scheme, parts.netloc, usage_path, "", ""))
    headers = {"Accept": "application/json", "User-Agent": "CircuitLLM/0.1"}
    headers["x-api-key" if config.provider == "anthropic" else "Authorization"] = (
        config.api_key if config.provider == "anthropic" else "Bearer " + config.api_key)
    request = Request(url, headers=headers)
    host = (parts.hostname or "").lower()
    handlers = [ProxyHandler({})] if host in {"localhost", "127.0.0.1", "::1"} else (
        [ProxyHandler({"http": config.proxy_url, "https": config.proxy_url})] if config.proxy_url else [ProxyHandler()])
    try:
        with build_opener(_NoRedirect(), *handlers).open(request, timeout=min(config.timeout_seconds, 30)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise DemoDataError(f"余额查询失败（HTTP {exc.code}）；未继续生成") from None
    except (URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError):
        raise DemoDataError("余额无法确认；未继续生成") from None
    if not isinstance(data, dict):
        return None
    currency = data.get("currency", data.get("unit"))
    for key in ("remaining_usd", "balance_usd", "available_balance_usd", "remaining", "balance", "available_balance"):
        val = data.get(key)
        if isinstance(val, dict):
            currency = val.get("currency", currency)
            val = val.get("amount", val.get("value"))
        if key.endswith("_usd"):
            currency = "USD"
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val) and str(currency).upper() in {"USD", "US$"}:
            return float(val)
    return None


def _fingerprint(config, args, cases_hash: str) -> str:
    parts = urlsplit(config.base_url)
    # 不把密钥纳入指纹；完整请求配置用于阻止不同解码设置或超时的结果混跑。
    value = {"provider": config.provider, "origin": f"{parts.scheme}://{parts.netloc}",
             "path": parts.path, "api": config.openai_api, "requested_model": config.model_id,
             "run_label": args.run_label, "step": args.step0, "max_output_tokens": args.max_output_tokens,
             "timeout_seconds": config.timeout_seconds, "max_retries": config.max_retries,
             "openai_max_tokens_field": config.openai_max_tokens_field,
             "extra_body": config.extra_body,
             "cases_sha256": cases_hash}
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def _validate_reference_config(config) -> None:
    """拒绝可能引入工具或跨请求服务端会话状态的自定义参数。"""
    forbidden = {"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
                 "previous_response_id", "conversation",
                 "conversation_id", "thread_id", "session_id", "assistant_id", "run_id"}
    if not isinstance(config.extra_body, dict):
        raise DemoDataError("Teacher extra_body 配置无效；强参考必须逐请求独立")
    keys = {key.lower() for key in config.extra_body if isinstance(key, str)}
    if keys.intersection(forbidden):
        raise DemoDataError("Teacher extra_body 含工具或服务端会话状态；强参考必须逐请求独立")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model-id", default="claude-opus-5-5")
    parser.add_argument("--run-label", default="claude-opus-5-5")
    parser.add_argument("--step0", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-state", type=Path, default=Path("本地资料/claude参考预算.json"))
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="只检查 cases，不读取凭据或联网")
    args = parser.parse_args(argv)
    if args.step0 < 0 or args.max_output_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("step0 必须非负，max-output-tokens 和 limit 必须为正数")
    try:
        cases = _read_cases(args.cases, args.limit)
        cases_hash = _sha(args.cases.read_bytes())
        if args.dry_run:
            print(json.dumps({"dry_run": True, "cases": len(cases), "cases_sha256": cases_hash}, ensure_ascii=False))
            return 0
        from teacher_api import TeacherError, call_teacher_messages, load_config
        config = load_config(args.env_file)
        config.model_id = args.model_id
        config.max_output_tokens = args.max_output_tokens
        config.max_retries = 0
        _validate_reference_config(config)
        fingerprint = _fingerprint(config, args, cases_hash)
        output = args.output.resolve()
        rows: list[dict] = []
        by_id: dict[str, dict] = {}
        if output.exists():
            if not args.resume:
                raise DemoDataError("输出已存在；复用结果请明确指定 --resume")
            try:
                for line in output.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("run_fingerprint") != fingerprint or item.get("case_id") in by_id:
                        raise DemoDataError("输出运行指纹不符或 case_id 重复")
                    by_id[item["case_id"]] = item
                    rows.append(item)
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
                raise DemoDataError("已有输出损坏或无法读取") from None
        balance = _wallet(config)
        if balance is None or not math.isfinite(balance) or balance < 0.50:
            raise DemoDataError("USD 余额不可确认或低于 $0.50 保留额；停止请求")
        state_path = args.budget_state
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise DemoDataError("预算状态文件损坏") from None
            if state.get("cases_sha256") != cases_hash or state.get("run_fingerprint") != fingerprint:
                raise DemoDataError("预算状态指纹不符；拒绝混用不同运行预算")
            starting = state.get("starting_balance")
            state["reserve_usd"] = 0.50
            _atomic_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        else:
            starting = balance
            state = {"cases_sha256": cases_hash, "run_fingerprint": fingerprint,
                     "starting_balance": balance, "reserve_usd": 0.50}
            state_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        if not isinstance(starting, (int, float)) or not math.isfinite(starting):
            raise DemoDataError("预算起始余额无效")
        requests = 0
        for case in cases:
            item = by_id.get(case["case_id"])
            if item is not None and item.get("status") == "completed":
                continue
            if item is None:
                item = {"case_id": case["case_id"], "kind": case["kind"], "split": case["split"],
                        "model": args.run_label, "step": args.step0,
                        "requested_model": config.model_id, "reported_model": None, "reported_models": [],
                        "config": {"provider": config.provider, "api": config.openai_api,
                                   "max_output_tokens": config.max_output_tokens,
                                   "timeout_seconds": config.timeout_seconds, "max_retries": config.max_retries,
                                   "openai_max_tokens_field": config.openai_max_tokens_field,
                                   "extra_body": config.extra_body,
                                   "stream": False, "decoding": "service_default" if not config.extra_body else "configured"},
                        "run_fingerprint": fingerprint, "cases_sha256": cases_hash,
                        "status": "partial", "turns": [], "answers_by_turn": [], "full_dialogue": list(case["input_messages"])}
                by_id[case["case_id"]] = item
                rows.append(item)
            history = list(item["full_dialogue"])
            turns = [None, *case["user_turns"]]
            completed_turns = len(item["turns"])
            for turn_idx in range(completed_turns, len(turns)):
                user_turn = turns[turn_idx]
                if user_turn is not None:
                    history.append({"role": "user", "content": user_turn})
                balance = _wallet(config)
                spent = float(starting) - balance if balance is not None else float("inf")
                if balance is None or balance < 0.50 or spent >= float(starting) - 0.50:
                    raise DemoDataError("USD 余额触及预算或 $0.50 保留额；停止请求")
                began = time.perf_counter()
                try:
                    response = call_teacher_messages(config, history)
                except TeacherError as exc:
                    raise DemoDataError(f"教师接口停止：{exc}；已保存成功的逐轮检查点") from None
                elapsed = time.perf_counter() - began
                answer = response["text"]
                if config.api_key and config.api_key in answer:
                    raise DemoDataError("模型输出包含凭据特征；拒绝保存")
                usage = response.get("usage") or {}
                finish = response.get("finish_reason")
                truncated = finish not in {"completed", "stop", "end_turn", "stop_sequence"}
                turn = {"user_turn": user_turn, "answer": answer,
                        "reported_model": response.get("reported_model"),
                        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                        "latency_seconds": round(elapsed, 4), "finish_reason": finish,
                        "truncated": truncated,
                        "service_default_decoding": not bool(config.extra_body)}
                item["turns"].append(turn)
                item["answers_by_turn"].append(answer)
                item["reported_models"].append(turn["reported_model"])
                observed_models = {name for name in item["reported_models"] if isinstance(name, str) and name}
                item["reported_model"] = next(iter(observed_models)) if len(observed_models) == 1 else None
                history.append({"role": "assistant", "content": answer})
                item["full_dialogue"] = history
                item["input_tokens"] = sum((t.get("input_tokens") or 0) for t in item["turns"])
                item["output_tokens"] = sum((t.get("output_tokens") or 0) for t in item["turns"])
                item["latency_seconds"] = round(sum(t["latency_seconds"] for t in item["turns"]), 4)
                item["truncated"] = any(t["truncated"] for t in item["turns"])
                requests += 1
                _save_rows(output, rows)
                if truncated:
                    raise DemoDataError("教师回复未正常完成；已保存部分检查点，停止后续请求")
            item["answer"] = "\n\n".join(item["answers_by_turn"])
            item["status"] = "completed"
            item["finish_reason"] = "completed"
            item["created_utc"] = datetime.now(timezone.utc).isoformat()
            _save_rows(output, rows)
        balance = _wallet(config)
        if balance is None:
            raise DemoDataError("结束余额无法确认；输出已保存")
        print(json.dumps({"completed": sum(r.get("status") == "completed" for r in rows),
                          "requests_this_run": requests, "ending_balance_usd": balance,
                          "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (DemoDataError, OSError, ValueError) as exc:
        print(f"拒绝执行：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
