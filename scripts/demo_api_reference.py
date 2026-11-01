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
             "teacher_profile": getattr(args, "effective_teacher_profile", getattr(args, "teacher_profile", "primary")),
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


def _run_account(args, cases: list[dict], cases_hash: str) -> int:
    """按账户实际余额运行独立参考；未决请求不会在续跑时自动重发。"""
    from build_demo_data import ROOT, BuildError, _account_generation_lock
    from teacher_api import TeacherError, call_teacher_messages, load_config, load_profile_order

    private_root = (ROOT / "本地资料").resolve()
    output, state_path = args.output.resolve(), args.budget_state.resolve()
    try:
        output.relative_to(private_root)
        state_path.relative_to(private_root)
    except ValueError:
        raise DemoDataError("账户余额参考运行的输出与账本必须位于本地资料") from None
    cases_path = args.cases.resolve()
    if len({output, state_path, cases_path}) != 3:
        raise DemoDataError("参考输出、账户账本和题目文件必须是不同文件")
    profile = getattr(args, "teacher_profile", "primary")
    if profile != "primary":
        raise DemoDataError("强参考账户模式只允许 primary profile；未发送参考请求")
    try:
        if load_profile_order(args.env_file, "reference") != ("primary",):
            raise DemoDataError("TEACHER_PROFILE_ORDER_REFERENCE 必须为 primary；未发送参考请求")
    except TeacherError as exc:
        raise DemoDataError(f"参考 profile 顺序无效；未发送参考请求：{exc}") from None
    args.effective_teacher_profile = profile
    for case in cases:
        roles = [m["role"] for m in case["input_messages"]]
        # 冻结的多轮题可带有学生提供的既有 assistant 历史；_read_cases 已白名单化并剥离评分/答案等字段。
        if (not getattr(args, "allow_student_history", False)
                and roles not in (["user"], ["system", "user"])):
            raise DemoDataError("强参考的初始消息只能包含学生材料；后续回答使用本模型自己的历史")
    try:
        config = load_config(args.env_file) if profile == "primary" else load_config(args.env_file, profile=profile)
    except TeacherError as exc:
        raise DemoDataError(f"{profile} API 配置无效；未发送参考请求：{exc}") from None
    config.model_id = args.model_id or config.model_id
    config.max_output_tokens = args.max_output_tokens
    config.max_retries = 0
    _validate_reference_config(config)
    fingerprint = _fingerprint(config, args, cases_hash)
    by_case = {case["case_id"]: case for case in cases}
    total_turns = sum(1 + len(case["user_turns"]) for case in cases)

    try:
        with _account_generation_lock(output.parent):
            rows: list[dict] = []
            by_id: dict[str, dict] = {}
            if output.exists():
                if not args.resume:
                    raise DemoDataError("参考输出已存在；续跑请使用 --resume")
                try:
                    for line in output.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        cid = item.get("case_id")
                        if (cid not in by_case or cid in by_id or item.get("run_fingerprint") != fingerprint
                                or not isinstance(item.get("turns"), list)):
                            raise DemoDataError("参考输出包含未知题目、重复题目或不同运行指纹")
                        expected = list(by_case[cid]["input_messages"])
                        planned_turns = [None, *by_case[cid]["user_turns"]]
                        if len(item["turns"]) > len(planned_turns):
                            raise DemoDataError("参考输出的轮数超过题目")
                        for index, turn in enumerate(item["turns"]):
                            if not isinstance(turn, dict) or turn.get("user_turn") != planned_turns[index] or not isinstance(turn.get("answer"), str):
                                raise DemoDataError("参考输出的对话轮次无效")
                            if planned_turns[index] is not None:
                                expected.append({"role": "user", "content": planned_turns[index]})
                            expected.append({"role": "assistant", "content": turn["answer"]})
                        if item.get("full_dialogue") != expected:
                            raise DemoDataError("参考历史与本模型已保存的回答不一致")
                        if item.get("answers_by_turn") != [turn["answer"] for turn in item["turns"]]:
                            raise DemoDataError("参考输出的回答与逐轮记录不一致")
                        if any(turn.get("truncated") for turn in item["turns"]):
                            raise DemoDataError("已保存的参考回复不完整，需要先人工核对")
                        if item.get("status") not in {"partial", "completed"}:
                            raise DemoDataError("参考输出状态无效")
                        if item.get("status") == "completed" and len(item["turns"]) != len(planned_turns):
                            raise DemoDataError("参考输出标记完成但轮数不足")
                        by_id[cid] = item
                        rows.append(item)
                except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
                    raise DemoDataError("已有参考输出无法读取") from None
            if state_path.exists():
                if not args.resume:
                    raise DemoDataError("账户账本已存在；续跑请使用 --resume")
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    raise DemoDataError("参考账户账本无法读取") from None
                if (not isinstance(state, dict) or state.get("mode") != "account_balance_reference"
                        or state.get("teacher_profile", "primary") != profile
                        or state.get("run_fingerprint") != fingerprint or state.get("cases_sha256") != cases_hash):
                    raise DemoDataError("参考账户账本与当前题目或配置不一致")
            else:
                if rows:
                    raise DemoDataError("已有参考输出但缺少账户账本")
                state = {"mode": "account_balance_reference", "run_fingerprint": fingerprint,
                         "teacher_profile": profile,
                         "cases_sha256": cases_hash, "starting_balance": None, "last_balance": None}

            def save(status: str, **extra) -> None:
                completed_turns = sum(len(item["turns"]) for item in rows)
                state.update({"status": status, "total_cases": len(cases),
                              "completed_cases": sum(item.get("status") == "completed" for item in rows),
                              "total_turns": total_turns, "completed_turns": completed_turns,
                              "pending_turns": total_turns - completed_turns,
                              "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                state.update(extra)
                _atomic_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")

            # 崩溃时如果对应轮的完整输出已落盘，则可继续；否则不能判断是否计费。
            if state.get("status") == "request_unknown":
                cid, index = state.get("running_case_id"), state.get("running_turn_index")
                item = by_id.get(cid)
                if (not isinstance(index, int) or isinstance(index, bool) or index < 0
                        or item is None or len(item["turns"]) <= index):
                    raise DemoDataError("上一轮参考请求结果未知；请先核对服务记录，避免重复请求")
                save("active", running_case_id=None, running_turn_index=None)

            pending = [case for case in cases if by_id.get(case["case_id"], {}).get("status") != "completed"]
            if not pending:
                save("complete", running_case_id=None, running_turn_index=None)
                print("强参考已完成；账户账本记录了逐题和逐轮进度。")
                return 0
            if args.limit is not None:
                pending = pending[:args.limit]
            save("active", running_case_id=None, running_turn_index=None)
            requests = 0
            for case in pending:
                item = by_id.get(case["case_id"])
                if item is None:
                    item = {"case_id": case["case_id"], "kind": case["kind"], "split": case["split"],
                            "model": args.run_label, "step": args.step0, "requested_model": config.model_id,
                            "reported_model": None, "reported_models": [], "run_fingerprint": fingerprint,
                            "config": {"provider": config.provider, "api": config.openai_api,
                                       "max_output_tokens": config.max_output_tokens,
                                       "timeout_seconds": config.timeout_seconds, "max_retries": config.max_retries,
                                       "openai_max_tokens_field": config.openai_max_tokens_field,
                                       "extra_body": config.extra_body, "stream": False,
                                       "decoding": "service_default" if not config.extra_body else "configured"},
                            "cases_sha256": cases_hash, "status": "partial", "turns": [],
                            "answers_by_turn": [], "full_dialogue": list(case["input_messages"])}
                    by_id[case["case_id"]] = item
                    rows.append(item)
                history = list(item["full_dialogue"])
                planned_turns = [None, *case["user_turns"]]
                for index in range(len(item["turns"]), len(planned_turns)):
                    user_turn = planned_turns[index]
                    if user_turn is not None:
                        history.append({"role": "user", "content": user_turn})
                    try:
                        balance = _wallet(config)
                    except DemoDataError:
                        save("balance_unknown", running_case_id=None, running_turn_index=None, last_balance=None)
                        raise
                    if not isinstance(balance, (int, float)) or isinstance(balance, bool) or not math.isfinite(balance):
                        save("balance_unknown", running_case_id=None, running_turn_index=None, last_balance=None)
                        raise DemoDataError("余额无法确认；停止参考请求")
                    if state.get("starting_balance") is None:
                        state["starting_balance"] = balance
                    if balance <= 0:
                        save("balance_zero", running_case_id=None, running_turn_index=None, last_balance=balance)
                        raise DemoDataError("账户余额为零；已保存参考进度")
                    save("request_unknown", running_case_id=case["case_id"], running_turn_index=index,
                         last_balance=balance, last_error_category=None, last_http_status=None, last_error_reason=None)
                    began = time.perf_counter()
                    try:
                        response = call_teacher_messages(config, history)
                    except TeacherError as exc:
                        status, category = getattr(exc, "http_status", None), getattr(exc, "category", None)
                        reason = getattr(exc, "reason", None)
                        definite = isinstance(status, int) and not isinstance(status, bool) and status in {400, 401, 402, 403, 404, 413, 422, 429}
                        outcome = "balance_insufficient" if definite and reason == "insufficient_balance" else ("request_failed" if definite else "request_unknown")
                        save(outcome, running_case_id=None if definite else case["case_id"],
                             running_turn_index=None if definite else index,
                             last_error_category=category if category in {"quota_billing", "auth", "rate_limit", "unknown"} else "unknown",
                             last_http_status=status if isinstance(status, int) and not isinstance(status, bool) else None,
                             last_error_reason="insufficient_balance" if reason == "insufficient_balance" else None)
                        raise DemoDataError("强参考请求停止；已保存处理进度") from None
                    answer = response["text"]
                    if config.api_key and config.api_key in answer:
                        raise DemoDataError("模型输出包含凭据特征；保留未决状态并停止")
                    usage = response.get("usage") or {}
                    finish = response.get("finish_reason")
                    truncated = finish not in {"completed", "stop", "end_turn", "stop_sequence"}
                    turn = {"user_turn": user_turn, "answer": answer, "reported_model": response.get("reported_model"),
                            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                            "latency_seconds": round(time.perf_counter() - began, 4), "finish_reason": finish,
                            "truncated": truncated, "service_default_decoding": not bool(config.extra_body)}
                    item["turns"].append(turn)
                    item["answers_by_turn"].append(answer)
                    item["reported_models"].append(turn["reported_model"])
                    observed = {name for name in item["reported_models"] if isinstance(name, str) and name}
                    item["reported_model"] = next(iter(observed)) if len(observed) == 1 else None
                    history.append({"role": "assistant", "content": answer})
                    item["full_dialogue"] = history
                    item["input_tokens"] = sum(t.get("input_tokens") or 0 for t in item["turns"])
                    item["output_tokens"] = sum(t.get("output_tokens") or 0 for t in item["turns"])
                    item["latency_seconds"] = round(sum(t["latency_seconds"] for t in item["turns"]), 4)
                    item["truncated"] = any(t["truncated"] for t in item["turns"])
                    requests += 1
                    # 先落盘完整回复，再清除未决状态；即使中途停止也不会直接重发。
                    _save_rows(output, rows)
                    save("active", running_case_id=None, running_turn_index=None)
                    if truncated:
                        raise DemoDataError("强参考回复未正常完成；已保存检查点")
                item["answer"] = "\n\n".join(item["answers_by_turn"])
                item["status"] = "completed"
                item["finish_reason"] = "completed"
                item["created_utc"] = datetime.now(timezone.utc).isoformat()
                _save_rows(output, rows)
                save("active", running_case_id=None, running_turn_index=None)
            status = "complete" if all(item.get("status") == "completed" for item in rows) and len(rows) == len(cases) else "active"
            save(status, running_case_id=None, running_turn_index=None)
            print(json.dumps({"completed_cases": state["completed_cases"], "total_cases": len(cases),
                              "completed_turns": state["completed_turns"], "requests_this_run": requests}, ensure_ascii=False))
            return 0
    except BuildError as exc:
        raise DemoDataError(str(exc)) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model-id", help="强参考模型 ID；账户余额模式默认使用环境配置")
    parser.add_argument("--teacher-profile", choices=("primary", "backup", "tertiary"), default="primary",
                        help="强参考账户模式仅允许 primary profile")
    parser.add_argument("--run-label", default="claude-opus-5-5")
    parser.add_argument("--step0", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-student-history", action="store_true",
                        help="允许冻结题输入中已有的对话历史作为上下文；不包含目标参考答案、rubric、teacher scores、工具或服务端会话状态")
    parser.add_argument("--budget-state", type=Path, default=Path("本地资料/claude参考预算.json"))
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="只检查 cases，不读取凭据或联网")
    parser.add_argument("--use-account-balance", action="store_true", help="使用账户实时余额，不设置旧 Demo 保留额")
    args = parser.parse_args(argv)
    if args.step0 < 0 or args.max_output_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("step0 必须非负，max-output-tokens 和 limit 必须为正数")
    try:
        cases = _read_cases(args.cases, None if args.use_account_balance else args.limit)
        cases_hash = _sha(args.cases.read_bytes())
        if args.dry_run:
            print(json.dumps({"dry_run": True, "cases": len(cases), "cases_sha256": cases_hash}, ensure_ascii=False))
            return 0
        if args.use_account_balance:
            return _run_account(args, cases, cases_hash)
        # 旧预算模式始终使用现有主配置，CLI 的账户模式profile不影响它。
        args.effective_teacher_profile = "primary"
        from teacher_api import TeacherError, call_teacher_messages, load_config
        config = load_config(args.env_file)
        config.model_id = args.model_id or "claude-opus-5-5"
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
