"""对冻结的 Demo cases 批量运行本机 Base 或 Adapter 对照。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from demo_data_utils import DemoDataError
from qwen35_common import (CHAT_STOP_POLICY_VERSION, DEFAULT_MODEL_PATH,
                           effective_chat_stop_ids, load_unsloth_model, require_adapter)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cases(path: Path, limit: int | None) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DemoDataError("cases 文件无法以 UTF-8 读取") from exc
    cases, seen = [], set()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DemoDataError(f"cases 第 {line_no} 行不是有效 JSON") from exc
        if not isinstance(row, dict):
            raise DemoDataError(f"cases 第 {line_no} 行必须是 JSON 对象")
        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise DemoDataError(f"cases 第 {line_no} 行 case_id 缺失或重复")
        seen.add(cid)
        if row.get("kind") not in {"domain", "general"} or row.get("split") not in {"dev", "test"}:
            raise DemoDataError(f"case_id={cid} 的 kind 或 split 无效")
        messages = row.get("input_messages")
        if not isinstance(messages, list) or not messages:
            raise DemoDataError(f"case_id={cid} 缺少 input_messages")
        for i, msg in enumerate(messages):
            if (not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(msg.get("content"), str) or not msg["content"].strip()):
                raise DemoDataError(f"case_id={cid} 的 input_messages[{i}] 无效")
        if messages[-1]["role"] != "user":
            raise DemoDataError(f"case_id={cid} 的 input_messages 必须以 user 结束")
        turns = row.get("user_turns", [])
        if not isinstance(turns, list) or any(not isinstance(x, str) or not x.strip() for x in turns):
            raise DemoDataError(f"case_id={cid} 的 user_turns 必须为非空字符串数组")
        if any("<think>" in x.lower() or "</think>" in x.lower() for m in messages for x in [m["content"]]) or any(
                "<think>" in x.lower() or "</think>" in x.lower() for x in turns):
            raise DemoDataError(f"case_id={cid} 含禁止的 think 标签")
        cases.append({"case_id": cid, "kind": row["kind"], "split": row["split"],
                      "input_messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                      "user_turns": turns})
        # rubric/gold 等其它字段故意丢弃，不会传入 tokenizer 或模型。
        if limit and len(cases) >= limit:
            break
    if not cases:
        raise DemoDataError("cases 文件没有有效记录")
    return cases


def _append_case_atomic(path: Path, existing: list[dict], row: dict) -> None:
    existing.append(row)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in existing), encoding="utf-8")
    temp.replace(path)


def _run_fingerprint(cases_path: Path, model_path: Path, adapter_path: Path | None,
                    run_label: str, step: int, seq_length: int,
                    max_new_tokens: int, seed: int) -> tuple[str, str]:
    """用输入、基座配置与适配器真实权重标识一次不可混淆的运行。"""
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise DemoDataError("本地基座缺少 config.json，无法建立运行指纹")
    adapter_meta = None
    if adapter_path:
        adapter_path = adapter_path.resolve()
        weight = next((p for p in (adapter_path / "adapter_model.safetensors",
                                   adapter_path / "adapter_model.bin") if p.is_file()), None)
        if weight is None:
            raise DemoDataError("adapter 缺少权重文件，无法建立运行指纹")
        adapter_meta = {"path": str(adapter_path),
                        "config_sha256": _sha(adapter_path / "adapter_config.json"),
                        "weight_file": weight.name, "weight_sha256": _sha(weight)}
    cases_hash = _sha(cases_path)
    generation_config = model_path / "generation_config.json"
    payload = {"cases_sha256": cases_hash, "model_path": str(model_path),
               "model_config_sha256": _sha(config_path), "adapter": adapter_meta,
               "generation_config_sha256": _sha(generation_config) if generation_config.is_file() else None,
               "stop_policy_version": CHAT_STOP_POLICY_VERSION,
               "run_label": run_label, "step": step, "seq_length": seq_length,
               "max_new_tokens": max_new_tokens, "seed": seed}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), cases_hash


def _load_existing_output(path: Path, fingerprint: str) -> tuple[list[dict], set[str]]:
    """检查续跑文件属于唯一运行且没有重复 case，避免静默拼接实验。"""
    rows: list[dict] = []
    seen: set[str] = set()
    if not path.exists():
        return rows, seen
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DemoDataError("已有推理输出不可读取") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DemoDataError(f"已有推理输出第 {line_no} 行不是有效 JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise DemoDataError(f"已有推理输出第 {line_no} 行缺少 case_id")
        if row.get("run_fingerprint") != fingerprint:
            raise DemoDataError("输出文件包含不同运行指纹；请为不同模型/检查点使用独立输出文件")
        if row["case_id"] in seen:
            raise DemoDataError(f"已有推理输出重复 case_id：{row['case_id']}")
        seen.add(row["case_id"])
        rows.append(row)
    return rows, seen


def _append_turn(history: list[dict], user_turn: str | None, answer: str) -> list[dict]:
    """将本轮用户更正和本模型回答加入历史，供下一轮生成继续使用。"""
    updated = list(history)
    if user_turn is not None:
        updated.append({"role": "user", "content": user_turn})
    updated.append({"role": "assistant", "content": answer})
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    # 开发集首批384-token回答常在结论前截断；统一给Base和FT更充分的生成余量。
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-label", default="Base")
    parser.add_argument("--step", type=int, default=0,
                        help="模型检查点训练步数；基座或无训练步骤记录用 0")
    args = parser.parse_args(argv)
    if min(args.seq_length, args.max_new_tokens) < 1 or args.step < 0 or (args.limit is not None and args.limit < 1):
        parser.error("seq-length、max-new-tokens、limit 必须为正数")
    try:
        cases = _read_cases(args.cases, args.limit)
        adapter = require_adapter(args.adapter_path) if args.adapter_path else None
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fingerprint, input_hash = _run_fingerprint(args.cases, args.model_path, adapter,
                                                   args.run_label, args.step, args.seq_length,
                                                   args.max_new_tokens, args.seed)
        existing, done = _load_existing_output(output, fingerprint)
        pending = [case for case in cases if case["case_id"] not in done]
        if not pending:
            print(json.dumps({"resumed": True, "completed": len(done)}, ensure_ascii=False))
            return 0

        from unsloth import FastLanguageModel
        import torch
        FastLanguageModel, model, tokenizer, torch = load_unsloth_model(args.model_path, args.seq_length, "qlora")
        if adapter:
            from transformers import AutoTokenizer
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True, trust_remote_code=False)
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
        FastLanguageModel.for_inference(model)
        model.eval()
        stop_ids = effective_chat_stop_ids(model, tokenizer)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for case in pending:
            history = list(case["input_messages"])
            turns = [None, *case["user_turns"]]
            turn_results = []
            for user_turn in turns:
                if user_turn is not None:
                    # 更正必须在生成前进入本模型历史，否则第二轮仍会回答旧问题。
                    history.append({"role": "user", "content": user_turn})
                encoded = tokenizer.apply_chat_template(history, enable_thinking=False,
                                                        add_generation_prompt=True, return_tensors="pt", return_dict=True)
                input_tokens = int(encoded["input_ids"].shape[-1])
                if input_tokens + args.max_new_tokens > args.seq_length:
                    raise DemoDataError(
                        f"case_id={case['case_id']} 输入与最大生成长度合计 {input_tokens + args.max_new_tokens} "
                        f"超过 seq-length={args.seq_length}；请显式增大上下文或缩短输入/生成上限"
                    )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, use_cache=True,
                                               eos_token_id=stop_ids)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                generated_ids = generated[0, input_tokens:]
                output_tokens = int(generated_ids.numel())
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                last_token_id = int(generated_ids[-1]) if output_tokens else None
                stopped_on_eos = last_token_id in stop_ids
                hit_limit = output_tokens >= args.max_new_tokens and not stopped_on_eos
                stop_reason = "max_new_tokens" if hit_limit else "eos_or_stop"
                finish_reason = "truncated_at_limit" if hit_limit else "completed"
                # 用户回合已加入，只追加回答；保存的历史与实际模型输入一致。
                history = _append_turn(history, None, answer)
                turn_results.append({"user_turn": user_turn, "answer": answer,
                                     "stop_token_ids": stop_ids,
                                     "stop_policy_version": CHAT_STOP_POLICY_VERSION,
                                     "last_token_id": last_token_id,
                                     "stopped_on_eos": stopped_on_eos,
                                     "input_tokens": input_tokens, "output_tokens": output_tokens,
                                     "latency_seconds": round(elapsed, 4), "finish_reason": finish_reason,
                                     "stop_reason": stop_reason, "truncated": hit_limit,
                                     "peak_memory": {"allocated_GiB": round(torch.cuda.max_memory_allocated()/1024**3, 3),
                                                     "reserved_GiB": round(torch.cuda.max_memory_reserved()/1024**3, 3)}})
                del encoded, generated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            all_answers = [turn["answer"] for turn in turn_results]
            result = {"case_id": case["case_id"], "kind": case["kind"], "split": case["split"],
                      "model": args.run_label, "step": args.step,
                      "adapter_label": "adapter" if adapter else None,
                      "adapter_path_label": adapter.name if adapter else None,
                      "stop_token_ids": stop_ids,
                      "stop_policy_version": CHAT_STOP_POLICY_VERSION,
                      "run_fingerprint": fingerprint, "cases_sha256": input_hash,
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "answer": "\n\n".join(all_answers), "answers_by_turn": all_answers,
                      "turns": turn_results, "input_tokens": sum(t["input_tokens"] for t in turn_results),
                      "output_tokens": sum(t["output_tokens"] for t in turn_results),
                      "latency_seconds": round(sum(t["latency_seconds"] for t in turn_results), 4),
                      "finish_reason": "truncated_at_limit" if any(t["truncated"] for t in turn_results) else "completed",
                      "truncated": any(t["truncated"] for t in turn_results),
                      "full_dialogue": history}
            _append_case_atomic(output, existing, result)
            print(f"完成 {case['case_id']}")
        return 0
    except (DemoDataError, OSError, ValueError) as exc:
        print(f"拒绝执行：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
