"""Import manually obtained external teacher JSONL as unapproved candidates.

Hashes are local fingerprints of the source row, supplied prompt version,
manual configuration, and issues. prompt_hash represents input and prompt
version; it does not store WorkBuddy's full prompt or prove a model call.
This script is offline and never reads credentials or environment variables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


class ImportErrorWithContext(Exception):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ImportErrorWithContext(f"{path} 第 {line_no} 行 JSON 无效：{exc.msg}") from None
                if not isinstance(row, dict):
                    raise ImportErrorWithContext(f"{path} 第 {line_no} 行必须是 JSON 对象")
                rows.append((line_no, row))
    except OSError as exc:
        raise ImportErrorWithContext(f"无法读取 {path}: {exc}") from None
    return rows


def validate_messages(messages: Any, line_no: int) -> None:
    if not isinstance(messages, list) or not 2 <= len(messages) <= 12:
        raise ImportErrorWithContext(f"教师文件第 {line_no} 行 messages 必须有 2 至 12 条消息")
    expected = "user"
    for index, message in enumerate(messages, 1):
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message.get("role") != expected or not isinstance(message.get("content"), str)
                or not message["content"].strip()):
            raise ImportErrorWithContext(
                f"教师文件第 {line_no} 行第 {index} 条消息无效：需非空文本且 user/assistant 交替")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ImportErrorWithContext(f"教师文件第 {line_no} 行最后一条消息必须是 assistant")


def run(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(args.source_pack)
    teacher_rows = read_jsonl(args.teacher_jsonl)
    if not source_rows:
        raise ImportErrorWithContext("来源包为空，没有可匹配的训练来源")
    if not teacher_rows:
        raise ImportErrorWithContext("教师 JSONL 为空，没有可导入的候选")
    for name, value in (("provider", args.provider), ("model", args.model),
                        ("prompt-version", args.prompt_version)):
        if not isinstance(value, str) or not value.strip():
            raise ImportErrorWithContext(f"{name} 不能为空")
    sources: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_no, row in source_rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise ImportErrorWithContext(f"来源包第 {line_no} 行缺少有效 sample_id")
        if not isinstance(row.get("source_id"), str) or not row["source_id"].strip():
            raise ImportErrorWithContext(f"来源包第 {line_no} 行缺少有效 source_id")
        if sid in sources:
            raise ImportErrorWithContext(f"来源包第 {line_no} 行 sample_id 重复：{sid}")
        sources[sid] = (line_no, row)
    seen: set[str] = set()
    candidates = []
    for line_no, result in teacher_rows:
        sid = result.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行缺少有效 sample_id")
        if sid in seen:
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行 sample_id 重复：{sid}")
        seen.add(sid)
        if sid not in sources:
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行 sample_id 不在来源包中：{sid}")
        _, source = sources[sid]
        if source.get("split") != "train":
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行对应来源不是 split=train：{sid}")
        if result.get("source_id") != source.get("source_id"):
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行 source_id 与来源包不一致：{sid}")
        validate_messages(result.get("messages"), line_no)
        issues = result.get("issues")
        if not isinstance(issues, (str, list)):
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行 issues 必须是字符串或列表")
        if isinstance(issues, list) and any(not isinstance(item, str) for item in issues):
            raise ImportErrorWithContext(f"教师文件第 {line_no} 行 issues 列表只能包含字符串")
        for field in ("group_id", "project_id", "task", "split", "sources"):
            if field not in source:
                raise ImportErrorWithContext(f"来源包第 {sources[sid][0]} 行缺少候选必需字段 {field}")
        if not isinstance(source.get("group_id"), str) or not source["group_id"].strip():
            raise ImportErrorWithContext(f"来源包第 {sources[sid][0]} 行缺少有效 group_id")
        if not isinstance(source.get("task"), str) or not source["task"].strip():
            raise ImportErrorWithContext(f"来源包第 {sources[sid][0]} 行缺少有效 task")
        if not isinstance(source.get("sources"), list) or not source["sources"]:
            raise ImportErrorWithContext(f"来源包第 {sources[sid][0]} 行 sources 必须是非空列表")
        row_hash = digest(source)
        prompt_hash = digest({"source_row": source, "prompt_version": args.prompt_version})
        config = {"provider": args.provider, "model": args.model,
                  "prompt_version": args.prompt_version, "mode": "external_manual_import"}
        review_note_hash = digest({"issues": issues})
        candidate = {field: source[field] for field in
                     ("sample_id", "source_id", "group_id", "project_id", "task", "split")}
        candidate.update({"messages": result["messages"], "sources": source["sources"],
                          "teacher": {"provider": args.provider, "model": args.model,
                                      "origin": "external_manual", "issues": issues},
                          "teacher_profile": args.provider,
                          "evidence_scope": "外部手工导入；内容与工程事实仍待主 Agent 审核，未批准用于 Demo",
                          "modifications": ["从外部教师 JSONL 手工导入；保留 issues，待主审"],
                          "review": {"status": "pending_main_review", "approved_for_demo": False},
                          "row_hash": row_hash, "prompt_hash": prompt_hash,
                          "config_fingerprint": digest(config), "review_note_hash": review_note_hash})
        candidates.append(candidate)
    out = args.out
    if out.exists():
        raise ImportErrorWithContext(f"输出文件已存在，拒绝覆盖：{out}")
    if not args.apply:
        print(f"dry-run：校验通过 {len(candidates)} 条；未写入 {out}。加 --apply 才会写入。")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8", newline="\n") as stream:
            for candidate in candidates:
                stream.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError as exc:
        raise ImportErrorWithContext(f"无法创建输出文件 {out}: {exc}") from None
    print(f"已写入 {len(candidates)} 条待审核候选：{out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="离线导入外部教师 JSONL 为未批准候选")
    parser.add_argument("--source-pack", required=True, type=Path)
    parser.add_argument("--teacher-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", required=True,
                        help="提示词版本标识；prompt_hash 基于来源行与此版本，不保存完整提示词，也不证明模型调用")
    parser.add_argument("--apply", action="store_true", help="写入新文件；默认仅 dry-run")
    args = parser.parse_args()
    try:
        return run(args)
    except ImportErrorWithContext as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
