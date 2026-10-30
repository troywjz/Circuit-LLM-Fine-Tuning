"""把正式实验的来源候选按家族分组，准备训练、开发和最终集的候选名单。

本程序只划分已经收集的来源，不调用教师模型，不批准训练，也不生成考题。
运行前应先准备旧实验的保留来源清单；结果保存在“本地资料/正式实验”。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "本地资料" / "正式实验"
RESERVED_STATUS = "proposed_reserved_sources_not_final_cases"
SPLITS = ("train", "development", "final")


def _title(value: Any) -> str:
    """用小写、去标点后的标题检查近似题；原始标题保持不变。"""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\W+", "", html.unescape(value).casefold(), flags=re.UNICODE)


def _similar(left: str, right: str, threshold: float) -> bool:
    if not left or not right:
        return False
    # 长度差过大时，相似度理论上达不到门槛，无需继续逐字符比较。
    if 2 * min(len(left), len(right)) / (len(left) + len(right)) < threshold:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold


def _source_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _question_id(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip():
        raise ValueError(f"{name} 必须是有效的原题 ID")
    return str(value).strip()


def _read_reserved(reserved: dict[str, Any]) -> tuple[set[str], set[str], set[str], list[str]]:
    """只取旧实验的来源标识和标题，实际测试题、答案不参与划分。"""
    if not isinstance(reserved, dict) or reserved.get("status") != RESERVED_STATUS:
        raise ValueError("旧评测保留清单的状态无效")
    sources = reserved.get("sources")
    excluded_ids = reserved.get("excluded_question_ids", [])
    if not isinstance(sources, list) or not isinstance(excluded_ids, list):
        raise ValueError("旧评测保留清单结构无效")
    ids: set[str] = set()
    groups: set[str] = set()
    questions = {_question_id(x, "excluded_question_ids") for x in excluded_ids}
    titles: list[str] = []
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("旧评测来源条目无效")
        source_ids = item.get("source_ids", [])
        if not isinstance(source_ids, list):
            raise ValueError("旧评测 source_ids 应为数组")
        for value in source_ids + [item.get("source_id")]:
            if isinstance(value, str) and value.strip():
                ids.add(value.strip())
        for name in ("group_id", "project_id"):
            value = item.get(name)
            if isinstance(value, str) and value.strip():
                groups.add(value.strip())
        if item.get("question_id") is not None:
            questions.add(_question_id(item["question_id"], "旧评测 question_id"))
        title = _title(item.get("title_original"))
        if title:
            titles.append(title)
    return ids, groups, questions, titles


def build_plan(rows: list[dict[str, Any]], reserved: dict[str, Any], seed: int = 3407) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """分组并给来源分配候选用途，结果仍待内容和许可审核。"""
    if not isinstance(rows, list) or not rows or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("候选来源或随机种子无效")
    reserved_ids, reserved_groups, reserved_qids, reserved_titles = _read_reserved(reserved)
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("每条候选来源必须是对象")
        sid = _source_id(row.get("source_id"), "source_id")
        if sid in seen:
            raise ValueError("候选来源 source_id 重复")
        seen.add(sid)
        _source_id(row.get("group_id"), "group_id")
        _source_id(row.get("project_id"), "project_id")
        _question_id(row.get("question_id"), "question_id")
        if row.get("training_approved") is not False:
            raise ValueError("此入口只处理尚未批准训练的来源候选")
        raw = row.get("raw_record")
        if not isinstance(raw, dict) or not isinstance(raw.get("question"), dict):
            raise ValueError("候选缺少原始问题元数据")
        duplicates = raw.get("duplicate_question_ids", [])
        if not isinstance(duplicates, list):
            raise ValueError("duplicate_question_ids 应为数组")
        for duplicate in duplicates:
            _question_id(duplicate, "duplicate_question_ids")
        for field in ("question_text", "answer_text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{field} 必须是非空文本")
        # 按来源 ID 排序，使同一批数据的输入顺序不会影响最终分配。
        prepared.append(dict(row))
    prepared.sort(key=lambda x: x["source_id"])

    # 同一项目、原题、显式重复题或高度相似标题合并成一个家族。
    parent = list(range(len(prepared)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    groups: dict[tuple[str, str], int] = {}
    qid_index: dict[str, int] = {}
    for index, row in enumerate(prepared):
        for name in ("project_id", "group_id"):
            key = (name, row[name])
            if key in groups:
                union(index, groups[key])
            else:
                groups[key] = index
        qid = str(row["question_id"]).strip()
        if qid in qid_index:
            union(index, qid_index[qid])
        else:
            qid_index[qid] = index
    titles = [_title(r["raw_record"]["question"].get("title")) for r in prepared]
    for index, row in enumerate(prepared):
        for duplicate in row["raw_record"].get("duplicate_question_ids", []):
            other = qid_index.get(str(duplicate).strip())
            if other is not None:
                union(index, other)
        for other in range(index):
            if _similar(titles[index], titles[other], 0.92):
                union(index, other)

    families: dict[int, list[int]] = {}
    for index in range(len(prepared)):
        families.setdefault(find(index), []).append(index)
    family_rows = sorted(families.values(), key=lambda group: prepared[group[0]]["source_id"])
    old: set[int] = set()
    media: set[int] = set()
    for family_index, members in enumerate(family_rows):
        for index in members:
            row = prepared[index]
            duplicates = {str(x).strip() for x in row["raw_record"].get("duplicate_question_ids", [])}
            matches_old = (
                row["source_id"] in reserved_ids
                or row["group_id"] in reserved_groups
                or row["project_id"] in reserved_groups
                or str(row["question_id"]).strip() in reserved_qids
                or bool(duplicates & reserved_qids)
                or str(row.get("isolation_status", "")).startswith("isolated_reserved")
                or any(_similar(titles[index], title, 0.86) for title in reserved_titles)
            )
            if matches_old:
                old.add(family_index)
            if row.get("source_media_review") or row["raw_record"].get("needs_media"):
                media.add(family_index)

    # 先隔离旧评测家族和需要图片核对的家族，再给其余家族按固定种子分配用途。
    eligible = [i for i in range(len(family_rows)) if i not in old and i not in media]
    random.Random(seed).shuffle(eligible)
    total = len(eligible)
    assignments: dict[int, str] = {}
    for position, family_index in enumerate(eligible):
        assignments[family_index] = (
            "train" if position < int(total * 0.8)
            else "development" if position < int(total * 0.9)
            else "final"
        )

    seen_families: set[str] = set()
    for family_index, members in enumerate(family_rows):
        source_ids = sorted(prepared[i]["source_id"] for i in members)
        family_id = hashlib.sha256("\0".join(source_ids).encode("utf-8")).hexdigest()[:20]
        if family_id in seen_families:
            raise ValueError("家族标识发生碰撞，请检查输入")
        seen_families.add(family_id)
        status = "isolated_reserved_pending_review" if family_index in old else (
            "media_pending_review" if family_index in media else "unreviewed_candidate"
        )
        for index in members:
            prepared[index]["family_id"] = family_id
            prepared[index]["split_candidate"] = assignments.get(family_index, "excluded_pending_review")
            prepared[index]["split_status"] = status
    split_counts = Counter(r["split_candidate"] for r in prepared)
    status_counts = Counter(r["split_status"] for r in prepared)
    summary = {
        "rows": len(prepared),
        "families": len(family_rows),
        "eligible_families": len(eligible),
        "split_counts": {name: split_counts.get(name, 0) for name in (*SPLITS, "excluded_pending_review")},
        "status_counts": dict(sorted(status_counts.items())),
        "seed": seed,
        "training_approved": False,
        "note": "来源候选分组；内容和许可待审，尚未冻结正式题目。",
    }
    return prepared, summary


def validate_output_path(target: Path, private_root: Path) -> Path:
    """仅允许在正式实验私有目录写新文件，杜绝覆盖现有记录。"""
    root = Path(private_root).resolve()
    path = Path(target).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("输出必须位于本地资料/正式实验目录") from exc
    if not relative.parts or path.exists():
        raise FileExistsError("输出文件已经存在或指向目录")
    return path


def _write_new_pair(output: Path, content: bytes, summary_path: Path, summary: bytes) -> None:
    """两个目标均以独占方式创建；发生错误时清理本次创建的不完整文件。"""
    if output == summary_path:
        raise ValueError("候选文件和摘要必须使用不同路径")
    created: list[Path] = []
    try:
        for path, payload in ((output, content), (summary_path, summary)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                created.append(path)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="未批准训练的来源候选 JSONL")
    parser.add_argument("--reserved", type=Path, required=True, help="旧评测保留来源清单")
    parser.add_argument("--output", type=Path, required=True, help="正式实验目录内的新候选文件")
    parser.add_argument("--summary", type=Path, required=True, help="正式实验目录内的新摘要文件")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    output = validate_output_path(args.output, PRIVATE_ROOT)
    summary_path = validate_output_path(args.summary, PRIVATE_ROOT)
    raw = args.input.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    reserved_raw = args.reserved.read_bytes()
    reserved = json.loads(reserved_raw.decode("utf-8-sig"))
    planned, stats = build_plan(records, reserved, args.seed)
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in planned).encode("utf-8")
    stats.update({"input_sha256": hashlib.sha256(raw).hexdigest(),
                  "reserved_sha256": hashlib.sha256(reserved_raw).hexdigest(),
                  "output_sha256": hashlib.sha256(content).hexdigest()})
    summary = (json.dumps(stats, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _write_new_pair(output, content, summary_path, summary)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
