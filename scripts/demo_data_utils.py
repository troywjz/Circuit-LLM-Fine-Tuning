"""QLoRA Demo 的 JSONL 校验与 assistant-only 标签构造。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DemoDataError(ValueError):
    """输入数据违反 Demo 训练契约。"""


def read_jsonl(path: Path, *, require_approval: bool = True) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL，只保留训练必需字段，绝不把审核元数据送入模型。"""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DemoDataError(f"无法读取 UTF-8 JSONL：{path.name}") from exc
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DemoDataError(f"第 {line_no} 行不是有效 JSON") from exc
        if not isinstance(item, dict):
            raise DemoDataError(f"第 {line_no} 行必须是 JSON 对象")
        sid = item.get("sample_id")
        if not isinstance(sid, str) or not sid.strip() or sid != sid.strip() or sid in seen:
            raise DemoDataError(f"第 {line_no} 行 sample_id 缺失、空白、格式无效或重复")
        seen.add(sid)
        for key in ("source_id", "group_id"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise DemoDataError(f"sample_id={sid} 的 {key} 必须是非空字符串")
        if item.get("split") not in {"train", "validation"}:
            raise DemoDataError(f"sample_id={sid} 的 split 必须是 train 或 validation")
        if require_approval and item.get("approved_for_demo") is not True:
            raise DemoDataError(f"sample_id={sid} 未明确标记 approved_for_demo=true")
        messages = item.get("messages")
        _validate_messages(sid, messages)
        # 白名单字段是结构元数据，messages 才是唯一模型输入。
        rows.append({"sample_id": sid, "source_id": item["source_id"],
                     "group_id": item["group_id"], "split": item["split"],
                     "messages": [{"role": m["role"], "content": m["content"]} for m in messages]})
    if not rows:
        raise DemoDataError(f"数据文件没有有效记录：{path.name}")
    return rows


def _validate_messages(sample_id: str, messages: Any) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        raise DemoDataError(f"sample_id={sample_id} 的 messages 至少需要一轮对话")
    if messages[0].get("role") == "system" if isinstance(messages[0], dict) else False:
        start = 1
    else:
        start = 0
    if start and len(messages) < 3:
        raise DemoDataError(f"sample_id={sample_id} 只有 system 消息，没有对话")
    if start and (not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip()):
        raise DemoDataError(f"sample_id={sample_id} 的 system 内容为空")
    # 输入正文不能伪造模板角色边界，否则会误把用户文字当成助手训练目标。
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            if any(marker in message["content"] for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")):
                raise DemoDataError(f"sample_id={sample_id} 正文含保留模板标记")
    expected = "user"
    for i in range(start, len(messages)):
        message = messages[i]
        if not isinstance(message, dict) or message.get("role") != expected:
            raise DemoDataError(f"sample_id={sample_id} 的消息角色必须 user/assistant 交替，位置 {i}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DemoDataError(f"sample_id={sample_id} 的消息内容不能为空，位置 {i}")
        if "<think>" in content.lower() or "</think>" in content.lower():
            raise DemoDataError(f"sample_id={sample_id} 含禁止的 think 标签")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1].get("role") != "assistant":
        raise DemoDataError(f"sample_id={sample_id} 必须以 assistant 回答结束")


def _ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                          enable_thinking=False)
    if isinstance(value, dict) or (hasattr(value, "keys") and "input_ids" in value.keys()):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def tokenize_assistant_only(tokenizer, messages: list[dict[str, str]], max_length: int) -> dict[str, Any]:
    """用实际 chat template 生成整段 token，并只给 assistant 回合及其 EOS 计损失。"""
    full_ids = _ids(tokenizer, messages)
    if len(full_ids) > max_length:
        raise DemoDataError(f"样本含 {len(full_ids)} tokens，超过 seq-length={max_length}；未截断")
    labels = [-100] * len(full_ids)
    assistant_tokens = 0
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    endthink_id = tokenizer.convert_tokens_to_ids("</think>")
    if None in (start_id, end_id, think_id, endthink_id):
        raise DemoDataError("tokenizer 缺少 Qwen chat template 必需的消息边界 token")
    # 从真实完整序列识别角色边界。逐前缀渲染会让动态模板把前一轮回答重写成 think 格式。
    role_header = tokenizer.encode("assistant\n", add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    blank_line_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    if not role_header or not newline_ids:
        raise DemoDataError("无法编码 Qwen assistant 消息头")
    position = 0
    while position < len(full_ids):
        if full_ids[position] != start_id:
            position += 1
            continue
        assistant_start = position + 1
        role_end = assistant_start + len(role_header)
        if full_ids[assistant_start:role_end] == role_header:
            stop = next((i for i in range(role_end, len(full_ids)) if full_ids[i] == end_id), None)
            if stop is None:
                raise DemoDataError("assistant 回合缺少 EOS/im_end token")
            answer_start = role_end
            # enable_thinking=False 的 Qwen 模板仍会在最后一轮生成空 think 包装；
            # 推理时模板也会提供它，因此不把这几个控制 token 当作 assistant 答案监督。
            if answer_start < stop and full_ids[answer_start] == think_id:
                closing = next((i for i in range(answer_start + 1, stop) if full_ids[i] == endthink_id), None)
                if closing is None:
                    raise DemoDataError("检测到未闭合 think 包装，拒绝训练")
                answer_start = closing + 1
                while True:
                    if blank_line_ids and full_ids[answer_start:answer_start + len(blank_line_ids)] == blank_line_ids:
                        answer_start += len(blank_line_ids)
                    elif full_ids[answer_start:answer_start + len(newline_ids)] == newline_ids:
                        answer_start += len(newline_ids)
                    else:
                        break
            for index in range(answer_start, stop + 1):
                labels[index] = full_ids[index]
            assistant_tokens += stop + 1 - answer_start
            position = stop + 1
        else:
            position += 1
    if assistant_tokens == 0 or all(x == -100 for x in labels):
        raise DemoDataError("样本没有可监督的 assistant tokens")
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not any(labels[i] == eos_id for i in range(len(labels))):
        raise DemoDataError("assistant 目标中未找到 tokenizer EOS token")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels, "token_count": len(full_ids),
            "assistant_label_tokens": assistant_tokens}
