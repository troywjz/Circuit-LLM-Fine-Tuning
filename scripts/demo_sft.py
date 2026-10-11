"""在本地以 assistant-only loss 运行 QLoRA Demo。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from demo_data_utils import DemoDataError, read_jsonl, tokenize_assistant_only
from qwen35_common import DEFAULT_MODEL_PATH, load_unsloth_model

DEFAULT_OUTPUT = Path("本地资料/Demo-V1/实验/run-001")
SEED = 3407


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _batch(rows):
    return [{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-steps", default="15,30,60")
    parser.add_argument("--max-samples", type=int, help="只用于小型 smoke 的样本上限")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据与本地 tokenizer，不加载 GPU 模型")
    args = parser.parse_args(argv)
    if min(args.seq_length, args.rank, args.alpha, args.steps, args.gradient_accumulation) < 1:
        parser.error("长度、rank、alpha、steps 和 gradient-accumulation 必须为正数")
    if args.learning_rate <= 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("learning-rate 和 max-samples 必须为正数")
    try:
        save_steps = sorted({int(x) for x in args.save_steps.split(",") if x.strip()})
    except ValueError:
        parser.error("save-steps 需为逗号分隔的正整数")
    if not save_steps or any(x < 1 or x > args.steps for x in save_steps):
        parser.error("save-steps 必须是 steps 范围内的正整数")

    try:
        train_rows = read_jsonl(args.train)
        if any(row["split"] != "train" for row in train_rows):
            raise DemoDataError("--train 文件中所有记录的 split 必须为 train")
        if args.validation:
            valid_rows = read_jsonl(args.validation)
            if any(row["split"] != "validation" for row in valid_rows):
                raise DemoDataError("--validation 文件中所有记录的 split 必须为 validation")
            train_ids = {row["sample_id"] for row in train_rows}
            train_groups = {row["group_id"] for row in train_rows}
            if train_ids & {row["sample_id"] for row in valid_rows}:
                raise DemoDataError("train 与 validation 存在重复 sample_id")
            if train_groups & {row["group_id"] for row in valid_rows}:
                raise DemoDataError("train 与 validation 存在相同 group_id，可能造成切分泄漏")
        else:
            valid_rows = []
        if args.max_samples:
            train_rows = train_rows[:args.max_samples]
        if not train_rows:
            raise DemoDataError("筛选后没有训练样本")
        # 正式训练时 Unsloth 必须在 Transformers/torch 前初始化；dry-run 只加载 tokenizer。
        if not args.dry_run:
            from unsloth import FastLanguageModel
        # 加载本机 tokenizer 读取真实模板；dry-run 路径不会初始化 CUDA 模型。
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path.resolve()), local_files_only=True,
                                                  trust_remote_code=False)
        encoded_train = [tokenize_assistant_only(tokenizer, row["messages"], args.seq_length) for row in train_rows]
        encoded_valid = [tokenize_assistant_only(tokenizer, row["messages"], args.seq_length) for row in valid_rows]
        if args.dry_run:
            print(json.dumps({"dry_run": True, "train_samples": len(encoded_train),
                              "validation_samples": len(encoded_valid),
                              "train_token_counts": [r["token_count"] for r in encoded_train],
                              "assistant_label_tokens": [r["assistant_label_tokens"] for r in encoded_train],
                              "max_length": args.seq_length}, ensure_ascii=False))
            return 0

        out = args.output_dir.resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise DemoDataError("输出目录已存在且非空；请使用新的 run 目录")
        out.mkdir(parents=True, exist_ok=True)
        # Unsloth 必须先导入，再导入其余训练栈，符合本地版本的初始化要求。
        from unsloth import FastLanguageModel  # noqa: F401
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments
        from datasets import Dataset

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        try:
            _, model, tokenizer, torch = load_unsloth_model(args.model_path, args.seq_length, "qlora")
            model = FastLanguageModel.get_peft_model(
                model, r=args.rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=args.alpha, lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth", random_state=args.seed,
                temporary_location=str(out / "unsloth_temp"),
            )
        except (torch.cuda.OutOfMemoryError, SystemExit) as exc:
            message = str(exc)
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "OOM" in message:
                _atomic_json(out / "失败报告.json", {"status": "cuda_oom", "error_type": type(exc).__name__,
                                                      "message": message[:500], "global_step": 0,
                                                      "created_utc": datetime.now(timezone.utc).isoformat()})
            raise

        class DataCollator:
            def __call__(self, features):
                width = max(len(x["input_ids"]) for x in features)
                pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (width-len(x["input_ids"])) for x in features]),
                        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (width-len(x["input_ids"])) for x in features]),
                        "labels": torch.tensor([x["labels"] + [-100] * (width-len(x["labels"])) for x in features])}

        metrics_path = out / "训练指标.jsonl"
        class DemoCallback(TrainerCallback):
            def on_log(self, _args, state, _control, logs=None, **_kwargs):
                row = {"global_step": state.global_step, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "sample_lengths": [x["token_count"] for x in encoded_train],
                       "assistant_label_tokens": [x["assistant_label_tokens"] for x in encoded_train]}
                for key in ("loss", "grad_norm", "learning_rate"):
                    if logs and key in logs:
                        value = float(logs[key])
                        if not math.isfinite(value):
                            raise FloatingPointError(f"non-finite {key}")
                        row[key] = value
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            def on_step_end(self, _args, state, _control, model=None, **_kwargs):
                if state.global_step in save_steps:
                    save_adapter(model, tokenizer, out / f"adapter-step-{state.global_step}")

        def save_adapter(current_model, current_tokenizer, dest: Path):
            dest.mkdir(parents=True, exist_ok=True)
            current_model.save_pretrained(str(dest))
            current_tokenizer.save_pretrained(str(dest))

        train_dataset = Dataset.from_list(_batch(encoded_train))
        validation_dataset = Dataset.from_list(_batch(encoded_valid)) if encoded_valid else None
        config = TrainingArguments(
            output_dir=str(out / "trainer"), per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
            max_steps=args.steps, learning_rate=args.learning_rate, bf16=True, fp16=False,
            optim="adamw_8bit", logging_steps=1, logging_nan_inf_filter=False,
            save_strategy="no", report_to="none", seed=args.seed,
            dataloader_num_workers=0, remove_unused_columns=False,
            gradient_checkpointing=False,
        )
        trainer = Trainer(model=model, args=config, train_dataset=train_dataset,
                          eval_dataset=validation_dataset, data_collator=DataCollator(),
                          callbacks=[DemoCallback()])
        torch.cuda.reset_peak_memory_stats()
        try:
            result = trainer.train()
            if trainer.state.global_step != args.steps:
                raise RuntimeError(f"实际步数 {trainer.state.global_step} 与目标 {args.steps} 不同")
            save_adapter(model, tokenizer, out / "adapter-final")
            report = {"result_type": "demo_sft_run", "status": "completed",
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "model_path": str(args.model_path.resolve()), "train_file": str(args.train.resolve()),
                      "validation_file": str(args.validation.resolve()) if args.validation else None,
                      "sample_count": len(train_rows), "validation_count": len(valid_rows),
                      "sample_token_counts": [x["token_count"] for x in encoded_train],
                      "assistant_label_tokens": [x["assistant_label_tokens"] for x in encoded_train],
                      "seq_length": args.seq_length, "rank": args.rank, "alpha": args.alpha,
                      "learning_rate": args.learning_rate, "requested_steps": args.steps,
                      "global_step": trainer.state.global_step, "gradient_accumulation": args.gradient_accumulation,
                      "seed": args.seed, "train_loss": float(result.metrics["train_loss"]),
                      "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                      "limitations": "Demo 训练结果不代表物理验收或 Benchmark 通过。"}
            if not math.isfinite(report["train_loss"]):
                raise FloatingPointError("non-finite train_loss")
            _atomic_json(out / "训练报告.json", report)
        except Exception as exc:
            # OOM 与数值异常记录失败状态，不更改参数，不伪报成功。
            status = "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed"
            _atomic_json(out / "失败报告.json", {"status": status, "error_type": type(exc).__name__,
                                                  "message": str(exc)[:500], "global_step": trainer.state.global_step,
                                                  "created_utc": datetime.now(timezone.utc).isoformat()})
            raise
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (DemoDataError, OSError, ValueError) as exc:
        print(f"拒绝执行：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
