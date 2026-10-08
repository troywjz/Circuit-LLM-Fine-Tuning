"""运行极小的本地 QLoRA 环境冒烟试训，不是能力 Benchmark。

脚本检查从数据格式化、训练步执行到 adapter 保存的链路；玩具对话和短步数
不能说明正式数据质量、上下文上限稳定性或模型电路能力有所提升。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from qwen35_common import DEFAULT_MODEL_PATH, load_unsloth_model, peak_memory_gib, require_local_model

SEED = 3407  # 固定随机种子便于做相近条件的对照，但不保证不同环境逐位一致。
DEFAULT_OUTPUT = Path("experiments/qlora/checkpoints/qwen35-4b-smoke")
# 这 8 条只是流程演示用的玩具样本，不是经过正式审查的训练集，也不是评测集。
TOY_DIALOGUES = [
    ("你好。", "你好！有什么电路规划问题需要一起梳理？"),
    ("欧姆定律是什么？", "欧姆定律表示在线性电阻条件下，电压等于电流乘以电阻：V=IR。"),
    ("mA 是什么单位？", "mA 是毫安，1 mA 等于 0.001 A。"),
    ("帮我设计一个采集传感器的电路。", "还需要确认传感器型号或输出类型、供电条件，以及采样速率和精度要求。"),
    ("做一个电池供电的数据采集器。", "请先补充电池类型与电压范围、采集信号类型，以及目标续航和采样指标。"),
    ("怎么选 MCU？", "先明确接口、处理负载、功耗、供电和成本约束，再比较候选 MCU；具体型号还需结合需求核实。"),
    ("传感器的引脚怎么接？", "需要传感器准确型号及其数据手册，才能可靠确认引脚和外围连接。"),
    ("需要一个通信接口。", "请说明对端设备、距离、速率和环境约束，以便判断合适的接口方案。"),
]


def main() -> int:
    # 先解析参数并拒绝明显无效值；用户可单独运行 --help，不必加载模型。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--method", choices=("qlora", "lora"), default="qlora")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.seq_length, args.rank, args.steps) < 1:
        parser.error("seq-length、rank 和 steps 必须为正整数")
    # 输出目录非空就停止，保护此前实验；每次重试请显式选新目录。
    require_local_model(args.model_path)
    out_dir = args.output_dir.resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"输出路径已存在但不是目录：{out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，请换一个新目录：{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一走 common 的本地检查与 QLoRA/LoRA 加载条件，再设置随机种子。
    FastLanguageModel, model, tokenizer, torch = load_unsloth_model(args.model_path, args.seq_length, args.method)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # PEFT 在基座上增加可训练 LoRA 矩阵；冻结的基座权重不随训练更新。
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,  # 新增低秩矩阵的维度；影响适配器大小，不会改变基座参数规模。
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 只给名称匹配的层加 LoRA 适配器。
        lora_alpha=2 * args.rank,  # LoRA 更新的缩放系数；这里按 rank 的两倍设置。
        lora_dropout=0,  # 训练时不随机丢弃 LoRA 输入；小型冒烟配置设为 0。
        bias="none",  # 不为目标层额外训练 bias 参数。
        use_gradient_checkpointing="unsloth",  # 需要时重算中间激活，以额外计算换较低显存占用。
        random_state=SEED,  # 固定适配器初始化相关随机性，便于相近条件对照。
        temporary_location=str(out_dir / "unsloth_temp"),  # Unsloth 临时文件放在本次输出目录下。
    )

    # 把 user/assistant 对话按模型模板串成纯文本样本，训练时不额外拼推理提示。
    # 这里把完整对话作为 text 交给训练器，没有设置只计算助手回答部分的损失；
    # 它用于跑通环境，正式训练数据与损失范围需要另外设计。
    rows = []
    for user, assistant in TOY_DIALOGUES:
        rows.append({"text": tokenizer.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )})
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    # 训练日志若出现 NaN/无穷 loss 或梯度范数，就立即中止，避免把异常结果当成功。
    class FiniteMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"检测到非有限 {key}={logs[key]}，终止试训")
            return control

    dataset = Dataset.from_list(rows)
    # 记录每条样本真实 token 数；seq_length=512 是允许上限，短样本不能证明 512 稳定。
    sample_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in dataset["text"]]
    sample_max_tokens = max(sample_lengths)
    # processing_class/tokenizer 负责把文本转成模型可读 token；数据列名由 dataset_text_field 指定。
    # max_steps 是优化器更新次数；batch=1、累积 4 次时每次更新通常合并 4 个 microbatch。
    # BF16 用作计算精度；adamw_8bit 压缩优化器状态，与基座是否 4-bit 量化是两回事。
    config = SFTConfig(
        max_length=args.seq_length,  # 每条输入允许的 token 上限；短样本不会自动填满该长度。
        dataset_num_proc=1,  # 数据预处理用一个进程，减少小型本地任务的并行开销。
        dataloader_num_workers=0,  # 数据加载在主进程执行，避免 Windows 多进程启动复杂度。
        packing=False,  # 不把多条短样本拼成一条长输入，便于保持样本边界清楚。
        dataset_text_field="text",  # 告诉 SFTTrainer 从每条记录的 text 字段取训练内容。
        per_device_train_batch_size=1,  # 每张 GPU 一次前向/反向只放一条样本。
        gradient_accumulation_steps=4,  # 累积 4 个 microbatch 的梯度后再做一次参数更新。
        max_steps=args.steps,  # 限定优化器更新次数；10 步通常约处理 40 个 microbatch。
        learning_rate=2e-4,  # LoRA 可训练参数的学习率，是试运行设置而非已验证最佳值。
        bf16=True,  # 使用 BF16 训练计算；基座量化设置由 common 的加载方法控制。
        fp16=False,  # 不同时启用 FP16，避免与 BF16 精度开关冲突。
        optim="adamw_8bit",  # 优化器状态用 8-bit 表示，不代表基座权重是 8-bit 或 4-bit。
        logging_steps=1,  # 每个优化器更新步记录一次训练日志，方便观察短跑情况。
        report_to="none",  # 不把日志发送到 W&B 等外部跟踪服务。
        save_strategy="no",  # 不保存中途 checkpoint；脚本结尾单独保存 adapter。
        seed=SEED,  # 将同一随机种子传给 Trainer，帮助尽量复现实验条件。
        logging_nan_inf_filter=False,  # 不过滤日志中的 NaN/Inf；保留异常供回调明确检查。
        output_dir=str(out_dir / "trainer"),  # Trainer 的预留输出目录；当前关闭中途保存，目录可能为空。
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,  # 提供分词器，使 Trainer 能将 text 字段转换成 token。
        callbacks=[FiniteMetricsCallback()],
    )
    # 真正更新参数发生在 trainer.train()；OOM 明确退出，不降级为 CPU 或伪报成功。
    # 从当前已有显存占用重新统计后续峰值；这不会释放模型或清空实际显存。
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        if args.method == "lora":
            advice = (
                "BF16 LoRA 的未量化基座本身会占用大量显存，降低 rank 不能消除基座占用；"
                "请关闭其他显存占用后重试，或改用默认 QLoRA。"
            )
        else:
            advice = "可尝试 --seq-length 256 --rank 4，并指定新的 --output-dir。"
        raise SystemExit(
            f"CUDA OOM：{advice} 脚本不会在失败后自动更改参数、改用 CPU 或将失败记作成功。"
        ) from exc
    actual_steps = int(trainer.state.global_step)
    train_loss = result.metrics.get("train_loss")
    logs = trainer.state.log_history
    if actual_steps != args.steps or train_loss is None or not math.isfinite(float(train_loss)):
        raise SystemExit(f"试训结果异常：global_step={actual_steps}, train_loss={train_loss}")
    for log in logs:
        for key in ("loss", "grad_norm"):
            if key in log and not math.isfinite(float(log[key])):
                raise SystemExit(f"发现非有限训练指标 {key}={log[key]}；不保存为成功结果")

    # 只保存 PEFT adapter 和 tokenizer，不覆盖本机原始基座权重目录。
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    required_adapter_files = [adapter_dir / "adapter_config.json", adapter_dir / "tokenizer_config.json"]
    adapter_weights = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    missing_saved = [path.name for path in required_adapter_files if not path.is_file()]
    if not any(path.is_file() and path.stat().st_size > 0 for path in adapter_weights):
        missing_saved.append("adapter_model.safetensors 或 adapter_model.bin")
    if missing_saved:
        raise SystemExit(f"adapter 保存后检查失败，缺少文件：{', '.join(missing_saved)}")
    memory = peak_memory_gib(torch)
    # 报告记录环境版本、样本长度、训练步数和进程显存峰值，便于追溯这次冒烟运行。
    # 峰值 allocated/reserved 不是整卡 nvidia-smi 占用；UTC 时间用于跨时区记录。
    package_versions = {}
    for package in ("torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "bitsandbytes"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    report = {
        "result_type": "environment_smoke_test",
        "method": args.method,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "configured_max_length": args.seq_length,
        "sample_max_tokens": sample_max_tokens,
        "sample_count": len(rows),
        "rank": args.rank,
        "requested_steps": args.steps,
        "global_step": actual_steps,
        "train_loss": float(train_loss),
        "seed": SEED,
        "package_versions": package_versions,
        "gpu_name": torch.cuda.get_device_name(0),
        **memory,
        "limitations": "短玩具对话只用于环境冒烟检查；不是真实训练集或Benchmark，不证明电路能力提升，也不证明配置长度稳定性。",
    }
    # 报告只描述本次小规模流程结果，不代表 Benchmark 通过或能力提升。
    (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Adapter saved to: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
