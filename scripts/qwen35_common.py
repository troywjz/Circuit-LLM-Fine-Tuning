"""本地 Qwen3.5 脚本共用的检查与模型加载入口。

推理和试训都从这里进入，集中处理本地文件的基本检查、CUDA 前置条件和
Unsloth 加载参数，避免两个入口各自形成略有差异的加载逻辑。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = Path(r"D:\models\Qwen3.5-4B")


def require_local_model(model_path: Path) -> None:
    """检查模型目录的基本完整性，不联网，也不逐字节校验权重内容。

    配置 JSON、分片索引引用、文件存在且非空只能挡住常见的缺文件情况；
    不能证明权重没有损坏，也不能代替真正加载模型。
    """
    # 统一成绝对路径，后续错误信息和索引相对路径都以同一目录为准。
    model_path = model_path.expanduser().resolve()
    config = model_path / "config.json"
    if not model_path.is_dir() or not config.is_file():
        _missing_model(model_path, "未找到模型目录或 config.json")
    # 配置可读且能解析成 JSON，说明结构基本可读；这里不验证模型架构兼容性。
    try:
        json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"模型配置不可读：{config}\n{exc}") from exc

    # 分片模型通常由 index 的 weight_map 列出文件名；先检查索引再检查其引用。
    weights = [p for p in model_path.glob("*.safetensors") if p.is_file()]
    indexes = sorted(model_path.glob("*.safetensors.index.json"))
    if indexes:
        try:
            index = json.loads(indexes[0].read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            shards = sorted(set(weight_map.values()))
            if not weight_map or not shards:
                raise ValueError("weight_map 为空")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SystemExit(f"权重索引不可读：{indexes[0]}\n{exc}") from exc
        except ValueError as exc:
            raise SystemExit(f"权重索引无有效分片：{indexes[0]}\n{exc}") from exc
        missing = [name for name in shards if not (model_path / name).is_file()]
        if missing:
            _missing_model(model_path, "权重索引引用的分片不完整：" + ", ".join(missing))
        empty = [name for name in shards if (model_path / name).stat().st_size == 0]
        if empty:
            _missing_model(model_path, "权重索引分片为空文件：" + ", ".join(empty))
    elif not weights or not any(p.stat().st_size > 0 for p in weights):
        _missing_model(model_path, "目录中没有 safetensors 权重文件")


def require_adapter(adapter_path: Path) -> Path:
    """检查 PEFT adapter 的必要文件名是否齐全，不代表 adapter 已能成功加载。"""
    # 返回规范化路径，供 tokenizer 和 PEFT 后续只从本机目录读取。
    adapter_path = adapter_path.expanduser().resolve()
    if not adapter_path.is_dir():
        raise SystemExit(f"adapter 目录不存在：{adapter_path}")
    required = [adapter_path / "adapter_config.json", adapter_path / "tokenizer_config.json"]
    weight_files = [adapter_path / "adapter_model.safetensors", adapter_path / "adapter_model.bin"]
    missing = [str(p.name) for p in required if not p.is_file()]
    if not any(p.is_file() for p in weight_files):
        missing.append("adapter_model.safetensors 或 adapter_model.bin")
    if missing:
        raise SystemExit(f"adapter 目录缺少必需文件：{', '.join(missing)} ({adapter_path})")
    return adapter_path


def _missing_model(path: Path, reason: str) -> None:
    """用明确原因提示本地模型未就绪，并以状态码 2 结束当前命令。"""
    print(
        f"本地模型未就绪：{reason}\n模型目录：{path}\n"
        "脚本不会联网下载。请先在已激活的开发环境中手动执行：\n"
        f'modelscope download Qwen/Qwen3.5-4B --local-dir "{path}"',
        file=sys.stderr,
    )
    raise SystemExit(2)


def load_unsloth_model(model_path: Path, seq_length: int, method: str = "qlora"):
    """检查运行条件并加载基座；不负责训练，也不在本地改写基座文件。"""
    # method 与量化开关一一对应：默认 qlora；lora 则走 16-bit 未量化基座。
    if method not in ("qlora", "lora"):
        raise ValueError(f"不支持的训练方法：{method}")
    require_local_model(model_path)
    # 延迟导入让 --help 和缺模型快速失败不需要初始化 GPU 训练栈；Unsloth 要先于
    # torch/transformers/peft/trl 导入，以便它能先配置自己的兼容与优化路径。
    from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("未检测到可用 CUDA GPU；此脚本要求 CUDA 与 BF16 支持。")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("当前 CUDA GPU 不支持 BF16；请确认设备与驱动，脚本不会自动切换精度。")
    # BF16 主要指定计算及未量化部分的数据类型；QLoRA 的量化权重仍按 4-bit 存储，
    # 整个约 9 GB 文件也不会因此简单变成 1/4，嵌入等部分可能保留较高精度。
    # full_finetuning=False 配合 PEFT 只训练额外适配器；text_only=True 选择文本加载路径。
    # 本地读取、精确模型名、禁用远程代码共同避免意外联网或执行上游自定义代码。
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path.resolve()),  # 从这份已下载的模型目录读取权重和配置。
            max_seq_length=seq_length,  # 模型可处理的配置长度上限；实际输入仍受显存约束。
            dtype=torch.bfloat16,  # 请求 BF16（也是 16 位）；框架仍可能给数值敏感运算保留 FP32。
            load_in_4bit=(method == "qlora"),  # 仅 method 恰为 qlora 时为 True。
            load_in_16bit=(method == "lora"),  # lora 方法选择 16-bit 未量化基座路径。
            full_finetuning=False,  # 不训练全部原始权重；试训入口随后会加上可训练的 LoRA 适配器。
            text_only=True,  # 选择文本模型加载路径，不加载多模态视觉相关组件。
            fast_inference=False,  # 不启动 vLLM 等额外快速推理引擎，沿普通加载路径运行。
            use_exact_model_name=True,  # 按传入的本地模型目录精确解析，不自动改用其他名称。
            local_files_only=True,  # 只读本机缓存/目录，不向远端请求文件。
            trust_remote_code=False,  # 不执行模型仓库自带的自定义 Python 代码。
        )
    # 显存不足时停止并给出针对方法的建议，不悄悄改 CPU、精度或模型。
    except torch.cuda.OutOfMemoryError as exc:
        hint = (
            "BF16 LoRA 的未量化基座本身会占用大量显存；降低 rank 不能消除基座占用。"
            "请关闭其他显存占用后重试。"
            if method == "lora"
            else "请关闭占用显存的程序后重试，或尝试 --seq-length 256。"
        )
        raise SystemExit(f"加载模型时 CUDA OOM：{hint} 脚本不会在失败后自动改用 CPU 或更换模型。") from exc
    return FastLanguageModel, model, tokenizer, torch


def peak_memory_gib(torch):
    """读取本进程 CUDA 峰值；allocated/reserved 不等于 nvidia-smi 整卡占用。"""
    # GPU 操作通常异步执行；先等已提交的工作结束，再读取统计值。
    torch.cuda.synchronize()
    return {
        # allocated 是 PyTorch 张量实际占用的显存；这里取统计期间的最高值。
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        # reserved 是 PyTorch 向驱动预留的内存池，包括尚未交还的空闲块，因此可能更大。
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
    }
