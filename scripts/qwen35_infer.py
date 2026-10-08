"""用 Unsloth 在本机运行 Qwen3.5-4B 纯文本推理。

默认以 QLoRA 方式加载 4-bit 基座；仅指定 adapter 时才叠加已训练的增量权重。
这个入口只生成回答，不训练、不保存新权重，也不改动原始模型目录。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qwen35_common import DEFAULT_MODEL_PATH, load_unsloth_model, peak_memory_gib, require_adapter


def main() -> int:
    # 参数解析放在模型导入之前，因此 --help 不需要 CUDA 或模型文件。
    parser = argparse.ArgumentParser(description=__doc__)
    # method 默认 qlora；改成 lora 时加载 BF16 基座。前后对照建议保持相同加载精度，
    # 这样回答变化更容易归因于适配器；并非适配器只能在一种量化精度上使用。
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--method", choices=("qlora", "lora"), default="qlora")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt", default="你好，请用一句话介绍欧姆定律。")
    parser.add_argument("--adapter-path", type=Path, help="可选：同基座训练所得 PEFT adapter")
    args = parser.parse_args()
    if args.seq_length < 1 or args.max_new_tokens < 1:
        parser.error("seq-length 和 max-new-tokens 必须为正整数")

    # 不带 adapter 就是基座模型推理；带路径才检查并加载该目录中的 LoRA 增量。
    adapter_path = require_adapter(args.adapter_path) if args.adapter_path else None
    FastLanguageModel, model, tokenizer, torch = load_unsloth_model(args.model_path, args.seq_length, args.method)
    if adapter_path:
        # adapter 附带的 tokenizer 配置需与训练时保持一致；权重仍从本地 adapter 目录读。
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), local_files_only=True, trust_remote_code=False
        )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    # 切到 Unsloth 的推理模式，让框架使用适合生成回答的执行方式。
    FastLanguageModel.for_inference(model)
    # eval() 让 Dropout 等层进入评估模式；它本身不会关闭梯度计算。
    model.eval()
    # 用 chat template 补上模型所需的 user 角色边界；关闭 thinking，并请求词典形式，
    # 这样 input_ids 与 attention_mask 等张量能作为命名参数交给 generate。
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        enable_thinking=False,
        add_generation_prompt=True,  # 在用户消息后补上‘该助手接着回答了’的模板标记。
        return_tensors="pt",  # 返回 PyTorch 能直接计算的张量，而不是普通数字列表。
        return_dict=True,  # 分开保留 input_ids、attention_mask 等字段。
    )
    # tokenizer 通常先在 CPU 产出张量，逐项搬到模型所在设备后才能一起推理。
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    # 输入长度加最大续写长度不能超过配置上限；短 prompt 也会实际占用上下文预算。
    input_len = inputs["input_ids"].shape[-1]
    if input_len + args.max_new_tokens > args.seq_length:
        raise SystemExit(
            f"输入与生成长度合计 {input_len + args.max_new_tokens} 超过 seq-length={args.seq_length}；"
            "请缩短 prompt 或调整 --seq-length / --max-new-tokens。"
        )
    # 从当前占用重新统计后续峰值，不释放模型，也不会把当前显存占用清零。
    torch.cuda.reset_peak_memory_stats()
    # inference_mode() 才是这里关闭梯度记录的地方；推理无需保存反向传播信息。
    with torch.inference_mode():
        output = model.generate(
            **inputs,  # 把词典的各个字段展开，分别交给生成函数。
            max_new_tokens=args.max_new_tokens,  # 最多新生成多少 token，不包含输入长度。
            do_sample=False,  # 每步选最高分的 token，减少随机采样造成的回答差异。
            use_cache=True,  # 缓存已算过的生成状态，避免每写一个 token 都重算整个前文。
        )
    # 输出序列前 input_len 个 token 是原输入，裁掉后只展示模型新续写的部分。
    answer = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
    print("回答：\n" + answer)
    print("CUDA peak memory (GiB): " + str(peak_memory_gib(torch)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
