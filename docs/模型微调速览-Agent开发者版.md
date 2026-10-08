# 模型微调速览：Agent 开发者版

给已有 LangChain/LangGraph 使用经验的开发者，约 10–15 分钟读完。重点是把熟悉的 Agent 概念接到模型训练，不展开入门内容。

## 1. Agent 哪些部分会变

**SOURCE_FACT｜**LangGraph 管状态、路由和工具调用；Prompt 与 RAG 提供一次运行时的指令和资料。它们影响模型“这次看到什么、流程怎么走”。本项目的参数高效微调只更新额外的一小部分参数，让某些回答习惯更容易出现。基座加 adapter 可以封装成推理服务再接入 LangGraph；具体接口、工具调用和对话行为仍需对接测试。

PEFT 是 Parameter-Efficient Fine-Tuning（参数高效微调）这一类方法，也有 Hugging Face 的 PEFT 库。本项目的 adapter 使用 LoRA：附加在指定模型层上的小矩阵增量。它不是 LangChain 的工具适配器，不是 prompt，也不是能脱离基座独立运行的新模型。

## 2. LoRA adapter 到底存了什么

**DERIVED｜**把某个线性层简写为：

`y = Wx + sBAx`，其中 `s = alpha / r`

`W` 是原有权重，训练时冻结；`A`、`B` 是额外的小矩阵，参与训练。QLoRA 将基座 `W` 以量化形式冻结保存，计算时使用较高精度路径，但仍然训练 LoRA 的 `A/B`。

举一个纯数学例子：假设某一层是 4096×4096，普通矩阵有 16,777,216 个参数；若 LoRA rank `r=8`，两个低秩矩阵合计 `4096×8 + 8×4096 = 65,536`，约为原层的 0.39%。这是**假设尺寸**，不是 Qwen 4B 实际某层尺寸，也不能据此推算所有模型的 adapter 大小或保证训练效果。

增加 rank 通常会增加可训练参数与 adapter 文件体积，也会改变显存和训练成本；它本身不保证答案更好。改 `target_modules` 则会改变哪些层能被适配，实验比较时应一次只改少量变量，并记录配置，避免把差异错归因给模型能力。

`r` 控制增量容量；`alpha` 控制缩放；`target_modules` 决定哪些层加入 adapter。当前冒烟脚本默认 `r=8`、`alpha=16`，目标为 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。保存的 `adapter_model.safetensors` 是增量权重，`adapter_config.json` 记录方法、基座等配置。加载时需要相容的基座和版本、分词器与 chat template，不能拿去套任意其他模型。合并是把增量加到基座对应权重上；另行保存后，才会导出一份包含增量的新模型。这和继续训练或量化转换是不同操作。当前脚本只保存 adapter，不合并，也不覆盖原始权重。

因此，adapter 文件很小不代表部署不需要基座；推理时仍要先载入相容基座，再叠加 adapter。若换了基座 revision、目标模块名或 tokenizer/template，即使文件能读，也可能无法正确加载或生成格式异常。部署记录最好把基座身份、adapter、分词器/template 和量化方式作为一组版本化资产来检查。

## 3. 权重里的数字怎么理解

| 格式 | 每个值的典型存储 | 位字段与直觉 |
|---|---:|---|
| FP32 | 4 bytes | 1 位符号、8 位指数、23 位尾数；常用参考精度 |
| FP16 | 2 bytes | 1/5/10；尾数比 BF16 细，但可表示范围窄，最大有限值 65504 |
| BF16 | 2 bytes | 1/8/7；范围接近 FP32，尾数较粗 |
| NF4 | 约 0.5 byte/量化值 | 16 个非均匀等级；不是 IEEE 4-bit 浮点，也不是“四位小数” |

**SOURCE_FACT｜**CPU 示例中，1.01 转成 FP32、FP16、BF16 后约为 1.00999999、1.009765625、1.0078125，精度不同会有舍入。NF4 通常还需块级 scale 等元数据；敏感部分也可能保留较高精度，所以不能把整个模型文件简单除以四。量化是近似表示，会引入误差。按十进制理想估算，4B 个权重约需 FP32 16GB、BF16 8GB、完全 4-bit 2GB；这只是权重的量级，不是总显存，实际模型也不保证每个权重都能按 4-bit 存储。

当前配置里的三个精度名词各管一件事：`bf16=True` 指训练计算精度请求；`load_in_4bit` 控制 QLoRA 基座存储；`adamw_8bit` 指优化器状态的表示。框架仍可能为数值敏感计算保留 FP32。不能把这三者当作同一个“模型精度开关”。

## 4. 从对话样本到一次参数更新

先区分两种分类：SFT 描述“用示例文本怎样监督学习”；全参数微调、LoRA 描述“哪些参数参与更新”。SFT 可以采用全参数、LoRA 或 QLoRA，SFT 与 LoRA 并不是互斥的训练路线。

SFT（监督微调）把示例答案转成 token，让模型预测目标 token 并计算 loss；随后反向传播求梯度，优化器据此更新可训练的 adapter。推理时模型则按前文逐 token 自回归生成。**Teacher forcing** 是训练时提供真实前文 token 来预测下一个 token 的方式，和推理时逐步读入模型自己生成的前文不同。

冻结基座不等于训练时只计算 A/B：前向仍要经过基座；为把梯度传回较早层的 adapter，也需要相应反向计算。当前脚本将 8 条玩具对话套 chat template 转成文本，没有专门限制为只对 assistant 答案计算 loss。正式数据需弄清 template、labels、`-100` mask、EOS 和截断；不能假设加一个 `assistant_only_loss=True` 就对任何模板都正确。

脚本的 batch=1、梯度累积=4，在单 GPU 下通常相当于每次更新累计 4 个 microbatch；10 次 optimizer step 约处理 40 个 microbatch。若只有 8 条样本，就大约重复看 5 遍，并不等于新增了 40 条数据。`max_length=512` 是上限，短样本不会因此证明 512 token 上下文稳定。2026-10-08 已完成 8 条玩具对话的 10 步链路验证，具体证据见[试训记录](10步QLoRA试训记录-2026-10-08.md)；正式领域数据训练与评测仍未完成。

loss 只说明模型对训练目标 token 的预测误差，不会自动告诉你回答是否电气正确、是否会追问缺失需求，或是否损伤普通聊天能力。短跑里 loss 下降可以作为链路和数值状态的观察项；是否有用仍要在冻结、未参与训练的项目样本上，与基座及 Prompt/SKILL 基线比较。若样本来自同一项目，拆成不同对话后分别放进训练集与测试集也会造成泄漏，所以切分单位应优先是项目。

## 5. 建议补齐的六项知识

| 顺序 | 学什么、为什么 |
|---|---|
| 1 | LoRA 与数值表示：能读懂当前 `load_unsloth_model` 和 `get_peft_model` 配置，分清基座与 adapter。 |
| 2 | SFT 数据和答案 mask：弄清 chat template 怎样变成 token，哪些 token 参与 loss。 |
| 3 | 训练旋钮：理解学习率、rank、steps、epoch、梯度累积分别影响什么。 |
| 4 | 显存账本：区分权重、激活、梯度、优化器状态和推理 KV cache；知道 OOM 出在加载、前向、反向还是生成。权重是模型参数本身，训练激活随层数、序列长度和 batch 增长，生成缓存随上下文增长；降低 LoRA rank 只减少增量部分，不能解决所有显存问题。梯度检查点用额外重算换激活显存。 |
| 5 | 评测设计：冻结 Benchmark、按项目切分；比较量化前后基线，并检查通用对话、多轮追问和不确定时拒绝猜测。 |
| 6 | 保存与恢复：记录基座 revision、template、量化配置并匹配加载。完整恢复训练通常还需 optimizer、RNG 等状态；只有 adapter 不等于能从原训练状态接续。 |

## 6. 三个最小练习（只作学习计划）

1. 用 CPU 对比 FP16/BF16 数值差异，观察舍入与范围，不推断真实模型权重分布。
2. 在相同量化配置和 prompt 下，对比基座与加载 adapter 后的回答。玩具训练题答得更像样，不代表泛化提升。
3. 读一份 smoke report，辨认 `global_step`、loss 是否有限、`sample_max_tokens` 和显存字段；报告没有 Benchmark 对照，就不能推出能力提升。

分布式训练、RLHF/DPO、从零预训练和深入推导 Attention 可以先放一边。当前目标仍是验证“自然语言硬件需求 → Circuit Design Intent”的电路规划效果；小试验与训练日志可以形成环境验证记录，但不能据此宣布电路能力提升或 V0 验收通过。

## 延伸阅读

- [PEFT LoRA 概念指南](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [PEFT checkpoint 格式](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint)
- [Transformers bitsandbytes 量化](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [NVIDIA TensorRT 精度注意事项](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
