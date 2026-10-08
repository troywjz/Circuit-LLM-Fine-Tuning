# 本机 Qwen3.5-4B QLoRA 冒烟操作

本文给出本机 4B 模型的文本推理和短环境试训命令。环境试训只检查依赖、显存和训练保存链路，不构成项目训练结果或能力评测。

初次阅读代码与目录用途可先看[代码与目录说明](代码与目录说明.md)。本机依赖版本观测快照见[环境依赖快照-2026-10-09](../experiments/reports/2026-10-08-10步QLoRA/环境依赖快照-2026-10-09.txt)；10 步链路记录见[试训记录](10步QLoRA试训记录-2026-10-08.md)。

硬件与软件环境信息核查日期：2026-10-08。模型页面：[ModelScope Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B)、[Hugging Face Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)；训练接口参考 [Unsloth 文档](https://unsloth.ai/docs/models/qwen3.5/fine-tune)。

## 证据边界

- `SOURCE_FACT`：当前开发环境已检查 Python 3.12.13、PyTorch 2.14.1+cu130、Transformers 5.17.0、TRL 1.13.0、Unsloth/unsloth_zoo 2026.10.2；`uv pip check` 通过。RTX 5060 Ti 显存 8 GB，检查时空闲约 6.83 GiB，支持 BF16。
- `SOURCE_FACT`（脚本创建时的核查快照，2026-10-08）：当时 `D:\models\Qwen3.5-4B` 尚不存在。上游全模型目录约 9.34 GB，分为约 5.33 GB 与 3.99 GB 两个权重分片。后续目录已出现完整上游文件；用户报告本机推理已结束。
- `DERIVED`：8 GB 显存下先走 4-bit QLoRA 冒烟比直接尝试未量化基座的 BF16 LoRA 更合适；实际是否可运行仍受上下文、显存碎片和后台占用影响。
- `SOURCE_FACT`：2026-10-08 已在本机完成五个阶段的 QLoRA 冒烟链路，均以退出码 0 结束，含 10 步训练、adapter 保存和独立进程重载推理；具体范围和限制见[试训记录](10步QLoRA试训记录-2026-10-08.md)。该结果不是正式数据训练或能力评测。
- 试训最多使用 8 条短的对话/需求澄清玩具样本。它们不是正式训练数据，也不是 Benchmark；短样本的最大 token 数会写入报告，不代表 `configured_max_length` 的整段长度已通过稳定性验证。试训成功不证明电路规划能力提升。

## 下载模型（手动）

本说明针对 Windows、Python 3.12 与本机 GPU 环境；依赖快照只记录 2026-10-09 这台机器的已安装版本，不承诺跨平台可重建。新建环境时应显式使用 `uv pip install --torch-backend=cu130 ...` 选择 CUDA 13.0 PyTorch wheel 来源，不能只依赖默认 PyPI 选择 CUDA 轮子。

在项目 PowerShell 中激活已有开发环境：

```powershell
cd "D:\code\Circuit LLM Fine-Tuning"
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

确认磁盘空间充足后，手动下载官方模型到脚本默认目录（约 9.34 GB）：

```powershell
modelscope download Qwen/Qwen3.5-4B --local-dir "D:\models\Qwen3.5-4B"
```

脚本只读本地模型，不会自动联网下载。首次加载可能触发 CUDA/内核编译，需预留额外时间。

## 本地推理

```powershell
python .\scripts\qwen35_infer.py --prompt "你好，请用一句话介绍欧姆定律。"
```

## 10 步环境试训

```powershell
python .\scripts\qwen35_qlora_smoke.py --steps 10
```

默认 adapter 与 JSON 报告写入 `experiments/qlora/checkpoints/qwen35-4b-smoke`。目录非空时脚本拒绝覆盖，改用新的 `--output-dir`。输出报告中的显存是本进程 CUDA `peak_allocated_GiB` 与 `peak_reserved_GiB`，不是整卡占用。

用刚保存的 adapter 重新推理：

```powershell
python .\scripts\qwen35_infer.py --adapter-path .\experiments\qlora\checkpoints\qwen35-4b-smoke\adapter --prompt "mA 是什么单位？"
```

也可尝试 BF16 基座 LoRA（已有同一份本地模型权重即可，不需要再次下载）。该模式未量化基座，预计在 8 GB 显存上大概率 OOM，尚未验证；降低 rank 不能消除基座权重占用。请使用独立输出路径：

```powershell
python .\scripts\qwen35_qlora_smoke.py --method lora --steps 10 --output-dir .\experiments\lora\checkpoints\qwen35-4b-smoke
```

加载该 LoRA adapter 推理时建议保持相同方法，避免量化差异影响对照：

```powershell
python .\scripts\qwen35_infer.py --method lora --adapter-path .\experiments\lora\checkpoints\qwen35-4b-smoke\adapter --prompt "mA 是什么单位？"
```

若遇到 OOM，可缩小上下文和 LoRA rank，并使用全新的输出目录：

```powershell
python .\scripts\qwen35_qlora_smoke.py --seq-length 256 --rank 4 --steps 10 --output-dir .\experiments\qlora\checkpoints\qwen35-4b-smoke-small
```

默认优先试验 4B QLoRA；能否运行及是否有效都需实测。4B BF16 LoRA 可作为单独的探索性尝试，不作为本机 8 GB 显存默认路线。这里的短试训不能代替正式数据集、冻结 Benchmark、通用对话退化检查或项目 M6/M7 验收。请只用本脚本自己生成的 adapter；不要下载现成 GGUF 并将其称为本机微调结果。
