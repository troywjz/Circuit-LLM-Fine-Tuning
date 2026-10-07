# 环境搭建：Windows + RTX 5060 Ti 8GB

目标：在本机跑通 Qwen3.5-4B 的推理与 QLoRA 微调。  
预计耗时 1–2 小时，其中大头是下载（模型权重约 9.3GB）。

> 本页命令均在 **命令提示符（cmd）** 或 **PowerShell** 中执行，专业名词随文解释。

## 0. 检查显卡驱动

按 `Win + R`，输入 `cmd`，回车，执行：

```bat
nvidia-smi
```

看右上角 **CUDA Version** 是否 ≥ 12.8（这是驱动支持的上限，不是已安装的 CUDA）。

- ≥ 12.8 → 继续下一步。
- < 12.8 → 去 <https://www.nvidia.cn/geforce/drivers/> 下载最新驱动安装。

## 1. 确认 Python 版本（本机已够用，无需新装）

本机现状（2026-10-07 实测）：

| 位置 | 版本 | 说明 |
| --- | --- | --- |
| 系统默认 `python` | 3.13.14 | 日常使用，不动它 |
| uv 托管的 CPython | 3.12.13 | 训练环境用它 |

3.13 本身可用——PyTorch cu128、bitsandbytes、triton 都提供对应的 3.13 轮子（实测已确认）。但**训练环境仍建议用 3.12**：各家微调框架测试最充分，遇到库的兼容问题时少一层变量；而 3.12 本机已经有了，用 uv 建环境零成本，不必重新下载。

验证命令：`python --version`（应显示 3.13.14）、`uv python list`（应能看到 3.12.13）。

## 2. 安装 Visual Studio 生成工具 2022

本机现状（2026-10-07 实测）：**未安装**，也没有 `cl.exe`。

**VSCode 替代不了这一步。** VSCode 只是编辑器，本身不含编译器；插件市场里的 C/C++ 插件，官方说明第一段就写了"不包含 C++ 编译器和调试器，需要另行安装"。两个名字很像，但不是同一件东西：

| 名称 | 是什么 | 与本项目的关系 |
| --- | --- | --- |
| Visual Studio Code（VSCode） | 轻量代码编辑器 | 你已有，用来读代码、看脚本 |
| Visual Studio 生成工具（Build Tools） | MSVC 编译器 + Windows SDK | 本步要装的 |

装法（本机 winget 1.29 可用，两种任选）：

- **命令行（推荐）**：以**管理员身份**打开 PowerShell，执行

  ```bat
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e
  ```

  弹出安装界面后勾选 **使用 C++ 的桌面开发**，并确认右侧包含 `MSVC v143 ... 生成工具` 与 `Windows 10/11 SDK`。
- **官网下载**：<https://visualstudio.microsoft.com/visual-cpp-build-tools/>，同样勾选 **使用 C++ 的桌面开发**（约 3–6GB）。

验证：**新开**一个窗口执行 `cl`，看到 `Microsoft (R) C/C++ ...` 的版本信息即成功。

> 为什么需要：加速库 Triton 第一次运行要现场编译显卡内核，机器上没有 C 语言编译器就会报 `Failed to find C compiler`。
>
> 补充：新版 triton-windows 的轮子已内置一个小型 C 编译器（TinyCC），所以**有可能**不装也能跑；但这条路径未经本项目实测，是网上报错的常见来源。如果这 3–6GB 下载是个负担，可以先跳过，等真的报错再回来装。
>
> 另外，在 VSCode 里装 C/C++ 插件是可选的便利项（代码高亮、跳转），它不提供编译器，装了也不能替代本步。

## 3. VC++ 运行库（本机已达标，可跳过）

**是什么**：程序运行时要用的一批共享 DLL（`vcruntime140.dll`、`msvcp140.dll` 等），提供内存分配、异常处理、C++ 标准库这些底层功能。编译器不会把这些代码打进每一个 exe，而是运行时从系统加载——所以缺了它，程序一启动就报错，而不是编译失败。

本机现状（2026-10-07 实测）：`C:\Windows\System32` 下已存在，版本 **14.51.36247.0**，高于 triton-windows 要求的 14.42，**本步可直接跳过**。

以后若出现 DLL 相关报错，再装最新版：<https://aka.ms/vs/17/release/vc_redist.x64.exe>

## 4. 创建虚拟环境

在项目目录下：

```bat
cd /d D:\code\Circuit LLM Fine-Tuning
uv venv .venv --python 3.12
.venv\Scripts\activate
```

uv 会直接使用已装好的 3.12.13，不会重新下载 Python。验证：`python --version` 应显示 `Python 3.12.13`。

> 虚拟环境（venv）是 Python 的一份独立副本，装在这里的库不会影响系统。看到命令行前面出现 `(.venv)` 就说明已激活；换新窗口要重新执行 `.venv\Scripts\activate`。

## 5. 安装 PyTorch（必须 cu128 版）

```bat
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 5060 Ti 是 Blackwell 架构（代号 sm_120），低于 CUDA 12.8 的版本跑不起来。装错版本的表现是报 `no kernel image is available`。

## 6. 安装微调工具链

```bat
uv pip install unsloth unsloth_zoo bitsandbytes transformers trl peft accelerate datasets modelscope
```

- `unsloth`：省显存的微调框架（官方称省 70% 显存），8GB 卡能用上它很关键。
- `bitsandbytes`：把模型以 4 位精度加载的库，QLoRA 靠它。
- `transformers` / `trl` / `peft`：HuggingFace 的标准微调三件套，做兜底方案用。
- `modelscope`：国内下载模型的命令行工具。

## 7. 验证环境

```bat
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

期望输出类似：`2.x.x+cu128 True NVIDIA GeForce RTX 5060 Ti (12, 0)`

- `True` = 显卡能被调用。
- `(12, 0)` = 计算能力 12.0，即 sm_120，说明 Blackwell 识别正常。

## 8. 下载模型

**微调用的原始权重**（BF16，约 9.3GB）：

```bat
modelscope download --model Qwen/Qwen3.5-4B --local_dir D:\models\Qwen3.5-4B
```

- 国内首选 ModelScope（阿里自家，速度稳定，无需代理）：<https://modelscope.cn/models/Qwen/Qwen3.5-4B>
- 国外源 HuggingFace：<https://huggingface.co/Qwen/Qwen3.5-4B> （走代理）

> 部署用的 GGUF 格式（体积小、给推理软件用）后面单独下，不要和这份混在一起。

## 9. 跑通一次推理

新建 `test_infer.py`，内容：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = r"D:\models\Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, dtype="auto", device_map="cuda")
msgs = [{"role": "user", "content": "用一个电路设计决策的例子解释什么是去耦电容。"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tok.decode(m.generate(ids, max_new_tokens=256)[0][ids.shape[1]:], skip_special_tokens=True))
```

执行 `python test_infer.py`，能吐出通顺回答就算环境通了。

## 10. 跑通一次微调

- 官方入门笔记本：<https://github.com/unslothai/unsloth> （Notebooks 一章里找 Qwen 的 QLoRA 例子）
- 8GB 显存建议参数：

| 参数                            | 建议值       | 说明                              |
| ----------------------------- | --------- | ------------------------------- |
| `max_seq_length`              | 1024–2048 | 单条样本的最大长度，越高越吃显存                |
| `load_in_4bit`                | True      | 4 位加载，省显存的关键开关                  |
| `per_device_train_batch_size` | 1         | 一批喂几条样本                         |
| `gradient_accumulation_steps` | 8         | 攒 8 批再更新一次参数，等效大批量              |
| `lora r`                      | 16        | 适配器的"容量"，越大越能学但越容易过拟合           |
| `num_train_epochs`            | 1–2       | epoch 即"把全部训练数据学完一遍"，学太多遍会变成背答案 |

**先拿 5–20 条样本试跑 10 步**，确认能跑通、显存不爆，再上正式数据。

## 11. 用起来（部署）

微调产出的是适配器文件，需要先合并回基座模型，再导出 GGUF 给推理软件：

1. 下载推理软件：LM Studio（图形界面，最省事）<https://lmstudio.ai> 或 Ollama <https://ollama.com>
2. 在 LM Studio 里直接搜索 `Qwen3.5-4B-GGUF`，选 `Q4_K_M` 量化版（约 2.8GB），即可对话。

## 12. 常见报错对照

| 现象                             | 原因                       | 解决                                                                          |
| ------------------------------ | ------------------------ | --------------------------------------------------------------------------- |
| `no kernel image is available` | PyTorch 装成了 CPU 版或 cu126 | 重装第 5 步的 cu128 版                                                            |
| `Failed to find C compiler`    | 缺 C 编译器                  | 回到第 2 步装生成工具，并用 x64 原生工具命令提示符运行                                             |
| `CUDA out of memory`           | 显存不足                     | 降 `max_seq_length`，确保 `load_in_4bit=True`                                   |
| Unsloth 提示不支持 `qwen3_5` 架构     | 框架还没跟上新架构                | 改用 `transformers + peft + trl` 标准 QLoRA，或换 Qwen3-4B-Instruct-2507（老架构，生态成熟） |
| 解包 DLL 失败                      | 缺运行库                     | 回到第 3 步                                                                     |
