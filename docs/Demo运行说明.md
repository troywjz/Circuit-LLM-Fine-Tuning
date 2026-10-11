# Demo 运行说明

所有命令从项目根目录运行。候选数据、模型、适配器、API 凭据和实验输出放在被 Git 忽略的 `本地资料/` 或本机模型目录；命令中的路径均为相对路径示例。

## 当前状态

**SOURCE_FACT**：教师 API 已恢复。首轮中文数据已完成审校；少量原始返回有正文截断或 JSON 引号问题，分别剔除或经人工核对后离线修复，原文保持不变。早期小样例也已验证训练、保存和重载链路。

完整来源池仍有 92 条领域训练来源候选、20 条通用情境和 19 条验证来源候选；来源数不等于已生成的中文对话数。首轮已实际冻结 30 条中文训练对话：14 条 Stack Exchange 领域来源改写、16 条自编通用与多轮情境，其中包含技术材料说明，并非纯日常闲聊。原先约 14+20 的设想调整为 14+16 以尽快获得反馈，未补造其余 4 条；完整来源池留待后续扩充。

首轮数据已审校与编码检查：每条 69–378 tokens，共 7,288 tokens、assistant labels 4,341，无截断。训练文件 SHA-256 为 `e6cfddb55a92342e7f48bbd18f7956a30576da77f9c8074fb94837430bbb68d3`。dev16（12 领域、4 通用）和 final50（40 领域、10 通用）保持冻结，含同任务族参数变体，不能当作独立任务或官方 Benchmark。旧 Base 分数因停句问题暂作诊断，新比较必须统一停止协议。首轮未传中文 validation，以 dev 选检查点。

**SOURCE_FACT**：两轮 60 步训练与修复停句后的开发对照已完成。第二轮 30 步领域提高 5.00 分、通用降低 11.25 分；三个版本均未通过既定门槛，当前只做最终诊断、不推荐部署。见[修订对照](修订开发集对照.md)。旧曲线与数字只作诊断，见[修复说明](停句配置修复说明.md)。第二轮混合数据与训练见[修订记录](修订数据与训练记录.md)。

**SOURCE_FACT**：冻结的 final50 三组诊断已完成并经主 Agent 逐轮复核。FT-revised step 30 相对 Base 的领域均分提高 3.56 分，通用均分降低 5.75 分，领域硬错题增加 2 题；它未通过开发门槛，仅作诊断，不推荐部署。结果与限制见[最终诊断对照](最终诊断对照.md)。若继续评估，必须使用新的独立终测集，不能用当前 final50 回训或调参。

## 首轮准备与训练

本轮首批实际使用 14 条领域来源改写和 16 条自编通用、多轮情境，已冻结并完成训练。领域材料由教师改写并由主 Agent 作实质工程修订，通用材料由项目自编；都不是纯教师原样答案。无需凑满来源包或用完预算。

后续批次仍须逐条审校，只有主 Agent 批准的对话才能进入 `approved_first_run_candidates.jsonl`。`export` 不会自动筛掉未审行；输入中只要含未批准行就会停止，因此先把通过审核的行另存为批准文件，再导出。已冻结首轮数据的入口为：

```powershell
python scripts/build_demo_data.py export --input 本地资料/Demo-V1/数据/approved_first_run_candidates.jsonl --output 本地资料/Demo-V1/数据/train_first_run.jsonl --approved-by main
python scripts/demo_sft.py --train 本地资料/Demo-V1/数据/train_first_run.jsonl --output-dir 本地资料/Demo-V1/实验/formal-first-001 --seq-length 1024 --rank 8 --steps 60 --save-steps 15,30,60 --seed 3407 --dry-run
```

dry-run 会加载本地 tokenizer 检查数据，不加载 GPU 模型。首轮验证来源候选尚未完成中文审校，因此训练时不传 `--validation`，使用冻结 dev16 比较 Base 与各 checkpoint，并据此选择 checkpoint。不要把候选验证源写成已验证数据。本轮训练已经完成，不要重复启动。

本轮正式训练参数如下。当前训练已完成，以下仅记录配置，不要再次执行：

```powershell
python scripts/demo_sft.py --train 本地资料/Demo-V1/数据/train_first_run.jsonl --output-dir 本地资料/Demo-V1/实验/formal-first-001 --model-path models/Qwen3.5-4B --seq-length 1024 --rank 8 --alpha 16 --learning-rate 1e-4 --gradient-accumulation 4 --steps 60 --save-steps 15,30,60 --seed 3407
```

正式训练从新的 Base 开始，使用 60 步并保存 15、30、60 步检查点；训练已完成。Base 与 checkpoint 使用相同冻结题集和配置：关闭 thinking、greedy 解码、`seq-length=4096`、`max-new-tokens=1024`。保留截断标记；完整题集跑完后才汇总比较。结论需结合逐题评分、硬错误和通用题回归。

## 推理与盲评

```powershell
python scripts/demo_infer.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/base.jsonl --model-path models/Qwen3.5-4B --seq-length 4096 --max-new-tokens 1024 --run-label Base
python scripts/demo_infer.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/ft-step-15.jsonl --model-path models/Qwen3.5-4B --adapter-path 本地资料/Demo-V1/实验/formal-first-001/adapter-step-15 --seq-length 4096 --max-new-tokens 1024 --run-label FT --step 15
python scripts/demo_eval_tools.py prepare-blind --answers 本地资料/Demo-V1/评测/answers.jsonl --blind 本地资料/Demo-V1/评测/blind.jsonl --mapping 本地资料/Demo-V1/评测/private_mapping.jsonl
```

Claude 参照通过当前 API 服务另发独立的 messages 请求，只接收学生材料和该请求自己的历史，不提供 teacher 答案、评分要点或工具。服务使用默认解码，与本地推理配置不同；gateway 返回的模型身份信息不是独立认证。开发 16 题和最终 50 题的 API 参照已完整收集，无需重跑。以下仅为使用程序的示例，先 dry-run 检查题集：

```powershell
python scripts/demo_api_reference.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/claude-opus-5-5.jsonl --env-file .env --model-id claude-opus-5-5 --run-label claude-opus-5-5 --dry-run
```

确认题集与预算后，去掉 `--dry-run` 才会发起真实 API 请求；需保留返回记录并审阅完整性。多个单条完整回复不代表整套参照已完成。

人工盲评后再运行 `demo_eval_tools.py aggregate` 与 `plot` 汇总。评分、图表和训练 loss 都不能单独证明能力提升；主 Agent 审阅完整结果后再更新结论。
