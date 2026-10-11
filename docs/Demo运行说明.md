# Demo 运行说明

所有命令从项目根目录运行。数据、模型、适配器、API 凭据和实验输出放在被 Git 忽略的 `本地资料/` 或本机模型目录；下文路径只是相对路径示例。脚本不会下载模型或安装依赖。

## 当前证据与待办

**SOURCE_FACT**：新版 assistant-only QLoRA 小试训使用 5 条人工审核中文样例，rank 8、seq-length 1024、seed 3407，完成 3 步；train loss 为 3.463，峰值 allocated 为 3.311 GiB。adapter 已保存，并在新进程中重载生成一条 128 token 回答；回答到达生成上限，已标记为截断。旧版另有 8 条玩具对话、10 步的链路记录。

**DERIVED**：这些结果只说明小样本训练、adapter 保存/重载和有限生成链路可运行，不说明回答质量、领域提升或正式训练效果。

本轮评测集已审核冻结为 dev16（12 条领域题、4 条通用题）与 final50（40 条领域题、10 条通用题），题目为自编合成诊断场景，不是官方 Benchmark。领域题包含同任务族参数变体，结果应报告为40个场景，不称为40类独立任务；评测case不关联原始来源。后续更换题目时，仍需在看模型回答前审核冻结。目前准备了 96 条领域来源候选、20 条通用情境及 20 条验证来源候选，尚未全部生成并审核为训练对话。计划训练 60 步并保存 15/30/60 步，评测结果允许下降。

比较本地 Base 与 FT 时，应使用同一量化基座和相同输入，关闭 thinking，采用 greedy 解码、`seq-length=4096`、`max-new-tokens=1024`。Claude Opus 5.5 参照是同一 API 服务环境中的独立 messages 请求，不接收评分要点、参考答案、旧教师回答或工具；它使用服务默认解码，因此单独标注，不声称与本地解码设置完全一致。目前接口返回限流类 HTTP 429，批量生成和完整参照运行尚未完成。

## 数据准备和审核

筛选来源、划分验证样本并生成自编通用情境。正式目标以命令明确指定：

```powershell
python scripts/build_demo_data.py prepare --domain-count 96 --validation-count 20 --general-count 20 --outdir 本地资料/Demo-V1/数据
```

主 Agent 审核候选后，才为获准的来源 ID 写生成许可。示例 ID 需替换为真实、已审核的 ID：

```powershell
python scripts/build_demo_data.py approve --approval-file 本地资料/Demo-V1/数据/generation_approval.json --source-id 已审核来源ID --reviewer 主Agent
python scripts/build_demo_data.py generate --input 本地资料/Demo-V1/数据/train_sourcepack.jsonl --approval-file 本地资料/Demo-V1/数据/generation_approval.json --outdir 本地资料/Demo-V1/数据 --limit 20 --dry-run
```

`generate` 的真实请求会连接已配置的教师 API；确认许可、凭据和运行范围后，去掉 `--dry-run` 才会发出请求。生成结果仍是待审候选。逐条核对来源、事实、授权与训练消息，并将通过项标为已批准后导出；训练和验证分别导出：

```powershell
python scripts/build_demo_data.py export --input 本地资料/Demo-V1/数据/train_candidates.jsonl --output 本地资料/Demo-V1/数据/train.jsonl --approved-by 主Agent
python scripts/build_demo_data.py export --input 本地资料/Demo-V1/数据/validation_candidates.jsonl --output 本地资料/Demo-V1/数据/validation.jsonl --approved-by 主Agent
```

导出脚本会拒绝未批准或消息格式不合格的候选。若审核后不足目标数量，应记录实际数量，不要把计划数写成已收集数。

## 训练和推理

以下正式试训命令对应当前计划；只有在样本审核、切分隔离和评测冻结后执行。`--train` 与 `--validation` 必须指向审核导出的文件。

```powershell
python scripts/demo_sft.py --train 本地资料/Demo-V1/数据/train.jsonl --validation 本地资料/Demo-V1/数据/validation.jsonl --output-dir 本地资料/Demo-V1/训练/run-001 --model-path models/Qwen3.5-4B --seq-length 1024 --rank 8 --steps 60 --save-steps 15,30,60 --seed 3407
```

同一冻结题集分别运行 Base 和已保存检查点。脚本默认 `seq-length=4096`、`max-new-tokens=1024`、关闭 thinking、greedy 解码；完整题目跑完后才可用于比较，截断回答保留截断标记。

```powershell
python scripts/demo_infer.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/base.jsonl --model-path models/Qwen3.5-4B --seq-length 4096 --max-new-tokens 1024 --run-label Base
python scripts/demo_infer.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/ft-step-15.jsonl --model-path models/Qwen3.5-4B --adapter-path 本地资料/Demo-V1/训练/run-001/adapter-step-15 --seq-length 4096 --max-new-tokens 1024 --run-label FT --step 15
```

**UNCONFIRMED**：API 参照完整运行尚未成功。先用 dry-run 检查冻结题集，不读取凭据或联网：

```powershell
python scripts/demo_api_reference.py --cases 本地资料/Demo-V1/评测/dev_inputs.jsonl --output 本地资料/Demo-V1/评测/claude-opus-5-5.jsonl --env-file .env --model-id claude-opus-5-5 --run-label claude-opus-5-5 --dry-run
```

准备发起真实参照请求时，移除 `--dry-run`；脚本有余额保留检查、默认不重试，并在失败时保存已有检查点。API 输出不含题目中的 rubric 或参考答案，HTTP 429 后先查明原因再重试。

## 盲评、汇总和绘图

将各模型回答合并为每题一行的 JSONL，至少含 `case_id`、`model`、`step`、`answer`。先生成供人工盲评的回答和私有映射；映射只留在本机：

```powershell
python scripts/demo_eval_tools.py prepare-blind --answers 本地资料/Demo-V1/评测/answers.jsonl --blind 本地资料/Demo-V1/评测/blind.jsonl --mapping 本地资料/Demo-V1/评测/private_mapping.jsonl
```

人工盲评后，将评分写为每行含 `run_id`、`case_id`、`scores`、`hard_errors` 的 JSONL，再汇总并绘图：

```powershell
python scripts/demo_eval_tools.py aggregate --cases 本地资料/Demo-V1/评测/dev_grader.jsonl --answers 本地资料/Demo-V1/评测/blind.jsonl --ratings 本地资料/Demo-V1/评测/ratings.jsonl --mapping 本地资料/Demo-V1/评测/private_mapping.jsonl --out 本地资料/Demo-V1/评测/aggregate.jsonl
python scripts/demo_eval_tools.py plot --aggregate 本地资料/Demo-V1/评测/aggregate.jsonl --loss 本地资料/Demo-V1/评测/loss.csv --out-prefix 本地资料/Demo-V1/评测/dev_curve --note "冻结题集版本与运行配置"
```

绘图所需 `loss.csv` 有 `step,loss` 两列；训练脚本实际写出的是 `训练指标.jsonl`，需据其中的 `global_step` 与 `loss` 另行整理成 CSV。评分每个适用维度取 0–2 分。单题需达到 80 分、每维至少 1 分且没有硬错误才算通过；漏答、漏评分或题集不完整会报错，不会自动补分。图表和训练 loss 都不是能力结论；结论须结合逐题评分、硬错误和通用题回归。所有题目、输出和评分仍需主 Agent 审阅。
