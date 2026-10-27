# 本机 LoRA SFT 与复现

训练器使用 PyTorch / Transformers / PEFT，在冻结基础权重上训练 LoRA；目标是最后一个 assistant completion 与 EOS 的标准 next-token loss。没有新的损失函数、强化学习或付费教师 API。

## 模型和环境

需要 Python 3.10+。原始运行环境为 M4 Pro / 24 GB，当前 `configs/main.json` 指定 MPS、bfloat16 和 gradient checkpointing。MPS 不可用时明确报错，不静默改到 CPU。其他设备可以另存配置，但结果应另列，不能冒充同次运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
```

安装区间见 `pyproject.toml`，实际运行的包版本写入 `run_manifest.json`。模型使用 `Qwen/Qwen2.5-0.5B-Instruct`，公开 revision 为 `7ae557604adf67be50417f59c2c2f167def9a775`。首次下载需要网络和本地磁盘空间；后续训练只加载本地文件，不调用模型 API。

```bash
python scripts/download_model.py --output models/qwen2.5-0.5b-instruct
```

主权重 `model.safetensors` 的预期 SHA-256 为 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。脚本默认从 Hugging Face 下载；网络较慢时可加 `--weight-mirror modelscope`，元数据仍来自固定 HF revision，镜像权重必须通过相同哈希检查。“同名模型”不足以保证同一文件。训练器对本地权重、tokenizer 与配置文件另做完整快照哈希，并绑定训练数据和配置。

## 数据与训练

仓库的 `data/main/` 是固定实验数据。审计跨 split 边界：

```bash
evidence-gate audit data/main/train.jsonl data/main/dev.jsonl data/main/test.jsonl
```

需要重新生成时使用新目录，例如 `evidence-gate generate --output data/reproduced --seed 20261027`，比较 manifest 后再训练。生成器拒绝覆盖已有 manifest，评测也拒绝覆盖现有输出。

以下命令使用新的复现目录；已存在的运行须显式恢复或改目录。

```bash
evidence-gate train --data data/main/train_answer_only.jsonl --model models/qwen2.5-0.5b-instruct --config configs/main.json --output runs/reproduce/answer-only
evidence-gate train --data data/main/train.jsonl --model models/qwen2.5-0.5b-instruct --config configs/main.json --output runs/reproduce/evidence
```

两者都用 rank 8 / alpha 16、`q_proj` 与 `v_proj`，预设 200 个 optimizer steps、microbatch 1 × accumulation 4、学习率 3e-4、最大序列长度 512、bfloat16 和 gradient checkpointing。answer-only 有 320 行，evidence 有 1,120 行；各处理 800 个样本位置，样本重复率和监督 token 总数不同。这是训练分布的对照，**不是等 token 消融，也不能单独证明配对结构带来增益**。

训练器按实际 chat template 形成输入，只监督 completion 与 EOS，屏蔽 prompt 和 padding。超过上下文上限会报错，不静默截断引用或证据。token 审计的最终值在 `run_manifest.json.data_audit`；不能从配置的 512 反推实际输入长度。

`configs/smoke.json` 仅运行 3 步，用于检查训练和保存链路。smoke loss 下降不构成任务质量结果。正式配置固定步数，没有自动 early stopping；不得用 test 挑 checkpoint。

## 为什么修改本机配置

最初的 float32 / microbatch 2 / accumulation 2 配置在第 67 步发生 MPS 内存不足，日志只记录了完成的 66 步；它不是完成的 200 步训练。失败运行保存在本地 `runs/failed-float32/`，对外只保留 `experiments/failed-float32/` 的失败说明、metrics 和 manifest，不发布未完成的权重。

随后将训练改为 bfloat16、microbatch 1 / accumulation 4、长度上限 512 和 gradient checkpointing，保持有效 batch 4 与主实验 200 步预算；3 步 smoke 已完成。这是资源适配，不是根据测试成绩选择配置。训练精度降低和激活重计算也会改变数值或耗时，不能把前后两次拼成一次连续训练；配置指纹不同，必须使用新运行目录。

## 恢复与产物

每 50 步保存 checkpoint。恢复只接受同一输出目录下、数据 / 配置 / 模型哈希一致且文件校验通过的 checkpoint：

```bash
evidence-gate train --data data/main/train.jsonl --model models/qwen2.5-0.5b-instruct --config configs/main.json --output runs/reproduce/evidence --resume runs/reproduce/evidence/checkpoints/step-000050
```

恢复包含 optimizer、scheduler、随机数状态和采样游标；状态使用 JSON 与 safetensors，不读取 pickle。恢复命令仅用于确实存在的中断 checkpoint，正常完成后无需再执行。

| 文件 | 作用 |
| --- | --- |
| `adapter/` | 保存的 adapter 配置、权重和 tokenizer；基础模型权重未合并进去 |
| `run_manifest.json` | 模型文件哈希、配置、数据哈希、依赖版本和 token 审计 |
| `metrics.json` | 每步 loss、grad norm、学习率、输入 / 监督 token 和累计耗时 |
| `summary.json` | 完成步数、可训练参数量、adapter 哈希与训练摘要 |
| `artifact_manifest.json` | 非 checkpoint 训练产物的文件校验清单 |

训练日志说明参数更新确实执行；质量必须靠下面的独立评测。单个种子 42 不证明多种子稳健性。

## 同输入的三方评测

正式对照统一使用相同 test、MPS、bfloat16、batch 4、greedy free generation 和 72 个新 token 上限。下面显式传入精度与 batch，避免依赖 CLI 的默认值；较小 batch 用于控制本机 MPS 缓存占用。模型输出不做 JSON 修补，也不用规则替换决策。

```bash
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name base --model models/qwen2.5-0.5b-instruct --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/base.json
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name answer-only-sft --model models/qwen2.5-0.5b-instruct --adapter runs/reproduce/answer-only/adapter --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/answer-only.json
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name evidence-sft --model models/qwen2.5-0.5b-instruct --adapter runs/reproduce/evidence/adapter --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/evidence.json
evidence-gate report experiments/reproduce/base.json experiments/reproduce/answer-only.json experiments/reproduce/evidence.json experiments/contract.json experiments/first-match.json --training runs/reproduce/answer-only/metrics.json runs/reproduce/evidence/metrics.json --output experiments/reproduce/report.json
evidence-gate serve --report experiments/reproduce/report.json
```

预测文件的 `model_provenance` 保存实际基础模型文件指纹和 adapter SHA，`inference` 保存精度、设备、batch 与解码参数。上述 `--training metrics.json` 会同时加载旁边的 `summary.json`，将曲线与已评测 adapter 的哈希匹配。报告拒绝合并不同数据哈希或不同模型推理设置的评测。如果重新生成了不同数据，需在该数据上重跑所有规则与模型基线。`mean_latency_ms` 是 batch wall time / batch size，不是单请求延迟或 p95；当前没有据此声明线上 QPS。

两个额外诊断数据已固定在 `data/diagnostics/`：paraphrase 与 redacted 各 56 行，均来自前 4 个 test group。可将 `evaluate --data` 分别指向这两个文件并使用新的输出路径，按各自数据哈希单独生成报告。它们不能合并进主测试当作新增独立样本，也不能用于挑选 checkpoint。具体变换见 [数据卡](data_card.md)。

可独立演示检索恢复：

```bash
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search --model models/qwen2.5-0.5b-instruct --adapter runs/reproduce/evidence/adapter --device mps --dtype bfloat16 --output experiments/reproduce/recovery.json
```

replay 每次只决策一个 observation，使用 bfloat16、greedy 和默认 72 个新 token 上限；它不是 batch 4 的吞吐评测。检查 trace 中确实发生检索、预算减少和新证据输入；`completed` 只是合法终止，不等于答案正确。指标定义与否定性对照见 [评测协议](evaluation_protocol.md)。未完成训练或评测的模型结果保持 `pending`，不以规则结果或训练 loss 代替。
