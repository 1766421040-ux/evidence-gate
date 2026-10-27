# 本机 LoRA SFT 与复现

训练器使用 PyTorch / Transformers / PEFT，在冻结基础权重上训练 LoRA；目标是最后一个 assistant completion 与 EOS 的标准 next-token loss。没有新的损失函数、强化学习或付费教师 API。

当前主流程是 **V3.1**：完成预登记的 2400 步本机训练，由 576 行 dev 在四个候选中选择第 2400 步，再对冻结 test 做一次正式确认。所选模型的七项门槛全部通过：严格成功 **1437/1440**、可答成功 **720/720**、整组全对 **58/60**；3 个失败均在 `future` 情境。证据见 [正式门槛](../experiments/v31/confirmation/quality-gate.json) 和 [独立 raw 复算](../experiments/v31/final-analysis.md)。这只支持冻结的有限合成任务，不等于业务上线认证。

V1/V2 的失败结果完整保留在下文。V2 当轮按协议停止后，用户明确要求进一步诊断并优化至质量门槛，才启动后续 train/dev 诊断、新实体数据和预登记训练；没有改写 V2 结论、重选 V2 checkpoint 或降低门槛。V3.1 的任务与验收边界见 [完整协议](v31_protocol.md)。

## 模型和环境

需要 Python 3.10+。原始运行环境为 M4 Pro / 24 GB，当前 `configs/v31.json` 指定 MPS、bfloat16 和 gradient checkpointing。MPS 不可用时明确报错，不静默改到 CPU。其他设备可以另存配置，但结果应另列，不能冒充同次运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
```

安装区间见 `pyproject.toml`，实际运行的包版本写入 `run_manifest.json`。模型使用 `Qwen/Qwen2.5-0.5B-Instruct`，公开 revision 为 `7ae557604adf67be50417f59c2c2f167def9a775`。首次下载需要网络和本地磁盘空间；后续训练只加载本地文件，不调用模型 API。

```bash
python scripts/download_model.py --output .cache/base-model
```

主权重 `model.safetensors` 的预期 SHA-256 为 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。脚本默认从 Hugging Face 下载；网络较慢时可加 `--weight-mirror modelscope`，元数据仍来自固定 HF revision，镜像权重必须通过相同哈希检查。“同名模型”不足以保证同一文件。训练器对本地权重、tokenizer 与配置文件另做完整快照哈希，并绑定训练数据和配置。

## V3.1：推荐复现入口

先审计已发布证据，无需下载模型或安装训练依赖：

```bash
PYTHONPATH=src python3 scripts/verify_v31_release.py --root .
```

该脚本从保存的 raw 输出重新评分，核验冻结数据/源码、dev 选择、七项门槛、adapter/checkpoint、所选训练前缀、完整预算和确认收据。它不重新执行模型。退出码 0 才是 `verified`；未完成的发布材料为 `pending`/2；不一致或质量失败为 1。正式测试通过与公开文件审计通过是两件事，缺失材料不能默认通过。

查看完整确认流程保存的报告：

```bash
PYTHONPATH=src python3 -m evidence_gate.cli serve --report experiments/v31/confirmation/report.json --port 8766
```

浏览器打开 `http://127.0.0.1:8766`。必须先有真实报告文件；页面浏览保存的模型输出，不进行在线推理。

需要完整重训时，完成上面的环境安装与固定底模下载，再执行：

```bash
PYTHONPATH=src python scripts/run_v31_experiment.py --model .cache/base-model --output runs/v31-reproduction
```

这是实际训练入口，固定 MPS，不提供 CPU/云端替代。它先核验 freeze、源码、训练配置、三份数据 SHA、底模和 V2 参考 adapter，以及本地 holdout 确认账本，才创建新输出目录。之后按以下顺序执行：

1. 登记固定配置和 600/1200/1800/2400 四个候选，完成全部 2400 步；不按中途 dev 结果提前停止预算。
2. 对四个 checkpoint 使用相同 dev、MPS/bfloat16、batch 4、greedy、72 新 token 上限进行自由生成，不修补 JSON 或用规则替换决策。
3. 从 raw 复算七门槛和回答分母资格，只在合格候选中按冻结顺序选点；无合格候选就保存 `no_eligible_candidate`、保留全部 dev 输出并退出 2，禁止 test。
4. 重新校验所选模型和训练绑定，再做一次 selected test，以及预先固定的 base、V2 adapter、contract 参考；规则结果不算模型结果。
5. 保存门槛、逐条结果、训练记录、收据和报告。确认通过退出 0；执行失败或最终质量失败退出 1，保留已有证据。

同一测试文件 SHA 对应的 `.cache/confirmation-attempts/` 账本一旦存在，入口在训练前拒绝。不得删除/更名账本来重试已见测试；已执行正式确认的当前工作机因此不能再次运行这条完整实验。新环境重做公开协议也只是**历史复现，不是新的独立盲测**。输出目录必须全新；入口不覆盖、不自动续跑，也没有跳过门槛或账本的开关。

### 固定训练配置与实际预算

| 项目 | V3.1 |
| --- | --- |
| 基础模型 | 固定 revision 的 Qwen2.5-0.5B-Instruct，未合并基础权重 |
| 数据 | 80 train groups × 24 单元 = 1920 行，seed 20261030 |
| LoRA | rank 8、alpha 16、dropout 0.05；q/k/v/o 与 gate/up/down 七个投影 |
| 训练 | seed 42，2400 optimizer steps，microbatch 1 × accumulation 4 |
| 优化 | 学习率 3e-4、warmup 60 步、随后线性衰减 |
| 本机内存设置 | bfloat16、长度 512、gradient checkpointing |
| 可训练参数 | 4,399,104 |
| 候选 | 每 600 步保存，完整预算后统一评估四个候选 |

[仅 train/dev 的 tokenizer 预检](../experiments/v31/tokenizer-preflight.json) 得到最大完整序列 499/497 token，保留 512 上限，不截断。训练按实际 chat template 检查 prompt 前缀，只监督最终 assistant completion 与 EOS，prompt/padding 标签为 `-100`；以有效监督 token 数归一梯度累积。使用标准交叉熵，没有 action 加权、辅助推理目标、教师蒸馏或新的优化算法。

[真实选中训练记录](../experiments/v31/confirmation/selected-training.json) 中完整预算为 2400 步，处理 9600 个样本位置，即每个训练样本恰好五轮；训练耗时 5568.61 秒（约 92.8 分钟，不含下载和评测）。这是本次 M4 Pro 24 GB 运行记录，不是其他机器的耗时保证。所选与最后一步恰好相同；审计仍分别绑定“所选前缀”和“完成预算”，不能用最终 loss 冒充任意较早 checkpoint 的结果。

### 开发选择和公开文件

1800 与 2400 步均通过 dev 七门槛。2400 的可答严格成功为 288/288，高于 1800 的 287/288，按预先冻结的首要排序项选中；不是看到 test 后选择。其 dev 为 576/576，不代表 test 必然全对。完整四候选结果见 [dev 独立复算](../experiments/v31/dev-analysis.md)，最终选点见 [selection.json](../experiments/v31/selection.json)。

| 位置 | 契约 |
| --- | --- |
| `runs/v31-reproduction/` | 私有复现树：registry、training、四份 dev、selection、可能产生的 confirmation、outcome/failure |
| `artifacts/evidence-v31-lora/` | 公开所选 adapter 权重、配置、checkpoint 元数据和模型卡；无 optimizer |
| `experiments/training/evidence-v31/` | 所选 checkpoint 的训练前缀、summary、run manifest、公开文件 manifest |
| `experiments/training/evidence-v31-full/` | 完成 2400 步的全部训练记录，另含原字节 `checkpoint-step-002400.json` |
| `experiments/v31/confirmation/` | 唯一正式确认的 raw、门槛、固定参考、报告与收据 |

一键复现**不自动导出或替换公开模型**，不能把 `verify_v31_release.py --root` 指向 `runs/v31-reproduction`；公开审计器需要完整仓库的约定布局。发布的 adapter 可推理和审计，但不包含恢复优化器所需状态。依赖采用安装区间，实际版本在运行 manifest；MPS 跨环境和新登记时间戳不保证逐字节复现。首次下载后训练/推理只读本地文件，不产生云训练或模型 API 费用。

## 历史实验：V1 / V2

以下命令和数据描述保留原实验语义，不是当前 V3.1 的主入口；已经公开的历史 test 不可重新当作未见测试。

## V1：已完成的对照与复现

仓库的 `data/main/` 是固定实验数据。审计跨 split 边界：

```bash
evidence-gate audit data/main/train.jsonl data/main/dev.jsonl data/main/test.jsonl
```

需要重新生成时使用新目录，例如 `evidence-gate generate --output data/reproduced --seed 20261027`，比较 manifest 后再训练。生成器拒绝覆盖已有 manifest，评测也拒绝覆盖现有输出。

以下命令使用新的复现目录；已存在的运行须显式恢复或改目录。

```bash
evidence-gate train --data data/main/train_answer_only.jsonl --model .cache/base-model --config configs/main.json --output runs/reproduce/answer-only
evidence-gate train --data data/main/train.jsonl --model .cache/base-model --config configs/main.json --output runs/reproduce/evidence
```

两者都用 rank 8 / alpha 16、`q_proj` 与 `v_proj`，预设 200 个 optimizer steps、microbatch 1 × accumulation 4、学习率 3e-4、最大序列长度 512、bfloat16 和 gradient checkpointing。answer-only 有 320 行，evidence 有 1,120 行；各处理 800 个样本位置，样本重复率和监督 token 总数不同。这是训练分布的对照，**不是等 token 消融，也不能单独证明配对结构带来增益**。

训练器按实际 chat template 形成输入，只监督 completion 与 EOS，屏蔽 prompt 和 padding。超过上下文上限会报错，不静默截断引用或证据。token 审计的最终值在 `run_manifest.json.data_audit`；不能从配置的 512 反推实际输入长度。

`configs/smoke.json` 仅运行 3 步，用于检查训练和保存链路。smoke loss 下降不构成任务质量结果。V1 正式配置固定 200 步，没有自动 early stopping；不得用 test 挑 checkpoint。V1 evidence 的总体 71.43% 与可答 25%、整组全通过 0% 必须一起阅读，不能仅凭总体分数宣布微调目标完成。

也可用运行器在新目录一次生成 V1 数据、训练两个 adapter、完成同输入评测并生成回放：

```bash
python scripts/run_experiment.py --model .cache/base-model --device mps --output runs/my-v1
evidence-gate serve --report runs/my-v1/report.json --port 8766
```

## V2：独立数据修正与开发集选点

V2 是观察 V1 的文档数量 / 预算捷径后制定的后续方案。它不修改 V1 数据或覆盖 V1 权重：新 seed `20261028`，新产品按 80 / 12 / 20 组划分，每组仍有 14 个变体；所有行固定三篇文档、两份 official 与一份 community，并共享注入尾句。具体日期平衡和检索快照边界见 [V2 协议](v2_protocol.md)，不能把消除几个已知线索写成“消除了全部捷径”。

已保存的 [选点记录](../experiments/v2/selection.json) 只使用 168 行、12 个 dev group。第 200 / 400 / 600 步的四动作等权严格成功率分别为 60.42% / 76.77% / 77.40%，因此按预设规则选中第 600 步。第 400 与 600 步的总体成功数同为 123/168，但总体成功率不是选点指标；不因最终 test 失败而改选其他权重。选中 adapter 的 SHA-256 为 `5af71346aa50dc90b3f7cfc130127a94e3b97c83076c84ac6503b62fb5d4c9e5`，并与 [test 预测](../experiments/v2/selected-v2.json) 和 [选中训练记录](../experiments/v2/selected-training.json) 绑定。

只查看已完成结果，无需加载模型或重训：

```bash
evidence-gate serve --report experiments/v2/report.json --port 8766
```

需要重新运行完整 V2 流程时，使用新目录：

```bash
python scripts/run_v2.py --model .cache/base-model --device mps --output runs/my-v2
evidence-gate serve --report runs/my-v2/report.json --port 8766
```

`run_v2.py` 会实际执行一次新的训练；新运行结束后才生成其最终报告。它拒绝任何已存在的输出目录，也不会自动覆盖、跳过失败阶段或启动额外调参轮次。执行顺序是：

1. 逐字节复制锁定的 `experiments/v2/protocol.json`，在 `runs/my-v2/data/` 独立生成并验证 v2 数据。
2. 仅用 train / dev 做 tokenizer 长度预检；从 512 开始，只在不足时增加到下一个 64 倍数，不截断，也不读 test 模型输出。
3. 按 `configs/v2.json` 从基础模型新建 LoRA，完成一次 600 步训练；每 200 步保存，候选限定为第 200、400、600 步。
4. 对三个候选执行同一 dev 评测：bfloat16、batch 4、greedy、最多 72 个新 token。按四种真值动作等权的**严格成功率均值**选点，精确并列选较早步数；它不是动作 macro F1，也不是总体成功率。
5. 保存 `selection.json`，核验被选 adapter 和 checkpoint 的实际哈希后，才允许任何 test 模型推理。正式报告包含 selected-v2、contract 规则和 base；规则不是模型训练结果。
6. 从原始生成文本复算质量门槛，保存 `quality-gate.json` 和最终 `report.json`。未达标也保留报告，不删除难例、不修补输出；该 V2 运行器不会依据 test 自动再训练。后续用户授权的独立开发沿革见本文开头。

V2 与 V1 都采用 rank 8 / alpha 16、q/v 投影、microbatch 1 × accumulation 4、学习率 3e-4 和梯度检查点；V2 的总步数是 600，warmup 是 15，V1 则为 200 / 5。本次预检得到 train / dev 的最大完整序列长度为 492 / 494，因此保留 512 上限，未截断样本。V2 数据与训练预算同时改变，不能与 V1 视作等预算因果消融，也不能把分数变化单独归因于数据修正。

复现目录中的 `selected-training.json` 只使用选中 checkpoint 自身的 history。其 `optimizer_steps / selected_step / adapter_sha256` 对应所选权重，`total_training_budget_steps` 单独记录总预算 600；不能把最终 600 步的 summary 或末步 loss 套给较早权重。`selection.json` 保存三个候选的 dev 分数与指纹，`reproduction.json` 保存最终入口和门槛状态。

以下六项须在同一次锁定 test 上同时满足，实际阈值由复制的协议读取。[已完成的门槛审计](../experiments/v2/quality-gate.json) 对 280 行、20 个 test group 的 raw 输出重新评分，核验了案例完整性、数据指纹和保存摘要，真实性检查通过，模型质量门槛失败：

| 指标 | 门槛 | V2 第 600 步实际结果 | 是否通过 |
| --- | ---: | ---: | --- |
| 严格任务成功率 | ≥ 85% | 204/280（72.86%） | 否 |
| 可答样本严格成功率 | ≥ 80% | 10/80（12.5%） | 否 |
| 四动作 macro F1 | ≥ 0.80 | 0.7123 | 否 |
| 整组 14 变体全部正确比例 | ≥ 50% | 0/20（0%） | 否 |
| 不应回答样本中的过度回答率 | ≤ 5% | 6/200（3%） | 是 |
| 格式合法率 | 100% | 280/280（100%） | 是 |

模型在 80 个可答样本中错误 search 37 次、abstain 33 次，核心问题仍是不能可靠地依据可用证据作答。过度回答门槛通过也不等于输出可靠：本次一共输出 16 个合法 answer，其中 6 个无依据，即 `unsupported_answer_rate = 6/16 = 37.5%`；其分母与 `over_answer_rate = 6/200 = 3%` 不同。按锁定协议保留失败结果，V2 当轮到此停止，没有根据该 test 重选 V2 checkpoint 或降低门槛。后续优化来自用户明确追加授权，使用新版本实体与单独冻结的确认协议；这些历史测试已被看过，不能当作后续独立验证。

这是有限合成任务的验收，不是生产认证。没有可解析 answer 时，风险门槛标为不具备评估条件，不能把数字上的 0 当作安全证据；空分母的 null 也不转换为 0。质量门槛失败与审计执行失败不同：默认退出 0 表示审计完成并保存真实 pass/fail，只有显式加 `--require-pass` 才在未达标时返回 2。数据、案例或摘要与 raw 不一致则报错。

无需模型依赖即可复核仓库中的正式结果，使用一个新输出文件：

```bash
python scripts/quality_gate.py --evaluation experiments/v2/selected-v2.json --data data/v2/test.jsonl --protocol experiments/v2/protocol.json --output runs/v2-quality-gate-recheck.json
```

若复核的是自己重新训练的结果，则改用该运行目录中的数据、协议和预测：

```bash
python scripts/quality_gate.py --evaluation runs/my-v2/evaluations/selected-v2-test.json --data runs/my-v2/data/test.jsonl --protocol runs/my-v2/protocol.json --output runs/my-v2/quality-gate-recheck.json
```

## 历史资源适配：为什么修改本机配置

最初的 float32 / microbatch 2 / accumulation 2 配置在第 67 步发生 MPS 内存不足，日志只记录了完成的 66 步；它不是完成的 200 步训练。失败运行保存在本地 `runs/failed-float32/`，对外只保留 `experiments/failed-float32/` 的失败说明、metrics 和 manifest，不发布未完成的权重。

随后将训练改为 bfloat16、microbatch 1 / accumulation 4、长度上限 512 和 gradient checkpointing，保持有效 batch 4 与 V1 的 200 步预算；通过 3 步 smoke 后，两次正式 V1 训练均已完成。这是资源适配，不是根据测试成绩选择配置。训练精度降低和激活重计算也会改变数值或耗时，不能把前后两次拼成一次连续训练；失败尝试与正式运行的数据 / 配置指纹不同，必须使用新运行目录。

## 训练器的手动恢复与产物（不等于 V3.1 一键流程续跑）

保存间隔由配置决定：V1 `configs/main.json` 每 50 步，V2 `configs/v2.json` 每 200 步，V3.1 `configs/v31.json` 每 600 步。恢复只接受同一输出目录下、数据 / 配置 / 模型哈希一致且文件校验通过的 checkpoint。下面是 V1 的手动恢复示例：

```bash
evidence-gate train --data data/main/train.jsonl --model .cache/base-model --config configs/main.json --output runs/reproduce/evidence --resume runs/reproduce/evidence/checkpoints/step-000050
```

恢复包含 optimizer、scheduler、随机数状态和采样游标；状态使用 JSON 与 safetensors，不读取 pickle。恢复命令仅用于确实存在的中断 checkpoint，正常完成后无需再执行。

V2 若需手动恢复训练，必须使用该次运行实际写出的配置，而不是另建一个配置冒充同一运行：

```bash
evidence-gate train --data runs/my-v2/data/train.jsonl --model .cache/base-model --config runs/my-v2/config.json --output runs/my-v2/training --resume runs/my-v2/training/checkpoints/step-000200
```

它只恢复训练阶段，不自动续跑整个 `run_v2.py`。复现运行器继续拒绝已有目录；后续 dev、选点和 test 必须仍按锁定协议执行，不能因中断绕过选点边界。

| 文件 | 作用 |
| --- | --- |
| `adapter/` | 保存的 adapter 配置、权重和 tokenizer；基础模型权重未合并进去 |
| `run_manifest.json` | 模型文件哈希、配置、数据哈希、依赖版本和 token 审计 |
| `metrics.json` | 每步 loss、grad norm、学习率、输入 / 监督 token 和累计耗时 |
| `summary.json` | 完成步数、可训练参数量、adapter 哈希与训练摘要 |
| `artifact_manifest.json` | 非 checkpoint 训练产物的文件校验清单 |

训练日志说明参数更新确实执行；质量必须靠下面的独立评测。单个种子 42 不证明多种子稳健性。

## V1 同输入的三方评测

正式对照统一使用相同 test、MPS、bfloat16、batch 4、greedy free generation 和 72 个新 token 上限。下面显式传入精度与 batch，避免依赖 CLI 的默认值；较小 batch 用于控制本机 MPS 缓存占用。模型输出不做 JSON 修补，也不用规则替换决策。

```bash
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name base --model .cache/base-model --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/base.json
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name answer-only-sft --model .cache/base-model --adapter runs/reproduce/answer-only/adapter --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/answer-only.json
evidence-gate evaluate --data data/main/test.jsonl --policy hf --name evidence-sft --model .cache/base-model --adapter runs/reproduce/evidence/adapter --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/evidence.json
evidence-gate report experiments/reproduce/base.json experiments/reproduce/answer-only.json experiments/reproduce/evidence.json experiments/contract.json experiments/first-match.json --training runs/reproduce/answer-only/metrics.json runs/reproduce/evidence/metrics.json --output experiments/reproduce/report.json
evidence-gate serve --report experiments/reproduce/report.json
```

预测文件的 `model_provenance` 保存实际基础模型文件指纹和 adapter SHA，`inference` 保存精度、设备、batch 与解码参数。上述 `--training metrics.json` 会同时加载旁边的 `summary.json`，将曲线与已评测 adapter 的哈希匹配。报告拒绝合并不同数据哈希或不同模型推理设置的评测。如果重新生成了不同数据，需在该数据上重跑所有规则与模型基线。`mean_latency_ms` 是 batch wall time / batch size，不是单请求延迟或 p95；当前没有据此声明线上 QPS。

两个额外诊断数据已固定在 `data/diagnostics/`：paraphrase 与 redacted 各 56 行，均来自前 4 个 test group，三种模型的诊断评测与闭环评测均已完成，结果见 [experiments/diagnostics](../experiments/diagnostics)。复算时可将 `evaluate --data` 分别指向这两个文件并使用新的输出路径，按各自数据哈希单独生成报告。它们不能合并进主测试当作新增独立样本，也不能用于挑选 checkpoint。具体变换见 [数据卡](data_card.md)。

可独立演示检索恢复：

```bash
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search --model .cache/base-model --adapter runs/reproduce/evidence/adapter --device mps --dtype bfloat16 --output experiments/reproduce/recovery.json
```

replay 每次只决策一个 observation，使用 bfloat16、greedy 和默认 72 个新 token 上限；它不是 batch 4 的吞吐评测。检查 trace 中确实发生检索、预算减少和新证据输入；`completed` 只是合法终止，严格闭环成功要求每一步符合当时可见证据。指标定义与否定性对照见 [评测协议](evaluation_protocol.md)。V1 诊断与 V2 最终 test 均已保存，但两者属于不同数据版本和评测范围；V2 的单步 test 不能冒充 V2 多步闭环实验，也不能以规则分数或训练 loss 代替模型质量。
