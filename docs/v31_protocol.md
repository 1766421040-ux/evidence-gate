# V3.1：证据决策 SFT 的冻结协议

V3.1 在固定英文合成任务中检验：模型能否根据当前可见证据选择 `answer / search / clarify / abstain`，并给出正确事实与完整引用集合。它使用标准 LoRA 和 assistant completion 交叉熵，没有提出新训练算法。主模型已经完成一次正式测试，七项门槛全部通过；可核查的结论是 **1437/1440 严格成功、720/720 可答成功、58/60 产品组全部正确**。剩余 3 个错误均是采用尚未生效的证据回答。[正式门槛](../experiments/v31/confirmation/quality-gate.json)与[独立 raw 复算](../experiments/v31/final-analysis.md)共同保留成功和失败。

## 为什么形成这个版本

V1 的总体分数隐藏了可答和澄清失败，并暴露文档数量等线索。V2 固定三文档并做 dev 选点，但所选模型仍只有 10/80 可答成功，六门槛仅两项通过。该轮失败没有被覆盖。用户随后明确要求深入诊断、继续优化至质量门槛，才启动后续开发。

train/dev 诊断核对了监督 mask、shifted CE 与模型 loss：低平均 loss 不表示自由生成决策已学会，剩余 NLL 主要集中在 action token。这是损失分布分析，不是梯度占比测量。仅将 LoRA 扩至七个投影的 V2 数据对照提高了部分可答召回，却增加了错误回答，不能单独认定容量扩大解决了问题。[诊断记录](../experiments/v3/diagnostics/README.md)与[容量对照](../experiments/v3/capacity-probe/comparison.json)保持原样。

随后 V3 的 train/dev 静态审查发现：一个不核对问题主题、只接受单个候选证据的规则仍能严格全对。因此在 V3 的模型确认前，V3.1 用新实体增加主题过滤和一致证据单元；没有使用 V3 dev/test 模型预测作此修订。V3 的冻结文件继续保留，变更理由及旧 SHA 见 [supersession.json](../experiments/v31/supersession.json)。

V3.1 同时修改数据覆盖、LoRA 目标模块和训练预算，属于组合工程改进，不能把相对 V2 的变化归因为某一个因素。旧测试已见，不再作为后续开发的独立证明。

## 数据和可见信息

固定数据 seed `20261030`；按完整产品组隔离为 train 80 / dev 24 / test 60 组，共 1920 / 576 / 1440 行。每组有 12 种语义，各含 budget 0/1，形成 24 个相关单元：

`supported, value_changed, distractor, reordered, missing, expired, future, conflict, ambiguous, untrusted, irrelevant_topic, consensus`。

完整名称列表以 [protocol.json](../experiments/v31/protocol.json) 的 `variants` 为准，不从字符串猜测或省略难例。每组真值为 answer 12、search 5、clarify 2、abstain 5。每行三篇文档、两篇 official 与一篇 community，日期、来源、适用产品/套餐/主题都来自公开 observation。新数据的查询、干扰和恢复实体与 main/v2/v3 各 split 隔离；模板并未隔离。生成器、文件 SHA 和已知捷径审计见[数据卡](data_card.md)与 [freeze.json](../experiments/v31/freeze.json)。

模型只接收原有 system contract 和 `messages_for(observation)`；标签、variant、split、design、gold 不进入推理提示词。训练入口审计序列化消息必须与 observation 相符，completion 是紧凑三字段 JSON。只监督最终 assistant completion 与 EOS；prompt/padding 被屏蔽，超长样本报错而不静默截断。query plan 缺失要求 clarify；证据充分时即使预算为零也应 answer；证据不足按预算 search 或 abstain。

有效期是 `[valid_from, valid_until)`，官方来源本身不足以支持答案。回答必须匹配产品、套餐、主题和时间，并引用全部适用且值一致的官方文档；冲突时不自行选择优先级。`irrelevant_topic` 只改变 conflict 第二篇证据的主题，`consensus` 只改变其值，两项同预算配对分别检查主题范围与证据一致性。完整边界见数据卡；不能称所有跨情景配对都是单字段因果干预。

## 训练预算和候选登记

[configs/v31.json](../configs/v31.json) 固定 Qwen2.5-0.5B-Instruct、seed 42、MPS/bfloat16、rank 8 / alpha 16 / dropout 0.05，LoRA 覆盖 q/k/v/o 与 gate/up/down 七个投影。microbatch 1 × accumulation 4，学习率 3e-4，warmup 60 步后线性衰减，gradient checkpointing，最长 512 token。

在任何本轮训练或 dev 推理前登记 **2400 optimizer steps**，候选限定 600、1200、1800、2400；完成整个预算后评估全部候选。1920 行共处理五轮。train/dev 的 tokenizer 最大完整长度为 499/497，预检没有读取 test，也未加载模型权重，因此保留 512 上限。[registry.json](../experiments/v31/registry.json) 绑定底模快照、数据、归一化配置、源配置与训练代码哈希。

实际训练更新 4,399,104 个参数，完整 2400 步耗时 5568.61 秒，具体设备、依赖版本、每步 loss/token/耗时与 SHA 保存在训练记录中。时间不含下载和评测；低训练 loss 不是验收证据。所选第 2400 步恰好等于预算终点，但记录仍分别表达 selected prefix 和 full budget。

## 七门槛、风险分母和 dev 选择

每个候选必须在**同一完整 dev** 上通过以下七项及回答分母资格，才有权参与排序。冻结 test 使用同样阈值；规则基线没有资格成为所选模型。

| 指标 | 定义与分母 | 阈值 | 正式 selected test |
| --- | --- | ---: | ---: |
| 严格成功率 | 格式、动作、value、引用集合同时正确 / 全部行 | ≥85% | 1437/1440（99.7917%） |
| 可答严格成功率 | 严格正确 / 真值 answer 行 | ≥80% | 720/720（100%） |
| 四动作 macro F1 | 四种真值动作等权；不是可答召回 | ≥0.80 | 0.998227 |
| 整组全对率 | 24 单元全部严格正确的组 / 全部组 | ≥50% | 58/60（96.6667%） |
| 过度回答率 | 真值非 answer 却输出合法 answer / 真值非 answer 行 | ≤5% | 3/720（0.4167%） |
| 格式合法率 | 严格三字段协议合法 / 全部行 | 100% | 1440/1440 |
| 已答合同错误率 | 实际合法 answer 中未严格成功的行 / 实际合法 answer 行 | ≤5% | 3/723（0.4149%） |

最后一项对应代码 `unsupported_answer_rate`，包括不该回答、错值、错引与漏引；不能直接改称事实幻觉率。两个回答风险门槛还要求实际合法 answer 数不少于 `ceil(0.8 × 真值可答数)`，并拒绝空分母/null：dev 至少 231 个，test 至少 576 个。本次 test 实际回答 723 个，满足资格。1200 步 dev 虽有较低风险比例，但仅回答 98 个，不能因低覆盖而通过风险验收。

合格候选按可答严格成功率、整组全对率、总体严格成功率依次降序，再按已答合同错误率、预登记顺序升序；计数比使用精确分数比较。没有合格候选则 `selected=null, test_allowed=false, status=no_eligible_candidate`，不运行 test。

| 候选步数 | dev 严格成功 / 576 | dev 可答成功 / 288 | 七门槛 |
| ---: | ---: | ---: | --- |
| 600 | 334 | 125 | fail |
| 1200 | 380 | 95 | fail |
| 1800 | 570 | 287 | pass |
| 2400 | 576 | 288 | pass，获选 |

2400 在首要排序项优于 1800，所以在测试前已被选中，无需用 test 打破平局。[selection.json](../experiments/v31/selection.json) 与 [dev 独立复算](../experiments/v31/dev-analysis.md) 保存完整证据。所选 adapter SHA-256 为 `8f4416485a369460f97b0cf47ec2e56e298588c46d3df60c3e94b31541715c19`。

## 一次正式确认和独立复算

模型推理固定为 MPS、bfloat16、batch 4、greedy 自由生成、最多 72 新 token，无候选答案评分、JSON 修复或规则代答。进入 test 前，确认器重算完整 dev 选择，验证模型/adapter/config/checkpoint/训练预算，以及冻结数据和源码；按 test 文件 SHA 排他创建本地确认账本，再保存开始收据。完整流程还执行预先固定的 base、V2 adapter 和 contract 参考，最终报告位置为 `experiments/v31/confirmation/report.json`；这些参考不能事后替换成更有利的比较对象。

已产生的 selected test raw 在 [selected-v3.json](../experiments/v31/confirmation/selected-v3.json)（文件名沿用通用 V3 确认器），由 [quality-gate.json](../experiments/v31/confirmation/quality-gate.json) 和[独立分析](../experiments/v31/final-analysis.md)逐条重新解析、评分、核验完整性与摘要。两者一致通过七门槛；错误集中在两个产品组的三个 `future` 单元。现在这些 test 已被看过，不能据此更换本轮 checkpoint、修补三个输出或再试同一确认来改写结果。

若最终确认失败也必须保留；本地账本在推理开始前即创建，失败或中断不会自动释放。账本是本地流程保护，不是保密测试服务，也不证明不存在未披露实验。公开审计另外检查该 holdout 的唯一公开开始/完成收据。

## 复现与证据布局

从仓库根目录复算公开文件，无需模型、网络或训练依赖：

```bash
PYTHONPATH=src python3 scripts/verify_v31_release.py --root .
```

只有 `verified`/退出码 0 表示公开发布证据完整通过；缺失的待发布文件是 pending/2，审计不一致或确认质量失败是 1。它核查 raw、门槛、选择、freeze、训练绑定与全预算，不重新运行模型，也不将缺少证据视为通过。已完成确认收据存在却缺失必需文件时直接失败。

完整 MPS 重训入口、固定底模下载和环境安装见[训练文档](training.md)：

```bash
PYTHONPATH=src python scripts/run_v31_experiment.py --model .cache/base-model --output runs/v31-reproduction
```

该入口只接受全新目录，先检查同 holdout 账本；已执行确认的本机应提前拒绝，不能删除账本来绕过。它运行固定预算和 dev 选择，仅合格才确认；不发布、不自动导出权重、不覆盖、不续跑，也不提供放宽门槛参数。新环境重做公开协议仍是历史复现，不是新的独立测试。

公开 adapter 在 `artifacts/evidence-v31-lora/`，只有推理与审计所需权重/config/checkpoint 元数据，没有 optimizer。所选训练前缀位于 `experiments/training/evidence-v31/`，完整 2400 步记录位于 `experiments/training/evidence-v31-full/`，后者保留原字节 `checkpoint-step-002400.json`。独立复现的 `runs/NEW` 不符合公开根目录契约，不能直接传给发布验证器作为 `--root`。冻结 protocol 中的初始 status 描述建档时刻；实际选点、质量和发布完整性应分别读取 selection、quality-gate、公开审计结果，不修改旧冻结状态字段。

## 结论的范围

这是一个数据生成种子、一个训练种子的有限英文模板任务。60 个 test group 才是组层面的样本；组内 24 单元、scope pairs、四个同次训练 checkpoint 都不是独立实验重复。按组 bootstrap 区间只描述本合成组分布，不包含训练种子、语言迁移或真实场景的不确定性。

没有通过本协议证明新的措辞、多轮 Agent、中文、长上下文、开放域检索或生产安全。此前 V1 的 paraphrase/redacted 不能充当 V3.1 压力测试，检索恢复目录的静态一致性审计也不是 V3.1 多步模型实验。公开结果证明的是：标准 LoRA 小模型在预先声明的证据合同及新合成实体上达到了七项门槛，并提供可复算的失败、选择和训练证据。
