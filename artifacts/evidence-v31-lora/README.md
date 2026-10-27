---
language:
- en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: adapter
pipeline_tag: text-generation
tags:
- lora
- sft
- evidence-gate
- synthetic-data
- agent-decision
---

# EvidenceGate Evidence LoRA V3.1

在当前 Mac M4 Pro / 24 GB 上完成的证据决策 SFT 适配器。底模是 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`。使用标准 LoRA、completion-only 交叉熵；不提出新的微调算法，也不使用云 GPU 或付费教师模型。

**在预先冻结的 V3.1 合成新实体测试中通过七项联合门槛：总体 1437/1440（99.79%），可回答样本 720/720（100%），整组 58/60（96.67%）。** 结果来自贪心自由生成，无 JSON 修复或规则替换。

**已知弱点：** 3 个错误全部提前使用下一天才生效的证据，且集中于 3 月 31 日到 4 月 1 日的跨月比较。测试中这类 future 边界共 4 行，仅 1 行正确。这是事后描述分层，不是新增门槛；总体达标不表示跨月日期推理可靠，更不表示生产可用。

## 输入与输出

模型接收问题、查询日期、剩余检索预算和三篇文档（文本、ID、来源、有效期），输出严格三字段 JSON：`action / value / citations`。动作是 `answer / search / clarify / abstain`；可答时给出天数并列出全部适用官方文档 ID，其他动作的值和引用为空。训练目标、变体名、split、gold 和隐藏检索目录不进入模型消息。

模型只负责有限任务中的一步证据决策。它不负责开放域检索、长文档阅读、自然语言解释或工具执行。完整规则可以解出这个固定语法任务，不能据此断言 SFT 优于规则。

## 数据与训练

| 项目 | 真实记录 |
| --- | --- |
| 数据 | 英文模板化虚构 SaaS；12 个证据状态 × 两个预算；train 80组/1920行，dev 24组/576行，test 60组/1440行 |
| 隔离 | 按产品实体组切分，与 V1/V2/未使用的 V3 共584个历史产品实体隔离 |
| 种子 | 数据 20261030；训练 42 |
| 起点与步数 | 从固定底模重新初始化；完整训练2400步；dev选择第2400步 |
| LoRA | rank 8，alpha 16，dropout 0.05；q/k/v/o/gate/up/down 七种线性投影 |
| 参数量 | 4,399,104 可训练 / 498,431,872 总参数 |
| 设备与精度 | Apple MPS；底模 bfloat16、LoRA 权重 F32；microbatch 1、累积4 |
| 优化 | AdamW；学习率3e-4；warmup60步后线性衰减；标准completion-only CE |
| 长度 | max_length512；真实最长499tokens；无截断；监督最后assistant和EOS |
| 训练预算 | 9600个样本位置，恰好五个打乱的完整epoch |
| 耗时 | 5568.608秒（约92.81分钟）；仅此正式训练，不含诊断、失败探针、开发/测试评测 |
| 框架 | torch2.14.1、transformers4.57.6、peft0.21.2 |
| 首步/末步loss | 1.2658387574 / 0.0001147323；loss不代替自由生成质量 |

配置见 [configs/v31.json](../../configs/v31.json)，[选中训练记录](../../experiments/training/evidence-v31/summary.json)与[完整训练记录](../../experiments/training/evidence-v31-full/summary.json)分开保存。两者这次恰好都为2400步；导出工具仍核验所选前缀和完整预算，不能把末步记录套给较早候选。

## 选择和最终确认

[冻结协议](../../experiments/v31/protocol.json)在本轮训练/预测前固定，四个候选来自同一轨迹。候选先同时满足七项 dev 门槛及回答数要求，再依次按可答成功、整组正确、总体成功、更低已答合同错误率、注册顺序排序。

| 候选步数 | dev严格成功 | dev可答成功 | 全对组 | 是否有选择资格 |
| --- | ---: | ---: | ---: | --- |
| 600 | 334/576 | 125/288 | 0/24 | 否 |
| 1200 | 380/576 | 95/288 | 0/24 | 否 |
| 1800 | 570/576 | 287/288 | 20/24 | 是 |
| **2400** | **576/576** | **288/288** | **24/24** | **是，获选** |

这些 dev 分数参与了选择。测试只对获选权重做一次正式确认；固定的 base、V2、contract 是对照，不能改变选择。推理采用 MPS/bfloat16、batch4、greedy、最多72新tokens。最终原始输出与每一条失败均保留。

| 最终测试指标 | 实测 | 预设门槛 |
| --- | ---: | ---: |
| 严格成功 | 1437/1440 · 99.79% | ≥85% |
| 可答严格成功 | 720/720 · 100% | ≥80% |
| action macro-F1 | 0.998227 | ≥0.80 |
| 全部24变体正确的组 | 58/60 · 96.67% | ≥50% |
| 不可答行中的冒答 | 3/720 · 0.4167% | ≤5% |
| JSON合法 | 1440/1440 · 100% | 100% |
| 已答中未通过证据合同 | 3/723 · 0.4149% | ≤5% |

最后一项包含应搜索/拒答却回答、错值、错引或漏引，不能直接称作事实幻觉率。风险资格要求至少输出576次可解析answer，实际为723。总体成功的按组bootstrap95%区间为99.44%–100%；组内24行相关，此区间不涵盖训练种子和真实场景的变化。

[逐条原始输出](../../experiments/v31/confirmation/selected-v3.json) · [七项验收](../../experiments/v31/confirmation/quality-gate.json) · [24单元与边界分析](../../experiments/v31/final-analysis.md) · [完整诊断与失败历史](../../docs/optimization.md)。

## 使用与审计

在仓库根目录直接复算已发布证据（无需加载模型）：

```bash
PYTHONPATH=src python3 scripts/verify_v31_release.py --root .
```

安装训练依赖、下载固定底模后，可以重放此历史评测，必须写入新路径。这不会产生新的独立测试结论：

```bash
python -m pip install -e '.[train]'
python scripts/download_model.py --output .cache/base-model
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m evidence_gate.cli evaluate \
  --policy hf --model .cache/base-model \
  --adapter artifacts/evidence-v31-lora --data data/v31/test.jsonl \
  --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 \
  --output runs/v31-replay-evaluation.json
```

完整重训入口、锁定确认账本及复现限制见 [训练说明](../../docs/training.md)。依赖版本、MPS实现或设备变化可能影响数值；不保证跨环境逐字节复现。

## 发布范围与局限

本目录包含17,640,136字节的增量权重、配置、原始checkpoint元数据、许可证与此模型卡。底模、tokenizer、optimizer状态张量不发布；checkpoint中的训练状态哈希用于来源说明，公开目录不能直接恢复optimizer。全部2400步训练日志与已发布权重可独立核对。

数据仍是有限英文模板、显式日期和来源字段、固定三篇文档。单一训练种子、同一语法的新实体留出不验证中文、自然措辞迁移、开放知识、多轮工具或广泛提示注入防御。V3.1同时改变数据、LoRA覆盖与预算；改善不能单独归因于某一个改动。旧版本的失败结果保持不变。

## 许可证与精确身份

适配器遵循Apache-2.0，见[LICENSE](LICENSE)；项目代码及自建数据为MIT。

| 对象 | SHA256 |
| --- | --- |
| adapter权重 | `8f4416485a369460f97b0cf47ec2e56e298588c46d3df60c3e94b31541715c19` |
| adapter配置 | `1b2f9300f84334d7d3024d34e77014d1f72e1c251ef5d434055b046ce814947d` |
| checkpoint | `025da321c750ee4319e2086f378efcef888b6c45814ee31cbac3235770266771` |
| 训练数据 | `f684041829641485a06650ebb1b7152c9bf9851b08b05f47b0c50ebb520e3876` |
| 标准化训练配置 | `0fce6557c90906eec4c7fdac6a96222e135ffa3c2e0f6b9491c03a83d5678781` |
| 基础模型快照 | `74bf6dd08049d862b3bd56ac2a84a8e1980ae93acfad71612a9ea00548e887b7` |
