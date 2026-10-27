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

# EvidenceGate Evidence LoRA v2

这是在本机 Apple MPS 上完成 600 个优化器更新步后，由开发集选中的 PEFT LoRA 适配器。基础模型为 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，锁定版本为 `7ae557604adf67be50417f59c2c2f167def9a775`。本次使用标准 LoRA 和 SFT；项目贡献在于可审计的数据、模型选择及失败分析流程，没有提出新的训练算法。

**最终质量门槛未通过：测试集严格成功 204/280（72.86%），可回答样本仅 10/80（12.50%）成功，6 项门槛通过 2 项。** 发布保留实际失败结果，供复现与诊断使用。

模型预测 `answer / search / clarify / abstain`，输出包含动作、答案值和引用 ID 的 JSON。输入限于问题、日期、预算、文档及其来源和有效期；训练标签、划分、变体名及隐藏恢复目录不进入模型消息。检索预算行为遵循明确约定的恢复协议，不代表学习了最优搜索规划。

## 数据与训练记录

v1 的实测动作与文档数量捷径吻合后，本次改用独立数据种子的 [v2 英文合成数据](../../data/v2/manifest.json)：每条观测固定 3 篇文档、2 篇官方来源和 1 篇社区来源，共享注入文本扰动。该设计移除了已发现的数量等线索，并不保证不存在其他捷径。训练、开发、测试按虚构产品问题组隔离；训练为 80 组、1120 条，开发为 12 组、168 条。

| 项目 | 实际记录 |
| --- | --- |
| 训练起点 / 预算 | 从锁定基础模型重新初始化；完整运行 600 步；获选第 600 步 |
| 设备与精度 | Apple MPS；底模 bfloat16，LoRA 张量 F32；本机执行，未使用付费云计算 |
| LoRA | rank 8，alpha 16，dropout 0.05；`q_proj`、`v_proj` |
| 可训练参数 | 540,672 / 494,573,440 |
| Micro batch / 梯度累积 | 1 / 4，有效 batch size 4 |
| 优化器 / 学习率 | AdamW；3e-4；warmup 15 步，之后线性衰减 |
| 长度与内存 | max_length 512；训练最长 492 tokens；梯度检查点；未截断样本 |
| 监督范围 | 最后一个 assistant completion 与 EOS；prompt 和 padding 标签屏蔽 |
| 随机种子 | 训练 42；数据生成 20261028 |
| 框架 | torch 2.14.1；transformers 4.57.6；peft 0.21.2 |
| 全部 600 步耗时 | 1322.86 秒；不含下载、开发集/测试集评测及此前失败尝试 |
| 首步 / 末步训练 loss | 1.292804 / 0.019277；这些数值不代表任务质量 |

完整配置见 [`configs/v2.json`](../../configs/v2.json)，训练来源见 [选中训练记录](../../experiments/v2/selected-training.json)。v2 同时改变了数据和训练预算，因此不能将相对 v1 的变化单独归因于其中某一项。

## 开发集选择与最终评测

按照训练前冻结的 [协议](../../experiments/v2/protocol.json)，仅比较 200、400、600 步三个 checkpoint，以开发集中四种真实动作各自的严格成功率取等权平均，最高者获选，精确平局选更早步。严格成功要求动作、答案值和引用全部正确。输出由贪心自由生成获得，未进行候选评分或 JSON 修复；推理设置统一为 bfloat16、batch size 4、最多 72 个新 tokens。

| 步数 | 开发集宏平均严格成功率 | 开发集总体严格成功 |
| --- | ---: | ---: |
| 200 | 60.42% | 113/168 |
| 400 | 76.77% | 123/168 |
| **600（获选）** | **77.40%** | **123/168** |

第 600 步在开发集各动作的成功数为：answer 13/48、search 52/60、clarify 12/12、abstain 46/48。选择程序从保存的原始输出重新评分，并核验数据、基座、训练配置和适配器哈希；完整记录见 [`selection.json`](../../experiments/v2/selection.json)。这些开发集分数参与了选择，不能作为独立最终测试分数。测试集没有参与 checkpoint 选择。

选定第 600 步后，冻结模型在 20 个测试问题组、280 条记录上完成一次最终评测。原始输出见 [`selected-v2.json`](../../experiments/v2/selected-v2.json)，原始输出重评分与门槛检查见 [`quality-gate.json`](../../experiments/v2/quality-gate.json)。

| 最终测试指标 | 结果 | 预定门槛 | 结论 |
| --- | ---: | ---: | --- |
| 总体严格成功率 | 204/280＝72.86% | ≥85% | 未通过 |
| 可回答样本成功率 | 10/80＝12.50% | ≥80% | 未通过 |
| 动作 macro-F1 | 0.7123 | ≥0.80 | 未通过 |
| 组内全部变体成功 | 0/20＝0% | ≥50% | 未通过 |
| 不可回答样本上的直接作答率 | 6/200＝3.00% | ≤5% | 通过 |
| JSON 格式有效率 | 280/280＝100% | 100% | 通过 |

总体成功率按问题组 bootstrap 的 95% 区间为 70.71%–75.00%。模型在 80 条可回答样本中有 70 条未答对；16 次实际直接作答中有 6 次缺乏正确证据支持（37.50%）。因此，表中以 200 条不可回答样本为分母的 3.00% 不能解释为已具备可靠回答能力。完整对照与失败分析见 [实验结果](../../docs/results.md)。

## 复现与发布范围

先按仓库根 README 安装依赖，并下载锁定且校验权重 SHA256 的基础模型，然后可复现冻结模型的最终评测：

```bash
python scripts/download_model.py --output .cache/base-model
python -m evidence_gate.cli evaluate \
  --policy hf --model .cache/base-model \
  --adapter artifacts/evidence-v2-lora \
  --data data/v2/test.jsonl --device mps --dtype bfloat16 \
  --batch-size 4 --max-new-tokens 72 \
  --output experiments/evidence-v2-lora-rerun.json
```

没有 MPS 的机器可改用 `--device cpu --dtype float32`；运行时间及数值可能变化。生成过程只接收公开观测，标签仅用于事后评分。

本目录发布增量权重、适配器配置、许可证和 `checkpoint.json`，不包含底模或分词器。checkpoint 元数据用于核验选中步数、训练绑定及推理文件；**未发布优化器状态张量 `training_state.safetensors`，本目录仅用于推理与审计，不能直接恢复优化器训练**。基础分词器从同一锁定模型快照读取。实际发布文件范围见 [发布校验清单](../manifest.json)。

## 适用范围与限制

- 数据是模板化、虚构的英文任务，未验证中文、开放域检索、真实业务文档或跨语言泛化。
- 单模型、单训练种子、少量相关变体的实验不足以证明通用 Agent 能力；本次最终测试已明确未通过质量门槛。
- 规则基线与模型输出分别报告；低训练 loss 和合法 JSON 都不能替代证据语义正确性。
- 这里的适配器是可复核的实验产物，不能据此宣称生产可用。

## 许可证与精确身份

适配器采用 Apache-2.0，见本目录 [LICENSE](LICENSE)。项目代码和自编数据的许可证见仓库根目录。

| 对象 | SHA256 |
| --- | --- |
| `adapter_model.safetensors` | `5af71346aa50dc90b3f7cfc130127a94e3b97c83076c84ac6503b62fb5d4c9e5` |
| `adapter_config.json` | `361c9e943c1fb27ad923633b32ce08a60976c0ef57478a0e92f81fc8cdf96308` |
| 训练数据文件 | `11629323ee99030472ccabd944301d9f4fec4a3a4c0624c574d1e00d5487f23a` |
| 标准化完整训练配置 | `b7db06d009008454127de3efa963bc959237792cd7e3e76cc9b324a219e97938` |
| 基础模型快照清单 | `74bf6dd08049d862b3bd56ac2a84a8e1980ae93acfad71612a9ea00548e887b7` |
