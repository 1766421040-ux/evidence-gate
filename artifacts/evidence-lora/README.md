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

# EvidenceGate Evidence LoRA

这是在当前电脑上真实完成 200 个优化器更新步的 PEFT LoRA 适配器。基础模型是 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，本目录只包含增量权重与适配器配置，不包含基础模型或分词器。

完整证据决策监督：同一问题配对构造有效、缺失、过期、冲突、非可信来源、预算耗尽等输入，训练 answer / search / clarify / abstain 四类动作。

## 任务和数据

输入是一个虚构协作软件问题、判断日期、查询预算以及带来源和有效期的文档；输出固定为 `{"action":"...","value":"...","citations":[...]}`。证据有效期是左闭右开区间，非官方来源不能支持直接作答。预算规则是明确约定的恢复协议，不代表学习了最优搜索规划。

本次训练使用仓库自编的英文合成数据 [`data/main/train.jsonl`](../../data/main/train.jsonl)：80 个训练问题组、1120 条记录。产品身份先分组划分，再生成变体；训练集与测试集不共享产品身份。观测只包含公开字段，`target`、`gold`、变体名称和划分标签不进入模型消息。该数据不是客户语料，也不是外部基准。

同一基础模型和配置用于两个适配器，均运行 200 个更新步、有效 batch size 为 4。由于数据集大小不同，训练数据覆盖次数不同；这是固定更新预算对照，不是等 epoch 对照。

## 实际训练记录

| 项目 | 记录 |
| --- | --- |
| 设备与精度 | Apple MPS · 底模 bfloat16；发布的 LoRA 张量为 F32；没有使用云 GPU |
| 参数方法 | LoRA，rank 8，alpha 16，dropout 0.05 |
| 目标模块 | `q_proj`、`v_proj` |
| 可训练参数 | 540,672 / 494,573,440 |
| 优化器更新 | 200 步；从第 0 步开始 |
| Micro batch / 梯度累积 | 1 / 4 |
| 最大序列长度 | 512；实际最长 476 tokens；未截断样本 |
| 学习率 / warmup | 3e-4 / 5 步 |
| 内存策略 | 梯度检查点；固定长度 padding；padding 标签屏蔽 |
| 监督范围 | 最后一个 assistant completion 和 EOS；输入 prompt 不计入 loss |
| 随机种子 | 42 |
| 框架版本 | torch 2.14.1 · transformers 4.57.6 · peft 0.21.2 |
| 首步 / 末步训练 loss | 1.116279 / 0.021831869 |
| 已记录训练耗时 | 389.36 秒；不含下载、评测和失败尝试 |

这些 loss 和耗时来自已保存的训练日志，不能当作任务质量分数。是否提升了决策能力，应查看严格评分、配对输入以及实际模型原始输出。完整评测见 [实验结果与限制](../../docs/results.md)；质量结论以该结果文档及原始输出的严格复算为准。

日志与来源：

- [逐步训练记录](../../experiments/training/evidence/metrics.json)
- [训练摘要](../../experiments/training/evidence/summary.json)
- [模型、数据、配置绑定清单](../../experiments/training/evidence/run_manifest.json)
- [原始运行产物哈希](../../experiments/training/evidence/artifact_manifest.json)
- [发布文件校验清单](../manifest.json)

原始运行清单还包含本地保存过的 tokenizer 文件哈希；为避免重复发布基础模型资源，本仓库只发布适配器、许可证和训练证据。发布文件的实际范围由 `artifacts/manifest.json` 定义。分词器应从锁定的基础模型快照读取。

## 复现与使用

在仓库根目录执行。先按根 README 安装训练依赖，再下载被仓库脚本锁定版本且校验权重 SHA256 的基础模型：

```bash
python scripts/download_model.py --output .cache/base-model
python -m evidence_gate.cli evaluate \
  --policy hf --model .cache/base-model \
  --adapter artifacts/evidence-lora \
  --data data/main/test.jsonl --device mps --dtype bfloat16 \
  --batch-size 8 --max-new-tokens 72 \
  --output experiments/evidence-lora-rerun.json
```

没有 MPS 的机器可使用 `--device cpu --dtype float32`；耗时和数值可能不同。该命令使用本地快照和贪心自由生成，不替模型修复 JSON，也不在生成时读取测试标签。重新训练使用同一数据文件和 [`configs/main.json`](../../configs/main.json)，并指定尚不存在的输出目录。

## 使用范围与限制

- 本适配器用于证据约束下的受控行为实验，不能凭此声称通用 Agent 能力或真实业务可用性。
- 训练语料为模板化英文合成数据；中文表现、开放域检索和真实文档泛化没有由这些训练记录证明。
- 单一训练种子、固定预算的小模型结果，不能证明方法对其他模型规模或训练预算同样有效。
- 规则基线表现必须与模型实测分开；较低训练 loss 不代表较少无依据作答。
- 基础模型版本为 `7ae557604adf67be50417f59c2c2f167def9a775`；本次基础权重和数据的精确身份见 `run_manifest.json`。

## 许可证与校验

此适配器采用 Apache-2.0；随目录附带从基础模型快照复制的 [Apache 2.0 许可证](LICENSE)。项目实现代码的许可证独立见仓库根目录。

`adapter_model.safetensors` SHA256：

```text
1a15242a23305e91fe1e487ba88153afa348a887a27632bd59b44830ad3d4283
```
