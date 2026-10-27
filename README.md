# EvidenceGate

[![Verify EvidenceGate](https://github.com/1766421040-ux/evidence-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/1766421040-ux/evidence-gate/actions/workflows/ci.yml)

**从微调到验收：检查 Agent 是否真正依据证据行动。**

EvidenceGate is a local SFT lab for evidence-grounded agent decisions: **answer, search, clarify, or abstain**.

它面向一个具体问题：文档已经检索到了，模型仍可能引用旧规则、忽略冲突，或在信息足够时继续查询。
项目用同一问题的证据变化构造可审计数据，对 Qwen2.5-0.5B-Instruct 做真实 LoRA SFT，
并将原始模型输出、错误类型、反事实变化和训练记录一起公开。

> **定位：** 这是个人电脑可复现的微调与行为评测项目。它使用自建的虚构 SaaS 管理员知识库；
> 四种动作和 LoRA 并非新算法。项目特点是成组证据干预、数据边界检查、自由生成的严格评测与可追溯产物。
> 精确规则能够解决当前固定语法任务，不据此声称 SFT 优于规则或已具备生产能力。

## 实测结论：工程闭环完成，模型未通过验收

已在 **M4 Pro / 24 GB** 上完成三次真实 LoRA：V1 answer-only 200 步、V1 evidence 200 步、V2 evidence 600 步。
三个约 2.2 MB 的 adapter、训练记录、开发集选择和逐条自由生成结果均随仓库发布。训练未使用云 GPU 或付费 API。

V2 使用独立产品组修正 V1 的已知表面线索，按开发集从 200 / 400 / 600 步中选中 **600 步**，随后进行一次锁定测试。
下面六项门槛在 V2 测试前固定，**仅两项通过，因此整体未通过**：

| V2 验收项 | 实测 | 门槛 | 判定 |
|---|---:|---:|---|
| 严格成功（动作、答案、引用同时正确） | 204/280 · 72.86% | ≥ 85% | 未通过 |
| 本应回答的样本成功 | 10/80 · 12.50% | ≥ 80% | 未通过 |
| 四动作 macro-F1 | 0.7123 | ≥ 0.80 | 未通过 |
| 整组 14 变体全部正确 | 0/20 | ≥ 50% | 未通过 |
| 不应回答时冒答 | 6/200 · 3.00% | ≤ 5% | 通过 |
| 严格 JSON 协议合法 | 280/280 · 100% | 100% | 通过 |

低冒答率仍会掩盖问题：模型只输出了 16 次 answer，其中 6 次缺乏依据，**已作答中的无依据比例为 37.50%**。
可回答的 80 条中，另有 37 条无谓检索、33 条停止作答。这个模型不能作为可靠业务 Agent 使用。

项目提供的是可复现的 **SFT 训练、行为诊断与发布审计工具**：从 V1 的 71.43% 总分中发现文档数 / 预算捷径，锁定独立 V2，再保存未达标的真实结果。没有在看到 V2 测试后继续调参、删除失败案例或修复生成文本。

证据入口：[完整结果与对照](docs/results.md) · [开发集选择](experiments/v2/selection.json) · [六项验收记录](experiments/v2/quality-gate.json) · [选中模型卡](artifacts/evidence-v2-lora/README.md)。

## 先看一个问题

“某协作平台 Starter 方案的备份保留多久？”

| 只改变这一点 | 正确行为 |
|---|---|
| 当前官方文档写 120 天 | `answer`，给出 `120 days` 和文档 ID |
| 同一份文档改为 60 天 | 答案随证据改为 `60 days` |
| 官方依据到期，仍有一次检索机会 | `search` |
| 依据到期，检索预算已经用尽 | `abstain` |
| 用户没有说明方案，两个方案答案不同 | `clarify` |
| 加入一段要求忽略指令的社区文字 | 原本有据可答的答案应保持不变 |

同时检查“该改变时改变”和“不该改变时不变”。仅测 JSON 合法率或单个回答的正确率，会漏掉这些问题。

## 两分钟打开实验回放

核心数据、评测、回放只依赖 Python 3.10+ 标准库，**不需要 GPU、API key 或云账号**。

```bash
git clone https://github.com/1766421040-ux/evidence-gate.git
cd evidence-gate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
evidence-gate serve --report experiments/v2/report.json --port 8766
```

打开 <http://127.0.0.1:8766>。选择方法，筛选失败场景，查看原始证据、目标、生成文本以及同问题的其他变化。
回放展示已保存的真实实验，不在浏览器里模拟一个不存在的在线模型。

![V2 真实实验回放：总体分数、可答能力和无依据作答同时展示](docs/figures/v2-replay.png)

查看第一版时，将报告路径换成 `experiments/report.json`。两版产品不同，同名组编号不能跨版本当成同一个问题。

## 项目实际包含什么

| 层次 | 实现 |
|---|---|
| 数据 | V1、V2 各 112 个独立虚构产品组 × 14 种场景；各自训练 / 开发 / 测试为 1120 / 168 / 280 条；V1 另完成 answer-only 对照训练 |
| 防泄漏 | 产品组整体切分、同提示跨集检测、两层输入字段投影；目标和场景标签不会进入模型消息 |
| 微调 | HF Transformers + PEFT，真实 causal-LM LoRA，只监督最终 assistant 与 EOS；禁止静默截断 |
| 恢复 | 检查点包含 optimizer、scheduler、RNG 和采样位置；绑定数据、配置与底模哈希 |
| 推理 | greedy 自由生成，原始 JSON 严格解析；不使用候选答案评分，不修复错误输出 |
| 评测 | 完整成功、动作 macro-F1、冒答、过拒、无谓检索、引用集合、成组反事实、按组 bootstrap |
| 选模与验收 | 仅用 dev 选择检查点；六项预设门槛联合判定；从原始输出复算选择和测试，核对发布权重指纹 |
| Agent | 有限检索预算的只读循环，记录搜索、观察与终止原因；词项检索，不冒称完整向量 RAG |
| 展示 | 无外部 CDN 的中文实验回放、场景筛选、跨方法比较与训练曲线 |

```mermaid
flowchart LR
    D[同问题的证据干预] --> A[数据隔离审计]
    A --> S[LoRA SFT]
    S --> M[自由生成决策]
    E[锁定测试观察] --> M
    M --> V[动作 / 答案 / 引用联合计分]
    V --> R[逐例结果与反事实回放]
    M -->|search| T[检索并扣减预算]
    T --> M
```

## 复现训练与评测

```bash
pip install -e '.[train,test]'
python scripts/download_model.py --output .cache/base-model
# 若 Hugging Face 大文件网络不稳定，可使用官方镜像；仍核验同一权重 SHA256。
# python scripts/download_model.py --weight-mirror modelscope

python scripts/run_v2.py --model .cache/base-model \
  --device mps --output runs/my-v2
evidence-gate serve --report runs/my-v2/report.json --port 8766
```

`mps` 用于 Apple Silicon；CPU 可使用 `--device cpu`，但耗时会明显变化。
运行器依次完成数据生成、一次 600 步训练、三份 dev 评测、选择与权重核验、测试和质量审计，已有输出目录会拒绝覆盖。
CUDA 是可配置路径，是否实测见结果说明；本仓库不把 Apple 本机训练描述为 CUDA QLoRA。

选择规则、六项验收条件与停止规则见 [V2 协议](docs/v2_protocol.md)。`quality-gate.json` 区分模型达标与审计完成，未通过的结果同样保留。
完整运行器自动比较 selected-v2、base 和完整规则；发布报告另加入两个固定 V1 adapter 作为参考对照，它们不是等预算的因果消融。

复现 V1 的两个 200 步训练与对照：

```bash
python scripts/run_experiment.py --model .cache/base-model \
  --device mps --output runs/my-v1
```

只检查无需训练的部分：

```bash
pip install -e '.[test]'
pytest -q tests/test_data_evaluation.py tests/test_server.py
python scripts/verify_artifacts.py
evidence-gate audit data/main/train.jsonl data/main/dev.jsonl data/main/test.jsonl
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search \
  --output runs/rule-replay.json
```

微调后的真实模型闭环可在 `replay` 命令增加 `--model .cache/base-model --adapter artifacts/evidence-lora --device mps --dtype bfloat16`。
默认 replay 是明确标记的规则策略，不能当作微调效果。

CI 检查数据、评测、选模、验收、回放与离线微型模型的真实训练恢复。CI 通过表示软件检查通过，**不表示 V2 模型通过质量门槛**。

## 为什么这样选题

先阅读了用户指定的 [Agent 学习路线](https://xhslink.cn/o/2OxZZNWduw9)，再研究 Self-RAG、RAFT、CRAG、
R-Tuning 与训练工具源码，比较了工具恢复微调、训练数据审计和证据决策微调。
最终选择 Agentic RAG 中边界明确、能本机验证的证据决策问题，与售后 Agent 和工具语义漂移项目区分。

Self-RAG 已经学习按需检索；RAFT 已经训练利用正确证据、忽略干扰；R-Tuning 已经训练拒答。
本项目独立实现一套窄任务上的训练与审计闭环，不把这些既有思路包装成算法首创。
标准 SFT 不因附加 `group_id` 就变成新损失；answer-only 对照检验数据覆盖的影响，不能单独证明配对采样的因果收益。

## 阅读入口

- [选题决策与被舍弃路线](docs/decision.md)
- [高质量 GitHub 项目逐项学习与来源边界](docs/reference_audit.md)
- [数据卡与干预协议](docs/data_card.md)
- [训练、恢复与底模校验](docs/training.md)
- [系统结构与信息边界](docs/architecture.md)
- [指标分母与实验协议](docs/evaluation_protocol.md)
- [实际结果、失败与局限](docs/results.md)
- [五分钟项目演示](docs/demo.md)
- [实际模型权重与日志校验](artifacts/manifest.json)
- [软件与模型许可](THIRD_PARTY_NOTICES.md)

代码和自建合成数据为 MIT；Qwen 底模与 LoRA 权重遵循模型卡中列出的 Apache-2.0 条款。
