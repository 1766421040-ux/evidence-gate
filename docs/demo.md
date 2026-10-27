# 五分钟演示：证据变了，Agent 的决定是否跟着变

本演示面向 Agent / 大模型应用开发面试。先用已完成的 V1 对照解释失败，再切换到独立后续实验 V2：V2 已完成 600 步训练、dev 选点和最终 test，六项质量门槛只通过两项，结论为 `fail`。交付的是完整可复现的 SFT 诊断工具，当前模型未达业务使用标准。所有回放读取真实保存的结果，不在现场重训。

## 演示前准备

在仓库根目录完成 Python 3.10+ 环境和基础安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
evidence-gate serve --report experiments/report.json --port 8766
```

打开 `http://127.0.0.1:8766`。仓库已包含 `experiments/report.json`，无需下载底模就能查看 V1 的原始预测和训练记录。它是只读实验回放，不是实时训练或推理。服务在启动时读取报告快照；更换报告后应重启服务。演示过程中可以另开一个已激活环境的终端运行后面的命令。

## 0:00–0:45：提出一个具体错误

“管理员问某套餐的日志保留天数，检索可能返回过期、冲突或另一套餐的说明。我的项目检查模型能否根据当前证据，选择回答、继续检索、询问用户或停止作答。”

指出输出只有 `action / value / citations`。这是可审计的有限任务；检索有固定恢复规则，不是学习最优规划。最近邻和选题取舍见 [decision.md](decision.md) 与 [reference_audit.md](reference_audit.md)。

## 0:45–1:45：展示同一问题的变化

在案例列表中查看同一组 `g0092` 的 `supported`、`value_changed`、`missing_search`、`missing_stop` 和 `reordered`，对照原始 observation、expected 与模型 raw 输出。

- 证据值替换时，问题不变，两侧都应回答各自的值。
- 文档顺序改变时，答案和引用集合应保持正确。
- 缺失证据且有预算时应 search；同样缺失证据但预算为零时应 abstain。

强调：每个配对两侧都正确才算通过；不是“输出变了就算成功”。train / dev / test 按完整产品组隔离，组内变体没有拆开。可现场执行边界审计：

```bash
evidence-gate audit data/main/train.jsonl data/main/dev.jsonl data/main/test.jsonl
```

它核查数据边界，不评价模型质量。变体定义、数据哈希和诊断来源见 [data_card.md](data_card.md)。

## 1:45–2:45：比较训练前后，讲清分母

在报告中并排查看 base、answer-only SFT、evidence SFT 与两种规则对照。先看严格成功率，再看 `answerable_success`、`over_answer_rate`、`over_refusal_rate` 与配对分数；点击一个失败案例，指出究竟错在动作、值还是引用。

优先展示这三个已验证的事实，而不是只报一个总分：

- Evidence SFT 总体严格成功为 200/280（71.43%），但本应回答的样本只成功 20/80（25%），整组 14 变体全通过为 0/20。
- Answer-only SFT 的可答样本成功 80/80，却在不应回答的 200 个样本上全部作答。
- Base 的 280 个输出均未通过严格格式解析。界面因此将无依据作答比例标为“不可估计”；不能把没有合法 answer 当成安全，也不能把协议失败解释为知识能力为零。

V1 evidence 模型的动作与文档数 / 预算启发式完全一致，故总体成绩不足以证明它学会了证据判断。统计和逐例证据见 [结果说明](results.md)。UI 的“无依据作答比例”以合法输出的 answer 为分母；`over_answer_rate` 则以本不应回答的样本为分母，两者不要混称。

“两次 SFT 都是标准 LoRA，各固定 200 步、有效 batch 4。answer-only 只训练可回答样本；evidence 覆盖异常证据和非回答动作。两者 token 数和样本重复率不同，不能把差异单独归因于配对采样。”

在预测记录的 `inference` 中核对正式三方推理都使用 bfloat16、batch 4、greedy 和 72 个新 token 上限。训练的有效 batch 与推理 batch 是两项独立设置。

规则满分并不尴尬：当前固定语法可由规则完整解决，模型没有被证明是部署必需品。规则分数不能当成 SFT 分数。证据位置是 `experiments/report.json` 中各 method 的 `cases / summary / model_provenance`，以及报告中关联的训练日志。

## 2:45–3:45：运行一次检索闭环

不需要模型下载的规则演示：

```bash
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search --output runs/demo-rule-replay.json
```

打开 `runs/demo-rule-replay.json`，展示两步 trace：第一次缺少证据，选择 search；词项检索返回文档，预算从 1 减到 0；第二次依新证据作答。这个命令默认使用规则策略，必须明确说明。

如果已提前安装训练依赖、下载底模，并已有发布的 adapter，可把同一流程交给真实微调模型：

```bash
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search --model .cache/base-model --adapter artifacts/evidence-lora --device mps --dtype bfloat16 --output runs/demo-model-replay.json
```

此处按单实例逐步推理，仍使用 bfloat16、greedy 与 72 个新 token 上限。不要预测它一定成功，直接看 raw 输出和 trace。`completed` 仅代表合法终止；闭环严格成功要求每一步都符合当时可见证据，包括检索前不能猜中隐藏目录就算正确。批量主评测的均摊耗时与这里的单例生成耗时不可混用。文件已存在时换一个输出名；工具拒绝覆盖已有实验记录。

## 3:45–4:30：展示一次失败及处理

打开 `experiments/failed-float32/` 的失败记录：原 float32 / microbatch 2 运行完成 66 步，在第 67 步内存不足。随后改为 bfloat16、microbatch 1 / accumulation 4、512 长度和 gradient checkpointing，通过 3 步 smoke 后，两个 V1 运行均从基础模型重新开始并完成各自的 200 步。

这说明处理过真实资源约束，也说明没有把中断训练当作完成。不要从 smoke loss 或主训练 loss 推导准确率；正式行为结果来自留出集。具体配置、哈希和恢复机制见 [training.md](training.md)。

## 4:30–5:00：说明现在能证明什么

展示 `data/diagnostics/manifest.json` 和 [已完成的诊断结果](results.md#v1-诊断与真实检索闭环)：措辞改写与去证据诊断各 56 行，均来自前四个 test group。它们与主测试共享组，不能相加当作更多独立样本；去证据后真值已重算。闭环结果也要求逐步正确，不以合法终止率替代成功率。

“目前交付的是本机可重现的数据、LoRA、严格评测和失败回放。它能检验这个有限协议上的证据依赖，不能直接证明真实企业文档的泛化、复杂多轮规划或生产可靠性。下一步需要独立自然文档的失败集，而不是仅增加合成模板数量。”

## 随后切换 V2：修正已知捷径后，仍未通过质量门槛

打开 [预设后续协议](v2_protocol.md)：新 seed 和新产品组、每行固定三篇文档、统一来源组成和注入尾句，修正已发现的表面线索。训练预算锁定为 600 步，按 dev 在 200 / 400 / 600 步中选择一次，再进入 test。实际 [选点记录](../experiments/v2/selection.json) 选中第 600 步，选点没有使用 test。

保留 V1 页面，在另一个终端启动 V2 只读回放：

```bash
evidence-gate serve --report experiments/v2/report.json --port 8767
```

打开 `http://127.0.0.1:8767`，选择 `selected-v2`，并对照 [原始 test 输出](../experiments/v2/selected-v2.json) 与 [质量门槛审计](../experiments/v2/quality-gate.json)：

- 严格任务成功 204/280（72.86%），本应回答的样本成功 10/80（12.5%），整组 14 个变体全部正确为 0/20。
- 四动作 macro F1 为 0.7123，低于 0.80 门槛。可答样本中 37 个被错误地继续检索，33 个被错误地拒答，不能将保守行为等同于证据判断成功。
- 格式合法率 280/280、过度回答率 6/200（3%）通过各自门槛，但另外四项失败，联合结论仍为 `fail`。模型只输出了 16 个合法 answer，其中 6 个无依据；界面的无依据作答比例是 6/16（37.5%），不是 3%。

无需加载模型即可从数据与 raw 输出重新核验门槛，使用尚不存在的输出文件：

```bash
python scripts/quality_gate.py --evaluation experiments/v2/selected-v2.json --data data/v2/test.jsonl --protocol experiments/v2/protocol.json --output runs/demo-v2-quality-gate.json
```

默认退出 0 代表完成真实性审计，不代表模型达标；保存的 `status` 仍是 `fail`。需要让自动化检查在质量失败时返回 2，可加 `--require-pass`。这轮数据与训练预算同时改变，不能把 V1 与 V2 的分数差异解释为某一个改动的因果增益；也不能在看到 test 后改选 checkpoint 或降低门槛。完整重训命令见 [训练文档](training.md#v2独立数据修正与开发集选点)。

## 面试官追问时直接打开的代码

| 问题 | 证据入口 |
| --- | --- |
| 如何避免把 gold 喂给模型？ | [data.py](../src/evidence_gate/data.py) 的 `messages_for / audit`，以及训练入口再次审计 |
| 微调到底训练了什么？ | [training.py](../src/evidence_gate/training.py) 的 completion mask、LoRA 配置、token 加权 loss 和 checkpoint |
| 为什么不只看准确率？ | [evaluation.py](../src/evidence_gate/evaluation.py) 的严格计分、明确分母和配对聚合 |
| 检索是否真的发生？ | [agent.py](../src/evidence_gate/agent.py) 的 `retrieve / run_episode` 与已保存 trace |
| 结果能否对应到这个 adapter？ | 预测的 `model_provenance.adapter_sha256` 与训练 summary 的 `adapter_sha256` |
| 为什么选 600 步，失败后有没有换门槛？ | [selection.json](../experiments/v2/selection.json) 的 dev 选点、[锁定协议](../experiments/v2/protocol.json) 与 [quality-gate.json](../experiments/v2/quality-gate.json) |
| 哪些结论没有证据？ | [evaluation_protocol.md](evaluation_protocol.md) 的否定性对照与范围限制 |

这条讲解路径用于说明可验证的工作，不应改写成“解决通用幻觉”“生产级可靠”或未经实测的性能简历表述。
