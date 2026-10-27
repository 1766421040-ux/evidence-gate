# V2：针对 V1 捷径的预设后续实验

本文记录 V2 的设计和验收规则，不预填模型成绩。机器可读协议为 [experiments/v2/protocol.json](../experiments/v2/protocol.json)；它在 V2 模型测试前固定。V2 是观察 V1 失败后的后续实验，不是事先未见任何项目结果的首轮研究。

## 为什么保留 V1，并另建 V2

V1 的 evidence adapter 倾向于在特定干扰模板出现时回答，其余输入按预算选择检索或停止。文档数量与注入短句均能复述其动作行为，普通证据充分的样本和需要澄清的样本仍失败。总体成功率不能证明学会了证据判断；原始预测、负结果和数据均保留，见 [V1 结果](results.md)。

V2 使用独立的 [build_v2.py](../scripts/build_v2.py)，不修改 V1 的 `src/evidence_gate/data.py` 或 `data/main/`。新 seed 为 `20261028`，重新生成产品与事实，仍按产品组切分 80 / 12 / 20 groups，每组 14 变体，对应 1,120 / 168 / 280 行。六字母产品名与 V1 的五字母命名空间不同，构建时也显式检查两版产品零重叠。

组编号可以重用，因此跨版本引用必须同时带数据版本或哈希；`v1:g0092` 与 `v2:g0092` 不是同一产品。V2 文件、动作分布、来源与日期审计均绑定在 [data/v2/manifest.json](../data/v2/manifest.json)。数据已经锁定，训练过程中不改样本或标签。

## 改掉了哪些线索，保留了哪些任务语义

每行固定三篇文档、两篇 official 和一篇 community。每行 community 都使用相同的事实句式和注入尾句，不再只有 distractor 出现注入文本。文档 ID 随组随机分配，组内各动作使用同一组 ID；初始排列也随机分配，reordered 再反转它。ID 不表示权威性，位置不表示相关性。

以下变化与 V1 不同，不能只按旧变体名字猜具体构造：

| 变体 | V2 的构造 |
| --- | --- |
| supported / value_changed | 保留当前有效官方事实；值置换同时改变该反事实世界的恢复目录 |
| distractor | 不增加文档；将已有 community 文档中的无关产品改成被问产品，保留不可信数字与共同注入短句 |
| missing | 将主要官方文档改为另一产品；其余官方说明属于另一套餐 |
| untrusted | 唯一同时匹配问题产品与套餐的说法来自 community；两篇官方文档分别属于另一套餐和另一产品 |
| conflict | 两篇当前有效官方文档匹配同一范围，但数值不同 |
| expired / future | 与问题匹配的官方证据分别处于到期边界或尚未生效；仍按可见预算决定后续动作 |
| ambiguous | 问题省略套餐，按固定合同必须 clarify；不能用检索代替询问条件 |

所有输入依旧由 `messages_for` 投影。`variant`、`group_id`、`target` 和 `gold` 不进入提示词。目标由生成器构造，再用只读取公开 observation 的 `contract_rule` 核对；这证明有限语法下的标签一致性，不证明模型能力。

官方日期组合使用 VV、VE、VF：V 为当前有效，E 为已到期，F 为尚未生效；community 始终当前有效，但仍不具有权威性。冲突要求 VV，过期要求 VE，未来证据要求 VF，这是任务语义。其他变体安排同样的日期组合，例如正确答案旁也可以放过期的无关官方文档。

每个 split 的三种日期组合均包含 answer、search、clarify、abstain，单按这个无序来源 / 日期组合查表不能超过多数类基线。数量、来源数量和注入出现次数三个固定特征与动作的互信息为零。**这只覆盖列出的特征，不等于全部元数据组合、文本模式或位置特征都无相关性。** 预算本身仍有较强类别先验，必须同时检查可回答任务和澄清类表现。

## 检索目录与配对的解释边界

`gold.catalogue` 是预先构造的“当前官方恢复快照”：使用新的官方文档 ID，提供当前可核对的事实。它不是一个同时保留全部历史版本和冲突版本的静态检索库。只有实际执行 search 后返回的文档才成为下一步可见证据；同名旧文档不会被静默覆盖成新内容。

value_changed 的恢复快照使用置换后的值，不能恢复成原答案。构建时独立检查公开观察的 target，并检查检索返回后对恢复事实的回答。clarify 仍优先于检索；结束流程不等于整段任务成功。

值置换、文档重排和缺失证据的预算对保留相应的其他可见字段；证据缺失对使用与 supported 相同预算。每行均为三文档，因此数量变化无法解释这些对。

为平衡日期组合，部分 supported 与 expired 配对还会改变无关官方文档的日期状态。这些 expiry 配对检查的是两个情境是否都正确，**不能全部解释成只改变一个日期字段的因果实验**。同理，distractor 的检查是 community 内容相关性改变后的输出保持，不再是增加文档的实验。配对通过仍要求两侧动作、事实值和引用全部正确。

## 固定训练预算与开发集选择

从同一 Qwen2.5-0.5B-Instruct 基础快照重新初始化 LoRA，不从 V1 adapter 继续训练。训练目标仍是最终 assistant completion 与 EOS 的标准 next-token loss，没有新的配对损失。

[configs/v2.json](../configs/v2.json) 固定 600 个 optimizer steps、seed 42、MPS / bfloat16、microbatch 1 × accumulation 4、rank 8 / alpha 16、q/v 投影、学习率 3e-4、warmup 15 和 gradient checkpointing。600 步意味着 2,400 个样本位置，不代表 2,400 个独立样本。

长度规则在训练前约定：先用真实 tokenizer 预检，512 不足才增加至能容纳数据的下一个 64 倍数，禁止静默截断。实际预检仅检查 train / dev，保留 512；证据见 [tokenizer-preflight.json](../experiments/v2/tokenizer-preflight.json)。这项长度检查没有读取测试模型输出。

只保存并比较预设的第 200、400、600 步候选。三个候选在同一 dev 上使用 bfloat16、batch 4、greedy 和最多 72 个新 token，保留未修补输出。选择指标为：

```text
dev_macro_strict_success = (1/4) × Σ_action
    [该真值动作类别中 action、value、citations 全正确的样本数 / 该类别样本数]
```

它不是 action macro F1，也不是按样本数加权的总体成功率。四种真值动作等权，避免数量较少的 clarify 被掩盖；精确并列时选较早步数。

[select_checkpoint.py](../scripts/select_checkpoint.py) 从 raw 输出重新计分，要求同一 dev、训练数据 / 配置 / 底模与推理设置，并保存选中步数、候选分母和 adapter / checkpoint 指纹。只读取 dev 结果；开发成绩参与了选择，不能当作无偏的最终测试估计。

在新的复现目录中可按下列顺序执行，已有运行不应重复覆盖：

```bash
evidence-gate train --data data/v2/train.jsonl --model .cache/base-model --config configs/v2.json --output runs/reproduce/v2
for step in 000200 000400 000600; do
  evidence-gate evaluate --data data/v2/dev.jsonl --policy hf --name v2-dev-$step --model .cache/base-model --adapter runs/reproduce/v2/checkpoints/step-$step --device mps --dtype bfloat16 --batch-size 4 --max-new-tokens 72 --output experiments/reproduce/v2-dev-$step.json
done
PYTHONPATH=src python scripts/select_checkpoint.py experiments/reproduce/v2-dev-000200.json experiments/reproduce/v2-dev-000400.json experiments/reproduce/v2-dev-000600.json --data data/v2/dev.jsonl --output experiments/reproduce/v2-selection.json
```

## 锁定测试与发布条件

在完成上述 dev 选择前，不进行 V2 测试模型推理。先保存选择记录，再对选中的 adapter 做一次正式 test 评测。固定 base 与两个 V1 adapter 可以作为同一 V2 输入上的参考；规则仍须单独标记。数据构造阶段的 schema、哈希和规则检查不属于模型测试评测。

以下六项是 [预设协议](../experiments/v2/protocol.json) 中的联合验收条件，必须在同一次锁定 test 上同时满足，不能挑选较好的指标宣布通过：

| 条件 | 固定门槛 |
| --- | ---: |
| 严格任务成功率 | ≥ 85% |
| 可回答样本严格成功率 | ≥ 80% |
| 四动作 macro F1 | ≥ 0.80 |
| 整组 14 变体全部正确比例 | ≥ 50% |
| 不应回答样本中的过度回答率 | ≤ 5% |
| 格式合法率 | 100% |

门槛用于本项目的有限任务验收，不是生产系统认证。未完成评测时保持 pending；任一项不满足就明确记录未通过，仍可公开代码、预测和失败分析。查看 V2 test 后不再启动第三轮数据或模型修改，也不通过删除难例、修补模型输出或更换统计口径“过关”。

## 哪些比较不能作因果结论

V2 同时改变数据、训练总步数、warmup / 学习率调度和 checkpoint 选择流程。V1 两个 adapter 各训练 200 步，V2 预设训练 600 步后用 dev 选择。即使最终选到第 200 步，它也属于 600 步调度中的 checkpoint，不等同于 V1 的 200 步运行。

因此新旧结果只能说明这套后续方案的观察表现，不能把差异单独归因于数据修正、成组关系、训练时长或某个日期设计。`train_answer_only.jsonl` 虽已提供，存在文件不代表已训练对应的等预算 V2 对照。配对、逐动作与整组指标也不能替代独立自然文档上的泛化证据。
