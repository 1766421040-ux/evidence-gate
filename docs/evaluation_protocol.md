# 评测协议：动作、证据和格式分别算

当前确认协议是 **V3.1：每组 24 情境、60 个全新测试组、七项联合门槛**。其问题是：标准 LoRA SFT 能否让小模型遵守当前可见证据下的有限决策协议？它不评价通用知识能力，也不证明最优 Agent 策略或生产可靠性。

以 [冻结协议](../experiments/v31/protocol.json)、[候选注册表](../experiments/v31/registry.json)、[选模记录](../experiments/v31/selection.json) 和 [最终质量审计](../experiments/v31/confirmation/quality-gate.json) 为机器可核验证据。协议文件中的 `status` 是冻结当时的记录，不会在训练后改写；后续状态由注册表、选择和确认结果记录。V1/V2 与原 V3 的历史文件保留，不用新协议重写旧成绩。

## 数据边界、选模与最终确认

V3.1 使用 seed `20261030`；train / dev / test 分别为 80 / 24 / 60 个产品组，共 1,920 / 576 / 1,440 行。同组全部情境归属同一 split。查询产品、干扰产品及恢复目录中的实体，均与其他 split 及 main / V2 / V3 的已见实体隔离。字节哈希、代码哈希、独立审计和旧版本替代记录见 [manifest](../data/v31/manifest.json)、[freeze](../experiments/v31/freeze.json) 与 [supersession](../experiments/v31/supersession.json)。这是可审计的使用约束，不是保密测试服务器。

模型 prompt、三字段输出、标准 completion-only SFT loss 和评分器保持不变。正式训练使用 `configs/v31.json`：LoRA rank 8、七类线性投影、2,400 个 optimizer steps、microbatch 1 / accumulation 4，有效 batch 4。训练样本各经历五个 epoch；数据覆盖、LoRA 容量和预算均发生变化，因此它不是相对于 V2 的等预算因果消融。

600 / 1,200 / 1,800 / 2,400 步候选在训练前登记。先对全部候选完成同一 dev 评测，再按如下规则选择：

1. 同一候选须通过全部七门槛，并满足已答风险的分母资格。
2. 合格候选依次比较可答成功率、整组成功率、总体成功率（降序）、已答风险（升序），最后按登记顺序。计数比例用精确分数比较。
3. 无合格候选则记录 `no_eligible_candidate`、`selected=null`、`test_allowed=false`，禁止最终模型推理。
4. 完成登记的总训练预算，核验底模、配置、数据、checkpoint 和 adapter 身份，再对选定候选做一次最终确认。保存所有失败；已见 test 不可用于继续挑点后再宣称独立验证。

本轮选中 2,400 步；1,800 步也合格，但按预设顺序排名较后。dev 的 576/576 是适应性选模证据。最终 test 严格成功 1,437/1,440、可答 720/720、整组 58/60，七门槛全部通过；三个失败均为提前采信尚未生效的官方证据。主结论限于这次有限合成协议。

所有正式模型推理统一 MPS、bfloat16、batch 4、greedy free generation、最多 72 个新 token，不修补输出。训练有效 batch 与推理 batch 是两项设置。单实例 replay 的生成耗时不等于批量均摊耗时，二者都不是请求 p95。

## 24 情境与引用合同

十二种语义各配预算 0 / 1，命名为 `<semantic>_b0 / _b1`。每行均有两篇 official、一篇 community；模型只看到序列化的公开观察，不看到 variant、目标、组 ID 或恢复目录。公开观察本身的证据 ID 可用于引用。

| 语义 | 正确行为与检验点 |
| --- | --- |
| `supported` | 适用且当前有效的官方证据支持作答 |
| `value_changed` | 只替换适用事实值；回答随证据改变 |
| `distractor` | 社区干扰变化，不应改变正确作答 |
| `reordered` | 顺序变化，不应改变值或引用集合 |
| `missing` | 无适用官方事实；一部分组含混合主题，另一部分组的两篇 official 都是相同错误主题 |
| `expired` / `future` | 相关证据已过期 / 尚未生效，不能作当前依据 |
| `conflict` | 两篇当前有效的适用 official 值冲突，不能任取一个 |
| `ambiguous` | 缺少套餐范围，应 clarify，预算不改变这个要求 |
| `untrusted` | 相关事实只来自非可信来源，不能按官方依据作答 |
| `irrelevant_topic` | 两篇 official 均当前有效且同产品套餐，但第二篇主题无关；只引用适用证据作答 |
| `consensus` | 两篇当前适用 official 的值一致，应回答并引用两个 ID |

在没有可答依据且范围明确时，预算 1 应 search，预算 0 应 abstain。有充分依据的情境在两种预算下都应 answer。当前合同要求**全部适用官方引用**：引用排列不计顺序，但少一个、多一个、引用错误 ID 或重复引用都不能通过。`consensus` 用原有集合评分即可判定，不需要为新情境修改评分器。

相关产品、适用主题、来源权威性、有效日期和剩余预算是任务所需的语义信息；目标不是让所有公开字段与标签统计独立。独立审计检查已知的非因果捷径，如“所有文档提到同一产品”“两官方数值相同”“任意不同主题就拒答”“官方候选数不是一就拒答”，并要求相应反例。审计只能排除所检查的规则，不能证明完全不存在其他捷径。

## 当前固定参照与历史对照

V3.1 最终确认固定比较 `selected-v3`（方法名沿用，实际协议 V3.1）、未微调 base、历史 V2 adapter 和 `contract`。前三者须使用完全相同的可见输入、序列化、解码和评分。历史 V2 adapter 在 V3.1 上的新结果，是固定参照评测；不可把它在旧 V2 测试集上的分数放在同一表内冒充同输入比较。

`contract` 只解析公开字段，检查产品、套餐、主题、时间、来源和事实一致性，不读取 gold。它是有限语法的精确规则参照，不是通用语义核验器。规则成功不构成 SFT 成功，也不支持“部署必须用微调”的结论。若将来使用规则守卫改写模型输出，应分别报告原始模型与带守卫系统。

历史 V1 还包括 answer-only SFT、evidence SFT 和忽略日期/来源/冲突的 `first-match`。V1 两个 SFT 均为 200 步，但分别用 320 行可答数据和 1,120 行全部情境，token 量和覆盖率不等；它们不是等 token 消融，也不是上游 RAFT 的完整复现。

## 严格评分顺序

1. 保存未经修改的 raw 输出。
2. 只接受恰好 `action / value / citations` 的单个 JSON 对象。拒绝重复键、额外字段、markdown 包裹、重复引用和非法类型。
3. 检查动作枚举及字段约束。非 answer 的 value / citations 须为空；answer 必须有非空值和至少一个引用。非法输出是失败，不自动改成 abstain。
4. 比较动作、值和完整引用集合。值仅去首尾空白、忽略大小写，不做单位换算或同义匹配；引用比较忽略排列顺序。
5. 按案例、变体、组和合法配对聚合，保留每个分母。独立审计从 data 与 raw 重新评分，同时核对案例身份、完整性、原始指标和摘要。

真值来自冻结的数据合同，独立规则审计检查可见输入的一致性；评分器是精确目标比较，不是额外的语义蕴含模型或付费 LLM。不能在看到 test 后扩充答案匹配规则来提高成绩。

## 指标、精确分母与七项门槛

令 N 为所有案例数，A 为本应回答的案例数，U 为本不应回答的案例数，E 为模型输出**可解析且字段合法的 answer** 数，G 为完整问题组数。本轮 N=1,440、A=720、U=720、E=723、G=60。invalid 计入失败，并在混淆矩阵独立列出。

| 字段 / 展示名 | 定义 | V3.1 test / 联合门槛 |
| --- | --- | --- |
| `success_rate` / 严格成功 | 动作、值、引用全部正确数 / N | 1,437/1,440；≥85% |
| `answerable_success` / 可答成功 | A 中严格成功数 / A | 720/720；≥80% |
| `action_macro_f1` | answer、search、clarify、abstain 四类 F1 等权平均；各类使用 TP/FP/FN，并计入 invalid 导致的 FN | 0.99822677；≥0.80 |
| `all_variants_correct_rate` / 整组全通过 | 组内全部 24 情境严格正确的组数 / G | 58/60；≥50% |
| `over_answer_rate` / 不应作答却作答 | U 中输出合法 answer 数 / U | 3/720；≤5% |
| `format_valid_rate` / 格式有效 | 合法结构和字段输出数 / N | 1,440/1,440；100% |
| `unsupported_answer_rate` / 证据合同未通过的回答比例 | E 中不应作答、值错或引用集合错（含缺漏）的数量 / E | 3/723；≤5% |

七项必须同时通过。已答风险另外要求 E ≥ `ceil(0.8 × A)`：dev 至少 231 次、test 至少 576 次合法 answer。没有合法 answer 或分母未定义不能通过风险门槛；格式失败不能解释为安全。此保护防止仅用少数容易案例回答就宣称低风险。

`unsupported_answer_rate` 不等于事实幻觉率：即使数值正确，只漏一个必须引用的 official，也会失败。它的分母是 E；`over_answer_rate` 的分母是 U。本轮分别为 3/723（0.4149%）和 3/720（0.4167%），界面一位小数都可能显示 0.4%，应结合原始分数读取。

其余已实现诊断字段：

| 字段 | 定义 / 边界 |
| --- | --- |
| `action_accuracy` | 动作正确数 / N；不检查值与引用 |
| `answer_coverage` | E / N；说明模型实际回答多少 |
| `over_refusal_rate` | A 中输出 clarify 或 abstain 的数量 / A；不含 search |
| `unnecessary_search_rate` | A 中输出 search 的数量 / A；与上一项相加才覆盖合法输出中的多余延迟 |
| `citation_exact_rate` | E 中引用集合正确数 / E；不单独检查值，无 answer 时为 null |
| `by_variant` | 每个变体的数量与严格成功率；V3.1 test 每个变体 60 行 |
| `success_group_bootstrap_95ci` | 整个 group 为重采样单位的严格成功区间；不是训练种子不确定性 |

主单步摘要没有独立 `budget_violations` 列；闭环对零预算 search 报 `budget_violation`。不能把未实现的指标写成现有汇总字段。

## 配对检查不能替代完整组检查

所有 pair 要求两侧**动作、值和引用都正确**；仅动作发生变化、字符串保持一致或最终猜中值都不够。固定预算比较证据干预，固定证据比较预算干预。

| 主摘要 pair | V3.1 关系 | test 分母 |
| --- | --- | ---: |
| `value_change` | supported ↔ value_changed，分别匹配 b0、b1 | 120 |
| `distractor_invariance` | supported ↔ distractor，同预算 | 120 |
| `order_invariance` | supported ↔ reordered，同预算 | 120 |
| `evidence_removal` | supported ↔ missing，同预算 | 120 |
| `expiry_boundary` | supported ↔ expired，同预算 | 120 |
| `budget_exhaustion` | missing_b0 ↔ missing_b1 | 60 |
| `answer_budget_invariance` | supported_b0 ↔ supported_b1 | 60 |
| `clarify_budget_invariance` | ambiguous_b0 ↔ ambiguous_b1 | 60 |

V3.1 另用 [score_scope_pairs.py](../scripts/score_scope_pairs.py) 检查 `topic_filter`（conflict ↔ irrelevant_topic）和 `conflict_agreement`（conflict ↔ consensus），各自按两种预算匹配，test 各有 120 对。它们从原 raw 重评分，单独保存 [诊断结果](../experiments/v31/final-scope-pairs.json)，不改写已冻结主摘要或七门槛。

本轮上述 pair 全通过，仍有三个 future 情境错误；现有 pair 不是穷尽所有干预。`all_variants_correct_rate` 检查全部 24 情境，因此仍是 58/60。120 个相关配对也不能当作 120 个独立产品组。

## 历史协议与否定性诊断

| 版本 | 保留的实验边界 |
| --- | --- |
| V1 | 14 情境、20 test 组、280 行；两个 200 步 SFT。主摘要六种 pair 各为 20 对。完整报告见 [experiments/report.json](../experiments/report.json) |
| V2 | 仍为 14 情境、20 test 组；新数据修正、600 步预算，dev 在 200/400/600 中选点；原六门槛通过两项，最终 fail。见 [V2 协议](v2_protocol.md) 与 [审计](../experiments/v2/quality-gate.json) |
| V3 | 20 情境、每种语义有两个预算；train/dev 静态审计发现主题和候选数量捷径，在最终模型确认前被 V3.1 替代。原冻结与分析全部保留 |
| V3.1 | 24 情境、24 dev 组与 60 test 组；新实体与新 seed；七门槛及作答数量保护，先合格再排序选模 |

V1 的 `paraphrase.jsonl / redacted.jsonl` 各 56 行，来自原 test 前四组。它们是历史诊断，不能证明 V3.1 的措辞泛化。redacted 删除文档并重算目标，不沿用原 gold；它没有应回答样本，相关空分母指标保持 null。共享测试组的探针不能相加成更多独立样本，也不能直接沿用原变体名来解释干预效果。

V3.1 的主确认没有自动包含自然文档、措辞迁移或多轮压力测试。未实际运行并冻结的挑战保持 pending。若只提高格式合法性而不提高证据与配对正确性，应报告为格式适配；若减少误答却增加不必要拒答或检索，应同时披露代价。

## 检索闭环需要单独验收

[agent.py](../src/evidence_gate/agent.py) 提供 `search → 新证据 → 再决策`：词项重叠检索 catalogue，扣减预算，用检索结果替换 documents，再调用同一策略。没有向量检索、训练 reranker 或模型生成查询。

`completed` 只表示非 search 的合法终止。严格闭环成功要求每一步都符合当步可见证据和最终恢复目标；检索前猜中隐藏目录事实不算正确，终止率不能替代成功率。规则闭环分数和 V3.1 单步模型成绩均不能冒充 V3.1 的真实模型多步结果。

## CPU 复算、身份与不确定性

无需模型或 GPU，即可审计已保存的最终输出：

```bash
python scripts/selection_v3.py audit-test \
  --evaluation experiments/v31/confirmation/selected-v3.json \
  --data data/v31/test.jsonl \
  --protocol experiments/v31/protocol.json \
  --selection experiments/v31/selection.json \
  --output runs/v31-audit-replay.json
```

输出路径须尚不存在；退出 0 表示审计完成，质量结论查看 `status / passed / checks`。不要用仅处理 V2 六门槛的 `quality_gate.py` 替代它。`run_v3.py confirm` 属于真正推理流程，已确认的 test 哈希有使用记录，不应为了浏览或复算再次执行。

每份模型预测绑定底模快照与 adapter SHA；选择记录还绑定数据、配置和 checkpoint。训练图必须来自所选 checkpoint 自身的 history，同时披露本轮实际完成步数、总预算与总成本。不得把后续步骤 loss 归到早期选中权重；只匹配模型名称或数据集不能建立身份。

同组情境相关，bootstrap 必须整组重采样。单次训练种子的区间只描述测试组抽样变化，不代表训练随机性稳健性或跨自然文档泛化。保留训练日志、失败、中间候选、原始预测、哈希和所有审计；未运行结果不补零，失败结果不删除，看到 test 后的下一轮研究须建立新的确认边界。
