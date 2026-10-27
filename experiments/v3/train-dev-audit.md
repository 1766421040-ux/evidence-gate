# V3 train/dev 覆盖审计

本次只读 `data/v3/train.jsonl`、`dev.jsonl` 和生成/评分源码；未读新 test 样本或任何模型预测，未修改冻结数据。完整交叉表、来源 SHA、训练集拟合的查表探针和四个 train-only 反例见 [train-dev-audit.json](train-dev-audit.json)。以下规则分数不是模型成绩。

**确认存在一个可修复的覆盖缺口：官方文档 topic 不一致仅出现在 missing 负例。** 一个不将文档 topic 与问题 topic 匹配、也不比较多条事实值的公开输入规则，严格成功达到 train 1,600/1,600、dev 480/480。这说明当前数据不足以检验 topic 范围判断与多官方一致性；不说明训练中的模型已经采用了该规则。

## 确认的缺口

| official 的 topic 是否混合 | train：answer / search / abstain / clarify | dev：answer / search / abstain / clarify |
| --- | --- | --- |
| 是 | 0 / 80 / 80 / 0 | 0 / 24 / 24 / 0 |
| 否 | 640 / 320 / 320 / 160 | 192 / 96 / 96 / 48 |

可行的捷径规则是：缺套餐则 clarify；按 **official + product + plan + 日期** 筛选，若候选不是恰好一个，或两个 official 的 topic 不同，就按预算 search/abstain；否则复制唯一候选的数字与 ID。它从未把 official 的 topic 与问题中的 topic 比较，也未检查两条事实的值是否一致，却在两份已见开发数据上严格全对。

这里混合了两类机制：product/plan、权威来源及相关证据的有效日期是任务必须使用的条件；“任意 official topic 不同就拒答”和“候选数不等于一就拒答”则不是合同要求。后者成立是因为缺少“另一个 topic 的无关 official 不影响回答”和“两篇适用 official 一致时都应引用”的正例。当前合同本来就支持这两种情况，不需要改变任务定义。

JSON 保存了从 `v3-train-g0000:supported_b0` 构造的反例：保留正确事实、只将无关 official 改为其他 topic 时应回答，但捷径拒答；两个 official 都改成同一个错误 topic 时应 abstain，但捷径回答；两篇适用当前 official 同值时应带两引用回答，但捷径拒答。它们是 CPU 规则反例，没有加入冻结集，也没有预测模型会如何表现。

## 其他检查没有形成同等强度的捷径证据

- **日期组合**：dev 的 VV、VE、VF 各 160 行，每种均为 answer 64、search 40、abstain 40、clarify 16。单用日期组合、来源排列或来源按数字 ID 排序的查表器，train 拟合后 dev 动作正确率都是 40%，等于多数类；加预算与是否缺套餐后为 60%，不高于相同语义的简单基线。dev 有 128/192 个可答样本同时含失效的无关 official，“看到任意失效文档就拒答”有正例反驳。需要判断**相关**证据的有效期，这种关联不应被强行消除。
- **ID / 位置 / 来源**：dev 的正确答案引用在数字 ID 第 1/2/3 位分别为 72/56/64 次，在输入第 1/2/3 位为 56/96/40 次，均非固定位置或固定 ID 排名。primary 始终为 official 是来源合同的构造选择；primary 身份来自同组 gold，只用于生成器审计，未作为查表探针的可见特征。24 个 dev group 仍可能产生有限样本偏斜，不能据这些统计宣称无任何位置捷径。
- **同组数字等价**：dev 两个 official 同值时为 answer 64、search 32、abstain 32、clarify 16；不同值时为 answer 128、search 88、abstain 88、clarify 32。无关 official 的值等于原值/干预值/两者都不等，在 dev 各 8 组，train 为 27/27/26 组；`value_changed` 保持无关 official 不变。因此旧“同值即 answer”已被反例打破。**适用范围内**两条证据的一致性仍是另一项能力，不能用上述无关数字平衡代替。

## 最小 V3.1 建议：只补四个单元

新增 `irrelevant_topic_b0 / irrelevant_topic_b1 / consensus_b0 / consensus_b1`，从同预算的 `conflict` 情景派生；原 20 单元保持，不再增加其他变体：

| 新情景 | 对 conflict 的唯一公开干预 | 正确输出 |
| --- | --- | --- |
| irrelevant_topic | 第二个当前 official 只改 topic，仍同产品/套餐、不同数值；社区、日期、顺序不动 | answer 原值，仅引用仍适用的 primary |
| consensus | 第二个当前 official 只把值改成 primary 的值，其他字段不动 | answer 同值，引用两份适用 official |

错误 topic 应在另外两个 topic 中按组平衡选择，避免永久固定“下一个 topic”的方向关系。恢复目录使用独立 ID，公开事实与 recovery target 的值和引用集合相互一致。保留所有原日期与预算反例；不追求让任务必需的语义条件与标签独立。

每组变为 24 单元：train 1,920 行，dev 576 行（可答 288 / 非可答 288），test 1,440 行（可答 720 / 非可答 720）；这是设计数量，未生成新数据。七项阈值不变，整组通过需全部 24 单元正确。主 test 的整数要求相应为严格成功至少 1,224/1,440、可答至少 576/720、整组至少 30/60、过度回答最多 36/720、格式全合法、合法 answer 至少 576，再同时满足 macro F1 与已答风险门槛。

`score` 和 `parse_prediction` **无需修改**：现有引用集合比较已支持两 ID。schema / 完整性审计须读取显式 24 单元名单；pair 报告新增同预算的 `conflict → irrelevant_topic`（topic 作用域）与 `conflict → consensus`（值一致性），每对两边都严格正确才算成功。新增类型预算 0/1 均必须严格正确，不能只检查动作相同。

若采纳，在任何 V3 模型推理前创建 V3.1 的新目录、新协议和新哈希，并说明它由 train/dev 静态覆盖审计触发；旧 V3 数据和冻结记录保留、标记被替代。不得把新设计写成已经验证有效，也不将旧 holdout 包装成已获独立模型确认。此次最小补充针对已证实的两条规则漏洞，仍不是全部语言与证据组合的完备证明。
