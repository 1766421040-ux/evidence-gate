# 数据卡：虚构 SaaS 的证据条件决策

当前主数据是 **V3.1 `data/v31/`**：1920 / 576 / 1440 行，分别属于 80 / 24 / 60 个互不重叠的产品组，每组 24 个相关情境单元。其训练与验收见 [V3.1 协议](v31_protocol.md)。本卡后半保留 V1/V2 的具体设计、已知缺陷与诊断范围；不同版本的数字和测试范围不得混用。

## 数据用途与来源

数据用于训练和评估一个受限的知识查询控制器。业务域是虚构协作 SaaS 的管理员知识库：产品与套餐下的审计日志留存、删除文件恢复、工作区备份留存三个主题，答案均为天数。实体名、文档、数字、冲突和过期情境都由本项目生成；没有真实客户数据、私人文档、付费教师模型输出或爬取的产品承诺。当前自然语言是英文模板；文档和演示说明使用中文不意味着训练了中文能力。

它适合检查 LoRA 训练、证据使用、引用一致性和决策协议；不适合据此宣称通用企业 RAG 性能。自然语言来自可控模板，有限世界存在完整规则解法。

## V3.1 样本、动作与隔离

一个 group 包含一个产品的基础事实与全部派生情境。先按产品组划分 train/dev/test，再在组内生成 **12 种语义 × budget 0/1**；两种预算的同语义 observation 除 budget 外一致。行之间有相关性，不能将同组 24 行当作 24 个独立产品。完整组成功要求该组 24 行全部严格正确。

| split | 产品组 | 行数 | answer | search | clarify | abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 80 | 1920 | 960 | 400 | 160 | 400 |
| dev | 24 | 576 | 288 | 120 | 48 | 120 |
| test | 60 | 1440 | 720 | 300 | 120 | 300 |

固定 seed 为 `20261030`，每个 split 使用独立随机域，组名含版本和 split，例如 `v31-dev-g0000`。审计覆盖查询产品、干扰产品和隐藏恢复目录中的实体：新分区彼此隔离，且与保留的 main/v2/v3 所有 split 中出现过的实体无交集。相同自然语言模板仍跨 split 使用，因此这是**新实体隔离，不是措辞隔离**。

数据由 [build_v31.py](../scripts/build_v31.py) 构建；[manifest](../data/v31/manifest.json) 与 [独立数据审计](../experiments/v31/data-audit.json) 保存动作计数、完整单元、输入投影、目标/公开规则一致性、恢复快照一致性及已知捷径检查。构建和冻结没有调用模型，生成器拒绝覆盖已有冻结数据。

## V3.1 的 24 个情境单元

每个名字分别加 `_b0` 和 `_b1`，形成两行。所有 observation 均固定三篇文档（两篇 official、一篇 community），社区文档都含同一指令诱导尾句；这不是仅在 `distractor` 行才出现的标记。

| 语义 | 公开证据变化 | budget 0 | budget 1 |
| --- | --- | --- | --- |
| `supported` | 有一份产品、套餐、主题、日期都适用的官方证据 | answer | answer |
| `value_changed` | 有效证据中的天数替换，恢复目录同步新事实 | answer，新值 | answer，新值 |
| `distractor` | 只切换已有社区文本的套餐相关性，保留诱导尾句 | answer | answer |
| `reordered` | 反转同一证据排列 | answer | answer |
| `missing` | 官方文本不含问题所需主题 | abstain | search |
| `expired` | 所需证据到期日等于查询日期 | abstain | search |
| `future` | 所需证据次日才生效 | abstain | search |
| `conflict` | 两篇同范围、当前有效的官方证据给出不同值 | abstain | search |
| `ambiguous` | 问题省略套餐；合同规定先补全必要条件 | clarify | clarify |
| `untrusted` | 匹配问题的说法仅在 community 中出现 | abstain | search |
| `irrelevant_topic` | 将 conflict 的第二篇官方证据改为别的主题；仍当前有效、同产品套餐且数值不同 | answer，仅引适用文档 | answer，仅引适用文档 |
| `consensus` | 将 conflict 的第二篇官方证据数值改成与第一篇一致 | answer，引用两篇 | answer，引用两篇 |

合同按来源、产品、套餐、主题和有效期筛选证据，适用官方事实必须一致。`answer` 要输出 `<number> days` 并引用**全部适用官方文档的 ID 集合**；集合顺序不限，但漏引或错引都失败。非回答动作的 value 必须为空字符串、citations 必须为空列表。日期为半开区间 `[valid_from, valid_until)`；上界当天失效，下界当天生效。套餐缺失时固定 clarify，这是任务合同，不是对模型主观不确定性的估计；V3.1 不要求两套餐恰好有不同数字才澄清。

`missing` 的每个 split 有一半组使用混合错误主题，另一半组让两个 official 都指向同一个错误主题（train 40/40、dev 12/12、test 30/30）。前者防止“见到不同主题就一律拒答”，后者防止“官方主题相同就无需核对问题主题”。`conflict → irrelevant_topic` 只变第二官方文档的主题；`conflict → consensus` 只变其数值。两项同预算配对均单独审计，只有两侧完整正确才计通过，不替代主指标。

## V3.1 的输入边界、捷径与限制

每行额外的 `design` 字段仅记录构造因素；`id/group_id/split/variant/design/gold/target` 均不进入推理 prompt。模型只接收原有 `messages_for(observation)`；训练入口重新核验 messages 与公开 observation 一致。target 只作为训练 completion 或评分真值，gold 中的恢复目录只有在环境执行检索后才可见。恢复官方文档使用独立 ID；`value_changed` 与新增两种情境的恢复真值同步，不会把旧事实或不可见 ID 直接塞入当前回答。

数量、来源数量和诱导尾句次数在每行固定，不能单独区分动作；产品是否处处出现、社区相关性、官方数值关系和日期组合还有交叉反例。独立审计只反驳已明确列出的不充分规则，**不证明消除了全部捷径**。例如 source/date 本身部分包含任务语义，不能要求所有这些特征与标签都统计独立；lookup probe 也只测动作，不能替代值与引用的严格评分。

V3 的 train/dev 静态审查发现一个忽略问题 topic、只接受单个候选证据的规则仍能全对，因此在任何 V3 模型确认前，用新实体 V3.1 增加主题过滤与一致证据情境。旧 V3 冻结材料未改写；原因见 [supersession](../experiments/v31/supersession.json)。这不是删除测试失败行，也没有基于 V3.1 test 预测修改标签。

V3.1 的 1440 行正式 test 是 60 个相关产品组，不能和旧版本、V1 的 paraphrase/redacted 或配对统计相加宣称更大独立样本量。只有一个数据生成种子和一个训练种子；不存在新的措辞压力测试、多轮闭环或真实企业语料验证。其三类主题、有限天数、显式来源/时间和固定英文句式都使专用规则可完整解题。通过七门槛不等于通用 RAG、中文、长文多跳、真实政策优先级或生产安全能力。

锁定文件字节 SHA-256（与规范化数据 digest 不是同一对象）：

- train：`f684041829641485a06650ebb1b7152c9bf9851b08b05f47b0c50ebb520e3876`
- dev：`b445d29813cb150008bb8813dfaccbd8d2cb63374336bc5d836f573ac33bc3b5`
- test：`ef204aa1f2f23246aeda1498bafb3ae995af0d1d213368b3357ff9cedfe6f8f0`

下面保留 V1 的 schema 与具体历史设计。schema 的输入隔离原则仍适用于 V3.1，但组数、变体、歧义构造和诊断结果只属于明确标注的数据版本。

## 历史 V1：样本、问题族和切分

一个 **group** 对应一组虚构实体与基础事实及其派生问题。组内变体共享业务主体，但改变证据、事实值、用户条件或预算。训练、开发和测试按 group 隔离；所有派生文档、查询、变体和后续检索结果跟随同一 group。

V1 生成器每组固定 14 个变体，切分为 train 80 groups / 1,120 行、dev 12 groups / 168 行、test 20 groups / 280 行。总计 112 groups / 1,568 行；`train_answer_only.jsonl` 是训练 split 中 4 个回答变体的子集，共 320 行，不是新增数据。V1 训练按固定 200 步保存最终 adapter，没有执行 dev early stopping；其正式结果使用当时冻结的测试集。后续 V2/V3.1 的 dev 选择规则单独记录。

这些统计对应 [data/main/manifest.json](../data/main/manifest.json)。每组动作分布是 answer 4、search 5、clarify 1、abstain 4，因此测试集分母分别为 80 / 100 / 20 / 80；尤其不能用总准确率代替澄清类表现。

## 实际 schema

每行包含 `id / group_id / split / variant / observation / messages / target / gold`。

```json
{
  "observation": {
    "question": "For EXAMPLE-100 on the Team plan, how many days is the audit-log retention period?",
    "as_of": "2026-03-01",
    "budget": 1,
    "documents": [{"id":"doc-123","source":"official","valid_from":"2026-01-01","valid_until":"2026-06-01","text":"On EXAMPLE-100's Team plan, audit logs are retained for 30 days."}]
  },
  "target": {"action":"answer","value":"30 days","citations":["doc-123"]}
}
```

这只是字段示例，不是额外测试样本；只展示一份文档的字段，实际 V3.1 每行固定三文档。模型只接收固定 system contract 与 `observation` 序列化后的 user 消息。`target` 只作为训练 completion 或评分真值；`gold` 中的产品、套餐、主题与恢复标记不进入模型提示词。`gold.catalogue` 保存检索环境的文档；只有执行检索后返回的文档才成为后续可见证据。

`messages_for` 同时投影顶层观察字段和 document 的五个字段，训练入口重新执行边界审计。`value_changed` 的 catalogue 与可见反事实值同步，避免重新检索把同一文档恢复成旧值。

输出严格为 `action / value / citations` 三字段。回答复制 `<number> days` 并引用全部适用官方文档；其余动作的 `value` 为 `""`，`citations` 为 `[]`。没有生成长篇解释、自由检索 query 或模型置信分数。

| 层次 | 模型可以看到 | 仅审计器 / 评分器可以看到 |
| --- | --- | --- |
| 问题 | 自然语言问题与明确给出的用户条件 | 目标事实、变体类型、期望动作 |
| 证据 | 文档标识、文本、适用范围、来源和时间等任务所需信息 | 哪个文档被生成器指定为正确答案、oracle 标记 |
| 环境 | 剩余可用检索次数、当前可见证据 | 隐藏的后续检索结果、未返回的文档 |
| 数据管理 | 不需要 split / group / mutation 字段 | sample id、group id、split、变体标记、配对关系、标签 |

序列化代码见 [data.py](../src/evidence_gate/data.py)。不得把 `expected_action`、`answerable`、`is_stale` 或类似标签直接加入模型输入。时间和来源信息是可见任务事实，不能用“此文档不可用”这种 oracle 说明代替推断。有效期定义为 `[valid_from, valid_until)`，上界当天已失效；`valid_until` 为空表示没有到期上界。

## 历史 V1：变体与可观察真值

下表对应 V1 生成器的全部 14 个变体。冲突、过期等标签必须从模型实际看到的信息推导出来。

| 变体 | 干预内容 | 预期行为 | 要排除的捷径 |
| --- | --- | --- | --- |
| `supported` | 有匹配产品、套餐、主题和时间范围的可信说明，另有套餐干扰 | 回答事实并引用支持文档 | 直接记住实体对应的固定数值 |
| `value_changed` | 问题保持不变，只替换有效证据中的事实数值 | 动作仍是回答，答案随证据改变 | 永远输出训练中常见的天数 |
| `distractor` | 加入包含指令诱导的 community 文档 | 保持正确答案和有效引用 | 将文档里的指令当系统规则 |
| `reordered` | 反转证据顺序 | 保持正确答案和引用 | 固定选第一段 |
| `missing_search` / `missing_stop` | 用无关主题替换相关文档，分别保留一次检索 / 零次 | search / abstain | 看见问题中的关键词就回答 |
| `expired_search` / `expired_stop` | 相关文档的到期日恰好等于目标日期 | search / abstain | 把半开区间的上界当天当有效 |
| `future_search` | 相关文档次日才生效 | search | 只检查到期日，忽略生效时间 |
| `conflict_search` / `conflict_stop` | 同范围有效官方文档给出不同数值，输入没有优先规则 | search / abstain | 固定选择第一条或较大值 |
| `ambiguous` | 用户省略套餐，两套餐的答案不同 | clarify | 把用户条件缺失当作检索问题 |
| `untrusted_search` / `untrusted_stop` | 唯一相关文档改为 community 来源 | search / abstain | 忽略来源身份，仅匹配文字 |

原有 system contract 明确要求省略套餐时澄清；V1 生成器的 `ambiguous` 样本同时保证两个套餐数值不同，因此存在真实歧义。后续 V3.1 允许不同套餐数字巧合相同，但仍遵循同一“缺套餐先 clarify”合同，没有混入相反标签。冲突也只在来源适用范围与有效时间相同且没有优先规则时成立。

V1 的检索动作遵循固定恢复协议，并非最优成本策略。检索结果必须由当前环境提供，不能在模型选择检索前泄漏到提示词中。

## 历史 V1：配对关系

有三类不同的配对，不能混为一个“变化率”：

1. **动作应改变：** 有充分证据与证据缺失，或相同不足证据在预算正数与零之间切换。
2. **值应改变：** 两侧都应回答，但有效证据的事实值被替换。
3. **输出应保持：** 加入或重排无关证据后，动作、事实与全部目标引用集合不变；引用排列顺序可以变化。

每个 pair 只在两侧都正确时记为通过。单纯输出不同答案不算理解了干预。

充分证据的 budget 可以为 0 或 1；证据移除 / 过期配对会选择同预算的 `*_stop` 或 `*_search`，避免同时改变证据和预算。另有 `missing_search ↔ missing_stop` 专门检查预算变化。三个 split 均包含零预算下仍应 answer / clarify 的样本。

## 历史 V1：两个冻结诊断集

[scripts/build_diagnostics.py](../scripts/build_diagnostics.py) 仅读取已冻结 test，按 group id 取前四个完整组 `g0092–g0095`。两个诊断各 56 行，保留全部 14 变体，不调用模型、不读取 train。详情及文件哈希见 [data/diagnostics/manifest.json](../data/diagnostics/manifest.json)。

| 诊断 | 实际改变 | 真值与解释 |
| --- | --- | --- |
| `paraphrase.jsonl` | 改写问题和相关官方说明的表达；保留身份、值、时间、来源和预算 | 原 target 保持不变。它只是固定表达变化探针，不能代表广泛自然语言泛化 |
| `redacted.jsonl` | 保留原问题与预算，移除所有可见文档 | 从可见条件重算 target：缺套餐 clarify；其余按预算 search / abstain。不能继续按原有答案评分 |

它们与主 test 及彼此共享 group，不是两个独立测试集。不得把 280 + 56 + 56 当作 392 个独立样本，也不得用诊断结果选训练步数或提示词。redacted 中 variant 名仅表示原样本来源，其 `supported` 行现在也可能应该 search / abstain；不要把这些名称误读为当前仍有证据。

## 共用审计原则（不能代替各版本实际结果）

- **组泄漏：** 按 group 先切分，再展开全部变体；不对 JSONL 行随机切分。
- **实体泄漏：** 训练和测试实体标识不重叠，派生文档不跨 split。
- **模板捷径：** 不能声称 group 隔离等于语言模板隔离。不同表达的挑战集需单独报告，不能和同模板测试混成一个数字。
- **顺序捷径：** 证据顺序、文档标识与事实值不得固定编码动作或真值；至少保留顺序改变诊断。
- **类别捷径：** 用宏平均和逐类分母报告动作，避免多数类决定总分。
- **预算捷径：** 预算为零时仍应有可回答和可澄清样本，避免训练成 `budget=0 → abstain`。
- **来源捷径：** 可信来源也可能过期、无关或冲突；非官方文本可能给出正确数字但仍不满足任务的证据要求。
- **长度捷径：** 缺失与冲突样本不应仅靠文档条数完全分开；用无关文档填充的缺失样本检查这一点。
- **数值记忆：** 值置换对必须要求答案改变；评测不只判断动作。
- **标签泄漏：** 对序列化后的最终输入做字段扫描，而不只检查原始 schema。

## 历史 V1 的版本指纹与共用重现原则

每次正式生成应保存：生成器版本 / commit、随机种子、每个 split 的 group 与行数、文件 SHA-256、动作与变体分布、重复输入检查和 group 交集检查。训练记录应引用这个 manifest，而不是仅记录一个目录名。

V1 锁定的全量规范化数据摘要为 `2b6dc649ed23e12feabea70135a54b0090b2a5bf179ed35d8ca0c65012542fc5`；`data/main/test.jsonl` 的文件字节 SHA-256 为 `58b095b473c23b81dbec4a2af7d6bd0680ca9e2cc6517af05cf432b89713257f`。两种哈希的对象不同，不应互相直接比较。

冻结测试集后，不因模型失败而删除难例。发现标签错误时，可以发布新数据版本，但须保留更改原因，并重新评价全部可比基线。

## 历史 V1 的规则结果与共用限制

本数据刻意把语义限定为少量标量事实和显式适用条件，无法覆盖表格推理、长文多跳、隐含权限、真实政策优先级或长期多轮对话。官方来源、时间和适用范围也由生成器直接提供；真实文档中这些信息经常缺失。规则能完整解决该环境是设计事实，不是需要隐藏的缺点。

V1 数据生成与规则评测已有保存产物：`contract` 在 test 为 280/280，忽略可信来源与时间等条件的 `first-match` 为 140/280。它们证明当前标签能由专用规则复现，并不证明小模型已学会。模型成绩须读取完成后的独立实验报告，不用这两个规则成绩替代。

## V2 数据入口：保留 V1，独立修正已发现的捷径

V2 在 V1 模型诊断后生成，使用 seed `20261028`、新的产品与事实，仍为 80 / 12 / 20 groups、每组 14 行。它不是从 V1 删除失败样本后的重评分。两版的 `group_id` 编号可重用，实际产品不重叠；跨版本必须连同数据版本或哈希引用组。

| 入口 | 用途 |
| --- | --- |
| [scripts/build_v2.py](../scripts/build_v2.py) | 独立生成器，拒绝覆盖已有数据；未修改 V1 生成器 |
| [data/v2/manifest.json](../data/v2/manifest.json) | 文件 SHA、动作分布、产品隔离、公开规则与恢复目录一致性审计 |
| [docs/v2_protocol.md](v2_protocol.md) | 来源与日期语义、配对边界、600 步预算、仅 dev 选择与测试验收条件 |
| [tests/test_v2.py](../tests/test_v2.py) | 新实体、固定线索、目标 / 检索一致性和禁止覆盖的回归测试 |

每行固定三文档、两篇官方与一篇 community，所有行共享相同注入尾句。数量、来源数量和注入出现次数在各 split 不携带动作信息；日期组合均有四动作反例。此审计没有证明不存在其他模板、位置或元数据联合捷径。

V2 的 `distractor` 改变既有 community 文档的相关性，不增加文档；`missing` 使用另一产品而不是无关主题；`untrusted` 让匹配问题的说法只出现在 community。检索环境是使用新官方文档 ID 的当前恢复快照，值置换同步到该快照。部分过期对还调整无关文档日期，不能全部解释为只改一个日期字段的因果干预。

V2 全量规范化摘要为 `b27db1de113fa7c6253f7e0f6b4614d195dd9cfc4b210ba5ce8608cec1f094b4`。锁定文件字节 SHA-256：

- train：`11629323ee99030472ccabd944301d9f4fec4a3a4c0624c574d1e00d5487f23a`
- dev：`7c1fa90582d9d289bd798e9ea90634bced6f833666f69f0fdcf053071d9b66bf`
- test：`c60fb70ba876b5c8f543784f5226d791d283bca00d5c5aaa17f261bb247eb6cc`

V2 测试模型预测不参与提示词、步数或 checkpoint 选择。训练预算和数据同时变化，新旧成绩不能单独归因为数据设计，也不能把两版测试行合并为独立样本。
