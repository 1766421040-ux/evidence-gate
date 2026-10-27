# V3.1 正式测试独立复算

本报告只读取唯一正式确认产生的原始输出，未运行模型、修改模型或重新选择 checkpoint。选中对象为 `crossed-r8-step2400`，由 dev 在测试前固定。逐条重算 1440 行的 raw、JSON 解析、动作/值/引用与汇总；**保存的 quality-gate 与独立复算完全相同**。正式联合门槛为 **pass**。

## 七项门槛

| 指标 | 复算值 | 预设门槛 | 判定 |
| --- | ---: | --- | --- |
| success_rate | 0.997917 | >= 0.85 | pass |
| answerable_success | 1.000000 | >= 0.8 | pass |
| action_macro_f1 | 0.998227 | >= 0.8 | pass |
| all_variants_correct_rate | 0.966667 | >= 0.5 | pass |
| over_answer_rate | 0.004167 | <= 0.05 | pass |
| format_valid_rate | 1.000000 | >= 1.0 | pass |
| unsupported_answer_rate | 0.004149 | <= 0.05 | pass |

分母：可答 720，不可答 720，实际已答 723；风险资格至少回答 576 行。整组统计共 60 组，2 组存在至少一个失败。总体成功的按组 bootstrap 95% 区间为 `[0.9944444444444445, 1.0]`，仅描述该合成组分布，不包含训练种子或真实场景不确定性。

## 24 个情境单元

每格为严格正确数/该单元总数；budget 两层分别报告。

| 情境 | budget 0 | budget 1 |
| --- | ---: | ---: |
| supported | 60/60 | 60/60 |
| value_changed | 60/60 | 60/60 |
| distractor | 60/60 | 60/60 |
| reordered | 60/60 | 60/60 |
| missing | 60/60 | 60/60 |
| expired | 60/60 | 60/60 |
| future | 59/60 | 58/60 |
| conflict | 60/60 | 60/60 |
| ambiguous | 60/60 | 60/60 |
| untrusted | 60/60 | 60/60 |
| irrelevant_topic | 60/60 | 60/60 |
| consensus | 60/60 | 60/60 |

## 四动作与新增配对

动作列只判断 action，完整成功仍要求 value 与引用集同时正确。

| 动作 | 正确动作/真实该动作 | 模型输出该动作数 | precision |
| --- | ---: | ---: | ---: |
| answer | 720/720 | 723 | 99.5851% |
| search | 298/300 | 298 | 100.0000% |
| clarify | 120/120 | 120 | 100.0000% |
| abstain | 299/300 | 299 | 100.0000% |

| 同预算配对 | 两侧严格全对 | budget 0 | budget 1 |
| --- | ---: | ---: | ---: |
| topic_filter | 120/120 | 60/60 | 60/60 |
| conflict_agreement | 120/120 | 60/60 | 60/60 |

配对仍是组内相关测量；[独立配对 raw 复算](final-scope-pairs.json)不改变主摘要或门槛。

## 失败与边界

错误计数：`{"wrong_action":3}`。正确 answer 动作下，错值 0 行、引用集错误 0 行；两者可能重叠，不应无条件相加。错误回答不可答行记为错动作。

- `v31-test-g0056:future_b1`：wrong_action；期望 `{"action":"search","citations":[],"value":""}`；实际 `{"action":"answer","citations":["doc-293"],"value":"180 days"}`。
- `v31-test-g0058:future_b0`：wrong_action；期望 `{"action":"abstain","citations":[],"value":""}`；实际 `{"action":"answer","citations":["doc-609"],"value":"21 days"}`。
- `v31-test-g0058:future_b1`：wrong_action；期望 `{"action":"search","citations":[],"value":""}`；实际 `{"action":"answer","citations":["doc-609"],"value":"21 days"}`。

三个错误都发生在 **as_of=2026-03-31、valid_from=2026-04-01**，模型提前一天使用了尚未生效的官方事实。对已有 future 输出做事后描述分层：同月未来一天为 **116/116**，跨月未来一天为 **1/4**，后者只有 **2 个相关产品组**。这个集中弱点必须与总体通过同时披露；不能把该小样本比例外推为通用日历错误率，也不能用这一已见测试分层重选 checkpoint 或修改本轮验收。没有新增或降低门槛。

已答合同错误率将错误值、应拒答/搜索却回答，以及错引/漏引都计入，不能直接改称事实幻觉率。结论只支持预先冻结的 V3.1 合成新实体协议；不证明通用检索、多轮 Agent、自然措辞或生产安全。已揭示的测试结果不能用于重选本轮候选。

证据：[逐条失败、完整混淆矩阵与来源 SHA](final-analysis.json)、[原始测试输出](confirmation/selected-v3.json)、[正式门槛](confirmation/quality-gate.json)、[测试前选择](selection.json)。
