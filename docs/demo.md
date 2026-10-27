# 五分钟演示：证据变了，Agent 的决定是否跟着变

主入口是 V3.1 的真实实验回放。Qwen2.5-0.5B-Instruct 的 LoRA 在全新产品组上严格成功 **1,437/1,440（99.79%）**，本应回答的样本成功 **720/720**，完整 24 情境全通过 **58/60 组**，七项预设门槛全部通过。仍有三个未生效证据被提前采信的错误；这证明的是有限合成协议上的可训练性，不能外推为生产可靠性。

## 演示前准备

在仓库根目录完成 Python 3.10+ 环境和基础安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
evidence-gate serve --report experiments/v31/confirmation/report.json --port 8766
```

打开 `http://127.0.0.1:8766`。报告已包含选中模型、固定V2、base及完整规则四种方法。也可直接查看 [模型原始结果](../experiments/v31/confirmation/selected-v3.json) 与 [七门槛审计](../experiments/v31/confirmation/quality-gate.json)。静态回放无需下载底模、不调用模型。服务在启动时读取报告和页面资源快照；更换报告或页面后须重启服务。

界面中选择 `selected-v3`：这是运行器沿用的方法名，其数据和协议是 **V3.1**。同输入方法对照分别标注模型实测与规则基线，不能把规则分数算作 SFT 效果。

## 0:00–0:45：说清问题与边界

“管理员问某产品、套餐和日期下的保留天数。检索可能返回错误主题、过期或尚未生效的文档，也可能出现两份一致或冲突的官方说明。项目检查小模型能否根据当前可见证据，选择回答、继续检索、请求澄清或拒答。”

输出严格限定为 `action / value / citations`。模型按原始 prompt 自由生成 JSON，没有规则修补、付费教师或隐藏答案注入。它是有限证据决策任务，不是通用事实核验器或最优 Agent 规划器；选题和已有工作的比较见 [decision.md](decision.md)、[reference_audit.md](reference_audit.md)。

## 0:45–1:45：用同一问题展示 24 个情境

搜索 `v31-test-g0000`，沿同问题对比表切换以下场景。每个情境都有 `_b0 / _b1`，代表证据相同、查询预算分别为 0 / 1。

| 场景 | 应展示的变化 |
| --- | --- |
| `supported_b0 → value_changed_b0` | 问题不变，适用证据的值从 30 days 变为 45 days；两侧各自正确作答 |
| `supported_b0 → reordered_b0` | 仅改变文档顺序，答案和引用集合保持正确 |
| `missing_b0 → missing_b1` | 固定证据，预算从 0 变 1，动作从 abstain 变 search |
| `conflict_b0 → irrelevant_topic_b0` | 只把第二篇官方文档改为无关主题；其日期仍有效，应回答并只引用 `doc-820` |
| `conflict_b0 → consensus_b0` | 只把第二篇官方事实值改为一致，应回答 30 days，并同时引用 `doc-820`、`doc-831` |

查看期望结果、真实模型 JSON 和文档日期。双引用要求集合完整，顺序不影响得分；只给一个引用仍失败。配对分数要求两侧都严格正确，不能仅凭“输出变了”判成功。主题和一致性两个额外配对各通过 120/120，见 [独立配对诊断](../experiments/v31/final-scope-pairs.json)；它们不增加独立测试样本数，也不改写主摘要。

## 1:45–2:45：先讲分母，再讲分数

打开 [质量门槛审计](../experiments/v31/confirmation/quality-gate.json)，与界面指标卡对照：

| 指标 | 真实结果 | 预设门槛 |
| --- | ---: | ---: |
| 严格成功 | 1,437/1,440（99.79%） | ≥85% |
| 可答成功 | 720/720（100%） | ≥80% |
| 四动作 Macro-F1 | 0.99823 | ≥0.80 |
| 整组全部正确 | 58/60（96.67%） | ≥50% |
| 不应作答却作答 | 3/720（0.4167%） | ≤5% |
| 格式有效 | 1,440/1,440（100%） | 100% |
| 证据合同未通过的回答比例 | 3/723（0.4149%） | ≤5% |

最后两种风险分母不同：720 是本不应回答的样本数；723 是模型实际输出合法 answer 的次数。“证据合同未通过”包含不应回答时回答、答案值错误、错引或缺引，不能直接等同于事实幻觉。零个合法 answer 时风险不可估计，不能显示为安全。本轮还满足最低作答数量保护：723 ≥ `ceil(0.8 × 720) = 576`。

切到训练页，核对所选检查点、实际完成步数和总预算均为 2,400；曲线只属于所选检查点的训练前缀，不能把损失下降当作测试能力。正式推理统一 bfloat16、MPS、batch 4、greedy、72 个新 token、不修补输出；显示的批量均摊耗时不是单请求延迟或 p95。

## 2:45–3:45：打开三个真实失败

将结果筛选设为“未通过”，得到：

- `v31-test-g0056:future_b1`：应继续查询，却引用 `doc-293` 回答 180 days。
- `v31-test-g0058:future_b0`：应拒答，却引用 `doc-609` 回答 21 days。
- `v31-test-g0058:future_b1`：应继续查询，却引用同一未生效证据回答 21 days。

逐项比较 `as_of`、文档 `valid_from` 和预算。这三个输出语法均有效，错误是把尚未生效的证据当作当前依据。它们全部集中于3月31日到4月1日的跨月比较；这个事后描述子集共4行、仅1行正确，不能用总体高分掩盖。两组因此没有通过“组内全部 24 情境正确”，即使其他配对都是满分，也不能隐藏这些错误。查看此 test 后，它已成为已见数据；不能针对这三例继续调参，再把同一 test 称为新的独立确认。

## 3:45–4:30：说明为什么选它，而不是只挑最高总分

打开 [冻结协议](../experiments/v31/protocol.json)、[候选注册表](../experiments/v31/registry.json) 和 [dev 选择记录](../experiments/v31/selection.json)。候选在训练前登记为 600 / 1,200 / 1,800 / 2,400 步；先要求同一候选在 dev 上七门槛全过，再依次按可答成功、整组成功、总体成功、较低已答风险和登记顺序排序。

实际合格候选是 1,800 与 2,400 步，最终选择 2,400 步；该点 dev 为 576/576，但最终 test 为 1,437/1,440。dev 是选模证据，不能当作最终泛化成绩。没有合格候选时，流程会保存 `no_eligible_candidate` 并禁止模型读取 test。原始 V1/V2 失败和未用于确认的 V3 冻结记录均保留。

## 4:30–5:00：现场只用 CPU 复算

在已安装项目的终端执行，输出文件必须尚不存在：

```bash
python scripts/selection_v3.py audit-test \
  --evaluation experiments/v31/confirmation/selected-v3.json \
  --data data/v31/test.jsonl \
  --protocol experiments/v31/protocol.json \
  --selection experiments/v31/selection.json \
  --output runs/demo-v31-quality-gate.json
```

这个命令读取保存的 raw 输出，重新检查数据身份、完整案例、评分、摘要与七项门槛；不生成新预测。退出 0 表示审计完成，是否通过须看保存的 `status / passed / checks`。不要用 V2 的六门槛脚本替代 V3.1 审计。

“这次交付的是有冻结协议、真实 LoRA、选模约束、原始预测、失败定位和可复算审计的项目。新产品组上的有限合成任务已达预设标准；自然文档、措辞迁移、复杂多轮检索和生产可靠性仍需各自独立验证。”

## 可选：用 V1/V2 说明失败如何推动协议修订

另开终端，查看原始历史回放：

```bash
evidence-gate serve --report experiments/report.json --port 8767
# 或停止该服务后切换：
evidence-gate serve --report experiments/v2/report.json --port 8767
```

V1 evidence SFT 总体 200/280（71.43%），可答仅 20/80（25%），整组 0/20；answer-only 可答 80/80，却在 200 个不应回答的样本上全部作答。Base 的 280 个输出均格式失败，风险不可估计，不代表安全，也不代表通用知识为零。V2 虽修正部分表面线索，仍只有严格 204/280、可答 10/80、整组 0/20，六项门槛通过两项。保守输出不能冒充证据推理成功。

V3.1 进一步增加预算配对、主题反例与双官方一致性，使用新实体和新 seed 冻结留出集，同时扩大 LoRA 覆盖和训练预算；它不是等预算单因素消融，不能把提升全归因于某一项改动。版本边界和风险分母见 [评测协议](evaluation_protocol.md)。

如需补充“检索确实发生”的演示，可运行历史规则闭环：

```bash
evidence-gate replay --data data/main/test.jsonl --id g0092:missing_search --output runs/demo-rule-replay.json
```

明确标注它使用规则策略，不能作为 V3.1 模型多步能力证据。`completed` 只表示合法终止；严格闭环成功要求每一步符合当时可见证据，检索前猜中隐藏目录不能算成功。现场无需运行模型。

## 面试追问的代码入口

| 问题 | 证据入口 |
| --- | --- |
| 训练有没有看到目标之外的隐藏字段？ | [data.py](../src/evidence_gate/data.py) 的 `messages_for` 与数据审计 |
| 如何构造主题和双引用反例？ | [build_v31.py](../scripts/build_v31.py)、[独立数据审计](../scripts/audit_v31_data.py) |
| 微调实际训练了什么？ | [training.py](../src/evidence_gate/training.py) 的 completion mask、LoRA、token 加权 loss 和 checkpoint |
| 如何避免拒答刷分与 test 选模？ | [selection_v3.py](../scripts/selection_v3.py)、[run_v3.py](../scripts/run_v3.py) |
| 引用遗漏和错误答案如何算？ | [evaluation.py](../src/evidence_gate/evaluation.py) 的严格集合计分 |
| 主题和一致性配对是否改写总分？ | [score_scope_pairs.py](../scripts/score_scope_pairs.py)，独立诊断不修改主摘要 |
| 结果是否对应所选权重？ | [selected-training.json](../experiments/v31/confirmation/selected-training.json) 与预测的 adapter / checkpoint SHA |
