# V2 低可回答成功率：训练与开发诊断

这是用户重新授权优化后的探索性诊断。它不改变已发布 V2 结果，不用于选择测试集表现，也没有启动新的训练。模型是 V2 开发集选中的第 600 步适配器，权重 SHA256 为 `5af71346aa50dc90b3f7cfc130127a94e3b97c83076c84ac6503b62fb5d4c9e5`。

原始记录：[v2-selected-64.json](v2-selected-64.json)。执行脚本：[diagnose_policy.py](../../../scripts/diagnose_policy.py)。

## 诊断设计

- 只读取 `data/v2/train.jsonl` 和 `data/v2/dev.jsonl`，未读取测试数据。
- 固定 seed 73，每个划分、每种真实动作抽 8 条，共 64 条。每种动作先轮流抽不同产品组，再考虑同组第二个变体；本次每种动作均覆盖 8 个不同产品组，train 共覆盖 28 组、dev 覆盖全部 12 组。所选 ID、源文件 SHA、种子均保存在原始记录中。
- 这是动作平衡的探索性样本，不等于原始任务分布，不能把其总体分数与完整测试分数直接比较。
- 自由生成先运行，仅接收经 `audit` 检查的公开 `messages`；使用 MPS/bfloat16、batch 4、greedy、最多 72 个新 tokens，无 JSON 修复。
- 随后单独进行 teacher-forced 前向：提供正确历史输出，计算各个监督 token 的条件负对数似然（NLL）。这些数值用于诊断，不替换模型原始输出。
- 使用真实 tokenizer offsets 定位 action、value、citations 字段；跨字段的 BPE token 归入与其重叠的内容字段，citation 统计包含列表括号。四个候选动作的首 token 概率仅用于分析，没有参与解码或重排。

## 结果：训练集也没有充分学会是否作答

| 指标 | train 探索样本 | dev 探索样本 |
| --- | ---: | ---: |
| 自由生成严格成功 | 26/32 | 25/32 |
| answer 严格成功 | 4/8 | 1/8 |
| search 严格成功 | 8/8 | 8/8 |
| clarify 严格成功 | 8/8 | 8/8 |
| abstain 严格成功 | 6/8 | 8/8 |
| 所有 completion token 平均 NLL | 0.023434 | 0.023752 |
| answer 首动作 token 平均 NLL | 0.738464 | 0.890913 |
| answer 首动作 token top-1 正确 | 3/8 | 1/8 |
| 正确前缀下 answer 值 token top-1 | 22/22 | 27/27 |
| 正确前缀下 answer 引用 token top-1 | 56/56 | 56/56 |

可回答样本中的错误主要发生在第一个动作词：例如开发样本 `g0085:supported` 的 `P(answer)=0.32052`、`P(search)=0.67854`。预算为 0 的多个可回答样本则在 answer 与 abstain 之间偏向 abstain。给定正确的 answer 前缀后，答案值与引用的条件 NLL 约为 `1e-4` 或更低。这说明当前首要瓶颈是证据充分性的动作判别；不能仅归因于字段复制失败，也不是只有开发集才出现的问题。

有 1 条训练样本的 teacher-forced 首动作与自由生成不同：`g0053:distractor`，其 `P(answer)` 与 `P(search)` 同为约 0.49918。完整序列前向与带 KV cache 的生成可能在低精度并列附近产生差异。这一条边界样本不能证明存在系统性缓存或位置对齐错误，更不能解释其余概率明显偏向错误动作的样本。

## 低平均 loss 不等于错误的 loss，也不等于梯度消失

本地安装版本的源码核查结果：

1. Transformers 4.57.6 的 `Qwen2ForCausalLM.forward` 将 `labels` 交给 `ForCausalLMLoss`；后者内部移动标签一位，忽略 `-100`，并将 logits 转为 FP32 后计算交叉熵。
2. 本项目屏蔽 prompt/padding，仅监督最终 assistant completion 和 EOS；梯度累积按各 micro-batch 的有效监督 token 数加权，没有重复除以梯度累积次数。
3. 逐例手算 shifted completion-only CE 与实际 `model.loss` 的最大绝对差为 **7.45e-9**。没有发现标签错位或 loss 计算错误。
4. PEFT 0.21.2 默认将低精度 LoRA 参数提升到 FP32，LoRA 结果最后转回底模输出精度；此次 bf16 底模和 FP32 adapter 的组合符合实现。

平均 loss 的分母包含格式、值、引用等容易预测的 completion token，所以单个困难的动作 token 可以被较多低损失 token 稀释。**输入 prompt 不在这个分母中。** 另一方面，首动作 token 仍贡献了本次剩余 NLL 的 **99.66%（train）与 99.68%（dev）**；不是格式 token 占据了大部分剩余误差。

NLL 占比不是梯度范数占比，本次没有计算按字段分解的梯度范数或梯度方向。[V2 最后 100 步训练日志](../../training/evidence-v2-full/metrics.json)中的平均整体梯度范数为 0.740，中位数 0.632，13% 的步数在裁剪前超过 1。仅给动作 loss 加权并没有引入新的判别信息；在残余误差已经集中于动作时，其作用可能主要表现为更新强度变化，但这一点仍需要对照验证，不能由 NLL 直接断言。

因此暂停动作加权改动，原有 `training.py` 和监督目标保持不变。4 项新增 CPU 单元测试验证了分组抽样、拒绝 test 输入、真实 tiny Qwen 的 loss 等价性，以及自由生成只收到公开消息。

## 下一步与本机代价边界

优先检验更广的可训练表示能力，同时保留同一 V2 数据、学习率轨迹和更新预算。当前 `q_proj/v_proj` 的 rank 8 LoRA 有 540,672 个参数；增加 `k_proj/o_proj/gate_proj/up_proj/down_proj` 后，按基础模型各层维度计算，rank 8 为 **4,399,104** 个参数。

仅计算 FP32 adapter 权重、梯度和两份 Adam 状态，理论存储约从 8.25 MiB 增至 67.13 MiB；这不包含激活、底模、缓存与框架开销，不能当作峰值显存。保持 bf16、batch 1、梯度累积 4、max_length 512、梯度检查点，先确认运行稳定；实际耗时由新运行记录，而不是按参数比例推断。

已登记的 [capacity-probe 开发计划](../development-plan.json)采用 all-7 modules、rank 8、**600 步**，其余设置与 V2 相同，便于隔离模块覆盖范围。未来如使用 1200 步，应在执行前另行登记：当前训练器的线性衰减依赖 `max_steps`，1200 步运行的前 600 步也具有不同的学习率轨迹，不能直接当作与原 600 步运行相同预算的纯容量对照。新数据修订和后续训练由开发计划分别记录，不在本诊断中宣称已经改善质量。
