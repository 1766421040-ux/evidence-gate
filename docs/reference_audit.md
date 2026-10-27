# 参考项目审查与复用边界

审查日期：2026-10-27。以下依据来自论文作者或官方仓库；未把社区文章的复述当作实现证据。链接指向审查时的公开页面，`main` 可能变化。这里没有宣称已复现上游论文结果。

## 用户提供的学习路线

选题起点是用户提供的 [小红书 Agent 学习路线笔记](https://xhslink.cn/o/2OxZZNWduw9)。主任务已阅读其五张图片。笔记用于确定 Agent、检索、工具与工程实践这一学习范围；**笔记本身未提供 SFT 方案**，本项目把用户另外提出的微调要求与该范围结合。图片中涉及的协议版本或吞吐指标没有在本项目中得到验证，因此不作为依赖版本、性能承诺或评测结果。

## 最近邻：哪些想法早已存在

| 项目 | 已核查的机制 | 对本项目的约束 | 实际借鉴 |
| --- | --- | --- | --- |
| [Self-RAG](https://github.com/AkariAsai/self-rag) / [论文](https://arxiv.org/abs/2310.11511) | 按需检索，生成过程中的反思标记，以及证据相关性和支持程度评价 | “让模型判断是否检索”本身不是本项目的新颖点 | 把检索决策作为可训练输出，并把证据支持与生成质量分开检查 |
| [RAFT / Gorilla](https://github.com/ShishirPatil/gorilla/tree/main/raft) / [论文](https://arxiv.org/abs/2403.10131) | 对问答附加相关和干扰文档，训练领域内的证据使用 | “RAG + SFT + 干扰文档”已有明确先例 | 设计同问题的相关 / 干扰上下文；答案引用和事实值单独评分 |
| [R-Tuning](https://github.com/shizhediao/R-Tuning) / [论文](https://arxiv.org/abs/2311.09677) | 根据模型知识边界构造拒答感知训练数据 | “训练模型说不知道”不是本项目的新颖点；其参数知识边界也不等同于我们的当前可见证据边界 | 单独报告误答与过度拒答，避免以拒答比例代替可靠性 |

以上工作已覆盖本方向的核心思想。本项目的差异是把**证据干预、预算边界、缺失用户条件和引用核验**放到一个可在个人电脑重放的小型实验中。这里的差异不构成对全部已有研究的首创性证明。

## 代码层面核查

### Self-RAG：训练目标与监督边界

已查看 [retrieval_lm/finetune.py](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/finetune.py) 中的 prompt/completion 编码、非 assistant 部分的标签屏蔽，以及 LoRA 参数入口。由此采用的实践是：明确监督哪些输出 token，并测试截断后仍有有效监督。EvidenceGate 不复制其训练器，不引入反思特殊词表，也不声称执行了 Self-RAG 的分段 beam search。

### RAFT：干扰文档与数据流水线

已查看 [raft.py](https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py) 的干扰数量、相关文档包含比例、问题生成与输出格式入口，以及仓库 README 的本地生成说明。EvidenceGate 借鉴“相关证据和干扰证据一起出现”的训练设计，但独立生成虚构事实与标签；不调用上游默认的付费生成接口，不复用其问答样本，不把上游的 CoT 文本当作本项目数据。

注意名称碰撞：检索“RAFT”也可能得到 `microsoft/RAFT` 等不同项目。本项目所说的 RAFT 是 Zhang、Patil 等人的 *Adapting Language Model to Domain Specific RAG*，对应上面的 Gorilla 子目录。

### R-Tuning：拒答真值的含义

已查看官方 README 的数据构造和训练说明。该工作关注模型参数中的已知与未知；EvidenceGate 则规定即使模型可能“知道”答案，也只能依当前证据作答。因此不能把本项目的 `abstain` 标签称作模型真实不确定性的测量，也不应直接比较两者的论文分数。

## 训练与评测工程：从源码学到了什么

下面四项是工程参考，不是本项目新增的研究机制，也不是已安装运行的四套框架。审查沿着具体函数和配置展开，不以仓库热度作为质量证据。

### Hugging Face TRL：区分 completion-only 与 assistant-only

[SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer) 区分 prompt-completion 与 conversational 数据；assistant-only 监督依赖 chat template 能返回 assistant 区间。[sft_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) 按 completion / assistant mask 将不参与监督的位置设为 `-100`。因此“输入用了 chat template”并不足以证明 loss 只作用于目标回答。

本项目在 [training.py](../src/evidence_gate/training.py) 中独立比较 prompt 与完整会话的 token 前缀，只监督最终 assistant completion 与 EOS，屏蔽 prompt 和 padding，超长样本直接报错。实际训练调用 Transformers / PEFT，**没有调用 TRL 的 `SFTTrainer`**，没有实现 TRL 的 packing 或其他训练目标。

### LlamaFactory：历史轮次、截断与 EOS 都会改变监督

[SupervisedDatasetProcessor](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/data/processor/supervised.py) 通过 `encode_multiturn` 编码；`train_on_prompt` 控制源文本标签，`mask_history` 可只监督最后一轮，并优先保留末轮；`infer_seqlen` 与 `efficient_eos` 分别处理长度分配和 EOS。这里可核查的是监督边界如何落实到 token，而不是配置名看起来是否合理。

本项目采用单轮决策，增加 prompt 前缀、有效监督和 EOS 审计；没有复用其多轮处理器、截断策略、packing 或 WebUI。拒绝静默截断是本项目的小任务选择，不能宣称比上游通用处理策略更先进。

### Open-Instruct：有效 batch、训练 token 与恢复状态分开记录

已查看 [open_instruct/finetune.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py)：有效 batch 由每设备 batch、并行规模和梯度累积共同决定；代码分别累计输入 token 与标签非 `-100` 的预测 token，并维护 checkpoint 恢复后的步数和数据位置。官方 [SFT 文档](https://allenai.github.io/open-instruct/algorithms/finetune/) 已将这个 Accelerate / DeepSpeed 入口标为 legacy，并另列 OLMo-core 路线；本项目没有把旧入口说成当前唯一推荐实现。

EvidenceGate 的小型训练循环独立记录输入 / 监督 token，以有效监督 token 数加权梯度累积，保存 optimizer、scheduler、随机数状态和采样游标。它没有调用 Open-Instruct、OLMo-core、DeepSpeed 或分布式启动器，也没有实现其 RLHF 流程。token 记账正是披露“两种 SFT 等步数但不等 token”的依据，不是新的优化方法。

### BFCL：能解析结构，不等于任务正确

Gorilla 的 [BFCL ast_checker.py](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py) 根据调用类别分派检查器；`simple_function_checker` 继续核查函数名、必需参数、意外参数、类型和值，并返回具体错误类型。只统计 JSON 可解析率会漏掉这些错误。

本项目在 [evaluation.py](../src/evidence_gate/evaluation.py) 中独立检查 JSON 协议、动作、事实值和引用集合，保留原始预测与失败明细。输出是有限 `action / value / citations`，**没有使用 BFCL 的 AST 检查器、数据集或评分脚本**，也未运行 BFCL benchmark。因此这里的严格成功率不是 BFCL 分数，不能据此声称通用函数调用能力。

## 复用清单

| 对象 | 本项目的处理 |
| --- | --- |
| Self-RAG、RAFT、R-Tuning 的研究机制 | 用于最近邻比较与任务设计；没有将上游结果计作本项目结果 |
| TRL、LlamaFactory、Open-Instruct、BFCL 的上述源码 | 作为实现审查参考；没有引入源码副本，也没有把这些项目作为运行依赖 |
| 上述项目的数据与已训练权重 | 不用于本项目合成语料或初始化；BFCL / RAFT 等上游 benchmark 未运行 |
| 基础语言模型 | 独立依赖；实际名称、revision、许可与下载来源必须在运行记录 / 模型说明中登记 |
| 实际训练依赖 | PyTorch / Transformers / PEFT；安装项另含 safetensors / accelerate，见 [pyproject.toml](../pyproject.toml) 与运行 manifest。PEFT 提供 LoRA，不伪装成自研底层训练引擎 |
| 数据生成器、动作协议、评测器、可视化 | 本项目实现；具体文件见仓库源码 |
| SaaS 实体、数值与文档 | 本项目合成，不代表任何真实服务的产品承诺 |

在复制任何新的第三方代码、样本或权重前，应分别核查其许可。仓库许可证、数据许可与模型许可不是一回事；仅能访问文件不代表可以再分发。当前审查仅支持上表明确列出的参考关系和实际依赖。

## 阅读转化为实现约束

1. 不只看 loss：保存 base 与 adapter 的逐条预测，分别审计动作、事实、引用和格式。
2. 不把“不会答”当作唯一安全动作：缺失用户条件需要澄清，可恢复的证据不足需要检索，预算耗尽才进入相应的停止路径。
3. 不在测试集上挑提示词、步数或 checkpoint：开发决策只使用 dev。
4. 不把同行结果迁移成自己的结果：论文模型、数据规模和硬件与本项目不同，所有本项目指标都必须实际运行。
5. 不用泛称“融合多个开源项目”掩盖边界：我们借鉴机制、依赖成熟框架，主要新增的是数据与评测闭环。

本项目的反对证据与停止条件见 [选题决策](decision.md)。

## V3.1 的后续实现边界

V1/V2 的否定结果保持原样。用户后来明确要求诊断并继续优化后，项目增加 train/dev 的 teacher-forcing 与自由生成诊断、七投影 LoRA 容量对照、24 单元新实体语料和更完整的 dev 选择门槛；这些不是上游论文的新算法复现，也没有据 V2 test 重选旧权重。低平均训练 loss 与 action 错误并存的诊断，最终没有转化为新的加权损失：V3.1 仍使用标准 completion 交叉熵。

V3.1 的实际贡献范围是本机可运行的任务构造、已知捷径反例、原始生成评分、证据绑定和可审计选择流程。数据、训练模块和预算一起改变，不足以单独归因某一种机制，也不构成首创性或生产可靠性的证明。七门槛与真实结果见 [V3.1 协议](v31_protocol.md)，历史失败和引用边界不因后续合成测试通过而删除。
