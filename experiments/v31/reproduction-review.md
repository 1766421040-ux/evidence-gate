# V3.1 复现与发布入口审查

本备忘录只核对源码、冻结协议与路径契约；没有启动模型、读取正在进行的正式确认结果或修改冻结文件。以下命令均在仓库根目录执行，要求 Python 3.10+。正式测试是否达标，应以完成后的确认记录和公开审计器输出为准。

## README 可采用的三个入口

**1. 复算公开证据（CPU，无模型下载、无训练依赖）**

```bash
PYTHONPATH=src python3 scripts/verify_v31_release.py --root .
```

此命令从保存的原始输出重算评分、七项门槛和开发集选择，并核验数据、冻结源码、模型/adapter 哈希、训练记录和确认收据。它不重新执行模型。只有 `status=verified` 且退出码 0 表示公开证据审计通过；缺少尚未完成的发布文件是 `pending`/退出码 2；记录不一致或质量未通过是退出码 1。已有完成收据却缺少所需文件属于失败，不会默认为通过。

**2. 查看已经保存的报告（CPU，无实时模型推理）**

```bash
PYTHONPATH=src python3 -m evidence_gate.cli serve \
  --report experiments/v31/confirmation/report.json --port 8766
```

打开 `http://127.0.0.1:8766`。报告必须先真实生成；不能把静态案例浏览描述为在线 Agent 推理服务。

**3. 可选：完整重做历史实验（本机 MPS）**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python scripts/download_model.py --output .cache/base-model
PYTHONPATH=src python scripts/run_v31_experiment.py \
  --model .cache/base-model --output runs/v31-reproduction
```

下载脚本固定 `Qwen/Qwen2.5-0.5B-Instruct` 的 revision `7ae557604adf67be50417f59c2c2f167def9a775`，并校验权重 SHA。下载需要网络；后续实验入口强制 Hugging Face 离线运行，不调用付费服务。完整入口固定 MPS、bfloat16、`configs/v31.json` 的 2400 步预算，以及开发集 600/1200/1800/2400 四个候选；推理固定 greedy、batch 4、最多 72 新 token、无输出修复。它不提供 CPU 训练替代、自动下载、续训、覆盖输出或绕过门槛的开关。

入口先检查 freeze、源码/config、数据和底模身份、V2 参考模型及本地确认账本，再创建全新输出目录。四个候选只有满足全部门槛及回答风险分母要求者才可进入选择；无候选合格则保存 `no_eligible_candidate` 和原始 dev 输出、退出码 2，绝不运行 test。合格后复用既有确认流程执行所选模型与固定 contract/base/V2 参考，最终成功退出码为 0，确认失败为 1。

## 两种目录契约不能混用

`runs/v31-reproduction/` 是独立复现记录，包含 `registry.json`、`training/`、四份 `dev-step-*.json`、`selection.json`、可能产生的 `confirmation/` 和 `outcome.json`；异常保留 `failure.json`。入口不导出公开 adapter，不替换仓库历史结果。**不要运行 `verify_v31_release.py --root runs/v31-reproduction`**：公开验证器的 root 是完整仓库，要求 `data/v31`、冻结源码、`artifacts/evidence-v31-lora`、`experiments/training/evidence-v31{,-full}` 和 `experiments/v31/confirmation` 等约定布局。

发布导出须保留选中 checkpoint 原始字节、选中训练前缀和完整 2400 步记录；完整训练目录还必须包含原始 `checkpoint-step-002400.json`。公开训练 manifest 使用公开文件映射，不能直接复制含私有运行文件的 manifest。新 adapter 权重/config 及相关训练记录还需加入全局 `artifacts/manifest.json`，否则既有 HFPolicy 的公开 adapter 加载检查会拒绝未登记文件。完成导出后才执行上面的公开审计。

## 必须随命令说明的限制

- 同一 holdout 的本地确认账本一旦存在，完整入口在训练前拒绝；当前工作机已启动正式确认，因此这条保护应生效。不能建议删除或更名账本来重跑。新机器上的历史协议复现同样不是新的独立盲测。
- 输出目录必须全新；复现 registry 的 ID/时间戳与历史记录不同。依赖配置是版本区间而非精确锁文件，实际版本见 `run_manifest.json`，不能承诺跨环境逐字节或指标完全相同。
- 项目采用标准 LoRA/SFT；结论限定在冻结的英文合成证据决策任务。通过这些门槛也不能直接证明真实企业文档、开放域问答、多轮 Agent 或新的语言分布上的效果。
- 公开验证器审计已发布记录及唯一公开确认收据，不证明所有未公开实验均不存在。README 的最终数值必须来自完成后的真实结果；dev 全对不能代替正式测试结论。
- 本次源码审查时 CI 仍调用原 `scripts/verify_artifacts.py`。发布前应确认新 V3.1 审计确已接入并实际通过，再写“V3.1 CI 验证通过”。
