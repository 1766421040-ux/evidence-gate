# V2 完整训练预算记录

本目录原样保存一次完整的 600 步训练记录，耗时 1322.86 秒。设备为当前 M4 Pro / 24 GB，使用 MPS / bfloat16，没有使用云 GPU 或付费 API。

这里的 `summary.json` 对应训练终点权重，不自动代表开发集选中的权重。选择记录在 [selection.json](../../v2/selection.json)，选中权重自己的训练前缀与摘要在 [evidence-v2](../evidence-v2)，推理适配器在 [artifacts/evidence-v2-lora](../../../artifacts/evidence-v2-lora)。

原始 `artifact_manifest.json` 描述本地完整运行曾产生的文件，其中也含未在 GitHub 重复发布的 tokenizer 文件。实际发布范围以根目录的 [artifacts/manifest.json](../../../artifacts/manifest.json) 为准。优化器恢复状态保留在本地 `runs/v2-evidence/`；本仓库发布用于推理与审计的权重和记录，没有声称包含所有训练恢复文件。
