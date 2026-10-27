# Attribution and licenses

EvidenceGate's application, synthetic-data generator and evaluation code are independently authored and MIT licensed.
The fictional records are generated locally; no Xiaohongshu post, benchmark dataset, private support ticket or scraped
customer content is redistributed as training data.

The project was motivated by [月亮向北的 Agent 学习路线](https://xhslink.cn/o/2OxZZNWduw9), specifically its Agentic RAG
and evaluation directions. Its prose and five images were read, but are not copied into this repository. The post does
not itself present an SFT recipe. Unverified framework-version and QPS claims from the post are not adopted.

Design references, not copied implementations:

- [Self-RAG](https://github.com/AkariAsai/self-rag), MIT: adaptive retrieval and reflection.
- [RAFT](https://github.com/ShishirPatil/gorilla/tree/main/raft), Apache-2.0: retrieval-aware training with distractors.
- [R-Tuning](https://github.com/shizhediao/R-Tuning): refusal-aware learning; no source code or data copied.
- [CRAG](https://github.com/HuskyInSalt/CRAG): corrective retrieval decisions; no source code or data copied.
- [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft),
  [LlamaFactory](https://github.com/hiyouga/LlamaFactory), [Open-Instruct](https://github.com/allenai/open-instruct):
  assistant masking, parameter-efficient adaptation, explicit role formats, and data-isolation practice.

Runtime dependencies retain their own licenses, including PyTorch (BSD-style) and Hugging Face Transformers/PEFT
(Apache-2.0). Dependencies are installed from their normal package distributions, not vendored here.

The optional Qwen2.5-0.5B-Instruct base model is by the Qwen team and uses Apache-2.0:
[official model card and license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
The base weight is not redistributed. Published LoRA adapters are modifications produced by this project's local SFT,
not official Qwen releases. Adapter licensing, provenance and limitations are stated in their model cards, with the
upstream Apache-2.0 license included alongside them. Software MIT licensing does not replace model-weight terms.
