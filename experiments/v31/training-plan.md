# V3.1 first optimization round

Registered before this round's training or dev predictions. V3.1 replaces the unused V3 confirmation protocol following a train/dev-only coverage audit; see `supersession.json`. V1/V2 outcomes and the 20-cell V3 freeze remain preserved.

The controlled V2 capacity probe improved answer recall but increased unsafe answers, with almost unchanged overall dev success. Broader LoRA alone is insufficient. This round uses the corrected 24-cell data: both budgets for each state, crossed text cues, two agreeing official sources, a current irrelevant-topic source, and uniform wrong-topic negatives. The original prompt, strict three-field output, standard completion-only CE and base model remain unchanged.

- Train: `data/v31/train.jsonl`, 1,920 rows in 80 entity-isolated groups. No old dev/test rows are added to training.
- Configuration: `configs/v31.json`, all seven linear projections, rank 8, 2,400 optimizer steps, effective batch 4. This exposes each train row exactly five times under the existing shuffled-epoch sampler.
- Warmup: 60 steps, then the existing linear decay. The longer schedule is explicitly part of this combined data/budget intervention; this is not an equal-budget causal comparison with V2.
- Checkpoint candidates: 600, 1,200, 1,800, 2,400. Evaluate all on the frozen 576-row dev set using identical greedy bf16/MPS/batch-4 inference, max 72 new tokens and no output repair. Every dev gate must pass before a candidate is eligible.
- Complete the registered budget; preserve every checkpoint and raw dev result. If no candidate passes, do not run the 1,440-row test. Revisit training diagnostics using train/dev only.
- A batch-4/accumulation-1 throughput smoke took 45.10 seconds for 20 steps, close to batch-1/accumulation-4. Retain the latter for continuity and lower peak activation memory; do not claim the smoke's weights as a candidate or accuracy result.
- Local MPS only. No cloud training, paid teacher, external inference API, or hidden rule replacement. A passing result establishes only the declared finite synthetic protocol, not production reliability.

The machine-readable candidate registry binds actual data bytes, normalized training configuration, base snapshot and source files. Final confirmation must revalidate it against the original frozen data/protocol and all raw dev outputs.
