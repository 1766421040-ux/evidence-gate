# V3.1 complete training budget

This directory preserves the original summary, all 2400 update records, run manifest and final checkpoint metadata from the completed local MPS run. `artifact_manifest.json` hashes these four files. The selected checkpoint is also step 2400 in this run; its separately exported inference adapter and selected-prefix records are in `artifacts/evidence-v31-lora/` and `experiments/training/evidence-v31/`.

Training took 5568.608 seconds, excluding diagnostics, failed probes and evaluations. Optimizer/RNG tensor state and the base model are not published; the checkpoint metadata retains their original source hashes but is not a resumable optimizer package. Quality is established by the separate raw-generation confirmation, not by training loss.
