# V3.1 evidence index

This directory records the local, user-authorized continuation after V2 failed. It keeps planning evidence, development selection and final confirmation separate. The selected 2400-step model passed the seven frozen test gates: 1437/1440 strict, 720/720 answerable, 58/60 complete groups. Three failures remain at a cross-month future-validity boundary; that post-hoc subgroup is only 1/4 correct.

| Stage | Records |
| --- | --- |
| Data design and freeze | `supersession.json`, `protocol.json`, `freeze.json`, `acceptance-plan.md`, `train-dev-audit.json`, `tokenizer-preflight.json` |
| Registration | `training-plan.md`, `registry.json` |
| Every development candidate | `dev-step-000600.json`, `dev-step-001200.json`, `dev-step-001800.json`, `dev-step-002400.json` |
| Selection and independent audit | `selection.json`, `dev-independent-selection.json`, `dev-analysis.md`, `dev-analysis.json`, `dev-scope-step-*.json` |
| One final confirmation | `confirmation/confirmation-start.json`, `confirmation/confirmation.json`, `confirmation/selected-v3.json`, `confirmation/quality-gate.json` |
| Fixed references and replay | `confirmation/base.json`, `confirmation/v2-evidence.json`, `confirmation/contract.json`, `confirmation/report.json` |
| Independent final diagnosis | `final-analysis.md`, `final-analysis.json`, `final-scope-pairs.json` |
| Review snapshots | `quality-claim-review.md`, `reproduction-review.md`, `fallback-decision.md` |

The review snapshots describe what was known at their writing time; their pending/fallback statements are not final experiment outcomes. The fallback was not used. `selected-v3` is a retained method identifier, while this directory's data/protocol version is V3.1.

Raw text is never repaired. Main summary and quality gates are re-scored from every saved raw output. Supplemental scope pairs and date-boundary slices do not alter selection or introduce new test gates. The final holdout is now seen; any further development requires preserving this result and cannot relabel a rerun as new independent confirmation.

Selected inference files live in `artifacts/evidence-v31-lora/`; selected-prefix and full-budget logs are in `experiments/training/evidence-v31/` and `experiments/training/evidence-v31-full/`. Private optimizer state and base-model/cache files are not released. Use `PYTHONPATH=src python3 scripts/verify_v31_release.py --root .` from the repository root to audit the complete public evidence without loading a model.
