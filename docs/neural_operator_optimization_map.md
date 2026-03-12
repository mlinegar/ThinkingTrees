# Neural Operator Optimization Map

This document maps neural-operator implementation choices to optimization objectives and audit-facing guarantees.

## Scope

Applies to:

- `src/tree/ctreepo_model.py`
- `src/training/ctreepo_trainer.py`
- `src/training/embedding_sketch.py`
- `scripts/train_ctreepo.py`
- `scripts/train_rile_embedding_sketch.py`
- `scripts/train_neural_operators.py`

## Operator Families

1. Approx-mergeable expressive operator (`CTreePO`)
- Merge variants: `gated`, `mlp`, `residual_gated`, `bilinear`, `avg`.
- Used when richer composition is preferred over strict mergeability.

2. Strictly mergeable operator (`MergeableEmbeddingSketch`)
- State is additive: `(sum_phi, count)`.
- Exact merge rule supports sharded and order-agnostic aggregation.

## Optimization Stack

### CTreePO

Primary losses:
- Root supervision (MSE on normalized target).
- Leaf local-law supervision / C1 span preservation (when oracle labels exist).
- Internal-node local-law supervision / C3 merge preservation (when oracle labels exist).

Structural regularizers:
- Consistency: parent prediction tracks weighted children prediction.
- Associativity penalty over random triplets.
- Cross-document contrastive penalty.

Training controls:
- Optimizer: `adam` or `adamw`.
- Scheduler: `none`, `cosine`, `linear`.
- Warmup + minimum LR floor.
- Gradient clipping.
- Early stopping based on validation MAE.

Evaluation diagnostics:
- Root MAE / MSE.
- Normalized MAE.
- 95% interval proxy coverage and width.
- Confidence calibration proxy error.
- Node-oracle label rate.
- Leaf/internal oracle MAE.
- Leaf/internal local-law violation rates.

Explicit local-law controls:
- `scripts/train_ctreepo.py` exposes `--root-weight`, `--leaf-audit-weight`, `--merge-audit-weight`, `--local-law-violation-threshold`.
- Preferred path: attach node-span supervision through `--local-law-oracle task` when the task/teacher setting already supplies a span oracle.
- Explicit callback path: `--local-law-oracle module.path:function_name`.
- Mechanical local-law supervision does not require an LM. The operator stays fully mechanical when the node-span oracle callback is mechanical.
- Model-backed node-span labeling is an explicit fallback teacher-labeling mode. The port-backed path now routes through the task's own local-law oracle interface rather than a hardcoded RILE scorer.
- The repo does not yet ship a built-in mechanical manifesto span oracle; manifesto runs can use the task-provided oracle/teacher path or an explicit callback.
- `--require-local-law-supervision` hard-fails training if positive local-law weights are requested but the training split has no attached node oracle labels.

### Mergeable Sketch

Primary losses:
- RILE regression head (sigmoid in `[0,1]`).
- Optional delta head in `[-1,1]`.

Optimization controls:
- Learned uncertainty-based multitask weighting or fixed weights.
- Gradient clipping.
- Optional semantic retrieval features.

## Shared Interface

`src/tree/neural_operator.py` defines:

- `OperatorPrediction` with point + interval + confidence.
- `CTreePOOperatorAdapter`.
- `MergeableSketchOperatorAdapter`.

This gives one contract for embedding/neural-operator scoring regardless of internal architecture.

## Pipeline Integration

The training pipeline (`src/training/run_pipeline.py`) includes Phase 1.3 neural-operator orchestration:

- `--train-neural-operators` runs `scripts/train_neural_operators.py`.
- Phase 1.3 now has first-class CTreePO local-law controls:
  - `--neural-operators-ctreepo-root-weight`
  - `--neural-operators-ctreepo-leaf-audit-weight`
  - `--neural-operators-ctreepo-merge-audit-weight`
  - `--neural-operators-ctreepo-local-law-oracle`
  - `--neural-operators-ctreepo-local-law-teacher-port` (fallback teacher-labeling path)
  - `--neural-operators-ctreepo-require-local-law-supervision`
- Artifacts are checkpointed in `checkpoints/phase1_3_neural_operators_complete.json`.
- When `--neural-operators-auto-wire-representation` is enabled, discovered model artifacts are auto-wired into:
  - `ctreepo_model_path`
  - `mergeable_sketch_model_path`
  - representation routing defaults (`llm,embedding,ctreepo,mergeable_sketch,ensemble`)
  - primary backend (`ensemble`)

This keeps one pipeline for LLM-only, embedding-only, and hybrid embedding+operator scoring.

## Hybrid Oracle-Seeded Ensemble

Representation routing supports a hybrid mode where LLM/oracle score is a seed and embedding/operator signals apply learned corrections:

- `hybrid_oracle_seeded_ensemble`
- `hybrid_seed_llm_min_weight`
- `hybrid_seed_llm_max_weight`
- `hybrid_operator_boost`

In hybrid mode, ensemble weights are adjusted per-document using available confidence/support diagnostics (e.g., CTreePO confidence, embedding neighbor support, mergeable window count).

## Theory Alignment Notes

1. Mergeability/approx-mergeability:
- Strict additive state supports exact mergeability.
- Proxy operators may use readout-aggregation or associativity regularizers, but those penalties are not local-law certificates.

2. Audit linkage:
- Confidence and interval proxies identify uncertain internal nodes for targeted audit.
- Local-law supervision directly enters training loss where labels are available.
- Mechanical local-law supervision and model-backed teacher labeling are reported separately in runtime summaries via the local-law label-source metadata.
- Positive local-law weights are not treated as "active" unless node oracle labels are actually attached; `require_local_law_supervision` converts that into a hard contract.

3. Risk reporting:
- Interval coverage and calibration diagnostics provide measurable uncertainty quality for downstream certification workflows.
