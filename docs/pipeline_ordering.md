# Pipeline Ordering and Feature Gating

This note documents *ordering constraints* between subsystems so we don’t
accidentally enable adaptive components before they have real signal.

The core pattern: **start with trusted truth labels / oracle signals**, then
train cheap approximators, then use those approximators to drive adaptive
policies (chunking, window refinement, query routing).

## Adaptive Embeddings (Cold Start Policy)

“Adaptive embeddings” in this repo refers to the **embedding-proxy + span-level
feedback** path used by adaptive chunking:

- Phase 1.25 fits a small head on `/v1/embeddings` to predict a trusted target
  score (currently `reference_score` in `[0, 1]`).
- During tree building, we can score *windows/spans* cheaply with that proxy and
  convert the result into `ChunkFeedbackSignal`s.

Important: **do not use embedding-based span feedback until the proxy head is
trained on trusted labels.** Untrained proxy scores behave like noise and will
push boundaries arbitrarily.

Current enforcement (see `src/training/run_pipeline.py`):

- Proxy training is Phase 1.25 (`train_embedding_proxy_from_phase1`).
- Span feedback is enabled only if Phase 1 results contain a saved proxy model
  artifact path (`proxy_model_artifact`). If no artifact exists, the run logs
  that span feedback is disabled and falls back to document-level feedback.

### What counts as “trusted” labels

Proxy training filters by `truth_label_source` (see `chunking.adaptive.embedding_proxy.allowed_truth_sources`
in `config/settings.yaml`).

Recommended meaning:
- `human`: manual labels
- `dataset`: trusted corpus labels
- `oracle`: trusted model-based labels (large judge / reward model / calibrated scorer)

If you don’t have any of these, **leave embedding proxy disabled** and rely on
fixed chunking + audits until you can collect them.

## Recommended Ordering (Training Pipeline)

For the main pipeline (`python -m src.training.run_pipeline`):

1. **Phase 1 (batched processing)**: build fixed-chunk trees and obtain trusted
   truth labels (`reference_score`) for each doc.
2. **Phase 1.25 (optional)**: train embedding proxy head on trusted doc labels
   and attach predictions + save `proxy_model_artifact`.
3. **Phase 1.5 (GenRM trees)**: build trees using adaptive chunking *only if*
   you have meaningful feedback signals:
   - document-level error proxies (needs Phase 1 scoring + reference labels), and/or
   - parser router hints, and/or
   - embedding span feedback (requires trained proxy artifact from Phase 1.25).
4. **Phase 2+ (optimization/auditing)**: only after the above is stable should
   you turn on more aggressive adaptivity (cross-fit folds, three-layer honesty,
   preference collection, etc.).

## Common Ordering Footguns

- **Adaptive chunking enabled, Phase 1 scoring disabled**:
  - doc-level error proxies cannot be computed (no `estimated_score`), so the
    chunker effectively has little/no signal (except parser hints).
- **Embedding proxy enabled without trusted labels**:
  - Phase 1.25 will skip due to `min_samples`/filters; span feedback stays off.
- **Embedding proxy enabled but embedding server down**:
  - Phase 1.25 fails; span feedback stays off. Set `EMBEDDING_URL` or start the
    embedding server (`./scripts/start_vllm.sh qwen3-embedding-8b --port 8003`).
- **Turning on span feedback without calibration**:
  - always treat proxy scores as *approximate*; prefer honest splits and (if
    possible) out-of-fold calibration for any reported evaluation claims.

## Similar “Order Matters” Constraints Elsewhere

- **Honest chunking**: boundary split should update policies; evaluation split
  should only report metrics (see `docs/adaptive_chunking_honesty.md`).
- **Three-layer honesty**: if you mix training/eval roles across chunker,
  summarizer, and oracle in the same fold, reported gains can be selection bias.
- **Cross-fitting**: don’t set `crossfit_folds > 1` unless you actually have
  enough documents per fold; otherwise you increase variance and instability.
- **GenRM tournaments**: only enable if the GenRM server is up and you’ve
  stabilized the summarizer prompts; tournament selection is expensive and can
  amplify noise in early iterations.

