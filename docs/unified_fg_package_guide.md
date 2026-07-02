# Unified F/G Package Guide

## Design Intent

This package should behave like one general compositional learning system, not a collection of benchmark-specific stacks.

The desired shape is:

- Choose a `space`
- Choose how `g` is learned on that space
- Choose how `f` is learned on that space
- Reuse the same split source, bundle layout, launch surface, and preference export path

The public API should therefore be benchmark-agnostic:

- no `manifesto`, `rile`, or `markov` names in the primary bundle/runner surface
- task-specific logic lives behind adapters, datasets, and evaluation helpers
- the same collection config should be able to launch PyTorch operator runs, LLM runs, and TRL runs side by side

## Core Principles

- `unified_g` means one shared learned summary function across leaves and merges
- in the contextual-sufficiency lane, `g(leafInput(embed(x)))` is the learned
  sufficient carrier state and `f` is the downstream readout on that carrier
- ideally `f` and `g` share the same program contract, even when they use different learner families
- the package API is orthogonal:
  - `space_kind`
  - `g_learner_kind`
  - `f_learner_kind`
  - split source
  - optimization/export backend
- token-first partitioning is the default for text and token-sequence spaces
- preference/TRL flows should attach to the same data provenance and split provenance as PyTorch or LLM runs

## Contextual Sufficiency Literature Hooks

The modern references for this lane are indexed in
[`docs/literature/contextual_sufficiency/README.md`](/home/mlinegar/ThinkingTrees/docs/literature/contextual_sufficiency/README.md).
The short mapping is:

- Chen et al. 2021 NASS: learn neural approximate sufficient statistics by
  infomax; original code at `cyz-ai/neural-approx-ss-lfi`.
- Chen/Gutmann/Weller 2023 SSS/NASSS: learn sufficient statistics through
  low-dimensional random slices; available in `sbijax.NASSS`.
- Dirmeier/Albert/Perez-Cruz 2025 SSNL: learn a lower-dimensional state used
  directly by downstream likelihood inference; implemented through `sbijax`
  and `surjectors`.
- Hybrid Summary Statistics 2024: keep exact known sketches as diagnostics or
  hybrid probes rather than hard-coding them into the learned state.

The current Lean-backed slice bridge lives in
`FormalProofs.OPT.SlicedContextualSufficiency`: selected SSS/NASSS slices are
treated as deterministic probes of `R_K(x)`, and finite sliced preservation plus
slice cover implies ordinary contextual sufficiency. Random slice sampling,
MI bounds, PAC generalization, and SSNL likelihood layers remain outside this
package contract.

The canonical runtime package for learned contextual-sufficiency experiments is
`sbijax==0.3.6`, installed through:

```bash
pip install -e ".[contextual_sbi]"
```

Use `ctreepo sim run contextual-sbijax` or the standalone
`ctreepo-contextual-sbijax` entrypoint for package-facing runs. When this lane
is used as a leaf/item grid on the t128 Markov hazard panel, include
`--fragment-len` rungs `1, 2, 4, 8, 16, 32, 64`; the `1, 2, 4` rungs are part
of the acceptance surface, not just tiny debug cases.

The JAX package lane is:

```bash
python scripts/probe_contextual_sbijax.py \
  --training-objective contextual_sufficiency \
  --data-source markov \
  --sbijax-trainer package \
  --doc-tokens 128 \
  --leaf-tokens 128 \
  --sbijax-method nasss \
  --context-samples-per-doc 2 \
  --response-signature-contexts 8 \
  --response-signature-slices 4
```

`--sbijax-trainer package` is the default CLI path and calls
`sbijax.NASS.fit` / `sbijax.NASSS.fit` directly. It learns package summaries
from normalized token-id inputs, then trains a small Haiku readout for
contextual diagnostics. `--sbijax-trainer repo` remains available as the
repo-owned mirrored-loss comparison path.

The JAX lane records package provenance for `sbijax`, `jax`, `jaxlib`, and
`surjectors`. By default it consumes the same official Markov data family as
the PyTorch probe; pass `--data-source synthetic` for the tiny local generator
used in smoke tests. The clean PyTorch probe remains the comparison lane, with
contextual dependence objectives in
`scripts/probe_clean_unified_no.py`:

```bash
--training-objective contextual_sufficiency \
--contextual-dependence-objective regression|dcorr|jsd|dv|wasserstein|infonce|none \
--response-signature-contexts K \
--response-signature-slices M
```

`sbijax.simulators.tree` is a useful later package smoke because it is a
hierarchical latent-variable SBI benchmark, but it is separate from this
Markov milestone. The acceptance target here is direct `sbijax` training on
our official Markov contextual datasets.

Small side-by-side official-Markov smoke:

```bash
python scripts/probe_contextual_sbijax.py \
  --training-objective contextual_sufficiency \
  --data-source markov \
  --sbijax-trainer package \
  --doc-tokens 24 \
  --leaf-tokens 24 \
  --train-docs 4 \
  --eval-docs 2 \
  --fragment-len 6 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 3 \
  --response-signature-slices 2 \
  --embedding-dim 8 \
  --state-dim 4 \
  --hidden-dim 8 \
  --n-iter 2 \
  --batch-size 4

python scripts/probe_clean_unified_no.py \
  --doc-tokens 24 \
  --leaf-tokens 24 \
  --train-docs 4 \
  --eval-docs 2 \
  --epochs 0 \
  --batch-size 2 \
  --channels 8 \
  --g-n-modes 4 \
  --g-n-layers 1 \
  --scorer-n-modes 4 \
  --scorer-n-layers 1 \
  --device cpu \
  --training-objective contextual_sufficiency \
  --context-samples-per-doc 1 \
  --response-signature-contexts 3 \
  --response-signature-slices 2 \
  --contextual-dependence-objective regression \
  --infomax-loss-weight 0 \
  --diagnostic-baselines none
```

## The Three Canonical Cases

### 1. Token IDs + FNO/FNO

Use this when the working space is a token-id sequence and both `g` and `f` are neural operators.

- Space: `token_id_sequence`
- Learners: `fno` for `g`, `fno` for `f`
- Public program family: `token_id_sequence__fno__fno`
- Bundle approaches:
  - `token_fno_smoke`
  - `token_fno_report`

### 2. Embedding Sequence + FNO/FNO

Use this when the working space is an ordered embedding sequence and both `g` and `f` are neural operators.

- Space: `embedding_sequence`
- Learners: `fno` for `g`, `fno` for `f`
- Public program family: `embedding_sequence__fno__fno`
- Bundle approaches:
  - `embedding_sequence_smoke`
  - `embedding_fno_train`

### 3. Text + LLM/LLM

Use this when the working space is raw text/tokens and both `g` and `f` run on the LLM side.

- Space: `text`
- Learners: `llm` for `g`, `llm` for `f`
- Public program family: `text__llm__llm`
- Bundle approaches:
  - `text_audit`
  - `text_batch`
  - `text_llm_train`
  - `text_dspy_optimize`

## Preference / TRL Companion Surface

TRL is exposed as a generic preference surface rather than a benchmark-specific path.

- `preference_data`
  - stage split ids plus a canonical supervision dataset
  - export DPO / GRPO / reward-model / scalar-reward records
- `preference_optimize`
  - same exports
  - optional TRL training on top

This means the same split provenance can be attached to:

- text LLM training
- embedding operator training
- preference optimization

## Primary Code Locations

### Canonical Contracts

- [parallel/unified_g_v1/src/unified_g_v1/core/specs.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/specs.py)
- [parallel/unified_g_v1/src/unified_g_v1/core/program.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/program.py)
- [parallel/unified_g_v1/src/unified_g_v1/core/splits.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/splits.py)
- [parallel/unified_g_v1/src/unified_g_v1/core/supervision.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/supervision.py)

### Bundle / Launch Surface

- [parallel/unified_g_v1/src/unified_g_v1/bundles.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/bundles.py)
- [parallel/unified_g_v1/src/unified_g_v1/bundle_runner.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/bundle_runner.py)
- [parallel/unified_g_v1/scripts/run_unified_g_bundle.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/scripts/run_unified_g_bundle.py)

### Text + LLM/LLM

- [parallel/unified_g_v1/src/unified_g_v1/core/artifact.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/artifact.py)
- [src/training/run_pipeline.py](/home/mlinegar/ThinkingTrees/src/training/run_pipeline.py)
- [scripts/run_training_pipeline.sh](/home/mlinegar/ThinkingTrees/scripts/run_training_pipeline.sh)

### Embedding Sequence + FNO/FNO

- [parallel/unified_g_v1/src/unified_g_v1/core/tensor_program.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/core/tensor_program.py)
- [parallel/unified_g_v1/src/unified_g_v1/realdoc/embedding_fno_training.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/realdoc/embedding_fno_training.py)
- [parallel/unified_g_v1/scripts/run_manifesto_embedding_fno_training.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/scripts/run_manifesto_embedding_fno_training.py)

### Token Sequence + FNO/FNO

- [parallel/unified_g_v1/src/unified_g_v1/markov/program.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/markov/program.py)
- [parallel/unified_g_v1/src/unified_g_v1/markov/runner.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/markov/runner.py)
- [parallel/unified_g_v1/src/unified_g_v1/markov/report.py](/home/mlinegar/ThinkingTrees/parallel/unified_g_v1/src/unified_g_v1/markov/report.py)

## Example Usage

### Python

```python
from unified_g_v1.bundles import (
    run_text_llm_train_bundle,
    run_embedding_fno_train_bundle,
    run_preference_optimization_bundle,
    run_token_sequence_fno_smoke_bundle,
)

run_text_llm_train_bundle(
    "outputs/text_run",
    split_ids_path="outputs/splits/split_ids.json",
    execute=False,
)

run_embedding_fno_train_bundle(
    "outputs/embedding_run",
    split_ids_path="outputs/splits/split_ids.json",
    embedding_api_base="http://localhost:8006/v1",
    embedding_model="google/embeddinggemma-300m",
    execute=False,
)

run_token_sequence_fno_smoke_bundle(
    "outputs/token_run",
    train_docs=1024,
    reuse_existing=False,
)

run_preference_optimization_bundle(
    "outputs/preference_run",
    supervision_path="outputs/preference/supervision.json",
    split_ids_path="outputs/splits/split_ids.json",
    train_mode="none",
)
```

### Collection Config

```json
{
  "collection_name": "parallel_example",
  "collection_root": "outputs/parallel_example/runs",
  "shared_params": {
    "split_ids_path": "outputs/splits/split_ids.json"
  },
  "runs": [
    {
      "approach": "text_llm_train",
      "params": {
        "execute": true,
        "port": 8005,
        "max_chunk_tokens": 1024
      }
    },
    {
      "approach": "embedding_fno_train",
      "params": {
        "execute": true,
        "embedding_api_base": "http://localhost:8006/v1",
        "embedding_model": "google/embeddinggemma-300m"
      }
    },
    {
      "approach": "preference_optimize",
      "params": {
        "supervision_path": "outputs/preference/supervision.json",
        "train_mode": "none"
      }
    }
  ]
}
```

Launch it with:

```bash
./venv/bin/python parallel/unified_g_v1/scripts/run_unified_g_bundle.py launch \
  --config outputs/parallel_example/collection.json
```

## Compatibility Policy

Old benchmark-specific approach names are still accepted as aliases at parse time for migration, but:

- manifests emit generic approach names
- `list_bundle_approaches()` reports generic names
- new code should use the generic names only
