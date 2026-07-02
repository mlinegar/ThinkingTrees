# Contextual `sbijax` Walkthrough

Date: 2026-05-04

This note builds the practical bridge between `sbijax` and the C-TreePO
contextual-sufficiency / unified-`g` lane. It is intentionally API-facing:
what the package does, how the generic finite-context dataset is shaped, and
which words should mean the same thing when discussing the `sbijax` lane and
the existing PyTorch `CleanUnifiedNO` probe.

Primary repo anchors:

- `src/ctreepo/sim/core/contextual_sbijax.py`
- `scripts/probe_contextual_sbijax.py`
- `scripts/probe_clean_unified_no.py`
- `tests/ctreepo/test_contextual_sbijax.py`
- `docs/minimal_unified_gf_contract_2026-05-03.md`

External anchors:

- `sbijax` docs: <https://sbijax.readthedocs.io/>
- `sbijax` source: <https://github.com/dirmeier/sbijax>

## What `sbijax` Does

`sbijax` is a JAX package for simulation-based inference (SBI). In ordinary
SBI, the user has:

- a prior over parameters or latent variables;
- a simulator that generates observations from those variables;
- a neural estimator that learns a posterior, likelihood, or useful summary
  statistic from simulated pairs.

The relevant package surface for this repo is:

- `sbijax.NASS(model_fns, summary_net)`: learns neural approximate summary
  statistics with an information/dependence-style objective.
- `sbijax.NASSS(model_fns, summary_net)`: the sliced-summary variant. It
  learns summaries using random low-dimensional projections of the inferential
  target, which is closer to our finite contextual response slices.
- `sbijax.SNLE(model_fns, network)`: sequential neural likelihood estimation.
  This is not the first implementation target here, but it gives useful
  vocabulary for a downstream estimator trained on learned states.

The common package pattern is:

```python
from sbijax import NASS, NASSS, SNLE
from sbijax.nn import make_nass_net, make_nasss_net

model_fns = (prior_fn, simulator_fn)
network = make_nasss_net(
    embedding_dim=state_dim,
    sec_embedding_dim=1,
    hidden_sizes=[64, 64],
)
model = NASSS(model_fns, network)
params = model.fit(rng_key, data, n_iter=1000, batch_size=128)
```

At the package level, batches are shaped around `y` and `theta`:

- `y`: simulated observation.
- `theta`: parameter or inferential variable that should remain recoverable
  from a learned summary of `y`.
- `summary(y)`: low-dimensional learned statistic.
- `critic(summary(y), theta)`: dependence/infomax scoring path used by NASS.
- `phi`: random unit-sphere slice used by NASSS to ask whether `summary(y)`
  preserves low-dimensional projections such as `phi^T theta`.

## Generic Repo Pattern

The theorem-level primitive is not a left/span/right triple. It is a contextual
query:

```text
query : Ctx -> X -> Y
ResponseSignature(query)(x) = fun c => query(c, x)
R_K(x) = [query(c_i, x)]_{i=1..K}
```

The repo API mirrors this through `ContextualQueryProblem`. A problem adapter
knows how to sample items `x`, sample finite contexts `c_i`, evaluate
`query(c_i, x)`, and serialize context metadata. The generic dataset fields are:

- `item_tokens`: encoded items `x`;
- `context_payloads`: serializable descriptions of sampled contexts;
- `context_tensors`: tensorized context banks for adapters that can execute
  contexts through a model;
- `response_signatures`: finite `R_K(x)` rows, with shape
  `(n_items, n_contexts)` for scalar queries or
  `(n_items, n_contexts, target_dim)` for vector-valued queries.

An adapter may optionally expose `predict_contextual_response(...)`. If present,
the PyTorch probe can enact contexts through `g`/`f`; if absent, training still
uses response-signature preservation.

## Markov Adapter Example

The first concrete adapter is `MarkovTwoSidedContextProblem`. Here `Ctx` happens
to be `(left_fragment, right_fragment)`, and the query is:

```text
query((left, right), x) = exact_count(left + x + right) / target_scale
```

Tokens are partitioned into regimes, and the oracle target is the number of
adjacent regime changes.

```python
from src.ctreepo.sim.core.contextual_sbijax import (
    MarkovTwoSidedContext,
    MarkovTwoSidedContextProblem,
    build_contextual_query_dataset,
    make_synthetic_markov_docs,
    palette_block_map,
)

block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
docs = make_synthetic_markov_docs(
    n_docs=4,
    doc_tokens=24,
    vocab_size=8,
    n_regimes=2,
    expected_boundaries=3.0,
    seed=1,
)
problem = MarkovTwoSidedContextProblem(
    block_by_token=block_by_token,
    vocab_size=8,
    target_scale=32.0,
)

dataset = build_contextual_query_dataset(
    docs,
    problem=problem,
    samples_per_source=2,
    item_len=6,
    n_contexts=3,
    seed=2,
)
```

The important arrays are:

- `item_tokens`: sampled item fragments. With the test settings above this has
  shape `(8, 6)`: four docs times two samples per doc, each padded or cropped
  to six tokens.
- `context_payloads`: three serialized Markov two-sided contexts.
- `context_tensors["left_tokens"]` and `context_tensors["right_tokens"]`: fixed
  Markov left/right context banks. With `n_contexts=3`, each has shape `(3, 6)`.
- `response_signatures`: the empirical finite-context response signature
  `R_K(x) = [fstar(left_i + x + right_i)]_i / target_scale`. With eight sampled
  items and three contexts, this has shape `(8, 3)`.

Validation and test datasets should reuse the train context banks when the goal
is side-by-side comparison on the same finite context set:

```python
shared_contexts = tuple(
    MarkovTwoSidedContext(left_tokens=left, right_tokens=right)
    for left, right in zip(
        dataset.context_left_raw,
        dataset.context_right_raw,
        strict=True,
    )
)
val = build_contextual_query_dataset(
    docs,
    problem=problem,
    samples_per_source=1,
    item_len=6,
    n_contexts=3,
    seed=3,
    contexts=shared_contexts,
)
```

`build_contextual_response_dataset(...)` remains as a compatibility wrapper for
older Markov-specific callers and still exposes `span_tokens`,
`context_left_raw`, and `context_right_raw` aliases.

## Package Pattern

For practical translation, read the `sbijax` sufficient-statistic learner as:

```text
observation y -> summary_net(y) -> summary state
summary state + theta -> dependence or sliced prediction objective
```

`NASS` asks the learned summary to retain dependence with `theta`. In the
package implementation this is typically done with shuffled negatives through a
critic, so paired `(summary(y_i), theta_i)` should score differently from
unpaired `(summary(y_i), theta_j)`.

`NASSS` keeps the same summary-learning goal but lowers variance by sampling
random directions `phi` and training against scalar or low-dimensional slices
of `theta`. For this repo, that is the cleanest package-side analogy for
`responses @ slice_matrix`: each slice asks whether the learned state preserved
a particular low-dimensional view of the finite contextual response signature.

`SNLE` is useful future vocabulary rather than the current path. It suggests a
stronger downstream test: train a calibrated likelihood or surrogate directly
on learned states, not only a scalar root readout. The current implementation
does not add this flow/likelihood layer.

## Our Reinterpretation

C-TreePO does not use `sbijax` to infer a posterior over hidden simulator
parameters. The target object is a contextual response fiber:

```text
R_K(x) = [query(c_i, x)]_{i=1..K}
z_x = g(leafInput(embed(x)))
```

The goal is for `z_x` to preserve the downstream responses that matter under
sampled contexts. For the clean compositional PyTorch operator probe, the
Markov two-sided adapter can enact those contexts through:

```text
z_x = g(leafInput(embed(x)))
z_y = g(leafInput(embed(y)))
z_xy = g(mergeInput(z_x, z_y))
score = f(z_xy)
```

In the JAX contextual lane, `fit_contextual_sbijax(...)` learns a state map from
item tokens to `z_x`, plus explicit readouts that predict the finite response
signature and any NASS/NASSS-style auxiliary target. The Markov
`(count, first, last)` sketch remains a validation witness for sufficiency. It
is not the definition of the learned state.

## Shared Vocabulary

| `sbijax` term | C-TreePO contextual term | Meaning in this repo |
| --- | --- | --- |
| `y` | `item_tokens` | The observation-like input: a sampled item `x`. |
| `theta` | `response_signatures = R_K(x)` | The inferential target: finite contextual query responses. |
| `summary_net(y)` | `g_state = z_x` | Learned summary/state for the item. |
| NASS critic | contextual dependence between `z_x` and `R_K(x)` | Auxiliary pressure to avoid state collapse across response-distinct items. |
| NASSS slice `phi^T theta` | `responses @ slice_matrix` | Random low-dimensional response-signature target. |
| validation posterior quality | contextual MAE, MSE, correlation, collision diagnostics | How well learned states preserve finite contextual responses. |
| exact handcrafted summary | Markov `(count, first, last)` witness | Known sufficient sketch used for diagnostics, not hard-coded state slots. |
| `SNLE` downstream estimator | future state-space likelihood/readout test | Possible future check that learned states support richer inference. |

## Current Implementation

The default CLI path now calls `sbijax.NASS.fit` or `sbijax.NASSS.fit`
directly on official Markov contextual datasets:

```text
--sbijax-trainer package
```

In that path, `src/ctreepo/sim/core/contextual_sbijax.py`:

- lazy-imports `haiku`, `jax`, `optax`, and `sbijax`;
- verifies that `sbijax` exposes `NASS`, `NASSS`, and `SNLE`;
- builds generic finite-context response signatures through
  `ContextualQueryProblem`;
- passes package-shaped data as `{"y": encoded_span_tokens, "theta": R_K(x)}`;
- uses normalized dense token ids as the v1 package input encoding;
- trains the package summary model, then trains a small Haiku MLP readout
  `f(z_x) -> R_K(x)` for the same contextual diagnostics;
- records package provenance in every run;
- keeps the explicit contextual response readout so diagnostics remain
  comparable with `scripts/probe_clean_unified_no.py`.

The older mirrored-loss comparison path remains available:

```text
--sbijax-trainer repo
```

That path does not call package `fit`; it mirrors NASS/NASSS-style objectives
in a repo-owned Haiku/Optax loop so we can compare the package learner against
the exact same diagnostic surface.

The repo-owned Markov local-law lane is the current exact-zero diagnostic path:

```text
--sbijax-trainer learned_local_laws
```

That path trains/evaluates against the exact Markov sketch
`(count, first, last)`, uses the deterministic Markov contextual decoder, and
reports the theorem-facing local-law residuals:

- `eps_leaf` for leaf preservation (`L1_LEAF`, paper C1)
- `eps_merge` for merge preservation (`L2_MERGE`, paper C3)
- `eps_idemp` for on-range idempotence (`L3_IDEMPOTENCE`, paper C2)

For the leaf=1 exact-zero smoke, use the affine learned-state family:

```text
--local-law-summary-family affine_probe
```

This learns an affine map from `regime_one_hot` inputs to the sketch-shaped
state, then applies canonical count/endpoint projection before exact decoding.
It is a learned sufficient-statistic check, not a learned readout workaround.

Future adapters can use contexts that are not compositional left/right pairs:
masks, retrieval probes, prompt conditions, local-law probes, or
domain-specific query objects. Only adapters with an enacted model-side context
executor need to define how `g` and `f` should consume the context.

## Leaf=1 Exact-Zero Diagnostic

The current diagnostic artifact is:

- [outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.md](../outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.md)

On the t128 hazard-panel bundle with `leaf_tokens=1` and `fragment_len=1`:

| candidate | input | decoder | contextual raw MAE | theta MAE | raw count MAE | first/last acc | eps leaf |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_zero_markov` | `regime_one_hot` | exact | 0.0 | 0.0 | 0.0 | 1.0 / 1.0 | 0.0 |
| `identity_theta` | `markov_exact_sketch` | exact | 0.0 | 0.0 | 0.0 | 1.0 / 1.0 | 0.0 |
| `learned_local_laws_affine` | `regime_one_hot` | exact | 0.0 | 0.0 | 0.0 | 1.0 / 1.0 | 0.0 |
| `learned_local_laws_mlp` | `regime_one_hot` | exact | 1.49e-4 | 7.10e-5 | 1.01e-4 | 1.0 / 1.0 | 7.10e-5 |
| `package_nass` | `regime_one_hot` | learned | 1.69e-4 | 0.625 | 0.939 | 0.035 / 0.035 | 0.625 |
| `package_nasss` | `regime_one_hot` | learned | 4.96e-2 | 0.207 | 0.251 | 0.0 / 0.176 | 0.207 |

The main lesson is that contextual MAE is not enough. `package_nass` can have a
small contextual readout error while failing the sufficient-state checks. Exact
zero should be judged by `theta_mae`, raw count MAE, first/last accuracy,
decoder kind, and `eps_*` law metrics.

## Post-Resolution Ablations

After the exact-zero lane was resolved, the follow-up ablations tested three
questions: whether NASS/NASSS helps when added to the local-law objective,
whether learned merge/readout variants work inside the JAX lane, and whether a
standalone general f/g operator can discover the law without exact Markov
structure.

Canonical handoff:
[`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md).
Full table report:
[`outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md).

The result is precise:

- Low-weight NASSS is a useful auxiliary inside `learned_local_laws`.
- Learned merge and learned decoder variants work because the sufficient sketch
  is still supervised by the local laws.
- `CleanUnifiedNO` is the current honest general f/g test. Its best row
  (`contextual_sufficiency/dep_none/leaf_tokens_16`) reaches root MAE `1.1451`
  and contextual MAE `1.1187`, so general f/g remains an open bridge problem.

## How To Run The Smallest Example

Install the optional dependency group when needed:

```bash
pip install -e ".[contextual_sbi]"
```

The current optional group includes `sbijax==0.3.6` and repo-specific bounds
for transitive compatibility:

```toml
contextual_sbi = [
    "sbijax==0.3.6",
    "fastprogress>=1.0.0,<1.1.0",
    "starlette>=0.40.0,<0.51.0",
]
```

Activate the repo environment and run the unit smoke:

```bash
source venv/bin/activate
pytest -q tests/ctreepo/test_contextual_sbijax.py
```

Run the smallest official-Markov CLI smoke:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/probe_contextual_sbijax.py \
  --training-objective contextual_sufficiency \
  --data-source markov \
  --sbijax-trainer package \
  --sbijax-method nasss \
  --train-docs 4 \
  --eval-docs 2 \
  --doc-tokens 24 \
  --leaf-tokens 24 \
  --fragment-len 6 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 3 \
  --response-signature-slices 2 \
  --embedding-dim 8 \
  --state-dim 4 \
  --hidden-dim 8 \
  --n-iter 2 \
  --batch-size 4 \
  --seed 13 \
  --output-root /tmp/contextual_sbijax_smoke
```

Repeat with `--sbijax-method nass` for the NASS variant. Use
`--sbijax-trainer repo` for the mirrored objective comparison.

To run the same control against the paper-facing hazard-panel data generated by
`scripts/prepare_markov_hazard_panel_data.py`, load the saved
`MarkovOPSDataBundle` directly. The package-facing command is
`ctreepo sim run contextual-sbijax`; `ctreepo-contextual-sbijax` is also
available after reinstalling the editable package entrypoints with
`python -m pip install -e ".[contextual_sbi]"`.

For leaf/item grids on the t128 hazard panel, use the full ladder
`1, 2, 4, 8, 16, 32, 64`. The `1, 2, 4` rungs are deliberate small-leaf
checks: they test whether the package summary can preserve exact Markov
statistics when the item is almost atomic, and they should travel with the
larger composition rungs in paper-facing runs.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false ctreepo sim run contextual-sbijax \
  --training-objective contextual_sufficiency \
  --data-source markov \
  --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --sbijax-trainer theta_supervised \
  --sbijax-method nasss \
  --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding markov_exact_sketch \
  --train-docs 16 \
  --val-docs 8 \
  --test-docs 8 \
  --fragment-len 8 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 3 \
  --response-signature-slices 2 \
  --embedding-dim 8 \
  --state-dim 8 \
  --hidden-dim 16 \
  --n-iter 1 \
  --batch-size 8 \
  --seed 0 \
  --output-root /tmp/contextual_sbijax_hazard_panel_bundle_smoke
```

The equivalent direct repo command remains
`python scripts/probe_contextual_sbijax.py ...` for development sessions.

That path infers the mixed panel capacity from bundle metadata
(`vocab_size=48`, `n_regimes=12` for `paper_hazard_panel_v1_t128`), preserves
condition IDs/counts in `data_source_metadata`, and still reports
`diagnostics.markov_exact_sketch_oracle`. The exact-sketch oracle should remain
at zero contextual error; any learned-model error is then attributable to the
chosen package objective/network settings rather than to the panel data format.

For the old local synthetic generator, add `--data-source synthetic`; that mode
keeps `--vocab-size`, `--n-regimes`, `--val-docs`, and `--test-docs` as the
split controls.

If the installed `jaxlib` is CPU-only on a GPU machine, JAX may warn that it is
falling back to CPU. That is fine for this smoke.

The main output is `/tmp/contextual_sbijax_smoke/summary.json`:

- `provenance`: package/runtime facts such as `backend_package`, installed
  `sbijax`, `jax`, `jaxlib`, and `surjectors` versions, selected method, and
  response-signature dimensions. The package path also records
  `trainer=package`, `input_encoding=normalized_token_ids`, and
  `downstream_readout=haiku_mlp_mse`.
- `data_source_metadata`: official Markov loader facts such as `benchmark`,
  `doc_tokens`, `leaf_tokens`, split sizes, generator settings, and seed.
- `context_bank_metadata`: confirms that val/test reuse the fixed two-sided
  context bank sampled from train.
- `history`: per-iteration train/validation loss rows. `train_contextual_mse`
  and `val_contextual_mse` are finite-context response-prediction errors.
  `train_package_loss` and `val_package_loss` are the NASS/NASSS-style
  auxiliary losses.
- `diagnostics.train`, `diagnostics.val`, `diagnostics.test`: contextual MAE,
  MSE, prediction mean/std, truth mean/std, prediction-truth correlation, and
  collision rate.
- `diagnostics.exact_root_witness`: deterministic oracle sanity check for the
  Markov docs. It should report zero root error because it is the exact count
  witness, separate from the learned JAX state.
- `train_dataset`, `val_dataset`, `test_dataset`: row counts, fragment length,
  context count, and sampling metadata for each split.

## Version Check

The venv used for this walkthrough reports:

```text
sbijax 0.3.6
jax 0.8.1
surjectors 0.3.3
symbols ['NASS', 'NASSS', 'SNLE']
```

Those values are also recorded automatically in `summary.json` when the
optional runtime is installed.

## Deferred Package Tree Smoke

`sbijax.simulators.tree` is a hierarchical latent-variable SBI benchmark. It
is useful for a later package-alignment smoke because it has tree structure,
but it is not the official Markov contextual-sufficiency milestone here. The
current acceptance target is still our Markov changepoint process and the exact
`(count, first, last)` witness.
