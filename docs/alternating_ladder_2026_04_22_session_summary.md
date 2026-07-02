# Alternating f/g Ladder — Session Summary (2026-04-22)

This document summarizes all code changes made on 2026-04-22 to the manifesto
f/g alternating ladder: the size-based leaf restructure, per-family batching
improvements, and the DSPy / TRL warmstart + metric corrections.

## Context

Starting state of the session: the ladder trained f and g as **one-shot
parallel fits** on pre-computed teacher traces, with a count-based leaf axis
(`leaf_grid=1,2,4,8,16`), silent-truncation throughout the embedding and LM
paths, and no cross-family warmstart. Every rung could *weaken* the previous
iterate — the opposite of the ladder's design intent.

End state: the ladder is an **alternating optimization** on a **size-based
leaf axis** (tokens per leaf), with cross-family warmstart invariants,
no-truncation guards, and a corrected g-training metric that rewards fidelity
to ground truth rather than absolute f-score.

## Load-bearing invariants (saved to memory)

Every future change should respect these rules. See
`~/.claude/projects/-home-mlinegar-ThinkingTrees/memory/` for the full text.

1. **FNO channel invariant** — `feedback_fno_channel_invariant.md`. Every f/g
   FNO has `in_channels=1` (f) or `2` (g), `out_channels=1`. No
   `state_channels` knob.
2. **Symmetric backend families** —
   `feedback_fgfg_alternating_and_symmetric_backends.md`. Three pure families
   (DSPy, TRL, FNO). No cross-family (g, f) pairs. Also:
   "f is 1×, g is 2× on the leaf/token domain" — every backend must expose
   this 1×/2× arity contract.
3. **Alternating optimization for fgfg+** — same memory. Letters count
   iterations, g at iteration k is scored by the **current** student f_k
   (not teacher, not expert); measure f-vs-f\* gap every iteration.
4. **Never reset between rungs** — `feedback_never_reset_between_rungs.md`.
   `train_f(f_init=f_current)` must **use** f_current as the warmstart /
   student program / checkpoint. Never instantiate a fresh blank.
5. **No truncation; size-based leaves** —
   `feedback_no_truncation_size_based_leaves.md`. Leaves are sized in tokens
   (512 / 1024 / 2048). No silent truncation at any LM or embedding
   boundary. FNO: one embedding per leaf (spatial axis = embedding_dim); if
   leaf_tokens > embedding_max_length, concatenate chunks along spatial axis
   (`chunks_per_leaf > 1`).
6. **2×leaf concatenation budget** — `feedback_two_leaf_concat_budget.md`.
   LM families must hard-error at construction if
   `2 * leaf_size_tokens + max_completion_tokens + overhead >
   lm_context_window_tokens`.
7. **Warmstart via identity default** —
   `feedback_warmstart_is_default_via_identity.md`. Identity init is the
   baseline; fg(leaf_001) must reduce to f at t=0.

## Restructure 1: size-based leaves (tokens, not counts)

**What changed**: replaced `--leaf-grid N1,N2,...` (counts per document) with
`--leaf-size-tokens T1,T2,...` (exact tokens per leaf). Number of leaves per
document is derived: `ceil(doc_tokens / T)`. The legacy count-based path is
kept behind `--leaf-grid` for backward compatibility.

**Tokenizer**: Google EmbeddingGemma-300m's tokenizer (a Gemma-family
tokenizer, vocab 262144) is the canonical token-counter used everywhere a
budget is checked.

**Files**:
- [`src/preprocessing/leaf_size_utils.py`](src/preprocessing/leaf_size_utils.py) — new. Cached tokenizer
  singleton, `char_windows_from_token_budget(text, T)` using the tokenizer's
  `offset_mapping` to produce exact-token char spans, `count_tokens(text)`,
  `assert_no_truncation(text, max_tokens)`.
- [`src/ctreepo/distillation.py`](src/ctreepo/distillation.py) —
  `_build_binary_node_specs` and `build_labeled_tree_from_text` now accept
  `explicit_char_windows: Optional[List[Tuple[int, int]]]` for token-exact
  leaves. Topology policy records `"kind": "explicit_char_windows"`.
- [`scripts/run_manifesto_teacher_fg_leaf_grid.py`](scripts/run_manifesto_teacher_fg_leaf_grid.py) — new CLI
  flag `--leaf-size-tokens` (mutex with `--leaf-grid`). Per-size output dir
  `leaf{TTT}tok/`. Per-node caches keyed on `axis_tag` (`tok_T` vs
  `count_N`). Preserves `--leaf-grid` legacy path.
- [`scripts/run_alternating_ladder.py`](scripts/run_alternating_ladder.py) — matching flag
  `--leaf-size-tokens`; resolver `_load_leaf_size_trees` reads per-size
  teacher-trace dirs.

## Restructure 2: teacher batching (doc-level + level-parallel)

**What changed**: teacher trace generation was ~1 LM call/sec because (a)
only 4 docs ran in parallel and (b) within each doc, every node was summarized
sequentially. Now we have two batching axes:

1. **Doc-level**: `--num-workers 64` (up from 4) — 64 docs processed
   concurrently via `ThreadPoolExecutor`. No code change needed; just a
   flag bump.
2. **Level-parallel within each doc**: new `--lm-concurrency 32` flag. Inside
   `_build_teacher_labeled_tree`, the tree topology is computed first (no LM
   calls), then summaries are fired level-by-level — all level-0 leaves
   concurrently, then all level-1 merges, etc. Merges depend on child
   summaries so each level is a barrier, but within a level it's `min(32,
   N_nodes_at_level)` concurrent calls. Scores are all independent; one
   barrier round at the end.

**Files**:
- [`scripts/run_manifesto_teacher_fg_leaf_grid.py:_build_teacher_labeled_tree`](scripts/run_manifesto_teacher_fg_leaf_grid.py) —
  new level-parallel prewarm (lines ~500-570) that level-walks the node specs
  and batches `cached_summary` calls via `ThreadPoolExecutor`.
  Then the subsequent `build_labeled_tree_from_text` call hits cache for
  every node. Similarly `_resolve_score` is extracted so all scores fire in
  one `ThreadPoolExecutor` round after the tree is built.
- CLI: `--lm-concurrency 32` (default).

**Observed speedup**: on economic manifestos at leaf_size_tokens=512, what
would have been a ~15–20 hour teacher run finished in ~6 minutes of LM work
(plus cache replay of ~20 small docs from a prior run). Raw throughput jump
was ~60× (4 → ~240 concurrent active LM calls).

## Restructure 3: FNO channel invariant + no-truncation embedding

**What changed**: Fourier operator for f / g now has `in_channels=1` / `2`
and `out_channels=1` hardcoded. The old `state_channels` knob (default 8,
arbitrary) was removed. Identity-init makes k=0 a neutral baseline
(prediction = midpoint 4.0).

**Files**:
- [`src/ctreepo/embedding_fno.py`](src/ctreepo/embedding_fno.py) — `state_channels` removed from
  `EmbeddingFNOModelConfig` and `EmbeddingCoordinateFNOTreeRegressor.__init__`.
  `leaf_fno` is `FNO(in_channels=1, out_channels=1, ...)` and `merge_fno` is
  `FNO(in_channels=2, out_channels=1, ...)`, both hardcoded.
  `initialize_as_identity()` zeros spectral convs and sets layernorm to
  identity so `encode_leaves(x) ≡ x` and `merge(a, b) ≡ 0.5*(a+b)` at t=0.
- `_prepare_trees` — new `embedding_max_tokens` + `chunks_per_leaf` +
  `enforce_no_truncation` parameters. For each leaf: tokenize, assert fit in
  one embedding call, or split into `chunks_per_leaf` non-overlapping
  512-token chunks and concat their embeddings along the spatial axis
  (`D_eff = chunks_per_leaf × 768` for EmbeddingGemma). Raises
  `RuntimeError("silent truncation in _prepare_trees: ...")` on any
  overflow.
- [`src/ctreepo/fno_family.py:FNOFamily.__init__`](src/ctreepo/fno_family.py) — hard-errors if
  `leaf_size_tokens > embedding_max_length_tokens` (forces 1 embedding per
  leaf for MVP; chunk-within-leaf is supported by the model but the
  `FNOFamilyConfig.__post_init__` doesn't allow it yet).
- [`src/ctreepo/fno_family.py:_run_training`](src/ctreepo/fno_family.py) — per-tree loss grad guard:
  skip losses with no grad_fn (single-leaf trees during g-training). Fixes
  the `element 0 of tensors does not require grad and does not have a
  grad_fn` error that appeared when mixed batches contained 1-leaf trees.

## Restructure 4: EmbeddingGemma-300m as the embedding model

**What changed**: the FNO embedding backend was Qwen3-Embedding-0.6B (1024
dim) in fp16 — the fp16 path produced NaN outputs via plain `AutoModel`
(missing sentence-transformers pooling projection). Swapped to Google's
EmbeddingGemma-300m (768 dim, 2048 max_position_embeddings, same tokenizer
family as Gemma-4-31B-IT). Client default now fp32 on CUDA; fp16 only via
the `CTREEPO_EMBEDDING_FP16=1` env var.

**Files**:
- [`scripts/run_manifesto_dimension_fit_existing_results.py:LocalHFEmbeddingClient._load`](scripts/run_manifesto_dimension_fit_existing_results.py) —
  dtype defaults fp32 on CUDA; fp16 opt-in only.
- [`scripts/run_alternating_ladder.py`](scripts/run_alternating_ladder.py) — `--embedding-model`
  default changed to `/mnt/data/models/google/embeddinggemma-300m`.
- `src/ctreepo/fno_family.py:FNOFamilyConfig` — `embedding_max_length_tokens:
  int = 2048`, `effective_embedding_dim: Optional[int] = 768`.

## Restructure 5: symmetric backend families + alternating trampoline

**What changed** (earlier in session, before the 2026-04-22 fixes):
replaced arbitrary (g, f) cross-products with three pure families — **DSPy**,
**TRL**, **FNO**. Introduced the alternating trampoline with `f_star_gap`
(internal_f_pearson - external_expert_pearson) reporting every iteration.

**Files**:
- [`src/ctreepo/alternating.py`](src/ctreepo/alternating.py) — `FamilyRuntime` protocol,
  `stage_name_for_iteration(k)` (0→fg, 1→fgf, 2→fgfg, ...),
  `run_alternating_family`, `IterationRecord`, `SplitMetrics`,
  `evaluate_iteration` (emits internal + external Pearson + gap per split).
- [`src/ctreepo/fno_family.py`](src/ctreepo/fno_family.py),
  [`src/ctreepo/dspy_family.py`](src/ctreepo/dspy_family.py),
  [`src/ctreepo/trl_family.py`](src/ctreepo/trl_family.py) — one per family.
- [`scripts/run_alternating_ladder.py`](scripts/run_alternating_ladder.py) — new entry point, replaces
  the legacy `build_manifesto_fg_ladder_legacy.py`. CLI axis is
  `--families {dspy,trl,fno,all}` × `--leaf-size-tokens` ×
  `--max-iterations`. Writes `grid_summary.{json,md}`.

## Restructure 6 (today): DSPy warmstart + correct metric

Three issues were compounding on DSPy:

1. **Bare signature discards teacher rubric**: the k=1 student was a fresh
   `dspy.Predict(CTreePOFSignature)` whose entire instruction was
   `"""Predict the normalized scalar score for a C-TreePO node summary."""`.
   The teacher's actual scoring prompt carried full dimension rubric,
   1-7 scale description, expert framing — all thrown away.
2. **Raw-score g-metric**: the metric for g-training was
   `f_current(summary).normalized_score` directly, which is a "higher =
   better" reward. g learned to reward-hack into summaries that made f
   output near-7 regardless of what the doc actually said. Observed
   `mean_pred` drifted from 3.70 at k=0 to 6.85 at k=2.
3. **No warmstart across rungs**: `train_f` always compiled a fresh
   `Predict`, never the prior iterate. k=1 had 1.0 internal Pearson → 0.31
   → −0.10 at k=2 instead of monotonic strengthening.

**Fixes**:
1. **Load GEPA-v2 tuned `DimensionScorer` as f_init**: the existing
   artifacts at `outputs/phase1_gepa_v2_rank/<dim>/optimized_scorer.json`
   (5/6 dimensions) are loaded by default. The bare-scorer fallback is a
   `DimensionScorer` instance (which already carries dimension rubric +
   scoring context internally via its `_scoring_context`).
2. **g-metric rewards target agreement, not raw f-score**:
   `reward = 1 - |f(candidate) - target| / scale`, where `target` is the
   ground-truth score for the node (from `target_score_raw` on the record's
   metadata). This rewards fidelity — g should produce summaries that let
   f recover the **known** score — not summaries f rates high.
3. **Warmstart from prior program**: `train_f` now passes `f_current` (the
   loaded `DimensionScorer`) as the `program` arg to
   `optimizer.compile(program=..., trainset=...)`. Same for `train_g`.
4. **Dimension rubric injected into g signature**:
   `CTreePOGSignature.__doc__` is populated with
   `get_preservation_rubric(dim) + get_scoring_context(dim)` so the bare
   Predict starts with teacher-grade task instructions.
5. **`_apply_f_normalized` signature updated**: accepts `summary` as the
   sole semantic input (the `prompt` kwarg is kept for backward compat but
   ignored). Calls `DimensionScorer.forward(summary=...)`, gets a 1-7 raw
   score, linear-rescales to [0, 1]. This is the fix for the
   10977-input-token context blowout where g's long "Summarize..." prompt
   was being passed to f.

**Files**:
- [`src/ctreepo/dspy_family.py`](src/ctreepo/dspy_family.py):
  - `DSPyFamilyConfig` — new fields `dimension`, `f_init_path`.
  - `_default_f_init_path()` — resolver for per-dimension GEPA-v2 artifact.
  - `_new_dimension_scorer()` — instantiate `DimensionScorer` for the dim.
  - `_load_f_program()` — loads `DimensionScorer` with GEPA-v2 state by
    default, or a bare `DimensionScorer` as fallback.
  - `_apply_f_normalized()` — calls `DimensionScorer.forward(summary=...)`;
    normalizes 1-7 to [0, 1].
  - `_g_signature()` — dimension rubric in docstring.
  - `train_f()` — warmstart path, correct metric.
  - `train_g()` — warmstart path, agreement-with-target metric (not raw
    f-score).

## Restructure 7 (today): TRL `--init-checkpoint` warmstart

**What changed**: TRL family's `train_f` and `train_g` were raising
`NotImplementedError` at k≥1. They now dispatch a subprocess to
`scripts/distill_ctreepo_students.py` with `--init-checkpoint <prior_dir>`,
which routes through the existing TRL SFT / scalar-regression paths using
the prior iteration's HF model directory as the warmstart.

**MVP limitation**: TRL train_g uses `--run-g-sft` (teacher-supervised SFT),
not true GRPO-with-f-as-reward. That's the followup. SFT still honors the
"never reset between rungs" rule by resuming from `g_init` when provided.

**Files**:
- [`scripts/distill_ctreepo_students.py`](scripts/distill_ctreepo_students.py):
  - New CLI flag `--init-checkpoint` (path to prior HF model dir).
  - `effective_g_model = args.init_checkpoint or args.g_model_name` passed
    to `GLMConfig.model_name`.
  - Same treatment for `effective_f_model` → `FLMConfig.model_name`.
- [`src/ctreepo/trl_family.py`](src/ctreepo/trl_family.py):
  - `_init_checkpoint(artifact, base_model)` — resolves passthrough /
    identity / None to base model; otherwise to the prior iteration's dir.
  - `_traces_artifact_path(traces)` — locates the labeled_trees.jsonl the
    subprocess needs to consume.
  - `train_f` — subprocess call to
    `distill_ctreepo_students.py --run-f-lm-regression
    --init-checkpoint=<f_init>`.
  - `train_g` — subprocess call to `--run-g-sft --init-checkpoint=<g_init>`
    (SFT; GRPO is the followup).
  - `score_roots_with_f` — unchanged; still passthrough for k=0. Real HF
    inference for k≥1 is a followup (requires a GPU the vLLM server
    doesn't occupy).

## Ladder entry point + CLI changes

**Files**:
- [`scripts/run_alternating_ladder.py`](scripts/run_alternating_ladder.py):
  - `--leaf-size-tokens` replaces count-based axis by default.
  - `--embedding-model` default: `/mnt/data/models/google/embeddinggemma-300m`.
  - `--embedding-max-length 2048` (matches EmbeddingGemma's native max).
  - `--dspy-max-tokens 1024` (2× leaf_size_tokens default).
  - `--dspy-lm-context-tokens`, `--dspy-prompt-overhead-tokens` feed the
    2×leaf config-time budget check.
  - Wires `dimension`, `tokenizer_model_path`, `embedding_max_length_tokens`
    through to each family's config.

## Summary of family behaviors after today's fixes

| family | f at k=0 | f at k≥1 | g at k=0 | g at k≥1 | Warmstart? |
|--------|----------|----------|----------|----------|------------|
| **DSPy** | GEPA-v2 `DimensionScorer` (tuned) | MIPRO/Bootstrap on the loaded scorer, target-agreement metric | passthrough (teacher summary from tree) | MIPRO/Bootstrap on loaded g_current, `1 − \|f(cand) − target\|` metric, rubric in signature | ✓ on both |
| **TRL** | teacher passthrough | subprocess to `distill_ctreepo_students.py --run-f-lm-regression --init-checkpoint=<prior>` | teacher passthrough | subprocess to `--run-g-sft --init-checkpoint=<prior>` (GRPO is followup) | ✓ on both |
| **FNO** | identity init (pred ≡ 4.0) | AdamW on `leaf_fno + score_head`, `merge_fno` frozen | identity init | AdamW on `merge_fno` only, `leaf_fno + score_head` frozen | ✓ via state_dict load |

## Known followups

- **DSPy**: MIPRO by default (instead of Bootstrap) is gated on LM speed.
  Currently default is `bootstrap` for smoke; switch to `mipro` + `--dspy-budget medium` for the real headline numbers.
- **DSPy g-side upgrade**: replace bare `Predict(CTreePOGSignature)` with
  `LeafSummarizer` / `MergeSummarizer` from
  `src/tasks/manifesto/summarizer.py` so g has the full rubric-aware
  signature instead of a generic "completion" field.
- **DSPy seed demos**: load Benoit masked summaries + expert ensemble means
  as labeled demos for the Predict student (`(raw_text, Benoit_summary)`
  for g; `(Benoit_summary, expert_score_1_7)` for f). Expected to lift k=1
  from the bare-student-distilling-teacher regime into the
  seeded-student-refining regime.
- **TRL GRPO**: add `--run-g-grpo` to `distill_ctreepo_students.py` that
  calls `src/training/trl_training.py:train_grpo` with `reward_funcs=[f_eval]`
  where `f_eval` loads the current-f HF regression model and returns per-
  sample scalar rewards. This is the real "train g against current f"
  alternation signal for TRL.
- **TRL inference**: `score_roots_with_f` for k≥1 needs to load HF models
  (g: causal LM, f: sequence-classification). Requires a GPU not occupied
  by the vLLM server — either a secondary device or pausing vLLM during
  the inference pass.
- **Teacher no-truncation budgets for longest manifestos**: 9/139 docs
  failed with `no-truncation guard: resummary input has 5150 chars but
  max_chars=5000`. Bump `--resummary-max-chars 7000 --node-summary-max-chars
  14000 --score-max-chars 8000` to recover those docs. Or scale these
  budgets with `leaf_size_tokens` instead of hardcoding.
- **Fan out to other 5 dimensions**: with teacher batching
  (`--num-workers 64 --lm-concurrency 32`), each dimension's teacher run
  should finish in ~10–20 min. Followed by a ladder run per dimension.

## Files touched in this session

```
src/ctreepo/alternating.py                       # trampoline, FamilyRuntime, metrics
src/ctreepo/distillation.py                      # explicit_char_windows support
src/ctreepo/dspy_family.py                       # rewrite: GEPA scorer load, warmstart, correct metric
src/ctreepo/embedding_fno.py                     # 1×/2× channel invariant, identity init, no-truncation
src/ctreepo/fno_family.py                        # train_f/train_g freeze helpers, grad guard, leaf_size guards
src/ctreepo/trl_family.py                        # train_f/train_g subprocess + --init-checkpoint
src/preprocessing/leaf_size_utils.py             # new: EmbeddingGemma tokenizer + exact char windows
scripts/build_manifesto_fg_ladder_legacy.py      # renamed from build_manifesto_fg_ladder.py, deprecated
scripts/distill_ctreepo_students.py              # --init-checkpoint flag, effective_g/f_model resolver
scripts/evaluate_ladder_vs_expert.py             # per-leaf Pearson-vs-expert eval (from Phase A)
scripts/run_alternating_ladder.py                # entry point, --leaf-size-tokens, EmbeddingGemma defaults
scripts/run_manifesto_dimension_fit_existing_results.py  # LocalHFEmbeddingClient fp32 default
scripts/run_manifesto_teacher_fg_leaf_grid.py    # --leaf-size-tokens, --lm-concurrency, level-parallel batching
scripts/teacher_run_status.sh                    # new helper for monitoring teacher runs
```

Memory files created:

```
~/.claude/projects/-home-mlinegar-ThinkingTrees/memory/
  feedback_fno_channel_invariant.md
  feedback_fgfg_alternating_and_symmetric_backends.md
  feedback_warmstart_is_default_via_identity.md
  feedback_two_leaf_concat_budget.md
  feedback_no_truncation_size_based_leaves.md
  feedback_never_reset_between_rungs.md
```
