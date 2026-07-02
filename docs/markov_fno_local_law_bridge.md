# Markov Local-Law Bridge: JAX → PyTorch FNO (planned, 2026-05-05)

This is the experiment-design handoff for testing whether the JAX
`learned_local_laws` result transfers to the PyTorch `CleanUnifiedNO` FNO
surface. It is a follow-up to:

- [`docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`](contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md)
- [`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md)
- [`outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md)

## Status / TL;DR

- **JAX `learned_local_laws` works.** Across `analytic`, `learned_merge`,
  `learned_decoder`, and `fully_learned` architectures, the local-law
  objective recovers the Markov sufficient sketch with first/last regime
  accuracy = 1.0 / 1.0 at leaf=1, theta accuracy 1.0 across leaves 2-64.
- **PyTorch `CleanUnifiedNO` (no laws) does not.** Best general f/g run
  bottoms out at root MAE ~1.15, and at leaf_tokens=128 collapses to
  ~1.80 root MAE across all dependence regularizers.
- **Bridge question:** does C1+C2+C3 supervision on the FNO encoder/merge/
  decoder transfer the JAX result? This experiment tests it.
- **Headline action:** the existing PyTorch probe already implements the
  needed objectives (`markov_node_witness`, `markov_local_laws_fno`); the
  experiment is to *run* them on the same hazard panel as the JAX control,
  with matched seeds and a unified metric schema. No new objectives need to
  be implemented.

### Update 2026-05-05 evening — Round 1 results and Round 2 Stage 1 in progress

- **Round 1 multi-leaf bridge campaign concluded; bridge is not solved.**
  Output:
  [`outputs/markov_fno_bridge_8h_20260505_065112/`](../outputs/markov_fno_bridge_8h_20260505_065112/).
  Best `markov_local_laws_fno` cell at t128: `leaf=32, ch=128, gm=16, ep=24`,
  test_root_mae **`1.94`**, leaf first/last `0.93/0.94`, merge first/last
  `0.78/0.80`, root rounded-count exact rate only `0.089`.
  `markov_node_witness` collapsed to constant prediction across configurations
  (`pred_std ≈ 2e-4`). t2048 composition stress did not move the
  `recoverable_v5_t2048` ~`2.13` floor.
- **Round 2 Stage 1 single-leaf encoder diagnostic — encoder confirmed working,
  capacity-vs-calibration story sharpened.**
  Output:
  [`outputs/markov_fno_round2_stage1_20260505_173903/`](../outputs/markov_fno_round2_stage1_20260505_173903/).
  At `n_leaves=1` (root=leaf), boundary BCE F1 stays at `0.99–1.00` from
  doc=32 through doc=64 (doc=128 in progress), witness/laws full_exact rate
  hits `0.83–0.99` at doc=32, theta_first/last accuracy is `1.0/1.0`.
  **The encoder is not the bottleneck.** Pool calibration accumulates error
  linearly with leaf length even when per-token classification is
  near-perfect. Wider channels (ch=128 → ch=256) substantially improve
  calibration at doc=64 (witness count_mae 0.39→0.27, full_exact 0.74→0.86)
  even though F1 was already ≥0.99 — additional capacity goes into tighter
  sigmoid scores, not better classification. The multi-leaf collapse must
  come from pooling structure and/or merge composition, not encoder
  capacity.
- **Round 2 Stage 2 (pooling alternatives) and Stage 3 (witness/root readout
  decoupling) are the next experiments**, plan drafted but gated on Stage 1
  doc=128 results.

## Cross-Architecture Parity Matrix

| Concept | JAX `learned_local_laws` (`contextual_sbijax.py`) | PyTorch `CleanUnifiedNO` (`clean_unified_fg.py`) | PyTorch `fno_family.py` |
|---|---|---|---|
| Backend | JAX + Haiku + sbijax | PyTorch + neuraloperator FNO | PyTorch + FNO |
| Local laws as training objective | C1+C2+C3 active; four arch variants | `markov_local_laws_fno` (C1 leaf calibration + C2 relational merge + C3 idempotence/range) | absent — `local_law_trace_metadata` is diagnostic only |
| Direct sketch witness supervision | `c2_merge_target=theta` | `markov_node_witness` (every leaf and merge state regressed onto exact `(count, first, last)`) | absent |
| `analytic` exact-Markov merge / decoder cell | yes | no — CleanUnifiedNO has no exact-merge module; the exact-Markov f/g lives in the production pipeline (`recoverable_v5_t2048`) | n/a |
| `learned_merge` vs `learned_decoder` separability | yes (`--law-architecture`) | no — encoder/g/decoder are jointly learned | n/a |
| `c2_merge_target {theta, self_consistency}` | CLI knob | `markov_node_witness` ↔ `c2_theta`; `markov_local_laws_fno` ↔ `c2_self_consistency` (relational only) | absent |
| Provenance fields | `law_architecture`, `c2_merge_target`, `merge_network`, `decoder_kind`, `local_law_package_*` | `c2_merge_target="decoded_child_relational"` hardcoded; no `law_architecture` field | absent |

The key parity insight: **PyTorch already has the two objectives that
correspond to JAX's two main C2 cells.** What's missing is a unified metric
schema and matched bundle, not new code paths.

## Existing Infrastructure (Already Implemented)

The probe and grid runner expose the needed objectives. Do not re-implement.

| Feature | Location |
|---|---|
| `--training-objective markov_node_witness` (decoded `(count, first, last)` MSE+CE on every leaf and merge state via `_MarkovWitnessReadout`) | [`scripts/probe_clean_unified_no.py:2728-2733`](../scripts/probe_clean_unified_no.py#L2728-L2733) |
| `--training-objective markov_local_laws_fno` (C1 leaf calibration MSE+CE, C2 relational merge under Markov boundary rule, C3 differentiable idempotence/range) | [`scripts/probe_clean_unified_no.py:2738-2742`](../scripts/probe_clean_unified_no.py#L2738-L2742); loss body at [`:1805-1936`](../scripts/probe_clean_unified_no.py#L1805) |
| Per-node witness target builder (balanced-tree merge-order alignment with `forward_doc`) | `_markov_node_witness_targets_for_leaves`, `_balanced_merge_state_triples` |
| Grid runner cells `root \| contextual_none \| markov_node_witness \| markov_local_laws_fno` | [`scripts/run_clean_unified_no_grid.py:341-342`](../scripts/run_clean_unified_no_grid.py#L341-L342) |
| Loss weight knobs | `--markov-law-{leaf,merge,idempotence,count,edge}-weight`, `--markov-law-readout {flatten,conv_pool}` ([`run_clean_unified_no_grid.py:397-407`](../scripts/run_clean_unified_no_grid.py#L397-L407)) |
| Witness weight knobs | `--markov-witness-{weight,count-weight,edge-weight,readout}` ([`run_clean_unified_no_grid.py:389-396`](../scripts/run_clean_unified_no_grid.py#L389-L396)) |

Because the prior CleanUnifiedNO ablation
([`outputs/clean_unified_fg_contextual_ablation_t128/`](../outputs/clean_unified_fg_contextual_ablation_t128/))
only ran `contextual_sufficiency` cells, neither `markov_node_witness` nor
`markov_local_laws_fno` has been exercised at scale yet.

## Required Probe Extension (Small)

The PyTorch summary needs three new metric columns to match JAX. Estimate
~30 lines of code, shared across both witness and laws cells.

| Column | Source |
|---|---|
| `theta_mae` | decode root state via `_MarkovWitnessReadout`; concatenate `(count_pred, first_onehot, last_onehot)` and compute MAE against the analytic sketch |
| `root_first_regime_accuracy`, `root_last_regime_accuracy` | argmax of root-node `first_logits` / `last_logits` from the same readout, vs analytic root sketch |
| `eps_leaf` | leaf-state decode error vs analytic `(count, first, last)` per leaf, averaged |
| `eps_merge` | parent decode vs `markov_compose(left_decode, right_decode)`, averaged |
| `eps_idemp` | already present as `idempotence_loss`; expose as eval-time metric |

These all derive from the existing `_MarkovWitnessReadout` head plus
`_balanced_merge_state_triples` — no new modeling components.

## Experiment Plan

### Bundle parity

- **JAX control and PyTorch t128 primary share the same data bundle:**
  `paper_hazard_panel_v1_t128`, train=10240, val=test=1024.
- PyTorch loads it via `--load-data-bundle`, not regenerated.
- t2048 composition stress uses `recoverable_v5_t2048` (separate).

### Lane 1 — JAX control grid

No code change. Reuses `learned_local_laws`.

- bundle: `paper_hazard_panel_v1_t128`
- inputs: `markov_exact_sketch`, `regime_one_hot`, `one_hot_token_ids`
- leaves: `1, 2, 4, 16, 64`
- architectures × C2: `analytic / c2_theta`,
  `learned_merge / c2_self_consistency`,
  `fully_learned / c2_self_consistency`
- seeds: `0, 1, 2`
- iterations: `n_iter=300`, `batch_size=256`, cosine LR

Launcher pattern: same as
[`scripts/run_optimize_to_zero_fg_architecture_ablation.sh`](../scripts/run_optimize_to_zero_fg_architecture_ablation.sh).

### Lane 2 — PyTorch CleanUnifiedNO primary grid (FNO bridge, t128)

Run via [`scripts/run_clean_unified_no_grid.py`](../scripts/run_clean_unified_no_grid.py)
with `--load-data-bundle` pointing at the same `paper_hazard_panel_v1_t128`
artifact.

- `doc_tokens=128`, `train_docs=10240`, `eval_docs=1024`
- `leaf_tokens`: `2, 16, 32, 64, 128` (leaf=2 is the cleanest merge-axis
  test against JAX `learned_merge`)
- `channels`: `64, 128`
- `g_n_modes`: `8, 16`
- objectives: `root`, `contextual_none`, `markov_node_witness`,
  `markov_local_laws_fno`
- `epochs=60`, `batch_size=16`
- seeds: `0, 1, 2`
- forward pass: `collect_full_trace=True` for witness and laws cells
  (load-bearing per CLAUDE.md; no per-node `.cpu()` / `.item()` calls)

Full grid is 5 × 2 × 2 × 4 × 3 = 240 cells. Stage it.

#### Stage A (gate test)

`leaf_tokens=64,128`, `channels=64`, `g_n_modes=16`, all 4 objectives, seeds
`0,1,2`. ~24 cells.

**Gate criterion:** at least one leaf size where `markov_local_laws_fno`
beats both `root` and `contextual_none` on root-count MAE *and* improves
root first/last regime accuracy. If that does not hold, do not proceed to
Stage B or t2048 — the laws are not transferring and we need to investigate
encoder capacity or readout placement before burning more compute.

#### Stage B (depth + capacity)

Conditional on Stage A passing the gate.

`leaf_tokens=2,16,32`, `channels={64,128}`, `g_n_modes={8,16}`, objectives
`{markov_local_laws_fno, markov_node_witness, root}`, seed `0` first; expand
to `1,2` only on the winning cell.

### Lane 3 — PyTorch composition-stress grid (t2048)

Run only after Stage A passes. The headline target is the ~2.13 zero-merge
root_mae floor on `recoverable_v5_t2048` documented in CLAUDE.md.

- benchmark `recoverable_v5_t2048`, `train_docs=10240`, `eval_docs=1024`
- `leaf_tokens`: `256, 2048`
- `channels=128`, `g_n_modes={8,16}`
- objectives: `{root, markov_node_witness, markov_local_laws_fno}`
- `epochs=20` (down from the proposed 40 — `batch_size=1` × 10240 × 40 =
  ~400k FNO-1D forwards, exceeds overnight budget)
- seed `0` only initially; seed expand on the winning cell
- pre-flight required: one cell at `epochs=1, train_docs=64` to confirm
  throughput before committing to the full grid

### Unified Metric Schema

Every cell in every lane reports the columns below in `summary.json` and
`grid_summary.csv`. JAX and PyTorch headers must be identical.

| Column | JAX source | PyTorch source |
|---|---|---|
| `root_count_mae` | derived from theta count component | existing `test_root_mae` |
| `theta_mae` | existing | new — root-node `_MarkovWitnessReadout` decode |
| `root_first_regime_accuracy` | `theta_first_regime_accuracy` | new — argmax of root `first_logits` |
| `root_last_regime_accuracy` | `theta_last_regime_accuracy` | new — argmax of root `last_logits` |
| `leaf_first_acc`, `leaf_last_acc` | n/a (leaf=root for sketch) | existing |
| `merge_first_acc`, `merge_last_acc` | n/a | existing |
| `eps_leaf` | existing | new — leaf decode vs analytic |
| `eps_merge` | existing | new — parent decode vs (left ⊕ right) under Markov rule |
| `eps_idemp` | existing | existing as `idempotence_loss` test value |
| `contextual_mae` | existing | existing where applicable |

Also report architecture/objective provenance:
`law_architecture`, `c2_merge_target`, `objective`, `seed`,
`bundle`, `leaf_tokens`/`leaves`, `channels`, `g_n_modes`.

## Reproduction Commands

(stamped roots, new launcher root per stage)

```bash
source venv/bin/activate
STAMP=$(date -u +%Y%m%d_%H%M%S)
ROOT=outputs/markov_fno_local_law_bridge_${STAMP}

# Lane 1: JAX control
./venv/bin/python scripts/long_job.py launch \
  --name jax_local_law_control_${STAMP} \
  --job-root ${ROOT}/jax_control/launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --no-replace-existing \
  -- ./scripts/run_optimize_to_zero_fg_architecture_ablation.sh 0
# (extend that script to sweep seeds {0,1,2} and inputs
# {markov_exact_sketch, regime_one_hot, one_hot_token_ids})

# Lane 2 Stage A: PyTorch t128 gate test
./venv/bin/python scripts/long_job.py launch \
  --name pytorch_t128_stage_a_${STAMP} \
  --job-root ${ROOT}/pytorch_t128_stage_a/launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --no-replace-existing \
  -- ./venv/bin/python scripts/run_clean_unified_no_grid.py \
       --benchmark paper_hazard_panel_v1 \
       --load-data-bundle <bundle path here> \
       --doc-tokens 128 --train-docs 10240 --eval-docs 1024 \
       --leaf-tokens-grid 64,128 --channels-grid 64 --g-n-modes-grid 16 \
       --objectives root,contextual_none,markov_node_witness,markov_local_laws_fno \
       --seeds 0,1,2 --epochs 60 --batch-size 16 \
       --output-root ${ROOT}/pytorch_t128_stage_a

# Stage B and Lane 3 commands: same pattern, gated on Stage A success.
```

If existing CleanUnifiedNO jobs are still running, queue rather than contend:

```bash
./venv/bin/python scripts/wait_for_long_job_then_run.py \
  --watch outputs/<existing job root>/launcher \
  -- <stage launch command>
```

## Lean Crosswalk Update

The bridge experiment is **empirical evidence under the
laws-realize hypothesis**, not a Lean theorem about FNO + SGD.

| Python / artifact concept | Lean anchor | Relation |
|---|---|---|
| Markov sketch supervision on FNO leaf and merge states | `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean` | Formal exact-sketch witness; the witness loss regresses learned states onto this sketch. |
| FNO encoder + learned merge + decoder under C1/C2/C3 | `lean3/FormalProofs/OPT/NeuralOperatorTheoremBridge.lean`, `lean3/FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean`, `lean3/FormalProofs/ML/FNOFormalization.lean` | Interface layer: *if* the FNO realizes the local laws within slack, theorem-side transport applies. Does not prove SGD finds that operator. |
| Two-sided contextual sufficiency | `lean3/FormalProofs/OPT/ContextualQuerySufficiency.lean`, `lean3/FormalProofs/OPT/MarkovSufficiency.lean` | Already covered by the JAX result; the bridge experiment tests transfer to the FNO surface. |

Practical rule for write-ups (load-bearing per
[`feedback_lean_alignment.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_lean_alignment.md)
and the resolution doc): say **"Lean proves the exact sketch, contextual,
and sliced sufficiency surfaces; the bridge experiment is empirical
evidence that the FNO trainer realizes the local laws."** Do not say
**"Lean proves FNO + SGD finds the sufficient state."**

## Verification

- **Parity unit test:** `_markov_node_witness_targets_for_leaves` (PyTorch)
  agrees with `_markov_witness_targets_for_spans` (JAX) on a small fixture.
  This is the load-bearing assumption that the two lanes target the same
  state. Add to [`tests/`](../tests/).
- **Trace fast-path check:** `collect_full_trace=True` is set in witness/
  laws training paths; static check `rg "\.cpu\(\)|\.item\(\)"` finds no
  per-batch host syncs (per
  [`feedback_no_per_node_cpu_sync_in_forward_doc.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_no_per_node_cpu_sync_in_forward_doc.md)).
- **Smoke run:** Stage A's smallest cell (`leaf_tokens=128`, `channels=64`,
  `g_n_modes=8`, `markov_local_laws_fno`, `epochs=1`, `train_docs=64`)
  finishes in <5 min and emits all unified metric columns.
- **Apples-to-apples sanity:** Stage A `leaf_tokens=128` JAX `regime_one_hot`
  cell vs PyTorch `markov_local_laws_fno` cell at matched bundle/seed
  show monotonic ranking on `root_count_mae` and root regime accuracy.
- **t2048 throughput pre-flight** before committing to full t2048 grid.
- **End-to-end:** all three lanes emit a column-equal `grid_summary.csv`.
  Cross-grid roll-up appended as a follow-up section in
  [`outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md)
  or a dated successor.

## Open Threads

1. **Maximally-general no-theta cell** carried over from
   [`project_general_fg_resolved.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/project_general_fg_resolved.md):
   `--c2-merge-target self_consistency --local-law-leaf-weight 0` on the
   JAX side; the PyTorch analog is already `markov_local_laws_fno` with
   `--markov-law-leaf-weight 0`.
2. **PyTorch `c2_theta` cell separate from `markov_node_witness`.** Currently
   the two coincide; if we want a "laws + theta merge target only" cell
   without leaf supervision, that's a small extension on top of
   `markov_local_laws_fno`.
3. **`fno_family.py` law objective.** Currently absent (laws are diagnostic
   only). If Stage A succeeds in CleanUnifiedNO, replicate the same
   objective in `fno_family.py` and re-test against the production
   pipeline's `recoverable_v5_t2048` zero-merge floor.
4. **Seed-sensitivity at leaf=2** for the eps_merge two-cluster pattern
   (open thread #2 from `project_general_fg_resolved.md`) — Lane 2 Stage B
   at leaf=2 with seeds 0,1,2 directly addresses this.

## Memory Pointers

- [`project_optimize_to_zero_resolved.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/project_optimize_to_zero_resolved.md) — JAX result that motivates this bridge.
- [`project_general_fg_resolved.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/project_general_fg_resolved.md) — JAX `learned_merge`/`fully_learned` result and `eps_merge` interpretation.
- [`feedback_judge_summary_by_theta_accuracy.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_judge_summary_by_theta_accuracy.md) — never judge on contextual MAE alone.
- [`feedback_no_per_node_cpu_sync_in_forward_doc.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_no_per_node_cpu_sync_in_forward_doc.md) — host-sync ban inside the forward path.
- [`feedback_head_capacity_was_not_the_bottleneck.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_head_capacity_was_not_the_bottleneck.md) — wider heads don't crack the t2048 floor; encoder/leaf-pooling is the suspect, which this bridge experiment also tests.
- [`feedback_lean_alignment.md`](../.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_lean_alignment.md) — all three laws required by Lean.
