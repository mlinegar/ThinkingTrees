# C-TreePO f/g Coupling Audit

## Why the HLL Diagnostic Was Misleading

The HLL exact-state FNO diagnostic was not training the readout on the same
state distribution it used at evaluation time. The legacy path materialized
exact rows once:

- `f_states`: exact HLL leaf/internal/root states.
- `g_left`, `g_right`: exact child states.
- `g_target_state`: exact parent states.

Then each `f` stage learned `readout(exact_state)`, while each evaluation
rolled out the learned merge operator recursively and applied `f` to those
learned states. Multi-leaf runs therefore tested `f(learned_g_rollout)` after
training mostly on `f(exact_state)`. The corrected default is now
`objective_mode=rollout_local_law`: start from exact leaves, roll out the
current learned `g`, and optimize the single-lambda objective on that realized
tree.

`objective_mode=exact_rows` remains available only as an ablation.

## Current Status

| Path | Coupling status | What is correct now | Remaining issue |
|---|---|---|---|
| HLL exact-state FNO diagnostic | Fixed first | Default training now adapts `f` to current learned `g` rollouts and trains `g` through the same root/local objective. | Run GPU mini-grid to compare rollout default against exact-row ablation. |
| Markov FNO local-law path | Coupled | `forward_doc_unified` builds the current learned tree and the training loss consumes those realized node predictions. It already uses the single-lambda root/local objective. | Keep this as the reference implementation for FNO objective semantics. |
| Generic FNO family runtime | Partially aligned | `train_f` loads current `g`; `train_g` loads current `f`; both forward the current tree through learned merge states. | Objective is still role-weighted MSE, not the centralized corrected local-law objective. |
| DSPy / LLM families | Partially coupled | `train_g` uses current `f` as the scoring metric, so generated summaries are optimized against the current scorer. | `train_f` still trains on fixed labeled trace summaries, not regenerated summaries from current `g`. Decide whether to add rollout trace regeneration for scorer updates. |
| TRL family | Not true alternating yet | Stages are warmstarted and do not reset. | `train_g` is still SFT on teacher targets, not GRPO/reward training with current `f`. |

## Required Follow-Ups

- Make generic FNO use the same loss-row objective as Markov FNO, or clearly
  mark its role-weighted objective as an ablation.
- For DSPy/LLM ladders, decide and implement whether each `f` stage should
  train from current `g` generated full-tree traces rather than fixed teacher
  traces.
- Replace TRL `g` SFT with a current-`f` reward path before treating it as a
  true f/g alternating result.
- Add a shared audit test that each family reports whether `train_f` consumes
  current `g` outputs and whether `train_g` consumes current `f` rewards.

## Acceptance Criteria

- Every method emits dense full-tree traces when it can expose node readouts.
- Every training/evaluation path documents whether it is optimizing the current
  realized surface `f(g(tree))` or an ablation/proxy.
- Paper-facing FNO runs use the single-lambda objective:

  `J = (1 - lambda) * L_root + lambda * L_corrected`.

- Sparse oracle labels affect the local-law correction rows, not the objective
  weighting formula.
