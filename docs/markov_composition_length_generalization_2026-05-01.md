# Markov Composition Length Generalization

Date: 2026-05-01

## Question

If the model is really learning the Markov merge operator `g`, training on
shallow trees should transfer to deeper trees. In the 128-token recoverable
DGP:

- `L=4` leaves means `leaf_tokens=32`.
- `L=16` leaves means `leaf_tokens=8`.
- The exact sketch `(count, first, last)` composes perfectly for both
  partitions.

So this is a clean composition test: train the learned merge on exact sketch
pairs from `L=4`, then evaluate the same learned merge on exact sketches from
`L=16`.

## Harness

Script:

```
scripts/test_markov_composition_length_generalization.py
```

Setup:

- DGP: `hazard_topic`, 128 tokens/document, 4 hidden regimes, 16 observed
  tokens, expected boundaries `5`.
- Train: exact sketch pairs from `L=4` balanced trees only.
- Select checkpoint: one-step pair count MAE on `L=4` validation only.
- Evaluate: one-step pairs and closed-loop tree rollout on both `L=4` and
  `L=16`.
- Exact sanity: oracle sketch + oracle merge has root MAE `0.0` for both
  `L=4` and `L=16`.

Primary run:

```
outputs/markov_composition_l4_to_l16_scaling_20260501
```

## Data Scaling Results

| train docs | train pairs | L16 row seen | L16 unique seen | L4 root MAE | L16 pair MAE | L16 pair exact | L16 join acc | L16 rollout root MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 0.776 | 0.177 | 0.0807 | 0.0409 | 0.9903 | 0.9770 | 0.4498 |
| 512 | 1,536 | 0.915 | 0.414 | 0.1015 | 0.0456 | 0.9917 | 0.9742 | 0.4633 |
| 2,048 | 6,144 | 0.961 | 0.679 | 0.0799 | 0.0686 | 0.9998 | 1.0000 | 0.9437 |
| 8,192 | 24,576 | 0.986 | 0.849 | 0.0400 | 0.0152 | 0.9999 | 1.0000 | 0.1572 |

This says two things at once:

1. More data eventually helps. At 8,192 train docs, the same `g_theta` trained
   only on `L=4` exact pairs transfers to `L=16` with `0.157` root MAE.
2. One-step exact-pair accuracy is not enough. The 2,048-doc checkpoint has
   `0.9998` L16 pair exact rate and perfect join accuracy, but closed-loop
   rollout root MAE is `0.944`. Small hidden/count drift compounds when merge
   outputs are fed back as inputs.

The non-monotone 2,048-doc result is therefore a useful failure, not noise:
checkpointing on one-step `L=4` count MAE can choose a merge surface that looks
excellent under teacher forcing but is unstable under repeated composition.

## Off-Manifold Stabilization

I then reran the 8,192-doc case with small count jitter during merge training:

```
outputs/markov_composition_l4_to_l16_jitter005_20260501
```

| train docs | count jitter | L4 root MAE | L16 pair MAE | L16 pair exact | L16 join acc | L16 rollout root MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.05 | 0.0208 | 0.0113 | 0.9999 | 1.0000 | 0.0707 |
| 32,768 | 0.05 | 0.0183 | 0.0099 | 1.0000 | 1.0000 | 0.0663 |

This is the current strongest direct evidence that the neural merge operator
is learning composition. It was trained on `L=4`, evaluated on `L=16`, and the
rollout error dropped by more than 2x relative to the non-jitter 8,192-doc run.
The 32,768-doc run also reaches exact discrete merge recovery on the `L=16`
test rollout:

```
Run: outputs/markov_composition_l4_to_l16_jitter005_n32768_20260501

L4 pair MAE / root MAE      : 0.0095 / 0.0183
L16 pair MAE / root MAE     : 0.0099 / 0.0663
L16 pair exact / join acc   : 1.0000 / 1.0000
L16 rollout merge exact     : 1.0000
L16 row / unique seen rates : 0.9965 / 0.9549
```

The remaining `0.0663` root MAE is continuous count calibration, not a
discrete sketch error.

## Discrete Count Output

The continuous-count diagnosis suggested an obvious head/runtime change:
project the learned merge count output back onto the integer sketch lattice
before feeding it into the next merge. I added this as:

```
runtime_count_discretization="st_round"
```

This is not the existing theorem-count classifier head. In the carrier
projection path the count is a direct state slot, so the relevant intervention
is straight-through rounding of that slot after `count_slot_merger`.

Results:

| train docs | count jitter | count output | L4 root MAE | L16 pair MAE | L16 pair exact | L16 rollout root MAE |
|---:|---:|---|---:|---:|---:|---:|
| 8,192 | 0.00 | `st_round` | 0.0010 | 0.0001 | 0.9999 | 0.0010 |
| 8,192 | 0.05 | `st_round` | 0.0029 | 0.0018 | 0.9982 | 0.0264 |
| 32,768 | 0.05 | `st_round` | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

Best run:

```
Run: outputs/markov_composition_l4_to_l16_discrete_jitter005_n32768_20260501

L4 pair MAE / root MAE      : 0.0000 / 0.0000
L16 pair MAE / root MAE     : 0.0000 / 0.0000
L16 pair exact / join acc   : 1.0000 / 1.0000
L16 rollout merge exact     : 1.0000
```

This fully solves the exact-sketch composition problem in the simple DGP:
`g_theta` is learned from `L=4` merge pairs and transfers exactly to `L=16`
rollout.

The 8,192-doc no-jitter run is also important:

```
Run: outputs/markov_composition_l4_to_l16_discrete_nojitter_n8192_20260501

L4 root MAE                 : 0.0010
L16 pair count MAE          : 0.0001
L16 pair exact rate         : 0.9999
L16 rollout root MAE        : 0.0010
```

So the discrete lattice projection is doing most of the stabilization. Jitter
helped the continuous head, but once the count slot is discrete it is not the
main ingredient.

## Rollout Fine-Tuning Check

I also added a closed-loop rollout fine-tuning phase to the same harness. It
feeds learned merge outputs back into later merges during training and
checkpoints on validation rollout root MAE.

```
Run: outputs/markov_composition_l4_to_l16_rollout_l4only_20260501
```

This improves the `L=4` validation/test rollout, but it does not improve
`L=16` transfer:

| train docs | rollout train partition | L4 root MAE | L16 pair MAE | L16 rollout root MAE |
|---:|---|---:|---:|---:|
| 8,192 | `L=4` only | 0.0296 | 0.0177 | 0.1735 |

An upper-bound attempt that rollout-fine-tuned directly on the `L=16`
partition was worse:

```
Run: outputs/markov_composition_l4_to_l16_rollout_l16upper_20260501

L16 rollout root MAE : 1.3416
```

So the current rollout objective is not yet the right stabilizer. It can select
a good `L=4` closed-loop checkpoint, but the unnormalized tree behavior is
still fragile. The simple count-jitter pair objective is better right now.

## Interpretation

The answer to the current question is yes, but with an important qualification:

- Increasing data does move the learned operator closer when the objective and
  checkpoint are aligned with stable composition.
- Exact one-step merge supervision is necessary but not sufficient.
- Discretizing the count output is the correct architectural match for this
  theorem sketch. The remaining continuous-head error was count calibration,
  not a failure to learn the merge law.
- The composition objective must expose `g_theta` to its own induced state
  distribution, or at least to small perturbations around the exact theorem
  lattice.
- The current naive rollout loss is not enough; it needs better scaling and
  checkpointing if we want it to beat jitter.

This is consistent with the earlier endpoint-surface fix. We first had to put
endpoint slots back on the canonical theorem surface. Now the remaining issue
is rollout stability of the count/state channel under repeated learned merge.

## Next Objective

The next pipeline change should not be another root-only attempt. It should add
a closed-loop composition term:

1. Pretrain `g` on exact `L=4` sketch pairs.
2. Use discrete count slots for Markov-count sketches when the oracle target is
   integer-valued.
3. Roll out `g` through deeper exact-sketch trees (`L=8`, `L=16`) and supervise
   all internal count/endpoint targets.
4. Use count-scale-aware rollout losses or root-weighted rollout losses; the
   first naive normalized rollout objective was unstable.
5. Checkpoint on rollout validation, not only one-step pair count MAE.
6. Add learned-state replay or count jitter so `g` remains stable off the exact
   lattice.
7. Reintroduce the learned leaf encoder only after exact-sketch rollout is
   stable.

That is the clean path from "we can learn the local law" to "we can learn the
composition law that the tree actually uses."
