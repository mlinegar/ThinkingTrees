# Markov Composition Surface Results

Date: 2026-05-01

This note isolates why the Markov neural tree was not composing learned
sketches reliably, and records the current lower-stack controls.

## Question

The simple DGP is exactly mergeable by the theorem sketch
`(count, first, last)`. If the leaf encoder learns the sketch and the merge
operator learns `g`, then repeated tree composition should work. The failure
mode was that each part could look plausible in isolation while the learned
leaf states and learned merge did not compose.

## Surface Fixes

We found three surface issues and patched them:

- `encode_summary()` now preserves direct `(count, first, last)` carrier
  summaries before the `unified_g` wide-summary branch. Exact theorem
  summaries no longer pass through an unrelated learned encoder.
- Direct carrier slots are validated as `(1, n_regimes, n_regimes)`. The old
  inherited `8/8/8` dimensions were invalid for direct endpoint logits.
- Fixed-fused dense leaf evaluation now uses the same carrier-projection leaf
  path as flat/list training.

After those fixes, exact theorem summaries reenter the model state perfectly:

```
Run: outputs/markov_surface_carrier_v2_split_20260501

exact summaries -> state leaf count MAE : 0.0000
exact summaries -> state exact match    : 1.0000
exact summaries -> state first/last acc : 1.0000 / 1.0000
```

## What Learns

The carrier-projection merge MLP can learn `g` on the canonical sketch surface.

```
Run: outputs/markov_carrier_merge_surface_20260501

test pair count MAE                 : 0.0184
test pair exact rate                : 1.0000
test pair join accuracy             : 1.0000
exact leaves + learned merge root MAE: 0.0533
```

This rules out "the merge network cannot represent the Markov merge" as the
primary explanation.

The learned leaf encoder also learns most of the sketch:

```
Run: outputs/markov_carrier_merge_surface_with_leaf_fixedeval_20260501

leaf count MAE                      : 0.1353
leaf exact match                    : 0.9492
leaf first/last accuracy            : 1.0000 / 1.0000
learned leaves + exact merge root MAE: 0.5954
```

So the leaf encoder contains enough information for a good root when the
merge is exact-projected.

## What Still Failed

The bad case was learned leaves through learned `g`:

```
Run: outputs/markov_carrier_merge_surface_with_leaf_fixedeval_20260501

learned leaves + learned merge root MAE : 7.4303
merge exact-state match                 : 0.1423
merge first/last accuracy               : 1.0000 / 1.0000
merge join accuracy                     : 0.9939
```

Adding Gaussian count jitter during direct pair training did not solve this:

```
Run: outputs/markov_carrier_merge_surface_jitter05_leaf_20260501

learned leaves + learned merge root MAE : 6.6708
learned leaves + exact merge root MAE   : 0.5908
```

The interpretation is a composition surface mismatch, not missing state. The
merge MLP was trained on exact-lattice endpoint slots, but learned leaves
produce raw endpoint logits. Argmax endpoints were correct, yet the raw logits
were off the canonical sketch surface. Exact projected merge canonicalizes by
decoding endpoints; learned `g` had been consuming the raw off-manifold slots.

## Composition Works After Canonical Endpoint Merge

The current patch canonicalizes direct carrier endpoint slots before learned
merge:

- the learned count merger still predicts the parent count;
- endpoint identities are passed as straight-through one-hot slots;
- parent first/last slots are propagated in canonical one-hot form.

This preserves learnability of `g` while making the learned composition input
match the theorem sketch surface. The success criterion was:

```
learned leaves + learned merge root MAE ~= learned leaves + exact merge root MAE
```

The patched run meets that criterion:

```
Run: outputs/markov_carrier_merge_surface_canonical_endpoints_20260501

merge pretrain pair count MAE           : 0.0184
merge pretrain pair exact rate          : 1.0000
leaf count MAE                          : 0.1353
leaf exact match                        : 0.9492
leaf first/last accuracy                : 1.0000 / 1.0000
learned leaves + exact merge root MAE   : 0.5954
learned leaves + learned merge root MAE : 0.5970
merge exact-state match                 : 0.7723
merge first/last/join accuracy          : 1.0000 / 1.0000 / 1.0000
```

This is the missing composition demonstration. The learned leaf encoder emits
an approximate theorem sketch; the learned count merger composes those states
through the tree; and the learned root path is as accurate as exact projected
merge on the same learned leaves. The remaining root error is leaf count
error, not learned merge failure.

Before canonical endpoint merge, the same setup had:

```
learned leaves + exact merge root MAE   : 0.5954
learned leaves + learned merge root MAE : 7.4303
```

So the decisive issue was not that composition was absent from the objective.
It was that the learned merge consumed off-surface endpoint logits even though
the Markov law is defined on endpoint identities. Canonical endpoint slots put
the learned `g_theta` back on the theorem surface.

## Main Split Harness Check

The ordinary two-stage split harness improves under the same surface patch, but
does not yet match the sequential lab:

```
Run: outputs/markov_surface_carrier_v2_split_canonical_20260501

exact summaries -> state exact match     : 1.0000
exact leaves + learned merge root MAE    : 2.5957
exact leaves + learned merge exact rate  : 0.2891
learned leaves + learned merge root MAE  : 1.2471
learned leaves + exact merge root MAE    : 4.6326
learned leaf first/last accuracy         : 0.4961 / 0.3481
```

Compared to the pre-canonical run, exact leaves through learned merge improve
from `8.9882` to `2.5957`, so the endpoint-surface patch is moving the right
failure mode. But the joint schedule still fails to train the learned leaf
endpoints well, and learned leaves through exact merge remain poor. This is
why the sequential lab is the stronger proof of learnable composition: it
separates "can leaf learn sketch?", "can `g` learn merge?", and "do they
compose after surface canonicalization?".

## L=4 to L=16 Composition Check

The newest harness isolates length generalization:

```
scripts/test_markov_composition_length_generalization.py
```

It trains `g_theta` only on exact sketch pairs from `L=4` trees, then evaluates
the same learned merge in closed-loop rollout on `L=16` exact-sketch trees from
the same 128-token DGP. Full details are in
`docs/markov_composition_length_generalization_2026-05-01.md`.

Primary scaling run:

```
outputs/markov_composition_l4_to_l16_scaling_20260501
```

| train docs | L4 root MAE | L16 pair MAE | L16 pair exact | L16 rollout root MAE |
|---:|---:|---:|---:|---:|
| 128 | 0.0807 | 0.0409 | 0.9903 | 0.4498 |
| 512 | 0.1015 | 0.0456 | 0.9917 | 0.4633 |
| 2,048 | 0.0799 | 0.0686 | 0.9998 | 0.9437 |
| 8,192 | 0.0400 | 0.0152 | 0.9999 | 0.1572 |

The 8,192-doc point shows the expected data-scaling direction: the `L=4`
operator transfers substantially better to `L=16`. But the 2,048-doc point is
important: one-step exact-pair accuracy can be almost perfect while closed-loop
composition is unstable. This is teacher-forcing mismatch, not a failure of
the Markov sketch.

Adding small count jitter during merge training improves rollout stability:

```
Run: outputs/markov_composition_l4_to_l16_jitter005_20260501

L4 root MAE                 : 0.0208
L16 pair count MAE          : 0.0113
L16 pair exact rate         : 0.9999
L16 join accuracy           : 1.0000
L16 rollout root MAE        : 0.0707
```

Scaling jittered L4-pair training further preserves the trend:

```
Run: outputs/markov_composition_l4_to_l16_jitter005_n32768_20260501

train docs                   : 32,768
L4 root MAE                  : 0.0183
L16 pair count MAE           : 0.0099
L16 pair exact rate          : 1.0000
L16 join accuracy            : 1.0000
L16 rollout merge exact      : 1.0000
L16 rollout root MAE         : 0.0663
```

The remaining error is continuous count calibration rather than a discrete
composition failure.

I also tested a naive closed-loop rollout fine-tune. Training rollout on `L=4`
only improved `L=4` root MAE to `0.0296`, but did not transfer better to `L=16`
(`0.1735` root MAE). Rollout fine-tuning directly on `L=16` was worse
(`1.3416` root MAE). So the current rollout loss needs better scaling; count
jitter is the stronger stabilizer for the continuous count head.

### Discrete Count Slot

The remaining continuous-head error was count calibration. I added a
straight-through discrete count mode for the carrier merge output:

```
runtime_count_discretization="st_round"
```

This rounds the learned `count_slot_merger` output onto the integer sketch
lattice before the next merge, while preserving a straight-through gradient.

The resulting length-generalization run solves the exact-sketch composition
problem:

```
Run: outputs/markov_composition_l4_to_l16_discrete_jitter005_n32768_20260501

train docs                   : 32,768
trained on                   : L=4 exact sketch pairs
evaluated on                 : L=16 rollout
L16 pair count MAE           : 0.0000
L16 pair exact rate          : 1.0000
L16 join accuracy            : 1.0000
L16 rollout merge exact      : 1.0000
L16 rollout root MAE         : 0.0000
```

Even without jitter, the discrete output nearly solves the transfer at 8,192
train docs:

```
Run: outputs/markov_composition_l4_to_l16_discrete_nojitter_n8192_20260501

L16 pair count MAE           : 0.0001
L16 pair exact rate          : 0.9999
L16 rollout root MAE         : 0.0010
```

So yes: for this integer-valued Markov sketch, the learned merge should output
a discrete count slot. The continuous head was solving the wrong geometry.

For exact theorem sketches, this solves composition across tree length. The
remaining work is to carry the same discrete merge surface back to learned
leaf states, where learned-state replay/off-manifold training still matters.

## Next Ladder

The next diagnostic is to move this sequential-teacher-forcing success back
into the main training pipeline:

1. Train `g` on exact sketch pairs.
2. Use the discrete count-slot runtime for integer Markov counts.
3. Train leaf encoder with `g` frozen.
4. Train `g` on learned-leaf child states with internal full-sketch targets.
5. Fine-tune jointly only after the off-grid merge is stable.

That ladder directly tests composition on the distribution induced by the
learned summarizer, not just the exact theorem lattice.
