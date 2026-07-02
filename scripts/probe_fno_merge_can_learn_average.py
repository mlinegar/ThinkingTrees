#!/usr/bin/env python3
"""Can the FNO neural-operator MERGE learn simple binary ops (avg / max / mass-avg)?

The manifesto experiment showed the LLM (dgemma) g cannot learn the mass-weighted
ratio merge. The objection: a neural operator should EASILY learn something as
simple as average or max. This probe settles that empirically on the REAL merge
(``EmbeddingCoordinateFNOTreeRegressor.merge``) with synthetic two-child data, so
the result is about the operator's capacity, not the manifesto data or dgemma.

For each target function we train ONLY the merge path to fit it and report final
MAE. Targets:
  - avg:        0.5*(l + r)                     -- the trivial mean
  - max:        elementwise max(l, r)           -- needs maxpool/gated/mlp
  - wavg_in:    mass-weighted avg, mass ENCODED as a dim of the state vector
                (the merge CAN see it) -> should be learnable by gated/mlp
  - wavg_out:   mass-weighted avg, mass NOT in the state (a separate scalar the
                merge never receives) -> SHOULD FAIL: the weight is unobservable.

The wavg_in vs wavg_out contrast is the crux: if wavg_in is learnable but
wavg_out is not, the manifesto failure is an INPUT BOTTLENECK (mass isn't fed to
the merge), not a capacity gap -- the fix is to feed mass, not to hand-code the
merge.

Usage:
  python scripts/probe_fno_merge_can_learn_average.py \
      --merge-modes mean,gated,maxpool,mlp --dim 16 --steps 2000
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List

import torch


def _make_model(*, dim: int, mode: str, hidden: int):
    from src.ctreepo.embedding_fno import EmbeddingCoordinateFNOTreeRegressor

    return EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=dim,
        hidden_channels=hidden,
        n_modes=min(dim, 16),
        n_layers=2,
        head_hidden_dim=hidden,
        target_min=0.0,
        target_max=1.0,
        merge_mode=mode,
        merge_gate_hidden_dim=hidden,
    )


def _targets(dim: int):
    """Return {name: (make_inputs, target_fn)}.

    Inputs are (B,1,D) child states. For the mass variants, the mass is encoded
    in dim 0 of the state (wavg_in) or supplied as a separate weight the target
    uses but the merge never sees (wavg_out).
    """
    def rand_state(B):
        return torch.rand(B, 1, dim)

    def avg(l, r):
        return 0.5 * (l + r)

    def mx(l, r):
        return torch.maximum(l, r)

    def wavg_in(l, r):
        # mass = state dim 0 (>0). weighted avg over ALL dims by those masses.
        ml = l[:, :, 0:1].clamp_min(1e-3)
        mr = r[:, :, 0:1].clamp_min(1e-3)
        w = ml / (ml + mr)
        return w * l + (1.0 - w) * r

    def make_wavg_out(B):
        l, r = rand_state(B), rand_state(B)
        ml = torch.rand(B, 1, 1) + 0.1  # mass NOT in the state vector
        mr = torch.rand(B, 1, 1) + 0.1
        w = ml / (ml + mr)
        tgt = w * l + (1.0 - w) * r
        return (l, r), tgt

    return {
        "avg": (lambda B: (rand_state(B), rand_state(B)), avg),
        "max": (lambda B: (rand_state(B), rand_state(B)), mx),
        "wavg_in": (lambda B: (rand_state(B), rand_state(B)), wavg_in),
        "wavg_out": ("special", make_wavg_out),
    }


def _train_one(model, name, spec, *, dim, steps, batch, lr) -> float:
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    make, tgt_fn = spec
    for step in range(steps):
        opt.zero_grad()
        if make == "special":  # wavg_out: inputs+target come together
            (l, r), tgt = tgt_fn(batch)
        else:
            l, r = make(batch)
            tgt = tgt_fn(l, r)
        pred = model.merge(l, r)
        loss = loss_fn(pred, tgt)
        loss.backward()
        opt.step()
    # eval on a fresh batch
    with torch.no_grad():
        if make == "special":
            (l, r), tgt = tgt_fn(4096)
        else:
            l, r = make(4096)
            tgt = tgt_fn(l, r)
        mae = torch.nn.functional.l1_loss(model.merge(l, r), tgt).item()
    return mae


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge-modes", default="mean,gated,maxpool,mlp")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.merge_modes.split(",") if m.strip()]
    targets = _targets(args.dim)
    print(f"FNO merge capacity probe: dim={args.dim} hidden={args.hidden} "
          f"steps={args.steps} (final MAE on held-out batch; lower=better)\n")
    header = f'{"mode":>8} | ' + " | ".join(f'{t:>9}' for t in targets)
    print(header)
    print("-" * len(header))
    results: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        row = {}
        for tname, spec in targets.items():
            model = _make_model(dim=args.dim, mode=mode, hidden=args.hidden)
            mae = _train_one(model, tname, spec, dim=args.dim, steps=args.steps,
                             batch=args.batch, lr=args.lr)
            row[tname] = mae
        results[mode] = row
        print(f'{mode:>8} | ' + " | ".join(f'{row[t]:>9.5f}' for t in targets))
    print("\nReading: avg/max ~0 = operator learns the trivial op. "
          "wavg_in ~0 but wavg_out >> 0 => the merge CAN learn mass-weighting "
          "WHEN mass is in the state, and CANNOT when it isn't (input bottleneck, "
          "not a capacity gap).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
