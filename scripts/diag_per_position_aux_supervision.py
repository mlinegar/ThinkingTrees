#!/usr/bin/env python3
"""Experiment C: per-position auxiliary supervision.

Setup:
  The current f's forward is mathematically equal to a per-position scalar
  output followed by sum-pool over L:
    f.forward(state) = sum_l f.linear(f.fno(state)[:, :, l])

  So for the same f.linear we can simultaneously expose:
    - root_pred_norm  = sum_l per_pos_norm[l]         (B,)
    - per_pos_unnorm  = per_pos_norm * target_scale   (B, L)

  Aux supervision:
    boundary_indicator[i] = 1 if block(t[i]) != block(t[i-1]) else 0  (and 0 at i=0)
    aux_loss = mean over l of (per_pos_unnorm[l] - boundary_indicator[l])^2

  Hypothesis: the architecture can express per-position indicators (it's
  basically the witness in differentiable form). Per-position MSE gives
  O(1) per-position gradient, vs O(1/L) for root-only. If we're right
  about gradient density being the obstacle, root MAE -> ~0 for free
  (because root = sum of per-pos predictions).

Variants:
  C1: aux only, no root MSE     (cleanest expressivity test)
  C2: aux + root MSE jointly    (the practical fix)
  C3: B-baseline replay (palette-init, root MSE only) for an apples-to-apples comparison
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.probe_clean_unified_no import (  # noqa: E402
    _doc_arrays,
    _load_split_docs,
    _palette_block_map,
    _to_tensor,
)
from src.ctreepo.sim.core.clean_unified_fg import (  # noqa: E402
    CleanUnifiedNO,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _palette_init_embedding(model, *, vocab_size, n_regimes, freeze):
    block_by_token = _palette_block_map(vocab_size=vocab_size, n_regimes=n_regimes)
    emb = model.token_embedding.embedding
    new_w = torch.zeros_like(emb.weight)
    for tok_id, blk in enumerate(block_by_token):
        new_w[tok_id, int(blk)] = 1.0
    with torch.no_grad():
        emb.weight.copy_(new_w)
    if freeze:
        emb.weight.requires_grad = False


def _boundary_indicator(tokens, *, block_by_token):
    """tokens: 1D list of ints. returns (L,) float tensor of 0/1, indicator at i
    means 'transition between i-1 and i'. Position 0 is 0."""
    blocks = [block_by_token[int(t)] for t in tokens]
    out = [0.0] + [1.0 if blocks[i] != blocks[i-1] else 0.0 for i in range(1, len(blocks))]
    return out


def _f_per_position(model, state):
    """state: (B, C, L). Returns (B, L) per-position scalar in NORMALIZED units."""
    x = model.f.fno(state)                               # (B, C, L)
    per_pos_norm = model.f.linear(x.transpose(-1, -2)).squeeze(-1)  # (B, L)
    return per_pos_norm


def _forward_per_position_and_root(model, leaf_tokens):
    """Returns per_pos_norm (1, L) and root_pred_norm (1,) jointly.
    Replicates model.forward_doc for the one-leaf case but exposes per-position."""
    embedded = model.token_embedding(leaf_tokens)        # (1, C, L)
    state = model.g(embedded)                            # (1, C, L)
    per_pos_norm = _f_per_position(model, state)         # (1, L)
    root_pred_norm = per_pos_norm.sum(dim=-1)            # (1,)
    return per_pos_norm, root_pred_norm


def _eval_pred_stats(model, docs, *, device, target_scale, block_by_token):
    model.eval()
    preds, truths = [], []
    aux_mses = []
    arrays = _doc_arrays(docs)
    with torch.no_grad():
        for di in range(len(docs)):
            tok_list = arrays[0][di][0]  # one leaf, take its token list
            tok = _to_tensor(arrays[0][di], dtype=torch.long, device=device)
            per_pos_norm, root_pred_norm = _forward_per_position_and_root(model, tok)
            per_pos_unnorm = per_pos_norm * target_scale
            bnd = torch.tensor(
                _boundary_indicator(tok_list, block_by_token=block_by_token),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)  # (1, L)
            aux_mses.append(float(((per_pos_unnorm - bnd) ** 2).mean()))
            preds.append(float(root_pred_norm) * target_scale)
            truths.append(float(arrays[3][di]))
    p = np.asarray(preds); t = np.asarray(truths)
    return {
        "mae": float(np.mean(np.abs(p - t))),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "truth_mean": float(np.mean(t)),
        "truth_std": float(np.std(t)),
        "corr": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-6 else float("nan"),
        "aux_mse": float(np.mean(aux_mses)),
    }


def _precompute_boundary_tensors(docs, *, block_by_token, device):
    out = []
    for d in docs:
        tok_list = list(d.leaf_token_ids[0])
        bnd = _boundary_indicator(tok_list, block_by_token=block_by_token)
        out.append(torch.tensor(bnd, dtype=torch.float32, device=device))
    return out


def run_cell(
    *,
    train,
    val,
    target_scale,
    block_by_token,
    bnd_train,
    bnd_val,
    use_aux,
    use_root,
    aux_weight,
    root_weight,
    epochs,
    lr,
    bs,
    label,
    device,
    log,
):
    log(f"\n=== {label} ===")
    log(f"use_aux={use_aux} use_root={use_root} aux_w={aux_weight} root_w={root_weight} "
        f"epochs={epochs} lr={lr} bs={bs}")
    torch.manual_seed(0)
    model = CleanUnifiedNO(
        vocab_size=16,
        target_scale=target_scale,
        channels=4,
        g_n_modes=16,
        g_n_layers=2,
        scorer_n_modes=8,
        scorer_n_layers=2,
        pooling_mode="sum",
    ).to(device)
    _palette_init_embedding(model, vocab_size=16, n_regimes=4, freeze=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params total={n_params:,}  trainable={n_trainable:,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    arrays = _doc_arrays(train)
    n_train = len(train)

    s = _eval_pred_stats(model, val, device=device, target_scale=target_scale,
                         block_by_token=block_by_token)
    log(f"epoch  0 (init): val_mae={s['mae']:.4f}  pred_std={s['pred_std']:.4f}  "
        f"corr={s['corr']:.3f}  aux_mse={s['aux_mse']:.4f}")

    best_mae = float("inf")
    best_epoch = 0
    history = [{"epoch": 0, **s}]
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(n_train).tolist()
        running = 0.0
        running_aux = 0.0
        running_root = 0.0
        nb = 0
        for bi in range(0, n_train, bs):
            optimizer.zero_grad()
            losses = []
            for di in order[bi:bi+bs]:
                tok = _to_tensor(arrays[0][di], dtype=torch.long, device=device)
                per_pos_norm, root_pred_norm = _forward_per_position_and_root(model, tok)
                # Aux: per-position MSE in COUNT units
                bnd = bnd_train[di].unsqueeze(0)                      # (1, L)
                per_pos_unnorm = per_pos_norm * target_scale          # (1, L)
                if use_aux:
                    aux = ((per_pos_unnorm - bnd) ** 2).mean()
                else:
                    aux = torch.zeros((), device=device)
                # Root: MSE in NORMALIZED units (matches root_mse_loss)
                if use_root:
                    root_t_unnorm = float(arrays[3][di])
                    root_t_norm = root_t_unnorm / target_scale
                    root_loss = (root_pred_norm.squeeze(0) - root_t_norm) ** 2
                else:
                    root_loss = torch.zeros((), device=device)
                losses.append(aux_weight * aux + root_weight * root_loss)
                running_aux += float(aux.detach())
                running_root += float(root_loss.detach())
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            running += float(batch_loss.detach())
            nb += 1
        scheduler.step()
        s = _eval_pred_stats(model, val, device=device, target_scale=target_scale,
                             block_by_token=block_by_token)
        s["epoch"] = epoch
        s["train_loss"] = running / max(1, nb)
        history.append(s)
        if s["mae"] < best_mae:
            best_mae = s["mae"]
            best_epoch = epoch
        log(f"epoch {epoch:3d}: tloss={running/nb:.4f} "
            f"(aux={running_aux/n_train:.4f} root={running_root/n_train:.4f})  "
            f"val_mae={s['mae']:.4f}  pred_std={s['pred_std']:.4f}  "
            f"corr={s['corr']:.3f}  aux_mse={s['aux_mse']:.4f}  "
            f"best={best_mae:.4f}@{best_epoch}  ({time.time()-t0:.1f}s)")

    return {
        "label": label,
        "best_val_mae": best_mae,
        "best_val_epoch": best_epoch,
        "final": history[-1],
        "n_params_total": n_params,
        "n_params_trainable": n_trainable,
        "history": history,
    }


def main() -> None:
    out_root = REPO / "outputs" / f"diag_per_position_aux_supervision_{_ts()}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"output dir: {out_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    log("loading recoverable_v5_t2048, train_docs=512, leaf_tokens=2048")
    train, val, _ = _load_split_docs(
        benchmark="recoverable_v5_t2048",
        train_docs=512,
        fixed_leaf_tokens=2048,
        seed=0,
    )
    log(f"train={len(train)} val={len(val)} leaves/doc={len(train[0].leaf_token_ids)}")
    target_scale = float(max(d.root_count for d in train))

    block_by_token = _palette_block_map(vocab_size=16, n_regimes=4)
    bnd_train = _precompute_boundary_tensors(train, block_by_token=block_by_token, device=device)
    bnd_val = _precompute_boundary_tensors(val, block_by_token=block_by_token, device=device)

    cells = []

    # C1: aux only, no root loss
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        block_by_token=block_by_token,
        bnd_train=bnd_train, bnd_val=bnd_val,
        use_aux=True, use_root=False,
        aux_weight=1.0, root_weight=0.0,
        epochs=40, lr=1e-3, bs=16,
        label="C1_aux_only",
        device=device, log=log,
    ))

    # C2: joint aux + root
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        block_by_token=block_by_token,
        bnd_train=bnd_train, bnd_val=bnd_val,
        use_aux=True, use_root=True,
        aux_weight=1.0, root_weight=1.0,
        epochs=40, lr=1e-3, bs=16,
        label="C2_aux_plus_root",
        device=device, log=log,
    ))

    # C3: root only, palette init, frozen embedding (replay of B1 for sanity)
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        block_by_token=block_by_token,
        bnd_train=bnd_train, bnd_val=bnd_val,
        use_aux=False, use_root=True,
        aux_weight=0.0, root_weight=1.0,
        epochs=40, lr=1e-3, bs=16,
        label="C3_root_only_baseline",
        device=device, log=log,
    ))

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "benchmark": "recoverable_v5_t2048",
            "train_docs": len(train),
            "val_docs": len(val),
            "leaf_tokens": 2048,
            "target_scale": target_scale,
            "cells": cells,
        }, f, indent=2)
    log(f"\nsummary saved to {summary_path}")

    log("\n=== headline ===")
    for c in cells:
        log(f"  {c['label']}: best_val_mae={c['best_val_mae']:.4f} "
            f"@epoch {c['best_val_epoch']}  "
            f"final_pred_std={c['final']['pred_std']:.4f}  "
            f"final_corr={c['final']['corr']:.3f}  "
            f"final_aux_mse={c['final']['aux_mse']:.4f}")


if __name__ == "__main__":
    main()
