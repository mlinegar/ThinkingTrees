#!/usr/bin/env python3
"""Experiment B: palette-initialized embedding, train g + f.

Hypothesis test:
  Bigram OLS reaches MAE=0 with 17 params. Architecture is sum_pool-linear
  over per-position features. So the only hard part is producing per-position
  bigram-indicator features. We hand the model the embedding (the part that
  identifies palette blocks) and ask SGD to find the rest under root-only MSE.

Variants:
  - frozen=True : embedding weights frozen at palette one-hots
  - frozen=False: embedding initialized to palette one-hots, trainable

Channels are set equal to n_regimes so the one-hots fill the channel axis
exactly. n_regimes=4, channels=4. We also report a wider channel run
(channels=16, init_first_4_with_one_hot, rest=zero) for capacity comparison.

Output:
  outputs/diag_palette_init_embedding/<ts>/run.log + per-cell summary.
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
    root_mse_loss,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _eval_pred_stats(model, docs, *, device, target_scale):
    model.eval()
    preds, truths = [], []
    arrays = _doc_arrays(docs)
    with torch.no_grad():
        for di in range(len(docs)):
            tok = _to_tensor(arrays[0][di], dtype=torch.long, device=device)
            out = model(tok)
            preds.append(float(out.root_count_norm) * target_scale)
            truths.append(float(arrays[3][di]))
    p = np.asarray(preds); t = np.asarray(truths)
    return {
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "truth_mean": float(np.mean(t)),
        "truth_std": float(np.std(t)),
        "corr": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-6 else float("nan"),
    }


def _palette_init_embedding(model, *, vocab_size, n_regimes, freeze):
    """Set token_embedding to palette one-hot in first n_regimes channels;
    zero elsewhere. Optionally freeze."""
    block_by_token = _palette_block_map(vocab_size=vocab_size, n_regimes=n_regimes)
    emb = model.token_embedding.embedding
    C = emb.weight.shape[1]
    if C < n_regimes:
        raise ValueError(f"channels={C} < n_regimes={n_regimes}")
    new_w = torch.zeros_like(emb.weight)
    for tok_id, blk in enumerate(block_by_token):
        new_w[tok_id, int(blk)] = 1.0
    # padding row stays zero
    with torch.no_grad():
        emb.weight.copy_(new_w)
    if freeze:
        emb.weight.requires_grad = False


def run_cell(
    *,
    train,
    val,
    target_scale,
    channels,
    g_n_modes,
    g_n_layers,
    scorer_n_modes,
    scorer_n_layers,
    epochs,
    lr,
    bs,
    freeze_embedding,
    label,
    device,
    log,
):
    log(f"\n=== {label} ===")
    log(f"channels={channels} g_modes={g_n_modes} g_layers={g_n_layers} "
        f"scorer_modes={scorer_n_modes} scorer_layers={scorer_n_layers} "
        f"freeze_emb={freeze_embedding} epochs={epochs} lr={lr} bs={bs}")
    torch.manual_seed(0)
    model = CleanUnifiedNO(
        vocab_size=16,
        target_scale=target_scale,
        channels=channels,
        g_n_modes=g_n_modes,
        g_n_layers=g_n_layers,
        scorer_n_modes=scorer_n_modes,
        scorer_n_layers=scorer_n_layers,
        pooling_mode="sum",
    ).to(device)
    _palette_init_embedding(model, vocab_size=16, n_regimes=4, freeze=freeze_embedding)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params total={n_params:,}  trainable={n_trainable:,}  target_scale={target_scale:.1f}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    arrays = _doc_arrays(train)
    n_train = len(train)

    s = _eval_pred_stats(model, val, device=device, target_scale=target_scale)
    log(f"epoch  0 (init): val_mae={s['mae']:.4f}  pred_std={s['pred_std']:.4f}  "
        f"truth_std={s['truth_std']:.2f}  corr={s['corr']:.3f}")

    best_mae = float("inf")
    best_epoch = 0
    history = [{"epoch": 0, **s}]
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(n_train).tolist()
        running = 0.0
        nb = 0
        for bi in range(0, n_train, bs):
            optimizer.zero_grad()
            losses = []
            for di in order[bi:bi+bs]:
                tok = _to_tensor(arrays[0][di], dtype=torch.long, device=device)
                out = model(tok)
                root_t = _to_tensor(arrays[3][di], dtype=torch.float32, device=device)
                losses.append(root_mse_loss(out, root_count=root_t, target_scale=target_scale))
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            running += float(batch_loss.detach())
            nb += 1
        scheduler.step()
        s = _eval_pred_stats(model, val, device=device, target_scale=target_scale)
        s["epoch"] = epoch
        s["train_loss"] = running / max(1, nb)
        history.append(s)
        if s["mae"] < best_mae:
            best_mae = s["mae"]
            best_epoch = epoch
        log(f"epoch {epoch:3d}: tloss={running/nb:.4f}  val_mae={s['mae']:.4f}  "
            f"pred_std={s['pred_std']:.4f}  corr={s['corr']:.3f}  "
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
    out_root = REPO / "outputs" / f"diag_palette_init_embedding_{_ts()}"
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

    cells = []

    # B1: channels=4 == n_regimes, frozen palette one-hot embedding
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        channels=4, g_n_modes=16, g_n_layers=2,
        scorer_n_modes=8, scorer_n_layers=2,
        epochs=40, lr=1e-3, bs=16,
        freeze_embedding=True,
        label="B1_C4_frozen_emb",
        device=device, log=log,
    ))

    # B2: channels=4, palette one-hot init, embedding trainable
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        channels=4, g_n_modes=16, g_n_layers=2,
        scorer_n_modes=8, scorer_n_layers=2,
        epochs=40, lr=1e-3, bs=16,
        freeze_embedding=False,
        label="B2_C4_trainable_emb",
        device=device, log=log,
    ))

    # B3: channels=16, first-4 palette one-hot, rest zero, frozen embedding
    cells.append(run_cell(
        train=train, val=val, target_scale=target_scale,
        channels=16, g_n_modes=16, g_n_layers=2,
        scorer_n_modes=8, scorer_n_layers=2,
        epochs=40, lr=1e-3, bs=16,
        freeze_embedding=True,
        label="B3_C16_frozen_emb",
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
            f"final_corr={c['final']['corr']:.3f}")


if __name__ == "__main__":
    main()
