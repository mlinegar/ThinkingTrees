#!/usr/bin/env python3
"""Confirm prediction-std collapse for one-leaf root-only training.

Train a small CleanUnifiedNO on recoverable_v5_t2048 one-leaf for a few
epochs. Each epoch, log the std of the model's predictions on val.

If the constant-predictor hypothesis is right, pred_std should collapse
toward 0 while truth_std stays at ~4.4 and val MAE plateaus at ~3.45.
"""

from __future__ import annotations

import sys
import time
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
    _to_tensor,
)
from src.ctreepo.sim.core.clean_unified_fg import (  # noqa: E402
    CleanUnifiedNO,
    root_mse_loss,
)


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
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "truth_mean": float(np.mean(t)),
        "truth_std": float(np.std(t)),
        "corr": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-6 else float("nan"),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train, val, _ = _load_split_docs(
        benchmark="recoverable_v5_t2048",
        train_docs=512,
        fixed_leaf_tokens=2048,
        seed=0,
    )
    print(f"train={len(train)} val={len(val)} leaves/doc={len(train[0].leaf_token_ids)}")

    target_scale = float(max(d.root_count for d in train))
    model = CleanUnifiedNO(
        vocab_size=16,
        target_scale=target_scale,
        channels=64,
        g_n_modes=16,
        g_n_layers=2,
        scorer_n_modes=8,
        scorer_n_layers=2,
        pooling_mode="sum",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  target_scale={target_scale:.1f}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    arrays = _doc_arrays(train)
    n_train = len(train)
    bs = 8

    # Initial
    s = _eval_pred_stats(model, val, device=device, target_scale=target_scale)
    print(f"epoch  0 (init): val mae={s['mae']:.3f}  pred_mean={s['pred_mean']:.2f} "
          f"pred_std={s['pred_std']:.4f}  truth_std={s['truth_std']:.2f}  corr={s['corr']:.3f}")

    for epoch in range(1, 21):
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(batch_loss.detach())
            nb += 1
        scheduler.step()
        s = _eval_pred_stats(model, val, device=device, target_scale=target_scale)
        print(f"epoch {epoch:2d}: train_loss={running/nb:.4f}  val_mae={s['mae']:.3f}  "
              f"pred_mean={s['pred_mean']:.2f} pred_std={s['pred_std']:.4f}  "
              f"corr={s['corr']:.3f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
