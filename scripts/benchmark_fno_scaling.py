#!/usr/bin/env python3
"""
Focused FNO scaling benchmark: push toward zero error on the Markov
changepoint-count recoverable task.

Sweeps over data size, epochs, model width, and number of Fourier modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    _eval_root_predictions,
    _exact_match_rate,
    _token_sequence_arrays,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    HAS_NEURAL_OPERATOR,
    _class_setup,
)
from src.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)

if not HAS_NEURAL_OPERATOR:
    print("ERROR: neuraloperator not installed", file=sys.stderr)
    sys.exit(1)

from neuralop.models import FNO as _NeuralOpFNO


class FNOCountPredictorV2(nn.Module):
    """Enhanced FNO for changepoint counting with cosine LR and optional MSE loss."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        n_modes: int,
        width: int,
        n_layers: int,
        n_count_classes: int,
        target_max: float,
    ) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.target_max = float(target_max)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1, int(embed_dim), padding_idx=self.pad_id
        )
        self.input_proj = nn.Linear(int(embed_dim), int(width))
        self.fno = _NeuralOpFNO(
            n_modes=(int(n_modes),),
            in_channels=int(width),
            out_channels=int(width),
            hidden_channels=int(width),
            n_layers=int(n_layers),
        )
        head_hidden = max(64, int(width) // 2)
        self.count_classifier = nn.Sequential(
            nn.Linear(int(width), head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, int(n_count_classes)),
        )
        # Scalar regression head (alternative to classification)
        self.regression_head = nn.Sequential(
            nn.Linear(int(width), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def _pool(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        emb = self.token_embedding(tokens)
        x = self.input_proj(emb)
        x = x.permute(0, 2, 1)
        x = self.fno(x)
        mask = token_mask.unsqueeze(1)
        return (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    def forward_logits(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = self._pool(tokens, token_mask)
        return self.count_classifier(pooled)

    def forward_regression(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = self._pool(tokens, token_mask)
        return self.regression_head(pooled).squeeze(-1) * self.target_max


def train_fno(
    *,
    train_docs,
    val_docs,
    test_docs,
    vocab_size: int,
    embed_dim: int,
    width: int,
    n_modes: int,
    n_layers: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    loss_mode: str,  # "ce", "mse", or "ce+mse"
    device: torch.device,
    seed: int = 42,
) -> dict:
    pad_id = int(vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max, class_values, class_index, class_values_arr = _class_setup(
        train_y, val_y, test_y
    )

    model = FNOCountPredictorV2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_modes=n_modes,
        width=width,
        n_layers=n_layers,
        n_count_classes=len(class_values),
        target_max=target_max,
    ).to(device=device)

    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    rng = np.random.default_rng(seed)

    train_tok_t = torch.tensor(train_tokens, dtype=torch.long, device=device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_tok_t = torch.tensor(val_tokens, dtype=torch.long, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.float32, device=device)
    test_tok_t = torch.tensor(test_tokens, dtype=torch.long, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.float32, device=device)

    train_target_class = torch.tensor(
        [int(class_index[int(round(float(v)))]) for v in train_y.tolist()],
        dtype=torch.long, device=device,
    )
    train_target_scalar = torch.tensor(train_y.tolist(), dtype=torch.float32, device=device)

    n_train = int(train_tokens.shape[0])
    bs = min(batch_size, n_train)

    best_state = clone_module_state(model)
    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, bs):
            idx = torch.tensor(perm[start:start + bs], dtype=torch.long, device=device)
            b_tok = train_tok_t.index_select(0, idx)
            b_mask = train_mask_t.index_select(0, idx)
            b_class = train_target_class.index_select(0, idx)
            b_scalar = train_target_scalar.index_select(0, idx)

            opt.zero_grad(set_to_none=True)

            if loss_mode == "ce":
                logits = model.forward_logits(b_tok, token_mask=b_mask)
                loss = F.cross_entropy(logits, b_class)
            elif loss_mode == "mse":
                pred = model.forward_regression(b_tok, token_mask=b_mask)
                loss = F.mse_loss(pred, b_scalar)
            elif loss_mode == "ce+mse":
                logits = model.forward_logits(b_tok, token_mask=b_mask)
                pred = model.forward_regression(b_tok, token_mask=b_mask)
                loss = F.cross_entropy(logits, b_class) + 0.5 * F.mse_loss(pred, b_scalar)
            else:
                raise ValueError(f"unknown loss_mode: {loss_mode}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        # Validation
        if val_y.size > 0:
            model.eval()
            with torch.no_grad():
                if loss_mode == "mse":
                    val_pred_t = model.forward_regression(val_tok_t, token_mask=val_mask_t)
                    val_pred = val_pred_t.detach().cpu().numpy()
                else:
                    logits = model.forward_logits(val_tok_t, token_mask=val_mask_t)
                    pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                    val_pred = class_values_arr[pred_idx]
            val_mae = float(np.mean(np.abs(val_pred - val_y.astype(np.float64))))
            if val_mae < best_val_mae - 1e-9:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = clone_module_state(model)

    # Restore best
    if val_y.size > 0:
        restore_module_state(model, best_state)

    # Final eval
    model.eval()
    with torch.no_grad():
        if loss_mode == "mse":
            test_pred = model.forward_regression(test_tok_t, token_mask=test_mask_t).cpu().numpy()
            train_pred = model.forward_regression(train_tok_t, token_mask=train_mask_t).cpu().numpy()
        else:
            test_logits = model.forward_logits(test_tok_t, token_mask=test_mask_t)
            test_pred = class_values_arr[torch.argmax(test_logits, dim=1).cpu().numpy()]
            train_logits = model.forward_logits(train_tok_t, token_mask=train_mask_t)
            train_pred = class_values_arr[torch.argmax(train_logits, dim=1).cpu().numpy()]

    test_mae = float(np.mean(np.abs(test_pred - test_y.astype(np.float64))))
    test_exact = float(np.mean((np.rint(test_pred) == np.rint(test_y)).astype(np.float64)))
    train_mae = float(np.mean(np.abs(train_pred - train_y.astype(np.float64))))
    train_exact = float(np.mean((np.rint(train_pred) == np.rint(train_y)).astype(np.float64)))

    return {
        "test_root_mae": test_mae,
        "test_exact_match": test_exact,
        "train_root_mae": train_mae,
        "train_exact_match": train_exact,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO scaling benchmark")
    parser.add_argument("--bundle-1x", required=True, help="1x data bundle")
    parser.add_argument("--bundle-10x", default="", help="10x data bundle (optional)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/fno_scaling.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle_1x = MarkovOPSDataBundle.load(Path(args.bundle_1x))
    bundle_10x = None
    if args.bundle_10x.strip():
        p = Path(args.bundle_10x)
        if p.exists():
            bundle_10x = MarkovOPSDataBundle.load(p)

    configs = []

    # Sweep: epochs × data size × width × loss_mode
    for label_suffix, bundle in [("1x", bundle_1x)] + ([("10x", bundle_10x)] if bundle_10x else []):
        for n_epochs in [100, 200, 500]:
            for width in [64, 128]:
                for n_modes in [16, 32]:
                    for loss_mode in ["ce", "mse"]:
                        configs.append({
                            "label": f"fno_w{width}_m{n_modes}_{loss_mode}_{n_epochs}ep_{label_suffix}",
                            "bundle": bundle,
                            "n_epochs": n_epochs,
                            "width": width,
                            "n_modes": n_modes,
                            "loss_mode": loss_mode,
                            "embed_dim": max(64, width),
                            "n_layers": 4,
                            "batch_size": 32,
                            "lr": 3e-4 if loss_mode == "ce" else 1e-3,
                        })

    results = {}
    print(f"Running {len(configs)} configurations...\n")
    print(f"{'Config':<55} {'Test MAE':>10} {'Exact':>8} {'Train MAE':>10} {'Time':>6}")
    print("-" * 95)

    for cfg in configs:
        t0 = time.time()
        r = train_fno(
            train_docs=cfg["bundle"].train_docs,
            val_docs=cfg["bundle"].val_docs,
            test_docs=cfg["bundle"].test_docs,
            vocab_size=96,
            embed_dim=cfg["embed_dim"],
            width=cfg["width"],
            n_modes=cfg["n_modes"],
            n_layers=cfg["n_layers"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            loss_mode=cfg["loss_mode"],
            device=device,
        )
        dt = time.time() - t0
        r["wall_seconds"] = round(dt, 1)
        results[cfg["label"]] = r
        print(
            f"  {cfg['label']:<53} {r['test_root_mae']:>10.6f} {r['test_exact_match']:>8.4f}"
            f" {r['train_root_mae']:>10.6f} {dt:>5.0f}s"
        )
        sys.stdout.flush()

    # Summary: best configs
    print("\n" + "=" * 95)
    print("Top 5 by test MAE:")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["test_root_mae"])
    for label, r in ranked[:5]:
        print(f"  {label:<53} MAE={r['test_root_mae']:.6f}  exact={r['test_exact_match']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
