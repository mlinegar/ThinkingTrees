#!/usr/bin/env python3
"""Probe the smallest non-trivial Markov tree: 2 leaves + 1 merge (the "triangle").

Builds a synthetic DGP with the same rule as recoverable_v5_t128
(regime = token // 4) but with configurable leaf size and a controllable
left/right structure. Trains end-to-end (leaf encoder + merge) on (count,
first, last) supervision at every node.

Sweeps:

  - leaf_encoder in {transition_table, mlp, fno}
  - merge in {additive_join_table, mlp}
  - count_head in {mse, ce}     # NEW: discrete classifier vs continuous regression
  - leaf_tokens in {4, 8, 16}

The question: at what point does the simplest triangle stop being solvable,
and does swapping the count head from MSE-regression to CE-classification
fix the count-memorization gap that shows up at L>=8?
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# ------------------------- synthetic DGP ---------------------------------------


def synthesize_triangle_docs(*, n_docs: int, leaf_tokens: int,
                             n_regimes: int = 4, vocab_per_regime: int = 4,
                             seed: int = 0):
    rng = np.random.default_rng(int(seed))
    docs = []
    for _ in range(n_docs):
        doc = {"leaves": [], "leaf_count": [], "leaf_first": [], "leaf_last": []}
        for _leaf in range(2):
            k = int(rng.integers(0, min(leaf_tokens, n_regimes)))
            positions = sorted(rng.choice(range(1, leaf_tokens), size=k, replace=False).tolist()) if k > 0 else []
            cur = int(rng.integers(0, n_regimes))
            regimes = []
            pos_idx = 0
            for t in range(leaf_tokens):
                if pos_idx < len(positions) and t == positions[pos_idx]:
                    new = int(rng.integers(0, n_regimes - 1))
                    if new >= cur: new += 1
                    cur = new
                    pos_idx += 1
                regimes.append(cur)
            tokens = [r * vocab_per_regime + int(rng.integers(0, vocab_per_regime)) for r in regimes]
            doc["leaves"].append(tokens)
            doc["leaf_count"].append(float(len(positions)))
            doc["leaf_first"].append(int(regimes[0]))
            doc["leaf_last"].append(int(regimes[-1]))
        c0, c1 = doc["leaf_count"]
        l0_last = doc["leaf_last"][0]
        l1_first = doc["leaf_first"][1]
        doc["root_count"] = float(c0 + c1 + (1.0 if l0_last != l1_first else 0.0))
        doc["root_first"] = doc["leaf_first"][0]
        doc["root_last"] = doc["leaf_last"][1]
        doc["join_bit"] = 1 if l0_last != l1_first else 0
        docs.append(doc)
    return docs


def docs_to_tensors(docs, device):
    tokens = torch.tensor(
        [[d["leaves"][0], d["leaves"][1]] for d in docs],
        dtype=torch.long, device=device,
    )
    leaf_count = torch.tensor(
        [d["leaf_count"] for d in docs], dtype=torch.float32, device=device,
    )
    leaf_first = torch.tensor(
        [d["leaf_first"] for d in docs], dtype=torch.long, device=device,
    )
    leaf_last = torch.tensor(
        [d["leaf_last"] for d in docs], dtype=torch.long, device=device,
    )
    root_count = torch.tensor([d["root_count"] for d in docs], dtype=torch.float32, device=device)
    root_first = torch.tensor([d["root_first"] for d in docs], dtype=torch.long, device=device)
    root_last = torch.tensor([d["root_last"] for d in docs], dtype=torch.long, device=device)
    join_bit = torch.tensor([d["join_bit"] for d in docs], dtype=torch.float32, device=device)
    return tokens, leaf_count, leaf_first, leaf_last, root_count, root_first, root_last, join_bit


# ------------------------- count head helpers ----------------------------------


def _ce_count_head(in_dim: int, max_count: int) -> nn.Linear:
    return nn.Linear(int(in_dim), int(max_count) + 1)


def _expected_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: (..., max_count+1) -> expected scalar
    probs = torch.softmax(logits, dim=-1)
    values = torch.arange(probs.size(-1), dtype=probs.dtype, device=probs.device)
    return (probs * values).sum(dim=-1)


# ------------------------- leaf encoders ---------------------------------------


class TransitionTableLeaf(nn.Module):
    """Per-token regime classifier; count derived from transition formula."""
    def __init__(self, vocab_size: int, n_regimes: int, count_head_kind: str = "mse",
                 max_leaf_count: int | None = None):
        super().__init__()
        self.token_to_regime = nn.Embedding(int(vocab_size), int(n_regimes))
        self.n_regimes = int(n_regimes)
        self.count_head_kind = count_head_kind
        # transition_table doesn't need a separate count head — count is derived.
        # When count_head_kind == 'ce', we still report count_logits via differentiable
        # discretization of the expected count for diagnostic comparability.
        self.max_leaf_count = max_leaf_count

    def forward(self, tokens):
        logits = self.token_to_regime(tokens)
        probs = torch.softmax(logits, dim=-1)
        same = (probs[..., :-1, :] * probs[..., 1:, :]).sum(dim=-1)
        count = (1.0 - same).sum(dim=-1)
        first = logits[..., 0, :]
        last = logits[..., -1, :]
        # No CE count logits for transition_table (count is derived structurally).
        return {"count": count, "count_logits": None, "first": first, "last": last}


class MLPLeaf(nn.Module):
    def __init__(self, vocab_size: int, n_regimes: int, leaf_tokens: int,
                 count_head_kind: str = "mse", max_leaf_count: int | None = None,
                 embed_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.net = nn.Sequential(
            nn.Linear(int(leaf_tokens) * int(embed_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.count_head_kind = count_head_kind
        if count_head_kind == "mse":
            self.count_head = nn.Linear(int(hidden_dim), 1)
        elif count_head_kind == "ce":
            assert max_leaf_count is not None
            self.count_head = _ce_count_head(int(hidden_dim), int(max_leaf_count))
        else:
            raise ValueError(f"unknown count_head_kind: {count_head_kind!r}")
        self.first_head = nn.Linear(int(hidden_dim), int(n_regimes))
        self.last_head = nn.Linear(int(hidden_dim), int(n_regimes))
        self.embed_dim = int(embed_dim)
        self.leaf_tokens = int(leaf_tokens)

    def forward(self, tokens):
        B, L, T = tokens.shape
        emb = self.embedding(tokens).reshape(B * L, T * self.embed_dim)
        h = self.net(emb)
        if self.count_head_kind == "mse":
            c = self.count_head(h).squeeze(-1)
            c_logits = None
        else:
            c_logits = self.count_head(h)
            c = _expected_from_logits(c_logits)
            c_logits = c_logits.reshape(B, L, -1)
        c = c.reshape(B, L)
        f = self.first_head(h).reshape(B, L, -1)
        ll = self.last_head(h).reshape(B, L, -1)
        return {"count": c, "count_logits": c_logits, "first": f, "last": ll}


class SpectralConv1d(nn.Module):
    def __init__(self, ic: int, oc: int, n_modes: int):
        super().__init__()
        self.ic, self.oc, self.n_modes = int(ic), int(oc), int(n_modes)
        scale = 1.0 / (self.ic * self.oc)
        self.weights = nn.Parameter(scale * torch.randn(self.ic, self.oc, self.n_modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.oc, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        m = min(self.n_modes, x_ft.size(-1))
        out_ft[..., :m] = torch.einsum("bcm,com->bom", x_ft[..., :m], self.weights[..., :m])
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class FNOLeaf(nn.Module):
    def __init__(self, vocab_size: int, n_regimes: int,
                 count_head_kind: str = "mse", max_leaf_count: int | None = None,
                 fno_width: int = 64, n_modes: int = 4, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(fno_width))
        self.layers = nn.ModuleList([
            SpectralConv1d(fno_width, fno_width, n_modes) for _ in range(int(n_layers))
        ])
        self.skip = nn.ModuleList([
            nn.Conv1d(fno_width, fno_width, 1) for _ in range(int(n_layers))
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, fno_width) for _ in range(int(n_layers))
        ])
        self.count_head_kind = count_head_kind
        if count_head_kind == "mse":
            self.count_head = nn.Sequential(
                nn.Linear(fno_width, fno_width), nn.GELU(), nn.Linear(fno_width, 1),
            )
        elif count_head_kind == "ce":
            assert max_leaf_count is not None
            self.count_head = nn.Sequential(
                nn.Linear(fno_width, fno_width), nn.GELU(),
                _ce_count_head(fno_width, int(max_leaf_count)),
            )
        else:
            raise ValueError(f"unknown count_head_kind: {count_head_kind!r}")
        self.first_head = nn.Linear(fno_width, int(n_regimes))
        self.last_head = nn.Linear(fno_width, int(n_regimes))
        self.fno_width = int(fno_width)

    def forward(self, tokens):
        B, L, T = tokens.shape
        x = self.embedding(tokens).reshape(B * L, T, self.fno_width).transpose(1, 2)
        for layer, skip, norm in zip(self.layers, self.skip, self.norms):
            x = norm(F.gelu(layer(x) + skip(x)))
        h_mean = x.mean(dim=-1)
        h_first = x[..., 0]
        h_last = x[..., -1]
        if self.count_head_kind == "mse":
            c = self.count_head(h_mean).squeeze(-1)
            c_logits = None
        else:
            c_logits = self.count_head(h_mean)
            c = _expected_from_logits(c_logits)
            c_logits = c_logits.reshape(B, L, -1)
        c = c.reshape(B, L)
        f = self.first_head(h_first).reshape(B, L, -1)
        ll = self.last_head(h_last).reshape(B, L, -1)
        return {"count": c, "count_logits": c_logits, "first": f, "last": ll}


class StructuralMLPLeaf(nn.Module):
    """MLP that produces per-token regime logits; count derived via transition formula.

    Like transition_table, but with a per-token MLP feature extractor instead of
    an embedding lookup. Tests whether the structural inductive bias for count
    transfers to a richer-capacity per-token feature path.
    """
    def __init__(self, vocab_size: int, n_regimes: int, leaf_tokens: int,
                 count_head_kind: str = "mse", max_leaf_count: int | None = None,
                 embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.token_net = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(n_regimes)),
        )
        self.n_regimes = int(n_regimes)
        self.count_head_kind = count_head_kind  # consumed but ignored: count is structural
        self.leaf_tokens = int(leaf_tokens)

    def forward(self, tokens):
        B, L, T = tokens.shape
        emb = self.embedding(tokens)  # (B, L, T, E)
        logits = self.token_net(emb)  # (B, L, T, R)
        probs = torch.softmax(logits, dim=-1)
        same = (probs[..., :-1, :] * probs[..., 1:, :]).sum(dim=-1)
        count = (1.0 - same).sum(dim=-1)  # (B, L)
        first = logits[..., 0, :]  # (B, L, R)
        last = logits[..., -1, :]
        return {"count": count, "count_logits": None, "first": first, "last": last}


class StructuralFNOLeaf(nn.Module):
    """FNO that emits per-token regime logits via Linear(C, R); count derived structurally."""
    def __init__(self, vocab_size: int, n_regimes: int,
                 count_head_kind: str = "mse", max_leaf_count: int | None = None,
                 fno_width: int = 64, n_modes: int = 4, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(fno_width))
        self.layers = nn.ModuleList([
            SpectralConv1d(fno_width, fno_width, n_modes) for _ in range(int(n_layers))
        ])
        self.skip = nn.ModuleList([
            nn.Conv1d(fno_width, fno_width, 1) for _ in range(int(n_layers))
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, fno_width) for _ in range(int(n_layers))
        ])
        self.regime_head = nn.Linear(int(fno_width), int(n_regimes))
        self.n_regimes = int(n_regimes)
        self.count_head_kind = count_head_kind
        self.fno_width = int(fno_width)

    def forward(self, tokens):
        B, L, T = tokens.shape
        x = self.embedding(tokens).reshape(B * L, T, self.fno_width).transpose(1, 2)
        for layer, skip, norm in zip(self.layers, self.skip, self.norms):
            x = norm(F.gelu(layer(x) + skip(x)))
        # x: (B*L, C, T) -> per-position regime logits via Linear(C, R)
        per_token_logits = self.regime_head(x.transpose(1, 2))  # (B*L, T, R)
        per_token_logits = per_token_logits.reshape(B, L, T, -1)
        probs = torch.softmax(per_token_logits, dim=-1)
        same = (probs[..., :-1, :] * probs[..., 1:, :]).sum(dim=-1)
        count = (1.0 - same).sum(dim=-1)
        first = per_token_logits[..., 0, :]
        last = per_token_logits[..., -1, :]
        return {"count": count, "count_logits": None, "first": first, "last": last}


# ------------------------- merge -----------------------------------------------


class AdditiveJoinTableMerge(nn.Module):
    """Merge that adds counts and an expected join bit. When count_head_kind=='ce',
    we ALSO emit a discrete root-count classifier head over {0..max_root_count}
    fed by leaf count logits (when available), expected counts, and endpoints."""
    def __init__(self, n_regimes: int, count_head_kind: str = "mse",
                 max_leaf_count: int | None = None, max_root_count: int | None = None):
        super().__init__()
        self.join_logits = nn.Parameter(torch.zeros(int(n_regimes), int(n_regimes)))
        self.count_head_kind = count_head_kind
        self.n_regimes = int(n_regimes)
        if count_head_kind == "ce":
            assert max_leaf_count is not None and max_root_count is not None
            self.max_root_count = int(max_root_count)
            in_dim = 2 * (int(max_leaf_count) + 1) + 2 * int(n_regimes) + 2
            self.root_count_head = nn.Sequential(
                nn.Linear(in_dim, 128), nn.GELU(),
                nn.Linear(128, int(max_root_count) + 1),
            )

    def forward(self, l, r):
        # l, r: dicts with count, count_logits, first, last
        l_last_p = torch.softmax(l["last"], dim=-1)
        r_first_p = torch.softmax(r["first"], dim=-1)
        join = torch.sigmoid(self.join_logits)
        expected_join = torch.einsum("...i,ij,...j->...", l_last_p, join, r_first_p)
        merged_count = l["count"] + r["count"] + expected_join
        out = {"count": merged_count, "count_logits": None,
               "first": l["first"], "last": r["last"]}
        if self.count_head_kind == "ce" and l["count_logits"] is not None and r["count_logits"] is not None:
            in_dim = self.root_count_head[0].in_features
            need_dim = (in_dim - 2 * self.n_regimes - 2) // 2
            l_logits = l["count_logits"][..., :need_dim]
            r_logits = r["count_logits"][..., :need_dim]
            feats = torch.cat([
                l_logits, r_logits, l_last_p, r_first_p,
                expected_join.unsqueeze(-1), (l["count"] + r["count"]).unsqueeze(-1),
            ], dim=-1)
            out["count_logits"] = self.root_count_head(feats)
        return out


class MLPMerge(nn.Module):
    def __init__(self, n_regimes: int, count_head_kind: str = "mse",
                 max_leaf_count: int | None = None, max_root_count: int | None = None,
                 hidden_dim: int = 128):
        super().__init__()
        self.count_head_kind = count_head_kind
        self.n_regimes = int(n_regimes)
        self.feat_dim = 2 + 2 * int(n_regimes)
        self.body = nn.Sequential(
            nn.Linear(self.feat_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.hidden_dim = int(hidden_dim)
        if count_head_kind == "mse":
            self.head = nn.Linear(int(hidden_dim), 1)
        elif count_head_kind == "ce":
            assert max_root_count is not None
            self.max_root_count = int(max_root_count)
            self.head = nn.Linear(int(hidden_dim), int(max_root_count) + 1)
        else:
            raise ValueError(f"unknown count_head_kind: {count_head_kind!r}")

    def forward(self, l, r):
        l_last_p = torch.softmax(l["last"], dim=-1)
        r_first_p = torch.softmax(r["first"], dim=-1)
        feats = torch.cat([
            l["count"].unsqueeze(-1), r["count"].unsqueeze(-1),
            l_last_p, r_first_p,
        ], dim=-1)
        h = self.body(feats)
        if self.count_head_kind == "mse":
            c = self.head(h).squeeze(-1)
            c_logits = None
        else:
            c_logits = self.head(h)
            c = _expected_from_logits(c_logits)
        return {"count": c, "count_logits": c_logits, "first": l["first"], "last": r["last"]}


class TriangleModel(nn.Module):
    def __init__(self, leaf_encoder: nn.Module, merge: nn.Module):
        super().__init__()
        self.leaf = leaf_encoder
        self.merge = merge

    def forward(self, tokens):
        out = self.leaf(tokens)
        l = {k: out[k][:, 0] if out[k] is not None else None for k in out}
        r = {k: out[k][:, 1] if out[k] is not None else None for k in out}
        merged = self.merge(l, r)
        return {
            "leaf_count": out["count"], "leaf_count_logits": out["count_logits"],
            "leaf_first": out["first"], "leaf_last": out["last"],
            "root_count": merged["count"], "root_count_logits": merged["count_logits"],
            "root_first": merged["first"], "root_last": merged["last"],
        }


# ------------------------- training & eval -------------------------------------


def joint_loss(out, leaf_count, leaf_first, leaf_last, root_count, root_first, root_last,
               count_head_kind: str = "mse", count_loss_scale: float = 1.0):
    if count_head_kind == "mse" or out["leaf_count_logits"] is None:
        leaf_count_loss = F.mse_loss(out["leaf_count"], leaf_count)
    else:
        leaf_count_loss = F.cross_entropy(
            out["leaf_count_logits"].reshape(-1, out["leaf_count_logits"].size(-1)),
            leaf_count.reshape(-1).long().clamp(0, out["leaf_count_logits"].size(-1) - 1),
        )
    leaf_loss = (
        float(count_loss_scale) * leaf_count_loss
        + F.cross_entropy(out["leaf_first"].reshape(-1, out["leaf_first"].size(-1)), leaf_first.reshape(-1))
        + F.cross_entropy(out["leaf_last"].reshape(-1, out["leaf_last"].size(-1)), leaf_last.reshape(-1))
    )
    if count_head_kind == "mse" or out["root_count_logits"] is None:
        root_count_loss = F.mse_loss(out["root_count"], root_count)
    else:
        root_count_loss = F.cross_entropy(
            out["root_count_logits"],
            root_count.long().clamp(0, out["root_count_logits"].size(-1) - 1),
        )
    root_loss = (
        float(count_loss_scale) * root_count_loss
        + F.cross_entropy(out["root_first"], root_first)
        + F.cross_entropy(out["root_last"], root_last)
    )
    return leaf_loss + root_loss


@torch.no_grad()
def eval_model(model, tokens, leaf_count, leaf_first, leaf_last,
               root_count, root_first, root_last, batch_size=256):
    model.eval()
    metrics = {
        "leaf_count_exact": 0, "leaf_first_acc": 0, "leaf_last_acc": 0,
        "root_count_exact": 0, "root_first_acc": 0, "root_last_acc": 0,
        "root_count_mae": 0.0,
    }
    n_leaf = leaf_count.numel()
    n_root = root_count.numel()
    for i in range(0, tokens.size(0), batch_size):
        ts = tokens[i:i+batch_size]
        out = model(ts)
        # Predicted counts: argmax if logits available, else round of expected
        if out["leaf_count_logits"] is not None:
            pred_lc = out["leaf_count_logits"].argmax(-1).float()
        else:
            pred_lc = out["leaf_count"].round()
        pred_lf = out["leaf_first"].argmax(-1)
        pred_ll = out["leaf_last"].argmax(-1)
        if out["root_count_logits"] is not None:
            pred_rc = out["root_count_logits"].argmax(-1).float()
        else:
            pred_rc = out["root_count"].round()
        pred_rf = out["root_first"].argmax(-1)
        pred_rl = out["root_last"].argmax(-1)
        metrics["leaf_count_exact"] += (pred_lc == leaf_count[i:i+batch_size]).sum().item()
        metrics["leaf_first_acc"] += (pred_lf == leaf_first[i:i+batch_size]).sum().item()
        metrics["leaf_last_acc"] += (pred_ll == leaf_last[i:i+batch_size]).sum().item()
        metrics["root_count_exact"] += (pred_rc == root_count[i:i+batch_size]).sum().item()
        metrics["root_first_acc"] += (pred_rf == root_first[i:i+batch_size]).sum().item()
        metrics["root_last_acc"] += (pred_rl == root_last[i:i+batch_size]).sum().item()
        metrics["root_count_mae"] += (out["root_count"] - root_count[i:i+batch_size]).abs().sum().item()
    return {
        "leaf_count_exact": metrics["leaf_count_exact"] / n_leaf,
        "leaf_first_acc": metrics["leaf_first_acc"] / n_leaf,
        "leaf_last_acc": metrics["leaf_last_acc"] / n_leaf,
        "root_count_exact": metrics["root_count_exact"] / n_root,
        "root_first_acc": metrics["root_first_acc"] / n_root,
        "root_last_acc": metrics["root_last_acc"] / n_root,
        "root_count_mae": metrics["root_count_mae"] / n_root,
    }


def train_triangle(model, train, val, *, epochs: int, lr: float, batch_size: int = 64,
                   device="cpu", count_head_kind: str = "mse", count_loss_scale: float = 1.0,
                   count_loss_warmup_epochs: int = 0):
    """Single-L training: train and val are tuples of doc tensors at one L."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_train = train[0].size(0)
    history = []
    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0
        n_batches = 0
        # Linear warmup of count loss scale from 0 -> count_loss_scale.
        if count_loss_warmup_epochs > 0:
            ramp = min(1.0, max(0.0, (ep - 1) / float(count_loss_warmup_epochs)))
            effective_count_loss_scale = float(count_loss_scale) * ramp
        else:
            effective_count_loss_scale = float(count_loss_scale)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch = [t[idx] for t in train]
            out = model(batch[0])
            loss = joint_loss(out, *batch[1:7], count_head_kind=count_head_kind,
                              count_loss_scale=effective_count_loss_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        train_m = eval_model(model, *train[:7])
        val_m = eval_model(model, *val[:7])
        history.append({
            "epoch": ep, "loss": ep_loss / max(1, n_batches),
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        score = -(val_m["root_count_exact"] + val_m["root_first_acc"] + val_m["root_last_acc"])
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_triangle_multiL(model, trains_by_L, vals_by_L, *, epochs: int, lr: float,
                          batch_size: int = 64, device="cpu", count_head_kind: str = "mse",
                          count_loss_scale: float = 1.0, count_loss_warmup_epochs: int = 0):
    """Multi-L training: trains_by_L and vals_by_L are dicts {L: doc tensors}.
    Each batch samples a random L (uniformly) and pulls a batch from that L's training set.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    L_list = sorted(trains_by_L.keys())
    n_train_per_L = {L: trains_by_L[L][0].size(0) for L in L_list}
    history = []
    best_val = float("inf")
    best_state = None
    rng = np.random.default_rng(0)
    # Total batches per epoch = sum across all L's
    n_batches_per_epoch = sum(max(1, n // batch_size) for n in n_train_per_L.values())
    for ep in range(1, epochs + 1):
        model.train()
        if count_loss_warmup_epochs > 0:
            ramp = min(1.0, max(0.0, (ep - 1) / float(count_loss_warmup_epochs)))
            effective_count_loss_scale = float(count_loss_scale) * ramp
        else:
            effective_count_loss_scale = float(count_loss_scale)
        # Build a shuffled list of (L, batch_index_in_L) pairs covering one pass over all L's
        epoch_batches = []
        for L in L_list:
            n = n_train_per_L[L]
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                epoch_batches.append((L, perm[i:i+batch_size]))
        rng.shuffle(epoch_batches)
        ep_loss = 0.0
        n_steps = 0
        for L, idx in epoch_batches:
            batch = [t[idx] for t in trains_by_L[L]]
            out = model(batch[0])
            loss = joint_loss(out, *batch[1:7], count_head_kind=count_head_kind,
                              count_loss_scale=effective_count_loss_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_steps += 1
        # Evaluate on each L
        ep_record = {"epoch": ep, "loss": ep_loss / max(1, n_steps)}
        val_score_sum = 0.0
        for L in L_list:
            tm = eval_model(model, *trains_by_L[L][:7])
            vm = eval_model(model, *vals_by_L[L][:7])
            ep_record[f"L{L}_train_root_count_exact"] = tm["root_count_exact"]
            ep_record[f"L{L}_val_root_count_exact"] = vm["root_count_exact"]
            ep_record[f"L{L}_val_root_first_acc"] = vm["root_first_acc"]
            ep_record[f"L{L}_val_root_last_acc"] = vm["root_last_acc"]
            val_score_sum += vm["root_count_exact"] + vm["root_first_acc"] + vm["root_last_acc"]
        history.append(ep_record)
        score = -val_score_sum / max(1, len(L_list))
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def make_leaf(kind: str, vocab_size: int, n_regimes: int, leaf_tokens: int,
              count_head_kind: str, max_leaf_count: int):
    if kind == "transition_table":
        return TransitionTableLeaf(vocab_size, n_regimes,
                                   count_head_kind=count_head_kind,
                                   max_leaf_count=max_leaf_count)
    if kind == "mlp":
        return MLPLeaf(vocab_size, n_regimes, leaf_tokens,
                       count_head_kind=count_head_kind, max_leaf_count=max_leaf_count)
    if kind == "fno":
        return FNOLeaf(vocab_size, n_regimes,
                       count_head_kind=count_head_kind, max_leaf_count=max_leaf_count)
    if kind == "mlp_structural":
        return StructuralMLPLeaf(vocab_size, n_regimes, leaf_tokens,
                                 count_head_kind=count_head_kind,
                                 max_leaf_count=max_leaf_count)
    if kind == "fno_structural":
        return StructuralFNOLeaf(vocab_size, n_regimes,
                                 count_head_kind=count_head_kind,
                                 max_leaf_count=max_leaf_count)
    raise ValueError(f"unknown leaf encoder: {kind}")


def make_merge(kind: str, n_regimes: int, count_head_kind: str,
               max_leaf_count: int, max_root_count: int):
    if kind == "additive_join_table":
        return AdditiveJoinTableMerge(n_regimes, count_head_kind=count_head_kind,
                                      max_leaf_count=max_leaf_count,
                                      max_root_count=max_root_count)
    if kind == "mlp":
        return MLPMerge(n_regimes, count_head_kind=count_head_kind,
                        max_leaf_count=max_leaf_count, max_root_count=max_root_count)
    raise ValueError(f"unknown merge: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-tokens", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--leaf-encoders", type=str, nargs="+",
                        default=["transition_table", "mlp", "fno"])
    parser.add_argument("--merges", type=str, nargs="+",
                        default=["additive_join_table", "mlp"])
    parser.add_argument("--count-heads", type=str, nargs="+", default=["mse", "ce"],
                        help="Which count head variants to sweep.")
    parser.add_argument("--count-loss-scale", type=float, default=1.0,
                        help="Multiplier on count-loss terms (helps structural at large L).")
    parser.add_argument("--count-loss-warmup-epochs", type=int, default=0,
                        help="Linearly ramp count-loss scale from 0 over the first N epochs.")
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-per-regime", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "outputs" / f"markov_triangle_probes_{_ts()}"))
    parser.add_argument("--length-transfer-eval", type=int, nargs="*", default=None,
                        help="Additional leaf_tokens to evaluate the trained model on (length-transfer test).")
    parser.add_argument("--multi-l-train", type=int, nargs="*", default=None,
                        help="If set, train on a MIXTURE of these leaf sizes (each batch picks one L). "
                             "Replaces --leaf-tokens for the training step but the model is then evaluated "
                             "on each L individually plus any --length-transfer-eval targets.")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root}")

    device = torch.device(args.device)
    vocab_size = int(args.n_regimes) * int(args.vocab_per_regime)

    summary = {"args": vars(args), "results": []}

    # Multi-L training path (optional)
    if args.multi_l_train:
        train_L_list = sorted(set(int(L) for L in args.multi_l_train))
        max_leaf_count = max(train_L_list)
        max_root_count = 2 * max_leaf_count + 1
        # Pre-generate per-L train/val/test docs
        trains_by_L = {}
        vals_by_L = {}
        tests_by_L = {}
        for L in train_L_list:
            train_docs = synthesize_triangle_docs(
                n_docs=int(args.n_train), leaf_tokens=int(L),
                n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
                seed=int(args.seed),
            )
            val_docs = synthesize_triangle_docs(
                n_docs=int(args.n_val), leaf_tokens=int(L),
                n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
                seed=int(args.seed) + 1,
            )
            test_docs = synthesize_triangle_docs(
                n_docs=int(args.n_test), leaf_tokens=int(L),
                n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
                seed=int(args.seed) + 2,
            )
            trains_by_L[L] = docs_to_tensors(train_docs, device)
            vals_by_L[L] = docs_to_tensors(val_docs, device)
            tests_by_L[L] = docs_to_tensors(test_docs, device)

        # Length-flexible models only
        for leaf_kind in args.leaf_encoders:
            if leaf_kind in {"mlp", "fno"}:
                print(f"skipping {leaf_kind} for multi-L training (fixed-length encoder)")
                continue
            for merge_kind in args.merges:
                for count_head_kind in args.count_heads:
                    tag = f"multiL_{'-'.join(str(L) for L in train_L_list)}_{leaf_kind}_{merge_kind}_count_{count_head_kind}"
                    print(f"\n=== {tag} ===")
                    leaf = make_leaf(leaf_kind, vocab_size, int(args.n_regimes), int(train_L_list[0]),
                                     count_head_kind, max_leaf_count)
                    merge = make_merge(merge_kind, int(args.n_regimes),
                                       count_head_kind, max_leaf_count, max_root_count)
                    model = TriangleModel(leaf, merge).to(device)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  params: {n_params:,}")

                    history = train_triangle_multiL(model, trains_by_L, vals_by_L,
                                                    epochs=int(args.epochs), lr=float(args.lr),
                                                    batch_size=64, device=device,
                                                    count_head_kind=count_head_kind,
                                                    count_loss_scale=float(args.count_loss_scale),
                                                    count_loss_warmup_epochs=int(args.count_loss_warmup_epochs))
                    # Test on each train_L
                    per_L_test = {}
                    for L in train_L_list:
                        m = eval_model(model, *tests_by_L[L][:7])
                        per_L_test[f"L{L}"] = m
                        print(f"  test  L{L}: root_count={m['root_count_exact']:.4f} "
                              f"first={m['root_first_acc']:.4f} last={m['root_last_acc']:.4f}")

                    # Length-transfer to OOD L's
                    length_transfer = {}
                    transfer_targets = []
                    if args.length_transfer_eval is not None:
                        transfer_targets = [int(L_eval) for L_eval in args.length_transfer_eval
                                            if int(L_eval) not in train_L_list]
                    for L_eval in transfer_targets:
                        try:
                            transfer_test_docs = synthesize_triangle_docs(
                                n_docs=int(args.n_test), leaf_tokens=int(L_eval),
                                n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
                                seed=int(args.seed) + 2,
                            )
                            transfer_test = docs_to_tensors(transfer_test_docs, device)
                            metrics = eval_model(model, *transfer_test[:7])
                            length_transfer[f"L{L_eval}"] = metrics
                            print(f"  transfer L{L_eval}: root_count={metrics['root_count_exact']:.4f} "
                                  f"first={metrics['root_first_acc']:.4f} last={metrics['root_last_acc']:.4f}")
                        except Exception as e:
                            length_transfer[f"L{L_eval}"] = {"error": f"{type(e).__name__}: {e}"}
                            print(f"  transfer L{L_eval}: ERROR {e}")

                    cell_dir = out_root / tag
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), cell_dir / "model_state.pt")
                    (cell_dir / "history.json").write_text(json.dumps(history, indent=2))
                    summary["results"].append({
                        "multi_L_train": train_L_list,
                        "leaf_encoder": leaf_kind, "merge": merge_kind,
                        "count_head": count_head_kind, "n_params": int(n_params),
                        "per_L_test": per_L_test,
                        "length_transfer": length_transfer,
                    })

        # Write summary and exit (skip the single-L loop)
        md = ["# Markov Triangle Probe — Multi-L Training", "",
              f"Synthetic DGP: regime = token // {args.vocab_per_regime}, n_regimes={args.n_regimes}.",
              f"Trained on a MIXTURE of L = {train_L_list}.",
              f"n_train={args.n_train} per L, n_val={args.n_val} per L, epochs={args.epochs}, lr={args.lr}.", "",
              "## Per-L test (each cell trained on the mixture, evaluated separately at each L)", "",
              "| leaf_enc | merge | count_head | params | "
              + " | ".join(f"L{L} root_count" for L in train_L_list) + " |",
              "|---|---|---|---:|" + "---:|" * len(train_L_list)]
        for r in summary["results"]:
            cells = [f"{r['per_L_test'][f'L{L}']['root_count_exact']:.4f}" for L in train_L_list]
            md.append(f"| {r['leaf_encoder']} | {r['merge']} | {r['count_head']} | {r['n_params']:,} | "
                      + " | ".join(cells) + " |")
        if any(r.get("length_transfer") for r in summary["results"]):
            xfer_targets = sorted({k for r in summary["results"] for k in (r.get("length_transfer") or {}).keys()},
                                   key=lambda x: int(x[1:]))
            md += ["", "## Length transfer (zero-shot eval at L not in training mixture)", "",
                   "| leaf_enc | merge | count_head | "
                   + " | ".join(f"eval {t}" for t in xfer_targets) + " |",
                   "|---|---|---|" + "---:|" * len(xfer_targets)]
            for r in summary["results"]:
                cells = []
                for t in xfer_targets:
                    m = (r.get("length_transfer") or {}).get(t, {})
                    if "error" in m:
                        cells.append("ERR")
                    elif m:
                        cells.append(f"{m['root_first_acc']:.3f}/{m['root_last_acc']:.3f}/{m['root_count_exact']:.3f}")
                    else:
                        cells.append("-")
                md.append(f"| {r['leaf_encoder']} | {r['merge']} | {r['count_head']} | "
                          + " | ".join(cells) + " |")
        md += [""]
        (out_root / "report.md").write_text("\n".join(md) + "\n")
        (out_root / "report.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote: {out_root / 'report.md'}")
        return

    for L in args.leaf_tokens:
        max_leaf_count = int(L)  # at most L-1 changepoints, but reserve L+1 buckets
        max_root_count = 2 * int(L) + 1
        train_docs = synthesize_triangle_docs(
            n_docs=int(args.n_train), leaf_tokens=int(L),
            n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
            seed=int(args.seed),
        )
        val_docs = synthesize_triangle_docs(
            n_docs=int(args.n_val), leaf_tokens=int(L),
            n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
            seed=int(args.seed) + 1,
        )
        test_docs = synthesize_triangle_docs(
            n_docs=int(args.n_test), leaf_tokens=int(L),
            n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
            seed=int(args.seed) + 2,
        )
        train = docs_to_tensors(train_docs, device)
        val = docs_to_tensors(val_docs, device)
        test = docs_to_tensors(test_docs, device)

        for leaf_kind in args.leaf_encoders:
            for merge_kind in args.merges:
                for count_head_kind in args.count_heads:
                    tag = f"L{L}_{leaf_kind}_{merge_kind}_count_{count_head_kind}"
                    print(f"\n=== {tag} ===")
                    leaf = make_leaf(leaf_kind, vocab_size, int(args.n_regimes), int(L),
                                     count_head_kind, max_leaf_count)
                    merge = make_merge(merge_kind, int(args.n_regimes),
                                       count_head_kind, max_leaf_count, max_root_count)
                    model = TriangleModel(leaf, merge).to(device)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  params: {n_params:,}")

                    history = train_triangle(model, train, val,
                                             epochs=int(args.epochs), lr=float(args.lr),
                                             batch_size=64, device=device,
                                             count_head_kind=count_head_kind,
                                             count_loss_scale=float(args.count_loss_scale),
                                             count_loss_warmup_epochs=int(args.count_loss_warmup_epochs))
                    test_m = eval_model(model, *test[:7])
                    train_m = eval_model(model, *train[:7])
                    print(f"  test:  root_count_exact={test_m['root_count_exact']:.4f} "
                          f"root_first={test_m['root_first_acc']:.4f} "
                          f"root_last={test_m['root_last_acc']:.4f} "
                          f"leaf_count_exact={test_m['leaf_count_exact']:.4f} "
                          f"leaf_first={test_m['leaf_first_acc']:.4f}")
                    print(f"  train: root_count_exact={train_m['root_count_exact']:.4f} "
                          f"root_first={train_m['root_first_acc']:.4f} "
                          f"root_last={train_m['root_last_acc']:.4f}")

                    # Length-transfer evaluation: re-evaluate the trained model on
                    # synthetic data with different leaf sizes (no retraining).
                    length_transfer = {}
                    transfer_targets = []
                    if args.length_transfer_eval is not None:
                        transfer_targets = [int(L_eval) for L_eval in args.length_transfer_eval if int(L_eval) != int(L)]
                    for L_eval in transfer_targets:
                        # Only the structural variants and transition_table are length-flexible.
                        # mlp/fno (vanilla) have fixed-size flatten layers and will error.
                        if leaf_kind in {"mlp", "fno"}:
                            length_transfer[f"L{L_eval}"] = {"error": "fixed-length encoder"}
                            continue
                        try:
                            transfer_test_docs = synthesize_triangle_docs(
                                n_docs=int(args.n_test), leaf_tokens=int(L_eval),
                                n_regimes=int(args.n_regimes), vocab_per_regime=int(args.vocab_per_regime),
                                seed=int(args.seed) + 2,
                            )
                            transfer_test = docs_to_tensors(transfer_test_docs, device)
                            metrics = eval_model(model, *transfer_test[:7])
                            length_transfer[f"L{L_eval}"] = metrics
                        except Exception as e:
                            length_transfer[f"L{L_eval}"] = {"error": f"{type(e).__name__}: {e}"}

                    cell_dir = out_root / tag
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), cell_dir / "model_state.pt")
                    (cell_dir / "history.json").write_text(json.dumps(history, indent=2))
                    summary["results"].append({
                        "leaf_tokens": int(L), "leaf_encoder": leaf_kind, "merge": merge_kind,
                        "count_head": count_head_kind, "n_params": int(n_params),
                        "train": train_m, "test": test_m,
                        "length_transfer": length_transfer,
                    })
                    if length_transfer:
                        for tgt, m in sorted(length_transfer.items()):
                            if "error" in m:
                                print(f"  transfer {tgt}: ERROR {m['error']}")
                            else:
                                print(f"  transfer {tgt}: root_count_exact={m['root_count_exact']:.4f} "
                                      f"root_first={m['root_first_acc']:.4f} root_last={m['root_last_acc']:.4f}")

    md = ["# Markov Triangle Probe (2 leaves + 1 merge), with count-head swap", "",
          f"Synthetic DGP: regime = token // {args.vocab_per_regime}, n_regimes={args.n_regimes}.",
          f"n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}, epochs={args.epochs}, lr={args.lr}.", "",
          "## Test metrics", "",
          "| L | leaf_enc | merge | count_head | params | root_first | root_last | root_count_exact | root_count_mae | leaf_first | leaf_last | leaf_count_exact |",
          "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["results"]:
        t = r["test"]
        md.append(f"| {r['leaf_tokens']} | {r['leaf_encoder']} | {r['merge']} | {r['count_head']} | {r['n_params']:,} | "
                  f"{t['root_first_acc']:.4f} | {t['root_last_acc']:.4f} | {t['root_count_exact']:.4f} | "
                  f"{t['root_count_mae']:.4f} | {t['leaf_first_acc']:.4f} | {t['leaf_last_acc']:.4f} | "
                  f"{t['leaf_count_exact']:.4f} |")
    md += ["", "## Train metrics", "",
           "| L | leaf_enc | merge | count_head | root_first | root_last | root_count_exact |",
           "|---:|---|---|---|---:|---:|---:|"]
    for r in summary["results"]:
        t = r["train"]
        md.append(f"| {r['leaf_tokens']} | {r['leaf_encoder']} | {r['merge']} | {r['count_head']} | "
                  f"{t['root_first_acc']:.4f} | {t['root_last_acc']:.4f} | {t['root_count_exact']:.4f} |")
    md += [""]

    # Length-transfer table
    has_transfer = any(r.get("length_transfer") for r in summary["results"])
    if has_transfer:
        md += ["## Length transfer (zero-shot eval at L != L_train)", "",
               "Reads: root_first / root_last / root_count_exact at each evaluated L.", ""]
        # Determine evaluated columns
        all_targets = sorted({k for r in summary["results"] for k in (r.get("length_transfer") or {}).keys()},
                              key=lambda x: int(x[1:]))
        md.append("| L_train | leaf_enc | merge | count_head | " +
                  " | ".join(f"eval {t}" for t in all_targets) + " |")
        md.append("|---:|---|---|---|" + "---:|" * len(all_targets))
        for r in summary["results"]:
            cells = []
            for t in all_targets:
                m = (r.get("length_transfer") or {}).get(t, {})
                if "error" in m:
                    cells.append("ERR")
                elif m:
                    cells.append(f"{m['root_first_acc']:.3f}/{m['root_last_acc']:.3f}/{m['root_count_exact']:.3f}")
                else:
                    cells.append("-")
            md.append(f"| {r['leaf_tokens']} | {r['leaf_encoder']} | {r['merge']} | {r['count_head']} | " +
                      " | ".join(cells) + " |")
        md += [""]

    (out_root / "report.md").write_text("\n".join(md) + "\n")
    (out_root / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
