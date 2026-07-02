#!/usr/bin/env python3
"""Probe whether a leaf encoder learns the recoverable_v5_t128 rule (regime = token // 4)
or interpolates through length/position-conditional features.

Trains three small leaf-encoder variants on recoverable_v5_t128:

  - transition_table : nn.Embedding(vocab_size, n_regimes)            (~64 params)
  - mlp              : nn.Embedding -> Flatten -> MLP                  (~150K params)
  - fno              : nn.Embedding -> SpectralConv1d FNO -> head      (~600K params, mimics v3 leaf encoder)

Then runs four diagnostics on the trained leaf encoders:

  Probe B (token-swap-within-regime): predictions invariant?
  Probe C (length-transfer)         : zero-shot on length 8 and 32 when trained at 16?
  Probe D (composition)             : exact (count, first, last) -> encoder -> decoder round trip?
  Probe E (held-out-token)          : synthetic leaves with rare token compositions?

Outputs JSON + Markdown in outputs/markov_rule_learning_probes_<ts>/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _load_fno_docs,
    prepare_markov_full_doc_anchor_diagnostics_data,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# ------------------------- data ------------------------------------------------


def _load_split_docs(*, train_docs: int, seed: int = 0):
    payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name="recoverable_v5_t128",
        seeds=(int(seed),),
        train_doc_counts=(int(train_docs),),
        use_cuda=False,
        torch_threads=1,
    )
    prepared = dict(payload["prepared"][0])
    return (
        list(_load_fno_docs(Path(str(prepared["train_fno_docs_json"]))))[: int(train_docs)],
        list(_load_fno_docs(Path(str(prepared["val_fno_docs_json"])))),
        list(_load_fno_docs(Path(str(prepared["test_fno_docs_json"])))),
    )


def _shape(docs):
    n_leaves = len(docs[0].leaf_token_ids)
    leaf_tokens = len(docs[0].leaf_token_ids[0])
    n_regimes = 1 + max(
        max(int(v) for d in docs for v in d.leaf_first_regimes),
        max(int(v) for d in docs for v in d.leaf_last_regimes),
    )
    vocab_size = 1 + max(
        int(t) for d in docs for leaf in d.leaf_token_ids for t in leaf
    )
    return n_leaves, leaf_tokens, n_regimes, vocab_size


def _doc_tensors(docs, device):
    tokens = torch.tensor(
        [[list(map(int, leaf)) for leaf in d.leaf_token_ids] for d in docs],
        dtype=torch.long,
        device=device,
    )
    counts = torch.tensor(
        [[float(c) for c in d.leaf_counts] for d in docs],
        dtype=torch.float32,
        device=device,
    )
    first = torch.tensor(
        [[int(v) for v in d.leaf_first_regimes] for d in docs],
        dtype=torch.long,
        device=device,
    )
    last = torch.tensor(
        [[int(v) for v in d.leaf_last_regimes] for d in docs],
        dtype=torch.long,
        device=device,
    )
    return tokens, counts, first, last


# ------------------------- models ----------------------------------------------


class TransitionTableEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_regimes: int):
        super().__init__()
        self.token_to_regime = nn.Embedding(int(vocab_size), int(n_regimes))
        self.n_regimes = int(n_regimes)
        self.kind = "transition_table"

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, n_leaves, T)
        logits = self.token_to_regime(tokens)
        probs = torch.softmax(logits, dim=-1)
        same = (probs[..., :-1, :] * probs[..., 1:, :]).sum(dim=-1)
        count = (1.0 - same).sum(dim=-1)
        first_logits = logits[..., 0, :]
        last_logits = logits[..., -1, :]
        return count, first_logits, last_logits


class MLPEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_regimes: int, leaf_tokens: int,
                 embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.net = nn.Sequential(
            nn.Linear(int(leaf_tokens) * int(embed_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.count_head = nn.Linear(int(hidden_dim), 1)
        self.first_head = nn.Linear(int(hidden_dim), int(n_regimes))
        self.last_head = nn.Linear(int(hidden_dim), int(n_regimes))
        self.n_regimes = int(n_regimes)
        self.leaf_tokens = int(leaf_tokens)
        self.kind = "mlp"

    def forward(self, tokens: torch.Tensor):
        B, L, T = tokens.shape
        emb = self.embedding(tokens).reshape(B * L, T * self.embedding.embedding_dim)
        h = self.net(emb)
        count = self.count_head(h).squeeze(-1).reshape(B, L)
        first = self.first_head(h).reshape(B, L, self.n_regimes)
        last = self.last_head(h).reshape(B, L, self.n_regimes)
        return count, first, last


class SpectralConv1d(nn.Module):
    """1D spectral convolution layer (FNO building block)."""
    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = int(n_modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        m = min(self.n_modes, x_ft.size(-1))
        out_ft[..., :m] = torch.einsum("bcm,com->bom", x_ft[..., :m], self.weights[..., :m])
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class FNOEncoder(nn.Module):
    """Mimics the v3 leaf FNO encoder shape."""
    def __init__(self, vocab_size: int, n_regimes: int,
                 fno_width: int = 128, n_modes: int = 8, n_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(fno_width))
        self.layers = nn.ModuleList([
            SpectralConv1d(int(fno_width), int(fno_width), int(n_modes))
            for _ in range(int(n_layers))
        ])
        self.skip = nn.ModuleList([
            nn.Conv1d(int(fno_width), int(fno_width), 1)
            for _ in range(int(n_layers))
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, int(fno_width)) for _ in range(int(n_layers))
        ])
        self.count_head = nn.Sequential(
            nn.Linear(int(fno_width), int(fno_width)),
            nn.GELU(),
            nn.Linear(int(fno_width), 1),
        )
        self.first_head = nn.Linear(int(fno_width), int(n_regimes))
        self.last_head = nn.Linear(int(fno_width), int(n_regimes))
        self.n_regimes = int(n_regimes)
        self.fno_width = int(fno_width)
        self.kind = "fno"

    def forward(self, tokens: torch.Tensor):
        B, L, T = tokens.shape
        x = self.embedding(tokens).reshape(B * L, T, self.fno_width).transpose(1, 2)
        for layer, skip, norm in zip(self.layers, self.skip, self.norms):
            x = norm(F.gelu(layer(x) + skip(x)))
        # Mean-pool over tokens for count head; first/last from end positions
        h_mean = x.mean(dim=-1)
        h_first = x[..., 0]
        h_last = x[..., -1]
        count = self.count_head(h_mean).squeeze(-1).reshape(B, L)
        first = self.first_head(h_first).reshape(B, L, self.n_regimes)
        last = self.last_head(h_last).reshape(B, L, self.n_regimes)
        return count, first, last


# ------------------------- training & eval -------------------------------------


def _losses(count, first, last, t_count, t_first, t_last):
    return (
        F.mse_loss(count, t_count.float())
        + F.cross_entropy(first.reshape(-1, first.size(-1)), t_first.reshape(-1))
        + F.cross_entropy(last.reshape(-1, last.size(-1)), t_last.reshape(-1))
    )


@torch.no_grad()
def _accuracy(model, tokens, t_count, t_first, t_last, batch_size=64):
    model.eval()
    n_first_correct = n_last_correct = n_count_correct = 0
    n_total = 0
    count_mae_sum = 0.0
    for i in range(0, tokens.size(0), batch_size):
        ts = tokens[i:i+batch_size]
        c, f, l = model(ts)
        pred_first = f.argmax(dim=-1)
        pred_last = l.argmax(dim=-1)
        pred_count = c.round()
        n_first_correct += (pred_first == t_first[i:i+batch_size]).sum().item()
        n_last_correct += (pred_last == t_last[i:i+batch_size]).sum().item()
        n_count_correct += (pred_count == t_count[i:i+batch_size]).sum().item()
        count_mae_sum += (c - t_count[i:i+batch_size].float()).abs().sum().item()
        n_total += ts.size(0) * ts.size(1)
    return {
        "leaf_first_acc": n_first_correct / n_total,
        "leaf_last_acc": n_last_correct / n_total,
        "leaf_count_exact": n_count_correct / n_total,
        "leaf_count_mae": count_mae_sum / n_total,
    }


def train_model(model, train_docs, val_docs, *, epochs: int, lr: float, device,
                batch_size: int = 32):
    train_tokens, train_count, train_first, train_last = _doc_tensors(train_docs, device)
    val_tokens, val_count, val_first, val_last = _doc_tensors(val_docs, device)
    n_train = train_tokens.size(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            c, f, l = model(train_tokens[idx])
            loss = _losses(c, f, l, train_count[idx], train_first[idx], train_last[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        train_metrics = _accuracy(model, train_tokens, train_count, train_first, train_last)
        val_metrics = _accuracy(model, val_tokens, val_count, val_first, val_last)
        ep = {
            "epoch": int(epoch),
            "train_loss": float(ep_loss / max(1, n_train // batch_size)),
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        history.append(ep)
        # Track best by val_leaf_first+last combined
        score = -(val_metrics["leaf_first_acc"] + val_metrics["leaf_last_acc"])
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ------------------------- probes ----------------------------------------------


@torch.no_grad()
def probe_token_swap(model, tokens, *, n_regimes: int, vocab_per_regime: int, device):
    """Probe B: swap each token to a different token in the same regime bucket."""
    model.eval()
    # Original predictions
    base_first_logits = []
    base_last_logits = []
    base_count = []
    swap_first_logits = []
    swap_last_logits = []
    swap_count = []
    for i in range(0, tokens.size(0), 32):
        ts = tokens[i:i+32]
        c0, f0, l0 = model(ts)
        base_count.append(c0); base_first_logits.append(f0); base_last_logits.append(l0)
        # Swap: token t -> ((t // vocab_per_regime) * vocab_per_regime) + ((t + 1) % vocab_per_regime)
        bucket = ts // vocab_per_regime
        within = ts % vocab_per_regime
        swapped = bucket * vocab_per_regime + (within + 1) % vocab_per_regime
        c1, f1, l1 = model(swapped)
        swap_count.append(c1); swap_first_logits.append(f1); swap_last_logits.append(l1)
    base_first = torch.cat(base_first_logits).argmax(-1)
    base_last = torch.cat(base_last_logits).argmax(-1)
    base_c = torch.cat(base_count).round()
    swap_first = torch.cat(swap_first_logits).argmax(-1)
    swap_last = torch.cat(swap_last_logits).argmax(-1)
    swap_c = torch.cat(swap_count).round()
    n = base_first.numel()
    return {
        "first_agreement": float((base_first == swap_first).sum().item() / n),
        "last_agreement": float((base_last == swap_last).sum().item() / n),
        "count_agreement": float((base_c == swap_c).sum().item() / n),
    }


def probe_length_transfer(model, train_docs, *, target_lengths: Sequence[int], device,
                          n_regimes: int):
    """Probe C: build synthetic leaves at each target length and evaluate."""
    rng = np.random.default_rng(0)
    out = {}
    n_test = 256
    n_leaves = 8
    for L in target_lengths:
        # Synthesize per-leaf: pick first regime, pick last regime, count = #changepoints inside
        # Simplest: each leaf gets a random number of changepoints (0 to L-1)
        all_tokens = []
        all_count = []
        all_first = []
        all_last = []
        for d in range(n_test):
            doc_tokens = []
            doc_count = []
            doc_first = []
            doc_last = []
            for leaf in range(n_leaves):
                # Random regime sequence within the leaf
                k = int(rng.integers(0, min(L, 4)))  # 0 to 3 changepoints
                positions = sorted(rng.choice(range(1, L), size=k, replace=False).tolist()) if k > 0 else []
                regimes = []
                cur = int(rng.integers(0, n_regimes))
                pos_idx = 0
                for t in range(L):
                    if pos_idx < len(positions) and t == positions[pos_idx]:
                        # Pick a different regime
                        new = int(rng.integers(0, n_regimes - 1))
                        if new >= cur: new += 1
                        cur = new
                        pos_idx += 1
                    regimes.append(cur)
                # Each token: random within bucket
                tokens = [r * 4 + int(rng.integers(0, 4)) for r in regimes]
                doc_tokens.append(tokens)
                doc_count.append(float(len(positions)))
                doc_first.append(int(regimes[0]))
                doc_last.append(int(regimes[-1]))
            all_tokens.append(doc_tokens)
            all_count.append(doc_count)
            all_first.append(doc_first)
            all_last.append(doc_last)
        tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
        count = torch.tensor(all_count, dtype=torch.float32, device=device)
        first = torch.tensor(all_first, dtype=torch.long, device=device)
        last = torch.tensor(all_last, dtype=torch.long, device=device)
        try:
            metrics = _accuracy(model, tokens, count, first, last)
        except Exception as e:
            metrics = {"error": f"{type(e).__name__}: {e}"}
        out[f"L{L}"] = metrics
    return out


@torch.no_grad()
def probe_held_out_combinations(model, *, leaf_tokens: int, n_regimes: int, n_leaves: int,
                                vocab_per_regime: int, device):
    """Probe E: check synthetic leaves with rare token compositions."""
    rng = np.random.default_rng(0)
    cases = {}

    def eval_case(tokens, expect_count, expect_first, expect_last):
        # tokens: (n_leaves, T)
        # We construct n_test docs each with these n_leaves
        n_docs = 64
        ts = torch.tensor([list(tokens) for _ in range(n_docs)], dtype=torch.long, device=device)
        # All-same target per leaf in this batch
        ec = torch.tensor([list(expect_count) for _ in range(n_docs)], dtype=torch.float32, device=device)
        ef = torch.tensor([list(expect_first) for _ in range(n_docs)], dtype=torch.long, device=device)
        el = torch.tensor([list(expect_last) for _ in range(n_docs)], dtype=torch.long, device=device)
        return _accuracy(model, ts, ec, ef, el)

    # Case 1: all tokens equal to 0 in every leaf (regime 0, count 0)
    cases["all_zeros"] = eval_case(
        [[0]*leaf_tokens for _ in range(n_leaves)],
        expect_count=[0.0]*n_leaves,
        expect_first=[0]*n_leaves,
        expect_last=[0]*n_leaves,
    )
    # Case 2: alternating regime 0 / regime 2 every step in every leaf
    alt = []
    for t in range(leaf_tokens):
        alt.append(0 if t % 2 == 0 else 8)  # token 0 -> regime 0, token 8 -> regime 2
    cases["alt_0_8"] = eval_case(
        [list(alt) for _ in range(n_leaves)],
        expect_count=[float(leaf_tokens - 1)]*n_leaves,
        expect_first=[0]*n_leaves,
        expect_last=[2 if (leaf_tokens - 1) % 2 == 1 else 0]*n_leaves,
    )
    # Case 3: token=15 throughout (regime 3, count 0)
    cases["all_fifteens"] = eval_case(
        [[15]*leaf_tokens for _ in range(n_leaves)],
        expect_count=[0.0]*n_leaves,
        expect_first=[3]*n_leaves,
        expect_last=[3]*n_leaves,
    )
    # Case 4: single regime change at position leaf_tokens // 2 (regime 0 -> regime 1)
    half = leaf_tokens // 2
    half_seq = [0]*half + [4]*(leaf_tokens - half)  # token 4 is regime 1
    cases["single_change_mid"] = eval_case(
        [list(half_seq) for _ in range(n_leaves)],
        expect_count=[1.0]*n_leaves,
        expect_first=[0]*n_leaves,
        expect_last=[1]*n_leaves,
    )
    # Case 5: every token a different bucket (4 regimes cycling)
    cyc = [(t % n_regimes) * vocab_per_regime for t in range(leaf_tokens)]
    expect_last_cyc = ((leaf_tokens - 1) % n_regimes)
    cases["cyclic_regimes"] = eval_case(
        [list(cyc) for _ in range(n_leaves)],
        expect_count=[float(leaf_tokens - 1)]*n_leaves,
        expect_first=[0]*n_leaves,
        expect_last=[expect_last_cyc]*n_leaves,
    )
    return cases


# Probe D (composition): trivial for transition_table by construction; for MLP/FNO we
# simply note that they don't accept exact (count, first, last) summaries as input
# (their input is tokens). The composition probe is meaningful for the v3 shared-feature
# encoder, not for simple leaf-only encoders. We capture this fact in the report.


# ------------------------- main ------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-docs", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=str, default=str(REPO / "outputs" / f"markov_rule_learning_probes_{_ts()}"))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["transition_table", "mlp", "fno"])
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root}")

    train_docs, val_docs, test_docs = _load_split_docs(train_docs=int(args.train_docs), seed=int(args.seed))
    n_leaves, leaf_tokens, n_regimes, vocab_size = _shape(train_docs)
    vocab_per_regime = vocab_size // n_regimes
    print(f"data: n_leaves={n_leaves} leaf_tokens={leaf_tokens} n_regimes={n_regimes} vocab_size={vocab_size} vocab_per_regime={vocab_per_regime}")
    print(f"train_docs={len(train_docs)} val_docs={len(val_docs)} test_docs={len(test_docs)}")

    device = torch.device(args.device)

    summary = {
        "args": vars(args),
        "data": {
            "n_leaves": n_leaves, "leaf_tokens": leaf_tokens,
            "n_regimes": n_regimes, "vocab_size": vocab_size, "vocab_per_regime": vocab_per_regime,
            "n_train": len(train_docs), "n_val": len(val_docs), "n_test": len(test_docs),
        },
        "models": {},
    }

    test_tokens, test_count, test_first, test_last = _doc_tensors(test_docs, device)

    for model_kind in args.models:
        print(f"\n=== training {model_kind} ===")
        if model_kind == "transition_table":
            model = TransitionTableEncoder(vocab_size, n_regimes).to(device)
        elif model_kind == "mlp":
            model = MLPEncoder(vocab_size, n_regimes, leaf_tokens).to(device)
        elif model_kind == "fno":
            model = FNOEncoder(vocab_size, n_regimes).to(device)
        else:
            raise ValueError(f"unknown model: {model_kind}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        history = train_model(model, train_docs, val_docs,
                              epochs=int(args.epochs), lr=float(args.lr), device=device)
        # Save model & history
        model_dir = out_root / model_kind
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model_state.pt")
        (model_dir / "train_history.json").write_text(json.dumps(history, indent=2))

        test_metrics = _accuracy(model, test_tokens, test_count, test_first, test_last)
        print(f"  test:  first={test_metrics['leaf_first_acc']:.4f} last={test_metrics['leaf_last_acc']:.4f} count_exact={test_metrics['leaf_count_exact']:.4f} count_mae={test_metrics['leaf_count_mae']:.4f}")

        # Train accuracy for gap
        train_tokens, train_count_t, train_first_t, train_last_t = _doc_tensors(train_docs, device)
        train_metrics = _accuracy(model, train_tokens, train_count_t, train_first_t, train_last_t)

        # Probe B: token-swap
        probe_b = probe_token_swap(model, test_tokens, n_regimes=n_regimes,
                                   vocab_per_regime=vocab_per_regime, device=device)
        print(f"  probe_B (token-swap):       first_agree={probe_b['first_agreement']:.4f} last={probe_b['last_agreement']:.4f} count={probe_b['count_agreement']:.4f}")

        # Probe C: length transfer
        probe_c = probe_length_transfer(model, train_docs,
                                        target_lengths=[8, 16, 32], device=device,
                                        n_regimes=n_regimes)
        for L, m in probe_c.items():
            if "error" in m:
                print(f"  probe_C ({L}): ERROR {m['error']}")
            else:
                print(f"  probe_C ({L}): first={m['leaf_first_acc']:.4f} last={m['leaf_last_acc']:.4f} count_exact={m['leaf_count_exact']:.4f}")

        # Probe E: held-out combinations
        probe_e = probe_held_out_combinations(model, leaf_tokens=leaf_tokens, n_regimes=n_regimes,
                                              n_leaves=n_leaves, vocab_per_regime=vocab_per_regime,
                                              device=device)
        for case, m in probe_e.items():
            print(f"  probe_E ({case}): first={m['leaf_first_acc']:.4f} last={m['leaf_last_acc']:.4f} count_exact={m['leaf_count_exact']:.4f}")

        summary["models"][model_kind] = {
            "n_params": int(n_params),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "gap_first": float(train_metrics["leaf_first_acc"] - test_metrics["leaf_first_acc"]),
            "gap_last": float(train_metrics["leaf_last_acc"] - test_metrics["leaf_last_acc"]),
            "probe_B_token_swap": probe_b,
            "probe_C_length_transfer": probe_c,
            "probe_E_held_out_combinations": probe_e,
        }

    # Markdown report
    md = ["# Markov Rule-Learning Probes", "",
          f"DGP: recoverable_v5_t128, n_train={len(train_docs)}, leaf_tokens={leaf_tokens}, n_regimes={n_regimes}, vocab={vocab_size} (per-regime bucket {vocab_per_regime})", ""]
    md += ["## Capacity & gap", "",
           "| model | params | test_first | test_last | train_first | train_last | gap_first | gap_last |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in summary["models"].items():
        md.append(f"| {k} | {v['n_params']:,} | {v['test_metrics']['leaf_first_acc']:.4f} | {v['test_metrics']['leaf_last_acc']:.4f} | "
                  f"{v['train_metrics']['leaf_first_acc']:.4f} | {v['train_metrics']['leaf_last_acc']:.4f} | "
                  f"{v['gap_first']:.4f} | {v['gap_last']:.4f} |")
    md += ["", "## Probe B: token-swap-within-regime invariance", "",
           "| model | first_agreement | last_agreement | count_agreement |",
           "|---|---:|---:|---:|"]
    for k, v in summary["models"].items():
        b = v["probe_B_token_swap"]
        md.append(f"| {k} | {b['first_agreement']:.4f} | {b['last_agreement']:.4f} | {b['count_agreement']:.4f} |")
    md += ["", "Rule-learning predicts 1.0000 on all three columns.", ""]
    md += ["## Probe C: length-transfer (model trained at this leaf_tokens)", "",
           "| model | L=8 first/last/count | L=16 first/last/count | L=32 first/last/count |",
           "|---|---:|---:|---:|"]
    for k, v in summary["models"].items():
        cells = []
        for L in ["L8", "L16", "L32"]:
            m = v["probe_C_length_transfer"][L]
            if "error" in m:
                cells.append(f"ERR")
            else:
                cells.append(f"{m['leaf_first_acc']:.3f}/{m['leaf_last_acc']:.3f}/{m['leaf_count_exact']:.3f}")
        md.append(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    md += ["", "## Probe E: held-out token compositions", ""]
    cases = list(next(iter(summary["models"].values()))["probe_E_held_out_combinations"].keys())
    md += ["| model | " + " | ".join(cases) + " |",
           "|---|" + "---:|" * len(cases)]
    for k, v in summary["models"].items():
        cells = []
        for c in cases:
            m = v["probe_E_held_out_combinations"][c]
            cells.append(f"{m['leaf_first_acc']:.3f}/{m['leaf_last_acc']:.3f}/{m['leaf_count_exact']:.3f}")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "Each cell: first_acc / last_acc / count_exact. Rule-learning predicts 1.000/1.000/1.000.", ""]

    (out_root / "report.md").write_text("\n".join(md) + "\n")
    (out_root / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
