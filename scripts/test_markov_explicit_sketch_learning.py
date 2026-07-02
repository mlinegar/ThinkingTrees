#!/usr/bin/env python3
"""Train an explicit Markov sketch learner on the recoverable DGP.

This is intentionally smaller than the full C-TreePO training pipeline.  The
state is the theorem sketch itself:

    (count / target_scale, first logits, last logits)

The script first learns the merge count update on oracle sketch leaves, then
trains a token-to-sketch leaf summarizer while keeping the learned merge in the
loop.  It tests the user-facing question directly: can we learn the Markov
sketch and its merge once the objective actually matches the DGP?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    _FNOCountDoc,
    _balanced_exact_sketch_targets,
    _set_global_seed,
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_docs(
    *,
    benchmark: str,
    train_docs: int,
    seed: int,
) -> tuple[tuple[_FNOCountDoc, ...], tuple[_FNOCountDoc, ...], tuple[_FNOCountDoc, ...], Dict[str, Any]]:
    prepared_payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=str(benchmark),
        seeds=(int(seed),),
        train_doc_counts=(int(train_docs),),
        use_cuda=False,
        torch_threads=1,
    )
    prepared = dict(prepared_payload["prepared"][0])
    return (
        tuple(_load_fno_docs(Path(str(prepared["train_fno_docs_json"])))[: int(train_docs)]),
        tuple(_load_fno_docs(Path(str(prepared["val_fno_docs_json"])))),
        tuple(_load_fno_docs(Path(str(prepared["test_fno_docs_json"])))),
        prepared,
    )


def _shape_info(docs: Sequence[_FNOCountDoc]) -> tuple[int, int, int, int]:
    if not docs:
        raise ValueError("docs must be non-empty")
    n_leaves = len(docs[0].leaf_token_ids)
    leaf_tokens = len(docs[0].leaf_token_ids[0])
    n_regimes = 1 + max(
        max(int(v) for doc in docs for v in doc.leaf_first_regimes),
        max(int(v) for doc in docs for v in doc.leaf_last_regimes),
    )
    vocab_size = 1 + max(
        int(tok)
        for doc in docs
        for leaf in doc.leaf_token_ids
        for tok in leaf
    )
    for doc in docs:
        if len(doc.leaf_token_ids) != n_leaves:
            raise ValueError("all docs must have the same leaf count")
        if any(len(leaf) != leaf_tokens for leaf in doc.leaf_token_ids):
            raise ValueError("all leaves must have the same token count")
    return int(n_leaves), int(leaf_tokens), int(n_regimes), int(vocab_size)


def _target_scale(docs: Sequence[_FNOCountDoc]) -> float:
    return float(max(1, max(int(round(float(doc.root_count))) for doc in docs)))


def _doc_batch_tensors(
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = torch.tensor(
        [[list(leaf) for leaf in doc.leaf_token_ids] for doc in docs],
        device=device,
        dtype=torch.long,
    )
    leaf_counts = torch.tensor(
        [[float(v) for v in doc.leaf_counts] for doc in docs],
        device=device,
        dtype=torch.float32,
    )
    leaf_first = torch.tensor(
        [[int(v) for v in doc.leaf_first_regimes] for doc in docs],
        device=device,
        dtype=torch.long,
    )
    leaf_last = torch.tensor(
        [[int(v) for v in doc.leaf_last_regimes] for doc in docs],
        device=device,
        dtype=torch.long,
    )
    merge_targets = [
        _balanced_exact_sketch_targets(
            leaf_counts=doc.leaf_counts,
            leaf_first_regimes=doc.leaf_first_regimes,
            leaf_last_regimes=doc.leaf_last_regimes,
        )
        for doc in docs
    ]
    merge_counts = torch.tensor(
        [[float(item[0]) for item in target["merge"]] for target in merge_targets],
        device=device,
        dtype=torch.float32,
    )
    merge_first = torch.tensor(
        [[int(item[1]) for item in target["merge"]] for target in merge_targets],
        device=device,
        dtype=torch.long,
    )
    merge_last = torch.tensor(
        [[int(item[2]) for item in target["merge"]] for target in merge_targets],
        device=device,
        dtype=torch.long,
    )
    root_counts = torch.tensor(
        [float(doc.root_count) for doc in docs],
        device=device,
        dtype=torch.float32,
    )
    return {
        "tokens": tokens,
        "leaf_counts": leaf_counts,
        "leaf_first": leaf_first,
        "leaf_last": leaf_last,
        "merge_counts": merge_counts,
        "merge_first": merge_first,
        "merge_last": merge_last,
        "root_counts": root_counts,
    }


class ExplicitMarkovSketchModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        leaf_tokens: int,
        n_regimes: int,
        target_scale: float,
        embed_dim: int = 48,
        hidden_dim: int = 256,
        exact_logit: float = 8.0,
        leaf_encoder: str = "mlp",
        merge_count_mode: str = "mlp",
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.leaf_tokens = int(leaf_tokens)
        self.n_regimes = int(n_regimes)
        self.target_scale = float(target_scale)
        self.exact_logit = float(exact_logit)
        if str(leaf_encoder) not in {"mlp", "transition_table"}:
            raise ValueError(f"unsupported leaf_encoder={leaf_encoder!r}")
        if str(merge_count_mode) not in {"mlp", "additive_join_table"}:
            raise ValueError(f"unsupported merge_count_mode={merge_count_mode!r}")
        self.leaf_encoder = str(leaf_encoder)
        self.merge_count_mode = str(merge_count_mode)
        if self.leaf_encoder == "mlp":
            self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
            self.leaf_net = nn.Sequential(
                nn.Linear(int(leaf_tokens) * int(embed_dim), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            )
            self.leaf_count = nn.Linear(int(hidden_dim), 1)
            self.leaf_first = nn.Linear(int(hidden_dim), int(n_regimes))
            self.leaf_last = nn.Linear(int(hidden_dim), int(n_regimes))
        else:
            self.token_regime_logits = nn.Embedding(int(vocab_size), int(n_regimes))
        if self.merge_count_mode == "mlp":
            self.merge_count = nn.Sequential(
                nn.Linear(2 + 2 * int(n_regimes), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.join_logits = nn.Parameter(torch.zeros(int(n_regimes), int(n_regimes)))

    @property
    def state_dim(self) -> int:
        return 1 + 2 * int(self.n_regimes)

    def exact_leaf_states(
        self,
        leaf_counts: torch.Tensor,
        leaf_first: torch.Tensor,
        leaf_last: torch.Tensor,
    ) -> torch.Tensor:
        count_norm = leaf_counts / float(self.target_scale)
        first_logits = F.one_hot(
            leaf_first.to(dtype=torch.long),
            num_classes=int(self.n_regimes),
        ).to(dtype=count_norm.dtype) * float(self.exact_logit)
        last_logits = F.one_hot(
            leaf_last.to(dtype=torch.long),
            num_classes=int(self.n_regimes),
        ).to(dtype=count_norm.dtype) * float(self.exact_logit)
        return torch.cat(
            [count_norm.unsqueeze(-1), first_logits, last_logits],
            dim=-1,
        )

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, n_leaves, leaf_tokens = tokens.shape
        if self.leaf_encoder == "transition_table":
            token_logits = self.token_regime_logits(tokens)
            token_probs = torch.softmax(token_logits, dim=-1)
            same_prob = (token_probs[:, :, :-1, :] * token_probs[:, :, 1:, :]).sum(dim=-1)
            count_norm = (1.0 - same_prob).sum(dim=-1) / float(self.target_scale)
            return torch.cat(
                [
                    count_norm.unsqueeze(-1),
                    token_logits[:, :, 0, :],
                    token_logits[:, :, -1, :],
                ],
                dim=-1,
            )
        emb = self.embedding(tokens).reshape(int(bsz * n_leaves), -1)
        h = self.leaf_net(emb)
        count_norm = self.leaf_count(h).squeeze(-1)
        first_logits = self.leaf_first(h)
        last_logits = self.leaf_last(h)
        return torch.cat(
            [count_norm.unsqueeze(-1), first_logits, last_logits],
            dim=-1,
        ).reshape(int(bsz), int(n_leaves), self.state_dim)

    def split(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(self.n_regimes)
        count_norm = states[..., 0]
        first_logits = states[..., 1 : 1 + n]
        last_logits = states[..., 1 + n :]
        return count_norm, first_logits, last_logits

    def merge_pairs(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_count, left_first, left_last = self.split(left)
        right_count, _right_first, right_last = self.split(right)
        right_first = self.split(right)[1]
        left_last_prob = torch.softmax(left_last, dim=-1)
        right_first_prob = torch.softmax(right_first, dim=-1)
        if self.merge_count_mode == "mlp":
            features = torch.cat(
                [
                    left_count.unsqueeze(-1),
                    right_count.unsqueeze(-1),
                    left_last_prob,
                    right_first_prob,
                ],
                dim=-1,
            )
            merged_count = self.merge_count(features).squeeze(-1)
        else:
            join_value = torch.sigmoid(self.join_logits)
            expected_join = torch.einsum(
                "...i,ij,...j->...",
                left_last_prob,
                join_value,
                right_first_prob,
            )
            merged_count = left_count + right_count + expected_join / float(self.target_scale)
        return torch.cat(
            [merged_count.unsqueeze(-1), left_first, right_last],
            dim=-1,
        )

    def merge_tree(self, leaf_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = leaf_states
        merge_levels = []
        while int(current.shape[1]) > 1:
            n_pairs = int(current.shape[1] // 2)
            left = current[:, 0 : 2 * n_pairs : 2, :]
            right = current[:, 1 : 2 * n_pairs : 2, :]
            merged = self.merge_pairs(left, right)
            merge_levels.append(merged)
            if int(current.shape[1]) % 2 == 1:
                current = torch.cat([merged, current[:, -1:, :]], dim=1)
            else:
                current = merged
        return current[:, 0, :], torch.cat(merge_levels, dim=1)

    def count(self, states: torch.Tensor) -> torch.Tensor:
        return self.split(states)[0] * float(self.target_scale)

    def leaf_parameters(self) -> list[nn.Parameter]:
        if self.leaf_encoder == "transition_table":
            return list(self.token_regime_logits.parameters())
        return [
            *self.embedding.parameters(),
            *self.leaf_net.parameters(),
            *self.leaf_count.parameters(),
            *self.leaf_first.parameters(),
            *self.leaf_last.parameters(),
        ]

    def merge_parameters(self) -> list[nn.Parameter]:
        if self.merge_count_mode == "mlp":
            return list(self.merge_count.parameters())
        return [self.join_logits]


def _loss_for_batch(
    model: ExplicitMarkovSketchModel,
    batch: Mapping[str, torch.Tensor],
    *,
    exact_leaves: bool,
    root_weight: float,
    leaf_weight: float,
    merge_weight: float,
    leaf_count_loss_weight: float = 1.0,
    leaf_endpoint_loss_weight: float = 1.0,
    merge_count_loss_weight: float = 1.0,
    merge_endpoint_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if exact_leaves:
        leaf_states = model.exact_leaf_states(
            batch["leaf_counts"],
            batch["leaf_first"],
            batch["leaf_last"],
        )
    else:
        leaf_states = model.encode_tokens(batch["tokens"])
    root_state, merge_states = model.merge_tree(leaf_states)
    leaf_count, leaf_first_logits, leaf_last_logits = model.split(leaf_states)
    merge_count, merge_first_logits, merge_last_logits = model.split(merge_states)
    root_count = model.count(root_state)
    leaf_count_loss = F.mse_loss(leaf_count * float(model.target_scale), batch["leaf_counts"])
    leaf_first_loss = F.cross_entropy(
        leaf_first_logits.reshape(-1, int(model.n_regimes)),
        batch["leaf_first"].reshape(-1),
    )
    leaf_last_loss = F.cross_entropy(
        leaf_last_logits.reshape(-1, int(model.n_regimes)),
        batch["leaf_last"].reshape(-1),
    )
    merge_count_loss = F.mse_loss(merge_count * float(model.target_scale), batch["merge_counts"])
    merge_first_loss = F.cross_entropy(
        merge_first_logits.reshape(-1, int(model.n_regimes)),
        batch["merge_first"].reshape(-1),
    )
    merge_last_loss = F.cross_entropy(
        merge_last_logits.reshape(-1, int(model.n_regimes)),
        batch["merge_last"].reshape(-1),
    )
    root_loss = F.mse_loss(root_count, batch["root_counts"])
    if exact_leaves:
        leaf_term = leaf_count_loss.new_zeros(())
    else:
        leaf_term = (
            float(leaf_count_loss_weight) * leaf_count_loss
            + float(leaf_endpoint_loss_weight) * (leaf_first_loss + leaf_last_loss)
        )
    merge_term = (
        float(merge_count_loss_weight) * merge_count_loss
        + float(merge_endpoint_loss_weight) * (merge_first_loss + merge_last_loss)
    )
    loss = (
        float(root_weight) * root_loss
        + float(leaf_weight) * leaf_term
        + float(merge_weight) * merge_term
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "root_loss": float(root_loss.detach().cpu()),
        "leaf_count_loss": float(leaf_count_loss.detach().cpu()),
        "leaf_first_loss": float(leaf_first_loss.detach().cpu()),
        "leaf_last_loss": float(leaf_last_loss.detach().cpu()),
        "merge_count_loss": float(merge_count_loss.detach().cpu()),
        "merge_first_loss": float(merge_first_loss.detach().cpu()),
        "merge_last_loss": float(merge_last_loss.detach().cpu()),
    }
    return loss, metrics


def _iter_batches(
    docs: Sequence[_FNOCountDoc],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Sequence[tuple[_FNOCountDoc, ...]]:
    indices = np.arange(len(docs))
    rng.shuffle(indices)
    return [
        tuple(docs[int(i)] for i in indices[start : start + int(batch_size)])
        for start in range(0, len(indices), int(batch_size))
    ]


def _train_phase(
    model: ExplicitMarkovSketchModel,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    exact_leaves: bool,
    root_weight: float,
    leaf_weight: float,
    merge_weight: float,
    leaf_count_loss_weight: float = 1.0,
    leaf_endpoint_loss_weight: float = 1.0,
    merge_count_loss_weight: float = 1.0,
    merge_endpoint_loss_weight: float = 1.0,
    seed: int,
    progress_path: Path,
    phase: str,
    train_leaf: bool = True,
    train_merge: bool = True,
) -> Dict[str, float]:
    params: list[nn.Parameter] = []
    if bool(train_leaf):
        params.extend(model.leaf_parameters())
    if bool(train_merge):
        params.extend(model.merge_parameters())
    if not params:
        raise ValueError("at least one of train_leaf or train_merge must be true")
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    rng = np.random.default_rng(int(seed))
    last_metrics: Dict[str, float] = {}
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_metrics: Dict[str, list[float]] = {}
        for batch_docs in _iter_batches(docs, batch_size=int(batch_size), rng=rng):
            batch = _doc_batch_tensors(batch_docs, device=device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = _loss_for_batch(
                model,
                batch,
                exact_leaves=bool(exact_leaves),
                root_weight=float(root_weight),
                leaf_weight=float(leaf_weight),
                merge_weight=float(merge_weight),
                leaf_count_loss_weight=float(leaf_count_loss_weight),
                leaf_endpoint_loss_weight=float(leaf_endpoint_loss_weight),
                merge_count_loss_weight=float(merge_count_loss_weight),
                merge_endpoint_loss_weight=float(merge_endpoint_loss_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(float(value))
        last_metrics = {
            key: float(np.mean(values)) for key, values in epoch_metrics.items()
        }
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            payload = {
                "phase": str(phase),
                "epoch": int(epoch),
                "epochs": int(epochs),
                **last_metrics,
            }
            progress_path.open("a", encoding="utf-8").write(
                json.dumps(payload, sort_keys=True) + "\n"
            )
            print(json.dumps(payload, sort_keys=True), flush=True)
    return last_metrics


@torch.inference_mode()
def _eval(
    model: ExplicitMarkovSketchModel,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    batch_size: int,
    exact_leaves: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "root_abs": 0.0,
        "root_exact": 0.0,
        "leaf_count_abs": 0.0,
        "leaf_exact": 0.0,
        "leaf_first": 0.0,
        "leaf_last": 0.0,
        "merge_count_abs": 0.0,
        "merge_exact": 0.0,
        "merge_first": 0.0,
        "merge_last": 0.0,
        "n_docs": 0.0,
        "n_leaf": 0.0,
        "n_merge": 0.0,
    }
    for start in range(0, len(docs), int(batch_size)):
        batch_docs = tuple(docs[start : start + int(batch_size)])
        batch = _doc_batch_tensors(batch_docs, device=device)
        leaf_states = (
            model.exact_leaf_states(
                batch["leaf_counts"],
                batch["leaf_first"],
                batch["leaf_last"],
            )
            if exact_leaves
            else model.encode_tokens(batch["tokens"])
        )
        root_state, merge_states = model.merge_tree(leaf_states)
        leaf_count, leaf_first_logits, leaf_last_logits = model.split(leaf_states)
        merge_count, merge_first_logits, merge_last_logits = model.split(merge_states)
        root_pred = model.count(root_state)
        leaf_pred_count = leaf_count * float(model.target_scale)
        merge_pred_count = merge_count * float(model.target_scale)
        leaf_first = torch.argmax(leaf_first_logits, dim=-1)
        leaf_last = torch.argmax(leaf_last_logits, dim=-1)
        merge_first = torch.argmax(merge_first_logits, dim=-1)
        merge_last = torch.argmax(merge_last_logits, dim=-1)
        leaf_exact = (
            torch.round(leaf_pred_count).to(torch.long).eq(
                torch.round(batch["leaf_counts"]).to(torch.long)
            )
            & leaf_first.eq(batch["leaf_first"])
            & leaf_last.eq(batch["leaf_last"])
        )
        merge_exact = (
            torch.round(merge_pred_count).to(torch.long).eq(
                torch.round(batch["merge_counts"]).to(torch.long)
            )
            & merge_first.eq(batch["merge_first"])
            & merge_last.eq(batch["merge_last"])
        )
        totals["root_abs"] += float(torch.abs(root_pred - batch["root_counts"]).sum().cpu())
        totals["root_exact"] += float(
            torch.round(root_pred).to(torch.long).eq(
                torch.round(batch["root_counts"]).to(torch.long)
            ).to(torch.float32).sum().cpu()
        )
        totals["leaf_count_abs"] += float(
            torch.abs(leaf_pred_count - batch["leaf_counts"]).sum().cpu()
        )
        totals["leaf_exact"] += float(leaf_exact.to(torch.float32).sum().cpu())
        totals["leaf_first"] += float(leaf_first.eq(batch["leaf_first"]).to(torch.float32).sum().cpu())
        totals["leaf_last"] += float(leaf_last.eq(batch["leaf_last"]).to(torch.float32).sum().cpu())
        totals["merge_count_abs"] += float(
            torch.abs(merge_pred_count - batch["merge_counts"]).sum().cpu()
        )
        totals["merge_exact"] += float(merge_exact.to(torch.float32).sum().cpu())
        totals["merge_first"] += float(merge_first.eq(batch["merge_first"]).to(torch.float32).sum().cpu())
        totals["merge_last"] += float(merge_last.eq(batch["merge_last"]).to(torch.float32).sum().cpu())
        totals["n_docs"] += float(len(batch_docs))
        totals["n_leaf"] += float(batch["leaf_counts"].numel())
        totals["n_merge"] += float(batch["merge_counts"].numel())
    return {
        "root_mae": totals["root_abs"] / max(1.0, totals["n_docs"]),
        "root_exact_match": totals["root_exact"] / max(1.0, totals["n_docs"]),
        "leaf_count_mae": totals["leaf_count_abs"] / max(1.0, totals["n_leaf"]),
        "leaf_exact_match": totals["leaf_exact"] / max(1.0, totals["n_leaf"]),
        "leaf_first_accuracy": totals["leaf_first"] / max(1.0, totals["n_leaf"]),
        "leaf_last_accuracy": totals["leaf_last"] / max(1.0, totals["n_leaf"]),
        "merge_count_mae": totals["merge_count_abs"] / max(1.0, totals["n_merge"]),
        "merge_exact_match": totals["merge_exact"] / max(1.0, totals["n_merge"]),
        "merge_first_accuracy": totals["merge_first"] / max(1.0, totals["n_merge"]),
        "merge_last_accuracy": totals["merge_last"] / max(1.0, totals["n_merge"]),
    }


def _render_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Explicit Markov Sketch Learning",
        "",
        f"- Benchmark: `{summary['run']['benchmark']}`",
        f"- Train docs: `{summary['run']['train_docs']}`",
        f"- Device: `{summary['run']['device']}`",
        "",
        "| split | exact leaves root MAE | exact leaves merge exact | learned leaves root MAE | learned leaf exact | learned merge exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        exact = summary["splits"][split]["exact_leaves"]
        learned = summary["splits"][split]["learned_leaves"]
        lines.append(
            f"| {split} | {exact['root_mae']:.6g} | {exact['merge_exact_match']:.6g} | "
            f"{learned['root_mae']:.6g} | {learned['leaf_exact_match']:.6g} | "
            f"{learned['merge_exact_match']:.6g} |"
        )
    lines.extend(["", "## Test Details", "", "```json"])
    lines.append(json.dumps(summary["splits"]["test"], indent=2, sort_keys=True))
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="recoverable_v5_t128")
    parser.add_argument("--train-docs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--merge-epochs", type=int, default=200)
    parser.add_argument("--leaf-endpoint-pretrain-epochs", type=int, default=0)
    parser.add_argument("--leaf-epochs", type=int, default=200)
    parser.add_argument("--merge-lr", type=float, default=2e-3)
    parser.add_argument("--leaf-endpoint-lr", type=float, default=1e-2)
    parser.add_argument("--leaf-lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--exact-logit", type=float, default=8.0)
    parser.add_argument(
        "--leaf-encoder",
        choices=("mlp", "transition_table"),
        default="mlp",
        help="Leaf summarizer used for learned leaves.",
    )
    parser.add_argument(
        "--merge-count-mode",
        choices=("mlp", "additive_join_table"),
        default="mlp",
        help="Merge count update used for learned g_theta.",
    )
    parser.add_argument(
        "--freeze-merge-in-leaf-stage",
        action="store_true",
        help="Keep the pretrained merge fixed while training the token leaf summarizer.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_explicit_sketch_{_timestamp()}"),
    )
    args = parser.parse_args()

    _set_global_seed(int(args.seed))
    device = (
        torch.device(f"cuda:{int(args.cuda_device)}")
        if bool(args.use_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    train_docs, val_docs, test_docs, prepared = _load_docs(
        benchmark=str(args.benchmark),
        train_docs=int(args.train_docs),
        seed=int(args.seed),
    )
    all_docs = tuple(train_docs) + tuple(val_docs) + tuple(test_docs)
    _n_leaves, leaf_tokens, n_regimes, vocab_size = _shape_info(all_docs)
    model = ExplicitMarkovSketchModel(
        vocab_size=int(vocab_size),
        leaf_tokens=int(leaf_tokens),
        n_regimes=int(n_regimes),
        target_scale=_target_scale(train_docs),
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        exact_logit=float(args.exact_logit),
        leaf_encoder=str(args.leaf_encoder),
        merge_count_mode=str(args.merge_count_mode),
    ).to(device)

    merge_final = _train_phase(
        model,
        train_docs,
        device=device,
        batch_size=int(args.batch_size),
        epochs=int(args.merge_epochs),
        lr=float(args.merge_lr),
        exact_leaves=True,
        root_weight=0.2,
        leaf_weight=0.0,
        merge_weight=1.0,
        seed=int(args.seed) + 100,
        progress_path=progress_path,
        phase="merge_oracle_sketch",
        train_leaf=False,
        train_merge=True,
    )
    leaf_endpoint_pretrain_final: Dict[str, float] | None = None
    if int(args.leaf_endpoint_pretrain_epochs) > 0:
        leaf_endpoint_pretrain_final = _train_phase(
            model,
            train_docs,
            device=device,
            batch_size=int(args.batch_size),
            epochs=int(args.leaf_endpoint_pretrain_epochs),
            lr=float(args.leaf_endpoint_lr),
            exact_leaves=False,
            root_weight=0.0,
            leaf_weight=1.0,
            merge_weight=0.0,
            leaf_count_loss_weight=0.0,
            leaf_endpoint_loss_weight=1.0,
            seed=int(args.seed) + 150,
            progress_path=progress_path,
            phase="token_endpoint_pretrain",
            train_leaf=True,
            train_merge=False,
        )
    leaf_final = _train_phase(
        model,
        train_docs,
        device=device,
        batch_size=int(args.batch_size),
        epochs=int(args.leaf_epochs),
        lr=float(args.leaf_lr),
        exact_leaves=False,
        root_weight=0.2,
        leaf_weight=1.0,
        merge_weight=1.0,
        seed=int(args.seed) + 200,
        progress_path=progress_path,
        phase="token_leaf_to_sketch",
        train_leaf=True,
        train_merge=not bool(args.freeze_merge_in_leaf_stage),
    )
    splits = {}
    for name, docs in (("train", train_docs), ("val", val_docs), ("test", test_docs)):
        splits[name] = {
            "exact_leaves": _eval(
                model,
                docs,
                device=device,
                batch_size=int(args.batch_size),
                exact_leaves=True,
            ),
            "learned_leaves": _eval(
                model,
                docs,
                device=device,
                batch_size=int(args.batch_size),
                exact_leaves=False,
            ),
        }
    summary = {
        "run": {
            "benchmark": str(args.benchmark),
            "train_docs": int(args.train_docs),
            "seed": int(args.seed),
            "device": str(device),
            "target_scale": float(_target_scale(train_docs)),
            "n_regimes": int(n_regimes),
            "vocab_size": int(vocab_size),
            "leaf_tokens": int(leaf_tokens),
            "exact_logit": float(args.exact_logit),
            "leaf_encoder": str(args.leaf_encoder),
            "merge_count_mode": str(args.merge_count_mode),
            "freeze_merge_in_leaf_stage": bool(args.freeze_merge_in_leaf_stage),
            "prepared_data_root": str(prepared.get("prepared_data_root", "")),
        },
        "train_final": {
            "merge_oracle_sketch": merge_final,
            "token_endpoint_pretrain": leaf_endpoint_pretrain_final,
            "token_leaf_to_sketch": leaf_final,
        },
        "splits": splits,
    }
    torch.save(model.state_dict(), output_root / "model_state.pt")
    summary["run"]["model_state"] = str(output_root / "model_state.pt")
    (output_root / "explicit_sketch_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "explicit_sketch_summary.md").write_text(
        _render_md(_jsonable(summary)),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary["splits"]["test"]), indent=2, sort_keys=True))
    print(output_root / "explicit_sketch_summary.json")
    print(output_root / "explicit_sketch_summary.md")


if __name__ == "__main__":
    main()
