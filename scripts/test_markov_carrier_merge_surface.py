#!/usr/bin/env python3
"""Directly overtrain the carrier-projection Markov merge surface.

This bypasses the leaf FNO and full tree objective.  It trains the actual
FNOCountSketch carrier-projection count-slot merger on exact Markov merge
pairs from prepared recoverable-DGP documents, then evaluates exact leaves
through the learned merge.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _load_fno_docs,
    prepare_markov_full_doc_anchor_diagnostics_data,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    FNOCountSketch,
    _FNOCountDoc,
    _eval_fno_exact_sketch_direct_metrics,
    _set_global_seed,
)
from test_markov_exact_progression import (  # noqa: E402
    _evaluate_exact_leaf_merger,
    _root_support_max,
    _validate_uniform_leaf_shape,
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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_prepared_docs(
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
    train_all = _load_fno_docs(Path(str(prepared["train_fno_docs_json"])))
    val_docs = _load_fno_docs(Path(str(prepared["val_fno_docs_json"])))
    test_docs = _load_fno_docs(Path(str(prepared["test_fno_docs_json"])))
    return tuple(train_all[: int(train_docs)]), tuple(val_docs), tuple(test_docs), prepared


def _vocab_size(docs: Sequence[_FNOCountDoc]) -> int:
    max_token = 0
    for doc in docs:
        for leaf in doc.leaf_token_ids:
            for token in leaf:
                max_token = max(max_token, int(token))
    return int(max_token + 1)


def _one_hot(index: int, *, n_regimes: int) -> list[float]:
    return [1.0 if int(i) == int(index) else 0.0 for i in range(int(n_regimes))]


def _summary_tuple(
    *,
    count: float,
    first: int,
    last: int,
    target_scale: float,
    n_regimes: int,
) -> list[float]:
    return (
        [float(count) / float(target_scale)]
        + _one_hot(int(first), n_regimes=int(n_regimes))
        + _one_hot(int(last), n_regimes=int(n_regimes))
    )


def _merge_pair_tensors(
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: float,
    n_regimes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_rows: list[list[float]] = []
    right_rows: list[list[float]] = []
    target_counts: list[float] = []
    target_joins: list[float] = []
    for doc in docs:
        current = [
            (float(count), int(first), int(last))
            for count, first, last in zip(
                doc.leaf_counts,
                doc.leaf_first_regimes,
                doc.leaf_last_regimes,
            )
        ]
        while len(current) > 1:
            next_level: list[tuple[float, int, int]] = []
            pair_count = len(current) // 2
            for pair_idx in range(pair_count):
                left = current[2 * pair_idx]
                right = current[2 * pair_idx + 1]
                join = 0.0 if int(left[2]) == int(right[1]) else 1.0
                merged = (float(left[0]) + float(right[0]) + join, int(left[1]), int(right[2]))
                left_rows.append(
                    _summary_tuple(
                        count=float(left[0]),
                        first=int(left[1]),
                        last=int(left[2]),
                        target_scale=float(target_scale),
                        n_regimes=int(n_regimes),
                    )
                )
                right_rows.append(
                    _summary_tuple(
                        count=float(right[0]),
                        first=int(right[1]),
                        last=int(right[2]),
                        target_scale=float(target_scale),
                        n_regimes=int(n_regimes),
                    )
                )
                target_counts.append(float(merged[0]))
                target_joins.append(float(join))
                next_level.append(merged)
            if len(current) % 2 == 1:
                next_level.append(current[-1])
            current = next_level
    return (
        torch.tensor(left_rows, dtype=torch.float32, device=device),
        torch.tensor(right_rows, dtype=torch.float32, device=device),
        torch.tensor(target_counts, dtype=torch.float32, device=device),
        torch.tensor(target_joins, dtype=torch.float32, device=device),
    )


def _build_model(
    *,
    docs: Sequence[_FNOCountDoc],
    target_scale: float,
    n_regimes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> FNOCountSketch:
    return FNOCountSketch(
        vocab_size=_vocab_size(docs),
        leaf_tokens=int(len(docs[0].leaf_token_ids[0])),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        fno_width=int(args.fno_width),
        fno_n_modes=int(args.fno_n_modes),
        fno_n_layers=int(args.fno_n_layers),
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="carrier_projection",
        score_merge_mode="gated_affine",
        tree_model_version="v2",
        runtime_count_discretization=str(
            getattr(args, "runtime_count_discretization", "continuous")
        ),
    ).to(device=device)


@torch.inference_mode()
def _pair_metrics(
    model: FNOCountSketch,
    left_summary: torch.Tensor,
    right_summary: torch.Tensor,
    target_count: torch.Tensor,
    target_join: torch.Tensor,
    *,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    count_abs = 0.0
    count_exact = 0.0
    join_hits = 0.0
    total = int(target_count.shape[0])
    for start in range(0, total, int(batch_size)):
        stop = min(total, start + int(batch_size))
        left_state = model.encode_summary(left_summary[start:stop])
        right_state = model.encode_summary(right_summary[start:stop])
        merged = model._merge_summary_spec_states(left_state, right_state)
        pred_count = model.predict_count_from_state(merged)
        join_prob = model.predict_join_prob_from_states(left_state, right_state)
        count_abs += float(torch.abs(pred_count - target_count[start:stop]).sum().detach().cpu())
        count_exact += float(
            torch.round(pred_count)
            .to(dtype=torch.long)
            .eq(torch.round(target_count[start:stop]).to(dtype=torch.long))
            .to(dtype=torch.float32)
            .sum()
            .detach()
            .cpu()
        )
        join_hits += float(
            join_prob.ge(0.5)
            .eq(target_join[start:stop].ge(0.5))
            .to(dtype=torch.float32)
            .sum()
            .detach()
            .cpu()
        )
    denom = float(max(1, total))
    return {
        "pair_count_mae": float(count_abs / denom),
        "pair_count_exact": float(count_exact / denom),
        "pair_join_accuracy": float(join_hits / denom),
        "n_pairs": float(total),
    }


def _leaf_tensors(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[Sequence[int]] = []
    counts: list[float] = []
    first: list[int] = []
    last: list[int] = []
    for doc in docs:
        for tokens, count, first_regime, last_regime in zip(
            doc.leaf_token_ids,
            doc.leaf_counts,
            doc.leaf_first_regimes,
            doc.leaf_last_regimes,
        ):
            rows.append(tuple(int(token) for token in tokens))
            counts.append(float(count))
            first.append(int(first_regime))
            last.append(int(last_regime))
    max_len = max((len(row) for row in rows), default=int(model.leaf_tokens))
    token_tensor = torch.full(
        (len(rows), int(max_len)),
        int(model.pad_id),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((len(rows), int(max_len)), dtype=torch.float32, device=device)
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        token_tensor[row_idx, : len(row)] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
        mask[row_idx, : len(row)] = 1.0
    return (
        token_tensor,
        mask,
        torch.tensor(counts, dtype=torch.float32, device=device),
        torch.tensor(first, dtype=torch.long, device=device),
        torch.tensor(last, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def _leaf_metrics(
    model: FNOCountSketch,
    leaf_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    tokens, mask, target_count, target_first, target_last = leaf_data
    count_abs = 0.0
    exact_hits = 0.0
    first_hits = 0.0
    last_hits = 0.0
    total = int(tokens.shape[0])
    for start in range(0, total, int(batch_size)):
        stop = min(total, start + int(batch_size))
        states = model.encode_leaf_tokens_batch(
            tokens[start:stop],
            token_mask=mask[start:stop],
            device=tokens.device,
        )
        pred_count = model.predict_count_from_state(states)
        _count_norm, first_logits, last_logits = model._decode_markov_summary_components(
            states
        )
        pred_first = torch.argmax(first_logits, dim=-1)
        pred_last = torch.argmax(last_logits, dim=-1)
        count_abs += float(torch.abs(pred_count - target_count[start:stop]).sum().detach().cpu())
        first_match = pred_first.eq(target_first[start:stop])
        last_match = pred_last.eq(target_last[start:stop])
        exact = (
            torch.round(pred_count)
            .to(dtype=torch.long)
            .eq(torch.round(target_count[start:stop]).to(dtype=torch.long))
            & first_match
            & last_match
        )
        exact_hits += float(exact.to(dtype=torch.float32).sum().detach().cpu())
        first_hits += float(first_match.to(dtype=torch.float32).sum().detach().cpu())
        last_hits += float(last_match.to(dtype=torch.float32).sum().detach().cpu())
    denom = float(max(1, total))
    return {
        "leaf_count_mae": float(count_abs / denom),
        "leaf_exact": float(exact_hits / denom),
        "leaf_first_accuracy": float(first_hits / denom),
        "leaf_last_accuracy": float(last_hits / denom),
        "n_leaves": float(total),
    }


def _train_surface(
    model: FNOCountSketch,
    left_summary: torch.Tensor,
    right_summary: torch.Tensor,
    target_count: torch.Tensor,
    target_join: torch.Tensor,
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.train()
    params = list(model.count_slot_merger.parameters())
    if model.join_bit_head is not None:
        params.extend(model.join_bit_head.parameters())
    if model.residual_slot_merger is not None:
        params.extend(model.residual_slot_merger.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    n = int(target_count.shape[0])
    best: Dict[str, Any] = {"epoch": 0, "val_pair_count_mae": float("inf")}
    history: list[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        order = torch.randperm(n, device=target_count.device)
        total_loss = 0.0
        for start in range(0, n, int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            left_batch = left_summary.index_select(0, idx)
            right_batch = right_summary.index_select(0, idx)
            truth_count = target_count.index_select(0, idx)
            if float(args.merge_count_jitter) > 0.0:
                left_batch = left_batch.clone()
                right_batch = right_batch.clone()
                left_noise = torch.randn_like(truth_count) * float(args.merge_count_jitter)
                right_noise = torch.randn_like(truth_count) * float(args.merge_count_jitter)
                left_batch[..., 0] = (
                    left_batch[..., 0] * float(model.target_scale) + left_noise
                ) / float(model.target_scale)
                right_batch[..., 0] = (
                    right_batch[..., 0] * float(model.target_scale) + right_noise
                ) / float(model.target_scale)
                truth_count = truth_count + left_noise + right_noise
            left_state = model.encode_summary(left_batch)
            right_state = model.encode_summary(right_batch)
            merged = model._merge_summary_spec_states(left_state, right_state)
            pred_norm = model.predict_norm_from_state(merged)
            truth_norm = truth_count / float(model.target_scale)
            count_loss = F.mse_loss(pred_norm, truth_norm)
            join_logits = model.predict_join_logit_from_states(left_state, right_state)
            join_loss = F.binary_cross_entropy_with_logits(
                join_logits,
                target_join.index_select(0, idx),
            )
            loss = count_loss + float(args.join_weight) * join_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(idx.shape[0])
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            train_metrics = _pair_metrics(
                model,
                left_summary,
                right_summary,
                target_count,
                target_join,
                batch_size=int(args.eval_batch_size),
            )
            val_metrics = _pair_metrics(
                model,
                *val_data,
                batch_size=int(args.eval_batch_size),
            )
            row = {
                "epoch": int(epoch),
                "loss": float(total_loss / float(max(1, n))),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            if float(val_metrics["pair_count_mae"]) < float(best["val_pair_count_mae"]):
                best = {
                    "epoch": int(epoch),
                    "val_pair_count_mae": float(val_metrics["pair_count_mae"]),
                    "val_pair_count_exact": float(val_metrics["pair_count_exact"]),
                    "state_dict": {
                        key: (
                            value.detach().cpu().clone()
                            if isinstance(value, torch.Tensor)
                            else copy.deepcopy(value)
                        )
                        for key, value in model.state_dict().items()
                        if str(key) != "_metadata"
                    },
                }
            print(json.dumps(_jsonable(row), sort_keys=True), flush=True)
    if "state_dict" in best:
        model.load_state_dict(best["state_dict"])
    best.pop("state_dict", None)
    return {"best": best, "history": history}


def _freeze_merge_surface(model: FNOCountSketch) -> None:
    modules = (
        model.count_slot_merger,
        model.join_bit_head,
        model.residual_slot_merger,
    )
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(False)


def _train_leaf_surface(
    model: FNOCountSketch,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if int(args.leaf_epochs) <= 0:
        return {"skipped": True}
    _freeze_merge_surface(model)
    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(
        trainable,
        lr=float(args.leaf_lr),
        weight_decay=float(args.leaf_weight_decay),
    )
    tokens, mask, target_count, target_first, target_last = train_data
    n = int(tokens.shape[0])
    history: list[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "val_leaf_count_mae": float("inf")}
    for epoch in range(1, int(args.leaf_epochs) + 1):
        model.train()
        order = torch.randperm(n, device=tokens.device)
        total_loss = 0.0
        for start in range(0, n, int(args.leaf_batch_size)):
            idx = order[start : start + int(args.leaf_batch_size)]
            states = model.encode_leaf_tokens_batch(
                tokens.index_select(0, idx),
                token_mask=mask.index_select(0, idx),
                device=tokens.device,
            )
            count_norm, first_logits, last_logits = model._decode_markov_summary_components(
                states
            )
            truth_norm = target_count.index_select(0, idx) / float(model.target_scale)
            loss = (
                F.mse_loss(count_norm, truth_norm)
                + F.cross_entropy(first_logits, target_first.index_select(0, idx))
                + F.cross_entropy(last_logits, target_last.index_select(0, idx))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(idx.shape[0])
        if (
            epoch == 1
            or epoch % int(args.leaf_eval_interval) == 0
            or epoch == int(args.leaf_epochs)
        ):
            train_metrics = _leaf_metrics(
                model,
                train_data,
                batch_size=int(args.eval_batch_size),
            )
            val_metrics = _leaf_metrics(
                model,
                val_data,
                batch_size=int(args.eval_batch_size),
            )
            row = {
                "epoch": int(epoch),
                "loss": float(total_loss / float(max(1, n))),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            if float(val_metrics["leaf_count_mae"]) < float(best["val_leaf_count_mae"]):
                best = {
                    "epoch": int(epoch),
                    "val_leaf_count_mae": float(val_metrics["leaf_count_mae"]),
                    "val_leaf_exact": float(val_metrics["leaf_exact"]),
                }
            print(json.dumps(_jsonable({"leaf": row}), sort_keys=True), flush=True)
    return {"best": best, "history": history}


def _render_markdown(summary: Mapping[str, Any]) -> str:
    run = dict(summary.get("run") or {})
    best = dict((summary.get("training") or {}).get("best") or {})
    test_pairs = dict((summary.get("pair_metrics") or {}).get("test") or {})
    test_tree = dict((summary.get("tree_metrics") or {}).get("test") or {})
    learned_tree = dict((summary.get("learned_tree_metrics") or {}).get("test") or {})
    lines = [
        "# Markov Carrier Merge Surface",
        "",
        f"- Benchmark: `{run.get('benchmark')}`",
        f"- Train docs: `{run.get('train_docs')}`",
        f"- Device: `{run.get('device')}`",
        f"- Best epoch: `{best.get('epoch')}`",
        f"- Test pair count MAE: `{test_pairs.get('pair_count_mae')}`",
        f"- Test pair exact rate: `{test_pairs.get('pair_count_exact')}`",
        f"- Test exact-leaf learned-merge root MAE: `{test_tree.get('step1_root_mae')}`",
        f"- Test learned-leaf learned-merge root MAE: `{learned_tree.get('root_direct_count_mae')}`",
        "",
        "```json",
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="recoverable_v5_t128")
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--join-weight", type=float, default=0.2)
    parser.add_argument("--merge-count-jitter", type=float, default=0.0)
    parser.add_argument(
        "--runtime-count-discretization",
        choices=("continuous", "st_round"),
        default="continuous",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--leaf-epochs", type=int, default=0)
    parser.add_argument("--leaf-batch-size", type=int, default=512)
    parser.add_argument("--leaf-eval-interval", type=int, default=25)
    parser.add_argument("--leaf-lr", type=float, default=1e-3)
    parser.add_argument("--leaf-weight-decay", type=float, default=0.0)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--fno-width", type=int, default=128)
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_carrier_merge_surface_{_timestamp()}"),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _set_global_seed(int(args.seed))
    device = (
        torch.device(f"cuda:{int(args.cuda_device)}")
        if bool(args.use_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_docs, val_docs, test_docs, prepared = _load_prepared_docs(
        benchmark=str(args.benchmark),
        train_docs=int(args.train_docs),
        seed=int(args.seed),
    )
    all_docs = tuple(train_docs) + tuple(val_docs) + tuple(test_docs)
    n_leaves, n_regimes = _validate_uniform_leaf_shape(all_docs)
    target_scale = float(_root_support_max(train_docs))
    model = _build_model(
        docs=all_docs,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        args=args,
        device=device,
    )
    train_pairs = _merge_pair_tensors(
        train_docs,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    val_pairs = _merge_pair_tensors(
        val_docs,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    test_pairs = _merge_pair_tensors(
        test_docs,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    training = _train_surface(
        model,
        *train_pairs,
        val_pairs,
        args=args,
    )
    pair_metrics = {
        "train": _pair_metrics(model, *train_pairs, batch_size=int(args.eval_batch_size)),
        "val": _pair_metrics(model, *val_pairs, batch_size=int(args.eval_batch_size)),
        "test": _pair_metrics(model, *test_pairs, batch_size=int(args.eval_batch_size)),
    }
    leaf_training = _train_leaf_surface(
        model,
        _leaf_tensors(model, train_docs, device=device),
        _leaf_tensors(model, val_docs, device=device),
        args=args,
    )
    tree_metrics = {
        "train": _evaluate_exact_leaf_merger(
            model,
            train_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
        ),
        "val": _evaluate_exact_leaf_merger(
            model,
            val_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
        ),
        "test": _evaluate_exact_leaf_merger(
            model,
            test_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
        ),
    }
    learned_tree_metrics = {}
    if int(args.leaf_epochs) > 0:
        learned_tree_metrics = {
            "train": _eval_fno_exact_sketch_direct_metrics(
                model,
                train_docs,
                device=device,
                pack_mode="fixed_fused",
                runtime_bucket_mode="leaf_count_auto_queue",
                max_docs=max(1, len(train_docs)),
                phi_pair_calibration_max_nodes=512,
            ),
            "val": _eval_fno_exact_sketch_direct_metrics(
                model,
                val_docs,
                device=device,
                pack_mode="fixed_fused",
                runtime_bucket_mode="leaf_count_auto_queue",
                max_docs=max(1, len(val_docs)),
                phi_pair_calibration_max_nodes=512,
            ),
            "test": _eval_fno_exact_sketch_direct_metrics(
                model,
                test_docs,
                device=device,
                pack_mode="fixed_fused",
                runtime_bucket_mode="leaf_count_auto_queue",
                max_docs=max(1, len(test_docs)),
                phi_pair_calibration_max_nodes=512,
            ),
        }
    torch.save(model.state_dict(), output_root / "final_model_state.pt")
    summary: Dict[str, Any] = {
        "run": {
            "benchmark": str(args.benchmark),
            "train_docs": int(args.train_docs),
            "seed": int(args.seed),
            "device": str(device),
            "n_leaves": int(n_leaves),
            "n_regimes": int(n_regimes),
            "target_scale": float(target_scale),
            "prepared_data_root": str(prepared.get("prepared_data_root", "")),
            "final_model_state": str(output_root / "final_model_state.pt"),
        },
        "training": _jsonable(training),
        "leaf_training": _jsonable(leaf_training),
        "pair_metrics": _jsonable(pair_metrics),
        "tree_metrics": _jsonable(tree_metrics),
        "learned_tree_metrics": _jsonable(learned_tree_metrics),
    }
    summary_json = output_root / "carrier_merge_surface_summary.json"
    summary_md = output_root / "carrier_merge_surface_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(str(summary_json))
    print(str(summary_md))


if __name__ == "__main__":
    main()
