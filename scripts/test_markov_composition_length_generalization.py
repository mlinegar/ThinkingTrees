#!/usr/bin/env python3
"""Train Markov merge composition at one tree length and test another.

This harness isolates the composition question.  It uses exact leaf sketches
``(count, first, last)`` and trains only the carrier-projection merge surface
on internal merge pairs from a shallow balanced tree.  The same learned merge is
then evaluated on a deeper partition of the exact same DGP.
"""

from __future__ import annotations

import argparse
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

from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    _FNOCountDoc,
    _prepare_fno_count_docs,
    _set_global_seed,
)
from test_markov_carrier_merge_surface import (  # noqa: E402
    _build_model,
    _jsonable,
    _merge_pair_tensors,
    _pair_metrics,
    _train_surface,
)
from test_markov_exact_progression import (  # noqa: E402
    _evaluate_exact_leaf_merger,
    _root_support_max,
    _validate_uniform_leaf_shape,
    step0_exact,
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _leaf_tokens_for_l(*, doc_tokens: int, leaves: int) -> int:
    if int(leaves) <= 0:
        raise ValueError("leaf count L must be positive")
    if int(doc_tokens) % int(leaves) != 0:
        raise ValueError(
            f"doc_tokens={doc_tokens} must be divisible by L={leaves} for this harness"
        )
    return int(doc_tokens) // int(leaves)


def _make_config(args: argparse.Namespace, *, max_train_docs: int) -> OPSCountConfig:
    mean_seg_len = int(
        round(
            float(args.doc_tokens)
            / max(1.0, 0.5 * (float(args.min_segments) + float(args.max_segments)))
        )
    )
    return OPSCountConfig(
        generator_profile="hazard_topic",
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.doc_tokens),
        max_tokens=int(args.doc_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(mean_seg_len),
        max_seg_len=int(mean_seg_len),
        hazard_switch_prob=float(args.expected_boundaries)
        / float(max(1, int(args.doc_tokens) - 1)),
        fixed_leaf_tokens=_leaf_tokens_for_l(
            doc_tokens=int(args.doc_tokens),
            leaves=int(args.train_leaves),
        ),
        train_docs=int(max_train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        data_seed=int(args.seed),
        model_seed=int(args.seed),
    )


def _vocab_max(docs: Sequence[_FNOCountDoc]) -> int:
    return max(
        1,
        max(
            int(token)
            for doc in docs
            for leaf in doc.leaf_token_ids
            for token in leaf
        )
        + 1,
    )


def _pair_input_signatures(docs: Sequence[_FNOCountDoc]) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for doc in docs:
        current = [
            (int(round(float(count))), int(first), int(last))
            for count, first, last in zip(
                doc.leaf_counts,
                doc.leaf_first_regimes,
                doc.leaf_last_regimes,
            )
        ]
        while len(current) > 1:
            nxt: list[tuple[int, int, int]] = []
            pair_count = len(current) // 2
            for pair_idx in range(pair_count):
                left = current[2 * pair_idx]
                right = current[2 * pair_idx + 1]
                rows.append(
                    (
                        int(left[0]),
                        int(left[1]),
                        int(left[2]),
                        int(right[0]),
                        int(right[1]),
                        int(right[2]),
                    )
                )
                join = 0 if int(left[2]) == int(right[1]) else 1
                nxt.append((int(left[0]) + int(right[0]) + int(join), int(left[1]), int(right[2])))
            if len(current) % 2 == 1:
                nxt.append(current[-1])
            current = nxt
    return tuple(rows)


def _signature_metrics(
    train_docs: Sequence[_FNOCountDoc],
    eval_docs: Sequence[_FNOCountDoc],
) -> Dict[str, float]:
    train_rows = _pair_input_signatures(train_docs)
    eval_rows = _pair_input_signatures(eval_docs)
    train_unique = set(train_rows)
    eval_unique = set(eval_rows)
    seen = sum(1 for row in eval_rows if row in train_unique)
    return {
        "train_pair_rows": float(len(train_rows)),
        "train_unique_input_signatures": float(len(train_unique)),
        "eval_pair_rows": float(len(eval_rows)),
        "eval_unique_input_signatures": float(len(eval_unique)),
        "eval_input_seen_rate": float(seen) / float(max(1, len(eval_rows))),
        "eval_unique_seen_rate": (
            float(len(eval_unique & train_unique)) / float(max(1, len(eval_unique)))
        ),
    }


def _tree_eval(
    model: Any,
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    device: torch.device,
) -> Dict[str, Any]:
    return _evaluate_exact_leaf_merger(
        model,
        docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )


def _leaf_summary_tensor(
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for doc in docs:
        count = torch.tensor(
            [float(value) / float(target_scale) for value in doc.leaf_counts],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        first = F.one_hot(
            torch.tensor(
                [int(value) for value in doc.leaf_first_regimes],
                dtype=torch.long,
                device=device,
            ),
            num_classes=int(n_regimes),
        ).to(dtype=torch.float32)
        last = F.one_hot(
            torch.tensor(
                [int(value) for value in doc.leaf_last_regimes],
                dtype=torch.long,
                device=device,
            ),
            num_classes=int(n_regimes),
        ).to(dtype=torch.float32)
        rows.append(torch.cat([count, first, last], dim=-1))
    return torch.stack(rows, dim=0)


def _leaf_target_tensors(
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = torch.tensor(
        [[float(value) for value in doc.leaf_counts] for doc in docs],
        dtype=torch.float32,
        device=device,
    )
    first = torch.tensor(
        [[int(value) for value in doc.leaf_first_regimes] for doc in docs],
        dtype=torch.long,
        device=device,
    )
    last = torch.tensor(
        [[int(value) for value in doc.leaf_last_regimes] for doc in docs],
        dtype=torch.long,
        device=device,
    )
    return count, first, last


def _rollout_loss_for_docs(
    model: Any,
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    device: torch.device,
    join_weight: float,
) -> Dict[str, torch.Tensor]:
    if not docs:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return {
            "loss": zero,
            "count_loss": zero,
            "join_loss": zero,
            "n_merges": zero,
        }
    leaf_summary = _leaf_summary_tensor(
        docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    batch_size, n_leaves, _ = leaf_summary.shape
    current = model.encode_summary(leaf_summary.reshape(batch_size * n_leaves, -1))
    current = current.reshape(batch_size, n_leaves, -1)
    exact_count, exact_first, exact_last = _leaf_target_tensors(docs, device=device)

    count_terms: list[torch.Tensor] = []
    join_terms: list[torch.Tensor] = []
    n_merges = 0
    while int(current.shape[1]) > 1:
        n_nodes = int(current.shape[1])
        n_pairs = int(n_nodes // 2)
        left = current[:, 0 : 2 * n_pairs : 2, :]
        right = current[:, 1 : 2 * n_pairs : 2, :]
        left_count = exact_count[:, 0 : 2 * n_pairs : 2]
        right_count = exact_count[:, 1 : 2 * n_pairs : 2]
        left_first = exact_first[:, 0 : 2 * n_pairs : 2]
        right_first = exact_first[:, 1 : 2 * n_pairs : 2]
        left_last = exact_last[:, 0 : 2 * n_pairs : 2]
        right_last = exact_last[:, 1 : 2 * n_pairs : 2]

        target_join = left_last.ne(right_first).to(dtype=torch.float32)
        target_count = left_count + right_count + target_join
        merged = model._merge_state_pairs(
            left.reshape(batch_size * n_pairs, -1),
            right.reshape(batch_size * n_pairs, -1),
        ).reshape(batch_size, n_pairs, -1)
        pred_norm = model.predict_norm_from_state(
            merged.reshape(batch_size * n_pairs, -1)
        ).reshape(batch_size, n_pairs)
        count_terms.append(
            F.mse_loss(pred_norm, target_count / float(target_scale))
        )
        join_logits = model.predict_join_logit_from_states(
            left.reshape(batch_size * n_pairs, -1),
            right.reshape(batch_size * n_pairs, -1),
        ).reshape(batch_size, n_pairs)
        join_terms.append(F.binary_cross_entropy_with_logits(join_logits, target_join))
        n_merges += int(batch_size * n_pairs)

        next_count = target_count
        next_first = left_first
        next_last = right_last
        if n_nodes % 2 == 1:
            merged = torch.cat([merged, current[:, -1:, :]], dim=1)
            next_count = torch.cat([next_count, exact_count[:, -1:]], dim=1)
            next_first = torch.cat([next_first, exact_first[:, -1:]], dim=1)
            next_last = torch.cat([next_last, exact_last[:, -1:]], dim=1)
        current = merged
        exact_count = next_count
        exact_first = next_first
        exact_last = next_last

    count_loss = torch.stack(count_terms).mean()
    join_loss = (
        torch.stack(join_terms).mean()
        if join_terms
        else torch.zeros((), dtype=torch.float32, device=device)
    )
    loss = count_loss + float(join_weight) * join_loss
    return {
        "loss": loss,
        "count_loss": count_loss,
        "join_loss": join_loss,
        "n_merges": torch.tensor(float(n_merges), dtype=torch.float32, device=device),
    }


@torch.inference_mode()
def _rollout_loss_metrics(
    model: Any,
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    device: torch.device,
    join_weight: float,
) -> Dict[str, float]:
    model.eval()
    terms = _rollout_loss_for_docs(
        model,
        docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
        join_weight=float(join_weight),
    )
    return {
        "rollout_loss": float(terms["loss"].detach().cpu()),
        "rollout_count_loss": float(terms["count_loss"].detach().cpu()),
        "rollout_join_loss": float(terms["join_loss"].detach().cpu()),
        "n_merges": float(terms["n_merges"].detach().cpu()),
    }


def _merge_surface_params(model: Any) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = list(model.count_slot_merger.parameters())
    if model.join_bit_head is not None:
        params.extend(model.join_bit_head.parameters())
    if model.residual_slot_merger is not None:
        params.extend(model.residual_slot_merger.parameters())
    return params


def _train_rollout_surface(
    model: Any,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    *,
    args: argparse.Namespace,
    target_scale: int,
    n_regimes: int,
    device: torch.device,
) -> Dict[str, Any]:
    if int(args.rollout_epochs) <= 0:
        return {"skipped": True}
    params = _merge_surface_params(model)
    opt = torch.optim.AdamW(
        params,
        lr=float(args.rollout_lr),
        weight_decay=float(args.rollout_weight_decay),
    )
    best: Dict[str, Any] = {"epoch": 0, "val_rollout_root_mae": float("inf")}
    history: list[Dict[str, Any]] = []
    train_docs = tuple(train_docs)
    val_docs = tuple(val_docs)
    for epoch in range(1, int(args.rollout_epochs) + 1):
        model.train()
        order = torch.randperm(len(train_docs)).detach().cpu().tolist()
        total_loss = 0.0
        total_docs = 0
        for start in range(0, len(order), int(args.rollout_batch_docs)):
            idx = order[start : start + int(args.rollout_batch_docs)]
            batch = tuple(train_docs[int(i)] for i in idx)
            terms = _rollout_loss_for_docs(
                model,
                batch,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
                join_weight=float(args.rollout_join_weight),
            )
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
            opt.step()
            total_loss += float(terms["loss"].detach().cpu()) * int(len(batch))
            total_docs += int(len(batch))
        if (
            epoch == 1
            or epoch % int(args.rollout_eval_interval) == 0
            or epoch == int(args.rollout_epochs)
        ):
            train_loss = _rollout_loss_metrics(
                model,
                train_docs[: min(len(train_docs), int(args.rollout_eval_docs))],
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
                join_weight=float(args.rollout_join_weight),
            )
            val_loss = _rollout_loss_metrics(
                model,
                val_docs,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
                join_weight=float(args.rollout_join_weight),
            )
            val_tree = _tree_eval(
                model,
                val_docs,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            )
            row = {
                "epoch": int(epoch),
                "loss": float(total_loss / float(max(1, total_docs))),
                "train": train_loss,
                "val": val_loss,
                "val_tree": val_tree,
            }
            history.append(row)
            val_root = float(val_tree["step1_root_mae"])
            if val_root < float(best["val_rollout_root_mae"]):
                best = {
                    "epoch": int(epoch),
                    "val_rollout_root_mae": float(val_root),
                    "val_rollout_loss": float(val_loss["rollout_loss"]),
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                        if str(key) != "_metadata"
                    },
                }
            print(json.dumps(_jsonable({"rollout": row}), sort_keys=True), flush=True)
    if "state_dict" in best:
        model.load_state_dict(best["state_dict"])
    best.pop("state_dict", None)
    return {"best": best, "history": history}


def _render_markdown(summary: Mapping[str, Any]) -> str:
    run = dict(summary.get("run") or {})
    lines = [
        "# Markov Composition Length Generalization",
        "",
        f"- DGP: `hazard_topic`, doc tokens `{run.get('doc_tokens')}`, regimes `{run.get('n_regimes')}`",
        f"- Train partition: `L={run.get('train_leaves')}` leaves (`leaf_tokens={run.get('train_leaf_tokens')}`)",
        f"- Eval partition: `L={run.get('eval_leaves')}` leaves (`leaf_tokens={run.get('eval_leaf_tokens')}`)",
        f"- Selection split: validation pairs from `L={run.get('train_leaves')}` only",
        "",
        "| train docs | L4 test pair MAE | L4 root MAE | L16 test pair MAE | L16 root MAE | L16 input seen |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("runs", []):
        n = int(row.get("train_docs", 0))
        train_label = str(run.get("train_label", "train_L"))
        eval_label = str(run.get("eval_label", "eval_L"))
        train_pair = ((row.get("pair_metrics") or {}).get(train_label) or {}).get("test") or {}
        eval_pair = ((row.get("pair_metrics") or {}).get(eval_label) or {}).get("test") or {}
        train_tree = ((row.get("tree_metrics") or {}).get(train_label) or {}).get("test") or {}
        eval_tree = ((row.get("tree_metrics") or {}).get(eval_label) or {}).get("test") or {}
        sig = ((row.get("signature_metrics") or {}).get(f"{train_label}_train_to_{eval_label}_test") or {})
        lines.append(
            "| "
            f"{n} | "
            f"{float(train_pair.get('pair_count_mae', float('nan'))):.6g} | "
            f"{float(train_tree.get('step1_root_mae', float('nan'))):.6g} | "
            f"{float(eval_pair.get('pair_count_mae', float('nan'))):.6g} | "
            f"{float(eval_tree.get('step1_root_mae', float('nan'))):.6g} | "
            f"{float(sig.get('eval_input_seen_rate', float('nan'))):.3f} |"
        )
    lines.extend(
        [
            "",
            "```json",
            json.dumps(_jsonable(summary), indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_one(
    *,
    args: argparse.Namespace,
    train_docs_n: int,
    train_docs_l_train: Sequence[_FNOCountDoc],
    train_docs_l_eval: Sequence[_FNOCountDoc],
    val_docs_l_train: Sequence[_FNOCountDoc],
    test_docs_l_train: Sequence[_FNOCountDoc],
    val_docs_l_eval: Sequence[_FNOCountDoc],
    test_docs_l_eval: Sequence[_FNOCountDoc],
    all_docs: Sequence[_FNOCountDoc],
    n_regimes: int,
    target_scale: int,
    device: torch.device,
    output_root: Path,
) -> Dict[str, Any]:
    _set_global_seed(int(args.seed) + int(train_docs_n))
    train_docs = tuple(train_docs_l_train[: int(train_docs_n)])
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
    val_pairs_l_train = _merge_pair_tensors(
        val_docs_l_train,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    test_pairs_l_train = _merge_pair_tensors(
        test_docs_l_train,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    val_pairs_l_eval = _merge_pair_tensors(
        val_docs_l_eval,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    test_pairs_l_eval = _merge_pair_tensors(
        test_docs_l_eval,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )

    training = _train_surface(
        model,
        *train_pairs,
        val_pairs_l_train,
        args=args,
    )
    if str(args.rollout_partition) == "eval":
        rollout_train_docs = tuple(train_docs_l_eval[: int(train_docs_n)])
        rollout_val_docs = tuple(val_docs_l_eval)
    else:
        rollout_train_docs = tuple(train_docs)
        rollout_val_docs = tuple(val_docs_l_train)
    rollout_training = _train_rollout_surface(
        model,
        rollout_train_docs,
        rollout_val_docs,
        args=args,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    train_label = f"L{int(args.train_leaves)}"
    eval_label = f"L{int(args.eval_leaves)}"
    pair_metrics = {
        train_label: {
            "train": _pair_metrics(model, *train_pairs, batch_size=int(args.eval_batch_size)),
            "val": _pair_metrics(model, *val_pairs_l_train, batch_size=int(args.eval_batch_size)),
            "test": _pair_metrics(model, *test_pairs_l_train, batch_size=int(args.eval_batch_size)),
        },
        eval_label: {
            "val": _pair_metrics(model, *val_pairs_l_eval, batch_size=int(args.eval_batch_size)),
            "test": _pair_metrics(model, *test_pairs_l_eval, batch_size=int(args.eval_batch_size)),
        },
    }
    tree_metrics = {
        train_label: {
            "train": _tree_eval(
                model,
                train_docs,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            ),
            "val": _tree_eval(
                model,
                val_docs_l_train,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            ),
            "test": _tree_eval(
                model,
                test_docs_l_train,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            ),
        },
        eval_label: {
            "val": _tree_eval(
                model,
                val_docs_l_eval,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            ),
            "test": _tree_eval(
                model,
                test_docs_l_eval,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            ),
        },
    }
    signature_metrics = {
        f"{train_label}_train_to_{train_label}_test": _signature_metrics(
            train_docs,
            test_docs_l_train,
        ),
        f"{train_label}_train_to_{eval_label}_test": _signature_metrics(
            train_docs,
            test_docs_l_eval,
        ),
    }
    state_path = output_root / f"model_state_train_docs_{int(train_docs_n)}.pt"
    torch.save(model.state_dict(), state_path)
    return {
        "train_docs": int(train_docs_n),
        "model_state": str(state_path),
        "runtime_count_discretization": str(args.runtime_count_discretization),
        "training": _jsonable(training),
        "rollout_training": _jsonable(rollout_training),
        "pair_metrics": _jsonable(pair_metrics),
        "tree_metrics": _jsonable(tree_metrics),
        "signature_metrics": _jsonable(signature_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-tokens", type=int, default=128)
    parser.add_argument("--train-leaves", type=int, default=4)
    parser.add_argument("--eval-leaves", type=int, default=16)
    parser.add_argument("--train-doc-grid", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--val-docs", type=int, default=256)
    parser.add_argument("--test-docs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=6)
    parser.add_argument("--expected-boundaries", type=float, default=5.0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
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
    parser.add_argument("--rollout-epochs", type=int, default=0)
    parser.add_argument("--rollout-batch-docs", type=int, default=256)
    parser.add_argument("--rollout-eval-docs", type=int, default=2048)
    parser.add_argument("--rollout-eval-interval", type=int, default=10)
    parser.add_argument("--rollout-lr", type=float, default=3e-4)
    parser.add_argument("--rollout-weight-decay", type=float, default=0.0)
    parser.add_argument("--rollout-join-weight", type=float, default=0.2)
    parser.add_argument("--rollout-partition", choices=("train", "eval"), default="train")
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--fno-width", type=int, default=128)
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=4)
    parser.add_argument("--target-scale", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_composition_l_generalization_{_timestamp()}"),
    )
    args = parser.parse_args()

    if int(args.vocab_size) < int(args.n_regimes):
        raise ValueError("vocab-size must be at least n-regimes for disjoint palettes")
    train_leaf_tokens = _leaf_tokens_for_l(
        doc_tokens=int(args.doc_tokens),
        leaves=int(args.train_leaves),
    )
    eval_leaf_tokens = _leaf_tokens_for_l(
        doc_tokens=int(args.doc_tokens),
        leaves=int(args.eval_leaves),
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _set_global_seed(int(args.seed))
    device = (
        torch.device(f"cuda:{int(args.cuda_device)}")
        if bool(args.use_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_grid = tuple(sorted({int(value) for value in args.train_doc_grid}))
    max_train_docs = max(train_grid)
    config = _make_config(args, max_train_docs=int(max_train_docs))
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    train_l_train_all = _prepare_fno_count_docs(bundle.train_docs, leaf_tokens=int(train_leaf_tokens))
    val_l_train = _prepare_fno_count_docs(bundle.val_docs, leaf_tokens=int(train_leaf_tokens))
    test_l_train = _prepare_fno_count_docs(bundle.test_docs, leaf_tokens=int(train_leaf_tokens))
    train_l_eval_all = _prepare_fno_count_docs(bundle.train_docs, leaf_tokens=int(eval_leaf_tokens))
    val_l_eval = _prepare_fno_count_docs(bundle.val_docs, leaf_tokens=int(eval_leaf_tokens))
    test_l_eval = _prepare_fno_count_docs(bundle.test_docs, leaf_tokens=int(eval_leaf_tokens))

    _validate_uniform_leaf_shape(train_l_train_all + val_l_train + test_l_train)
    _validate_uniform_leaf_shape(train_l_eval_all + val_l_eval + test_l_eval)
    all_docs = (
        tuple(train_l_train_all)
        + tuple(val_l_train)
        + tuple(test_l_train)
        + tuple(train_l_eval_all)
        + tuple(val_l_eval)
        + tuple(test_l_eval)
    )
    target_scale = (
        int(args.target_scale)
        if int(args.target_scale) > 0
        else int(_root_support_max(all_docs))
    )
    n_regimes = int(args.n_regimes)
    vocab_size_observed = _vocab_max(all_docs)
    if vocab_size_observed > int(args.vocab_size):
        raise ValueError(
            f"observed token id exceeds configured vocab: observed={vocab_size_observed} "
            f"configured={args.vocab_size}"
        )

    exact_sanity = {
        f"L{int(args.train_leaves)}": {
            "val": step0_exact(val_l_train),
            "test": step0_exact(test_l_train),
        },
        f"L{int(args.eval_leaves)}": {
            "val": step0_exact(val_l_eval),
            "test": step0_exact(test_l_eval),
        },
    }
    runs: list[Dict[str, Any]] = []
    for train_docs_n in train_grid:
        runs.append(
            _run_one(
                args=args,
                train_docs_n=int(train_docs_n),
                train_docs_l_train=train_l_train_all,
                train_docs_l_eval=train_l_eval_all,
                val_docs_l_train=val_l_train,
                test_docs_l_train=test_l_train,
                val_docs_l_eval=val_l_eval,
                test_docs_l_eval=test_l_eval,
                all_docs=all_docs,
                n_regimes=int(n_regimes),
                target_scale=int(target_scale),
                device=device,
                output_root=output_root,
            )
        )
        summary_so_far = {
            "run": {
                "doc_tokens": int(args.doc_tokens),
                "train_leaves": int(args.train_leaves),
                "eval_leaves": int(args.eval_leaves),
                "train_leaf_tokens": int(train_leaf_tokens),
                "eval_leaf_tokens": int(eval_leaf_tokens),
                "train_label": f"L{int(args.train_leaves)}",
                "eval_label": f"L{int(args.eval_leaves)}",
                "n_regimes": int(n_regimes),
                "vocab_size": int(args.vocab_size),
                "target_scale": int(target_scale),
                "runtime_count_discretization": str(args.runtime_count_discretization),
                "seed": int(args.seed),
                "device": str(device),
                "output_root": str(output_root),
            },
            "config": _jsonable(config.__dict__),
            "exact_sanity": _jsonable(exact_sanity),
            "runs": _jsonable(runs),
        }
        (output_root / "composition_length_generalization_summary.json").write_text(
            json.dumps(summary_so_far, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_root / "composition_length_generalization_summary.md").write_text(
            _render_markdown(summary_so_far),
            encoding="utf-8",
        )

    print(str(output_root / "composition_length_generalization_summary.json"))
    print(str(output_root / "composition_length_generalization_summary.md"))


if __name__ == "__main__":
    main()
