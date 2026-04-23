#!/usr/bin/env python3
"""Markov merge-signal feasibility study.

Uses the exact progression harness as a small lab for the merger question:

  Step 0: exact sketch + exact merge
  Phase 1: exact leaves + learned merge under different merge objectives

The main study stays on the clean `recoverable_v4` benchmark and keeps the
latent carrier opaque by using the shared-theorem-surface path rather than any
slotwise latent decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.diffusion.markov_toy import (
    MarkovToySketch,
    changepoint_count,
    encode_markov_path,
    merge_markov_sketch,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    _load_fno_docs,
    prepare_markov_full_doc_anchor_diagnostics_data,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    FNOCountSketch,
    VALID_MARKOV_MERGE_OBJECTIVE_MODES,
    VALID_MARKOV_MERGE_WEIGHTING_MODES,
    _FNOCountDoc,
    _balanced_exact_sketch_targets,
    _markov_merge_objective_terms_batched,
    _set_global_seed,
    _summary_spec_supervision_terms_batched,
)


# ---------------------------------------------------------------------------
# Exact sketch algebra / Lean worked example
# ---------------------------------------------------------------------------


def _exact_tree_reduce(
    sketches: Sequence[MarkovToySketch],
) -> Tuple[MarkovToySketch, List[MarkovToySketch]]:
    current = list(sketches)
    intermediates: List[MarkovToySketch] = []
    while len(current) > 1:
        nxt = []
        i = 0
        while i + 1 < len(current):
            merged = merge_markov_sketch(current[i], current[i + 1])
            nxt.append(merged)
            intermediates.append(merged)
            i += 2
        if i < len(current):
            nxt.append(current[i])
        current = nxt
    return current[0], intermediates


def test_lean_worked_example() -> None:
    calm, spike = "0", "1"
    leafA = encode_markov_path([calm])
    leafB = encode_markov_path([spike])
    leafC = encode_markov_path([spike])
    leafD = encode_markov_path([calm])

    assert leafA.changepoints == 0 and leafA.start_state == calm and leafA.end_state == calm
    assert leafB.changepoints == 0 and leafB.start_state == spike and leafB.end_state == spike

    naive_sum = (
        leafA.changepoints
        + leafB.changepoints
        + leafC.changepoints
        + leafD.changepoints
    )
    assert naive_sum == 0

    mergeAB = merge_markov_sketch(leafA, leafB)
    mergeCD = merge_markov_sketch(leafC, leafD)
    assert mergeAB.changepoints == 1
    assert mergeCD.changepoints == 1

    root = merge_markov_sketch(mergeAB, mergeCD)
    full_oracle = changepoint_count([calm, spike, spike, calm])
    assert root.changepoints == 2
    assert full_oracle == 2
    assert abs(root.changepoints - full_oracle) == 0


def step0_exact(
    docs: Sequence[_FNOCountDoc],
) -> dict[str, float]:
    errors = []
    max_error = 0.0
    for doc in docs:
        leaf_sketches = []
        for leaf_tokens, first, last, count in zip(
            doc.leaf_token_ids,
            doc.leaf_first_regimes,
            doc.leaf_last_regimes,
            doc.leaf_counts,
        ):
            if len(leaf_tokens) <= 0:
                continue
            sketch = MarkovToySketch(
                changepoints=int(round(float(count))),
                start_state=str(int(first)),
                end_state=str(int(last)),
                length=len(leaf_tokens),
            )
            leaf_sketches.append(sketch)
        root, _ = _exact_tree_reduce(leaf_sketches)
        error = abs(float(root.changepoints) - float(doc.root_count))
        errors.append(error)
        max_error = max(max_error, float(error))
    return {
        "root_mae": float(np.mean(errors)) if errors else 0.0,
        "root_max_error": float(max_error),
        "n_docs": float(len(docs)),
    }


# ---------------------------------------------------------------------------
# Study config / helpers
# ---------------------------------------------------------------------------


DEFAULT_TRAIN_DOC_COUNTS = (256, 1024, 4096)
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class MergeStudyRunSpec:
    benchmark: str
    train_docs: int
    seed: int
    merge_objective: str
    root_loss_weight: float
    merge_weighting: str
    count_head_mode: str
    state_dim: int
    hidden_dim: int
    theorem_feature_dim: int
    theorem_feature_hidden_dim: int
    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    @property
    def label(self) -> str:
        return (
            f"{self.merge_objective}"
            f"__root{self.root_loss_weight:g}"
            f"__{self.merge_weighting}"
            f"__head_{self.count_head_mode}"
            f"__n{self.train_docs}"
            f"__seed{self.seed}"
        )


@dataclass(frozen=True)
class MergeStudyRunResult:
    label: str
    benchmark: str
    train_docs: int
    seed: int
    merge_objective: str
    root_loss_weight: float
    merge_weighting: str
    count_head_mode: str
    state_dim: int
    hidden_dim: int
    theorem_feature_dim: int
    theorem_feature_hidden_dim: int
    n_epochs: int
    best_epoch: int
    best_val_step1_root_mae: float
    train_local_loss_last: float
    train_root_loss_last: float
    train_total_loss_last: float
    merger_grad_norm_root: float
    merger_grad_norm_local: float
    merger_grad_ratio_root_to_local: float
    step1_root_mae: float
    step1_merge_exact_summary_match_rate: float
    step1_count_only_root_mae: float
    step1_endpoint_only_root_mae: float
    merge_first_accuracy: float
    merge_last_accuracy: float
    merge_join_accuracy: float
    per_depth_merge_exact_summary_match_rate: Dict[str, float]
    n_train_used: int
    n_val_used: int
    n_test_used: int


def _timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _resolved_epochs(train_docs: int) -> int:
    if int(train_docs) >= 4096:
        return 150
    return 100


def _balanced_merge_levels(n_leaves: int) -> Tuple[Tuple[int, ...], ...]:
    current = list(range(int(n_leaves)))
    merge_idx = 0
    levels: List[Tuple[int, ...]] = []
    while len(current) > 1:
        level: List[int] = []
        nxt: List[int] = []
        i = 0
        while i < len(current):
            if i + 1 >= len(current):
                nxt.append(current[i])
                i += 1
                continue
            nxt.append(int(n_leaves) + merge_idx)
            level.append(merge_idx)
            merge_idx += 1
            i += 2
        if level:
            levels.append(tuple(level))
        current = nxt
    return tuple(levels)


def _doc_exact_targets_by_level(
    doc: _FNOCountDoc,
) -> Dict[str, Any]:
    exact_targets = _balanced_exact_sketch_targets(
        leaf_counts=doc.leaf_counts,
        leaf_first_regimes=doc.leaf_first_regimes,
        leaf_last_regimes=doc.leaf_last_regimes,
    )
    merge_levels: List[Tuple[Tuple[float, int, int], ...]] = []
    cursor = 0
    current_leaf_count = int(len(doc.leaf_counts))
    while current_leaf_count > 1:
        level_pairs = current_leaf_count // 2
        merge_levels.append(tuple(exact_targets["merge"][cursor : cursor + level_pairs]))
        cursor += level_pairs
        current_leaf_count = level_pairs + (current_leaf_count % 2)
    return {
        "leaf": tuple(exact_targets["leaf"]),
        "merge": tuple(exact_targets["merge"]),
        "merge_join_bits": tuple(exact_targets["merge_join_bits"]),
        "merge_levels": tuple(merge_levels),
        "root": tuple(exact_targets["root"]),
    }


def _load_recoverable_v4_split_docs(
    *,
    benchmark: str,
    train_docs: int,
) -> Dict[str, Any]:
    prepared = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=str(benchmark),
        train_doc_counts=(int(train_docs),),
        seeds=(0,),
    )["prepared"][0]
    return {
        "prepared": prepared,
        "train_all": tuple(_load_fno_docs(Path(prepared["train_fno_docs_json"]))),
        "val": tuple(_load_fno_docs(Path(prepared["val_fno_docs_json"]))),
        "test": tuple(_load_fno_docs(Path(prepared["test_fno_docs_json"]))),
    }


def _select_train_prefix(
    docs: Sequence[_FNOCountDoc],
    *,
    train_docs: int,
    seed: int,
) -> Tuple[_FNOCountDoc, ...]:
    if int(train_docs) > len(docs):
        raise ValueError(f"requested {train_docs} train docs but only {len(docs)} available")
    indices = list(range(len(docs)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    chosen = indices[: int(train_docs)]
    return tuple(docs[idx] for idx in chosen)


def _validate_uniform_leaf_shape(docs: Sequence[_FNOCountDoc]) -> tuple[int, int]:
    if not docs:
        raise ValueError("docs must be non-empty")
    n_leaves = int(len(docs[0].leaf_counts))
    n_regimes = 1 + max(
        max(int(value) for value in docs[0].leaf_first_regimes),
        max(int(value) for value in docs[0].leaf_last_regimes),
    )
    for doc in docs:
        if int(len(doc.leaf_counts)) != n_leaves:
            raise ValueError("all docs must share the same balanced leaf count")
        n_regimes = max(
            n_regimes,
            1 + max(int(value) for value in doc.leaf_first_regimes),
            1 + max(int(value) for value in doc.leaf_last_regimes),
        )
    return n_leaves, n_regimes


def _root_support_max(docs: Sequence[_FNOCountDoc]) -> int:
    return int(
        max(
            1,
            max(int(round(float(doc.root_count))) for doc in docs),
        )
    )


def _leaf_summary_batch(
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: float,
    n_regimes: int,
    device: "torch.device",
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    rows: List[torch.Tensor] = []
    for doc in docs:
        per_doc: List[torch.Tensor] = []
        for count, first, last in zip(
            doc.leaf_counts,
            doc.leaf_first_regimes,
            doc.leaf_last_regimes,
        ):
            count_norm = torch.tensor(
                [float(count) / float(target_scale)],
                device=device,
                dtype=torch.float32,
            )
            first_oh = F.one_hot(
                torch.tensor(int(first), device=device),
                num_classes=int(n_regimes),
            ).to(dtype=torch.float32)
            last_oh = F.one_hot(
                torch.tensor(int(last), device=device),
                num_classes=int(n_regimes),
            ).to(dtype=torch.float32)
            per_doc.append(torch.cat([count_norm, first_oh, last_oh], dim=-1))
        rows.append(torch.stack(per_doc, dim=0))
    return torch.stack(rows, dim=0)


def _merge_parameter_tensors(model: FNOCountSketch) -> tuple["torch.nn.Parameter", ...]:
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "summary_state_merger" in name or "join_bit_head" in name:
            params.append(param)
    return tuple(params)


def _grad_norm(grads: Sequence["torch.Tensor | None"]) -> float:
    import torch

    pieces = []
    for grad in grads:
        if grad is None:
            continue
        pieces.append(torch.sum(grad.detach() ** 2))
    if not pieces:
        return 0.0
    return float(torch.sqrt(torch.stack(pieces).sum()).item())


def _aggregate_level_means(
    *,
    level_means: Sequence["torch.Tensor"],
    level_counts: Sequence[int],
    weighting_mode: str,
) -> "torch.Tensor":
    import torch

    normalized = str(weighting_mode or "flat_mean").strip().lower() or "flat_mean"
    if normalized not in VALID_MARKOV_MERGE_WEIGHTING_MODES:
        raise ValueError(
            "merge weighting must be one of "
            f"{VALID_MARKOV_MERGE_WEIGHTING_MODES}; got {weighting_mode!r}"
        )
    if not level_means:
        raise ValueError("level_means must be non-empty")
    if normalized == "depth_balanced":
        return torch.stack(list(level_means)).mean()
    numer = torch.zeros_like(level_means[0])
    denom = 0.0
    for level_mean, count in zip(level_means, level_counts):
        numer = numer + level_mean * float(count)
        denom += float(count)
    return numer / max(denom, 1.0)


def _build_model(
    *,
    n_regimes: int,
    vocab_size: int,
    leaf_tokens: int,
    target_scale: int,
    spec: MergeStudyRunSpec,
    device: "torch.device",
) -> FNOCountSketch:
    import torch

    model = FNOCountSketch(
        state_dim=int(spec.state_dim),
        hidden_dim=int(spec.hidden_dim),
        n_regimes=int(n_regimes),
        vocab_size=int(vocab_size),
        fno_width=32,
        fno_n_modes=4,
        fno_n_layers=2,
        leaf_tokens=int(leaf_tokens),
        target_scale=float(target_scale),
        tree_model_version="v2",
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_feature_dim=int(spec.theorem_feature_dim),
        theorem_feature_hidden_dim=int(spec.theorem_feature_hidden_dim),
        theorem_score_dim=1,
        theorem_fiber_dim=max(1, int(spec.theorem_feature_dim) - 1),
        theorem_aux_dim=0,
        score_merge_mode="gated_affine",
        join_bit_weight=1.0,
        c2_mode="reconstruction",
        root_supervision_kind="mse",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="shared_feature",
        theorem_count_head_mode=str(spec.count_head_mode),
        theorem_count_ordinal_weight=1.0,
        theorem_count_scalar_aux_weight=0.25,
        theorem_feature_adapter="markov_count_sketch",
        theorem_count_dim=0,
        theorem_first_dim=0,
        theorem_last_dim=0,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def _forward_exact_leaf_batch(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    merge_objective: str,
    merge_weighting: str,
    root_loss_weight: float,
    device: "torch.device",
) -> Dict[str, Any]:
    import torch

    leaf_summary = _leaf_summary_batch(
        docs,
        target_scale=float(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    batch_size, n_leaves, summary_dim = leaf_summary.shape
    leaf_states = model.encode_summary(
        leaf_summary.reshape(int(batch_size) * int(n_leaves), int(summary_dim))
    ).reshape(int(batch_size), int(n_leaves), -1)

    level_means: List[torch.Tensor] = []
    level_count_means: List[torch.Tensor] = []
    level_first_means: List[torch.Tensor] = []
    level_last_means: List[torch.Tensor] = []
    level_counts: List[int] = []
    current = leaf_states
    exact_targets = [_doc_exact_targets_by_level(doc) for doc in docs]
    for level_idx in range(len(exact_targets[0]["merge_levels"])):
        n_pairs = int(current.shape[1] // 2)
        if n_pairs <= 0:
            break
        left = current[:, 0 : 2 * n_pairs : 2, :]
        right = current[:, 1 : 2 * n_pairs : 2, :]
        merged = model._merge_state_pairs(
            left.reshape(int(batch_size) * int(n_pairs), -1),
            right.reshape(int(batch_size) * int(n_pairs), -1),
        ).reshape(int(batch_size), int(n_pairs), -1)
        level_targets = [targets["merge_levels"][level_idx] for targets in exact_targets]
        truth_counts = torch.tensor(
            [float(item[0]) for per_doc in level_targets for item in per_doc],
            device=device,
            dtype=torch.float32,
        )
        truth_first = torch.tensor(
            [int(item[1]) for per_doc in level_targets for item in per_doc],
            device=device,
            dtype=torch.long,
        )
        truth_last = torch.tensor(
            [int(item[2]) for per_doc in level_targets for item in per_doc],
            device=device,
            dtype=torch.long,
        )
        terms = _markov_merge_objective_terms_batched(
            model,
            left.reshape(int(batch_size) * int(n_pairs), -1),
            right.reshape(int(batch_size) * int(n_pairs), -1),
            merged.reshape(int(batch_size) * int(n_pairs), -1),
            truth_counts=truth_counts,
            truth_first=truth_first,
            truth_last=truth_last,
            objective_mode=str(merge_objective),
        )
        level_means.append(terms["mean_total_loss"])
        level_count_means.append(terms["mean_count_loss"])
        level_first_means.append(terms["mean_first_loss"])
        level_last_means.append(terms["mean_last_loss"])
        level_counts.append(int(batch_size) * int(n_pairs))
        if int(current.shape[1]) % 2 == 1:
            current = torch.cat([merged, current[:, -1:, :]], dim=1)
        else:
            current = merged

    if not level_means:
        raise RuntimeError("exact-leaf batch produced no merge levels")

    local_loss = _aggregate_level_means(
        level_means=level_means,
        level_counts=level_counts,
        weighting_mode=str(merge_weighting),
    )
    local_count_loss = _aggregate_level_means(
        level_means=level_count_means,
        level_counts=level_counts,
        weighting_mode=str(merge_weighting),
    )
    local_first_loss = _aggregate_level_means(
        level_means=level_first_means,
        level_counts=level_counts,
        weighting_mode=str(merge_weighting),
    )
    local_last_loss = _aggregate_level_means(
        level_means=level_last_means,
        level_counts=level_counts,
        weighting_mode=str(merge_weighting),
    )

    root_truth = torch.tensor(
        [float(doc.root_count) for doc in docs],
        device=device,
        dtype=torch.float32,
    )
    root_states = current[:, 0, :]
    root_terms = _summary_spec_supervision_terms_batched(
        model,
        root_states,
        truth_counts=root_truth,
        supervise_count=True,
        supervise_endpoints=False,
    )
    root_loss = root_terms["total_loss"].mean()
    total_loss = local_loss + float(root_loss_weight) * root_loss
    return {
        "total_loss": total_loss,
        "local_loss": local_loss,
        "local_count_loss": local_count_loss,
        "local_first_loss": local_first_loss,
        "local_last_loss": local_last_loss,
        "root_loss": root_loss,
    }


def _evaluate_exact_leaf_merger(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    target_scale: int,
    n_regimes: int,
    device: "torch.device",
) -> Dict[str, Any]:
    import torch

    model.eval()
    n_leaves, _ = _validate_uniform_leaf_shape(docs)
    layout = model._balanced_tree_layout(int(n_leaves))
    merge_levels = _balanced_merge_levels(int(n_leaves))

    root_errors: List[float] = []
    count_only_root_errors: List[float] = []
    endpoint_only_root_errors: List[float] = []
    merge_exact_matches = 0
    merge_total = 0
    merge_first_hits = 0
    merge_last_hits = 0
    merge_join_hits = 0
    depth_hits: Dict[str, int] = {}
    depth_totals: Dict[str, int] = {}

    with torch.no_grad():
        for doc in docs:
            exact_targets = _doc_exact_targets_by_level(doc)
            leaf_summary = _leaf_summary_batch(
                (doc,),
                target_scale=float(target_scale),
                n_regimes=int(n_regimes),
                device=device,
            )
            current = model.encode_summary(leaf_summary.reshape(int(n_leaves), -1))
            current = current.reshape(1, int(n_leaves), -1)

            pred_count_only_path = model.predict_count_from_state(current[0]).detach().cpu().tolist()
            endpoint_only_path = [float(value) for value in doc.leaf_counts]
            oracle_first_path = [int(value) for value in doc.leaf_first_regimes]
            oracle_last_path = [int(value) for value in doc.leaf_last_regimes]

            for level_idx, level_merge_indices in enumerate(merge_levels):
                n_pairs = int(current.shape[1] // 2)
                left = current[:, 0 : 2 * n_pairs : 2, :]
                right = current[:, 1 : 2 * n_pairs : 2, :]
                merged = model._merge_state_pairs(
                    left.reshape(int(n_pairs), -1),
                    right.reshape(int(n_pairs), -1),
                ).reshape(1, int(n_pairs), -1)

                pred_counts = model.predict_count_from_state(merged[0]).detach().cpu().tolist()
                _h, first_logits, last_logits = model._split_state(merged[0])
                pred_first = torch.argmax(first_logits, dim=-1).detach().cpu().tolist()
                pred_last = torch.argmax(last_logits, dim=-1).detach().cpu().tolist()
                join_probs = model.predict_join_prob_from_states(
                    left.reshape(int(n_pairs), -1),
                    right.reshape(int(n_pairs), -1),
                ).detach().cpu().tolist()
                level_targets = list(exact_targets["merge_levels"][level_idx])

                next_count_only: List[float] = []
                next_endpoint_only: List[float] = []
                next_oracle_first: List[int] = []
                next_oracle_last: List[int] = []

                carry_count_only: float | None = None
                carry_endpoint_only: float | None = None
                carry_first: int | None = None
                carry_last: int | None = None
                if int(current.shape[1]) % 2 == 1:
                    carry_idx = int(current.shape[1]) - 1
                    carry_state = current[0, carry_idx : carry_idx + 1]
                    carry_count_only = float(model.predict_count_from_state(carry_state).item())
                    carry_endpoint_only = float(endpoint_only_path[-1])
                    carry_first = int(oracle_first_path[-1])
                    carry_last = int(oracle_last_path[-1])

                for local_idx, target in enumerate(level_targets):
                    truth_count, truth_first, truth_last = target
                    pred_count = float(pred_counts[local_idx])
                    pred_first_i = int(pred_first[local_idx])
                    pred_last_i = int(pred_last[local_idx])
                    pred_join = 1 if float(join_probs[local_idx]) >= 0.5 else 0
                    truth_join = (
                        0
                        if int(oracle_last_path[2 * local_idx])
                        == int(oracle_first_path[2 * local_idx + 1])
                        else 1
                    )
                    merge_total += 1
                    merge_exact_matches += int(
                        int(round(pred_count)) == int(round(float(truth_count)))
                        and pred_first_i == int(truth_first)
                        and pred_last_i == int(truth_last)
                    )
                    merge_first_hits += int(pred_first_i == int(truth_first))
                    merge_last_hits += int(pred_last_i == int(truth_last))
                    merge_join_hits += int(pred_join == int(truth_join))
                    depth = str(
                        int(
                            layout["depth_by_global_idx"][
                                int(n_leaves) + int(level_merge_indices[local_idx])
                            ]
                        )
                    )
                    depth_hits[depth] = depth_hits.get(depth, 0) + int(
                        int(round(pred_count)) == int(round(float(truth_count)))
                        and pred_first_i == int(truth_first)
                        and pred_last_i == int(truth_last)
                    )
                    depth_totals[depth] = depth_totals.get(depth, 0) + 1

                    next_count_only.append(
                        float(pred_count_only_path[2 * local_idx])
                        + float(pred_count_only_path[2 * local_idx + 1])
                        + float(truth_join)
                    )
                    next_endpoint_only.append(
                        float(endpoint_only_path[2 * local_idx])
                        + float(endpoint_only_path[2 * local_idx + 1])
                        + float(pred_join)
                    )
                    next_oracle_first.append(int(truth_first))
                    next_oracle_last.append(int(truth_last))

                if carry_count_only is not None:
                    next_count_only.append(float(carry_count_only))
                    next_endpoint_only.append(float(carry_endpoint_only))
                    next_oracle_first.append(int(carry_first))
                    next_oracle_last.append(int(carry_last))

                pred_count_only_path = next_count_only
                endpoint_only_path = next_endpoint_only
                oracle_first_path = next_oracle_first
                oracle_last_path = next_oracle_last
                current = (
                    torch.cat([merged, current[:, -1:, :]], dim=1)
                    if int(current.shape[1]) % 2 == 1
                    else merged
                )

            root_pred = float(model.predict_count_from_state(current[0, 0:1]).item())
            root_truth = float(doc.root_count)
            root_errors.append(abs(root_pred - root_truth))
            count_only_root_errors.append(abs(float(pred_count_only_path[0]) - root_truth))
            endpoint_only_root_errors.append(abs(float(endpoint_only_path[0]) - root_truth))

    return {
        "step1_root_mae": float(np.mean(root_errors)) if root_errors else 0.0,
        "step1_merge_exact_summary_match_rate": (
            float(merge_exact_matches) / float(max(merge_total, 1))
        ),
        "step1_count_only_root_mae": (
            float(np.mean(count_only_root_errors)) if count_only_root_errors else 0.0
        ),
        "step1_endpoint_only_root_mae": (
            float(np.mean(endpoint_only_root_errors)) if endpoint_only_root_errors else 0.0
        ),
        "merge_first_accuracy": float(merge_first_hits) / float(max(merge_total, 1)),
        "merge_last_accuracy": float(merge_last_hits) / float(max(merge_total, 1)),
        "merge_join_accuracy": float(merge_join_hits) / float(max(merge_total, 1)),
        "per_depth_merge_exact_summary_match_rate": {
            str(depth): float(depth_hits.get(depth, 0)) / float(max(depth_totals.get(depth, 0), 1))
            for depth in sorted(depth_totals.keys(), key=lambda value: int(value))
        },
    }


def _train_exact_leaf_merger(
    spec: MergeStudyRunSpec,
    *,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    test_docs: Sequence[_FNOCountDoc],
    device: "torch.device",
    output_dir: Path,
) -> MergeStudyRunResult:
    import torch

    _set_global_seed(int(spec.seed))
    all_docs = tuple(train_docs) + tuple(val_docs) + tuple(test_docs)
    n_leaves, n_regimes = _validate_uniform_leaf_shape(all_docs)
    vocab_size = (
        max(
            int(token)
            for doc in all_docs
            for leaf in doc.leaf_token_ids
            for token in leaf
        )
        + 1
    )
    leaf_tokens = int(len(all_docs[0].leaf_token_ids[0]))
    target_scale = _root_support_max(all_docs)
    model = _build_model(
        n_regimes=int(n_regimes),
        vocab_size=int(vocab_size),
        leaf_tokens=int(leaf_tokens),
        target_scale=int(target_scale),
        spec=spec,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.lr),
        weight_decay=float(spec.weight_decay),
    )
    merger_params = _merge_parameter_tensors(model)

    best_state = None
    best_epoch = 0
    best_val = math.inf
    grad_norm_root_log: List[float] = []
    grad_norm_local_log: List[float] = []
    train_local_last = 0.0
    train_root_last = 0.0
    train_total_last = 0.0

    batches = [
        tuple(train_docs[idx : idx + int(spec.batch_size)])
        for idx in range(0, len(train_docs), int(spec.batch_size))
    ]
    if not batches:
        raise ValueError("training set is empty")

    for epoch in range(1, int(spec.n_epochs) + 1):
        model.train()
        epoch_local: List[float] = []
        epoch_root: List[float] = []
        epoch_total: List[float] = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            losses = _forward_exact_leaf_batch(
                model,
                batch,
                target_scale=int(target_scale),
                n_regimes=int(n_regimes),
                merge_objective=str(spec.merge_objective),
                merge_weighting=str(spec.merge_weighting),
                root_loss_weight=float(spec.root_loss_weight),
                device=device,
            )
            local_obj = losses["local_loss"]
            root_obj = float(spec.root_loss_weight) * losses["root_loss"]
            root_norm = 0.0
            local_norm = 0.0
            if merger_params:
                if float(spec.root_loss_weight) > 0.0:
                    root_grads = torch.autograd.grad(
                        root_obj,
                        merger_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    root_norm = _grad_norm(root_grads)
                local_grads = torch.autograd.grad(
                    local_obj,
                    merger_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                local_norm = _grad_norm(local_grads)
            grad_norm_root_log.append(float(root_norm))
            grad_norm_local_log.append(float(local_norm))
            total = losses["total_loss"]
            total.backward()
            optimizer.step()
            epoch_local.append(float(losses["local_loss"].detach().cpu().item()))
            epoch_root.append(float(losses["root_loss"].detach().cpu().item()))
            epoch_total.append(float(total.detach().cpu().item()))

        train_local_last = float(np.mean(epoch_local)) if epoch_local else 0.0
        train_root_last = float(np.mean(epoch_root)) if epoch_root else 0.0
        train_total_last = float(np.mean(epoch_total)) if epoch_total else 0.0

        val_metrics = _evaluate_exact_leaf_merger(
            model,
            val_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
        )
        if float(val_metrics["step1_root_mae"]) < float(best_val):
            best_val = float(val_metrics["step1_root_mae"])
            best_epoch = int(epoch)
            best_state = {
                key: (
                    value.detach().cpu().clone()
                    if hasattr(value, "detach")
                    else value
                )
                for key, value in model.state_dict().items()
                if str(key) != "_metadata"
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = _evaluate_exact_leaf_merger(
        model,
        test_docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    ratio_values = []
    for root_norm, local_norm in zip(grad_norm_root_log, grad_norm_local_log):
        if local_norm > 0.0:
            ratio_values.append(float(root_norm) / float(local_norm))
        elif root_norm > 0.0:
            ratio_values.append(float("inf"))
        else:
            ratio_values.append(0.0)

    result = MergeStudyRunResult(
        label=spec.label,
        benchmark=str(spec.benchmark),
        train_docs=int(spec.train_docs),
        seed=int(spec.seed),
        merge_objective=str(spec.merge_objective),
        root_loss_weight=float(spec.root_loss_weight),
        merge_weighting=str(spec.merge_weighting),
        count_head_mode=str(spec.count_head_mode),
        state_dim=int(spec.state_dim),
        hidden_dim=int(spec.hidden_dim),
        theorem_feature_dim=int(spec.theorem_feature_dim),
        theorem_feature_hidden_dim=int(spec.theorem_feature_hidden_dim),
        n_epochs=int(spec.n_epochs),
        best_epoch=int(best_epoch),
        best_val_step1_root_mae=float(best_val),
        train_local_loss_last=float(train_local_last),
        train_root_loss_last=float(train_root_last),
        train_total_loss_last=float(train_total_last),
        merger_grad_norm_root=float(np.mean(grad_norm_root_log)) if grad_norm_root_log else 0.0,
        merger_grad_norm_local=float(np.mean(grad_norm_local_log)) if grad_norm_local_log else 0.0,
        merger_grad_ratio_root_to_local=(
            float(np.mean(ratio_values))
            if ratio_values and not any(math.isinf(value) for value in ratio_values)
            else (float("inf") if ratio_values else 0.0)
        ),
        step1_root_mae=float(test_metrics["step1_root_mae"]),
        step1_merge_exact_summary_match_rate=float(
            test_metrics["step1_merge_exact_summary_match_rate"]
        ),
        step1_count_only_root_mae=float(test_metrics["step1_count_only_root_mae"]),
        step1_endpoint_only_root_mae=float(test_metrics["step1_endpoint_only_root_mae"]),
        merge_first_accuracy=float(test_metrics["merge_first_accuracy"]),
        merge_last_accuracy=float(test_metrics["merge_last_accuracy"]),
        merge_join_accuracy=float(test_metrics["merge_join_accuracy"]),
        per_depth_merge_exact_summary_match_rate={
            str(key): float(value)
            for key, value in dict(
                test_metrics["per_depth_merge_exact_summary_match_rate"]
            ).items()
        },
        n_train_used=int(len(train_docs)),
        n_val_used=int(len(val_docs)),
        n_test_used=int(len(test_docs)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _default_specs(args: argparse.Namespace) -> List[MergeStudyRunSpec]:
    specs: List[MergeStudyRunSpec] = []
    for train_docs in [int(value) for value in args.train_doc_counts]:
        n_epochs = int(args.n_epochs) if int(args.n_epochs) > 0 else _resolved_epochs(int(train_docs))
        for seed in [int(value) for value in args.seeds]:
            specs.extend(
                [
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="strict_c3",
                        root_loss_weight=0.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="strict_c3",
                        root_loss_weight=1.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="strict_c3",
                        root_loss_weight=0.0,
                        merge_weighting="depth_balanced",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="strict_c3",
                        root_loss_weight=1.0,
                        merge_weighting="depth_balanced",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_count",
                        root_loss_weight=0.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_count",
                        root_loss_weight=1.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_full_sketch",
                        root_loss_weight=0.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_full_sketch",
                        root_loss_weight=1.0,
                        merge_weighting="flat_mean",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_full_sketch",
                        root_loss_weight=0.0,
                        merge_weighting="depth_balanced",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                    MergeStudyRunSpec(
                        benchmark=str(args.benchmark),
                        train_docs=int(train_docs),
                        seed=int(seed),
                        merge_objective="teacher_parent_full_sketch",
                        root_loss_weight=1.0,
                        merge_weighting="depth_balanced",
                        count_head_mode=str(args.count_head_mode),
                        state_dim=int(args.state_dim),
                        hidden_dim=int(args.hidden_dim),
                        theorem_feature_dim=int(args.theorem_feature_dim),
                        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
                        n_epochs=int(n_epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                    ),
                ]
            )
    return specs


def _summarize_results(results: Sequence[MergeStudyRunResult]) -> Dict[str, Any]:
    grouped: Dict[str, List[MergeStudyRunResult]] = {}
    for result in results:
        group_key = (
            f"{result.merge_objective}"
            f"__root{result.root_loss_weight:g}"
            f"__{result.merge_weighting}"
            f"__head_{result.count_head_mode}"
            f"__n{result.train_docs}"
        )
        grouped.setdefault(group_key, []).append(result)
    aggregate_rows: List[Dict[str, Any]] = []
    for key, runs in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "label": key,
                "n_runs": int(len(runs)),
                "step1_root_mae_mean": float(mean(run.step1_root_mae for run in runs)),
                "step1_root_mae_std": float(
                    pstdev(run.step1_root_mae for run in runs) if len(runs) > 1 else 0.0
                ),
                "step1_merge_exact_summary_match_rate_mean": float(
                    mean(run.step1_merge_exact_summary_match_rate for run in runs)
                ),
                "step1_count_only_root_mae_mean": float(
                    mean(run.step1_count_only_root_mae for run in runs)
                ),
                "step1_endpoint_only_root_mae_mean": float(
                    mean(run.step1_endpoint_only_root_mae for run in runs)
                ),
                "merge_first_accuracy_mean": float(mean(run.merge_first_accuracy for run in runs)),
                "merge_last_accuracy_mean": float(mean(run.merge_last_accuracy for run in runs)),
                "merge_join_accuracy_mean": float(mean(run.merge_join_accuracy for run in runs)),
                "merger_grad_norm_root_mean": float(
                    mean(run.merger_grad_norm_root for run in runs)
                ),
                "merger_grad_norm_local_mean": float(
                    mean(run.merger_grad_norm_local for run in runs)
                ),
                "merger_grad_ratio_root_to_local_mean": float(
                    mean(run.merger_grad_ratio_root_to_local for run in runs)
                ),
            }
        )
    aggregate_rows.sort(key=lambda row: float(row["step1_root_mae_mean"]))
    return {
        "n_runs": int(len(results)),
        "results": [asdict(result) for result in results],
        "aggregate": aggregate_rows,
    }


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Markov Merge-Signal Feasibility Study",
        "",
        f"Runs: {int(summary.get('n_runs', 0))}",
        "",
        "## Aggregate",
        "",
        "| label | runs | step1 root mae | merge exact | count-only root mae | endpoint-only root mae | first acc | last acc | join acc | grad root | grad local | ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(summary.get("aggregate") or []):
        lines.append(
            "| {label} | {n_runs} | {step1_root_mae_mean:.4f} | "
            "{step1_merge_exact_summary_match_rate_mean:.4f} | "
            "{step1_count_only_root_mae_mean:.4f} | "
            "{step1_endpoint_only_root_mae_mean:.4f} | "
            "{merge_first_accuracy_mean:.4f} | {merge_last_accuracy_mean:.4f} | "
            "{merge_join_accuracy_mean:.4f} | {merger_grad_norm_root_mean:.4f} | "
            "{merger_grad_norm_local_mean:.4f} | {merger_grad_ratio_root_to_local_mean:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _run_study(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    test_lean_worked_example()
    dataset_payload = _load_recoverable_v4_split_docs(
        benchmark=str(args.benchmark),
        train_docs=max(int(value) for value in args.train_doc_counts),
    )
    full_train_docs = tuple(dataset_payload["train_all"])
    val_docs = tuple(dataset_payload["val"])
    test_docs = tuple(dataset_payload["test"])
    exact_baseline = step0_exact(test_docs)
    if float(exact_baseline["root_mae"]) != 0.0:
        raise RuntimeError(
            f"exact Step 0 baseline must be zero on prepared FNO docs; got {exact_baseline['root_mae']}"
        )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_root = output_root / "runs"
    device = (
        torch.device("cuda")
        if bool(args.use_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    results: List[MergeStudyRunResult] = []
    specs = _default_specs(args)
    if int(args.max_runs) > 0:
        specs = specs[: int(args.max_runs)]
    for spec in specs:
        run_train_docs = _select_train_prefix(
            full_train_docs,
            train_docs=int(spec.train_docs),
            seed=int(spec.seed),
        )
        run_dir = summary_root / spec.label
        result = _train_exact_leaf_merger(
            spec,
            train_docs=run_train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            device=device,
            output_dir=run_dir,
        )
        results.append(result)

    summary = _summarize_results(results)
    summary["benchmark"] = str(args.benchmark)
    summary["exact_step0"] = exact_baseline
    summary["prepared_data_root"] = str(dataset_payload["prepared"]["prepared_data_root"])
    summary["device"] = str(device)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=str, default="recoverable_v4")
    parser.add_argument(
        "--train-doc-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_DOC_COUNTS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--count-head-mode", type=str, default="scalar_mse")
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--theorem-feature-dim", type=int, default=128)
    parser.add_argument("--theorem-feature-hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO / "outputs" / f"markov_merge_signal_feasibility_{_timestamp()}"),
    )
    args = parser.parse_args()

    summary = _run_study(args)
    output_root = Path(args.output_root).resolve()
    (output_root / "merge_signal_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "merge_signal_feasibility_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"][:5], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
