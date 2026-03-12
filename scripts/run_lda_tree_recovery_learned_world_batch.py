#!/usr/bin/env python3
"""Run a fixed-world batched learned LDA tree-recovery sweep for one (dtc, seed) bundle."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core import lda_tree_recovery as _base  # noqa: E402
from src.ctreepo.sim.core import lda_tree_recovery_learned as _learned  # noqa: E402
from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import build_leaf_spans  # noqa: E402


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _rows_from_summary(summary: _learned.LDATreeRecoveryLearnedSummary) -> List[dict]:
    cfg = dict(summary.config)
    learning = dict(summary.learning)
    exact_recovery = dict(summary.exact_reference.get("exact_recovery", {}))
    world_stats = dict(summary.exact_reference.get("world_stats", {}))

    rows: List[dict] = []
    methods = summary.methods if isinstance(summary.methods, dict) else {}
    for method, metrics in methods.items():
        if not isinstance(metrics, dict):
            continue
        method_diag = dict(learning.get(str(method), {})) if isinstance(learning.get(str(method), {}), dict) else {}
        row = {
            "method": str(method),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{f"world_{k}": v for k, v in world_stats.items()},
            **{f"exact_{k}": v for k, v in exact_recovery.items()},
            **{f"diag_{k}": v for k, v in method_diag.items() if not isinstance(v, dict)},
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _output_base(
    *,
    output_root: Path,
    doc_topic_concentration: float,
    quadratic_utility_weight: float,
    leaf_tokens: int,
    train_docs: int,
    state_dim: int,
    seed: int,
) -> Path:
    return (
        output_root
        / f"dtc_{doc_topic_concentration:g}"
        / f"qweight_{quadratic_utility_weight:g}"
        / f"leaf_{leaf_tokens}"
        / f"train_{train_docs}"
        / f"state_{state_dim}"
        / f"seed_{seed}"
    )


def _cache_load_or_compute(cache_path: Path, build_fn):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                return pickle.load(handle)
        value = build_fn()
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        return value


def _prepare_common_docs(
    docs: Sequence[_base.LDATreeRecoveryDoc],
    *,
    vocab_size: int,
    topics_phi: Sequence[np.ndarray],
    inference_prior_mass: float,
    inference_max_iter: int,
    inference_tol: float,
) -> Tuple[_learned._PreparedDoc, ...]:
    out: List[_learned._PreparedDoc] = []
    for doc in docs:
        counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(vocab_size))
        pi_true = np.asarray(doc.topic_weights, dtype=np.float64)
        pi_full = _base._infer_topic_mixture_from_counts(
            counts_full,
            topics_phi=topics_phi,
            prior_mass=float(inference_prior_mass),
            max_iter=int(inference_max_iter),
            tol=float(inference_tol),
        )
        loglik_full = _base._doc_log_likelihood(counts_full, pi=pi_full, topics_phi=topics_phi)
        out.append(
            _learned._PreparedDoc(
                counts_full=np.asarray(counts_full, dtype=np.float64).copy(),
                pi_true=np.asarray(pi_true, dtype=np.float64).copy(),
                pi_full=np.asarray(pi_full, dtype=np.float64).copy(),
                utility_true=float("nan"),
                utility_full=float("nan"),
                loglik_full=float(loglik_full),
                leaf_counts=tuple(),
                balanced_node_counts=tuple(),
            )
        )
    return tuple(out)


def _attach_leaf_counts(
    common_docs: Sequence[_learned._PreparedDoc],
    raw_docs: Sequence[_base.LDATreeRecoveryDoc],
    *,
    leaf_tokens: int,
    vocab_size: int,
) -> Tuple[_learned._PreparedDoc, ...]:
    out: List[_learned._PreparedDoc] = []
    for common, raw in zip(common_docs, raw_docs):
        leaf_spans = build_leaf_spans(len(raw.tokens), leaf_tokens=int(leaf_tokens))
        leaf_counts = tuple(
            _base._counts_from_tokens(raw.tokens[int(start) : int(end)], vocab_size=int(vocab_size))
            for start, end in leaf_spans
        )
        out.append(
            _learned._PreparedDoc(
                counts_full=np.asarray(common.counts_full, dtype=np.float64).copy(),
                pi_true=np.asarray(common.pi_true, dtype=np.float64).copy(),
                pi_full=np.asarray(common.pi_full, dtype=np.float64).copy(),
                utility_true=float("nan"),
                utility_full=float("nan"),
                loglik_full=float(common.loglik_full),
                leaf_counts=tuple(np.asarray(x, dtype=np.float64).copy() for x in leaf_counts),
                balanced_node_counts=_learned._balanced_node_counts(leaf_counts),
            )
        )
    return tuple(out)


def _evaluate_svd_reconstruction(
    fit: _learned._SVDCountSketch,
    node_counts: np.ndarray,
    *,
    vocab_size: int,
) -> Dict[str, float]:
    if node_counts.size == 0:
        return {"count_l1_mean": float("nan"), "count_rmse_mean": float("nan")}
    l1s: List[float] = []
    rmses: List[float] = []
    for row in np.asarray(node_counts, dtype=np.float64):
        pred = fit.decode(fit.encode(row))
        clipped = _learned._project_counts_to_histogram(pred, total_tokens=int(np.sum(row)), vocab_size=int(vocab_size))
        diff = clipped - row
        l1s.append(float(np.sum(np.abs(diff))))
        rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    return {
        "count_l1_mean": _base._safe_stat(l1s, kind="mean"),
        "count_rmse_mean": _base._safe_stat(rmses, kind="mean"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a batched fixed-world learned LDA tree-recovery sweep.")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--world-cache-dir", type=str, default="")
    p.add_argument("--prepared-cache-dir", type=str, default="")

    p.add_argument("--doc-topic-concentration", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--leaf-tokens-grid", type=str, required=True)
    p.add_argument("--train-docs-grid", type=str, required=True)
    p.add_argument("--state-dims", type=str, required=True)
    p.add_argument(
        "--quadratic-weight-grid",
        "--lambda-grid",
        dest="quadratic_weight_grid",
        type=str,
        required=True,
    )
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--topic-concentration", type=float, default=0.2)
    p.add_argument("--emission-mode", type=str, default="anchored")
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--anchor-multiplier", type=float, default=25.0)
    p.add_argument("--relevant-topics", type=int, default=2)
    p.add_argument("--theta-scale", type=float, default=1.0)
    p.add_argument("--zero-diagonal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-train-docs-capacity", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=512)
    p.add_argument("--inference-prior-mass", type=float, default=0.25)
    p.add_argument("--inference-max-iter", type=int, default=200)
    p.add_argument("--inference-tol", type=float, default=1e-9)
    p.add_argument("--full-hidden-dim", type=int, default=256)
    p.add_argument("--full-n-layers", type=int, default=3)
    p.add_argument("--supervise-all-balanced-nodes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    leaf_tokens_grid = _parse_ints(args.leaf_tokens_grid)
    train_docs_grid = _parse_ints(args.train_docs_grid)
    state_dims = _parse_ints(args.state_dims)
    lambda_grid = _parse_floats(args.quadratic_weight_grid)
    if not leaf_tokens_grid or not train_docs_grid or not state_dims or not lambda_grid:
        raise ValueError("all grids must be non-empty")

    pending: List[Tuple[int, int, int, float]] = []
    for leaf_tokens in leaf_tokens_grid:
        for train_docs in train_docs_grid:
            for state_dim in state_dims:
                for lam in lambda_grid:
                    base = _output_base(
                        output_root=output_root,
                        doc_topic_concentration=float(args.doc_topic_concentration),
                        quadratic_utility_weight=float(lam),
                        leaf_tokens=int(leaf_tokens),
                        train_docs=int(train_docs),
                        state_dim=int(state_dim),
                        seed=int(args.seed),
                    )
                    json_path = base.with_suffix(".json")
                    csv_path = base.with_suffix(".csv")
                    if bool(args.skip_existing) and json_path.exists() and csv_path.exists():
                        continue
                    pending.append((int(leaf_tokens), int(train_docs), int(state_dim), float(lam)))
    if not pending:
        print(json.dumps({"seed": int(args.seed), "doc_topic_concentration": float(args.doc_topic_concentration), "completed": 0, "skipped_all": True}, sort_keys=True))
        return 0

    max_train_docs = int(args.max_train_docs_capacity) if int(args.max_train_docs_capacity) > 0 else max(train_docs_grid)
    base_cfg = _learned.LDATreeRecoveryLearnedConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        relevant_topics=int(args.relevant_topics),
        theta_scale=float(args.theta_scale),
        zero_diagonal=bool(args.zero_diagonal),
        lambda_multiplier=0.0,
        leaf_tokens=min(leaf_tokens_grid),
        train_docs=int(max_train_docs),
        test_docs=int(args.test_docs),
        inference_prior_mass=float(args.inference_prior_mass),
        inference_max_iter=int(args.inference_max_iter),
        inference_tol=float(args.inference_tol),
        full_hidden_dim=int(args.full_hidden_dim),
        full_n_layers=int(args.full_n_layers),
        state_dim=max(state_dims),
        supervise_all_balanced_nodes=bool(args.supervise_all_balanced_nodes),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
        seed=int(args.seed),
    )
    _learned._validate_config(base_cfg)
    _learned._set_global_seed(int(base_cfg.seed), torch_threads=int(base_cfg.torch_threads))
    device = _learned._resolve_device(base_cfg)

    world_cache_dir = Path(args.world_cache_dir) if str(args.world_cache_dir).strip() else (output_root / "world_cache")
    prepared_cache_dir = Path(args.prepared_cache_dir) if str(args.prepared_cache_dir).strip() else (output_root / "prepared_cache")
    world_cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_cache_dir.mkdir(parents=True, exist_ok=True)

    base_world_cfg = _learned._base_config(base_cfg)
    world_sig = _base.lda_tree_recovery_world_cache_signature(
        base_world_cfg,
        train_docs_capacity=int(max_train_docs),
        test_docs_capacity=int(args.test_docs),
    )
    world_key = hashlib.sha256(json.dumps(world_sig, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    world_cache_path = world_cache_dir / f"{world_key}.pkl"
    world = _cache_load_or_compute(
        world_cache_path,
        lambda: _base.sample_lda_tree_recovery_world(
            base_world_cfg,
            train_docs_capacity=int(max_train_docs),
            test_docs_capacity=int(args.test_docs),
        ),
    )

    topics_phi = tuple(np.asarray(t, dtype=np.float64) for t in world.topics_phi)
    theta_true = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)

    common_sig = {
        **world_sig,
        "inference_prior_mass": float(args.inference_prior_mass),
        "inference_max_iter": int(args.inference_max_iter),
        "inference_tol": float(args.inference_tol),
    }
    common_key = hashlib.sha256(json.dumps(common_sig, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    common_cache_path = prepared_cache_dir / f"{common_key}.pkl"

    common_train, common_test = _cache_load_or_compute(
        common_cache_path,
        lambda: (
            _prepare_common_docs(
                world.docs_train[: int(max_train_docs)],
                vocab_size=int(args.vocab_size),
                topics_phi=topics_phi,
                inference_prior_mass=float(args.inference_prior_mass),
                inference_max_iter=int(args.inference_max_iter),
                inference_tol=float(args.inference_tol),
            ),
            _prepare_common_docs(
                world.docs_test[: int(args.test_docs)],
                vocab_size=int(args.vocab_size),
                topics_phi=topics_phi,
                inference_prior_mass=float(args.inference_prior_mass),
                inference_max_iter=int(args.inference_max_iter),
                inference_tol=float(args.inference_tol),
            ),
        ),
    )

    exact_cache: Dict[Tuple[int, float], _base.LDATreeRecoverySummary] = {}
    full_doc_diag_cache: Dict[int, Dict[str, object]] = {}
    full_doc_metrics_cache: Dict[Tuple[int, float], _learned.LearnedMethodMetrics] = {}
    leaf_docs_cache: Dict[int, Tuple[Tuple[_learned._PreparedDoc, ...], Tuple[_learned._PreparedDoc, ...]]] = {}
    svd_system_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[_learned._PreparedDoc, ...], Tuple[_learned._PreparedDoc, ...]]] = {}

    pending_by_train: Dict[int, List[Tuple[int, int, float]]] = {}
    pending_by_leaf_lambda: Dict[Tuple[int, float], bool] = {}
    for leaf_tokens, train_docs, state_dim, lam in pending:
        pending_by_train.setdefault(int(train_docs), []).append((int(leaf_tokens), int(state_dim), float(lam)))
        pending_by_leaf_lambda[(int(leaf_tokens), float(lam))] = True

    completed = 0
    skipped_existing = 0

    for train_docs in sorted(pending_by_train):
        train_prefix = tuple(common_train[: int(train_docs)])
        train_fit_docs, val_fit_docs = _learned._split_train_val(train_prefix)
        _learned._set_global_seed(int(base_cfg.seed), torch_threads=int(base_cfg.torch_threads))
        full_doc_model, full_doc_diag = _learned._train_full_doc_operator(
            train_fit_docs,
            val_fit_docs,
            config=base_cfg,
            device=device,
        )
        full_doc_diag_cache[int(train_docs)] = dict(full_doc_diag)
        needed_lambdas = sorted({float(lam) for _leaf, _state, lam in pending_by_train[int(train_docs)]})
        for lam in needed_lambdas:
            cfg_lam = _learned.LDATreeRecoveryLearnedConfig(
                **{
                    **asdict(base_cfg),
                    "train_docs": int(train_docs),
                    "lambda_multiplier": float(lam),
                }
            )
            full_doc_metrics_cache[(int(train_docs), float(lam))] = _learned._eval_full_doc_operator(
                full_doc_model,
                common_test,
                config=cfg_lam,
                topics_phi=topics_phi,
                theta_true=theta_true,
                W_base=W_base,
                device=device,
            )

    for (leaf_tokens, lam) in sorted(pending_by_leaf_lambda):
        exact_cfg = _base.LDATreeRecoveryConfig(
            **{
                **asdict(_learned._base_config(base_cfg)),
                "leaf_tokens": int(leaf_tokens),
                "train_docs": int(max_train_docs),
                "lambda_multiplier": float(lam),
            }
        )
        exact_cache[(int(leaf_tokens), float(lam))] = _base.run_lda_tree_recovery_experiment_from_world(exact_cfg, world)

    for leaf_tokens in sorted({int(x[0]) for x in pending}):
        leaf_sig = {**common_sig, "leaf_tokens": int(leaf_tokens)}
        leaf_key = hashlib.sha256(json.dumps(leaf_sig, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        leaf_cache_path = prepared_cache_dir / f"{leaf_key}.pkl"
        leaf_docs_cache[int(leaf_tokens)] = _cache_load_or_compute(
            leaf_cache_path,
            lambda lt=int(leaf_tokens): (
                _attach_leaf_counts(
                    common_train,
                    world.docs_train[: int(max_train_docs)],
                    leaf_tokens=int(lt),
                    vocab_size=int(args.vocab_size),
                ),
                _attach_leaf_counts(
                    common_test,
                    world.docs_test[: int(args.test_docs)],
                    leaf_tokens=int(lt),
                    vocab_size=int(args.vocab_size),
                ),
            ),
        )

    pending_by_leaf_train: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for leaf_tokens, train_docs, state_dim, lam in pending:
        pending_by_leaf_train.setdefault((int(leaf_tokens), int(train_docs)), []).append((int(state_dim), float(lam)))

    for (leaf_tokens, train_docs), state_lams in sorted(pending_by_leaf_train.items()):
        train_leaf_docs, test_leaf_docs = leaf_docs_cache[int(leaf_tokens)]
        train_prefix = tuple(train_leaf_docs[: int(train_docs)])
        train_fit_docs, val_fit_docs = _learned._split_train_val(train_prefix)

        train_node_counts = np.stack(
            [
                np.asarray(node, dtype=np.float64)
                for doc in train_fit_docs
                for node in (
                    doc.balanced_node_counts
                    if bool(args.supervise_all_balanced_nodes)
                    else (doc.counts_full,)
                )
            ],
            axis=0,
        )
        val_node_counts = np.stack(
            [
                np.asarray(node, dtype=np.float64)
                for doc in val_fit_docs
                for node in (
                    doc.balanced_node_counts
                    if bool(args.supervise_all_balanced_nodes)
                    else (doc.counts_full,)
                )
            ],
            axis=0,
        ) if val_fit_docs else np.zeros((0, int(args.vocab_size)), dtype=np.float64)

        singular_values, right_vectors = _learned._right_singular_system(train_node_counts)
        svd_system_cache[(int(leaf_tokens), int(train_docs))] = (
            singular_values,
            right_vectors,
            train_node_counts,
            val_node_counts,
            train_fit_docs,
            val_fit_docs,
        )

        needed_states = sorted({int(x[0]) for x in state_lams})
        for state_dim in needed_states:
            fit, tree_diag = _learned._svd_fit_from_right_singular_system(
                singular_values,
                right_vectors,
                state_dim=int(state_dim),
                vocab_size=int(args.vocab_size),
            )
            tree_diag = {
                **tree_diag,
                "train_docs_fit": int(len(train_fit_docs)),
                "val_docs_fit": int(len(val_fit_docs)),
                "train_node_examples": int(train_node_counts.shape[0]),
                "val_node_examples": int(val_node_counts.shape[0]),
                "supervise_all_balanced_nodes": bool(args.supervise_all_balanced_nodes),
                "train_reconstruction": _evaluate_svd_reconstruction(fit, train_node_counts, vocab_size=int(args.vocab_size)),
                "val_reconstruction": _evaluate_svd_reconstruction(fit, val_node_counts, vocab_size=int(args.vocab_size)),
            }
            needed_lambdas = sorted({float(lam) for s, lam in state_lams if int(s) == int(state_dim)})
            for lam in needed_lambdas:
                base = _output_base(
                    output_root=output_root,
                    doc_topic_concentration=float(args.doc_topic_concentration),
                    quadratic_utility_weight=float(lam),
                    leaf_tokens=int(leaf_tokens),
                    train_docs=int(train_docs),
                    state_dim=int(state_dim),
                    seed=int(args.seed),
                )
                json_path = base.with_suffix(".json")
                csv_path = base.with_suffix(".csv")
                if bool(args.skip_existing) and json_path.exists() and csv_path.exists():
                    skipped_existing += 1
                    continue

                cfg_lam = _learned.LDATreeRecoveryLearnedConfig(
                    **{
                        **asdict(base_cfg),
                        "leaf_tokens": int(leaf_tokens),
                        "train_docs": int(train_docs),
                        "state_dim": int(state_dim),
                        "lambda_multiplier": float(lam),
                    }
                )
                tree_metrics = _learned._eval_tree_svd_sketch(
                    fit,
                    test_leaf_docs,
                    config=cfg_lam,
                    topics_phi=topics_phi,
                    theta_true=theta_true,
                    W_base=W_base,
                )
                exact_summary = exact_cache[(int(leaf_tokens), float(lam))]
                exact_reference = {
                    "exact_recovery": dict(exact_summary.exact_recovery),
                    "methods": dict(exact_summary.methods),
                    "world_stats": {
                        **dict(exact_summary.world_stats),
                        "train_docs_reserved": int(train_docs),
                    },
                }
                summary = _learned.LDATreeRecoveryLearnedSummary(
                    config={**asdict(cfg_lam), "quadratic_utility_weight": float(cfg_lam.lambda_multiplier)},
                    topic_meta=dict(world.topic_meta),
                    utility_truth=dict(exact_summary.utility_truth),
                    exact_reference=exact_reference,
                    learning={
                        "train_docs_requested": int(train_docs),
                        "test_docs_requested": int(args.test_docs),
                        "full_doc_operator": dict(full_doc_diag_cache[(int(train_docs))]),
                        "tree_svd_sketch": dict(tree_diag),
                    },
                    methods={
                        "full_doc_operator": asdict(full_doc_metrics_cache[(int(train_docs), float(lam))]),
                        "tree_svd_sketch": asdict(tree_metrics),
                    },
                )
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(summary.to_json(), encoding="utf-8")
                _write_csv(csv_path, _rows_from_summary(summary))
                completed += 1

    print(
        json.dumps(
            {
                "doc_topic_concentration": float(args.doc_topic_concentration),
                "seed": int(args.seed),
                "pending_initial": int(len(pending)),
                "completed": int(completed),
                "skipped_existing": int(skipped_existing),
                "world_cache_path": str(world_cache_path),
                "common_cache_path": str(common_cache_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
