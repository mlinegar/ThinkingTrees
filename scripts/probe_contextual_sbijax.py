#!/usr/bin/env python3
"""JAX/sbijax contextual-sufficiency probe for learned g states.

This is a separate runtime lane from the PyTorch ``CleanUnifiedNO`` probe.  It
uses ``sbijax`` as the package surface and trains a JAX state map from generic
finite-context response signatures.  Markov two-sided contexts are the first
adapter, not the shape of the method.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ctreepo.sim.core.contextual_sbijax import (  # noqa: E402
    CONTEXTUAL_SBI_INSTALL_MSG,
    ContextualSBIJAXConfig,
    HLLUnionContext,
    HLLUnionContextProblem,
    MarkovTwoSidedContext,
    MarkovTwoSidedContextProblem,
    build_contextual_query_dataset,
    contextual_sbijax_provenance,
    evaluate_contextual_sbijax,
    exact_root_witness_diagnostics,
    fit_contextual_sbijax,
    hll_register_sketch_targets_for_dataset,
    hybrid_summary_diagnostics_for_contextual_sbijax,
    load_markov_contextual_splits,
    load_markov_contextual_splits_from_bundle,
    markov_exact_sketch_oracle_diagnostics,
    markov_exact_sketch_targets_for_dataset,
    make_synthetic_markov_docs,
    palette_block_map,
    with_package_theta_target,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    test = _nested(payload, ("diagnostics", "test")) or {}
    oracle = _nested(payload, ("diagnostics", "markov_exact_sketch_oracle", "test")) or {}
    root_witness = _nested(payload, ("diagnostics", "exact_root_witness", "test")) or {}
    downstream = str(provenance.get("downstream_readout", ""))
    decoder_kind = provenance.get("decoder_kind")
    if decoder_kind is None:
        decoder_kind = "exact" if downstream == "deterministic_markov_exact_sketch" else "learned"
    return {
        "trainer": provenance.get("trainer", payload.get("trainer")),
        "method": provenance.get("method"),
        "package_theta": payload.get("package_theta"),
        "input_encoding": payload.get("input_encoding"),
        "law_architecture": provenance.get("law_architecture"),
        "c2_merge_target": provenance.get("c2_merge_target"),
        "learned_merge_active": bool(provenance.get("learned_merge_hidden_dim", 0)),
        "learned_decoder_active": str(decoder_kind) == "learned_mlp",
        "local_law_package_weight": provenance.get("local_law_package_weight"),
        "local_law_package_objective": provenance.get("local_law_package_objective"),
        "local_law_package_aux_active": provenance.get("local_law_package_aux_active"),
        "decoder_kind": decoder_kind,
        "decoder_exact": str(decoder_kind) == "exact",
        "exact_zero_claim": bool(provenance.get("exact_zero_claim", False)),
        "baseline_role": provenance.get("baseline_role"),
        "theta_mae": test.get("theta_mae"),
        "hll_register_mae": test.get("hll_register_mae"),
        "hll_estimate_raw_mae": test.get("hll_estimate_raw_mae"),
        "raw_count_mae": test.get("theta_count_raw_mae"),
        "first_accuracy": test.get("theta_first_regime_accuracy"),
        "last_accuracy": test.get("theta_last_regime_accuracy"),
        "exact_oracle_mae": oracle.get("contextual_mae"),
        "root_witness_mae": root_witness.get("root_mae"),
        "contextual_mae": test.get("contextual_mae"),
        "contextual_raw_mae": test.get("contextual_raw_mae"),
        "law_set_id": test.get("law_set_id", provenance.get("law_set_id")),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        "eps_idemp": test.get("eps_idemp"),
        "pred_truth_corr": test.get("pred_truth_corr"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a JAX/sbijax contextual-sufficiency g-state probe."
    )
    parser.add_argument(
        "--training-objective",
        default="contextual_sufficiency",
        choices=["root", "contextual_sufficiency"],
        help="Only contextual_sufficiency trains the sbijax lane; root is rejected.",
    )
    parser.add_argument(
        "--data-source",
        default="markov",
        choices=["markov", "synthetic", "hll"],
        help=(
            "markov uses the official Markov data family; synthetic keeps the "
            "tiny local Markov generator; hll uses random token streams with "
            "HLL cardinality contexts."
        ),
    )
    parser.add_argument("--benchmark", default="recoverable_v5_t2048")
    parser.add_argument(
        "--load-data-bundle",
        default=None,
        help=(
            "Load an existing MarkovOPSDataBundle JSON/PKL instead of generating "
            "or preparing a named Markov benchmark. Use this for the paper "
            "hazard-panel bundles."
        ),
    )
    parser.add_argument(
        "--sbijax-trainer",
        default="package",
        choices=[
            "repo",
            "package",
            "theta_supervised",
            "identity_theta",
            "exact_zero_markov",
            "learned_local_laws",
            "posterior",
            "npe",
            "nass_nle",
        ],
    )
    parser.add_argument("--sbijax-method", default="nasss", choices=["nass", "nasss"])
    parser.add_argument(
        "--sbijax-package-theta",
        default="response_signature",
        choices=["response_signature", "markov_exact_sketch", "hll_register_sketch"],
        help=(
            "Package-native theta target: finite response signatures or the exact "
            "Markov sufficient sketch [count/scale, first one-hot, last one-hot], "
            "or normalized HLL registers."
        ),
    )
    parser.add_argument(
        "--sbijax-input-encoding",
        default="normalized_token_ids",
        choices=[
            "normalized_token_ids",
            "one_hot_token_ids",
            "regime_ids",
            "regime_one_hot",
            "markov_exact_sketch",
        ],
        help=(
            "Input representation for package-direct summary learning. Regime "
            "encodings expose the known Markov token->regime partition."
        ),
    )
    parser.add_argument(
        "--sbijax-summary-activation",
        default="relu",
        choices=["relu", "tanh", "gelu", "swish", "silu", "elu", "leaky_relu"],
        help="Activation passed to sbijax.nn.make_nass_net/make_nasss_net.",
    )
    parser.add_argument(
        "--local-law-supervision-mode",
        default="dual",
        choices=["dense_exact", "sparse_ipw", "dual"],
        help="Observation mode for learned_local_laws.",
    )
    parser.add_argument("--local-law-weight", type=float, default=1.0)
    parser.add_argument("--local-law-leaf-weight", type=float, default=1.0)
    parser.add_argument("--local-law-merge-weight", type=float, default=1.0)
    parser.add_argument("--local-law-idempotence-weight", type=float, default=1.0)
    parser.add_argument("--local-law-contextual-weight", type=float, default=1.0)
    parser.add_argument(
        "--local-law-package-weight",
        type=float,
        default=0.0,
        help=(
            "Optional NASS/NASSS-style auxiliary weight inside "
            "learned_local_laws. Uses --sbijax-method to choose the auxiliary "
            "objective and defaults to 0 so exact-zero baselines are unchanged."
        ),
    )
    parser.add_argument(
        "--local-law-hll-estimate-weight",
        type=float,
        default=0.0,
        help=(
            "HLL-only auxiliary weight for the normalized HLL cardinality "
            "estimate implied by predicted leaf/merge registers. Defaults off."
        ),
    )
    parser.add_argument("--local-law-leaf-rate", type=float, default=1.0)
    parser.add_argument("--local-law-merge-rate", type=float, default=1.0)
    parser.add_argument("--local-law-idempotence-rate", type=float, default=1.0)
    parser.add_argument(
        "--local-law-summary-family",
        default="mlp",
        choices=["mlp", "affine_probe", "regime_transition_sum", "jax_fno", "norax_fno"],
        help=(
            "Summary optimizer family for learned_local_laws. 'jax_fno' uses "
            "the repo's internal JAX FFT-based 1-D FNO over the leaf's spatial "
            "axis (reshapes flat features to (B, fragment_len, input_width), "
            "adds a normalized position channel, and reads off via aggregate + "
            "first + last-position concat -> theta_dim). 'norax_fno' is kept as "
            "a backward-compatible alias for the internal implementation; norax "
            "is a design reference, not a runtime dependency. Compatible with "
            "--sbijax-input-encoding "
            "{regime_one_hot, one_hot_token_ids, regime_ids, normalized_token_ids}."
        ),
    )
    parser.add_argument(
        "--local-law-summary-fno-n-modes",
        type=int,
        default=16,
        help="FNO spectral modes for the summary network (clamped to L//2+1).",
    )
    parser.add_argument(
        "--local-law-summary-fno-n-layers",
        type=int,
        default=2,
        help="Number of stacked FNO layers in the summary network.",
    )
    parser.add_argument(
        "--local-law-summary-fno-pooling-mode",
        default="sum",
        choices=["sum", "mean"],
        help="Spatial pool over the FNO summary's L axis before projecting to theta_dim.",
    )
    parser.add_argument(
        "--local-law-explicit-state-decoder",
        action="store_true",
        help=(
            "Use literal paper factorization: g maps leaves/summaries to a "
            "learned summary z, then explicit f decodes z -> theta before "
            "local-law/readout losses. Requires learned_merge or fully_learned."
        ),
    )
    parser.add_argument(
        "--local-law-summary-dim",
        type=int,
        default=0,
        help=(
            "Learned summary z width for --local-law-explicit-state-decoder. "
            "0 auto-sizes to max(theta_dim, --state-dim)."
        ),
    )
    parser.add_argument(
        "--local-law-state-decoder-head",
        default="mlp",
        choices=["mlp", "linear"],
        help="Explicit f head for --local-law-explicit-state-decoder.",
    )
    parser.add_argument(
        "--local-law-count-only",
        action="store_true",
        help=(
            "Switch C1/C2 from full-sketch MSE to count-only MSE via a learned "
            "count_readout(rep) head. The summary outputs a learned rep of width "
            "--local-law-rep-dim (default 2*theta_dim); (first, last) become "
            "emergent diagnostics rather than supervision targets. Requires "
            "--law-architecture fully_learned."
        ),
    )
    parser.add_argument(
        "--local-law-rep-dim",
        type=int,
        default=0,
        help=(
            "Width of the learned rep when --local-law-count-only is set. "
            "0 (default) auto-sizes to 2 * theta_dim ('big d, fair "
            "comparison' vs the existing sketch-shape supervision)."
        ),
    )
    parser.add_argument(
        "--local-law-merge-loss",
        default="mse",
        choices=["mse", "nass_jsd", "nasss_jsd"],
        help=(
            "C2 merge-loss form. 'mse' is the existing element-wise MSE "
            "between merge_states and the merge truth. 'nass_jsd' replaces "
            "it with sbijax NASS-style JSD contrastive supervision over the "
            "full merge-target vector. 'nasss_jsd' is the sliced variant "
            "(NASSS): project the merge target onto random unit slices and "
            "average per-slice JSD MI bounds — mirrors the f-side NASSS."
        ),
    )
    parser.add_argument(
        "--merge-family",
        default="mlp",
        choices=["mlp", "fno_rep"],
        help=(
            "Architecture for the learned merge g(s_L, s_R). 'mlp' (default): "
            "asymmetric concat MLP. 'fno_rep': 1D FNO over the length-2 "
            "(left, right) rep-sequence with rep_dim channels."
        ),
    )
    parser.add_argument(
        "--merge-fno-n-modes",
        type=int,
        default=16,
        help=(
            "Spectral modes kept along the rep-dim spatial axis. Capped "
            "at state_dim//2 + 1. 16-32 is reasonable for state_dim>=64."
        ),
    )
    parser.add_argument(
        "--merge-fno-n-layers",
        type=int,
        default=2,
        help="Number of FNO blocks in the merge family.",
    )
    parser.add_argument(
        "--merge-fno-hidden-channels",
        type=int,
        default=32,
        help="Lifted channel dim for the FNO merge family.",
    )
    parser.add_argument(
        "--decoder-head",
        default="mlp",
        choices=["mlp", "linear"],
        help=(
            "Decoder head architecture (response readout from merge state). "
            "'mlp' (default): 2-layer MLP. 'linear': single Dense projection."
        ),
    )
    parser.add_argument(
        "--merge-nasss-n-slices",
        type=int,
        default=16,
        help="Number of random unit-vector slices for nasss_jsd merge loss.",
    )
    parser.add_argument(
        "--law-architecture",
        default="analytic",
        choices=["analytic", "learned_merge", "learned_decoder", "fully_learned"],
        help=(
            "Which pieces of the f/g pipeline are learned. 'analytic' uses the "
            "exact Markov merge and decoder (current zero baseline). "
            "'learned_merge' replaces the merge with an asymmetric MLP "
            "g(s_L, s_R). 'learned_decoder' replaces the analytic Markov "
            "response readout with an MLP f(state). 'fully_learned' uses both."
        ),
    )
    parser.add_argument(
        "--c2-merge-target",
        default="theta",
        choices=["theta", "self_consistency"],
        help=(
            "C2 merge supervision target. 'theta' uses the analytic full-item "
            "Markov sketch. 'self_consistency' uses "
            "stop_gradient(g_summary(full_item_features)) so merges only have "
            "to agree with the encoder, no theta on merges."
        ),
    )
    parser.add_argument(
        "--learned-merge-hidden-dim",
        type=int,
        default=0,
        help="Hidden width for the learned merge MLP (0 = use --hidden-dim).",
    )
    parser.add_argument(
        "--learned-decoder-hidden-dim",
        type=int,
        default=0,
        help="Hidden width for the learned decoder MLP (0 = use --hidden-dim).",
    )
    parser.add_argument(
        "--posterior-estimator",
        "--sbijax-posterior-estimator",
        dest="posterior_estimator",
        default="npe",
        choices=["npe", "fmpe", "cmpe", "nle", "snle", "nre"],
        help="Package-native posterior/likelihood estimator for --sbijax-trainer=posterior.",
    )
    parser.add_argument(
        "--density-family",
        "--sbijax-density-family",
        dest="density_family",
        default="mdn",
        choices=["mdn", "maf", "spf", "cnf", "cm", "resnet"],
        help="Package density/network family paired with --posterior-estimator.",
    )
    parser.add_argument("--train-docs", type=int, default=32)
    parser.add_argument("--val-docs", type=int, default=16)
    parser.add_argument("--test-docs", type=int, default=16)
    parser.add_argument(
        "--eval-docs",
        type=int,
        default=None,
        help="If set for --data-source=markov, use this cap for both val and test docs.",
    )
    parser.add_argument("--doc-tokens", type=int, default=128)
    parser.add_argument(
        "--leaf-tokens",
        type=int,
        default=128,
        help="Official Markov leaf size; used only when --data-source=markov.",
    )
    parser.add_argument("--fragment-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--hll-precision", type=int, default=4)
    parser.add_argument("--hll-hash-bits", type=int, default=64)
    parser.add_argument("--expected-boundaries", type=float, default=None)
    parser.add_argument("--target-scale", type=float, default=0.0)
    parser.add_argument("--context-samples-per-doc", type=int, default=2)
    parser.add_argument("--response-signature-contexts", type=int, default=8)
    parser.add_argument("--response-signature-slices", type=int, default=4)
    parser.add_argument(
        "--include-hybrid-diagnostics",
        action="store_true",
        help=(
            "Add Makinen-style base/neural/hybrid finite-response collision "
            "diagnostics to summary.json. Uses the exact Markov sketch as the "
            "base statistic when available."
        ),
    )
    parser.add_argument("--contextual-loss-weight", type=float, default=1.0)
    parser.add_argument("--infomax-loss-weight", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        default="constant",
        choices=["constant", "cosine"],
        help=(
            "Learning rate schedule for trainers that support it (currently "
            "learned_local_laws). 'cosine' decays init_value -> 0 over "
            "n_iter*steps_per_epoch steps; combined with best-by-val-law "
            "params return, this prevents Adam-overshoot drift."
        ),
    )
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=32,
        help="Monte Carlo samples for package posterior-mean diagnostics.",
    )
    parser.add_argument(
        "--posterior-eval-samples",
        type=int,
        default=0,
        help=(
            "Override posterior Monte Carlo samples used for diagnostics; "
            "0 reuses --posterior-samples."
        ),
    )
    parser.add_argument(
        "--posterior-eval-batch-size",
        type=int,
        default=0,
        help=("Batch size for amortized posterior diagnostic sampling; " "0 reuses --batch-size."),
    )
    parser.add_argument(
        "--posterior-sampler",
        default="nuts",
        choices=["nuts", "slice"],
        help="MCMC sampler for NLE/SNLE/NRE posterior diagnostics.",
    )
    parser.add_argument(
        "--density-components",
        type=int,
        default=5,
        help="Mixture components for package-native MDN density estimators.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.training_objective) != "contextual_sufficiency":
        raise ValueError(
            "--training-objective=root is not implemented for the sbijax lane; "
            "use --training-objective=contextual_sufficiency"
        )

    output_root = (
        Path(str(args.output_root))
        if args.output_root is not None
        else REPO / "outputs" / f"contextual_sbijax_{_ts()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    block_by_token = palette_block_map(
        vocab_size=int(args.vocab_size),
        n_regimes=int(args.n_regimes),
    )
    target_scale = (
        float(args.target_scale)
        if float(args.target_scale) > 0.0
        else float(
            max(
                1,
                (2 if str(args.data_source) == "hll" else 3) * int(args.fragment_len) - 1,
            )
        )
    )
    if str(args.data_source) == "markov":
        val_docs_requested = (
            int(args.eval_docs) if args.eval_docs is not None else int(args.val_docs)
        )
        test_docs_requested = (
            int(args.eval_docs) if args.eval_docs is not None else int(args.test_docs)
        )
        if args.load_data_bundle:
            markov_splits = load_markov_contextual_splits_from_bundle(
                Path(str(args.load_data_bundle)),
                train_docs=int(args.train_docs),
                val_docs=int(val_docs_requested),
                test_docs=int(test_docs_requested),
            )
        else:
            markov_splits = load_markov_contextual_splits(
                benchmark=str(args.benchmark),
                doc_tokens=int(args.doc_tokens),
                train_docs=int(args.train_docs),
                val_docs=int(val_docs_requested),
                test_docs=int(test_docs_requested),
                leaf_tokens=int(args.leaf_tokens),
                expected_boundaries=args.expected_boundaries,
                seed=int(args.seed),
                vocab_size=int(args.vocab_size),
                n_regimes=int(args.n_regimes),
            )
        train_docs = markov_splits.train_docs
        val_docs = markov_splits.val_docs
        test_docs = markov_splits.test_docs
        train_root_counts = markov_splits.train_root_counts
        val_root_counts = markov_splits.val_root_counts
        test_root_counts = markov_splits.test_root_counts
        block_by_token = markov_splits.block_by_token
        data_source_metadata = markov_splits.metadata
    elif str(args.data_source) == "synthetic":
        resolved_expected_boundaries = (
            float(args.expected_boundaries) if args.expected_boundaries is not None else 5.0
        )
        train_docs = make_synthetic_markov_docs(
            n_docs=int(args.train_docs),
            doc_tokens=int(args.doc_tokens),
            vocab_size=int(args.vocab_size),
            n_regimes=int(args.n_regimes),
            expected_boundaries=float(resolved_expected_boundaries),
            seed=int(args.seed),
        )
        val_docs = make_synthetic_markov_docs(
            n_docs=int(args.val_docs),
            doc_tokens=int(args.doc_tokens),
            vocab_size=int(args.vocab_size),
            n_regimes=int(args.n_regimes),
            expected_boundaries=float(resolved_expected_boundaries),
            seed=int(args.seed) + 101,
        )
        test_docs = make_synthetic_markov_docs(
            n_docs=int(args.test_docs),
            doc_tokens=int(args.doc_tokens),
            vocab_size=int(args.vocab_size),
            n_regimes=int(args.n_regimes),
            expected_boundaries=float(resolved_expected_boundaries),
            seed=int(args.seed) + 202,
        )
        data_source_metadata = {
            "data_source": "synthetic",
            "doc_tokens": int(args.doc_tokens),
            "leaf_tokens": None,
            "train_docs": int(len(train_docs)),
            "val_docs": int(len(val_docs)),
            "test_docs": int(len(test_docs)),
            "vocab_size": int(args.vocab_size),
            "n_regimes": int(args.n_regimes),
            "expected_boundaries": float(resolved_expected_boundaries),
            "seed": int(args.seed),
        }
        train_root_counts = None
        val_root_counts = None
        test_root_counts = None
    else:

        def _random_token_docs(n_docs: int, *, seed_offset: int) -> list[list[int]]:
            local_rng = np.random.default_rng(int(args.seed) + int(seed_offset))
            return [
                [
                    int(tok)
                    for tok in local_rng.integers(
                        0,
                        int(args.vocab_size),
                        size=int(args.doc_tokens),
                    )
                ]
                for _ in range(int(n_docs))
            ]

        train_docs = _random_token_docs(int(args.train_docs), seed_offset=0)
        val_docs = _random_token_docs(int(args.val_docs), seed_offset=101)
        test_docs = _random_token_docs(int(args.test_docs), seed_offset=202)
        data_source_metadata = {
            "data_source": "hll",
            "doc_tokens": int(args.doc_tokens),
            "leaf_tokens": None,
            "train_docs": int(len(train_docs)),
            "val_docs": int(len(val_docs)),
            "test_docs": int(len(test_docs)),
            "vocab_size": int(args.vocab_size),
            "hll_precision": int(args.hll_precision),
            "hll_hash_bits": int(args.hll_hash_bits),
            "seed": int(args.seed),
        }
        train_root_counts = None
        val_root_counts = None
        test_root_counts = None
    data_vocab_size = int(data_source_metadata.get("vocab_size", int(args.vocab_size)))
    if str(args.data_source) == "hll":
        contextual_problem = HLLUnionContextProblem(
            vocab_size=int(data_vocab_size),
            target_scale=float(target_scale),
            precision=int(args.hll_precision),
            hash_bits=int(args.hll_hash_bits),
        )
    else:
        contextual_problem = MarkovTwoSidedContextProblem(
            block_by_token=block_by_token,
            vocab_size=int(data_vocab_size),
            target_scale=float(target_scale),
        )

    train_dataset = build_contextual_query_dataset(
        train_docs,
        problem=contextual_problem,
        samples_per_source=int(args.context_samples_per_doc),
        item_len=int(args.fragment_len),
        n_contexts=int(args.response_signature_contexts),
        seed=int(args.seed) + 303,
    )
    if str(args.data_source) == "hll":
        train_contexts = tuple(
            HLLUnionContext(tokens=tuple(int(tok) for tok in payload.get("tokens", ())))
            for payload in train_dataset.context_payloads
        )
    else:
        train_contexts = tuple(
            MarkovTwoSidedContext(left_tokens=left, right_tokens=right)
            for left, right in zip(
                train_dataset.context_left_raw,
                train_dataset.context_right_raw,
                strict=True,
            )
        )
    val_dataset = build_contextual_query_dataset(
        val_docs,
        problem=contextual_problem,
        samples_per_source=max(1, int(args.context_samples_per_doc)),
        item_len=int(args.fragment_len),
        n_contexts=int(args.response_signature_contexts),
        seed=int(args.seed) + 404,
        contexts=train_contexts,
    )
    test_dataset = build_contextual_query_dataset(
        test_docs,
        problem=contextual_problem,
        samples_per_source=max(1, int(args.context_samples_per_doc)),
        item_len=int(args.fragment_len),
        n_contexts=int(args.response_signature_contexts),
        seed=int(args.seed) + 505,
        contexts=train_contexts,
    )
    if str(args.sbijax_package_theta) == "markov_exact_sketch":
        n_regimes_for_sketch = int(data_source_metadata.get("n_regimes", int(args.n_regimes)))
        train_dataset = with_package_theta_target(
            train_dataset,
            name="markov_exact_sketch",
            targets=markov_exact_sketch_targets_for_dataset(
                train_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
        )
        val_dataset = with_package_theta_target(
            val_dataset,
            name="markov_exact_sketch",
            targets=markov_exact_sketch_targets_for_dataset(
                val_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
        )
        test_dataset = with_package_theta_target(
            test_dataset,
            name="markov_exact_sketch",
            targets=markov_exact_sketch_targets_for_dataset(
                test_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
        )
    if str(args.sbijax_package_theta) == "hll_register_sketch":
        train_dataset = with_package_theta_target(
            train_dataset,
            name="hll_register_sketch",
            targets=hll_register_sketch_targets_for_dataset(
                train_dataset,
                precision=int(args.hll_precision),
                hash_bits=int(args.hll_hash_bits),
            ),
        )
        val_dataset = with_package_theta_target(
            val_dataset,
            name="hll_register_sketch",
            targets=hll_register_sketch_targets_for_dataset(
                val_dataset,
                precision=int(args.hll_precision),
                hash_bits=int(args.hll_hash_bits),
            ),
        )
        test_dataset = with_package_theta_target(
            test_dataset,
            name="hll_register_sketch",
            targets=hll_register_sketch_targets_for_dataset(
                test_dataset,
                precision=int(args.hll_precision),
                hash_bits=int(args.hll_hash_bits),
            ),
        )

    config = ContextualSBIJAXConfig(
        trainer=str(args.sbijax_trainer),
        method=str(args.sbijax_method),
        package_theta=str(args.sbijax_package_theta),
        input_encoding=str(args.sbijax_input_encoding),
        summary_activation=str(args.sbijax_summary_activation),
        vocab_size=int(data_vocab_size),
        embedding_dim=int(args.embedding_dim),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        response_signature_contexts=int(args.response_signature_contexts),
        response_signature_slices=int(args.response_signature_slices),
        contextual_loss_weight=float(args.contextual_loss_weight),
        infomax_loss_weight=float(args.infomax_loss_weight),
        local_law_supervision_mode=str(args.local_law_supervision_mode),
        local_law_weight=float(args.local_law_weight),
        local_law_leaf_weight=float(args.local_law_leaf_weight),
        local_law_merge_weight=float(args.local_law_merge_weight),
        local_law_idempotence_weight=float(args.local_law_idempotence_weight),
        local_law_contextual_weight=float(args.local_law_contextual_weight),
        local_law_package_weight=float(args.local_law_package_weight),
        local_law_hll_estimate_weight=float(args.local_law_hll_estimate_weight),
        local_law_leaf_rate=float(args.local_law_leaf_rate),
        local_law_merge_rate=float(args.local_law_merge_rate),
        local_law_idempotence_rate=float(args.local_law_idempotence_rate),
        local_law_summary_family=str(args.local_law_summary_family),
        local_law_summary_fno_n_modes=int(args.local_law_summary_fno_n_modes),
        local_law_summary_fno_n_layers=int(args.local_law_summary_fno_n_layers),
        local_law_summary_fno_pooling_mode=str(args.local_law_summary_fno_pooling_mode),
        local_law_explicit_state_decoder=bool(args.local_law_explicit_state_decoder),
        local_law_summary_dim=int(args.local_law_summary_dim),
        local_law_state_decoder_head=str(args.local_law_state_decoder_head),
        local_law_count_only=bool(args.local_law_count_only),
        local_law_rep_dim=int(args.local_law_rep_dim),
        local_law_merge_loss=str(args.local_law_merge_loss),
        merge_family=str(args.merge_family),
        merge_fno_n_modes=int(args.merge_fno_n_modes),
        merge_fno_n_layers=int(args.merge_fno_n_layers),
        merge_fno_hidden_channels=int(args.merge_fno_hidden_channels),
        decoder_head=str(args.decoder_head),
        hll_precision=int(args.hll_precision),
        hll_hash_bits=int(args.hll_hash_bits),
        merge_nasss_n_slices=int(args.merge_nasss_n_slices),
        law_architecture=str(args.law_architecture),
        c2_merge_target=str(args.c2_merge_target),
        learned_merge_hidden_dim=int(args.learned_merge_hidden_dim),
        learned_decoder_hidden_dim=int(args.learned_decoder_hidden_dim),
        learning_rate=float(args.learning_rate),
        lr_schedule=str(args.lr_schedule),
        n_iter=int(args.n_iter),
        batch_size=int(args.batch_size),
        posterior_samples=int(args.posterior_samples),
        posterior_estimator=str(args.posterior_estimator),
        density_family=str(args.density_family),
        posterior_eval_samples=int(args.posterior_eval_samples),
        posterior_eval_batch_size=int(args.posterior_eval_batch_size),
        posterior_sampler=str(args.posterior_sampler),
        density_components=int(args.density_components),
        seed=int(args.seed),
    )

    try:
        result = fit_contextual_sbijax(train_dataset, val_dataset, config=config)
        test_diagnostics = evaluate_contextual_sbijax(
            params=result.params,
            apply_fn=result.apply_fn,
            dataset=test_dataset,
        )
        train_diagnostics = result.train_diagnostics
        val_diagnostics = result.val_diagnostics
        history = result.history
        provenance = result.provenance
    except ImportError as exc:
        if CONTEXTUAL_SBI_INSTALL_MSG not in str(exc):
            raise
        provenance = contextual_sbijax_provenance(
            method=str(args.sbijax_method),
            response_signature_contexts=int(args.response_signature_contexts),
            response_signature_slices=int(args.response_signature_slices),
            trainer=str(args.sbijax_trainer),
            input_encoding=str(args.sbijax_input_encoding),
            summary_activation=str(args.sbijax_summary_activation),
        )
        payload = {
            "status": "missing_dependency",
            "error": str(exc),
            "provenance": provenance,
            "args": vars(args),
        }
        _write_json(output_root / "summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    payload = {
        "status": "ok",
        "args": vars(args),
        "data_source": str(args.data_source),
        "trainer": str(args.sbijax_trainer),
        "package_theta": str(args.sbijax_package_theta),
        "input_encoding": str(args.sbijax_input_encoding),
        "summary_activation": str(args.sbijax_summary_activation),
        "posterior_estimator": str(args.posterior_estimator),
        "density_family": str(args.density_family),
        "posterior_eval_samples": int(args.posterior_eval_samples),
        "posterior_eval_batch_size": int(args.posterior_eval_batch_size),
        "posterior_sampler": str(args.posterior_sampler),
        "data_source_metadata": data_source_metadata,
        "context_bank_metadata": {
            "problem_id": str(contextual_problem.problem_id),
            "context_kind": str(contextual_problem.context_kind),
            "source_split": "train",
            "response_signature_contexts": int(args.response_signature_contexts),
            "context_tensor_shapes": {
                str(name): list(values.shape)
                for name, values in train_dataset.context_tensors.items()
            },
            "left_context_shape": list(train_dataset.context_left_tokens.shape),
            "right_context_shape": list(train_dataset.context_right_tokens.shape),
        },
        "target_scale": float(target_scale),
        "train_dataset": train_dataset.metadata,
        "val_dataset": val_dataset.metadata,
        "test_dataset": test_dataset.metadata,
        "provenance": provenance,
        "history": history,
        "diagnostics": {
            "train": train_diagnostics,
            "val": val_diagnostics,
            "test": test_diagnostics,
        },
    }
    if str(args.data_source) != "hll":
        payload["diagnostics"]["exact_root_witness"] = {
            "train": exact_root_witness_diagnostics(
                train_docs,
                block_by_token=block_by_token,
                root_counts=train_root_counts,
            ),
            "val": exact_root_witness_diagnostics(
                val_docs,
                block_by_token=block_by_token,
                root_counts=val_root_counts,
            ),
            "test": exact_root_witness_diagnostics(
                test_docs,
                block_by_token=block_by_token,
                root_counts=test_root_counts,
            ),
        }
    if str(args.data_source) == "markov":
        n_regimes_for_sketch = int(data_source_metadata.get("n_regimes", int(args.n_regimes)))
        payload["diagnostics"]["markov_exact_sketch_oracle"] = {
            "train": markov_exact_sketch_oracle_diagnostics(
                train_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
            "val": markov_exact_sketch_oracle_diagnostics(
                val_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
            "test": markov_exact_sketch_oracle_diagnostics(
                test_dataset,
                block_by_token=block_by_token,
                target_scale=float(target_scale),
                n_regimes=int(n_regimes_for_sketch),
            ),
        }
    payload["metric_summary"] = _metric_summary(payload)
    if bool(args.include_hybrid_diagnostics):
        n_regimes_for_sketch = int(data_source_metadata.get("n_regimes", int(args.n_regimes)))

        def _hybrid_diagnostics_for_dataset(dataset):
            if str(args.sbijax_package_theta) == "hll_register_sketch":
                base_states = hll_register_sketch_targets_for_dataset(
                    dataset,
                    precision=int(args.hll_precision),
                    hash_bits=int(args.hll_hash_bits),
                )
            else:
                base_states = markov_exact_sketch_targets_for_dataset(
                    dataset,
                    block_by_token=block_by_token,
                    target_scale=float(target_scale),
                    n_regimes=int(n_regimes_for_sketch),
                )
            return hybrid_summary_diagnostics_for_contextual_sbijax(
                params=result.params,
                apply_fn=result.apply_fn,
                dataset=dataset,
                base_states=base_states,
            )

        payload["diagnostics"]["hybrid_summary"] = {
            "base_statistic": str(args.sbijax_package_theta),
            "neural_statistic": "sbijax_state",
            "train": _hybrid_diagnostics_for_dataset(train_dataset),
            "val": _hybrid_diagnostics_for_dataset(val_dataset),
            "test": _hybrid_diagnostics_for_dataset(test_dataset),
        }
    _write_json(output_root / "summary.json", payload)
    with (output_root / "history.jsonl").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
