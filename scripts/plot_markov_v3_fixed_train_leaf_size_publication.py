#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    build_budgeted_train_supervision_manifest,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID,
    STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP,
    resolve_full_doc_diagnostic_benchmark,
)
from scripts.report_markov_optimization_tradeoffs import (
    FNO_OFFICIAL_COLOR,
    NEUTRAL_COLOR,
    SUPERVISION_RECOVERY_STRUCTURAL_CELL,
    TREE_PRIMARY_COLOR,
    _effective_fixed_leaf_tokens,
    _effective_leaves_per_doc,
    _preferred_recovery_row,
    _safe_float,
    _safe_int,
    _scope_label_from_recovery,
)

DEFAULT_REPORT_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "markov_v3_rolling_partial_report_current"
    / "summary.json"
)

FNO_EQUIVALENT_COLOR = "#b45309"
MASS_EQUIVALENT_COLOR = "#1d4ed8"
EMPIRICAL_MEAN_BASELINE_COLOR = "#6b7280"
EMPIRICAL_BAYES_BASELINE_COLOR = "#0f766e"

PUBLICATION_TIERS = {"publication_fullval", "publication_fullval_v3", "publication_xlarge"}

PRIMARY_ROOT_ONLY_BUNDLE_PREFERENCE_BY_LEAF = {
    128: [
        "oneleaf_root_budget_publication_fullval",
        "root_budget_ladder_large_train",
    ],
    64: [
        "root_budget_publication_multileaf_fullval",
        "root_budget_ladder_large_train",
    ],
    32: [
        "root_budget_publication_multileaf_fullval",
        "root_budget_ladder_large_train",
    ],
    16: [
        "root_budget_ladder_large_train",
        "root_budget_publication_multileaf_fullval",
    ],
    8: [
        "root_budget_ladder_large_train",
        "root_budget_publication_multileaf_fullval",
    ],
}

PRIMARY_ONELEAF_LINEAGE_TOKENS = (
    "manifestfix",
    "subsetfix_v2",
    "subsetfix",
    "exactcollapse_identity",
    "publication_fullval_v3_oneleaf",
    "publication_fullval",
)

ONELEAF_LOCAL_LAW_BUNDLE_PREFERENCE = (
    "oneleaf_local_law_root_sweep_fullval",
    "oneleaf_local_law_root_sweep_xlarge",
    "preset_ablation_full_laws",
)

ONELEAF_LOCAL_LAW_LINEAGE_TOKENS = (
    "oneleaf_local_law_root_sweep",
    "one_leaf_parity_diag",
    "overnight_fill",
    "check_basics_fixed",
)

ALTERNATE_ROOT_ONLY_BUNDLE_PREFERENCE_BY_LEAF = {
    128: [
        "oneleaf_root_budget_longschedule_fill_fullval",
        "oneleaf_root_budget_longschedule_fill_xlarge",
        "root_budget_ladder_large_train_longschedule",
    ],
    64: ["root_budget_ladder_large_train_longschedule"],
    32: ["root_budget_ladder_large_train_longschedule"],
    16: ["root_budget_ladder_large_train_longschedule"],
    8: ["root_budget_ladder_large_train_longschedule"],
}

ALTERNATE_ROOT_ONLY_LINEAGE_TOKENS = (
    "oneleaf_root_budget_longschedule_fill",
    "root_budget_ladder_large_train_longschedule",
    "large_train_tuning",
)

RECOVERABLE_SCOPE_KEYS = (
    "recoverable_v5_t128",
    "recoverable_v4_t128",
    "recoverable_v5",
    "recoverable_v4",
)
RECOVERABLE_SCOPE_PREFERENCE_BY_LEAF = {
    128: ("recoverable_v5_t128", "recoverable_v4_t128", "recoverable_v5", "recoverable_v4"),
}
STRUCTURAL_SCOPE_DEFAULT = str(DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID or SUPERVISION_RECOVERY_STRUCTURAL_CELL)
STRUCTURAL_SCOPE_ALIASES: Dict[str, tuple[str, ...]] = {}
for _legacy_alias, _canonical in dict(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP).items():
    STRUCTURAL_SCOPE_ALIASES.setdefault(str(_canonical), tuple())
for _canonical in list(STRUCTURAL_SCOPE_ALIASES.keys()):
    STRUCTURAL_SCOPE_ALIASES[_canonical] = tuple(
        alias
        for alias, mapped in dict(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP).items()
        if str(mapped) == str(_canonical)
    )

SCOPE_PRESENTATION = {
    "recoverable_v5_t128": {
        "title": "Counting Topic Changes (Simple Case)",
        "subtitle": (
            "DGP: 128-token documents, 4 hidden regimes, and a simple stay/switch regime process calibrated "
            "to about 5 expected regime changes per document. Regimes use disjoint token palettes with 4 observed "
            "tokens per regime (16 total tokens), and the label is the number of regime changes across the full document."
        ),
    },
    "recoverable_v5": {
        "title": "Counting Topic Changes (Simple Case)",
        "subtitle": (
            "DGP: 128-token documents, 4 hidden regimes, and a simple stay/switch regime process calibrated "
            "to about 5 expected regime changes per document. Regimes use disjoint token palettes with 4 observed "
            "tokens per regime (16 total tokens), and the label is the number of regime changes across the full document."
        ),
    },
    "recoverable_v4": {
        "title": "Counting Topic Changes (Simple Case)",
        "subtitle": (
            "DGP: 128-token documents, 4 hidden regimes, and 2-6 contiguous segments per document. "
            "Regimes use disjoint token palettes with 4 observed tokens per regime (16 total tokens), and the label is the "
            "number of regime changes across the full document. On this benchmark's evaluation split, that "
            "count takes values 3, 4, or 5."
        ),
    },
    "r12_seg10to12": {
        "title": "Counting Topic Changes (Harder Case)",
        "subtitle": (
            "DGP: 128-token documents, 12 hidden regimes, and 10-12 contiguous segments per document. "
            "Regimes use disjoint token palettes with 4 observed tokens per regime (48 total tokens), and the label is the "
            "number of regime changes across the full document. This version is harder because the same 128-token budget "
            "must represent many more topic identities and many more boundaries."
        ),
    },
    "r12_seg10to12__sticky": {
        "title": "Counting Topic Changes (Harder Case)",
        "subtitle": (
            "DGP: 128-token documents, 12 hidden regimes, and a simple stay/switch regime process calibrated "
            "to the high-density boundary setting used by this structural anchor. Regimes use disjoint token "
            "palettes with 4 observed tokens per regime (48 total tokens), and the label is the number of "
            "regime changes across the full document. This version is harder because the same 128-token budget "
            "must represent many more topic identities and many more boundaries."
        ),
    },
}


def _preferred_available_scope_key(
    recovery: Mapping[str, Any],
    scope_keys: Sequence[str],
) -> str:
    scopes = dict(recovery.get("scopes") or {})
    for key in scope_keys:
        if key in scopes:
            return str(key)
    return str(scope_keys[0]) if scope_keys else ""


def _scope_presentation_key(
    recovery: Mapping[str, Any],
    *,
    primary_scope_key: str,
) -> str:
    normalized_primary = str(primary_scope_key or "").strip()
    if normalized_primary == "r12_seg10to12":
        scope_label = str(
            (dict(recovery.get("scopes") or {}).get(normalized_primary) or {}).get("scope_label", "")
            or ""
        ).strip()
        if "structural_core_v2" in scope_label:
            return str(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP.get(normalized_primary, normalized_primary))
        return "r12_seg10to12"
    if normalized_primary.startswith("recoverable_v5"):
        return _preferred_available_scope_key(recovery, RECOVERABLE_SCOPE_KEYS)
    structural_candidates = _structural_scope_candidates(normalized_primary)
    if structural_candidates:
        scopes = dict(recovery.get("scopes") or {})
        for key in structural_candidates:
            if key in scopes:
                scope_label = str((dict(scopes.get(key) or {})).get("scope_label", "") or "").strip()
                if "structural_core_v2" in scope_label and key in STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP:
                    return str(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP.get(key, key))
                return str(key)
        return str(structural_candidates[0])
    return str(normalized_primary)


def _structural_scope_candidates(primary_scope_key: str) -> tuple[str, ...]:
    normalized = str(primary_scope_key or "").strip()
    if not normalized:
        return tuple()
    canonical = str(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP.get(normalized, normalized))
    candidates: list[str] = []
    for value in (canonical, *STRUCTURAL_SCOPE_ALIASES.get(canonical, ()), normalized):
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return tuple(candidates)


def _benchmark_name_for_scope_key(
    scope_key: str,
    *,
    recovery: Mapping[str, Any],
) -> str:
    normalized = str(scope_key or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("recoverable_v5"):
        return str(_preferred_available_scope_key(recovery, RECOVERABLE_SCOPE_KEYS))
    structural_candidates = _structural_scope_candidates(normalized)
    if structural_candidates:
        return f"structural_core_v2_t128::{structural_candidates[0]}"
    return str(normalized)


def _sticky_scope_presentation(
    benchmark_name: str,
    *,
    title: str,
) -> Dict[str, str]:
    try:
        benchmark = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    except Exception:
        return {}
    config_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    doc_tokens = int(
        _safe_int(
            config_overrides.get("max_tokens"),
            config_overrides.get("min_tokens"),
        )
    )
    regime_count = int(_safe_int(getattr(benchmark, "regime_count", 0), 0) or 0)
    if regime_count <= 0:
        regime_count = int(_safe_int(config_overrides.get("n_regimes"), 0) or 0)
    switch_prob = _safe_float(
        getattr(benchmark, "hazard_switch_prob", float("nan")),
        float("nan"),
    )
    expected_changes = (
        float(switch_prob) * float(max(0, doc_tokens - 1))
        if math.isfinite(switch_prob) and doc_tokens > 0
        else float("nan")
    )
    vocab_size = int(
        _safe_int(config_overrides.get("vocab_size"), 0)
    )
    palette_size = int(round(float(vocab_size) / float(max(1, regime_count))))
    total_vocab = int(vocab_size if vocab_size > 0 else palette_size * regime_count)
    if (
        doc_tokens <= 0
        or regime_count <= 0
        or not math.isfinite(switch_prob)
        or palette_size <= 0
        or total_vocab <= 0
    ):
        return {}
    subtitle = (
        f"DGP: {int(doc_tokens)}-token documents, {int(regime_count)} hidden regimes, and a simple "
        f"stay/switch regime process with per-token switch probability {float(switch_prob):.5f}, "
        f"implying about {expected_changes:.0f} expected regime changes per document. "
        f"Regimes use disjoint token palettes with {int(palette_size)} observed tokens per regime "
        f"({int(total_vocab)} total tokens), and the label is the number of regime changes across the full document."
    )
    return {
        "title": str(title),
        "subtitle": str(subtitle),
    }


def _scope_presentation(
    recovery: Mapping[str, Any],
    *,
    primary_scope_key: str,
) -> Dict[str, str]:
    benchmark_name = _benchmark_name_for_scope_key(
        primary_scope_key,
        recovery=recovery,
    )
    if str(benchmark_name).startswith("recoverable_v5"):
        sticky = _sticky_scope_presentation(
            benchmark_name,
            title="Counting Topic Changes (Simple Case)",
        )
        if sticky:
            return sticky
    structural_candidates = _structural_scope_candidates(primary_scope_key)
    if structural_candidates:
        sticky = _sticky_scope_presentation(
            f"structural_core_v2_t128::{structural_candidates[0]}",
            title="Counting Topic Changes (Harder Case)",
        )
        if sticky and str(structural_candidates[0]) == "r12_p079":
            return sticky
    presentation_key = _scope_presentation_key(recovery, primary_scope_key=primary_scope_key)
    scope_presentation = dict(SCOPE_PRESENTATION.get(presentation_key) or {})
    if scope_presentation:
        return scope_presentation
    if "_p" in str(presentation_key) and str(presentation_key).startswith("r"):
        try:
            benchmark = resolve_full_doc_diagnostic_benchmark(
                f"structural_core_v2_t128::{presentation_key}"
            )
        except Exception:
            return {}
        regime_count = int(getattr(benchmark, "regime_count", 0) or 0)
        density_label = (
            "Lower-Switch Case"
            if "lower" in str(getattr(benchmark, "segment_density_band", "")).lower()
            else "Higher-Switch Case"
        )
        sticky = _sticky_scope_presentation(
            f"structural_core_v2_t128::{presentation_key}",
            title=f"Counting Topic Changes ({int(regime_count)} Regimes, {density_label})",
        )
        if sticky:
            return sticky
    return {}

SECONDARY_TREE_SERIES_CHOICES = ("leaf_mass_eq", "depth_equal_mass_eq")

SECONDARY_TREE_SERIES_CONFIG: Dict[str, Dict[str, str]] = {
    "leaf_mass_eq": {
        "legend_label": "Tree, same root-label budget as panel, with missing supervision reallocated to leaves",
        "caption_label": (
            "The blue dashed line keeps the training set fixed and keeps the panel's root-label budget fixed, "
            "but reallocates the missing supervision mass to count-only leaf labels so the total supervision "
            "mass matches the full100 budget."
        ),
        "filename_suffix": "with_leaf_mass_equivalent",
    },
    "depth_equal_mass_eq": {
        "legend_label": "Tree, same root-label budget as panel, with missing supervision reallocated across leaves and merges",
        "caption_label": (
            "The blue dashed line keeps the training set fixed and keeps the panel's root-label budget fixed, "
            "but reallocates the missing supervision mass across leaves and non-root merge depths so the total "
            "supervision mass matches the full100 budget."
        ),
        "filename_suffix": "with_depth_equal_mass_equivalent",
    },
}

LABEL_TREE_ROOT_ONLY = "Tree, root supervision only"
LABEL_TREE_NO_LOCAL_LEAF128 = "Tree without local laws at leaf128"
LABEL_TREE_DUPLICATE_COUNT_ONLY = "One-leaf tree with duplicate count labels only"
LABEL_TREE_DUPLICATE_LOCAL = "One-leaf tree with richer duplicate local labels"
LABEL_FNO_SAME_BUDGET = "Official FNO at the same training set and root-label budget"
LABEL_EMPIRICAL_MEAN = "Empirical-mean guess from reviewed root docs"
LABEL_EMPIRICAL_BAYES_LIMIT = "Empirical Bayes limit with DGP known"


def _mass_matched_rate_suffix(rate_percent: float) -> str:
    return f"{float(rate_percent):.1f}".replace(".", "p")


def _leaf_mass_preserving_package_name(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return f"r{int(root_share)}_leaf_mass_eq_{_mass_matched_rate_suffix(local_mass_percent)}"


def _depth_equal_mass_preserving_package_name(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return (
        f"r{int(root_share)}_depth_equal_mass_eq_"
        f"{_mass_matched_rate_suffix(local_mass_percent)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed-train-doc leaf-size publication plots from the current "
            "Markov v3 rolling summary."
        )
    )
    parser.add_argument(
        "--report-summary",
        type=Path,
        default=DEFAULT_REPORT_SUMMARY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / f"markov_v3_publication_leaf_size_fixed_docs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--train-doc-count",
        type=int,
        default=10240,
    )
    parser.add_argument(
        "--train-doc-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of fixed train-doc counts to render in one bundle.",
    )
    parser.add_argument(
        "--all-available-train-doc-counts",
        action="store_true",
        help="Auto-discover all available train-doc counts from the rolling summary and render each one.",
    )
    parser.add_argument(
        "--root-shares",
        type=int,
        nargs="+",
        default=[100, 90, 80, 70, 50, 20, 10],
    )
    parser.add_argument(
        "--secondary-tree-series",
        type=str,
        nargs="*",
        choices=SECONDARY_TREE_SERIES_CHOICES,
        default=[],
        help=(
            "Optional sibling figure variants that overlay a second tree series on the "
            "same scaffold. Current options are the leaf-only and depth-equal "
            "equal-total-mass families."
        ),
    )
    parser.add_argument(
        "--structural-scope-key",
        type=str,
        default=STRUCTURAL_SCOPE_DEFAULT,
        help=(
            "Structural scope key to render for the non-recoverable panel family. "
            "New explicit sticky cases use ids such as r4_p031, r12_p031, r4_p079, r12_p079."
        ),
    )
    parser.add_argument(
        "--empirical-bayes",
        choices=("off", "collapsed_hmm"),
        default="off",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_train_doc_counts(
    recovery: Mapping[str, Any],
    *,
    scope_keys: Sequence[str],
) -> List[int]:
    discovered: set[int] = set()
    scopes = dict(recovery.get("scopes") or {})
    for scope_key in scope_keys:
        scope_payload = dict(scopes.get(str(scope_key)) or {})
        rows_by_train_docs = dict(scope_payload.get("rows_by_train_docs") or {})
        for train_doc_key, payload in rows_by_train_docs.items():
            train_doc_count = int(_safe_int(train_doc_key, 0))
            if train_doc_count <= 0:
                continue
            rows = list(dict(payload or {}).get("rows") or [])
            if rows:
                discovered.add(train_doc_count)
    return sorted(discovered)


_RAW_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}
_BUNDLE_CACHE: Dict[str, MarkovOPSDataBundle] = {}
_EMPIRICAL_MEAN_BASELINE_CACHE: Dict[str, Dict[str, float | int | str]] = {}
_EMPIRICAL_BAYES_BASELINE_CACHE: Dict[str, Dict[str, float | int | str]] = {}


def _root_only_package(root_share: int) -> str:
    return f"full{int(root_share)}"


def _raw_source_summary(path_text: str) -> Dict[str, Any]:
    normalized = str(path_text or "").strip()
    if not normalized:
        return {}
    cached = _RAW_SUMMARY_CACHE.get(normalized)
    if cached is not None:
        return cached
    path = Path(normalized)
    if not path.exists():
        _RAW_SUMMARY_CACHE[normalized] = {}
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        payload = {}
    _RAW_SUMMARY_CACHE[normalized] = payload
    return payload


def _load_bundle(path_text: str) -> MarkovOPSDataBundle | None:
    normalized = str(path_text or "").strip()
    if not normalized:
        return None
    cached = _BUNDLE_CACHE.get(normalized)
    if cached is not None:
        return cached
    path = Path(normalized)
    if not path.exists():
        return None
    try:
        bundle = MarkovOPSDataBundle.load(path)
    except Exception:
        return None
    _BUNDLE_CACHE[normalized] = bundle
    return bundle


def _row_raw_run_config(row: Mapping[str, Any]) -> Dict[str, Any]:
    raw_summary = _raw_source_summary(str(row.get("source_summary_json", "") or ""))
    return dict(raw_summary.get("config") or {})


def _row_bundle_path(row: Mapping[str, Any]) -> str:
    raw_summary = _raw_source_summary(str(row.get("source_summary_json", "") or ""))
    bundle_manifest = dict(raw_summary.get("bundle_manifest") or {})
    for value in bundle_manifest.values():
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    benchmark_spec = dict(raw_summary.get("benchmark_spec") or {})
    train_doc_count = int(
        _safe_int(
            row.get("train_doc_count"),
            _safe_int(
                (_row_raw_run_config(row) or {}).get("train_docs"),
                0,
            ),
        )
    )
    canonical_capacity = int(_safe_int(benchmark_spec.get("canonical_train_docs_capacity"), 0))
    expanded_capacity = int(_safe_int(benchmark_spec.get("expanded_train_docs_capacity"), 0))
    canonical_path = str(benchmark_spec.get("canonical_bundle_path", "") or "").strip()
    expanded_path = str(benchmark_spec.get("expanded_bundle_path", "") or "").strip()
    if expanded_path and expanded_capacity > 0 and train_doc_count > max(0, canonical_capacity):
        return expanded_path
    if canonical_path:
        return canonical_path
    if expanded_path:
        return expanded_path
    return ""


def _ops_config_from_row(row: Mapping[str, Any]) -> OPSCountConfig | None:
    config_map = dict(_row_raw_run_config(row) or {})
    if not config_map:
        return None
    known_fields = set(OPSCountConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_map.items() if k in known_fields}
    try:
        return OPSCountConfig(**filtered)
    except Exception:
        return None


def _oracle_root_count(doc: Any) -> float:
    return float(len(tuple(getattr(doc, "true_boundaries", ()) or ())))


def _empirical_mean_baseline_from_row(row: Mapping[str, Any]) -> Dict[str, float | int | str]:
    source_summary_json = str(row.get("source_summary_json", "") or "").strip()
    if not source_summary_json:
        return {}
    cached = _EMPIRICAL_MEAN_BASELINE_CACHE.get(source_summary_json)
    if cached is not None:
        return dict(cached)

    bundle_path = _row_bundle_path(row)
    bundle = _load_bundle(bundle_path)
    config = _ops_config_from_row(row)
    train_doc_count = int(
        _safe_int(
            row.get("train_doc_count"),
            _safe_int(getattr(config, "train_docs", 0) if config is not None else 0, 0),
        )
    )
    baseline_family = str(
        row.get("baseline_family")
        or (_row_raw_run_config(row) or {}).get("baseline_family")
        or "tree_neural"
    ).strip()
    if bundle is None or config is None or train_doc_count <= 0:
        return {}
    if len(bundle.train_docs) < train_doc_count or not bundle.test_docs:
        return {}

    train_docs = tuple(bundle.train_docs[: int(train_doc_count)])
    try:
        manifest = build_budgeted_train_supervision_manifest(
            docs=train_docs,
            config=config,
            baseline_family=baseline_family,
            seed=int(getattr(config, "seed", 0)),
        )
    except Exception:
        return {}
    if manifest is None:
        return {}

    reviewed_docs = [
        doc
        for doc, plan in zip(train_docs, manifest.doc_plans)
        if str(plan.document_mode or "").strip()
    ]
    if not reviewed_docs:
        return {}

    train_mean = float(
        sum(_oracle_root_count(doc) for doc in reviewed_docs)
        / float(max(1, len(reviewed_docs)))
    )
    test_mae = float(
        sum(abs(_oracle_root_count(doc) - train_mean) for doc in bundle.test_docs)
        / float(max(1, len(bundle.test_docs)))
    )
    payload: Dict[str, float | int | str] = {
        "train_mean_root_count": float(train_mean),
        "test_root_mae": float(test_mae),
        "reviewed_root_docs": int(len(reviewed_docs)),
        "bundle_path": str(bundle_path),
    }
    _EMPIRICAL_MEAN_BASELINE_CACHE[source_summary_json] = dict(payload)
    return payload


def _benchmark_name_from_row(row: Mapping[str, Any]) -> str:
    raw_summary = _raw_source_summary(str(row.get("source_summary_json", "") or ""))
    benchmark_spec = dict(raw_summary.get("benchmark_spec") or {})
    benchmark_name = str(benchmark_spec.get("name", "") or "").strip()
    if benchmark_name:
        return benchmark_name
    scope_key = str(row.get("scope_key", "") or "").strip()
    hardness_grid = str(row.get("hardness_grid", "") or "").strip()
    if scope_key.startswith("recoverable_v"):
        return str(scope_key)
    if hardness_grid:
        return f"{hardness_grid}::{scope_key}"
    structural_candidates = _structural_scope_candidates(scope_key)
    if structural_candidates:
        return f"structural_core_v2_t128::{structural_candidates[0]}"
    return str(scope_key)


def _token_regime_lookup_from_benchmark_name(
    benchmark_name: str,
) -> Dict[int, int]:
    benchmark = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    config_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    n_regimes = int(_safe_int(getattr(benchmark, "regime_count", 0), 0) or 0)
    if n_regimes <= 0:
        n_regimes = int(_safe_int(config_overrides.get("n_regimes"), 0) or 0)
    vocab_size = int(
        _safe_int(config_overrides.get("vocab_size"), 0)
    )
    if n_regimes <= 0 or vocab_size <= 0:
        return {}
    base_block = int(vocab_size // max(1, n_regimes))
    if base_block <= 0 or int(base_block * n_regimes) != int(vocab_size):
        return {}
    return {
        int(token): int(token // base_block)
        for token in range(int(vocab_size))
    }


def _observed_change_count_from_tokens(
    tokens: Sequence[int],
    *,
    token_to_regime: Mapping[int, int],
) -> float:
    mapped = [
        int(token_to_regime[int(token)])
        for token in list(tokens or [])
        if int(token) in token_to_regime
    ]
    if len(mapped) < 2:
        return 0.0
    return float(
        sum(1 for left, right in zip(mapped[:-1], mapped[1:]) if int(left) != int(right))
    )


def _empirical_bayes_baseline_from_row(
    row: Mapping[str, Any],
) -> Dict[str, float | int | str | bool]:
    source_summary_json = str(row.get("source_summary_json", "") or "").strip()
    if not source_summary_json:
        return {}
    cached = _EMPIRICAL_BAYES_BASELINE_CACHE.get(source_summary_json)
    if cached is not None:
        return dict(cached)

    bundle_path = _row_bundle_path(row)
    bundle = _load_bundle(bundle_path)
    config = _ops_config_from_row(row)
    train_doc_count = int(
        _safe_int(
            row.get("train_doc_count"),
            _safe_int(getattr(config, "train_docs", 0) if config is not None else 0, 0),
        )
    )
    baseline_family = str(
        row.get("baseline_family")
        or (_row_raw_run_config(row) or {}).get("baseline_family")
        or "tree_neural"
    ).strip()
    benchmark_name = _benchmark_name_from_row(row)
    if bundle is None or config is None or train_doc_count <= 0 or not benchmark_name:
        return {}
    if len(bundle.train_docs) < train_doc_count or not bundle.test_docs:
        return {}

    token_to_regime = _token_regime_lookup_from_benchmark_name(benchmark_name)
    if not token_to_regime:
        return {}

    train_docs = tuple(bundle.train_docs[: int(train_doc_count)])
    try:
        manifest = build_budgeted_train_supervision_manifest(
            docs=train_docs,
            config=config,
            baseline_family=baseline_family,
            seed=int(getattr(config, "seed", 0)),
        )
    except Exception:
        return {}
    if manifest is None:
        return {}

    reviewed_docs = [
        doc
        for doc, plan in zip(train_docs, manifest.doc_plans)
        if str(plan.document_mode or "").strip()
    ]
    if not reviewed_docs:
        return {}

    reviewed_transitions = sum(max(0, len(tuple(doc.tokens)) - 1) for doc in reviewed_docs)
    reviewed_changes = sum(
        _observed_change_count_from_tokens(
            tuple(getattr(doc, "tokens", ()) or ()),
            token_to_regime=token_to_regime,
        )
        for doc in reviewed_docs
    )
    fitted_switch_prob = float(
        (float(reviewed_changes) + 0.5) / float(max(1, reviewed_transitions) + 1.0)
    )
    test_predictions = [
        _observed_change_count_from_tokens(
            tuple(getattr(doc, "tokens", ()) or ()),
            token_to_regime=token_to_regime,
        )
        for doc in bundle.test_docs
    ]
    test_truths = [_oracle_root_count(doc) for doc in bundle.test_docs]
    test_mae = float(
        sum(abs(float(truth) - float(pred)) for truth, pred in zip(test_truths, test_predictions))
        / float(max(1, len(test_truths)))
    )
    payload: Dict[str, float | int | str | bool] = {
        "train_switch_prob_hat": float(fitted_switch_prob),
        "test_root_mae": float(test_mae),
        "reviewed_root_docs": int(len(reviewed_docs)),
        "reviewed_transitions": int(reviewed_transitions),
        "benchmark_name": str(benchmark_name),
        "bundle_path": str(bundle_path),
        "posterior_collapse_via_disjoint_palettes": True,
    }
    _EMPIRICAL_BAYES_BASELINE_CACHE[source_summary_json] = dict(payload)
    return payload


def _plot_tree_root_mae_from_row(row: Mapping[str, Any]) -> float:
    raw_value = _safe_float(
        row.get("tree_test_root_mae"),
        _safe_float(row.get("test_root_mae_mean"), float("nan")),
    )
    source_summary_json = str(row.get("source_summary_json", "") or "").strip()
    comparison_mode = str(row.get("comparison_mode", "") or "").strip()
    is_one_leaf = int(_effective_fixed_leaf_tokens(row)) == 128 and int(_effective_leaves_per_doc(row)) == 1
    if source_summary_json and comparison_mode == "exact_collapse" and is_one_leaf:
        raw_summary = _raw_source_summary(source_summary_json)
        aggregate_rows = list(raw_summary.get("aggregate_rows") or [])
        if aggregate_rows:
            exact_value = _safe_float(
                dict(aggregate_rows[0]).get("test_root_mae_mean"),
                float("nan"),
            )
            if math.isfinite(exact_value):
                return float(exact_value)
    return raw_value


def _official_fno_actual_value_from_row(row: Mapping[str, Any]) -> float:
    source_summary_json = str(row.get("source_summary_json", "") or "").strip()
    comparison_mode = str(row.get("comparison_mode", "") or "").strip()
    is_one_leaf = int(_effective_fixed_leaf_tokens(row)) == 128 and int(_effective_leaves_per_doc(row)) == 1
    if source_summary_json and comparison_mode == "exact_collapse" and is_one_leaf:
        sibling_path = Path(source_summary_json)
        sibling_text = str(sibling_path)
        if "__tree_neural__" in sibling_text:
            sibling_text = sibling_text.replace("__tree_neural__", "__fno__")
            sibling_summary = _raw_source_summary(sibling_text)
            aggregate_rows = list(sibling_summary.get("aggregate_rows") or [])
            if aggregate_rows:
                sibling_value = _safe_float(
                    dict(aggregate_rows[0]).get("test_root_mae_mean"),
                    float("nan"),
                )
                if math.isfinite(sibling_value):
                    return float(sibling_value)
    family_rows = dict(row.get("fno_family_rows") or {})
    return _safe_float(
        dict(family_rows.get("official_fno") or {}).get("test_root_mae"),
        float("nan"),
    )


def _best_full_data_fno_value_from_row(row: Mapping[str, Any]) -> float:
    value = _safe_float(row.get("canonical_official_fno_full100_test_root_mae"), float("nan"))
    if math.isfinite(value):
        return value
    if str(row.get("full100_fno_family", "") or "").strip() == "official_fno":
        value = _safe_float(row.get("full100_fno_test_root_mae"), float("nan"))
        if math.isfinite(value):
            return value
    if str(row.get("best_full100_fno_family", "") or "").strip() == "official_fno":
        value = _safe_float(row.get("best_full100_fno_test_root_mae"), float("nan"))
        if math.isfinite(value):
            return value
    return float("nan")


def _best_full_data_fno_family_from_row(row: Mapping[str, Any]) -> str:
    value = str(row.get("canonical_official_fno_full100_family", "") or "").strip()
    if value:
        return value
    if str(row.get("full100_fno_family", "") or "").strip() == "official_fno":
        return "official_fno"
    if str(row.get("best_full100_fno_family", "") or "").strip() == "official_fno":
        return "official_fno"
    return "official_fno"


def _rows_for_scope_and_train_docs(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    train_doc_count: int,
) -> List[Dict[str, Any]]:
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payload = dict((scope.get("rows_by_train_docs") or {}).get(str(int(train_doc_count))) or {})
    return [dict(row) for row in list(payload.get("rows") or [])]


def _is_root_only_row(row: Mapping[str, Any]) -> bool:
    package_semantics = str(row.get("package_semantics", "") or "").strip()
    if package_semantics and package_semantics != "full_doc_only":
        return False
    leaf_mass = _safe_float(row.get("tree_computed_leaf_mass_per_doc"), float("nan"))
    internal_mass = _safe_float(
        row.get("tree_computed_internal_mass_per_doc"),
        float("nan"),
    )
    local_mass = _safe_float(row.get("tree_computed_local_mass_per_doc"), float("nan"))
    if math.isfinite(local_mass) and abs(local_mass) > 1e-9:
        return False
    if math.isfinite(leaf_mass) and abs(leaf_mass) > 1e-9:
        return False
    if math.isfinite(internal_mass) and abs(internal_mass) > 1e-9:
        return False
    return True


def _bundle_rank_for_leaf(row: Mapping[str, Any], leaf_tokens: int) -> int:
    bundle_name = str(row.get("source_bundle_name", "") or "").strip()
    preference = list(PRIMARY_ROOT_ONLY_BUNDLE_PREFERENCE_BY_LEAF.get(int(leaf_tokens), []))
    if bundle_name in preference:
        return preference.index(bundle_name)
    return len(preference) + 100


def _is_publication_row(row: Mapping[str, Any]) -> bool:
    tier = str(row.get("source_tier_label", "") or "").strip()
    return tier in PUBLICATION_TIERS


def _is_one_leaf_canary_row(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("comparison_mode", "") or "").strip() == "exact_collapse"
        and int(_effective_leaves_per_doc(row)) == 1
    )


def _is_one_leaf_local_law_row(row: Mapping[str, Any]) -> bool:
    return bool(
        int(_effective_fixed_leaf_tokens(row)) == 128
        and int(_effective_leaves_per_doc(row)) == 1
        and str(row.get("comparison_mode", "") or "").strip() != "exact_collapse"
        and "leaf_full100_internal_count100" in str(row.get("package_name", "") or "")
    )


def _effective_reviewed_docs(train_doc_count: int, root_share: int) -> int:
    return int(round(float(train_doc_count) * float(root_share) / 100.0))


def _wrapped_scope_subtitle(text: str, *, width: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    return textwrap.fill(normalized, width=width)


def _canonical_local_law_package(root_share: int) -> str:
    return f"full{int(root_share)}_leaf_full100_internal_count100"


def _lineage_rank(row: Mapping[str, Any], preferred_tokens: Sequence[str]) -> int:
    lineage = " ".join(
        (
            str(row.get("source_lineage_label", "") or ""),
            str(row.get("source_attempt_lineage", "") or ""),
            str(row.get("source_root_prefix", "") or ""),
        )
    ).lower()
    for idx, token in enumerate(preferred_tokens):
        if str(token).strip().lower() in lineage:
            return idx
    return len(preferred_tokens) + 100


def _select_primary_root_only_leaf_row(
    candidates: Sequence[Mapping[str, Any]],
    *,
    leaf_tokens: int,
) -> Dict[str, Any]:
    normalized = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not normalized:
        return {}
    root_only = [row for row in normalized if _is_root_only_row(row)]
    preferred_pool = root_only or normalized
    preference = list(PRIMARY_ROOT_ONLY_BUNDLE_PREFERENCE_BY_LEAF.get(int(leaf_tokens), []))
    allowed = [
        row
        for row in preferred_pool
        if str(row.get("source_bundle_name", "") or "").strip() in preference
    ]
    ranked_pool = allowed or preferred_pool

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        return (
            -1 * _bundle_rank_for_leaf(row, int(leaf_tokens)),
            -1 if _is_publication_row(row) else 0,
            -1
            * (
                1
                if (
                    int(leaf_tokens) == 128
                    and str(row.get("comparison_mode", "") or "").strip() == "exact_collapse"
                )
                else 0
            ),
            -1 * _lineage_rank(row, PRIMARY_ONELEAF_LINEAGE_TOKENS),
            int(_safe_int(row.get("source_tier_rank"), 0)),
            1 if bool(row.get("contract_headline_eligible")) else 0,
            str(row.get("source_summary_json", "") or ""),
        )

    return max(ranked_pool, key=_sort_key)


def _select_alternate_root_only_leaf_row(
    candidates: Sequence[Mapping[str, Any]],
    *,
    leaf_tokens: int,
) -> Dict[str, Any]:
    normalized = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not normalized:
        return {}
    root_only = [row for row in normalized if _is_root_only_row(row)]
    preferred_pool = root_only or normalized
    preference = list(ALTERNATE_ROOT_ONLY_BUNDLE_PREFERENCE_BY_LEAF.get(int(leaf_tokens), []))
    allowed = [
        row
        for row in preferred_pool
        if str(row.get("source_bundle_name", "") or "").strip() in preference
    ]
    ranked_pool = allowed or []
    if not ranked_pool:
        return {}

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        bundle_name = str(row.get("source_bundle_name", "") or "").strip()
        bundle_rank = preference.index(bundle_name) if bundle_name in preference else len(preference) + 100
        return (
            -1 * bundle_rank,
            -1 * _lineage_rank(row, ALTERNATE_ROOT_ONLY_LINEAGE_TOKENS),
            int(_safe_int(row.get("source_tier_rank"), 0)),
            str(row.get("source_summary_json", "") or ""),
        )

    return max(ranked_pool, key=_sort_key)


def _select_oneleaf_local_law_row(
    candidates: Sequence[Mapping[str, Any]],
    *,
    root_share: int,
) -> Dict[str, Any]:
    normalized = [
        dict(row)
        for row in candidates
        if isinstance(row, Mapping)
        and _is_one_leaf_local_law_row(row)
        and str(row.get("package_name", "") or "").strip()
        == _canonical_local_law_package(int(root_share))
    ]
    if not normalized:
        return {}

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        bundle_name = str(row.get("source_bundle_name", "") or "").strip()
        bundle_rank = (
            ONELEAF_LOCAL_LAW_BUNDLE_PREFERENCE.index(bundle_name)
            if bundle_name in ONELEAF_LOCAL_LAW_BUNDLE_PREFERENCE
            else len(ONELEAF_LOCAL_LAW_BUNDLE_PREFERENCE) + 100
        )
        return (
            -1 * bundle_rank,
            -1 * _lineage_rank(row, ONELEAF_LOCAL_LAW_LINEAGE_TOKENS),
            int(_safe_int(row.get("source_tier_rank"), 0)),
            str(row.get("source_summary_json", "") or ""),
        )

    return max(normalized, key=_sort_key)


def _select_full_root_oneleaf_tree_row(
    candidates: Sequence[Mapping[str, Any]],
    *,
    preferred_scope_order: Sequence[str] = (),
) -> Dict[str, Any]:
    normalized = [
        dict(row)
        for row in candidates
        if isinstance(row, Mapping)
        and str(row.get("baseline_family", "") or "").strip() == "tree_neural"
        and str(row.get("package_name", "") or "").strip() == "full100"
        and int(_effective_fixed_leaf_tokens(row)) == 128
        and int(_effective_leaves_per_doc(row)) == 1
        and _is_root_only_row(row)
    ]
    if preferred_scope_order:
        for preferred_scope_key in preferred_scope_order:
            narrowed = [
                dict(row)
                for row in normalized
                if str(row.get("scope_key", "") or "").strip() == str(preferred_scope_key)
            ]
            if narrowed:
                normalized = narrowed
                break
    if not normalized:
        return {}
    return _select_primary_root_only_leaf_row(normalized, leaf_tokens=128)


def _best_root_only_tree_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    alternate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_leaf: Dict[int, List[Dict[str, Any]]] = {}
    for row in list(primary_rows) + list(alternate_rows):
        if not isinstance(row, Mapping):
            continue
        leaf_tokens = int(_effective_fixed_leaf_tokens(row))
        if leaf_tokens <= 0:
            continue
        by_leaf.setdefault(leaf_tokens, []).append(dict(row))
    out: List[Dict[str, Any]] = []
    for leaf_tokens in sorted(by_leaf.keys(), reverse=True):
        candidates = [dict(row) for row in by_leaf.get(leaf_tokens, [])]
        if not candidates:
            continue
        if int(leaf_tokens) == 128:
            oneleaf = _selected_leaf128_root_only_row(candidates)
            if oneleaf:
                out.append(dict(oneleaf))
                continue
        best = min(
            candidates,
            key=lambda row: (
                _plot_tree_root_mae_from_row(row)
                if math.isfinite(_plot_tree_root_mae_from_row(row))
                else float("inf"),
                -1 * _lineage_rank(row, ALTERNATE_ROOT_ONLY_LINEAGE_TOKENS),
                str(row.get("source_summary_json", "") or ""),
            ),
        )
        out.append(dict(best))
    return out


def _selected_leaf128_root_only_row(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not normalized:
        return {}
    exact_oneleaf = [row for row in normalized if _is_one_leaf_canary_row(row)]
    if exact_oneleaf:
        return dict(_select_primary_root_only_leaf_row(exact_oneleaf, leaf_tokens=128))
    leaf128_rows = [
        row
        for row in normalized
        if int(_effective_fixed_leaf_tokens(row)) == 128 and _is_root_only_row(row)
    ]
    if leaf128_rows:
        return dict(_select_primary_root_only_leaf_row(leaf128_rows, leaf_tokens=128))
    return {}


def _secondary_tree_package_name(root_share: int, series_key: str) -> str:
    normalized = str(series_key).strip().lower()
    if normalized == "leaf_mass_eq":
        return _leaf_mass_preserving_package_name(int(root_share))
    if normalized == "depth_equal_mass_eq":
        return _depth_equal_mass_preserving_package_name(int(root_share))
    raise ValueError(f"Unsupported secondary tree series: {series_key!r}")


def _select_secondary_tree_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    root_share: int,
    tree_family: str,
    secondary_series_key: str,
) -> List[Dict[str, Any]]:
    package_name = _secondary_tree_package_name(int(root_share), secondary_series_key)
    candidates = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("baseline_family", "") or "").strip() == str(tree_family)
        and str(row.get("package_name", "") or "").strip() == package_name
        and int(_effective_fixed_leaf_tokens(row)) > 0
    ]
    grouped_by_leaf: Dict[int, List[Dict[str, Any]]] = {}
    for row in candidates:
        grouped_by_leaf.setdefault(int(_effective_fixed_leaf_tokens(row)), []).append(dict(row))
    selected: List[Dict[str, Any]] = []
    for leaf_tokens in sorted(grouped_by_leaf.keys(), reverse=True):
        leaf_candidates = [dict(row) for row in grouped_by_leaf.get(leaf_tokens, [])]
        if not leaf_candidates:
            continue
        best = min(
            leaf_candidates,
            key=lambda row: (
                _plot_tree_root_mae_from_row(row)
                if math.isfinite(_plot_tree_root_mae_from_row(row))
                else float("inf"),
                int(_safe_int(row.get("source_tier_rank"), 0)),
                str(row.get("source_summary_json", "") or ""),
            ),
        )
        selected.append(dict(best))
    return selected


def _select_official_fno_leaf128_row(
    candidates: Sequence[Mapping[str, Any]],
    *,
    root_share: int,
    preferred_scope_order: Sequence[str] = (),
) -> Dict[str, Any]:
    package_name = _root_only_package(int(root_share))
    normalized = [
        dict(row)
        for row in candidates
        if isinstance(row, Mapping)
        and str(row.get("baseline_family", "") or "").strip() == "official_fno"
        and str(row.get("package_name", "") or "").strip() == package_name
        and int(_effective_fixed_leaf_tokens(row)) == 128
    ]
    if preferred_scope_order:
        for preferred_scope_key in preferred_scope_order:
            narrowed = [
                dict(row)
                for row in normalized
                if str(row.get("scope_key", "") or "").strip() == str(preferred_scope_key)
            ]
            if narrowed:
                normalized = narrowed
                break
    if not normalized:
        return {}

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        bundle_name = str(row.get("source_bundle_name", "") or "").strip()
        bundle_rank = _bundle_rank_for_leaf(row, 128)
        if bundle_name == "oneleaf_root_budget_publication_xlarge":
            bundle_rank = min(bundle_rank, 1)
        return (
            -1 * bundle_rank,
            -1 * _lineage_rank(row, PRIMARY_ONELEAF_LINEAGE_TOKENS),
            int(_safe_int(row.get("source_tier_rank"), 0)),
            str(row.get("source_summary_json", "") or ""),
        )

    return max(normalized, key=_sort_key)


def _selected_full_root_official_fno_anchor_rows(
    recovery: Mapping[str, Any],
    *,
    scope_keys: Sequence[str],
    scope_preference_by_leaf: Mapping[int, Sequence[str]] | None = None,
) -> List[Dict[str, Any]]:
    rows_by_scope_and_docs: Dict[int, List[Dict[str, Any]]] = {}
    for scope_key in scope_keys:
        scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
        for train_doc_key, payload in dict(scope.get("rows_by_train_docs") or {}).items():
            train_doc_count = int(_safe_int(train_doc_key, 0))
            if train_doc_count <= 0:
                continue
            rows_by_scope_and_docs.setdefault(train_doc_count, []).extend(
                [dict(row) for row in list(dict(payload or {}).get("rows") or [])]
            )
    preferred_scope_order = tuple(
        scope_preference_by_leaf.get(128, ())
    ) if scope_preference_by_leaf else ()
    anchors: List[Dict[str, Any]] = []
    for train_doc_count in sorted(rows_by_scope_and_docs.keys()):
        selected = _select_full_root_oneleaf_tree_row(
            rows_by_scope_and_docs.get(train_doc_count, []),
            preferred_scope_order=preferred_scope_order,
        )
        if not selected:
            continue
        mae = _official_fno_actual_value_from_row(selected)
        if not math.isfinite(mae):
            continue
        anchors.append(
            {
                "train_doc_count": int(train_doc_count),
                "official_fno_root_mae": float(mae),
                "source_lineage_label": str(
                    selected.get("source_lineage_label", "") or ""
                ).strip(),
            }
        )
    return anchors


def _interpolate_official_fno_at_train_docs(
    anchors: Sequence[Mapping[str, Any]],
    *,
    target_train_docs: float,
) -> Dict[str, Any]:
    target = float(target_train_docs)
    points = sorted(
        [
            (
                int(_safe_int(row.get("train_doc_count"), 0)),
                float(_safe_float(row.get("official_fno_root_mae"), float("nan"))),
            )
            for row in anchors
            if int(_safe_int(row.get("train_doc_count"), 0)) > 0
            and math.isfinite(_safe_float(row.get("official_fno_root_mae"), float("nan")))
        ],
        key=lambda item: item[0],
    )
    if not points or not math.isfinite(target):
        return {"value": float("nan"), "relation": "", "min_train_docs": 0, "max_train_docs": 0}
    for docs, value in points:
        if int(docs) == int(round(target)):
            return {
                "value": float(value),
                "relation": "exact",
                "min_train_docs": int(docs),
                "max_train_docs": int(docs),
            }
    if target < float(points[0][0]) or target > float(points[-1][0]):
        return {
            "value": float("nan"),
            "relation": "out_of_range",
            "min_train_docs": int(points[0][0]),
            "max_train_docs": int(points[-1][0]),
        }
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if float(x0) <= target <= float(x1):
            if int(x0) == int(x1):
                value = float(y0)
            else:
                alpha = (target - float(x0)) / float(max(1e-12, float(x1 - x0)))
                value = float(y0 + alpha * (y1 - y0))
            return {
                "value": float(value),
                "relation": "interpolated",
                "min_train_docs": int(x0),
                "max_train_docs": int(x1),
            }
    return {"value": float("nan"), "relation": "", "min_train_docs": 0, "max_train_docs": 0}


def _leaf_size_panel_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_keys: Sequence[str],
    train_doc_count: int,
    root_shares: Sequence[int],
    scope_preference_by_leaf: Mapping[int, Sequence[str]] | None = None,
    secondary_tree_series_key: str | None = None,
    empirical_bayes_mode: str = "off",
) -> List[Dict[str, Any]]:
    rows_by_scope = {
        str(scope_key): _rows_for_scope_and_train_docs(
            recovery,
            scope_key=str(scope_key),
            train_doc_count=int(train_doc_count),
        )
        for scope_key in scope_keys
    }
    rows = [
        dict(row)
        for scope_key in scope_keys
        for row in rows_by_scope.get(str(scope_key), [])
    ]
    if not rows:
        return []
    tree_family = str(recovery.get("tree_family", "tree_neural") or "tree_neural")
    out: List[Dict[str, Any]] = []
    global_preferred_scope_order = tuple(
        scope_preference_by_leaf.get(128, ())
    ) if scope_preference_by_leaf else ()
    full_root_official_fno_anchor_rows = _selected_full_root_official_fno_anchor_rows(
        recovery,
        scope_keys=scope_keys,
        scope_preference_by_leaf=scope_preference_by_leaf,
    )
    full_root_oneleaf_tree_row = _select_full_root_oneleaf_tree_row(
        rows,
        preferred_scope_order=global_preferred_scope_order,
    )
    full_root_official_fno_root_mae = _official_fno_actual_value_from_row(
        full_root_oneleaf_tree_row
    )
    for root_share in root_shares:
        package_name = _root_only_package(int(root_share))
        package_tree_rows = [
            row
            for row in rows
            if str(row.get("baseline_family", "") or "") == tree_family
            and str(row.get("package_name", "") or "") == package_name
            and int(_effective_fixed_leaf_tokens(row)) > 0
            and _is_root_only_row(row)
        ]
        grouped_by_leaf_tokens: Dict[int, List[Dict[str, Any]]] = {}
        for row in package_tree_rows:
            grouped_by_leaf_tokens.setdefault(int(_effective_fixed_leaf_tokens(row)), []).append(dict(row))
        primary_rows: List[Dict[str, Any]] = []
        alternate_rows: List[Dict[str, Any]] = []
        for leaf_tokens, candidates in grouped_by_leaf_tokens.items():
            if int(leaf_tokens) <= 0:
                continue
            preferred_scope_order = tuple(
                scope_preference_by_leaf.get(int(leaf_tokens), ())
            ) if scope_preference_by_leaf else ()
            narrowed_candidates = list(candidates)
            for preferred_scope_key in preferred_scope_order:
                scope_specific = [
                    dict(row)
                    for row in narrowed_candidates
                    if str(row.get("scope_key", "") or "").strip() == str(preferred_scope_key)
                ]
                if scope_specific:
                    narrowed_candidates = scope_specific
                    break
            primary_rows.append(
                dict(
                    _select_primary_root_only_leaf_row(
                        narrowed_candidates,
                        leaf_tokens=int(leaf_tokens),
                    )
                )
            )
            alternate_row = _select_alternate_root_only_leaf_row(
                narrowed_candidates,
                leaf_tokens=int(leaf_tokens),
            )
            if alternate_row:
                alternate_rows.append(dict(alternate_row))
        primary_rows = [row for row in primary_rows if bool(row)]
        alternate_rows = [row for row in alternate_rows if bool(row)]
        primary_rows = _best_root_only_tree_rows(primary_rows, alternate_rows)
        alternate_rows = []
        primary_rows.sort(
            key=lambda row: (
                -int(_effective_fixed_leaf_tokens(row)),
                int(_effective_leaves_per_doc(row)) or 10**9,
            )
        )
        alternate_rows.sort(
            key=lambda row: (
                -int(_effective_fixed_leaf_tokens(row)),
                int(_effective_leaves_per_doc(row)) or 10**9,
            )
        )
        primary_leaf_tokens = [
            int(_effective_fixed_leaf_tokens(row))
            for row in primary_rows
            if int(_effective_fixed_leaf_tokens(row)) > 0
        ]
        alternate_leaf_tokens = [
            int(_effective_fixed_leaf_tokens(row))
            for row in alternate_rows
            if int(_effective_fixed_leaf_tokens(row)) > 0
        ]
        show_alternate_rows = bool(
            primary_leaf_tokens
            and alternate_leaf_tokens
            and alternate_leaf_tokens == primary_leaf_tokens
        )
        exemplar_candidates = [
            row
            for row in primary_rows
            if bool(row)
        ] or package_tree_rows
        exemplar_row = (
            dict(
                _select_primary_root_only_leaf_row(
                    exemplar_candidates,
                    leaf_tokens=int(
                        max(
                            (
                                int(_effective_fixed_leaf_tokens(row))
                                for row in exemplar_candidates
                                if int(_effective_fixed_leaf_tokens(row)) > 0
                            ),
                            default=128,
                        )
                    ),
                )
            )
            if exemplar_candidates
            else {}
        )
        leaf128_selected_row = _selected_leaf128_root_only_row(primary_rows)
        oneleaf_local_law_row = _select_oneleaf_local_law_row(rows, root_share=int(root_share))
        empirical_mean_baseline = _empirical_mean_baseline_from_row(
            leaf128_selected_row or exemplar_row
        )
        empirical_bayes_baseline = (
            _empirical_bayes_baseline_from_row(leaf128_selected_row or exemplar_row)
            if str(empirical_bayes_mode).strip().lower() == "collapsed_hmm"
            else {}
        )
        secondary_rows: List[Dict[str, Any]] = []
        if secondary_tree_series_key:
            if int(root_share) == 100:
                secondary_rows = [dict(row) for row in primary_rows]
            else:
                secondary_rows = _select_secondary_tree_rows(
                    rows,
                    root_share=int(root_share),
                    tree_family=tree_family,
                    secondary_series_key=str(secondary_tree_series_key),
                )
        out.append(
            {
                "root_share": int(root_share),
                "package_name": package_name,
                "primary_rows": primary_rows,
                "alternate_rows": alternate_rows if show_alternate_rows else [],
                "secondary_rows": secondary_rows,
                "oneleaf_local_law_row": dict(oneleaf_local_law_row),
                "official_fno_actual_root_mae": _safe_float(
                    _official_fno_actual_value_from_row(leaf128_selected_row),
                    _official_fno_actual_value_from_row(exemplar_row),
                ),
                "official_fno_actual_source": str(
                    dict(leaf128_selected_row).get("source_lineage_label", "") or ""
                ).strip(),
                "best_full_data_fno_root_mae": _safe_float(
                    full_root_official_fno_root_mae,
                    _best_full_data_fno_value_from_row(exemplar_row),
                ),
                "empirical_mean_guess_root_mae": _safe_float(
                    empirical_mean_baseline.get("test_root_mae"),
                    float("nan"),
                ),
                "empirical_bayes_root_mae": _safe_float(
                    empirical_bayes_baseline.get("test_root_mae"),
                    float("nan"),
                ),
                "empirical_bayes_train_switch_prob_hat": _safe_float(
                    empirical_bayes_baseline.get("train_switch_prob_hat"),
                    float("nan"),
                ),
                "empirical_bayes_reviewed_root_docs": int(
                    _safe_int(empirical_bayes_baseline.get("reviewed_root_docs"), 0)
                ),
                "empirical_mean_guess_train_mean_root_count": _safe_float(
                    empirical_mean_baseline.get("train_mean_root_count"),
                    float("nan"),
                ),
                "empirical_mean_guess_reviewed_root_docs": int(
                    _safe_int(empirical_mean_baseline.get("reviewed_root_docs"), 0)
                ),
                "best_full_data_fno_family": (
                    "official_fno"
                    if math.isfinite(full_root_official_fno_root_mae)
                    else _best_full_data_fno_family_from_row(exemplar_row)
                ),
                "equivalent_budget_train_docs": float(
                    _effective_reviewed_docs(int(train_doc_count), int(root_share))
                ),
                "equivalent_budget_official_fno": _interpolate_official_fno_at_train_docs(
                    full_root_official_fno_anchor_rows,
                    target_train_docs=float(
                        _effective_reviewed_docs(int(train_doc_count), int(root_share))
                    ),
                ),
                "lineage_label": str(exemplar_row.get("source_lineage_label", "") or "").strip(),
            }
        )
    return out


def _plot_fixed_train_leaf_size_root_only(
    recovery: Mapping[str, Any],
    *,
    scope_keys: Sequence[str],
    primary_scope_key: str,
    train_doc_count: int,
    root_shares: Sequence[int],
    output_path: Path,
    scope_preference_by_leaf: Mapping[int, Sequence[str]] | None = None,
    secondary_tree_series_key: str | None = None,
    empirical_bayes_mode: str = "off",
) -> Dict[str, Any]:
    payloads = _leaf_size_panel_payloads(
        recovery,
        scope_keys=scope_keys,
        train_doc_count=train_doc_count,
        root_shares=root_shares,
        scope_preference_by_leaf=scope_preference_by_leaf,
        secondary_tree_series_key=secondary_tree_series_key,
        empirical_bayes_mode=empirical_bayes_mode,
    )
    if not payloads:
        return {"rendered": False, "reason": "no rows"}
    n_panels = len(payloads)
    ncols = min(4, max(1, n_panels))
    nrows = int(math.ceil(float(n_panels) / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10.5, 4.3 * ncols), max(4.2, 3.8 * nrows)),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes.ravel())
    rendered_panels = 0
    panel_summaries: List[Dict[str, Any]] = []
    emitted_labels: Dict[str, bool] = {}
    secondary_series_config = dict(
        SECONDARY_TREE_SERIES_CONFIG.get(str(secondary_tree_series_key or ""), {}) or {}
    )
    secondary_legend_label = str(secondary_series_config.get("legend_label", "") or "").strip()
    secondary_caption_label = str(secondary_series_config.get("caption_label", "") or "").strip()
    for ax, payload in zip(axes_list, payloads):
        root_share = int(_safe_int(payload.get("root_share"), 0))
        primary_rows = [dict(row) for row in list(payload.get("primary_rows") or [])]
        alternate_rows = [dict(row) for row in list(payload.get("alternate_rows") or [])]
        secondary_rows = [dict(row) for row in list(payload.get("secondary_rows") or [])]
        rows = list(primary_rows)
        x = list(range(len(rows)))
        tree_y = [_plot_tree_root_mae_from_row(row) for row in rows]
        if rows and any(math.isfinite(value) for value in tree_y):
            rendered_panels += 1
            tick_labels = [
                f"{int(_effective_fixed_leaf_tokens(row))}\n({int(_effective_leaves_per_doc(row))})"
                for row in rows
            ]
            canary_idx = [
                idx for idx, row in enumerate(rows) if _is_one_leaf_canary_row(row)
            ]
            ax.plot(
                x,
                tree_y,
                color=TREE_PRIMARY_COLOR,
                linewidth=2.3,
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=(
                    LABEL_TREE_ROOT_ONLY
                    if not emitted_labels.get("tree_root_only")
                    else None
                ),
            )
            emitted_labels["tree_root_only"] = True
            if secondary_rows:
                x_by_leaf = {
                    int(_effective_fixed_leaf_tokens(row)): idx
                    for idx, row in enumerate(rows)
                    if int(_effective_fixed_leaf_tokens(row)) > 0
                }
                secondary_points = [
                    (
                        x_by_leaf[int(_effective_fixed_leaf_tokens(row))],
                        _plot_tree_root_mae_from_row(row),
                        int(_effective_fixed_leaf_tokens(row)),
                        dict(row),
                    )
                    for row in secondary_rows
                    if int(_effective_fixed_leaf_tokens(row)) in x_by_leaf
                    and math.isfinite(_plot_tree_root_mae_from_row(row))
                ]
                if secondary_points:
                    ax.plot(
                        [item[0] for item in secondary_points],
                        [item[1] for item in secondary_points],
                        color=MASS_EQUIVALENT_COLOR,
                        linewidth=2.1,
                        linestyle="--",
                        marker="s",
                        markersize=5.0,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=(
                            secondary_legend_label
                            if secondary_legend_label
                            and not emitted_labels.get("secondary_tree_series")
                            else None
                        ),
                    )
                    emitted_labels["secondary_tree_series"] = True
                    secondary_leaf128 = next(
                        (
                            item
                            for item in secondary_points
                            if int(item[2]) == 128
                        ),
                        None,
                    )
                    if secondary_leaf128 is not None:
                        ax.scatter(
                            [secondary_leaf128[0]],
                            [secondary_leaf128[1]],
                            color=MASS_EQUIVALENT_COLOR,
                            edgecolor="black",
                            linewidth=1.0,
                            marker="s",
                            s=74,
                            zorder=7,
                            label=(
                                LABEL_TREE_DUPLICATE_COUNT_ONLY
                                if not emitted_labels.get("tree_oneleaf_duplicate_count_only")
                                else None
                            ),
                        )
                        emitted_labels["tree_oneleaf_duplicate_count_only"] = True
            if canary_idx:
                ax.scatter(
                    [x[idx] for idx in canary_idx],
                    [tree_y[idx] for idx in canary_idx],
                    facecolors="none",
                    edgecolor="black",
                    linewidth=1.0,
                        marker="D",
                        s=68,
                        zorder=6,
                        label=(
                            LABEL_TREE_NO_LOCAL_LEAF128
                            if not emitted_labels.get("tree_oneleaf_no_local")
                            else None
                        ),
                )
                emitted_labels["tree_oneleaf_no_local"] = True
            oneleaf_local_law_row = dict(payload.get("oneleaf_local_law_row") or {})
            local_law_y = _plot_tree_root_mae_from_row(oneleaf_local_law_row)
            if oneleaf_local_law_row and math.isfinite(local_law_y):
                local_law_leaf = int(_effective_fixed_leaf_tokens(oneleaf_local_law_row))
                local_law_x = next(
                    (
                        idx
                        for idx, row in enumerate(rows)
                        if int(_effective_fixed_leaf_tokens(row)) == local_law_leaf
                    ),
                    None,
                )
                if local_law_x is not None:
                    ax.scatter(
                        [local_law_x],
                        [local_law_y],
                        color=NEUTRAL_COLOR,
                        edgecolor="black",
                        linewidth=0.9,
                        marker="^",
                        s=68,
                        zorder=7,
                        label=(
                            LABEL_TREE_DUPLICATE_LOCAL
                            if not emitted_labels.get("tree_oneleaf_duplicate_local")
                            else None
                        ),
                    )
                    emitted_labels["tree_oneleaf_duplicate_local"] = True
            best_full_data_fno = _safe_float(
                payload.get("best_full_data_fno_root_mae"),
                float("nan"),
            )
            official_fno_actual = _safe_float(
                payload.get("official_fno_actual_root_mae"),
                float("nan"),
            )
            empirical_mean_guess = _safe_float(
                payload.get("empirical_mean_guess_root_mae"),
                float("nan"),
            )
            empirical_bayes_guess = _safe_float(
                payload.get("empirical_bayes_root_mae"),
                float("nan"),
            )
            if math.isfinite(official_fno_actual):
                ax.plot(
                    x,
                    [official_fno_actual] * len(x),
                    color=FNO_EQUIVALENT_COLOR,
                    linestyle=":",
                    linewidth=2.0,
                    label=(
                        LABEL_FNO_SAME_BUDGET
                        if not emitted_labels.get("fno_equivalent_budget")
                        else None
                    ),
                )
                emitted_labels["fno_equivalent_budget"] = True
            if (
                math.isfinite(empirical_mean_guess)
                and str(empirical_bayes_mode).strip().lower() != "collapsed_hmm"
            ):
                ax.plot(
                    x,
                    [empirical_mean_guess] * len(x),
                    color=EMPIRICAL_MEAN_BASELINE_COLOR,
                    linestyle="-.",
                    linewidth=1.9,
                    label=(
                        LABEL_EMPIRICAL_MEAN
                        if not emitted_labels.get("empirical_mean_guess")
                        else None
                    ),
                )
                emitted_labels["empirical_mean_guess"] = True
            if math.isfinite(empirical_bayes_guess):
                ax.plot(
                    x,
                    [empirical_bayes_guess] * len(x),
                    color=EMPIRICAL_BAYES_BASELINE_COLOR,
                    linestyle=(0, (5, 2)),
                    linewidth=1.9,
                    label=(
                        LABEL_EMPIRICAL_BAYES_LIMIT
                        if not emitted_labels.get("empirical_bayes_guess")
                        else None
                    ),
                )
                emitted_labels["empirical_bayes_guess"] = True
            if math.isfinite(official_fno_actual):
                leaf128_idx = next(
                    (
                        idx
                        for idx, row in enumerate(rows)
                        if int(_effective_fixed_leaf_tokens(row)) == 128
                    ),
                    None,
                )
                if leaf128_idx is not None:
                    ax.scatter(
                        [leaf128_idx],
                        [official_fno_actual],
                        color=FNO_OFFICIAL_COLOR,
                        edgecolor="white",
                        linewidth=0.8,
                        marker="X",
                        s=82,
                        zorder=7,
                        label=(
                            "Official FNO actual point at leaf128"
                            if not emitted_labels.get("fno_leaf128_actual")
                            else None
                        ),
                    )
                    emitted_labels["fno_leaf128_actual"] = True
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Leaf tokens\n(leaves/doc)")
            ax.grid(True, alpha=0.25)
            y_formatter = ScalarFormatter(useMathText=True)
            y_formatter.set_scientific(True)
            y_formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(y_formatter)
            best_idx = min(
                range(len(tree_y)),
                key=lambda idx: tree_y[idx] if math.isfinite(tree_y[idx]) else float("inf"),
            )
            if math.isfinite(tree_y[best_idx]) and best_idx not in canary_idx:
                ax.scatter(
                    [best_idx],
                    [tree_y[best_idx]],
                    color=TREE_PRIMARY_COLOR,
                    edgecolor="white",
                    linewidth=0.8,
                    s=60,
                    zorder=5,
                )
            panel_summaries.append(
                {
                    "root_share": root_share,
                    "available_leaf_tokens": [int(_effective_fixed_leaf_tokens(row)) for row in rows],
                    "tree_root_mae_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): float(_plot_tree_root_mae_from_row(row))
                        for row in rows
                        if math.isfinite(_plot_tree_root_mae_from_row(row))
                    },
                    "tree_source_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): str(
                            row.get("source_lineage_label", "") or ""
                        ).strip()
                        for row in rows
                    },
                    "tree_source_tier_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): str(
                            row.get("source_tier_label", "") or ""
                        ).strip()
                        for row in rows
                    },
                    "secondary_tree_series_key": str(secondary_tree_series_key or ""),
                    "secondary_tree_root_mae_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): float(_plot_tree_root_mae_from_row(row))
                        for row in secondary_rows
                        if math.isfinite(_plot_tree_root_mae_from_row(row))
                    },
                    "secondary_tree_source_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): str(
                            row.get("source_lineage_label", "") or ""
                        ).strip()
                        for row in secondary_rows
                    },
                    "one_leaf_duplicate_count_only_root_mae": (
                        float(
                            next(
                                (
                                    _plot_tree_root_mae_from_row(row)
                                    for row in secondary_rows
                                    if int(_effective_fixed_leaf_tokens(row)) == 128
                                    and math.isfinite(_plot_tree_root_mae_from_row(row))
                                ),
                                float("nan"),
                            )
                        )
                        if any(
                            int(_effective_fixed_leaf_tokens(row)) == 128
                            and math.isfinite(_plot_tree_root_mae_from_row(row))
                            for row in secondary_rows
                        )
                        else None
                    ),
                    "one_leaf_duplicate_count_only_source": str(
                        next(
                            (
                                str(row.get("source_lineage_label", "") or "").strip()
                                for row in secondary_rows
                                if int(_effective_fixed_leaf_tokens(row)) == 128
                            ),
                            "",
                        )
                    ),
                    "alternate_tree_root_mae_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): float(_plot_tree_root_mae_from_row(row))
                        for row in alternate_rows
                        if math.isfinite(_plot_tree_root_mae_from_row(row))
                    },
                    "alternate_tree_source_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(row))): str(
                            row.get("source_lineage_label", "") or ""
                        ).strip()
                        for row in alternate_rows
                    },
                    "one_leaf_canary_leaf_tokens": [
                        int(_effective_fixed_leaf_tokens(rows[idx]))
                        for idx in canary_idx
                    ],
                    "one_leaf_canary_root_mae_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(rows[idx]))): float(tree_y[idx])
                        for idx in canary_idx
                        if math.isfinite(tree_y[idx])
                    },
                    "one_leaf_canary_source_by_leaf_tokens": {
                        str(int(_effective_fixed_leaf_tokens(rows[idx]))): str(
                            rows[idx].get("source_lineage_label", "") or ""
                        ).strip()
                        for idx in canary_idx
                    },
                    "one_leaf_duplicate_local_label_root_mae": (
                        float(local_law_y) if math.isfinite(local_law_y) else None
                    ),
                    "one_leaf_duplicate_local_label_source": str(
                        oneleaf_local_law_row.get("source_lineage_label", "") or ""
                    ).strip(),
                    "empirical_mean_guess_root_mae": (
                        float(empirical_mean_guess)
                        if math.isfinite(empirical_mean_guess)
                        else None
                    ),
                    "empirical_bayes_root_mae": (
                        float(empirical_bayes_guess)
                        if math.isfinite(empirical_bayes_guess)
                        else None
                    ),
                    "empirical_bayes_train_switch_prob_hat": (
                        float(
                            _safe_float(
                                payload.get("empirical_bayes_train_switch_prob_hat"),
                                float("nan"),
                            )
                        )
                        if math.isfinite(
                            _safe_float(
                                payload.get("empirical_bayes_train_switch_prob_hat"),
                                float("nan"),
                            )
                        )
                        else None
                    ),
                    "empirical_bayes_reviewed_root_docs": int(
                        _safe_int(payload.get("empirical_bayes_reviewed_root_docs"), 0)
                    ),
                    "empirical_mean_guess_train_mean_root_count": (
                        float(
                            _safe_float(
                                payload.get("empirical_mean_guess_train_mean_root_count"),
                                float("nan"),
                            )
                        )
                        if math.isfinite(
                            _safe_float(
                                payload.get("empirical_mean_guess_train_mean_root_count"),
                                float("nan"),
                            )
                        )
                        else None
                    ),
                    "empirical_mean_guess_reviewed_root_docs": int(
                        _safe_int(payload.get("empirical_mean_guess_reviewed_root_docs"), 0)
                    ),
                    "official_fno_actual_root_mae": official_fno_actual,
                    "official_fno_actual_source": str(
                        payload.get("official_fno_actual_source", "") or ""
                    ).strip(),
                    "best_full_data_fno_root_mae": best_full_data_fno,
                    "best_full_data_fno_family": str(
                        payload.get("best_full_data_fno_family", "") or ""
                    ).strip(),
                    "equivalent_budget_train_docs": _safe_float(
                        payload.get("equivalent_budget_train_docs"),
                        float("nan"),
                    ),
                    "equivalent_budget_official_fno_root_mae": None,
                    "equivalent_budget_official_fno_relation": "",
                    "equivalent_budget_official_fno_min_train_docs": 0,
                    "equivalent_budget_official_fno_max_train_docs": 0,
                    "lineage_label": str(payload.get("lineage_label", "") or ""),
                }
            )
        else:
            ax.axis("off")
            ax.text(
                0.5,
                0.55,
                "Awaiting tree rows",
                ha="center",
                va="center",
                fontsize=10,
                color=NEUTRAL_COLOR,
            )
        effective_docs = _effective_reviewed_docs(int(train_doc_count), int(root_share))
        ax.set_title(
            f"R{root_share}: {effective_docs:,} root-labeled docs\n"
            f"({root_share}% of {int(train_doc_count):,} training docs)"
        )
    for ax in axes_list[len(payloads) :]:
        ax.axis("off")
    axes_list[0].set_ylabel("Test root MAE")
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()
    for ax in axes_list[: len(payloads)]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            normalized = str(label or "").strip()
            if not normalized or normalized in seen_labels:
                continue
            seen_labels.add(normalized)
            legend_handles.append(handle)
            legend_labels.append(normalized)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.125),
            ncol=min(3, len(legend_labels)),
            frameon=False,
            fontsize=9,
        )
    scope_presentation = _scope_presentation(
        recovery,
        primary_scope_key=primary_scope_key,
    )
    scope_title = str(
        scope_presentation.get(
            "title",
            _scope_label_from_recovery(recovery, primary_scope_key),
        )
    )
    scope_subtitle = str(scope_presentation.get("subtitle", "") or "")
    r100_docs = _effective_reviewed_docs(int(train_doc_count), 100)
    r10_docs = _effective_reviewed_docs(int(train_doc_count), 10)
    fig.suptitle(
        f"{scope_title}\nPerformance vs leaf size at fixed training set size ({int(train_doc_count):,} docs)",
        y=0.99,
    )
    fig.text(
        0.5,
        0.93,
        _wrapped_scope_subtitle(scope_subtitle),
        ha="center",
        va="top",
        fontsize=9,
    )
    caption_text = (
        f"Panels vary the root-supervision share: R100 means root labels on all {r100_docs:,} training documents, "
        f"while R10 means root labels on {r10_docs:,} documents. "
        "The solid tree curve uses the best available converged root-only tree surface at each leaf size. "
        + (
            f"{secondary_caption_label} "
            if secondary_caption_label
            else ""
        )
        + (
            "So, for example, at R10 the green curve uses root labels on only 10% of the training documents, "
            "while the blue curve uses that same root-label budget plus additional count-only local labels so the total supervision mass matches full100. "
            "At R100, the blue and green curves coincide by construction because there is no non-root supervision mass left to reallocate. "
            if secondary_caption_label
            else ""
        )
        + "The hollow diamond marks the tree's leaf128 point without local laws. "
        + f"The dotted amber line is official FNO trained on the same {int(train_doc_count):,} training documents and the same root-label budget as the panel, repeated across leaf sizes because the FNO baseline does not depend on leaf size. "
        + (
            "The dark teal dashed line is the empirical-Bayes limit when the DGP family is treated as known: it fits the sticky switch probability on the panel's reviewed training documents, then predicts changepoint count from observed tokens alone at test time. "
            if str(empirical_bayes_mode).strip().lower() == "collapsed_hmm"
            else "The gray dash-dot line predicts a constant equal to the empirical mean root count on the panel's reviewed root-labeled training documents, then measures MAE on the fixed test split. "
        )
        + "The red X is the official FNO point paired with that same selected leaf128 no-local-law tree run, trained on the same root-supervision budget as the panel. "
        + "When parity is correct, the leaf128 diamond and the leaf128 FNO X should coincide. "
        + "If a panel shows scientific notation such as ×10^-5 above the y-axis, that factor applies to the y-axis values in that panel. "
        + (
            "When the blue series is present, its leaf128 square highlights the one-leaf count-only duplicate-label case: "
            "the model keeps the same training set and root-label budget, then adds a duplicate count target on the single whole-document leaf. "
            if secondary_caption_label
            else ""
        )
        + "The triangle marks the richer one-leaf duplicate-local-label check. "
        + "At one leaf, this is not really a merge effect: the model is getting extra whole-document side targets on that single leaf, especially richer local sketch and endpoint supervision in addition to the final count. "
        + "We would not treat that as a practical baseline in general, because it assumes access to an oracle or teacher that can decompose the task into these tree-local targets, meaning the label must factor through the tree structure in the way the method expects."
    )
    fig.text(
        0.015,
        0.012,
        caption_text,
        ha="left",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.canvas.draw()
    for ax in axes_list[: len(payloads)]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(9)
        offset_text.set_fontweight("bold")
    fig.tight_layout(rect=(0.0, 0.225, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "rendered": bool(rendered_panels > 0),
        "scope_key": str(primary_scope_key),
        "scope_label": str(scope_title),
        "train_doc_count": int(train_doc_count),
        "root_shares": [int(_safe_int(item.get("root_share"), 0)) for item in payloads],
        "rendered_panel_count": int(rendered_panels),
        "panel_summaries": panel_summaries,
        "figure_path": str(output_path),
        "scope_subtitle": str(scope_subtitle),
    }


def main() -> int:
    args = _parse_args()
    report_summary = _load_json(Path(args.report_summary))
    recovery = dict(report_summary.get("supervision_recovery") or {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    if bool(args.all_available_train_doc_counts):
        train_doc_counts = _available_train_doc_counts(
            recovery,
            scope_keys=tuple(RECOVERABLE_SCOPE_KEYS) + _structural_scope_candidates(args.structural_scope_key),
        )
    else:
        train_doc_counts = [int(value) for value in (args.train_doc_counts or [args.train_doc_count])]
    figure_variants: List[tuple[str, str | None]] = [("root_only", None)]
    for series_key in args.secondary_tree_series:
        figure_variants.append((str(series_key), str(series_key)))
    summaries: Dict[str, Any] = {}
    figure_paths: Dict[str, Any] = {}
    for train_doc_count in train_doc_counts:
        summaries_by_scope: Dict[str, Any] = {}
        figure_paths_by_scope: Dict[str, str] = {}
        for scope_key, scope_prefix, scope_keys, scope_preference_by_leaf in (
            (
                "recoverable_v5_t128",
                "recoverable_root_only_leaf_size_fixed",
                RECOVERABLE_SCOPE_KEYS,
                RECOVERABLE_SCOPE_PREFERENCE_BY_LEAF,
            ),
            (
                str(args.structural_scope_key),
                "structural_root_only_leaf_size_fixed",
                _structural_scope_candidates(str(args.structural_scope_key)),
                None,
            ),
        ):
            for variant_key, secondary_tree_series_key in figure_variants:
                variant_suffix = ""
                if secondary_tree_series_key:
                    variant_suffix = "_" + str(
                        SECONDARY_TREE_SERIES_CONFIG.get(secondary_tree_series_key, {}).get(
                            "filename_suffix",
                            secondary_tree_series_key,
                        )
                    )
                label_key = f"{scope_prefix}{variant_suffix}_train{int(train_doc_count)}"
                output_path = figures_dir / f"{label_key}.png"
                payload = _plot_fixed_train_leaf_size_root_only(
                    recovery,
                    scope_keys=scope_keys,
                    primary_scope_key=scope_key,
                    train_doc_count=int(train_doc_count),
                    root_shares=list(args.root_shares),
                    output_path=output_path,
                    scope_preference_by_leaf=scope_preference_by_leaf,
                    secondary_tree_series_key=secondary_tree_series_key,
                    empirical_bayes_mode=str(args.empirical_bayes),
                )
                storage_key = (
                    str(scope_key)
                    if variant_key == "root_only"
                    else f"{scope_key}__{variant_key}"
                )
                summaries_by_scope[storage_key] = payload
                if bool(payload.get("rendered")):
                    figure_paths_by_scope[storage_key] = str(output_path)
        summaries[str(int(train_doc_count))] = summaries_by_scope
        figure_paths[str(int(train_doc_count))] = figure_paths_by_scope
    markdown_lines = [
        "# Markov V3 Fixed-Train Leaf-Size Publication View",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"- Source summary: `{Path(args.report_summary)}`",
        f"- Fixed train docs: `{', '.join(str(int(value)) for value in train_doc_counts)}`",
        f"- Root shares: `{', '.join(str(int(value)) for value in args.root_shares)}`",
        "",
    ]
    for train_doc_count in train_doc_counts:
        markdown_lines.extend([f"## Fixed Train Docs = {int(train_doc_count):,}", ""])
        for scope_key, title, variant_key, variant_title in (
            ("recoverable_v5_t128", "Recoverable", "root_only", "Root-only supervision"),
            ("recoverable_v5_t128", "Recoverable", "leaf_mass_eq", "Root-only plus equal-total-mass leaf labels"),
            ("recoverable_v5_t128", "Recoverable", "depth_equal_mass_eq", "Root-only plus equal-total-mass leaf+merge labels"),
            (str(args.structural_scope_key), "Structural", "root_only", "Root-only supervision"),
            (str(args.structural_scope_key), "Structural", "leaf_mass_eq", "Root-only plus equal-total-mass leaf labels"),
            (str(args.structural_scope_key), "Structural", "depth_equal_mass_eq", "Root-only plus equal-total-mass leaf+merge labels"),
        ):
            if variant_key != "root_only" and variant_key not in args.secondary_tree_series:
                continue
            storage_key = scope_key if variant_key == "root_only" else f"{scope_key}__{variant_key}"
            payload = dict((summaries.get(str(int(train_doc_count))) or {}).get(storage_key) or {})
            markdown_lines.extend([f"### {title}: {variant_title}", ""])
            if payload.get("rendered"):
                figure_path = ((figure_paths.get(str(int(train_doc_count))) or {}).get(storage_key) or "")
                rel_path = Path(figure_path).relative_to(args.output_dir) if figure_path else None
                if rel_path is not None:
                    markdown_lines.append(f"![{title} fixed-train leaf-size view]({rel_path.as_posix()})")
                    markdown_lines.append("")
                markdown_lines.append(
                    f"- Rendered panels: `{int(_safe_int(payload.get('rendered_panel_count'), 0))}`."
                )
                for item in list(payload.get("panel_summaries") or []):
                    markdown_lines.append(
                        f"- `R{int(_safe_int(item.get('root_share'), 0))}`: leaf tokens `{', '.join(str(v) for v in item.get('available_leaf_tokens', []))}`, "
                        f"`official_fno_actual={item.get('official_fno_actual_root_mae')}`."
                    )
                    source_by_leaf = dict(item.get("tree_source_by_leaf_tokens") or {})
                    if source_by_leaf:
                        markdown_lines.append(
                            f"  sources: "
                            + ", ".join(
                                f"`{leaf}` → `{source_by_leaf[leaf]}`"
                                for leaf in sorted(
                                    source_by_leaf.keys(),
                                    key=lambda value: -int(_safe_int(value, 0)),
                                )
                            )
                        )
                    alternate_source_by_leaf = dict(item.get("alternate_tree_source_by_leaf_tokens") or {})
                    if alternate_source_by_leaf:
                        markdown_lines.append(
                            "  longer-schedule root-only sources: "
                            + ", ".join(
                                f"`{leaf}` → `{alternate_source_by_leaf[leaf]}`"
                                for leaf in sorted(
                                    alternate_source_by_leaf.keys(),
                                    key=lambda value: -int(_safe_int(value, 0)),
                                )
                            )
                        )
                    secondary_source_by_leaf = dict(item.get("secondary_tree_source_by_leaf_tokens") or {})
                    if secondary_source_by_leaf:
                        markdown_lines.append(
                            "  secondary tree sources: "
                            + ", ".join(
                                f"`{leaf}` → `{secondary_source_by_leaf[leaf]}`"
                                for leaf in sorted(
                                    secondary_source_by_leaf.keys(),
                                    key=lambda value: -int(_safe_int(value, 0)),
                                )
                            )
                        )
                    canary_by_leaf = dict(item.get("one_leaf_canary_source_by_leaf_tokens") or {})
                    if canary_by_leaf:
                        markdown_lines.append(
                            "  one-leaf canary: "
                            + ", ".join(
                                f"`{leaf}` → `{canary_by_leaf[leaf]}`"
                                for leaf in sorted(
                                    canary_by_leaf.keys(),
                                    key=lambda value: -int(_safe_int(value, 0)),
                                )
                            )
                        )
                    duplicate_local_source = str(
                        item.get("one_leaf_duplicate_local_label_source") or ""
                    ).strip()
                    if duplicate_local_source:
                        markdown_lines.append(
                            "  one-leaf duplicate-local check: "
                            f"`{duplicate_local_source}`"
                        )
            else:
                markdown_lines.append("- No rendered panels yet.")
            markdown_lines.append("")
    if len(train_doc_counts) == 1:
        only_count = str(int(train_doc_counts[0]))
        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_report_summary": str(Path(args.report_summary)),
            "train_doc_count": int(train_doc_counts[0]),
            "train_doc_counts": [int(value) for value in train_doc_counts],
            "root_shares": [int(value) for value in args.root_shares],
            "figures": dict(figure_paths.get(only_count) or {}),
            "scopes": dict(summaries.get(only_count) or {}),
        }
    else:
        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_report_summary": str(Path(args.report_summary)),
            "train_doc_count": int(train_doc_counts[0]),
            "train_doc_counts": [int(value) for value in train_doc_counts],
            "root_shares": [int(value) for value in args.root_shares],
            "figures": figure_paths,
            "scopes": summaries,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_json": str(args.output_dir / "summary.json"),
                "markdown": str(args.output_dir / "report.md"),
                "figures": figure_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
