#!/usr/bin/env python3
"""Paper-facing official-implementation comparison reports.

This script is intentionally narrow at the top level: it does not replace the
broader classical-sketch benchmark bundle.  Instead it gives the paper a stable
reproduction path for the current learned HLL/JAX grid comparison against the
Apache DataSketches implementation, plus a cheap inventory check for the
general DataSketches and Markov official-FNO lanes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TREEPO_SRC = REPO_ROOT / "treepo" / "src"
if str(TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(TREEPO_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HLL_GRID_SUMMARIES = (
    REPO_ROOT
    / "outputs"
    / "hll_jax_local_law_round4_overnight_grid_20260508_065221"
    / "grid_summary.csv",
    REPO_ROOT
    / "outputs"
    / "hll_jax_local_law_vocab512_screen_20260508_223207"
    / "grid_summary.csv",
)

HLL_COLUMNS = [
    "source_label",
    "leaf_tokens",
    "leaf_size_label",
    "leaves_per_doc",
    "full_doc_leaf",
    "selected_row",
    "group",
    "train_docs",
    "test_docs",
    "summary_dim",
    "selection_metric",
    "selection_metric_value",
    "learned_vs_native_hll_mae",
    "learned_vs_exact_unique_mae",
    "native_hll_vs_exact_unique_mae",
    "datasketches_official_vs_exact_unique_mae",
    "datasketches_full_doc_exact_mae",
    "datasketches_tree_flat_mae",
    "datasketches_tree_flat_max_abs",
    "native_full_doc_exact_mae",
    "native_tree_flat_mae",
    "native_tree_flat_max_abs",
    "test_hll_estimate_raw_mae",
    "test_hll_register_mae",
    "test_contextual_raw_mae",
    "precision",
    "hash_bits",
    "vocab_size",
    "doc_tokens",
    "seed",
    "official_implementation",
    "datasketches_version",
    "grid_summary",
    "summary_json",
]

INVENTORY_CHECKS = [
    {
        "area": "general_datasketches",
        "path": "scripts/run_classical_sketches_paper_bundle.py",
        "tokens": ("classical-sketches", "leaf-sizes", "audit_tree_bundle_contracts.py"),
        "note": "Broad paper bundle for Apache DataSketches and learned sketch assets.",
    },
    {
        "area": "general_datasketches",
        "path": "treepo/src/treepo/sketches/adapters/hll_datasketches.py",
        "tokens": ("datasketches.hll_sketch", "hll_union", "HLLDatasketchesAdapter"),
        "note": "Official Apache DataSketches HLL adapter.",
    },
    {
        "area": "general_datasketches",
        "path": "treepo/src/treepo/bench/classical_sketches.py",
        "tokens": (
            "make_hll_adapter(backend=\"datasketches\"",
            "make_cpc_adapter",
            "make_theta_adapter",
            "make_kll_floats_adapter",
            "make_count_min_adapter",
        ),
        "note": "Benchmark surface for HLL/CPC/Theta/frequency/quantile/sampling families.",
    },
    {
        "area": "general_datasketches",
        "path": "treepo/src/treepo/bench/reports/classical_sketches.py",
        "tokens": ("classical_sketches_aggregate.csv", "classical_sketches_grid.tex"),
        "note": "Report writer for cross-sketch tables and figures.",
    },
    {
        "area": "markov",
        "path": "scripts/run_markov_publication_bundle.py",
        "tokens": ("official_fno", "official_fno_sumlen"),
        "note": "Publication bundle includes official-FNO parity families.",
    },
    {
        "area": "markov",
        "path": "scripts/run_markov_optimization_tradeoff_pipeline.py",
        "tokens": ("official_fno", "official_fno_sumlen", "CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES"),
        "note": "Optimization tradeoff pipeline carries official-FNO comparison lanes.",
    },
    {
        "area": "markov",
        "path": "src/ctreepo/sim/core/full_doc_anchor_diagnostics.py",
        "tokens": ("official_fno", "official_fno_sumlen", "_official_fno_locked_config_for_benchmark"),
        "note": "Markov full-doc diagnostics construct the locked official-FNO comparator.",
    },
    {
        "area": "markov",
        "path": "src/ctreepo/sim/core/markov_changepoint_ops_count.py",
        "tokens": ("official_fno", "official_fno_sumlen", "official_neuraloperator_fno"),
        "note": "Core Markov simulator exposes official neural-operator baseline families.",
    },
    {
        "area": "markov",
        "path": "scripts/report_markov_optimization_tradeoffs.py",
        "tokens": ("official_fno_root_mae", "delta_vs_official_fno", "official_fno_sumlen"),
        "note": "Paper report compares learned tree rows against official-FNO rows.",
    },
]


@dataclass(frozen=True)
class SelectedGridRow:
    grid_summary: Path
    row: Mapping[str, str]
    source_label: str

    @property
    def name(self) -> str:
        return str(self.row.get("name") or "")

    @property
    def output_dir(self) -> Path:
        raw = str(self.row.get("output_dir") or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else (REPO_ROOT / p)
        return self.grid_summary.parent / self.name

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "summary.json"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    f = _safe_float(value)
    if not math.isfinite(f):
        return default
    return int(f)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n")


def _datasketches_version() -> str:
    try:
        return importlib.metadata.version("datasketches")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _format_metric(value: Any) -> str:
    f = _safe_float(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.4g}"


def _mae(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    n = min(int(aa.size), int(bb.size))
    return float(np.mean(np.abs(aa[:n] - bb[:n])))


def _max_abs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    n = min(int(aa.size), int(bb.size))
    return float(np.max(np.abs(aa[:n] - bb[:n])))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _label_for_grid(path: Path) -> str:
    parent = path.parent.name
    if "vocab512" in parent:
        return "vocab512_screen"
    if "round4" in parent:
        return "vocab128_round4"
    return parent or path.stem


def _parse_groups(value: str) -> set[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def _default_grid_summaries() -> list[Path]:
    return [path for path in DEFAULT_HLL_GRID_SUMMARIES if path.exists()]


def _select_hll_rows(
    grid_summaries: Sequence[Path],
    *,
    groups: set[str] | None,
    train_docs: int | None,
    selection_metric: str,
) -> list[SelectedGridRow]:
    by_key: dict[tuple[str, int], SelectedGridRow] = {}
    by_score: dict[tuple[str, int], float] = {}
    for grid_summary in grid_summaries:
        grid_summary = grid_summary.resolve()
        if not grid_summary.exists():
            raise FileNotFoundError(f"grid summary not found: {grid_summary}")
        source_label = _label_for_grid(grid_summary)
        for row in _read_csv_rows(grid_summary):
            if str(row.get("status") or "").strip().lower() != "ok":
                continue
            group = str(row.get("group") or "").strip()
            if groups is not None and group not in groups:
                continue
            row_train_docs = _safe_int(row.get("train_docs"))
            if train_docs is not None and row_train_docs != int(train_docs):
                continue
            leaf_tokens = _safe_int(row.get("fragment_len"))
            if leaf_tokens <= 0:
                continue
            score = _safe_float(row.get(selection_metric))
            if not math.isfinite(score):
                continue
            key = (source_label, leaf_tokens)
            if key not in by_score or score < by_score[key]:
                by_score[key] = score
                by_key[key] = SelectedGridRow(
                    grid_summary=grid_summary,
                    row=row,
                    source_label=source_label,
                )
    return [by_key[key] for key in sorted(by_key, key=lambda x: (x[0], x[1]))]


def _sample_token_fragment(
    tokens: Sequence[int],
    *,
    fragment_len: int,
    rng: np.random.Generator,
) -> list[int]:
    if not tokens:
        raise ValueError("cannot sample a fragment from an empty token sequence")
    length = max(1, min(int(fragment_len), len(tokens)))
    if len(tokens) == length:
        return [int(tok) for tok in tokens]
    start = int(rng.integers(0, len(tokens) - length + 1))
    return [int(tok) for tok in tokens[start : start + length]]


def _random_token_docs(
    *,
    n_docs: int,
    doc_tokens: int,
    vocab_size: int,
    seed: int,
    seed_offset: int,
) -> list[list[int]]:
    rng = np.random.default_rng(int(seed) + int(seed_offset))
    return [
        [
            int(tok)
            for tok in rng.integers(
                0,
                int(vocab_size),
                size=int(doc_tokens),
            )
        ]
        for _ in range(int(n_docs))
    ]


def _reconstruct_hll_test_docs_and_fragments(
    *,
    test_docs: int,
    doc_tokens: int,
    vocab_size: int,
    fragment_len: int,
    seed: int,
    samples_per_doc: int,
) -> tuple[list[list[int]], list[list[int]]]:
    docs = _random_token_docs(
        n_docs=int(test_docs),
        doc_tokens=int(doc_tokens),
        vocab_size=int(vocab_size),
        seed=int(seed),
        seed_offset=202,
    )
    item_rng = np.random.default_rng(int(seed) + 505)
    fragments: list[list[int]] = []
    for doc in docs:
        for _ in range(int(samples_per_doc)):
            fragments.append(
                _sample_token_fragment(
                    doc,
                    fragment_len=int(fragment_len),
                    rng=item_rng,
                )
            )
    return docs, fragments


def _tree_reduce(adapter: Any, sketches: Sequence[Any]) -> Any:
    if not sketches:
        raise ValueError("sketches must be non-empty")
    cur = list(sketches)
    while len(cur) > 1:
        nxt: list[Any] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
            else:
                nxt.append(adapter.merge(cur[i], cur[i + 1]))
                i += 2
        cur = nxt
    return cur[0]


def _estimate_flat(adapter: Any, tokens: Sequence[int]) -> float:
    return float(adapter.query(adapter.encode(tokens)))


def _estimate_tree(adapter: Any, tokens: Sequence[int], *, leaf_tokens: int) -> float:
    chunks = [tokens[i : i + int(leaf_tokens)] for i in range(0, len(tokens), int(leaf_tokens))]
    sketches = [adapter.encode(chunk) for chunk in chunks if chunk]
    return float(adapter.query(_tree_reduce(adapter, sketches)))


def _load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary JSON not found: {path}")
    with path.open(encoding="utf-8") as f:
        return dict(json.load(f))


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _summary_args(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    args = summary.get("args")
    return args if isinstance(args, Mapping) else {}


def _summary_data_metadata(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = summary.get("data_source_metadata")
    return meta if isinstance(meta, Mapping) else {}


def _make_hll_adapters(*, precision: int, hash_bits: int) -> tuple[Any, Any]:
    from treepo.bench.sketches.adapters.hll_datasketches import HLLDatasketchesAdapter
    from src.tree.hll import HLLNativeAdapter

    return (
        HLLDatasketchesAdapter(precision=int(precision)),
        HLLNativeAdapter(precision=int(precision), hash_bits=int(hash_bits)),
    )


def _hll_metrics_for_selected_row(
    selected: SelectedGridRow,
    *,
    selection_metric: str,
) -> dict[str, Any]:
    row = dict(selected.row)
    summary = _load_summary_json(selected.summary_json)
    args = _summary_args(summary)
    data_meta = _summary_data_metadata(summary)
    test_diag = _nested(summary, "diagnostics", "test")
    if not isinstance(test_diag, Mapping):
        test_diag = {}

    precision = _safe_int(row.get("precision"), _safe_int(args.get("hll_precision"), 8))
    hash_bits = _safe_int(row.get("hash_bits"), _safe_int(args.get("hll_hash_bits"), 64))
    vocab_size = _safe_int(row.get("vocab_size"), _safe_int(data_meta.get("vocab_size"), 0))
    doc_tokens = _safe_int(row.get("doc_tokens"), _safe_int(data_meta.get("doc_tokens"), 0))
    test_docs = _safe_int(row.get("test_docs"), _safe_int(data_meta.get("test_docs"), 0))
    train_docs = _safe_int(row.get("train_docs"), _safe_int(data_meta.get("train_docs"), 0))
    fragment_len = _safe_int(row.get("fragment_len"), _safe_int(args.get("fragment_len"), 0))
    seed = _safe_int(row.get("seed"), _safe_int(args.get("seed"), 0))
    samples_per_doc = _safe_int(args.get("context_samples_per_doc"), 1)

    learned_pred = list(test_diag.get("per_leaf_hll_estimate_pred_raw") or [])
    native_truth = list(test_diag.get("per_leaf_hll_estimate_truth_raw") or [])
    if learned_pred:
        inferred = max(1, int(round(len(learned_pred) / max(1, test_docs))))
        samples_per_doc = max(1, samples_per_doc, inferred)

    docs, fragments = _reconstruct_hll_test_docs_and_fragments(
        test_docs=int(test_docs),
        doc_tokens=int(doc_tokens),
        vocab_size=int(vocab_size),
        fragment_len=int(fragment_len),
        seed=int(seed),
        samples_per_doc=int(samples_per_doc),
    )
    if learned_pred:
        fragments = fragments[: len(learned_pred)]

    ds_adapter, native_adapter = _make_hll_adapters(precision=precision, hash_bits=hash_bits)

    exact_fragment_counts = [float(len(set(fragment))) for fragment in fragments]
    ds_fragment_estimates = [_estimate_flat(ds_adapter, fragment) for fragment in fragments]
    if native_truth:
        native_fragment_estimates = [float(x) for x in native_truth[: len(fragments)]]
    else:
        native_fragment_estimates = [_estimate_flat(native_adapter, fragment) for fragment in fragments]

    ds_full_flat: list[float] = []
    ds_full_tree: list[float] = []
    native_full_flat: list[float] = []
    native_full_tree: list[float] = []
    exact_full_counts: list[float] = []
    for doc in docs:
        exact_full_counts.append(float(len(set(doc))))
        ds_full_flat.append(_estimate_flat(ds_adapter, doc))
        ds_full_tree.append(_estimate_tree(ds_adapter, doc, leaf_tokens=fragment_len))
        native_full_flat.append(_estimate_flat(native_adapter, doc))
        native_full_tree.append(_estimate_tree(native_adapter, doc, leaf_tokens=fragment_len))

    leaves_per_doc = int(math.ceil(float(doc_tokens) / float(fragment_len))) if fragment_len else 0
    full_doc_leaf = bool(fragment_len >= doc_tokens > 0)
    leaf_size_label = f"{fragment_len} (full)" if full_doc_leaf else str(fragment_len)
    datasketches_impl = "Apache DataSketches hll_sketch/hll_union"

    return {
        "source_label": selected.source_label,
        "leaf_tokens": int(fragment_len),
        "leaf_size_label": leaf_size_label,
        "leaves_per_doc": int(leaves_per_doc),
        "full_doc_leaf": str(full_doc_leaf).lower(),
        "selected_row": selected.name,
        "group": row.get("group", ""),
        "train_docs": int(train_docs),
        "test_docs": int(test_docs),
        "summary_dim": _safe_int(row.get("summary_dim"), 0),
        "selection_metric": selection_metric,
        "selection_metric_value": _safe_float(row.get(selection_metric)),
        "learned_vs_native_hll_mae": _mae(learned_pred, native_fragment_estimates)
        if learned_pred
        else float("nan"),
        "learned_vs_exact_unique_mae": _mae(learned_pred, exact_fragment_counts)
        if learned_pred
        else float("nan"),
        "native_hll_vs_exact_unique_mae": _mae(native_fragment_estimates, exact_fragment_counts),
        "datasketches_official_vs_exact_unique_mae": _mae(
            ds_fragment_estimates,
            exact_fragment_counts,
        ),
        "datasketches_full_doc_exact_mae": _mae(ds_full_tree, exact_full_counts),
        "datasketches_tree_flat_mae": _mae(ds_full_tree, ds_full_flat),
        "datasketches_tree_flat_max_abs": _max_abs_delta(ds_full_tree, ds_full_flat),
        "native_full_doc_exact_mae": _mae(native_full_tree, exact_full_counts),
        "native_tree_flat_mae": _mae(native_full_tree, native_full_flat),
        "native_tree_flat_max_abs": _max_abs_delta(native_full_tree, native_full_flat),
        "test_hll_estimate_raw_mae": _safe_float(row.get("test_hll_estimate_raw_mae")),
        "test_hll_register_mae": _safe_float(row.get("test_hll_register_mae")),
        "test_contextual_raw_mae": _safe_float(row.get("test_contextual_raw_mae")),
        "precision": int(precision),
        "hash_bits": int(hash_bits),
        "vocab_size": int(vocab_size),
        "doc_tokens": int(doc_tokens),
        "seed": int(seed),
        "official_implementation": datasketches_impl,
        "datasketches_version": _datasketches_version(),
        "grid_summary": str(selected.grid_summary),
        "summary_json": str(selected.summary_json),
    }


def _hll_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Official HLL Implementation Comparison",
        "",
        "Learned rows are selected from each grid by the requested metric. "
        "Official rows use Apache DataSketches `hll_sketch`/`hll_union`; "
        "native rows use the repo's deterministic register-max HLL.",
        "",
        "| source | leaf size | leaves/doc | learned vs native HLL MAE | learned vs exact MAE | native HLL vs exact MAE | DataSketches leaf vs exact MAE | DataSketches full-doc vs exact MAE | DataSketches tree-flat MAE | row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {source} | {leaf} | {leaves} | {learned_native} | {learned_exact} | "
            "{native_exact} | {ds_leaf} | {ds_full} | {ds_tree_flat} | `{name}` |".format(
                source=row.get("source_label", ""),
                leaf=row.get("leaf_size_label", row.get("leaf_tokens", "")),
                leaves=row.get("leaves_per_doc", ""),
                learned_native=_format_metric(row.get("learned_vs_native_hll_mae")),
                learned_exact=_format_metric(row.get("learned_vs_exact_unique_mae")),
                native_exact=_format_metric(row.get("native_hll_vs_exact_unique_mae")),
                ds_leaf=_format_metric(row.get("datasketches_official_vs_exact_unique_mae")),
                ds_full=_format_metric(row.get("datasketches_full_doc_exact_mae")),
                ds_tree_flat=_format_metric(row.get("datasketches_tree_flat_mae")),
                name=row.get("selected_row", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `leaf size` includes `full` when one document is one leaf.",
            "- Leaf-level exact MAE uses exact unique-token counts on the sampled test fragments.",
            "- Full-doc metrics build a tree from each test document at the same leaf size, then compare with exact document cardinality or a flat official sketch.",
            "",
        ]
    )
    return "\n".join(lines)


def run_hll_jax(args: argparse.Namespace) -> int:
    grid_summaries = [Path(p) for p in args.grid_summary]
    if not grid_summaries:
        grid_summaries = _default_grid_summaries()
    if not grid_summaries:
        raise SystemExit(
            "no --grid-summary supplied and no default HLL grid summaries were found"
        )

    selected = _select_hll_rows(
        grid_summaries,
        groups=_parse_groups(args.groups),
        train_docs=args.train_docs,
        selection_metric=str(args.selection_metric),
    )
    if not selected:
        raise SystemExit("no completed HLL rows matched the requested filters")

    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "outputs" / f"official_implementation_comparison_{_utc_stamp()}"
    )
    out_dir = out_dir.resolve()
    rows = [
        _hll_metrics_for_selected_row(row, selection_metric=str(args.selection_metric))
        for row in selected
    ]
    _write_csv(out_dir / "official_hll_jax_comparison.csv", rows, HLL_COLUMNS)
    (out_dir / "official_hll_jax_comparison.md").write_text(
        _hll_markdown(rows),
        encoding="utf-8",
    )
    _write_json(
        out_dir / "official_hll_jax_comparison_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "command_args": {
                key: value for key, value in vars(args).items() if key != "func"
            },
            "grid_summaries": [str(path.resolve()) for path in grid_summaries],
            "selection_metric": str(args.selection_metric),
            "groups": str(args.groups),
            "train_docs": args.train_docs,
            "rows": rows,
        },
    )
    if args.stage_paper_assets:
        assets_dir = REPO_ROOT / "paper" / "ctreepo" / "assets" / "sketches" / "tables"
        _write_csv(assets_dir / "official_hll_jax_comparison.csv", rows, HLL_COLUMNS)
        (assets_dir / "official_hll_jax_comparison.md").write_text(
            _hll_markdown(rows),
            encoding="utf-8",
        )
    print(f"wrote {out_dir / 'official_hll_jax_comparison.csv'}")
    print(f"wrote {out_dir / 'official_hll_jax_comparison.md'}")
    return 0


def _inventory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check in INVENTORY_CHECKS:
        rel_path = str(check["path"])
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        tokens = tuple(str(tok) for tok in check.get("tokens", ()))
        missing = [tok for tok in tokens if tok not in text]
        rows.append(
            {
                "area": str(check["area"]),
                "path": rel_path,
                "exists": bool(path.exists()),
                "tokens_checked": list(tokens),
                "missing_tokens": missing,
                "status": "ok" if path.exists() and not missing else "missing",
                "note": str(check.get("note", "")),
            }
        )
    return rows


def _inventory_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Official Implementation Inventory",
        "",
        "| area | status | path | missing tokens | note |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        missing = ", ".join(str(x) for x in row.get("missing_tokens", []) or [])
        lines.append(
            "| {area} | {status} | `{path}` | {missing} | {note} |".format(
                area=row.get("area", ""),
                status=row.get("status", ""),
                path=row.get("path", ""),
                missing=missing,
                note=str(row.get("note", "")).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "General DataSketches paper runner:",
            "",
            "```bash",
            "./venv/bin/python scripts/run_classical_sketches_paper_bundle.py --leaf-sizes 16,32,64,128,256,512",
            "```",
            "",
            "Focused learned-HLL vs official-HLL report:",
            "",
            "```bash",
            "./venv/bin/python scripts/report_official_implementation_comparison.py hll-jax",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_inventory(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "outputs" / f"official_implementation_inventory_{_utc_stamp()}"
    )
    out_dir = out_dir.resolve()
    rows = _inventory_rows()
    _write_json(
        out_dir / "official_implementation_inventory.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "rows": rows,
        },
    )
    (out_dir / "official_implementation_inventory.md").write_text(
        _inventory_markdown(rows),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'official_implementation_inventory.md'}")
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed and bool(args.strict):
        for row in failed:
            print(
                f"missing inventory check: {row.get('path')} "
                f"missing={row.get('missing_tokens')}",
                file=sys.stderr,
            )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hll = sub.add_parser(
        "hll-jax",
        help="Compare learned HLL/JAX grid rows with Apache DataSketches HLL.",
    )
    hll.add_argument(
        "--grid-summary",
        action="append",
        default=[],
        type=Path,
        help=(
            "Path to a grid_summary.csv. May be repeated. Defaults to the latest "
            "round4 vocab128 and vocab512 grid summaries if present."
        ),
    )
    hll.add_argument(
        "--groups",
        default="main",
        help="Comma-separated grid groups to consider, or 'all'. Default: main.",
    )
    hll.add_argument(
        "--train-docs",
        type=int,
        default=102400,
        help="Select rows with this train_docs value. Use -1 for no filter.",
    )
    hll.add_argument(
        "--selection-metric",
        default="test_hll_estimate_raw_mae",
        help="Metric minimized within each source/leaf-size cell.",
    )
    hll.add_argument("--output-dir", type=Path, default=None)
    hll.add_argument(
        "--stage-paper-assets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also copy CSV/Markdown table into paper/ctreepo/assets/sketches/tables.",
    )
    hll.set_defaults(func=run_hll_jax)

    inventory = sub.add_parser(
        "inventory",
        help="Verify general DataSketches and Markov official-comparator code paths.",
    )
    inventory.add_argument("--output-dir", type=Path, default=None)
    inventory.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return nonzero if an inventory check is missing.",
    )
    inventory.set_defaults(func=run_inventory)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", "") == "hll-jax" and int(args.train_docs) < 0:
        args.train_docs = None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
