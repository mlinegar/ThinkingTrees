#!/usr/bin/env python3
"""Build a canonical all-dimension Manifesto coverage split.

The first production use is the all-six Benoit policy-dimension pool with a
soft bias toward shorter training documents.  The output is deliberately small:
``split_ids.json`` for downstream scripts, ``coverage_split_summary.json`` for
audit/debugging, and ``run_manifest.json`` for RunManifest v1 provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.script_io import stable_digest as _stable_digest  # noqa: E402
from src.tasks.manifesto.result_rows import load_rows_by_dimension as _load_rows_by_dimension  # noqa: E402
from src.ctreepo.contracts import run_manifest_metadata  # noqa: E402
from src.tasks.manifesto.script_utils import (  # noqa: E402
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    parse_csv as _parse_csv,
    write_json as _write_json,
)
from src.experiments import SamplingPlan  # noqa: E402
from src.tasks.manifesto import ManifestoDataset  # noqa: E402
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_NORMALIZED_1_7,
    resolve_benoit_expert_target,
)


DEFAULT_DIMENSIONS = (
    "economic",
    "social",
    "decentralization",
    "environment",
    "eu",
    "immigration",
)
SPLIT_SCHEMA_VERSION = "ctreepo.manifesto_coverage_split.v1"
DEFAULT_STRATEGY = "all6_soft_inverse_sqrt_length"
DEFAULT_FULL_DOC_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "manifesto_corpus_benoit"




def _resolve_mp_data_dir(path: Optional[Path]) -> Optional[Path]:
    if path is not None:
        return Path(path)
    return DEFAULT_FULL_DOC_DATA_DIR if DEFAULT_FULL_DOC_DATA_DIR.exists() else None



def _eligible_documents(
    *,
    rows_by_dimension: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dimensions: Sequence[str],
    dataset: ManifestoDataset,
    min_doc_chars: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ids_by_dim = {dim: set(rows_by_dimension[dim]) for dim in dimensions}
    union_ids = set().union(*ids_by_dim.values()) if ids_by_dim else set()
    all_dim_ids = set.intersection(*ids_by_dim.values()) if ids_by_dim else set()
    coverage_counts = {
        str(k): sum(1 for doc_id in union_ids if sum(doc_id in ids_by_dim[d] for d in dimensions) >= k)
        for k in range(1, len(dimensions) + 1)
    }
    docs: List[Dict[str, Any]] = []
    skipped = {"missing_text": 0, "missing_label": 0, "too_short": 0}
    for doc_id in sorted(all_dim_ids):
        sample = dataset.get_sample(str(doc_id))
        text = str(getattr(sample, "text", "") or "") if sample is not None else ""
        if not text.strip():
            skipped["missing_text"] += 1
            continue
        if int(min_doc_chars) > 0 and len(text) < int(min_doc_chars):
            skipped["too_short"] += 1
            continue
        labels: Dict[str, float] = {}
        missing_label = False
        for dim in dimensions:
            label = resolve_benoit_expert_target(
                rows_by_dimension[dim][doc_id],
                dimension=str(dim),
                scale=EXPERT_SCALE_NORMALIZED_1_7,
            )
            if label is None:
                missing_label = True
                break
            labels[str(dim)] = float(label)
        if missing_label:
            skipped["missing_label"] += 1
            continue
        docs.append(
            {
                "manifesto_id": str(doc_id),
                "char_len": int(len(text)),
                "party_id": int(getattr(sample, "party_id", 0) or 0),
                "year": int(getattr(sample, "year", 0) or 0),
                "labels_1_7": labels,
            }
        )
    stats = {
        "dimension_counts": {dim: len(ids_by_dim[dim]) for dim in dimensions},
        "union_count": len(union_ids),
        "all_dimensions_count": len(all_dim_ids),
        "coverage_at_least_k": coverage_counts,
        "skipped": skipped,
    }
    return docs, stats


def _training_weight(
    char_len: int,
    *,
    length_floor_chars: int,
    min_raw_weight: float,
    max_weight_ratio: float,
) -> float:
    length = max(float(char_len), float(length_floor_chars), 1.0)
    raw = 1.0 / math.sqrt(length)
    cap = float(min_raw_weight) * float(max_weight_ratio)
    return float(min(raw, cap))


def _weighted_sample_without_replacement(
    docs: Sequence[Mapping[str, Any]],
    n: int,
    *,
    rng: random.Random,
) -> List[Mapping[str, Any]]:
    pool = [dict(doc) for doc in docs]
    selected: List[Mapping[str, Any]] = []
    if n > len(pool):
        raise ValueError(f"cannot sample {n} training docs from pool of {len(pool)}")
    for _ in range(n):
        total = sum(float(doc["sampling_weight"]) for doc in pool)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_idx = len(pool) - 1
        for idx, doc in enumerate(pool):
            cumulative += float(doc["sampling_weight"])
            if cumulative >= threshold:
                chosen_idx = idx
                break
        selected.append(pool.pop(chosen_idx))
    return selected


def build_coverage_split(
    *,
    source_root: Path,
    output_dir: Path,
    dimensions: Sequence[str],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    mp_data_dir: Optional[Path],
    length_floor_chars: int,
    max_weight_ratio: float,
    min_doc_chars: int = 0,
    strategy: str = DEFAULT_STRATEGY,
) -> Dict[str, Any]:
    rows_by_dimension = _load_rows_by_dimension(source_root, dimensions)
    resolved_mp_data_dir = _resolve_mp_data_dir(mp_data_dir)
    dataset = ManifestoDataset(data_dir=resolved_mp_data_dir, require_text=True)
    docs, coverage_stats = _eligible_documents(
        rows_by_dimension=rows_by_dimension,
        dimensions=dimensions,
        dataset=dataset,
        min_doc_chars=int(min_doc_chars),
    )
    needed = int(train_n) + int(val_n) + int(test_n)
    if len(docs) < needed:
        raise SystemExit(
            f"not enough eligible all-dimension docs: have {len(docs)}, need {needed}"
        )
    raw_weights = [
        1.0 / math.sqrt(max(float(doc["char_len"]), float(length_floor_chars), 1.0))
        for doc in docs
    ]
    min_raw_weight = min(raw_weights) if raw_weights else 1.0
    docs_by_id = {str(doc["manifesto_id"]): dict(doc) for doc in docs}
    for doc in docs_by_id.values():
        doc["sampling_weight"] = _training_weight(
            int(doc["char_len"]),
            length_floor_chars=int(length_floor_chars),
            min_raw_weight=float(min_raw_weight),
            max_weight_ratio=float(max_weight_ratio),
        )

    rng = random.Random(int(seed))
    uniform_pool = [str(doc["manifesto_id"]) for doc in docs]
    rng.shuffle(uniform_pool)
    val_ids = uniform_pool[: int(val_n)]
    test_ids = uniform_pool[int(val_n) : int(val_n) + int(test_n)]
    holdout = set(val_ids) | set(test_ids)
    train_pool = [docs_by_id[doc_id] for doc_id in sorted(docs_by_id) if doc_id not in holdout]
    train_docs = _weighted_sample_without_replacement(train_pool, int(train_n), rng=rng)
    train_ids = [str(doc["manifesto_id"]) for doc in train_docs]

    split_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    sampling_plan = SamplingPlan(
        seed=int(seed),
        split="train,val,test",
        strategy=strategy,
        sample_budget=needed,
        unit="document",
        frame="manifesto_all6_labeled",
        metadata={
            "dimensions": list(dimensions),
            "length_floor_chars": int(length_floor_chars),
            "max_weight_ratio": float(max_weight_ratio),
            "min_doc_chars": int(min_doc_chars),
            "train_sampling": "weighted_without_replacement",
            "val_test_sampling": "uniform_without_replacement",
        },
    ).to_dict()
    digest_payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_root": str(source_root),
        "mp_data_dir": str(resolved_mp_data_dir or ""),
        "dimensions": list(dimensions),
        "sampling_plan": sampling_plan,
        "split_ids": split_ids,
        "doc_lengths": {
            doc_id: int(docs_by_id[doc_id]["char_len"])
            for split in ("train", "val", "test")
            for doc_id in split_ids[split]
        },
    }
    split_digest = _stable_digest(digest_payload)
    selected_docs = []
    for split, ids in split_ids.items():
        for rank, doc_id in enumerate(ids):
            doc = dict(docs_by_id[doc_id])
            doc["split"] = split
            doc["split_rank"] = int(rank)
            selected_docs.append(doc)
    summary = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "split_manifest_digest": split_digest,
        "source_root": str(source_root),
        "mp_data_dir": str(resolved_mp_data_dir or ""),
        "text_source_kind": "raw_manifesto_full_document",
        "dimensions": list(dimensions),
        "sampling_plan": sampling_plan,
        "coverage": coverage_stats,
        "eligible_count": len(docs),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "selected_docs": selected_docs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = _write_json(output_dir / "split_ids.json", split_ids)
    summary_path = _write_json(output_dir / "coverage_split_summary.json", summary)
    run_manifest = run_manifest_metadata(
        run_id=f"manifesto.coverage_split.{split_digest[:12]}",
        domain="manifesto_rile",
        role="coverage_split_builder",
        backend="data_prep",
        status="completed",
        input_contracts=[
            {
                "kind": "manifesto_dimension_results",
                "schema_version": "ctreepo.manifesto_dimension_results.v1",
                "source_root": str(source_root),
                "dimensions": list(dimensions),
            }
        ],
        output_artifacts=[
            {"kind": "split_ids", "uri": str(split_path)},
            {"kind": "coverage_split_summary", "uri": str(summary_path)},
            {"kind": "coverage_split_directory", "uri": str(output_dir)},
        ],
        audit_results={"ok": True, "eligible_count": len(docs)},
        quarantine={"classification": "valid_run_manifest_v1"},
        command=sys.argv,
        publication_ready=True,
        metadata={
            "runner": "scripts/build_manifesto_coverage_split.py",
            "sampling_plan": sampling_plan,
            "split_manifest_digest": split_digest,
            "mp_data_dir": str(resolved_mp_data_dir or ""),
            "text_source_kind": "raw_manifesto_full_document",
        },
    )
    run_manifest_path = _write_json(output_dir / "run_manifest.json", run_manifest)
    summary["artifacts"] = {
        "split_ids": str(split_path),
        "coverage_split_summary": str(summary_path),
        "run_manifest": str(run_manifest_path),
    }
    _write_json(output_dir / "coverage_split_summary.json", summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/overnight_benoit/full_pipeline"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dimensions", default=",".join(DEFAULT_DIMENSIONS))
    parser.add_argument("--train-n", type=int, default=90)
    parser.add_argument("--val-n", type=int, default=30)
    parser.add_argument("--test-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mp-data-dir", type=Path, default=None)
    parser.add_argument("--length-floor-chars", type=int, default=2000)
    parser.add_argument("--max-weight-ratio", type=float, default=8.0)
    parser.add_argument("--min-doc-chars", type=int, default=0)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or Path("outputs") / "manifesto_coverage_splits" / f"all6_soft_inverse_{_now_stamp()}"
    summary = build_coverage_split(
        source_root=Path(args.source_root),
        output_dir=Path(output_dir),
        dimensions=_parse_csv(args.dimensions),
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        test_n=int(args.test_n),
        seed=int(args.seed),
        mp_data_dir=args.mp_data_dir,
        length_floor_chars=int(args.length_floor_chars),
        max_weight_ratio=float(args.max_weight_ratio),
        min_doc_chars=int(args.min_doc_chars),
        strategy=str(args.strategy),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "split_manifest_digest": summary["split_manifest_digest"],
                "split_counts": summary["split_counts"],
                "eligible_count": summary["eligible_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
