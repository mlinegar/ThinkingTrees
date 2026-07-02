#!/usr/bin/env python3
"""Regenerate per-document C-TreePO prediction records from saved f/g artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.alternating import evaluate_iteration
from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig
from src.ctreepo.joint_dspy_family import DIMENSION_ORDER, JointDSPyFamily, JointDSPyFamilyConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    root: Path
    teacher_root: Path
    dimension: Optional[str]
    is_joint: bool
    manifest_path: Path
    leaf_size_tokens: int
    iteration: Mapping[str, Any]
    score_by_dimension: Mapping[str, Optional[float]]
    macro_score: Optional[float]

    @property
    def row_dir(self) -> Path:
        return self.manifest_path.parent

    @property
    def record_path(self) -> Path:
        iteration = int(self.iteration["iteration"])
        return self.row_dir / "prediction_records" / f"iter_{iteration:02d}_post_eval.jsonl"

    @property
    def leaf_label(self) -> str:
        return f"leaf{int(self.leaf_size_tokens):04d}tok"

    @property
    def teacher_path(self) -> Path:
        return self.teacher_root / self.leaf_label / "labeled_trees.jsonl"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_mapping(values: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"expected KEY=PATH mapping, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise SystemExit(f"empty key in mapping {raw!r}")
        out[key] = Path(value).expanduser()
    return out


def _parse_root_mappings(values: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"expected KEY=PATH mapping, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise SystemExit(f"empty key in mapping {raw!r}")
        out.append((key, Path(value).expanduser()))
    return out


def _split_metrics(iteration: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    metrics = iteration.get("split_metrics") or {}
    if not isinstance(metrics, Mapping):
        return {}
    selected = metrics.get(split) or metrics.get("all") or {}
    return selected if isinstance(selected, Mapping) else {}


def _candidate_scores(
    iteration: Mapping[str, Any],
    *,
    split: str,
    scalar_dimension: Optional[str],
) -> Tuple[Dict[str, Optional[float]], Optional[float]]:
    metrics = _split_metrics(iteration, split)
    macro_score = _safe_float(metrics.get("external_expert_pearson"))
    per_dimension = metrics.get("per_dimension")
    scores: Dict[str, Optional[float]] = {}
    if isinstance(per_dimension, Mapping) and per_dimension:
        for dim, raw_metrics in per_dimension.items():
            if isinstance(raw_metrics, Mapping):
                scores[str(dim)] = _safe_float(raw_metrics.get("external_expert_pearson"))
    elif scalar_dimension is not None:
        scores[str(scalar_dimension)] = macro_score
    return scores, macro_score


def _iter_candidates_for_root(
    *,
    root: Path,
    teacher_root: Path,
    scalar_dimension: Optional[str],
    is_joint: bool,
    split: str,
) -> Iterable[Candidate]:
    for manifest_path in sorted((root / "ladder" / "dspy").glob("leaf*tok/iteration_history.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        leaf_size = manifest.get("leaf_size_tokens") or manifest.get("axis_value")
        if leaf_size is None:
            LOGGER.warning("Skipping %s: no leaf_size_tokens", manifest_path)
            continue
        for iteration in manifest.get("iterations", []) or []:
            if not isinstance(iteration, Mapping) or iteration.get("iteration") is None:
                continue
            if iteration.get("extra", {}).get("error") or iteration.get("error"):
                continue
            scores, macro_score = _candidate_scores(
                iteration, split=split, scalar_dimension=scalar_dimension
            )
            if not scores and macro_score is None:
                continue
            yield Candidate(
                root=root,
                teacher_root=teacher_root,
                dimension=scalar_dimension,
                is_joint=is_joint,
                manifest_path=manifest_path,
                leaf_size_tokens=int(leaf_size),
                iteration=iteration,
                score_by_dimension=scores,
                macro_score=macro_score,
            )


def _select_best_by_dimension(candidates: Sequence[Candidate], dimensions: Sequence[str]) -> Dict[str, Candidate]:
    selected: Dict[str, Candidate] = {}
    selected_score: Dict[str, float] = {}
    for candidate in candidates:
        for dim in dimensions:
            score = candidate.score_by_dimension.get(dim)
            if score is None:
                continue
            if dim not in selected or float(score) > selected_score[dim]:
                selected[dim] = candidate
                selected_score[dim] = float(score)
    return selected


def _filter_eval_trees(trees: Sequence[Any], split: str) -> List[Any]:
    if split == "all":
        return list(trees)
    out = []
    for tree in trees:
        tree_split = str((getattr(tree, "metadata", None) or {}).get("split") or "").lower()
        if tree_split == split.lower():
            out.append(tree)
    return out


def _lm_config(args: argparse.Namespace, *, leaf_size_tokens: int) -> Dict[str, Any]:
    max_tokens = int(args.max_completion_tokens)
    if max_tokens <= 0:
        max_tokens = max(1, 2 * int(leaf_size_tokens))
    return {
        "model": str(args.dspy_model),
        "api_base": str(args.dspy_api_base),
        "api_key": str(args.dspy_api_key),
        "max_tokens": int(max_tokens),
    }


def _make_family(candidate: Candidate, args: argparse.Namespace) -> Any:
    max_tokens = int(args.max_completion_tokens)
    if max_tokens <= 0:
        max_tokens = max(1, 2 * int(candidate.leaf_size_tokens))
    common = dict(
        optimizer="mipro",
        budget="medium",
        num_threads=int(args.num_threads),
        target_min=1.0,
        target_max=7.0,
        lm_config=_lm_config(args, leaf_size_tokens=int(candidate.leaf_size_tokens)),
        lm_transport=str(args.lm_transport),
        batch_max_concurrent=int(args.batch_max_concurrent),
        batch_size=int(args.batch_size),
        batch_timeout=float(args.batch_timeout),
        batch_request_timeout=float(args.batch_request_timeout),
        batch_await_response_timeout=args.batch_await_response_timeout,
        batch_routing_policy=str(args.batch_routing_policy),
        leaf_size_tokens=int(candidate.leaf_size_tokens),
        lm_context_window_tokens=int(args.context_tokens),
        max_completion_tokens=int(max_tokens),
        prompt_template_overhead_tokens=int(args.prompt_overhead_tokens),
        tokenizer_model_path=str(args.tokenizer_model),
    )
    if candidate.is_joint:
        return JointDSPyFamily(
            config=JointDSPyFamilyConfig(
                **common,
                dimension="combined",
                dimensions=tuple(args.dimensions or DIMENSION_ORDER),
                f_init_path=str(args.joint_f_init_path) if args.joint_f_init_path else None,
            )
        )
    if candidate.dimension is None:
        raise ValueError(f"scalar candidate lacks dimension: {candidate}")
    return DSPyFamily(
        config=DSPyFamilyConfig(
            **common,
            dimension=str(candidate.dimension),
        )
    )


def _regenerate_candidate(candidate: Candidate, args: argparse.Namespace) -> Dict[str, Any]:
    if not candidate.teacher_path.exists():
        raise FileNotFoundError(f"missing labeled trees: {candidate.teacher_path}")
    if candidate.record_path.exists() and not args.overwrite:
        LOGGER.info("Skipping existing prediction records: %s", candidate.record_path)
        return {
            "record_path": str(candidate.record_path),
            "skipped_existing": True,
            "leaf_size_tokens": candidate.leaf_size_tokens,
            "iteration": candidate.iteration.get("iteration"),
            "stage_label": candidate.iteration.get("stage_label"),
        }

    trees = load_labeled_trees(candidate.teacher_path)
    eval_trees = _filter_eval_trees(trees, str(args.split))
    if not eval_trees:
        eval_trees = list(trees)
    family = _make_family(candidate, args)
    split_metrics = evaluate_iteration(
        family=family,
        f=candidate.iteration.get("f_artifact"),
        g=candidate.iteration.get("g_artifact"),
        trees=eval_trees,
        prediction_records_path=candidate.record_path,
    )
    return {
        "record_path": str(candidate.record_path),
        "teacher_path": str(candidate.teacher_path),
        "n_eval_trees": len(eval_trees),
        "leaf_size_tokens": candidate.leaf_size_tokens,
        "iteration": candidate.iteration.get("iteration"),
        "stage_label": candidate.iteration.get("stage_label"),
        "f_artifact": candidate.iteration.get("f_artifact"),
        "g_artifact": candidate.iteration.get("g_artifact"),
        "split_metrics": {
            key: value.__dict__ if hasattr(value, "__dict__") else value
            for key, value in split_metrics.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-root", action="append", default=[], help="DIM=ROOT scalar ladder root.")
    parser.add_argument("--teacher-root", action="append", default=[], help="DIM=TEACHER_ROOT for scalar roots.")
    parser.add_argument("--joint-root", type=Path, default=None)
    parser.add_argument("--joint-teacher-root", type=Path, default=None)
    parser.add_argument("--joint-f-init-path", default="outputs/phase2/joint_gepa/optimized_program.json")
    parser.add_argument("--dimension", action="append", dest="dimensions", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument("--lm-transport", choices=["batch", "litellm"], default="batch")
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--batch-max-concurrent", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-timeout", type=float, default=0.02)
    parser.add_argument("--batch-request-timeout", type=float, default=300.0)
    parser.add_argument("--batch-await-response-timeout", type=float, default=None)
    parser.add_argument("--batch-routing-policy", default="affinity_load_aware")
    parser.add_argument("--context-tokens", type=int, default=51200)
    parser.add_argument("--max-completion-tokens", type=int, default=0)
    parser.add_argument("--prompt-overhead-tokens", type=int, default=1500)
    parser.add_argument("--tokenizer-model", default="/mnt/data/models/google/embeddinggemma-300m")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dimensions = tuple(str(dim) for dim in (args.dimensions or DIMENSION_ORDER))
    scalar_roots = _parse_root_mappings(args.scalar_root)
    teacher_roots = _parse_mapping(args.teacher_root)

    candidates: List[Candidate] = []
    for dim, root in scalar_roots:
        teacher_root = teacher_roots.get(dim, root / "teacher")
        candidates.extend(
            _iter_candidates_for_root(
                root=root,
                teacher_root=teacher_root,
                scalar_dimension=dim,
                is_joint=False,
                split=str(args.split),
            )
        )
    if args.joint_root is not None:
        joint_root = Path(args.joint_root)
        joint_teacher_root = Path(args.joint_teacher_root) if args.joint_teacher_root else joint_root / "teacher"
        candidates.extend(
            _iter_candidates_for_root(
                root=joint_root,
                teacher_root=joint_teacher_root,
                scalar_dimension=None,
                is_joint=True,
                split=str(args.split),
            )
        )

    selected = _select_best_by_dimension(candidates, dimensions)
    if not selected:
        raise SystemExit("No usable saved ladder rows found.")

    unique_candidates: Dict[Tuple[str, int], Candidate] = {}
    selected_rows = []
    for dim in dimensions:
        candidate = selected.get(dim)
        if candidate is None:
            LOGGER.warning("No saved candidate found for %s", dim)
            continue
        score = candidate.score_by_dimension.get(dim)
        key = (str(candidate.manifest_path), int(candidate.iteration["iteration"]))
        unique_candidates[key] = candidate
        selected_rows.append(
            {
                "dimension": dim,
                "score": score,
                "root": str(candidate.root),
                "teacher_root": str(candidate.teacher_root),
                "manifest_path": str(candidate.manifest_path),
                "record_path": str(candidate.record_path),
                "leaf_size_tokens": candidate.leaf_size_tokens,
                "iteration": candidate.iteration.get("iteration"),
                "stage_label": candidate.iteration.get("stage_label"),
                "is_joint": candidate.is_joint,
            }
        )

    payload: Dict[str, Any] = {
        "selected": selected_rows,
        "regenerated": [],
    }
    print(json.dumps({"selected": selected_rows}, indent=2, sort_keys=True))
    if args.plan_only:
        return 0

    for candidate in unique_candidates.values():
        LOGGER.info(
            "Regenerating %s iter=%s stage=%s -> %s",
            candidate.manifest_path.parent.name,
            candidate.iteration.get("iteration"),
            candidate.iteration.get("stage_label"),
            candidate.record_path,
        )
        payload["regenerated"].append(_regenerate_candidate(candidate, args))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
