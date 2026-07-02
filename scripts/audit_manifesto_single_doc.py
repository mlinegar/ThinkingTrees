#!/usr/bin/env python3
"""Build + audit a single manifesto doc for local-law sanity checks.

This script is intentionally "small loop" friendly and supports two-stage
execution so the student summarizer and teacher scorer do not need to be
running simultaneously:

1) build_tree_only:
   - Uses the student DSPy LM to build baseline + optimized trees
   - Precomputes idempotence + substitution artifacts that require g

2) audit_only:
   - Uses the teacher scorer to evaluate C1/C2/C3 checks over the saved trees

Outputs per variant (baseline/optimized):
  - tree.json
  - audit_report.json
  - summary_root.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.openai_chat import OpenAIChatClient
from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.core.logged_supervision import write_logged_observations_jsonl
from src.core.protocols import format_merge_input
from src.core.scoring import SimilarityScorer, UNIT_SCALE
from src.tasks.manifesto import ManifestoDataset, RILE_PRESERVATION_RUBRIC
from src.tree.builder import BuildConfig, TreeBuilder
from src.tree.auditor import AuditConfig, Auditor
from src.tree.audit_serialization import audit_problem_manifest, audit_report_to_dict
from src.core.strategy import DSPyStrategy
from src.core.data_models import Tree
from src.tasks.manifesto.lawstress_bootstrap_program import UnifiedG
from src.tasks.manifesto.lawstress_generator import normalize_rile


LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"



_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_score(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError):
        return None
    return max(-100.0, min(100.0, value))


def _parse_last_number(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _build_teacher_score_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    cache: Dict[str, float] = {}

    def _score_norm(text: str) -> float:
        rendered = str(text or "")
        key = rendered  # short-lived per-run cache; text may be large but n is small for single doc
        if key in cache:
            return cache[key]
        response = client.chat(
            system="Return exactly one numeric RILE score in [-100,100] for directional information signal.",
            user=(
                "Score this text on a RILE-style directional scale. Return only one number.\n\n"
                f"TEXT:\n{rendered}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _parse_score(response)
        retry = None
        if parsed is None:
            retry = client.chat(
                system=(
                    "Output exactly one numeric RILE score in [-100,100]. "
                    "No words, no explanation, no JSON."
                ),
                user=(
                    "Extract and return only the numeric RILE score.\n"
                    "Output format example: -12.50\n\n"
                    f"TEXT:\n{rendered}"
                ),
                temperature=0.0,
                max_tokens=max(8, int(max_tokens)),
            )
            parsed = _parse_score(retry)
        if parsed is None:
            salvage = _parse_last_number(f"{response}\n{retry or ''}")
            if salvage is not None:
                parsed = max(-100.0, min(100.0, float(salvage)))
        if parsed is None:
            raise ValueError(f"Could not parse score responses: first={response!r} retry={retry!r}")
        norm = float(normalize_rile(float(parsed)))
        cache[key] = norm
        return norm

    return _score_norm


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a single manifesto doc with teacher scorer.")
    parser.add_argument("--id", type=str, default="51320_198306", help="Manifesto id to audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimized-module", type=Path, required=True, help="Path to unified_g_final.json")
    parser.add_argument("--mode", type=str, default="full", choices=["build_tree_only", "audit_only", "full"])

    # Student summarizer LM (DSPy).
    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument("--student-max-tokens", type=int, default=800)

    # Teacher scorer LM.
    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=32)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for teacher scoring calls (default: disabled).",
    )

    # Tree building.
    parser.add_argument("--chunk-size", type=int, default=8000)
    parser.add_argument("--min-chunk-chars", type=int, default=400)

    # Audit thresholds (normalized RILE in [0,1]).
    parser.add_argument("--discrepancy-threshold", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifesto_text(doc_id: str) -> str:
    dataset = ManifestoDataset(require_text=True)
    sample = dataset.get_sample(str(doc_id))
    if sample is None or not sample.text:
        raise ValueError(f"Could not load manifesto text for id={doc_id!r}")
    return str(sample.text)


def _build_tree(
    *,
    text: str,
    rubric: str,
    g_module: UnifiedG,
    chunk_size: int,
    min_chunk_chars: int,
) -> Tree:
    strategy = DSPyStrategy(leaf_module=g_module, merge_module=None, unified_mode=True)
    builder = TreeBuilder(
        strategy=strategy,
        config=BuildConfig(
            max_chunk_chars=int(chunk_size),
            min_chunk_chars=int(min_chunk_chars),
            chunk_strategy="axis",
        ),
    )
    result = builder.build_sync(text, rubric)
    return result.tree


def _precompute_idempotence_and_substitution(tree: Tree, g_module: UnifiedG) -> None:
    # Idempotence: store resummary for internal nodes.
    for node in tree.traverse_preorder():
        if node.is_leaf:
            continue
        original = str(node.summary or "")
        resummary = str(g_module(content=original, rubric=tree.rubric) or "").strip()
        node.metadata["idempotence_resummary"] = resummary

    # Substitution: store joint vs disjoint summaries for adjacent leaf pairs.
    pairs: List[Dict[str, Any]] = []
    leaves = tree.leaves
    for i in range(max(0, len(leaves) - 1)):
        left = leaves[i]
        right = leaves[i + 1]
        raw_left = str(left.raw_text_span or "")
        raw_right = str(right.raw_text_span or "")
        joint_raw = format_merge_input(raw_left, raw_right)
        joint_summary = str(g_module(content=joint_raw, rubric=tree.rubric) or "").strip()

        concat_summaries = format_merge_input(str(left.summary or ""), str(right.summary or ""))
        disjoint_summary = str(g_module(content=concat_summaries, rubric=tree.rubric) or "").strip()

        pairs.append(
            {
                "left_id": left.id,
                "right_id": right.id,
                "joint_summary": joint_summary,
                "disjoint_summary": disjoint_summary,
            }
        )

    tree.metadata["substitution_pairs"] = pairs


def _make_cached_summarizer(tree: Tree) -> Any:
    """Return a summarizer(text, rubric) that replays precomputed g outputs."""

    mapping: Dict[str, str] = {}

    # Leaf raw spans -> leaf summaries (safety fallback for substitution).
    for leaf in tree.leaves:
        raw = str(leaf.raw_text_span or "")
        summ = str(leaf.summary or "")
        if raw and summ:
            mapping[raw] = summ

    # Internal node summaries -> idempotence resummaries.
    for node in tree.traverse_preorder():
        if node.is_leaf:
            continue
        original = str(node.summary or "")
        resummary = str((node.metadata or {}).get("idempotence_resummary") or "")
        if original and resummary:
            mapping[original] = resummary

    # Substitution: joint_raw -> joint_summary and concat_summaries -> disjoint_summary.
    id_map = {n.id: n for n in tree.traverse_preorder()}
    for pair in (tree.metadata or {}).get("substitution_pairs", []) or []:
        left_id = str(pair.get("left_id") or "")
        right_id = str(pair.get("right_id") or "")
        left = id_map.get(left_id)
        right = id_map.get(right_id)
        if left is None or right is None:
            continue
        joint_summary = str(pair.get("joint_summary") or "")
        disjoint_summary = str(pair.get("disjoint_summary") or "")
        if not joint_summary or not disjoint_summary:
            continue

        raw_left = str(left.raw_text_span or "")
        raw_right = str(right.raw_text_span or "")
        joint_raw = format_merge_input(raw_left, raw_right)
        mapping[joint_raw] = joint_summary

        left_summary = str(left.summary or mapping.get(raw_left, "") or "")
        right_summary = str(right.summary or mapping.get(raw_right, "") or "")
        concat_summaries = format_merge_input(left_summary, right_summary)
        mapping[concat_summaries] = disjoint_summary

    def _summarizer(text: str, rubric: str) -> str:
        key = str(text or "")
        if key in mapping:
            return str(mapping[key])
        raise KeyError(
            "Cached summarizer missing key. Re-run build_tree_only to precompute idempotence/substitution artifacts."
        )

    return _summarizer

def _audit_tree_with_auditor(
    *,
    tree: Tree,
    teacher_score_norm: Any,
    discrepancy_threshold: float,
    output_dir: Path,
) -> Dict[str, Any]:
    scorer = SimilarityScorer(
        value_extractor=teacher_score_norm,
        scale=UNIT_SCALE,
        name="rile_norm",
        cache_size=0,
    )
    summarizer = _make_cached_summarizer(tree)
    config = AuditConfig(
        sample_budget=100_000,
        discrepancy_threshold=float(discrepancy_threshold),
        audit_leaves=True,
        audit_internal=True,
        audit_idempotence=True,
        audit_substitution=True,
        idempotence_budget=100_000,
        substitution_budget=100_000,
        sampling_probability=1.0,
    )
    report = Auditor(oracle=scorer, config=config, summarizer=summarizer).audit_tree(tree)
    if report.logged_observations:
        artifact = write_logged_observations_jsonl(
            Path(output_dir) / "logged_observations.jsonl",
            report.logged_observations,
            channel_name="sampled_substructure_supervision",
        )
        report.logged_observation_artifacts = {artifact.channel_name: artifact.to_dict()}
    return audit_report_to_dict(report)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(args.id)
    rubric = str(RILE_PRESERVATION_RUBRIC).strip()
    audit_manifest: Dict[str, Any] = {}

    baseline_dir = output_dir / "baseline"
    optimized_dir = output_dir / "optimized"
    baseline_tree_path = baseline_dir / "tree.json"
    optimized_tree_path = optimized_dir / "tree.json"

    if args.mode in ("build_tree_only", "full"):
        text = _load_manifesto_text(doc_id)

        local_inference = resolve_local_inference_config(
            {
                "port": int(args.student_port),
                "model": args.student_model,
                "temperature": float(args.student_temperature),
                "max_tokens": int(args.student_max_tokens),
            }
        )
        lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
        configure_dspy(lm=lm)

        baseline_g = UnifiedG()
        optimized_g = UnifiedG()
        optimized_g.load(str(args.optimized_module))

        baseline_tree = _build_tree(
            text=text,
            rubric=rubric,
            g_module=baseline_g,
            chunk_size=int(args.chunk_size),
            min_chunk_chars=int(args.min_chunk_chars),
        )
        _precompute_idempotence_and_substitution(baseline_tree, baseline_g)
        baseline_tree.metadata.setdefault("doc_id", doc_id)
        baseline_tree.metadata.setdefault("variant", "baseline")
        baseline_tree.save(baseline_tree_path)
        (baseline_dir / "summary_root.txt").write_text(baseline_tree.final_summary.strip() + "\n", encoding="utf-8")

        optimized_tree = _build_tree(
            text=text,
            rubric=rubric,
            g_module=optimized_g,
            chunk_size=int(args.chunk_size),
            min_chunk_chars=int(args.min_chunk_chars),
        )
        _precompute_idempotence_and_substitution(optimized_tree, optimized_g)
        optimized_tree.metadata.setdefault("doc_id", doc_id)
        optimized_tree.metadata.setdefault("variant", "optimized")
        optimized_tree.save(optimized_tree_path)
        (optimized_dir / "summary_root.txt").write_text(optimized_tree.final_summary.strip() + "\n", encoding="utf-8")

        LOGGER.info("Wrote baseline tree: %s", baseline_tree_path)
        LOGGER.info("Wrote optimized tree: %s", optimized_tree_path)

    if args.mode in ("audit_only", "full"):
        teacher_client = OpenAIChatClient(
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            api_key=args.teacher_api_key,
            timeout_seconds=float(args.teacher_timeout_seconds),
            enable_thinking=bool(args.enable_thinking),
        )
        teacher_score_norm = _build_teacher_score_fn(
            teacher_client,
            temperature=float(args.teacher_temperature),
            max_tokens=int(args.teacher_max_tokens),
        )

        baseline_tree = Tree.load(baseline_tree_path)
        optimized_tree = Tree.load(optimized_tree_path)

        baseline_report = _audit_tree_with_auditor(
            tree=baseline_tree,
            teacher_score_norm=teacher_score_norm,
            discrepancy_threshold=float(args.discrepancy_threshold),
            output_dir=baseline_dir,
        )
        optimized_report = _audit_tree_with_auditor(
            tree=optimized_tree,
            teacher_score_norm=teacher_score_norm,
            discrepancy_threshold=float(args.discrepancy_threshold),
            output_dir=optimized_dir,
        )

        _save_json(baseline_dir / "audit_report.json", baseline_report)
        _save_json(optimized_dir / "audit_report.json", optimized_report)

        compare = {
            "created_at": datetime.utcnow().isoformat(),
            "doc_id": doc_id,
            "threshold": float(args.discrepancy_threshold),
            "baseline": baseline_report.get("violation_rates"),
            "optimized": optimized_report.get("violation_rates"),
        }
        _save_json(output_dir / "compare.json", compare)

        LOGGER.info("Wrote baseline audit: %s", baseline_dir / "audit_report.json")
        LOGGER.info("Wrote optimized audit: %s", optimized_dir / "audit_report.json")
        audit_manifest = {
            "baseline": audit_problem_manifest(baseline_report),
            "optimized": audit_problem_manifest(optimized_report),
        }

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "doc_id": doc_id,
        "mode": str(args.mode),
        "optimized_module": str(Path(args.optimized_module)),
        "paths": {
            "baseline_tree": str(baseline_tree_path),
            "optimized_tree": str(optimized_tree_path),
            "baseline_audit": str(baseline_dir / "audit_report.json"),
            "optimized_audit": str(optimized_dir / "audit_report.json"),
            "compare": str(output_dir / "compare.json"),
        },
    }
    if audit_manifest:
        manifest["audit"] = audit_manifest
    _save_json(output_dir / "manifest.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
