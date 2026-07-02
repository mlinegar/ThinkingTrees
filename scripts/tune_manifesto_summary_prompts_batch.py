#!/usr/bin/env python3
"""Batch prompt-tune a unified manifesto summarizer from summary-pair traces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dspy

from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.tasks.manifesto.lawstress_bootstrap_program import UnifiedG
from src.tasks.manifesto.lawstress_eval import RILE_RUBRIC


LOGGER = logging.getLogger(__name__)
DEFAULT_STUDENT_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_batch_rubric(
    *,
    base_rubric: str,
    hop: int,
    source_rile_raw: float,
    include_score_conditioning: bool = False,
) -> str:
    lines = [
        str(base_rubric or RILE_RUBRIC).strip(),
        f"Resummary hop: {int(hop)}.",
        "Do not mention numeric scores or the word RILE in the summary.",
    ]
    if include_score_conditioning:
        lines.append(f"Directional signal to preserve (teacher): {float(source_rile_raw):.2f}")
    return "\n".join(lines).strip()


def build_pair_examples(
    rows: Sequence[Dict[str, Any]],
    *,
    base_rubric: str,
    include_score_conditioning: bool = False,
) -> Tuple[List[dspy.Example], List[Dict[str, Any]]]:
    examples: List[dspy.Example] = []
    snapshots: List[Dict[str, Any]] = []
    for row in rows:
        input_text = str(row.get("input_text", "") or "").strip()
        target_summary = str(row.get("target_summary", "") or "").strip()
        if not input_text or not target_summary:
            continue
        hop = int(row.get("hop", 1) or 1)
        source_rile_raw = float(row.get("source_rile_raw", 0.0) or 0.0)
        rubric = build_batch_rubric(
            base_rubric=base_rubric,
            hop=hop,
            source_rile_raw=source_rile_raw,
            include_score_conditioning=bool(include_score_conditioning),
        )
        snapshots.append(
            {
                "id": str(row.get("id", "") or ""),
                "example_id": str(row.get("example_id", "") or ""),
                "split": str(row.get("split", "") or ""),
                "hop": hop,
                "source_rile_raw": source_rile_raw,
                "input_text": input_text,
                "target_summary": target_summary,
                "rubric": rubric,
            }
        )
        examples.append(
            dspy.Example(
                input_text=input_text,
                rubric=rubric,
                target_summary=target_summary,
                hop=hop,
                source_rile_raw=source_rile_raw,
            ).with_inputs("input_text", "rubric")
        )
    return examples, snapshots


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = _tokenize(prediction)
    target_tokens = _tokenize(target)
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, target_counts.get(token, 0))

    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(target_tokens))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _char_bigram_jaccard(prediction: str, target: str) -> float:
    pred = str(prediction or "")
    gold = str(target or "")
    if not pred and not gold:
        return 1.0
    if len(pred) < 2 or len(gold) < 2:
        return 0.0
    pred_grams = {pred[i : i + 2] for i in range(len(pred) - 1)}
    gold_grams = {gold[i : i + 2] for i in range(len(gold) - 1)}
    union = pred_grams | gold_grams
    if not union:
        return 0.0
    return len(pred_grams & gold_grams) / len(union)


class SummaryPairBatchProgram(dspy.Module):
    """Unified summary program over batch-mixed hop examples."""

    def __init__(self, g: Optional[UnifiedG] = None) -> None:
        super().__init__()
        self.g = g or UnifiedG()

    def forward(self, input_text: str, rubric: str) -> dspy.Prediction:
        summary = self.g(content=str(input_text or ""), rubric=str(rubric or ""))
        return dspy.Prediction(summary=str(summary or "").strip())


def create_prompt_batch_metric(
    *,
    max_length_ratio: float,
    length_penalty_weight: float,
    min_length_penalty_input_chars: int,
):
    def _metric(example: dspy.Example, prediction: dspy.Prediction, *_: Any) -> Dict[str, Any]:
        pred_summary = str(getattr(prediction, "summary", "") or "").strip()
        target_summary = str(getattr(example, "target_summary", "") or "").strip()
        input_text = str(getattr(example, "input_text", "") or "")

        token_f1 = _token_f1(pred_summary, target_summary)
        char_sim = _char_bigram_jaccard(pred_summary, target_summary)
        base_score = 0.85 * token_f1 + 0.15 * char_sim

        length_penalty = 0.0
        input_chars = len(input_text)
        if input_chars >= int(min_length_penalty_input_chars):
            ratio = len(pred_summary) / max(1, input_chars)
            overflow = max(0.0, ratio - float(max_length_ratio))
            length_penalty = min(1.0, overflow / max(1e-6, float(max_length_ratio))) * float(length_penalty_weight)

        score = max(0.0, min(1.0, base_score - length_penalty))
        return {
            "score": float(score),
            "token_f1": float(token_f1),
            "char_bigram_jaccard": float(char_sim),
            "length_penalty": float(length_penalty),
        }

    return _metric


def _mean_metric(metric_fn: Any, program: dspy.Module, examples: Sequence[dspy.Example]) -> Dict[str, Any]:
    if not examples:
        return {"n": 0, "mean_score": None}
    scores: List[float] = []
    for example in examples:
        pred = program(
            input_text=getattr(example, "input_text"),
            rubric=getattr(example, "rubric"),
        )
        out = metric_fn(example, pred, None, None, None)
        value = out.get("score") if isinstance(out, dict) else out
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return {"n": len(scores), "mean_score": (sum(scores) / len(scores)) if scores else None}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch prompt-tune unified manifesto summarizer from train/val summary-pair JSONL."
    )
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--eval-pairs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=0,
        help="Max tokens for student DSPy generations (<=0 uses model/context-window default).",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for DSPy calls (default: disabled).",
    )
    parser.add_argument(
        "--gepa-reflection-model",
        type=str,
        default=None,
        help="Optional model id for GEPA reflection (default: same as student model).",
    )
    parser.add_argument(
        "--gepa-reflection-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for GEPA reflection generations.",
    )
    parser.add_argument(
        "--gepa-reflection-max-tokens",
        type=int,
        default=0,
        help="Max tokens for GEPA reflection generations (<=0 uses model/context-window default).",
    )

    parser.add_argument("--gepa-budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-rubric", type=str, default=RILE_RUBRIC)
    parser.add_argument("--max-length-ratio", type=float, default=0.65)
    parser.add_argument("--length-penalty-weight", type=float, default=0.20)
    parser.add_argument("--min-length-penalty-input-chars", type=int, default=200)
    parser.add_argument(
        "--include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include teacher source_rile_raw in rubric text. "
            "Disabled by default to avoid label leakage into deployed prompts."
        ),
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.train_pairs.exists():
        raise FileNotFoundError(f"Train pairs not found: {args.train_pairs}")
    if args.eval_pairs is not None and not args.eval_pairs.exists():
        raise FileNotFoundError(f"Eval pairs not found: {args.eval_pairs}")

    output_dir = args.output_dir or (Path("outputs") / f"summary_prompt_batch_{_now_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(args.train_pairs)
    eval_rows = _load_jsonl(args.eval_pairs) if args.eval_pairs is not None else []

    trainset, train_snapshot = build_pair_examples(
        train_rows,
        base_rubric=str(args.base_rubric),
        include_score_conditioning=bool(args.include_score_conditioning),
    )
    valset, val_snapshot = build_pair_examples(
        eval_rows,
        base_rubric=str(args.base_rubric),
        include_score_conditioning=bool(args.include_score_conditioning),
    )
    if not trainset:
        raise ValueError("No usable train examples after filtering empty rows")

    train_snapshot_path = output_dir / "train_dataset_snapshot.jsonl"
    eval_snapshot_path = output_dir / "eval_dataset_snapshot.jsonl"
    _write_jsonl(train_snapshot_path, train_snapshot)
    _write_jsonl(eval_snapshot_path, val_snapshot)

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "status": "pending",
        "config": {
            "train_pairs": str(args.train_pairs),
            "eval_pairs": str(args.eval_pairs) if args.eval_pairs is not None else None,
            "student_port": int(args.student_port),
            "student_model": str(args.student_model),
            "student_temperature": float(args.student_temperature),
            "student_max_tokens": (int(args.student_max_tokens) if int(args.student_max_tokens) > 0 else None),
            "enable_thinking": bool(args.enable_thinking),
            "gepa_budget": str(args.gepa_budget),
            "gepa_reflection_model": str(args.gepa_reflection_model) if args.gepa_reflection_model else None,
            "gepa_reflection_temperature": float(args.gepa_reflection_temperature),
            "gepa_reflection_max_tokens": (
                int(args.gepa_reflection_max_tokens) if int(args.gepa_reflection_max_tokens) > 0 else None
            ),
            "num_threads": int(args.num_threads),
            "seed": int(args.seed),
            "include_score_conditioning": bool(args.include_score_conditioning),
            "max_length_ratio": float(args.max_length_ratio),
            "length_penalty_weight": float(args.length_penalty_weight),
            "min_length_penalty_input_chars": int(args.min_length_penalty_input_chars),
        },
        "counts": {
            "train_examples": len(trainset),
            "eval_examples": len(valset),
        },
        "artifacts": {
            "train_dataset_snapshot": str(train_snapshot_path),
            "eval_dataset_snapshot": str(eval_snapshot_path),
            "module_dir": str(output_dir / "trained_modules"),
            "unified_g": None,
        },
    }
    if args.dry_run:
        manifest["status"] = "dry_run"
        _write_json(output_dir / "prompt_batch_manifest.json", manifest)
        LOGGER.info("Dry run complete. Output: %s", output_dir)
        return 0

    llm_extra_kwargs: Dict[str, Any] = {}
    if not bool(args.enable_thinking):
        llm_extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    student_max_tokens = int(args.student_max_tokens)
    student_inference = resolve_local_inference_config(
        {
            "port": int(args.student_port),
            "model": str(args.student_model),
            "temperature": float(args.student_temperature),
            "max_tokens": student_max_tokens if student_max_tokens > 0 else None,
        }
    )
    lm = create_local_engine_lm(**student_inference.dspy_kwargs(cache=True), **llm_extra_kwargs)
    configure_dspy(lm=lm)

    reflection_max_tokens = int(args.gepa_reflection_max_tokens)
    reflection_lm = create_local_engine_lm(
        engine=student_inference.engine,
        endpoints=student_inference.endpoints,
        model=str(args.gepa_reflection_model) if args.gepa_reflection_model else str(args.student_model),
        temperature=float(args.gepa_reflection_temperature),
        max_tokens=(reflection_max_tokens if reflection_max_tokens > 0 else None),
        cache=True,
        **llm_extra_kwargs,
    )

    metric_fn = create_prompt_batch_metric(
        max_length_ratio=float(args.max_length_ratio),
        length_penalty_weight=float(args.length_penalty_weight),
        min_length_penalty_input_chars=int(args.min_length_penalty_input_chars),
    )
    # GEPA parallel evaluator expects scalar metric values.
    def metric_for_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        out = metric_fn(gold, pred, trace, pred_name, pred_trace)
        if isinstance(out, dict):
            return float(out.get("score", 0.0))
        return float(out)

    baseline_program = SummaryPairBatchProgram(g=UnifiedG())
    baseline_stats = _mean_metric(metric_fn, baseline_program, valset or trainset)

    optimizer = dspy.GEPA(
        metric=metric_for_gepa,
        auto=str(args.gepa_budget),
        num_threads=int(args.num_threads),
        reflection_lm=reflection_lm,
        log_dir=str(output_dir / "checkpoints" / "gepa" / "summary_prompt_batch"),
        track_stats=True,
        seed=int(args.seed),
    )
    compile_kwargs: Dict[str, Any] = {"student": baseline_program, "trainset": trainset}
    if valset:
        compile_kwargs["valset"] = valset
    optimized_program = optimizer.compile(**compile_kwargs)
    optimized_stats = _mean_metric(metric_fn, optimized_program, valset or trainset)

    module_dir = output_dir / "trained_modules"
    module_dir.mkdir(parents=True, exist_ok=True)
    g_module = getattr(optimized_program, "g", None)
    if g_module is None:
        raise RuntimeError("Optimized prompt program missing unified g module")
    module_path = module_dir / "unified_g_final.json"
    g_module.save(str(module_path))

    manifest["status"] = "completed"
    manifest["metrics"] = {
        "baseline": baseline_stats,
        "optimized": optimized_stats,
    }
    manifest["artifacts"]["unified_g"] = str(module_path)
    _write_json(output_dir / "prompt_batch_manifest.json", manifest)
    LOGGER.info("Batch prompt tuning complete. Unified g: %s", module_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
