#!/usr/bin/env python3
"""Train/evaluate a raw full-document global DSPy f scorer for Manifesto.

This is the f-only, one-leaf/document baseline:

* the supplied raw Manifesto text is the only leaf and root state;
* g is the identity/no-op because there are no merges;
* one shared DSPy predictor f(document, dimension, rubric) -> score is trained
  over all requested dimensions and documents.

The script intentionally uses the same coverage split, ExperimentContext
sidecars, and RunManifest envelope as the full-doc direct benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.script_io import (  # noqa: E402
    append_jsonl as _append_jsonl,
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    stable_digest as _stable_digest,
    stable_hash as _stable_hash,
    write_json as _write_json,
)
from src.experiments.script_parse import (  # noqa: E402
    parse_csv as _parse_csv,
    safe_float as _safe_float,
)
from src.tasks.manifesto.coverage_split import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    SPLIT_SCHEMA_VERSION,
    load_split_ids as _load_split_ids,
    load_split_summary as _load_split_summary,
    resolve_full_doc_data_dir as _resolve_full_doc_data_dir,
)
from src.tasks.manifesto.full_doc_helpers import (  # noqa: E402
    DIMENSION_BY_NAME as _DIM_FROM_NAME,
    dimension_metrics as _dimension_metrics,
    limit_text as _limit_text,
    load_tokenizer as _load_tokenizer,
    parse_response as _parse_response,
)
from src.tasks.manifesto.result_rows import load_rows_by_dimension as _load_rows_by_dimension  # noqa: E402
from src.ctreepo.contracts import objective_metadata, run_manifest_metadata  # noqa: E402
from src.experiments import (  # noqa: E402
    ARTIFACT_CALLS_JSONL,
    ARTIFACT_PREDICTIONS_JSONL,
    ARTIFACT_SUMMARY_JSON,
    ExperimentContext,
    JsonlCallTraceSink,
    SamplingPlan,
    benchmark_ref_from_parts,
    chat_role_ref,
    experiment_method_ref,
    oracle_ref,
)
from src.tasks.manifesto import ManifestoDataset  # noqa: E402
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context  # noqa: E402
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_NORMALIZED_1_7,
    resolve_benoit_expert_target,
)


SCRIPT_NAME = "scripts/run_manifesto_full_doc_dspy_global_f.py"
TREE_BUNDLE_KIND = "raw_manifesto_single_leaf_document"
METHOD_FAMILY = "full_doc_dspy_global_f"
ADAPTER_ID = "manifesto_full_doc_dspy_global_f"
SCHEMA_VERSION = "ctreepo.manifesto_full_doc_dspy_global_f.v1"


def _dspy_model_name(model: str) -> str:
    return str(model) if str(model).startswith("openai/") else f"openai/{model}"


def _forced_numeric_task_context(dimension: str) -> str:
    base = get_benoit_scoring_context(_DIM_FROM_NAME[str(dimension)])
    return (
        base
        + "\n\nFor this supervised f-training benchmark, always return your best numeric estimate "
        "on the 1-7 scale. Do not return NA. If the document is short, partial, or ambiguous, "
        "infer the most likely expert-mean score from the available text and the training examples. "
        "Return only one number, allowing decimals when appropriate."
    )


def _token_cache_path(
    *,
    token_cache_dir: Optional[Path],
    doc_id: str,
    raw_text: str,
    max_input_chars: int,
    max_input_tokens: int,
    tokenizer_model: str,
) -> Optional[Path]:
    if token_cache_dir is None:
        return None
    payload = {
        "doc_id": str(doc_id),
        "raw_sha256": _stable_hash(raw_text, algorithm="sha256"),
        "raw_chars": len(raw_text),
        "max_input_chars": int(max_input_chars),
        "max_input_tokens": int(max_input_tokens),
        "tokenizer_model": str(tokenizer_model or ""),
        "cache_schema": "ctreepo.manifesto_tokenized_text.v1",
    }
    digest = _stable_digest(payload, length=20)
    safe_doc_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(doc_id))
    return Path(token_cache_dir) / f"{safe_doc_id}.{digest}.json"


def _limit_text_cached(
    raw_text: str,
    *,
    doc_id: str,
    max_input_chars: int,
    max_input_tokens: int,
    tokenizer: Any,
    tokenizer_model: str,
    token_cache_dir: Optional[Path],
) -> Tuple[str, Dict[str, Any]]:
    cache_path = _token_cache_path(
        token_cache_dir=token_cache_dir,
        doc_id=str(doc_id),
        raw_text=raw_text,
        max_input_chars=int(max_input_chars),
        max_input_tokens=int(max_input_tokens),
        tokenizer_model=str(tokenizer_model or ""),
    )
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, Mapping) and isinstance(cached.get("document"), str):
                meta = dict(cached.get("limit_meta") or {})
                meta["token_cache_hit"] = True
                meta["token_cache_path"] = str(cache_path)
                return str(cached["document"]), meta
        except Exception:
            pass
    limited, meta = _limit_text(
        raw_text,
        max_input_chars=int(max_input_chars),
        max_input_tokens=int(max_input_tokens),
        tokenizer=tokenizer,
    )
    meta = dict(meta)
    meta["token_cache_hit"] = False
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": "ctreepo.manifesto_tokenized_text.v1",
                    "doc_id": str(doc_id),
                    "tokenizer_model": str(tokenizer_model or ""),
                    "max_input_chars": int(max_input_chars),
                    "max_input_tokens": int(max_input_tokens),
                    "raw_chars": len(raw_text),
                    "document": limited,
                    "limit_meta": meta,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        meta["token_cache_path"] = str(cache_path)
    return limited, meta


def _prediction_score(pred: Any) -> Optional[float]:
    if isinstance(pred, Mapping):
        for key in ("parsed_score", "score", "prediction"):
            value = _safe_float(pred.get(key))
            if value is not None:
                return max(1.0, min(7.0, value))
        return _parse_response(str(pred.get("raw_score") or ""))
    value = _safe_float(getattr(pred, "parsed_score", None))
    if value is not None:
        return max(1.0, min(7.0, value))
    value = _safe_float(getattr(pred, "score", None))
    if value is not None:
        return max(1.0, min(7.0, value))
    return _parse_response(str(getattr(pred, "score", "") or pred))


def _raw_score(pred: Any) -> str:
    if isinstance(pred, Mapping):
        return str(pred.get("score") or pred.get("raw_score") or "")
    return str(getattr(pred, "score", "") or pred)


def _doc_weight(char_len: int, *, strategy: str, length_floor_chars: int) -> float:
    strategy = str(strategy or "soft_inverse_sqrt_length").strip().lower()
    if strategy in {"uniform", "none"}:
        return 1.0
    floored = max(int(char_len), max(1, int(length_floor_chars)))
    if strategy in {"soft_inverse_length", "inverse_length"}:
        return 1.0 / float(floored)
    if strategy in {"soft_inverse_sqrt_length", "inverse_sqrt_length", "coverage_soft_inverse_sqrt_length"}:
        return 1.0 / math.sqrt(float(floored))
    raise ValueError(f"unknown sampling strategy {strategy!r}")


def _weighted_sample_without_replacement(
    items: Sequence[Mapping[str, Any]],
    *,
    n: Optional[int],
    seed: int,
    strategy: str,
    length_floor_chars: int,
) -> List[Mapping[str, Any]]:
    pool = [dict(item) for item in items]
    if n is None or int(n) < 0 or int(n) >= len(pool):
        return pool
    rng = random.Random(int(seed))
    selected: List[Mapping[str, Any]] = []
    while pool and len(selected) < int(n):
        weights = [
            _doc_weight(
                int(item.get("char_len_full") or 0),
                strategy=strategy,
                length_floor_chars=length_floor_chars,
            )
            for item in pool
        ]
        total = sum(weights)
        if total <= 0.0:
            idx = rng.randrange(len(pool))
        else:
            draw = rng.random() * total
            running = 0.0
            idx = len(pool) - 1
            for candidate, weight in enumerate(weights):
                running += float(weight)
                if running >= draw:
                    idx = candidate
                    break
        selected.append(pool.pop(idx))
    return selected


def _split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "val": 100_000, "validation": 100_000, "test": 200_000}
    return int(seed) + offsets.get(str(split), 300_000)


def _load_doc_items(
    *,
    split_ids: Mapping[str, Sequence[str]],
    rows_by_dimension: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dimensions: Sequence[str],
    dataset: ManifestoDataset,
    split: str,
    max_docs: Optional[int],
    seed: int,
    sampling_strategy: str,
    length_floor_chars: int,
    max_input_chars: int,
    max_input_tokens: int,
    min_doc_chars: int,
    tokenizer: Any,
    tokenizer_model: str,
    token_cache_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_items: List[Dict[str, Any]] = []
    for doc_id in split_ids.get(str(split), []):
        sample = dataset.get_sample(str(doc_id))
        raw_text = str(getattr(sample, "text", "") or "") if sample is not None else ""
        if not raw_text.strip():
            continue
        if int(min_doc_chars) > 0 and len(raw_text) < int(min_doc_chars):
            continue
        limited_text, limit_meta = _limit_text_cached(
            raw_text,
            doc_id=str(doc_id),
            max_input_chars=max_input_chars,
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
            tokenizer_model=str(tokenizer_model or ""),
            token_cache_dir=token_cache_dir,
        )
        raw_items.append(
            {
                "manifesto_id": str(doc_id),
                "split": str(split),
                "document": limited_text,
                "char_len_full": len(raw_text),
                "char_len_input": len(limited_text),
                **limit_meta,
            }
        )

    selected_docs = _weighted_sample_without_replacement(
        raw_items,
        n=max_docs,
        seed=_split_seed(seed, split),
        strategy=sampling_strategy,
        length_floor_chars=length_floor_chars,
    )
    examples: List[Dict[str, Any]] = []
    for item in selected_docs:
        doc_id = str(item["manifesto_id"])
        for dim in dimensions:
            source_row = rows_by_dimension.get(str(dim), {}).get(doc_id)
            if source_row is None:
                continue
            expert = resolve_benoit_expert_target(
                source_row,
                dimension=str(dim),
                scale=EXPERT_SCALE_NORMALIZED_1_7,
            )
            if expert is None:
                continue
            examples.append(
                {
                    **item,
                    "dimension": str(dim),
                    "task_context": _forced_numeric_task_context(str(dim)),
                    "expert_score_1_7": float(expert),
                }
            )
    return selected_docs, examples


def _examples_for_dspy(rows: Sequence[Mapping[str, Any]]) -> List[Any]:
    import dspy

    return [
        dspy.Example(
            dimension=str(row["dimension"]),
            task_context=str(row["task_context"]),
            document=str(row["document"]),
            score=str(float(row["expert_score_1_7"])),
        ).with_inputs("dimension", "task_context", "document")
        for row in rows
    ]


def _build_context(
    *,
    output_dir: Path,
    split_dir: Path,
    split_summary: Mapping[str, Any],
    dimensions: Sequence[str],
    model: str,
    base_url: str,
    seed: int,
    command: Sequence[str],
) -> ExperimentContext:
    split_digest = str(split_summary.get("split_manifest_digest") or "")
    benchmark_ref = benchmark_ref_from_parts(
        family="manifesto_rile",
        scope="full_doc_global_f",
        dataset_id=str(split_dir),
        name="Manifesto all-six raw full-document global f",
        metadata={
            "split_manifest_digest": split_digest,
            "dimensions": list(dimensions),
            "tree_bundle_kind": TREE_BUNDLE_KIND,
        },
    )
    method_ref = experiment_method_ref(
        family=METHOD_FAMILY,
        variant="raw_document_one_leaf",
        engine="dspy_vllm_openai",
        model=str(model),
        adapter=ADAPTER_ID,
        roles={
            "scorer": chat_role_ref(
                role="scorer",
                engine="vllm",
                model=str(model),
                base_url=str(base_url),
            )
        },
        oracle=oracle_ref(
            kind="benchmark_labels",
            source="benoit_expert_means",
            metadata={"target_scale": EXPERT_SCALE_NORMALIZED_1_7},
        ),
        metadata={
            "split_manifest_digest": split_digest,
            "tree_bundle_kind": TREE_BUNDLE_KIND,
            "g_init": "identity_single_leaf",
        },
    )
    sampling_plan = SamplingPlan(
        seed=int(seed),
        split="train,val,test",
        strategy=str(
            ((split_summary.get("sampling_plan") or {}).get("strategy"))
            or "manifesto_coverage_split"
        ),
        unit="document",
        frame="manifesto_all6_labeled",
        metadata={
            "split_manifest_digest": split_digest,
            "split_dir": str(split_dir),
            "dimensions": list(dimensions),
            "tree_bundle_kind": TREE_BUNDLE_KIND,
        },
    )
    return ExperimentContext(
        output_root=output_dir,
        benchmark_ref=benchmark_ref,
        method_ref=method_ref,
        title="manifesto_full_doc_dspy_global_f",
        adapter_id=ADAPTER_ID,
        phases=("train", "evaluate"),
        sampling=sampling_plan,
        report_profiles=("runtime_eval_summary",),
        launch_command=tuple(str(item) for item in command),
        metadata={
            "split_summary_schema_version": str(split_summary.get("schema_version") or ""),
            "split_manifest_digest": split_digest,
            "tree_bundle_kind": TREE_BUNDLE_KIND,
            "tree_state_source": "raw_manifesto_text",
            "single_leaf_per_document": True,
        },
    )


def _configure_dspy_lm(args: argparse.Namespace) -> Any:
    import dspy
    from src.config.dspy_config import configure_dspy
    from src.core.dspy_batch_client import BatchedDSPyLM

    lm_kwargs = {
        "model": _dspy_model_name(str(args.model)),
        "api_base": str(args.base_url).rstrip("/"),
        "api_key": str(args.api_key),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "cache": not bool(args.disable_dspy_cache),
        "timeout": float(args.timeout_seconds),
        "num_retries": int(args.num_retries),
    }
    if bool(args.use_batched_lm):
        lm = BatchedDSPyLM(
            model=lm_kwargs["model"],
            api_base=lm_kwargs["api_base"],
            api_key=lm_kwargs["api_key"],
            temperature=lm_kwargs["temperature"],
            max_tokens=lm_kwargs["max_tokens"],
            cache=lm_kwargs["cache"],
            max_concurrent=int(args.batch_max_concurrent),
            batch_size=int(args.batch_size),
            batch_timeout=float(args.batch_timeout),
            request_timeout=float(args.timeout_seconds),
            await_response_timeout=float(args.timeout_seconds),
        )
    else:
        lm = dspy.LM(**lm_kwargs)
    configure_dspy(lm=lm)
    return lm


def _make_mipro_prompt_lm(args: argparse.Namespace) -> Any:
    """Use a separate output budget for instruction proposals."""
    import dspy

    return dspy.LM(
        model=_dspy_model_name(str(args.model)),
        api_base=str(args.base_url).rstrip("/"),
        api_key=str(args.api_key),
        temperature=float(args.mipro_prompt_temperature),
        max_tokens=int(args.mipro_prompt_max_tokens),
        cache=not bool(args.disable_dspy_cache),
        timeout=float(args.timeout_seconds),
        num_retries=int(args.num_retries),
    )


def _make_program(max_output_tokens: int) -> Any:
    import dspy

    class FullDocGlobalFSignature(dspy.Signature):
        """Predict the expert-mean 1-7 score for one policy dimension.

        Use only the provided raw manifesto text and task context. Always output
        one numeric 1-7 score. Never output NA.
        """

        dimension: str = dspy.InputField(desc="Policy dimension name.")
        task_context: str = dspy.InputField(desc="Dimension scale and scoring guidance.")
        document: str = dspy.InputField(desc="Raw manifesto document text. This is the one leaf/root state.")
        score: str = dspy.OutputField(desc="Single numeric score from 1 to 7. Never NA.")

    class FullDocGlobalF(dspy.Module):
        def __init__(self, *, max_output_tokens: int) -> None:
            super().__init__()
            self.max_output_tokens = int(max_output_tokens)
            self.predictor = dspy.Predict(FullDocGlobalFSignature)

        def forward(self, dimension: str, task_context: str, document: str) -> Any:
            result = self.predictor(
                dimension=dimension,
                task_context=task_context,
                document=document,
                config={"max_tokens": self.max_output_tokens},
            )
            raw = str(getattr(result, "score", "") or "")
            parsed = _parse_response(raw)
            return dspy.Prediction(score=raw, parsed_score=parsed)

    return FullDocGlobalF(max_output_tokens=max_output_tokens)


def _metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
    del trace, unused, kwargs
    target = _safe_float(getattr(gold, "score", None))
    prediction = _prediction_score(pred)
    if target is None or prediction is None:
        return 0.0
    return max(0.0, 1.0 - abs(float(prediction) - float(target)) / 6.0)


def _compile_program(
    *,
    args: argparse.Namespace,
    program: Any,
    trainset: Sequence[Any],
    valset: Sequence[Any],
) -> Any:
    import dspy

    optimizer_name = str(args.optimizer).strip().lower()
    if optimizer_name in {"none", "zero_shot", "zeroshot"}:
        return program
    if optimizer_name == "mipro":
        optimizer_cls = dspy.MIPROv2
        if bool(args.mipro_skip_bootstrap):
            class InstructionOnlyMIPRO(dspy.MIPROv2):
                def _bootstrap_fewshot_examples(self, program: Any, trainset: list, seed: int, teacher: Any) -> list | None:
                    del program, trainset, seed, teacher
                    import logging

                    logging.getLogger("dspy.teleprompt.mipro_optimizer_v2").info(
                        "\n==> STEP 1: SKIP FEWSHOT EXAMPLE BOOTSTRAP <==\n"
                        "Instruction-only mode: no bootstrapped/labeled examples are built for prompt context."
                    )
                    return None

            optimizer_cls = InstructionOnlyMIPRO
        manual = args.mipro_num_trials is not None or args.mipro_num_candidates is not None
        num_candidates = args.mipro_num_candidates
        if manual and num_candidates is None:
            num_candidates = max(1, int(args.mipro_num_trials or 1))
        prompt_model = None
        if int(args.mipro_prompt_max_tokens) > 0:
            prompt_model = _make_mipro_prompt_lm(args)
        optimizer = optimizer_cls(
            metric=_metric,
            prompt_model=prompt_model,
            auto=None if manual else str(args.dspy_budget),
            num_candidates=num_candidates,
            num_threads=int(args.dspy_num_threads),
            max_bootstrapped_demos=int(args.max_bootstrapped_demos),
            max_labeled_demos=int(args.max_labeled_demos),
            seed=int(args.seed),
        )
        compile_kwargs: Dict[str, Any] = {
            "trainset": list(trainset),
            "valset": list(valset),
        }
        if args.mipro_num_trials is not None:
            compile_kwargs["num_trials"] = int(args.mipro_num_trials)
        if args.mipro_minibatch_size is not None:
            compile_kwargs["minibatch_size"] = int(args.mipro_minibatch_size)
        if args.mipro_minibatch_full_eval_steps is not None:
            compile_kwargs["minibatch_full_eval_steps"] = int(args.mipro_minibatch_full_eval_steps)
        compile_kwargs["program_aware_proposer"] = bool(args.mipro_program_aware_proposer)
        compile_kwargs["data_aware_proposer"] = bool(args.mipro_data_aware_proposer)
        compile_kwargs["tip_aware_proposer"] = bool(args.mipro_tip_aware_proposer)
        compile_kwargs["fewshot_aware_proposer"] = bool(args.mipro_fewshot_aware_proposer)
        compile_kwargs["view_data_batch_size"] = int(args.mipro_view_data_batch_size)
        return optimizer.compile(program, **compile_kwargs)
    if optimizer_name in {"bootstrap", "bootstrap_fewshot"}:
        optimizer = dspy.BootstrapFewShot(
            metric=_metric,
            max_bootstrapped_demos=int(args.max_bootstrapped_demos),
            max_labeled_demos=int(args.max_labeled_demos),
            max_errors=int(args.max_errors),
        )
        try:
            return optimizer.compile(program, trainset=list(trainset), valset=list(valset))
        except TypeError:
            return optimizer.compile(program, trainset=list(trainset))
    raise ValueError(f"unsupported optimizer {args.optimizer!r}")


def _save_program(program: Any, program_root: Path) -> Dict[str, Any]:
    """Save both a loadable DSPy program and a lightweight state JSON."""
    program_root.mkdir(parents=True, exist_ok=True)
    program_dir = program_root / "dspy_program"
    state_json = program_root / "program_state.json"
    result: Dict[str, Any] = {
        "ok": False,
        "program_dir": str(program_dir),
        "program_state_json": str(state_json),
        "errors": [],
    }
    try:
        program.save(str(program_dir), save_program=True)
        result["program_dir_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["program_dir_ok"] = False
        result["errors"].append({"artifact": "program_dir", "error": str(exc)})

    try:
        program.save(str(state_json), save_program=False)
        result["program_state_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["program_state_ok"] = False
        result["errors"].append({"artifact": "program_state_json", "error": str(exc)})

    result["ok"] = bool(result.get("program_dir_ok") or result.get("program_state_ok"))
    if not result["ok"]:
        err_path = program_root / "program.save_error.json"
        _write_json(err_path, result)
        result["error_path"] = str(err_path)
    return result


def _score_one(program: Any, row: Mapping[str, Any], *, mock_predictions: bool) -> Dict[str, Any]:
    started = time.time()
    error = ""
    if mock_predictions:
        prediction = max(1.0, min(7.0, round(float(row["expert_score_1_7"]))))
        raw_response = str(prediction)
    else:
        try:
            pred = program(
                dimension=str(row["dimension"]),
                task_context=str(row["task_context"]),
                document=str(row["document"]),
            )
            prediction = _prediction_score(pred)
            raw_response = _raw_score(pred)
        except Exception as exc:  # noqa: BLE001
            prediction = None
            raw_response = ""
            error = str(exc)
    latency_ms = 1000.0 * (time.time() - started)
    out = {
        "manifesto_id": str(row["manifesto_id"]),
        "split": str(row["split"]),
        "dimension": str(row["dimension"]),
        "expert_score_1_7": float(row["expert_score_1_7"]),
        "prediction": prediction,
        "is_na": prediction is None,
        "raw_response": str(raw_response)[:500],
        "error": error,
        "latency_ms": latency_ms,
        "char_len_full": int(row.get("char_len_full") or 0),
        "char_len_input": int(row.get("char_len_input") or 0),
        "truncated": bool(row.get("truncated")),
        "limit_kind": str(row.get("limit_kind") or ""),
        "input_tokens_estimated": int(row.get("input_tokens_estimated") or 0),
        "full_tokens_estimated": int(row.get("full_tokens_estimated") or 0),
        "coverage_ratio": float(row.get("coverage_ratio") or 0.0),
    }
    return out


def _evaluate_rows(
    *,
    program: Any,
    rows: Sequence[Mapping[str, Any]],
    context: ExperimentContext,
    calls_path: Path,
    predictions_path: Path,
    model: str,
    eval_num_threads: int,
    mock_predictions: bool,
) -> List[Dict[str, Any]]:
    predictions_live_path = predictions_path.with_name(f"{predictions_path.stem}.live{predictions_path.suffix}")
    progress_path = predictions_path.with_name("prediction_progress.json")
    for stale_path in (predictions_live_path, progress_path):
        if stale_path.exists():
            stale_path.unlink()
    call_sink = JsonlCallTraceSink(
        calls_path,
        defaults={"surface": "dspy", "engine": "vllm", "model": model},
    )
    predictions: List[Dict[str, Any]] = []
    total = len(rows)
    started_at = time.time()

    def run(row: Mapping[str, Any]) -> Dict[str, Any]:
        return _score_one(program, row, mock_predictions=mock_predictions)

    by_key = {
        (str(row["manifesto_id"]), str(row["dimension"]), str(row["split"])): dict(row)
        for row in rows
    }

    def record(pred: Dict[str, Any]) -> None:
        key = (str(pred["manifesto_id"]), str(pred["dimension"]), str(pred["split"]))
        source = by_key.get(key, {})
        call_meta = context.call_metadata(
            role="scorer",
            request_kind="full_doc_dspy_global_f_score",
            problem_id=f"{pred['manifesto_id']}:{pred['dimension']}",
            runner_id=ADAPTER_ID,
            artifacts={
                "predictions_jsonl": str(predictions_path),
                "predictions_live_jsonl": str(predictions_live_path),
                "prediction_progress_json": str(progress_path),
            },
        )
        call_sink(
            {
                **call_meta,
                "document_id": str(pred["manifesto_id"]),
                "unit_id": f"{pred['manifesto_id']}:{pred['dimension']}",
                "latency_ms": float(pred.get("latency_ms") or 0.0),
                "usage": {},
                "error": str(pred.get("error") or ""),
                "metadata": {
                    "dimension": str(pred["dimension"]),
                    "split": str(pred["split"]),
                    "tree_bundle_kind": TREE_BUNDLE_KIND,
                    "char_len_full": int(source.get("char_len_full") or pred.get("char_len_full") or 0),
                    "truncated": bool(source.get("truncated") or pred.get("truncated")),
                },
            }
        )
        predictions.append(pred)
        _append_jsonl(predictions_live_path, [pred])
        completed = len(predictions)
        elapsed = max(0.0, time.time() - started_at)
        rate = completed / elapsed if elapsed > 0 else None
        remaining = (total - completed) / rate if rate else None
        _write_json(
            progress_path,
            {
                "schema_version": f"{SCHEMA_VERSION}.prediction_progress.v1",
                "completed": completed,
                "total": total,
                "percent_complete": (100.0 * completed / total) if total else 100.0,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": remaining,
                "predictions_live_jsonl": str(predictions_live_path),
                "calls_jsonl": str(calls_path),
                "final_predictions_jsonl": str(predictions_path),
                "last_completed": {
                    "manifesto_id": str(pred["manifesto_id"]),
                    "dimension": str(pred["dimension"]),
                    "split": str(pred["split"]),
                },
            },
        )

    if int(eval_num_threads) <= 1:
        for row in rows:
            record(run(row))
    else:
        with ThreadPoolExecutor(max_workers=int(eval_num_threads)) as executor:
            futures = [executor.submit(run, row) for row in rows]
            for future in as_completed(futures):
                record(future.result())
    predictions.sort(key=lambda row: (str(row["split"]), str(row["manifesto_id"]), str(row["dimension"])))
    return predictions


def _length_summary(docs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lengths = sorted(int(row.get("char_len_full") or 0) for row in docs)
    if not lengths:
        return {"doc_count": 0}
    def pct(q: float) -> int:
        idx = min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))
        return lengths[idx]
    return {
        "doc_count": len(lengths),
        "min_chars": lengths[0],
        "p25_chars": pct(0.25),
        "median_chars": pct(0.50),
        "p75_chars": pct(0.75),
        "max_chars": lengths[-1],
    }


def run_global_f(args: argparse.Namespace) -> Dict[str, Any]:
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = _load_split_ids(split_dir)
    split_summary = _load_split_summary(split_dir)
    dimensions = _parse_csv(args.dimensions or ",".join(split_summary.get("dimensions") or DEFAULT_DIMENSIONS))
    source_root = Path(args.source_root or split_summary.get("source_root") or "outputs/overnight_benoit/full_pipeline")
    rows_by_dimension = _load_rows_by_dimension(source_root, dimensions)
    resolved_mp_data_dir = _resolve_full_doc_data_dir(args.mp_data_dir, project_root=PROJECT_ROOT)
    dataset = ManifestoDataset(data_dir=resolved_mp_data_dir, require_text=True)
    tokenizer = _load_tokenizer(str(args.tokenizer_model)) if args.tokenizer_model else None
    token_cache_dir = Path(args.token_cache_dir) if args.token_cache_dir else output_dir / "token_cache"
    train_max_input_chars = (
        int(args.train_max_input_chars)
        if args.train_max_input_chars is not None
        else int(args.max_input_chars)
    )
    val_max_input_chars = (
        int(args.val_max_input_chars)
        if args.val_max_input_chars is not None
        else int(args.max_input_chars)
    )
    test_max_input_chars = (
        int(args.test_max_input_chars)
        if args.test_max_input_chars is not None
        else int(args.max_input_chars)
    )
    train_max_input_tokens = (
        int(args.train_max_input_tokens)
        if args.train_max_input_tokens is not None
        else int(args.max_input_tokens)
    )
    val_max_input_tokens = (
        int(args.val_max_input_tokens)
        if args.val_max_input_tokens is not None
        else int(args.max_input_tokens)
    )
    test_max_input_tokens = (
        int(args.test_max_input_tokens)
        if args.test_max_input_tokens is not None
        else int(args.max_input_tokens)
    )
    model = str(args.model)
    base_url = str(args.base_url).rstrip("/")
    context = _build_context(
        output_dir=output_dir,
        split_dir=split_dir,
        split_summary=split_summary,
        dimensions=dimensions,
        model=model,
        base_url=base_url,
        seed=int(args.seed),
        command=sys.argv,
    )

    train_docs, train_rows = _load_doc_items(
        split_ids=split_ids,
        rows_by_dimension=rows_by_dimension,
        dimensions=dimensions,
        dataset=dataset,
        split="train",
        max_docs=args.train_docs,
        seed=int(args.seed),
        sampling_strategy=str(args.train_sampling_strategy),
        length_floor_chars=int(args.length_floor_chars),
        max_input_chars=train_max_input_chars,
        max_input_tokens=train_max_input_tokens,
        min_doc_chars=int(args.min_doc_chars),
        tokenizer=tokenizer,
        tokenizer_model=str(args.tokenizer_model or ""),
        token_cache_dir=token_cache_dir,
    )
    val_docs, val_rows = _load_doc_items(
        split_ids=split_ids,
        rows_by_dimension=rows_by_dimension,
        dimensions=dimensions,
        dataset=dataset,
        split="val",
        max_docs=args.val_docs,
        seed=int(args.seed),
        sampling_strategy="uniform",
        length_floor_chars=int(args.length_floor_chars),
        max_input_chars=val_max_input_chars,
        max_input_tokens=val_max_input_tokens,
        min_doc_chars=int(args.min_doc_chars),
        tokenizer=tokenizer,
        tokenizer_model=str(args.tokenizer_model or ""),
        token_cache_dir=token_cache_dir,
    )
    test_docs, test_rows = _load_doc_items(
        split_ids=split_ids,
        rows_by_dimension=rows_by_dimension,
        dimensions=dimensions,
        dataset=dataset,
        split="test",
        max_docs=args.test_docs,
        seed=int(args.seed),
        sampling_strategy="uniform",
        length_floor_chars=int(args.length_floor_chars),
        max_input_chars=test_max_input_chars,
        max_input_tokens=test_max_input_tokens,
        min_doc_chars=int(args.min_doc_chars),
        tokenizer=tokenizer,
        tokenizer_model=str(args.tokenizer_model or ""),
        token_cache_dir=token_cache_dir,
    )
    if not train_rows:
        raise RuntimeError("no training rows resolved from split/source labels")
    if not val_rows:
        raise RuntimeError("no validation rows resolved from split/source labels")
    if not test_rows:
        raise RuntimeError("no test rows resolved from split/source labels")

    trainset = _examples_for_dspy(train_rows)
    valset = _examples_for_dspy(val_rows)
    if args.initial_program_dir:
        import dspy

        # Local, self-produced program artifacts (cloudpickle) are trusted.
        program = dspy.load(str(args.initial_program_dir), allow_pickle=True)
        if hasattr(program, "max_output_tokens"):
            program.max_output_tokens = int(args.max_tokens)
        f_init = "loaded_dspy_global_f_raw_document"
    else:
        program = _make_program(max_output_tokens=int(args.max_tokens))
        f_init = "dspy_global_f_raw_document"
    compile_started = time.time()
    if bool(args.mock_predictions):
        compiled = program
    else:
        _configure_dspy_lm(args)
        compiled = _compile_program(
            args=args,
            program=program,
            trainset=trainset,
            valset=valset,
        )
    compile_seconds = time.time() - compile_started

    program_save = _save_program(compiled, output_dir / "program")
    program_dir_path = str(program_save.get("program_dir") or (output_dir / "program" / "dspy_program"))
    program_state_path = str(
        program_save.get("program_state_json") or (output_dir / "program" / "program_state.json")
    )
    examples_manifest = {
        "schema_version": f"{SCHEMA_VERSION}.examples",
        "tree_bundle_kind": TREE_BUNDLE_KIND,
        "dimensions": list(dimensions),
        "train_docs": len(train_docs),
        "train_examples": len(train_rows),
        "val_docs": len(val_docs),
        "val_examples": len(val_rows),
        "test_docs": len(test_docs),
        "test_examples": len(test_rows),
        "length_summary": {
            "train": _length_summary(train_docs),
            "val": _length_summary(val_docs),
            "test": _length_summary(test_docs),
        },
        "train_sampling_strategy": str(args.train_sampling_strategy),
        "single_leaf_per_document": True,
    }
    examples_manifest_path = _write_json(output_dir / "examples_manifest.json", examples_manifest)

    predictions_path = output_dir / "predictions.jsonl"
    calls_path = output_dir / "calls.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    if calls_path.exists():
        calls_path.unlink()
    prediction_rows = _evaluate_rows(
        program=compiled,
        rows=test_rows,
        context=context,
        calls_path=calls_path,
        predictions_path=predictions_path,
        model=model,
        eval_num_threads=int(args.eval_num_threads),
        mock_predictions=bool(args.mock_predictions),
    )
    _append_jsonl(predictions_path, prediction_rows)

    per_dimension = {dim: _dimension_metrics(prediction_rows, dimension=dim) for dim in dimensions}
    pearsons = [
        float(metrics["pearson"].get("pearson_r"))
        for metrics in per_dimension.values()
        if metrics["pearson"].get("pearson_r") is not None
    ]
    macro_pearson = sum(pearsons) / len(pearsons) if pearsons else None
    metrics: Dict[str, Any] = {
        "macro_external_expert_pearson": macro_pearson,
        "prediction_rows": len(prediction_rows),
        "na_count": sum(1 for row in prediction_rows if row.get("prediction") is None),
        "compile_seconds": compile_seconds,
        "train_docs": len(train_docs),
        "train_examples": len(train_rows),
        "val_docs": len(val_docs),
        "val_examples": len(val_rows),
        "test_docs": len(test_docs),
        "test_examples": len(test_rows),
    }
    for dim, dim_metrics in per_dimension.items():
        metrics[f"{dim}_external_expert_pearson"] = dim_metrics["pearson"].get("pearson_r")
        metrics[f"{dim}_mae"] = dim_metrics.get("mae")
        metrics[f"{dim}_n_scored"] = dim_metrics.get("n_scored")

    summary = {
        "created_at": _now_iso(),
        "schema_version": SCHEMA_VERSION,
        "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
        "split_summary_schema_version": str(split_summary.get("schema_version") or ""),
        "split_dir": str(split_dir),
        "source_root": str(source_root),
        "mp_data_dir": str(resolved_mp_data_dir or ""),
        "text_source_kind": "raw_manifesto_full_document",
        "dimensions": list(dimensions),
        "model": model,
        "base_url": base_url,
        "optimizer": str(args.optimizer),
        "dspy_budget": str(args.dspy_budget),
        "max_input_chars": int(args.max_input_chars),
        "max_input_tokens": int(args.max_input_tokens),
        "split_input_caps": {
            "train": {
                "max_input_chars": train_max_input_chars,
                "max_input_tokens": train_max_input_tokens,
            },
            "val": {
                "max_input_chars": val_max_input_chars,
                "max_input_tokens": val_max_input_tokens,
            },
            "test": {
                "max_input_chars": test_max_input_chars,
                "max_input_tokens": test_max_input_tokens,
            },
        },
        "tokenizer_model": str(args.tokenizer_model or ""),
        "token_cache_dir": str(token_cache_dir),
        "tree_bundle_kind": TREE_BUNDLE_KIND,
        "g_init": "identity_single_leaf",
        "f_training_surface": "raw_manifesto_document",
        "min_doc_chars": int(args.min_doc_chars),
        "mock_predictions": bool(args.mock_predictions),
        "initial_program_dir": str(args.initial_program_dir) if args.initial_program_dir else "",
        "program_save": program_save,
        "examples_manifest": str(examples_manifest_path),
        "per_dimension": per_dimension,
        "metrics": metrics,
        "artifacts": {
            ARTIFACT_PREDICTIONS_JSONL: str(predictions_path),
            ARTIFACT_CALLS_JSONL: str(calls_path),
            ARTIFACT_SUMMARY_JSON: str(output_dir / "summary.json"),
            "program_dir": program_dir_path,
            "program_state_json": program_state_path,
            "program_json": program_state_path,
            "examples_manifest_json": str(examples_manifest_path),
        },
    }
    summary_path = _write_json(output_dir / "summary.json", summary)
    artifacts = {
        ARTIFACT_PREDICTIONS_JSONL: str(predictions_path),
        ARTIFACT_CALLS_JSONL: str(calls_path),
        ARTIFACT_SUMMARY_JSON: str(summary_path),
        "program_dir": program_dir_path,
        "program_state_json": program_state_path,
        "program_json": program_state_path,
        "examples_manifest_json": str(examples_manifest_path),
    }
    context.record(
        {"metrics": metrics, "artifacts": artifacts, "metadata": {"summary": METHOD_FAMILY}},
        phase="evaluate",
        artifacts=artifacts,
        metadata={
            "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
            "dimensions": list(dimensions),
            "tree_bundle_kind": TREE_BUNDLE_KIND,
            "single_leaf_per_document": True,
        },
        train_docs=len(train_docs),
        state="completed",
    )

    run_manifest = run_manifest_metadata(
        run_id=f"manifesto.full_doc_dspy_global_f.{_now_stamp()}",
        domain="manifesto_rile",
        role="full_doc_dspy_global_f",
        backend="dspy",
        status="completed",
        input_contracts=[
            {
                "kind": "manifesto_coverage_split",
                "schema_version": SPLIT_SCHEMA_VERSION,
                "digest": str(split_summary.get("split_manifest_digest") or ""),
                "uri": str(split_dir),
            },
            {
                "kind": "tree_bundle",
                "schema_version": "ctreepo.tree_bundle_manifest.v1",
                "tree_bundle_kind": TREE_BUNDLE_KIND,
                "tree_state_source": "raw_manifesto_text",
                "single_leaf_per_document": True,
            },
        ],
        f_init=f_init,
        g_init="identity_single_leaf",
        f_lineage={
            "model": model,
            "base_url": base_url,
            "optimizer": str(args.optimizer),
            "initial_program_dir": str(args.initial_program_dir) if args.initial_program_dir else "",
            "program_dir": program_dir_path,
            "program_state_json": program_state_path,
            "program_json": program_state_path,
        },
        g_lineage={"init": "identity_single_leaf", "merge_depth": 0},
        objective=objective_metadata(
            objective_family="manifesto_root_expert_supervised_global_f",
            local_law_estimator="none",
            root_share=1.0,
            local_law_component_weights={},
            metadata={
                "dimensions": list(dimensions),
                "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
                "target_scale": EXPERT_SCALE_NORMALIZED_1_7,
                "tree_bundle_kind": TREE_BUNDLE_KIND,
            },
        ),
        optimizer_config={
            "optimizer": str(args.optimizer),
            "dspy_budget": str(args.dspy_budget),
            "dspy_num_threads": int(args.dspy_num_threads),
            "max_bootstrapped_demos": int(args.max_bootstrapped_demos),
            "max_labeled_demos": int(args.max_labeled_demos),
            "mipro_num_trials": args.mipro_num_trials,
            "mipro_num_candidates": args.mipro_num_candidates,
            "mipro_minibatch_size": args.mipro_minibatch_size,
            "mipro_minibatch_full_eval_steps": args.mipro_minibatch_full_eval_steps,
            "mipro_program_aware_proposer": bool(args.mipro_program_aware_proposer),
            "mipro_data_aware_proposer": bool(args.mipro_data_aware_proposer),
            "mipro_tip_aware_proposer": bool(args.mipro_tip_aware_proposer),
            "mipro_fewshot_aware_proposer": bool(args.mipro_fewshot_aware_proposer),
            "mipro_view_data_batch_size": int(args.mipro_view_data_batch_size),
            "mipro_skip_bootstrap": bool(args.mipro_skip_bootstrap),
            "mipro_prompt_max_tokens": int(args.mipro_prompt_max_tokens),
            "mipro_prompt_temperature": float(args.mipro_prompt_temperature),
            "initial_program_dir": str(args.initial_program_dir) if args.initial_program_dir else "",
            "token_cache_dir": str(token_cache_dir),
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "train_sampling_strategy": str(args.train_sampling_strategy),
            "min_doc_chars": int(args.min_doc_chars),
        },
        output_artifacts=[
            {"kind": ARTIFACT_SUMMARY_JSON, "uri": str(summary_path)},
            {"kind": ARTIFACT_PREDICTIONS_JSONL, "uri": str(predictions_path)},
            {"kind": ARTIFACT_CALLS_JSONL, "uri": str(calls_path)},
            {"kind": "program_dir", "uri": program_dir_path},
            {"kind": "program_state_json", "uri": program_state_path},
            {"kind": "examples_manifest_json", "uri": str(examples_manifest_path)},
            {"kind": "experiment_directory", "uri": str(output_dir)},
        ],
        audit_results={
            "ok": True,
            "prediction_rows": len(prediction_rows),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "test_examples": len(test_rows),
            "tree_bundle_kind": TREE_BUNDLE_KIND,
        },
        quarantine={"classification": "valid_run_manifest_v1"},
        command=sys.argv,
        publication_ready=True,
        metadata={
            "runner": SCRIPT_NAME,
            "experiment_api": "ExperimentContext",
            "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
            "tree_bundle_kind": TREE_BUNDLE_KIND,
            "single_leaf_per_document": True,
        },
    )
    _write_json(output_dir / "run_manifest.json", run_manifest)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--dimensions", default="")
    parser.add_argument("--mp-data-dir", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8010/v1")
    parser.add_argument("--model", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--num-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--train-max-input-chars", type=int, default=None)
    parser.add_argument("--val-max-input-chars", type=int, default=None)
    parser.add_argument("--test-max-input-chars", type=int, default=None)
    parser.add_argument("--train-max-input-tokens", type=int, default=None)
    parser.add_argument("--val-max-input-tokens", type=int, default=None)
    parser.add_argument("--test-max-input-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--token-cache-dir", type=Path, default=None)
    parser.add_argument("--train-docs", type=int, default=50)
    parser.add_argument("--val-docs", type=int, default=30)
    parser.add_argument("--test-docs", type=int, default=30)
    parser.add_argument("--train-sampling-strategy", default="soft_inverse_sqrt_length")
    parser.add_argument("--length-floor-chars", type=int, default=2000)
    parser.add_argument("--min-doc-chars", type=int, default=0)
    parser.add_argument("--optimizer", default="mipro", choices=("mipro", "bootstrap", "bootstrap_fewshot", "none"))
    parser.add_argument("--dspy-budget", default="light", choices=("light", "medium", "heavy"))
    parser.add_argument(
        "--initial-program-dir",
        type=Path,
        default=None,
        help="Optional loadable DSPy program directory to use as the starting global f.",
    )
    parser.add_argument("--dspy-num-threads", type=int, default=128)
    parser.add_argument("--eval-num-threads", type=int, default=128)
    parser.add_argument("--max-bootstrapped-demos", type=int, default=0)
    parser.add_argument("--max-labeled-demos", type=int, default=0)
    parser.add_argument("--mipro-num-trials", type=int, default=8)
    parser.add_argument("--mipro-num-candidates", type=int, default=None)
    parser.add_argument("--mipro-minibatch-size", type=int, default=24)
    parser.add_argument("--mipro-minibatch-full-eval-steps", type=int, default=2)
    parser.add_argument("--mipro-program-aware-proposer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mipro-data-aware-proposer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mipro-tip-aware-proposer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mipro-fewshot-aware-proposer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mipro-view-data-batch-size", type=int, default=10)
    parser.add_argument("--mipro-skip-bootstrap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mipro-prompt-max-tokens", type=int, default=2048)
    parser.add_argument("--mipro-prompt-temperature", type=float, default=0.7)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--use-batched-lm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-max-concurrent", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-timeout", type=float, default=0.02)
    parser.add_argument("--disable-dspy-cache", action="store_true")
    parser.add_argument("--mock-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("outputs") / "manifesto_full_doc_dspy_global_f" / _now_stamp()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_global_f(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "optimizer": summary["optimizer"],
                "prediction_rows": summary["metrics"]["prediction_rows"],
                "na_count": summary["metrics"]["na_count"],
                "train_examples": summary["metrics"]["train_examples"],
                "macro_external_expert_pearson": summary["metrics"]["macro_external_expert_pearson"],
                "summary": str(Path(args.output_dir) / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
