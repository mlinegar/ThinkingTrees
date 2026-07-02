#!/usr/bin/env python3
"""Run manifesto f/g baseline scorer combinations on the ladder split.

This runner is for the external / non-alternating baselines that do not fit
the literal ``fg -> fgf -> fgfg`` ladder rows:

- ``f^1 g^{benoit}``: GEPA-v2 optimized scorer on Benoit GPT-4o summaries.
- ``f^1 g^0``: GEPA-v2 optimized scorer on the stored baseline root summaries
  from ``outputs/overnight_benoit/full_pipeline/<dim>/per_manifesto.jsonl``.
- ``f^0 g^{benoit}``: exact Benoit raw prompt on Benoit's GPT-4o summaries.
- ``f^0 g^0``: exact Benoit raw prompt on the same stored baseline root
  summaries, with party names masked to ``<PARTY>`` before scoring.

Outputs are written per combo as ``per_manifesto.jsonl`` plus ``report.json``.
The report includes split-wise metrics for the same train/val/test split that
the current DSPy ladder uses, so the later plot integration can read the test
metric directly.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.script_io import (
    now_iso as _utc_now,
    read_json as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from src.experiments.script_parse import safe_float as _safe_float
from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.dimensions import PolicyDimension, get_dimension
from src.tasks.manifesto.expert_benchmarks import (
    benoit_ensemble_mean,
    load_benoit_expert_means,
    load_benoit_llm_scores,
    load_benoit_masked_summaries,
    load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.expert_scale import (
    EXPERT_SCALE_CHOICES,
    EXPERT_SCALE_NORMALIZED_1_7,
    EXPERT_SCALE_RAW,
    expert_scale_bounds,
    expert_scale_metadata,
    raw_benoit_expert_from_row,
    resolve_benoit_expert_target,
    scorer_1_7_to_expert_target,
)
from src.tasks.manifesto.resume_utils import load_resume_rows

LOGGER = logging.getLogger(__name__)

_DIM_FROM_NAME = {dim.value: dim for dim in PolicyDimension}
_INT_RE = re.compile(r"([1-7])")
PARTY_MASK_MODES = ("safe_boundary", "legacy", "none")

DEFAULT_SPLIT_IDS = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_fg_alternating"
    / "economic_benoit_moreleaves_dspy_medium_20260422_192229"
    / "teacher"
    / "split_ids.json"
)
DEFAULT_F1_SCORER = (
    PROJECT_ROOT / "outputs" / "phase1_gepa_v2_rank" / "economic" / "optimized_scorer.json"
)
DEFAULT_G0_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "overnight_benoit"
    / "full_pipeline"
    / "economic"
    / "per_manifesto.jsonl"
)

BENOIT_FIGURE1_PUBLISHED = {
    PolicyDimension.ECONOMIC: 0.87,
    PolicyDimension.SOCIAL: 0.92,
    PolicyDimension.IMMIGRATION: 0.89,
    PolicyDimension.EU: 0.91,
    PolicyDimension.ENVIRONMENT: 0.82,
    PolicyDimension.DECENTRALIZATION: 0.49,
}

COMBO_SPECS: dict[str, dict[str, Any]] = {
    "f1g_benoit": {
        "display_label": r"f^1 g^{benoit}",
        "f_kind": "optimized_dimension_scorer",
        "g_kind": "benoit_masked_summary",
    },
    "f1g0": {
        "display_label": r"f^1 g^0",
        "f_kind": "optimized_dimension_scorer",
        "g_kind": "stored_baseline_summary",
    },
    "f0g_benoit": {
        "display_label": r"f^0 g^{benoit}",
        "f_kind": "raw_benoit_prompt",
        "g_kind": "benoit_masked_summary",
    },
    "f0g0": {
        "display_label": r"f^0 g^0",
        "f_kind": "raw_benoit_prompt",
        "g_kind": "stored_baseline_summary_masked",
    },
}


def _resolve_expert_target_scale(args: argparse.Namespace, dimension: PolicyDimension) -> str:
    raw = getattr(args, "expert_target_scale", None)
    if raw:
        return str(raw)
    return EXPERT_SCALE_NORMALIZED_1_7


def _prediction_on_target_scale(
    score_1_7: Any,
    *,
    dimension: PolicyDimension,
    expert_target_scale: str,
) -> Optional[float]:
    if expert_target_scale == EXPERT_SCALE_NORMALIZED_1_7:
        return _safe_float(score_1_7)
    return scorer_1_7_to_expert_target(
        score_1_7,
        dimension=dimension,
        scale=expert_target_scale,
    )


def _discover_model(client: OpenAI) -> str:
    models = client.models.list().data
    if not models:
        raise RuntimeError("vLLM server returned no models")
    return str(models[0].id)


def _parse_raw_prompt_score(text: str) -> tuple[Optional[float], str]:
    stripped = str(text or "").strip()
    if not stripped:
        return None, ""
    upper = stripped.upper()
    if upper.startswith("NA") or upper == "N/A":
        return None, stripped
    match = _INT_RE.search(stripped)
    if match is None:
        return None, stripped
    return float(match.group(1)), stripped


def _load_split_map(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    payload = _read_json(path)
    split_map: dict[str, str] = {}
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        ids = [str(v) for v in payload.get(split, []) if str(v).strip()]
        split_ids[split] = ids
        for manifesto_id in ids:
            split_map[manifesto_id] = split
    return split_map, split_ids


def _is_invalid_party_alias(text: str) -> bool:
    normalized = str(text or "").strip().casefold()
    return normalized in {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _alias_word_chars(text: str) -> str:
    return re.sub(r"[^\w]+", "", str(text or ""), flags=re.UNICODE)


def _party_aliases(sample: Any, *, include_invalid: bool = True) -> list[str]:
    aliases = []
    for value in (
        getattr(sample, "party_name", None),
        getattr(sample, "party_abbrev", None),
    ):
        text = str(value or "").strip()
        if not text or (not include_invalid and _is_invalid_party_alias(text)):
            continue
        aliases.append(text)
        aliases.append(text.replace("’", "'"))
        aliases.append(text.replace("'", "’"))
    out: list[str] = []
    seen = set()
    for item in aliases:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _mask_party_names(summary: str, sample: Any, *, mode: str = "safe_boundary") -> str:
    if mode not in PARTY_MASK_MODES:
        raise ValueError(f"unknown party mask mode {mode!r}; expected one of {PARTY_MASK_MODES}")
    masked = str(summary or "")
    if mode == "none":
        return masked
    if mode == "legacy":
        for alias in sorted(_party_aliases(sample, include_invalid=True), key=len, reverse=True):
            masked = re.sub(re.escape(alias), "<PARTY>", masked, flags=re.IGNORECASE)
        return masked

    aliases = []
    for alias in _party_aliases(sample, include_invalid=False):
        if len(_alias_word_chars(alias)) <= 1:
            continue
        aliases.append(alias)
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        masked = re.sub(pattern, "<PARTY>", masked, flags=re.IGNORECASE)
    return masked


def _load_g0_rows(
    *,
    path: Path,
    dimension: PolicyDimension,
    dataset: ManifestoDataset,
    split_map: Mapping[str, str],
    party_mask_mode: str,
    expert_target_scale: str = EXPERT_SCALE_RAW,
) -> list[dict[str, Any]]:
    sample_by_id = {
        str(manifesto_id): dataset.get_sample(str(manifesto_id))
        for manifesto_id in split_map
    }
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(path, skip_bad=True):
        manifesto_id = str(row.get("manifesto_id") or "").strip()
        if not manifesto_id or manifesto_id not in split_map:
            continue
        summary = str(row.get("summary") or "").strip()
        expert = resolve_benoit_expert_target(row, dimension=dimension, scale=expert_target_scale)
        expert_1_7 = resolve_benoit_expert_target(
            row,
            dimension=dimension,
            scale=EXPERT_SCALE_NORMALIZED_1_7,
        )
        expert_raw = raw_benoit_expert_from_row(row, dimension=dimension)
        sample = sample_by_id.get(manifesto_id)
        if not summary or expert is None or sample is None:
            continue
        rows.append(
            {
                "manifesto_id": manifesto_id,
                "split": str(split_map[manifesto_id]),
                "summary": summary,
                "masked_summary": _mask_party_names(summary, sample, mode=party_mask_mode),
                "party_mask_mode": party_mask_mode,
                "expert_score": float(expert),
                "expert_score_native": float(expert_raw) if expert_raw is not None else None,
                "expert_score_1_7": float(expert_1_7) if expert_1_7 is not None else None,
                "expert_score_raw_benoit": expert_raw,
                "source_score_1_7": _safe_float(row.get("llm_score_1_7")),
                "source_score": _prediction_on_target_scale(
                    row.get("llm_score_1_7"),
                    dimension=dimension,
                    expert_target_scale=expert_target_scale,
                ),
                "summary_source": "stored_baseline_summary",
                "source_path": str(path),
                "party_name": getattr(sample, "party_name", ""),
                "party_abbrev": getattr(sample, "party_abbrev", ""),
            }
        )
    rows.sort(key=lambda item: (item["split"], item["manifesto_id"]))
    return rows


def _load_benoit_rows(
    *,
    dimension: PolicyDimension,
    dataset: ManifestoDataset,
    split_map: Mapping[str, str],
    expert_target_scale: str = EXPERT_SCALE_RAW,
) -> list[dict[str, Any]]:
    crosswalk = load_benoit_mp_crosswalk()
    benoit_to_py = {
        str(row.manifesto).removesuffix(".txt"): (int(row.party), int(row.year))
        for row in crosswalk.itertuples()
    }
    py_to_mid = {
        (int(sample.party_id), int(sample.year)): str(sample.manifesto_id)
        for sample in dataset
    }
    expert_lookup = {
        str(row.manifesto).removesuffix(".txt"): (
            float(row.expert_mean),
            float(row.expert_mean_1_7),
        )
        for row in load_benoit_expert_means(dimension).itertuples()
    }
    summaries = load_benoit_masked_summaries(dimension=dimension)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in summaries.itertuples():
        benoit_key = str(row.manifesto_stem)
        py_key = benoit_to_py.get(benoit_key)
        if py_key is None:
            skipped += 1
            continue
        manifesto_id = py_to_mid.get(py_key)
        if manifesto_id is None or manifesto_id not in split_map:
            skipped += 1
            continue
        expert_pair = expert_lookup.get(benoit_key)
        summary = str(row.summary or "").strip()
        if expert_pair is None or not summary:
            skipped += 1
            continue
        expert_raw, expert_1_7 = expert_pair
        expert = expert_raw if expert_target_scale == EXPERT_SCALE_RAW else expert_1_7
        rows.append(
            {
                "manifesto_id": manifesto_id,
                "split": str(split_map[manifesto_id]),
                "summary": summary,
                "masked_summary": summary,
                "expert_score": float(expert),
                "expert_score_native": float(expert_raw),
                "expert_score_1_7": float(expert_1_7),
                "expert_score_raw_benoit": float(expert_raw),
                "source_score_1_7": _safe_float(getattr(row, "benoit_score", None)),
                "source_score": _prediction_on_target_scale(
                    getattr(row, "benoit_score", None),
                    dimension=dimension,
                    expert_target_scale=expert_target_scale,
                ),
                "summary_source": "benoit_masked_summary",
                "source_key": benoit_key,
            }
        )
    rows.sort(key=lambda item: (item["split"], item["manifesto_id"]))
    LOGGER.info(
        "Loaded %d Benoit summary rows on the active split (%d skipped during mapping/filter)",
        len(rows),
        skipped,
    )
    return rows


def _pair_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    pred_key: str,
    truth_key: str,
    pred_mean_key: str,
    truth_mean_key: str,
    values_are_1_7: bool = False,
) -> dict[str, Any]:
    row_list = list(rows)
    preds: list[float] = []
    truths: list[float] = []
    for row in row_list:
        pred = _safe_float(row.get(pred_key))
        truth = _safe_float(row.get(truth_key))
        if pred is None or truth is None:
            continue
        preds.append(float(pred))
        truths.append(float(truth))
    if not preds:
        return {
            "n": 0,
            "n_na": len(row_list),
            "n_scored": 0,
            "pearson_r": None,
            "pearson_ci_low": None,
            "pearson_ci_high": None,
            "spearman_r": None,
            "mae": None,
            "mae_1_7": None,
            pred_mean_key: None,
            truth_mean_key: None,
        }
    if len(preds) >= 3:
        report = compute_corpus_pearson_r(preds, truths).as_dict()
    else:
        report = {
            "pearson_r": None,
            "pearson_ci_low": None,
            "pearson_ci_high": None,
            "spearman_r": None,
            "n": int(len(preds)),
            "n_na": int(len(row_list) - len(preds)),
        }
    report["mae"] = float(sum(abs(p - t) for p, t in zip(preds, truths)) / len(preds))
    report["mae_1_7"] = report["mae"] if values_are_1_7 else None
    report[pred_mean_key] = float(sum(preds) / len(preds))
    report[truth_mean_key] = float(sum(truths) / len(truths))
    report["n_scored"] = int(len(preds))
    return report


def _score_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return _pair_metrics(
        rows,
        pred_key="pred_score",
        truth_key="expert_score",
        pred_mean_key="mean_prediction",
        truth_mean_key="mean_expert",
    )


def _split_report(
    rows: list[dict[str, Any]],
    *,
    pred_key: str = "pred_score",
    truth_key: str = "expert_score",
    pred_mean_key: str = "mean_prediction",
    truth_mean_key: str = "mean_expert",
    values_are_1_7: bool = False,
) -> dict[str, Any]:
    by_split: dict[str, Any] = {
        "all": _pair_metrics(
            rows,
            pred_key=pred_key,
            truth_key=truth_key,
            pred_mean_key=pred_mean_key,
            truth_mean_key=truth_mean_key,
            values_are_1_7=values_are_1_7,
        )
    }
    for split in ("train", "val", "test"):
        subset = [row for row in rows if str(row.get("split")) == split]
        by_split[split] = _pair_metrics(
            subset,
            pred_key=pred_key,
            truth_key=truth_key,
            pred_mean_key=pred_mean_key,
            truth_mean_key=truth_mean_key,
            values_are_1_7=values_are_1_7,
        )
    by_split["n_total_rows"] = int(len(rows))
    by_split["n_na_rows"] = int(
        sum(_safe_float(row.get(pred_key)) is None or _safe_float(row.get(truth_key)) is None for row in rows)
    )
    return by_split


def _comparison_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_vs_expert": _split_report(
            rows,
            pred_key="pred_score",
            truth_key="expert_score",
            pred_mean_key="mean_prediction",
            truth_mean_key="mean_expert",
        ),
        "source_vs_expert": _split_report(
            rows,
            pred_key="source_score",
            truth_key="expert_score",
            pred_mean_key="mean_source_score",
            truth_mean_key="mean_expert",
        ),
        "model_vs_source": _split_report(
            rows,
            pred_key="pred_score_1_7",
            truth_key="source_score_1_7",
            pred_mean_key="mean_prediction_1_7",
            truth_mean_key="mean_source_score_1_7",
            values_are_1_7=True,
        ),
    }


def _benoit_ensemble_reference_metrics(dimension: PolicyDimension) -> dict[str, Any]:
    refs: dict[str, Any] = {
        "reported_ensemble_published_pearson_r": BENOIT_FIGURE1_PUBLISHED[dimension],
    }
    experts = load_benoit_expert_means(dimension)
    expert_lookup = {
        str(row.manifesto): float(row.expert_mean)
        for row in experts.itertuples()
    }
    for kind in ("reported", "openweight"):
        try:
            scores = load_benoit_llm_scores(kind=kind, dimension=dimension)
            ensemble = benoit_ensemble_mean(scores)
            rows = [
                {
                    "pred": _prediction_on_target_scale(
                        row.score_llm_mean,
                        dimension=dimension,
                        expert_target_scale=EXPERT_SCALE_RAW,
                    ),
                    "expert": expert_lookup.get(str(row.manifesto)),
                }
                for row in ensemble.itertuples()
            ]
            refs[f"{kind}_ensemble_reproduced"] = _pair_metrics(
                rows,
                pred_key="pred",
                truth_key="expert",
                pred_mean_key="mean_ensemble_score",
                truth_mean_key="mean_expert",
                values_are_1_7=False,
            )
        except Exception as exc:  # noqa: BLE001
            refs[f"{kind}_ensemble_reproduced_error"] = str(exc)
    return refs


def _configure_dspy_lm(*, port: int, model: Optional[str], temperature: float, max_tokens: int) -> str:
    local_inference = resolve_local_inference_config(
        {
            "port": int(port),
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
    )
    lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
    configure_dspy(lm=lm)
    resolved = getattr(lm, "model", None) or model
    return str(resolved or "")


def _score_with_f1(
    *,
    rows: list[dict[str, Any]],
    dimension: PolicyDimension,
    scorer_json: Path,
    max_output_tokens: int,
    expert_target_scale: str,
) -> list[dict[str, Any]]:
    spec = get_dimension(dimension)
    scorer = DimensionScorer(spec, use_cot=False, max_output_tokens=int(max_output_tokens))
    scorer.load(str(scorer_json))
    out: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, row in enumerate(rows, start=1):
        result = scorer(summary=str(row["summary"]))
        pred = result.get("score") if isinstance(result, dict) else None
        out.append(
            {
                **row,
                "pred_score_1_7": _safe_float(pred),
                "pred_score": _prediction_on_target_scale(
                    pred,
                    dimension=dimension,
                    expert_target_scale=expert_target_scale,
                ),
                "pred_is_na": pred is None,
                "pred_reasoning": str((result or {}).get("reasoning") or "")[:800],
            }
        )
        if idx % 25 == 0 or idx == len(rows):
            LOGGER.info("f1 scoring %d/%d (%.1fs)", idx, len(rows), time.time() - t0)
    return out


def _score_with_f0_raw_prompt(
    *,
    rows: list[dict[str, Any]],
    dimension: PolicyDimension,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    expert_target_scale: str,
) -> list[dict[str, Any]]:
    system_prompt = get_benoit_scoring_context(dimension)
    out: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, row in enumerate(rows, start=1):
        summary = str(row.get("masked_summary") or row.get("summary") or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze the following political text:\n\n{summary}"},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                extra_body={"seed": 0, "top_p": 1.0},
            )
            raw_text = str(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Raw prompt call failed on %s: %s", row.get("manifesto_id"), exc)
            raw_text = ""
        pred, normalized = _parse_raw_prompt_score(raw_text)
        out.append(
            {
                **row,
                "pred_score_1_7": pred,
                "pred_score": _prediction_on_target_scale(
                    pred,
                    dimension=dimension,
                    expert_target_scale=expert_target_scale,
                ),
                "pred_is_na": pred is None,
                "pred_reasoning": normalized[:200],
            }
        )
        if idx % 25 == 0 or idx == len(rows):
            LOGGER.info("f0 raw prompt scoring %d/%d (%.1fs)", idx, len(rows), time.time() - t0)
    return out


def _write_combo_outputs(
    *,
    combo: str,
    combo_spec: Mapping[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
    run_meta: Mapping[str, Any],
    references: Mapping[str, Any],
) -> None:
    combo_dir = output_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)
    per_path = combo_dir / "per_manifesto.jsonl"
    _write_jsonl(per_path, rows, ensure_ascii=False)
    comparisons = _comparison_reports(rows)
    report = {
        "generated_at": _utc_now(),
        "combo": combo,
        "display_label": combo_spec.get("display_label"),
        "f_kind": combo_spec.get("f_kind"),
        "g_kind": combo_spec.get("g_kind"),
        "run": dict(run_meta),
        "metrics": comparisons["model_vs_expert"],
        "comparisons": comparisons,
        "references": dict(references),
        "artifacts": {
            "per_manifesto": str(per_path),
        },
    }
    _write_json(combo_dir / "report.json", report)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimension", choices=sorted(_DIM_FROM_NAME), default="economic")
    parser.add_argument(
        "--combo",
        action="append",
        choices=sorted(COMBO_SPECS),
        help="Repeat to run a subset; default runs all supported combos.",
    )
    parser.add_argument("--split-ids", type=Path, default=DEFAULT_SPLIT_IDS)
    parser.add_argument("--mp-data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "manifesto_corpus_benoit")
    parser.add_argument("--g0-results", type=Path, default=DEFAULT_G0_RESULTS)
    parser.add_argument("--f1-scorer-json", type=Path, default=DEFAULT_F1_SCORER)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--f1-max-tokens", type=int, default=256)
    parser.add_argument("--f0-max-tokens", type=int, default=8)
    parser.add_argument(
        "--expert-target-scale",
        choices=EXPERT_SCALE_CHOICES,
        default=None,
        help="Omit for normalized_1_7; pass raw_benoit only to reproduce older raw-scale metrics.",
    )
    parser.add_argument(
        "--party-mask-mode",
        choices=PARTY_MASK_MODES,
        default="safe_boundary",
        help=(
            "How to anonymize party names in stored baseline summaries before raw-prompt scoring. "
            "safe_boundary skips invalid/one-letter aliases and masks only boundary-delimited aliases; "
            "legacy preserves the old unrestricted substitution behavior."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    combos = args.combo or list(COMBO_SPECS)
    split_map, split_ids = _load_split_map(args.split_ids)
    dimension = _DIM_FROM_NAME[str(args.dimension)]
    expert_target_scale = _resolve_expert_target_scale(args, dimension)
    target_min, target_max = expert_scale_bounds(
        dimension=dimension,
        scale=expert_target_scale,
    )
    dataset = ManifestoDataset(data_dir=args.mp_data_dir, require_text=True)

    g0_rows = _load_g0_rows(
        path=args.g0_results,
        dimension=dimension,
        dataset=dataset,
        split_map=split_map,
        party_mask_mode=str(args.party_mask_mode),
        expert_target_scale=expert_target_scale,
    )
    benoit_rows = _load_benoit_rows(
        dimension=dimension,
        dataset=dataset,
        split_map=split_map,
        expert_target_scale=expert_target_scale,
    )
    references = _benoit_ensemble_reference_metrics(dimension)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": _utc_now(),
        "dimension": str(args.dimension),
        "combos": combos,
        "split_ids_path": str(args.split_ids),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "g0_results_path": str(args.g0_results),
        "f1_scorer_json": str(args.f1_scorer_json),
        "party_mask_mode": str(args.party_mask_mode),
        "metrics_scale": expert_target_scale,
        **expert_scale_metadata(dimension=dimension, scale=expert_target_scale),
        "target_min": float(target_min),
        "target_max": float(target_max),
        "model": args.model,
        "port": int(args.port),
        "references": references,
        "sources": {
            "g0_split_rows": len(g0_rows),
            "benoit_split_rows": len(benoit_rows),
        },
    }

    need_f1 = any(COMBO_SPECS[combo]["f_kind"] == "optimized_dimension_scorer" for combo in combos)
    resolved_f1_model = None
    if need_f1:
        resolved_f1_model = _configure_dspy_lm(
            port=int(args.port),
            model=args.model,
            temperature=float(args.temperature),
            max_tokens=int(args.f1_max_tokens),
        )
        LOGGER.info("Configured DSPy LM for f1 scoring: %s", resolved_f1_model or "<unknown>")

    raw_prompt_client: Optional[OpenAI] = None
    raw_prompt_model: Optional[str] = None

    for combo in combos:
        combo_spec = COMBO_SPECS[combo]
        rows = benoit_rows if combo_spec["g_kind"] == "benoit_masked_summary" else g0_rows
        LOGGER.info("Running %s over %d rows", combo_spec["display_label"], len(rows))
        if combo_spec["f_kind"] == "optimized_dimension_scorer":
            scored_rows = _score_with_f1(
                rows=rows,
                dimension=dimension,
                scorer_json=args.f1_scorer_json,
                max_output_tokens=int(args.f1_max_tokens),
                expert_target_scale=expert_target_scale,
            )
            run_meta = {
                "scorer_kind": "optimized_dimension_scorer",
                "scorer_json": str(args.f1_scorer_json),
                "model": resolved_f1_model,
                "temperature": float(args.temperature),
                "party_mask_mode": str(args.party_mask_mode),
                "metrics_scale": expert_target_scale,
                **expert_scale_metadata(dimension=dimension, scale=expert_target_scale),
            }
        elif combo_spec["f_kind"] == "raw_benoit_prompt":
            if raw_prompt_client is None:
                base_url = f"http://{args.host}:{int(args.port)}/v1"
                raw_prompt_client = OpenAI(base_url=base_url, api_key="EMPTY")
                raw_prompt_model = str(args.model or _discover_model(raw_prompt_client))
                LOGGER.info("Configured raw prompt client: %s model=%s", base_url, raw_prompt_model)
            scored_rows = _score_with_f0_raw_prompt(
                rows=rows,
                dimension=dimension,
                client=raw_prompt_client,
                model=str(raw_prompt_model),
                temperature=float(args.temperature),
                max_tokens=int(args.f0_max_tokens),
                expert_target_scale=expert_target_scale,
            )
            run_meta = {
                "scorer_kind": "raw_benoit_prompt",
                "model": raw_prompt_model,
                "temperature": float(args.temperature),
                "max_tokens": int(args.f0_max_tokens),
                "party_mask_mode": str(args.party_mask_mode),
                "metrics_scale": expert_target_scale,
                **expert_scale_metadata(dimension=dimension, scale=expert_target_scale),
            }
        else:
            raise ValueError(f"Unsupported combo scorer kind: {combo_spec['f_kind']!r}")
        _write_combo_outputs(
            combo=combo,
            combo_spec=combo_spec,
            rows=scored_rows,
            output_dir=args.output_dir,
            run_meta=run_meta,
            references=references,
        )

    _write_json(args.output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote baseline manifest to %s", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
