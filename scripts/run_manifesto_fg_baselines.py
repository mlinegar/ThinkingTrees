#!/usr/bin/env python3
"""Run manifesto f/g baseline scorer combinations on the ladder split.

This runner is for the external / non-alternating baselines that do not fit
the literal ``fg -> fgf -> fgfg`` ladder rows:

- ``f^1 g^{benoit}``: GEPA-v2 optimized scorer on Benoit GPT-4o summaries.
- ``f^1 g^0``: GEPA-v2 optimized scorer on the stored baseline root summaries
  from ``outputs/overnight_benoit/full_pipeline/<dim>/per_manifesto.jsonl``.
- ``f^0 g^0``: exact Benoit raw prompt on the same stored baseline root
  summaries, with party names masked to ``<PARTY>`` before scoring.

Outputs are written per combo as ``per_manifesto.jsonl`` plus ``report.json``.
The report includes split-wise metrics for the same train/val/test split that
the current DSPy ladder uses, so the later plot integration can read the test
metric directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.dspy_config import configure_dspy, create_vllm_lm
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.dimensions import PolicyDimension, get_dimension
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_masked_summaries,
    load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.resume_utils import load_resume_rows

LOGGER = logging.getLogger(__name__)

_DIM_FROM_NAME = {dim.value: dim for dim in PolicyDimension}
_INT_RE = re.compile(r"([1-7])")

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
    "f0g0": {
        "display_label": r"f^0 g^0",
        "f_kind": "raw_benoit_prompt",
        "g_kind": "stored_baseline_summary_masked",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_map: dict[str, str] = {}
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        ids = [str(v) for v in payload.get(split, []) if str(v).strip()]
        split_ids[split] = ids
        for manifesto_id in ids:
            split_map[manifesto_id] = split
    return split_map, split_ids


def _party_aliases(sample: Any) -> list[str]:
    aliases = []
    for value in (
        getattr(sample, "party_name", None),
        getattr(sample, "party_abbrev", None),
    ):
        text = str(value or "").strip()
        if not text:
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


def _mask_party_names(summary: str, sample: Any) -> str:
    masked = str(summary or "")
    for alias in sorted(_party_aliases(sample), key=len, reverse=True):
        masked = re.sub(re.escape(alias), "<PARTY>", masked, flags=re.IGNORECASE)
    return masked


def _load_g0_rows(
    *,
    path: Path,
    dataset: ManifestoDataset,
    split_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    sample_by_id = {
        str(manifesto_id): dataset.get_sample(str(manifesto_id))
        for manifesto_id in split_map
    }
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        manifesto_id = str(row.get("manifesto_id") or "").strip()
        if not manifesto_id or manifesto_id not in split_map:
            continue
        summary = str(row.get("summary") or "").strip()
        expert = _safe_float(row.get("benoit_expert_mean"))
        sample = sample_by_id.get(manifesto_id)
        if not summary or expert is None or sample is None:
            continue
        rows.append(
            {
                "manifesto_id": manifesto_id,
                "split": str(split_map[manifesto_id]),
                "summary": summary,
                "masked_summary": _mask_party_names(summary, sample),
                "expert_score_1_7": float(expert),
                "source_score_1_7": _safe_float(row.get("llm_score_1_7")),
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
        str(row.manifesto).removesuffix(".txt"): float(row.expert_mean)
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
        expert = expert_lookup.get(benoit_key)
        summary = str(row.summary or "").strip()
        if expert is None or not summary:
            skipped += 1
            continue
        rows.append(
            {
                "manifesto_id": manifesto_id,
                "split": str(split_map[manifesto_id]),
                "summary": summary,
                "masked_summary": summary,
                "expert_score_1_7": float(expert),
                "source_score_1_7": _safe_float(getattr(row, "benoit_score", None)),
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


def _score_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    preds: list[float] = []
    truths: list[float] = []
    for row in rows:
        pred = _safe_float(row.get("pred_score_1_7"))
        truth = _safe_float(row.get("expert_score_1_7"))
        if pred is None or truth is None:
            continue
        preds.append(float(pred))
        truths.append(float(truth))
    if not preds:
        return {
            "n_scored": 0,
            "pearson_r": None,
            "pearson_ci_low": None,
            "pearson_ci_high": None,
            "mae_1_7": None,
            "mean_prediction_1_7": None,
            "mean_expert_1_7": None,
        }
    if len(preds) >= 3:
        report = compute_corpus_pearson_r(preds, truths).as_dict()
    else:
        report = {
            "pearson_r": None,
            "pearson_ci_low": None,
            "pearson_ci_high": None,
            "n": int(len(preds)),
        }
    report["mae_1_7"] = float(sum(abs(p - t) for p, t in zip(preds, truths)) / len(preds))
    report["mean_prediction_1_7"] = float(sum(preds) / len(preds))
    report["mean_expert_1_7"] = float(sum(truths) / len(truths))
    report["n_scored"] = int(len(preds))
    return report


def _split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {"all": _score_metrics(rows)}
    for split in ("train", "val", "test"):
        subset = [row for row in rows if str(row.get("split")) == split]
        by_split[split] = _score_metrics(subset)
    by_split["n_total_rows"] = int(len(rows))
    by_split["n_na_rows"] = int(sum(row.get("pred_score_1_7") is None for row in rows))
    return by_split


def _configure_dspy_lm(*, port: int, model: Optional[str], temperature: float, max_tokens: int) -> str:
    lm = create_vllm_lm(
        port=int(port),
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        cache=True,
    )
    configure_dspy(lm=lm)
    resolved = getattr(lm, "model", None) or model
    return str(resolved or "")


def _score_with_f1(
    *,
    rows: list[dict[str, Any]],
    dimension: PolicyDimension,
    scorer_json: Path,
    max_output_tokens: int,
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
) -> None:
    combo_dir = output_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)
    per_path = combo_dir / "per_manifesto.jsonl"
    with per_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "generated_at": _utc_now(),
        "combo": combo,
        "display_label": combo_spec.get("display_label"),
        "f_kind": combo_spec.get("f_kind"),
        "g_kind": combo_spec.get("g_kind"),
        "run": dict(run_meta),
        "metrics": _split_report(rows),
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
    dataset = ManifestoDataset(data_dir=args.mp_data_dir, require_text=True)

    g0_rows = _load_g0_rows(path=args.g0_results, dataset=dataset, split_map=split_map)
    benoit_rows = _load_benoit_rows(dimension=dimension, dataset=dataset, split_map=split_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": _utc_now(),
        "dimension": str(args.dimension),
        "combos": combos,
        "split_ids_path": str(args.split_ids),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "g0_results_path": str(args.g0_results),
        "f1_scorer_json": str(args.f1_scorer_json),
        "model": args.model,
        "port": int(args.port),
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
            )
            run_meta = {
                "scorer_kind": "optimized_dimension_scorer",
                "scorer_json": str(args.f1_scorer_json),
                "model": resolved_f1_model,
                "temperature": float(args.temperature),
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
            )
            run_meta = {
                "scorer_kind": "raw_benoit_prompt",
                "model": raw_prompt_model,
                "temperature": float(args.temperature),
                "max_tokens": int(args.f0_max_tokens),
            }
        else:
            raise ValueError(f"Unsupported combo scorer kind: {combo_spec['f_kind']!r}")
        _write_combo_outputs(
            combo=combo,
            combo_spec=combo_spec,
            rows=scored_rows,
            output_dir=args.output_dir,
            run_meta=run_meta,
        )

    _write_json(args.output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote baseline manifest to %s", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
