#!/usr/bin/env python3
"""Generate teacher traces from real manifesto anchors for summary training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    build_benchmark_docs,
    build_split_labels,
    build_summary_pair_rows,
    clip_source_text,
    select_seed_manifestos,
    strict_same_side_raw,
    summarize_teacher_trace_records,
    write_jsonl,
    write_teacher_trace_records_jsonl,
)
from src.ctreepo.distillation import (
    build_labeled_tree_from_text,
    write_labeled_trees_jsonl,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_MAIN_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"

try:
    import dspy
except Exception:  # pragma: no cover - optional runtime dependency guard
    dspy = None  # type: ignore[assignment]


def _http_error_detail(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = str(exc.response.text or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "..."
        return f"status={exc.response.status_code} body={body}"
    return str(exc)


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 180.0,
        enable_thinking: bool = False,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key)
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {
                "enable_thinking": bool(self.enable_thinking),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_SCORE_HINT_RE = re.compile(
    r"(?i)(?:rile(?:\s+score)?|score|value|prediction)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)"
)

if dspy is not None:

    class ExpansionRevisionGuidance(dspy.Signature):
        """Structured guidance to revise an expansion toward the target score."""

        source_text: str = dspy.InputField(
            desc="Source manifesto text that must remain semantically faithful"
        )
        current_expansion: str = dspy.InputField(
            desc="Current expanded English rewrite that drifted from target"
        )
        target_rile_raw: str = dspy.InputField(
            desc="Target RILE score in [-100,100] that should be matched"
        )
        current_rile_raw: str = dspy.InputField(
            desc="Current scored RILE value in [-100,100] of the expansion"
        )
        guidance: str = dspy.OutputField(
            desc=(
                "Concrete edit instructions to move score toward target while preserving "
                "source commitments, entities, qualifiers, and policy content."
            )
        )


def _parse_score(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None

    parsed_json = _extract_json_object(rendered)
    if parsed_json is not None:
        from_json = _extract_score_from_json_obj(parsed_json)
        if from_json is not None:
            return from_json

    hinted = _SCORE_HINT_RE.findall(rendered)
    for token in hinted:
        try:
            value = float(token)
        except (TypeError, ValueError):
            continue
        if -100.0 <= value <= 100.0:
            return float(value)

    raw_matches = _NUMERIC_RE.findall(rendered)
    values: List[float] = []
    for token in raw_matches:
        try:
            values.append(float(token))
        except (TypeError, ValueError):
            continue
    in_range = [value for value in values if -100.0 <= value <= 100.0]
    if in_range:
        # Prefer non-boundary values (avoid selecting rubric bounds "-100, 100").
        non_boundary = [value for value in in_range if abs(value) < 99.999]
        return float(non_boundary[0] if non_boundary else in_range[0])
    if values:
        return float(max(-100.0, min(100.0, values[-1])))
    return None


def _extract_score_from_json_obj(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        for key in ("score", "rile", "rile_score", "value", "prediction"):
            if key in payload:
                nested = _extract_score_from_json_obj(payload[key])
                if nested is not None:
                    return nested
        for value in payload.values():
            nested = _extract_score_from_json_obj(value)
            if nested is not None:
                return nested
        return None
    if isinstance(payload, list):
        for value in payload:
            nested = _extract_score_from_json_obj(value)
            if nested is not None:
                return nested
        return None
    try:
        number = float(payload)
    except (TypeError, ValueError):
        return None
    if -100.0 <= number <= 100.0:
        return float(number)
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    rendered = str(text or "").strip()
    if not rendered:
        return None

    candidates = [rendered]
    for match in _JSON_FENCE_RE.findall(rendered):
        block = str(match).strip()
        if block:
            candidates.append(block)

    for candidate in candidates:
        parsed = _try_parse_json_dict(candidate)
        if parsed is not None:
            return parsed
        sliced = _slice_first_json_object(candidate)
        if sliced:
            parsed = _try_parse_json_dict(sliced)
            if parsed is not None:
                return parsed
    return None


def _clip_for_prompt(text: str, max_chars: int) -> str:
    rendered = str(text or "")
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars]


def _try_parse_json_dict(text: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _slice_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _to_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        rows = [line.strip(" -\t") for line in value.splitlines() if line.strip()]
        return [row for row in rows if row]
    return []


def _build_score_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _score(text: str) -> float:
        response = client.chat(
            system=(
                "You are a strict directional coder for information extraction. "
                "Return exactly one numeric RILE score in [-100, 100]."
            ),
            user=(
                "Score this text on a RILE-style directional scale. "
                "Return only one number.\n\n"
                f"TEXT:\n{text}"
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
                    f"TEXT:\n{text}"
                ),
                temperature=0.0,
                max_tokens=max(8, int(max_tokens)),
            )
            parsed = _parse_score(retry)
        if parsed is None:
            raise ValueError(f"Could not parse score responses: first={response!r} retry={retry!r}")
        return float(parsed)

    return _score


def _build_dspy_guidance_fn(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    source_max_chars: int,
    expansion_max_chars: int,
):
    if dspy is None:
        raise RuntimeError("DSPy is not available but --use-dspy-guidance was requested.")

    from src.config.dspy_config import configure_dspy

    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=str(base_url).rstrip("/"),
        api_key=api_key,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        cache=False,
    )
    configure_dspy(lm=lm)
    predictor = dspy.Predict(ExpansionRevisionGuidance)

    def _guidance(
        *,
        source_text: str,
        current_expansion: str,
        target_rile_raw: float,
        current_rile_raw: float,
    ) -> str:
        prediction = predictor(
            source_text=_clip_for_prompt(str(source_text), max_chars=int(source_max_chars)),
            current_expansion=_clip_for_prompt(str(current_expansion), max_chars=int(expansion_max_chars)),
            target_rile_raw=f"{float(target_rile_raw):.2f}",
            current_rile_raw=f"{float(current_rile_raw):.2f}",
        )
        return str(getattr(prediction, "guidance", "") or "").strip()

    return _guidance


def _expand_document(
    *,
    client: OpenAIChatClient,
    source_text: str,
    source_rile_raw: float,
    source_manifesto_id: str,
    attempt: int,
    previous_expansion: Optional[str],
    previous_score: Optional[float],
    revision_guidance: Optional[str],
    temperature: float,
    max_tokens: int,
    previous_expansion_max_chars: int,
    revision_guidance_max_chars: int,
) -> str:
    correction = ""
    if previous_expansion:
        guidance_block = ""
        if revision_guidance:
            guidance_block = (
                "\nAdditional revision guidance (follow strictly while preserving source fidelity):\n"
                f"{_clip_for_prompt(str(revision_guidance), max_chars=int(revision_guidance_max_chars))}"
            )
        correction = (
            f"\n\nPrevious attempt score: {previous_score:.2f} vs target {source_rile_raw:.2f}.\n"
            "Revise to move closer to the target score without dropping source commitments.\n"
            f"PREVIOUS_ATTEMPT:\n{_clip_for_prompt(str(previous_expansion), max_chars=int(previous_expansion_max_chars))}"
            f"{guidance_block}"
        )

    return client.chat(
        system=(
            "You produce faithful English policy documents for information extraction training. "
            "Preserve substantive commitments and caveats while improving structure and clarity. "
            "Do not output Q&A, math exercises, code, or commentary. "
            "Do not mention any numeric scores (including RILE) in the output."
        ),
        user=(
            f"Source manifesto id: {source_manifesto_id}\n"
            f"Target RILE score: {source_rile_raw:.2f}\n"
            f"Attempt: {attempt}\n\n"
            "Task:\n"
            "1. Rewrite and expand the source into a coherent English policy document.\n"
            "2. Preserve all directional commitments, named entities, and qualifiers.\n"
            "3. Keep policy content faithful; avoid adding unrelated claims.\n"
            "4. Output only the rewritten document.\n"
            "5. Do NOT mention the target score, the word RILE, or any numeric score.\n\n"
            f"SOURCE_TEXT:\n{source_text}"
            f"{correction}"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()


def _summarize_text(
    *,
    client: OpenAIChatClient,
    text: str,
    source_rile_raw: float,
    hop: int,
    temperature: float,
    max_tokens: int,
) -> str:
    return client.chat(
        system=(
            "Summarize for information extraction while preserving directional stance, "
            "factual commitments, and qualifying caveats."
        ),
        user=(
            f"Target directional score to preserve: {source_rile_raw:.2f}\n"
            f"Resummary hop: {hop}\n"
            "Do NOT mention any numeric score or the term RILE.\n"
            "Return only summary text.\n\n"
            f"TEXT:\n{text}"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()


def _build_labeled_tree_node_summary_fn(
    *,
    client: OpenAIChatClient,
    source_rile_raw: float,
    temperature: float,
    max_tokens: int,
    max_chars: int,
) -> Callable[[str, Dict[str, Any]], str]:
    def _summarize_node(text: str, context: Dict[str, Any]) -> str:
        clipped_text = _clip_for_prompt(str(text), max_chars=int(max_chars))
        if bool(context.get("is_leaf")):
            user = (
                f"Target directional score to preserve: {float(source_rile_raw):.2f}\n"
                "Summarize this C-TreePO leaf span for later score prediction.\n"
                "Preserve directional stance, entities, factual commitments, and caveats.\n"
                "Do NOT mention any numeric score or the term RILE.\n"
                "Return only summary text.\n\n"
                f"LEAF_SPAN:\n{clipped_text}"
            )
        else:
            left_summary = str(context.get("left_summary") or "").strip()
            right_summary = str(context.get("right_summary") or "").strip()
            user = (
                f"Target directional score to preserve: {float(source_rile_raw):.2f}\n"
                "Merge these two child summaries into a C-TreePO parent summary.\n"
                "Preserve all score-relevant commitments and caveats. Do not add unrelated claims.\n"
                "Do NOT mention any numeric score or the term RILE.\n"
                "Return only summary text.\n\n"
                f"LEFT_CHILD_SUMMARY:\n{left_summary}\n\n"
                f"RIGHT_CHILD_SUMMARY:\n{right_summary}\n\n"
                f"PARENT_SPAN_REFERENCE:\n{clipped_text}"
            )
        return client.chat(
            system=(
                "Summarize for tree-indexed information extraction distillation. "
                "Outputs must be concise, faithful, and score preserving."
            ),
            user=user,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ).strip()

    return _summarize_node


def _extract_trace(
    *,
    client: OpenAIChatClient,
    source_text: str,
    expanded_text: str,
    summary1: str,
    summary2: str,
    source_rile_raw: float,
    temperature: float,
    max_tokens: int,
    source_max_chars: int,
    expanded_max_chars: int,
) -> Dict[str, Any]:
    response = client.chat(
        system=(
            "Extract structured preservation traces for summarization training. "
            "Return strict JSON only."
        ),
        user=(
            "Return a JSON object with keys:\n"
            "- critical_points: list[str]\n"
            "- entities: list[str]\n"
            "- qualifiers: list[str]\n"
            "- invariants: list[str]\n"
            "- notes: str\n\n"
            f"TARGET_RILE: {source_rile_raw:.2f}\n\n"
            f"SOURCE_TEXT:\n{_clip_for_prompt(source_text, max_chars=int(source_max_chars))}\n\n"
            f"EXPANDED_TEXT:\n{_clip_for_prompt(expanded_text, max_chars=int(expanded_max_chars))}\n\n"
            f"SUMMARY1:\n{summary1}\n\n"
            f"SUMMARY2:\n{summary2}\n"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    parsed = _extract_json_object(response)
    if parsed is None:
        return {
            "critical_points": [],
            "entities": [],
            "qualifiers": [],
            "invariants": [],
            "notes": str(response).strip(),
        }
    return {
        "critical_points": _to_list_of_strings(parsed.get("critical_points")),
        "entities": _to_list_of_strings(parsed.get("entities")),
        "qualifiers": _to_list_of_strings(parsed.get("qualifiers")),
        "invariants": _to_list_of_strings(parsed.get("invariants")),
        "notes": str(parsed.get("notes", "") or "").strip(),
    }


def _scale_split_counts(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    available: int,
) -> Tuple[int, int, int]:
    requested = int(train_size) + int(val_size) + int(test_size)
    if requested <= 0:
        return 0, 0, 0
    if requested == available:
        return int(train_size), int(val_size), int(test_size)

    ratios = [
        float(train_size) / requested,
        float(val_size) / requested,
        float(test_size) / requested,
    ]
    scaled = [int(round(r * available)) for r in ratios]
    delta = available - sum(scaled)
    order = sorted(range(3), key=lambda idx: ratios[idx], reverse=True)
    for idx in order:
        if delta == 0:
            break
        scaled[idx] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
    scaled = [max(0, value) for value in scaled]
    total = sum(scaled)
    if total != available:
        scaled[0] += (available - total)
    return int(scaled[0]), int(scaled[1]), int(scaled[2])


def _process_seed_doc(
    *,
    seed_doc: Any,
    split: str,
    example_id: str,
    args: argparse.Namespace,
    teacher_client: OpenAIChatClient,
    score_fn: Callable[[str], float],
    dspy_guidance_fn: Optional[Callable[..., str]],
    dspy_guidance_enabled: bool,
) -> Tuple[Optional[TeacherTraceRecord], Optional[Dict[str, Any]]]:
    source_text_prompt = clip_source_text(seed_doc.source_text, max_chars=int(args.max_source_chars))
    target_raw = float(seed_doc.source_rile_raw)

    accepted: Optional[Tuple[str, float, int]] = None
    prev_expansion: Optional[str] = None
    prev_score: Optional[float] = None
    attempt_scores: List[float] = []
    guidance_note: Optional[str] = None

    for attempt in range(1, max(1, int(args.max_attempts)) + 1):
        revision_guidance = None
        if dspy_guidance_fn is not None and prev_expansion and prev_score is not None:
            try:
                revision_guidance = dspy_guidance_fn(
                    source_text=source_text_prompt,
                    current_expansion=prev_expansion,
                    target_rile_raw=target_raw,
                    current_rile_raw=prev_score,
                )
                guidance_note = revision_guidance[:220] if revision_guidance else guidance_note
            except Exception as exc:
                LOGGER.warning(
                    "DSPy guidance failed for %s attempt %d: %s",
                    seed_doc.manifesto_id,
                    attempt,
                    exc,
                )
        try:
            expanded = _expand_document(
                client=teacher_client,
                source_text=source_text_prompt,
                source_rile_raw=target_raw,
                source_manifesto_id=seed_doc.manifesto_id,
                attempt=attempt,
                previous_expansion=prev_expansion,
                previous_score=prev_score,
                revision_guidance=revision_guidance,
                temperature=float(args.expand_temperature),
                max_tokens=int(args.expand_max_tokens),
                previous_expansion_max_chars=int(args.previous_expansion_max_chars),
                revision_guidance_max_chars=int(args.revision_guidance_max_chars),
            )
        except Exception as exc:
            LOGGER.warning(
                "Expansion failed for %s attempt %d: %s",
                seed_doc.manifesto_id,
                attempt,
                _http_error_detail(exc),
            )
            if bool(args.allow_source_shrink_on_error):
                current_chars = len(source_text_prompt)
                if current_chars > 12000:
                    reduced_chars = max(12000, int(current_chars * 0.8))
                    if reduced_chars < current_chars:
                        source_text_prompt = clip_source_text(seed_doc.source_text, max_chars=reduced_chars)
                        LOGGER.warning(
                            "Reduced source prompt for %s from %d to %d chars after failure.",
                            seed_doc.manifesto_id,
                            current_chars,
                            len(source_text_prompt),
                        )
            prev_expansion = None
            prev_score = None
            continue
        if not expanded:
            prev_expansion = ""
            prev_score = None
            continue
        try:
            score = float(score_fn(expanded))
        except Exception as exc:
            LOGGER.warning(
                "Scoring expanded doc failed for %s attempt %d: %s",
                seed_doc.manifesto_id,
                attempt,
                _http_error_detail(exc),
            )
            prev_expansion = expanded
            prev_score = None
            continue
        attempt_scores.append(score)
        delta = abs(score - target_raw)
        if delta <= float(args.score_tolerance_raw):
            accepted = (expanded, score, attempt)
            break
        prev_expansion = expanded
        prev_score = score

    if accepted is None:
        return None, {
            "source_manifesto_id": seed_doc.manifesto_id,
            "split": split,
            "source_rile_raw": target_raw,
            "last_score_raw": prev_score,
            "attempt_scores_raw": attempt_scores,
            "score_tolerance_raw": float(args.score_tolerance_raw),
            "dspy_guidance_used": bool(dspy_guidance_enabled),
            "guidance_note": guidance_note,
        }

    expanded_text, expanded_score_raw, attempts_used = accepted
    try:
        summary1 = _summarize_text(
            client=teacher_client,
            text=expanded_text,
            source_rile_raw=target_raw,
            hop=1,
            temperature=float(args.summary_temperature),
            max_tokens=int(args.summary_max_tokens),
        )
        summary2 = _summarize_text(
            client=teacher_client,
            text=summary1,
            source_rile_raw=target_raw,
            hop=2,
            temperature=float(args.summary_temperature),
            max_tokens=int(args.summary_max_tokens),
        )

        summary1_score_raw = float(score_fn(summary1))
        summary2_score_raw = float(score_fn(summary2))
        trace_payload = _extract_trace(
            client=teacher_client,
            source_text=source_text_prompt,
            expanded_text=expanded_text,
            summary1=summary1,
            summary2=summary2,
            source_rile_raw=target_raw,
            temperature=float(args.trace_temperature),
            max_tokens=int(args.trace_max_tokens),
            source_max_chars=int(args.trace_source_max_chars),
            expanded_max_chars=int(args.trace_expanded_max_chars),
        )
    except Exception as exc:
        LOGGER.warning(
            "Post-processing failed for %s after accepted expansion: %s",
            seed_doc.manifesto_id,
            _http_error_detail(exc),
        )
        return None, {
            "source_manifesto_id": seed_doc.manifesto_id,
            "split": split,
            "source_rile_raw": target_raw,
            "last_score_raw": float(expanded_score_raw),
            "attempt_scores_raw": attempt_scores,
            "score_tolerance_raw": float(args.score_tolerance_raw),
            "dspy_guidance_used": bool(dspy_guidance_enabled),
            "guidance_note": guidance_note,
            "error": str(exc),
            "stage": "postprocess",
        }

    record = TeacherTraceRecord(
        example_id=example_id,
        split=split,
        source_manifesto_id=seed_doc.manifesto_id,
        source_party_abbrev=seed_doc.party_abbrev,
        source_country_name=seed_doc.country_name,
        source_year=int(seed_doc.year),
        source_rile_raw=target_raw,
        source_bin_name=seed_doc.source_bin_name,
        source_text=seed_doc.source_text,
        expanded_text=expanded_text,
        expanded_score_raw=float(expanded_score_raw),
        expanded_delta_raw=float(expanded_score_raw - target_raw),
        summary1=summary1,
        summary1_score_raw=float(summary1_score_raw),
        summary1_delta_raw=float(summary1_score_raw - target_raw),
        summary2=summary2,
        summary2_score_raw=float(summary2_score_raw),
        summary2_delta_raw=float(summary2_score_raw - target_raw),
        summary2_vs_summary1_delta_raw=float(summary2_score_raw - summary1_score_raw),
        same_side_summary1=strict_same_side_raw(summary1_score_raw, target_raw, neutral_raw=0.0),
        same_side_summary2=strict_same_side_raw(summary2_score_raw, target_raw, neutral_raw=0.0),
        trace_critical_points=[str(v) for v in trace_payload.get("critical_points", [])],
        trace_entities=[str(v) for v in trace_payload.get("entities", [])],
        trace_qualifiers=[str(v) for v in trace_payload.get("qualifiers", [])],
        trace_invariants=[str(v) for v in trace_payload.get("invariants", [])],
        trace_notes=str(trace_payload.get("notes", "") or ""),
        attempts_used=int(attempts_used),
    )
    return record, None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher traces from real manifesto anchors")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--manifesto-ids", type=str, nargs="*", default=None)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--balanced-bins", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--score-tolerance-raw", type=float, default=10.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Concurrent worker count for per-document trace generation (must be >=2 for multi-doc runs).",
    )

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--scorer-base-url", type=str, default=None)
    parser.add_argument("--scorer-model", type=str, default=None)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=180.0)

    parser.add_argument("--expand-temperature", type=float, default=0.4)
    parser.add_argument("--expand-max-tokens", type=int, default=4096)
    parser.add_argument("--summary-temperature", type=float, default=0.2)
    parser.add_argument("--summary-max-tokens", type=int, default=1200)
    parser.add_argument("--trace-temperature", type=float, default=0.1)
    parser.add_argument("--trace-max-tokens", type=int, default=1200)
    parser.add_argument("--score-temperature", type=float, default=0.0)
    parser.add_argument("--score-max-tokens", type=int, default=32)
    parser.add_argument(
        "--emit-labeled-trees",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also emit Stage-0 C-TreePO labeled-tree artifacts with leaf/internal "
            "node scores, summary targets, sibling triples, and idempotence pairs."
        ),
    )
    parser.add_argument(
        "--labeled-tree-leaf-size-chars",
        type=int,
        default=8000,
        help="Fixed leaf size used to build labeled-tree artifacts.",
    )
    parser.add_argument(
        "--labeled-tree-window-overlap-chars",
        type=int,
        default=0,
        help="Window overlap used to build labeled-tree artifacts.",
    )
    parser.add_argument(
        "--labeled-tree-target-leaves-per-doc",
        "--target-leaves-per-doc",
        type=int,
        default=None,
        help=(
            "Optional target number of exact artifact leaves per document. "
            "When set, this overrides fixed-size leaf construction for labeled trees."
        ),
    )
    parser.add_argument(
        "--labeled-tree-node-summary-mode",
        choices=["teacher", "identity", "partial"],
        default="teacher",
        help=(
            "How to populate non-root node summary targets: live teacher calls, "
            "span identity fallback, or partial artifact mode with missing G targets."
        ),
    )
    parser.add_argument(
        "--labeled-tree-node-summary-max-chars",
        type=int,
        default=12000,
        help="Max chars of a node span included in live node-summary prompts.",
    )
    parser.add_argument(
        "--labeled-tree-node-summary-max-tokens",
        type=int,
        default=700,
        help="Max tokens for live node-summary teacher calls.",
    )
    parser.add_argument(
        "--labeled-tree-node-summary-temperature",
        type=float,
        default=0.2,
        help="Temperature for live node-summary teacher calls.",
    )
    parser.add_argument(
        "--labeled-tree-label-source",
        type=str,
        default="manifesto_teacher_trace_model_backed",
        help="Provenance string stored on emitted labeled-tree node labels.",
    )
    parser.add_argument(
        "--dspy-guidance-source-max-chars",
        type=int,
        default=262144,
        help="Max chars for source text in DSPy guidance prompt (<=0 disables clipping).",
    )
    parser.add_argument(
        "--dspy-guidance-expansion-max-chars",
        type=int,
        default=262144,
        help="Max chars for expansion text in DSPy guidance prompt (<=0 disables clipping).",
    )
    parser.add_argument(
        "--previous-expansion-max-chars",
        type=int,
        default=262144,
        help="Max chars from previous expansion included when retrying (<=0 disables clipping).",
    )
    parser.add_argument(
        "--revision-guidance-max-chars",
        type=int,
        default=65536,
        help="Max chars from DSPy guidance block included in retry prompt.",
    )
    parser.add_argument(
        "--trace-source-max-chars",
        type=int,
        default=262144,
        help="Max chars for source text in trace extraction prompt (<=0 disables clipping).",
    )
    parser.add_argument(
        "--trace-expanded-max-chars",
        type=int,
        default=262144,
        help="Max chars for expanded text in trace extraction prompt (<=0 disables clipping).",
    )
    parser.add_argument(
        "--allow-source-shrink-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow shrinking source prompt after request failures (default: disabled).",
    )
    parser.add_argument("--use-dspy-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dspy-guidance-base-url", type=str, default=None)
    parser.add_argument("--dspy-guidance-model", type=str, default=None)
    parser.add_argument("--dspy-guidance-temperature", type=float, default=0.1)
    parser.add_argument("--dspy-guidance-max-tokens", type=int, default=420)
    parser.add_argument(
        "--allow-concurrent-dspy-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow DSPy guidance with num-workers>1. Disabled by default due shared DSPy runtime state; "
            "when disabled, guidance is automatically turned off for concurrent runs."
        ),
    )

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data") / "teacher_traces" / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_requested = int(args.train_size) + int(args.val_size) + int(args.test_size)
    if total_requested <= 0:
        raise ValueError("train_size + val_size + test_size must be positive")

    dataset = ManifestoDataset(require_text=True)
    seed_docs = select_seed_manifestos(
        dataset,
        n_docs=total_requested,
        seed=int(args.seed),
        min_source_chars=int(args.min_source_chars),
        manifesto_ids=args.manifesto_ids,
        balanced_bins=bool(args.balanced_bins),
    )
    if not seed_docs:
        raise RuntimeError("No seed manifestos selected. Relax filters or provide manifesto IDs.")

    if len(seed_docs) < total_requested:
        LOGGER.warning(
            "Requested %d docs but only selected %d eligible manifesto anchors; scaling split counts.",
            total_requested,
            len(seed_docs),
        )
    train_size, val_size, test_size = _scale_split_counts(
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        available=len(seed_docs),
    )
    split_labels = build_split_labels(
        total_docs=len(seed_docs),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=int(args.seed),
    )

    scorer_base_url = args.scorer_base_url or args.teacher_base_url
    scorer_model = args.scorer_model or args.teacher_model

    teacher_client = OpenAIChatClient(
        base_url=args.teacher_base_url,
        model=args.teacher_model,
        api_key=args.teacher_api_key,
        timeout_seconds=float(args.teacher_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )
    scorer_client = OpenAIChatClient(
        base_url=scorer_base_url,
        model=scorer_model,
        api_key=args.scorer_api_key,
        timeout_seconds=float(args.scorer_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )
    score_fn = _build_score_fn(
        scorer_client,
        temperature=float(args.score_temperature),
        max_tokens=int(args.score_max_tokens),
    )
    requested_workers = int(args.num_workers)
    if requested_workers < 1:
        raise ValueError(f"--num-workers must be >= 1 (got {requested_workers})")
    if len(seed_docs) > 1 and requested_workers < 2:
        raise ValueError(
            "Single-worker trace generation is disabled for multi-doc runs. "
            f"Set --num-workers >= 2 (got {requested_workers}, docs={len(seed_docs)})."
        )

    max_workers = requested_workers
    dspy_guidance_requested = bool(args.use_dspy_guidance)
    dspy_guidance_enabled = bool(dspy_guidance_requested)
    if (
        max_workers > 1
        and dspy_guidance_enabled
        and not bool(args.allow_concurrent_dspy_guidance)
    ):
        LOGGER.warning(
            "Disabling DSPy guidance because num_workers=%d and --allow-concurrent-dspy-guidance is false.",
            max_workers,
        )
        dspy_guidance_enabled = False

    dspy_guidance_fn: Optional[Callable[..., str]] = None
    dspy_guidance_base_url = args.dspy_guidance_base_url or args.teacher_base_url
    dspy_guidance_model = args.dspy_guidance_model or args.teacher_model
    if dspy_guidance_enabled:
        dspy_guidance_fn = _build_dspy_guidance_fn(
            base_url=str(dspy_guidance_base_url),
            model=str(dspy_guidance_model),
            api_key=str(args.teacher_api_key),
            temperature=float(args.dspy_guidance_temperature),
            max_tokens=int(args.dspy_guidance_max_tokens),
            source_max_chars=int(args.dspy_guidance_source_max_chars),
            expansion_max_chars=int(args.dspy_guidance_expansion_max_chars),
        )
        LOGGER.info(
            "DSPy guidance enabled: model=%s base_url=%s",
            dspy_guidance_model,
            dspy_guidance_base_url,
        )

    work_items: List[Tuple[int, Any, str, str]] = []
    split_ordinals: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for idx, (seed_doc, split) in enumerate(zip(seed_docs, split_labels)):
        split_ordinals[split] = split_ordinals.get(split, 0) + 1
        example_id = f"teacher_trace_{split}_{split_ordinals[split]:04d}"
        work_items.append((idx, seed_doc, split, example_id))

    worker_count = min(max_workers, max(1, len(work_items)))
    if worker_count > 1:
        LOGGER.info("Processing %d seed docs with %d concurrent workers", len(work_items), worker_count)

    result_slots: List[Optional[Tuple[Optional[TeacherTraceRecord], Optional[Dict[str, Any]]]]] = [
        None
    ] * len(work_items)
    accepted_count = 0

    if worker_count > 1 and len(work_items) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {}
            for idx, seed_doc, split, example_id in work_items:
                future = pool.submit(
                    _process_seed_doc,
                    seed_doc=seed_doc,
                    split=split,
                    example_id=example_id,
                    args=args,
                    teacher_client=teacher_client,
                    score_fn=score_fn,
                    dspy_guidance_fn=dspy_guidance_fn,
                    dspy_guidance_enabled=dspy_guidance_enabled,
                )
                future_to_idx[future] = int(idx)

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                seed_doc = seed_docs[idx]
                split = split_labels[idx]
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = (
                        None,
                        {
                            "source_manifesto_id": seed_doc.manifesto_id,
                            "split": split,
                            "source_rile_raw": float(seed_doc.source_rile_raw),
                            "score_tolerance_raw": float(args.score_tolerance_raw),
                            "dspy_guidance_used": bool(dspy_guidance_enabled),
                            "error": str(exc),
                            "stage": "worker_exception",
                        },
                    )
                result_slots[idx] = outcome
                if outcome[0] is not None:
                    accepted_count += 1
                completed += 1
                if completed % 10 == 0:
                    LOGGER.info("Processed %d/%d seed docs; accepted=%d", completed, len(seed_docs), accepted_count)
    else:
        for idx, seed_doc, split, example_id in work_items:
            outcome = _process_seed_doc(
                seed_doc=seed_doc,
                split=split,
                example_id=example_id,
                args=args,
                teacher_client=teacher_client,
                score_fn=score_fn,
                dspy_guidance_fn=dspy_guidance_fn,
                dspy_guidance_enabled=dspy_guidance_enabled,
            )
            result_slots[idx] = outcome
            if outcome[0] is not None:
                accepted_count += 1
            if (idx + 1) % 10 == 0:
                LOGGER.info("Processed %d/%d seed docs; accepted=%d", idx + 1, len(seed_docs), accepted_count)

    records: List[TeacherTraceRecord] = []
    rejected_rows: List[Dict[str, Any]] = []
    for item in result_slots:
        if item is None:
            continue
        record, rejected = item
        if record is not None:
            records.append(record)
        if rejected is not None:
            rejected_rows.append(rejected)

    records_path = output_dir / "teacher_trace_records.jsonl"
    benchmark_path = output_dir / "benchmark_docs.jsonl"
    summary_pairs_path = output_dir / "summary_training_pairs.jsonl"
    trace_rows_path = output_dir / "trace_artifacts.jsonl"
    rejected_path = output_dir / "rejected_records.jsonl"

    write_teacher_trace_records_jsonl(records_path, records)
    write_jsonl(benchmark_path, build_benchmark_docs(records))
    write_jsonl(summary_pairs_path, build_summary_pair_rows(records))
    write_jsonl(
        trace_rows_path,
        (
            {
                "example_id": row.example_id,
                "split": row.split,
                "source_manifesto_id": row.source_manifesto_id,
                "trace_critical_points": row.trace_critical_points,
                "trace_entities": row.trace_entities,
                "trace_qualifiers": row.trace_qualifiers,
                "trace_invariants": row.trace_invariants,
                "trace_notes": row.trace_notes,
            }
            for row in records
        ),
    )
    if rejected_rows:
        write_jsonl(rejected_path, rejected_rows)

    labeled_trees_path: Optional[Path] = None
    labeled_tree_count = 0
    labeled_tree_failures: List[Dict[str, Any]] = []
    if bool(args.emit_labeled_trees):
        labeled_trees = []
        for idx, row in enumerate(records, start=1):
            try:
                LOGGER.info(
                    "Building labeled tree artifact %d/%d for %s",
                    idx,
                    len(records),
                    row.example_id,
                )
                node_summary_mode = str(args.labeled_tree_node_summary_mode)
                node_summary_fn = (
                    _build_labeled_tree_node_summary_fn(
                        client=teacher_client,
                        source_rile_raw=float(row.source_rile_raw),
                        temperature=float(args.labeled_tree_node_summary_temperature),
                        max_tokens=int(args.labeled_tree_node_summary_max_tokens),
                        max_chars=int(args.labeled_tree_node_summary_max_chars),
                    )
                    if node_summary_mode == "teacher"
                    else None
                )
                labeled_trees.append(
                    build_labeled_tree_from_text(
                        doc_id=str(row.example_id),
                        text=str(row.expanded_text),
                        document_score=float(row.source_rile_raw),
                        split=str(row.split),
                        score_fn=score_fn,
                        window_size=int(args.labeled_tree_leaf_size_chars),
                        window_overlap=int(args.labeled_tree_window_overlap_chars),
                        target_leaves_per_doc=args.labeled_tree_target_leaves_per_doc,
                        label_source=str(args.labeled_tree_label_source),
                        root_summary=str(row.summary1),
                        resummary_target=str(row.summary2),
                        node_summary_fn=node_summary_fn,
                        fill_missing_summaries_from_span=(node_summary_mode == "identity"),
                        summary_source=(
                            "teacher_trace_node_summaries"
                            if node_summary_mode == "teacher"
                            else (
                                "span_identity_fallback"
                                if node_summary_mode == "identity"
                                else "teacher_trace_root_only_partial"
                            )
                        ),
                        extra_metadata={
                            "source_manifesto_id": str(row.source_manifesto_id),
                            "source_rile_raw": float(row.source_rile_raw),
                            "expanded_score_raw": float(row.expanded_score_raw),
                            "summary1_score_raw": float(row.summary1_score_raw),
                            "summary2_score_raw": float(row.summary2_score_raw),
                            "teacher_model": str(args.teacher_model),
                            "teacher_base_url": str(args.teacher_base_url),
                            "scorer_model": str(scorer_model),
                            "scorer_base_url": str(scorer_base_url),
                            "fixed_leaf_size_identity": {
                                "leaf_size_chars": int(args.labeled_tree_leaf_size_chars),
                                "window_overlap_chars": int(args.labeled_tree_window_overlap_chars),
                            },
                            "summary_target_policy": {
                                "root": "teacher_trace_summary1",
                                "idempotence": "teacher_trace_summary2",
                                "non_root": node_summary_mode,
                            },
                        },
                    )
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to build labeled tree artifact for %s: %s",
                    row.example_id,
                    _http_error_detail(exc),
                )
                labeled_tree_failures.append(
                    {
                        "example_id": row.example_id,
                        "split": row.split,
                        "source_manifesto_id": row.source_manifesto_id,
                        "error": str(exc),
                    }
                )
        labeled_trees_path = output_dir / "labeled_trees.jsonl"
        labeled_tree_count = len(labeled_trees)
        write_labeled_trees_jsonl(labeled_trees_path, labeled_trees)

    metrics = summarize_teacher_trace_records(records)
    manifest = {
        "requested_docs": total_requested,
        "selected_seed_docs": len(seed_docs),
        "accepted_docs": len(records),
        "rejected_docs": len(rejected_rows),
        "split_counts_requested": {
            "train": int(args.train_size),
            "val": int(args.val_size),
            "test": int(args.test_size),
        },
        "split_counts_effective": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "teacher": {
            "base_url": args.teacher_base_url,
            "model": args.teacher_model,
        },
        "scorer": {
            "base_url": scorer_base_url,
            "model": scorer_model,
        },
        "score_tolerance_raw": float(args.score_tolerance_raw),
        "num_workers": int(worker_count),
        "allow_source_shrink_on_error": bool(args.allow_source_shrink_on_error),
        "dspy_guidance": {
            "requested": bool(dspy_guidance_requested),
            "enabled": bool(dspy_guidance_enabled),
            "allow_concurrent": bool(args.allow_concurrent_dspy_guidance),
            "base_url": str(dspy_guidance_base_url),
            "model": str(dspy_guidance_model),
            "temperature": float(args.dspy_guidance_temperature),
            "max_tokens": int(args.dspy_guidance_max_tokens),
            "source_max_chars": int(args.dspy_guidance_source_max_chars),
            "expansion_max_chars": int(args.dspy_guidance_expansion_max_chars),
        },
        "prompt_clipping": {
            "previous_expansion_max_chars": int(args.previous_expansion_max_chars),
            "revision_guidance_max_chars": int(args.revision_guidance_max_chars),
            "trace_source_max_chars": int(args.trace_source_max_chars),
            "trace_expanded_max_chars": int(args.trace_expanded_max_chars),
        },
        "labeled_trees": {
            "emitted": bool(args.emit_labeled_trees),
            "leaf_size_chars": int(args.labeled_tree_leaf_size_chars),
            "window_overlap_chars": int(args.labeled_tree_window_overlap_chars),
            "target_leaves_per_doc": (
                int(args.labeled_tree_target_leaves_per_doc)
                if args.labeled_tree_target_leaves_per_doc is not None
                else None
            ),
            "node_summary_mode": str(args.labeled_tree_node_summary_mode),
            "node_summary_max_chars": int(args.labeled_tree_node_summary_max_chars),
            "node_summary_max_tokens": int(args.labeled_tree_node_summary_max_tokens),
            "label_source": str(args.labeled_tree_label_source),
            "accepted_artifacts": int(labeled_tree_count),
            "failures": labeled_tree_failures,
        },
        "metrics": metrics,
        "paths": {
            "records": str(records_path),
            "benchmark_docs": str(benchmark_path),
            "summary_training_pairs": str(summary_pairs_path),
            "trace_artifacts": str(trace_rows_path),
            "rejected_records": str(rejected_path) if rejected_rows else None,
            "labeled_trees": str(labeled_trees_path) if labeled_trees_path else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOGGER.info("Teacher trace generation complete: accepted=%d rejected=%d", len(records), len(rejected_rows))
    LOGGER.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
