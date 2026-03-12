#!/usr/bin/env python3
"""
Quality-aware GenRM A/B test on the same documents and candidate pools.

For each manifesto document:
1. Generate the same candidate summaries once.
2. Run tournament selection with GenRM fast mode.
3. Run tournament selection with GenRM think mode.
4. Score each mode's winner with an oracle scorer prompt.
5. Compare winner quality against the document's reference RILE score.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_detection import detect_model_async  # noqa: E402
from src.tasks.manifesto import ManifestoDataset, RILE_TASK_CONTEXT  # noqa: E402
from src.training.preference.genrm import is_genrm_error  # noqa: E402
from src.training.preference.genrm_batch import AsyncBatchGenRMClient, GenRMComparisonRequest  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class WinnerABRow:
    doc_id: str
    country_code: int
    year: int
    reference_rile: float
    n_candidates: int
    fast_winner_idx: int
    think_winner_idx: int
    same_winner: bool
    fast_winner_score: float
    think_winner_score: float
    fast_abs_error: float
    think_abs_error: float
    better_mode: str
    fast_comparison_errors: int
    think_comparison_errors: int
    candidate_gen_seconds: float
    fast_tournament_seconds: float
    think_tournament_seconds: float
    oracle_scoring_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_temperatures(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Temperature list is empty")
    return values


def _candidate_messages(text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You summarize political manifesto text. Preserve policy content and ideological "
                "signals. Return only the summary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize this manifesto excerpt in 2-4 sentences, preserving policy details and "
                "left-right positioning clues.\n\n"
                f"{text}\n\n"
                "Summary:"
            ),
        },
    ]


def _oracle_messages(summary: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert CMP manifesto coder. Return exactly one numeric RILE score "
                "between -100 and +100."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{RILE_TASK_CONTEXT}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                "Output only the numeric RILE score in [-100, +100]."
            ),
        },
    ]


async def _chat_completion(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if extra_payload:
        payload.update(extra_payload)
    async with session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        data = json.loads(text)
    return str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")


async def _generate_candidates(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    doc_text: str,
    temperatures: Sequence[float],
    max_tokens: int,
    unique_only: bool = True,
) -> List[str]:
    tasks: List[asyncio.Task] = []
    for temp in temperatures:
        task = asyncio.create_task(
            _chat_completion(
                session=session,
                base_url=base_url,
                model=model,
                messages=_candidate_messages(doc_text),
                max_tokens=max_tokens,
                temperature=temp,
            )
        )
        tasks.append(task)
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    candidates: List[str] = []
    for item in raw:
        if isinstance(item, Exception):
            continue
        text = str(item).strip()
        if not text:
            continue
        candidates.append(text)
    if unique_only:
        deduped = []
        seen = set()
        for c in candidates:
            key = c.strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        candidates = deduped
    return candidates


async def _score_summary_with_oracle(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    summary: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool,
    force_json_response: bool,
) -> float:
    payload: Dict[str, Any] = {}
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if force_json_response:
        payload["response_format"] = {"type": "json_object"}

    attempts: List[Optional[Dict[str, Any]]] = [payload or None]
    if payload:
        attempts.append(None)

    last_response = ""
    for extra_payload in attempts:
        try:
            response = await _chat_completion(
                session=session,
                base_url=base_url,
                model=model,
                messages=_oracle_messages(summary),
                max_tokens=max_tokens,
                temperature=temperature,
                extra_payload=extra_payload,
            )
        except Exception:
            continue
        last_response = response
        parsed = _parse_oracle_score_response(response)
        if parsed is not None:
            return float(parsed)

    raise ValueError(f"Could not parse oracle score from: {last_response[:160]}")


async def _run_tournament(
    *,
    client: AsyncBatchGenRMClient,
    doc_id: str,
    original_text: str,
    candidates: Sequence[str],
) -> Tuple[int, List[float], int]:
    """
    Run full pairwise tournament and return winner index.

    Returns:
        winner_idx, wins_per_candidate, error_count
    """
    k = len(candidates)
    if k < 2:
        return 0, [0.0] * k, 0

    wins = [0.0 for _ in range(k)]
    error_count = 0
    pair_info: List[Tuple[int, int, asyncio.Task]] = []
    for i in range(k):
        for j in range(i + 1, k):
            req = GenRMComparisonRequest(
                request_id=f"{doc_id}_{i}_{j}",
                context="Preserve ideological and policy information.",
                original_text=original_text,
                summary_a=candidates[i],
                summary_b=candidates[j],
                law_type="sufficiency",
            )
            pair_info.append((i, j, asyncio.create_task(client.call(req))))

    results = await asyncio.gather(*[t for _, _, t in pair_info], return_exceptions=True)
    for (i, j, _), result in zip(pair_info, results):
        if isinstance(result, Exception):
            error_count += 1
            wins[i] += 0.5
            wins[j] += 0.5
            continue
        if is_genrm_error(result):
            error_count += 1
            wins[i] += 0.5
            wins[j] += 0.5
            continue
        preferred = str(getattr(result, "preferred", "tie")).lower()
        if preferred == "a":
            wins[i] += 1.0
        elif preferred == "b":
            wins[j] += 1.0
        else:
            wins[i] += 0.5
            wins[j] += 0.5

    winner_idx = max(range(k), key=lambda idx: (wins[idx], -idx))
    return winner_idx, wins, error_count


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _safe_rate(numer: int, denom: int) -> float:
    return (float(numer) / float(denom)) if denom > 0 else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return float(ordered[idx])


def _strip_think_content(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _parse_numeric_line(text: str) -> Optional[float]:
    for line in str(text).splitlines():
        raw = line.strip()
        if not raw:
            continue
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw):
            value = float(raw)
            if -100.0 <= value <= 100.0:
                return value
    return None


def _to_valid_score(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if -100.0 <= parsed <= 100.0:
        return parsed
    return None


def _extract_score_hint(text: str) -> Optional[float]:
    if not text:
        return None
    matches = re.findall(
        r"(?i)(?:rile(?:\s+score)?|score)\D{0,24}?([-+]?\d+(?:\.\d+)?)",
        str(text),
    )
    valid: List[float] = []
    for token in matches:
        parsed = _to_valid_score(token)
        if parsed is not None:
            valid.append(parsed)
    if valid:
        return float(valid[-1])
    return None


def _parse_score_from_json_obj(obj: Any) -> Optional[float]:
    preferred_keys = ("score", "rile", "rile_score", "value")
    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                parsed_key_value = _to_valid_score(obj[key])
                if parsed_key_value is not None:
                    return parsed_key_value
        for value in obj.values():
            parsed = _parse_score_from_json_obj(value)
            if parsed is not None:
                return parsed
    if isinstance(obj, list):
        for value in obj:
            parsed = _parse_score_from_json_obj(value)
            if parsed is not None:
                return parsed
    if isinstance(obj, str):
        line_value = _parse_numeric_line(obj)
        if line_value is not None:
            return line_value
        hinted = _extract_score_hint(obj)
        if hinted is not None:
            return hinted
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", obj)
        valid_numbers = [_to_valid_score(token) for token in matches]
        valid_numbers = [v for v in valid_numbers if v is not None]
        if valid_numbers:
            return float(valid_numbers[-1])
    return None


def _parse_score_from_json_text(text: str) -> Optional[float]:
    raw = str(text).strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed_json = json.loads(raw)
    except Exception:
        return None
    return _parse_score_from_json_obj(parsed_json)


def _parse_oracle_score_response(response: str) -> Optional[float]:
    cleaned = _strip_think_content(response)
    candidates = [cleaned, str(response)]
    for text in candidates:
        parsed = _parse_numeric_line(text)
        if parsed is not None:
            return parsed
    for text in candidates:
        parsed = _parse_score_from_json_text(text)
        if parsed is not None:
            return parsed
    for text in candidates:
        parsed = _extract_score_hint(text)
        if parsed is not None:
            return float(parsed)
    for text in candidates:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        numeric = []
        for token in matches:
            try:
                value = float(token)
            except ValueError:
                continue
            if -100.0 <= value <= 100.0:
                numeric.append(value)
        if numeric:
            return float(numeric[-1])
    return None


def _write_csv(path: Path, rows: Sequence[WinnerABRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].to_dict().keys()) if rows else [
        "doc_id",
        "country_code",
        "year",
        "reference_rile",
        "n_candidates",
        "fast_winner_idx",
        "think_winner_idx",
        "same_winner",
        "fast_winner_score",
        "think_winner_score",
        "fast_abs_error",
        "think_abs_error",
        "better_mode",
        "fast_comparison_errors",
        "think_comparison_errors",
        "candidate_gen_seconds",
        "fast_tournament_seconds",
        "think_tournament_seconds",
        "oracle_scoring_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _parse_countries(raw: str) -> Optional[List[int]]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-focused fast-vs-think GenRM tournament A/B test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fast-url", default="http://localhost:8001/v1")
    parser.add_argument("--think-url", default="http://localhost:8002/v1")
    parser.add_argument(
        "--candidate-url",
        default=None,
        help="Endpoint used to generate candidate summaries (default: fast-url)",
    )
    parser.add_argument(
        "--oracle-url",
        default="http://localhost:8000/v1",
        help=(
            "Endpoint used to score winners on RILE scale. "
            "Use a scorer/task model endpoint (avoid GenRM endpoint)."
        ),
    )

    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--countries", default="51,41")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=None)

    parser.add_argument("--k-candidates", type=int, default=4)
    parser.add_argument("--candidate-temperatures", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--doc-max-chars", type=int, default=6000)
    parser.add_argument("--candidate-max-tokens", type=int, default=256)
    parser.add_argument("--oracle-max-tokens", type=int, default=64)
    parser.add_argument("--oracle-temperature", type=float, default=0.0)
    parser.add_argument(
        "--oracle-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request non-thinking mode for oracle scoring when supported.",
    )
    parser.add_argument(
        "--oracle-force-json-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request JSON output for oracle scoring when supported.",
    )

    parser.add_argument("--doc-concurrency", type=int, default=4)
    parser.add_argument("--genrm-max-concurrent", type=int, default=8)
    parser.add_argument("--genrm-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/genrm_winner_ab_test.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/genrm_winner_ab_test.csv"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    candidate_url = str(args.candidate_url or args.fast_url)
    oracle_url = str(args.oracle_url or args.think_url)
    countries = _parse_countries(args.countries)
    temperatures = _parse_temperatures(args.candidate_temperatures)[: int(args.k_candidates)]

    dataset = ManifestoDataset(
        countries=countries,
        min_year=args.min_year,
        max_year=args.max_year,
        require_text=True,
    )
    sample_ids = dataset.get_all_ids()[: int(args.max_docs)]
    samples = [dataset.get_sample(sid) for sid in sample_ids]
    samples = [s for s in samples if s is not None and s.text]
    if not samples:
        raise RuntimeError("No samples available for the configured filters")

    fast_model = await detect_model_async(args.fast_url, fallback="default", timeout=10.0)
    think_model = await detect_model_async(args.think_url, fallback="default", timeout=10.0)
    candidate_model = await detect_model_async(candidate_url, fallback="default", timeout=10.0)
    oracle_model = await detect_model_async(oracle_url, fallback="default", timeout=10.0)
    logger.info(
        "Models: fast=%s think=%s candidate=%s oracle=%s",
        fast_model,
        think_model,
        candidate_model,
        oracle_model,
    )
    if "genrm" in str(oracle_model).lower():
        logger.warning(
            "Oracle endpoint model appears to be GenRM (%s). "
            "For meaningful winner-quality comparison, prefer a non-GenRM scorer endpoint "
            "(e.g., task/oracle model on port 8000).",
            oracle_model,
        )

    timeout = aiohttp.ClientTimeout(total=float(args.http_timeout_seconds))
    connector = aiohttp.TCPConnector(limit=max(8, int(args.doc_concurrency) * 4))
    sem = asyncio.Semaphore(max(1, int(args.doc_concurrency)))

    fast_client = AsyncBatchGenRMClient(
        base_url=str(args.fast_url),
        max_concurrent=int(args.genrm_max_concurrent),
        model=fast_model,
        request_timeout=float(args.genrm_timeout_seconds),
        max_tokens=256,
        temperature=0.6,
        top_p=0.95,
        disable_thinking=True,
        force_json_response=True,
    )
    think_client = AsyncBatchGenRMClient(
        base_url=str(args.think_url),
        max_concurrent=int(args.genrm_max_concurrent),
        model=think_model,
        request_timeout=float(args.genrm_timeout_seconds),
        max_tokens=256,
        temperature=0.6,
        top_p=0.95,
        disable_thinking=False,
        force_json_response=False,
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with fast_client, think_client:
            started = datetime.now(timezone.utc)

            async def process_one(sample_idx: int, sample) -> Optional[WinnerABRow]:
                async with sem:
                    doc_text = sample.text[: int(args.doc_max_chars)]
                    candidate_start = time.monotonic()
                    candidates = await _generate_candidates(
                        session=session,
                        base_url=candidate_url,
                        model=candidate_model,
                        doc_text=doc_text,
                        temperatures=temperatures,
                        max_tokens=int(args.candidate_max_tokens),
                        unique_only=True,
                    )
                    candidate_gen_seconds = float(time.monotonic() - candidate_start)
                    if len(candidates) < 2:
                        logger.warning("Skipping %s: only %d candidates", sample.manifesto_id, len(candidates))
                        return None

                    async def _timed(coro):
                        started = time.monotonic()
                        result = await coro
                        return result, float(time.monotonic() - started)

                    (fast_result, fast_tournament_seconds), (think_result, think_tournament_seconds) = await asyncio.gather(
                        _timed(
                            _run_tournament(
                                client=fast_client,
                                doc_id=f"{sample.manifesto_id}_fast",
                                original_text=doc_text,
                                candidates=candidates,
                            )
                        ),
                        _timed(
                            _run_tournament(
                                client=think_client,
                                doc_id=f"{sample.manifesto_id}_think",
                                original_text=doc_text,
                                candidates=candidates,
                            )
                        ),
                    )
                    fast_idx, _, fast_errs = fast_result
                    think_idx, _, think_errs = think_result

                    fast_summary = candidates[fast_idx]
                    think_summary = candidates[think_idx]
                    oracle_scoring_start = time.monotonic()
                    fast_score, think_score = await asyncio.gather(
                        _score_summary_with_oracle(
                            session=session,
                            base_url=oracle_url,
                            model=oracle_model,
                            summary=fast_summary,
                            max_tokens=int(args.oracle_max_tokens),
                            temperature=float(args.oracle_temperature),
                            disable_thinking=bool(args.oracle_disable_thinking),
                            force_json_response=bool(args.oracle_force_json_response),
                        ),
                        _score_summary_with_oracle(
                            session=session,
                            base_url=oracle_url,
                            model=oracle_model,
                            summary=think_summary,
                            max_tokens=int(args.oracle_max_tokens),
                            temperature=float(args.oracle_temperature),
                            disable_thinking=bool(args.oracle_disable_thinking),
                            force_json_response=bool(args.oracle_force_json_response),
                        ),
                    )
                    oracle_scoring_seconds = float(time.monotonic() - oracle_scoring_start)

                    reference = float(sample.rile)
                    fast_abs_error = abs(fast_score - reference)
                    think_abs_error = abs(think_score - reference)
                    if abs(fast_abs_error - think_abs_error) < 1e-6:
                        better_mode = "tie"
                    elif fast_abs_error < think_abs_error:
                        better_mode = "fast"
                    else:
                        better_mode = "think"

                    return WinnerABRow(
                        doc_id=sample.manifesto_id,
                        country_code=int(sample.country_code),
                        year=int(sample.year),
                        reference_rile=reference,
                        n_candidates=len(candidates),
                        fast_winner_idx=int(fast_idx),
                        think_winner_idx=int(think_idx),
                        same_winner=bool(fast_idx == think_idx),
                        fast_winner_score=float(fast_score),
                        think_winner_score=float(think_score),
                        fast_abs_error=float(fast_abs_error),
                        think_abs_error=float(think_abs_error),
                        better_mode=better_mode,
                        fast_comparison_errors=int(fast_errs),
                        think_comparison_errors=int(think_errs),
                        candidate_gen_seconds=float(candidate_gen_seconds),
                        fast_tournament_seconds=float(fast_tournament_seconds),
                        think_tournament_seconds=float(think_tournament_seconds),
                        oracle_scoring_seconds=float(oracle_scoring_seconds),
                    )

            tasks = [asyncio.create_task(process_one(i, s)) for i, s in enumerate(samples)]
            raw_rows = await asyncio.gather(*tasks, return_exceptions=True)
            rows: List[WinnerABRow] = []
            failures = 0
            for item in raw_rows:
                if isinstance(item, Exception):
                    failures += 1
                    logger.warning("Document failed: %s", item)
                    continue
                if item is None:
                    continue
                rows.append(item)

            ended = datetime.now(timezone.utc)

    if not rows:
        raise RuntimeError("No rows produced. Check endpoints and candidate generation.")

    fast_errors = [r.fast_abs_error for r in rows]
    think_errors = [r.think_abs_error for r in rows]
    candidate_gen_seconds = [r.candidate_gen_seconds for r in rows]
    fast_tournament_seconds = [r.fast_tournament_seconds for r in rows]
    think_tournament_seconds = [r.think_tournament_seconds for r in rows]
    oracle_scoring_seconds = [r.oracle_scoring_seconds for r in rows]
    fast_better = sum(1 for r in rows if r.better_mode == "fast")
    think_better = sum(1 for r in rows if r.better_mode == "think")
    ties = sum(1 for r in rows if r.better_mode == "tie")
    same_winner = sum(1 for r in rows if r.same_winner)
    elapsed_seconds = max((ended - started).total_seconds(), 1e-9)

    summary = {
        "n_rows": len(rows),
        "n_failures": int(failures),
        "same_winner_rate": _safe_rate(same_winner, len(rows)),
        "fast_better_rate": _safe_rate(fast_better, len(rows)),
        "think_better_rate": _safe_rate(think_better, len(rows)),
        "tie_rate": _safe_rate(ties, len(rows)),
        "fast_abs_error_mean": _mean(fast_errors),
        "fast_abs_error_median": _median(fast_errors),
        "think_abs_error_mean": _mean(think_errors),
        "think_abs_error_median": _median(think_errors),
        "candidate_gen_seconds_mean": _mean(candidate_gen_seconds),
        "candidate_gen_seconds_p95": _p95(candidate_gen_seconds),
        "fast_tournament_seconds_mean": _mean(fast_tournament_seconds),
        "fast_tournament_seconds_p95": _p95(fast_tournament_seconds),
        "think_tournament_seconds_mean": _mean(think_tournament_seconds),
        "think_tournament_seconds_p95": _p95(think_tournament_seconds),
        "oracle_scoring_seconds_mean": _mean(oracle_scoring_seconds),
        "oracle_scoring_seconds_p95": _p95(oracle_scoring_seconds),
        "docs_per_minute": float(len(rows) / elapsed_seconds * 60.0),
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "fast_genrm_stats": str(fast_client.stats),
        "think_genrm_stats": str(think_client.stats),
    }

    print("=" * 96)
    print("GenRM Winner A/B Summary")
    print("=" * 96)
    print(f"rows={summary['n_rows']} failures={summary['n_failures']}")
    print(
        f"same_winner={100.0 * summary['same_winner_rate']:.1f}% "
        f"fast_better={100.0 * summary['fast_better_rate']:.1f}% "
        f"think_better={100.0 * summary['think_better_rate']:.1f}% "
        f"tie={100.0 * summary['tie_rate']:.1f}%"
    )
    print(
        f"fast_abs_error mean/median={summary['fast_abs_error_mean']:.3f}/"
        f"{summary['fast_abs_error_median']:.3f}"
    )
    print(
        f"think_abs_error mean/median={summary['think_abs_error_mean']:.3f}/"
        f"{summary['think_abs_error_median']:.3f}"
    )
    print(
        "stage_seconds mean/p95: "
        f"candidate={summary['candidate_gen_seconds_mean']:.2f}/{summary['candidate_gen_seconds_p95']:.2f} "
        f"fast_tournament={summary['fast_tournament_seconds_mean']:.2f}/{summary['fast_tournament_seconds_p95']:.2f} "
        f"think_tournament={summary['think_tournament_seconds_mean']:.2f}/{summary['think_tournament_seconds_p95']:.2f} "
        f"oracle={summary['oracle_scoring_seconds_mean']:.2f}/{summary['oracle_scoring_seconds_p95']:.2f}"
    )
    print(f"throughput: docs_per_minute={summary['docs_per_minute']:.2f}")
    print()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "fast_url": args.fast_url,
            "think_url": args.think_url,
            "candidate_url": candidate_url,
            "oracle_url": oracle_url,
            "max_docs": int(args.max_docs),
            "countries": countries,
            "min_year": args.min_year,
            "max_year": args.max_year,
            "k_candidates": int(args.k_candidates),
            "candidate_temperatures": temperatures,
            "doc_max_chars": int(args.doc_max_chars),
            "candidate_max_tokens": int(args.candidate_max_tokens),
            "oracle_max_tokens": int(args.oracle_max_tokens),
            "oracle_temperature": float(args.oracle_temperature),
            "oracle_disable_thinking": bool(args.oracle_disable_thinking),
            "oracle_force_json_response": bool(args.oracle_force_json_response),
            "doc_concurrency": int(args.doc_concurrency),
            "genrm_max_concurrent": int(args.genrm_max_concurrent),
            "genrm_timeout_seconds": float(args.genrm_timeout_seconds),
            "http_timeout_seconds": float(args.http_timeout_seconds),
            "models": {
                "fast": fast_model,
                "think": think_model,
                "candidate": candidate_model,
                "oracle": oracle_model,
            },
        },
        "summary": summary,
        "rows": [r.to_dict() for r in rows],
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(args.output_csv, rows)
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
