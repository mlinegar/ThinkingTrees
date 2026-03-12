#!/usr/bin/env python3
"""
Engram multilingual smoke test for ThinkingTrees.

This script loads a few manifesto texts (optionally in different languages),
runs summarization with and without Engram-style STATIC MEMORY injection, and
reports how many extracted memory items are preserved verbatim in the summary.

This is *not* a semantic retrieval test. It is a prompt-level check that:
  - Engram extraction produces sane items on non-English text.
  - Injecting STATIC MEMORY improves verbatim preservation of those items.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


DEFAULT_IDS = [
    # Sweden: Social Democratic Labour Party (Swedish)
    "11320_199809",
    # Spain: PSOE (Spanish)
    "33320_199603",
]

DEFAULT_RUBRIC = (
    "Preserve named entities, organizations, locations, numeric quantities (including %), "
    "and dates exactly as written in the source text. "
    "Keep the summary in the same language as the source (do not translate). "
    "Be concise."
)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_manifesto_text(manifesto_id: str) -> str:
    path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "texts" / f"{manifesto_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifesto text file: {path}")
    return _read_text_file(path)


def _load_manifesto_metadata(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    csv_path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "manifesto_maindataset.csv"
    if not csv_path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return {}
    try:
        df["manifesto_id"] = df["party"].astype(str) + "_" + df["date"].astype(str)
    except Exception:
        return {}

    wanted = set(str(x) for x in ids)
    sub = df[df["manifesto_id"].astype(str).isin(wanted)]
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in sub.iterrows():
        manifesto_id = str(row.get("manifesto_id", "")).strip()
        if not manifesto_id:
            continue
        out[manifesto_id] = {
            "partyname": row.get("partyname"),
            "countryname": row.get("countryname"),
            "date": row.get("date"),
            "rile": row.get("rile"),
            "parfam": row.get("parfam"),
        }
    return out


def _clip_text(text: str, max_chars: int) -> str:
    return _slice_text(text, start_char=0, max_chars=max_chars)


def _slice_text(text: str, *, start_char: int, max_chars: int) -> str:
    raw = str(text or "")
    start = max(0, int(start_char))
    if start >= len(raw):
        return ""
    out = raw[start:]
    if max_chars <= 0:
        return out
    return out[:max_chars]


def _preservation_stats(items: List[str], summary: str) -> Dict[str, Any]:
    summary_text = str(summary or "")
    preserved = [item for item in items if item and item in summary_text]
    missing = [item for item in items if item and item not in summary_text]
    denom = max(1, len(items))
    return {
        "items_total": len(items),
        "preserved": len(preserved),
        "missing": len(missing),
        "recall": len(preserved) / float(denom),
        "preserved_items": preserved,
        "missing_items": missing,
    }


def _summarize(
    *,
    client: Any,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    from src.core.prompting import clean_summary_text

    response = client.chat(
        messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )
    content = clean_summary_text(getattr(response, "content", ""))
    usage = {}
    try:
        usage = {
            "model": getattr(response, "model", None),
            "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(response, "total_tokens", 0) or 0),
        }
    except Exception:
        usage = {}
    return content, usage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multilingual Engram STATIC MEMORY smoke test (summarize with/without injection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="Manifesto IDs to test")
    parser.add_argument("--port", type=int, default=8000, help="Task model port (vLLM OpenAI-compatible)")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Max chars per document after --start-char (0 = no clip; may exceed model context)",
    )
    parser.add_argument(
        "--start-char",
        type=int,
        default=0,
        help="Start offset into the document before clipping (useful to skip TOC/headers)",
    )
    parser.add_argument("--max-tokens", type=int, default=450, help="Max tokens for summary generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--rubric", type=str, default=DEFAULT_RUBRIC, help="Preservation rubric")
    parser.add_argument("--engram-max-items", type=int, default=32, help="STATIC MEMORY max items")
    parser.add_argument("--engram-max-chars", type=int, default=1200, help="STATIC MEMORY max total chars")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="Skip model calls; only show extraction stats")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ids = [str(x).strip() for x in (args.ids or []) if str(x).strip()]
    if not ids:
        logger.error("No ids provided")
        return 2

    meta = _load_manifesto_metadata(ids)

    from src.core.engram_memory import EngramMemoryConfig, extract_engram_memory_items
    from src.core.engram_prompting import wrap_summarize_prompt_with_engram_memory
    from src.core.prompting import default_summarize_prompt

    cfg = EngramMemoryConfig(
        enabled=True,
        max_items=int(args.engram_max_items),
        max_chars=int(args.engram_max_chars),
    )
    engram_prompt = wrap_summarize_prompt_with_engram_memory(default_summarize_prompt, cfg)

    client = None
    model_id = None
    if not args.dry_run:
        from src.core.llm_client import LLMClient, LLMConfig

        client = LLMClient(
            LLMConfig.vllm(
                model="default",
                port=int(args.port),
                timeout=120.0,
                max_retries=1,
            ),
            enable_cache=True,
            cache_size=2048,
        )
        model_id = client.config.model

    runs: List[Dict[str, Any]] = []

    for manifesto_id in ids:
        raw = _load_manifesto_text(manifesto_id)
        text = _slice_text(raw, start_char=int(args.start_char), max_chars=int(args.max_chars))
        items = extract_engram_memory_items(text, cfg)

        info = meta.get(manifesto_id, {})
        header = f"{manifesto_id}"
        if info.get("countryname") or info.get("partyname"):
            header += f" | {info.get('countryname', 'unknown')} | {info.get('partyname', 'unknown')}"
        logger.info("=" * 88)
        logger.info(header)
        logger.info("Chars: %d (raw=%d) | Engram items: %d", len(text), len(raw), len(items))
        if items:
            preview_items = ", ".join(items[:10])
            logger.info("STATIC MEMORY preview: %s%s", preview_items, " ..." if len(items) > 10 else "")

        record: Dict[str, Any] = {
            "manifesto_id": manifesto_id,
            "meta": info,
            "text_chars": len(text),
            "raw_chars": len(raw),
            "start_char": int(args.start_char),
            "engram_config": asdict(cfg),
            "engram_items": items,
        }

        if args.dry_run:
            runs.append(record)
            continue

        rubric = str(args.rubric or "").strip()
        baseline_messages = default_summarize_prompt(text, rubric)
        engram_messages = engram_prompt(text, rubric)

        baseline_summary, baseline_usage = _summarize(
            client=client,
            messages=baseline_messages,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
        )
        engram_summary, engram_usage = _summarize(
            client=client,
            messages=engram_messages,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
        )

        baseline_stats = _preservation_stats(items, baseline_summary)
        engram_stats = _preservation_stats(items, engram_summary)

        logger.info(
            "Baseline recall: %.3f (%d/%d) | Engram recall: %.3f (%d/%d) | Δ=%.3f",
            baseline_stats["recall"],
            baseline_stats["preserved"],
            baseline_stats["items_total"],
            engram_stats["recall"],
            engram_stats["preserved"],
            engram_stats["items_total"],
            float(engram_stats["recall"]) - float(baseline_stats["recall"]),
        )

        record.update(
            {
                "rubric": rubric,
                "model": model_id,
                "baseline": {
                    "summary": baseline_summary,
                    "usage": baseline_usage,
                    "preservation": baseline_stats,
                },
                "engram": {
                    "summary": engram_summary,
                    "usage": engram_usage,
                    "preservation": engram_stats,
                },
            }
        )
        runs.append(record)

    payload = {
        "model": model_id,
        "port": int(args.port),
        "start_char": int(args.start_char),
        "max_chars": int(args.max_chars),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "rubric": str(args.rubric or "").strip(),
        "runs": runs,
    }

    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else (PROJECT_ROOT / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved JSON: %s", out_path)

    # Print a compact per-doc summary to stdout (useful even with --json-out).
    if not args.dry_run:
        print("")
        print("Engram multilingual smoke summary")
        print(f"  model={model_id} port={int(args.port)} max_chars={int(args.max_chars)} max_tokens={int(args.max_tokens)}")
        print("")
        for rec in runs:
            mid = rec["manifesto_id"]
            info = rec.get("meta", {}) or {}
            label = mid
            if info.get("countryname") and info.get("partyname"):
                label += f" | {info.get('countryname')} | {info.get('partyname')}"
            base = (rec.get("baseline", {}) or {}).get("preservation", {}) or {}
            eng = (rec.get("engram", {}) or {}).get("preservation", {}) or {}
            print(
                f"- {label}\n"
                f"  items={rec.get('engram_items') and len(rec['engram_items']) or 0} "
                f"baseline={base.get('preserved', 0)}/{base.get('items_total', 0)} ({base.get('recall', 0.0):.3f}) "
                f"engram={eng.get('preserved', 0)}/{eng.get('items_total', 0)} ({eng.get('recall', 0.0):.3f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
