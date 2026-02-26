"""
Engram-style "conditional memory" utilities for ThinkingTrees.

This module implements a cheap, deterministic "static memory" extractor for
local/stereotyped patterns (named entities, IDs, URLs, etc.) that are easy to
lose during compression. The extracted items can be injected into prompts so
the model can treat them like a lookup table rather than reconstructing them
through generation.

Implementation note:
The text normalization pipeline is adapted from DeepSeek's Engram demo
(`deepseek-ai/Engram`, `engram_demo_v1.py`, Apache-2.0). We keep a stdlib
fallback to avoid a hard dependency on `tokenizers`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Tuple

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)
_URL_RE = re.compile(r"\bhttps?://[^\s<>()\"']+\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]{8,}\b")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


class EngramTextNormalizer:
    """
    Normalize text to a canonical form suitable for deduplication.

    Tries to match Engram's demo normalizer:
      NFKC → NFD → strip accents → lowercase → whitespace collapse → strip
    """

    def __init__(self):
        self._use_tokenizers = False
        self._normalizer = None
        try:
            from tokenizers import Regex, normalizers  # type: ignore

            sentinel = "\uE000"
            self._normalizer = normalizers.Sequence(
                [
                    normalizers.NFKC(),
                    normalizers.NFD(),
                    normalizers.StripAccents(),
                    normalizers.Lowercase(),
                    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                    normalizers.Replace(Regex(r"^ $"), sentinel),
                    normalizers.Strip(),
                    normalizers.Replace(sentinel, " "),
                ]
            )
            self._use_tokenizers = True
        except Exception:
            self._use_tokenizers = False
            self._normalizer = None

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        raw = str(text)
        if self._use_tokenizers and self._normalizer is not None:
            try:
                return str(self._normalizer.normalize_str(raw))
            except Exception:
                # Fall through to stdlib path.
                pass

        out = unicodedata.normalize("NFKC", raw)
        out = _strip_accents(out)
        out = out.casefold()
        out = re.sub(r"[ \t\r\n]+", " ", out).strip()
        return out


@dataclass(frozen=True)
class EngramMemoryConfig:
    """Controls extraction and formatting of Engram-style static memory."""

    enabled: bool = False
    max_items: int = 32
    max_chars: int = 1200

    include_named_entities: bool = True
    include_single_proper_nouns: bool = True
    include_urls: bool = True
    include_emails: bool = True
    include_uuids: bool = True
    include_hex: bool = True
    include_numbers: bool = True
    include_identifiers: bool = True

    # Heuristics / thresholds.
    max_named_entity_words: int = 6
    min_single_proper_len: int = 10
    min_number_digits: int = 4
    min_identifier_len: int = 10


def _find_spans(pattern: re.Pattern[str], text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


def _dedup_by_normalized(
    spans: Iterable[Tuple[int, int, str]],
    *,
    normalizer: EngramTextNormalizer,
) -> List[Tuple[int, int, str]]:
    seen: set[str] = set()
    out: List[Tuple[int, int, str]] = []
    for start, end, value in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        key = normalizer.normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((start, end, value))
    return out


def _extract_named_entities(
    text: str,
    *,
    max_words: int,
    include_single: bool,
    min_single_len: int,
) -> List[Tuple[int, int, str]]:
    connectors = r"(?:of|the|and|de|van|von|da|dos|di|la|le|du|del|&)"
    multi = re.compile(
        rf"\b[A-Z][a-zA-Z]+(?:\s+(?:[A-Z][a-zA-Z]+|{connectors})){{1,{max(1, max_words - 1)}}}\b"
    )
    spans: List[Tuple[int, int, str]] = _find_spans(multi, text)

    if include_single and min_single_len > 0:
        single = re.compile(rf"\b[A-Z][a-zA-Z]{{{int(min_single_len) - 1},}}\b")
        spans.extend(_find_spans(single, text))

    return spans


def _extract_numbers(text: str, *, min_digits: int) -> List[Tuple[int, int, str]]:
    if min_digits <= 0:
        min_digits = 1
    number = re.compile(rf"\b\d[\d,]{{{int(min_digits) - 1},}}(?:\.\d+)?%?\b")
    return _find_spans(number, text)


def _extract_identifiers(text: str, *, min_len: int) -> List[Tuple[int, int, str]]:
    if min_len <= 0:
        min_len = 1
    snake = re.compile(rf"\b[a-zA-Z_][a-zA-Z0-9_]{{{int(min_len) - 1},}}\b")
    camel = re.compile(rf"\b[a-z]+[A-Z][A-Za-z0-9]{{{max(1, int(min_len) - 2)},}}\b")
    spans: List[Tuple[int, int, str]] = []
    for start, end, value in _find_spans(snake, text):
        if "_" in value or any(ch.isdigit() for ch in value):
            spans.append((start, end, value))
    spans.extend(_find_spans(camel, text))
    return spans


def extract_engram_memory_items(text: str, config: EngramMemoryConfig) -> List[str]:
    """
    Extract a compact list of verbatim strings worth preserving exactly.

    Returns:
        A list of strings in original surface form.
    """
    if not config.enabled:
        return []
    if not text or not str(text).strip():
        return []

    normalizer = EngramTextNormalizer()
    raw = str(text)

    spans: List[Tuple[int, int, str]] = []
    if config.include_urls:
        spans.extend(_find_spans(_URL_RE, raw))
    if config.include_emails:
        spans.extend(_find_spans(_EMAIL_RE, raw))
    if config.include_uuids:
        spans.extend(_find_spans(_UUID_RE, raw))
    if config.include_hex:
        spans.extend(_find_spans(_HEX_RE, raw))
        spans.extend(_find_spans(re.compile(r"\b[0-9a-fA-F]{16,}\b"), raw))

    if config.include_numbers:
        spans.extend(_extract_numbers(raw, min_digits=config.min_number_digits))

    if config.include_identifiers:
        spans.extend(_extract_identifiers(raw, min_len=config.min_identifier_len))

    if config.include_named_entities:
        spans.extend(
            _extract_named_entities(
                raw,
                max_words=config.max_named_entity_words,
                include_single=config.include_single_proper_nouns,
                min_single_len=config.min_single_proper_len,
            )
        )

    deduped = _dedup_by_normalized(spans, normalizer=normalizer)

    items: List[str] = []
    total_chars = 0
    max_items = max(0, int(config.max_items))
    max_chars = max(0, int(config.max_chars))

    for _, __, value in deduped:
        if max_items and len(items) >= max_items:
            break
        if max_chars and (total_chars + len(value)) > max_chars:
            continue
        items.append(value)
        total_chars += len(value)

    return items


def format_engram_memory_block(items: List[str]) -> str:
    """Render memory items as a prompt-ready block."""
    if not items:
        return ""
    lines = [
        "STATIC MEMORY (verbatim strings from the input; preserve exactly if relevant):",
        *[f"- {item}" for item in items],
    ]
    return "\n".join(lines).strip()

