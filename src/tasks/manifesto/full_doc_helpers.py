from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import PolicyDimension


DIMENSION_BY_NAME = {dim.value: dim for dim in PolicyDimension}
_INT_RE = re.compile(r"([1-7](?:\.\d+)?)")


def parse_response(text: str) -> Optional[float]:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    upper = stripped.upper()
    if upper.startswith("NA") or upper == "N/A":
        return None
    match = _INT_RE.search(stripped)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return max(1.0, min(7.0, value))


def usage_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return dict(usage.model_dump())
        except Exception:
            return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    out: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def load_tokenizer(model: str):
    if not model:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def limit_text(
    text: str,
    *,
    max_input_chars: int,
    max_input_tokens: int,
    tokenizer: Any,
) -> Tuple[str, Dict[str, Any]]:
    raw = str(text or "")
    if int(max_input_tokens) > 0:
        if tokenizer is not None:
            token_ids = tokenizer.encode(raw, add_special_tokens=False)
            if len(token_ids) > int(max_input_tokens):
                kept_ids = token_ids[: int(max_input_tokens)]
                limited = tokenizer.decode(kept_ids, skip_special_tokens=True)
                return limited, {
                    "truncated": True,
                    "limit_kind": "tokenizer",
                    "input_tokens_estimated": len(kept_ids),
                    "full_tokens_estimated": len(token_ids),
                    "coverage_ratio": len(limited) / max(len(raw), 1),
                }
            return raw, {
                "truncated": False,
                "limit_kind": "tokenizer",
                "input_tokens_estimated": len(token_ids),
                "full_tokens_estimated": len(token_ids),
                "coverage_ratio": 1.0,
            }
        char_limit = max(1, int(max_input_tokens))
        if len(raw) > char_limit:
            limited = raw[:char_limit]
            return limited, {
                "truncated": True,
                "limit_kind": "token_heuristic_1char",
                "input_tokens_estimated": int(max_input_tokens),
                "full_tokens_estimated": int(math.ceil(len(raw) / 4.0)),
                "coverage_ratio": len(limited) / max(len(raw), 1),
            }
    if int(max_input_chars) > 0 and len(raw) > int(max_input_chars):
        limited = raw[: int(max_input_chars)]
        return limited, {
            "truncated": True,
            "limit_kind": "chars",
            "input_tokens_estimated": int(math.ceil(len(limited) / 4.0)),
            "full_tokens_estimated": int(math.ceil(len(raw) / 4.0)),
            "coverage_ratio": len(limited) / max(len(raw), 1),
        }
    return raw, {
        "truncated": False,
        "limit_kind": "none",
        "input_tokens_estimated": int(math.ceil(len(raw) / 4.0)),
        "full_tokens_estimated": int(math.ceil(len(raw) / 4.0)),
        "coverage_ratio": 1.0,
    }


def dimension_metrics(rows: Sequence[Mapping[str, Any]], *, dimension: str) -> Dict[str, Any]:
    dim_rows = [row for row in rows if row.get("dimension") == dimension]
    preds = [row.get("prediction") for row in dim_rows]
    truths = [row.get("expert_score_1_7") for row in dim_rows]
    pairs = [
        (float(pred), float(truth))
        for pred, truth in zip(preds, truths)
        if pred is not None and truth is not None
    ]
    try:
        pearson_dict = compute_corpus_pearson_r(preds, truths).as_dict()
    except (ValueError, ZeroDivisionError) as exc:
        pearson_dict = {
            "n": len(pairs),
            "n_na": len(dim_rows) - len(pairs),
            "pearson_r": None,
            "pearson_ci_low": None,
            "pearson_ci_high": None,
            "spearman_r": None,
            "mae_rescaled": None,
            "rmse_rescaled": None,
            "pearson_defined": False,
            "spearman_defined": False,
            "undefined_reason": str(exc),
        }
    mae = sum(abs(pred - truth) for pred, truth in pairs) / len(pairs) if pairs else None
    return {
        "n_rows": len(dim_rows),
        "n_scored": len(pairs),
        "n_na": sum(1 for row in dim_rows if row.get("prediction") is None),
        "pearson": pearson_dict,
        "mae": mae,
    }


__all__ = [
    "DIMENSION_BY_NAME",
    "dimension_metrics",
    "limit_text",
    "load_tokenizer",
    "parse_response",
    "usage_dict",
]
