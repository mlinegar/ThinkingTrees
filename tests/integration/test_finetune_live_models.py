"""Optional live smoke tests for ``treepo.finetune`` rows.

These tests verify that exported fine-tune rows can be consumed by the local
OpenAI-compatible embedding endpoint and small Gemma-4 chat endpoints without
using sentence-transformers, TRL, PEFT, or Accelerate.

Enable with:

    TREEPO_RUN_FINETUNE_LIVE=1 \
      uv run pytest -q tests/integration/test_finetune_live_models.py

Useful local servers:

    ./scripts/start_embedding_server.sh
    ./scripts/start_vllm.sh gemma-4-e2b-it --port 8010 --cuda-devices 0
    ./scripts/start_vllm.sh gemma-4-e4b-it --port 8011 --cuda-devices 1

Environment overrides:

    TREEPO_FINETUNE_EMBED_URL=http://localhost:8003/v1
    TREEPO_FINETUNE_EMBED_MODEL=Qwen/Qwen3-Embedding-8B
    TREEPO_FINETUNE_GEMMA_URLS=http://localhost:8010/v1,http://localhost:8011/v1
    TREEPO_FINETUNE_GEMMA_MODEL=google/gemma-4-E2B-it
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Mapping

import pytest

from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState
from treepo.finetune import build_finetune_views


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_DEFAULT_GEMMA_URLS = (
    "http://localhost:8000/v1",
    "http://localhost:8010/v1",
    "http://localhost:8011/v1",
    "http://localhost:8012/v1",
    "http://localhost:8013/v1",
)
_SMALL_GEMMA_MARKERS = ("gemma-4-e2b", "gemma-4-e4b")


def test_finetune_live_fixture_builds_expected_rows() -> None:
    views = _views()

    assert len(views["embedding_pairs"]) >= 2
    assert len(views["embedding_triplets"]) == 1
    assert len(views["sft"]) == len(views["embedding_pairs"])
    assert views["embedding_pairs"][0]["metadata"]["tree_id"] == "live_doc"
    assert views["dpo"][0]["metadata"]["unit_type"] == "qsentence"


def test_finetune_embedding_pairs_embed_on_local_endpoint() -> None:
    _require_live_enabled()
    base_url = _embedding_url()
    model_ids = _models_or_skip(base_url, label="embedding")
    model = os.getenv("TREEPO_FINETUNE_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL") or model_ids[0]

    rows = _views()["embedding_pairs"][:2]
    texts = [row["anchor"] for row in rows] + [row["positive"] for row in rows]
    payload = {"model": model, "input": texts}
    response = _post_json(f"{base_url}/embeddings", payload, timeout=60)

    data = list(response.get("data") or [])
    assert len(data) == len(texts)
    vectors = [_embedding_vector(item) for item in sorted(data, key=lambda item: int(item.get("index", 0)))]
    dims = {len(vector) for vector in vectors}
    assert len(dims) == 1
    assert next(iter(dims)) > 8
    for vector in vectors:
        assert any(abs(value) > 1e-12 for value in vector)
        assert all(math.isfinite(value) for value in vector)


def test_finetune_sft_and_dpo_rows_generate_on_small_gemma4_endpoint() -> None:
    _require_live_enabled()
    base_url, model = _find_gemma_endpoint_or_skip()
    views = _views()

    prompt = (
        "You are validating exported fine-tuning rows. "
        "Return exactly READY if both rows have prompt text and tree metadata.\n\n"
        f"SFT row:\n{json.dumps(_compact_row(views['sft'][0]), sort_keys=True)}\n\n"
        f"DPO row:\n{json.dumps(_compact_row(views['dpo'][0]), sort_keys=True)}"
    )
    response = _post_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return exactly READY for valid rows."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        },
        timeout=120,
    )
    content = _chat_content(response)

    assert content.strip()
    assert "ready" in content.lower()


def _views() -> dict[str, list[dict[str, Any]]]:
    return build_finetune_views(_preference_dataset())


def _preference_dataset() -> PreferenceDataset:
    state = TaskState(
        kind="manifesto_policy",
        counts={"market": 1.0, "welfare": 2.0},
        measures={"rile": -0.35},
        text="welfare investment with market safeguards",
    )
    return PreferenceDataset(
        [
            PreferenceRecord(
                record_id="root_sft",
                unit_id="live_doc:root",
                unit_type="root",
                target="f",
                context="Estimate the document-level RILE score.",
                tree_id="live_doc",
                doc_id="live_doc",
                node_id="root",
                level=1,
                position=0,
                left_child_id="q0",
                right_child_id="q1",
                candidates=(Candidate(id="gold", value="RILE score: -0.35", score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="qsentence_sft",
                unit_id="live_doc:q0",
                unit_type="qsentence",
                target="g",
                context="Encode this qsentence as a policy state: Invest in health and schools.",
                tree_id="live_doc",
                doc_id="live_doc",
                node_id="q0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value=state, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="qsentence_pair",
                unit_id="live_doc:q0:pair",
                unit_type="qsentence",
                target="g",
                context="Choose the better policy-state encoding for the qsentence.",
                tree_id="live_doc",
                doc_id="live_doc",
                node_id="q0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(
                    Candidate(id="specific", value=state, score=0.9, preferred=True),
                    Candidate(id="generic", value="general campaign language", score=0.2),
                ),
            ),
        ]
    )


def _require_live_enabled() -> None:
    if str(os.getenv("TREEPO_RUN_FINETUNE_LIVE", "")).strip().lower() not in _TRUE_VALUES:
        pytest.skip("Set TREEPO_RUN_FINETUNE_LIVE=1 to run live model smoke tests.")


def _embedding_url() -> str:
    return _normalize_base_url(
        os.getenv("TREEPO_FINETUNE_EMBED_URL")
        or os.getenv("EMBEDDING_URL")
        or "http://localhost:8003/v1"
    )


def _gemma_urls() -> list[str]:
    raw = os.getenv("TREEPO_FINETUNE_GEMMA_URLS") or os.getenv("TREEPO_FINETUNE_GEMMA_URL")
    if raw:
        return [_normalize_base_url(item) for item in raw.split(",") if item.strip()]
    task_model_url = os.getenv("TASK_MODEL_URL")
    urls = ([task_model_url] if task_model_url else []) + list(_DEFAULT_GEMMA_URLS)
    return list(dict.fromkeys(_normalize_base_url(url) for url in urls if url))


def _find_gemma_endpoint_or_skip() -> tuple[str, str]:
    requested_model = os.getenv("TREEPO_FINETUNE_GEMMA_MODEL")
    available: list[tuple[str, list[str]]] = []
    for base_url in _gemma_urls():
        try:
            model_ids = _models(base_url)
        except RuntimeError:
            continue
        available.append((base_url, model_ids))
        if requested_model:
            if requested_model in model_ids:
                return base_url, requested_model
            continue
        lowered = [(model_id, model_id.lower()) for model_id in model_ids]
        for model_id, lower in lowered:
            if any(marker in lower for marker in _SMALL_GEMMA_MARKERS):
                return base_url, model_id
        for model_id, lower in lowered:
            if "gemma-4" in lower:
                return base_url, model_id
    pytest.skip(
        "No Gemma-4 endpoint found. Checked: "
        + ", ".join(f"{url} -> {ids}" for url, ids in available)
    )


def _models_or_skip(base_url: str, *, label: str) -> list[str]:
    try:
        return _models(base_url)
    except RuntimeError as exc:
        pytest.skip(f"{label} endpoint not available at {base_url}: {exc}")


def _models(base_url: str) -> list[str]:
    payload = _get_json(f"{base_url}/models", timeout=5)
    rows = list(payload.get("data") or [])
    model_ids = [str(row.get("id") or "") for row in rows if row.get("id")]
    if not model_ids:
        raise RuntimeError(f"empty /models response from {base_url}")
    return model_ids


def _normalize_base_url(url: str) -> str:
    return str(url).rstrip("/")


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(str(exc)) from exc


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(dict(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(str(exc)) from exc


def _embedding_vector(item: Mapping[str, Any]) -> list[float]:
    vector = item.get("embedding")
    assert isinstance(vector, list)
    return [float(value) for value in vector]


def _chat_content(response: Mapping[str, Any]) -> str:
    choices = list(response.get("choices") or [])
    assert choices, response
    first = dict(choices[0])
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    return {
        key: row[key]
        for key in ("prompt", "completion", "chosen", "rejected", "sample_weight")
        if key in row
    } | {
        "metadata": {
            key: metadata.get(key)
            for key in ("tree_id", "doc_id", "node_id", "unit_id", "unit_type")
        }
    }
