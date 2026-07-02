from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.batch_processor import BatchRequest, BatchResponse
from src.experiments.call_tracing import batch_request_call_row
from src.experiments.contracts import ResultRow, benchmark_ref_from_parts, method_ref_from_parts
from src.experiments.roles import (
    ROLE_SCORER,
    chat_role_ref,
    method_ref_with_roles,
    oracle_ref,
)
from src.experiments.sidecars import sidecar_root_for_output_file, write_canonical_sidecars


def test_role_helpers_attach_metadata_without_schema_change() -> None:
    method_ref = method_ref_from_parts(family="demo", variant="v1", adapter="test")
    updated = method_ref_with_roles(
        method_ref,
        roles={
            ROLE_SCORER: chat_role_ref(
                role=ROLE_SCORER,
                model="demo-model",
                base_url="http://localhost:8000/v1",
            )
        },
        oracle=oracle_ref(kind="benchmark_labels", source="fixture"),
    )

    assert updated.method_id == method_ref.method_id
    assert updated.metadata["roles"]["scorer"]["model"] == "demo-model"
    assert updated.metadata["oracle"]["kind"] == "benchmark_labels"


def test_sidecar_writer_handles_file_output_root(tmp_path: Path) -> None:
    output = tmp_path / "example.json"
    output.write_text("{}", encoding="utf-8")
    sidecar_root = sidecar_root_for_output_file(output)
    benchmark_ref = benchmark_ref_from_parts(family="fixture", name="fixture")
    method_ref = method_ref_from_parts(family="method", adapter="test")

    spec = write_canonical_sidecars(
        sidecar_root,
        title="fixture",
        adapter_id="test",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("dry_run",),
        artifacts={"output_json": str(output)},
        result_rows=(
            ResultRow(
                experiment_id="",
                phase="dry_run",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name="ok",
                metric_value=True,
                artifact_refs=("output_json",),
            ),
        ),
        state="dry_run",
    )

    assert (sidecar_root / "experiment_manifest.json").exists()
    rows = (sidecar_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["experiment_id"] == spec.experiment_id


def test_batch_call_trace_row_is_compact_and_role_aware() -> None:
    request = BatchRequest(
        request_id="r1",
        messages=[{"role": "user", "content": "secret prompt text"}],
        document_id="doc-1",
        request_type="score",
        call_metadata={"method_id": "m", "runner_id": "r"},
    )
    response = BatchResponse(
        request_id="r1",
        content="secret answer",
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        latency_ms=12.5,
    )

    row = batch_request_call_row(request, response, model="model-a")

    rendered = json.dumps(row)
    assert row["role"] == "scorer"
    assert row["surface"] == "chat_openai"
    assert row["document_id"] == "doc-1"
    assert "run_id" not in row
    assert "secret prompt text" not in rendered
    assert "secret answer" not in rendered


def test_runtime_umbrella_inventory_json_has_no_missing_supported_entries() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/audit_runtime_umbrella_coverage.py",
            "--json",
            "--fail-on-unclassified",
        ],
        cwd=repo_root,
        text=True,
    )
    report = json.loads(out)
    assert report["unclassified"] == []
    assert report["missing_supported"] == []
    assert report["supported_policy_violations"] == []
    assert any(item["path"] == "scripts/run_runtime_eval.py" for item in report["supported"])
    assert any(
        item["path"] == "scripts/audit_runtime_umbrella_coverage.py"
        and item["status"] == "canonical_tool"
        for item in report["supported"]
    )
