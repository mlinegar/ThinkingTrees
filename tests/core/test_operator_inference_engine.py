from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.core.engines import EngineSurface, EngineType
from src.core.inference_engine import NativeOperatorRegistry, build_inference_engine
from src.runtime.contracts import InferenceRequest, OperatorInput, OperatorOutput


def test_native_operator_surface_dispatches_registered_handler() -> None:
    NativeOperatorRegistry.clear()

    def _handler(payload: OperatorInput) -> OperatorOutput:
        return OperatorOutput(
            data={
                "operation": payload.operation,
                "selected_indices": [1],
                "selected_chunks": ["beta evidence"],
            },
            artifacts={"backend": "native"},
        )

    NativeOperatorRegistry.register("select_evidence", _handler)
    try:
        engine = build_inference_engine(
            "native_operator",
            surface=EngineSurface.OPERATOR,
            model="fake-selector",
        )

        response = engine.execute(
            InferenceRequest(
                surface=EngineSurface.OPERATOR,
                input=OperatorInput(
                    operation="select_evidence",
                    inputs={"question": "Which chunk?", "chunks": ["alpha", "beta evidence"]},
                ),
            )
        )

        assert response.surface is EngineSurface.OPERATOR
        assert response.engine is EngineType.NATIVE_OPERATOR
        assert isinstance(response.output, OperatorOutput)
        assert response.output.data["selected_indices"] == [1]
        assert response.output.artifacts["backend"] == "native"
        assert response.artifacts["operation"] == "select_evidence"
        assert response.telemetry["operation"] == "select_evidence"
    finally:
        NativeOperatorRegistry.clear()


def test_custom_http_operator_surface_posts_typed_payload() -> None:
    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("Authorization")
            captured["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps(
                {
                    "model": "served-selector",
                    "data": {"ok": True, "selected_indices": [0]},
                    "artifacts": {"backend": "served"},
                    "telemetry": {"remote": True},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        engine = build_inference_engine(
            "custom_http",
            surface=EngineSurface.OPERATOR,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="client-selector",
            api_key="test-key",
        )
        response = engine.execute(
            InferenceRequest(
                surface=EngineSurface.OPERATOR,
                input=OperatorInput(
                    operation="select_evidence",
                    inputs={"chunks": ["alpha"]},
                    options={"top_k": 1},
                ),
                metadata={"request_type": "operator_test"},
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert captured["path"] == "/v1/operators/execute"
    assert captured["authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "client-selector"
    assert payload["operation"] == "select_evidence"
    assert payload["inputs"] == {"chunks": ["alpha"]}
    assert response.engine is EngineType.CUSTOM_HTTP
    assert isinstance(response.output, OperatorOutput)
    assert response.model_id == "served-selector"
    assert response.output.data == {"ok": True, "selected_indices": [0]}
    assert response.output.artifacts["backend"] == "served"
    assert response.telemetry["remote"] is True
