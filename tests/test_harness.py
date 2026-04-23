"""Tests for the lightweight audit harness."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.core.llm_client import MockLLMClient
from src.harness import TreeAudit, AuditBudget, AuditCertificate, HarnessResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _word_count_oracle(text: str) -> float:
    """Trivial oracle: normalized word count (clamped to [0, 1])."""
    return min(1.0, len(text.split()) / 200.0)


SAMPLE_DOC = (
    "The central limit theorem states that the sum of many independent random "
    "variables tends toward a normal distribution. This fundamental result in "
    "probability theory has wide applications in statistics, machine learning, "
    "and the natural sciences. The theorem was first proved by Abraham de Moivre "
    "in a special case and later generalized by Pierre-Simon Laplace. Modern "
    "formulations allow for dependent variables under mixing conditions. "
    "Applications include confidence interval construction, hypothesis testing, "
    "and the justification of many Bayesian and frequentist procedures. The "
    "Berry-Esseen theorem provides quantitative bounds on the rate of convergence "
    "to the normal distribution. Extensions to multivariate and functional settings "
    "are active areas of research."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuditCertificate:
    """Test the certificate data class."""

    def test_round_trip_json(self):
        cert = AuditCertificate(
            guarantee_level="EXACT",
            violation_bound=0.0,
            confidence=0.95,
            sufficiency_rate=0.0,
            merge_rate=0.0,
            idempotence_rate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            n_documents=3,
            n_nodes_audited=15,
            n_leaves_total=12,
            n_merges_total=11,
            effective_sample_size=15.0,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            model_id="test-model",
            timestamp="2026-02-17T00:00:00Z",
            seed=42,
            run_id="test123",
        )
        j = cert.to_json()
        restored = AuditCertificate.from_json(j)
        assert restored.guarantee_level == "EXACT"
        assert restored.violation_bound == 0.0
        assert restored.n_documents == 3
        assert restored.seed == 42

    def test_to_dict(self):
        cert = AuditCertificate(
            guarantee_level="EMPIRICAL",
            violation_bound=0.15,
            confidence=0.95,
            sufficiency_rate=0.05,
            merge_rate=0.03,
            idempotence_rate=0.0,
            ci_low=0.01,
            ci_high=0.12,
            n_documents=1,
            n_nodes_audited=10,
            n_leaves_total=8,
            n_merges_total=7,
            effective_sample_size=9.5,
            token_usage={},
            model_id="mock",
            timestamp="now",
        )
        d = cert.to_dict()
        assert isinstance(d, dict)
        assert d["guarantee_level"] == "EMPIRICAL"


class TestTreeAuditEndToEnd:
    """End-to-end test with mock LLM."""

    def test_basic_run(self):
        """Build tree, audit, get certificate -- no real LLM server needed."""
        mock_client = MockLLMClient()

        audit = TreeAudit(
            oracle=_word_count_oracle,
            budget=AuditBudget(
                delta=0.05,
                epsilon=0.20,
                sample_budget=5,
                audit_idempotence=True,
                audit_substitution=True,
            ),
            chunk_chars=200,
            seed=42,
            _client_override=mock_client,
        )

        result = audit.run_sync([SAMPLE_DOC])

        # Certificate exists and has sensible fields
        cert = result.certificate
        assert cert.guarantee_level in ("EXACT", "UNION_BOUND", "EMPIRICAL", "NONE")
        assert cert.confidence == pytest.approx(0.95)
        assert cert.n_documents == 1
        assert cert.n_nodes_audited >= 0
        assert cert.n_leaves_total > 0
        assert math.isfinite(cert.violation_bound)

        # Trees were built
        assert len(result.trees) == 1
        assert result.trees[0].leaf_count > 0

        # Trace was recorded
        assert len(result.trace) == 1
        assert result.trace[0]["doc_id"] == "doc_0000"

    def test_multiple_documents(self):
        """Process multiple documents, get aggregated certificate."""
        mock_client = MockLLMClient()

        docs = [SAMPLE_DOC, SAMPLE_DOC + " Extra paragraph for variety."]

        audit = TreeAudit(
            oracle=_word_count_oracle,
            budget=AuditBudget(sample_budget=3),
            chunk_chars=200,
            seed=123,
            _client_override=mock_client,
        )

        result = audit.run_sync(docs)

        assert result.certificate.n_documents == 2
        assert len(result.trees) == 2
        assert len(result.trace) == 2

    def test_no_oracle(self):
        """Without an oracle, auditing is skipped, certificate is NONE."""
        mock_client = MockLLMClient()

        audit = TreeAudit(
            oracle=None,
            chunk_chars=200,
            _client_override=mock_client,
        )

        result = audit.run_sync([SAMPLE_DOC])

        assert result.certificate.guarantee_level == "NONE"
        assert len(result.audit_reports) == 0
        assert len(result.trees) == 1

    def test_save_artifacts(self):
        """Artifacts save to disk correctly."""
        mock_client = MockLLMClient()

        audit = TreeAudit(
            oracle=_word_count_oracle,
            budget=AuditBudget(sample_budget=3),
            chunk_chars=200,
            seed=7,
            _client_override=mock_client,
        )

        result = audit.run_sync([SAMPLE_DOC])

        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "certificate.json").exists()
            assert (Path(tmpdir) / "supervision.json").exists()
            assert (Path(tmpdir) / "trace.jsonl").exists()
            assert (Path(tmpdir) / "audit_reports.json").exists()
            assert (Path(tmpdir) / "trees").is_dir()

            # Certificate is valid JSON
            cert_data = json.loads((Path(tmpdir) / "certificate.json").read_text())
            assert "guarantee_level" in cert_data
            assert "violation_bound" in cert_data

            audit_reports = json.loads((Path(tmpdir) / "audit_reports.json").read_text())
            assert isinstance(audit_reports, list)
            if audit_reports:
                assert "compositional_learning_problem" in audit_reports[0]
