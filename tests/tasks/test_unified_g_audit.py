from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_audit_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "audit_unified_g_usage.py"
    spec = importlib.util.spec_from_file_location("audit_unified_g_usage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_classifier_buckets_active_split_and_unified_markers():
    audit = _load_audit_module()
    text = """
from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer
g = UnifiedManifestoG()
combined = format_merge_input(left, right)
strategy = DSPyStrategy(leaf_module=g, merge_module=None, unified_mode=True)
"""
    findings = audit.classify_text("scripts/phase3_full_pipeline_optimize.py", text)
    active_split = [f for f in findings if f.bucket == "active_split_paths"]
    active_unified = [f for f in findings if f.bucket == "active_unified_paths"]
    assert any(f.symbol == "ManifestoMerger" for f in active_split)
    assert any(f.symbol == "ManifestoSummarizer" for f in active_split)
    assert any(f.symbol == "UnifiedManifestoG" for f in active_unified)
    assert any(f.symbol == "format_merge_input" for f in active_unified)
    assert any(f.symbol == "unified_mode" for f in active_unified)


def test_audit_classifier_keeps_split_class_definitions_legacy():
    audit = _load_audit_module()
    text = """
class ManifestoMerger(dspy.Module):
    pass

def create_merge_summarizer(self):
    return self.create_summarizer()
"""
    findings = audit.classify_text("src/tasks/manifesto/pipeline.py", text)
    legacy = [f for f in findings if f.bucket == "legacy_compatible_split_classes"]
    assert any(f.symbol == "ManifestoMerger" for f in legacy)
    assert any(f.symbol == "create_merge_summarizer" for f in legacy)
