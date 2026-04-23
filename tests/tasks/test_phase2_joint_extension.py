from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.expert_benchmarks import load_joint_train_pairs


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combined_pipeline_summarize_uses_chunk_text(monkeypatch) -> None:
    mod = _load_script_module("phase2_combined_pipeline.py")
    chunks = [SimpleNamespace(text="alpha"), SimpleNamespace(text="beta")]
    seen_texts: list[str] = []

    class FakeSummarizer:
        def __call__(self, *, text: str, rubric: str) -> str:
            seen_texts.append(text)
            return f"S[{text}]"

    class FakeMerger:
        def __call__(self, *, summary1: str, summary2: str, rubric: str) -> str:
            return f"{summary1}|{summary2}"

    monkeypatch.setattr(mod, "chunk_for_ops", lambda text, max_chars, strategy: chunks)

    summary = mod._summarize(
        "ignored",
        FakeSummarizer(),
        FakeMerger(),
        "joint-rubric",
        chunk_chars=24000,
        max_workers=2,
    )

    assert seen_texts == ["alpha", "beta"]
    assert summary == "S[alpha]|S[beta]"


def test_combined_pipeline_main_does_not_force_max_tokens(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script_module("phase2_combined_pipeline.py")
    recorded_kwargs = {}

    class FakeDataset:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_all_ids(self) -> list[str]:
            return []

    def _fake_create_vllm_lm(**kwargs):
        recorded_kwargs.update(kwargs)
        return SimpleNamespace(model="fake-model")

    empty_experts = pd.DataFrame(columns=["manifesto", "expert_mean"])
    empty_crosswalk = pd.DataFrame(columns=["manifesto", "party", "year"])

    monkeypatch.setattr(mod, "create_vllm_lm", _fake_create_vllm_lm)
    monkeypatch.setattr(mod, "configure_dspy", lambda lm: None)
    monkeypatch.setattr(mod, "ManifestoDataset", FakeDataset)
    monkeypatch.setattr(mod, "load_benoit_expert_means", lambda dim: empty_experts.copy())
    monkeypatch.setattr(mod, "load_benoit_mp_crosswalk", lambda: empty_crosswalk.copy())
    monkeypatch.setattr(mod.sys, "argv", ["phase2_combined_pipeline.py", "--output-dir", str(tmp_path)])

    rc = int(mod.main())

    assert rc == 0
    assert "max_tokens" not in recorded_kwargs


def test_load_joint_train_pairs_excludes_global_holdout(monkeypatch) -> None:
    def _fake_summaries(dimension: PolicyDimension, dataverse_dir=None) -> pd.DataFrame:
        by_dim = {
            PolicyDimension.ECONOMIC: pd.DataFrame(
                {
                    "manifesto_stem": ["economic_keep", "shared_holdout"],
                    "summary": ["econ keep", "econ holdout"],
                }
            ),
            PolicyDimension.SOCIAL: pd.DataFrame(
                {
                    "manifesto_stem": ["social_keep", "shared_holdout"],
                    "summary": ["social keep", "social holdout"],
                }
            ),
        }
        return by_dim.get(
            dimension,
            pd.DataFrame({"manifesto_stem": [f"{dimension.value}_keep"], "summary": [dimension.value]}),
        ).copy()

    def _fake_lookup(dim: PolicyDimension, pool: str) -> dict[str, float]:
        return {
            "economic_keep": 1.0,
            "social_keep": 2.0,
            "shared_holdout": 3.0,
            f"{dim.value}_keep": 4.0,
        }

    monkeypatch.setattr(
        "src.tasks.manifesto.expert_benchmarks.load_benoit_masked_summaries",
        _fake_summaries,
    )
    monkeypatch.setattr(
        "src.tasks.manifesto.expert_benchmarks._train_lookup_for_pool",
        _fake_lookup,
    )

    out = load_joint_train_pairs(
        "openweight",
        test_keys_per_dim={PolicyDimension.ECONOMIC: {"shared_holdout"}},
        global_holdout_keys={"shared_holdout"},
    )

    assert "shared_holdout" not in set(out["manifesto_stem"])
    assert {"economic_keep", "social_keep"}.issubset(set(out["manifesto_stem"]))


def test_joint_optimize_main_omits_max_tokens_by_default(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script_module("phase2_joint_optimize.py")
    recorded_kwargs = {}

    def _fake_create_vllm_lm(**kwargs):
        recorded_kwargs.update(kwargs)
        return SimpleNamespace(model="fake-model")

    train_rows = pd.DataFrame(
        [
            {
                "manifesto_stem": f"train_{i}",
                "dimension": PolicyDimension.ECONOMIC.value,
                "summary": f"summary {i}",
                "label": float((i % 6) + 1),
            }
            for i in range(12)
        ]
    )

    def _fake_test_examples(dim: PolicyDimension) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "manifesto_stem": [f"holdout_{dim.value}"],
                "summary": [f"held out {dim.value}"],
                "label": [3.0],
                "dimension": [dim.value],
            }
        )

    class FakeJointScorer:
        def __init__(self, use_cot: bool = False) -> None:
            self.use_cot = use_cot

    monkeypatch.setattr(mod, "create_vllm_lm", _fake_create_vllm_lm)
    monkeypatch.setattr(mod, "configure_dspy", lambda lm: None)
    monkeypatch.setattr(mod, "_load_test_examples", _fake_test_examples)
    monkeypatch.setattr(mod, "load_joint_train_pairs", lambda *args, **kwargs: train_rows.copy())
    monkeypatch.setattr(
        mod,
        "_per_dim_report",
        lambda program, label, output_dir: {
            "per_dim": {
                dim.value: {
                    "pearson_r": 0.1,
                    "n": 3,
                    "pearson_ci_low": 0.0,
                    "pearson_ci_high": 0.2,
                }
                for dim in mod._ORDER
            },
            "macro_pearson_r": 0.1,
        },
    )
    monkeypatch.setattr(mod, "JointDimensionScorer", FakeJointScorer)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "phase2_joint_optimize.py",
            "--optimizer",
            "none",
            "--output-dir",
            str(tmp_path),
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    assert "max_tokens" not in recorded_kwargs
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["run"]["max_tokens"] is None


def test_roundup_finds_plain_and_timestamped_phase2_dirs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script_module("roundup_overnight.py")
    phase2_root = tmp_path / "phase2"
    overnight_root = tmp_path / "overnight"
    out_dir = tmp_path / "out"
    phase2_root.mkdir()
    overnight_root.mkdir()

    joint_dir = phase2_root / "joint_optimize"
    joint_dir.mkdir()
    (joint_dir / "report.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "per_dim": {
                        "economic": {
                            "pearson_r": 0.11,
                            "n": 3,
                            "pearson_ci_low": 0.0,
                            "pearson_ci_high": 0.2,
                        }
                    },
                    "macro_pearson_r": 0.11,
                }
            }
        ),
        encoding="utf-8",
    )

    combined_dir = phase2_root / "combined_pipeline_20260419_0100"
    combined_dir.mkdir()
    (combined_dir / "report.json").write_text(
        json.dumps(
            {
                "per_dim": {
                    "economic": {
                        "pearson_r": 0.22,
                        "n": 4,
                        "pearson_ci_low": 0.1,
                        "pearson_ci_high": 0.3,
                    }
                },
                "macro_pearson_r": 0.22,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "roundup_overnight.py",
            "--root",
            str(overnight_root),
            "--phase2-root",
            str(phase2_root),
            "--out-dir",
            str(out_dir),
            "--economic-old",
            str(tmp_path / "missing_economic_old.json"),
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    payload = json.loads((out_dir / "roundup.json").read_text(encoding="utf-8"))
    assert payload["dimensions"]["economic"]["joint_baseline"]["pearson_r"] == 0.11
    assert payload["dimensions"]["economic"]["combined_pipeline"]["pearson_r"] == 0.22
    assert payload["macro"]["joint_baseline"] == 0.11
    assert payload["macro"]["combined_pipeline"] == 0.22
