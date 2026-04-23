from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from src.ctreepo.sim.manifest import RunSpec
from src.ctreepo.sim.suite.common import SuiteGroupRuns


REQUIRED_SUITE_META_KEYS = {
    "schema_version",
    "suite_name",
    "suite_role",
    "profile",
    "policy",
    "manifest_file",
    "group_manifest_files",
    "selected_groups",
}


def _fake_runs(*, family: str, group_key: str, n: int = 1) -> list[RunSpec]:
    return [
        RunSpec.create(
            family=family,
            config={"suite_group": group_key, "index": idx},
            outputs={"json_summary": f"/tmp/{group_key}_{idx}.json"},
            command=f"{sys.executable} -c \"print('{group_key}:{idx}')\"",
        )
        for idx in range(n)
    ]


def _assert_suite_meta(path: Path, *, suite_name: str, suite_role: str) -> dict[str, object]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert REQUIRED_SUITE_META_KEYS <= set(meta)
    assert meta["schema_version"] == "v2"
    assert meta["suite_name"] == suite_name
    assert meta["suite_role"] == suite_role
    assert Path(str(meta["manifest_file"])).exists()
    for item in (meta.get("group_manifest_files", {}) or {}).values():
        assert Path(str(item)).exists()
    return meta


def _fake_queue(**kwargs: object) -> dict[str, object]:
    manifest_paths = [str(path) for path in kwargs.get("manifest_paths", [])]
    return {
        "manifest_paths": manifest_paths,
        "log_dir": str(kwargs.get("log_dir")),
        "gpu_tokens": [],
        "summary": {"n_fail": 0, "n_ok": len(manifest_paths)},
    }


def _capture_report(monkeypatch, module) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}

    def _fake_main(argv: Sequence[str] | None = None) -> int:
        payload["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(module, "main", _fake_main)
    return payload


def _arg_value(argv: Sequence[str], flag: str) -> str:
    idx = list(argv).index(flag)
    return str(list(argv)[idx + 1])


def test_cpu_megasweep_v2_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.cpu_megasweep as cpu_report
    import src.ctreepo.sim.cli.report.cpu_megasweep_readable as readable_report
    import src.ctreepo.sim.suite.cpu_megasweep_v2 as suite

    def _fake_legacy_build(argv: Sequence[str]) -> int:
        out_cmds = Path(_arg_value(argv, "--out-cmds"))
        out_plot_cmds = Path(_arg_value(argv, "--out-plot-cmds"))
        out_meta = Path(_arg_value(argv, "--out-meta"))
        out_cmds.parent.mkdir(parents=True, exist_ok=True)
        out_plot_cmds.parent.mkdir(parents=True, exist_ok=True)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_cmds.with_name(f"{out_cmds.stem}_markov.txt").write_text("python -c \"print('markov')\"\n", encoding="utf-8")
        out_cmds.with_name(f"{out_cmds.stem}_segment_lda_ops.txt").write_text(
            "python -c \"print('segment')\"\n", encoding="utf-8"
        )
        out_cmds.with_name(f"{out_cmds.stem}_segmented_lda_ctreepo.txt").write_text(
            "python -c \"print('ctree')\"\n", encoding="utf-8"
        )
        out_cmds.write_text("", encoding="utf-8")
        out_plot_cmds.write_text("", encoding="utf-8")
        out_meta.write_text(json.dumps({"n_plot_commands_total": 0}), encoding="utf-8")
        return 0

    monkeypatch.setattr(suite, "legacy_build_main", _fake_legacy_build)
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    cpu_seen = _capture_report(monkeypatch, cpu_report)
    readable_seen = _capture_report(monkeypatch, readable_report)

    out_root = tmp_path / "cpu_megasweep"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "markov"]) == 0
    meta = _assert_suite_meta(out_root / "suite_meta.json", suite_name="cpu-megasweep", suite_role="paper")
    assert meta["selected_groups"] == ["markov"]
    assert suite.main(["run", "--output-root", str(out_root), "--groups", "markov", "--jobs", "1", "--gpu-tokens", "none"]) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--skip-plots", "--no-emit-pdf"]) == 0
    assert cpu_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]
    assert readable_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]


def test_simulation_buildout_v2_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.simulation_buildout as buildout_report
    import src.ctreepo.sim.suite.simulation_buildout_v2 as suite

    def _fake_legacy_build(argv: Sequence[str]) -> int:
        out_cmds = Path(_arg_value(argv, "--out-cmds"))
        out_plot_cmds = Path(_arg_value(argv, "--out-plot-cmds"))
        out_meta = Path(_arg_value(argv, "--out-meta"))
        out_cmds.parent.mkdir(parents=True, exist_ok=True)
        out_plot_cmds.parent.mkdir(parents=True, exist_ok=True)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_cmds.write_text(
            "\n".join(
                [
                    "python -c \"print('hard_markov')\"",
                    "python -c \"print('hard_segment')\"",
                    "python -c \"print('hard_ctree')\"",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_plot_cmds.write_text("", encoding="utf-8")
        out_meta.write_text(
            json.dumps(
                {
                    "counts_by_suite": {
                        "item2_hard_markov": 1,
                        "item2_hard_segment_lda_ops": 1,
                        "item2_hard_ctreepo": 1,
                        "item3_estimator_stress_segment_lda_ops": 0,
                        "item4_guidance_frontier_ctreepo": 0,
                        "item5_ipw_expanded": 0,
                    },
                    "n_plot_commands_total": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(suite, "legacy_build_main", _fake_legacy_build)
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, buildout_report)

    out_root = tmp_path / "simulation_buildout"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "item2_hard_markov"]) == 0
    meta = _assert_suite_meta(out_root / "suite_meta.json", suite_name="simulation-buildout", suite_role="paper")
    assert meta["selected_groups"] == ["item2_hard_markov"]
    assert suite.main(
        ["run", "--output-root", str(out_root), "--groups", "item2_hard_markov", "--jobs", "1", "--gpu-tokens", "none"]
    ) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--skip-plots", "--no-emit-pdf"]) == 0
    assert report_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]


def test_identifiable_zero_neural_operator_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.identifiable_zero_neural_operator as report_mod
    import src.ctreepo.sim.suite.identifiable_zero_neural_operator as suite

    monkeypatch.setattr(
        suite,
        "_build_groups",
        lambda **_: [
            SuiteGroupRuns(
                key="markov_schedule_consistency",
                family="markov_changepoint_ops_count",
                runs=_fake_runs(family="markov_changepoint_ops_count", group_key="markov_schedule_consistency"),
            ),
            SuiteGroupRuns(
                key="ctree_operator_family",
                family="segmented_lda_ctreepo",
                runs=_fake_runs(family="segmented_lda_ctreepo", group_key="ctree_operator_family"),
            ),
        ],
    )
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, report_mod)

    out_root = tmp_path / "neural_operator"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "markov_schedule_consistency"]) == 0
    meta = _assert_suite_meta(
        out_root / "suite_meta.json",
        suite_name="identifiable-zero-neural-operator",
        suite_role="appendix",
    )
    assert meta["selected_groups"] == ["markov_schedule_consistency"]
    assert suite.main(
        ["run", "--output-root", str(out_root), "--groups", "markov_schedule_consistency", "--jobs", "1", "--gpu-tokens", "none"]
    ) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--no-emit-pdf"]) == 0
    assert report_seen["argv"][:2] == ["--overnight-output-root", str(out_root.resolve())]


def test_identifiable_zero_lda_leafnoise_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.identifiable_zero_lda_leafnoise as report_mod
    import src.ctreepo.sim.suite.identifiable_zero_lda_leafnoise as suite

    monkeypatch.setattr(
        suite,
        "_build_groups",
        lambda **_: [
            SuiteGroupRuns(
                key="leaf_8",
                family="segmented_lda_ctreepo",
                runs=_fake_runs(family="segmented_lda_ctreepo", group_key="leaf_8"),
            )
        ],
    )
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, report_mod)

    out_root = tmp_path / "leafnoise"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "leaf_8"]) == 0
    meta = _assert_suite_meta(
        out_root / "suite_meta.json",
        suite_name="identifiable-zero-lda-leafnoise",
        suite_role="appendix",
    )
    assert meta["selected_groups"] == ["leaf_8"]
    assert suite.main(["run", "--output-root", str(out_root), "--groups", "leaf_8", "--jobs", "1", "--gpu-tokens", "none"]) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--no-emit-pdf"]) == 0
    assert report_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]


def test_identifiable_zero_dtm_lda_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.identifiable_zero_dtm_lda as report_mod
    import src.ctreepo.sim.suite.identifiable_zero_dtm_lda as suite

    monkeypatch.setattr(
        suite,
        "_build_groups",
        lambda **_: [
            SuiteGroupRuns(
                key="ctree_lda",
                family="segmented_lda_ctreepo",
                runs=_fake_runs(family="segmented_lda_ctreepo", group_key="ctree_lda"),
            )
        ],
    )
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, report_mod)

    out_root = tmp_path / "dtm_lda"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "ctree_lda"]) == 0
    meta = _assert_suite_meta(out_root / "suite_meta.json", suite_name="identifiable-zero-dtm-lda", suite_role="appendix")
    assert meta["selected_groups"] == ["ctree_lda"]
    assert suite.main(["run", "--output-root", str(out_root), "--groups", "ctree_lda", "--jobs", "1", "--gpu-tokens", "none"]) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--no-emit-pdf"]) == 0
    assert report_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]


def test_lda_tree_recovery_progress_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.lda_tree_recovery_progress as report_mod
    import src.ctreepo.sim.suite.lda_tree_recovery_progress as suite

    def _fake_builder(argv: Sequence[str]) -> int:
        out_cmds = Path(_arg_value(argv, "--out-cmds"))
        out_cmds.parent.mkdir(parents=True, exist_ok=True)
        out_cmds.write_text("python -c \"print('ok')\"\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(suite, "exact_builder_main", _fake_builder)
    monkeypatch.setattr(suite, "world_batch_builder_main", _fake_builder)
    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, report_mod)

    out_root = tmp_path / "lda_tree_recovery"
    assert suite.main(["build", "--output-root", str(out_root), "--groups", "exact_cpu"]) == 0
    meta = _assert_suite_meta(
        out_root / "suite_meta.json",
        suite_name="lda-tree-recovery-progress",
        suite_role="diagnostic",
    )
    assert meta["selected_groups"] == ["exact_cpu"]
    assert suite.main(["run", "--output-root", str(out_root), "--groups", "exact_cpu", "--jobs", "1", "--gpu-tokens", "none"]) == 0
    assert suite.main(["report", "--output-root", str(out_root)]) == 0
    assert report_seen["argv"][:2] == ["--input-root", str(out_root.resolve())]


def test_learned_sketch_smoke_suite_build_run_report_smoke(tmp_path: Path, monkeypatch) -> None:
    import src.ctreepo.sim.cli.report.learned_sketch_smoke as report_mod
    import src.ctreepo.sim.suite.learned_sketch_smoke as suite

    monkeypatch.setattr(suite, "run_manifest_queue_suite", _fake_queue)
    report_seen = _capture_report(monkeypatch, report_mod)

    out_root = tmp_path / "learned_sketch_smoke"
    assert suite.main(["build", "--output-root", str(out_root)]) == 0
    meta = _assert_suite_meta(out_root / "suite_meta.json", suite_name="learned-sketch-smoke", suite_role="diagnostic")
    assert meta["selected_groups"] == ["proxy_baseline"]
    assert meta["profile"] == "smoke"
    assert suite.main(["run", "--output-root", str(out_root), "--jobs", "1", "--gpu-tokens", "none"]) == 0
    assert suite.main(["report", "--output-root", str(out_root), "--no-emit-pdf"]) == 0
    assert report_seen["argv"][:2] == ["--output-root", str(out_root.resolve())]


def test_learned_sketch_smoke_cli_dispatch_build_smoke(tmp_path: Path) -> None:
    from src.ctreepo.cli import main as cli_main

    out_root = tmp_path / "learned_sketch_cli"
    assert (
        cli_main(
            [
                "sim",
                "suite",
                "learned-sketch-smoke",
                "build",
                "--output-root",
                str(out_root),
            ]
        )
        == 0
    )
    meta = _assert_suite_meta(out_root / "suite_meta.json", suite_name="learned-sketch-smoke", suite_role="diagnostic")
    assert meta["selected_groups"] == ["proxy_baseline"]
