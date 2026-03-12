from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
DENSE_LLW = "0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0"


def test_markov_local_law_learnability_longrun_uses_dense_interior_grid() -> None:
    script = (REPO_ROOT / "scripts" / "run_markov_local_law_learnability_longrun.sh").read_text(
        encoding="utf-8"
    )
    assert f'MARKOV_LOCAL_LAW_WEIGHTS="${{MARKOV_LOCAL_LAW_WEIGHTS:-{DENSE_LLW}}}"' in script


def test_markov_local_law_grid_overnight_uses_dense_grid_and_full_cpu_default() -> None:
    script = (REPO_ROOT / "scripts" / "run_markov_local_law_grid_overnight.sh").read_text(
        encoding="utf-8"
    )
    assert f'MARKOV_LOCAL_LAW_WEIGHTS="${{MARKOV_LOCAL_LAW_WEIGHTS:-{DENSE_LLW}}}"' in script
    assert 'TOTAL_CPUS="$(detect_cpu_count)"' in script
    assert 'JOBS="${JOBS:-${TOTAL_CPUS}}"' in script


def test_markov_ops_builders_default_to_dense_local_law_grid() -> None:
    builder = (REPO_ROOT / "scripts" / "build_markov_changepoint_ops_count_cmds.py").read_text(
        encoding="utf-8"
    )
    sweep = (
        REPO_ROOT / "src" / "ctreepo" / "sim" / "cli" / "sweep_markov_changepoint_ops_count.py"
    ).read_text(encoding="utf-8")
    assert f'default="{DENSE_LLW}"' in builder
    assert f'default="{DENSE_LLW}"' in sweep


def test_markov_generic_sweep_defaults_use_equal_relative_law_weights() -> None:
    builder = (REPO_ROOT / "scripts" / "build_markov_changepoint_ops_count_cmds.py").read_text(
        encoding="utf-8"
    )
    sweep = (
        REPO_ROOT / "src" / "ctreepo" / "sim" / "cli" / "sweep_markov_changepoint_ops_count.py"
    ).read_text(encoding="utf-8")
    for text in (builder, sweep):
        assert re.search(r'--c1-relative-weights".*?default="1\.0"', text, re.S)
        assert re.search(r'--c2-relative-weights".*?default="1\.0"', text, re.S)
        assert re.search(r'--c3-relative-weights".*?default="1\.0"', text, re.S)


def test_lda_generic_launchers_default_to_equal_thirds() -> None:
    runner = (REPO_ROOT / "scripts" / "run_leaf_local_mixture_utility_simulation.py").read_text(
        encoding="utf-8"
    )
    generic_builder = (
        REPO_ROOT / "scripts" / "build_leaf_local_mixture_utility_cmds.py"
    ).read_text(encoding="utf-8")
    companion_builder = (
        REPO_ROOT / "scripts" / "build_tree_relevant_lda_local_law_cmds.py"
    ).read_text(encoding="utf-8")
    for text in (runner, generic_builder, companion_builder):
        assert re.search(r'--law-c1-weight".*?default=1\.0 / 3\.0', text, re.S)
        assert re.search(r'--law-c2-proxy-weight".*?default=1\.0 / 3\.0', text, re.S)
        assert re.search(r'--law-c3-weight".*?default=1\.0 / 3\.0', text, re.S)


def test_lda_generic_launchers_default_to_all_laws_package() -> None:
    runner = (REPO_ROOT / "scripts" / "run_leaf_local_mixture_utility_simulation.py").read_text(
        encoding="utf-8"
    )
    generic_builder = (
        REPO_ROOT / "scripts" / "build_leaf_local_mixture_utility_cmds.py"
    ).read_text(encoding="utf-8")
    companion_builder = (
        REPO_ROOT / "scripts" / "build_tree_relevant_lda_local_law_cmds.py"
    ).read_text(encoding="utf-8")
    for text in (runner, generic_builder, companion_builder):
        assert re.search(r'--law-package".*?default="all_laws"', text, re.S)
