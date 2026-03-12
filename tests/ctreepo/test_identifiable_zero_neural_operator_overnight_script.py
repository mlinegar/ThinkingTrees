from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_identifiable_zero_neural_operator_overnight_explicit_local_law_grid() -> None:
    script = REPO_ROOT / "scripts" / "run_identifiable_zero_neural_operator_overnight.sh"
    content = script.read_text(encoding="utf-8")

    assert 'MARKOV_LOCAL_LAW_WEIGHTS="${MARKOV_LOCAL_LAW_WEIGHTS:-0 0.25 0.5 1.0}"' in content
    assert '--local-law-weights "${MARKOV_LOCAL_LAW_WEIGHTS}"' in content
    assert '--c1-relative-weights "${MARKOV_C1_RELATIVE_WEIGHTS}"' in content
    assert '--c3-relative-weights "${MARKOV_C3_RELATIVE_WEIGHTS}"' in content
