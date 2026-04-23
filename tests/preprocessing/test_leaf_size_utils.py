from __future__ import annotations

from src.preprocessing.leaf_size_utils import (
    char_windows_from_token_budget,
    count_tokens,
)


def test_char_windows_from_token_budget_covers_text_without_overbudget_chunks() -> None:
    text = (
        " ".join(f"policy_term_{idx}" for idx in range(80))
        + "\n\nA final sentence keeps trailing whitespace covered.  "
    )
    windows = char_windows_from_token_budget(text, 32)

    assert windows[0][0] == 0
    assert windows[-1][1] == len(text)
    assert all(windows[idx][1] == windows[idx + 1][0] for idx in range(len(windows) - 1))
    assert "".join(text[start:end] for start, end in windows) == text
    assert all(count_tokens(text[start:end]) <= 32 for start, end in windows)
