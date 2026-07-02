#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_GUIDE = REPO_ROOT / "docs" / "markov_report_archive.md"


def archived_report_message(
    *,
    legacy_script: str,
    replacements: Sequence[str],
    note: str = "",
) -> str:
    lines = [
        f"{legacy_script} is archived and no longer part of the supported Markov v3 reporting surface.",
        "Use one of the v3-compatible entrypoints instead:",
    ]
    for replacement in replacements:
        lines.append(f"- {replacement}")
    if note:
        lines.append(note)
    lines.append(f"Archive guide: {ARCHIVE_GUIDE}")
    return "\n".join(lines)


def archived_report_exit(
    *,
    legacy_script: str,
    replacements: Sequence[str],
    note: str = "",
) -> int:
    print(
        archived_report_message(
            legacy_script=legacy_script,
            replacements=replacements,
            note=note,
        ),
        file=sys.stderr,
    )
    return 2

