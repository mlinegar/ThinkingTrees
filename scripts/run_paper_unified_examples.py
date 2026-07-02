#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.paper_unified_examples import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
