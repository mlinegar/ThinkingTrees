from __future__ import annotations

import sys

from src.experiments import scheduler as _scheduler

sys.modules[__name__] = _scheduler
