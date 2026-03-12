from __future__ import annotations

import sys
from pathlib import Path

from src.ctreepo.sim.cli.exec_cmds import main as exec_main


def test_exec_cmds_parallel_smoke(tmp_path: Path) -> None:
    cmds_path = tmp_path / "cmds.txt"
    log_dir = tmp_path / "logs"
    py = sys.executable

    cmds = [
        f'{py} -c "print(\'cmd0\')"',
        f'{py} -c "print(\'cmd1\')"',
        f'{py} -c "import time; time.sleep(0.1); print(\'cmd2\')"',
    ]
    cmds_path.write_text("\n".join(cmds) + "\n", encoding="utf-8")

    rc = int(exec_main(["--cmds", str(cmds_path), "--jobs", "2", "--log-dir", str(log_dir)]))
    assert rc == 0

    logs = sorted(log_dir.glob("cmd_*.log"))
    assert len(logs) == len(cmds)
    assert all(p.stat().st_size >= 0 for p in logs)
