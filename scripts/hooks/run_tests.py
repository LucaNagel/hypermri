"""Run the commit/CI unit gate against this checkout using the active Python."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    """Propagate pytest's exit code, including failed collection or missing pytest."""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    # Import this checkout even if another hypermri installation is active.
    env["PYTHONPATH"] = str(root / "src")
    env["MPLBACKEND"] = "Agg"
    # A test subprocess may not inherit a pre-commit-managed Python path.
    env.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "-o",
        "addopts=",
        "-q",
        "--tb=short",
        *sys.argv[1:],
    ]
    # Stashing/restoring an equal-size edit can leave bytecode with the same
    # timestamp. Use a fresh cache so the gate always imports current source.
    with TemporaryDirectory(prefix="hypermri-test-bytecode-") as bytecode_cache:
        env["PYTHONPYCACHEPREFIX"] = bytecode_cache
        return subprocess.run(command, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
