"""Exercise the real local test hook, including pre-commit's staged-file view."""

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_POSIX_HOOK_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="The Bash hook fixture models POSIX executable lookup and PATH isolation.",
)


def _hook_environment(tmp_path):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GIT_"):
            env.pop(key)
    env["PRE_COMMIT_HOME"] = str(tmp_path / "pre-commit-cache")
    env["PYTEST_ADDOPTS"] = "--version"
    # Model a GUI client with Git/bash available but no development Python.
    env["PATH"] = str(tmp_path / "gui-bin")
    Path(env["PATH"]).mkdir()
    for name in ("git", "bash"):
        Path(env["PATH"], name).symlink_to(shutil.which(name))
    return env


@_POSIX_HOOK_ONLY
def test_commit_gate_rejects_staged_failure_and_restores_unstaged_fix(tmp_path):
    """A passing unstaged fix cannot hide a failing staged source revision."""
    pytest.importorskip("pre_commit")
    root = tmp_path / "checkout with spaces"
    (root / "scripts/hooks").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests/unit").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/hooks/run_tests.py",
        root / "scripts/hooks/run_tests.py",
    )
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/hooks/run_tests.sh",
        root / "scripts/hooks/run_tests.sh",
    )
    source = root / "src/value.py"
    source.write_text("VALUE = 0\n")
    (root / "tests/unit/test_value.py").write_text(
        "from value import VALUE\ndef test_value():\n    assert VALUE == 1\n"
    )
    (root / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n"
        "      - id: unit-tests\n        name: unit-tests\n"
        "        entry: bash scripts/hooks/run_tests.sh\n"
        "        language: system\n        pass_filenames: false\n"
        "        always_run: true\n"
    )
    env = _hook_environment(tmp_path)

    def run(*command):
        return subprocess.run(
            command, cwd=root, env=env, text=True, capture_output=True
        )

    assert run("git", "init", "-q").returncode == 0
    assert (
        run(
            "git", "config", "--local", "hypermri.testPython", sys.executable
        ).returncode
        == 0
    )
    assert run("git", "add", ".").returncode == 0
    source.write_text("VALUE = 1\n")
    failed = run(sys.executable, "-m", "pre_commit", "run", "unit-tests")
    assert failed.returncode != 0
    assert "1 failed" in failed.stdout + failed.stderr
    assert source.read_text() == "VALUE = 1\n"
    assert run("git", "show", ":src/value.py").stdout == "VALUE = 0\n"

    assert run("git", "add", "src/value.py").returncode == 0
    passed = run(sys.executable, "-m", "pre_commit", "run", "unit-tests")
    assert passed.returncode == 0, passed.stdout + passed.stderr

    for configured in ("missing", "invalid"):
        if configured == "missing":
            assert (
                run("git", "config", "--unset", "hypermri.testPython").returncode == 0
            )
        else:
            assert (
                run(
                    "git", "config", "hypermri.testPython", "/missing/python"
                ).returncode
                == 0
            )
        unavailable = run(sys.executable, "-m", "pre_commit", "run", "unit-tests")
        assert unavailable.returncode != 0
        assert "Unit-test Python unavailable" in unavailable.stdout + unavailable.stderr


@_POSIX_HOOK_ONLY
def test_commit_gate_uses_active_python_when_unconfigured(tmp_path):
    """Terminal use keeps the active environment when no local path is set."""
    env = _hook_environment(tmp_path)
    Path(env["PATH"], "python").symlink_to(sys.executable)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "scripts/hooks").mkdir(parents=True)
    (tmp_path / "tests/unit").mkdir(parents=True)
    (tmp_path / "tests/unit/test_value.py").write_text(
        "def test_value():\n    assert 1 == 1\n"
    )
    for name in ("run_tests.sh", "run_tests.py"):
        shutil.copyfile(
            Path(__file__).resolve().parents[2] / "scripts/hooks" / name,
            tmp_path / "scripts/hooks" / name,
        )
    launcher = tmp_path / "scripts/hooks/run_tests.sh"
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_python_runner_uses_checkout_and_propagates_test_results(tmp_path):
    """The portable runner selects its checkout and rejects stale bytecode."""
    root = tmp_path / "checkout with spaces"
    (root / "scripts/hooks").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests/unit").mkdir(parents=True)
    runner = root / "scripts/hooks/run_tests.py"
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/hooks/run_tests.py", runner
    )
    source = root / "src/value.py"
    source.write_text("VALUE = 0\n", encoding="utf-8")
    (root / "tests/unit/test_value.py").write_text(
        "from value import VALUE\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    unrelated_source = tmp_path / "other source"
    unrelated_source.mkdir()
    (unrelated_source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(unrelated_source)
    env["PYTEST_ADDOPTS"] = "--version"

    def run():
        return subprocess.run(
            [sys.executable, str(runner)],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    failed = run()
    assert failed.returncode == pytest.ExitCode.TESTS_FAILED
    assert "1 failed" in failed.stdout + failed.stderr

    py_compile.compile(str(source), doraise=True)
    original_stat = source.stat()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    passed = run()
    assert passed.returncode == pytest.ExitCode.OK, passed.stdout + passed.stderr
    assert "1 passed" in passed.stdout + passed.stderr
