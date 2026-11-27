"""Tests of the pre-commit hook that refuses large images."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "check_image_sizes.py"
)


def _hook():
    spec = importlib.util.spec_from_file_location("check_image_sizes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_images_up_to_the_limit_pass_and_larger_ones_fail(tmp_path: Path, capsys):
    """A file of exactly 500 kB passes; one byte more is reported and fails."""
    hook = _hook()
    small = tmp_path / "small.png"
    large = tmp_path / "large.svg"
    small.write_bytes(b"\0" * hook.LIMIT_BYTES)
    large.write_bytes(b"\0" * (hook.LIMIT_BYTES + 1))

    assert hook.main([str(small)]) == 0
    assert hook.main([str(small), str(large), str(tmp_path / "deleted.png")]) == 1
    out = capsys.readouterr().out
    assert "large.svg" in out
    assert "small.png" not in out
