"""Tests for package stream handling and opt-in diagnostic file logging."""

from __future__ import annotations

import io
import logging
import subprocess
import sys

import numpy as np
import pytest

from hypermri.utils.system.logging import (
    debug_log,
    init_default_logger,
    set_hypermri_log_level,
)


@pytest.fixture
def isolated_logging(monkeypatch):
    """Give each test a fresh logger tree without changing pytest's handlers."""
    root = logging.RootLogger(logging.WARNING)
    monkeypatch.setattr(logging.Logger, "manager", logging.Manager(root))
    return root


@pytest.mark.parametrize("configure_first", [False, True])
def test_package_owns_one_stream_handler(isolated_logging, capsys, configure_first):
    """Child initialization and level changes commute and never duplicate output."""
    if configure_first:
        set_hypermri_log_level("error")
    child = init_default_logger("hypermri.study")
    for _ in range(3):
        set_hypermri_log_level("error")
        assert init_default_logger("hypermri.study") is child
    sibling = init_default_logger("hypermri.loading.scans")
    parent = logging.getLogger("hypermri")

    assert len(parent.handlers) == 1
    assert child.handlers == sibling.handlers == []
    child.error("synthetic loading failure")
    sibling.error("synthetic scanner failure")
    output = capsys.readouterr().err
    assert output.count("synthetic loading failure") == 1
    assert output.count("synthetic scanner failure") == 1


def test_level_changes_apply_to_existing_and_future_children(isolated_logging, capsys):
    """Later imports inherit the chosen level, including after repeated changes."""
    existing = init_default_logger("hypermri.study")
    set_hypermri_log_level("error")
    existing.debug("hidden diagnostic")
    set_hypermri_log_level("debug")
    future = init_default_logger("hypermri.loading.scans")
    existing.debug("existing diagnostic")
    future.debug("future diagnostic")
    set_hypermri_log_level(logging.ERROR)
    future.warning("hidden warning")
    output = capsys.readouterr().err
    assert "hidden" not in output
    assert output.count("existing diagnostic") == 1
    assert output.count("future diagnostic") == 1


@pytest.mark.parametrize("owner", ["root", "hypermri", "hypermri.study"])
def test_user_handler_and_configuration_are_preserved(isolated_logging, owner):
    """Defaults neither replace user handlers nor reset explicit child levels."""
    user_logger = isolated_logging if owner == "root" else logging.getLogger(owner)
    user_logger.setLevel(logging.WARNING)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter("user: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.WARNING)
    user_logger.addHandler(handler)
    child = logging.getLogger("hypermri.study")
    child.setLevel(logging.WARNING)
    child.propagate = owner != "hypermri.study"

    init_default_logger(child.name)
    set_hypermri_log_level("debug")

    assert child.level == logging.WARNING
    assert child.propagate == (owner != "hypermri.study")
    assert user_logger.handlers == [handler]
    assert handler.formatter is formatter
    assert handler.level == logging.WARNING
    if owner == "root":
        assert user_logger.level == logging.WARNING
        assert logging.getLogger("hypermri").handlers == []
    child.debug("hidden user diagnostic")
    child.error("user diagnostic")
    assert stream.getvalue() == "user: user diagnostic\n"


def test_similar_namespace_is_not_reconfigured(isolated_logging):
    logger = logging.getLogger("hypermri_external")
    logger.setLevel(logging.WARNING)
    set_hypermri_log_level("debug")
    assert logger.level == logging.WARNING
    assert logger.handlers == []


@pytest.mark.parametrize("owner", ["root", "hypermri", "hypermri.study"])
def test_later_application_handler_takes_precedence(isolated_logging, capsys, owner):
    child = init_default_logger("hypermri.study")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    user_logger = isolated_logging if owner == "root" else logging.getLogger(owner)
    user_logger.addHandler(handler)
    set_hypermri_log_level("debug")
    child.error("later user configuration")
    assert stream.getvalue() == "later user configuration\n"
    assert capsys.readouterr().err == ""


def test_non_package_logger_defaults_remain_supported(isolated_logging, capsys):
    logger = init_default_logger("external", default_mode="info", fstring="%(message)s")
    assert init_default_logger("external") is logger
    logger.info("external diagnostic")
    assert len(logger.handlers) == 1
    assert capsys.readouterr().err == "external diagnostic\n"


def test_notset_remains_inherited_after_later_imports(isolated_logging):
    set_hypermri_log_level("notset")
    child = init_default_logger("hypermri.late")
    assert logging.getLogger("hypermri").level == logging.NOTSET
    assert child.getEffectiveLevel() == isolated_logging.level


def test_initial_package_default_and_module_formats(isolated_logging, capsys):
    """The first default supplies the level; each module retains its format."""
    first = init_default_logger(
        "hypermri.first", default_mode="info", fstring="first: %(message)s"
    )
    second = init_default_logger("hypermri.second", fstring="second: %(message)s")
    first.info("one")
    second.info("two")
    assert capsys.readouterr().err == "first: one\nsecond: two\n"


@pytest.mark.parametrize("configure_first", [False, True])
def test_study_import_order_in_a_fresh_process(configure_first, package_subprocess_env):
    """Exercise the real import boundary without inherited pytest logging state."""
    source = """
import logging
import importlib
from hypermri.utils.system.logging import init_default_logger, set_hypermri_log_level
if CONFIGURE_FIRST:
    set_hypermri_log_level('debug')
from hypermri import Study
set_hypermri_log_level('debug')
set_hypermri_log_level('debug')
from hypermri.sequences.FLASH import FLASH
modules = ['hypermri.sequences.' + name for name in
           ('CSI', 'STEAM', 'semiLASER', 'RFPROFILE', 'NSPECT', 'PRESS',
            'base_spectroscopy')]
modules.append('hypermri.analysis.spectroscopy.single_voxel')
for name in modules:
    importlib.import_module(name)
    assert logging.getLogger(name).getEffectiveLevel() == logging.DEBUG, name
logging.getLogger('hypermri.study').error('synthetic import failure')
logging.getLogger('hypermri.sequences.FLASH').debug('synthetic late diagnostic')
print(sum(len(logger.handlers) for name, logger in
          logging.root.manager.loggerDict.items()
          if (name == 'hypermri' or name.startswith('hypermri.'))
          and isinstance(logger, logging.Logger)))
root = logging.getLogger()
root.setLevel(logging.WARNING)
assert init_default_logger(None, default_mode='info') is root
assert root.level == logging.WARNING
root.warning('synthetic root diagnostic')
""".replace("CONFIGURE_FIRST", repr(configure_first))
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=package_subprocess_env,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "1"
    assert result.stderr.count("synthetic import failure") == 1
    assert result.stderr.count("synthetic late diagnostic") == 1
    assert result.stderr.count("synthetic root diagnostic") == 1


def test_debug_log_full_array_formatting_does_not_change_numpy_global_options(tmp_path):
    """Explicit full-array logging must not affect later application formatting."""
    original_options = np.get_printoptions().copy()
    log_path = tmp_path / "diagnostics" / "debug.log"

    debug_log(
        np.arange(100),
        file_name=log_path,
        write_to_file=True,
        numpy_full_print=True,
    )

    assert np.get_printoptions() == original_options
    assert "99" in log_path.read_text(encoding="utf-8")


def test_debug_log_does_not_evaluate_lazy_messages_when_disabled(tmp_path):
    """Disabled file logging avoids constructing expensive diagnostic messages."""
    evaluated = False

    def build_message():
        nonlocal evaluated
        evaluated = True
        return np.arange(1_000_000)

    debug_log(
        build_message,
        file_name=tmp_path / "debug.log",
        write_to_file=False,
    )

    assert not evaluated
    assert not (tmp_path / "debug.log").exists()
