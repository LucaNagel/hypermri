import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "hypermri"

REMOVED_MODULES = (
    "hypermri.analysis.motion_correction.dipy",
    "hypermri.sequences.umpireFLASH",
    "hypermri.sequences.umpireMGE",
    "hypermri.utils.image.segmentation",
)

CONFIG_FILES = (
    SRC_ROOT / "sequences" / "config" / "sequence_mapping.yaml",
    SRC_ROOT / "reconstruction" / "selection.py",
)


def test_core_imports_without_removed_optional_features():
    for module_name in (
        "hypermri",
        "hypermri.sequences",
        "hypermri.visualization",
        "hypermri.analysis.fitting.phase_ideal",
        "hypermri.visualization.spectrum_simulator",
    ):
        importlib.import_module(module_name)


def test_sequence_public_api_excludes_removed_features():
    sequences = importlib.import_module("hypermri.sequences")

    for name in ("umpireFLASH", "umpireMGE"):
        assert name not in sequences.__all__
        assert not hasattr(sequences, name)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_are_not_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_removed_features_absent_from_release_configs():
    for config_file in CONFIG_FILES:
        text = config_file.read_text(encoding="utf-8").lower()
        assert "umpire" not in text, config_file


def test_removed_features_have_no_active_runtime_imports():
    for source_file in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots = {
                    alias.name.split(".", 1)[0].lower() for alias in node.names
                }
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots = {node.module.split(".", 1)[0].lower()}
            else:
                continue

            assert "umpire" not in imported_roots, source_file
            assert "dipy" not in imported_roots, source_file


def test_dipy_is_not_declared_as_package_dependency():
    pyproject = REPO_ROOT / "pyproject.toml"

    assert "dipy" not in pyproject.read_text(encoding="utf-8").lower()
