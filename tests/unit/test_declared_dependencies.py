import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "hooks" / "validate_dependencies.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "hypermri_validate_dependencies",
        VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_imports_are_declared_in_pyproject():
    validator = _load_validator_module()

    missing = validator.missing_dependency_report(REPO_ROOT)

    assert not missing, validator.format_missing_report(missing, REPO_ROOT)
