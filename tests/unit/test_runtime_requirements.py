"""Guard the supported Python and notebook-kernel installation contract."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]
_KERNEL_REQUIREMENTS = [
    requirement
    for declaration in _PROJECT["dependencies"]
    if (requirement := Requirement(declaration)).name.lower() == "ipykernel"
]


@pytest.mark.parametrize(
    ("python_version", "supported"),
    [
        ("3.10.16", False),
        ("3.11.10", False),
        ("3.12.0", True),
        ("3.12.14", True),
        ("3.13.0", False),
        ("3.14.0", False),
    ],
)
def test_only_python_312_patch_versions_are_supported(python_version, supported):
    """Allow Python 3.12 patch updates and reject other minor versions."""
    assert (python_version in SpecifierSet(_PROJECT["requires-python"])) is supported


def test_core_installation_requires_the_supported_kernel():
    """Require the fixed kernel in every installation."""
    assert len(_KERNEL_REQUIREMENTS) == 1
    assert _KERNEL_REQUIREMENTS[0].marker is None


@pytest.mark.parametrize(
    ("kernel_version", "supported"),
    [
        ("6.31.0", False),
        ("7.3.0", False),
        ("7.4.0", True),
        ("7.4.1", True),
        ("7.9.0", True),
        ("8.0.0", False),
    ],
)
def test_kernel_requirement_excludes_affected_and_new_major_versions(
    kernel_version, supported
):
    """Exclude the affected release while allowing fixes within kernel 7."""
    assert (kernel_version in _KERNEL_REQUIREMENTS[0].specifier) is supported


def test_installed_kernel_satisfies_the_runtime_requirement():
    """Check that the test environment uses an allowed notebook kernel."""
    assert version("ipykernel") in _KERNEL_REQUIREMENTS[0].specifier


def test_core_installation_includes_the_viewer_notebook_and_twix_stack():
    """``pip install hypermri`` provides the viewer, Jupyter and TWIX reading."""
    names = {Requirement(value).name.lower() for value in _PROJECT["dependencies"]}
    assert {
        "ipython",
        "ipywidgets",
        "ipykernel",
        "notebook",
        "plotly",
        "anywidget",
        "traitlets",
        "ipympl",
        "cmap",
        "twixtools",
    } <= names
    assert names.isdisjoint({"nbclassic", "simpleitk", "gitpython", "sympy"})


def test_extras_are_limited_to_motion_tests_development_and_docs():
    """Only the large motion backend and development tooling stay optional."""
    extras = _PROJECT["optional-dependencies"]
    assert set(extras) == {"motion", "test", "dev", "docs"}
    assert "hypermri[test]" in extras["dev"]
    kernels = [
        value
        for group in [_PROJECT["dependencies"], *extras.values()]
        for value in group
        if Requirement(value).name.lower() == "ipykernel"
    ]
    assert len(kernels) == 1
