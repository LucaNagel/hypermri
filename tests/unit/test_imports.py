"""
Smoke tests for module imports.
This ensures that recent refactoring hasn't introduced circular dependencies
or broken critical import paths.
"""

import importlib

import pytest

MODULES_TO_TEST = [
    "hypermri",
    "hypermri.vendors.bruker",
    "hypermri.vendors.agilent",
    "hypermri.vendors.magritek",
    "hypermri.vendors.siemens",
    "hypermri.vendors.external",
    "hypermri.utils.physics",
    "hypermri.analysis.anatomical.masks",
    "hypermri.vendors.registry",
]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_imports(module_name):
    """Verify that each module can be imported without error."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Failed to import {module_name}: {e}")


def test_bruker_handler_load():
    """Specifically test loading the Bruker handler via the registry."""
    from hypermri.vendors.registry import get_vendor_registry

    registry = get_vendor_registry()
    entry = registry.get("bruker")
    assert entry is not None
    assert entry["handler_class"].__name__ == "BrukerHandler"
    assert entry["handler_class"].__module__ == "hypermri.vendors.bruker"


def test_legacy_vendor_import_paths_reexport_new_handlers():
    """Old root-level vendor modules should remain compatibility shims."""
    from hypermri.vendors.agilent import AgilentHandler
    from hypermri.vendors.bruker import BrukerHandler
    from hypermri.vendors.external import ExternalDataHandler
    from hypermri.vendors.magritek import MagritekHandler
    from hypermri.vendors.siemens import SiemensHandler


ROOT_PUBLIC_NAMES = [
    "__version__",
    "MRIExperiment",
    "load_scan",
    "create_experiment",
    "load_external_experiment",
    "Study",
]


def test_root_exports_every_public_name_as_the_real_object():
    """No root name is replaced by ``None`` or ``object`` after a failed import."""
    import hypermri

    assert hypermri.__all__ == ROOT_PUBLIC_NAMES
    for name in ROOT_PUBLIC_NAMES:
        assert getattr(hypermri, name) not in (None, object), name


@pytest.mark.parametrize(
    "name",
    [
        "version",
        "PackageNotFoundError",
        "install_matplotlib_usetex_fallback",
        "BrukerHandler",
        "AgilentHandler",
        "MagritekHandler",
        "BrukerTopSpinHandler",
        "StandardizedMRIMixin",
        "StandardizedDataArrayMixin",
        "StandardizedPathMixin",
    ],
)
def test_removed_root_aliases_raise_attribute_error(name):
    """The former root aliases were removed on 15 November 2026."""
    import hypermri

    with pytest.raises(AttributeError, match=name):
        getattr(hypermri, name)
    assert name not in dir(hypermri)
