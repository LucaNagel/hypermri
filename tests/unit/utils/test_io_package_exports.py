"""Public export surface of ``hypermri.utils.io``."""

import warnings

import hypermri.utils.io as io_package


def test_every_name_in_all_is_importable():
    missing = [name for name in io_package.__all__ if not hasattr(io_package, name)]

    assert missing == []


def test_wildcard_import_succeeds():
    namespace: dict = {}

    exec("from hypermri.utils.io import *", namespace)

    for name in io_package.__all__:
        assert name in namespace


def test_package_import_emits_no_deprecation_warning():
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(io_package)

    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_loader_and_file_utilities_are_exported_from_their_modules():
    from hypermri.utils.io import file_utils, loaders

    assert io_package.load_pc_sam_temp_file is loaders.load_pc_sam_temp_file
    assert io_package.ensure_directory is file_utils.ensure_directory
