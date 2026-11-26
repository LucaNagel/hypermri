"""Canonical scientific modules and the removal of their former import paths."""

import doctest
import importlib
import pickle

import pytest

# Former warning-only module paths removed before the first public release by
# the decision of 13 November 2026. Each maps to its canonical owner.
REMOVED_MODULES = {
    "hypermri.core": "hypermri.loading",
    "hypermri.core.data_types": "hypermri.loading.contracts",
    "hypermri.core.exceptions": "hypermri.exceptions",
    "hypermri.core.experiment": "hypermri.loading.experiment",
    "hypermri.core.factory": "hypermri.loading.scans",
    "hypermri.core.interfaces": "hypermri.loading.protocols",
    "hypermri.core.mapping_files": "hypermri.config.mappings",
    "hypermri.core.precision": "hypermri.reconstruction.precision",
    "hypermri.functions": "hypermri.analysis",
    "hypermri.functions.IDEAL": "hypermri.analysis.fitting.ideal",
    "hypermri.functions.pHIDEAL": "hypermri.analysis.fitting.phase_ideal",
    "hypermri.functions.pHconverter": "hypermri.analysis.spectroscopy.ph",
    "hypermri.functions.nsa_simulation": "hypermri.visualization.nsa_dashboard",
    "hypermri.functions.specsimulator": "hypermri.visualization.spectrum_simulator",
    "hypermri.functions.csimulator": "hypermri.visualization.csi_simulator",
    "hypermri.functions.dashboard_template": (
        "hypermri.visualization.dashboard_template"
    ),
    "hypermri.vendorhandler": "hypermri.vendors",
    "hypermri.topspin_exp": "hypermri.vendors.topspin",
    "hypermri.analysis.anatomical.core": "hypermri.analysis.anatomical",
    "hypermri.analysis.spectroscopy.core": "hypermri.analysis.spectroscopy",
    "hypermri.analysis.fitting.core": "hypermri.analysis.fitting",
    "hypermri.analysis.fitting.core.conversions": "hypermri.analysis.dce.conversions",
    "hypermri.analysis.fitting.core.execution": "hypermri.analysis.fitting.execution",
    "hypermri.analysis.fitting.core.executors": "hypermri.analysis.fitting.voxelwise",
    "hypermri.analysis.fitting.core.metadata": (
        "hypermri.analysis.fitting.result_metadata"
    ),
    "hypermri.visualization.core.colormap_policy": (
        "hypermri.visualization.colormap_policy"
    ),
    "hypermri.visualization.core.interfaces": "hypermri.visualization.contracts",
    "hypermri.visualization.core.model": "hypermri.visualization.render_data",
    "hypermri.visualization.core.state": "hypermri.visualization.state",
    "hypermri.visualization.core.state_persistence": (
        "hypermri.visualization.state_persistence"
    ),
    "hypermri.utils.calculations": "hypermri.utils.mathematics.calculations",
    "hypermri.utils.channel_combination": "hypermri.utils.mri.channels",
    "hypermri.utils.io.data_io": "hypermri.utils.io.loaders",
    "hypermri.utils.mathematics.utils_fit": "hypermri.analysis.fitting.distributions",
    "hypermri.utils.plot.utils_plot": "hypermri.utils.plot.matplotlib_helpers",
    "hypermri.utils.spectroscopy.channels": "hypermri.utils.mri.channels",
    "hypermri.utils.utils_interpolation": "hypermri.utils.image.interpolation",
    "hypermri.utils.utils_overlay_source_images_over_reference_images": (
        "hypermri.visualization.image_overlay"
    ),
    "hypermri.utils.utils_plot_fit_results_overview": (
        "hypermri.visualization.fit_results_overview"
    ),
    "hypermri.utils.utils_plot_fit_spectra_overlay": (
        "hypermri.visualization.fit_spectra_overlay"
    ),
    "hypermri.utils.utils_sequences": "hypermri.utils.sequence.generation",
    "hypermri.utils.utils_study": "hypermri.utils.study.manipulation",
}


@pytest.mark.parametrize("old_path", sorted(REMOVED_MODULES))
def test_removed_module_paths_no_longer_import(old_path):
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(old_path)
    # A removed package reports itself for every former submodule.
    assert old_path.startswith(error.value.name)


@pytest.mark.parametrize("new_path", sorted(set(REMOVED_MODULES.values())))
def test_replacement_modules_import(new_path):
    assert importlib.import_module(new_path).__name__ == new_path


@pytest.mark.parametrize(
    "pickled",
    [
        b"chypermri.analysis.fitting.core.execution\nExecutionConfig\n.",
        b"chypermri.analysis.spectroscopy.core\nget_freq_axis\n.",
        b"chypermri.utils.plot.utils_plot\nget_overlapping_slices\n.",
    ],
)
def test_pickles_naming_removed_modules_fail_with_module_error(pickled):
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(pickled)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("hypermri.analysis.fitting.execution", "ExecutionConfig"),
        ("hypermri.reconstruction.axes", "get_freq_axis"),
        ("hypermri.utils.image.slices", "get_overlapping_slices"),
    ],
)
def test_canonical_definitions_pickle_by_their_own_module(module, name):
    value = getattr(importlib.import_module(module), name)
    assert value.__module__ == module
    assert pickle.loads(pickle.dumps(value)) is value


@pytest.mark.parametrize(
    "module",
    [
        "hypermri.analysis.fitting.execution",
        "hypermri.analysis.fitting.result_metadata",
        "hypermri.analysis.fitting.voxelwise",
        "hypermri.analysis.dce.conversions",
        "hypermri.analysis.spectroscopy.ph",
        "hypermri.analysis.spectroscopy.frequency",
        "hypermri.analysis.fitting.distributions",
        "hypermri.utils.plot.matplotlib_helpers",
        "hypermri.utils.image.slices",
    ],
)
def test_domain_examples_execute_with_the_shipped_implementation(module):
    result = doctest.testmod(importlib.import_module(module))
    assert result.attempted > 0
    assert result.failed == 0
