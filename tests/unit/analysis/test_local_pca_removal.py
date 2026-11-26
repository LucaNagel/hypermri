"""Removed local PCA workflows cannot silently resolve to global PCA."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["hypermri.analysis.denoising", "hypermri.analysis.denoising.algorithms"],
)
def test_removed_local_pca_apis_are_not_available(module_name):
    module = importlib.import_module(module_name)
    namespace = {}
    exec(f"from {module_name} import *", {}, namespace)

    for name in (
        "genpca",
        "mp_distribution",
        "median_filt",
        "tpca_denoise",
        "tpca_denoise_dce",
        "_gpca_classifier",
        "_pca_classifier_slow",
        "_classify_noise_components",
        "_calc_tau_factor",
    ):
        assert not hasattr(module, name)
        assert name not in namespace
    assert callable(module.pca_denoise)
    assert namespace["pca_denoise"] is module.pca_denoise


def test_removed_dce_denoising_module_is_unavailable():
    name = "hypermri.analysis.dce.denoising"
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(name)
    assert error.value.name == name
