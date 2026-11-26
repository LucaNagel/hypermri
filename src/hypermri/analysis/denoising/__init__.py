"""Expose the global and patch-wise PCA denoisers through explicit imports.

The algorithms module owns the global ``pca_denoise``; the patch_pca module
owns patch-wise MP-PCA, threshold PCA and general PCA and their eigenvalue
classifiers. Each module owns its numerical implementation, units and dtype
contracts. Imports do not process arrays or import matplotlib. ``__all__``
lists only these denoising names; imported child modules are not public
wildcard exports.
"""

from .algorithms import pca_denoise
from .patch_pca import (
    PCANoiseClassification,
    compute_marchenko_pastur_upper_edge,
    denoise_gpca,
    denoise_mppca,
    denoise_tpca,
    estimate_gpca_rank,
    estimate_mppca_rank,
    estimate_tpca_rank,
)

__all__ = [
    "PCANoiseClassification",
    "compute_marchenko_pastur_upper_edge",
    "denoise_gpca",
    "denoise_mppca",
    "denoise_tpca",
    "estimate_gpca_rank",
    "estimate_mppca_rank",
    "estimate_tpca_rank",
    "pca_denoise",
]
