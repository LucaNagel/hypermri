Denoising Algorithms
====================

Global PCA is available as ``pca_denoise`` from the generic package and the
algorithms module. It decomposes the whole image grid at once.

Patch-wise MP-PCA, threshold PCA (TPCA) and general PCA (GPCA) are available as
``denoise_mppca``, ``denoise_tpca`` and ``denoise_gpca`` from the generic
package and the ``patch_pca`` module. They are implemented from Veraart et al.,
NeuroImage 142 (2016) 394-406, and Henriques et al., Imaging Neuroscience 1
(2023), Eqs. 8 and 9. MP-PCA estimates the noise level from the eigenvalues;
TPCA and GPCA take an a-priori noise variance and are robust to spatially
correlated noise. The eigenvalue classifiers ``estimate_mppca_rank``,
``estimate_tpca_rank`` and ``estimate_gpca_rank`` work on plain NumPy arrays.
Global PCA and the patch-wise methods are different operations and are not
substituted for each other.

The earlier local implementations were removed on 10 October 2026 because
redistribution permission for the copied source code was not established. The
removed APIs ``genpca``, ``mp_distribution``, ``median_filt``, ``tpca_denoise``
and the ``hypermri.analysis.dce.denoising`` module stay removed; the new
functions have different names and signatures and are not drop-in
replacements.

Example for a CSI series with redundant spectral points per voxel and a noise
variance measured in a signal-free region:

.. code-block:: python

   from hypermri.analysis.denoising import denoise_tpca

   result = denoise_tpca(
       csi,
       feature_dims=("spectral_idx", "repetition_idx"),
       sample_dims=("read_idx", "phase_idx", "slice_idx"),
       noise_variance=noise_variance,
       patch_shape={"read_idx": 5, "phase_idx": 5, "slice_idx": 1},
   )
   denoised = result["denoised"]

.. automodule:: hypermri.analysis.denoising.algorithms
   :members: pca_denoise

.. automodule:: hypermri.analysis.denoising.patch_pca
   :members:
