"""Global PCA denoising for labelled MRI arrays.

``pca_denoise`` decomposes the whole image grid at once. Patch-wise MP-PCA,
threshold PCA and general PCA live in :mod:`hypermri.analysis.denoising.patch_pca`;
both modules share its Marchenko-Pastur upper edge. Global PCA is a separate
operation and not a replacement for the patch-wise methods.
"""

from numbers import Integral
from typing import Union

import numpy as np
import xarray as xr
from scipy.linalg import eigh, svd

from ...utils.system.logging import init_default_logger
from .patch_pca import compute_marchenko_pastur_upper_edge

logger = init_default_logger(__name__)

__all__ = ["pca_denoise"]

_NOISE_ESTIMATION_METHODS = ("auto", "corner", "last_reps", "manual")
# Fraction of the flattened redundant samples that "last_reps" treats as noise.
_LAST_REPS_NOISE_FRACTION = 0.1


def pca_denoise(
    input_xarray: xr.DataArray,
    components: Union[int, list, slice] = None,
    pca_method: str = "eig",
    plot: bool = False,
    db_mode: bool = False,
    spatial_dims_present: list = None,
    redundant_dims_present: list = None,
    noise_threshold: float = None,
    noise_estimation_method: str = "auto",
    treat_as_redundant: list = None,
) -> xr.DataArray:
    """Denoise labelled MRI data using a global principal-component projection.

    Spatial samples are pooled into one matrix, centred across pixels for each
    redundant sample, projected onto selected principal components and restored
    to their original labelled layout. This is global PCA, not a local
    patch-based TPCA, GPCA or MPPCA denoiser. Automatic component selection
    keeps the eigenvalues above the Marchenko-Pastur upper edge
    ``sigma^2 * (1 + sqrt(n_repetitions / n_pixels))^2`` for the noise
    variance ``sigma^2`` (Henriques et al., Imaging Neuroscience 1, 2023,
    Eq. 9) and always retains at least one component.

    Parameters
    ----------
    input_xarray : xarray.DataArray
        Real or complex signal in arbitrary but consistent signal units, with
        at least one pixel dimension and one redundant measurement dimension.
        Dimension order is arbitrary. Default spatial dimensions are the
        present read_idx, phase_idx and slice_idx axes; acquisition_idx is the
        fallback pixel axis when no spatial axis is present. Coordinates need
        no particular orientation or physical calibration.
    components : int, list of int, slice, or None, optional
        Components to retain, numbered from the largest eigenvalue. An integer
        (Python or NumPy, not bool) keeps that many leading components; a list
        or slice selects indices.
        None selects components using the noise-variance threshold. Indices
        should be nonnegative and less than the redundant sample count.
    pca_method : str, optional
        Decomposition method, 'eig' for Hermitian covariance decomposition or
        'svd' for singular-value decomposition; spelling is case-insensitive.
    plot : bool, optional
        Display linear and logarithmic scree plots after decomposition.
    db_mode : bool, optional
        Print dimension, decomposition and component-selection diagnostics.
    spatial_dims_present : list of str, optional
        Explicit pixel dimensions. Together with redundant_dims_present these
        must partition the input dimensions without overlap or duplication.
        None infers the present conventional spatial axes.
    redundant_dims_present : list of str, optional
        Explicit measurement dimensions to flatten together, for example
        repetitions or echoes. None uses all non-spatial dimensions.
    noise_threshold : float, optional
        Noise variance in squared input signal units for automatic component
        selection. When given, it is used instead of any estimate and the
        estimator is not evaluated; it is required for 'manual'. Ignored when
        components is explicit.
    noise_estimation_method : str, optional
        Variance estimator used only when components is None. 'corner' uses
        endpoints of read_idx and phase_idx and requires both axes;
        'last_reps' uses the final 10% of flattened redundant samples and
        requires at least ten of them. 'auto' uses 'corner' when read_idx and
        phase_idx are both present and 'last_reps' otherwise, and records the
        reason in the result attributes. 'manual' requires noise_threshold.
        These heuristics assume the selected samples represent noise; they do
        not validate that scientific assumption.
    treat_as_redundant : list of str, optional
        Axes to exclude from inferred spatial dimensions and include among
        redundant samples. Use with inferred dimension groups, for example
        when slice_idx actually labels repeated acquisitions.

    Returns
    -------
    xarray.DataArray
        Eagerly computed projected signal with the input shape, dimension order,
        coordinates, name, attributes and signal units. Calculation/output
        precision follows NumPy and SciPy promotion; no storage-dtype cast is
        imposed. The attributes are a shallow copy of the input attributes.
        When components is None they also hold
        ``pca_noise_estimation_method`` ('manual', 'corner' or 'last_reps',
        the source of the noise variance used) and, when 'auto' used
        'last_reps', ``pca_noise_estimation_fallback_reason``.

    Raises
    ------
    ValueError
        If pixel or redundant dimensions are missing, dimension groups cannot
        partition the input, the decomposition/component/estimator option is
        invalid, manual variance is absent, 'corner' lacks read_idx or
        phase_idx, or 'last_reps' has fewer than ten redundant samples.
    numpy.linalg.LinAlgError
        If the decomposition does not converge.

    Notes
    -----
    The operation eagerly reads array values, does not mutate the input or cache
    results, and is deterministic for fixed inputs/options within numerical
    linear-algebra precision. Non-finite samples are replaced by zero in the
    calculation with a log warning. Plotting and diagnostic printing occur only
    when requested; matplotlib is imported only for ``plot=True``. Equal or nearly equal eigenvalues can yield different bases
    across linear-algebra implementations.

    Examples
    --------
    >>> signal = xr.DataArray(
    ...     np.ones((3, 4)), dims=('read_idx', 'repetition_idx')
    ... )
    >>> result = pca_denoise(signal, components=1)
    >>> result.dims == signal.dims
    True
    """
    # --- 1. Identify Spatial and Redundant Dimensions ---
    if db_mode:
        print("--- Step 1: Identifying Dimensions ---")

    SPATIAL_DIMS = ["read_idx", "phase_idx", "slice_idx"]

    # Handle treat_as_redundant parameter for spectral data
    if treat_as_redundant is None:
        treat_as_redundant = []

    if spatial_dims_present is None:
        # Exclude dimensions that should be treated as redundant
        spatial_dims_present = [
            dim
            for dim in SPATIAL_DIMS
            if dim in input_xarray.dims and dim not in treat_as_redundant
        ]

    if redundant_dims_present is None:
        # Include all non-spatial dims plus those forced to be redundant
        redundant_dims_present = [
            dim
            for dim in input_xarray.dims
            if dim not in spatial_dims_present or dim in treat_as_redundant
        ]

    # Validate dimensions
    if not spatial_dims_present:
        # For spectral-only data, we need at least acquisition_idx as "spatial" dimension
        if "acquisition_idx" in input_xarray.dims:
            spatial_dims_present = ["acquisition_idx"]
            redundant_dims_present = [
                dim for dim in input_xarray.dims if dim != "acquisition_idx"
            ]
            if db_mode:
                print(
                    "Note: Using 'acquisition_idx' as spatial dimension for spectral data."
                )
        else:
            raise ValueError(
                "No spatial or pixel dimensions found. "
                "For spectral data, 'acquisition_idx' is required. "
                "Available dimensions: " + str(list(input_xarray.dims))
            )

    if not redundant_dims_present:
        raise ValueError(
            "No redundant dimensions for PCA found. "
            "PCA denoising requires multiple measurements (e.g., repetitions, slices, echoes). "
            "Available dimensions: " + str(list(input_xarray.dims))
        )

    if db_mode:
        print(f"Spatial dimensions identified: {spatial_dims_present}")
        print(f"Redundant dimensions identified: {redundant_dims_present}")
        if treat_as_redundant:
            print(f"Dimensions forced to be redundant: {treat_as_redundant}")
        print()

    # --- 2. Reshape Data for PCA ---
    if db_mode:
        print("--- Step 2: Reshaping Data for PCA ---")

    original_dims = input_xarray.dims
    transposed_xarray = input_xarray.transpose(
        *redundant_dims_present, *spatial_dims_present
    )
    n_pixels = np.prod([transposed_xarray.sizes[dim] for dim in spatial_dims_present])
    n_repetitions = np.prod(
        [transposed_xarray.sizes[dim] for dim in redundant_dims_present]
    )
    X = transposed_xarray.values.reshape(n_repetitions, n_pixels).T

    # Check for NaNs or Infs
    if not np.all(np.isfinite(X)):
        msg = "Input data contains NaNs or Infs. Replacing with 0 for PCA denoising."
        if db_mode:
            print(f"Warning: {msg}")
        logger.warning(msg)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if db_mode:
        print(f"Shape of reshaped matrix X: {X.shape}\n")

    # --- 3. Center the Data ---
    mean_of_each_repetition = X.mean(axis=0)
    X_centered = X - mean_of_each_repetition

    # --- 4. Perform Eigendecomposition ---
    if db_mode:
        print(f"--- Step 4: Performing Eigendecomposition using '{pca_method}'---")
    M, N = X_centered.shape

    if pca_method.lower() == "eig":
        # Scale data to avoid overflow during matrix multiplication
        # covariance = (1/M) * (X.T @ X) = (X/sqrt(M)).T @ (X/sqrt(M))
        X_scaled = X_centered / np.sqrt(M)
        covariance_matrix = X_scaled.conj().T @ X_scaled
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    elif pca_method.lower() == "svd":
        # Scale data to avoid overflow during squared singular value calculation
        X_scaled = X_centered / np.sqrt(M)
        U, s, Vt = svd(X_scaled, full_matrices=False)
        eigenvalues = s**2
        eigenvectors = Vt.conj().T
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    # Ensure eigenvalues is a numpy array (not xarray DataArray)
    if hasattr(eigenvalues, "values"):
        eigenvalues = eigenvalues.values

    if db_mode:
        print("Eigendecomposition complete.\n")

    # --- 5. Determine Components to Keep ---
    if db_mode:
        print("--- Step 5: Determining Components to Keep ---")

    # Indices of components to keep
    components_to_keep_indices = None
    noise_attrs = {}

    if components is None:
        # Convert noise_threshold to scalar if it's an xarray or numpy array
        if noise_threshold is not None:
            if hasattr(noise_threshold, "values"):
                # xarray DataArray
                noise_threshold = float(noise_threshold.values)
            elif isinstance(noise_threshold, np.ndarray):
                # numpy array - take mean or first element
                noise_threshold = float(np.mean(noise_threshold))
            else:
                noise_threshold = float(noise_threshold)

        noise_variance_prior, noise_source, fallback_reason = _resolve_noise_variance(
            input_xarray, X, noise_estimation_method, noise_threshold
        )
        noise_attrs = {"pca_noise_estimation_method": noise_source}
        if fallback_reason is not None:
            noise_attrs["pca_noise_estimation_fallback_reason"] = fallback_reason
        if db_mode:
            print(f"Noise variance from {noise_source}: {noise_variance_prior:.4f}")
            if fallback_reason is not None:
                print(f"Fallback: {fallback_reason}")

        # Marchenko-Pastur upper edge of X^H X / n_pixels; the aspect ratio is
        # the covariance size over the number of pixel samples.
        lambda_t_tpca = compute_marchenko_pastur_upper_edge(
            noise_variance_prior, aspect_ratio=n_repetitions / n_pixels
        )
        n_components_to_keep = max(1, np.sum(eigenvalues > lambda_t_tpca))
        components_to_keep_indices = np.arange(n_components_to_keep)

        if db_mode:
            print(f"Noise variance estimate: {noise_variance_prior:.4f}")
            print(f"Gamma (repetitions/pixels): {n_repetitions / n_pixels:.4f}")
            print(f"TPCA Threshold (lambda_t): {lambda_t_tpca:.4f}")
            print(f"Keeping top {n_components_to_keep} components.\n")

    elif isinstance(components, Integral) and not isinstance(components, bool):
        if db_mode:
            print(f"Keeping top {components} components (user-specified).")
        components_to_keep_indices = np.arange(int(components))
    elif isinstance(components, list):
        if db_mode:
            print(f"Keeping specific components: {components} (user-specified).")
        components_to_keep_indices = np.array(components)
    elif isinstance(components, slice):
        if db_mode:
            print(f"Keeping range of components: {components} (user-specified).")
        components_to_keep_indices = np.arange(n_repetitions)[components]
    else:
        raise ValueError(
            "pca_denoise: components must be an int, list, slice, or None. "
            f"Got {type(components)}: {components}"
        )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(eigenvalues.real, "o-")
        ax[0].set_title("Scree Plot: Eigenvalues")
        ax[0].set_xlabel("Principal Component Number")
        ax[0].set_ylabel("Eigenvalue (Variance Explained)")
        if components is None:
            ax[0].axhline(
                y=lambda_t_tpca,
                color="r",
                linestyle="--",
                label=f"TPCA Threshold ({lambda_t_tpca:.2f})",
            )

        # Highlight kept components
        for idx in components_to_keep_indices:
            if idx < len(eigenvalues):
                ax[0].plot(idx, eigenvalues[idx].real, "go")

        ax[0].legend()
        ax[1].plot(eigenvalues.real, "o-")
        ax[1].set_yscale("log")
        ax[1].set_title("Scree Plot (Log Scale)")
        ax[1].set_xlabel("Principal Component Number")
        ax[1].set_ylabel("Log(Eigenvalue)")
        if components is None:
            ax[1].axhline(y=lambda_t_tpca, color="r", linestyle="--")
        plt.tight_layout()
        plt.show()

    # --- 6. Denoise and Reconstruct Data ---
    if db_mode:
        print("--- Step 6: Denoising and Reconstructing ---")
    projected_data = X_centered @ eigenvectors
    projected_data_filtered = np.zeros_like(projected_data)

    # Keep only the selected components
    valid_indices = components_to_keep_indices[
        components_to_keep_indices < n_repetitions
    ]
    projected_data_filtered[:, valid_indices] = projected_data[:, valid_indices]

    X_denoised_centered = projected_data_filtered @ eigenvectors.conj().T
    X_denoised = X_denoised_centered + mean_of_each_repetition

    # --- 7. Reshape Data Back to Original Xarray Format ---
    if db_mode:
        print("--- Step 7: Reshaping Data to Original Format ---")

    denoised_multidim_array = X_denoised.T.reshape(transposed_xarray.shape)
    denoised_transposed_xarray = xr.DataArray(
        data=denoised_multidim_array,
        coords=transposed_xarray.coords,
        dims=transposed_xarray.dims,
    )
    input_xarray_dn = denoised_transposed_xarray.transpose(*original_dims)
    input_xarray_dn.attrs = {**input_xarray.attrs, **noise_attrs}
    input_xarray_dn.name = input_xarray.name

    if db_mode:
        print("\n--- Denoising Complete ---")
        print(f"Shape of final denoised xarray: {input_xarray_dn.shape}\n")

    return input_xarray_dn


def _resolve_noise_variance(data, samples, method, noise_threshold):
    """Return the noise variance, its source and an ``auto`` fallback reason.

    ``samples`` is the pixel-by-redundant-sample matrix with non-finite
    values set to zero. A given ``noise_threshold`` is used without
    evaluating an estimator, so its source is reported as ``"manual"``.
    """
    if method not in _NOISE_ESTIMATION_METHODS:
        valid = ", ".join(repr(name) for name in _NOISE_ESTIMATION_METHODS)
        raise ValueError(
            f"pca_denoise: unknown noise_estimation_method {method!r}; "
            f"use one of {valid}."
        )
    if noise_threshold is not None:
        return noise_threshold, "manual", None
    if method == "manual":
        raise ValueError(
            "noise_estimation_method='manual' requires noise_threshold to be provided."
        )
    has_corners = {"read_idx", "phase_idx"} <= set(data.dims)
    if method == "corner" and not has_corners:
        raise ValueError(
            "pca_denoise: noise_estimation_method='corner' requires read_idx and "
            f"phase_idx; the input has {list(data.dims)}. Use 'last_reps' or "
            "pass noise_threshold."
        )
    if method in ("corner", "auto") and has_corners:
        corners = data.isel(read_idx=[0, -1], phase_idx=[0, -1]).values
        return np.var(corners), "corner", None

    n_redundant = samples.shape[1]
    n_noise = int(n_redundant * _LAST_REPS_NOISE_FRACTION)
    if n_noise < 1:
        raise ValueError(
            f"pca_denoise: noise_estimation_method={method!r} estimates the noise "
            "from the last 10% of the redundant samples and needs at least 10 "
            f"redundant samples; got {n_redundant}. Pass noise_threshold "
            "(noise variance in squared signal units) instead."
        )
    fallback_reason = None
    if method == "auto":
        fallback_reason = (
            "read_idx and phase_idx are not both present, so the corner "
            "estimate is unavailable; used last_reps."
        )
    return np.var(samples[:, -n_noise:]), "last_reps", fallback_reason
