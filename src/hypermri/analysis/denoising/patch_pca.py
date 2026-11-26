"""Patch-wise PCA denoising with MP-PCA, threshold PCA and general PCA.

Each method builds, for every sliding patch, a matrix with one row per voxel
of the patch and one column per redundant measurement (for example
repetitions or spectral points). It subtracts the column means, decomposes the
matrix by principal component analysis, keeps the components that carry
signal and averages the projected patches back onto the voxel grid with equal
weights (overcomplete averaging). The methods differ only in how they separate
signal components from noise components:

* MP-PCA estimates the noise variance from the eigenvalues themselves by
  moment matching to the Marchenko-Pastur distribution (Veraart et al.,
  NeuroImage 142, 2016, 394-406).
* Threshold PCA (TPCA) and general PCA (GPCA) use an a-priori noise variance
  (Henriques et al., "Efficient PCA denoising of spatially correlated
  redundant MRI data", Imaging Neuroscience 1, 2023, Eqs. 8 and 9). They are
  robust to spatially correlated noise, which biases the MP-PCA estimate.

The functions here are implemented from these publications. The NumPy
kernels classify one eigenvalue spectrum; the labelled ``denoise_*``
functions resolve dimensions, cut patches and restore labels.

Conventions shared by kernels and labelled functions: a centred matrix with
``n_rows`` effective voxel rows and ``n_cols`` measurement columns has
``min(n_rows, n_cols)`` nonzero eigenvalues of its Gram matrix divided by
``max(n_rows, n_cols)``. With this normalisation pure noise of variance
``sigma^2`` per matrix entry gives eigenvalues inside
``sigma^2 * (1 +/- sqrt(gamma))^2`` with ``gamma = min / max``. For complex
data the variance is ``E|noise|^2``, the sum of the real and imaginary
variances.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "PCANoiseClassification",
    "compute_marchenko_pastur_upper_edge",
    "denoise_gpca",
    "denoise_mppca",
    "denoise_tpca",
    "estimate_gpca_rank",
    "estimate_mppca_rank",
    "estimate_tpca_rank",
]

_REFERENCES = {
    "mppca": "Veraart et al., NeuroImage 142 (2016) 394-406",
    "tpca": "Henriques et al., Imaging Neuroscience 1 (2023), Eq. 9",
    "gpca": "Henriques et al., Imaging Neuroscience 1 (2023), Eq. 8",
}


@dataclass(frozen=True)
class PCANoiseClassification:
    """Split of one eigenvalue spectrum into signal and noise components.

    Parameters
    ----------
    n_signal_components : int
        Number of largest eigenvalues classified as signal, from 0 to the
        number of eigenvalues. These components are kept by denoising.
    noise_variance : float
        Noise variance per matrix entry in squared signal units that the
        classification used or estimated. For complex data it is
        ``E|noise|^2``.
    """

    n_signal_components: int
    noise_variance: float


def compute_marchenko_pastur_upper_edge(
    noise_variance: float, *, aspect_ratio: float
) -> float:
    """Return the upper edge of the Marchenko-Pastur eigenvalue distribution.

    For a matrix of pure noise with variance ``noise_variance`` per entry, the
    eigenvalues of its normalised Gram matrix lie below
    ``noise_variance * (1 + sqrt(aspect_ratio))^2`` in the limit of large
    matrices (Henriques et al. 2023, Eq. 4 and Eq. 9).

    Parameters
    ----------
    noise_variance : float
        Noise variance per matrix entry in squared signal units, nonnegative.
    aspect_ratio : float
        Ratio ``gamma`` of the Gram-matrix size to the number of samples that
        the Gram matrix averages over, positive. For the conventions of this
        module it is ``min(n_rows, n_cols) / max(n_rows, n_cols)``.

    Returns
    -------
    float
        Upper edge in the units of the eigenvalues (squared signal units).

    Raises
    ------
    ValueError
        If ``noise_variance`` is negative or ``aspect_ratio`` is not positive.

    Examples
    --------
    >>> round(compute_marchenko_pastur_upper_edge(1.0, aspect_ratio=0.25), 6)
    2.25
    """
    if not noise_variance >= 0:
        raise ValueError(
            "compute_marchenko_pastur_upper_edge needs a nonnegative "
            f"noise_variance; got {noise_variance}."
        )
    if not aspect_ratio > 0:
        raise ValueError(
            "compute_marchenko_pastur_upper_edge needs a positive aspect_ratio; "
            f"got {aspect_ratio}."
        )
    return float((1 + np.sqrt(aspect_ratio)) ** 2 * noise_variance)


def _sorted_eigenvalues(eigenvalues, operation):
    """Return finite eigenvalues as a descending float64 vector."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"{operation} needs a nonempty one-dimensional eigenvalue array; "
            f"got shape {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{operation} received non-finite eigenvalues.")
    return np.sort(values)[::-1]


def _aspect_ratio(n_rows, n_cols, n_eigenvalues, operation):
    """Return ``min / max`` and check that the spectrum matches the shape."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError(
            f"{operation} needs positive n_rows and n_cols; got {n_rows} and {n_cols}."
        )
    if n_eigenvalues != min(n_rows, n_cols):
        raise ValueError(
            f"{operation} expects min(n_rows, n_cols) = {min(n_rows, n_cols)} "
            f"eigenvalues; got {n_eigenvalues}."
        )
    return min(n_rows, n_cols) / max(n_rows, n_cols)


def _check_noise_variance(noise_variance, operation):
    """Return a positive finite prior noise variance as float."""
    value = float(noise_variance)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(
            f"{operation} needs a positive finite noise_variance; got {noise_variance}."
        )
    return value


def estimate_mppca_rank(
    eigenvalues: np.ndarray, *, n_rows: int, n_cols: int
) -> PCANoiseClassification:
    """Classify PCA components by Marchenko-Pastur moment matching (MP-PCA).

    Starting with all eigenvalues as noise, the largest one is moved to the
    signal group until the mean of the ``c`` remaining eigenvalues is at
    least the noise variance implied by their spread,
    ``(max - min) / (4 * sqrt(c / max(n_rows, n_cols)))`` (Veraart et al.
    2016; Henriques et al. 2023, Eqs. 6 and 7). The mean of the remaining
    eigenvalues is the noise-variance estimate. No prior noise level is
    needed.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        One-dimensional real eigenvalues, in any order, of the Gram matrix of
        a centred ``n_rows x n_cols`` matrix divided by
        ``max(n_rows, n_cols)``; exactly ``min(n_rows, n_cols)`` values.
    n_rows, n_cols : int
        Effective matrix shape, positive. After column centring of a patch
        with ``V`` voxels the effective row count is ``V - 1``.

    Returns
    -------
    PCANoiseClassification
        Number of signal components (0 for pure noise) and the estimated noise
        variance in the units of the eigenvalues.

    Raises
    ------
    ValueError
        If the eigenvalue array is empty, non-finite, not one-dimensional or
        does not match the matrix shape.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.standard_normal((250, 20))
    >>> values = np.linalg.eigvalsh(noise.T @ noise / 250)
    >>> estimate_mppca_rank(values, n_rows=250, n_cols=20).n_signal_components
    0
    """
    operation = "estimate_mppca_rank"
    values = _sorted_eigenvalues(eigenvalues, operation)
    _aspect_ratio(n_rows, n_cols, values.size, operation)
    n_larger = max(n_rows, n_cols)
    for n_signal in range(values.size):
        noise = values[n_signal:]
        n_noise = noise.size
        mean = float(noise.mean())
        spread_variance = (noise[0] - noise[-1]) / (4 * np.sqrt(n_noise / n_larger))
        if mean >= spread_variance:
            return PCANoiseClassification(n_signal, max(mean, 0.0))
    # A single remaining eigenvalue has zero spread and always satisfies the
    # condition, so the loop returns before this point.
    raise AssertionError("estimate_mppca_rank did not terminate")


def estimate_tpca_rank(
    eigenvalues: np.ndarray, *, n_rows: int, n_cols: int, noise_variance: float
) -> PCANoiseClassification:
    """Classify PCA components with the Marchenko-Pastur upper edge (TPCA).

    Components whose eigenvalues are at least
    ``noise_variance * (1 + sqrt(gamma))^2`` with
    ``gamma = min(n_rows, n_cols) / max(n_rows, n_cols)`` are signal
    (Henriques et al. 2023, Eq. 9).

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        One-dimensional real eigenvalues as for :func:`estimate_mppca_rank`.
    n_rows, n_cols : int
        Effective matrix shape, positive.
    noise_variance : float
        A-priori noise variance per matrix entry in the units of the
        eigenvalues, positive.

    Returns
    -------
    PCANoiseClassification
        Number of eigenvalues at or above the edge and the prior variance.

    Raises
    ------
    ValueError
        If the eigenvalues do not match the matrix shape or the variance is
        not positive and finite.
    """
    operation = "estimate_tpca_rank"
    values = _sorted_eigenvalues(eigenvalues, operation)
    gamma = _aspect_ratio(n_rows, n_cols, values.size, operation)
    variance = _check_noise_variance(noise_variance, operation)
    edge = compute_marchenko_pastur_upper_edge(variance, aspect_ratio=gamma)
    return PCANoiseClassification(int(np.sum(values >= edge)), variance)


def estimate_gpca_rank(
    eigenvalues: np.ndarray, *, noise_variance: float
) -> PCANoiseClassification:
    """Classify PCA components by the mean of the noise eigenvalues (GPCA).

    The noise group is the largest set of smallest eigenvalues whose mean
    does not exceed ``noise_variance`` (Henriques et al. 2023, Eq. 8). The
    criterion needs no Marchenko-Pastur shape and therefore no matrix shape.
    When even the smallest eigenvalue exceeds the prior, all components are
    signal. For pure noise the mean of all eigenvalues equals the prior only
    in expectation, so GPCA keeps one component for roughly half of the
    noise realisations.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        One-dimensional real eigenvalues as for :func:`estimate_mppca_rank`.
    noise_variance : float
        A-priori noise variance per matrix entry in the units of the
        eigenvalues, positive.

    Returns
    -------
    PCANoiseClassification
        Number of signal components and the prior variance.

    Raises
    ------
    ValueError
        If the eigenvalue array is invalid or the variance is not positive and
        finite.
    """
    operation = "estimate_gpca_rank"
    values = _sorted_eigenvalues(eigenvalues, operation)
    variance = _check_noise_variance(noise_variance, operation)
    # Mean of the smallest c eigenvalues for c = size, size - 1, ..., 1.
    tail_means = np.cumsum(values[::-1])[::-1] / np.arange(values.size, 0, -1)
    noise_groups = np.flatnonzero(tail_means <= variance)
    n_signal = int(noise_groups[0]) if noise_groups.size else values.size
    return PCANoiseClassification(n_signal, variance)


def _project_patch(matrix, classify):
    """Denoise one ``voxels x measurements`` matrix and return its rank and variance."""
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    n_voxels, n_measurements = centred.shape
    n_rows = n_voxels - 1
    if n_measurements <= n_rows:
        gram = centred.conj().T @ centred / n_rows
        eigenvalues, vectors = np.linalg.eigh(gram)
        classification = classify(eigenvalues, n_rows, n_measurements)
        basis = vectors[:, ::-1][:, : classification.n_signal_components]
        projected = centred @ basis @ basis.conj().T
    else:
        gram = centred @ centred.conj().T / n_measurements
        eigenvalues, vectors = np.linalg.eigh(gram)
        # Column centring leaves one zero eigenvalue with the constant vector.
        classification = classify(eigenvalues[1:], n_rows, n_measurements)
        basis = vectors[:, ::-1][:, : classification.n_signal_components]
        projected = basis @ (basis.conj().T @ centred)
    return projected + mean, classification


def _resolve_dims(data, feature_dims, sample_dims, operation):
    """Check that feature and sample dimensions partition the data dimensions."""
    features = tuple(feature_dims)
    samples = tuple(sample_dims)
    for name, group in (("feature_dims", features), ("sample_dims", samples)):
        if not group:
            raise ValueError(f"{operation} needs at least one entry in {name}.")
        if len(set(group)) != len(group):
            raise ValueError(f"{operation}: {name} {group} contains duplicates.")
    overlap = set(features) & set(samples)
    if overlap:
        raise ValueError(
            f"{operation}: dimensions {sorted(overlap)} are in both feature_dims "
            "and sample_dims."
        )
    listed = set(features) | set(samples)
    if listed != set(data.dims):
        missing = [dim for dim in data.dims if dim not in listed]
        unknown = sorted(listed - set(data.dims))
        raise ValueError(
            f"{operation}: feature_dims and sample_dims must list every data "
            f"dimension exactly once. Data dimensions: {list(data.dims)}; "
            f"unlisted: {missing}; not in the data: {unknown}."
        )
    ordered_samples = tuple(dim for dim in data.dims if dim in samples)
    ordered_features = tuple(dim for dim in data.dims if dim in features)
    return ordered_samples, ordered_features


def _resolve_patch_shape(data, sample_dims, patch_shape, operation):
    """Return one patch length per sample dimension in data order."""
    if patch_shape is None:
        return tuple(data.sizes[dim] for dim in sample_dims)
    unknown = sorted(set(patch_shape) - set(sample_dims))
    if unknown:
        raise ValueError(
            f"{operation}: patch_shape names {unknown}, which are not sample "
            f"dimensions {list(sample_dims)}."
        )
    lengths = []
    for dim in sample_dims:
        length = patch_shape.get(dim, data.sizes[dim])
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
            raise ValueError(
                f"{operation}: patch_shape[{dim!r}] must be an integer; got {length!r}."
            )
        if not 1 <= length <= data.sizes[dim]:
            raise ValueError(
                f"{operation}: patch_shape[{dim!r}] = {length} must lie between 1 "
                f"and the dimension length {data.sizes[dim]}."
            )
        lengths.append(int(length))
    return tuple(lengths)


def _patch_noise_variance(noise_variance, data, sample_dims, operation):
    """Return the prior variance as a sample-grid array, or None for MP-PCA."""
    if noise_variance is None:
        return None
    if isinstance(noise_variance, xr.DataArray):
        unknown = sorted(set(noise_variance.dims) - set(sample_dims))
        if unknown:
            raise ValueError(
                f"{operation}: noise_variance has dimensions {unknown} that are "
                f"not sample dimensions {list(sample_dims)}."
            )
        template = data.isel(
            {dim: 0 for dim in data.dims if dim not in sample_dims}, drop=True
        )
        grid = noise_variance.broadcast_like(template).transpose(*sample_dims)
        values = np.asarray(grid.values, dtype=np.float64)
        if values.shape != tuple(data.sizes[dim] for dim in sample_dims):
            raise ValueError(
                f"{operation}: noise_variance does not align with the sample grid "
                f"{dict((dim, data.sizes[dim]) for dim in sample_dims)}; check its "
                "coordinates."
            )
    else:
        values = np.full(
            tuple(data.sizes[dim] for dim in sample_dims),
            float(noise_variance),
            dtype=np.float64,
        )
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError(
            f"{operation} needs a positive finite noise_variance at every voxel."
        )
    return values


def _denoise_patches(
    data: xr.DataArray,
    *,
    method: str,
    feature_dims: Sequence[str],
    sample_dims: Sequence[str],
    patch_shape: Mapping[str, int] | None,
    noise_variance: float | xr.DataArray | None,
    classify: Callable,
) -> xr.Dataset:
    """Run one patch-wise PCA denoising with the given component classifier."""
    operation = f"denoise_{method}"
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{operation} needs an xarray.DataArray; got {type(data)}.")
    samples, features = _resolve_dims(data, feature_dims, sample_dims, operation)
    window = _resolve_patch_shape(data, samples, patch_shape, operation)
    prior = _patch_noise_variance(noise_variance, data, samples, operation)

    values = data.transpose(*samples, *features).values
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{operation}: the data contain NaN or infinite samples; remove or "
            "mask them before denoising."
        )
    is_complex = np.iscomplexobj(values)
    work_dtype = np.complex128 if is_complex else np.float64
    grid_shape = values.shape[: len(samples)]
    n_features = int(np.prod(values.shape[len(samples) :]))
    n_voxels = int(np.prod(window))
    if n_voxels < 2 or n_features < 2:
        raise ValueError(
            f"{operation} needs at least two voxels per patch and two measurements; "
            f"got {n_voxels} voxels and {n_features} measurements."
        )
    matrix_grid = values.reshape(*grid_shape, n_features).astype(work_dtype)

    sum_signal = np.zeros_like(matrix_grid)
    sum_sigma = np.zeros(grid_shape)
    sum_rank = np.zeros(grid_shape)
    counts = np.zeros(grid_shape)
    sample_axes = tuple(range(len(samples)))
    patches = sliding_window_view(matrix_grid, window, axis=sample_axes)
    prior_patches = (
        None if prior is None else sliding_window_view(prior, window, axis=sample_axes)
    )
    for corner in np.ndindex(patches.shape[: len(samples)]):
        # sliding_window_view puts the window axes last: (features, *window).
        patch = np.moveaxis(patches[corner], 0, -1).reshape(n_voxels, n_features)
        if prior_patches is None:
            patch_prior = None
        else:
            patch_prior = float(np.median(prior_patches[corner]))
        denoised, classification = _project_patch(
            patch,
            lambda eig, rows, cols, p=patch_prior: classify(eig, rows, cols, p),
        )
        region = tuple(
            slice(start, start + length)
            for start, length in zip(corner, window, strict=True)
        )
        sum_signal[region] += denoised.reshape(*window, n_features)
        sum_sigma[region] += np.sqrt(classification.noise_variance)
        sum_rank[region] += classification.n_signal_components
        counts[region] += 1

    averaged = sum_signal / counts[..., np.newaxis]
    if not is_complex:
        averaged = averaged.real
    out_dtype = values.dtype if np.issubdtype(values.dtype, np.inexact) else work_dtype
    denoised_values = averaged.reshape(values.shape).astype(out_dtype, copy=False)
    denoised = xr.DataArray(
        denoised_values,
        dims=(*samples, *features),
        coords=data.coords,
        attrs=dict(data.attrs),
    ).transpose(*data.dims)

    sample_coords = {
        name: coord
        for name, coord in data.coords.items()
        if set(coord.dims) <= set(samples)
    }
    noise_sigma = xr.DataArray(
        sum_sigma / counts,
        dims=samples,
        coords=sample_coords,
        attrs={"description": "noise standard deviation per sample, signal units"},
    )
    n_components = xr.DataArray(
        sum_rank / counts,
        dims=samples,
        coords=sample_coords,
        attrs={"description": "mean number of signal components over patches"},
    )
    return xr.Dataset(
        {
            "denoised": denoised,
            "noise_sigma": noise_sigma,
            "n_components": n_components,
        },
        attrs={
            "denoising_method": method,
            "denoising_reference": _REFERENCES[method],
            "feature_dims": list(features),
            "sample_dims": list(samples),
            "patch_shape": [int(length) for length in window],
        },
    )


def denoise_mppca(
    data: xr.DataArray,
    *,
    feature_dims: Sequence[str],
    sample_dims: Sequence[str],
    patch_shape: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Denoise labelled data with Marchenko-Pastur PCA (MP-PCA).

    Every sliding patch over ``sample_dims`` forms a ``voxels x measurements``
    matrix; :func:`estimate_mppca_rank` estimates its noise variance from the
    eigenvalues and keeps the signal components. Overlapping patches are
    averaged with equal weights. MP-PCA assumes independent noise between
    voxels; with spatially correlated noise, for example after zero-filling
    or k-space filtering, prefer :func:`denoise_tpca` with a measured noise
    variance (Henriques et al. 2023).

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex signal in arbitrary signal units with finite values.
        Its dimensions are split exactly into ``feature_dims`` and
        ``sample_dims``; their order in ``data`` does not matter.
    feature_dims : sequence of str
        Redundant measurement dimensions that form the matrix columns, for
        example ``("repetition_idx",)`` or ``("spectral_idx", "time_idx")``.
        Together at least two measurements.
    sample_dims : sequence of str
        Dimensions of the voxel grid over which patches slide, for example
        ``("read_idx", "phase_idx")``.
    patch_shape : mapping of str to int, optional
        Patch length per sample dimension; unlisted dimensions use their full
        length. None uses one patch covering the whole grid (global PCA with
        the MP-PCA classifier). Patches slide with step 1.

    Returns
    -------
    xarray.Dataset
        ``denoised`` with the input dimension order, coordinates and
        attributes, in the input dtype for floating or complex input
        (float64 otherwise; calculation in float64 or complex128);
        ``noise_sigma`` (signal units, ``sqrt(E|noise|^2)`` for complex data)
        and ``n_components`` (mean number of kept components) over
        ``sample_dims``, averaged over the patches that contain each voxel.
        Dataset attributes record the method, reference, dimensions and
        patch shape.

    Raises
    ------
    TypeError
        If ``data`` is not a DataArray.
    ValueError
        If the dimension groups do not partition the data dimensions, the
        patch shape is invalid, a patch has fewer than two voxels, there are
        fewer than two measurements, or the data are not finite.

    Notes
    -----
    The function is eager and deterministic, does not modify ``data`` and
    computes one eigendecomposition of size
    ``min(voxels - 1, measurements)`` per patch.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> signal = np.outer(rng.standard_normal(64), rng.standard_normal(20))
    >>> data = xr.DataArray(
    ...     (signal + 0.1 * rng.standard_normal((64, 20))).reshape(8, 8, 20),
    ...     dims=("read_idx", "phase_idx", "repetition_idx"),
    ... )
    >>> result = denoise_mppca(
    ...     data,
    ...     feature_dims=("repetition_idx",),
    ...     sample_dims=("read_idx", "phase_idx"),
    ... )
    >>> int(result["n_components"][0, 0])
    1
    """

    def classify(eigenvalues, n_rows, n_cols, _prior):
        return estimate_mppca_rank(eigenvalues, n_rows=n_rows, n_cols=n_cols)

    return _denoise_patches(
        data,
        method="mppca",
        feature_dims=feature_dims,
        sample_dims=sample_dims,
        patch_shape=patch_shape,
        noise_variance=None,
        classify=classify,
    )


def denoise_tpca(
    data: xr.DataArray,
    *,
    feature_dims: Sequence[str],
    sample_dims: Sequence[str],
    noise_variance: float | xr.DataArray,
    patch_shape: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Denoise labelled data with threshold PCA (TPCA) and a known noise level.

    Every sliding patch keeps the components whose eigenvalues reach the
    Marchenko-Pastur upper edge for the a-priori noise variance
    (:func:`estimate_tpca_rank`, Henriques et al. 2023, Eq. 9). A noise map is
    reduced to its median within each patch. Overlapping patches are averaged
    with equal weights.

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex finite signal, as for :func:`denoise_mppca`.
    feature_dims, sample_dims : sequence of str
        Measurement and voxel-grid dimensions, as for :func:`denoise_mppca`.
    noise_variance : float or xarray.DataArray
        Noise variance per sample in squared signal units, positive; for
        complex data ``E|noise|^2``. A DataArray is a noise map over some or
        all ``sample_dims`` and is broadcast to the grid. Estimate it
        independently of the denoised data, for example from repetitions or a
        signal-free region.
    patch_shape : mapping of str to int, optional
        Patch length per sample dimension, as for :func:`denoise_mppca`.

    Returns
    -------
    xarray.Dataset
        As for :func:`denoise_mppca`; ``noise_sigma`` is the square root of
        the prior used in each patch.

    Raises
    ------
    TypeError
        If ``data`` is not a DataArray.
    ValueError
        For the conditions of :func:`denoise_mppca`, or if the noise variance
        is not positive and finite everywhere or has non-sample dimensions.

    Notes
    -----
    The function is eager and deterministic and does not modify its inputs.
    """

    def classify(eigenvalues, n_rows, n_cols, prior):
        return estimate_tpca_rank(
            eigenvalues, n_rows=n_rows, n_cols=n_cols, noise_variance=prior
        )

    return _denoise_patches(
        data,
        method="tpca",
        feature_dims=feature_dims,
        sample_dims=sample_dims,
        patch_shape=patch_shape,
        noise_variance=noise_variance,
        classify=classify,
    )


def denoise_gpca(
    data: xr.DataArray,
    *,
    feature_dims: Sequence[str],
    sample_dims: Sequence[str],
    noise_variance: float | xr.DataArray,
    patch_shape: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Denoise labelled data with general PCA (GPCA) and a known noise level.

    Every sliding patch treats as noise the largest set of smallest
    eigenvalues whose mean does not exceed the a-priori noise variance
    (:func:`estimate_gpca_rank`, Henriques et al. 2023, Eq. 8). A noise map is
    reduced to its median within each patch. Overlapping patches are averaged
    with equal weights.

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex finite signal, as for :func:`denoise_mppca`.
    feature_dims, sample_dims : sequence of str
        Measurement and voxel-grid dimensions, as for :func:`denoise_mppca`.
    noise_variance : float or xarray.DataArray
        Noise variance per sample, as for :func:`denoise_tpca`.
    patch_shape : mapping of str to int, optional
        Patch length per sample dimension, as for :func:`denoise_mppca`.

    Returns
    -------
    xarray.Dataset
        As for :func:`denoise_tpca`.

    Raises
    ------
    TypeError
        If ``data`` is not a DataArray.
    ValueError
        As for :func:`denoise_tpca`.

    Notes
    -----
    The function is eager and deterministic and does not modify its inputs.
    """

    def classify(eigenvalues, _n_rows, _n_cols, prior):
        return estimate_gpca_rank(eigenvalues, noise_variance=prior)

    return _denoise_patches(
        data,
        method="gpca",
        feature_dims=feature_dims,
        sample_dims=sample_dims,
        patch_shape=patch_shape,
        noise_variance=noise_variance,
        classify=classify,
    )
