"""Physical Gaussian/Fermi filtering along labelled Read and Phase axes.

Window evaluation and Fourier filtering are independent of acquisition order,
sequence type and interactive tools. Frequencies use cycles/mm, not radians/mm.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import xarray as xr
from scipy.special import expit

from ..reconstruction.centred_dft import centered_fft, centered_ifft

__all__ = [
    "SpatialFilterOptions",
    "kspace_filter_weights",
    "apply_spatial_filter",
    "spatial_to_kspace",
]


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or value is None:
        raise ValueError(f"Spatial filter {name} must be a positive finite scalar.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Spatial filter {name} must be a positive finite scalar."
        ) from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Spatial filter {name} must be positive and finite.")
    return value


@dataclass(frozen=True)
class SpatialFilterOptions:
    """Specify an isotropic radial low-pass window in physical k-space units.

    Parameters
    ----------
    kind : {'gaussian', 'fermi'}, optional
        Window profile, default 'gaussian'. Distances are from physical k=0,
        not a measured signal maximum or a fraction of the acquired radius.
    sigma_per_mm : float or None, optional
        Positive Gaussian standard deviation in cycles/mm, required for
        'gaussian'. The window is exp(-r**2 / (2*sigma_per_mm**2)).
    radius_per_mm : float or None, optional
        Positive Fermi half-height radius in cycles/mm, required for 'fermi'.
    transition_per_mm : float or None, optional
        Positive Fermi logistic transition scale in cycles/mm, required for
        'fermi'. The window is 1/(1+exp((r-radius_per_mm)/transition_per_mm)).
        It is not divided by its value at zero; its half-height remains 0.5.

    Raises
    ------
    ValueError
        For unsupported profiles, absent/nonpositive/nonfinite widths or fields
        belonging to the other profile. Widths have no inferred defaults.

    Notes
    -----
    Immutable, deterministic and free of data reads, mutation or widget state.
    Gaussian gain at zero is one; Fermi gain at zero can be below one for a
    broad transition. This keeps the explicit logistic-window convention.

    Examples
    --------
    >>> gaussian = SpatialFilterOptions(sigma_per_mm=0.1)
    >>> fermi = SpatialFilterOptions(kind="fermi", radius_per_mm=0.1,
    ...                              transition_per_mm=0.02)
    """

    kind: Literal["gaussian", "fermi"] = "gaussian"
    sigma_per_mm: float | None = None
    radius_per_mm: float | None = None
    transition_per_mm: float | None = None

    def __post_init__(self):
        if self.kind not in {"gaussian", "fermi"}:
            raise ValueError("Spatial filter kind must be 'gaussian' or 'fermi'.")
        required = (
            ("sigma_per_mm",)
            if self.kind == "gaussian"
            else ("radius_per_mm", "transition_per_mm")
        )
        for name in ("sigma_per_mm", "radius_per_mm", "transition_per_mm"):
            value = getattr(self, name)
            if name in required:
                object.__setattr__(self, name, _positive(value, name))
            elif value is not None:
                raise ValueError(
                    f"Spatial filter {name} is not used by {self.kind}; omit it."
                )


def kspace_filter_weights(
    k_read_per_mm: np.ndarray,
    k_phase_per_mm: np.ndarray,
    options: SpatialFilterOptions,
) -> np.ndarray:
    """Evaluate a physical radial window at Cartesian or paired k-space points.

    Parameters
    ----------
    k_read_per_mm, k_phase_per_mm : array_like
        Real finite coordinates in cycles/mm. NumPy broadcasting applies:
        (n_read, 1) and (1, n_phase) give a grid; equally shaped vectors give
        paired acquired points. Coordinates can descend or be nonuniform.
    options : SpatialFilterOptions
        Explicit physical profile and widths. Both axes use the same metric.

    Returns
    -------
    numpy.ndarray
        Eager float64 dimensionless weights in the broadcast shape, in [0, 1].
        Values depend on hypot(k_read, k_phase), independently of grid size,
        point order, observed signal amplitude and FOV.

    Raises
    ------
    TypeError
        If options is not SpatialFilterOptions or coordinates are complex.
    ValueError
        If coordinates are not finite real numbers or cannot broadcast.

    Notes
    -----
    Pure and deterministic, without caching or input mutation. The stable
    logistic implementation avoids overflow for a narrow Fermi transition.

    Examples
    --------
    >>> kspace_filter_weights(np.array([0, 0.1]), np.array([0, 0]),
    ...                       SpatialFilterOptions(sigma_per_mm=0.1))
    array([1.        , 0.60653066])
    """
    if not isinstance(options, SpatialFilterOptions):
        raise TypeError("Spatial filter options must be SpatialFilterOptions.")
    if np.iscomplexobj(k_read_per_mm) or np.iscomplexobj(k_phase_per_mm):
        raise TypeError("Spatial filter k-space coordinates must be real.")
    read, phase = np.broadcast_arrays(
        np.asarray(k_read_per_mm, dtype=float), np.asarray(k_phase_per_mm, dtype=float)
    )
    if not np.all(np.isfinite(read)) or not np.all(np.isfinite(phase)):
        raise ValueError("Spatial filter k-space coordinates must be finite.")
    radius = np.hypot(read, phase)
    if options.kind == "gaussian":
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(-0.5 * (radius / options.sigma_per_mm) ** 2)
    with np.errstate(over="ignore"):
        return expit((options.radius_per_mm - radius) / options.transition_per_mm)


def _coordinate_line(data, dim):
    coord = data.coords.get(dim)
    if coord is None or dim not in coord.dims:
        return None, "", None
    line = coord.isel({d: 0 for d in coord.dims if d != dim})
    if np.iscomplexobj(line.values):
        raise ValueError(f"Spatial filter {dim} coordinates must be real.")
    values = np.asarray(line.values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Spatial filter {dim} coordinates must be finite.")
    expanded = line.broadcast_like(coord).transpose(*coord.dims)
    if not np.allclose(coord.values, expanded.values, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"Spatial filter {dim} coordinates must form a separable axis."
        )
    units = (
        str(coord.attrs.get("units") or coord.attrs.get("unit") or "").strip().lower()
    )
    return values, units, coord.attrs.get("resolution_mm")


def _spacing_mm(data, dim, explicit=None):
    values, units, resolution = _coordinate_line(data, dim)
    if data.sizes[dim] == 1 and explicit is None:
        return None
    factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    spacing = None
    if units in factors and values is not None and values.size > 1:
        differences = np.diff(values) * factors[units]
        if differences[0] == 0 or not np.allclose(
            differences, differences[0], rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"Spatial filter {dim} image coordinates require a uniform nonzero spacing."
            )
        spacing = abs(float(differences[0]))
    elif resolution is not None:
        spacing = _positive(resolution, f"{dim} resolution_mm")
    if explicit is not None:
        explicit = _positive(explicit, f"{dim} spacing_mm")
        if spacing is not None and not np.isclose(
            spacing, explicit, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"Spatial filter {dim} spacing_mm conflicts with coordinate spacing."
            )
        spacing = explicit
    if spacing is None:
        if data.sizes[dim] == 1:
            return None
        raise ValueError(
            f"Spatial filter cannot resolve {dim} spacing in mm; supply physical coordinates or spatial_spacing_mm."
        )
    return spacing


def _frequency_axis(data, dim, domain, spacing):
    if domain == "kspace":
        values, units, _ = _coordinate_line(data, dim)
        factors = {
            "1/mm": 1.0,
            "mm^-1": 1.0,
            "mm**-1": 1.0,
            "cycles/mm": 1.0,
            "1/cm": 0.1,
            "1/m": 0.001,
        }
        if units in factors and values is not None:
            if np.unique(values).size != values.size:
                raise ValueError(
                    f"Spatial filter {dim} k-space coordinates must be unique."
                )
            if spacing is not None:
                raise ValueError(
                    f"Spatial filter {dim} already has physical k-space coordinates; omit spatial_spacing_mm."
                )
            return values * factors[units], None
        if spacing is None:
            raise ValueError(
                f"Spatial filter {dim} requires k-space coordinates in 1/mm or explicit spatial_spacing_mm."
            )
        spacing = _positive(spacing, f"{dim} spacing_mm")
    else:
        spacing = _spacing_mm(data, dim, spacing)
    if spacing is None:
        return np.zeros(data.sizes[dim]), None
    return np.fft.fftshift(np.fft.fftfreq(data.sizes[dim], d=spacing)), spacing


def _centred_kspace(values, axes, scanner_grid):
    """Return the centred k-space of image values along ``axes``.

    Bruker's CSI scanner grid carries the centred-frequency modulation in its
    complex voxel phases already, so it gets no output shift. Every other image
    uses the reconstruction's centred transform, whose zero frequency is at
    index ``n // 2`` for even and odd lengths.
    """
    if scanner_grid:
        return np.fft.fftn(np.fft.ifftshift(values, axes=axes), axes=axes)
    return centered_fft(values, axes)


def _spatial_filter_provenance(options, domain, spacing, scanner_grid=False):
    return json.dumps(
        {
            "options": asdict(options),
            "axes": ["read_idx", "phase_idx"],
            "domain": domain,
            "spacing_mm": list(spacing),
            "frequency_unit": "1/mm",
            "scanner_grid": scanner_grid,
        },
        sort_keys=True,
    )


def apply_spatial_filter(
    data: xr.DataArray,
    options: SpatialFilterOptions,
    *,
    spatial_spacing_mm: tuple[float, float] | None = None,
    domain: Literal["spatial", "kspace"] | None = None,
) -> xr.DataArray:
    """Filter Read and Phase using a radial physical Gaussian or Fermi window.

    Parameters
    ----------
    data : xarray.DataArray
        Numeric labelled signal containing read_idx and phase_idx in any order.
        Other dimensions (including slice, echo, FID, repetition and channel)
        are independent planes and never transformed or combined. Image axes
        require separable, uniform mm/cm/m coordinates or resolution_mm; k-space
        axes require finite unique coordinates in 1/mm, cycles/mm, mm^-1,
        mm**-1, 1/cm or 1/m. Nonuniform k-space coordinates are accepted directly.
        Complex quadratures are retained; filtering magnitude-only images cannot
        recover their previously discarded phase.
    options : SpatialFilterOptions
        Physical widths in cycles/mm. Filtering is disabled by not calling this
        function; a width is never inferred from the largest acquired radius.
    spatial_spacing_mm : tuple of float or None, optional
        Explicit positive (Read, Phase) image sampling distances in mm, independent
        of input dimension order. Supplies missing calibration and must agree
        with known image-coordinate spacing. For legacy k-space arrays lacking
        reciprocal coordinates, explicitly defines their centred Fourier axes.
        Omit when k-space coordinates already carry reciprocal units.
    domain : {'spatial', 'kspace'} or None, optional
        Explicit input/output domain. None reads data.attrs['domain'], defaulting
        to 'spatial' when absent. Unknown declarations are rejected. For spatial
        input, centred forward/inverse FFTs use NumPy backward normalization;
        csi_spatial_reconstruction='scanner_grid' retains the existing scanner
        modulation convention without an extra forward output shift.

    Returns
    -------
    xarray.DataArray
        New signal in the same domain, shape, dimension order and coordinate
        orientation, retaining all coordinate/unit attributes. Calculation uses
        float64/complex128; floating/complex input storage dtype is restored,
        while integer inputs yield float64. Real images yield real images,
        except phase-modulated scanner grids, whose inverse requires complex128.
        Existing attributes remain, with JSON spatial_filter provenance recording
        widths, axes, domain and calibration. Signal units are unchanged.

    Raises
    ------
    TypeError
        For nonnumeric/non-DataArray input or incorrect options type.
    ValueError
        For missing/empty Read or Phase, unknown domain, invalid widths, absent
        or conflicting calibration, nonuniform image spacing, inseparable axes
        or ambiguous k-space coordinates. Supply explicit spacing or a calibrated
        grid rather than allowing an index spacing to become a physical distance.

    Notes
    -----
    Eager, deterministic and non-mutating; no reconstruction, file access,
    widgets, data-dependent centre estimation or caching occurs. Gaussian DC
    gain is one; the unnormalized Fermi logistic may attenuate DC. Nonfinite
    signal values retain normal multiplication/FFT propagation. Uniform-grid
    validation uses rtol=1e-6 and atol=1e-9 in mm. A singleton image axis has
    only k=0 and requires no invented spacing. Reapplying intentionally filters
    again; callers should pass their unchanged original for interactive previews.

    Examples
    --------
    >>> options = SpatialFilterOptions(sigma_per_mm=0.1)
    >>> filtered = apply_spatial_filter(image, options,
    ...                                 spatial_spacing_mm=(2., 3.))  # doctest: +SKIP
    >>> fermi = SpatialFilterOptions(kind="fermi", radius_per_mm=0.15,
    ...                              transition_per_mm=0.02)
    >>> filtered_kspace = apply_spatial_filter(kspace, fermi)  # doctest: +SKIP
    """
    if not isinstance(data, xr.DataArray) or not np.issubdtype(data.dtype, np.number):
        raise TypeError("apply_spatial_filter requires a numeric labelled DataArray.")
    if not isinstance(options, SpatialFilterOptions):
        raise TypeError("Spatial filter options must be SpatialFilterOptions.")
    dims = ("read_idx", "phase_idx")
    if any(dim not in data.dims or data.sizes[dim] == 0 for dim in dims):
        raise ValueError(
            "apply_spatial_filter requires nonempty read_idx and phase_idx dimensions."
        )
    domain = domain if domain is not None else data.attrs.get("domain", "spatial")
    if domain not in {"spatial", "kspace"}:
        raise ValueError(
            "Spatial filter domain must be 'spatial' or 'kspace'; declare it explicitly."
        )
    if spatial_spacing_mm is not None and np.shape(spatial_spacing_mm) != (2,):
        raise ValueError(
            "spatial_spacing_mm must contain Read and Phase spacing in mm."
        )
    spacing = spatial_spacing_mm if spatial_spacing_mm is not None else (None, None)
    read, dx = _frequency_axis(data, dims[0], domain, spacing[0])
    phase, dy = _frequency_axis(data, dims[1], domain, spacing[1])
    window = kspace_filter_weights(read[:, None], phase[None, :], options)
    window = (
        xr.DataArray(window, dims=dims)
        .broadcast_like(data)
        .transpose(*data.dims)
        .values
    )
    axes = tuple(data.get_axis_num(dim) for dim in dims)
    values = np.asarray(
        data.values, dtype=np.complex128 if np.iscomplexobj(data.data) else np.float64
    )
    scanner_grid = data.attrs.get("csi_spatial_reconstruction") == "scanner_grid"
    if domain == "spatial":
        values = _centred_kspace(values, axes, scanner_grid)
    values = values * window
    if domain == "spatial":
        if scanner_grid:
            values = np.fft.fftshift(np.fft.ifftn(values, axes=axes), axes=axes)
        else:
            values = centered_ifft(values, axes)
        if not np.iscomplexobj(data.data) and not scanner_grid:
            values = values.real
    dtype = data.dtype if np.issubdtype(data.dtype, np.inexact) else np.float64
    if np.iscomplexobj(values) and not np.issubdtype(dtype, np.complexfloating):
        dtype = np.complex128 if scanner_grid else dtype
    result = data.copy(deep=True, data=values.astype(dtype))
    result.attrs["spatial_filter"] = _spatial_filter_provenance(
        options, domain, (dx, dy), scanner_grid
    )
    return result


def spatial_to_kspace(
    data: xr.DataArray,
    *,
    spatial_spacing_mm: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Return the centred spatial k-space of an image along Read and Phase.

    Uses the forward transform of :func:`apply_spatial_filter`, so a filter
    window drawn over the result lies where that function applies it.

    Parameters
    ----------
    data : xarray.DataArray
        Numeric labelled image-space signal with ``read_idx`` and
        ``phase_idx`` (any order) and ``attrs["domain"]`` ``"spatial"`` or
        missing. Other dimensions (slice, FID or spectrum, repetition,
        channel) are independent planes and stay unchanged; the slice
        direction is not transformed.
    spatial_spacing_mm : tuple of float or None, optional
        Read and phase voxel spacing in mm when the image axes carry no
        mm/cm/m coordinates or ``resolution_mm``; it must agree with
        calibrated axes.

    Returns
    -------
    xarray.DataArray
        New complex128 array with the same dimensions in the same order.
        ``read_idx`` and ``phase_idx`` carry centred spatial frequencies in
        1/mm (cycles/mm, ascending, k = 0 at index ``n // 2``);
        ``attrs["domain"]`` is ``"kspace"``. Other attributes and
        coordinates are kept, except coordinates that depend on read or
        phase position (for example scanner x, y, z), which have no meaning
        in k-space and are dropped. Signal units are unchanged (NumPy
        unnormalized forward FFT).

    Raises
    ------
    TypeError
        For non-numeric or non-DataArray input.
    ValueError
        If ``read_idx`` or ``phase_idx`` is missing, the data are already in
        k-space, or a spacing cannot be resolved.

    Notes
    -----
    Eager, deterministic and non-mutating.

    Examples
    --------
    >>> kspace = spatial_to_kspace(image)  # doctest: +SKIP
    >>> kspace.read_idx.attrs["units"]  # doctest: +SKIP
    '1/mm'
    """
    if not isinstance(data, xr.DataArray) or not np.issubdtype(data.dtype, np.number):
        raise TypeError("spatial_to_kspace requires a numeric labelled DataArray.")
    dims = ("read_idx", "phase_idx")
    if any(dim not in data.dims or data.sizes[dim] == 0 for dim in dims):
        raise ValueError(
            "spatial_to_kspace requires nonempty read_idx and phase_idx dimensions."
        )
    if data.attrs.get("domain", "spatial") != "spatial":
        raise ValueError(
            "spatial_to_kspace: the data are declared as "
            f"{data.attrs.get('domain')!r}; it transforms image-space data only."
        )
    if spatial_spacing_mm is not None and np.shape(spatial_spacing_mm) != (2,):
        raise ValueError(
            "spatial_to_kspace: spatial_spacing_mm must contain Read and Phase "
            "spacing in mm."
        )
    spacing = spatial_spacing_mm if spatial_spacing_mm is not None else (None, None)
    axes_k = [
        _frequency_axis(data, dim, "spatial", explicit)[0]
        for dim, explicit in zip(dims, spacing, strict=True)
    ]
    axes = tuple(data.get_axis_num(dim) for dim in dims)
    scanner_grid = data.attrs.get("csi_spatial_reconstruction") == "scanner_grid"
    values = _centred_kspace(
        np.asarray(data.values, dtype=np.complex128), axes, scanner_grid
    )
    dependent = [
        name
        for name, coord in data.coords.items()
        if name not in data.dims and set(coord.dims) & set(dims)
    ]
    result = data.drop_vars(dependent).copy(data=values)
    result = result.assign_coords(
        {
            dim: xr.DataArray(
                axis, dims=(dim,), attrs={"units": "1/mm", "long_name": f"k {dim}"}
            )
            for dim, axis in zip(dims, axes_k, strict=True)
        }
    )
    result.attrs = {**data.attrs, "domain": "kspace"}
    return result
