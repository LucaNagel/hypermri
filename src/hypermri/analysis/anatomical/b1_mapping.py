"""Interactive B1 mapping utilities for anatomical calibration experiments."""

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike as _ArrayLike
from numpy.typing import NDArray as _NDArray

from hypermri.analysis.fitting.api import fit
from hypermri.analysis.fitting.fit_maps import (
    fit_quantitative_maps as fit_quantitative_maps,
)
from hypermri.analysis.quantitative.b1_calibration import (
    _DEFAULT_PERIOD_SAMPLES,
    _DEFAULT_REFINEMENT_ITERATIONS,
    _DEFAULT_ZOOM_FACTOR,
    _resolve_b1_calibration_window,
)
from hypermri.analysis.quantitative.b1_calibration import (
    fit_b1_calibration as _fit_b1_calibration,
)
from hypermri.utils.system.logging import debug_log, init_default_logger

if TYPE_CHECKING:
    from hypermri.loading.experiment import MRIExperiment

DB_MSG = True  # Change this to False to hide detailed error messages


DEBUG_FILE = "/tmp/b1_mapping_debug.log"
DEBUG_LOG = False


def debug_log_b1_mapping(
    message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG, numpy_full_print=True
):
    """
    Debug logging system for B1 mapping operations.

    Parameters
    ----------
    message : str, optional
        The debug message to log. Default is empty string.
    file_name : str, optional
        Path to the log file. Default is DEBUG_FILE constant.
    write_to_file : bool, optional
        Whether to write the log to a file. Default is DEBUG_LOG constant.
    numpy_full_print : bool, optional
        Whether to print full numpy arrays without truncation. Default is True.

    Returns
    -------
    None
    """
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
        numpy_full_print=numpy_full_print,
    )


def fit_single_voxel(
    values: _ArrayLike,
    refpower_W: _ArrayLike,
    plot: bool = False,
    initial_guess_cent: float | None = None,
    initial_guess_extent: float | None = None,
    npoints: int = _DEFAULT_PERIOD_SAMPLES,
    niter: int = _DEFAULT_REFINEMENT_ITERATIONS,
    zoom_factor: float = _DEFAULT_ZOOM_FACTOR,
    stop_index_extra: int = 0,
) -> tuple[
    complex,
    float,
    _NDArray[np.complex128],
    _NDArray[np.float64],
    _NDArray[np.float64],
]:
    """Fit a single-voxel B1 calibration and optionally display its search.

    Delegates once to
    :func:`hypermri.analysis.quantitative.b1_calibration.fit_b1_calibration`,
    retaining the historical iterative complex sinusoidal model, defaults and
    five-element result. The eager fit is deterministic and preserves both
    inputs. When requested, a Matplotlib figure displays the final fitted signal
    and every period-search trajectory after the numerical calculation.

    Parameters
    ----------
    values : array_like
        Finite real or complex samples, shape ``(n_powers,)``, in arbitrary signal
        units, with at least three entries, ordered along RF power.
    refpower_W : array_like
        Finite, nonnegative powers in watts, shape ``(n_powers,)``, paired with
        ``values``. Samples retain their supplied order.
    plot : bool, optional
        Display the fitted signal and residual-search histories. No plotting
        occurs when False; True creates a Matplotlib figure.
    initial_guess_cent : float or None, optional
        Positive initial period-search center in sqrt(W). ``None`` selects
        3.0 sqrt(W); values below 0.25 are clamped to 0.25.
    initial_guess_extent : float or None, optional
        Positive initial search half-width in sqrt(W). ``None`` selects
        3.0 sqrt(W); values below 1e-6 are clamped to 1e-6.
    npoints : int, optional
        Number of period candidates per iteration; at least two.
    niter : int, optional
        Number of period-refinement iterations; at least one.
    zoom_factor : float, optional
        Finite factor greater than one by which the search half-width shrinks.
    stop_index_extra : int, optional
        Retained historical integer setting, which does not change this
        full-signal single-voxel fit. The original calculation accepted it
        without applying an automatic cutoff.

    Returns
    -------
    amplitude : complex
        Best complex amplitude in the input signal units.
    period : float
        Best period parameter in sqrt(W). NaN for an all-zero signal.
    fit_data : numpy.ndarray
        Fitted complex128 signal, shape ``(n_powers,)``. All zeros for an all-zero
        signal.
    local_r2 : numpy.ndarray
        Float64 residual sums of squares in squared signal units, shape
        ``(niter, npoints)``; the historical name denotes RSS, not R-squared.
        All NaN for an all-zero signal.
    local_per : numpy.ndarray
        Float64 candidate periods in sqrt(W), shape ``(niter, npoints)``, in
        iteration and candidate order. All NaN for an all-zero signal.

    Raises
    ------
    ValueError
        If arrays or search options violate the documented fitting contract.

    Examples
    --------
    >>> power = np.linspace(0.0, 16.0, 17)
    >>> signal = (2.0 + 0.5j) * np.sin(np.pi / 4.0 * np.sqrt(power))
    >>> amplitude, period, curve, rss, periods = fit_single_voxel(signal, power)
    >>> rss.shape
    (10, 1000)
    """
    result = _fit_b1_calibration(
        values=values,
        refpower_W=refpower_W,
        initial_guess_cent=initial_guess_cent,
        initial_guess_extent=initial_guess_extent,
        npoints=npoints,
        niter=niter,
        zoom_factor=zoom_factor,
        stop_index_extra=stop_index_extra,
    )
    if plot:
        _plot_b1_calibration(values, refpower_W, result)
    return result


def _plot_b1_calibration(data, refpower_W, result):
    """Render fitted samples and stored residuals without repeating the fit."""
    amplitude, period, fit_data, rss, periods = result
    sqrt_power = np.sqrt(np.asarray(refpower_W, dtype=float))
    signal = np.asarray(data)
    fig, (ax_signal, ax_search) = plt.subplots(1, 2, figsize=(14, 5))
    for component, values, curve, marker in (
        ("real", np.real(signal), np.real(fit_data), "."),
        ("imag", np.imag(signal), np.imag(fit_data), "x"),
        ("abs", np.abs(signal), np.abs(fit_data), "o"),
    ):
        ax_signal.plot(sqrt_power, values, marker, label=f"Data ({component})")
        ax_signal.plot(sqrt_power, curve, "--", label=f"Fit ({component})")
    ax_signal.set_xlabel("sqrt(RF power) [sqrt(W)]")
    ax_signal.set_ylabel("Signal")
    ax_signal.set_title(f"B1 calibration: |amplitude| = {abs(amplitude):.4g}")
    for iteration, (candidates, residuals) in enumerate(zip(periods, rss, strict=True)):
        ax_search.plot(candidates, residuals, "o-", label=f"Iteration {iteration + 1}")
    if np.isfinite(period):
        ax_search.axvline(period, color="red", linestyle="--", label="Best period")
    ax_search.set_xlabel("Period [sqrt(W)]")
    ax_search.set_ylabel("Residual sum of squares")
    ax_search.set_title("Period optimization")
    for axis in (ax_signal, ax_search):
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    plt.show()


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


def fit_b1_surface_coil(
    data: xr.DataArray,
    *,
    fit_params: dict | None = None,
    dynamic_dim: str | None = None,
    sort_by_power: bool = True,
) -> xr.Dataset:
    """
    Fit voxel-wise B1 maps for surface-coil calibrations.

    The generic fitting API prepares the B1 options and runs the existing period
    sweep solver over all non-RF-power dimensions. Input values and metadata are
    preserved; when requested, a one-dimensional RF-power coordinate is sorted
    on a derived view before fitting.

    Parameters
    ----------
    data : xarray.DataArray
        Complex-valued FLASH calibration samples. The array must have an RF-power
        dimension with a one-dimensional coordinate in the units expected by the
        B1 fitting options, normally watts; other dimensions are treated as map
        dimensions and retain their labels and order in the result.
    fit_params : dict or None, optional
        B1 fitting options, including the required pulse calibration values:
        ``fa_deg`` in degrees, ``trf_ms`` in milliseconds and dimensionless
        ``sint``. Execution settings are optional. The mapping is copied before
        adding the resolved RF-power dimension.
    dynamic_dim : str or None, optional
        RF-power dimension name. If omitted, the function searches for
        ``rfpower_idx``, ``power_idx``, ``rf_power_idx`` and ``rfpower`` in that
        order.
    sort_by_power : bool, optional
        Sort samples by a one-dimensional RF-power coordinate before fitting.
        Multidimensional coordinates are left in their existing order.

    Returns
    -------
    xarray.Dataset
        Fitted amplitude, period, reference power at 90 degrees in watts, residual
        statistics and diagnostic fields, with the spatial labels and order of the
        input array.

    Raises
    ------
    ValueError
        If the RF-power dimension cannot be resolved or has no coordinate values.

    Examples
    --------
    Fit a labelled complex calibration with powers in watts:

    >>> powers = np.linspace(0.0, 16.0, 17)
    >>> samples = (2.0 + 0.5j) * np.sin(np.pi / 4.0 * np.sqrt(powers))
    >>> calibration = xr.DataArray(
    ...     samples[:, None],
    ...     dims=("rfpower_idx", "voxel_idx"),
    ...     coords={"rfpower_idx": powers, "voxel_idx": [0]},
    ... )
    >>> maps = fit_b1_surface_coil(
    ...     calibration,
    ...     fit_params={
    ...         "fa_deg": 90.0, "trf_ms": 1.0, "sint": 1.0,
    ...         "use_multiprocessing": False,
    ...     },
    ... )
    """
    if fit_params is None:
        fit_params = {}
    fit_params_local = dict(fit_params)  # shallow copy to avoid in-place mutation

    if dynamic_dim is not None:
        fit_params_local["dynamic_dim"] = dynamic_dim

    candidate_dims = fit_params_local.get(
        "dynamic_dim_candidates",
        ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
    )
    dynamic_dim_resolved = fit_params_local.get("dynamic_dim")
    if dynamic_dim_resolved is None:
        dynamic_dim_resolved = next(
            (dim for dim in candidate_dims if dim in data.dims), None
        )
        if dynamic_dim_resolved is None:
            raise ValueError(
                "Unable to determine RF power dimension. Supply 'dynamic_dim' in fit_params "
                "or pass it explicitly to fit_b1_surface_coil()."
            )
        fit_params_local["dynamic_dim"] = dynamic_dim_resolved

    if dynamic_dim_resolved not in data.dims:
        raise ValueError(
            f"Resolved dynamic dimension '{dynamic_dim_resolved}' not found in data dims {data.dims}."
        )

    if dynamic_dim_resolved not in data.coords:
        raise ValueError(
            f"Input data must provide coordinate values for '{dynamic_dim_resolved}'."
        )

    working_data = data
    if sort_by_power:
        coord = data.coords[dynamic_dim_resolved]
        if coord.ndim == 1:
            working_data = data.sortby(dynamic_dim_resolved)
        else:
            logger.warning(
                "Skipping sort along '%s' because its coordinate spans multiple dimensions.",
                dynamic_dim_resolved,
            )

    logger.info(
        "Starting B1 fitting on data with shape %s (dynamic dim '%s', length %d).",
        working_data.shape,
        dynamic_dim_resolved,
        working_data.sizes[dynamic_dim_resolved],
    )

    result_ds = fit(
        data=working_data,
        model="b1",
        fit_params=fit_params_local,
    )

    logger.info("B1 fitting finished. Result variables: %s", list(result_ds.data_vars))
    return result_ds


def find_nth_highest_signal_voxel(data, n: int = 1, plot: bool = False):
    """
    Find the voxel with the nth highest signal after averaging along repetitions.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_powers, 1, 1, nx, ny, n_repetitions, 1) or (nx, ny).
        For 7D data, it represents multiple RF powers, spatial dimensions (nx, ny),
        repetitions, and singletons. For 2D data, it represents a single image (nx, ny).
    n : int, optional
        The rank of the voxel to return (1 for highest, 2 for second highest, etc.).
        Default is 1 (highest signal voxel).
    plot : bool, optional
        Whether to plot the signal and the selected voxel. Default is False.

    Returns
    -------
    tuple
        (x, y) coordinates of the nth highest signal voxel.

    Notes
    -----
    This function handles both 7D and 2D input data. It averages the signal over repetitions
    and then finds the voxel with the nth highest total signal magnitude across all powers.
    """
    # Average along repetitions (axis 5) and remove singleton dimensions
    if np.ndim(data) == 7:
        averaged_data = np.squeeze(np.nanmean(data, axis=(1, 2, 5, 6)))
    else:
        averaged_data = data

    # Sum signal magnitude across all powers
    total_signal = np.nansum(np.abs(averaged_data), axis=0)

    # Flatten the array and get the indices that would sort it in descending
    # order
    flat_indices = np.argsort(total_signal.ravel())[::-1]

    # Get the nth highest index (subtracting 1 because n starts from 1)
    nth_highest_flat_index = flat_indices[n - 1]

    # Convert the flat index back to 2D coordinates
    max_index = np.unravel_index(nth_highest_flat_index, total_signal.shape)

    if plot:
        plt.figure()
        plt.imshow(np.squeeze(np.abs(total_signal)), cmap="gray")
        plt.scatter(
            max_index[1],
            max_index[0],
            c="r",
            s=100,
            marker="x",
            label=f"{n}th highest signal voxel",
        )
        plt.legend()
        plt.show()

    return max_index


def interactive_b1_fit_viewer(
    b1_dataset: xr.Dataset,
    data: xr.DataArray,
    *,
    dynamic_dim: str = "rfpower_idx",
    static_indexers: dict | None = None,
    abs_amplitude: bool = True,
    rfpulse_duration_ms: float = 1.0,
    rfpulse_sint: float = 1.0,
):
    """
    Launch an interactive viewer for inspecting B1 fits.

    Clicking on the amplitude image plots the raw signal, fitted curve,
    fit window (center/extent), and stop index for the selected voxel.

    Parameters
    ----------
    b1_dataset : xr.Dataset
        Output from the fitting API containing at least the DataArrays
        ``amplitude``, ``period``, ``stop_index``, ``fit_window_center``,
        and ``fit_window_extent``.
    data : xr.DataArray
        Original measurement data passed to the fitting API.
    dynamic_dim : str, optional
        Name of the RF power dimension. Default is ``"rfpower_idx"``.
    static_indexers : dict, optional
        Dimension selections (passed to ``isel``) for reducing additional
        dimensions such as slices or coils before plotting.
    abs_amplitude : bool, optional
        When True (default), the displayed image is ``|amplitude|``.
        Set to False to view the real part instead.
    rfpulse_duration_ms : float, optional
        Duration in milliseconds used to normalize the displayed 90-degree
        reference power. Defaults to 1 ms.
    rfpulse_sint : float, optional
        Pulse-shape integral used to normalize the displayed 90-degree reference
        power. Defaults to 1.

    Returns
    -------
    None
        Displays a Matplotlib figure with a fit inspection panel. The function
        registers a click callback on that figure and does not modify its inputs.

    Raises
    ------
    ValueError
        If required fit variables are missing, the RF-power dimension is absent,
        or the selected data do not reduce to exactly two spatial dimensions.
    """
    import matplotlib.pyplot as plt

    if static_indexers is None:
        static_indexers = {}

    if "amplitude" not in b1_dataset:
        raise ValueError("b1_dataset must contain an 'amplitude' DataArray.")
    required_fields = {
        "period",
        "stop_index",
    }
    missing = required_fields.difference(b1_dataset.data_vars)
    if missing:
        raise ValueError(f"b1_dataset is missing required fields: {sorted(missing)}")

    if dynamic_dim not in data.dims:
        raise ValueError(f"dynamic_dim '{dynamic_dim}' not found in data.")

    # Reduce extra dimensions up front for both dataset and input data
    b1_ds_view = b1_dataset.isel(**static_indexers).squeeze(drop=True)
    input_view = data.isel(**static_indexers).squeeze(drop=True)

    spatial_dims = [dim for dim in b1_ds_view["amplitude"].dims if dim != dynamic_dim]
    if dynamic_dim in spatial_dims:
        spatial_dims.remove(dynamic_dim)
    if len(spatial_dims) != 2:
        raise ValueError(
            "Viewer requires exactly two non-dynamic spatial dimensions after applying static_indexers."
        )

    amp_da = b1_ds_view["amplitude"]
    period_da = b1_ds_view["period"]
    stop_da = b1_ds_view["stop_index"]
    refpower_90deg_W_da = b1_ds_view["refpower_90deg_W"]

    # cent_da = b1_ds_view["fit_window_center"]
    # ext_da = b1_ds_view["fit_window_extent"]

    if abs_amplitude:
        display_data = np.abs(amp_da.values)
    else:
        display_data = np.real(amp_da.values)

    if dynamic_dim not in input_view.coords:
        raise ValueError(f"data is missing coordinate values for '{dynamic_dim}'.")
    # sqrt_refpower = np.sqrt(np.asarray(input_view.coords[dynamic_dim].values, dtype=float))
    sqrt_refpower = np.asarray(input_view.coords[dynamic_dim].values, dtype=float)

    # Prepare figure with image and plot panels
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(8, 5))
    im = ax_img.imshow(
        display_data,
        origin="lower",
        cmap="magma",
    )
    ax_img.set_title("B1 amplitude" + (" (abs)" if abs_amplitude else " (real)"))
    ax_img.set_xlabel(spatial_dims[1])
    ax_img.set_ylabel(spatial_dims[0])
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    marker = ax_img.scatter(
        [], [], s=60, facecolors="none", edgecolors="cyan", linewidths=1.5, marker="x"
    )

    ax_plot.set_xlabel("sqrt(refpower) [W$^{1/2}$]")
    ax_plot.set_ylabel("Signal")
    ax_plot.set_title("Click a voxel to inspect fit")

    def update_plot(col_idx: int, row_idx: int):
        if (
            row_idx < 0
            or row_idx >= display_data.shape[0]
            or col_idx < 0
            or col_idx >= display_data.shape[1]
        ):
            return

        marker.set_offsets([[col_idx, row_idx]])

        isel_dict = {spatial_dims[0]: row_idx, spatial_dims[1]: col_idx}
        signal = input_view.isel(isel_dict).values
        amplitude = amp_da.isel(isel_dict).item()
        period = period_da.isel(isel_dict).item()
        stop_index_val = stop_da.isel(isel_dict).item()
        refpower_90deg_W_val = refpower_90deg_W_da.isel(isel_dict).item()
        refpower_90deg_W_val = (
            refpower_90deg_W_val / (rfpulse_duration_ms * rfpulse_sint) ** 2
        )

        # center_val = cent_da.isel(isel_dict).item()
        # extent_val = ext_da.isel(isel_dict).item()

        ax_plot.clear()
        ax_plot.set_xlabel("sqrt(RF Power) [W$^{1/2}$]")
        ax_plot.set_ylabel("Signal")

        ax_plot.plot(
            sqrt_refpower,
            np.real(signal),
            "o-",
            label="Signal (real)",
            linewidth=0.25,
            color="k",
        )
        ax_plot.plot(
            sqrt_refpower,
            np.imag(signal),
            "x-",
            label="Signal (imag)",
            alpha=0.7,
            linewidth=0.25,
            color="k",
        )
        ax_plot.plot(
            sqrt_refpower,
            np.abs(signal),
            ">-",
            label="Signal (abs)",
            alpha=0.7,
            linewidth=1.5,
            color="k",
        )

        sqrt_refpower_high = np.linspace(sqrt_refpower.min(), sqrt_refpower.max(), 500)

        if np.isfinite(period) and period > 0:
            fit_curve = amplitude * np.sin(np.pi / period * np.sqrt(sqrt_refpower_high))
            ax_plot.plot(
                sqrt_refpower_high,
                np.real(fit_curve),
                "o-",
                label="Fit (real)",
                linewidth=0.25,
                color="r",
            )
            ax_plot.plot(
                sqrt_refpower_high,
                np.imag(fit_curve),
                "x-",
                label="Fit (imag)",
                alpha=0.7,
                linewidth=0.25,
                color="r",
            )
            ax_plot.plot(
                sqrt_refpower_high,
                np.abs(fit_curve),
                ">-",
                label="Fit (abs)",
                alpha=0.77,
                linewidth=1.5,
                color="r",
            )

        # if np.isfinite(center_val) and np.isfinite(extent_val):
        # lower = max(center_val - extent_val, sqrt_refpower.min())
        # upper = min(center_val + extent_val, sqrt_refpower.max())
        # ax_plot.axvspan(lower, upper, color="orange", alpha=0.1, label="Fit window")
        # ax_plot.axvline(center_val, color="orange", linestyle="--", linewidth=1)

        ax_plot.axvline(refpower_90deg_W_val)

        if np.isfinite(stop_index_val):
            stop_idx = int(stop_index_val)
            stop_idx = np.clip(stop_idx, 0, sqrt_refpower.size - 1)
            ax_plot.axvline(
                sqrt_refpower[stop_idx],
                color="red",
                linestyle=":",
                linewidth=1.2,
                label=f"Stop index ({stop_idx})",
            )

        ax_plot.legend(loc="upper right")
        ax_plot.set_title(
            f"{spatial_dims[0]}={row_idx}, {spatial_dims[1]}={col_idx}, stop_idx={stop_index_val:.1f}\nReference Power={refpower_90deg_W_val}W"
        )
        plt.tight_layout()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        update_plot(col, row)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    # return fig


def refpow_parameter_adjustment(
    refpower_W: float,
    scan: Optional["MRIExperiment"] = None,
    trf_ms: float = 1,
    sint: float = 1,
    fa_deg: float = 90,
):
    """Scale reference power according to the applied pulse parameters.

    Parameters
    ----------
    refpower_W : float or array_like
        Reference power in Watts measured during calibration.
    scan : MRIExperiment, optional
        Sequence object that exposes ``rfpulse_duration_ms``, ``flipangle_deg``,
        and ``rfpulse_sint`` attributes. When provided the remaining parameters
        are read from the object.
    trf_ms : float, optional
        RF pulse duration in milliseconds when ``scan`` is ``None``.
    sint : float, optional
        Integrated RF pulse shape factor.
    fa_deg : float, optional
        Reference flip angle in degrees.

    Returns
    -------
    numpy.ndarray or float
        Adjusted reference power that compensates for the actual pulse setup.
    """
    if scan is None:
        pass
    else:
        trf_ms = scan.rfpulse_duration_ms
        fa_deg = scan.flipangle_deg
        sint = scan.rfpulse_sint

    refpower_parameter_adjusted_W = refpower_W * (90 * trf_ms * sint / fa_deg) ** 2
    return refpower_parameter_adjusted_W


def refpow_from_sin_fit(periods: float = 0):
    """Convert sine-fit periods into reference power estimates. This assumes that a blockpulse
    of 1ms duration and integration factor of 1 was used during calibration.

    If that is not the case, the resulting reference power values have to be adjusted
    using ``refpow_parameter_adjustment``.

    Fitted curve: S = A * sin(π / period * sqrt(refpower))
    Reference power for 90 degree flip angle: refpower_90deg = (0.5 * period) ** 2

    Parameters
    ----------
    periods : array_like or float
        Period values obtained from the sinusoidal calibration curve.

    Returns
    -------
    numpy.ndarray or float
        Reference power values in Watts computed as ``(0.5 * periods)**2``.
    """
    refpow = (0.5 * periods) ** 2
    return refpow


def plot_refpow_map_overlay(
    anatomical_image,
    refpow_map,
    anatomical_extent,
    b1_extent,
    fa_deg=90,
    trf_ms=1,
    sint=1,
    scan: "MRIExperiment" = None,
    rfpow_W=None,
    plot_params=None,
):
    """
    Plot an interactive overlay of B1 map and resulting flip angle map onto an anatomical image using ipywidgets.

    Parameters
    ----------
    anatomical_image : ndarray
        Array containing the anatomical image data. Expected shape: (1, slices, x, y, 1, 1)
    refpow_map : ndarray
        Array containing the B1 map data (reference power map). Expected shape: (1, 1, x, y, 1, 1)
    anatomical_extent : tuple
        (left, right, bottom, top) values for the anatomical image extent.
    b1_extent : tuple
        (left, right, bottom, top) values for the B1 map extent.
    fa_deg : float, optional
        Initial defined flip angle for reference power. Default is 90.
    trf_ms : float, optional
        Initial pulse duration in milliseconds. Default is 1.
    sint : float, optional
        Initial integration factor. Default is 1.
    scan : MRIExperiment, optional
        Data object containing method parameters. Default is None.
    rfpow_W : float, optional
        Initial RF power in Watts. Default is 1.
    plot_params : dict, optional
        Additional plotting parameters. Default is an empty dictionary.

    Returns
    -------
    None
    """
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import HBox, Layout, VBox, fixed, interactive

    from ...utils.image.interpolation import img_interp
    from ...utils.physics import calc_fa_deg

    if plot_params is None:
        plot_params = {}
    if scan is not None:
        fa_deg = scan.flipangle_deg
        trf_ms = scan.rfpulse_duration_ms
        sint = scan.rfpulse_sint
        rfpow_W = scan.rfpulse_peakpow_w

    if np.ndim(refpow_map) != 6:
        # assume 2D B1 map:
        refpow_map = np.expand_dims(refpow_map, axis=(0, 1, 4, 5))

    anatomical_image = np.rot90(anatomical_image, k=1, axes=(2, 3))
    refpow_map = np.rot90(refpow_map, k=1, axes=(2, 3))

    def mri_signal_intensity(flip_angle):
        return np.abs(np.sin(np.deg2rad(flip_angle)))

    def interpolate_refpow_map(refpow_map, factor):
        if factor == 1:
            return refpow_map
        refpow_map_temp = refpow_map.copy()
        refpow_map_temp = np.where(refpow_map_temp == np.nan, 0, refpow_map_temp)
        interpolated_refow_map = img_interp(data=refpow_map_temp, interp_factor=factor)
        return interpolated_refow_map

    figsize = plot_params.get("figsize", (9, 6))
    fig, ax = plt.subplots(figsize=figsize)

    def update(
        rfpow_W,
        alpha,
        interp_factor,
        fa_deg,
        trf_ms,
        sint,
        slice_index,
        map_type,
        mri_signal_mode,
    ):
        # Clear any previous output
        global click_info
        ax.clear()  # Clear the axes for the new plot
        # Create a text annotation for displaying click information

        click_info = ax.annotate(
            "Click in the image to get information",
            xy=(0, 0),
            xytext=(0, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        click_info.set_fontsize(5)
        click_info.set_visible(True)

        if anatomical_image.shape[1] > 1:
            anatomical_slice = anatomical_image[0, slice_index, :, :, 0, 0]
        else:
            anatomical_slice = anatomical_image[0, 0, :, :, 0, 0]

        b1_slice = interpolate_refpow_map(refpow_map, interp_factor)[0, 0, :, :, 0, 0]

        # Plot the anatomical image
        ax.imshow(
            np.squeeze(anatomical_slice),
            extent=anatomical_extent,
            cmap="bone",
        )
        flip_angle_map = calc_fa_deg(
            rfpow_W=rfpow_W,
            reference_pow_W=b1_slice,
            trf_ms=trf_ms,
            sint=sint,
            fa_deg=fa_deg,
        )
        if map_type == "B1 Map":
            overlay_data = b1_slice
        else:
            if mri_signal_mode:
                overlay_data = mri_signal_intensity(flip_angle_map)
            else:
                overlay_data = flip_angle_map

        # Plot the B1 map overlay
        ax.imshow(
            np.squeeze(overlay_data),
            extent=b1_extent,
            cmap="viridis",
            alpha=alpha,
        )

        title = f"RF Power: {rfpow_W: .3f} W, Set FA: {fa_deg: .1f}°, \nPulse Duration: {trf_ms: .1f} ms, Int. Factor: {sint: .2f}"
        if anatomical_image.shape[1] > 1:
            title = f"Slice {slice_index}, " + title
        ax.set_title(title)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata

            # Calculate the position in the refpow_map array
            col = int(
                (x - b1_extent[0]) / (b1_extent[1] - b1_extent[0]) * refpow_map.shape[2]
            )
            row = int(
                (b1_extent[3] - y) / (b1_extent[3] - b1_extent[2]) * refpow_map.shape[3]
            )
            flip_angle_map = calc_fa_deg(
                rf_power_widget.value,
                refpow_map,
                trf_widget.value,
                sint_widget.value,
                fa_widget.value,
            )

            if 0 <= row < refpow_map.shape[3] and 0 <= col < refpow_map.shape[2]:
                ref_power = refpow_map[0, 0, row, col, 0, 0]

                # Update the click_info annotation
                click_info.xy = (x, y)
                click_info.set_text(
                    f"Reference Power: {ref_power:.4f} W,\nFlip angle: {flip_angle_map[0, 0, row, col, 0, 0]:.2f}°"
                )
                click_info.set_visible(True)
                fig.canvas.draw_idle()
                print(f"x={x:.2f}, y={y:.2f}, ref_power={ref_power:.4f}")
            else:
                click_info.set_visible(False)
                fig.canvas.draw_idle()

            title = f"RF Power: {rf_power_widget.value: .3f} W, Set FA: {fa_widget.value: .1f}°, \nPulse Duration: {trf_widget.value: .1f} ms, Int. Factor: {sint_widget.value: .2f}\nReference Power: {ref_power: .4f}\nFlipangle: {flip_angle_map[0, 0, row, col, 0, 0]: .2f}°"
            ax.set_title(title)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Create ipywidgets with consistent widths
    slider_layout = Layout(width="600px")
    rf_power_widget = widgets.FloatSlider(
        value=rfpow_W,
        min=0.001,
        max=1.000,
        step=0.001,
        description="RF Power (W)",
        layout=slider_layout,
        readout_format=".3f",
    )
    alpha_widget = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Transparency",
        layout=slider_layout,
    )
    interp_widget = widgets.IntSlider(
        value=1, min=1, max=5, step=1, description="Interp Factor", layout=slider_layout
    )
    fa_widget = widgets.FloatSlider(
        value=fa_deg,
        min=0,
        max=180,
        step=1,
        description="Set FA (deg)",
        layout=slider_layout,
    )
    trf_widget = widgets.FloatSlider(
        value=trf_ms,
        min=0.1,
        max=10,
        step=0.1,
        description="Pulse Duration (ms)",
        layout=slider_layout,
    )
    sint_widget = widgets.FloatSlider(
        value=sint,
        min=0.1,
        max=1,
        step=0.01,
        description="Int. Factor",
        layout=slider_layout,
    )

    map_type_widget = widgets.RadioButtons(
        options=["B1 Map", "Flip Angle"],
        value="B1 Map",
        description="Map Type",
        layout=Layout(width="200px"),
    )
    mri_signal_widget = widgets.Checkbox(
        value=False, description="MRI Signal", layout=Layout(width="150px")
    )

    if anatomical_image.shape[1] > 1:
        slice_widget = widgets.IntSlider(
            value=0,
            min=0,
            max=anatomical_image.shape[1] - 1,
            step=1,
            description="Slice",
            layout=slider_layout,
        )
    else:
        slice_widget = fixed(0)

    # Organize widgets into two rows
    row1 = HBox([rf_power_widget, fa_widget, trf_widget])
    row2 = HBox([alpha_widget, interp_widget, sint_widget])

    if anatomical_image.shape[1] > 1:
        row2 = HBox([alpha_widget, interp_widget, sint_widget, slice_widget])

    control_column = VBox([map_type_widget, mri_signal_widget])

    # Combine all widgets
    all_widgets = VBox([row1, row2, control_column])

    # Create the interactive plot
    interactive(
        update,
        rfpow_W=rf_power_widget,
        alpha=alpha_widget,
        interp_factor=interp_widget,
        fa_deg=fa_widget,
        trf_ms=trf_widget,
        sint=sint_widget,
        slice_index=slice_widget,
        map_type=map_type_widget,
        mri_signal_mode=mri_signal_widget,
    )

    # Display the interactive plot
    display(VBox([all_widgets, fig.canvas]))

    # Initial update to show the plot
    update(rfpow_W, 0.5, 1, fa_deg, trf_ms, sint, 0, "B1 Map", False)


def b1_calibration(
    scan: "MRIExperiment | None" = None,
    fit_params: dict | None = None,
    plot: bool = True,
    db: bool = False,
    *,
    local_stop: int | None = -1,
    initial_guess_cent: float | None = 3.1,
    initial_guess_extent: float | None = None,
    n_jobs: int | None = 1,
) -> float:
    """
    Estimate pulse-adjusted 90-degree reference power from an RF-power series.

    Spectral pseudo-inverse fitting supplies a complex amplitude per repetition
    and receiver. The historical iterative sinusoidal fit then estimates the
    period in sqrt(W), from which pulse duration and shape determine the returned
    reference power. This is an eager calculation; scientific determinism follows
    the selected spectral fitting options.

    Parameters
    ----------
    scan : MRIExperiment, optional
        Required scan or compatible handler despite the retained None default.
        It must provide ``n_repetitions``, ``n_receivers``, pulse duration in
        ``rfpulse_duration_ms`` and the dimensionless ``rfpulse_sint`` integral.
        Its handler supplies a complex FID in repetition, receiver, acquisition
        sample order and a ``method`` dictionary with ``PulseVectorStart`` and
        ``PulseVectorEnd`` in watts and the pulse ``Method`` identifier. For one
        receiver the singleton receiver axis may be omitted. The FID sample count
        must be divisible by repetitions times receivers. The handler's
        ``dwell_times_s`` gives the sampling interval in seconds.

    fit_params : dict, optional
        A dictionary of fitting parameters to be passed to the spectral fitting function.
        If not provided, default parameters will be used. The dictionary may contain:

            - 'metabs': list of str, default is ["peak1"]
            - 'metabs_freqs_hz': float, default is 0
            - 'cut_off': int, default is 70
            - 'signal_domain': str, default is "temporal"

        Any other option accepted by
        :func:`hypermri.analysis.fitting.spectroscopy.def_fit_params` may be
        given. ``signal_domain`` must be ``"temporal"`` because the acquired
        FID is fitted. The B1-specific key ``stop_index_extra`` (int, default
        0) is passed to the sinusoidal fit instead. The mapping is copied and
        not mutated. Frequencies are in Hz and amplitudes in arbitrary signal
        units.

    plot : bool, optional
        If True, generates plots of the fitted data and calibration results. Default is True.

    db : bool, optional
        If True, additional diagnostic plots will be generated. Default is False.

    local_stop : int or None, optional
        Exclusive Python slice end applied to both fitted amplitudes and powers
        after spectral fitting. The default -1 excludes the final sample.
        Negative ends count from the end; positive ends beyond the series are
        clipped to its length. None selects all samples. At least three samples
        must remain. This setting never truncates the acquired FID.
    initial_guess_cent : float or None, optional
        Positive period-search center in sqrt(W), default 3.1. Explicit None
        selects the numerical solver's historical 3.0 sqrt(W) center. Values
        below 0.25 are clamped to 0.25.
    initial_guess_extent : float or None, optional
        Positive search half-width in sqrt(W). An explicit value is applied to
        every receiver. None follows the supplied center before its clamp,
        preserving the previously effective window; None for both selects
        3.0 sqrt(W) for both. Half-widths below 1e-6 are clamped to 1e-6.
    n_jobs : int or None, optional
        Number of parallel workers for the spectral fit, passed to
        :func:`hypermri.analysis.fitting.spectroscopy.fit_data_pseudo_inv`.
        The default 1 fits serially; -1 uses all CPU cores. This does not
        change the iterative single-voxel period search.

    Returns
    -------
    refpower_W : float
        Pulse-adjusted reference power in watts for a 90-degree flip angle,
        calculated as ``(period / 2)**2 * (rfpulse_duration_ms * rfpulse_sint)**2``.
        For multiple receivers every receiver is fitted and printed, but only
        the last receiver's power is returned, preserving the legacy contract.

    Raises
    ------
    KeyError
        If the method file lacks ``PulseVectorStart``, ``PulseVectorEnd`` or
        ``Method``; the message names the parameter and the scan.
    AttributeError
        If FID or pulse metadata are missing.
    ValueError
        If FID dimensions or fitting inputs are inconsistent, the search window
        is invalid, fewer than three power samples are selected, the dwell
        time is not positive, ``signal_domain`` is not temporal, or
        ``n_jobs`` is not an integer or None.
    TypeError
        If an unknown keyword is supplied. Use the explicitly named calibration
        options; pass spectral options through ``fit_params``.

    Notes
    -----
    Vendor dictionaries and acquired samples are read explicitly from
    scan.handler for a scan; handlers remain valid inputs. This eager
    calculation preserves the acquired FID and the supplied fit_params and
    creates plots only when requested. Samples are
    repetition/channel/acquisition ordered. The complex amplitude of the first
    fitted metabolite is the signal of each repetition and receiver. Spectral fitting retains its
    precision policy; sinusoidal fitting calculates in complex128 with float64
    powers and search histories.

    RF powers span the declared start/end in repetition order. The
    ``<User:lucaB1shapelist_1>`` method samples linearly in sqrt(W); other methods
    sample linearly in W. Spectral amplitudes and powers must remain paired in
    that order. Pulse duration is in ms and amplitudes are in arbitrary signal
    units. No automatic flip-angle cutoff is inferred.

    Receiver progress and fitted powers are printed even when plotting is
    disabled. ``db=True`` creates figures of stored residual sums of squares
    against candidate periods independently of ``plot``.

    Examples
    --------
    For an already loaded compatible calibration scan:

    >>> reference_power = b1_calibration(  # doctest: +SKIP
    ...     scan, plot=False, db=False, n_jobs=1
    ... )
    """
    if initial_guess_extent is None:
        initial_guess_extent = initial_guess_cent
    initial_guess_cent, initial_guess_extent = _resolve_b1_calibration_window(
        initial_guess_cent, initial_guess_extent
    )
    if n_jobs is not None and (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
    ):
        raise ValueError("B1 calibration n_jobs must be an integer or None.")
    if local_stop is not None and (
        isinstance(local_stop, (bool, np.bool_))
        or not isinstance(local_stop, (int, np.integer))
    ):
        raise ValueError(
            "B1 calibration local_stop must be an integer slice end or None."
        )
    handler = getattr(scan, "handler", scan)
    fit_params = dict(fit_params) if fit_params else dict(_DEFAULT_B1_SPECTRAL_FIT)
    stop_index_extra = int(fit_params.pop("stop_index_extra", 0))
    if fit_params.get("signal_domain", "temporal") != "temporal":
        raise ValueError(
            "B1 calibration fits the acquired FID; fit_params['signal_domain'] "
            "must be 'temporal'."
        )
    fit_params["signal_domain"] = "temporal"
    rfpower_start = handler.method_parameter("PulseVectorStart")
    rfpower_end = handler.method_parameter("PulseVectorEnd")
    n_repetitions = scan.n_repetitions
    n_channels = scan.n_receivers
    fit_slice = slice(None, local_stop)
    stop_index = fit_slice.indices(n_repetitions)[1]
    if stop_index < 3:
        raise ValueError(
            "B1 calibration local_stop must select at least three RF-power samples."
        )
    if handler.method_parameter("Method") == "<User:lucaB1shapelist_1>":
        rfpower_array = (
            np.linspace(np.sqrt(rfpower_start), np.sqrt(rfpower_end), n_repetitions)
            ** 2
        )
    else:
        rfpower_array = np.linspace(rfpower_start, rfpower_end, n_repetitions)
    fit_powers = rfpower_array[fit_slice]

    from ..fitting.spectroscopy import (
        def_fit_params,
        fit_data_pseudo_inv,
        reconstruct_fitted_signal,
    )

    data = _b1_calibration_fid(
        handler.fid, n_repetitions, n_channels, handler.dwell_times_s
    )
    prepared = def_fit_params(data, fit_params)
    fit_results = fit_data_pseudo_inv(data, prepared, n_jobs=n_jobs)
    amplitudes = (
        fit_results["fit_amps"]
        .isel(metabolite=0)
        .transpose("repetition_idx", "channel_idx", ...)
        .values
    )
    if amplitudes.size != n_repetitions * n_channels:
        raise ValueError(
            "B1 calibration needs one fitted amplitude per repetition and "
            f"receiver ({n_repetitions} x {n_channels}); the spectral fit "
            f"returned shape {amplitudes.shape}."
        )
    amplitudes = amplitudes.reshape(n_repetitions, n_channels)
    if plot:
        n_fitted = len(prepared.attrs["sampling_time_axis_s"])
        fitted_samples = data.isel(acquisition_idx=slice(0, n_fitted))
        fitted_signal = reconstruct_fitted_signal(fit_results, fitted_samples)

    for channel in range(n_channels):
        label = f" Ch{channel + 1}" if n_channels > 1 else ""
        if n_channels > 1:
            print(f"Channel {channel + 1}/{n_channels}")
        if plot:
            _plot_b1_spectra(
                measured=fitted_samples.isel(channel_idx=channel),
                fitted=fitted_signal.isel(channel_idx=channel),
                freq_axis_hz=prepared.attrs["freq_axis_hz"],
                rfpower_W=rfpower_array,
            )
        amplitude, period, _, local_r2, local_per = fit_single_voxel(
            values=amplitudes[fit_slice, channel],
            refpower_W=fit_powers,
            plot=plot,
            initial_guess_cent=initial_guess_cent,
            initial_guess_extent=initial_guess_extent,
            npoints=_DEFAULT_PERIOD_SAMPLES,
            niter=_DEFAULT_REFINEMENT_ITERATIONS,
            zoom_factor=_DEFAULT_ZOOM_FACTOR,
            stop_index_extra=stop_index_extra,
        )
        refpower_W = refpow_parameter_adjustment(
            refpower_W=refpow_from_sin_fit(periods=period),
            trf_ms=scan.rfpulse_duration_ms,
            sint=scan.rfpulse_sint,
            fa_deg=90,
        )
        if db:
            _plot_b1_period_search(local_r2, local_per)
        if plot:
            _plot_b1_power_series(
                amplitudes=amplitudes[:, channel],
                sinusoid=amplitude * np.sin(np.pi / period * np.sqrt(rfpower_array)),
                rfpower_W=rfpower_array,
                refpower_W=refpower_W,
                stop_line=(
                    (rfpower_array[stop_index], f"Stop index = {local_stop}")
                    if local_stop != -1 and stop_index < n_repetitions
                    else None
                ),
                title=(
                    f"B1 Calibration{label}\n TRF: {scan.rfpulse_duration_ms}ms\n "
                    f"sint: {scan.rfpulse_sint}"
                ),
            )
        print(f"Fitted refpower{label}: {refpower_W:.6f} W")
        if scan.rfpulse_duration_ms != 1 or scan.rfpulse_sint != 1:
            print(
                "If the shown reference power (red line) is not at the peak of "
                "the sinusoidal fit [ ^= 90 degree flipangle], \nconsider that "
                f"the pulse parameters are TRF = {scan.rfpulse_duration_ms} ms "
                f"and integration factor={scan.rfpulse_sint} and that the "
                "reference power was \nscaled by these parameters."
            )
    return refpower_W


_DEFAULT_B1_SPECTRAL_FIT = {
    "metabs": ["peak1"],
    "metabs_freqs_hz": 0,
    "cut_off": 70,
    "signal_domain": "temporal",
}


def _b1_calibration_fid(fid, n_repetitions, n_channels, dwell_time_s):
    """Label the calibration FID as repetitions x channels x samples."""
    from hypermri.analysis.spectroscopy.schema import (
        ensure_standard_spectroscopy_dataarray,
    )

    dwell_time_s = float(dwell_time_s)
    if not np.isfinite(dwell_time_s) or dwell_time_s <= 0:
        raise ValueError(
            "B1 calibration needs a positive dwell time; the scan handler "
            f"reports dwell_times_s={dwell_time_s}."
        )
    fid = np.asarray(fid)
    if fid.size % (n_repetitions * n_channels):
        raise ValueError(
            f"B1 calibration FID has {fid.size} samples, which is not divisible "
            f"by {n_repetitions} repetitions x {n_channels} receivers."
        )
    fid = fid.reshape(n_repetitions, n_channels, -1)
    data = ensure_standard_spectroscopy_dataarray(
        data=fid, provided_dims=["reps", "channels", "spec"]
    )
    times_s = np.arange(fid.shape[-1]) * dwell_time_s
    return data.assign_coords(sampling_time_axis_s=("acquisition_idx", times_s))


def _plot_b1_spectra(measured, fitted, freq_axis_hz, rfpower_W):
    """Plot measured and fitted spectra for every RF power of one receiver."""
    from ...utils.plot.matplotlib_helpers import subplot_auto_size

    def spectra(signal):
        shape = (signal.sizes["repetition_idx"], signal.sizes["acquisition_idx"])
        values = signal.transpose("repetition_idx", "acquisition_idx", ...).values
        values = values.reshape(shape)
        return np.real(np.fft.fftshift(np.fft.fft(values, axis=1), axes=1))

    measured_spectra = spectra(measured)
    fitted_spectra = spectra(fitted)
    row, cols = subplot_auto_size(len(rfpower_W))
    _, ax = plt.subplots(row, cols, figsize=(10, 10))
    ax = np.atleast_1d(ax).flatten()
    for r, power in enumerate(rfpower_W):
        ax[r].plot(freq_axis_hz, fitted_spectra[r])
        ax[r].plot(freq_axis_hz, measured_spectra[r], alpha=0.5)
        if measured_spectra.max() > measured_spectra.min():
            ax[r].set_ylim(1.1 * measured_spectra.min(), 1.1 * measured_spectra.max())
        ax[r].set_xlabel("Frequency [Hz]")
        ax[r].set_title(f"RF Power {power:.4f} W")
    plt.tight_layout()
    plt.show()


def _plot_b1_period_search(local_r2, local_per):
    """Plot the residual sum of squares against every searched period."""
    from ...utils.plot.matplotlib_helpers import subplot_auto_size

    row, cols = subplot_auto_size(local_r2.shape[0])
    _, ax = plt.subplots(row, cols)
    ax = np.atleast_1d(ax).flatten()
    for a in range(local_r2.shape[0]):
        ax[a].plot(local_per[a, :], local_r2[a, :])
        ax[a].set_xlabel("Period")
        ax[a].set_ylabel("Residual sum of squares")
    plt.tight_layout()


def _plot_b1_power_series(
    amplitudes, sinusoid, rfpower_W, refpower_W, stop_line, title
):
    """Plot fitted amplitudes and the sinusoid against sqrt(power) and power."""
    _, ax = plt.subplots(1, 2, figsize=(10, 4))
    for axis, x, xlabel in (
        (ax[0], np.sqrt(rfpower_W), "sqrt(refpower) [W]"),
        (ax[1], rfpower_W, "refpower [W]"),
    ):
        axis.plot(x, np.real(amplitudes), "o-", label="Real part")
        axis.plot(x, np.imag(amplitudes), "--", label="Imaginary part")
        axis.plot(x, np.abs(amplitudes), "*-", label="Magnitude")
        axis.plot(x, np.abs(sinusoid), "--", label="Fit")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Signal")
        axis.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax[0].set_title("B1 Calibration (sinusoidal fit)")
    ax[1].set_title(title)
    ax[1].axvline(
        refpower_W,
        label=f"Fitted refpower = {np.round(refpower_W, 4)}W",
        color="red",
    )
    if stop_line is not None:
        ax[1].axvline(stop_line[0], label=stop_line[1], color="black")
    for axis in ax:
        axis.legend()
    plt.tight_layout()
    plt.show()
