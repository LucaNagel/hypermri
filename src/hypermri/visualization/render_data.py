"""Prepare viewer data/payloads while delegating coordinate calculations.

The geometry module owns pure scanner-space operations. This model retains
scientific preprocessing, alignment cache lifetime, signal/fit overlays and
render payload assembly; their numerical behavior and unit/precision policies
are unchanged by structural extraction. Initial display selection uses the
existing global spectral/repetition peak and middle spatial indices. It is
computed eagerly only when the caller supplies no positional selection.
"""

import copy
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import freq_hz_to_ppm, get_freq_axis
from hypermri.utils.coordinates.alignment import (
    ScanAlignmentResult,
    compute_scan_alignment,
    get_2d_extent_from_alignment,
)
from hypermri.utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from hypermri.utils.mathematics.calculations import part_func
from hypermri.utils.plot.data_preprocessor import DataPreprocessor
from hypermri.utils.system.logging import LOG_MODES, debug_log, session_log_path
from hypermri.visualization import geometry
from hypermri.visualization.colormap_policy import (
    DEFAULT_ANATOMY_CMAP,
    DEFAULT_SPATIAL_CMAP_NON_SPECTRO,
    DEFAULT_SPATIAL_CMAP_OVER_ANATOMY,
    DEFAULT_SPATIAL_CMAP_SPECTRO,
)
from hypermri.visualization.contracts import RenderPayload
from hypermri.visualization.geometry import _looks_like_default_index_axis
from hypermri.visualization.geometry import extract_1d_coord as extract_1d_coord
from hypermri.visualization.geometry import (
    format_spatial_axis_label as format_spatial_axis_label,
)
from hypermri.visualization.geometry import (
    format_xyz_axis_label as format_xyz_axis_label,
)
from hypermri.visualization.geometry import (
    get_spatial_dim_label as get_spatial_dim_label,
)
from hypermri.visualization.geometry import (
    normalize_coordinate_mode as normalize_coordinate_mode,
)
from hypermri.visualization.spatial_overlay import SpatialOverlayLayer
from hypermri.visualization.state import (
    AnatomyState,
    OverlayState,
    PlotState,
    SpatialState,
    SpectralState,
    _state_field_was_supplied,
)
from hypermri.visualization.view_dimensions import (
    default_curve_dim,
    displayed_spatial_dims,
)

LOGGER = logging.getLogger(__name__)
DEBUG_FILE = session_log_path("visualization-model")
DEBUG_LOG = False

# Dedicated log for tracing the spectral x-axis / overlay bug
OVERLAY_DEBUG_FILE = session_log_path("visualization-overlay")


_FIT_OVERLAY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]


def fit_metabolite_colors(metabolites: list[str]) -> dict[str, str]:
    """Return the display color of every fitted metabolite.

    Parameters
    ----------
    metabolites : list of str
        Complete metabolite coordinate of the fit results, in coordinate
        order. Pass the full coordinate, not a filtered or amplitude-sorted
        subset, so each name keeps its color in every voxel and selection.

    Returns
    -------
    dict of str to str
        Hex color per metabolite name, cycling through the fit overlay palette.
    """
    return {
        name: _FIT_OVERLAY_COLORS[idx % len(_FIT_OVERLAY_COLORS)]
        for idx, name in enumerate(metabolites)
    }


def select_complex_part(values: np.ndarray, part: str) -> np.ndarray:
    """Return the requested real-valued representation of complex data."""
    return np.asarray(part_func(np.asarray(values), part))


def curve_range_mean(curves: np.ndarray, first: int, last: int) -> np.ndarray:
    """Return the mean of every voxel's curve over a range of curve indices.

    Used for the time course of a spectral range: the mean of the displayed
    values (the chosen complex part, e.g. magnitude) between two spectral
    points, for each voxel and time point.

    Parameters
    ----------
    curves : numpy.ndarray
        Curves of shape (rows, columns, n_curve, n_time), such as
        :attr:`RenderPayload.curve_time_block`. Not modified.
    first, last : int
        Inclusive curve indices of the range in either order; values
        outside ``0 .. n_curve - 1`` are clipped. Equal values give that
        point's value.

    Returns
    -------
    numpy.ndarray
        Means of shape (rows, columns, n_time) in the units and dtype of
        ``curves``. Non-finite samples make their mean non-finite.

    Examples
    --------
    >>> curves = np.arange(8.0).reshape(1, 1, 4, 2)
    >>> curve_range_mean(curves, 1, 2)[0, 0].tolist()
    [3.0, 4.0]
    """
    n_curve = curves.shape[2]
    low, high = sorted((int(first), int(last)))
    low = min(max(low, 0), n_curve - 1)
    high = min(max(high, 0), n_curve - 1)
    return curves[:, :, low : high + 1, :].mean(axis=2, dtype=curves.dtype)


def _display_preprocessing_options(state: PlotState, prefix: str) -> dict[str, Any]:
    """Translate display state once for selected-view and whole-volume processing."""
    options = state.to_dict(prefix=prefix)
    options.update(
        {
            f"{prefix}_interpolate_image": state.spatial.interpolation,
            f"{prefix}_spec_part": state.spectral.spec_part,
            f"{prefix}_view_toggle": state.spectral.show_fid,
            f"{prefix}_vmin": state.spatial.vmin,
            f"{prefix}_vmax": state.spatial.vmax,
            f"{prefix}_range_mode": state.spatial.range_mode,
        }
    )
    return options


def _initial_display_selection(
    data: xr.DataArray, show_fid: bool
) -> tuple[dict[str, int], int | None]:
    """Return the existing initial positional selection without mutating data.

    The full-volume shifted acquisition FFT determines the global peak in
    spectral mode; temporal mode uses acquired magnitudes (signed values for
    real input). Spatial axes select their middle index, while other axes use
    the first global maximum in input dimension order. Calculation keeps the
    existing NumPy precision and is eager, deterministic and independent of UI.
    On the existing failure path all axes use their middle index and None leaves
    the prior selected spectral index unchanged. No state or acquired metadata is
    modified, and no result is cached.
    """
    try:
        vals = data.values
        if "acquisition_idx" in data.dims and not show_fid:
            acq_axis = list(data.dims).index("acquisition_idx")
            mag = np.abs(
                np.fft.fftshift(np.fft.fft(vals, axis=acq_axis), axes=acq_axis)
            )
        else:
            mag = np.abs(vals) if np.iscomplexobj(vals) else vals
            mag = np.nan_to_num(mag, nan=np.nanmin(mag))

        max_idx_full = np.unravel_index(np.nanargmax(mag), mag.shape)
        indices = {}
        spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
        max_priority_dims = ["acquisition_idx", "repetition_idx"]
        for i, dim in enumerate(data.dims):
            if dim in spatial_dims:
                indices[dim] = data.sizes[dim] // 2
            elif dim in max_priority_dims:
                indices[dim] = int(max_idx_full[i])
            else:
                indices[dim] = int(max_idx_full[i])

        spectral_index = None
        if "acquisition_idx" in data.dims:
            acq_idx_pos = list(data.dims).index("acquisition_idx")
            spectral_index = int(max_idx_full[acq_idx_pos])
        return indices, spectral_index
    except Exception:
        return {dim: data.sizes[dim] // 2 for dim in data.dims}, None


def apply_reference_layer_coordinates(
    state: PlotState,
    *,
    has_reference_layers: bool,
    coordinate_mode_explicit: bool,
) -> None:
    """Switch to scanner coordinates when anatomy or overlays are drawn.

    An anatomy image or an overlay usually comes from another scan with its
    own grid; the two are only aligned in scanner coordinates (x, y, z in
    mm). Both viewers therefore start in ``"xyz"`` mode when such a layer is
    supplied, unless the caller chose a coordinate mode.

    Parameters
    ----------
    state : PlotState
        Display state; ``state.spatial.coordinate_mode`` is set in place.
    has_reference_layers : bool
        True when an anatomy image or an overlay is shown with the data.
    coordinate_mode_explicit : bool
        True when the caller passed a coordinate mode; it is then kept.

    Returns
    -------
    None
        The state is updated in place.
    """
    if has_reference_layers and not coordinate_mode_explicit:
        state.spatial.coordinate_mode = "xyz"


def apply_display_defaults(
    state: PlotState,
    data: xr.DataArray,
    model: "PlotDataModel",
    *,
    has_overlays: bool = False,
) -> None:
    """Fill display settings that the caller left at their factory defaults.

    This is the single owner of the start-up display state of the browser
    viewer. Sequence defaults from
    ``model.get_sequence_defaults()`` replace untouched colormap,
    interpolation, spectral unit, rotation, range mode and scalebar values;
    explicit options and saved settings recorded on the state win. Empty
    ``slice_indices`` receive the initial selection of
    ``_initial_display_selection`` (middle spatial voxel, global spectral
    peak), the slicing dimension falls back to the first of slice, phase and
    read, and interpolation is switched off for grids with fewer than 32
    voxels along a spatial axis.

    Parameters
    ----------
    state : PlotState
        Display state to complete. It is mutated in place.
    data : xarray.DataArray
        Labelled samples shown by the viewer. Only read; dimension sizes and,
        for an empty selection, the samples themselves are inspected.
    model : PlotDataModel
        Data preparation owner supplying sequence defaults and the optional
        anatomy reference.
    has_overlays : bool, optional
        True when overlay layers are shown; their opacity then starts at 1
        unless the state came from saved settings or was set explicitly.

    Returns
    -------
    None
        The state is updated in place. The initial selection computes one
        full-volume FFT for spectroscopic data; nothing is cached.
    """
    seq_defaults = model.get_sequence_defaults()

    if (
        has_overlays
        and not getattr(state, "_loaded_from_persistence", False)
        and state.overlay.alpha == OverlayState.alpha
    ):
        state.overlay.alpha = 1.0

    if (
        "cmap" in seq_defaults
        and not _state_field_was_supplied(state, "spatial.cmap")
        and state.spatial.cmap == SpatialState.cmap
    ):
        state.spatial.cmap = seq_defaults["cmap"]
    if (
        "interpolation" in seq_defaults
        and not _state_field_was_supplied(state, "spatial.interpolation")
        and state.spatial.interpolation == SpatialState.interpolation
    ):
        state.spatial.interpolation = seq_defaults["interpolation"]
    if (
        "spec_unit" in seq_defaults
        and state.spectral.spec_unit == SpectralState.spec_unit
    ):
        state.spectral.spec_unit = seq_defaults["spec_unit"]
    if "rotation" in seq_defaults and state.spatial.rotation == SpatialState.rotation:
        state.spatial.rotation = seq_defaults["rotation"]
    if (
        "range_mode" in seq_defaults
        and not _state_field_was_supplied(state, "spatial.range_mode")
        and state.spatial.range_mode == SpatialState.range_mode
    ):
        state.spatial.range_mode = seq_defaults["range_mode"]
    if (
        "show_scalebar" in seq_defaults
        and not getattr(state, "_loaded_from_persistence", False)
        and "spatial.show_scalebar"
        not in getattr(state, "_explicit_state_overrides", set())
        and state.spatial.show_scalebar == SpatialState.show_scalebar
    ):
        state.spatial.show_scalebar = bool(seq_defaults["show_scalebar"])
    if state.overlay.mode not in {"normal", "subtract", "percent_difference"}:
        state.overlay.mode = "normal"
    state.spatial.coordinate_mode = normalize_coordinate_mode(
        state.spatial.coordinate_mode
    )

    if model.anatomy_obj is not None and state.anatomy.cmap == AnatomyState.cmap:
        state.anatomy.cmap = DEFAULT_ANATOMY_CMAP

    if not state.spatial.slice_indices:
        indices, spectral_index = _initial_display_selection(
            data, state.spectral.show_fid
        )
        state.spatial.slice_indices = indices
        if spectral_index is not None:
            state.spectral.selected_spectral_index = spectral_index

    if state.spatial.slicing_dim is None:
        for dim in ["slice_idx", "phase_idx", "read_idx"]:
            if dim in data.dims:
                state.spatial.slicing_dim = dim
                break

    # Low-resolution grids such as CSI look misleading when interpolated.
    if state.spatial.interpolation == SpatialState.interpolation:
        for dim in ["read_idx", "phase_idx", "slice_idx"]:
            if dim in data.dims and 1 < data.sizes[dim] < 32:
                state.spatial.interpolation = False
                break


def debug_log_vis_core_model(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Log model diagnostics, with optional explicit file output."""
    LOGGER.debug("%s", message)
    if write_to_file:
        debug_log(message=message, file_name=file_name, write_to_file=True)


# Import fit_maps to register the hp accessor on xarray datasets
try:
    from hypermri.analysis.fitting import fit_maps  # noqa: F401
except ImportError:
    pass


def get_dataarray_unit(data: xr.DataArray | None) -> str | None:
    """Return the declared unit for a DataArray, if one is present."""
    if data is None:
        return None
    for key in ("units", "unit"):
        unit = str(data.attrs.get(key, "")).strip()
        if unit:
            return unit
    return None


def format_colorbar_label(
    data: xr.DataArray | None,
    *,
    base_label: str,
) -> str:
    """Build a colorbar label that includes the DataArray unit when available."""
    unit = get_dataarray_unit(data)
    return f"{base_label} [{unit}]" if unit else base_label


def format_spectral_cursor_readout(value: float, axis_label: str | None) -> str:
    """Format the spectral cursor position as text with its axis unit.

    Parameters
    ----------
    value : float
        Cursor position in the units of the displayed spectral axis.
    axis_label : str or None
        Axis label such as ``"Frequency (ppm)"``, ``"Frequency (Hz)"``,
        ``"Time (ms)"`` or ``"name [unit]"``. The trailing bracketed unit is
        appended to the number; labels without a unit give the bare number.

    Returns
    -------
    str
        For example ``"4.700 ppm"``, ``"-125.0 Hz"`` or ``"12.34 ms"``.
    """
    match = re.search(r"[\(\[]([^\)\]]+)[\)\]]\s*$", axis_label or "")
    unit = match.group(1).strip() if match else ""
    value = float(value)
    if unit == "ppm":
        text = f"{value:.3f}"
    elif unit == "Hz":
        text = f"{value:.1f}"
    else:
        text = f"{value:.4g}"
    return f"{text} {unit}".strip()


def _axis_profile(data: xr.DataArray, dim: str) -> np.ndarray | None:
    """Return the values of the coordinate ``dim`` along its own dimension.

    Scanner data may carry ``read_idx``, ``phase_idx`` and ``slice_idx`` as
    multi-dimensional coordinates (the voxel-centre position in mm on the
    whole ``(read_idx, phase_idx, slice_idx)`` grid), and a selected plane
    drops one of these dimensions. Only the order along ``dim`` matters for
    translating positions, so the coordinate is reduced to its first line
    along ``dim``. Returns ``None`` when the coordinate is missing or does not
    vary over ``dim``.
    """
    if dim not in data.coords or dim not in data.coords[dim].dims:
        return None
    coord = data.coords[dim]
    line = coord.isel({other: 0 for other in coord.dims if other != dim})
    return np.asarray(line.values)


def _shown_source_indices(
    shown: xr.DataArray,
    index_coord_names: dict[str, str],
    shown_dims: list[str],
    processed_data: xr.DataArray,
    source_data: xr.DataArray,
) -> dict[str, np.ndarray]:
    """Return the source position of every shown row and column.

    Parameters
    ----------
    shown
        Displayed plane after selection, rotation and orientation, still
        carrying the positional index coordinates attached by
        ``geometry._index_selection_coordinates``.
    index_coord_names
        Name of the index coordinate of each spatial dimension; empty when
        no index coordinates were attached.
    shown_dims
        Displayed [row, column] dimensions.
    processed_data, source_data
        Processed array the plane was selected from and the source array
        given to ``get_render_payload``; display preprocessing may have
        reversed a dimension between them.

    Returns
    -------
    dict of str to numpy.ndarray
        Integer source positions keyed by the two shown dimensions;
        ``result[shown_dims[0]][row]`` is the source position of shown row
        ``row``. Empty without index coordinates, and when display
        preprocessing replaced the spatial grid (the k-space view shows
        spatial frequencies, not source voxels).
    """
    if not all(dim in index_coord_names for dim in shown_dims):
        return {}
    try:
        return {
            dim: np.asarray(
                _matching_positions(
                    np.asarray(shown.coords[index_coord_names[dim]].values, dtype=int),
                    dim,
                    processed_data,
                    source_data,
                ),
                dtype=int,
            )
            for dim in shown_dims
        }
    except ValueError:
        return {}


def _matching_positions(
    positions: np.ndarray | int,
    dim: str,
    from_data: xr.DataArray,
    to_data: xr.DataArray,
) -> np.ndarray | int:
    """Translate positions along ``dim`` from one array to another.

    Display preprocessing may reverse a spatial dimension (descending
    ``read_idx`` or ascending ``slice_idx`` coordinates), so a position in the
    processed data is not always the same position in the source. The
    coordinate values along ``dim`` identify the voxel; without such
    coordinates the order is unchanged. A reversal is its own inverse, so the
    same rule translates source positions to processed positions and back.

    Parameters
    ----------
    positions
        Integer position or array of positions along ``dim`` in ``from_data``.
    dim
        Dimension whose positions are translated.
    from_data, to_data
        Arrays that hold the same voxels along ``dim``, in the same or in
        reversed order.

    Returns
    -------
    numpy.ndarray or int
        Positions along ``dim`` in ``to_data`` that hold the same voxels.

    Raises
    ------
    ValueError
        If the coordinates of the two arrays are not the same values in the
        same or reversed order.
    """
    from_values = _axis_profile(from_data, dim)
    to_values = _axis_profile(to_data, dim)
    if from_values is None or to_values is None:
        return positions
    if np.array_equal(from_values, to_values):
        return positions
    if np.array_equal(from_values, to_values[::-1]):
        return to_values.size - 1 - positions
    raise ValueError(
        f"plot: the {dim} coordinates changed during display preprocessing "
        "beyond a reversal; the selected voxel cannot be traced."
    )


def _selectors_in_processed_order(
    selectors: dict[str, int],
    source: xr.DataArray | None,
    processed: xr.DataArray,
) -> dict[str, int]:
    """Translate source voxel indices to positions in the processed data.

    ``PlotState`` stores the selected voxel as source indices, while display
    preprocessing may reverse ``read_idx`` or ``slice_idx``. Only spatial
    dimensions are translated: preprocessing never reorders the other
    dimensions, and their coordinates may legitimately change (for example the
    time axis after the Fourier transform).

    Parameters
    ----------
    selectors
        Integer source index per dimension name.
    source
        Array before display preprocessing, or ``None`` to keep ``selectors``
        unchanged.
    processed
        Array after display preprocessing.

    Returns
    -------
    dict[str, int]
        A new dictionary with the positions that select the same voxel in
        ``processed``. ``selectors`` is not modified.
    """
    if source is None:
        return dict(selectors)
    return {
        dim: (
            int(_matching_positions(index, dim, source, processed))
            if dim in ("read_idx", "phase_idx", "slice_idx")
            and dim in source.dims
            and dim in processed.dims
            and source.sizes[dim] == processed.sizes[dim]
            and 0 <= index < processed.sizes[dim]
            else index
        )
        for dim, index in selectors.items()
    }


def _reject_deferred_anatomy_alignment(enabled: bool | None) -> None:
    """Reject the deferred reference-RPS alignment without reading scan data."""
    if enabled:
        raise NotImplementedError(
            "anatomy_align=True is not available; show the anatomy with "
            "coordinate_mode='xyz' instead."
        )


class PlotDataModel:
    """Prepare labelled MRI data and overlays for the existing render payload.

    Parameters
    ----------
    scan : object or None, optional
        Acquisition context for primary geometry and scientific preprocessing.
        Labelled sample arrays are supplied to rendering methods explicitly.
    anatomy_obj : object or None, optional
        Anatomical reference context and its primary_data representation.
    anatomy_align : bool, optional
        Reserved reference-RPS alignment flag. True is rejected because the
        reference-RPS viewer mode is disabled; False retains ordinary grid matching, including the shared XYZ
        plane selection and orientation when XYZ mode is selected.

    Raises
    ------
    NotImplementedError
        If anatomy_align=True requests the deferred reference-RPS viewer mode.
        Omit the flag and use ordinary XYZ anatomy/overlay plotting.

    Notes
    -----
    Coordinate functions delegate to visualization.geometry and retain labelled
    dimensions, physical direction/units and dtype. Scientific preprocessing and
    calculation precision stay with their existing owners. Construction creates no
    widgets and performs no fitting. The dormant reference-RPS cache and its
    invalidation are kept for a later decision on that mode. Source
    samples are not mutated by coordinate functions. The separate reference-RPS
    anatomy alignment remains disabled; no alignment geometry
    is computed for rejected requests.

    Examples
    --------
    >>> model = PlotDataModel(scan)  # doctest: +SKIP
    >>> payload = model.get_render_payload(scan.primary_data, state)  # doctest: +SKIP
    """

    def __init__(
        self,
        scan: Any = None,
        anatomy_obj: Any = None,
        anatomy_align: bool = False,
    ):
        """Initialize the model.

        Parameters
        ----------
        scan : Any, optional
            The sequence object providing primary data.
        anatomy_obj : Any, optional
            The sequence object providing anatomical reference.
        anatomy_align : bool, default False
            True requests the deferred reference-RPS mode and raises
            NotImplementedError. False retains ordinary coordinate matching;
            XYZ plane selection and orientation use the common geometry owner.
            No alignment geometry is calculated for rejected requests.

        Raises
        ------
        NotImplementedError
            If anatomy_align=True requests the disabled reference-RPS mode.
            Omit the flag and use ordinary XYZ anatomy/overlay plotting.
        """
        self.data_obj = scan
        self.anatomy_obj = anatomy_obj
        self.anatomy_align = anatomy_align
        self.preprocessor = DataPreprocessor(scan)
        self.anatomy_preprocessor = (
            DataPreprocessor(anatomy_obj) if anatomy_obj is not None else None
        )
        self.anatomy_data_override: xr.DataArray | None = None
        # Alignment cache; invalidate_alignment clears it when the source changes.
        self._alignment_cache: Optional[ScanAlignmentResult] = None
        self._alignment_computed: bool = False
        # Deferred reference-RPS alignment is rejected before rendering.
        self._get_alignment()

    def get_anatomy_data(self) -> xr.DataArray | None:
        """Return the anatomy samples: a DataArray or the scan's primary data.

        Returns
        -------
        xarray.DataArray or None
            ``anatomy_data_override`` when set, else the anatomy itself when
            it is a DataArray, else its ``primary_data``; None without an
            anatomy.
        """
        if self.anatomy_data_override is not None:
            return self.anatomy_data_override
        if self.anatomy_obj is None:
            return None
        data = (
            self.anatomy_obj
            if isinstance(self.anatomy_obj, xr.DataArray)
            else getattr(self.anatomy_obj, "primary_data", None)
        )
        return data if isinstance(data, xr.DataArray) else None

    def _get_alignment(self) -> Optional[ScanAlignmentResult]:
        """Reject deferred alignment; keep its implementation for a later decision."""
        _reject_deferred_anatomy_alignment(self.anatomy_align)
        if not self.anatomy_align or self.data_obj is None or self.anatomy_obj is None:
            return None
        if not self._alignment_computed:
            self._alignment_cache = compute_scan_alignment(
                source_obj=self.data_obj,
                reference_obj=self.anatomy_obj,
            )
            self._alignment_computed = True
        return self._alignment_cache

    def invalidate_alignment(self) -> None:
        """Discard this model's alignment for recomputation on the next use.

        Returns
        -------
        None
            Clears the existing model cache and its computed flag. No geometry is
            calculated and no scan data or optional UI are loaded by this call.

        Notes
        -----
        The next requested aligned payload uses the same scan-alignment operation, units in
        mm and physical position tolerance. Input samples, dimensions, coordinates,
        attrs and storage dtype are not mutated. Repeated calls are deterministic and leave the cache empty.

        Examples
        --------
        >>> model.invalidate_alignment()  # doctest: +SKIP
        """
        self._alignment_cache = None
        self._alignment_computed = False

    def get_sequence_defaults(self) -> dict[str, Any]:
        """Return the default display settings for the shown data.

        Spectroscopic data start with the ``"turbo"`` colour map, without
        interpolation and with the ``"Min/Max"`` range. Image data start
        with ``"bone"``, or with ``"turbo"`` when an anatomy lies under them
        (the anatomy itself keeps ``"bone"``), with interpolation and with
        a scale bar.

        Returns
        -------
        dict of str to Any
            Defaults for ``cmap``, ``interpolation``, ``spec_unit``,
            ``rotation``, ``range_mode`` and ``show_scalebar``. Without a
            data object the image defaults are returned.
            The call reads only metadata and has no side effects.
        """
        image_cmap = (
            DEFAULT_SPATIAL_CMAP_OVER_ANATOMY
            if self.anatomy_obj is not None
            else DEFAULT_SPATIAL_CMAP_NON_SPECTRO
        )
        defaults = {
            "cmap": image_cmap,
            "interpolation": False,
            "spec_unit": "ppm",
            "rotation": 0,
            "range_mode": "Percentage",
            "show_scalebar": False,
        }

        if self.data_obj is None:
            return defaults

        is_spectroscopic = getattr(self.data_obj, "is_spectroscopic", False)
        debug_log_vis_core_model(f"is_spectroscopoc={is_spectroscopic}")

        if is_spectroscopic:
            defaults["cmap"] = DEFAULT_SPATIAL_CMAP_SPECTRO
            defaults["interpolation"] = False
            defaults["range_mode"] = "Min/Max"
        else:
            defaults["cmap"] = image_cmap
            defaults["interpolation"] = True
            defaults["rotation"] = 0
            defaults["range_mode"] = "Min/Max"
            defaults["show_scalebar"] = True

        return defaults

    @staticmethod
    def apply_overlay_mode(
        overlay_values: np.ndarray,
        base_values: np.ndarray | None,
        mode: str,
    ) -> np.ndarray:
        """Apply spatial overlay composition mode to the overlay values."""
        resolved_mode = str(mode or "normal").strip().lower()
        if resolved_mode == "normal" or base_values is None:
            return overlay_values

        ov = np.asarray(overlay_values, dtype=float)
        base = np.asarray(base_values, dtype=float)
        if ov.shape != base.shape:
            return overlay_values

        if resolved_mode == "subtract":
            return ov - base

        if resolved_mode == "percent_difference":
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = 100.0 * (ov - base) / np.abs(base)
            return np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

        return overlay_values

    def _get_spectral_reference_axis(
        self,
        npoints: int,
        state: PlotState,
        source_data: xr.DataArray | None = None,
    ) -> np.ndarray | None:
        """Generate the spectral/temporal reference axis for mapping values to indices."""
        if state.spectral.show_fid:
            dt = self._spectral_dwell_time_s(source_data)
            if dt is None:
                return None
            return np.arange(npoints, dtype=float) * dt * 1000.0
        else:
            try:
                return get_freq_axis(
                    self.data_obj,
                    state.spectral.spec_unit,
                    getattr(self.data_obj, "cut_off", 0),
                    npoints,
                )
            except Exception:
                return None

    def _spectral_dwell_time_s(
        self, source_data: xr.DataArray | None = None
    ) -> float | None:
        """Return a finite positive dwell time, preferring the displayed array."""
        try:
            primary_data = getattr(self.data_obj, "primary_data", None)
        except Exception:
            primary_data = None

        data_candidates: list[xr.DataArray] = []
        for candidate in (source_data, primary_data):
            if isinstance(candidate, xr.DataArray) and all(
                candidate is not existing for existing in data_candidates
            ):
                data_candidates.append(candidate)

        def axis_step(candidate: Any) -> float | None:
            try:
                axis = np.asarray(candidate, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            if axis.size < 2:
                return None
            diffs = np.diff(axis)
            diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
            if not diffs.size:
                return None
            dt = float(np.median(diffs))
            return dt if np.isfinite(dt) and dt > 0 else None

        def scalar_dwell(candidate: Any) -> float | None:
            try:
                dt = float(candidate)
            except (TypeError, ValueError):
                return None
            return dt if np.isfinite(dt) and dt > 0 else None

        # Explicit ``plot(data, ..., scan=scan)`` metadata must win over the
        # sequence object's primary data, because derived arrays can have a
        # different sampling grid.
        for candidate_data in data_candidates:
            for coord_name in ("acquisition_time_s", "sampling_time_axis_s"):
                if coord_name in candidate_data.coords:
                    dt = axis_step(candidate_data.coords[coord_name].values)
                    if dt is not None:
                        return dt
            for attr_name in ("sampling_time_axis_s", "sampling_time_axis"):
                dt = axis_step(candidate_data.attrs.get(attr_name))
                if dt is not None:
                    return dt
            dt = scalar_dwell(candidate_data.attrs.get("dwell_time_s"))
            if dt is not None:
                return dt

        for attr_name in ("sampling_time_axis_s", "sampling_time_axis"):
            try:
                dt = axis_step(getattr(self.data_obj, attr_name, None))
            except Exception:
                continue
            if dt is not None:
                return dt

        try:
            dt = scalar_dwell(getattr(self.data_obj, "dwell_time_s", None))
        except Exception:
            dt = None
        if dt is not None:
            return dt
        return None

    def _extract_display_1d_coord(
        self,
        data: xr.DataArray,
        dim_name: str,
    ) -> tuple[np.ndarray, str, str]:
        """Extract the display axis for a dimension, preferring physical time over raw indices."""
        primary_values = extract_1d_coord(data, dim_name)
        if dim_name != "repetition_idx" or not _looks_like_default_index_axis(
            primary_values
        ):
            unit = "idx"
            if dim_name in data.coords:
                coord_attrs = data.coords[dim_name].attrs
                unit = coord_attrs.get("units", coord_attrs.get("unit", unit))
            return primary_values, dim_name, unit

        for alt_name in ("time_s", "time", "acquisition_time_s"):
            if alt_name not in data.coords:
                continue
            alt_coord = data.coords[alt_name]
            if dim_name not in alt_coord.dims:
                continue
            alt_values = self._coerce_axis_values(
                alt_coord.values,
                data.sizes.get(dim_name),
            )
            if alt_values is not None:
                coord_attrs = alt_coord.attrs
                unit = coord_attrs.get("units", coord_attrs.get("unit", "s"))
                return alt_values, alt_name, unit

        if dim_name in data.coords:
            axis_s = data.coords[dim_name].attrs.get("axis_s")
            axis_values = self._coerce_axis_values(axis_s, data.sizes.get(dim_name))
            if axis_values is not None:
                return axis_values, "time", "s"

        for attr_name in ("frame_time_s", "time_points_s"):
            axis_values = self._coerce_axis_values(
                data.attrs.get(attr_name),
                data.sizes.get(dim_name),
            )
            if axis_values is not None:
                return axis_values, "time", "s"

        if self.data_obj is not None:
            for attr_name in ("frame_time_s", "time_points_s"):
                axis_values = self._coerce_axis_values(
                    getattr(self.data_obj, attr_name, None),
                    data.sizes.get(dim_name),
                )
                if axis_values is not None:
                    return axis_values, "time", "s"

            primary_data = getattr(self.data_obj, "primary_data", None)
            if isinstance(primary_data, xr.DataArray):
                for alt_name in ("time_s", "time"):
                    if alt_name not in primary_data.coords:
                        continue
                    alt_coord = primary_data.coords[alt_name]
                    if dim_name not in alt_coord.dims:
                        continue
                    axis_values = self._coerce_axis_values(
                        extract_1d_coord(primary_data, alt_name),
                        data.sizes.get(dim_name),
                    )
                    if axis_values is not None:
                        coord_attrs = alt_coord.attrs
                        unit = coord_attrs.get("units", coord_attrs.get("unit", "s"))
                        return axis_values, alt_name, unit

                if dim_name in primary_data.coords:
                    axis_s = primary_data.coords[dim_name].attrs.get("axis_s")
                    axis_values = self._coerce_axis_values(
                        axis_s,
                        data.sizes.get(dim_name),
                    )
                    if axis_values is not None:
                        return axis_values, "time", "s"

        return primary_values, dim_name, "idx"

    @staticmethod
    def _coerce_axis_values(
        values: Any,
        expected_size: int | None,
    ) -> np.ndarray | None:
        """Normalize candidate axis values and validate their length."""
        if values is None:
            return None
        axis_values = np.asarray(values)
        if axis_values.ndim != 1 or axis_values.size == 0:
            return None
        if expected_size is not None and axis_values.size < expected_size:
            return None
        if expected_size is not None and axis_values.size > expected_size:
            axis_values = axis_values[:expected_size]
        return axis_values.astype(float, copy=False)

    _uses_xyz_coordinates = staticmethod(geometry._uses_xyz_coordinates)

    def _ensure_xyz_coordinates(
        self,
        data: xr.DataArray,
        state: PlotState,
        *,
        scan: Any | None = None,
    ) -> xr.DataArray:
        return geometry._ensure_xyz_coordinates(
            data, state, scan=scan if scan is not None else self.data_obj
        )

    _coord_unit = staticmethod(geometry._coord_unit)

    _coord_values_in_display_order = staticmethod(
        geometry._coord_values_in_display_order
    )

    _axis_line_values = staticmethod(geometry._axis_line_values)

    _finite_span = staticmethod(geometry._finite_span)

    _line_is_descending = staticmethod(geometry._line_is_descending)

    _xyz_display_coordinate_data = staticmethod(geometry._xyz_display_coordinate_data)

    _select_xyz_display_coords = staticmethod(geometry._select_xyz_display_coords)

    _xyz_axis_coord_by_display_axis = staticmethod(
        geometry._xyz_axis_coord_by_display_axis
    )

    _canonical_xyz_display_pair = staticmethod(geometry._canonical_xyz_display_pair)

    _orient_spatial_slice_for_xyz = staticmethod(geometry._orient_spatial_slice_for_xyz)

    _axis_extent_from_coord_values = staticmethod(
        geometry._axis_extent_from_coord_values
    )

    _calculate_xyz_spatial_extent = staticmethod(geometry._calculate_xyz_spatial_extent)

    _coord_line_along_dim = staticmethod(geometry._coord_line_along_dim)

    _data_obj_value = staticmethod(geometry._data_obj_value)

    _spatial_dim_resolution_mm = staticmethod(geometry._spatial_dim_resolution_mm)

    _direct_xyz_vector_for_dim = staticmethod(geometry._direct_xyz_vector_for_dim)

    _xyz_vector_for_dim = staticmethod(geometry._xyz_vector_for_dim)

    _dominant_xyz_coord_for_dim = staticmethod(geometry._dominant_xyz_coord_for_dim)

    _slice_xyz_position = staticmethod(geometry._slice_xyz_position)

    _slice_xyz_positions = staticmethod(geometry._slice_xyz_positions)

    _finite_slice_positions = staticmethod(geometry._finite_slice_positions)

    _nearest_slice_index_for_xyz_position = staticmethod(
        geometry._nearest_slice_index_for_xyz_position
    )

    _position_outside_slice_fov = staticmethod(geometry._position_outside_slice_fov)

    _xyz_plane_signature = staticmethod(geometry._xyz_plane_signature)

    _dominant_anatomy_dim_for_xyz_coord = staticmethod(
        geometry._dominant_anatomy_dim_for_xyz_coord
    )

    _target_plane_from_primary_xyz = staticmethod(
        geometry._target_plane_from_primary_xyz
    )

    def reduce_spatial_overlay(
        self, data: xr.DataArray, state: PlotState
    ) -> xr.DataArray:
        """Reduce overlay samples to one metabolite and the shown repetitions.

        Parameters
        ----------
        data : xarray.DataArray
            One overlay layer, for example the ``fit_amps`` variable of a fit
            result, with labelled spatial dimensions and optionally metabolite
            and repetition_idx. Not mutated.
        state : PlotState
            ``state.overlay.active_metabolite`` selects the metabolite (the
            first one when unset or unknown). The repetition average or
            cumulative sum and its range follow the spectral settings of the
            shown scan (``average_repetitions``, ``cum_sum_repetitions``,
            ``rep_avg_start``, ``rep_avg_end``).

        Returns
        -------
        xarray.DataArray
            The layer in standard dimension order without metabolite and, when
            a repetition reduction is selected, without repetition_idx. Units
            and attributes are those of data.
        """
        data = self.preprocessor.standardize_dimension_order(data)
        if "metabolite" in data.dims:
            metabolites = [str(m) for m in data.metabolite.values]
            if state.overlay.active_metabolite in metabolites:
                data = data.sel(metabolite=state.overlay.active_metabolite)
            else:
                data = data.isel(metabolite=0)
        if "repetition_idx" in data.dims:
            start = state.spectral.rep_avg_start
            end = state.spectral.rep_avg_end
            if state.spectral.cum_sum_repetitions:
                data = self.preprocessor.cumulative_sum_repetitions_range(
                    data, start=start, end=end
                )
            elif state.spectral.average_repetitions:
                data = self.preprocessor.average_repetitions_range(
                    data, start=start, end=end
                )
        return data

    def preprocess_spatial_overlay(
        self, data: xr.DataArray, state: PlotState
    ) -> xr.DataArray:
        """Match a reduced overlay to the spectral display of the shown scan.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay layer after :meth:`reduce_spatial_overlay`. Not mutated.
        state : PlotState
            Spectral display settings of the shown scan (FID or spectrum,
            complex part, line broadening, channel combination, denoising)
            and the coordinate mode.

        Returns
        -------
        xarray.DataArray
            The layer with scanner x/y/z coordinates in scanner coordinate
            mode. A layer with acquisition_idx is processed like the shown
            scan, so its map at the selected spectral position is comparable;
            repetitions are not reduced a second time. Other layers keep their
            values.
        """
        data = self._ensure_xyz_coordinates(data, state, scan=data)
        if "acquisition_idx" not in data.dims:
            return data
        prefix = "active_overlay"
        options = state.to_dict(prefix=prefix)
        options.update(
            {
                f"{prefix}_interpolate_image": state.spatial.interpolation,
                f"{prefix}_spec_part": state.spectral.spec_part,
                f"{prefix}_view_toggle": state.spectral.show_fid,
                f"{prefix}_vmin": state.overlay.vmin,
                f"{prefix}_vmax": state.overlay.vmax,
                f"{prefix}_range_mode": state.overlay.range_mode,
                # reduce_spatial_overlay already reduced the repetitions.
                f"{prefix}_average_repetitions": False,
                f"{prefix}_cum_sum_repetitions": False,
            }
        )
        return self.preprocessor.apply_preprocessing(
            data, options, widget_prefix=prefix
        )

    def _apply_spatial_overlay(
        self,
        payload: RenderPayload,
        state: PlotState,
        layer: SpatialOverlayLayer,
        *,
        primary_data: xr.DataArray,
        spatial_dims: list[str],
        display_spatial_dims: list[str],
        primary_spatial_slice: xr.DataArray,
    ) -> None:
        """Add the overlay plane matching the shown map to the payload.

        The plane is cut like the map: positional selection and rotation by
        :meth:`get_spatial_slice`, then scanner orientation. spatial_dims are
        the map's row/column dimensions before rotation and
        display_spatial_dims after it.
        """
        overlay_state = state.overlay
        payload.overlay_cmap = overlay_state.cmap
        payload.overlay_alpha = overlay_state.alpha
        if layer.value_range is not None:
            low, high = layer.value_range
            if overlay_state.range_mode == "Percentage":
                span = high - low
                payload.overlay_clim = (
                    low + (overlay_state.vmin / 100.0) * span,
                    low + (overlay_state.vmax / 100.0) * span,
                )
            else:
                payload.overlay_clim = (overlay_state.vmin, overlay_state.vmax)
        try:
            selectors, keep_dims = self._secondary_plane_selection(
                layer.data,
                state,
                overlay_state,
                primary_data=primary_data,
                primary_spatial_slice=primary_spatial_slice,
                primary_spatial_dims=display_spatial_dims,
            )
            payload.metadata["overlay_plane_selection"] = {
                dim: int(index) for dim, index in selectors.items()
            }
            if overlay_state.slice_warning:
                payload.metadata["overlay_slice_warning"] = overlay_state.slice_warning
            if (
                overlay_state.auto_hide_non_overlapping
                and overlay_state.slice_overlap_status == "primary_outside_secondary"
            ):
                payload.metadata["overlay_auto_hidden_non_overlap"] = True
                return

            plane_dims = list(spatial_dims)
            if len(keep_dims) == 2 and set(keep_dims) != set(spatial_dims):
                plane_dims = list(keep_dims)
            plane_state = state
            if selectors:
                plane_state = copy.copy(state)
                plane_state.spatial = copy.copy(state.spatial)
                plane_state.spatial.slice_indices = {
                    **state.spatial.slice_indices,
                    **selectors,
                }
            plane = self.get_spatial_slice(
                layer.data,
                plane_state,
                prefix=f"ov_{layer.name}",
                spatial_dims=plane_dims,
            )
            if plane.ndim != 2:
                return
            plane_display_dims = list(plane.dims)
            plane, plane_display_dims = self._orient_spatial_slice_for_xyz(
                plane,
                plane_display_dims,
                state,
                geometry_data=layer.data,
            )
            values = np.asarray(plane.values)
            if np.iscomplexobj(values):
                values = np.abs(values)
            values = self.apply_overlay_mode(
                overlay_values=np.asarray(values, dtype=float),
                base_values=payload.spatial_array,
                mode=overlay_state.mode,
            )
            if state.spatial.min_to_nan and payload.overlay_clim is not None:
                values = np.asarray(values, dtype=float).copy()
                values[values < payload.overlay_clim[0]] = np.nan

            payload.spatial_overlays[layer.name] = values
            if self._uses_xyz_coordinates(state):
                extent = self._calculate_xyz_spatial_extent(
                    plane, plane_display_dims, geometry_data=layer.data
                )
                if extent is not None:
                    payload.spatial_overlay_extents[layer.name] = extent[0]
            payload.metadata["overlay_mode"] = overlay_state.mode
            payload.labels["overlay_colorbar"] = format_colorbar_label(
                plane, base_label="Overlay"
            )
        except Exception as exc:
            debug_log_vis_core_model(f"model.spatial_overlay_error: {exc!r}")

    def _secondary_plane_selection(
        self,
        target_data: xr.DataArray,
        state: PlotState,
        secondary_state: Any,
        *,
        primary_data: xr.DataArray | None = None,
        primary_spatial_slice: xr.DataArray | None = None,
        primary_spatial_dims: list[str] | None = None,
    ) -> tuple[dict[str, int], list[str]]:
        explicit = geometry._explicit_secondary_plane(target_data, secondary_state)
        if explicit is not None:
            return explicit
        if (
            self._uses_xyz_coordinates(state)
            and primary_spatial_slice is None
            and primary_data is not None
        ):
            try:
                primary_spatial_slice = self.get_spatial_slice(
                    primary_data, state, spatial_dims=primary_spatial_dims
                )
                primary_spatial_dims = list(primary_spatial_slice.dims)
            except Exception:
                primary_spatial_slice = None
        return geometry._secondary_plane_from_primary(
            target_data,
            state,
            primary_data=primary_data,
            primary_spatial_slice=primary_spatial_slice,
            primary_spatial_dims=primary_spatial_dims,
            scan=self.data_obj,
        )

    def _primary_plane_signature(
        self, primary_data: xr.DataArray, state: PlotState
    ) -> tuple[xr.DataArray, str, float] | None:
        """Return primary data with x/y/z and the scanner position of the shown plane.

        The result is ``(data, normal_coordinate, position_mm)``; None outside
        scanner coordinate mode or when the plane has no single position.
        """
        if not self._uses_xyz_coordinates(state):
            return None
        try:
            primary_data = self._ensure_xyz_coordinates(
                primary_data, state, scan=self.data_obj
            )
            spatial_dims = self._determine_spatial_dims(primary_data, state, "ds0")
            if len(spatial_dims) != 2:
                return None
            primary_spatial_slice = self.get_spatial_slice(
                primary_data, state, spatial_dims=spatial_dims
            )
            signature = self._xyz_plane_signature(
                primary_spatial_slice,
                list(primary_spatial_slice.dims),
                geometry_data=primary_data,
                slicing_dim=state.spatial.slicing_dim,
                scan=self.data_obj,
            )
        except Exception:
            return None
        if signature is None:
            return None
        normal_coord, position = signature
        return primary_data, normal_coord, float(position)

    def primary_slice_for_secondary_slice(
        self,
        primary_data: xr.DataArray,
        state: PlotState,
        target_data: xr.DataArray,
        target_dim: str,
        target_index: int,
        *,
        label: str,
    ) -> tuple[str | None, int | None, str | None] | None:
        """Return the primary slice at the scanner position of a secondary slice.

        Used when a viewer lets the user choose the slice of an anatomy or
        overlay layer in scanner coordinate mode: the shown primary plane
        moves to the slice nearest to the chosen one, as in ``plot``.

        Parameters
        ----------
        primary_data : xarray.DataArray
            Shown scan data with labelled spatial dimensions. Not mutated.
        state : PlotState
            Display state; read only. ``state.spatial.coordinate_mode`` must
            be ``"xyz"`` for a match.
        target_data : xarray.DataArray
            Anatomy or overlay samples with scanner x/y/z coordinates in mm.
        target_dim : str
            Dimension of ``target_data`` normal to its shown plane, for
            example ``"slice_idx"``.
        target_index : int
            Zero-based chosen slice along ``target_dim``.
        label : str
            Layer name used in the warning, for example ``"overlay"``.

        Returns
        -------
        tuple or None
            None outside scanner coordinate mode or when the chosen slice
            has no scanner position. Otherwise ``(primary_dim,
            primary_index, warning)``: the primary dimension and zero-based
            index to show, both None when the primary data have a single
            slice along that coordinate, and a warning text when the chosen
            slice lies outside the primary slices (else None). A single
            primary slice covers its slice thickness around its position.
        """
        if not self._uses_xyz_coordinates(state) or target_dim not in target_data.dims:
            return None
        try:
            prepared = self._ensure_xyz_coordinates(
                primary_data, state, scan=self.data_obj
            )
        except Exception:
            return None
        return geometry._nearest_primary_slice_from_secondary(
            prepared,
            target_data,
            target_dim,
            int(target_index),
            source_label=label,
            primary_signature=self._primary_plane_signature(primary_data, state),
            primary_thickness_mm=geometry._spatial_dim_resolution_mm(
                prepared, state.spatial.slicing_dim, scan=self.data_obj
            ),
        )

    _nearest_slice_from_xyz = staticmethod(geometry._nearest_slice_from_xyz)

    def _xyz_display_point(
        self,
        data: xr.DataArray,
        state: PlotState,
        spatial_dims: list[str],
    ) -> tuple[float, float] | None:
        return geometry._xyz_display_point(
            data, state, spatial_dims, scan=self.data_obj
        )

    def _xyz_indices_from_display_point(
        self,
        data: xr.DataArray,
        state: PlotState,
        x: float,
        y: float,
        *,
        prefix: str = "ds0",
        spatial_dims: list[str] | None = None,
    ) -> dict[str, int] | None:
        if (
            not self._uses_xyz_coordinates(state)
            or not has_xyz_coordinates(data)
            or not np.isfinite(x)
            or not np.isfinite(y)
        ):
            return None
        actual_spatial_dims = (
            list(spatial_dims)
            if spatial_dims is not None
            else self._determine_spatial_dims(data, state, prefix)
        )
        if len(actual_spatial_dims) != 2:
            return None
        prepared = geometry._index_selection_coordinates(data, actual_spatial_dims)
        if prepared is None:
            return None
        selection_data, index_coord_names = prepared
        try:
            spatial_slice = self.get_spatial_slice(
                selection_data, state, prefix=prefix, spatial_dims=actual_spatial_dims
            )
        except Exception:
            return None
        return geometry._xyz_indices_from_prepared_slice(
            data,
            selection_data,
            spatial_slice,
            actual_spatial_dims,
            index_coord_names,
            state,
            x,
            y,
            scan=self.data_obj,
        )

    @staticmethod
    def _secondary_preprocessing_kwargs(
        proc_kwargs: dict[str, Any],
        *,
        target_dim: str | None,
        prefix: str,
    ) -> dict[str, Any]:
        """Preserve the secondary axis when preprocessing would otherwise remove it."""
        adjusted = dict(proc_kwargs)

        # The secondary view cannot plot along a dimension that preprocessing has
        # already collapsed. Keep the requested axis intact for the 1D signal path.
        if target_dim == "repetition_idx":
            adjusted[f"{prefix}_average_repetitions"] = False
        elif target_dim == "channel_idx":
            adjusted[f"{prefix}_combine_channels"] = False

        return adjusted

    def get_spatial_slice(
        self,
        data: xr.DataArray,
        state: PlotState,
        prefix: str = "ds0",
        spatial_dims: list[str] | None = None,
        *,
        keep_dims: tuple[str, ...] = (),
    ) -> xr.DataArray:
        """Select the displayed plane, or the plane with kept dimensions.

        Parameters
        ----------
        data : xarray.DataArray
            Processed labelled samples with the spatial dimensions and any
            further selection dimensions such as acquisition_idx,
            repetition_idx or channel_idx.
        state : PlotState
            Positional selections, the selected spectral index or value and the
            display rotation in degrees (0, 90, 180 or 270). Not mutated.
        prefix : str, optional
            Preprocessor option prefix used to determine spatial dimensions.
        spatial_dims : list of str or None, optional
            Displayed row and column dimensions before rotation. None
            determines them from data and state.
        keep_dims : tuple of str, optional
            Non-spatial dimensions that stay whole instead of being selected,
            for example ``("acquisition_idx",)`` for the spectra of every voxel.
            They follow the two spatial dimensions in the result.

        Returns
        -------
        xarray.DataArray
            The rotated plane in displayed row/column order, followed by the
            kept dimensions. Values, units, attributes and dtype are those of
            data; only selection, transposition and reversal are applied.
        """
        if spatial_dims is None:
            spatial_dims = self._determine_spatial_dims(data, state, prefix)

        spectral_dims = [dim for dim in ["acquisition_idx"] if dim not in keep_dims]

        # Base slice selectors
        spatial_selectors = {
            dim: state.spatial.slice_indices.get(dim, 0)
            for dim in data.dims
            if dim not in spatial_dims
            and dim not in spectral_dims
            and dim not in keep_dims
        }

        # Handle spectral/temporal dimension index
        if state.spectral.selected_spectral_value is not None:
            for spec_dim in spectral_dims:
                if spec_dim in data.dims:
                    npoints = data.sizes[spec_dim]
                    ref_axis = self._get_spectral_reference_axis(npoints, state, data)
                    if ref_axis is not None:
                        idx = int(
                            np.argmin(
                                np.abs(
                                    ref_axis - state.spectral.selected_spectral_value
                                )
                            )
                        )
                        spatial_selectors[spec_dim] = idx
                        break
        elif state.spectral.selected_spectral_index is not None:
            for spec_dim in spectral_dims:
                if spec_dim in data.dims:
                    spatial_selectors[spec_dim] = min(
                        state.spectral.selected_spectral_index, data.sizes[spec_dim] - 1
                    )
                    break

        # Apply slicing
        valid_selectors = {
            k: v
            for k, v in spatial_selectors.items()
            if k in data.dims and v < data.sizes[k]
        }
        spatial_slice = data.isel(valid_selectors)

        # Squeeze remaining non-spatial dims
        auto_selectors = {
            d: 0
            for d in spatial_slice.dims
            if d not in spatial_dims and d not in keep_dims
        }
        if auto_selectors:
            spatial_slice = spatial_slice.isel(auto_selectors)

        # Transpose and rotate
        if len(spatial_dims) == 2:
            dim_y, dim_x = spatial_dims
            if all(d in spatial_slice.dims for d in [dim_y, dim_x]):
                spatial_slice = spatial_slice.transpose(dim_y, dim_x, ...)
                rot = state.spatial.rotation % 360
                if rot == 90:
                    spatial_slice = spatial_slice.transpose(dim_x, dim_y, ...).isel(
                        {dim_y: slice(None, None, -1)}
                    )
                elif rot == 180:
                    spatial_slice = spatial_slice.isel(
                        {dim_y: slice(None, None, -1), dim_x: slice(None, None, -1)}
                    )
                elif rot == 270:
                    spatial_slice = spatial_slice.transpose(dim_x, dim_y, ...).isel(
                        {dim_x: slice(None, None, -1)}
                    )

        return spatial_slice

    def _get_display_domain_range(
        self,
        data: xr.DataArray,
        state: PlotState,
        payload: RenderPayload | None = None,
    ) -> tuple[float, float] | None:
        """Return finite global bounds after display preprocessing.

        All labelled dimensions of data participate; unlike payload preparation,
        no selected-plane slicing precedes processing. FFT axes and PCA scope
        therefore remain those of the complete volume. This eager operation
        preserves acquired values/state and uses the existing calculation
        precision without caching. Bounds are Python floats in processed signal
        units; phase has the canonical interval [-pi, pi] in radians. A successful
        all-nonfinite result returns None. Only preprocessing failure uses finite
        values of the optional already displayed payload as a fallback.
        """
        try:
            prefix = "ds0"
            proc_kwargs = _display_preprocessing_options(state, prefix)
            processed = self.preprocessor.apply_preprocessing(
                data, proc_kwargs, widget_prefix=prefix
            )
            vals = np.asarray(processed.values)
            if processed.attrs.get("spec_part_applied") != state.spectral.spec_part:
                vals = select_complex_part(vals, state.spectral.spec_part)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                return None
            dmin = float(np.nanmin(finite))
            dmax = float(np.nanmax(finite))
            if state.spectral.spec_part == "Magnitude":
                dmin = max(0.0, dmin)
                dmax = max(dmin, dmax)
            elif state.spectral.spec_part == "Phase":
                # A canonical phase interval avoids rescaling a plane that spans
                # only a small part of the phase circle.
                dmin, dmax = -float(np.pi), float(np.pi)
            return dmin, dmax
        except Exception:
            if payload is not None and payload.spatial_array is not None:
                arr = np.asarray(payload.spatial_array, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    dmin = float(np.nanmin(finite))
                    dmax = float(np.nanmax(finite))
                    if state.spectral.spec_part == "Magnitude":
                        return max(0.0, dmin), max(0.0, dmax)
                    if state.spectral.spec_part == "Phase":
                        return -float(np.pi), float(np.pi)
                    return dmin, dmax
            return None

    def get_render_payload(
        self,
        data: xr.DataArray,
        state: PlotState,
        prefix: str = "ds0",
        overlays: dict[str, Any] | None = None,
        *,
        include_curve_block: bool = False,
        time_dim: str | None = None,
        spatial_overlay: SpatialOverlayLayer | None = None,
    ) -> RenderPayload:
        """Prepare selected spatial and secondary views from labelled data.

        Parameters
        ----------
        data
            Signal or parameter-map array with labelled spatial dimensions such
            as phase_idx/read_idx/slice_idx and optional acquisition_idx,
            repetition_idx or channel_idx. Positional selections use these
            labels rather than assuming input dimension order. Signal units,
            domain, sampling-time/frequency axes and scanner geometry come from
            array metadata or the model's sequence object.
        state
            Selections, display preprocessing and geometry choices. Percentage
            colour bounds refer to the selected processed plane; physical bounds
            use its signal units, or radians for phase. The state is not mutated.
        prefix
            Prefix of the explicit preprocessor option keys; normally ds0 for
            the primary view.
        overlays
            Named labelled signal arrays or fitted Datasets for the secondary
            view. Compatible dimensions share positional selectors and display
            preprocessing. Fitted signals use the selected voxel and the
            Dataset's scientific reconstructor; named scan.fit_results
            dictionaries are supported. Show Fit controls curve visibility
            independently of spatial parameter-map selection. Unsupported
            overlay values are ignored. Inputs are not mutated.
        include_curve_block
            Also return the processed curves of every displayed voxel along
            the curve dimension (``acquisition_idx`` spectra, or for example
            ``repetition_idx``) as ``payload.curve_block`` when a
            two-dimensional map is shown. They come from the plane of the
            secondary view: as there, averaging repetitions or combining
            channels reduces only the map when the curve runs along that
            dimension, and the curves keep every entry (as in ``plot``).
        time_dim
            With ``include_curve_block``, also return those curves at every
            index of this dimension (for example ``repetition_idx``) as
            ``payload.curve_time_block``, shape (rows, columns, curve, time).
            The dimension is already kept by the shared preprocessing, so
            this only selects and copies the plane. None, the curve
            dimension itself, or a dimension the display processing removes or
            reduces to one entry (combined channels, averaged repetitions)
            leaves it None.
        spatial_overlay
            Overlay layer drawn over the map while ``state.overlay.visible``.
            It is cut to the shown plane with the same selection, rotation and
            orientation as the map, so a layer on the map's grid lines up
            pixel by pixel; a layer on another grid is matched to the shown
            plane through its scanner coordinates (``state.overlay`` keeps the
            selected plane, slice warning and overlap status). Colour limits
            follow ``state.overlay.range_mode`` and ``vmin``/``vmax``; the
            result is ``payload.spatial_overlays[spatial_overlay.name]`` with
            its scanner extent in ``payload.spatial_overlay_extents`` in
            scanner coordinate mode.

        Returns
        -------
        RenderPayload
            Spatial array ordered as spatial_dims (display y, display x), with
            corresponding coordinates, extents, labels, colour bounds and
            optional background. The secondary signal is one-dimensional with
            its displayed axis, units and overlays. Rotation and scanner
            orientation follow state/geometry metadata. Scanner-space extents
            are in millimetres; other axis units follow the display mode and
            labelled coordinates. Missing views have None
            arrays. Processing failures are reported in spatial_error or
            spectral_error rather than raising during view preparation.

        Notes
        -----
        Preparation is eager and creates no widgets or plotting backend. It
        slices dimensions unused by either view before applying the existing
        numerical stages. This scope differs from whole-volume slider bounds:
        spatial FFT or denoising can depend on the retained dimensions. Secondary
        dimension reductions are disabled when they would remove its axis.
        Acquired values, coordinates and attributes remain intact; computation
        keeps each stage's existing precision, while optional masking copies the
        displayed plane to floating point. Geometry alignment may populate the
        model's existing lazy cache. Processed signals are not cached.
        Numerical preparation is deterministic for unchanged inputs and metadata.

        Examples
        --------
        >>> state = PlotState()
        >>> state.spatial.slice_indices = {"slice_idx": 0}
        >>> payload = model.get_render_payload(scan.primary_data, state)  # doctest: +SKIP
        """
        debug_log_vis_core_model("get_render_payload: Start")
        data = self._ensure_xyz_coordinates(data, state, scan=self.data_obj)
        payload = RenderPayload()
        spectral_dims = ["acquisition_idx"]

        proc_kwargs = _display_preprocessing_options(state, prefix)

        # Set overlay alpha (0 if hidden)
        payload.overlay_alpha = state.overlay.alpha if state.overlay.visible else 0.0
        payload.overlay_cmap = state.overlay.cmap

        # 2. SHARED PREPROCESSING
        try:
            # Identify dimensions for both views
            spatial_dims_raw = self._determine_spatial_dims(data, state, prefix)
            target_dim = self._determine_secondary_target_dim(
                data, state, spatial_dims_raw
            )

            # Construct 'keep_dims' (union of dimensions needed for any view or preprocessor)
            keep_dims = list(set(spatial_dims_raw))
            if target_dim and target_dim not in keep_dims:
                keep_dims.append(target_dim)

            for d in ["repetition_idx", "channel_idx", "acquisition_idx"]:
                if d in data.dims and d not in keep_dims:
                    keep_dims.append(d)

            # Pre-slice to reduce volume before processing
            shared_selectors = {
                dim: state.spatial.slice_indices.get(dim, 0)
                for dim in data.dims
                if dim not in keep_dims
            }
            valid_shared_selectors = {
                k: v
                for k, v in shared_selectors.items()
                if k in data.dims and v < data.sizes[k]
            }

            shared_subset = data.isel(valid_shared_selectors)

            # Apply preprocessing for the shared spatial path.
            processed_data = self.preprocessor.apply_preprocessing(
                shared_subset, proc_kwargs, widget_prefix=prefix
            )

            secondary_proc_kwargs = self._secondary_preprocessing_kwargs(
                proc_kwargs,
                target_dim=target_dim,
                prefix=prefix,
            )
            secondary_data = processed_data
            if secondary_proc_kwargs != proc_kwargs:
                secondary_data = self.preprocessor.apply_preprocessing(
                    shared_subset,
                    secondary_proc_kwargs,
                    widget_prefix=prefix,
                )

            # 3. SPATIAL VIEW PROCESSING
            if len(spatial_dims_raw) < 2:
                payload.spatial_array = None
                actual_spatial_dims = []
            else:
                actual_spatial_dims = self._determine_spatial_dims(
                    processed_data, state, prefix
                )
                indexed = geometry._index_selection_coordinates(
                    processed_data, actual_spatial_dims
                )
                index_coord_names: dict[str, str] = {}
                selection_data = processed_data
                if indexed is not None:
                    selection_data, index_coord_names = indexed
                spatial_slice = self.get_spatial_slice(
                    selection_data,
                    state,
                    prefix=prefix,
                    spatial_dims=actual_spatial_dims,
                )

                display_spatial_dims = list(actual_spatial_dims)
                if state.spatial.rotation % 180 != 0:
                    display_spatial_dims = [
                        display_spatial_dims[1],
                        display_spatial_dims[0],
                    ]
                rotated_spatial_dims = list(display_spatial_dims)
                spatial_slice, display_spatial_dims = (
                    self._orient_spatial_slice_for_xyz(
                        spatial_slice,
                        display_spatial_dims,
                        state,
                        geometry_data=data,
                        scan=self.data_obj,
                    )
                )

                payload.spatial_source_indices = _shown_source_indices(
                    spatial_slice,
                    index_coord_names,
                    display_spatial_dims,
                    processed_data,
                    data,
                )
                spatial_slice = spatial_slice.drop_vars(
                    list(index_coord_names.values())
                )
                payload.spatial_dims = display_spatial_dims
                payload.spatial_unrotated_dims = list(actual_spatial_dims)
                if self._uses_xyz_coordinates(state):
                    payload.spatial_unrotated_dims = list(display_spatial_dims)
                if (
                    spatial_slice.attrs.get("spec_part_applied")
                    == state.spectral.spec_part
                ):
                    spatial_vals = np.asarray(spatial_slice.values)
                else:
                    spatial_vals = select_complex_part(
                        spatial_slice.values,
                        state.spectral.spec_part,
                    )
                payload.spatial_array = spatial_vals
                payload.spatial_coords = {
                    dim: spatial_slice.coords[dim].values
                    for dim in display_spatial_dims
                    if dim in spatial_slice.coords
                }
                payload.spatial_cmap = state.spatial.cmap
                payload.spatial_interpolation = state.spatial.interpolation
                payload.spatial_zsmooth = state.spatial.zsmooth_mode

                extent, origin, xlabel, ylabel = self._calculate_spatial_extent(
                    data, spatial_slice, display_spatial_dims, state
                )
                payload.spatial_extent = extent
                payload.spatial_origin = origin
                payload.labels["spatial_x"] = xlabel
                payload.labels["spatial_y"] = ylabel
                if state.spectral.spec_part == "Phase":
                    payload.labels["spatial_colorbar"] = "Phase [rad]"
                else:
                    payload.labels["spatial_colorbar"] = format_colorbar_label(
                        spatial_slice,
                        base_label="Intensity",
                    )
                payload.spatial_scalebar = self._build_spatial_scalebar(
                    spatial_slice=spatial_slice,
                    extent=extent,
                    plot_dim_x=display_spatial_dims[1],
                    state=state,
                )

                self._apply_anatomical_background(
                    payload,
                    state,
                    data,
                    spatial_dims=display_spatial_dims,
                    primary_spatial_slice=spatial_slice,
                )
                self._apply_spatial_clim(payload, state)
                self._apply_spatial_crosshair(
                    payload, state, display_spatial_dims, data=data
                )
                if spatial_overlay is not None and state.overlay.visible:
                    self._apply_spatial_overlay(
                        payload,
                        state,
                        spatial_overlay,
                        primary_data=data,
                        spatial_dims=actual_spatial_dims,
                        display_spatial_dims=display_spatial_dims,
                        primary_spatial_slice=spatial_slice,
                    )
                # The curves keep their dimension as in the secondary
                # view: averaged repetitions or combined channels reduce
                # the image only, the curve still runs along them.
                if (
                    include_curve_block
                    and target_dim is not None
                    and target_dim in secondary_data.dims
                ):

                    def curves_along(dims: tuple[str, ...]) -> np.ndarray:
                        return self._curve_block(
                            secondary_data,
                            state,
                            prefix=prefix,
                            curve_dims=dims,
                            spatial_dims=actual_spatial_dims,
                            rotated_spatial_dims=rotated_spatial_dims,
                            display_spatial_dims=display_spatial_dims,
                            geometry_data=data,
                        )

                    payload.curve_block = curves_along((target_dim,))
                    if (
                        time_dim is not None
                        and time_dim != target_dim
                        and time_dim in secondary_data.dims
                        and secondary_data.sizes[time_dim] > 1
                    ):
                        payload.curve_time_block = curves_along((target_dim, time_dim))

            payload.metadata["show_tooltips"] = state.global_settings.show_tooltips
            payload.metadata["show_grid_lines"] = state.global_settings.show_grid_lines

            # 4. SECONDARY VIEW PROCESSING
            self._prepare_secondary_view(
                payload,
                secondary_data,
                state,
                actual_spatial_dims,
                spectral_dims,
                overlays,
                preprocess_kwargs=secondary_proc_kwargs,
                source_data=shared_subset,
            )

            # Couple Y-Limits
            if state.spectral.couple_ylims and payload.spatial_clim:
                if state.spectral.spec_part in ["Real", "Imaginary", "Phase"]:
                    max_abs = max(abs(float(value)) for value in payload.spatial_clim)
                    if state.spectral.spec_part == "Phase":
                        max_abs = min(max_abs, float(np.pi))
                    payload.spectral_clim = (-max_abs, max_abs)
                else:
                    payload.spectral_clim = (
                        max(0.0, float(payload.spatial_clim[0])),
                        max(0.0, float(payload.spatial_clim[1])),
                    )
            _apply_spectral_display_range(payload, state)

        except Exception as e:
            import traceback

            payload.spatial_error = f"{e}\n{traceback.format_exc()}"

        return payload

    def _curve_block(
        self,
        processed_data: xr.DataArray,
        state: PlotState,
        *,
        prefix: str,
        curve_dims: tuple[str, ...],
        spatial_dims: list[str],
        rotated_spatial_dims: list[str],
        display_spatial_dims: list[str],
        geometry_data: xr.DataArray,
    ) -> np.ndarray:
        """Return the curves of every voxel of the displayed plane.

        The plane is selected, rotated and oriented by the same functions as
        the displayed map; only ``curve_dims`` stay whole and follow the two
        spatial axes in the given order. Rows and columns therefore match the
        map and payload.spatial_source_indices. The complex part follows the
        state like the map and the curve.
        """
        block = self.get_spatial_slice(
            processed_data,
            state,
            prefix=prefix,
            spatial_dims=spatial_dims,
            keep_dims=curve_dims,
        )
        block, _ = self._orient_spatial_slice_for_xyz(
            block,
            rotated_spatial_dims,
            state,
            geometry_data=geometry_data,
            scan=self.data_obj,
        )
        block = block.transpose(*display_spatial_dims, *curve_dims)
        if block.attrs.get("spec_part_applied") == state.spectral.spec_part:
            return np.asarray(block.values)
        return select_complex_part(block.values, state.spectral.spec_part)

    def _determine_secondary_target_dim(
        self, data: xr.DataArray, state: PlotState, spatial_dims: list[str]
    ) -> str | None:
        """Return the dimension the secondary (curve) view runs along.

        A requested ``state.spectral.secondary_dim`` present in ``data`` wins;
        otherwise :func:`default_curve_dim` chooses. When every remaining
        dimension has one entry, the first of them is used (a one-point curve).
        """
        if state.spectral.secondary_dim and state.spectral.secondary_dim in data.dims:
            return state.spectral.secondary_dim
        chosen = default_curve_dim(data, spatial_dims)
        if chosen is not None:
            return chosen
        return next((str(d) for d in data.dims if d not in spatial_dims), None)

    def _determine_spatial_dims(
        self, data: xr.DataArray, state: PlotState, prefix: str
    ) -> list[str]:
        """Return the displayed spatial dimensions as ``[y_dim, x_dim]``.

        Delegates to :func:`displayed_spatial_dims` with the view plane in
        ``state.spatial.slicing_dim``; ``prefix`` is not used.
        """
        return displayed_spatial_dims(data, state.spatial.slicing_dim)

    def _apply_spatial_clim(self, payload: RenderPayload, state: PlotState) -> None:
        """Calculate and set spatial color limits."""
        if payload.spatial_array is not None:
            dmin, dmax = (
                np.nanmin(payload.spatial_array),
                np.nanmax(payload.spatial_array),
            )
            if state.spatial.range_mode == "Percentage":
                r = dmax - dmin
                vmin = dmin + (state.spatial.vmin / 100.0) * r
                vmax = dmin + (state.spatial.vmax / 100.0) * r
            else:
                vmin, vmax = state.spatial.vmin, state.spatial.vmax
            payload.spatial_clim = (vmin, vmax)
            if state.spatial.min_to_nan:
                arr = payload.spatial_array.astype(float, copy=True)
                arr[arr < vmin] = np.nan
                payload.spatial_array = arr

    def _apply_spatial_crosshair(
        self,
        payload: RenderPayload,
        state: PlotState,
        spatial_dims: list[str],
        data=None,
    ) -> None:
        """Calculate crosshair coordinates."""
        if not state.global_settings.show_crosshair:
            payload.spatial_crosshair = None
            return
        if len(spatial_dims) != 2 or payload.spatial_array is None:
            return
        dim_y, dim_x = spatial_dims
        idx_x, idx_y = (
            state.spatial.slice_indices.get(dim_x, 0),
            state.spatial.slice_indices.get(dim_y, 0),
        )
        if (
            data is not None
            and state.spatial.show_image_space
            and self._uses_xyz_coordinates(state)
        ):
            xyz_point = self._xyz_display_point(data, state, spatial_dims)
            if xyz_point is not None:
                payload.spatial_crosshair = xyz_point
                return
        if data is not None and dim_x in data.coords and dim_y in data.coords:
            x_vals, y_vals = (
                extract_1d_coord(data, dim_x),
                extract_1d_coord(data, dim_y),
            )
            px, py = (
                float(x_vals[max(0, min(idx_x, len(x_vals) - 1))]),
                float(y_vals[max(0, min(idx_y, len(y_vals) - 1))]),
            )
        elif payload.spatial_extent:
            xmin, xmax, ymin, ymax = payload.spatial_extent
            h, w = payload.spatial_array.shape[:2]
            px, py = (
                xmin + (max(0, min(idx_x, w - 1)) + 0.5) * (xmax - xmin) / w,
                ymin + (max(0, min(idx_y, h - 1)) + 0.5) * (ymax - ymin) / h,
            )
        else:
            px, py = float(idx_x), float(idx_y)
        payload.spatial_crosshair = (px, py)

    def _prepare_secondary_view(
        self,
        payload: RenderPayload,
        data: xr.DataArray,
        state: PlotState,
        spatial_dims: list[str],
        spectral_dims: list[str],
        overlays: dict | None = None,
        prefix: str = "ds0",
        preprocess_kwargs: dict[str, Any] | None = None,
        source_data: xr.DataArray | None = None,
    ) -> None:
        """Prepare the 1D signal of the selected voxel for the secondary view.

        ``state.spatial.slice_indices`` holds source indices. ``data`` is the
        display-preprocessed signal, so the indices are translated to its
        order before selection; fit overlays keep the source indices because
        fit results are stored in source order. ``source_data`` is the array
        before display preprocessing; ``None`` takes the indices unchanged.
        """
        target_dim = self._determine_secondary_target_dim(data, state, spatial_dims)
        if target_dim is None:
            return
        full_selectors = {
            d: state.spatial.slice_indices.get(d, 0)
            for d in set(list(data.dims) + list(state.spatial.slice_indices.keys()))
            if d != target_dim
        }
        processed_selectors = _selectors_in_processed_order(
            full_selectors, source_data, data
        )
        valid_selectors = {
            k: v
            for k, v in processed_selectors.items()
            if k in data.dims and v < data.sizes[k]
        }
        final_sig = data.isel(valid_selectors)
        while final_sig.ndim > 1:
            final_sig = final_sig.isel(
                {next(d for d in final_sig.dims if d != target_dim): 0}
            )
        vals = np.asarray(final_sig.values)
        if final_sig.attrs.get("spec_part_applied") != state.spectral.spec_part:
            vals = select_complex_part(vals, state.spectral.spec_part)
        payload.spectral_y = vals
        if state.spectral.spec_part == "Phase":
            payload.labels["spectral_y"] = "Phase [rad]"
        else:
            payload.labels["spectral_y"] = format_colorbar_label(
                final_sig,
                base_label="Intensity",
            )

        if target_dim in ["acquisition_idx"]:
            self._handle_spectral_x_axis(
                payload,
                target_dim,
                final_sig.sizes[target_dim],
                state,
                source_data=final_sig,
            )
        else:
            x_vals_raw, display_coord_name, unit = self._extract_display_1d_coord(
                final_sig, target_dim
            )
            sort_order = np.argsort(x_vals_raw)
            payload.metadata["spectral_sort_order"] = sort_order.tolist()
            payload.spectral_y = vals[sort_order]
            payload.spectral_x = x_vals_raw[sort_order]
            payload.spectral_invert_x = False
            payload.spectral_label = f"{display_coord_name} [{unit}]"
            if (
                target_dim == "repetition_idx"
                and state.spectral.average_repetitions
                and len(payload.spectral_x) == final_sig.sizes[target_dim]
            ):
                rep_count = final_sig.sizes[target_dim]
                rep_start = max(0, int(state.spectral.rep_avg_start))
                rep_end = (
                    rep_count - 1
                    if state.spectral.rep_avg_end is None
                    else min(int(state.spectral.rep_avg_end), rep_count - 1)
                )
                rep_end = max(rep_start, rep_end)
                inv_sort = np.empty(len(sort_order), dtype=int)
                inv_sort[sort_order] = np.arange(len(sort_order))
                payload.metadata["avg_rep_range"] = {
                    "target_dim": target_dim,
                    "start_index": rep_start,
                    "end_index": rep_end,
                    "start_position": int(inv_sort[rep_start]),
                    "end_position": int(inv_sort[rep_end]),
                }

        self._handle_data_overlays(
            payload,
            data,
            state,
            full_selectors,
            target_dim,
            overlays,
            is_spectroscopic_target=(target_dim == "acquisition_idx"),
            prefix=prefix,
            preprocess_kwargs=preprocess_kwargs,
        )

        self._handle_fit_overlays(
            payload,
            final_sig,
            state,
            full_selectors,
            target_dim,
            overlays,
            is_spectroscopic_target=(target_dim == "acquisition_idx"),
            prefix=prefix,
        )

        if (
            target_dim in spectral_dims
            and state.spectral.selected_spectral_value is not None
            and np.isfinite(state.spectral.selected_spectral_value)
        ):
            payload.metadata["selected_spectral_value"] = float(
                state.spectral.selected_spectral_value
            )
        elif (
            target_dim in spectral_dims
            and state.spectral.selected_spectral_index is not None
        ):
            if (
                payload.spectral_x is not None
                and state.spectral.selected_spectral_index < len(payload.spectral_x)
            ):
                orig_idx = state.spectral.selected_spectral_index
                sort_order = np.asarray(
                    payload.metadata.get(
                        "spectral_sort_order", np.arange(len(payload.spectral_x))
                    )
                )
                # Build inverse index for O(1) lookup instead of O(n) np.where scan
                inv_sort = np.empty(len(sort_order), dtype=sort_order.dtype)
                inv_sort[sort_order] = np.arange(len(sort_order))
                payload.metadata["selected_spectral_value"] = float(
                    payload.spectral_x[inv_sort[orig_idx]]
                )
        elif payload.spectral_x is not None and len(payload.spectral_x) > 0:
            try:
                # Use the coordinates from the data being plotted, not the original primary_data
                # which might only have indices while the plotted data has physical units (e.g. time).
                # x_vals_raw was already extracted from final_sig (a slice of data) above.
                raw_coords = x_vals_raw
                curr_idx = max(
                    0,
                    min(
                        state.spatial.slice_indices.get(target_dim, 0),
                        len(raw_coords) - 1,
                    ),
                )
                payload.metadata["selected_spectral_value"] = float(
                    payload.spectral_x[
                        np.argmin(
                            np.abs(payload.spectral_x - float(raw_coords[curr_idx]))
                        )
                    ]
                )
            except Exception:
                pass

        selected_value = payload.metadata.get("selected_spectral_value")
        if selected_value is not None:
            readout = format_spectral_cursor_readout(
                selected_value, payload.spectral_label
            )
            if target_dim == "acquisition_idx" and not state.spectral.show_fid:
                other = self._cursor_frequency_in_other_unit(
                    float(selected_value),
                    state.spectral.spec_unit,
                    final_sig.sizes[target_dim],
                )
                if other is not None:
                    readout += " | " + format_spectral_cursor_readout(*other)
            payload.metadata["selected_spectral_readout"] = readout

    def _cursor_frequency_in_other_unit(
        self, value: float, unit: str | None, npoints: int
    ) -> tuple[float, str] | None:
        """Convert a spectral cursor frequency between ppm and Hz.

        Parameters
        ----------
        value : float
            Cursor frequency in ``unit``.
        unit : str or None
            Unit of the displayed frequency axis, ``"ppm"`` or ``"Hz"``.
        npoints : int
            Number of spectral samples of the displayed spectrum.

        Returns
        -------
        tuple of (float, str) or None
            The frequency and the axis label of the other unit, for example
            ``(-163.2, "Frequency (Hz)")``; None for another unit or when the
            scan gives no frequency axes. Both axes come from
            :func:`get_freq_axis` of the same scan, so the conversion is the
            one of the displayed axis.
        """
        units = {"ppm": "Hz", "Hz": "ppm"}
        if unit not in units or npoints < 2:
            return None
        other_unit = units[unit]
        cut_off = getattr(self.data_obj, "cut_off", 0)
        try:
            axis = np.asarray(get_freq_axis(self.data_obj, unit, cut_off, npoints))
            other_axis = np.asarray(
                get_freq_axis(self.data_obj, other_unit, cut_off, npoints)
            )
        except Exception:
            return None
        if (
            axis.shape != (npoints,)
            or other_axis.shape != (npoints,)
            or not np.all(np.isfinite(axis))
            or not np.all(np.isfinite(other_axis))
            or np.ptp(axis) == 0
        ):
            return None
        # Both frequency axes are linear in the sample index.
        slope, offset = np.polyfit(axis, other_axis, 1)
        return float(slope * value + offset), f"Frequency ({other_unit})"

    def _handle_data_overlays(
        self,
        payload: RenderPayload,
        primary_sig: xr.DataArray,
        state: PlotState,
        selectors: dict,
        target_dim: str,
        overlays: dict | None,
        is_spectroscopic_target: bool = False,
        prefix: str = "ds0",
        preprocess_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Process general DataArray overlays for the 1D signal view.

        ``selectors`` are source indices; each overlay is preprocessed like the
        main signal and the indices are translated to its processed order.
        """
        if not overlays:
            return

        # Prepare preprocessing keywords from state
        proc_kwargs = preprocess_kwargs or state.to_dict(prefix=prefix)

        for name, ov_data in overlays.items():
            # Only handle DataArrays that are not already handled as fit results
            if isinstance(ov_data, xr.DataArray):
                try:
                    # 1. Apply full preprocessing (standardize, FFT, linebroadening, part selection)
                    # This ensures the overlay matches the main data's spectral domain and part.
                    ov_data_processed = self.preprocessor.apply_preprocessing(
                        ov_data, proc_kwargs, widget_prefix=prefix
                    )

                    # 2. Select the source voxel in the overlay's processed order
                    ov_selectors = {
                        k: v
                        for k, v in _selectors_in_processed_order(
                            selectors, ov_data, ov_data_processed
                        ).items()
                        if k in ov_data_processed.dims
                        and v < ov_data_processed.sizes[k]
                    }
                    ov_sig = ov_data_processed.isel(ov_selectors)

                    # 3. Ensure we reduce to 1D along target_dim
                    while ov_sig.ndim > 1:
                        # Find a dimension that is not the target_dim to reduce
                        other_dims = [d for d in ov_sig.dims if d != target_dim]
                        if not other_dims:
                            break
                        ov_sig = ov_sig.isel({other_dims[0]: 0})

                    if target_dim not in ov_sig.dims:
                        continue

                    vals = ov_sig.values

                    # Handle spectral/temporal sorting if applicable
                    if not is_spectroscopic_target:
                        sort_order = payload.metadata.get("spectral_sort_order")
                        if sort_order is not None and len(vals) == len(sort_order):
                            vals = vals[sort_order]

                    payload.overlays[name] = vals

                except Exception as e:
                    debug_log_vis_core_model(f"Error processing overlay '{name}': {e}")

    def _handle_spectral_x_axis(
        self,
        payload: RenderPayload,
        spec_dim: str,
        npoints: int,
        state: PlotState,
        source_data: xr.DataArray | None = None,
    ) -> None:
        ref_axis = self._get_spectral_reference_axis(npoints, state, source_data)
        if ref_axis is not None:
            payload.spectral_x, payload.spectral_label, payload.spectral_invert_x = (
                ref_axis,
                (
                    "Time (ms)"
                    if state.spectral.show_fid
                    else (
                        f"Frequency ({state.spectral.spec_unit})"
                        if state.spectral.spec_unit
                        else "Index"
                    )
                ),
                (
                    False
                    if state.spectral.show_fid
                    else state.spectral.spec_unit == "ppm"
                ),
            )
        else:
            payload.spectral_x, payload.spectral_label, payload.spectral_invert_x = (
                np.arange(npoints),
                "Index",
                False,
            )

    def _apply_anatomical_background(
        self,
        payload: RenderPayload,
        state: PlotState,
        primary_data: xr.DataArray | None = None,
        spatial_dims: list[str] | None = None,
        primary_spatial_slice: xr.DataArray | None = None,
    ) -> None:
        if (
            self.anatomy_obj is None
            or not self.anatomy_preprocessor
            or not state.anatomy.visible
            or not state.spatial.show_image_space
        ):
            return
        payload.background_cmap, payload.background_opacity = (
            state.anatomy.cmap,
            state.anatomy.opacity,
        )
        try:
            anat_data = self.get_anatomy_data()
            if anat_data is None:
                return
            anat_data = self._ensure_xyz_coordinates(
                anat_data,
                state,
                scan=self.anatomy_obj,
            )
            use_xyz = self._uses_xyz_coordinates(state)
            anat_sel, keep_dims = self._secondary_plane_selection(
                anat_data,
                state,
                state.anatomy,
                primary_data=primary_data,
                primary_spatial_slice=primary_spatial_slice,
                primary_spatial_dims=spatial_dims,
            )
            payload.metadata["anatomy_plane_selection"] = {
                dim: int(index) for dim, index in anat_sel.items()
            }
            keep_dims = set(keep_dims)
            if state.anatomy.slice_warning:
                payload.metadata["anatomy_slice_warning"] = state.anatomy.slice_warning
            if (
                state.anatomy.auto_hide_non_overlapping
                and state.anatomy.slice_overlap_status == "primary_outside_secondary"
            ):
                payload.metadata["anatomy_auto_hidden_non_overlap"] = True
                return
            anat_slice = anat_data.isel(anat_sel)
            if (
                state.anatomy.combine_channels
                and "channel_idx" in anat_slice.dims
                and anat_slice.sizes["channel_idx"] > 1
            ):
                anat_slice = self.anatomy_preprocessor.apply_channel_combination(
                    anat_slice, method="rss"
                )
            if (
                state.anatomy.average_repetitions
                and "repetition_idx" in anat_slice.dims
                and anat_slice.sizes["repetition_idx"] > 1
            ):
                anat_slice = self.anatomy_preprocessor.average_repetitions(anat_slice)
            drop_dims = {d: 0 for d in anat_slice.dims if d not in keep_dims}
            if drop_dims:
                anat_slice = anat_slice.isel(drop_dims)
            anatomy_spatial_dims = [
                dim for dim in (spatial_dims or []) if dim in anat_slice.dims
            ]
            if len(anatomy_spatial_dims) != 2:
                anatomy_spatial_dims = list(anat_slice.dims[-2:])
            try:
                anat_slice = anat_slice.transpose(*anatomy_spatial_dims)
            except Exception:
                anatomy_spatial_dims = list(anat_slice.dims[-2:])
            if use_xyz:
                anat_slice, anatomy_spatial_dims = self._orient_spatial_slice_for_xyz(
                    anat_slice,
                    anatomy_spatial_dims,
                    state,
                    geometry_data=anat_data,
                    scan=self.anatomy_obj,
                )
            bg = (
                np.abs(anat_slice.values)
                if np.iscomplexobj(anat_slice.values)
                else anat_slice.values
            )
            payload.spatial_background = bg
            payload.metadata["background_interpolation"] = bool(
                state.anatomy.interpolate_non_slice_views
                and state.spatial.slicing_dim != "slice_idx"
            )
            global_min, global_max = float(np.nanmin(bg)), float(np.nanmax(bg))
            r = global_max - global_min
            if state.anatomy.range_mode == "Percentage" and r > 0:
                payload.background_clim = (
                    global_min + (state.anatomy.vmin / 100.0) * r,
                    global_min + (state.anatomy.vmax / 100.0) * r,
                )
            elif state.anatomy.range_mode in {"Min/Max", "Absolute"}:
                payload.background_clim = (
                    float(state.anatomy.vmin),
                    float(state.anatomy.vmax),
                )
            else:
                payload.background_clim = (global_min, global_max)
            try:
                if anat_slice.ndim >= 2:
                    if use_xyz:
                        xyz_extent = self._calculate_xyz_spatial_extent(
                            anat_slice,
                            anatomy_spatial_dims,
                            geometry_data=anat_data,
                            scan=self.anatomy_obj,
                        )
                        if xyz_extent is not None:
                            payload.background_extent = xyz_extent[0]
                    dim_y, dim_x = anatomy_spatial_dims[-2], anatomy_spatial_dims[-1]
                    if dim_x in anat_slice.coords and dim_y in anat_slice.coords:
                        x_vals, y_vals = (
                            extract_1d_coord(anat_slice, dim_x),
                            extract_1d_coord(anat_slice, dim_y),
                        )
                        if len(x_vals) > 1 and len(y_vals) > 1:
                            dx, dy = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]
                            if payload.background_extent is None:
                                payload.background_extent = [
                                    float(x_vals[0] - dx / 2),
                                    float(x_vals[-1] + dx / 2),
                                    float(y_vals[0] - dy / 2),
                                    float(y_vals[-1] + dy / 2),
                                ]
            except Exception:
                pass
        except Exception as exc:
            debug_log_vis_core_model(f"model.anatomical_background_error: {exc!r}")

    def _calculate_spatial_extent(
        self,
        full_data: xr.DataArray,
        spatial_slice: xr.DataArray,
        spatial_dims: list[str],
        state: PlotState,
    ) -> tuple[list[float], str, str, str]:
        if state.spatial.show_image_space and self._uses_xyz_coordinates(state):
            extent = geometry.xyz_spatial_extent(
                spatial_slice,
                spatial_dims,
                geometry_data=full_data,
                scan=self.data_obj,
            )
            if extent is not None:
                return extent
        alignment = self._get_alignment() if state.spatial.show_image_space else None
        return geometry._calculate_rps_spatial_extent(
            spatial_slice, spatial_dims, alignment=alignment
        )

    _build_spatial_scalebar = staticmethod(geometry._build_spatial_scalebar)

    @staticmethod
    def _coerce_sampling_axis(values: Any) -> np.ndarray | None:
        if values is None:
            return None
        try:
            axis = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if axis.size < 2 or not np.all(np.isfinite(axis)):
            return None
        return axis

    def _resolve_dynamic_reconstruction_axis(
        self,
        *,
        sig_slice: xr.DataArray,
        target_dim: str,
    ) -> np.ndarray | None:
        """Resolve a finite dynamic axis for model reconstruction fallback."""
        if target_dim in sig_slice.coords:
            axis = self._coerce_sampling_axis(sig_slice.coords[target_dim].values)
            if axis is not None:
                return axis

        try:
            axis_values, _, _ = self._extract_display_1d_coord(sig_slice, target_dim)
        except Exception:
            return None
        return self._coerce_sampling_axis(axis_values)

    def _resolve_spectroscopic_fit_sampling_axis(
        self,
        *,
        sig_slice: xr.DataArray,
        target_dim: str,
        fit_results: xr.Dataset,
        npoints: int,
    ) -> np.ndarray | None:
        """Build a spectroscopic reconstruction time axis with the requested length."""
        source_axes = [
            fit_results.attrs.get("sampling_time_axis_s"),
            fit_results.attrs.get("sampling_time_axis"),
            getattr(self.data_obj, "sampling_time_axis_s", None),
            getattr(self.data_obj, "sampling_time_axis", None),
        ]
        if target_dim in sig_slice.coords:
            source_axes.append(sig_slice.coords[target_dim].values)

        for values in source_axes:
            axis = self._coerce_sampling_axis(values)
            if axis is None:
                continue
            diffs = np.diff(axis)
            finite_diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
            if finite_diffs.size == 0:
                continue
            dt = float(np.median(np.abs(finite_diffs)))
            start = float(axis[0])
            return start + np.arange(int(npoints), dtype=float) * dt

        dwell_time_s = getattr(self.data_obj, "dwell_time_s", None)
        try:
            dwell_time_s = float(dwell_time_s)
        except (TypeError, ValueError):
            dwell_time_s = None
        if dwell_time_s is not None and np.isfinite(dwell_time_s) and dwell_time_s > 0:
            return np.arange(int(npoints), dtype=float) * dwell_time_s

        return None

    def _fit_x_axis_for_sampling_axis(
        self,
        *,
        sampling_axis_s: np.ndarray,
        state: PlotState,
    ) -> np.ndarray:
        """Return the display x-axis matching a reconstructed spectroscopic fit."""
        if state.spectral.show_fid:
            return np.asarray(sampling_axis_s, dtype=float) * 1000.0

        try:
            return get_freq_axis(
                self.data_obj,
                state.spectral.spec_unit,
                getattr(self.data_obj, "cut_off", 0),
                len(sampling_axis_s),
            )
        except Exception:
            if len(sampling_axis_s) > 1:
                dt = float(np.median(np.diff(np.asarray(sampling_axis_s, dtype=float))))
                if np.isfinite(dt) and dt != 0:
                    return np.fft.fftshift(
                        np.fft.fftfreq(len(sampling_axis_s), d=abs(dt))
                    )
            return np.arange(len(sampling_axis_s), dtype=float)

    @staticmethod
    def _metabolite_fit_amplitudes(
        fit_results: xr.Dataset,
        fit_selectors: dict[str, int],
        metabolites: list[Any],
    ) -> dict[str, float]:
        """Return absolute fitted amplitudes for the selected voxel/metabolites."""
        if (
            "fit_amps" not in fit_results
            or "metabolite" not in fit_results["fit_amps"].dims
        ):
            return {}

        try:
            amp_da = fit_results["fit_amps"]
            amp_selectors = {
                dim: idx
                for dim, idx in fit_selectors.items()
                if dim in amp_da.dims and idx < amp_da.sizes[dim]
            }
            if amp_selectors:
                amp_da = amp_da.isel(amp_selectors)
            labels = [str(m) for m in metabolites]
            available = [m for m in amp_da.metabolite.values if str(m) in set(labels)]
            if available:
                amp_da = amp_da.sel(metabolite=available)
            while amp_da.ndim > 1:
                other_dims = [d for d in amp_da.dims if d != "metabolite"]
                if not other_dims:
                    break
                amp_da = amp_da.isel({other_dims[0]: 0})
            values = np.abs(np.asarray(amp_da.values, dtype=float).reshape(-1))
            return {
                str(m): float(values[idx])
                for idx, m in enumerate(amp_da.metabolite.values)
                if idx < values.size and np.isfinite(values[idx])
            }
        except Exception:
            return {}

    def _format_fit_parameter_legend_labels(
        self,
        fit_results: xr.Dataset,
        fit_selectors: dict[str, int],
        metabolites: list[Any],
        state: PlotState,
    ) -> dict[str, str]:
        """Return detailed legend labels for selected-voxel metabolite fits."""
        required = {"fit_amps", "fit_freqs", "fit_t2s"}
        if not required.issubset(set(fit_results.data_vars)):
            return {}

        labels: dict[str, str] = {}

        def _scalar(variable: str, metabolite: Any) -> complex | float:
            """Return one value of ``variable`` for the selected voxel and metabolite.

            Dimensions that are neither selected by ``fit_selectors`` nor the
            metabolite dimension (for example a singleton slice axis) are
            reduced to their first entry so that a label is always produced.
            """
            data = fit_results[variable]
            isel = {
                k: v
                for k, v in fit_selectors.items()
                if k in data.dims and v < data.sizes[k]
            }
            data = data.isel(isel).sel(metabolite=metabolite)
            while data.ndim > 0:
                data = data.isel({data.dims[0]: 0})
            return data.values.item()

        for metabolite in metabolites:
            name = str(metabolite)
            try:
                amp = _scalar("fit_amps", metabolite)
                freq_hz = float(np.real(_scalar("fit_freqs", metabolite)))
                t2_ms = float(np.real(_scalar("fit_t2s", metabolite))) * 1000.0
            except Exception:
                continue

            amp_abs = float(np.abs(amp))
            freq_value = freq_hz
            freq_unit = "Hz"
            if state.spectral.spec_unit == "ppm":
                try:
                    freq_value = float(
                        freq_hz_to_ppm(freq_hz=freq_hz, scan=self.data_obj)
                    )
                    freq_unit = "ppm"
                except Exception:
                    freq_value = freq_hz
                    freq_unit = "Hz"

            labels[name] = (
                f"{name}: A={amp_abs:.3g}, "
                f"T2*={t2_ms:.1f} ms, "
                f"f={freq_value:.3g} {freq_unit}"
            )

        return labels

    def _handle_fit_overlays(
        self,
        payload: RenderPayload,
        sig_slice: xr.DataArray,
        state: PlotState,
        selectors: dict,
        target_dim: str,
        overlays: dict | None,
        is_spectroscopic_target: bool = False,
        prefix: str = "ds0",
    ) -> None:
        """Process fit reconstruction overlays."""
        fit_results = getattr(self.data_obj, "fit_results", None)
        if not isinstance(fit_results, xr.Dataset):
            candidates = list((overlays or {}).values())
            if isinstance(fit_results, dict):
                candidates.extend(fit_results.values())
            fit_results = None
            for ov_data in candidates:
                if not isinstance(ov_data, xr.Dataset) or not hasattr(ov_data, "hp"):
                    continue
                dynamic_dim = ov_data.attrs.get("dynamic_dim")
                if dynamic_dim and dynamic_dim != target_dim:
                    continue

                model_name = str(ov_data.attrs.get("model_name", "")).strip()
                is_known_fit_dataset = bool(model_name) or any(
                    v in ov_data.data_vars
                    for v in [
                        "fit_amps",
                        "fit_freqs",
                        "t1",
                        "t2",
                        "adc",
                        "K_trans",
                        "Ktrans_1c",
                        "Ktrans_2c",
                    ]
                )
                if is_known_fit_dataset:
                    fit_results = ov_data
                    break

        if (
            fit_results is not None
            and state.spectral.show_fit
            and hasattr(fit_results, "hp")
        ):
            try:
                # 1. Prepare selectors for the fit model
                fit_selectors = {
                    k: v
                    for k, v in selectors.items()
                    if k in fit_results.dims and v < fit_results.sizes[k]
                }

                # 2. Handle custom x-axis for non-spectroscopic temporal fits (e.g. DCE/T1)
                reco_kwargs = {"voxel_coords": fit_selectors}
                use_custom_x = False
                if (
                    state.spectral.fit_npoints is not None
                    and not is_spectroscopic_target
                ):
                    try:
                        x_vals_raw, _, _ = self._extract_display_1d_coord(
                            sig_slice, target_dim
                        )
                        if len(x_vals_raw) > 1:
                            dense_x = np.linspace(
                                np.nanmin(x_vals_raw),
                                np.nanmax(x_vals_raw),
                                state.spectral.fit_npoints,
                            )
                            reco_kwargs["dynamic_dim"] = dense_x
                            payload.metadata["fit_x_axis"] = dense_x
                            use_custom_x = True
                    except Exception:
                        pass
                elif state.spectral.fit_npoints is not None and is_spectroscopic_target:
                    sampling_axis = self._resolve_spectroscopic_fit_sampling_axis(
                        sig_slice=sig_slice,
                        target_dim=target_dim,
                        fit_results=fit_results,
                        npoints=state.spectral.fit_npoints,
                    )
                    if sampling_axis is not None:
                        reco_kwargs["sampling_time_axis_s"] = sampling_axis
                        payload.metadata["fit_x_axis"] = (
                            self._fit_x_axis_for_sampling_axis(
                                sampling_axis_s=sampling_axis,
                                state=state,
                            )
                        )
                        use_custom_x = True

                # 3. Reconstruct model prediction
                try:
                    reconstructed = fit_results.hp.reconstruct(**reco_kwargs)
                except Exception as recon_exc:
                    dynamic_axis = self._resolve_dynamic_reconstruction_axis(
                        sig_slice=sig_slice,
                        target_dim=target_dim,
                    )
                    sampling_axis = None
                    for attr_name in ("sampling_time_axis_s", "sampling_time_axis"):
                        sampling_axis = self._coerce_sampling_axis(
                            fit_results.attrs.get(attr_name)
                        )
                        if sampling_axis is not None:
                            break
                    if sampling_axis is None:
                        for attr_name in (
                            "sampling_time_axis_s",
                            "sampling_time_axis",
                        ):
                            sampling_axis = self._coerce_sampling_axis(
                                getattr(self.data_obj, attr_name, None)
                            )
                            if sampling_axis is not None:
                                break
                    if sampling_axis is None and target_dim in sig_slice.coords:
                        sampling_axis = self._coerce_sampling_axis(
                            sig_slice.coords[target_dim].values
                        )

                    reconstructed = None
                    fallback_kwargs: list[dict[str, np.ndarray]] = []
                    if sampling_axis is not None:
                        fallback_kwargs.append({"sampling_time_axis_s": sampling_axis})
                    if dynamic_axis is not None:
                        fallback_kwargs.append({"dynamic_dim": dynamic_axis})

                    for extra_kwargs in fallback_kwargs:
                        try:
                            reconstructed = fit_results.hp.reconstruct(
                                **{**extra_kwargs, **reco_kwargs},
                            )
                            break
                        except Exception:
                            continue

                    if reconstructed is None:
                        raise recon_exc

                # Ensure consistent dimension order
                for dim in list(reconstructed.dims):
                    if (
                        dim not in [target_dim, "metabolite"]
                        and reconstructed.sizes[dim] == 1
                    ):
                        reconstructed = reconstructed.squeeze(dim)

                # 4. Domain Logic: Determine if we should FFT the reconstructed model
                # Consistency check: is_spectroscopic_target matches target_dim == "acquisition_idx"
                is_spectral = is_spectroscopic_target and not state.spectral.show_fid

                actual_npoints = sig_slice.sizes[target_dim]

                # 5. Handle sorting for temporal non-spectroscopic views
                if not is_spectroscopic_target:
                    raw_fit_x = None
                    if target_dim in reconstructed.dims:
                        try:
                            raw_fit_x, _, _ = self._extract_display_1d_coord(
                                reconstructed, target_dim
                            )
                        except Exception:
                            raw_fit_x = None
                    if raw_fit_x is None:
                        # Find coordinate matching target_dim size
                        for dim in reconstructed.dims:
                            if dim in reconstructed.coords and reconstructed.sizes[
                                dim
                            ] == reconstructed.sizes.get(target_dim, -1):
                                raw_fit_x = reconstructed.coords[dim].values.astype(
                                    float
                                )
                                break
                    if raw_fit_x is not None:
                        if not use_custom_x:
                            sort_order_meta = payload.metadata.get(
                                "spectral_sort_order"
                            )
                            if sort_order_meta is not None and len(raw_fit_x) == len(
                                sort_order_meta
                            ):
                                raw_fit_x = raw_fit_x[np.array(sort_order_meta)]
                            if len(raw_fit_x) != actual_npoints:
                                raw_fit_x = None
                        if raw_fit_x is not None:
                            payload.metadata["fit_x_axis"] = raw_fit_x

                def process_curve(curve_da):
                    """Apply domain transform and part selection to a curve."""
                    if target_dim in curve_da.dims:
                        for d in list(curve_da.dims):
                            if d != target_dim:
                                curve_da = curve_da.isel({d: 0})

                    vals = curve_da.values.ravel()

                    # Padding/Truncation to match data length if not using custom grid
                    if not use_custom_x:
                        if len(vals) > actual_npoints:
                            vals = vals[:actual_npoints]
                        elif len(vals) < actual_npoints:
                            vals = np.pad(vals, (0, actual_npoints - len(vals)))

                    # Domain Transform
                    if is_spectral:
                        # Reconstructed model is usually in temporal domain (FID)
                        vals = np.fft.fftshift(np.fft.fft(vals))

                    # Temporal sorting
                    if not is_spectroscopic_target and not use_custom_x:
                        sort_order = payload.metadata.get("spectral_sort_order")
                        if sort_order is not None and len(vals) == len(sort_order):
                            vals = vals[sort_order]

                    # Signal Part selection
                    if state.spectral.spec_part == "Real":
                        return np.real(vals)
                    if state.spectral.spec_part == "Imaginary":
                        return np.imag(vals)
                    if state.spectral.spec_part == "Phase":
                        return np.angle(vals)
                    return np.abs(vals)

                selected_metabolites = state.spectral.fit_metabolites
                if (
                    "metabolite" in reconstructed.dims
                    and selected_metabolites is not None
                ):
                    selected_labels = {str(name) for name in selected_metabolites}
                    available_labels = [
                        metabolite
                        for metabolite in reconstructed.metabolite.values
                        if str(metabolite) in selected_labels
                    ]
                    reconstructed_for_display = (
                        reconstructed.sel(metabolite=available_labels)
                        if available_labels
                        else reconstructed.isel(metabolite=slice(0, 0))
                    )
                else:
                    reconstructed_for_display = reconstructed

                # 6. Sum/Individual metabolites
                fill_fit_overlays = float(state.spectral.fit_fill_alpha) > 0.0
                payload.metadata["fill_fit_overlays"] = fill_fit_overlays
                payload.metadata["fit_fill_alpha"] = float(
                    state.spectral.fit_fill_alpha
                )
                overlay_colors = dict(
                    payload.metadata.get("spectral_overlay_colors", {})
                )
                overlay_colors.setdefault("Fit", "red")
                payload.metadata["spectral_overlay_colors"] = overlay_colors

                if state.spectral.fit_mode == "Sum":
                    payload.overlays["Fit"] = process_curve(
                        reconstructed_for_display.sum("metabolite")
                        if "metabolite" in reconstructed_for_display.dims
                        else reconstructed_for_display
                    )
                    if fill_fit_overlays:
                        payload.metadata["spectral_overlay_fill_names"] = ["Fit"]
                    if (
                        state.spectral.show_fit_results_legend
                        and "metabolite" in reconstructed_for_display.dims
                    ):
                        # A single summed curve carries all fit parameters in
                        # one multi-line legend entry.
                        legend_labels = self._format_fit_parameter_legend_labels(
                            fit_results,
                            fit_selectors,
                            list(reconstructed_for_display.metabolite.values),
                            state,
                        )
                        if legend_labels:
                            payload.metadata["spectral_overlay_legend_labels"] = {
                                "Fit": "<br>".join(["Fit", *legend_labels.values()])
                            }
                else:
                    if "metabolite" in reconstructed_for_display.dims:
                        metabolite_values = list(
                            reconstructed_for_display.metabolite.values
                        )
                        # Colors follow the position in the complete fit
                        # metabolite coordinate, as the metabolite buttons do,
                        # so they do not depend on the voxel or the selection.
                        overlay_colors.update(
                            fit_metabolite_colors(
                                [str(m) for m in reconstructed.metabolite.values]
                            )
                        )
                        active_metabolite = state.overlay.active_metabolite
                        if active_metabolite is not None and str(active_metabolite) in {
                            str(m) for m in metabolite_values
                        }:
                            payload.metadata["spectral_overlay_highlight"] = str(
                                active_metabolite
                            )

                        amp_by_metabolite = {}
                        for m in metabolite_values:
                            try:
                                component_display_curve = process_curve(
                                    reconstructed_for_display.sel(metabolite=m)
                                )
                                finite_component = np.asarray(
                                    component_display_curve, dtype=float
                                )
                                finite_component = finite_component[
                                    np.isfinite(finite_component)
                                ]
                                if finite_component.size:
                                    amp_by_metabolite[str(m)] = float(
                                        np.nanmax(np.abs(finite_component))
                                    )
                            except Exception:
                                continue
                        if not amp_by_metabolite:
                            amp_by_metabolite = self._metabolite_fit_amplitudes(
                                fit_results,
                                fit_selectors,
                                metabolite_values,
                            )
                        ordered_metabolites = sorted(
                            metabolite_values,
                            key=lambda m: amp_by_metabolite.get(str(m), -np.inf),
                            reverse=True,
                        )
                        if fill_fit_overlays:
                            payload.metadata["spectral_overlay_fill_names"] = [
                                str(m) for m in ordered_metabolites
                            ]

                        payload.overlays["Fit"] = process_curve(
                            reconstructed_for_display.sum("metabolite")
                        )
                        separate = state.spectral.fit_mode == "Separate"
                        stacked_curve = None
                        fill_modes = dict(
                            payload.metadata.get("spectral_overlay_fill_modes", {})
                        )
                        cumulative_reconstructed = None
                        for m in ordered_metabolites:
                            component_reconstructed = reconstructed_for_display.sel(
                                metabolite=m
                            )
                            if separate:
                                # Each metabolite is drawn on its own,
                                # without adding the preceding components.
                                fill_modes[str(m)] = "tozeroy"
                                payload.overlays[str(m)] = process_curve(
                                    component_reconstructed
                                )
                                continue
                            if cumulative_reconstructed is None:
                                cumulative_reconstructed = component_reconstructed
                                fill_modes[str(m)] = "tozeroy"
                            else:
                                cumulative_reconstructed = (
                                    cumulative_reconstructed + component_reconstructed
                                )
                                fill_modes[str(m)] = "tonexty"

                            stacked_curve = process_curve(cumulative_reconstructed)
                            payload.overlays[str(m)] = stacked_curve.copy()
                        payload.metadata["spectral_overlay_stack_names"] = [
                            str(m) for m in ordered_metabolites
                        ]
                        payload.metadata["spectral_overlay_amplitudes"] = {
                            str(m): amp_by_metabolite.get(str(m))
                            for m in ordered_metabolites
                        }
                        if state.spectral.show_fit_results_legend:
                            legend_labels = self._format_fit_parameter_legend_labels(
                                fit_results,
                                fit_selectors,
                                ordered_metabolites,
                                state,
                            )
                            if legend_labels:
                                payload.metadata["spectral_overlay_legend_labels"] = (
                                    legend_labels
                                )
                        payload.metadata["spectral_overlay_fill_modes"] = fill_modes
                        payload.metadata["spectral_overlay_colors"] = overlay_colors
                    else:
                        payload.overlays["Fit"] = process_curve(
                            reconstructed_for_display
                        )
                        if fill_fit_overlays:
                            payload.metadata["spectral_overlay_fill_names"] = ["Fit"]

            except Exception as e:
                debug_log_vis_core_model(f"Error processing fit overlays: {e}")


def _apply_spectral_display_range(payload: RenderPayload, state: PlotState) -> None:
    """Limit the spectrum plot to ``state.spectral.display_range``.

    The window applies only when the payload shows a frequency axis in
    ``display_range_unit``. Without coupled limits the y-limits span the
    finite signal and overlay samples inside the window with a 5 % margin.
    """
    spectral = state.spectral
    window = spectral.display_range
    if (
        window is None
        or spectral.show_fid
        or spectral.spec_unit != spectral.display_range_unit
        or payload.spectral_label != f"Frequency ({spectral.spec_unit})"
    ):
        return
    low, high = sorted(float(value) for value in window)
    payload.spectral_x_range = (low, high)
    if payload.spectral_clim is not None or payload.spectral_x is None:
        return

    spectral_x = np.ravel(np.asarray(payload.spectral_x, dtype=float))
    fit_x = payload.metadata.get("fit_x_axis")
    curves = [(spectral_x, payload.spectral_y)]
    for name, curve in payload.overlays.items():
        x_axis = payload.metadata.get(f"overlay_x_{name}", fit_x)
        curves.append((spectral_x if x_axis is None else x_axis, curve))

    visible = []
    for x_axis, curve in curves:
        x_values = np.ravel(np.asarray(x_axis, dtype=float))
        try:
            y_values = np.ravel(np.asarray(curve, dtype=float))
        except (TypeError, ValueError):
            continue
        if y_values.shape != x_values.shape:
            continue
        inside = y_values[(x_values >= low) & (x_values <= high)]
        visible.append(inside[np.isfinite(inside)])
    values = np.concatenate(visible) if visible else np.array([])
    if values.size == 0:
        return
    y_min, y_max = float(values.min()), float(values.max())
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    payload.spectral_clim = (y_min - margin, y_max + margin)


FIT_RESIDUAL_OVERLAY = "Residual"
FIT_RESIDUAL_ZERO_OVERLAY = "ResidualZero"
_FIT_RESIDUAL_GAP_FRACTION = 0.05


def add_fit_residual_overlay(payload: RenderPayload, state: PlotState) -> None:
    """Draw the fit residual below the measured and fitted curves.

    The residual is the displayed signal minus the displayed summed fit
    (``payload.spectral_y - payload.overlays["Fit"]``), sample by sample, so
    it is the difference of the curves the user sees: of the real or
    imaginary parts, or of the magnitudes when Magnitude is shown. Spectrum
    and FID displays are handled alike. The residual keeps its scale and is
    shifted down so that its maximum lies 5 % of the signal span below the
    lowest signal or fit sample; a dashed gray line ``"ResidualZero"`` marks
    its zero level. The legend names the shift. The y-limits are set to
    cover signal, fit and residual with a 5 % margin; existing limits
    (coupled limits or a frequency window) keep their upper limit and only
    lower their lower limit when the residual would be cut off.

    Parameters
    ----------
    payload : RenderPayload
        Render payload after all layers contributed their overlays. Changed
        in place: ``overlays``, overlay metadata and possibly
        ``spectral_clim``.
    state : PlotState
        Viewer state; ``state.spectral.show_fit_residual`` switches the
        residual on, and Phase displays get none, because a phase difference
        wraps.

    Returns
    -------
    None
        Nothing is added without a ``"Fit"`` overlay sampled like the signal,
        for a masked ROI curve, or when no finite residual sample exists.
    """
    spectral = state.spectral
    fit = payload.overlays.get("Fit")
    if (
        not spectral.show_fit_residual
        or spectral.spec_part == "Phase"
        or fit is None
        or payload.spectral_y is None
        or payload.spectral_x is None
        or payload.metadata.get("masked_signal")
        or "overlay_x_Fit" in payload.metadata
    ):
        return
    try:
        signal = np.ravel(np.asarray(payload.spectral_y, dtype=float))
        fit_values = np.ravel(np.asarray(fit, dtype=float))
    except (TypeError, ValueError):
        return
    x_values = np.ravel(np.asarray(payload.spectral_x, dtype=float))
    if not signal.shape == fit_values.shape == x_values.shape:
        return
    residual = signal - fit_values
    curves = np.concatenate([signal, fit_values])
    curves = curves[np.isfinite(curves)]
    if curves.size == 0 or not np.any(np.isfinite(residual)):
        return

    span = float(curves.max() - curves.min()) or 1.0
    offset = (
        float(curves.min())
        - _FIT_RESIDUAL_GAP_FRACTION * span
        - float(np.nanmax(residual))
    )
    payload.overlays[FIT_RESIDUAL_OVERLAY] = residual + offset
    payload.overlays[FIT_RESIDUAL_ZERO_OVERLAY] = np.array([offset, offset])
    payload.metadata[f"overlay_x_{FIT_RESIDUAL_ZERO_OVERLAY}"] = np.array(
        [float(np.nanmin(x_values)), float(np.nanmax(x_values))]
    )
    if f"overlay_x_{FIT_RESIDUAL_OVERLAY}" not in payload.metadata and (
        payload.metadata.get("fit_x_axis") is not None
    ):
        payload.metadata[f"overlay_x_{FIT_RESIDUAL_OVERLAY}"] = x_values

    colors = dict(payload.metadata.get("spectral_overlay_colors", {}))
    colors[FIT_RESIDUAL_OVERLAY] = "dimgray"
    colors[FIT_RESIDUAL_ZERO_OVERLAY] = "darkgray"
    payload.metadata["spectral_overlay_colors"] = colors
    labels = dict(payload.metadata.get("spectral_overlay_legend_labels", {}))
    labels[FIT_RESIDUAL_OVERLAY] = f"Residual (data − fit), shifted by {offset:.3g}"
    labels[FIT_RESIDUAL_ZERO_OVERLAY] = "Residual zero"
    payload.metadata["spectral_overlay_legend_labels"] = labels

    lowest = float(np.nanmin(residual)) + offset
    if payload.spectral_clim is None:
        high = float(curves.max())
        margin = _FIT_RESIDUAL_GAP_FRACTION * (high - lowest)
        payload.spectral_clim = (lowest - margin, high + margin)
    else:
        low, high = (float(value) for value in payload.spectral_clim)
        if lowest < low:
            margin = _FIT_RESIDUAL_GAP_FRACTION * (high - lowest)
            payload.spectral_clim = (lowest - margin, high)
