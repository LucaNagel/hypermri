"""Mutable display-state contracts, independent of scientific arrays and widgets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from hypermri.visualization.colormap_policy import (
    DEFAULT_ANATOMY_CMAP,
    DEFAULT_OVERLAY_CMAP,
    DEFAULT_SPATIAL_CMAP_NON_SPECTRO,
)


@dataclass
class SpatialState:
    """State for spatial display (images).

    Parameters
    ----------
    slice_indices : Dict[str, int], optional
        Selected zero-based indices keyed by labelled dimension.
    slicing_dim : Optional[str], optional
        Dimension normal to the current image plane; None selects automatically.
    view_dims : List[str], optional
        Displayed image dimensions in vertical then horizontal order.
    cmap : str, optional
        Colormap identifier for the corresponding image layer.
    data_source : Optional[str], optional
        Selected representation on the layer source object; None uses its default.
    clim : Optional[tuple], optional
        Effective lower and upper intensity limits in native signal units.
    vmin : float, optional
        Lower display control interpreted according to range_mode.
    vmax : float, optional
        Upper display control interpreted according to range_mode.
    range_mode : str, optional
        Display-limit interpretation: Percentage, Min/Max or supported Absolute mode.
    interpolation : bool, optional
        Whether the image renderer interpolates displayed samples.
    show_image_space : bool, optional
        Show image-space data instead of k-space when True.
    show_masks : bool, optional
        Whether stored mask overlays are visible.
    overlay_alpha : float, optional
        Mask overlay opacity from zero to one.
    rotation : int, optional
        Display rotation in degrees, normally 0, 90, 180 or 270.
    zsmooth_mode : str, optional
        Plotly image smoothing setting: off, fast or best.
    min_to_nan : bool, optional
        Mask intensities below the effective lower limit for transparent rendering.
    show_scalebar : bool, optional
        Display a scalebar when physical spacing is available.
    show_colorbar : bool, optional
        Draw the colour bar of the map in the browser view. Off by
        default; ``plot_scan`` switches it on for fitted maps.
    coordinate_mode : str, optional
        Spatial coordinate selection: rps or xyz.
    scalebar_fixed_value : Optional[float], optional
        Optional fixed scalebar length in the displayed physical axis unit.
    scalebar_color : str, optional
        Matplotlib-compatible scalebar color.
    scalebar_location : str, optional
        Scalebar placement name within the image.
    scalebar_fontsize : int, optional
        Scalebar label size in points.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    # Slice indices for each dimension
    slice_indices: Dict[str, int] = field(default_factory=dict)
    # The dimension currently being sliced (determines the view perspective)
    slicing_dim: Optional[str] = None
    # Active dimensions for the 2D view (e.g., ['phase_idx', 'read_idx'])
    view_dims: List[str] = field(default_factory=lambda: ["phase_idx", "read_idx"])
    # Colormap
    cmap: str = DEFAULT_SPATIAL_CMAP_NON_SPECTRO
    # Selected source on the primary scan object.
    data_source: Optional[str] = None
    # Intensity scaling (min, max) - derived from range_mode and sliders
    clim: Optional[tuple] = None
    vmin: float = 0.0
    vmax: float = 100.0
    # Range Mode (e.g., "Percentage", "Min/Max", "Absolute")
    range_mode: str = "Percentage"
    # Interpolation for image display (True=bilinear/bicubic, False=nearest)
    interpolation: bool = False
    # Show in image space (True) or k-space (False)
    show_image_space: bool = True
    # Mask overlays visibility
    show_masks: bool = True
    # Alpha for overlays
    overlay_alpha: float = 0.5
    # Rotation in degrees (0, 90, 180, 270)
    rotation: int = 0
    # Smoothing mode for image display ("off", "fast", "best")
    zsmooth_mode: str = "best"
    # When True, all values below the computed vmin are set to NaN (transparent)
    min_to_nan: bool = True
    # Show a spatial scalebar when physical axis spacing is available
    show_scalebar: bool = False
    # Colour bar beside the map of the browser view
    show_colorbar: bool = False
    # Coordinate system used for spatial extents: "rps" or "xyz"
    coordinate_mode: str = "rps"
    # Optional fixed scalebar length in the axis unit (typically mm)
    scalebar_fixed_value: Optional[float] = None
    # Scalebar appearance
    scalebar_color: str = "w"
    scalebar_location: str = "lower right"
    scalebar_fontsize: int = 15


@dataclass
class SpectralState:
    """State for spectral display (plots).

    Parameters
    ----------
    line_broadening_hz : float, optional
        Requested exponential line broadening in Hz.
    denoise_components : int, optional
        Retained component count; zero disables denoising.
    average_repetitions : bool, optional
        Average the selected repetition interval before display.
    cum_sum_repetitions : bool, optional
        Display a cumulative sum along the repetition axis.
    combine_channels : bool, optional
        Combine receive channels before display.
    combine_method : str, optional
        Channel-combination method, normally rss.
    show_spectral_domain : bool, optional
        Whether spectral-domain display is selected.
    show_fid : bool, optional
        Display acquired temporal data when available.
    spec_part : str, optional
        Displayed signal component: Magnitude, Real, Imaginary or Phase.
    spec_unit : str, optional
        Frequency display unit: ppm or Hz.
    display_range : Optional[tuple], optional
        Visible frequency window ``(start, end)`` of the spectrum plot in
        ``display_range_unit``; the order of the two values does not matter.
        It applies while the frequency axis is shown in that unit and keeps
        the window when another voxel is selected. Without coupled y-limits
        the y-axis is scaled to the signal and fit curves inside the window.
        None shows the full axis.
    display_range_unit : str, optional
        Unit of ``display_range``: ``"ppm"`` (default) or ``"Hz"``.
    scale_factor : float, optional
        Multiplicative display scaling of the selected signal.
    npoints : Optional[int], optional
        Optional Fourier/display point count.
    selected_spectral_index : Optional[int], optional
        Selected zero-based sample index for spectral maps.
    selected_spectral_value : Optional[float], optional
        Selected position in the displayed unit, such as Hz, ppm or ms.
    secondary_dim : Optional[str], optional
        Dimension used in the secondary plot; None selects automatically.
    time_course_dim : Optional[str], optional
        Dimension of the time-course panel of the browser view, for example
        ``repetition_idx``; None selects automatically.
    time_course_range : Optional[tuple], optional
        Inclusive spectral index range ``(first, last)`` averaged for the
        time course; None follows the selected spectral position.
    couple_ylims : bool, optional
        Use spatial intensity controls for secondary signal limits. False by
        default, so the spectrum's y-axis follows the shown signal.
    show_fit : bool, optional
        Show available fitted spectra.
    show_fit_results_legend : bool, optional
        Include fitted parameter values in selected-voxel curve legends.
    show_fit_residual : bool, optional
        Draw the residual (signal minus summed fit) below the signal and fit
        curves whenever a fit is shown. True by default.
    fill_fit_overlay : bool, optional
        Retained legacy fill flag; fit_fill_alpha controls the visible fill.
    fit_fill_alpha : float, optional
        Fitted-curve fill opacity from zero to one.
    fit_npoints : Optional[int], optional
        Optional reconstructed fit sample count; None uses measured length.
    fit_mode : str, optional
        Fit display selection: ``"Sum"`` draws the summed fit,
        ``"Individual"`` draws cumulative (stacked) metabolite curves and
        ``"Separate"`` draws every metabolite curve on its own.
    fit_metabolites : Optional[List[str]], optional
        Metabolite names to display; None includes all available components.
    rep_avg_start : int, optional
        Inclusive first repetition index for averaging.
    rep_avg_end : Optional[int], optional
        Inclusive last repetition index; None selects the last repetition.
    show_secondary_metabolite_curve : bool, optional
        Include the complementary split-acquisition metabolite curve.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    # Preprocessing
    line_broadening_hz: float = 0.0
    denoise_components: int = 0
    average_repetitions: bool = False
    cum_sum_repetitions: bool = False
    combine_channels: bool = True
    combine_method: str = "rss"
    # Domain (temporal vs spectral)
    show_spectral_domain: bool = True
    show_fid: bool = False
    # Part of complex signal ("Magnitude", "Real", "Imaginary", "Phase")
    spec_part: str = "Magnitude"
    # Spectral Unit ("ppm" or "Hz")
    spec_unit: str = "ppm"
    # Visible frequency window (start, end) in display_range_unit
    display_range: Optional[tuple] = None
    display_range_unit: str = "ppm"
    # Scaling factor
    scale_factor: float = 1.0
    # Number of points for FFT/display
    npoints: Optional[int] = None
    # Currently selected index along the spectral dimension (for frequency maps)
    selected_spectral_index: Optional[int] = None
    # Currently selected value along the spectral dimension (e.g. ppm, Hz, or Time in ms)
    selected_spectral_value: Optional[float] = None
    # Dimension to display in the secondary plot (None = Auto)
    secondary_dim: Optional[str] = None
    # Time-course panel of the browser view: dimension and spectral index range
    time_course_dim: Optional[str] = None
    time_course_range: Optional[tuple] = None
    # Couple spectral Y-limits to spatial Vmin/Vmax
    couple_ylims: bool = False
    # Show fit results on the spectrum
    show_fit: bool = True
    # Include selected-voxel fit parameters in spectral overlay legend labels
    show_fit_results_legend: bool = False
    # Draw the residual (signal - fit) below the curves when a fit is shown
    show_fit_residual: bool = True
    # Fill reconstructed fit spectra under the curve. Kept for compatibility;
    # the visible UI uses fit_fill_alpha as the effective fill control.
    fill_fit_overlay: bool = False
    # Alpha used for filled reconstructed fit spectra
    fit_fill_alpha: float = 0.0
    # Number of points for the fit reconstruction (None = same as measured)
    fit_npoints: Optional[int] = None
    # Fit display mode ("Sum", "Individual" = cumulative, or "Separate")
    fit_mode: str = "Sum"
    # List of metabolites to show in the fit (None = show all)
    fit_metabolites: Optional[List[str]] = None
    # Repetition range for averaging (inclusive indices; None = last rep)
    rep_avg_start: int = 0
    rep_avg_end: Optional[int] = None
    # For split two-metabolite acquisitions, show the other metabolite's curve.
    show_secondary_metabolite_curve: bool = True


@dataclass
class SecondarySliceState:
    """Slice selection and overlap state shared by secondary image layers.

    Parameters
    ----------
    slice_indices : Dict[str, int], optional
        Selected zero-based indices keyed by labelled dimension.
    slicing_dim : Optional[str], optional
        Dimension normal to the current image plane; None selects automatically.
    slice_warning : Optional[str], optional
        Optional user-facing plane-overlap warning.
    auto_hide_non_overlapping : bool, optional
        Hide a secondary plane without overlap instead of showing the nearest one.
    slice_overlap_status : Optional[str], optional
        Optional machine-readable overlap status for layer visibility.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    slice_indices: Dict[str, int] = field(default_factory=dict)
    slicing_dim: Optional[str] = None
    slice_warning: Optional[str] = None
    auto_hide_non_overlapping: bool = False
    slice_overlap_status: Optional[str] = None


@dataclass
class OverlayState(SecondarySliceState):
    """State for spatial overlays.

    Parameters
    ----------
    slice_indices : Dict[str, int], optional
        Selected zero-based indices keyed by labelled dimension.
    slicing_dim : Optional[str], optional
        Dimension normal to the current image plane; None selects automatically.
    slice_warning : Optional[str], optional
        Optional user-facing plane-overlap warning.
    auto_hide_non_overlapping : bool, optional
        Hide a secondary plane without overlap instead of showing the nearest one.
    slice_overlap_status : Optional[str], optional
        Optional machine-readable overlap status for layer visibility.
    active_variable : Optional[str], optional
        Selected overlay variable name.
    active_metabolite : Optional[str], optional
        Selected overlay metabolite when that dimension is available.
    visible : bool, optional
        Whether the layer is shown.
    cmap : str, optional
        Colormap identifier for the corresponding image layer.
    data_source : Optional[str], optional
        Selected representation on the layer source object; None uses its default.
    vmin : float, optional
        Lower display control interpreted according to range_mode.
    vmax : float, optional
        Upper display control interpreted according to range_mode.
    range_mode : str, optional
        Display-limit interpretation: Percentage, Min/Max or supported Absolute mode.
    mode : str, optional
        Overlay blend mode: normal, subtract or percent_difference.
    alpha : float, optional
        Overlay opacity from zero to one.
    smoothing : str, optional
        Plotly heatmap smoothing of the drawn overlay: ``"off"`` (each overlay
        voxel as a square), ``"fast"`` or ``"best"``. Only the drawing is
        smoothed; the overlay values stay unchanged.
    show_tooltips : bool, optional
        Add the overlay value to the hover label of the map.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    # Currently selected variable to display from the overlay dataset(s)
    active_variable: Optional[str] = None
    # Currently selected metabolite for heatmap display (if variable has metabolite dim)
    active_metabolite: Optional[str] = None
    # Visibility
    visible: bool = True
    # Colormap for overlays
    cmap: str = DEFAULT_OVERLAY_CMAP
    # Selected source on the active overlay scan object.
    data_source: Optional[str] = None
    # Intensity scaling
    vmin: float = 0.0
    vmax: float = 100.0
    range_mode: str = "Percentage"
    # Overlay blend mode for spatial overlay rendering.
    mode: str = "normal"  # "normal" | "subtract" | "percent_difference"
    # Alpha transparency
    alpha: float = 0.4
    smoothing: str = "off"
    show_tooltips: bool = False


@dataclass
class AnatomyState(SecondarySliceState):
    """State for the anatomical background image.

    Parameters
    ----------
    slice_indices : Dict[str, int], optional
        Selected zero-based indices keyed by labelled dimension.
    slicing_dim : Optional[str], optional
        Dimension normal to the current image plane; None selects automatically.
    slice_warning : Optional[str], optional
        Optional user-facing plane-overlap warning.
    auto_hide_non_overlapping : bool, optional
        Hide a secondary plane without overlap instead of showing the nearest one.
    slice_overlap_status : Optional[str], optional
        Optional machine-readable overlap status for layer visibility.
    visible : bool, optional
        Whether the layer is shown.
    opacity : float, optional
        Primary-image opacity above the anatomy, from zero to one.
    cmap : str, optional
        Colormap identifier for the corresponding image layer.
    data_source : Optional[str], optional
        Selected representation on the layer source object; None uses its default.
    vmin : float, optional
        Lower display control interpreted according to range_mode.
    vmax : float, optional
        Upper display control interpreted according to range_mode.
    range_mode : str, optional
        Display-limit interpretation: Percentage, Min/Max or supported Absolute mode.
    combine_channels : bool, optional
        Combine receive channels before display.
    average_repetitions : bool, optional
        Average the selected repetition interval before display.
    interpolate_non_slice_views : bool, optional
        Smooth anatomy in Read and Phase views while preserving acquired Slice views.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    visible: bool = True
    opacity: float = 0.6  # alpha of the MRI layer on top of the anatomy
    cmap: str = DEFAULT_ANATOMY_CMAP
    # Selected source on the anatomy scan object.
    data_source: Optional[str] = None
    vmin: float = 0.0
    vmax: float = 100.0
    range_mode: str = "Percentage"  # "Percentage" or "Min/Max"
    combine_channels: bool = True
    average_repetitions: bool = True
    interpolate_non_slice_views: bool = True


@dataclass
class GlobalState:
    """Overall plotter state.

    Parameters
    ----------
    active_dataset_id : str, optional
        Active dataset identifier used by the viewer context.
    theme : str, optional
        Display theme, normally light or dark.
    layout_mode : str, optional
        Horizontal or vertical viewer layout.
    show_tooltips : bool, optional
        Show data hints in viewer controls.
    show_grid_lines : bool, optional
        Show spatial and spectral axis grids.
    show_crosshair : bool, optional
        Show the currently selected voxel crosshair.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    # Data identifiers
    active_dataset_id: str = "ds0"
    # General UI settings
    theme: str = "light"
    # Layout mode (horizontal vs vertical)
    layout_mode: str = "horizontal"
    # Tooltips/Data hints
    show_tooltips: bool = False
    # Show axis grid lines in spatial/spectral plots
    show_grid_lines: bool = False
    # Show the current voxel crosshair overlay
    show_crosshair: bool = True


@dataclass
class PlotState:
    """Root state object combining all sub-states.

    Parameters
    ----------
    spatial : SpatialState, optional
        Mutable spatial display settings, created independently per root state.
    spectral : SpectralState, optional
        Mutable spectral display settings, created independently per root state.
    overlay : OverlayState, optional
        Mutable overlay display settings, created independently per root state.
    anatomy : AnatomyState, optional
        Mutable anatomical background settings, created independently per root state.
    global_settings : GlobalState, optional
        Mutable global UI settings, created independently per root state.

    Notes
    -----
    This mutable dataclass stores display choices without validating, transforming
    scientific arrays or constructing widgets. Defaults are owned here; constructing
    an instance does not trigger data loading, reconstruction or fitting.
    """

    spatial: SpatialState = field(default_factory=SpatialState)
    spectral: SpectralState = field(default_factory=SpectralState)
    overlay: OverlayState = field(default_factory=OverlayState)
    anatomy: AnatomyState = field(default_factory=AnatomyState)
    global_settings: GlobalState = field(default_factory=GlobalState)

    def to_dict(self, prefix: str = "ds0") -> Dict[str, Any]:
        """Export the retained flat subset of legacy viewer settings.

        Parameters
        ----------
        prefix : str, optional
            Dataset prefix attached to display-setting keys. Slice-index keys retain
            their dimension names without a prefix.

        Returns
        -------
        dict
            Fresh dictionary of legacy spatial/spectral settings. The ppm selector is a
            Boolean for compatibility. This is not a complete persistence snapshot;
            save_plot_state owns complete recursive snapshots. State is not mutated.
        """
        d = {}
        # Spatial
        for dim, idx in self.spatial.slice_indices.items():
            d[dim] = idx
        d[f"{prefix}_show_image_space"] = self.spatial.show_image_space
        d[f"{prefix}_range_mode"] = self.spatial.range_mode
        d[f"{prefix}_vmin"] = self.spatial.vmin
        d[f"{prefix}_vmax"] = self.spatial.vmax
        d[f"{prefix}_interpolate_image"] = self.spatial.interpolation
        d[f"{prefix}_zsmooth_mode"] = self.spatial.zsmooth_mode

        # Spectral
        d[f"{prefix}_linebroadening"] = self.spectral.line_broadening_hz
        d[f"{prefix}_denoise_components"] = self.spectral.denoise_components
        d[f"{prefix}_average_repetitions"] = self.spectral.average_repetitions
        d[f"{prefix}_cum_sum_repetitions"] = self.spectral.cum_sum_repetitions
        d[f"{prefix}_rep_avg_start"] = self.spectral.rep_avg_start
        d[f"{prefix}_rep_avg_end"] = self.spectral.rep_avg_end
        d[f"{prefix}_combine_channels"] = self.spectral.combine_channels
        d[f"{prefix}_combine_method"] = self.spectral.combine_method
        d[f"{prefix}_spec_part"] = self.spectral.spec_part
        d[f"{prefix}_spec_unit"] = (
            self.spectral.spec_unit == "ppm"
        )  # Legacy expects bool for ppm
        d[f"{prefix}_view_toggle"] = self.spectral.show_fid
        if self.spectral.npoints:
            d[f"{prefix}_npoints"] = self.spectral.npoints

        return d


def _state_field_was_supplied(state: PlotState, path: str) -> bool:
    """Check explicit options and recognized fields from the saved snapshot."""
    return path in getattr(
        state, "_explicit_state_overrides", set()
    ) or path in getattr(state, "_loaded_state_overrides", set())


def _primary_range_was_supplied(state: PlotState) -> bool:
    """Return True when the map colour limits must be kept at viewer start.

    Limits count as supplied when ``spatial.vmin`` or ``spatial.vmax`` came
    from explicit options or saved settings, or, for a state that was not
    restored from saved settings, when they differ from the default 0-100.
    Otherwise a viewer in an absolute range mode starts from the full value
    range of the shown data.
    """
    if any(
        _state_field_was_supplied(state, f"spatial.{field}")
        for field in ("vmin", "vmax")
    ):
        return True
    if getattr(state, "_loaded_from_persistence", False):
        return False
    return not (
        np.isclose(float(state.spatial.vmin), 0.0)
        and np.isclose(float(state.spatial.vmax), 100.0)
    )
