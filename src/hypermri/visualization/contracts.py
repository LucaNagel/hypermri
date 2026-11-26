"""Render payload: the prepared map, curve and overlays of one viewer update."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from hypermri.visualization.colormap_policy import (
    DEFAULT_ANATOMY_CMAP,
    DEFAULT_OVERLAY_CMAP,
    DEFAULT_SPATIAL_CMAP_NON_SPECTRO,
)


@dataclass
class RenderPayload:
    """Store prepared image, spectrum and overlay data for rendering.

    Parameters
    ----------
    spatial_array : Optional[np.ndarray], optional
        Rendered two-dimensional intensity samples in spatial_dims order; None has no image.
    spatial_coords : Dict[str, np.ndarray], optional
        Named displayed coordinate vectors/grids; units are recorded in units.
    spatial_source_indices : Dict[str, np.ndarray], optional
        Positional index in the input data of every displayed row and column
        of spatial_array (and curve_block), keyed by the two spatial_dims
        names; ``spatial_source_indices[spatial_dims[0]][row]`` is the source
        index of displayed row ``row``. It includes rotation, scanner
        orientation and any display reversal, so masks map between the shown
        plane and the data grid without coordinate values. Empty when no map
        is shown or when display preprocessing replaced the spatial grid
        (k-space view).
    spatial_dims : List[str], optional
        Image dimension names in displayed vertical then horizontal order.
    spatial_unrotated_dims : List[str], optional
        Original image dimension names before display rotation.
    spatial_extent : Optional[List[float]], optional
        Horizontal/vertical image bounds as xmin, xmax, ymin, ymax in displayed axis units.
    spatial_origin : str, optional
        Image origin convention: upper or lower.
    spatial_background : Optional[np.ndarray], optional
        Optional two-dimensional anatomical background in its displayed orientation.
    background_extent : Optional[List[float]], optional
        Background bounds as xmin, xmax, ymin, ymax in displayed axis units.
    background_cmap : str, optional
        Colormap identifier for the anatomical background.
    background_opacity : float, optional
        Background opacity from zero to one.
    background_clim : Optional[tuple], optional
        Lower and upper anatomical display limits in native signal units.
    spatial_cmap : str, optional
        Colormap identifier for the primary image.
    spatial_clim : Optional[tuple], optional
        Lower and upper primary image display limits in native signal units.
    spatial_interpolation : bool, optional
        Whether displayed image samples are interpolated.
    spatial_zsmooth : str, optional
        Plotly image smoothing selection, normally off, fast or best.
    spatial_crosshair : Optional[tuple], optional
        Optional selected horizontal/vertical position in plot coordinates.
    spatial_scalebar : Optional[Dict[str, Any]], optional
        Prepared scalebar geometry, label and appearance settings.
    spectral_x : Optional[np.ndarray], optional
        One-dimensional displayed frequency/time positions, with unit supplied by labels/units.
    spectral_y : Optional[np.ndarray], optional
        Displayed selected signal samples matching spectral_x, in native or display-scaled units.
    spectral_label : Optional[str], optional
        Optional displayed spectral axis label.
    spectral_invert_x : bool, optional
        Reverse the displayed spectral horizontal axis when True.
    spectral_clim : Optional[tuple], optional
        Lower and upper spectral vertical limits in displayed signal units.
    spectral_x_range : Optional[tuple], optional
        Ascending visible horizontal window in spectral_x units; None shows
        the full axis.
    curve_block : Optional[np.ndarray], optional
        Processed signals of every displayed voxel along the curve dimension
        (the dimension of spectral_x: ``acquisition_idx`` for spectra, or
        for example ``repetition_idx``), shape (n_rows, n_columns,
        n_curve). The first two axes follow spatial_array and spatial_dims;
        the last follows spectral_x. Units and complex-part selection equal
        spatial_array and spectral_y, so ``curve_block[:, :, k]`` holds the
        values of the map at curve index k and ``curve_block[row, column]``
        those of that voxel's curve. The block keeps the processing
        precision (float32 for complex64 samples); spatial_array may be
        promoted to float64 with unchanged values. Only filled on request
        and when the curve dimension survives the display processing of the
        map; None otherwise.
    curve_time_block : Optional[np.ndarray], optional
        curve_block with one more axis along a requested time dimension
        (for example ``repetition_idx``), shape (n_rows, n_columns,
        n_curve, n_time), in the order of that dimension. Same units,
        complex part and precision as curve_block;
        ``curve_time_block[..., t]`` equals curve_block at time index t.
        Only filled when curve_block is and the time dimension survives the
        display processing; None otherwise.
    overlays : Dict[str, Any], optional
        Named prepared overlay payloads such as fit curves and masks.
    spatial_overlays : Dict[str, np.ndarray], optional
        Named two-dimensional overlay arrays in their displayed spatial order.
    spatial_overlay_extents : Dict[str, List[float]], optional
        Optional per-overlay bounds overriding spatial_extent.
    overlay_cmap : str, optional
        Colormap identifier shared by prepared spatial overlays.
    overlay_alpha : float, optional
        Overlay opacity from zero to one.
    overlay_clim : Optional[tuple], optional
        Lower and upper overlay display limits in native overlay units.
    title : str, optional
        Prepared figure title.
    units : Dict[str, str], optional
        Named physical/display units for axis and layer data.
    labels : Dict[str, str], optional
        Prepared axis and layer display labels.
    metadata : Dict[str, Any], optional
        Additional prepared display context; carries no implicit analysis
        operation. ``"overlay_plane_selection"`` maps the overlay dimension
        normal to its shown plane to the zero-based overlay slice drawn (an
        empty dict when the overlay has no such dimension);
        ``"anatomy_plane_selection"`` does the same for the anatomy drawn
        under the map.
    spatial_error : Optional[str], optional
        Optional user-facing reason why a spatial view is unavailable.
    spectral_error : Optional[str], optional
        Optional user-facing reason why a spectral view is unavailable.

    Notes
    -----
    Construction stores arrays and mappings by reference without validation, copying,
    reconstruction, fitting or widget creation. The data preparation owner supplies
    orientation, units, precision and scientific choices before rendering. This
    mutable payload does not transform samples or infer missing physical geometry.
    """

    # Spatial View Data
    spatial_array: Optional[np.ndarray] = None
    spatial_coords: Dict[str, np.ndarray] = field(default_factory=dict)
    spatial_source_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    spatial_dims: List[str] = field(default_factory=list)  # Rotated [dim_y, dim_x]
    spatial_unrotated_dims: List[str] = field(
        default_factory=list
    )  # Raw [dim_y, dim_x]
    spatial_extent: Optional[List[float]] = None  # [xmin, xmax, ymin, ymax]
    spatial_origin: str = "upper"  # "upper" or "lower"
    spatial_background: Optional[np.ndarray] = (
        None  # Greyscale background image (e.g. Anatomy)
    )
    background_extent: Optional[List[float]] = None  # Extent for the background image
    background_cmap: str = DEFAULT_ANATOMY_CMAP
    background_opacity: float = 1.0
    background_clim: Optional[tuple] = None  # [vmin, vmax] for anatomy colormap
    spatial_cmap: str = DEFAULT_SPATIAL_CMAP_NON_SPECTRO
    spatial_clim: Optional[tuple] = None  # [vmin, vmax] for colormap
    spatial_interpolation: bool = False
    spatial_zsmooth: str = "best"
    spatial_crosshair: Optional[tuple] = None  # [x, y] in plot coordinates
    spatial_scalebar: Optional[Dict[str, Any]] = None

    # Spectral View Data
    spectral_x: Optional[np.ndarray] = None
    spectral_y: Optional[np.ndarray] = None
    spectral_label: Optional[str] = None
    spectral_invert_x: bool = False
    spectral_clim: Optional[tuple] = None  # [ymin, ymax] for spectral Y-axis
    spectral_x_range: Optional[tuple] = None
    curve_block: Optional[np.ndarray] = None
    curve_time_block: Optional[np.ndarray] = None

    # Overlays (e.g., masks, fit curves)
    # dict mapping overlay_name -> overlay_data (e.g., np.ndarray)
    overlays: Dict[str, Any] = field(default_factory=dict)

    # Spatial Overlays (e.g., metabolite maps, combined channels)
    # dict mapping overlay_name -> 2D np.ndarray
    spatial_overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    # Optional per-overlay extents, overriding the primary spatial extent
    spatial_overlay_extents: Dict[str, List[float]] = field(default_factory=dict)
    overlay_cmap: str = DEFAULT_OVERLAY_CMAP
    overlay_alpha: float = 0.4
    overlay_clim: Optional[tuple] = None

    # Metadata for display

    title: str = ""

    units: Dict[str, str] = field(default_factory=dict)

    labels: Dict[str, str] = field(
        default_factory=dict
    )  # e.g. {'x': 'ppm', 'y': 'Amplitude'}

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Arbitrary extra info (e.g. selected_x)

    # Error messages
    spatial_error: Optional[str] = None
    spectral_error: Optional[str] = None
