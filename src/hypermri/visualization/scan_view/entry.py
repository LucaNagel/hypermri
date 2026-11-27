"""Entry points :func:`plot_scan` and :func:`plot_side_by_side` of the browser viewer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from hypermri._optional import _import_optional
from hypermri.utils.coordinates.excited_volume import excited_volume
from hypermri.visualization.metabolite_split import (
    resolve_metabolite_split,
    split_interleaved_metabolites,
)
from hypermri.visualization.render_data import (
    PlotDataModel,
    apply_display_defaults,
    apply_reference_layer_coordinates,
)
from hypermri.visualization.spatial_overlay import (
    active_overlay_source_catalog,
    apply_overlay_start_state,
    normalize_overlays,
    overlay_source_catalogs,
)
from hypermri.visualization.state import (
    PlotState,
    _state_field_was_supplied,
)
from hypermri.visualization.state_persistence import (
    load_plot_state,
)
from hypermri.visualization.volume_outline import (
    outline_source,
)

# The browser viewer needs these at construction; plotly ships the Plotly.js
# bundle that the widget sends to the browser.
for _dependency in ("ipywidgets", "anywidget", "cmap", "plotly"):
    _import_optional(_dependency)

from hypermri.visualization.scan_view.data_choice import (  # noqa: E402
    _chosen_curve_dim,
    _chosen_variable,
    _is_single_voxel,
    _map_variables,
    _with_single_voxel_plane,
)
from hypermri.visualization.scan_view.layers import (  # noqa: E402
    _drawable_overlay_names,
)
from hypermri.visualization.scan_view.options import (  # noqa: E402
    SCAN_SETTINGS_CONTEXT,
)
from hypermri.visualization.scan_view.side_by_side import (  # noqa: E402
    ARRANGEMENTS,
    SideBySideView,
)
from hypermri.visualization.scan_view.widget import ScanView  # noqa: E402


def plot_scan(
    data: xr.DataArray | xr.Dataset | Any,
    *,
    variable: str | None = None,
    anatomy: Any | None = None,
    overlay: Any | None = None,
    coordinate_mode: str | None = None,
    curve_dim: str | None = None,
    split_metabolites: bool | None = None,
    scan: Any | None = None,
    state: PlotState | None = None,
    settings_context: str | None = None,
    name: str = "",
    image_only: bool = False,
    slice_indices: Mapping[str, int] | None = None,
) -> ScanView:
    """Show a CSI scan or an image as an interactive map with one voxel's curve.

    The map shows the processed signal at the
    selected spectral position; the plot on the right shows the spectrum of
    the selected voxel, or its curve along ``curve_dim``. Single-voxel
    spectra without a spatial plane show only the spectrum (and the time
    course over repetitions). Images without
    ``acquisition_idx`` (anatomy, multi-echo or dynamic images) show the
    curve along echoes, repetitions or another dimension, or only the image
    when no such dimension exists. Clicking either plot or using the arrow keys changes
    the selection instantly in the browser; slice, repetition, channel,
    complex part, frequency unit and line broadening are recomputed in
    Python. Masks can be drawn on the map; for a scan they are stored in
    ``scan.masks`` (see :class:`ScanView`). An anatomical scan can be shown
    under the map and parameter maps such as fit results above it, matched
    to the shown CSI plane as in ``plot``.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset or object
        Samples with at least two of ``read_idx``, ``phase_idx`` and
        ``slice_idx``, with dimension ``acquisition_idx`` (FID or spectrum)
        for spectroscopy or without it for images, or a scan whose
        ``primary_data`` supplies them. Spectra with fewer spatial
        dimensions (single voxel) get dimensions of size one. A Dataset
        such as ``scan.fit_results`` is shown as maps: every variable with
        two spatial dimensions and without ``acquisition_idx`` (fitted
        spectra are left out) can be chosen in the browser menu
        "variable"; a ``metabolite`` dimension becomes a slider labelled
        with the metabolite names. Such fitted maps start with their colour
        bar shown; other data start without it (``ScanView.set_display``).
        Pass the scan as ``scan`` for scanner coordinates and to store
        masks on it. The samples are not modified.
    variable : str or None, optional
        Dataset variable shown first. None shows ``fit_amps``, else
        ``fit_freqs``, else the first map variable. Only valid with a
        Dataset. Real-valued maps start with the complex part ``"Real"``
        unless ``state`` is given or saved settings exist, so that negative
        values such as frequency offsets keep their sign.
    anatomy : object or None, optional
        Anatomical reference scan (or a DataArray with scanner coordinates
        ``x``, ``y``, ``z``) drawn under the CSI map. It may also be a
        second image under an image, for example two anatomies of the same
        session. The plane that matches the shown plane is resampled
        automatically; the map is drawn above it with opacity ``state.anatomy.opacity`` (default 0.6,
        changeable in the browser or with
        :meth:`ScanView.set_anatomy_opacity`). For single-voxel (PRESS,
        STEAM) or slice spectroscopy the image panel instead shows the
        anatomy slice nearest the excited volume with the outline of the
        voxel or slab on it (see
        :func:`~hypermri.utils.coordinates.excited_volume.excited_volume`);
        a slider chooses the anatomy slice. This needs the scan itself (or
        ``scan``) for the voxel geometry and an anatomy with scanner
        coordinates. None shows no anatomy.
    overlay : object, dict or None, optional
        Parameter maps drawn above the CSI map, as for ``plot``: a
        DataArray, a Dataset such as fit results (one layer per variable,
        named ``"overlay_<variable>"``) or a dict of named DataArrays and
        Datasets. Layers need the spatial dimensions of a map and no
        ``acquisition_idx``; a ``metabolite`` dimension is selectable. Fit
        amplitudes are shown first, with opacity 1 and the whole value range
        as colour range; layer, metabolite, opacity and range can be changed
        in the browser or with :meth:`ScanView.set_overlay` and
        :meth:`ScanView.set_overlay_opacity`. The overlay is matched to the
        shown plane through scanner coordinates (or, without them, its
        index grid). None shows no overlay.
    coordinate_mode : {"rps", "xyz"} or None, optional
        Spatial axes of the map: ``"rps"`` uses the scan's read/phase
        positions, ``"xyz"`` scanner coordinates in mm. None keeps the
        state's mode, except that an ``anatomy`` or ``overlay`` switches it
        to ``"xyz"`` so that the layers are aligned in scanner space (as in
        ``plot``).
    curve_dim : str or None, optional
        Dimension the curve panel runs along, for example
        ``"repetition_idx"`` for the time course of every voxel of a dynamic
        CSI scan. The map then shows the selected curve position, and the
        spectral position becomes a slider. Any dimension with more than one
        entry outside the shown plane is allowed. None keeps
        ``state.spectral.secondary_dim`` (for example saved with
        :meth:`ScanView.save_settings`) when it is such a dimension, and
        otherwise shows the spectrum along ``acquisition_idx``. For images,
        None chooses the first of inversion, b-value, echo, repetition and
        channel dimensions present (see
        :data:`~hypermri.visualization.view_dimensions.CURVE_DIM_PREFERENCE`),
        or shows the image without a curve panel. Single-voxel spectra show
        no image panel; their repetitions appear in the time course panel,
        or as the curve with ``curve_dim="repetition_idx"``.
    split_metabolites : bool or None, optional
        Split repetitions that alternate between two metabolites into the
        dimension ``metabolite_idx`` (even repetitions are metabolite 0, odd
        ones metabolite 1), shown with its own slider, as ``plot`` does; see
        :func:`~hypermri.visualization.metabolite_split.split_interleaved_metabolites`.
        None splits the interleaved FISP sequence
        (:data:`~hypermri.visualization.metabolite_split.INTERLEAVED_METABOLITE_SEQUENCE`)
        automatically and nothing else; True splits any data with at least
        two repetitions; False never splits. The scan's data are not
        modified, and masks keep the scan's spatial grid.
    scan : object or None, optional
        Acquisition context (frequency axis, geometry) for an explicit
        DataArray. None uses the array itself. Do not combine with a scan
        supplied as ``data``.
    state : PlotState or None, optional
        Display state to start from; None uses the settings saved on the scan
        with :meth:`ScanView.save_settings`, or the shared viewer defaults when
        none are saved. The object is completed in place and kept by the
        widget. Without a given or saved state the view starts at the voxel
        of the shown plane with the strongest signal (largest magnitude along
        its curve) and with map colours over 0-80 % of the shown map, so
        that values above 80 % of the maximum share the top colour.
    settings_context : str or None, optional
        Key of the saved settings on the scan. None uses ``"plot_scan"``,
        separate from the settings of ``plot``. Views opened with the same
        key restore what :meth:`ScanView.save_settings` stored under it.
    name : str, optional
        Name shown above the plots, for example the scan's name; see
        :attr:`ScanView.name`. The empty string shows no name.
    image_only : bool, optional
        Show only the image, without the curve and time panels, for a view
        whose curve position follows another view of the same scan
        (:meth:`ScanView.link_frames`).
    slice_indices : mapping of str to int or None, optional
        Start position as source indices by dimension name, for example
        ``{"slice_idx": 3, "read_idx": 5, "phase_idx": 7}``. Names must be
        dimensions of the shown data (after a metabolite split) other than
        ``acquisition_idx``; values are positions, not coordinate values.
        Given positions override saved ones, and the view then does not move
        to the strongest voxel. None keeps the saved or automatic start.

    Returns
    -------
    ScanView
        Widget to display in a notebook. Construction prepares one curve
        block eagerly (one FFT of the displayed plane).

    Raises
    ------
    ValueError
        If a Dataset has no map variable or ``variable`` is not one of
        them, ``variable`` is given without a Dataset, ``scan`` is
        combined with a scan, the scan has no
        ``primary_data``, ``coordinate_mode`` is not ``"rps"`` or ``"xyz"``,
        ``curve_dim`` names no dimension with more than one entry outside the
        shown plane, ``split_metabolites=True`` is given for data with fewer
        than two repetitions, the data have no curves in a spatial plane, the
        overlay offers no map without ``acquisition_idx`` or the overlay
        state asks for a difference mode, or a single-voxel scan shown with
        an anatomy has no voxel or slice geometry, an unsupported patient
        position or an anatomy without scanner coordinates.
    TypeError
        If the resolved data are not an xarray.DataArray.
    ImportError
        If the optional visualization dependencies are unavailable.

    Examples
    --------
    >>> view = plot_scan(scan)  # doctest: +SKIP
    >>> view  # displays the widget in Jupyter  # doctest: +SKIP
    >>> view.state.spatial.slice_indices  # selected voxel  # doctest: +SKIP
    >>> view.add_mask("tumour")  # doctest: +SKIP
    >>> view.draw_polygon([(0, 0), (4, 0), (4, 4)], mask="tumour")  # doctest: +SKIP
    >>> view.masks["tumour"].attrs["weighted_voxel_count"]  # doctest: +SKIP
    >>> plot_scan(scan, curve_dim="repetition_idx")  # time course  # doctest: +SKIP
    >>> plot_scan(multi_echo_image)  # image with a curve over echoes  # doctest: +SKIP
    >>> plot_scan(press_scan)  # single-voxel spectrum and time course  # doctest: +SKIP
    >>> plot_scan(scan, anatomy=t2_scan)  # anatomy under the map  # doctest: +SKIP
    >>> view = plot_scan(scan, overlay=fit_results)  # fit maps above  # doctest: +SKIP
    >>> view.set_overlay("overlay_fit_amps", metabolite="lactate")  # doctest: +SKIP
    >>> plot_scan(scan.fit_results, scan=scan)  # fitted maps  # doctest: +SKIP
    >>> plot_scan(fisp_scan, split_metabolites=True)  # doctest: +SKIP
    """
    variables = None
    if isinstance(data, xr.Dataset):
        variables = _map_variables(data)
        if not variables:
            raise ValueError(
                "plot_scan found no map in the Dataset: every variable has "
                "acquisition_idx or fewer than two of read_idx, phase_idx and "
                f"slice_idx (variables {list(data.data_vars)})."
            )
        data = variables[_chosen_variable(variables, variable)]
    elif variable is not None:
        raise ValueError(
            "plot_scan variable is only valid with a Dataset such as "
            "scan.fit_results; a DataArray or scan has one variable."
        )
    if isinstance(data, xr.DataArray):
        source = data if scan is None else scan
    else:
        if scan is not None:
            raise ValueError(
                "plot_scan scan is only valid with an explicit DataArray; "
                "supply the scan once."
            )
        source = data
        data = getattr(source, "primary_data", None)
        if data is None:
            raise ValueError(
                "plot_scan requires a DataArray or a scan with primary_data."
            )
    if not isinstance(data, xr.DataArray):
        raise TypeError("plot_scan requires an xarray.DataArray.")
    split_available, split = resolve_metabolite_split(source, data, split_metabolites)
    if split_metabolites and not split_available:
        raise ValueError(
            "plot_scan split_metabolites=True needs at least two entries along "
            f"repetition_idx; the data have dimensions {dict(data.sizes)}."
        )
    metabolite_split = None
    if split_available:
        acquired = _with_single_voxel_plane(data)
        metabolite_split = (
            acquired,
            _with_single_voxel_plane(split_interleaved_metabolites(data)),
        )
        data = metabolite_split[1 if split else 0]
    else:
        data = _with_single_voxel_plane(data)
    if coordinate_mode is not None and coordinate_mode not in ("rps", "xyz"):
        raise ValueError(
            "plot_scan coordinate_mode must be 'rps' or 'xyz' (or None for the "
            f"automatic choice); got {coordinate_mode!r}."
        )
    outline = None
    if anatomy is not None and _is_single_voxel(data):
        # No map to draw the anatomy under: the anatomy is shown on its own
        # with the outline of the excited voxel or slab.
        outline = outline_source(anatomy, excited_volume(source))
        anatomy = None
    model = PlotDataModel(source, anatomy_obj=anatomy)
    context = settings_context or SCAN_SETTINGS_CONTEXT
    state_given = state is not None
    if state is None:
        state = PlotState()
        if not isinstance(source, xr.DataArray):
            state._loaded_from_persistence = load_plot_state(source, state, context)
    if (
        variables is not None
        and not state_given
        and not getattr(state, "_loaded_from_persistence", False)
        and not any(values.dtype.kind == "c" for values in variables.values())
    ):
        # Fitted maps are real; the magnitude would drop the sign of
        # frequencies or phases.
        state.spectral.spec_part = "Real"
    if (
        variables is not None
        and not state_given
        and not _state_field_was_supplied(state, "spatial.show_colorbar")
    ):
        # The values of a fitted map need their scale; a signal map does not.
        state.spatial.show_colorbar = True
    layer_chosen_by_caller = state_given and bool(state.overlay.active_variable)
    if coordinate_mode is not None:
        state.spatial.coordinate_mode = coordinate_mode
    apply_reference_layer_coordinates(
        state,
        has_reference_layers=anatomy is not None or overlay is not None,
        coordinate_mode_explicit=coordinate_mode is not None,
    )
    overlays, source_objects = normalize_overlays(
        overlay, state.spatial.coordinate_mode, geometry_reference=data
    )
    overlays, overlay_sources = overlay_source_catalogs(
        overlays,
        source_objects,
        requested=state.overlay.data_source,
        coordinate_mode=state.spatial.coordinate_mode,
        geometry_reference=data,
    )
    apply_display_defaults(state, data, model, has_overlays=bool(overlays))
    if slice_indices:
        state.spatial.slice_indices = {
            **state.spatial.slice_indices,
            **_checked_slice_indices(data, slice_indices),
        }
    if not state_given and not getattr(state, "_loaded_from_persistence", False):
        # The map starts at 0-80 % of its own values, so that weaker signal
        # is visible next to the strongest voxel, also for spectroscopic data
        # whose sequence default (Min/Max over the plane) suits plot.
        state.spatial.range_mode = "Percentage"
        state.spatial.vmin, state.spatial.vmax = 0.0, 80.0
    state.spectral.secondary_dim = _chosen_curve_dim(
        data, state, curve_dim, explicit=curve_dim is not None
    )
    apply_overlay_start_state(state.overlay, overlays)
    drawable = _drawable_overlay_names(overlays)
    if (
        not layer_chosen_by_caller
        and drawable
        and state.overlay.active_variable not in drawable
    ):
        # The shared default or a saved layer may be one with spectra, which
        # only plot shows, or one the overlay passed now does not offer.
        state.overlay.active_variable = drawable[0]
    _, catalog = active_overlay_source_catalog(
        overlays, overlay_sources, state.overlay.active_variable
    )
    state.overlay.data_source = None if catalog is None else catalog.current_key
    view = ScanView(
        data,
        model,
        state,
        overlays,
        overlay_sources=overlay_sources,
        settings_context=context,
        outline=outline,
        variables=variables,
        metabolite_split=metabolite_split,
        image_only=image_only,
    )
    if (
        not state_given
        and not slice_indices
        and not getattr(state, "_loaded_from_persistence", False)
    ):
        view._select_strongest_voxel()
    view.name = name
    return view


def _checked_slice_indices(
    data: xr.DataArray, slice_indices: Mapping[str, int]
) -> dict[str, int]:
    """Return ``slice_indices`` as integers after checking names and ranges."""
    checked = {}
    for dim, index in slice_indices.items():
        if dim not in data.dims or dim == "acquisition_idx":
            choices = [d for d in data.dims if d != "acquisition_idx"]
            raise ValueError(
                f"plot_scan slice_indices: {dim!r} is not a dimension of the "
                f"shown data; choose from {choices}."
            )
        size = data.sizes[dim]
        if isinstance(index, bool) or int(index) != index or not 0 <= index < size:
            raise ValueError(
                f"plot_scan slice_indices: {dim}={index!r} must be an integer "
                f"from 0 to {size - 1}."
            )
        checked[dim] = int(index)
    return checked


def plot_side_by_side(
    *images: Any,
    show_slices: bool = True,
    share_masks: bool = True,
    arrangement: str = "rows",
    under_menus: bool = True,
) -> SideBySideView:
    """Show several images in rows, each with the others' slices.

    Typical use is an axial and a coronal anatomy of the same session, for
    example to draw a mask on one while seeing where the other's slice lies,
    together with a CSI map whose spectrum needs the width of a row.
    Every image gets its own :class:`ScanView` with all controls, a coloured
    bar above it, and (with ``show_slices``) the shown slices of the other
    images as bands in their colours (see :meth:`ScanView.show_slices_of`).
    The bands are hidden until the display switch "other slices" is
    checked (``view.set_display(show_slice_lines=True)``). They follow when
    another view shows a new slice, also while its slice slider is dragged,
    and the mouse pointer over one image is marked at the same scanner
    position in the others (see :meth:`ScanView.link_pointer`). Masks are
    shared (see :meth:`ScanView.link_masks`): one mask tool draws on every
    image, and each scan stores the mask on its own grid. Only the first
    image shows the inputs of the mask tool; the others show the chosen mask
    with its volume on their own grid.

    Without given rows, the views that show only an image stand in the
    first row, and each view with a spectrum or time course (a curve panel)
    gets a row of its own below, in the given order. Pass one list of rows
    to place the views yourself.

    Parameters
    ----------
    *images : ScanView or object, or one list of rows
        Views from :func:`plot_scan`, or anything :func:`plot_scan` accepts
        (a scan or a DataArray), which is plotted with default settings.
        Pass a view to use options such as ``anatomy`` (another image drawn
        underneath) for one image. A single list of lists, such as
        ``[[axial, sagittal], [csi]]``, gives the rows from top to bottom.
    show_slices : bool, optional
        Draw the shown slices of the other images on each image and share
        the mouse pointer. This needs scanner coordinates ``x``, ``y``, ``z``
        for every image.
    share_masks : bool, optional
        Link the masks of all images with :meth:`ScanView.link_masks`. This
        needs scanner coordinates and a grid with at least two dimensions of
        more than one voxel for every image; pass False for images that
        cannot hold masks.
    arrangement : {"rows", "side by side", "stacked"}, optional
        Starting arrangement: the rows described above, all views in one
        row, or one view per row. The buttons "arrange" above the views
        change it while they are shown.
    under_menus : bool, optional
        Show a menu "under" per view that lays it under another view: the
        upper view is made anew with the lower image underneath (as with
        ``plot_scan(..., anatomy=...)``), in the plane of the lower image's
        slices, and the lower view is hidden until "none" is chosen again.
        The opacity slider of the upper view fades between both. See
        :meth:`SideBySideView.set_underlays`.

    Returns
    -------
    SideBySideView
        The views below the buttons "arrange"; ``box.views`` gives them in
        the given order and ``box.rows`` the views of each shown row, and
        ``box.arrangement`` changes the arrangement. In a row each view gets a share of the width in proportion to what it needs
        for its image at the full plot height (plus room for a curve); in a
        narrow row all images are lowered alike, so they keep one height,
        and a curve keeps at least 40 % of its view. The shares follow a new
        plane, a rotation and the plot height. The plots of the views in one
        row line up at the bottom.

    Raises
    ------
    ValueError
        If fewer than two images are given, *arrangement* is unknown, a
        given row is empty or not a list, or ``show_slices`` or ``share_masks`` is set and an image has
        no scanner coordinates or no grid for masks.

    Examples
    --------
    Two anatomies in the first row, the CSI with its spectrum below:

    >>> box = plot_side_by_side(axial_scan, sagittal_scan, csi_scan)  # doctest: +SKIP
    >>> (axial, sagittal), (csi,) = box.rows  # doctest: +SKIP
    >>> box.arrangement = "side by side"  # doctest: +SKIP

    The axial anatomy laid under the CSI map, as with the menu "under":

    >>> box.set_underlays({0: 2})  # doctest: +SKIP

    The same views in rows chosen by hand:

    >>> box = plot_side_by_side([[axial, csi], [sagittal]])  # doctest: +SKIP
    """
    if arrangement not in ARRANGEMENTS:
        raise ValueError(
            f"plot_side_by_side arrangement: {arrangement!r} is unknown; "
            f"choose from {list(ARRANGEMENTS)}."
        )
    given_rows = _given_side_by_side_rows(images)
    rows = [
        [image if isinstance(image, ScanView) else plot_scan(image) for image in row]
        for row in (given_rows if given_rows is not None else [images])
    ]
    views = [view for row in rows for view in row]
    if len(views) < 2:
        raise ValueError(
            f"plot_side_by_side needs at least two images; got {len(views)}."
        )
    return SideBySideView(
        views,
        None if given_rows is None else rows,
        arrangement=arrangement,
        show_slices=show_slices,
        share_masks=share_masks,
        under_menus=under_menus,
    )


def _given_side_by_side_rows(images: tuple[Any, ...]) -> list[list[Any]] | None:
    """Return the rows given as one list of lists, or None for plain images."""
    if not (len(images) == 1 and isinstance(images[0], (list, tuple))):
        return None
    rows = list(images[0])
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(
                f"plot_side_by_side: row {number} must be a non-empty list of "
                f"images, as in [[axial, sagittal], [csi]]; got {row!r}."
            )
    return [list(row) for row in rows]
