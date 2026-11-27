"""Variable, metabolite split, view plane, rotation and k-space of the shown data.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_DataChoice`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.analysis.spatial_filtering import (
    SpatialFilterOptions,
    apply_spatial_filter,
    spatial_to_kspace,
)
from hypermri.visualization.contracts import RenderPayload
from hypermri.visualization.spatial_overlay import (
    preferred_fit_variable,
)
from hypermri.visualization.state import (
    PlotState,
)
from hypermri.visualization.view_dimensions import (
    curve_dim_choices,
    default_curve_dim,
    displayed_spatial_dims,
    view_plane_choices,
)

_import_optional("anywidget")
_import_optional("cmap")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    _SPATIAL_FILTER_WIDTHS,
    SPATIAL_DIMS,
    SPECTRAL_DIM,
)


def _with_single_voxel_plane(data: xr.DataArray) -> xr.DataArray:
    """Give spectra without a spatial plane a plane of one voxel.

    Single-voxel spectroscopy (PRESS, STEAM, single pulse, ISIS, NSPECT)
    may lack read, phase or slice dimensions. Missing ones are added with
    size one, in the order read, phase, slice, until two exist, so that the
    one display path of :class:`ScanView` applies; the browser then hides
    the image panel. The added dimensions have no coordinates. Images and
    data that already have a plane are returned unchanged; the samples are
    never copied or modified.
    """
    present = [dim for dim in SPATIAL_DIMS if dim in data.dims]
    if SPECTRAL_DIM not in data.dims or len(present) >= 2:
        return data
    missing = [dim for dim in SPATIAL_DIMS if dim not in data.dims]
    return data.expand_dims({dim: 1 for dim in missing[: 2 - len(present)]})


def _is_single_voxel(data: xr.DataArray) -> bool:
    """Return whether spectra have no spatial resolution (one voxel or slab)."""
    return SPECTRAL_DIM in data.dims and all(
        data.sizes[dim] == 1 for dim in SPATIAL_DIMS if dim in data.dims
    )


def _map_variables(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return the variables of a fit result that can be shown as a map.

    A map has at least two of ``read_idx``, ``phase_idx`` and ``slice_idx``
    and no ``acquisition_idx`` (fitted spectra are shown by ``plot`` only).
    The order of ``dataset.data_vars`` is kept.
    """
    return {
        str(name): values
        for name, values in dataset.data_vars.items()
        if SPECTRAL_DIM not in values.dims
        and sum(dim in values.dims for dim in SPATIAL_DIMS) >= 2
    }


def _chosen_variable(variables: dict[str, xr.DataArray], requested: str | None) -> str:
    """Return the fit-result variable :func:`plot_scan` shows first."""
    if requested is None:
        return preferred_fit_variable(variables)
    if requested not in variables:
        raise ValueError(
            f"plot_scan variable must be one of the map variables "
            f"{list(variables)} of the Dataset; got {requested!r}."
        )
    return requested


def _average_range(payload: RenderPayload, curve_x: np.ndarray) -> list[float] | None:
    """Return the averaged repetitions as curve axis values, or None.

    ``plot`` marks the repetitions that the map averages on the repetition
    curve; the payload gives their positions on the curve axis.
    """
    shown = payload.metadata.get("avg_rep_range")
    if not shown or curve_x.size == 0:
        return None
    first, last = sorted(
        min(max(int(shown[key]), 0), curve_x.size - 1)
        for key in ("start_position", "end_position")
    )
    return [float(curve_x[first]), float(curve_x[last])]


def _chosen_curve_dim(
    data: xr.DataArray, state: PlotState, requested: str | None, *, explicit: bool
) -> str | None:
    """Return the curve dimension of :func:`plot_scan`, or None for image only.

    An explicit request must be one of
    :func:`~hypermri.visualization.view_dimensions.curve_dim_choices` for the
    shown plane; a curve dimension kept in ``state`` is used only when it
    still is one. Otherwise spectroscopic data show the spectrum along
    ``acquisition_idx``, and images a curve along
    :func:`~hypermri.visualization.view_dimensions.default_curve_dim` (for
    example echoes or repetitions). A spatial dimension such as the third
    slice axis is not chosen automatically: it stays a slider and the image
    is shown without a curve panel.
    """
    plane = displayed_spatial_dims(data, state.spatial.slicing_dim)
    choices = curve_dim_choices(data, plane)
    if explicit:
        if requested not in choices:
            raise ValueError(
                f"plot_scan curve_dim must be one of {choices} for the shown "
                f"plane {plane}; got {requested!r}."
            )
        return requested
    saved = state.spectral.secondary_dim
    if saved in choices:
        return saved
    if _has_spectrum(data):
        return SPECTRAL_DIM
    default = default_curve_dim(data, plane)
    return None if default in SPATIAL_DIMS else default


def _has_spectrum(data: xr.DataArray) -> bool:
    """Return whether the data hold a spectrum along ``acquisition_idx``.

    Images loaded from a scan keep ``acquisition_idx`` with one entry; one
    point is not a spectrum, so such images are shown as images.
    """
    return data.sizes.get(SPECTRAL_DIM, 0) > 1


class _DataChoice:
    """Variable, metabolite split, view plane, rotation and k-space of the shown data of :class:`ScanView`."""

    def set_variable(self, variable: str) -> None:
        """Show another variable of the fit result given to :func:`plot_scan`.

        The curve dimension is chosen again for the new variable (see
        :func:`plot_scan`), slider indices are kept where the new variable has
        the dimension and clipped to its size, and in an absolute range mode
        the colour range becomes the value range of the new map. Masks,
        overlay and anatomy are kept.

        Parameters
        ----------
        variable : str
            Name of a map variable of the Dataset, for example ``"fit_freqs"``.

        Returns
        -------
        None
            ``state`` is updated in place and a new block is sent.

        Raises
        ------
        ValueError
            If the view was not created from a Dataset or ``variable`` is not
            one of its map variables.

        Examples
        --------
        >>> view = plot_scan(scan.fit_results)  # doctest: +SKIP
        >>> view.set_variable("fit_freqs")  # doctest: +SKIP
        """
        self._change_variable(variable)
        self._controls = self._current_controls()

    def _change_variable(self, variable: str) -> None:
        """Switch the shown data to another map variable and render it."""
        if variable not in self._variables:
            raise ValueError(
                f"ScanView.set_variable: variable must be one of "
                f"{list(self._variables)}; got {variable!r}."
            )
        if variable == self._variable:
            return
        self._variable = variable
        self._show_data(self._variables[variable])

    def set_split_metabolites(self, split: bool) -> None:
        """Show the interleaved metabolites split or as acquired.

        Only available when :func:`plot_scan` found or was asked for an
        interleaved-metabolite sequence (``split_metabolites``). Split, the
        repetitions 0, 2, 4, ... and 1, 3, 5, ... become the two entries of
        ``metabolite_idx`` and ``repetition_idx`` has half the entries. Slider
        indices are kept where the shown data have the dimension and clipped
        to its size; masks, overlay and anatomy are kept.

        Parameters
        ----------
        split : bool
            True shows the metabolites split, False the repetitions as
            acquired.

        Returns
        -------
        None
            ``state`` is updated in place and a new block is sent.

        Raises
        ------
        ValueError
            If the view offers no metabolite split.

        Examples
        --------
        >>> view = plot_scan(scan)  # interleaved FISP: split  # doctest: +SKIP
        >>> view.set_split_metabolites(False)  # doctest: +SKIP
        """
        if self._metabolite_split is None:
            raise ValueError(
                "ScanView.set_split_metabolites: this view offers no metabolite "
                "split; create it with plot_scan(..., split_metabolites=True)."
            )
        data = self._metabolite_split[1 if split else 0]
        if data is not self._data:
            self._show_data(data)
        self._controls = self._current_controls()

    def set_view_plane(self, slicing_dim: str) -> None:
        """Show the image plane perpendicular to another spatial dimension.

        Only three-dimensional images (with ``read_idx``, ``phase_idx`` and
        ``slice_idx``) offer a choice. The chosen dimension becomes a slider
        and the image shows the other two, as the view-plane menu of ``plot``
        does. The curve dimension is chosen again when the old one is now
        shown in the image; slider indices, masks, overlay and anatomy are
        kept.

        Parameters
        ----------
        slicing_dim : str
            The through-plane dimension: ``"slice_idx"`` shows the read and
            phase plane, ``"read_idx"`` the phase and slice plane and
            ``"phase_idx"`` the read and slice plane.

        Returns
        -------
        None
            ``state.spatial.slicing_dim`` is updated in place and a new block
            is sent.

        Raises
        ------
        ValueError
            If ``slicing_dim`` is not one of the view planes of the data.

        Examples
        --------
        >>> view = plot_scan(scan.seq2d)  # doctest: +SKIP
        >>> view.set_view_plane("read_idx")  # doctest: +SKIP
        """
        choices = view_plane_choices(self._data)
        if slicing_dim not in choices:
            raise ValueError(
                f"ScanView.set_view_plane: slicing_dim must be one of {choices} "
                f"for data with dimensions {list(self._data.dims)}; got "
                f"{slicing_dim!r}."
            )
        if slicing_dim != self._view_plane():
            self.state.spatial.slicing_dim = slicing_dim
            self._show_data(self._data)
        self._controls = self._current_controls()

    def _view_plane(self) -> str:
        """Return the spatial dimension perpendicular to the shown image."""
        return next(
            dim
            for dim in view_plane_choices(self._data)
            if dim not in self._spatial_dims
        )

    def set_rotation(self, degrees: int) -> None:
        """Rotate the shown image like the rotate button of ``plot``.

        Parameters
        ----------
        degrees : int
            Display rotation in degrees: 0, 90, 180 or 270, with the meaning
            of ``state.spatial.rotation`` (90 shows ``numpy.rot90(plane,
            k=-1)`` of the plane in row and column order). Only the display
            turns; the data, voxel indices and masks keep their meaning.

        Returns
        -------
        None
            ``state.spatial.rotation`` is updated in place and a new block is
            sent.

        Raises
        ------
        ValueError
            If ``degrees`` is not one of 0, 90, 180 and 270.

        Examples
        --------
        >>> view = plot_scan(scan)  # doctest: +SKIP
        >>> view.set_rotation(90)  # doctest: +SKIP
        """
        if degrees not in (0, 90, 180, 270):
            raise ValueError(
                f"ScanView.set_rotation: degrees must be 0, 90, 180 or 270; "
                f"got {degrees!r}."
            )
        self.state.spatial.rotation = int(degrees)
        self._render()
        self._controls = self._current_controls()

    def set_kspace(
        self,
        show_kspace: bool,
        spatial_filter: SpatialFilterOptions | None = None,
    ) -> None:
        """Show the image or its spatial k-space, optionally spatially filtered.

        The view then shows a copy of its data: first filtered with
        :func:`~hypermri.analysis.spatial_filtering.apply_spatial_filter`,
        then, for k-space, transformed with
        :func:`~hypermri.analysis.spatial_filtering.spatial_to_kspace` along
        Read and Phase only (each slice on its own). The data of the view and
        of the scan are not changed. In k-space the map axes are spatial
        frequencies in cycles/mm, and anatomy, overlays, voxel outlines,
        masks, ROI mean, fit curves, slice lines and the linked pointer are
        not shown; they come back with the image. The browser's "k-space"
        controls call the same operation.

        Parameters
        ----------
        show_kspace : bool
            True shows k-space, False the image. Stored as
            ``state.spatial.show_image_space`` (its negation), so it is saved
            with the view settings.
        spatial_filter : SpatialFilterOptions or None, optional
            Gaussian or Fermi window over the spatial k-space, applied to the
            image and to the k-space shown. None shows the data unfiltered.
            A choice of this view, not stored in :attr:`state`.

        Raises
        ------
        ValueError
            If the data have no ``read_idx`` and ``phase_idx`` or no voxel
            spacing in mm; the view keeps what it showed.

        Examples
        --------
        >>> from hypermri.analysis.spatial_filtering import SpatialFilterOptions
        >>> view.set_kspace(True)  # doctest: +SKIP
        >>> view.set_kspace(
        ...     False, SpatialFilterOptions(kind="gaussian", sigma_per_mm=0.1)
        ... )  # doctest: +SKIP
        """
        if spatial_filter is not None and not isinstance(
            spatial_filter, SpatialFilterOptions
        ):
            raise TypeError(
                "ScanView.set_kspace: spatial_filter must be SpatialFilterOptions "
                f"or None; got {type(spatial_filter).__name__}."
            )
        # Compute first, so that a failure keeps the view as it was.
        self._shown_data(spatial_filter, bool(show_kspace))
        self._spatial_filter = spatial_filter
        if spatial_filter is not None:
            for name in _SPATIAL_FILTER_WIDTHS:
                if getattr(spatial_filter, name) is not None:
                    self._spatial_filter_widths[name] = getattr(spatial_filter, name)
        self.state.spatial.show_image_space = not show_kspace
        self._kspace_error = ""
        self._render(reset_map_range=True)

    def _kspace_shown(self) -> bool:
        """Return whether the map shows the spatial k-space of the data."""
        return not self.state.spatial.show_image_space

    def _shown_data(
        self,
        spatial_filter: SpatialFilterOptions | None | str = "current",
        kspace: bool | None = None,
    ) -> xr.DataArray:
        """Return the data the map and curves show (see :meth:`set_kspace`).

        Without filter and in image space this is the data of the view
        itself. The last result is kept until the data, the filter or the
        space changes.
        """
        if isinstance(spatial_filter, str):
            spatial_filter = self._spatial_filter
        if kspace is None:
            kspace = self._kspace_shown()
        if spatial_filter is None and not kspace:
            return self._data
        cache = self._shown_data_cache
        if (
            cache is not None
            and cache[0] is self._data
            and cache[1:3] == (spatial_filter, kspace)
        ):
            return cache[3]
        try:
            shown = self._data
            if spatial_filter is not None:
                shown = apply_spatial_filter(shown, spatial_filter)
            if kspace:
                shown = spatial_to_kspace(shown)
        except (TypeError, ValueError) as error:
            raise ValueError(f"ScanView.set_kspace: {error}") from error
        self._shown_data_cache = (self._data, spatial_filter, kspace, shown)
        return shown

    def _kspace_view(self) -> dict[str, Any]:
        """Return the image/k-space choice and the spatial filter for the browser."""
        options = self._spatial_filter
        return {
            "available": (
                {"read_idx", "phase_idx"} <= set(self._data.dims)
                and self._outline is None
            ),
            "shown": self._kspace_shown(),
            "filter": "none" if options is None else options.kind,
            **self._spatial_filter_widths,
            "error": self._kspace_error,
        }

    def _apply_browser_kspace(self, content: dict[str, Any]) -> None:
        """Apply the browser's image/k-space choice and spatial filter.

        A failure is shown below the inputs; the view keeps what it showed.
        """
        current = self._kspace_view()
        values = {
            name: content.get(name, current[name])
            for name in ("shown", "filter", *_SPATIAL_FILTER_WIDTHS)
        }
        kind = values["filter"]
        try:
            widths = {name: float(values[name]) for name in _SPATIAL_FILTER_WIDTHS}
            spatial_filter = (
                None
                if kind == "none"
                else SpatialFilterOptions(
                    kind=kind,
                    **{
                        name: widths[name]
                        for name in _SPATIAL_FILTER_WIDTHS
                        if (name == "sigma_per_mm") == (kind == "gaussian")
                    },
                )
            )
            self.set_kspace(bool(values["shown"]), spatial_filter)
        except (TypeError, ValueError) as error:
            self._kspace_error = str(error)
            self._view = {**self._view, "kspace": self._kspace_view()}
