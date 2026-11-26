"""Time course of a mask or voxel along a chosen dimension.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_TimeCourse`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypermri._optional import _import_optional
from hypermri.visualization.mask_editing import (
    DEFAULT_MASK_COLOR,
    plane_masked_mean,
)
from hypermri.visualization.render_data import (
    curve_range_mean,
)
from hypermri.visualization.view_dimensions import (
    curve_dim_choices,
    displayed_spatial_dims,
)

_import_optional("anywidget", "visualization")
_import_optional("cmap", "visualization")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    SPECTRAL_DIM,
    _as_float,
)


class _TimeCourse:
    """Time course of a mask or voxel along a chosen dimension of :class:`ScanView`."""

    # ── Time course ────────────────────────────────────────────────────

    def set_time_course_range(
        self, low: float | None = None, high: float | None = None
    ) -> None:
        """Choose the spectral range averaged for the time course panel.

        The time course panel appears when the curve panel shows spectra and
        the data have a further dimension with several entries, normally
        ``repetition_idx``. It shows, for the selected voxel and optionally
        inside a mask, the mean of the displayed spectral values (the chosen
        complex part, after line broadening) over this range at every entry of
        :attr:`time_course_dim`. Without a range it shows the value at the
        selected spectral point and follows that point.

        Parameters
        ----------
        low, high : float or None, optional
            Range limits in the displayed spectral unit (ppm or Hz), in either
            order and inclusive. Each limit is moved to the nearest spectral
            point. Both None follow the selected spectral point again.

        Raises
        ------
        ValueError
            If no time course panel is shown, if only one limit is given, or
            if a limit is not a finite number.

        Notes
        -----
        The range is stored as spectral indices in
        ``state.spectral.time_course_range``, so it stays at the same spectral
        points when the spectral unit changes. Only the time course is
        recomputed; the curve block is not.

        Examples
        --------
        >>> view = plot_scan(scan)  # doctest: +SKIP
        >>> view.set_time_course_range(170.0, 174.0)  # pyruvate, ppm  # doctest: +SKIP
        """
        self._require_time_course("set_time_course_range")
        if low is None and high is None:
            self.state.spectral.time_course_range = None
        elif low is None or high is None:
            raise ValueError(
                "set_time_course_range needs both limits, or neither to follow "
                f"the selected spectral point; got low={low!r}, high={high!r}."
            )
        else:
            for name, value in (("low", low), ("high", high)):
                if not np.isfinite(_as_float(value)):
                    raise ValueError(
                        f"set_time_course_range {name} must be a finite number; "
                        f"got {value!r}."
                    )
            first, last = (
                int(np.argmin(np.abs(self._curve_x - float(value))))
                for value in (low, high)
            )
            self.state.spectral.time_course_range = tuple(sorted((first, last)))
        self._send_time_course()

    def show_time_course_mask(self, mask: str | None) -> None:
        """Add the mean time course inside a mask to the time course panel.

        The mask curve is the coverage-weighted mean of the voxel time courses
        over the shown plane (:func:`~hypermri.visualization.mask_editing.plane_masked_mean`),
        i.e. of the same spectral range mean as the voxel curve. Voxels of the
        mask in other slices are not included. It follows later mask edits,
        range and plane changes.

        Parameters
        ----------
        mask : str or None
            Name of an existing mask, or None to show the voxel curve alone.

        Raises
        ------
        ValueError
            If no time course panel is shown or no mask has this name.
        """
        self._require_time_course("show_time_course_mask")
        if mask is not None:
            self._require_mask(mask, "show_time_course_mask")
        self._time_mask = mask
        self._send_time_course()

    def set_time_course_dim(self, dim: str) -> None:
        """Choose the dimension along which the time course panel runs.

        Parameters
        ----------
        dim : str
            A dimension with more than one entry that is neither shown as
            image nor spectral, such as ``"repetition_idx"``.

        Raises
        ------
        ValueError
            If no time course panel is shown or ``dim`` is not one of the
            dimensions offered for it.
        """
        self._require_time_course("set_time_course_dim")
        _, choices = self._chosen_time_dim()
        if dim not in choices:
            raise ValueError(
                f"set_time_course_dim: dim must be one of {choices}; got {dim!r}."
            )
        self.state.spectral.time_course_dim = dim
        self._render()

    @property
    def time_course_dim(self) -> str | None:
        """Return the dimension of the time course panel, or None without it."""
        return self._time_dim

    def _require_time_course(self, operation: str) -> None:
        """Fail when no time course panel is shown."""
        if self._time_dim is None:
            raise ValueError(
                f"{operation} needs the time course panel, which is shown only "
                "when the curve panel shows spectra and another dimension "
                "(such as repetition_idx) has several entries."
            )

    def _chosen_time_dim(self) -> tuple[str | None, list[str]]:
        """Return the time course dimension and the dimensions offered for it.

        Only spectra get a time course. Dimensions the display processing
        reduces to one entry (combined channels, averaged repetitions) are not
        offered. ``state.spectral.time_course_dim`` is used while it is
        offered, then ``repetition_idx``, then the first offered dimension.
        """
        if self._curve_dim != SPECTRAL_DIM:
            return None, []
        reduced = self._reduced_dims()
        plane = displayed_spatial_dims(self._data, self.state.spatial.slicing_dim)
        choices = [
            dim
            for dim in curve_dim_choices(self._data, plane)
            if dim != SPECTRAL_DIM and dim not in reduced
        ]
        for dim in (self.state.spectral.time_course_dim, "repetition_idx"):
            if dim in choices:
                return dim, choices
        return (choices[0] if choices else None), choices

    def _time_view(self, choices: list[str]) -> dict[str, Any] | None:
        """Describe the time axis for the browser, or None without time course."""
        dim = self._time_dim
        if dim is None:
            return None
        coord = self._data.coords.get(dim)
        if (
            coord is not None
            and coord.dims == (dim,)
            and np.issubdtype(coord.dtype, np.number)
        ):
            x = np.asarray(coord.values, dtype=float)
            units = coord.attrs.get("units")
            label = f"{dim} [{units}]" if units else dim
        else:
            x = np.arange(self._data.sizes[dim], dtype=float)
            label = dim
        return {
            "dim": dim,
            "x": x.tolist(),
            "label": label,
            "choices": choices,
            "index": self._other_index(dim),
        }

    def _time_range(self) -> tuple[int, int, bool]:
        """Return the averaged spectral index range and whether it follows the point."""
        n_curve = self._curve_x.size
        saved = self.state.spectral.time_course_range
        if saved is None:
            index = int(self.state.spectral.selected_spectral_index or 0)
            index = min(max(index, 0), n_curve - 1)
            return index, index, True
        first, last = sorted(min(max(int(value), 0), n_curve - 1) for value in saved)
        return first, last, False

    def _send_time_course(self) -> None:
        """Compute the voxel and mask time courses for the chosen range and send them."""
        if self._time_curves is None:
            self._time_mask = None
            self._time_block = b""
            self._time_course = {}
            return
        first, last, follows_point = self._time_range()
        means = curve_range_mean(self._time_curves, first, last)
        if self._time_mask not in self._mask_store:
            self._time_mask = None
        mask_y = None
        if self._time_mask is not None:
            coverage = self._shown_coverage(self._mask_store[self._time_mask])
            if coverage is not None:
                mask_y = [
                    float(value) if np.isfinite(value) else None
                    for value in plane_masked_mean(means, coverage)
                ]
        self._time_block = np.ascontiguousarray(means, dtype=np.float32).tobytes()
        self._time_course = {
            "range": [first, last],
            "range_values": [float(self._curve_x[first]), float(self._curve_x[last])],
            "follows_point": follows_point,
            "mask": self._time_mask,
            "mask_y": mask_y,
            "mask_color": self._mask_colors.get(self._time_mask, DEFAULT_MASK_COLOR),
        }

    def _apply_browser_time_course(self, content: dict[str, Any]) -> None:
        """Apply a time course change sent by the browser with the public methods."""
        if "dim" in content and content["dim"] != self._time_dim:
            self.set_time_course_dim(content["dim"])
        if "range" in content:
            limits = content["range"] or (None, None)
            self.set_time_course_range(*limits)
        if "mask" in content:
            self.show_time_course_mask(content["mask"] or None)
