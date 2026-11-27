"""Rows of views made by :func:`~hypermri.visualization.scan_view.plot_side_by_side`.

:class:`SideBySideView` holds the views and arranges them in rows of grids,
with a switch above them that changes the arrangement while the views are
shown, and one menu "under" per view that lays it under another view. Each
row shares its width between its views in proportion to the width each
image needs at the full plot height. :func:`link_side_by_side` links views
shown together, and :func:`views_over` makes the views of one image over
others; :func:`~hypermri.visualization.plot_study` uses both as well.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import ipywidgets
import traitlets

from hypermri.visualization.scan_view.options import SLICE_LINE_COLOURS
from hypermri.visualization.scan_view.widget import ScanView

#: Arrangements of :class:`SideBySideView`: ``"rows"`` (the given rows, or
#: image-only views in the first row and each view with a curve in a row of
#: its own), ``"side by side"`` (all views in one row) and ``"stacked"`` (one
#: view per row).
ARRANGEMENTS = ("rows", "side by side", "stacked")

_ARRANGEMENT_HINTS = (
    "Given rows, or the images in the first row and each spectrum or time "
    "course in a row of its own",
    "All views in one row",
    "One view per row, each with the full width",
)

#: Least width (px) that a view with a curve panel asks for its curves when
#: a row is shared between views; the curve panel's flex basis in
#: ``browser/viewer.css`` plus the gap before it.
SIDE_BY_SIDE_CURVE_WIDTH_PX = 304

#: Width of a curve panel per plot height that a view asks for when a row is
#: shared between views, as for a landscape plot of 3:2. With only the least
#: width a single plane next to spectra or time courses took more than its
#: share of the row.
SIDE_BY_SIDE_CURVE_ASPECT = 1.5

# Value of a menu "under" for no view underneath (a Dropdown treats None as
# no selection at all).
_NOT_UNDER = -1


class SideBySideView(ipywidgets.VBox):
    """Linked views in rows, with switches for arrangement and underlays.

    Made by :func:`~hypermri.visualization.scan_view.plot_side_by_side`.
    A row of buttons ("arrange") above the views sets :attr:`arrangement`;
    setting the attribute from Python moves the buttons too. The views are
    moved, not rebuilt, so their settings, selections and links are kept.

    A menu "under" per given view lays it under another given view (see
    :meth:`set_underlays`): the upper view is then shown again as a new
    view with the lower image underneath, in the plane of the lower
    image's slices, and the lower view is hidden. Several views can lie
    under one view; it then gets one new view over each of them, and only
    the first shows the curve and time course. Choosing "none" shows the
    given views again with their settings. After every change the shown
    views are linked again as by :func:`link_side_by_side`; if that fails
    (for example for an image without scanner coordinates), they are shown
    unlinked and a status line says why.

    Parameters
    ----------
    views : sequence of ScanView
        The given views in their order.
    rows : sequence of sequence of ScanView or None
        The rows of the arrangement ``"rows"`` from top to bottom; together
        they hold every view of *views* once. None places them with
        :func:`automatic_rows`.
    arrangement : str, optional
        Starting arrangement, one of :data:`ARRANGEMENTS`.
    show_slices, share_masks : bool, optional
        How the shown views are linked; see :func:`link_side_by_side`.
    under_menus : bool, optional
        Show the menus "under". :meth:`set_underlays` works without them.

    Raises
    ------
    traitlets.TraitError
        If *arrangement* is not one of :data:`ARRANGEMENTS`.
    ValueError
        If the given views cannot be linked as asked; see
        :func:`link_side_by_side`.

    Examples
    --------
    >>> box = plot_side_by_side(axial, sagittal, csi)  # doctest: +SKIP
    >>> box.arrangement = "stacked"  # doctest: +SKIP
    >>> box.rows  # doctest: +SKIP
    ((axial,), (sagittal,), (csi,))
    >>> box.set_underlays({0: 2})  # axial under the CSI  # doctest: +SKIP
    """

    arrangement = traitlets.Enum(ARRANGEMENTS, default_value="rows").tag(sync=True)

    def __init__(
        self,
        views: Sequence[ScanView],
        rows: Sequence[Sequence[ScanView]] | None,
        arrangement: str = "rows",
        *,
        show_slices: bool = True,
        share_masks: bool = True,
        under_menus: bool = True,
    ) -> None:
        self._given = tuple(views)
        index_of = {id(view): i for i, view in enumerate(self._given)}
        self._given_rows = (
            None
            if rows is None
            else tuple(tuple(index_of[id(view)] for view in row) for row in rows)
        )
        self._show_slices = show_slices
        self._share_masks = share_masks
        self._views = self._given
        # Lower view index to the index of the view it lies under.
        self._under_of: dict[int, int] = {}
        # Upper view index to (lower indices, the views over them).
        self._overlays: dict[int, tuple[tuple[int, ...], tuple[ScanView, ...]]] = {}
        self._grids: list[Any] = []
        self._switch = ipywidgets.ToggleButtons(
            options=ARRANGEMENTS,
            tooltips=list(_ARRANGEMENT_HINTS),
            description="arrange",
            style={"button_width": "auto", "description_width": "auto"},
        )
        self._menus = tuple(
            ipywidgets.Dropdown(
                options=[("none", _NOT_UNDER)]
                + [
                    (_view_label(other, j), j)
                    for j, other in enumerate(self._given)
                    if j != i
                ],
                value=_NOT_UNDER,
                description=f"{_view_label(view, i)} under",
                style={"description_width": "auto"},
                layout=ipywidgets.Layout(width="auto"),
            )
            for i, view in enumerate(self._given)
        )
        self._updating = False
        for menu in self._menus:
            menu.observe(self._on_menu_change, names="value")
        self._status = ipywidgets.HTML("")
        self._rows_box = ipywidgets.VBox(
            layout=ipywidgets.Layout(width="100%", grid_gap="12px")
        )
        controls = [self._switch]
        if under_menus:
            controls.append(
                ipywidgets.HBox(
                    list(self._menus), layout=ipywidgets.Layout(flex_flow="row wrap")
                )
            )
        super().__init__(
            [*controls, self._status, self._rows_box],
            layout=ipywidgets.Layout(width="100%"),
        )
        self.arrangement = arrangement
        self._switch.value = self.arrangement
        traitlets.link((self._switch, "value"), (self, "arrangement"))
        self.observe(self._arrange, names="arrangement")
        for view in self._given:
            self._observe_width(view)
        link_side_by_side(self._views, show_slices, share_masks)
        self._arrange()

    @property
    def views(self) -> tuple[ScanView, ...]:
        """Return the shown views in their order.

        Returns
        -------
        tuple of ScanView
            Without underlays the given views. A view laid under another
            one is left out, and a view with views underneath is replaced
            by its views over them (see :attr:`given_views`). The first view
            shows the inputs of a shared mask tool.
        """
        return self._views

    @property
    def given_views(self) -> tuple[ScanView, ...]:
        """Return the given views, shown or not.

        Returns
        -------
        tuple of ScanView
            The views passed in, in their order; the indices of
            :meth:`set_underlays` refer to it. A view laid under another one
            is kept with its settings while it is hidden.
        """
        return self._given

    @property
    def underlays(self) -> dict[int, int]:
        """Return which given view lies under which.

        Returns
        -------
        dict of int to int
            Index of a lower view in :attr:`given_views` to the index of the
            view it lies under. A new dict; changing it changes nothing.
        """
        return dict(self._under_of)

    @property
    def rows(self) -> tuple[tuple[ScanView, ...], ...]:
        """Return the views of each shown row, top to bottom, left to right.

        Returns
        -------
        tuple of tuple of ScanView
            The rows of the current :attr:`arrangement`. In given rows a
            view with views underneath stands where its views over them
            stand; a row whose views all lie under others is left out.
        """
        if self.arrangement == "side by side":
            return (self._views,)
        if self.arrangement == "stacked":
            return tuple((view,) for view in self._views)
        if self._given_rows is None:
            return tuple(tuple(row) for row in automatic_rows(list(self._views)))
        rows = (
            tuple(view for index in row for view in self._shown_for(index))
            for row in self._given_rows
        )
        return tuple(row for row in rows if row)

    def set_underlays(self, underlays: Mapping[int, int]) -> None:
        """Lay given views under others; the menus "under" follow.

        Parameters
        ----------
        underlays : mapping of int to int
            Index of a lower view in :attr:`given_views` to the index of the
            view drawn over it. An empty mapping shows the given views
            again. The upper view is made anew with
            :func:`~hypermri.visualization.scan_view.plot_scan` from its scan
            or DataArray, with the lower one's scan or DataArray as
            ``anatomy``, and shows the plane of the lower image's slices
            (:meth:`ScanView.show_anatomy_plane`). Views kept over the same
            lower views keep their settings; views no longer needed are
            closed.

        Raises
        ------
        ValueError
            If an index is not one of :attr:`given_views`, a view lies under
            itself, or a view both lies under another one and has one
            underneath. Nothing is changed then.
        Exception
            Any error of :func:`~hypermri.visualization.scan_view.plot_scan`
            for a new view. The views shown before stay shown.
        """
        wanted = _checked_underlays(len(self._given), underlays)
        overlays: dict[int, tuple[tuple[int, ...], tuple[ScanView, ...]]] = {}
        built: list[ScanView] = []
        try:
            for upper in sorted(set(wanted.values())):
                lowers = tuple(i for i in sorted(wanted) if wanted[i] == upper)
                kept = self._overlays.get(upper)
                if kept is not None and kept[0] == lowers:
                    overlays[upper] = kept
                    continue
                views = views_over(
                    _view_source(self._given[upper]),
                    [
                        (_view_source(self._given[i]), _view_label(self._given[i], i))
                        for i in lowers
                    ],
                    _view_label(self._given[upper], upper),
                )
                built.extend(views)
                overlays[upper] = (lowers, views)
        except Exception:
            for view in built:
                view.close()
            raise
        for view in (*self._given, *self._views):
            view.unlink()
        for upper, (_, views) in self._overlays.items():
            if overlays.get(upper, ((), ()))[1] is not views:
                for view in views:
                    view.close()
        for upper, (_, views) in overlays.items():
            for view in views:
                view.show_timings = self._given[upper].show_timings
        for view in built:
            self._observe_width(view)
        self._under_of = wanted
        self._overlays = overlays
        self._views = tuple(
            view for index in range(len(self._given)) for view in self._shown_for(index)
        )
        self._status.value = ""
        try:
            link_side_by_side(self._views, self._show_slices, self._share_masks)
        except ValueError as error:
            for view in self._views:
                view.unlink()
            link_side_by_side(self._views, show_slices=False, share_masks=False)
            self._status.value = f"Views not linked: {error}"
        for _, views in overlays.values():
            if len(views) > 1:
                views[0].link_frames(*views[1:])
        self._updating = True
        try:
            for index, menu in enumerate(self._menus):
                menu.value = wanted.get(index, _NOT_UNDER)
        finally:
            self._updating = False
        self._arrange()

    def _shown_for(self, index: int) -> tuple[ScanView, ...]:
        """Return the shown views that stand for the given view *index*."""
        if index in self._under_of:
            return ()
        overlay = self._overlays.get(index)
        return (self._given[index],) if overlay is None else overlay[1]

    def _on_menu_change(self, change: dict[str, Any]) -> None:
        """Lay the views as the menus say; report errors in the status line."""
        if self._updating:
            return
        wanted = {
            index: menu.value
            for index, menu in enumerate(self._menus)
            if menu.value != _NOT_UNDER
        }
        try:
            self.set_underlays(wanted)
        except Exception as error:  # noqa: BLE001 - shown in the status line
            self._updating = True
            try:
                for index, menu in enumerate(self._menus):
                    menu.value = self._under_of.get(index, _NOT_UNDER)
            finally:
                self._updating = False
            self._status.value = f"<b>Could not lay the views:</b> {error}"

    def _observe_width(self, view: ScanView) -> None:
        """Share the widths again when *view* changes the width it needs."""
        # A new plane (rotation, view plane) or plot height changes the width
        # each view needs, so the columns are worked out again.
        view.observe(self._update_columns, names=["_view", "plot_height", "_layout"])

    def _arrange(self, change: Any = None) -> None:
        """Place the views in one grid per row of the current arrangement.

        A grid instead of a wrapping row: each view spans three rows of it
        (controls, plots, tools; see viewer.css), so the plots of the views
        start and end at the same height whatever their controls take.
        """
        self._grids = [
            ipywidgets.GridBox(
                list(row),
                layout=ipywidgets.Layout(
                    grid_template_columns=_side_by_side_columns(list(row)),
                    grid_gap="0 16px",
                    width="100%",
                ),
            )
            for row in self.rows
        ]
        self._rows_box.children = self._grids

    def _update_columns(self, change: Any) -> None:
        """Share the width of every row again after a view changed its plane."""
        for grid in self._grids:
            grid.layout.grid_template_columns = _side_by_side_columns(
                list(grid.children)
            )


def link_side_by_side(
    views: Sequence[ScanView], show_slices: bool, share_masks: bool
) -> None:
    """Link views shown together and give each a coloured bar.

    Every view gets a bar in its colour of :data:`SLICE_LINE_COLOURS` above
    it. With *show_slices* each view draws the shown slices of the others
    in their colours (:meth:`ScanView.show_slices_of`) and the mouse
    pointer is shared (:meth:`ScanView.link_pointer`). With *share_masks*
    the masks are shared (:meth:`ScanView.link_masks`), and only the first
    view shows the inputs of the mask tool.

    Parameters
    ----------
    views : sequence of ScanView
        Views in their shown order; they should have no links yet
        (:meth:`ScanView.unlink`).
    show_slices : bool
        Draw the slices of the others and share the pointer; needs scanner
        coordinates ``x``, ``y``, ``z`` for every view.
    share_masks : bool
        Share the masks; needs scanner coordinates and a grid with at least
        two dimensions of more than one voxel for every view.

    Raises
    ------
    ValueError
        If a view lacks what *show_slices* or *share_masks* needs. Some
        links may then be made already; unlink the views before using them
        again.
    """
    colours = [
        SLICE_LINE_COLOURS[i % len(SLICE_LINE_COLOURS)] for i in range(len(views))
    ]
    for index, view in enumerate(views):
        if show_slices:
            others = [i for i in range(len(views)) if i != index]
            view.show_slices_of(
                *(views[i] for i in others), colours=[colours[i] for i in others]
            )
        view.layout.border_top = f"3px solid {colours[index]}"
    if show_slices and len(views) > 1:
        views[0].link_pointer(*views[1:])
    if share_masks and len(views) > 1:
        views[0].link_masks(*views[1:])
        for view in views[1:]:
            view._mask_tool_shown = False


def views_over(
    source: Any, lowers: Sequence[tuple[Any, str]], name: str
) -> tuple[ScanView, ...]:
    """Return one view of *source* over each lower image.

    Parameters
    ----------
    source : object
        Scan or DataArray of the upper image, as accepted by
        :func:`~hypermri.visualization.scan_view.plot_scan`.
    lowers : sequence of (object, str)
        Scan or DataArray of each lower image (``anatomy`` of ``plot_scan``)
        with its name.
    name : str
        Name of the upper image; each view is named "*name* over *lower
        name*".

    Returns
    -------
    tuple of ScanView
        The views in the order of *lowers*; only the first shows the curve
        and time course. Each shows the plane of its lower image's slices
        (:meth:`ScanView.show_anatomy_plane`). The views are not linked.

    Raises
    ------
    Exception
        Any error of ``plot_scan``; the views made before it are closed.
    """
    # Imported here: entry.py builds SideBySideView, so it imports this module.
    from hypermri.visualization.scan_view.entry import plot_scan

    views: list[ScanView] = []
    try:
        for position, (lower, lower_name) in enumerate(lowers):
            view = plot_scan(
                source,
                anatomy=lower,
                name=f"{name} over {lower_name}",
                image_only=position > 0,
            )
            views.append(view)
            view.show_anatomy_plane()
    except Exception:
        for view in views:
            view.close()
        raise
    return tuple(views)


def _checked_underlays(count: int, underlays: Mapping[int, int]) -> dict[int, int]:
    """Return *underlays* as a dict after checking indices and chains."""
    wanted = {}
    for lower, upper in underlays.items():
        for index in (lower, upper):
            if isinstance(index, bool) or index not in range(count):
                raise ValueError(
                    f"SideBySideView underlays: {index!r} is not the index of a "
                    f"given view; use 0 to {count - 1}."
                )
        if lower == upper:
            raise ValueError(
                f"SideBySideView underlays: view {lower} cannot lie under itself."
            )
        wanted[int(lower)] = int(upper)
    for lower, upper in wanted.items():
        if upper in wanted:
            raise ValueError(
                f"SideBySideView underlays: view {upper} has view {lower} "
                f"underneath and lies under view {wanted[upper]}; lay both "
                f"under view {wanted[upper]} instead."
            )
    return wanted


def _view_source(view: ScanView) -> Any:
    """Return the scan or DataArray *view* was made from."""
    return view._model.data_obj


def _view_label(view: ScanView, index: int) -> str:
    """Return the name of *view*, or its number when it has no name."""
    return view.name or f"view {index + 1}"


def automatic_rows(views: list[ScanView]) -> list[list[ScanView]]:
    """Return image-only views as the first row and each curve view as a row.

    Parameters
    ----------
    views : list of ScanView
        Views in their given order.

    Returns
    -------
    list of list of ScanView
        The rows from top to bottom, keeping the given order within the
        image-only row and between the curve rows. Without image-only views
        the first row is left out.
    """
    images = [view for view in views if not _shows_curve(view)]
    curves = [[view] for view in views if _shows_curve(view)]
    return ([images] if images else []) + curves


def _side_by_side_columns(views: list[ScanView]) -> str:
    """Return the CSS grid columns of one row of views.

    Each column gets a share of the width in proportion to what its view
    needs to show its image at the full plot height, plus room for a curve
    (see :func:`_side_by_side_width`). When the row is too
    narrow, every image is then lowered by about the same factor, so that the
    images keep one height.
    """
    return " ".join(f"minmax(0, {_side_by_side_width(view)}fr)" for view in views)


def _shows_curve(view: ScanView) -> bool:
    """Return whether *view* shows a curve panel next to its image."""
    return view._view.get("curve_dim") is not None and "curve" in view._layout.get(
        "panels", ()
    )


def _side_by_side_width(view: ScanView) -> int:
    """Return the width (px) *view* needs for its image at the full plot height.

    The image keeps equal scales on both axes, so its width is the plot
    height times the ratio of its horizontal to its vertical extent (from the
    axis ranges of the shown plane, or of the anatomy under a single voxel).
    A curve panel adds :data:`SIDE_BY_SIDE_CURVE_ASPECT` times the plot
    height, but at least :data:`SIDE_BY_SIDE_CURVE_WIDTH_PX`. Axis margins
    are left out; the value only sets the share of a column.
    """
    shown = view._view
    ranges = shown.get("outline") if shown.get("single_voxel") else shown
    width = float(view.plot_height)
    if ranges and ranges.get("x_range") and ranges.get("y_range"):
        span_x = abs(ranges["x_range"][-1] - ranges["x_range"][0])
        span_y = abs(ranges["y_range"][-1] - ranges["y_range"][0])
        if span_x > 0 and span_y > 0:
            width = view.plot_height * span_x / span_y
    if _shows_curve(view):
        width += max(
            SIDE_BY_SIDE_CURVE_WIDTH_PX,
            SIDE_BY_SIDE_CURVE_ASPECT * view.plot_height,
        )
    return max(round(width), 1)
