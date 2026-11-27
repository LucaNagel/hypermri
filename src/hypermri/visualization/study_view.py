"""Choose scans of a study and show them in the browser viewer.

:func:`plot_study` shows a collapsible list of the scans of a
:class:`~hypermri.Study` above the viewer. Ticking scans shows them with
:func:`~hypermri.visualization.scan_view.plot_scan`; several ticked scans are
shown next to each other as with
:func:`~hypermri.visualization.scan_view.plot_side_by_side`. A ticked scan
can be laid under another scan of the study: both then form one view of the
upper scan with the lower one underneath (``anatomy`` of ``plot_scan``), each
in its own colours. Several scans can lie under one scan, for example a
coronal and a sagittal anatomy under a 3D image: the upper scan then gets
one view over each of them, in the plane of that scan's slices, and only
the first view shows the curve and time course. Scans are read only when
they are shown; the view of a scan that is no longer shown is closed.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

from hypermri._optional import _import_optional
from hypermri.visualization.scan_view import ScanView, plot_scan, plot_side_by_side
from hypermri.visualization.scan_view.side_by_side import views_over

_import_optional("ipywidgets")

import ipywidgets  # noqa: E402

# Value of the menu "under" for no scan underneath (a Dropdown treats None as
# no selection at all).
NO_UNDERLAY = ""

# One value of ``underlays``: a scan key, several keys or None for none.
Underlays = Hashable | list[Hashable] | tuple[Hashable, ...] | None

# Columns of Study.dataframe that name a scan in the list, in this order.
SCAN_LABEL_COLUMNS = ("Scan Name", "SequenceClass", "Measurement Time")

# Column of Study.dataframe that orders the list: acquisition clock time as
# "HH:MM:SS", or "MISSING".
TIME_COLUMN = "Measurement Time"


class StudyView(ipywidgets.VBox):
    """Scan list of a study above the views of the chosen scans.

    Made by :func:`plot_study`; see there for the behaviour. The list sits
    in a collapsible section ("Scans"), ordered by measurement time; each
    row has a tick box that shows the scan and a menu "under" that lays the
    scan under another scan of the study. A status line reports scans that
    could not be shown and views that could not be linked.

    Parameters
    ----------
    study : hypermri.Study
        Study whose scans are listed. Its ``dataframe`` index gives the scan
        keys (experiment numbers), and ``study[key]`` gives a scan.

    Notes
    -----
    :attr:`views` gives the view of every shown scan and
    :attr:`underlay_views` its views over each lower scan (both read only);
    use :meth:`select` to change the shown scans.
    """

    def __init__(self, study: Any) -> None:
        keys = scans_by_time(study)
        self._study = study
        self._keys = keys
        # Scan key to (keys of the scans under it, one view per lower scan
        # or a single view without one).
        self._shown: dict[
            Hashable, tuple[tuple[Hashable, ...], tuple[ScanView, ...]]
        ] = {}
        self._updating = False
        self._ticks: dict[Hashable, ipywidgets.Checkbox] = {}
        self._unders: dict[Hashable, ipywidgets.Dropdown] = {}
        rows = []
        for key in keys:
            tick = ipywidgets.Checkbox(
                value=False,
                description=scan_label(study, key),
                indent=False,
                layout=ipywidgets.Layout(width="auto", flex="1 1 auto"),
            )
            under = ipywidgets.Dropdown(
                options=[("none", NO_UNDERLAY)]
                + [(_key_text(other), other) for other in keys if other != key],
                value=NO_UNDERLAY,
                description="under",
                disabled=True,
                layout=ipywidgets.Layout(width="200px"),
                style={"description_width": "40px"},
            )
            tick.observe(self._on_list_change, names="value")
            under.observe(self._on_list_change, names="value")
            self._ticks[key] = tick
            self._unders[key] = under
            rows.append(ipywidgets.HBox([tick, under]))
        self._list = ipywidgets.Accordion(
            children=[ipywidgets.VBox(rows)], selected_index=0
        )
        self._status = ipywidgets.HTML("")
        self._area = ipywidgets.Box(layout=ipywidgets.Layout(width="100%"))
        super().__init__([self._list, self._status, self._area])
        self._update_list_title()

    @property
    def views(self) -> dict[Hashable, ScanView]:
        """Return the view of every shown scan, keyed by scan key.

        Returns
        -------
        dict
            Scan key to :class:`~hypermri.visualization.scan_view.ScanView`,
            in the shown order (the order of the scan list). A scan with
            several scans underneath maps to its first view, the one with
            the curve and time course (see :attr:`underlay_views`). A scan
            laid under another one has no view of its own. A new dict;
            changing it changes nothing.
        """
        return {key: views[0] for key, (_, views) in self._shown.items()}

    @property
    def underlay_views(self) -> dict[tuple[Hashable, Hashable], ScanView]:
        """Return the view of every shown scan over each scan underneath.

        Returns
        -------
        dict
            ``(upper key, lower key)`` to the
            :class:`~hypermri.visualization.scan_view.ScanView` that shows the
            upper scan over the lower one, in the shown order. Scans with
            nothing underneath are left out. A new dict; changing it changes
            nothing.
        """
        return {
            (key, under): view
            for key, (unders, views) in self._shown.items()
            for under, view in zip(unders, views, strict=False)
        }

    def select(
        self,
        scans: Iterable[Hashable],
        underlays: Mapping[Hashable, Underlays] | None = None,
    ) -> None:
        """Show these scans, each with optional scans underneath.

        Views of scans that stay shown with the same scans underneath are
        kept with their settings; the others are closed. New views are made
        with :func:`~hypermri.visualization.scan_view.plot_scan`, which reads
        the scan's data if they are not loaded yet. Two or more views are
        shown next to each other with the slices of the others, a shared
        pointer and shared masks, as with
        :func:`~hypermri.visualization.scan_view.plot_side_by_side`; when a
        scan has no scanner coordinates for that, they are shown unlinked
        and the status line says why.

        Parameters
        ----------
        scans : iterable of hashable
            Keys of the scans to show with a view of their own, as in
            ``study.dataframe.index``. They are shown in the order of the
            scan list. An empty iterable shows nothing.
        underlays : mapping or None, optional
            Scan key to the key of another scan of the study drawn under it
            (an anatomy under a CSI map, or one image under another), or to
            a list or tuple of such keys. With several, the upper scan gets
            one view over each lower scan, in the given order; only the
            first shows the curve and time course, and the others show the
            same repetition (:meth:`ScanView.link_frames
            <hypermri.visualization.scan_view.ScanView.link_frames>`). Every
            view of a 3D image shows the plane in which its lower scan's
            slices lie (:meth:`ScanView.show_anatomy_plane
            <hypermri.visualization.scan_view.ScanView.show_anatomy_plane>`).
            Lower scans should share the session's scanner coordinates; they
            get no view of their own. None or an empty list draws nothing
            underneath.

        Raises
        ------
        ValueError
            If a key is not a scan of the study, a scan lies under itself or
            under two scans (or twice under one), a scan under another one is also chosen on its
            own, or an underlay is given for a scan that is not shown.
            Nothing is changed then.
        Exception
            Any error of :func:`~hypermri.visualization.scan_view.plot_scan`
            for a new scan, for example a scan without data. The scans
            shown before stay shown.
        """
        wanted = _checked_selection(self._keys, scans, underlays or {})
        self._show(wanted)
        # In the list, the lower scan's row names the scan it lies under.
        upper_of = {under: key for key, unders in wanted.items() for under in unders}
        self._updating = True
        try:
            for key in self._keys:
                ticked = key in wanted or key in upper_of
                self._ticks[key].value = ticked
                self._unders[key].value = _menu_value(upper_of.get(key))
                self._unders[key].disabled = not ticked
        finally:
            self._updating = False
        self._update_list_title()

    def _on_list_change(self, change: dict[str, Any]) -> None:
        """Show the scans ticked in the list; report errors in the status line."""
        if self._updating:
            return
        if change["owner"] in self._ticks.values() and not change["new"]:
            # Scans that lay under a scan no longer shown are shown alone.
            key = next(k for k, tick in self._ticks.items() if tick is change["owner"])
            for under in self._unders.values():
                if under.value == key:
                    self._updating = True
                    try:
                        under.value = NO_UNDERLAY
                    finally:
                        self._updating = False
        try:
            scans, underlays = _list_selection(
                self._keys,
                {key: self._ticks[key].value for key in self._keys},
                {key: _underlay_key(self._unders[key].value) for key in self._keys},
            )
            self.select(scans, underlays)
        except Exception as error:  # noqa: BLE001 - shown in the status line
            self.select(
                list(self._shown),
                {k: unders for k, (unders, _) in self._shown.items()},
            )
            self._status.value = f"<b>Could not show the scans:</b> {error}"

    def _show(self, wanted: dict[Hashable, tuple[Hashable, ...]]) -> None:
        """Build the views of ``wanted`` (key to underlay keys) and lay them out."""
        new_views: dict[Hashable, tuple[ScanView, ...]] = {}
        try:
            for key, unders in wanted.items():
                kept = self._shown.get(key)
                if kept is not None and kept[0] == unders:
                    continue
                new_views[key] = self._scan_views(key, unders)
        except Exception:
            for views in new_views.values():
                for view in views:
                    view.close()
            raise
        shown = {}
        for key, unders in wanted.items():
            views = new_views.get(key)
            shown[key] = (unders, self._shown[key][1] if views is None else views)
        for key, (_, views) in self._shown.items():
            if shown.get(key, ((), ()))[1] is not views:
                for view in views:
                    view.unlink()
                    view.close()
        self._shown = shown
        views = [view for _, scan_views in shown.values() for view in scan_views]
        for view in views:
            view.unlink()
        self._status.value = ""
        if len(views) >= 2:
            # A new set of views keeps the arrangement chosen for the last.
            shown_box = self._area.children[0] if self._area.children else None
            arrangement = getattr(shown_box, "arrangement", "rows")
            try:
                box = plot_side_by_side(
                    *views, arrangement=arrangement, under_menus=False
                )
            except ValueError as error:
                for view in views:
                    view.unlink()
                box = plot_side_by_side(
                    *views,
                    show_slices=False,
                    share_masks=False,
                    arrangement=arrangement,
                    under_menus=False,
                )
                self._status.value = f"Views not linked: {error}"
            self._area.children = [box]
        elif views:
            views[0].layout.border_top = None
            self._area.children = views
        else:
            self._area.children = []
        for _, scan_views in shown.values():
            if len(scan_views) > 1:
                scan_views[0].link_frames(*scan_views[1:])

    def _scan_views(
        self, key: Hashable, unders: tuple[Hashable, ...]
    ) -> tuple[ScanView, ...]:
        """Return the views of one scan: alone, or one over each lower scan."""
        label = scan_label(self._study, key)
        if not unders:
            return (plot_scan(self._study[key], name=label),)
        return views_over(
            self._study[key],
            [(self._study[under], _key_text(under)) for under in unders],
            label,
        )

    def _update_list_title(self) -> None:
        """Name the scan list with the number of shown scans."""
        shown = sum(tick.value for tick in self._ticks.values())
        self._list.set_title(0, f"Scans ({shown} of {len(self._keys)} shown)")


def scan_label(study: Any, key: Hashable) -> str:
    """Return the text that names a scan in the scan list.

    Parameters
    ----------
    study : hypermri.Study
        Study with a ``dataframe`` indexed by scan key.
    key : hashable
        Scan key, usually the experiment number.

    Returns
    -------
    str
        ``"E<key>"`` followed by the scan name, sequence class and
        measurement time, as far as the study's table has these columns
        (:data:`SCAN_LABEL_COLUMNS`), joined by ``" · "``.
    """
    row = study.dataframe.loc[key]
    parts = [_key_text(key)]
    parts += [str(row[column]) for column in SCAN_LABEL_COLUMNS if column in row]
    return " · ".join(parts)


def scans_by_time(study: Any) -> list[Hashable]:
    """Return the scan keys of a study ordered by measurement time.

    Parameters
    ----------
    study : hypermri.Study
        Study with a ``dataframe`` indexed by scan key.

    Returns
    -------
    list
        Scan keys, earliest acquisition first, by the clock time
        ``"HH:MM:SS"`` in the column :data:`TIME_COLUMN` (all scans of a
        study are taken to lie on one day). Scans without a time
        (``"MISSING"``) follow; scans with equal or no times keep the order
        of ``study.dataframe``. Without the column, that order is kept.
    """
    keys = list(study.dataframe.index)
    if TIME_COLUMN not in study.dataframe:
        return keys
    times = [str(time) for time in study.dataframe[TIME_COLUMN]]
    has_no_time = [len(time) != 8 or time.count(":") != 2 for time in times]
    order = sorted(range(len(keys)), key=lambda i: (has_no_time[i], times[i]))
    return [keys[i] for i in order]


def _list_selection(
    keys: list[Hashable],
    ticked: Mapping[Hashable, bool],
    lies_under: Mapping[Hashable, Hashable | None],
) -> tuple[list[Hashable], dict[Hashable, tuple[Hashable, ...]]]:
    """Return the shown scans and their underlays chosen in the list.

    ``lies_under`` holds the menu "under" of each row: the scan that the
    row's scan lies under, or None. A ticked row with a menu value draws its
    scan under that scan, which is then shown even if its own row is not
    ticked. Several rows may name the same scan; its underlays keep the
    order of the list.
    """
    lower = {key: lies_under[key] for key in keys if ticked[key]}
    lower = {key: upper for key, upper in lower.items() if upper is not None}
    for key, upper in lower.items():
        if upper in lower:
            raise ValueError(
                f"plot_study: {_key_text(upper)} lies under "
                f"{_key_text(lower[upper])}, so {_key_text(key)} cannot lie "
                f"under it; set one of the menus to none."
            )
    uppers = list(lower.values())
    scans = [key for key in keys if (ticked[key] and key not in lower) or key in uppers]
    underlays: dict[Hashable, tuple[Hashable, ...]] = {}
    for key, upper in lower.items():
        underlays[upper] = (*underlays.get(upper, ()), key)
    return scans, underlays


def _underlay_keys(value: Underlays) -> tuple[Hashable, ...]:
    """Return the underlay keys of one ``underlays`` value as a tuple."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _menu_value(under: Hashable | None) -> Hashable:
    """Return the value of the menu "under" for an underlay key or None."""
    return NO_UNDERLAY if under is None else under


def _underlay_key(value: Hashable) -> Hashable | None:
    """Return the underlay key for a value of the menu "under"."""
    return None if value == NO_UNDERLAY else value


def _key_text(key: Hashable) -> str:
    """Return ``"E5"`` for experiment number 5 and the key itself otherwise."""
    return f"E{key}" if isinstance(key, int) or str(key).isdigit() else str(key)


def _checked_selection(
    keys: list[Hashable],
    scans: Iterable[Hashable],
    underlays: Mapping[Hashable, Underlays],
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Return the shown scans in list order with their underlays, or raise."""
    chosen = list(scans)
    underlays = {key: _underlay_keys(value) for key, value in underlays.items()}
    lower = [under for unders in underlays.values() for under in unders]
    for key in [*chosen, *underlays, *lower]:
        if key not in keys:
            raise ValueError(
                f"plot_study: {key!r} is not a scan of the study; choose one of {keys}."
            )
    for key, unders in underlays.items():
        if not unders:
            continue
        if key not in chosen:
            raise ValueError(
                f"plot_study: a scan under {key!r} needs {key!r} to be shown; "
                "add it to the shown scans."
            )
        if key in unders:
            raise ValueError(f"plot_study: scan {key!r} cannot lie under itself.")
    for under in lower:
        if lower.count(under) > 1:
            raise ValueError(
                f"plot_study: scan {under!r} can lie under only one scan; "
                "choose another scan for the others."
            )
        if under in chosen:
            raise ValueError(
                f"plot_study: scan {under!r} lies under another scan and cannot "
                "also be shown on its own; remove it from the shown scans."
            )
    return {key: underlays.get(key, ()) for key in keys if key in chosen}


def plot_study(
    study: Any,
    scans: Iterable[Hashable] = (),
    underlays: Mapping[Hashable, Underlays] | None = None,
) -> StudyView:
    """Show a study's scan list and the chosen scans in the browser viewer.

    The scan list sits in a collapsible section above the viewer, ordered
    by measurement time (:func:`scans_by_time`). Ticking a scan shows it
    with :func:`~hypermri.visualization.scan_view.plot_scan`; with two or more
    shown scans the views stand next to each other as with
    :func:`~hypermri.visualization.scan_view.plot_side_by_side` (slice bands
    of the others, shared mouse pointer, clicked voxel and masks with one
    mask tool), in rows that the buttons "arrange" switch between rows, side
    by side and stacked; ticking more scans keeps the chosen arrangement. The menu "under" of a ticked row lays that scan under the
    chosen scan, for example an anatomy under a CSI map: one view of the
    chosen scan then shows it above the lower scan, each in its own colours,
    and the lower scan gets no view of its own. Several rows may lie under
    the same scan, for example a coronal and a sagittal anatomy under a 3D
    image: the scan then gets one view over each, in the plane of that
    anatomy's slices, and only the first shows the curve and time course.

    Parameters
    ----------
    study : hypermri.Study
        Study whose scans are listed, in the order of ``study.dataframe``.
        A study loaded with ``load_data="lazy"`` reads each scan's data the
        first time it is shown.
    scans : iterable of hashable, optional
        Keys (experiment numbers) of the scans shown with a view of their
        own at the start. Empty shows only the list.
    underlays : mapping or None, optional
        Scan key to the key of the scan drawn under it at the start, or to
        a list of keys, for example ``{34: 18}`` for anatomy 18 under CSI
        scan 34 or ``{26: [3, 25]}`` for scan 26 over an axial and over a
        coronal anatomy, two views (see :meth:`StudyView.select`). The upper scan must be
        in ``scans``, the lower ones not.

    Returns
    -------
    StudyView
        Widget to display in a notebook. :attr:`StudyView.views` gives the
        view of every shown scan and :meth:`StudyView.select` changes the
        shown scans from Python.

    Raises
    ------
    ValueError
        If a key is not a scan of the study or an underlay is invalid (see
        :meth:`StudyView.select`).
    ImportError
        If the optional visualization dependencies are unavailable.

    Notes
    -----
    A scan that is no longer shown has its view closed; showing it again
    starts a new view (from the settings saved on the scan, see
    :meth:`ScanView.save_settings
    <hypermri.visualization.scan_view.ScanView.save_settings>`). Masks stay in
    ``scan.masks``. The scan's data stay loaded in the study.

    Examples
    --------
    >>> study = Study("/data/study", load_data="lazy")  # doctest: +SKIP
    >>> view = plot_study(study)  # tick scans in the list  # doctest: +SKIP
    >>> view = plot_study(study, scans=[5, 7], underlays={7: 3})  # doctest: +SKIP
    >>> view = plot_study(study, scans=[26], underlays={26: [3, 25]})  # doctest: +SKIP
    >>> view.underlay_views[26, 25]  # scan 26 over scan 25  # doctest: +SKIP
    >>> view.select([5])  # doctest: +SKIP
    >>> view.views[5].add_mask("tumour")  # doctest: +SKIP
    """
    view = StudyView(study)
    scans = list(scans)
    if scans or underlays:
        view.select(scans, underlays)
    return view


__all__ = [
    "SCAN_LABEL_COLUMNS",
    "TIME_COLUMN",
    "StudyView",
    "plot_study",
    "scan_label",
    "scans_by_time",
]
