Viewer
======

:func:`hypermri.visualization.plot` and ``scan.plot`` show a scan or a
labelled array in one browser viewer: a map of the selected plane and the
spectrum (or another curve) of the selected voxel. The viewer is an
``anywidget``; clicking, the curve cursor, hover and crosshair run in the
browser, and Python prepares new data only when the processing changes.

Entry point
-----------

.. autofunction:: hypermri.visualization.plot

.. autoclass:: hypermri.visualization.PlotOptions

Use the same entry for scans or explicit labelled arrays::

   from hypermri.visualization import PlotOptions, plot

   view = plot(scan, PlotOptions(plot_height=480))
   view = plot(scan.scanner_data, scan=scan)
   view.tool = "masks"

``scan.plot`` takes the same settings as keywords, for example
``scan.plot(plot_height=480, slice_indices={"slice_idx": 2})``.
``scan.create_mask()`` and ``scan.interactive_fit()`` open the viewer with the
mask or fit tool selected. Options of the earlier Python-side plotter, such as
``return_controller`` or ``quick_fit``, raise a ValueError that names their
replacement.

Display geometry
----------------

The geometry functions accept labelled arrays and display state without
creating widgets. Source values, units, attrs and storage dtype remain intact.

.. automodule:: hypermri.visualization.geometry
   :members: orient_xyz_slice, xyz_spatial_extent, nearest_slice_index,
             extract_1d_coord, get_spatial_dim_label, format_spatial_axis_label,
             normalize_coordinate_mode, format_xyz_axis_label,
             display_voxel_centres, snap_display_point,
             mask_coverage_in_data_order, mask_coverage_in_display_order

Render data and state
---------------------

.. automodule:: hypermri.visualization.render_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hypermri.visualization.state
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: AnatomyState

.. autoclass:: hypermri.visualization.state.AnatomyState
   :show-inheritance:

Image plane and curve dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One module decides which two dimensions form the image and along which
dimension the curve (a spectrum, or a curve along echoes, repetitions or
b-values) runs. ``plot`` offers these choices in its menus and
``PlotDataModel.get_render_payload`` draws them, so the default shown in a
menu is the one drawn.

.. automodule:: hypermri.visualization.view_dimensions
   :members:

Spatial overlays
~~~~~~~~~~~~~~~~

:func:`hypermri.visualization.plot` and ``plot_scan`` accept the same
``overlay`` argument. One module registers it under names, names its layers,
chooses the first layer and computes the colour range; cutting the layer to
the shown plane belongs to ``PlotDataModel.get_render_payload``.

.. automodule:: hypermri.visualization.spatial_overlay
   :members:

Live fit monitor (Beta)
~~~~~~~~~~~~~~~~~~~~~~~

The Fit Monitor button of the ``interactive_fit`` workflow opens a local page
(``127.0.0.1`` only) that shows the trace of the selected voxel's fit:
measured and model signal, grid searches, frequency and T2* inside their
bounds, amplitudes and phases. With "follow voxel" the page follows the
viewer's voxel; Trace Voxel refreshes it on demand. The server stops when the
viewer closes.

.. automodule:: hypermri.visualization.fit_monitor
   :members: FitMonitor

Browser view
------------

``plot_scan`` shows a CSI map and the selected voxel's spectrum as one
anywidget. Python sends the processed spectra of all voxels of the shown plane
once; selecting a voxel or a spectral position, hovering and the colour range
then run in the browser without a Python call. Slice, repetition, channel,
part, unit and line broadening recompute the plane in Python.

``plot_scan(scan, curve_dim="repetition_idx")`` shows the selected voxel's
curve along another dimension instead of the spectrum, for example a time
course over repetitions. The allowed names are those of
:func:`~hypermri.visualization.view_dimensions.curve_dim_choices` for the
shown plane; other names raise ``ValueError``. The spectral position then
becomes an ordinary slider that recomputes the curves, and the ROI mean runs
along the chosen dimension. Without ``curve_dim`` a curve dimension saved in
the state is kept when it is still possible, otherwise the spectrum is shown.
While the display processing reduces the curve dimension (repetitions
averaged, channels combined), the map shows the processed image and the curve
keeps every entry, as in ``plot``; with averaged repetitions a red band marks
the averaged range on the curve.

The colour range of the map has the range modes of
:func:`hypermri.visualization.plot`: "Percentage" (0-100 % of the finite range
of the shown map; the browser menu also offers percentages of the whole
plane), "Min/Max" (shown as "full range of the data": limits in signal units
that return to the full plane range when the complex part or line broadening
changes) and "Absolute" (shown as "fixed values": limits that are kept). One
slider "range" with two handles sets the minimum and the maximum. A new view
without a given or saved state starts at 0-80 % of the shown map and of the
anatomy under it (maximum at 80 %, so that weaker signal stays visible) and at the voxel of the shown plane
with the strongest signal (largest magnitude along its curve). The menu, the
sliders and :meth:`~hypermri.visualization.scan_view.ScanView.set_map_range`
only redraw the map; a minimum that is not below the maximum is rejected.

"save view" (or :meth:`~hypermri.visualization.scan_view.ScanView.save_settings`)
stores the view settings, among them the colour range, the selected voxel,
part, unit, overlay layer, colormap and opacity, on the scan under the key
``settings_context`` (default ``"plot_scan"``). The next ``plot_scan(scan)``
without an explicit ``state`` starts from them. A view of a bare DataArray has
no scan to store them on.

The mask toolbar adds, colours, fades and deletes masks and draws polygon
(click corners, Enter or "Finish ROI" closes the outline and switches
drawing off; with freehand "Finish ROI" only switches drawing off, Backspace removes the last corner, Escape cancels) or freehand
outlines in add or erase mode. Each edit is stored at once
in ``scan.masks`` as float32 coverage between 0 and 1 per
voxel on the scan grid ``(read_idx, phase_idx, slice_idx)``. The drawn
outlines themselves are kept beside the coverage at the precision they were
drawn (``view.mask_outlines``, stored with the masks) and drawn as lines in
the mask colour over the voxel coverage; after an outline in erase mode,
the boundary of the drawn area that remains is shown instead. Changing the
slice while a polygon is being clicked closes and stores it first. The
display switch "outlines"
(``view.set_display(show_mask_outlines=False)``) hides them. The same edits are
available from Python; outline points are (x, y) in the units of the map
axes. "ROI mean" replaces the voxel curve by means inside the selected mask,
each voxel weighted by its coverage: the mean of the shown spectra (chosen
part, unit and line broadening) over the mask voxels of the shown slice, the
mean of their fitted spectra when fit results are shown, and, on its own axes
at the top and right, the mean magnitude of the supplied samples over the
whole mask (for FID data the mean FID), the same curve for every view. The fitted mean costs one fit evaluation per covered voxel, about
2 s for 170 voxels of 350 points.

Once fit results exist (from the fit tool or ``scan.fit_results``), "fit
statistics" (on by default) shows below the mask toolbar, for every mask and
fitted line, the mean ± standard deviation of amplitude, frequency (Hz and
ppm), T2* in ms and, for a Voigt fit, the Gaussian FWHM over all voxels of the
mask, each voxel weighted by its coverage and failed fits left out
(:func:`~hypermri.visualization.fit_tool.mask_fit_table`). The table follows
mask edits and new fits. In Python,
:func:`~hypermri.analysis.anatomical.roi.mask_statistics` returns the same
statistics for any parameter maps as a table:

.. code-block:: python

   from hypermri.analysis.anatomical.roi import mask_statistics

   table = mask_statistics(scan.fit_results, scan.masks)
   table[table.variable == "fit_t2s"][["mask", "metabolite", "mean", "std"]]

With a spectrum and more than one repetition a third panel shows a time
course of the selected voxel: its value at the selected spectral point over
the repetitions. Dragging across the spectrum, or typing limits into "from"
and "to" (in the spectral unit, snapped to the nearest point), replaces the
point by the mean over that spectral range, marked by a red band;
double-clicking the spectrum or "Follow point" returns to the point. "mask
mean" adds the same curve averaged over the selected mask, weighted by
coverage, and follows mask edits. Clicking a time point selects that
repetition. The dimension along which the course runs is chosen with "along"
when several are possible, and :meth:`~hypermri.visualization.scan_view.ScanView.set_time_course_range`,
:meth:`~hypermri.visualization.scan_view.ScanView.show_time_course_mask` and
:meth:`~hypermri.visualization.scan_view.ScanView.set_time_course_dim` do the
same from Python. The range is stored as spectral indices in
``state.spectral.time_course_range``. The panel is hidden when repetitions
are averaged, when only one exists, and when ``curve_dim`` shows another
curve. Python keeps the spectra of every repetition of the shown plane
(float32, rows x columns x spectral points x repetitions) and sends the
browser only the range mean, rows x columns x repetitions.

``plot_scan`` also shows images without ``acquisition_idx``, such as anatomy
or multi-echo images. The curve then runs along the first of inversion,
b-value, echo, repetition and channel dimensions present, for example the
signal over the echoes; an image without such a dimension is shown alone,
without a curve panel and with a slider for the third spatial axis
(``curve_dim="slice_idx"`` draws the curve along the slices instead).
Unit and line broadening are offered only for spectra. Single-voxel spectra
(PRESS, STEAM, single pulse, ISIS, NSPECT), which have fewer than two of
``read_idx``, ``phase_idx`` and ``slice_idx``, get the missing dimensions
with size one and are shown without an image panel: the spectrum and, with
several repetitions, its time course, or with
``curve_dim="repetition_idx"`` the value over the repetitions.

With ``anatomy=``, a single-voxel scan shows the anatomy slice nearest the
centre of the excited volume with the volume's outline in red: the box of a
PRESS or STEAM voxel or the band of a slice spectroscopy slab. The
anatomy slice slider above the plots moves through the anatomy; a slice that the volume
does not cross shows a note instead of the outline. The geometry is the
nominal one from the Bruker ``method`` parameters (``PVM_VoxArr*``, or
``PVM_SliceThick`` and ``PVM_SPackArr*`` for one slice package with one
slice) without chemical shift displacement, and only the patient position
``Head_Prone`` is supported. The slab path is tested with invented
parameters only. Scans without a selective excitation (single pulse) have
no outline and raise ``ValueError`` when an anatomy is given. The anatomy
colour range starts at the value range of the shown anatomy slice.

When a fit-result Dataset is given as overlay (``plot_scan(scan,
overlay=scan.fit_results)``) or the scan has ``fit_results``, the curve
panel draws the fitted spectrum of the selected voxel over the measured one,
with a legend. It is computed as in ``plot`` with ``fit_results.hp``
(``reconstruct`` for the selected voxel, then the same unit and part as the
data); ``state.spectral.fit_mode`` chooses the sum or the single
metabolites, and ``state.spectral.show_fit = False`` hides it. Python
computes it again only when the voxel changes.

A fit-result Dataset such as ``scan.fit_results`` can be shown as the data
itself, not only as overlay: ``plot_scan(scan.fit_results, scan=scan)``.
Every variable with two spatial dimensions and without ``acquisition_idx``
is a map; fitted spectra are left out. The "variable" menu (or
``view.set_variable("fit_freqs")``) switches between them, ``fit_amps`` is
shown first (``variable=`` chooses another). The curve panel shows the
value along repetitions when present, and the ``metabolite`` slider shows
the metabolite name next to its index. Real-valued maps start with the
complex part "Real", so that negative values such as frequency offsets keep
their sign.

Arrow keys move the selected voxel in the direction seen on screen, whatever
the rotation or coordinate direction; ``,`` and ``.`` move the spectral
position.

With ``anatomy=`` an anatomical scan is drawn under the map, as in
:func:`hypermri.visualization.plot`. The map then uses scanner coordinates
(``"xyz"``) unless ``coordinate_mode`` is given. The "map opacity" slider
above the plots sets ``state.anatomy.opacity``; the box "show" of the
"Anatomy" section hides the anatomy in the browser only, and its slider
"range" sets the colour range of the anatomy in percent of the shown slice's
value range (``state.anatomy.vmin`` and ``vmax``;
``view.set_anatomy_range(2, 98)``). Next to it, an
anatomy slice slider appears when the anatomy has several slices in the shown
orientation and works like the overlay slice slider below: "follow map" (the
default) draws the anatomy slice nearest to the map, and a chosen
slice is kept in ``state.anatomy`` (``view.set_anatomy_slice(5)``; None
follows the map again). In scanner coordinates a map with several slices moves
to its slice nearest to the chosen anatomy slice; a map with a single slice,
such as a 2D CSI, stays, and a warning appears when the anatomy slice lies
outside the map's slice thickness.

With ``overlay=`` a parameter map, for example fit results, is drawn
semi-transparently above the map with its own colour bar, cut to the shown
plane the same way as in :func:`hypermri.visualization.plot`. The controls
of the "Overlay" section choose the layer (named by its Dataset variable, for
example ``fit_amps``), the metabolite, the colormap and the colour limits, on
one slider with two handles, either in percent ("%") of the layer's finite
value range or as absolute values ("values") in the layer's unit; the
"overlay opacity" slider stands above the plots; a minimum that is not below the maximum is
rejected and leaves the settings unchanged. A new layer, metabolite or range
mode starts from the full range. The overlay slice slider appears when the
overlay has several slices in the shown orientation. "follow map" (the
default) draws the overlay plane matching the map. A chosen slice is kept in
``state.overlay``; in scanner coordinates (``"xyz"``) the map moves to its
nearest slice and a warning appears when the overlay slice lies outside the
map's slices, and moving the map lets the overlay follow it again, as in
:func:`hypermri.visualization.plot`. In ``"rps"`` the chosen slice stays until
"follow map" is selected. An overlay given as a scan offers a "source"
menu (hypermri, scanner or DICOM reconstruction; DICOM pixels are read on first
use); a new source starts from its full range and follows the map. Fit results of the shown scan inherit its
scanner coordinates when they keep its index grid. Overlay difference modes
are not offered: they subtract the overlay at the selected spectral position,
which the browser changes without Python. Fits are not supported yet; use
:func:`hypermri.visualization.plot` for them and for difference overlays::

   from hypermri.visualization.scan_view import plot_scan

   view = plot_scan(scan)                 # or plot_scan(scan, anatomy=t2w_scan)
   # plot_scan(scan, curve_dim="repetition_idx") shows a time course instead
   view                      # display in a notebook cell
   view.state.spatial.slice_indices   # follows the browser selection
   view.set_map_range("Absolute", vmin=0.0, vmax=5e4)
   view.save_settings()      # the next plot_scan(scan) starts from this view

   view = plot_scan(scan, overlay=fit_results, anatomy=t2w_scan)
   view.set_overlay(metabolite="lactate", range_mode="Min/Max", vmin=0.0, vmax=2e4)
   view.set_overlay(cmap="viridis")
   view.set_overlay_opacity(0.5)
   view.set_overlay_slice(2)     # None: follow the map again
   view.set_anatomy_slice(5)     # None: follow the map again

   view = plot_scan(scan, overlay=other_scan)
   view.set_overlay_source("dicom")   # "data", "scanner_data" or "dicom"

   view.add_mask("tumour", color="#00aa00")
   view.draw_polygon([(4.0, 6.0), (12.0, 6.0), (12.0, 14.0)], mask="tumour")
   scan.masks["tumour"]      # coverage on the scan grid
   view.mask_outlines["tumour"]  # the drawn outline, in data-grid voxel units

The interleaved FISP sequence ``user:fisp_grad_1metab_cs`` acquires two
metabolites in alternating repetitions. ``plot_scan`` splits them into a
``metabolite_idx`` dimension of size 2 (repetitions 0, 2, 4, ... and 1, 3,
5, ...), as ``plot`` does; an odd last repetition is dropped.
``split_metabolites=False`` keeps the repetitions as acquired and
``split_metabolites=True`` splits other data with at least two repetitions.
Whenever a split is possible, the browser shows a "split metabolites"
checkbox and ``view.set_split_metabolites(False)`` switches back to the
repetitions as acquired; slider indices are clipped to the new sizes, masks
and overlay are kept. With the metabolites split, the curve panel also draws
the curve of the other metabolite at the selected voxel (dotted, as in
``plot``); the "other metabolite" checkbox or
``state.spectral.show_secondary_metabolite_curve`` switches it off.

.. code-block:: python

   view = plot_scan(scan, curve_dim="repetition_idx", split_metabolites=True)

For three-dimensional images (with ``read_idx``, ``phase_idx`` and
``slice_idx``) the "view plane" menu chooses the dimension perpendicular to
the image, as the view-plane menu of ``plot``: Slice shows the read and phase
plane, Read the phase and slice plane, Phase the read and slice plane. The
perpendicular dimension becomes a slider; a curve dimension now shown in the
image is chosen again. The "rotate" button turns the image by a further 90
degrees (``state.spatial.rotation``); voxel indices and masks keep their
meaning. ``view.set_view_plane("read_idx")`` and ``view.set_rotation(90)`` do
the same from Python.

The "display" controls switch how the map is drawn, as the display
checkboxes of ``plot``: smoothing between voxel centres ("off", "fast" or
"best"; ``state.spatial.interpolation`` and ``zsmooth_mode``), "min → NaN"
(values below the colour minimum stay transparent, their tooltip still shows
the value), a scalebar in the lower right corner (only when the image axes
carry a physical voxel spacing), the colour bar of the map, grid lines in all
panels, hover tooltips (values under the mouse on the map, the spectrum and
the time course) and the crosshair. They start from the state, so image
data begin with "best" smoothing and the scalebar, and tooltips and grid lines
are off as in ``plot``. The colour bar starts on only for fitted maps such as
``plot_scan(scan.fit_results, scan=scan)``. ``view.set_display(smoothing="off", show_tooltips=True)`` sets them
from Python. "map colours" in the same group and "anatomy colours" in the
anatomy group choose the colormaps of the map and of the anatomy below it
(``state.spatial.cmap`` and ``state.anatomy.cmap``). With one image laid over
another, for example ``plot_scan(image, anatomy=other)``, two different
colormaps tell the two grey images apart. The lists offer common colormaps;
``set_display(map_cmap=...)`` accepts any name the ``cmap`` package knows.

.. code-block:: python

   view = plot_scan(scan)
   view.set_display(smoothing="off", show_scalebar=False, show_grid_lines=True)
   view.set_display(map_cmap="inferno", anatomy_cmap="gray")

The "k-space" controls (the switch "k-space | image") show the map and curves
of the image or of its spatial k-space, and apply a Gaussian or Fermi spatial filter with widths in cycles/mm
(``view.set_kspace(show_kspace, spatial_filter)``). The view shows a copy:
:func:`~hypermri.analysis.spatial_filtering.apply_spatial_filter` filters the
image, and :func:`~hypermri.analysis.spatial_filtering.spatial_to_kspace`
transforms it along Read and Phase, each slice on its own; the data of the
view and of the scan stay unchanged. In k-space the map axes are spatial
frequencies in 1/mm, and anatomy, overlays, voxel outlines, masks, ROI mean,
fit curves, slice bands and the linked pointer are off until the image is shown
again. The choice of space is ``state.spatial.show_image_space`` and is saved
with the view settings; the filter is a choice of the view.

.. code-block:: python

   from hypermri.analysis.spatial_filtering import SpatialFilterOptions

   view.set_kspace(True)
   view.set_kspace(False, SpatialFilterOptions(kind="gaussian", sigma_per_mm=0.1))

The "spectrum" controls also hold the signal processing of ``plot``, each
shown only where it applies: the switch "FID | Spec" shows the acquired time
signal instead of the spectrum, a switch "ppm | Hz" chooses the unit, "points" sets the number of spectral points (more points than
acquired fill the FID with zeros), "denoise" keeps that many principal
components, "repetitions" shows the mean ("average") or the running sum
("cumulative sum") over the given repetition range, and "combine channels"
combines the receive channels with the chosen method (rss, sum, mean or
adaptive). Averaged repetitions and combined channels lose their slider. The
settings are stored in ``state.spectral``; the acquired data are not
changed. ``view.set_processing(...)`` sets them from Python.

.. code-block:: python

   view = plot_scan(scan)
   view.set_processing(repetitions="average", repetition_range=(2, 10))
   view.set_processing(npoints=2048, combine_channels=True, combine_method="rss")

The "y-axis range" control chooses the y range of the curve: "fit to
curves" fits the shown curve, "whole plane" fits all voxels of the shown plane, and "map
colours" follows the colour range of the map (``plot``'s "Couple Y-Limits",
stored as ``state.spectral.couple_ylims``). For spectra, "limit" restricts the
frequency axis to a window in the shown unit and fits the y range to the
curve inside it; the window is stored in ``state.spectral.display_range`` and
applies only while the same unit is shown and not in the FID view.
``view.set_curve_scale(...)`` and ``view.set_frequency_window(...)`` set them
from Python.

.. code-block:: python

   view = plot_scan(scan)
   view.set_curve_scale("map")
   view.set_frequency_window((160.0, 185.0))
   view.set_frequency_window(None)

Several images of one session, for example an axial and a coronal anatomy,
can be shown next to each other with ``plot_side_by_side``. Image-only views
share the first row; each view with a spectrum or time course gets a row of
its own below, so that its curve has the width of the page. One list of rows,
such as ``plot_side_by_side([[axial, csi], [coronal]])``, places the views by
hand. The buttons "arrange" above the views switch at any time between these
rows, all views side by side in one row, and all views stacked one below the
other (``box.arrangement``, or ``arrangement="stacked"`` when plotting);
views, links and settings stay. ``plot_study`` keeps the chosen arrangement
when scans are ticked. The menu "under" of each view lays it under another
view later on: that view is shown again with the image underneath, in the
plane of its slices, and the lower view is hidden until "none" is chosen
(``box.set_underlays({0: 2})`` lays the first view under the third). The
opacity slider of the upper view then fades between both. Each image gets a
coloured bar, and the slice currently shown in each other image appears as a
semi-transparent band in that image's colour, where it crosses this one
(like the slice position lines of a scanner console). The band follows when
the other image shows a new slice, already while its slider is dragged;
parallel images draw no band. The mouse pointer is shared: over one image,
the others mark the same scanner position with a white ring, filled when
the position lies inside their shown slice and open when it lies in front
of or behind it. The band
width is the slice spacing, or ``slice_thickness_mm`` for a single slice.
Clicks pass through the bands, so masks can be drawn as usual. Masks are
shared: opening the mask tool, choosing a mask, the drawing tool and the
add/erase mode act on all images, and an outline drawn on one image is
copied to the others (each of their voxels takes the coverage of the drawn
image's voxel at its centre, for the edited slice only). Only the first
image shows the inputs of the mask tool; the others show the chosen mask
with its volume, and outlines can be drawn on every image: corners
clicked in another image are closed by "Finish ROI" of the first. Each scan stores
the mask on its own grid in ``scan.masks`` and uses it for its ROI mean and
volume. ``view.show_slices_of(other)``, ``view.link_pointer(other)`` and
``view.link_masks(other)`` link views made separately;
``plot_side_by_side(..., share_masks=False)`` keeps the masks apart. With
linked pointers, a click on a voxel of one image selects the voxel at the
same scanner position in the others and moves them to its slice. An image
whose stack does not reach that position takes its nearest slice; an image
whose plane does not contain the position beside it stays unchanged. This
needs scanner coordinates ``x``, ``y``, ``z`` (VisuCore geometry) for every
image.

To lay one image over another instead, pass the second as ``anatomy``: it is
resampled into the shown plane in scanner coordinates and drawn underneath,
and the opacity slider (or ``view.set_anatomy_opacity``) fades between both.
An image over an anatomy starts with the colormap ``turbo`` on an anatomy in
``bone``; CSI and other spectroscopic maps keep ``turbo``.
``view.show_anatomy_plane()`` turns a 3D image to the plane in which the
anatomy's slices lie
(:func:`~hypermri.visualization.geometry.view_plane_for_anatomy`), so the
anatomy keeps its in-plane resolution. ``plot_scan(..., image_only=True)``
shows the image panel alone, and ``view.link_frames(other)`` shows the same
repetition (or other non-spatial index) in several views of one scan. The
bands that show where the slices of linked views lie are off at the start;
the display switch "other slices" (``view.set_display(show_slice_lines=True)``)
shows them in the muted colour of each view's bar.
The display switch "masks" (``view.set_display(show_masks=False)``) hides
the masks over the map; they are kept and still count for the ROI mean.

.. code-block:: python

   box = plot_side_by_side(axial, coronal, csi)
   (axial_view, coronal_view), (csi_view,) = box.rows
   box.arrangement = "stacked"  # one view per row; "side by side": one row
   box.set_underlays({0: 2})  # the axial anatomy under the CSI map

   plot_side_by_side([[axial, csi], [coronal]])  # rows chosen by hand

   overlaid = plot_scan(axial, anatomy=other_axial)
   overlaid.set_anatomy_opacity(0.5)

``view.unlink()`` removes a view from all slice, pointer and mask links.

The status line below the plots shows the hints for the mouse and keys. How
long the browser took to redraw and how long Python took for the last data
block appear only with ``plot_scan(..., show_timings=True)``,
``plot_side_by_side(..., show_timings=True)`` or ``view.show_timings = True``.

To choose the scans of a study in the viewer, ``plot_study`` shows the scan
list of a :class:`~hypermri.Study` in a collapsible section ("Scans") above
the viewer, sorted by measurement time (scans without a time last).
Ticking scans shows them, side by side and linked as with
``plot_side_by_side`` when two or more are ticked; scans without scanner
coordinates are shown unlinked and the status line says why. Choosing
another scan in the menu "under" of a ticked row lays this row's scan under
the chosen scan, as ``anatomy`` does: both appear in one image, the lower
scan with its own colour map, and the lower scan is no longer shown on its
own. Several rows may lie under the same scan, for example a coronal and a
sagittal anatomy under a 3D bSSFP scan: the upper scan is then shown once
over each lower scan, in the plane of that scan's slices. Only the first of
these images has the curve panel; all show the same repetition
(``StudyView.underlay_views`` holds them by ``(upper, lower)``). In Python, ``underlays={upper: lower}`` or
``underlays={upper: [lower, ...]}`` does the same; the upper scan is in
``scans``, the lower ones are not. Scans are read only when they are
first shown; a scan
that is no longer shown has its view closed, but keeps its masks in
``scan.masks``.

.. code-block:: python

   from hypermri.visualization.study_view import plot_study

   view = plot_study(study)  # tick scans in the list
   view = plot_study(study, scans=[5, 7], underlays={7: 3})  # 3 under 7
   view = plot_study(study, scans=[26], underlays={26: [25, 28]})  # per plane
   view.views[5].add_mask("tumour")

.. automodule:: hypermri.visualization.scan_view
   :members: plot_scan, plot_side_by_side, ScanView, SideBySideView
   :inherited-members: AnyWidget, DOMWidget, Widget, LoggingHasTraits, HasTraits, HasDescriptors, object

Viewing without Python
~~~~~~~~~~~~~~~~~~~~~~

``view.to_html("scan.html")`` saves the shown plane as a standalone page.
It opens in any browser without Python: voxel choice, curve position, hover
values, colour range, curve axes and the display switches work as in the
notebook, while controls that need new data from Python are shown switched
off. ``plotly_js="cdn"`` keeps the file small by loading Plotly.js from the
web. A notebook opened without a running kernel loads the same Plotly.js
version from the web after a few seconds.

.. automodule:: hypermri.visualization.scan_view.snapshot
   :members:

.. automodule:: hypermri.visualization.study_view
   :members: plot_study, StudyView, scan_label, scans_by_time

.. automodule:: hypermri.visualization.metabolite_split
   :members: split_interleaved_metabolites, resolve_metabolite_split

.. automodule:: hypermri.visualization.volume_outline
   :members:

.. automodule:: hypermri.visualization.slice_lines
   :members:

Browser layout
--------------

The browser view is assembled from named panels (``image``, ``curve``) and
control groups (slice sliders, spectrum settings, colour range, anatomy,
overlay, masks and others). ``layout.py`` places each group in one of three
places: ``TOP_BAR`` (always visible above the plots), ``OPTION_SECTIONS``
(titled sections under the collapsible "Plot options" button, closed by
default) and ``BELOW_PLOTS`` (under the plots). Moving a name between these
tuples moves its inputs; ``browser_layout`` checks that every group is
placed once and leaves out sections without visible inputs. The browser
code is a set of small JavaScript files in ``visualization/browser/``, one
per panel or control group, which Python joins into one script when the
view is created. A new control group needs a builder function in its own
file, an entry in ``CONTROL_GROUP_BUILDERS`` in ``viewer.js`` and a name in
``CONTROL_GROUPS``.

All plot panels have the same height. The map panel is as wide as its
aspect ratio needs at that height, so the image fills it. In
``plot_side_by_side`` each row of views is one grid (an
``ipywidgets.GridBox``) whose rows are shared, so the plots of all views in
it start and end at the same height even when one view shows its plot
options. Each view gets a share of the row in proportion to the width its
image needs at the full plot height, plus room for a curve; in a narrow row
all images are lowered alike. A spectrum or time course keeps at least 40 %
of its view, and the map beside it is lowered instead. The colour bars of the map
and the overlay sit directly right of the map and have its height.
``plot_scan(..., name="...")`` shows a name in bold above the plots;
``plot_study`` uses the scan's entry of the scan list.

Tools such as the mask editor are chosen in the "tool" menu under
the plots, which shows the inputs of one tool at a time right after the
menu; "none" hides them all. Masks stay drawn on the map whichever tool is
chosen. ``TOOLS`` in ``layout.py`` lists the tools in menu order, and
``view.tool = "masks"`` chooses one from Python. A new tool is a control
group as above whose name is added to ``TOOLS`` instead of one of the
places. The mask drawing menu is labelled "draw". The mask and fit tools
also offer a "Tutorial" box; ticked, it lists the steps of the tool above
its inputs.

The "fit" tool is offered for spectra (not for fit results or images). It
holds a table of spectral lines. The row "Find lines" offers two ways to
fill it: "Add at cursor" adds a line at the spectral position chosen in the
spectrum, and "Use lines of earlier fit" is described below. Automatic line
detection is not offered in the tool;
:func:`~hypermri.visualization.fit_tool.detect_peaks` still fills the table
from Python. Added lines start with the search range and T2* bounds of the row "Defaults
for new lines". Each row of the table sets the start frequency and its full search width
in Hz, the T2* start value and bounds in ms, whether frequency or T2* stay
fixed and whether the line is fitted at all. "Run fit" fits every voxel of
the data with ``fit_spectrum`` (Lorentz or Voigt lineshape; "workers" is
``n_jobs``) and shows the result as overlay ``fit``, with its fitted
spectrum in the curve panel. On a scan the fit is stored as
``scan.fit_results_review`` and ``scan.interactive_fit_results`` (with the
fit parameters and a copy of the fitted data), where the review of
``interactive_fit`` finds it. Python owns the table
(``hypermri.visualization.fit_tool``); the browser only sends edits.

"Advanced" shows the remaining fit settings: smallest distance of fitted line
frequencies, repeats and grid points of the start-value search (0 chooses
them from the widest search window), fitted FID points (0 fits all), the
phase-band delay tau in ms (0 derives it from the scan timing with
``phase_band_tau_s_from_scan``) and whether T2* starts of coupled lines are
clipped into their ratio band. The table gains coupling columns: "T2*
leader" makes a line follow the T2* of another line with the ratio
``T2*(line) / T2*(leader)`` between "ratio min" and "max" (equal values fix
it), and "phase" joins the line to the common phase law
``phi0 + 2 pi f tau`` within "tol °" degrees.

After a fit, "Align spectra" shifts every voxel spectrum by the difference
between its amplitude-weighted fitted frequency and the mean of these over
all voxels (``align_spectra_by_fit_frequency``, as "Align Spectra" of
``interactive_fit``), so that B0 differences no longer move the lines. The
view then shows the aligned spectra and the shifted fit; "Run fit" fits the
aligned spectra. The scan keeps its acquired data, and the aligned fit and
a copy of the aligned samples are stored under the review names above.
Repeated alignment adds up (coordinate ``frequency_alignment_offset_hz``);
"Undo align" shows the spectra and the fit before the first alignment.

Start values can come from an earlier fit, separately for frequency and
T2*: the columns "freq" and "T2*" under "start from fit" replace the typed
start of that line in every voxel by the value the earlier fit found there
(``seed_start_values``); "win ms" then bounds T2* to a window around each
voxel's start instead of min/max. The earlier fit is the last fit of the
view, or else a fit stored on the scan (``interactive_fit_results``,
``fit_results_review`` or ``fit_results``), which the view loads as the
hidden overlay ``fit`` when its voxels match the shown data. "Use lines
of earlier fit" fills the table with the lines of that fit, both start values taken
from it. Each line has its own colour in the table and its frequency
search range is drawn as a band of that colour in the spectrum. "fit" and
"residual" in the spectrum options show the fitted spectrum and the
residual with a dashed zero line, as in ``plot``
(``ScanView.show_fit_curves``).

"result table" in the row "Review" (on by default) shows at the top of the tool, above the
settings, the fitted values of
the selected voxel, one row per line: amplitude (``|fit_amps|``), phase in
degrees, frequency in Hz (and ppm) and T2* in ms, plus the shared Gaussian
FWHM of a Voigt fit
(:func:`~hypermri.visualization.fit_tool.voxel_fit_table`). A frequency,
T2* or Gaussian width is tinted by its bound status: green within the
bounds, orange at the lower bound, red at the upper bound, grey when fixed.
The table follows the selected voxel.

"Line from residual" searches the residual of the selected voxel between
the two "residual Hz" values for one more line and adds it only when the
data need it. After a fit, an empty range starts at ±10 Hz around the start
frequency of the first line of the table.

"show parameters" below the plots (on its own line below an open tool) opens the measurement parameters of the
scan (``scan.get_parameters()``, grouped as ``PARAMETER_SCHEMA``) as a
table with a text filter. While it is open, "original names" adds the
scanner's own name after each parameter, for example ``PVM_Matrix`` after
``matrix`` (Bruker scans,
:attr:`~hypermri.vendors.bruker.BrukerHandler.PARAMETER_SOURCES`).

.. code-block:: python

   view = plot_scan(scan)
   view.tool = "fit"        # or choose "fit" in the tool menu
   # Add at cursor, edit the table, Run fit; then
   scan.fit_results_review  # the fit of all voxels

The tool "curve_fit" fits along the curve axis instead: ADC for b-values
(``bvalue_idx`` in s/mm2), T2 for echo times (``echo_idx`` in s) and T1 for
inversion times (``inversion_idx`` in ms, optionally with a baseline). The
map of all voxels becomes the overlay ``curve_fit``; the curve panel shows
fit and residual of the selected voxel. The result stays in the view until
it is saved (see below).

.. code-block:: python

   view = plot_scan(scan)    # curve axis bvalue_idx
   view.tool = "curve_fit"  # then "Run fit"
   view.curve_fit_result["adc"]  # mm²/s

The tool "ideal" is offered when the curve axis is ``echo_idx``. It
separates chemical components in complex multi-echo images with
:func:`~hypermri.analysis.quantitative.fit_ideal`. Enter the component
names and their frequencies in Hz relative to the reference frequency; the
echo spacing comes from the echo axis (in s or ms, equally spaced) unless it
is entered in ms. "field offset" and "Theta" choose the fitted unknowns
(both by default, as in the solver). The overlay ``ideal`` then holds one
magnitude map per component and ``field_offset`` in Hz; the complex result
stays in the view until it is saved (see below).

.. code-block:: python

   view = plot_scan(scan)  # curve axis echo_idx, complex images
   view.tool = "ideal"    # names "pyruvate, lactate", Hz "0, 390", Run IDEAL
   view.ideal_result["fit_amps"].sel(metabolite="lactate")

Each of the three tools has a row "Save" with a name field and "Save fit".
It copies the last fit into ``saved_fits`` of the plotted object: the
attribute ``scan.saved_fits`` of a scan, or ``data.attrs["saved_fits"]`` of
a plotted DataArray. Each entry is a
:class:`~hypermri.visualization.saved_fits.SavedFit` with the kind of fit,
the result Dataset and, for the spectral fit, its fit options. The view
keeps the maps as an overlay of that name, which stays when the tool runs
again, and a view opened later on the same object offers the saved fits as
overlays (hidden at the start). Nothing is written to disk; an empty name
chooses ``"<kind>_<n>"``.

.. code-block:: python

   view.save_fit("kidney_adc")           # same as the button
   scan.saved_fits["kidney_adc"].results["adc"]  # mm²/s
   del scan.saved_fits["kidney_adc"]     # forget it

.. automodule:: hypermri.visualization.saved_fits
   :members:

.. automodule:: hypermri.visualization.fit_tool
   :members:

.. automodule:: hypermri.visualization.curve_fit_tool
   :members:

.. automodule:: hypermri.visualization.ideal_tool
   :members:

Signal decay in the browser view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~hypermri.visualization.signal_decay.plot_signal_decay` opens CSI raw
data in the browser view with the tool "signal decay". A small plot shows
the decay over the acquired k-space points (mean magnitude or band power,
also cumulative) for the unfiltered and filtered data, a straight line
fitted from the signal end on, the signal end and a suggested signal end
(noise onset). The tool chooses the time-domain filter along the acquired
points ("None" keeps the points up to the signal end), a spatial Gaussian
or Fermi k-space filter and the signal end (number or click on the plot);
map and spectrum show the data reconstructed with that filter. "Confirm"
stores ``scan.signal_end`` and ``scan.filter_params``; acquired data are
never changed.

.. code-block:: python

   from hypermri.visualization.signal_decay import plot_signal_decay

   view = plot_signal_decay(csi, bands_hz=[(-330, -150), (200, 420)])
   view
   # after "Confirm":
   csi.filter_signal(**csi.filter_params)

.. automodule:: hypermri.visualization.signal_decay
   :members: plot_signal_decay

.. automodule:: hypermri.visualization.signal_decay_tool
   :members:

.. automodule:: hypermri.analysis.spectroscopy.peak_fit_setup
   :members:

.. automodule:: hypermri.analysis.spectroscopy.fit_seeding
   :members:

.. automodule:: hypermri.visualization.scan_parameters
   :members:

.. automodule:: hypermri.visualization.layout
   :members: browser_layout, browser_script
