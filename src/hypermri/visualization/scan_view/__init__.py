"""Interactive browser viewer of scans: map, voxel curve and tools.

:func:`plot_scan` (also reached through :func:`hypermri.visualization.plot`
and ``scan.plot``) shows a scan as an ``anywidget`` whose drawing and
voxel/spectral selection run in the browser. Python prepares the processed
curves of every voxel in the shown plane once and sends them as one float32
buffer; clicking a voxel, moving the curve cursor, hover and crosshair are
then handled by JavaScript. Only changes that need new processing (slice,
repetition, channel, complex part, frequency unit, line broadening) ask
Python for a new block. The browser rescales colours and axes but makes no
scientific decision; the processing is that of
:class:`~hypermri.visualization.render_data.PlotDataModel`.

The curve is the spectrum along ``acquisition_idx`` unless another curve
dimension such as ``repetition_idx`` is chosen. Images without
``acquisition_idx`` show a curve along echoes, repetitions or another
dimension, or the image alone.

Layers: an anatomical scan can be drawn under the map and parameter maps
such as fit results above it, matched to the shown plane in scanner
coordinates (:mod:`hypermri.visualization.spatial_overlay`).

Tools (``ScanView.tool``): "masks" draws polygon or freehand masks, stored in
``scan.masks`` (:mod:`hypermri.visualization.mask_editing`); "fit" fits all
voxels with a Lorentz or Voigt lineshape and reviews single voxels
(:mod:`hypermri.visualization.fit_tool`); further tools fit curves, show
signal decay and run IDEAL separation where the data allow it.

View settings can be saved on the scan and are restored by the next view
with the same settings context
(:mod:`hypermri.visualization.state_persistence`).
"""

from hypermri.visualization.scan_view.entry import plot_scan, plot_side_by_side
from hypermri.visualization.scan_view.side_by_side import SideBySideView
from hypermri.visualization.scan_view.widget import ScanView

__all__ = ["ScanView", "SideBySideView", "plot_scan", "plot_side_by_side"]
