# How the viewer works

`scan.plot()` returns a {class}`~hypermri.visualization.scan_view.ScanView`,
an [anywidget](https://anywidget.dev) that draws with Plotly.js in the
browser. The split between Python and browser is what makes it fast.

```{mermaid}
flowchart LR
    A[scan or DataArray] --> B["PlotDataModel<br/>(Python)"]
    B -- "one float32 block<br/>per shown plane" --> C["browser<br/>(JavaScript)"]
    C -- "slice, repetition,<br/>processing changes" --> B
    C --> D[map, curve,<br/>time course]
```

**Python** prepares the processed curves of every voxel in the shown plane
once and sends them as one binary block. All scientific decisions stay
there: phasing, line broadening, units, resampling of anatomy and overlays,
and the fits of the analysis tools.

**The browser** draws the map and curves from that block. Clicking a voxel,
the arrow keys, the spectral cursor, hover, the colour range and the
crosshair need no round trip to Python. Only a change that needs new
processing, such as another slice, repetition, channel or line broadening,
asks Python for a new block.

## Layout and tools

The viewer is built from named panels (`image`, `curve`, `time`) and control
groups, arranged by {mod}`hypermri.visualization.layout`. *Tools* are control
groups shown one at a time from the tool menu: masks, the spectral fit, curve
fits, signal decay and IDEAL separation. Each tool's numerical work lives in an
analysis module; the tool only collects inputs and shows results.

## Several scans

{func}`~hypermri.visualization.scan_view.plot_side_by_side` shows several
views that share the selected position and masks, and
{func}`~hypermri.visualization.study_view.plot_study` lets you tick scans of
a {class}`~hypermri.Study` and lay one under another.

## Snapshots

{meth}`ScanView.to_html <hypermri.visualization.scan_view.ScanView.to_html>`
writes the current view to one HTML file: the viewer code, the data of the
shown plane and either the full Plotly.js (`plotly_js="inline"`, works
offline) or a link to it (`"cdn"`, smaller). Everything that runs in the
browser keeps working there. With an anatomy under the map, every anatomy
slice is saved as well, so its slider still works. Other controls that need
Python are greyed out. The interactive frames in this documentation are such
snapshots.
