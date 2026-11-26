---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# Viewer

`scan.plot()` opens the HyperMRI viewer: a browser component that shows an
image, map or spectrum together with the curve under the mouse. It works in
JupyterLab and VS Code. Python computes the data; the browser draws it, so
moving the mouse and dragging the colour range need no round trip to Python
({doc}`../concepts/viewer`).

The frames on this page are saved viewers. They respond to the mouse; controls
that need a new computation are greyed out.

```{code-cell} ipython3
from pathlib import Path

study_path = Path("20200612_094625_lego_phantom_3_1_2")
```

```{code-cell} ipython3
:tags: [remove-cell]

# The documentation build reads its local copy of the phantom.
import os

study_path = Path(os.environ["HYPERMRI_DOCS_PHANTOM"])
```

```{code-cell} ipython3
from hypermri import Study

study = Study(study_path, verbose=False)
rare, csi, nspect = study[8], study[30], study[33]
```

## Spectroscopy on an anatomy

Scan 30 is a 16 × 16 chemical shift imaging (CSI) scan, scan 8 a RARE image
of the same phantom. With `anatomy=` the CSI map is drawn over the RARE image
at its true position in scanner coordinates, and the spectrum of the voxel
under the mouse is shown next to it.

```{code-cell} ipython3
view = csi.plot(anatomy=rare)
```

```{raw} html
<iframe class="hm-snapshot" src="../_static/snapshots/csi_map.html" loading="lazy"
        title="CSI map on the RARE anatomy"></iframe>
```

Things to try in Jupyter:

- click a voxel or move with the arrow keys; `view.selection` holds the
  current voxel;
- zoom into the spectrum, or show only a range of it with
  `view.set_frequency_window((4.0, 5.5), unit="ppm")`;
- change colour map, range and opacity in the controls above the image.

Scans without a spatial grid, such as the repeated single-voxel spectra of
scan 33, open with the spectrum and its time course:

```{code-cell} ipython3
nspect_view = nspect.plot()
```

```{raw} html
<iframe class="hm-snapshot" src="../_static/snapshots/nspect.html" loading="lazy"
        title="Repeated spectra in the HyperMRI viewer"></iframe>
```

## Masks

Masks are drawn in the viewer's mask tool: `rare.create_mask()` opens it, and
saved masks are stored in `rare.masks` on the scan's voxel grid. A mask can
also be made from code; here every voxel brighter than 30 % of the maximum,
which is the water around the Lego brick:

```{code-cell} ipython3
magnitude = abs(rare.primary_data)
rare.masks["water"] = (magnitude > 0.3 * magnitude.max()).astype("float32")
```

A mask belongs to the scan it was drawn on, but it can be applied to any other
scan of the session. {meth}`~hypermri.MRIExperiment.apply_mask_to` projects it
through scanner coordinates onto the CSI grid, weighting each CSI voxel by the
fraction it covers, and averages the FIDs inside:

```{code-cell} ipython3
water_fid = rare.apply_mask_to(csi, "water", reduction="mean")
water_fid.dims
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

spectrum = np.fft.fftshift(np.fft.fft(water_fid.squeeze().values))

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(csi.ppm, abs(spectrum), color="C0")
ax.set_xlabel("chemical shift (ppm)")
ax.set_ylabel("|spectrum| (a.u.)")
ax.invert_xaxis()
fig.tight_layout()
```

## Several scans

{func}`~hypermri.visualization.scan_view.plot_side_by_side` shows several
images next to each other. Each view shows where the others' slices cut it,
the mouse pointer is shared and masks drawn in one appear in all:

```python
from hypermri.visualization.scan_view import plot_side_by_side

plot_side_by_side(study[8], study[3])
```

Views with a spectrum or time course get a row of their own below the
images. To choose the rows yourself, pass one list of rows:
`plot_side_by_side([[study[8], study[3]], [csi]])`.
The buttons "arrange" above the views switch between these rows, all views
side by side, and all views one below the other while you look at them.
The menu "under" of a view lays it under another view while they are shown,
for example an anatomy under the CSI map; "none" shows both again. The slider
"anatomy" above the upper view then chooses the slice of the anatomy below,
also for a CSI with a single slice; "follow map" lets it follow the map again.

For a whole session, {func}`~hypermri.visualization.study_view.plot_study`
adds the scan table above the viewer; ticking a scan shows it, and the menu
"under" puts an anatomy below a map:

```python
from hypermri.visualization.study_view import plot_study

plot_study(study)
```

## Fit tool

The fit tool fits the spectrum of the selected voxel or mask with the model of
{doc}`spectroscopy_fitting` and draws the fitted lines over the data:

```python
view = csi.plot(anatomy=rare)
view.tool = "fit"
```

Choose metabolites and start values in the tool, run the fit, and keep a
result with **Save** or `view.save_fit("water")`. Saved fits are kept on the
scan (`csi.saved_fits`) and appear as map overlays in every view opened later;
`view.show_fit_curves()` draws the fitted spectrum and its residual.

## Sharing a view

{meth}`~hypermri.visualization.scan_view.ScanView.to_html` saves the current
view as one HTML file, as used on this page. It opens in any browser without
Python:

```python
view.to_html("csi_on_rare.html")
```
