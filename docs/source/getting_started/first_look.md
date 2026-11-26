---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# First look

This page loads a complete scanner session, picks one scan, looks at its data
and opens it in the viewer. It takes five lines of code.

## Example data

All tutorials use the public *Lego phantom* session of the BrukerAPI test
data set (Pšorn, Zenodo,
[doi:10.5281/zenodo.4048253](https://doi.org/10.5281/zenodo.4048253),
CC-BY-4.0): 45 Bruker ParaVision 6.0.1 scans of a Lego brick in water,
including FLASH, RARE, EPI, diffusion, CSI and single-voxel spectroscopy.
Download `20200612_094625_lego_phantom_3_1_2.zip` (780 MB) and unpack it.

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

## Load the session

{class}`~hypermri.Study` finds every scan folder, detects the vendor and the
sequence of each scan and sorts them by acquisition time. It reads parameters
and the scanner's own reconstruction, not the raw data.

```{code-cell} ipython3
from hypermri import Study

study = Study(study_path, verbose=False)
study.dataframe[["Sequence Type", "Scan Name", "Measurement Time"]].head(8)
```

## One scan

A study is indexed by the scan number. Scan 8 is a RARE image; its object
knows its sequence, parameters and reconstructions.

```{code-cell} ipython3
scan = study[8]
scan
```

The image is an {class}`xarray.DataArray`. Every axis has a name, the scanner
coordinates `x`, `y`, `z` (in mm) are attached, and attributes record the
orientation and where the values came from ({doc}`../concepts/data_model`).

```{code-cell} ipython3
scan.primary_data
```

## Open the viewer

In Jupyter, `scan.plot()` as the last line of a cell shows the interactive
viewer. Click the image or use the arrow keys to move through the voxels, and
change slice and colour range with the controls.

```{code-cell} ipython3
view = scan.plot()
```

```{raw} html
<iframe class="hm-snapshot" src="../_static/snapshots/rare.html" loading="lazy"
        title="RARE scan in the HyperMRI viewer"></iframe>
```

The frame above is that view saved with `view.to_html("rare.html")`: a single
HTML file that opens in any browser without Python. Controls that need a new
computation, such as another slice, stay greyed out there.

## Next

- {doc}`../tutorials/loading`: loading modes, single scans and the study table.
- {doc}`../tutorials/viewer`: anatomy under spectroscopy, masks and the fit
  tool.
