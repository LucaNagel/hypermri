---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# Reconstruction

A Bruker scan folder holds two versions of the same measurement: the raw
k-space data (`fid` or `rawdata.job0`) and the image ParaVision computed from
it (`2dseq`). HyperMRI reads the scanner image and also reconstructs the raw
data itself. This tutorial compares both for the FLASH scan of the phantom.

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
from hypermri import load_scan

flash = load_scan(study_path / "3")
flash.get_available_reconstructions()
```

Each entry names a reconstruction type, the class that performs it and its
default options. `seq2d` reads the scanner image; `raw` reconstructs k-space.

## Two reconstructions

```{code-cell} ipython3
scanner = flash.reconstruct("seq2d")
raw = flash.reconstruct("raw")
scanner.dtype, raw.dtype
```

The scanner image is real-valued (magnitude); HyperMRI's image is complex and
keeps the phase. Both have the same labelled dimensions and the same scanner
coordinates, so they can be compared voxel by voxel:

```{code-cell} ipython3
dict(raw.sizes) == dict(scanner.sizes), (raw.x == scanner.x).all().item()
```

ParaVision stores integers with its own scaling, so the two differ by one
global factor. After fitting that factor, the remaining difference is
rounding:

```{code-cell} ipython3
import numpy as np

a = abs(raw).isel(slice_idx=1).values.ravel()
b = scanner.isel(slice_idx=1).values.ravel()
scale = (a @ b) / (a @ a)
{
    "correlation": np.corrcoef(a, b)[0, 1],
    "scale": scale,
    "max relative difference": np.max(abs(scale * a - b)) / b.max(),
}
```

The viewer shows both reconstructions; drag through the voxels of either to
compare them.

```{raw} html
<div class="hm-snapshot-pair">
<iframe class="hm-snapshot" src="../_static/snapshots/flash_raw.html" loading="lazy"
        title="FLASH reconstructed by HyperMRI from raw k-space"></iframe>
<iframe class="hm-snapshot" src="../_static/snapshots/flash_scanner.html" loading="lazy"
        title="FLASH as reconstructed by ParaVision"></iframe>
</div>
```

*Left: HyperMRI from raw k-space. Right: ParaVision `2dseq`.*

The FFT convention behind the raw reconstruction (where k-space and image
centres lie, and how odd matrix sizes are handled) is described in
{doc}`../concepts/fourier_convention`.

## Options

Keyword arguments of {meth}`~hypermri.MRIExperiment.reconstruct` override the
default options of the reconstruction. Unknown names are rejected with a
{class}`~hypermri.exceptions.ReconstructionError` that lists the accepted
ones. The precision of the result is one such option:

```{code-cell} ipython3
flash.reconstruct("raw", data_precision="float64").dtype
```

```{code-cell} ipython3
from hypermri.exceptions import ReconstructionError

try:
    flash.reconstruct("raw", zero_filling=2)
except ReconstructionError as error:
    print(error.__cause__)
```

The result of the last successful call is stored on the scan:
`flash.data` holds the raw reconstruction and `flash.scanner_data` the
scanner image.

## Which reconstructions run automatically

The loading mode decides which reconstructions run when a scan is loaded. A
sequence declares this as its reconstruction policy:

```{code-cell} ipython3
flash.get_reconstruction_policy()
```

`"lazy"` runs none, `"minimal"` reads the scanner image and `"full"` also
reconstructs the raw data.

## Provenance

Every reconstruction records its geometry and origin as attributes: the
orientation, the signal domain, the slice thickness and the commit of
HyperMRI that produced it.

```{code-cell} ipython3
{key: raw.attrs[key] for key in ("domain", "slice_orientation", "slice_thickness_mm", "unit")}
```

## Next

- {doc}`spectroscopy_fitting`: fit spectra.
- {doc}`viewer`: look at reconstructions in the browser viewer.
