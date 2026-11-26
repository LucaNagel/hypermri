---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Hyperpolarized pyruvate in vivo

This page works with a real hyperpolarized ¹³C experiment: chemical shift
imaging (CSI) of a mouse abdomen after injection of hyperpolarized
[1-¹³C]pyruvate, acquired at two body temperatures. HyperMRI downloads the
data, loads the session, shows each CSI on its anatomy and compares the
spectra of the two temperatures.

## Example data

The session is published on Zenodo (Nagel,
[doi:10.5281/zenodo.20429962](https://doi.org/10.5281/zenodo.20429962),
CC-BY-4.0, 29 MB). It holds the data of

> Gottwald W, Nagel L, et al. *Temperature dependent chemical shifts of
> pyruvate and lactate enable in vivo hyperpolarized ¹³C MRSI thermometry.*
> npj Imaging 3, 19 (2025).
> [doi:10.1038/s44303-025-00081-3](https://doi.org/10.1038/s44303-025-00081-3)

Please cite the article when you use the data. Each CSI was started 10 s
after the end of the injection on a 7 T Bruker small-animal scanner
(ParaVision 7).

{func}`~hypermri.datasets.fetch_example_data` downloads the archive once,
checks its SHA-256 checksum, unpacks it into a local cache and returns the
session folder. Later calls only return the folder. The environment variable
`HYPERMRI_DATA` chooses another cache folder.

```{code-cell} ipython3
from hypermri.datasets import fetch_example_data

study_path = fetch_example_data("hp_pyruvate_csi_mouse", quiet=True)
```

## The session

```{code-cell} ipython3
from hypermri import Study

study = Study(study_path, verbose=False)
study.dataframe[["Sequence Type", "Scan Name"]]
```

Scans 13 and 22 are the two CSI scans, with the body temperature of the
mouse in their names. Scan 19 is an axial T2-weighted RARE image planned for
the CSI.

```{code-cell} ipython3
csi_38, csi_31, anatomy = study[13], study[22], study[19]
csi_38.primary_data.sizes
```

Each CSI has 14 × 12 voxels in one slice and 700 points per FID; the
spectral axis spans 155 to 195 ppm.

## CSI on the anatomy

```{code-cell} ipython3
view_38 = csi_38.plot(anatomy=anatomy)
```

```{code-cell} ipython3
:tags: [remove-cell]

# The documentation build saves the view next to the page; the file is not
# part of the repository.
from pathlib import Path

generated = Path("../_static/generated")
generated.mkdir(parents=True, exist_ok=True)
view_38.to_html(generated / "hp_pyruvate_csi_38c.html", plotly_js="cdn")
```

```{raw} html
<iframe class="hm-snapshot" src="../_static/generated/hp_pyruvate_csi_38c.html"
        loading="lazy" title="Pyruvate CSI at 38 °C on the T2 anatomy"></iframe>
```

Click a voxel or move with the arrow keys to see its spectrum. Pyruvate is
the large peak near 171 ppm; lactate (183 ppm), pyruvate hydrate (179 ppm)
and alanine (176.5 ppm) are its metabolic and chemical neighbours.

## Two temperatures

The article measures temperature from the chemical shifts of pyruvate and
lactate. A first look: sum the magnitude spectra of all voxels of each scan
and compare them.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np


def summed_magnitude_spectrum(csi):
    """Return the sum over all voxels of the magnitude spectra of a CSI scan."""
    fid = csi.primary_data.transpose("acquisition_idx", ...)
    spectra = np.fft.fftshift(np.fft.fft(fid.values, axis=0), axes=0)
    return np.abs(spectra).reshape(spectra.shape[0], -1).sum(axis=1)


fig, ax = plt.subplots(figsize=(7, 3))
for csi, label in [(csi_38, "38 °C"), (csi_31, "31 °C")]:
    spectrum = summed_magnitude_spectrum(csi)
    ax.plot(csi.ppm, spectrum / spectrum.max(), label=label)
ax.set_xlim(186, 166)
ax.set_xlabel("chemical shift (ppm)")
ax.set_ylabel("normalised sum (a.u.)")
ax.legend()
fig.tight_layout()
```

In these summed spectra the distance between the lactate and pyruvate
maxima differs by about 0.1 ppm between the two temperatures, a fifth of the
line width of about 0.5 ppm. Reading such small shifts needs a peak fit,
which the article does voxel by voxel. The
{doc}`spectroscopy_fitting` tutorial shows how to fit spectra with HyperMRI.
