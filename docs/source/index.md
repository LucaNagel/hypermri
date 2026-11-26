---
html_theme.sidebar_secondary.remove: true
---

# HyperMRI

**Load, reconstruct, fit and explore MRI and MR spectroscopy data in Python.**

HyperMRI reads scanner data (Bruker ParaVision first), reconstructs it on the
scanner's own voxel grid, keeps every result as labelled
[xarray](https://docs.xarray.dev) data with physical coordinates, and shows
images and spectra in an interactive browser viewer.

```python
from hypermri import Study

study = Study("path/to/study")   # every scan of a session, sorted by time
scan = study[8]                  # one scan, already resolved to its sequence
scan.plot()                      # interactive viewer in Jupyter
```

The viewer below is such a view, saved as a plain HTML page. Click a voxel or
use the arrow keys; no Python is running behind it.

```{raw} html
<iframe class="hm-snapshot" src="_static/snapshots/csi_map.html" loading="lazy"
        title="CSI scan of the Lego phantom in the HyperMRI viewer"></iframe>
```

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Start
:link: getting_started/index
:link-type: doc

Install HyperMRI and open your first scan.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/index
:link-type: doc

Loading, reconstruction, spectroscopy fitting and the viewer, step by step.
:::

:::{grid-item-card} {octicon}`light-bulb` Concepts
:link: concepts/index
:link-type: doc

Data model, coordinates, the Fourier convention and how the viewer works.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public function and class, grouped by task.
:::
::::

HyperMRI is research software and not a medical device; see
{doc}`getting_started/research_use`.

```{toctree}
:hidden:

getting_started/index
tutorials/index
concepts/index
api/index
development/index
```
