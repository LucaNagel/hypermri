# Tutorials

Each tutorial is a short notebook: the code cells are run while the
documentation is built, and the output below them is the real result.
Copy the cells into Jupyter to follow along.

The loading, reconstruction and viewer tutorials use the public Lego phantom
session described in {doc}`../getting_started/first_look`.

```{toctree}
:hidden:

loading
reconstruction
spectroscopy_fitting
viewer
```

::::{grid} 1 2 2 3
:gutter: 3
:class-container: hm-tutorial-cards

:::{grid-item-card} {octicon}`file-directory` Loading data
:link: loading
:link-type: doc

```{raw} html
<img class="hm-card-thumb" src="../_static/thumbnails/loading.png"
     alt="RARE image of the Lego phantom" loading="lazy" onerror="this.remove()">
```

A scanner session as a `Study`, single scans with `load_scan()`, loading
modes and scan parameters.
:::

:::{grid-item-card} {octicon}`sync` Reconstruction
:link: reconstruction
:link-type: doc

```{raw} html
<img class="hm-card-thumb" src="../_static/thumbnails/reconstruction.png"
     alt="FLASH image reconstructed from raw k-space" loading="lazy" onerror="this.remove()">
```

The scanner's image next to HyperMRI's reconstruction from raw k-space,
reconstruction options and where a result came from.
:::

:::{grid-item-card} {octicon}`graph` Spectroscopy fitting
:link: spectroscopy_fitting
:link-type: doc

```{raw} html
<img class="hm-card-thumb" src="../_static/thumbnails/spectroscopy_fitting.png"
     alt="Two-peak spectrum with fit and residual" loading="lazy" onerror="this.remove()">
```

Fit a synthetic two-peak FID, check the fit and estimate uncertainties.
Runs without any downloaded data.
:::

:::{grid-item-card} {octicon}`eye` Viewer
:link: viewer
:link-type: doc

```{raw} html
<img class="hm-card-thumb" src="../_static/thumbnails/viewer.png"
     alt="CSI map on the anatomy with the spectrum of one voxel" loading="lazy" onerror="this.remove()">
```

Spectroscopy on an anatomy, masks, several scans side by side and the fit
tool.
:::
::::
