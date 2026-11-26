# Tutorials

Each tutorial is a short notebook: the code cells are run while the
documentation is built, and the output below them is the real result.
Copy the cells into Jupyter to follow along.

```{toctree}
:maxdepth: 1

loading
reconstruction
spectroscopy_fitting
viewer
hp_pyruvate_csi
```

[Loading data](loading)
: A scanner session as a {class}`~hypermri.Study`, single scans with
  {func}`~hypermri.load_scan`, loading modes and scan parameters.

[Reconstruction](reconstruction)
: The scanner's image next to HyperMRI's reconstruction from raw k-space,
  reconstruction options and how a result records where it came from.

[Spectroscopy fitting](spectroscopy_fitting)
: Fit a synthetic two-peak FID, check the fit and estimate uncertainties.
  Runs without any downloaded data.

[Viewer](viewer)
: Spectroscopy on an anatomy, masks, several scans side by side and the fit
  tool.

[Hyperpolarized pyruvate in vivo](hp_pyruvate_csi)
: Download a published ¹³C pyruvate CSI of a mouse, show it on the anatomy
  and compare the spectra at two body temperatures.

The loading, reconstruction and viewer tutorials use the public Lego phantom
session described in {doc}`../getting_started/first_look`.
