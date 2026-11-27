# Concepts

Short explanations of the ideas the code is built on. Read them when a
tutorial step raises a *why*.

```{toctree}
:hidden:

data_model
loading_structure
scanner_geometry
coordinate_systems
reconstruction
fourier_convention
viewer
architecture
configuration
registry_overview
dce_model_dispatch
fid_phase_simulation
```

## Data and geometry

::::{grid} 1 2 2 4
:gutter: 3
:class-container: hm-overview-cards

:::{grid-item-card} {octicon}`database` Data model
:link: data_model
:link-type: doc

How images, spectra and fit results are stored: dimension names, order,
coordinates and attributes.
:::

:::{grid-item-card} {octicon}`package` Inside a loaded scan
:link: loading_structure
:link-type: doc

Vendor handler, `MRIExperiment`, sequence class, and where parameters and
images are found.
:::

:::{grid-item-card} {octicon}`move-to-end` Read/phase/slice to x/y/z
:link: scanner_geometry
:link-type: doc

How voxel indices become millimetres and how an anatomy slice lands under a
CSI map. With interactive figures.
:::

:::{grid-item-card} {octicon}`location` Coordinate systems
:link: coordinate_systems
:link-type: doc

Voxel indices, scanner coordinates in mm and how scans are aligned.
:::
::::

## Reconstruction and viewer

::::{grid} 1 2 2 4
:gutter: 3
:class-container: hm-overview-cards

:::{grid-item-card} {octicon}`sync` Reconstruction
:link: reconstruction
:link-type: doc

Scanner image versus HyperMRI's own reconstruction, and which code runs when.
:::

:::{grid-item-card} {octicon}`pulse` Fourier convention
:link: fourier_convention
:link-type: doc

The one rule that puts a raw reconstruction on the scanner's voxel grid.
:::

:::{grid-item-card} {octicon}`eye` How the viewer works
:link: viewer
:link-type: doc

What runs in Python and what in the browser, and what a snapshot contains.
:::
::::

## Code structure and background

::::{grid} 1 2 2 4
:gutter: 3
:class-container: hm-overview-cards

:::{grid-item-card} {octicon}`stack` Architecture
:link: architecture
:link-type: doc

How loading finds the vendor, sequence and reconstruction of a scan.
:::

:::{grid-item-card} {octicon}`gear` Configuration
:link: configuration
:link-type: doc

Which values belong in a configuration file and which in Python code.
:::

:::{grid-item-card} {octicon}`list-unordered` Registry overview
:link: registry_overview
:link-type: doc

How vendor and sequence types are found from detection patterns and
scanner aliases.
:::

:::{grid-item-card} {octicon}`beaker` Models and FID phase
Background for kinetic models
({doc}`dce_model_dispatch`) and the phase of a simulated FID
({doc}`fid_phase_simulation`).
:::
::::
