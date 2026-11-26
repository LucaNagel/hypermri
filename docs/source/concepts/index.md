# Concepts

Short explanations of the ideas the code is built on. Read them when a
tutorial step raises a *why*.

```{toctree}
:maxdepth: 1

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

{doc}`data_model`
: How images, spectra and fit results are stored: dimension names, order,
  coordinates and attributes.

{doc}`loading_structure`
: What a loaded scan is made of: vendor handler, `MRIExperiment`, sequence
  class, and where parameters and images are found.

{doc}`scanner_geometry`
: How read/phase/slice indices become x/y/z in mm, and how the viewer puts an
  anatomy slice under a CSI map. With interactive figures.

{doc}`coordinate_systems`
: Voxel indices, scanner coordinates in mm and how scans are aligned.

{doc}`reconstruction`
: Scanner image versus HyperMRI's own reconstruction, and which code runs when.

{doc}`fourier_convention`
: The one rule that puts a raw reconstruction on the scanner's voxel grid.

{doc}`viewer`
: What runs in Python and what in the browser, and what a snapshot contains.

{doc}`architecture`, {doc}`configuration`, {doc}`registry_overview`
: How loading finds the vendor, sequence and reconstruction of a scan.

{doc}`dce_model_dispatch`, {doc}`fid_phase_simulation`
: Background for kinetic models and FID phase.
