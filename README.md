# HyperMRI 🧲

**Python research tools for loading, reconstructing, and visualizing MRI data.**

## Research-use notice

HyperMRI is research software. It has not been validated or approved as a
medical device and is not intended for clinical diagnosis, treatment
decisions, or other clinical use. Its outputs may be incomplete, incorrect, or
unsuitable for a particular experiment. Users are responsible for independently
validating the methods and results and for complying with applicable ethical,
privacy, institutional, and regulatory requirements.

Do not publish patient data, identifying metadata, or confidential research
data in GitHub issues, discussions, or examples.

## Features

* **Bruker-first loading and reconstruction**: scoped stable support for
  Bruker ParaVision data, with automatic vendor and sequence detection.
  Siemens support is Beta; Agilent/Varian, Magritek, and Bruker TopSpin
  support is Experimental. See the
  [support matrix](https://github.com/LucaNagel/hypermri/blob/main/docs/source/support.rst)
  before relying on a workflow.
* **[xarray](https://docs.xarray.dev/en/stable/) data model**: MRI data is
  represented as labelled `xarray.DataArray` and `xarray.Dataset` objects with
  metadata and physical coordinates.
* **Interactive visualization**: the Plotly-based viewer supports
  multidimensional images, spectra, fit curves, and overlays in Jupyter.
* **Analysis and export**: reusable reconstruction and fitting workflows plus
  derived DICOM export for supported geometries.

## Documentation

The documentation is built from `docs/source`. Public hosting on Read the
Docs is planned but not live yet; build it locally with

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/_build/html
```

and open `docs/_build/html/index.html`. It contains:

*   **Getting started**: installation and a first look at a study.
*   **Tutorials**: loading, reconstruction, spectroscopy fitting and the viewer.
*   **Concepts**: data model, coordinates, reconstruction and architecture.
*   **Development**: setup, testing, tutorials and releases.

## Installation

HyperMRI is published on [PyPI](https://pypi.org/project/hypermri/) as
`hypermri`. Install the released package with `pip`, or install from a source
checkout to edit the package or build the documentation.

**Prerequisites:**

* **Python 3.12.x** in an isolated environment. Other Python minor versions
  are excluded by the package metadata. The test suite runs on Python 3.12.
  [Anaconda](https://anaconda.org/) or Miniconda is recommended.
* **Git** for a source checkout only. [Install Git](https://git-scm.com/downloads).

**1. Create an environment**

Create an environment named `hypermri` to keep package versions such as `numpy` and `matplotlib` consistent.

```bash
conda create -n hypermri python=3.12
conda activate hypermri
```

**2. Install from PyPI**

```bash
pip install "hypermri[notebook]"
```

For loading and numerical analysis alone, use `pip install hypermri`.

**3. Install from source (optional)**

Choose this path to edit the package or build the documentation.

```bash
conda install git
git clone https://github.com/LucaNagel/hypermri.git
cd hypermri
pip install -e ".[notebook]"
```

For loading and numerical analysis alone, use `pip install -e .`. The
`visualization` extra adds the viewer; `notebook` also adds Jupyter. Optional
Siemens TWIX loading uses `.[siemens]`, and motion correction uses `.[motion]`.

To build the documentation, install its additional dependencies with
`pip install -e ".[docs]"`. Quoting extras works reliably in shells such as
`zsh`.

The notebook extra requires `ipykernel >=7.4,<8`, which excludes the older
kernel affected by queued notebook cells. In VS Code, select the `hypermri`
environment as the notebook kernel and restart an already running kernel
after updating. Check the installed dependencies with `python -m pip check`.

**4. Build HTML (optional, source checkout only)**:

```bash
cd docs
make html
```

After building, open `docs/_build/html/index.html` in your web browser.


## Usage

```python
from hypermri import Study
from hypermri.datasets import fetch_example_data

# Download the released example session (29 MB, cached after the first call)
# or point Study to a Bruker study that you are authorised to use.
study = Study(fetch_example_data("hp_pyruvate_csi_mouse"))

# Show the CSI scan on its anatomical image
study[13].plot(anatomy=study[19])
```

The example session is hyperpolarized [1-¹³C]pyruvate CSI of a mouse abdomen
(Zenodo, [doi:10.5281/zenodo.20429962](https://doi.org/10.5281/zenodo.20429962),
CC BY 4.0). Please cite Gottwald et al., npj Imaging 3, 19 (2025),
[doi:10.1038/s44303-025-00081-3](https://doi.org/10.1038/s44303-025-00081-3),
when you use it.

The separate optional real-data integration suite uses Tomáš Pšorn's public
[BrukerAPI ParaVision 6.0.1 Lego-phantom corpus](https://doi.org/10.5281/zenodo.4048253)
(version 1.1.0, CC BY 4.0); it is not downloaded during installation or normal
unit tests.

For stock Bruker CSI, `raw-reco` removes the first 70
digital-filter/transient samples once per FID and rebuilds the time and
frequency axes for the remaining samples. The original handler FID remains
complete, and scanner-reconstructed `2dseq` data is not cut again. See the
[support matrix](https://github.com/LucaNagel/hypermri/blob/main/docs/source/support.rst)
for the exact reconstruction contract.

The tutorials in the documentation show more examples.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/LucaNagel/hypermri/blob/main/CONTRIBUTING.md) for the review,
testing, data-safety, and licensing rules. Pull requests must pass the required
checks and be approved by the maintainer before they are merged into `main`.

Historical and current contributors are credited in
[CONTRIBUTORS.md](https://github.com/LucaNagel/hypermri/blob/main/CONTRIBUTORS.md).
Because the public repository starts with a history-free root commit, that file
is the canonical public attribution rather than the private development history.

## License

HyperMRI is licensed under the
[GNU Affero General Public License v3.0 only](https://github.com/LucaNagel/hypermri/blob/main/LICENSE)
(`AGPL-3.0-only`).

Third-party attribution and modification notices are retained in
[NOTICE](https://github.com/LucaNagel/hypermri/blob/main/NOTICE).
The adapted BrukerMRI parameter readers retain their
[Apache License 2.0 text and attribution](https://github.com/LucaNagel/hypermri/blob/main/LICENSES/brukerMRI-Apache-2.0.txt).
Both notices and licence texts are included in the source and wheel packages.

Commercial use is permitted under the license. Its copyleft and source-offer
requirements apply when covered modified versions are conveyed and when users
interact with a modified version over a network. It does not by itself prohibit
a purely private internal fork. The exact license text controls; this summary
is not legal advice.

Copyright © HyperMRI contributors.
