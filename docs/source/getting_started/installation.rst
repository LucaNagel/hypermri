Installation
============

Follow these steps to install HyperMRI into an isolated environment. The
`hypermri project on PyPI <https://pypi.org/project/hypermri/>`_ is the approved
distribution name, but version ``0.1.0`` is not published yet. The version
currently returned by ``pip install hypermri`` is the historical ``0.0.3``
package from 2022, with legacy metadata and code. It is **not** the development
version described by this repository. Install the current version from a source
checkout until the ``0.1.0`` release is announced.

Prerequisites
-------------

* Python 3.12.x in an isolated environment; other Python minor versions are excluded
* Anaconda/Miniconda or a Python 3.12 interpreter with ``venv`` available
* ``ipykernel >=7.4,<8``, installed by the ``notebook`` extra
* The test suite and synthetic notebook-kernel test run on Python 3.12
* Git tooling for a source checkout
* JupyterLab or VS Code for the interactive viewer

Create an environment
---------------------

Conda environment:

.. code-block:: bash

   conda create -n hypermri python=3.12
   conda activate hypermri

Install from source
-------------------

Choose this path to edit the package or build the documentation.

GitHub Desktop workflow:

1. Click **Clone** on the repository page and copy the HTTPS URL.
2. In GitHub Desktop choose **Clone a repository from the Internet** and paste the URL.

Command-line workflow:

.. code-block:: bash

   conda install git
   git clone https://github.com/LucaNagel/hypermri.git
   cd hypermri
   pip install -e ".[notebook]"

Install the documentation dependencies only when you need them:

.. code-block:: bash

   pip install -e ".[docs]"

Quoting extras works reliably in shells such as ``zsh``.

Installation extras
-------------------

``pip install -e .`` installs loading, reconstruction and numerical analysis.
Choose optional features explicitly:

* ``pip install -e ".[visualization]"`` installs the browser viewer (Plotly and anywidget).
* ``pip install -e ".[notebook]"`` adds the viewer, Jupyter and the supported kernel.
* ``pip install -e ".[siemens]"`` adds the experimental Siemens TWIX reader.
* ``pip install -e ".[motion]"`` adds SimpleITK motion correction.
* ``pip install -e ".[test]"`` includes notebook/Siemens dependencies for the tests.
* ``pip install -e ".[dev,docs]"`` provides development, tests and documentation.

Missing optional features fail when requested, with the extra to install.
Numerical loading does not initialize notebook widgets. Matplotlib remains a
core dependency because existing analysis and sequence modules use its static
plotting helpers. GitPython and the unused SymPy dependency have been removed;
provenance uses the optional Git executable and retains an explicit missing label.

Notebook kernel
---------------

In VS Code, select the ``hypermri`` Python 3.12 environment as the notebook
kernel. Restart an already running kernel after updating its packages.
The required ipykernel range excludes versions affected by the queued-cell
reply defect fixed in `ipykernel 7.4
<https://github.com/ipython/ipykernel/pull/1529>`_.

Check the installed dependencies after installation or an environment update:

.. code-block:: bash

   python -m pip check

The version requirements are defined in ``pyproject.toml`` and also apply to
the test, development, documentation, and Binder installations. Automated
tests check synthetic scientific workflows, widgets, and queued kernel
requests; VS Code frontend rendering remains a manual check.

Troubleshooting
---------------

``ERROR: Cannot find command 'git'``
   Install git inside the active conda environment: ``conda install git``.

``import hypermri`` fails in a new session
   Ensure the correct environment is activated (``conda activate hypermri`` or
   ``source .venv/bin/activate``) before launching Python.

Installation reports an unsupported Python version
   Create a Python 3.12 environment. Python 3.10, 3.11, and 3.13 or newer are
   outside this release's supported runtime. Python 3.12 patch updates are
   accepted.

LaTeX
-----

Some report-generation workflows rely on a LaTeX distribution. Install TeX Live or MikTeX according to your platform if you plan to export PDFs from notebooks.
