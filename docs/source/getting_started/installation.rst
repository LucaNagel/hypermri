Installation
============

HyperMRI is published on `PyPI <https://pypi.org/project/hypermri/>`_ as
``hypermri`` and needs Python 3.12. Other Python minor versions are excluded
by the package metadata.

Install from PyPI
-----------------

These steps start from a computer without Python, conda or Git. Conda
installs Python 3.12 into the environment, so Python does not need to be
installed separately.

1. **Install conda.** `Miniforge <https://conda-forge.org/download/>`_ is a
   free conda installer that uses the conda-forge channel. Download the
   installer for your computer:

   * Windows: `Miniforge3-Windows-x86_64.exe
     <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>`_
   * macOS with an Apple chip (M1 or newer): `Miniforge3-MacOSX-arm64.sh
     <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>`_
   * macOS with an Intel chip: `Miniforge3-MacOSX-x86_64.sh
     <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh>`_
   * Linux on an x86-64 PC: `Miniforge3-Linux-x86_64.sh
     <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh>`_
   * Linux on ARM: `Miniforge3-Linux-aarch64.sh
     <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh>`_

   On Windows, run the ``.exe`` file, choose *Just Me* and keep the default
   options. On macOS and Linux, run the downloaded file in a terminal, for
   example ``bash ~/Downloads/Miniforge3-MacOSX-arm64.sh``, accept the licence
   and answer ``yes`` when the installer asks whether to initialize conda.
   Then close the terminal and open a new one.

2. **Open a terminal.** On Windows, open *Miniforge Prompt* from the Start
   menu; on macOS, *Terminal*; on Linux, the terminal of your desktop. The
   commands below are the same on every system. The line starts with
   ``(base)`` when conda is ready.

3. **Create an environment and install HyperMRI:**

   .. code-block:: bash

      conda create -n hypermri python=3.12
      conda activate hypermri
      pip install hypermri

4. **Start Jupyter** with ``jupyter notebook``, or select the ``hypermri``
   environment as the notebook kernel in VS Code. Each new terminal starts in
   the ``(base)`` environment; run ``conda activate hypermri`` again first.

``pip install hypermri`` installs loading, reconstruction, numerical
analysis, the interactive Plotly viewer, Jupyter Notebook with the supported
kernel and the Siemens TWIX reader. A ``venv`` environment based on a
Python 3.12 interpreter works as well.

Install with Anaconda Navigator
-------------------------------

`Anaconda Navigator <https://www.anaconda.com/docs/tools/anaconda-navigator/main>`_
is a graphical alternative to the terminal and is part of the
`Anaconda Distribution <https://www.anaconda.com/download/success>`_. Check
the `Anaconda terms of service
<https://www.anaconda.com/legal/terms/terms-of-service>`_ first:
organizations with more than 200 employees may need a paid licence.

1. Open Anaconda Navigator and go to *Environments*.
2. Click *Create*, enter the name ``hypermri``, select Python 3.12 and click
   *Create*.
3. Click the ▶ button next to ``hypermri`` and choose *Open Terminal*.
4. In this terminal, run ``pip install hypermri`` and then
   ``jupyter notebook``.

Next time, open the terminal of the ``hypermri`` environment the same way and
run ``jupyter notebook``.

Optional features
-----------------

* ``pip install "hypermri[motion]"`` adds SimpleITK motion correction.

Loading and numerical analysis do not import widgets or plotting packages;
the viewer and notebook packages are imported when a plot or styled display
is requested. If one of them is missing from a damaged environment, the
error names the package and asks to reinstall HyperMRI.

Install from source
-------------------

Choose this path to edit the package or build the documentation. Install Git
into the active environment and clone the repository:

.. code-block:: bash

   conda activate hypermri
   conda install git
   git clone https://github.com/LucaNagel/hypermri.git
   cd hypermri
   pip install -e ".[dev,docs]"

To contribute changes, first `fork the repository
<https://github.com/LucaNagel/hypermri/fork>`_ on GitHub and clone your fork
instead.

Git is also available from
`git-scm.com <https://git-scm.com/downloads>`_, and GitHub Desktop can clone
the repository instead. The ``test`` extra adds the test tools, ``dev`` adds
linting and packaging tools, and ``docs`` adds the documentation build.
Quoting extras works reliably in shells such as ``zsh``. Build the HTML
documentation with:

.. code-block:: bash

   cd docs
   make html

and open ``docs/_build/html/index.html``.

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
the development, documentation, and Binder installations. Automated
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
