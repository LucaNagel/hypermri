Development setup
=================

HyperMRI is developed on Python 3.12 with ``ipykernel >=7.4,<8``; CI uses the
same versions.

Environment
-----------

::

   conda create -n hypermri-dev python=3.12
   conda activate hypermri-dev
   pip install -e ".[dev,docs]"

Git hooks
---------

::

   pre-commit install
   git config --local hypermri.testPython "$(python -c 'import sys; print(sys.executable)')"

``.pre-commit-config.yaml`` is the only hook configuration. Before each commit
the hooks

- run Ruff, the single linter and Black-compatible formatter (do not add
  Black, isort or flake8);
- check whitespace, YAML syntax, merge markers and file size;
- block the local ``internal/`` folder and vendor manuals such as ParaVision
  PDFs, which must never be committed;
- run the unit tests with the interpreter stored in ``hypermri.testPython``.

Git clients with a graphical interface often do not inherit the activated
environment; the stored interpreter path solves this. Repeat the second
command after recreating the environment, and run ``pre-commit install`` again
if the hooks still point to a deleted environment. ``pre-commit run
--all-files`` runs all hooks by hand.

Checks before a pull request
----------------------------

::

   ruff check src tests scripts
   ruff format --check src tests scripts
   python scripts/hooks/validate_dependencies.py
   python scripts/hooks/run_tests.py
   sphinx-build -E -a -b html docs/source docs/_build/html -W --keep-going

GitHub Actions runs the same checks: unit tests and the synthetic notebook
kernel test on Python 3.12, ``pip check``, the documentation build with
warnings as errors, and an installation test on Ubuntu (required), macOS and
Windows (best effort). The real-data integration job stays off unless the
repository variable ``ENABLE_APPROVED_REAL_DATA_TESTS`` is ``true``.

Cleaning build output
---------------------

``python scripts/clean_build_artifacts.py`` lists packaging output (``build``,
``dist``, ``*.egg-info``); ``--all`` adds test caches, coverage files and
``docs/_build``. Nothing is deleted until ``--apply`` is added. The script
removes only ignored paths without tracked files and refuses symbolic links,
mount points and nested Git checkouts.
