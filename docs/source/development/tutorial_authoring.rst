Writing tutorials
=================

Tutorials live in ``docs/source/tutorials`` as MyST text notebooks
(``.md`` files with ``file_format: mystnb``). The documentation build runs
their code cells with myst-nb and shows the real output, so a tutorial that
no longer matches the code fails the build. Notebook files (``.ipynb``) are
not committed.

Use the public loading, analysis and viewer APIs. Keep scientific defaults in
the modules that own them instead of copying an algorithm into a page.

Start from synthetic data
-------------------------

A synthetic example runs everywhere and can be checked against known
parameters. State units, dimensions and physical assumptions. For example,
this snippet generates a noiseless 32-sample complex FID, with times in
seconds and frequency in Hz:

.. code-block:: python

   import numpy as np

   from hypermri.analysis.spectroscopy import generate_fid
   from hypermri.utils.time.temporal import calc_sampling_time_axis

   time_s = calc_sampling_time_axis(npoints=32, dwell_time_s=0.0005)
   fid = generate_fid(time_axis_s=time_s, t2_s=0.025, freq_hz=80.0, noise_std=0.0)
   assert fid.shape == time_s.shape

This exact snippet is executed by ``tests/unit/test_tutorial_authoring.py``.
The fitting tutorial follows the same rule: its signal is the fixture of
``tests/fixtures/synthetic_spectroscopy_uncertainty.py``, whose numbers are
checked by unit tests.

Page structure
--------------

A tutorial page starts with this front matter:

.. code-block:: yaml

   ---
   file_format: mystnb
   kernelspec:
     name: python3
     display_name: Python 3
   ---

Executed code goes into ``{code-cell} ipython3`` blocks. Interactive steps
that cannot run during the build (drawing a mask, clicking a button) go into
ordinary ``python`` code blocks. Show the viewer with a saved snapshot instead
of executing it in the build:

.. code-block:: html

   <iframe class="hm-snapshot" src="../_static/snapshots/rare.html"
           loading="lazy" title="RARE scan in the HyperMRI viewer"></iframe>

inside a ``{raw} html`` block. ``scripts/build_docs_snapshots.py`` writes
these files; add a new snapshot there and run it with the phantom folder.

Measurement data
----------------

Pages with measured data use the public Lego phantom (BrukerAPI test data,
Zenodo DOI 10.5281/zenodo.4048253). Each page shows the user's path in a
visible cell and reads the build's copy in a hidden one:

.. code-block:: python

   # cell tagged remove-cell
   import os

   study_path = Path(os.environ["HYPERMRI_DOCS_PHANTOM"])

``docs/source/conf.py`` looks for the phantom in ``HYPERMRI_DOCS_PHANTOM``,
``HYPERMRI_TEST_DATA``, ``tests/data`` and ``examples``. When it is missing,
the phantom pages listed in ``PHANTOM_PAGES`` are shown without running them;
add a new phantom page to that list.

Do not add private scanner data or identifying metadata to make a page run.
``src/hypermri/datasets/registry.json`` owns the metadata of every public
dataset.

Building
--------

::

   sphinx-build -E -a -b html docs/source docs/_build/html -W --keep-going

Executed outputs are cached in ``docs/_build/.jupyter_cache``; a page runs
again when its code changes. ``HYPERMRI_DOCS_EXECUTE=force`` runs every page,
``off`` none. The build must finish without warnings.
