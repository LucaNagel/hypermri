Stable public API
=================

This page is the allowlist for HyperMRI's stable Python API. Reference pages may
also describe Beta, Experimental, deprecated, or internal implementation
details; appearing elsewhere in the generated reference is not a stability
promise. Explicit maturity labels override the default.

The support boundary for vendors, sequences, workflows, Python versions, and
operating systems is defined in :doc:`../support`.

Package imports
---------------

The root ``__all__`` contains ``__version__``, ``MRIExperiment``, ``load_scan``,
``create_experiment``, ``load_external_experiment`` and ``Study``. These names
resolve lazily to their existing implementations. Importing ``hypermri`` reads
no scans, imports no vendor or notebook runtime, and changes no plotting hooks.
The maturity labels below remain unchanged.

The former root aliases were removed on 15 November 2026 and now raise
``AttributeError``. Import the handlers ``BrukerHandler``, ``AgilentHandler``,
``MagritekHandler`` and ``BrukerTopSpinHandler`` from ``hypermri.vendors`` and
the three ``Standardized*Mixin`` classes from ``hypermri.mixins``; ``version``
and ``PackageNotFoundError`` come from ``importlib.metadata`` and
``install_matplotlib_usetex_fallback`` from ``hypermri.utils.plot``.

``PlotOptions`` and numerical spectroscopy operations can be imported without
notebook packages. Calling ``plot`` requires ``hypermri[visualization]``; styled
``Study.display`` and legacy notebook controls require ``hypermri[notebook]``.
Core-only study loading uses ``Study(path, verbose=False)``; the existing
``verbose=True`` default requests notebook styling. See
:doc:`../getting_started/installation` for all extras. The previous automatic
global Matplotlib TeX retry hook is now opt-in::

    from hypermri.utils.plot import install_matplotlib_usetex_fallback

    install_matplotlib_usetex_fallback()

Stable entry points
-------------------

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Entry point
     - Stable scope
   * - :func:`hypermri.load_scan`
     - Detecting and loading a single Bruker ParaVision scan as documented in
       the support matrix.
   * - :func:`hypermri.create_experiment`
     - Compatibility wrapper for existing single-scan workflows; new code
       should use :func:`hypermri.load_scan`.
   * - :class:`hypermri.MRIExperiment`
     - The ``id``, ``type``, and ``metadata`` properties and
       ``reconstruct("seq2d")`` for supported Bruker ParaVision data.
   * - :func:`hypermri.visualization.plot` and
       :class:`hypermri.visualization.PlotOptions`
     - Displaying supported standardized ``xarray`` data in the interactive
       browser viewer; unknown display options fail before a widget is built.

Raw reconstruction becomes part of the stable API only for sequence
identifiers added to the explicit allowlist in :doc:`../support`. The registry
configuration alone does not grant stable status.

Beta entry points
-----------------

The following surfaces are public for evaluation, but are not covered by the
stable compatibility promise:

* :class:`hypermri.Study` and its directory-discovery/batch-loading workflow;
* :func:`hypermri.load_external_experiment`;
* vendor, sequence, reconstruction and fitting registries;
* :mod:`hypermri.analysis.fitting.api` and new fitting pipelines;
* :mod:`hypermri.analysis.fitting.spectroscopy` for direct spectroscopy fitting,
  iterative parameter updates and fit reconstruction;
* ``hypermri.analysis.fitting.reconstruct_fitted_signal`` for total spectroscopy
  signals on the input data's coordinates;
* DCE pipelines and interactive fit review.

Removed entry points
--------------------

The compatibility modules that were planned to stay until HyperMRI 0.2.0 were
removed before the first public release. Importing a former path, or
unpickling an object stored under it, raises ``ModuleNotFoundError``; the
new modules define the same objects, formulas, defaults and readers.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Former location
     - Import from
   * - ``hypermri.core`` and its submodules
     - ``hypermri.loading`` (experiment, scans, contracts, protocols),
       ``hypermri.exceptions``, ``hypermri.config.mappings`` and
       ``hypermri.reconstruction.precision``
   * - ``hypermri.vendorhandler``, ``hypermri.topspin_exp``
     - ``hypermri.vendors``
   * - ``hypermri.functions.*``
     - fitting modules (``ideal``, ``phase_ideal``) in
       ``hypermri.analysis.fitting``, ``hypermri.analysis.spectroscopy.ph``
       and the simulators and dashboards in ``hypermri.visualization``
   * - ``hypermri.analysis.*.core``
     - the domain modules of the same package, for example
       ``hypermri.analysis.spectroscopy.signals`` or ``.peaks``
   * - ``hypermri.utils.utils_*`` and other flat ``utils`` modules
     - the topic subpackages of ``hypermri.utils`` (``mathematics``,
       ``physics``, ``mri``, ``io``, ``image``, ``sequence``, ``study``) or
       ``hypermri.visualization``
   * - ``hypermri.visualization.core.*``
     - the module of the same name directly in ``hypermri.visualization``

:mod:`hypermri.analysis.fitting` and :mod:`hypermri.analysis.fitting.fit_maps`
no longer re-export model functions. Import them from their owners:
quantitative models from ``hypermri.analysis.fitting.models.quantitative``,
DCE models from ``hypermri.analysis.dce.models`` and its submodules,
``concentration_to_signal`` and ``signal_to_concentration`` from
``hypermri.analysis.dce.conversions``, and
``get_registry`` from :mod:`hypermri.analysis.fitting.registry`.
``fit_quantitative_maps`` and ``fit_maps.def_fit_params`` remain supported.

Removed plotting API
--------------------

The plotter factory, registry, adapters, ``sequence.plotter``,
``hypermri.core.BasePlotter`` and the ``hypermri.utils.plot`` modules are
removed, and so is the earlier Python-side viewer with its controller,
plugins and profiling log. :func:`hypermri.visualization.plot` and
``scan.plot`` show the browser viewer instead. For example, replace
``create_anatomy_plotter(scan).plot(data, height=480)`` with::

    from hypermri.visualization import PlotOptions, plot

    view = plot(data, PlotOptions(plot_height=480), scan=scan)

Options of the earlier viewer raise a ValueError that names their
replacement: masks and fits are tools of the returned view
(``view.tool = "masks"`` or ``"fit"``), and display settings are passed as
a ``PlotState`` in ``PlotOptions.state``. ``scan.create_mask()`` and
``scan.interactive_fit()`` open the view with that tool selected. Saved
view settings remain readable.

Versioning policy
-----------------

HyperMRI uses Semantic Versioning and tags releases as
``vMAJOR.MINOR.PATCH``. During ``0.x``, incompatible changes are restricted to
minor releases and must be announced. From ``1.0.0`` onward, an incompatible
change to a stable entry point requires a major release. Patch releases must
not remove or incompatibly change a stable entry point.
