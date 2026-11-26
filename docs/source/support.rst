Support and maturity
====================

This page is the canonical support contract for HyperMRI. A handler, sequence,
or strategy being present in a registry only means that an
implementation can be selected; it does not by itself make that implementation
supported.

Maturity levels
---------------

``stable``
   The documented public entry point is covered by the compatibility policy and
   is expected to work for the scope stated below.

``beta``
   The workflow is usable and tested, but its API, metadata schema, or numerical
   behaviour may still change in a ``0.x`` minor release.

``experimental``
   The implementation is intended for evaluation and development. It may be
   incomplete, may lack reference-data validation, and is not covered by the
   public compatibility promise.

``deprecated``
   The entry point remains only to guide migration. Its documented replacement
   should be used for new code.

``unsupported``
   HyperMRI does not provide a handler or a public compatibility promise for the
   system.

Vendor and import support
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 13 67

   * - System or interface
     - Maturity
     - Supported scope
   * - Bruker ParaVision
     - Stable, scoped
     - Scan detection and parameter loading are stable. Loading
       scanner-reconstructed ``2dseq`` data into the standard ``xarray`` model
       is stable. Raw reconstruction is stable only for sequence identifiers in
       the explicit allowlist below.
   * - Siemens
     - Beta
     - Detection, metadata handling, and the currently documented loading path
       are Beta. The registry raw-reconstruction strategy is still experimental
       and must not be treated as a stable Siemens reconstruction workflow.
   * - Agilent / Varian
     - Experimental
     - Spectroscopy-oriented detection, metadata translation, and loading only.
   * - Magritek
     - Experimental
     - Spectroscopy-oriented detection, metadata translation, and loading only.
   * - Bruker TopSpin
     - Experimental
     - TopSpin-format spectroscopy loading only.
   * - External import adapter
     - Beta
     - This is an explicit, standardized import adapter for caller-provided
       arrays and metadata, not a scanner vendor. It is never auto-detected.
   * - Philips
     - Unsupported
     - No handler is provided.
   * - GE
     - Unsupported
     - No handler is provided.

Polaris is intentionally absent from the vendor table: it is not a scanner
vendor supported by HyperMRI and no Polaris vendor handler exists. Names such
as ``PolarizationMonitoring13C`` in sequence mappings describe acquisition
methods, not vendors.

Bruker raw-reconstruction allowlist
-----------------------------------

Only rows in this table make a stable raw-reconstruction promise. Entries in
``sequence_mapping.yaml`` or ``hypermri.reconstruction.selection`` are not an implicit
allowlist. Canonical IDs and aliases refer to Bruker ParaVision input only;
identically named sequences from another vendor do not inherit this status.

.. note::

   These rows describe the intended ``0.1.0`` support contract. The executable
   Zenodo reference tests are present, but final scientific validation and
   release approval remain tracked in ``.github/release-readiness.json``. Do
   not publish a release candidate while that gate is pending.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Canonical sequence ID
     - Family
     - Stable raw-reconstruction scope
   * - ``flash``
     - Cartesian imaging
     - Raw-data reshape, Fourier reconstruction, standardized dimensions, and
       coordinates for supported ParaVision FLASH method mappings.
   * - ``rare``
     - RARE
     - Standard RARE acquisition ordering handled by the dedicated RARE
       strategy.
   * - ``fair_rare``
     - RARE
     - FAIR-RARE acquisition ordering handled by the dedicated RARE strategy.
   * - ``rarevtr``
     - RARE
     - RARE-VTR acquisition ordering handled by the dedicated RARE strategy.
   * - ``csi``
     - Spectroscopy
     - CSI sorting and spatial reconstruction handled by the dedicated CSI
       strategy. For stock ``<Bruker:CSI>``, the first 70
       digital-filter/transient samples are removed exactly once per raw FID.
   * - ``nspect``
     - Spectroscopy
     - Bruker NSPECT FID reconstruction through the spectroscopy strategy.
   * - ``press``
     - Spectroscopy
     - Bruker PRESS FID reconstruction through the spectroscopy strategy.
   * - ``steam``
     - Spectroscopy
     - Bruker STEAM FID reconstruction through the spectroscopy strategy.
   * - ``singlepulse``
     - Spectroscopy
     - Bruker single-pulse FID reconstruction through the spectroscopy
       strategy.
   * - ``isis``
     - Spectroscopy
     - Bruker ISIS FID reconstruction through the spectroscopy strategy.

Every other Bruker raw-reconstruction sequence is Experimental, even when the
registry can select a strategy for it. Exact method aliases inherit stable
status only when they resolve to one of the canonical IDs above and satisfy the
same documented data contract.

The machine-readable allowlist is
``src/hypermri/config/support_policy.json``. The selected contract must be
backed by numerical end-to-end tests against the pinned public BrukerAPI
ParaVision 6.0.1 Lego-phantom corpus (DOI
``10.5281/zenodo.4048253``, CC BY 4.0) before ``0.1.0rc1`` is published;
registry-resolution tests alone do not constitute scientific validation. The
783.7-MB archive is an opt-in integration-test dependency and is not downloaded
by installation, documentation builds, or the normal unit suite.

The imaging contract compares every FLASH, RARE, FAIR_RARE, and RAREVTR frame
with the corresponding ParaVision ``2dseq`` magnitude after independent
frame-wise L2 normalisation (L2 distance below ``1e-3`` and Pearson correlation
above ``0.99999``). Stock CSI is compared exactly with the published FID
reference after removing its first 70 samples along the acquisition dimension,
that is, against ``fid.npz[70:]``. The five stable single-voxel spectroscopy
families are compared after the documented cutoff and FFT. FAIR_RARE must
expose three inversion times and two FAIR modes, while RAREVTR must expose all
three variable-TR frames. These are implementation-regression references, not
independent ground truth for every scientific or quantitative claim.

.. note:: Stock-CSI digital-filter contract

   The cutoff belongs to the raw-reconstruction result, not to the stored raw
   payload. ``handler.fid`` remains complete and repeated reconstruction must
   remove the 70 points only once. Scanner-reconstructed ``2dseq`` data is
   presumed to have passed through the scanner reconstruction and is not cut
   again. For the pinned 2048-sample reference FID, ``raw-reco`` and its time
   and frequency axes contain 1978 points. Those axes are rebuilt for the
   processed length, and the processed time axis starts at zero.

Stable and Beta workflows
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Workflow
     - Maturity
     - Boundary
   * - Detect and load a Bruker ParaVision scan
     - Stable
     - Direct single-scan loading through the public experiment factory.
   * - Convert Bruker ``2dseq`` data to the standardized ``xarray`` model
     - Stable
     - Coordinates and metadata documented by the public data model.
   * - Display supported standardized data in the browser viewer
     - Stable
     - ``plot`` and ``scan.plot`` with PlotOptions.
       Legacy factory/registry/adapter imports have been removed; see :doc:`api/public`.
   * - Discover and batch-load a directory through ``Study``
     - Beta
     - Discovery rules and smart sequence assignment may change.
   * - DCE fitting
     - Beta
     - Model dispatch, parameter schemas, and scientific reference values are
       still being stabilized.
   * - Interactive fit review and refitting
     - Beta
     - ReviewFit and direct ``refit_spectroscopy_voxel`` calls share the Lorentz
       operation with immutable ``SpectroscopyRefitEdit`` rows. Widget
       behaviour, review options and result schemas may change.
   * - CSI peak detection through ``detect_csi_peaks``
     - Beta
     - Shared analysis operation with immutable detection settings;
       options and returned diagnostics may change.
   * - Runtime vendor, sequence, reconstruction and viewer plugin extensions
     - Beta
     - Registry APIs and configuration schemas may change.
   * - New analysis and reconstruction pipelines
     - Beta
     - A pipeline is stable only when it is promoted on the
       :doc:`api/public` page.
   * - Placeholders, simulators, and research prototypes
     - Experimental
     - These require an explicit experimental label in their documentation.

Python and operating systems
----------------------------

Only Python 3.12.x is supported, with ``ipykernel >=7.4,<8``. Package metadata
rejects other Python minor versions. Linux, macOS and Windows are required
release targets. The release workflows test the same built wheel on all three
systems before publication. The first public release requires all three jobs
to pass.

Scientific scope
----------------

Maturity describes software support, not clinical suitability or independent
scientific validation. See :doc:`getting_started/research_use`.
