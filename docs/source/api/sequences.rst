Sequences
=========

The sequence layer extends :class:`hypermri.loading.experiment.MRIExperiment` with anatomy- and spectroscopy-specific behaviour. ``BaseSequence`` owns the construction of a sequence object from a vendor handler or an existing experiment and the central automatic reconstruction policy; the anatomy and spectroscopy base classes add domain-specific helpers. Individual sequence modules reuse these patterns and are being refactored for Sphinx-friendly docstrings.

Base sequence interface
-----------------------

.. automodule:: hypermri.sequences.base_sequence
   :members:
   :undoc-members:
   :show-inheritance:

Base anatomy interface
----------------------

.. automodule:: hypermri.sequences.base_anatomy
   :members:
   :undoc-members:
   :show-inheritance:

Base spectroscopy interface
---------------------------

.. automodule:: hypermri.sequences.base_spectroscopy
   :members:
   :undoc-members:
   :show-inheritance:

CSI spectroscopy
----------------

Signal-decay review uses the shared Plotly viewer and returns its widget or
controller. Filters remain owned by ``BaseSpectroscopy.filter_signal`` and
reconstruction by the CSI strategy. The documented sequence methods describe
confirmation, explicit-source application and physical Read/Phase filtering.

.. autoclass:: hypermri.sequences.CSI.CSI
   :members: plot_signal_decay, create_filtered_data, find_peaks, get_plotting_grid
   :member-order: bysource
   :show-inheritance:

Additional sequences
--------------------

Implementation details for FLASH, RARE, PRESS, and other classes will be surfaced once their docstrings are cleaned up. For now, refer to :mod:`hypermri.sequences` and the architecture notes in :doc:`../concepts/registry_overview` for registry usage.

Legacy analysis prerequisites
-----------------------------

``CSI.find_peaks`` requires the legacy spectral iterator and peak/range
dictionaries. ``BSSFP.reco_scrambled_phase`` requires the legacy ``bssfp_fft``
implementation. The ``IVRecovery`` fitting and interactive methods require
``get_spectra`` or ``get_ppm`` preparation. These inputs are not provided by
current sequence construction. Calling the methods now raises an explanatory
``NotImplementedError`` before data mutation, plotting or widget construction.
Extensions that explicitly provide all prerequisites retain their existing
behavior. No current analysis function implements these complete legacy
workflows; ``BaseSpectroscopy.single_spec_find_peaks`` remains available for an
explicitly selected spectrum. ``CSImulator.simulate`` likewise explains a
missing legacy CSI iterator before allocating output arrays.

Vendor dictionaries and acquired FIDs belong to ``scan.handler``. Legacy
analysis adapters accepting a scan also accept its handler and now unwrap it
explicitly. Reading arbitrary handler attributes through the scan remains
deprecated until HyperMRI 0.2.0; the compatibility route is retained for that
window.
