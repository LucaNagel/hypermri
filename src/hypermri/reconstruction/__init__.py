"""Reconstruct labelled data from vendor raw data and scanner images.

:mod:`hypermri.reconstruction.selection` chooses the strategy class and its
default options for a vendor, sequence and reconstruction kind; the strategies
in :mod:`hypermri.reconstruction.strategies` perform the reconstruction.
Use :meth:`hypermri.loading.experiment.MRIExperiment.reconstruct` to run one.

Notes
-----
Importing this package imports no strategy, so loading a scan without
reconstruction stays lazy.
"""
