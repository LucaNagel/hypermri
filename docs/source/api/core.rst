Loading
========

Bruker slice geometry
---------------------

For one 2D slice package, ``scan.resolution_mm[2]`` is nominal thickness,
excluding gaps. Read and phase keep their FOV/matrix values; 3D uses
partition size. Multiple packages retain legacy resolution until per-package
geometry is supported. Use the property below for slice sampling distance.

.. autoproperty:: hypermri.vendors.bruker.BrukerHandler.slice_spacing_mm

Read it as ``scan.handler.slice_spacing_mm``; reading it through the scan
is deprecated (see ``MRIExperiment.__getattr__``).
Raw and scanner reconstructions copy the value to
``data.coords["slice_idx"].attrs["slice_spacing_mm"]`` when a slice coordinate
exists, and to ``data.attrs["slice_spacing_mm"]``. They also store physical
``slice_thickness_mm`` in both places: nominal thickness for 2D, partition
size for 3D. The handler's ``slice_thickness_mm`` still describes the
acquired slice or excitation slab. Array attributes retain physical sizes
when ``slice_idx`` is dropped. Fourier-axis preparation uses spacing before
the legacy resolution attribute; DICOM export uses physical thickness for
``SliceThickness`` and sampling distance for ``SpacingBetweenSlices``.
These attributes do not change image coordinates or sample values.

For a selected singleton ``slice_idx`` dimension, anatomical mask projection
uses half the thickness as its default plane tolerance, and the plot model
uses thickness for its fallback normal vector. Whole stacks retain XYZ
centre distances. Dropping the slice dimension still gives mask projection
its 0.001 mm fallback; preserving array thickness there is tracked separately.

Experiments
-----------

Scan properties
^^^^^^^^^^^^^^^

A scan reads its canonical acquisition parameters, such as ``nucleus``,
``bandwidth_hz``, ``dwell_time_s``, ``center_ppm``, ``n_acquisition``,
``reference_frequency_mhz`` (frequency of 0 ppm in MHz) and ``b0_tesla``,
from :attr:`~hypermri.loading.experiment.MRIExperiment.params` without a
warning. Every vendor handler sets ``b0_tesla`` with
:func:`hypermri.utils.physics.b0_tesla_from_reference_frequency`: the
reference frequency divided by the gyromagnetic ratio of the nucleus, for
example 75.4901665 MHz / 10.7084 MHz/T = 7.0496 T for 13C. The field is
None when the scan reports no reference frequency or no nucleus, and loading
fails with the scan path when the nucleus has no known ratio.

The frequency axes are properties of every scan and handler:

.. autoproperty:: hypermri.mixins.UniformDataPropertiesMixin.freq_axis_ppm

.. autoproperty:: hypermri.mixins.UniformDataPropertiesMixin.freq_axis_hz

.. autoproperty:: hypermri.mixins.UniformDataPropertiesMixin.ppm

.. autoproperty:: hypermri.mixins.UniformDataPropertiesMixin.hz

Spectroscopy sequences replace the short names with axes that match the
length of their loaded data:
:attr:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy.ppm` and
:attr:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy.hz`.

Studies
^^^^^^^

.. autoclass:: hypermri.study.Study
   :members: fit_maps

``fit_maps`` fits ADC and T2 maps of a whole study on request; its
docstring lists the settings, suitability checks and execution statuses.

.. automodule:: hypermri.loading.experiment
   :members:
   :show-inheritance:

Factories
---------

.. automodule:: hypermri.loading.scans
   :members:
   :show-inheritance:

Data types
----------

.. automodule:: hypermri.loading.contracts
   :members:
   :show-inheritance:

Exceptions
----------

.. automodule:: hypermri.exceptions
   :members:
   :show-inheritance:

Interfaces
----------

.. automodule:: hypermri.loading.protocols
   :members:
   :show-inheritance:

Mapping files
-------------

.. automodule:: hypermri.config.mappings
   :members:

.. automodule:: hypermri.sequence_mapping
   :members:

Precision
---------

.. automodule:: hypermri.reconstruction.precision
   :members:
