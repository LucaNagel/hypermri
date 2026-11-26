MRI Utilities
=============

Acquisition-axis spectral transforms retain NumPy normalization and calculation
precision. They differ from spatial reconstruction transforms that also centre
their inputs. Phase rotation and temporal padding preserve the corresponding
signal and dtype policies documented below.

.. automodule:: hypermri.utils.mri.fourier
   :members:

.. autofunction:: hypermri.utils.mri.signal.rotate_phase

.. autofunction:: hypermri.utils.mri.signal.zero_fill_temporal

.. automodule:: hypermri.utils.mri.channels
   :members:
   :undoc-members:
   :show-inheritance:
