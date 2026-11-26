Coordinate systems
==================

HyperMRI stores spatial metadata on every ``xarray.DataArray`` so voxels, masks, and overlays stay aligned across anatomy and spectroscopy datasets. Coordinates are emitted in two complementary frames:

Scanner coordinates (RPS)
-------------------------

- **R** (read) follows the frequency-encoding gradient.
- **P** (phase) follows the phase-encoding gradient.
- **S** (slice) follows the slice-selection gradient.

These axes are expressed in millimetres and respect the per-scan offsets recorded by the vendor handler. Array indices ``read_idx``, ``phase_idx``, and ``slice_idx`` are linked to the same orientation so slicer widgets and interpolation routines can traverse the data unambiguously.

Scanner/DICOM coordinates (XYZ)
-------------------------------

Bruker reconstructions also expose auxiliary ``x``, ``y`` and ``z``
coordinates.  These coordinates are stored per voxel with dimensions
``(read_idx, phase_idx, slice_idx)`` and are derived from ParaVision
``VisuCorePosition``, ``VisuCoreOrientation``, ``VisuCoreSize`` and
``VisuCoreExtent`` in the ``visu_pars`` file of the scan.  ``RECO_transposition``
in the ``reco`` file states for each frame whether read runs along the DICOM
columns or rows, so the slice packages of a multi-package scan may differ.
This matches the geometry written to DICOM, so scans with different patient positions, read
orientations, slice orientations, or offsets can be compared in one
scanner-space frame.  A scan without this geometry, for example one whose
``visu_pars`` is missing, has no ``x``, ``y`` and ``z`` coordinates, and
:func:`~hypermri.utils.coordinates.grids.get_xyz_grid` raises ``ValueError``
for it.  :doc:`scanner_geometry` explains this calculation step by step,
with an interactive figure.

The original ``read_idx``, ``phase_idx`` and ``slice_idx`` dimensions remain
the indexing axes.  For example, ``data.isel(slice_idx=5)`` still selects the
fifth stored slice.  The ``x/y/z`` coordinates describe where each selected
voxel lies in scanner/DICOM space; they are not independent 1D dimensions.

Transformations between scanner and acquisition frames live in
:mod:`hypermri.utils.coordinates.transformations`. Anatomical masks
automatically adopt the XYZ grid so a mask drawn on one acquisition can be
propagated to another using
:meth:`~hypermri.loading.experiment.MRIExperiment.apply_mask_to`.

Example
-------

.. code-block:: python

   from hypermri import Study

   study = Study("path/to/study", verbose=False)
   csi, anatomy = study[30], study[8]

   csi.primary_data.coords["x"]          # scanner x of each CSI voxel in mm
   anatomy.xyz_extent_mm                 # field of view of the anatomy

   # A mask drawn on the anatomy, transferred to the CSI grid.
   masked = anatomy.apply_mask_to(csi, mask_name="brick")

The viewer uses the same coordinates to resample an anatomy under a CSI map
(:doc:`viewer`).
