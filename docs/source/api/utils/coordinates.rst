Coordinates
===========

Coordinate transformations and grid generation.

Required geometry and multiple slice packages
---------------------------------------------

Grid operations require declared, finite, positive geometry. They report missing
or incomplete ``fov_mm``, ``matrix``, slice count or slice thickness with an
actionable ``ValueError``; the historical 100 mm and 64-point defaults have
been removed. Loading metadata does not require constructing a grid.
``define_grid_lines`` also accepts any explicit valid pair of field of view,
matrix size and resolution. An explicit ``rps_grid_mm`` supplies the axes directly.
``select_rps_extent_by_axis`` returns four flat limits for each valid spatial
axis and rejects unknown axis names.

Absent geometry is distinguished from invalid geometry. When a required value
is None, when a scan has several slice packages (RPS functions), or when all
four ``VisuCore`` parameters are missing (XYZ functions), the grid functions
raise ``GeometryUnavailableError``, a subclass of ``ValueError``. The
``rps_*_mm`` and ``xyz_*_mm`` properties of vendor handlers and scans return
None for this error only; incomplete or invalid declared geometry propagates
as ``ValueError``.

For scans with several slice packages, ``BrukerHandler.fov_mm[2]`` and
``resolution_mm[2]`` are NaN, meaning that a global slice value is unavailable.
Read and phase values and supported nominal slice thickness remain available.
Reconstructed arrays retain index coordinates for these scans and preserve
their physical ``x``, ``y`` and ``z`` coordinates from VisuCore metadata.
Select an individual package before requesting uniform RPS geometry; the global
slice count must not be treated as one regularly spaced stack. This policy does
not provide a new package-selection or DICOM-export interface.

The legacy ``define_xyz_grid`` and ``CSI.get_plotting_grid`` remain available
with their existing orientation limitations. Their numerical geometry has not
been corrected by removing the unused ``interpolate_array`` workflow.

``define_xyz_grid(scan)`` pairs the scan-derived field of view and matrix in
z/x/y order, using ``define_imageFOV_parameters`` and
``define_imagematrix_parameters`` respectively. If either vector is supplied
explicitly, it is paired element by element with the other vector; a supplied
vector must therefore also use z/x/y order. Supplying both vectors explicitly
retains their existing convention. For Head_Prone and Foot_Prone coronal L_R
scans, the shared FOV helper now assigns the phase length to z and the read
length to x, matching their voxel counts. This corrects the previous read/phase
length swap in derived grids, ``CSI.get_plotting_grid`` and legacy
``get_extent`` output. Caller-supplied FOV vectors retain their values.
Head_Prone and Foot_Prone support six orientation
pairs; Head_Supine only axial L_R works through both helpers and the plotting
table, and it ignores offsets. Other supine pairs warn and fail: axial A_P
reaches an unsupported CSI table entry, while sagittal/coronal pairs lack
matrix counts before that table is reached. The owner accepted these retained
limits on 11 October 2026.

These legacy lines start at minus half the assigned field of view and have
their midpoint half a voxel below the supported offset. Interpolation factors
only increase grid counts; they do not reconstruct or interpolate image data.
They are not scanner voxel positions. Use ``get_xyz_grid`` for positions from
the selected VisuCore geometry. Scan-default FOV and
matrix pairing now follows z/x/y order. The CSI plotting grid still retains its
documented supine and offset limitations.

Grids
-----
.. automodule:: hypermri.utils.coordinates.grids
   :members:
   :undoc-members:
   :show-inheritance:

Transformations
---------------
.. automodule:: hypermri.utils.coordinates.transformations
   :members:
   :undoc-members:
   :show-inheritance:

Image Shifts
------------
.. automodule:: hypermri.utils.coordinates.image_shifts
   :members:
   :undoc-members:
   :show-inheritance:

Excited Volume
--------------
.. automodule:: hypermri.utils.coordinates.excited_volume
   :members:
   :show-inheritance:

Scan Alignment
--------------
.. automodule:: hypermri.utils.coordinates.alignment
   :members:
   :undoc-members:
   :show-inheritance:
