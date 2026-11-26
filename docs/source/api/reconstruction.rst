Reconstruction
==============

.. note::

   The reconstruction table and new strategy pipelines are **Beta**. A configured strategy is not
   automatically a supported workflow. Stable vendor/sequence combinations are
   listed in :doc:`../support`.

.. automodule:: hypermri.reconstruction.selection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hypermri.reconstruction.strategies
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hypermri.reconstruction.strategies.bruker.epsi
   :members: epsi_dwell_time_ms, epsi_segment_count, correct_odd_even_echo_phase

.. automodule:: hypermri.reconstruction.consistency
   :members:

Spatial Fourier transforms
--------------------------

These four functions own every transform between spatial k-space and the
image grid of ``2dseq``. Zero frequency lies at index ``n // 2`` of every
spatial k-space axis; on the read and phase axes the FOV centre lies at index
``n / 2``, where ``VisuCorePosition`` puts it, and the 3D slice-encoding axis
takes the forward transform with its centre at ``(n - 1) / 2``. New code
calls these functions instead of writing its own ``fftshift``/``fft``
sequence for a spatial axis.

.. automodule:: hypermri.reconstruction.centred_dft
   :members:

.. _api-spectral-axes:

Spectral time and frequency axes
--------------------------------

The frequency axis of a spectrum is the FFT-shifted set of bins in Hz, or in
ppm after dividing by the scanner's Hz-per-ppm ratio and adding the centre
chemical shift. Its length is the number of FID samples left after the
leading ``cut_off`` samples were removed, and the time axis of the processed
FID starts at 0 s. Vendor handlers, sequences and analysis all read the axis
from this module.

.. automodule:: hypermri.reconstruction.axes
   :members: get_freq_axis, center_freq_axis, frequency_axes, processed_sampling_axis, resolve_cut_off, apply_processed_axis_metadata

CSI acquisition description
---------------------------

:class:`~hypermri.reconstruction.csi_acquisition.CSIAcquisition` describes one
2D CSI acquisition with named physical values: encoding order, one read/phase
grid position per acquired FID, matrix, field of view and its offset in mm,
samples per FID, dwell time in s, reference frequency in MHz and centre
chemical shift in ppm. It is immutable and holds no signal samples.
:func:`~hypermri.vendors.bruker_acquisition.bruker_to_csi_acquisition` fills it
from a parsed Bruker scan, taking the positions from the existing CSI
encoding-order parser; ``scan.handler.to_csi_acquisition()`` calls it, so
sequence and visualization code reach the description without importing
``hypermri.vendors``. The CSI reconstruction
(``BrukerCsiRecoStrategy.sort_kspace``), the signal-decay viewer, the k-space
ordering of ``filter_signal`` and the raw CSI spatial filter read grid,
positions, samples per FID and field of view from the description, so a scan
whose description fails validation is not reconstructed. ``fov_mm`` is
``None`` when the scan records no field of view; the spatial filter then needs
an explicit ``spatial_spacing_mm``. Storage fields of the raw ``fid`` (``ACQ_size``, ``GO_block_size``,
``GO_raw_data_format``, ``BYTORDA``) are not part of it: the Bruker CSI
reconstruction reads them and passes the padded record size, sample codec and
byte order to the vendor-neutral helpers in ``reconstruction.csi_raw``::

   acquisition = scan.handler.to_csi_acquisition()
   acquisition.matrix, acquisition.n_fids

.. automodule:: hypermri.reconstruction.csi_acquisition
   :members:

.. autofunction:: hypermri.vendors.bruker_acquisition.bruker_to_csi_acquisition

.. automethod:: hypermri.vendors.bruker.BrukerHandler.to_csi_acquisition

Bruker reconstruction stages and precision
------------------------------------------

Raw and MGE reconstruction share input selection, padding removal, FOV phase
shift, spatial Fourier transforms and storage conversion. Separate steps build
raw and MGE echo labels. RARE explicitly sorts 2D slice/echo-train or 3D phase
loops, separates FAIR and fat/water axes, and places measured partial-Fourier
lines in the final zero-filled grid. ``2dseq`` separates file loading,
dimension planning, singleton expansion, coordinate construction and metadata.
CSI separates the acquisition plan, payload selection, grid sorting, FOV phase
operator, spatial transform/noise propagation and provenance. Single-voxel
spectroscopy separates FID preparation, temporal labels and the spectral FFT.

Generic/FLASH and MGE place phase lines and encoded 3D partitions at their signed
scanner encoding step plus half the full matrix length. RARE, Cartesian and CSI
share this signed-step conversion. Missing phase positions are zero; sorting
and centred block filling must not substitute for declared asymmetric positions.
The RARE fallback without encoding steps aligns the acquired and target centres
as ``n // 2 - n_acq // 2``. Readout filling remains symmetric: partial-echo
placement and scanner-level partial-Fourier validation are not yet
implemented.

The stages pass fresh arrays directly and preserve calculation order,
normalization and spatial coordinate conventions. Reconstructions keep acquired
samples and timing/receiver metadata on the handler unchanged. The scan owns
stored reconstructions. Explicit ``set_kspace_xarray(write_to_object=True)`` and
``set_data(write_to_object=True)`` calls retain their documented side effects.
CSI's ``csi_sort_source_hash`` follows the current source files, so it
changes with source refactoring even when reconstructed values are identical.

The fourteen explicit ``float32`` imaging entries in
:data:`hypermri.reconstruction.selection.RECONSTRUCTIONS` are intentional:
imaging results are stored in single precision. CSI and single-voxel spectroscopy
retain their Python calculation defaults. Single-voxel spectroscopy stores
complex64 for ``float16``/``float32`` and complex128 for ``float64``. Shared
conversions live in :mod:`hypermri.reconstruction.precision`; vendor FID readers keep their
documented precision policies. Bruker scalar byte decoding returns complex128
quadrature samples (or float64 real-only FID samples), independently of requested
reconstruction storage precision. Magritek and Agilent retain complex64 inputs,
including signals with a zero imaginary component.
Siemens ``set_seq2d_rps_xarray`` accepts keyword-only ``data_precision`` and
defaults to float32 storage through the same conversion policy.

Frequency axes describe spectroscopy samples and MGE echoes; other imaging
arrays carry no spectral axes. MGE echo timing and frequency metadata are built
from the echo times without overwriting readout acquisition metadata. Handler
frequency accessors calculate their axes from current acquired parameters.

Failed preprocessing, FOV phase shifts, transforms and scanner reconstruction
reach the caller, preserving their original cause. Missing scanner payloads and
explicitly unsupported raw strategies may still return None. Verified scanner
layout fallbacks and incomplete optional geometry use narrow exception handling
and report their cause at warning level.

Optional Bruker scanner FOV phase ramp
--------------------------------------

Bruker raw reconstruction defaults to ``fov_shift_source="acquisition"``. This
keeps the independent acquisition-derived phase ramp and is unchanged by the
optional scanner mode. Select scanner mode explicitly only for the validated
stock ``<Bruker:FLASH>`` geometry using the generic 3D strategy: Head_Supine,
coronal, H_F, one slab/package, a full
matrix equal to ``PVM_EncMatrix``, and an even number of slice partitions. The
read and phase offsets must both be zero. ``RECO_rotate`` must have shape
``(3, 1)`` and come from ``pdata/1/reco``; its in-plane values must be 0.5.

Let :math:`d = (r_3 - 0.5)N_3` be the declared slice shift in voxels, where :math:`r_3` is
the third entry of ``RECO_rotate`` and :math:`N_3` is the slice-partition count. The
acquisition-derived slice shift :math:`a_3` must equal :math:`-(d + 0.5)` within
:math:`10^{-9}` voxel. Scanner mode applies the read/phase/slice shift
:math:`(0, 0, -(d + 0.5))`, the acquisition shift: the slice-encoding transform
already centres the slices on the ``2dseq`` slice grid, so scanner mode gives
the default image and adds only the metadata checks and provenance. It requires ``shift_xarray=True`` and
``fov_shift_sign=1``. Missing or inconsistent metadata, or another geometry,
fails before the FID is read. This option applies only the validated FOV phase
ramp; it does not reproduce other ParaVision scanner processing. For example::

   scanner_images = scan.reconstruct('raw', fov_shift_source='scanner')

The raw default remains the independent acquired-data reconstruction; scanner
mode is an explicit comparison option and does not change that default. Results
from scanner mode record ``fov_shift_source``, ``fov_shift_vox`` (the
read/phase/slice shifts in voxels), ``bruker_reco_rotate`` and
``bruker_reco_parameter_source``. The latter identifies the ``pdata/1/reco``
file used for the rotation values.

For callers using the lower-level ``reco_rawdata`` method directly,
``fov_shift_vox`` accepts a manual finite read/phase/slice voxel vector for 3D
k-space only. A positive entry moves the reconstructed image content to higher
indices along that axis. It requires ``shift_xarray=True`` and
``fov_shift_sign=1``; it does not select or validate scanner metadata.

For direct processed input, pass ``values`` and, when labelling is disabled,
``kspace_xarray`` to ``reco_rawdata``. Implicit processed handler inputs warn until
their removal in HyperMRI 0.2.0. For covariance-aware CSI use
``BrukerCsiRecoStrategy.reconstruct_with_noise`` to receive both signal and noise
results directly. The old noise option of ``reconstruct`` still stores the final
noise bundle on the handler with a deprecation warning until 0.2.0::

   signal, noise = strategy.reconstruct_with_noise(
       scan.handler, estimate, data_type="fid", cut_off=0
   )

Reader boundaries and supported layouts
-----------------------------------------

Bruker scanner reconstruction reads ``pdata/<n>/2dseq`` and its matching
``visu_pars``. The Bruker reader applies frame scaling and combines the actual
``FG_COMPLEX`` group exactly once; HyperMRI retains the remaining echo, cycle
and other frame groups. ``VisuFGOrderDesc`` describes these reconstructed image
frames. It does not determine the acquired loop order of ``fid``.

Missing or empty ``2dseq`` files produce the warning
``No seq2d data available for scan ...``, including the requested file path.
The reader returns an empty array without validating image-axis metadata.
Full loading keeps the scan available and can continue with its independent
raw reconstruction. Present image data still require valid matching
``visu_pars`` and ``RECO_transposition`` metadata. As with other log messages,
the configured ``log_mode`` determines whether the warning is displayed.

The strict scanner path checks every non-spatial axis against the method's
size before assigning labels; mismatches use the frame-group fallback and
record ``seq2d_fallback_reason``. Scalar ``VisuGroupDepVals`` dependencies
use the descriptor's dependency window and each parameter's own offset.
They supply echo/TR coordinates in seconds, inversion times in ms and
``VisuFGElemComment`` labels for parameter maps. Declared ``VisuFGElemId``
dependencies additionally supply the optional machine identifiers in
``frame_id``. Existing ``map_idx`` display labels remain unchanged; comments
are never interpreted as quantitative values or stable machine identifiers.
For example, a DTI map with the ID ``DTI_FA`` can be selected with
``maps.isel(map_idx=maps.frame_id == "DTI_FA")``.
Maps without element IDs retain their readable comments.
Auxiliary coordinates ``echo_time_s``, ``repetition_time_s``,
``inversion_time_ms``, ``frame_label`` and ``frame_id``
retain their source parameter; dependencies shared between groups append
the dimension name to the auxiliary coordinate name. Loop axes without
declared times keep index units. Readout sample times never supply loop times.
Compound cardiac timing records remain uninterpreted. MGE keeps its
``EffectiveTE`` echo times in seconds.

JCAMP parameter arrays are read through the next record boundary, preserving
complete wrapped strings and dependency structs. Physical line length and
string character capacity do not determine the number of frame labels.

T1 EPI reads method ``IRTime`` when FAIR inversion times are absent. Scalars,
nonempty vectors and single-row tables of shape ``(1, N)`` are accepted and
normalized to a one-dimensional timing vector. Empty, multi-row or
higher-dimensional tables and nonfinite or negative values raise an error.
Both raw and scanner reconstructions label multiple inversion times ``inversion_idx``
in ms, without applying FAIR slice offsets. FAIR retains its slice timing
calculation. Scanner scalar inversion dependencies take precedence over
method base times when declared.

For a single-slice, single-echo T1 EPI scan whose decoded object count equals
the declared inversion count, objects carry ``inversion_idx`` and repetitions
remain separate acquisitions. This also retains inversion labels when the raw
file has one repetition. The existing repetition-axis inversion layout remains
supported.

The 2D DTI EPI raw path also accepts the ``brukerapi`` ``dEPI`` layout with
its explicitly labelled trailing singleton ``k_space_encode_step_2`` axis.
Only that verified unused encoding axis is removed; object, repetition and
receiver axes retain their positions. Unknown extra axes and a non-singleton
third encoding axis remain unsupported on this 2D path.

``load_2dseq_file(set_reco_file=True)`` and ``load_reconstruction`` retain their
explicit selection of handler ``reco`` parameters. ``reco_parameter_source``
names the active scan-relative file. Scanner arrays record
``bruker_reconstruction_number`` for the data folder; scanner and spectroscopy
arrays snapshot ``bruker_reco_parameter_source``, ``bruker_reco_time`` and
``bruker_reco_objects_per_repetition`` when available, even with
``append_parameters=False``. This makes the parameters used by subsequent
spectroscopy and timestamp readers traceable. Later selections do not change
existing result snapshots. Spectroscopy also retains available ``patient_pos``
independently of full parameter attachment.

Raw ``fid``/``ser`` decoding uses ``GO_raw_data_format`` (signed int16/int32 or
float32) and ``BYTORDA``. ParaVision 360 may declare ``ACQ_word_size`` instead.
``AQ_mod=qf`` retains real samples; supported quadrature modes use interleaved
real/imaginary pairs. ``rawdata.job0`` additionally honors its matching
``ACQ_ScanPipeJobSettings`` storage declaration, including float64 jobs.
Nine-field ``ACQ_jobs`` records identify the settings by job name; older
eight-field records and absent job lists use position zero. Missing or ambiguous
job mappings raise an error.
Real-only jobs, sequential quadrature, missing/unknown declarations and
incomplete samples raise explanatory errors. Binary decoding retains padding;
reconstruction owns its removal and logical shape. Decoding support alone does
not establish support for every real-only reconstruction layout.

The actual Bruker reader owns ``ACQ_obj_order`` sorting for 2D BSSFP, DTI and
EPI. These strategies split the decoded object axis without applying its
permutation again. EPI consults the reader's mirroring state before undoing
sorted-row reversal and applying polarity within each shot. The 3D EPI stage
uses declared readout-train boundaries to remove 1024-byte padding, retaining
receivers, repetitions and shot-local polarity. Its numerical scanner
validation remains open; extra object loops and inconsistent declarations are
rejected.

Custom 3D BSSFP raw encoding and simultaneous 2D EPI slice/echo loops remain
unsupported until their acquired order and numerical references are validated.
Raw errors propagate through ``scan.data`` and the default plot path. Select
scanner images explicitly when they are the intended source::

   scanner_images = scan.reconstruct("seq2d")
   viewer = scan.plot(scanner_images)
   # Equivalently: scan.plot(scan.scanner_data)

Magritek spectral reconstruction retains the loaded export's repetition and
extra axes. FFT, digital-filter removal, windows and zero filling operate only
along acquisition. Unlabelled extra axes are rejected instead of flattened.
The physical interpretation of optional export counters still requires real
export validation; ``combine_channels=False`` preserves their values without
receiver reduction.

Siemens imaging k-space records ``fourier_encoded_dims`` from TWIX ``Col``,
``Lin`` and ``Par``. Independent ``Sli`` counters are not Fourier transformed.
The supported 2D preview retains individual receivers. A non-singleton encoded
``Par`` currently raises ``NotImplementedError`` and retains labelled k-space,
pending a 3D TWIX/scanner reference. The registered Siemens raw strategy remains
unsupported separately from this handler preview.

Selected scanner geometry and shared labels
-------------------------------------------

Scanner XYZ coordinates and spatial axis geometry use the selected
``pdata/<n>/visu_pars`` and ``reco`` without changing the handler's acquisition
geometry or its ``pdata/1`` transposition. ``bruker_visu_parameter_source``
records the selected geometry source. Missing selected physical positions
remain indexed rather than borrowing another reconstruction's positions.
When the selected orientation differs, physical direction vectors come from
that geometry. Acquisition-only categorical orientation labels are retained as
``acquisition_read_orientation`` and ``acquisition_slice_orientation`` rather
than describing the selected image with those labels.

Single-package imaging data receive read/phase/slice coordinates in mm when
geometry is unambiguous. These retain the established three-dimensional RPS
coordinate grids. Multiple packages keep index axes and auxiliary physical
XYZ coordinates. Singleton slices retain their dimension, so XYZ remains 3D.
``extent_mm`` is a minimum/maximum pair for each spatial axis. Slice centre
spacing and physical thickness remain distinct; gaps do not inflate thickness.
This does not resolve the separately deferred odd-matrix centre convention.

For uniform single-package stacks, raw and matching scanner slice labels use
the physical XYZ voxel centres projected onto increasing slice index. Equivalent
scalar and one-element ``RECO_transposition`` representations select the same
geometry. This fixes the opposite-sign method/scanner slice origin without
changing acquired samples, image values, read/phase labels or the 0.001-mm
geometry guard. Stacks without usable physical geometry retain acquisition
labels; singleton and multi-package behavior is unchanged. The available
external phantom scans validate the slice-origin correction independently of
their remaining FLASH magnitude disagreement.

Raw and scanner imaging results share ``to_imaging_layout`` at both experiment
and direct Bruker loading boundaries. Missing nonspatial axes and coordinates
receive integer indices. Map arrays retain their declared map layout; arrays
without a complete spatial grid retain their explicit dimensions.
Complete sampling/repetition times add ``acquisition_time_s`` and ``time`` in
seconds, including padded singleton axes. Declared processed/echo times take
precedence over acquisition readout metadata. An absent fat/water split uses
index zero instead of the former scanner-only ``"Unknown"`` placeholder;
two-component acquisitions retain ``"Water"`` and ``"Fat"`` labels.
CPMG spectra now share acquisition/echo/channel/repetition order. Echo labels
use complete ``EffectiveTE`` values in seconds or integer indices. The raw
``phase_idx`` echo axis and singleton localization placeholders are replaced
by ``echo_idx``; code indexing raw CPMG echoes must migrate accordingly.
Samples, FFT scale and acquired inputs are preserved. Scanner spectra may
retain different spectral lengths/bandwidths from raw FIDs due to scanner
processing and digital-filter removal; these are not relabelled as equal.
