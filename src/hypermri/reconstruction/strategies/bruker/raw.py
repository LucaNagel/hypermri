"""Generic Bruker raw-data reconstruction strategies.

Notes
-----
Acquisition metadata own the default FOV phase ramp. Generic FLASH additionally
supports an explicit, bounded scanner FOV convention; it does not adopt other
ParaVision processing. Stages preserve acquired inputs, labelled geometry and
units and use the shared calculation/storage precision policy. Reconstruction
is eager and deterministic; strategy construction does not load data.
"""

import time
import traceback
from typing import ClassVar, Literal

import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.reconstruction.centred_dft import (
    centered_ifft,
    slice_encoding_to_image,
)
from hypermri.reconstruction.layout import _assign_time_coordinate
from hypermri.reconstruction.precision import (
    convert_for_calculation,
    convert_for_storage,
    parse_data_precision,
)
from hypermri.utils.coordinates.image_shifts import (
    get_orientation_flip_multipliers,
    shift_image,
)
from hypermri.utils.system.logging import init_default_logger

from ...axes import _echo_frequency_axes
from .._common import _create_inversion_idx_coordinate, debug_log_recon_strategy
from ..base import BaseRecoStrategy, _has_spectral_axis
from ._block_padding import _strip_block_padding
from ._encoding import _encoding_step_positions
from ._scanner_shift import _scanner_fov_shift
from .geometry import _attach_image_geometry

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)

#: Default precision of the Bruker raw reconstruction.  Real
#: results are stored as ``float32`` and complex results as ``complex64``; a
#: complex FID is not downcast before the Fourier transform.  ``float16`` is
#: not used by default because it overflows above 65504 and keeps only about
#: three significant digits.
RAW_RECONSTRUCTION_PRECISION = np.float32


def _zero_fill_centred(kspace, matrix_axes):
    """Zero-fill acquired k-space symmetrically into the reconstruction matrix.

    Along each axis the acquired block of length ``n_acq`` starts at
    ``n // 2 - n_acq // 2`` of the target length ``n``.  The acquired
    k-space centre at index ``n_acq // 2`` therefore lands on index
    ``n // 2``, the centre of the centred Fourier transform, and the image
    gets no linear phase from the zero filling.  This assumes symmetric
    acquisitions; partial Fourier and partial echo place the acquired block
    asymmetrically.

    Parameters
    ----------
    kspace : numpy.ndarray
        Acquired k-space.
    matrix_axes : dict of int to int
        Target length for each zero-filled axis of ``kspace``.  Each target
        length must be at least the acquired length.

    Returns
    -------
    numpy.ndarray
        New zero-filled array with the dtype of ``kspace``.

    Raises
    ------
    ValueError
        If a target length is smaller than the acquired length.
    """
    target_shape = list(kspace.shape)
    region = [slice(None)] * kspace.ndim
    for axis, matrix_size in matrix_axes.items():
        acquired_size = kspace.shape[axis]
        matrix_size = int(matrix_size)
        if matrix_size < acquired_size:
            raise ValueError(
                "Bruker raw zero filling: the reconstruction matrix "
                f"({matrix_size}) is smaller than the encoding matrix "
                f"({acquired_size}) along k-space axis {axis}; check "
                "PVM_Matrix and PVM_EncMatrix."
            )
        start = matrix_size // 2 - acquired_size // 2
        target_shape[axis] = matrix_size
        region[axis] = slice(start, start + acquired_size)
    kspace_zeros = np.zeros(target_shape, dtype=kspace.dtype)
    kspace_zeros[tuple(region)] = kspace
    return kspace_zeros


#: Axes whose inverse DFT gives the ``2dseq`` voxel order without the flip
#: table.
_2DSEQ_ORDER_AXES = ("read_idx", "phase_idx")
#: Spatial axes transformed into the ``2dseq`` voxel grid: read and phase by
#: :func:`centered_ifft`, the 3D ``slice_idx`` by
#: :func:`slice_encoding_to_image` (forward FFT, see the Fourier convention page of the documentation).
_SPATIAL_AXES = (*_2DSEQ_ORDER_AXES, "slice_idx")


def _raw_stage_message(expnum, stage, error, db):
    """Describe a failed stage while preserving its original cause."""
    message = f"{expnum}: reco_rawdata()\nError during {stage}:\n{type(error).__name__}: {error}\n"
    if db:
        message += f"Full traceback:\n{traceback.format_exc()}"
    return message


def _raw_fourier_axes(spatial_dimension):
    """Return spatial FFT labels for 2D slices or a 3D volume."""
    if spatial_dimension == "2D":
        return ["read_idx", "phase_idx"]
    if spatial_dimension == "3D":
        return ["read_idx", "phase_idx", "slice_idx"]
    return []


def _transpose_raw_spatial_axes(kspace, n_inversion_time, mge):
    """Order labelled k-space as acquisition/read/phase/slice/.../channel.

    MGE omits absent dimensions; raw RARE preserves fat/water and inversion
    axes. Values, coordinates and dtype are preserved by the transpose.
    """
    order = ["acquisition_idx", "read_idx", "phase_idx", "slice_idx"]
    if not mge and "waterfat_idx" in kspace.dims:
        order.append("waterfat_idx")
    order.append("repetition_idx")
    if not mge and n_inversion_time > 1:
        order.append("inversion_idx")
    order.append("channel_idx")
    if mge:
        order = [dim for dim in order if dim in kspace.dims]
    return kspace.transpose(*order)


def _attach_raw_frame_time(reco_data, frame_time_s):
    """Delegate seconds-based labels to the shared reconstruction time owner."""
    return _assign_time_coordinate(reco_data, "time", "repetition_idx", frame_time_s)


def _create_labelled_kspace(data, dims, coords):
    """Create labelled k-space from samples, axis names and coordinate tuples.

    Input shape follows dims; coordinates retain units, dtype and order.
    Samples are not modified and the result can share their storage.
    Invalid dimension or coordinate shapes raise ValueError.
    """
    # Validate dimensions
    if len(data.shape) != len(dims):
        raise ValueError(
            f"Number of dimensions ({len(dims)}) must match the data shape ({len(data.shape)})"
        )

    # Create default coordinates if not provided
    if coords is None:
        coords = {
            dim: np.arange(size) for dim, size in zip(dims, data.shape, strict=False)
        }
    # Create or update xarray DataArray
    try:
        kspace_xarray = xr.DataArray(data, dims=dims, coords=coords)
    except ValueError as exc:
        coord_info = {name: np.array(coords[name][1]).shape for name in coords}
        raise ValueError(
            f"""\nCould not create xarray DataArray, shape of: \nData: {data.shape},
            \nDimensions: {dims}, \nCoordinates: {coord_info}\n
            """
        ) from exc

    return kspace_xarray


def _attach_kspace_frequency_axes(freq_axis_hz, freq_axis_ppm, kspace_xarray):
    """Attach aligned ppm and Hz coordinates along acquisition_idx.

    Frequency arrays must match acquisition length; other lengths are skipped.
    Sample values, dimensions and storage dtype remain unchanged.
    """
    if freq_axis_ppm is not None and len(freq_axis_ppm) == kspace_xarray.sizes.get(
        "acquisition_idx", 0
    ):
        kspace_xarray = kspace_xarray.assign_coords(
            {
                "freq_axis_ppm": (
                    "acquisition_idx",
                    freq_axis_ppm,
                    {"unit": "ppm"},
                )
            }
        )
    if freq_axis_hz is not None and len(freq_axis_hz) == kspace_xarray.sizes.get(
        "acquisition_idx", 0
    ):
        kspace_xarray = kspace_xarray.assign_coords(
            {
                "freq_axis_hz": (
                    "acquisition_idx",
                    freq_axis_hz,
                    {"unit": "Hz"},
                )
            }
        )

    return kspace_xarray


def _reorder_kspace_slices(acquisition_slice_order, kspace_xarray):
    """Reorder slice_idx samples by acquisition object order.

    slice_idx has len(acquisition_slice_order) entries. Coordinates retain their
    original order; values and dtype are permuted without interpolation.
    """
    slice_order = np.argsort(acquisition_slice_order)
    original_coords = kspace_xarray.coords
    slice_dim_index = kspace_xarray.dims.index("slice_idx")

    # Create a slicing object to reorder along the slice dimension
    slices = [slice(None)] * kspace_xarray.ndim
    slices[slice_dim_index] = slice_order

    reordered_data = kspace_xarray.values[tuple(slices)]

    # Create a new xarray with the reordered data and original
    # coordinates
    kspace_xarray = xr.DataArray(
        reordered_data,
        coords=original_coords,
        dims=kspace_xarray.dims,
    )

    return kspace_xarray


def _attach_kspace_axis_geometry(
    has_rps_positions,
    kspace_xarray,
    matrix,
    n_slices,
    resolution_mm,
    rps_axes_mm,
    rps_extent_mm,
):
    """Attach RPS axis lengths, resolution and extents to labelled k-space.

    Matrix is read/phase/slice; resolution, extent and axis positions use mm.
    Missing per-axis metadata retain the existing singleton fallbacks.
    Sample values, dimensions and coordinates are unchanged.
    """
    try:
        kspace_xarray["read_idx"].attrs["matrix"] = matrix[0]
        kspace_xarray["read_idx"].attrs["resolution_mm"] = resolution_mm[
            0
        ]  # should be change to kspace unit at some point
        if has_rps_positions:
            kspace_xarray["read_idx"].attrs["extent_mm"] = rps_extent_mm[0:2]
            kspace_xarray["read_idx"].attrs["axis_mm"] = rps_axes_mm[0]
    except (TypeError, IndexError) as exc:
        logger.warning("Optional k-space axis geometry is incomplete: %s", exc)
        kspace_xarray["read_idx"].attrs["matrix"] = 1
        kspace_xarray["read_idx"].attrs["resolution_mm"] = 0.0
        kspace_xarray["read_idx"].attrs["extent_mm"] = 0.0
        kspace_xarray["read_idx"].attrs["axis_mm"] = 0.0
    try:
        kspace_xarray["phase_idx"].attrs["matrix"] = matrix[1]
        kspace_xarray["phase_idx"].attrs["resolution_mm"] = resolution_mm[1]
        if has_rps_positions:
            kspace_xarray["phase_idx"].attrs["extent_mm"] = rps_extent_mm[2:4]
            kspace_xarray["phase_idx"].attrs["axis_mm"] = rps_axes_mm[1]
    except (TypeError, IndexError) as exc:
        logger.warning("Optional k-space axis geometry is incomplete: %s", exc)
        kspace_xarray["phase_idx"].attrs["matrix"] = 1
        kspace_xarray["phase_idx"].attrs["resolution_mm"] = 0.0
        kspace_xarray["phase_idx"].attrs["extent_mm"] = 0.0
        kspace_xarray["phase_idx"].attrs["axis_mm"] = 0.0
    try:
        kspace_xarray["slice_idx"].attrs["matrix"] = matrix[2]
        kspace_xarray["slice_idx"].attrs["resolution_mm"] = resolution_mm[2]
        if has_rps_positions:
            kspace_xarray["slice_idx"].attrs["extent_mm"] = rps_extent_mm[4:6]
            kspace_xarray["slice_idx"].attrs["axis_mm"] = rps_axes_mm[2]
    except (TypeError, IndexError) as exc:
        logger.warning("Optional k-space axis geometry is incomplete: %s", exc)
        kspace_xarray["slice_idx"].attrs["matrix"] = n_slices
        kspace_xarray["slice_idx"].attrs["resolution_mm"] = 0.0
        kspace_xarray["slice_idx"].attrs["extent_mm"] = 0.0
        kspace_xarray["slice_idx"].attrs["axis_mm"] = 0.0

    return kspace_xarray


def _fill_encoded_kspace(kspace, handler, *, read_axis):
    """Place acquired phase lines at their declared full-grid indices.

    Readout zero filling remains symmetric pending partial-echo references.
    Independent 2D slices also retain centred padding; 3D partitions use their
    signed encoding steps. Returns a new array without changing sample dtype.
    """
    centred_axes = {read_axis: handler.matrix[0]}
    if handler.spatial_dimension != "3D":
        centred_axes[2] = handler.matrix[2]
    result = _zero_fill_centred(kspace, centred_axes)
    encoded_axes = [(3, 1, "PVM_EncSteps1")]
    if handler.spatial_dimension == "3D":
        encoded_axes.append((2, 2, "PVM_EncSteps2"))
    for axis, matrix_axis, parameter in encoded_axes:
        matrix_size = int(handler.matrix[matrix_axis])
        try:
            steps = handler.method[parameter]
        except KeyError as error:
            raise ValueError(
                f"Bruker raw phase placement requires {parameter}; "
                "use scanner reconstruction when encoding positions are unavailable."
            ) from error
        positions = _encoding_step_positions(steps, matrix_size, parameter)
        if len(positions) != result.shape[axis]:
            raise ValueError(
                f"Invalid Bruker {parameter} length: expected {result.shape[axis]}, "
                f"got {len(positions)}."
            )
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"Invalid Bruker {parameter}: encoding steps repeat.")
        shape = list(result.shape)
        shape[axis] = matrix_size
        expanded = np.zeros(shape, dtype=result.dtype)
        region = [slice(None)] * result.ndim
        region[axis] = positions
        expanded[tuple(region)] = result
        result = expanded
    return result


def _reshape_2d_multislice_lines(fid, handler, *, line_shape, label):
    """Sort 2D multislice Bruker lines into slice-major phase order.

    Bruker acquires ``ACQ_phase_factor`` phase rows for one slice before it
    advances to the next slice; a factor of one is phase-major and a factor
    equal to the phase matrix is slice-major. Each acquired row holds the
    trailing ``line_shape`` samples, for example ``(receivers, read)`` or
    ``(echoes, receivers, read)``. The result has shape ``(1, repetitions,
    slices, phase rows, *line_shape)`` in acquisition slice order; it is a new
    array with the input dtype and the input samples are not modified.
    """
    n_phase = int(handler.enc_matrix[1])
    n_slices = int(handler.enc_matrix[2])
    acqp = getattr(handler, "acqp", None) or {}
    phase_factor = BrukerRawRecoStrategy._positive_integer_parameter(
        acqp.get("ACQ_phase_factor", 1), "ACQ_phase_factor"
    )
    if n_phase % phase_factor != 0:
        raise ValueError(
            f"Invalid Bruker ACQ_phase_factor for 2D raw reconstruction of {label}: "
            f"phase matrix {n_phase}, phase factor {phase_factor}."
        )
    line_shape = tuple(int(size) for size in line_shape)
    try:
        kspace = np.reshape(
            fid,
            (
                1,
                handler.n_repetitions,
                n_phase // phase_factor,
                n_slices,
                phase_factor,
                *line_shape,
            ),
        )
    except ValueError as error:
        raise ValueError(
            f"Failed to reshape Bruker 2D raw data of {label}: {error}"
        ) from error
    kspace = np.moveaxis(kspace, 3, 2)
    return np.reshape(
        kspace, (1, handler.n_repetitions, n_slices, n_phase, *line_shape)
    )


def _mge_readout_is_bipolar(handler):
    """Return whether every second MGE echo is read with reversed gradient.

    ParaVision MGE records ``EchoAcqMode``: ``allEchoes`` acquires echoes
    under alternating read gradients, so echoes 1, 3, 5, ... arrive with
    reversed read order; ``positiveEchoes`` uses flyback gradients and keeps
    every echo in the same order. A missing parameter means a monopolar
    readout, as in MGE versions without the bipolar option.

    Parameters
    ----------
    handler : VendorHandler
        Handler whose ``method`` mapping may contain ``EchoAcqMode``.

    Returns
    -------
    bool
        True for ``allEchoes``.

    Raises
    ------
    ValueError
        If ``EchoAcqMode`` has another value.
    """
    method = getattr(handler, "method", None) or {}
    mode = method.get("EchoAcqMode")
    if mode is None or str(mode) == "positiveEchoes":
        return False
    if str(mode) == "allEchoes":
        return True
    raise ValueError(
        f"Unsupported Bruker MGE EchoAcqMode {mode!r}: expected 'allEchoes' or "
        "'positiveEchoes'. Use scan.reconstruct('seq2d') for the scanner images."
    )


class BrukerRawRecoStrategy(BaseRecoStrategy):
    """Reconstruct generic Bruker Cartesian raw samples with explicit stages.

    Notes
    -----
    Construction takes no arguments and performs no reads or reconstruction.
    :meth:`reconstruct` accepts the handler and scientific options. Missing FID
    samples are loaded once and retained as acquired ``handler.fid``; all
    subsequent stages operate on copies and return labelled arrays directly.
    The scan, rather than this strategy or the handler, owns stored results.
    Acquisition-derived FOV shifting is the default; the optional scanner
    convention is restricted to the documented FLASH geometry.

    Examples
    --------
    >>> strategy = BrukerRawRecoStrategy()
    >>> image = strategy.reconstruct(handler)  # doctest: +SKIP
    """

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "postprocess_fid",
            "reshape_fid",
            "set_kspace_xarray",
            "shift_xarray",
            "fov_shift_sign",
            "fov_shift_source",
            "set_data",
            "db",
            "data_precision",
        }
    )

    @staticmethod
    def _load_fid_on_demand(handler) -> None:
        """Read the ``fid`` or ``ser`` file of a lazily loaded scan.

        A lazy or minimal load reads no raw data.  The file is read once,
        exactly as a full load reads it, and stored as ``handler.fid``, so
        the reconstruction equals the one of a full load.  A handler that already holds ``fid`` data is
        left unchanged.

        Parameters
        ----------
        handler : BrukerHandler
            Handler of the scan.  ``handler.fid`` is set as a side effect.

        Raises
        ------
        ValueError
            If the scan folder has neither a non-empty ``fid`` nor a
            non-empty ``ser`` file.
        """
        if getattr(handler, "fid", None) is not None:
            return
        fid = handler.load_fid_file()
        if fid.size == 0:
            raise ValueError(
                f"Raw reconstruction of scan {handler.path} needs a non-empty "
                "'fid' or 'ser' file in the scan folder; use the 'seq2d' "
                "reconstruction for scans without raw data."
            )
        handler.fid = fid

    def reconstruct(
        self,
        handler,
        *,
        fov_shift_source: Literal["acquisition", "scanner"] = "acquisition",
        **reco_options,
    ) -> xr.DataArray | None:
        """Reconstruct the raw FID of a Bruker scan into image space.

        Parameters
        ----------
        handler : BrukerHandler
            Handler of the scan.  A missing FID is read first (see
            :meth:`_load_fid_on_demand`). With ``reshape_fid=True``, the
            Cartesian encoding matrix from ``PVM_EncMatrix`` must be available
            as ``handler.enc_matrix``; it is never inferred from sample counts
            or scanner-reconstructed images.
        fov_shift_source
            Source of the FOV phase ramp. ``"acquisition"`` keeps the independent
            raw default. ``"scanner"`` explicitly selects the validated scanner
            convention: stock ``<Bruker:FLASH>`` using the generic 3D strategy,
            Head_Supine/coronal/H_F, one excited
            slab/package, a full even partition matrix, zero in-plane offsets,
            and finite ``RECO_rotate`` of shape (3, 1) from ``pdata/1/reco``.
            Its in-plane entries must be 0.5, and its declared slice shift
            plus 0.5 must equal the negative acquisition shift within 1e-9
            voxel. The applied slice ramp is the negative declared shift minus
            half a voxel, the acquisition ramp: the slice-encoding transform
            already centres the slices on the ``2dseq`` grid, so both sources
            give the same image and ``"scanner"`` only adds the metadata
            checks and provenance. Other ParaVision processing settings are
            not reproduced.
            Requires ``shift_xarray=True`` and ``fov_shift_sign=1``.
        **reco_options
            Stage controls: ``no_reconstruction=False`` skips the entire
            operation when True; ``postprocess_fid=True`` removes line padding;
            ``reshape_fid=True`` sorts acquisition loops into k-space;
            ``set_kspace_xarray=True`` labels k-space (False retains the legacy
            explicit-handler-input warning path); ``shift_xarray=True`` applies
            the FOV ramp; ``fov_shift_sign=1.0`` scales acquisition offsets;
            ``set_data=True`` transforms k-space into spatial data; and
            ``db=True`` includes tracebacks in stage errors without changing
            failure policy. ``data_precision`` defaults to float32 storage with
            complex128 calculations; float64 stores complex128 and float16 uses
            complex64 calculations/storage. See
            :mod:`hypermri.reconstruction.precision`.

        Returns
        -------
        xarray.DataArray or None
            The labelled reconstruction in the dimension order documented by
            :meth:`reco_rawdata`, retaining coordinate units and orientation.
            For spatial images of a uniform single-package stack, slice labels
            project available physical XYZ positions onto increasing slice index;
            acquisition read/phase labels and image samples are preserved.
            Stored real/complex dtype follows ``data_precision``. With
            ``set_data=False``, returns labelled k-space instead.
            ``None`` for ``no_reconstruction=True``. Scanner mode additionally
            records ``fov_shift_source``, ``fov_shift_vox`` (read/phase/slice
            voxels), ``bruker_reco_rotate`` and ``bruker_reco_parameter_source``.

        Raises
        ------
        ValueError
            If ``data_precision`` is not supported, if no FID can be read,
            if ``PVM_EncMatrix`` is missing while reshaping is requested, or
            if a stage of :meth:`reco_rawdata` fails, or if scanner mode has
            unsupported metadata or conflicting options (before reading FID).
            Missing encoding metadata
            are rejected before FID preprocessing; choose ``scanner_data``
            explicitly for the scanner reconstruction of an unsupported format.

        Notes
        -----
        The FID is converted to the calculation precision as a copy, so
        ``handler.fid`` keeps the acquired data and repeated
        reconstructions with different ``data_precision`` values start from
        the same data.  The method returns its result directly and does not store
        intermediate or final reconstruction arrays on the handler. This eager
        operation is deterministic for fixed acquired inputs and options.

        Examples
        --------
        >>> strategy = BrukerRawRecoStrategy()
        >>> image = strategy.reconstruct(handler, data_precision="float64")  # doctest: +SKIP
        """
        start_time = time.time()
        logger.debug(
            "BrukerRawRecoStrategy.reconstruct starting for scan "
            f"{getattr(handler, 'expnum', getattr(handler, 'id', 'unknown'))}"
        )
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerRawRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        postprocess_fid = reco_options.get("postprocess_fid", True)
        reshape_fid = reco_options.get("reshape_fid", True)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        fov_shift_sign = reco_options.get("fov_shift_sign", 1.0)
        set_data = reco_options.get("set_data", True)
        db = reco_options.get("db", True)
        DATA_PRECISION = parse_data_precision(
            reco_options.get("data_precision", RAW_RECONSTRUCTION_PRECISION)
        )

        if fov_shift_source not in ("acquisition", "scanner"):
            raise ValueError("fov_shift_source must be 'acquisition' or 'scanner'.")
        scanner_provenance = None
        explicit_shift = {}
        if fov_shift_source == "scanner":
            if type(self) is not BrukerRawRecoStrategy:
                raise ValueError(
                    "Scanner FOV shifting supports only generic FLASH raw reconstruction."
                )
            if not shift_xarray:
                raise ValueError(
                    "fov_shift_source='scanner' requires shift_xarray=True."
                )
            if fov_shift_sign != 1.0:
                raise ValueError(
                    "fov_shift_source='scanner' requires fov_shift_sign=1."
                )
            scanner_shift, scanner_provenance = _scanner_fov_shift(handler)
            explicit_shift["fov_shift_vox"] = scanner_shift

        self._load_fid_on_demand(handler)

        # Reconstruct the raw data into an xarray DataArray
        logger.debug("About to call reco_rawdata ->")
        reco_data = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=set_kspace_xarray,
            shift_xarray=shift_xarray,
            fov_shift_sign=fov_shift_sign,
            set_data=set_data,
            db=db,
            data_precision=DATA_PRECISION,
            values=handler.fid,
            **explicit_shift,
        )
        if scanner_provenance is not None:
            if any(
                reco_data.sizes.get(dim) != size
                for dim, size in zip(
                    ("read_idx", "phase_idx", "slice_idx"), handler.matrix, strict=True
                )
            ):
                raise ValueError(
                    "Scanner FOV shift requires k-space matching the full acquisition matrix."
                )
            reco_data.attrs.update(scanner_provenance)
        logger.info(
            f"[BrukerRawRecoStrategy] Reconstruction completed., took {time.time() - start_time:.3f}s"
        )
        return reco_data

    @staticmethod
    def _is_fat_water_enabled(handler) -> bool:
        """Return True when Bruker fat/water separation is explicitly enabled."""
        setting = getattr(handler, "fat_water_separation", "Off")
        return setting is True or str(setting).strip().lower() == "on"

    @staticmethod
    def _positive_integer_parameter(value, parameter_name: str) -> int:
        """Return one positive integral scanner parameter or fail explicitly."""
        values = np.asarray(value).reshape(-1)
        if values.size != 1:
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected one positive integer, "
                f"got {values.size} values."
            )

        scalar = values[0]
        try:
            numeric = float(scalar)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected one positive integer."
            ) from exc

        integer = int(numeric) if np.isfinite(numeric) else 0
        if integer <= 0 or numeric != integer:
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected one positive integer."
            )
        return integer

    @staticmethod
    def _spatial_coordinates(handler, dims, shape) -> dict:
        """Build physical RPS coordinates or safe index-coordinate fallbacks.

        A multi-slice-pack acquisition can contain different orientations and
        offsets per pack.  Such geometry has no single regular RPS grid, so
        the handler's ``rps_grid_mm`` is None for it, while the
        scanner-space ``x``, ``y`` and ``z`` coordinates can still be
        attached later from per-frame ``VisuCore`` metadata.  In that case the
        dimension coordinates must remain indices instead of aborting raw
        reconstruction.
        """
        shape_by_dim = dict(zip(dims, shape, strict=False))
        spatial_dims = ("read_idx", "phase_idx", "slice_idx")
        expected_shape = tuple(shape_by_dim[dim] for dim in spatial_dims)

        try:
            rps_grid = handler.rps_grid_mm
        except AttributeError:
            rps_grid = None

        if rps_grid is not None and all(
            np.shape(rps_grid[index]) == expected_shape for index in range(3)
        ):
            return {
                dim: (
                    list(spatial_dims),
                    rps_grid[index],
                    {"unit": "mm"},
                )
                for index, dim in enumerate(spatial_dims)
            }

        return {
            dim: (
                dim,
                np.arange(shape_by_dim[dim]),
                {"unit": "index"},
            )
            for dim in spatial_dims
        }

    @staticmethod
    def _fid_for_calculation(handler, postprocess_fid, data_precision, data=None):
        """Return the FID that :meth:`reco_rawdata` starts from.

        Explicit ``data`` is used when given, the acquired
        ``handler.fid`` otherwise, whatever ``postprocess_fid`` is.  The array
        is converted with
        :func:`~hypermri.reconstruction.precision.convert_for_calculation`, which
        returns a copy when it changes the dtype, so the handler keeps the
        acquired data.  Returns ``None`` when there is no FID.
        """
        if data is not None:
            return convert_for_calculation(data, data_precision)
        return convert_for_calculation(getattr(handler, "fid", None), data_precision)

    def reco_rawdata(
        self,
        handler=None,
        postprocess_fid=True,
        reshape_fid=True,
        set_kspace_xarray=True,
        shift_xarray=True,
        fov_shift_sign=1.0,
        set_data=True,
        db=True,
        data_precision=RAW_RECONSTRUCTION_PRECISION,
        *,
        values: np.ndarray | None = None,
        kspace_xarray: xr.DataArray | None = None,
        fov_shift_vox: tuple[float, float, float] | None = None,
    ):
        """Reconstruct raw fid data into k-space and spatial domain xarray DataArrays.

        Parameters
        ----------
        handler: VendorHandler
            Metadata source for acquired samples, geometry and units. With
            ``reshape_fid=True``, ``handler.enc_matrix`` must contain the
            Cartesian read, phase and slice/second-phase lengths derived from
            ``PVM_EncMatrix``. No encoding layout is inferred when it is missing.
        postprocess_fid: bool
            If True, postprocess the fid data before reconstruction.
        reshape_fid: bool
            If True, reshape the fid data into k-space.
        set_kspace_xarray: bool
            If True, create an xarray DataArray for k-space data.
        shift_xarray: bool
            If True, shift the k-space data with a linear phase ramp by the
            FOV offsets of the scan (``shift_image``).  A failed shift raises
            ``ValueError``; the reconstruction never continues unshifted.
        fov_shift_sign : float
            Multiplier of the FOV offsets applied by the phase ramp.
        set_data: bool
            If True, create an xarray DataArray for spatial-domain data. If
            False, return the k-space xarray instead.
        db: bool
            If True, include the full traceback in failed-stage errors.
            The error policy is identical for True and False.
        data_precision: str, numpy type or numpy.dtype
            Precision of the calculation and of the stored result; see
            :mod:`hypermri.reconstruction.precision`.  The FID is converted with
            :func:`~hypermri.reconstruction.precision.convert_for_calculation`, so a
            complex FID is only reduced to ``complex64`` for ``float16``.
            Defaults to ``RAW_RECONSTRUCTION_PRECISION`` (``np.float32``).

        values : numpy.ndarray, optional
            Explicit complex FID or sorted k-space samples; shape follows the
            selected reshaping stage. Defaults to acquired ``handler.fid``,
            also when ``postprocess_fid=False``. Inputs are never mutated.
        kspace_xarray : xarray.DataArray, optional
            Labelled k-space; required when ``set_kspace_xarray=False``. The
            array and its metadata are copied before phase shifting or spatial
            reconstruction.
        fov_shift_vox : tuple of float, optional
            Explicit finite read/phase/slice phase-ramp vector in voxels for
            3D k-space; a positive entry moves the reconstructed image
            content to higher indices. Replaces acquisition-derived offsets.
            Requires
            ``shift_xarray=True`` and ``fov_shift_sign=1``; defaults to None.

        Returns
        -------
        xr.DataArray
            The spatial-domain reconstruction when ``set_data`` is True;
            otherwise the k-space xarray. The default spatial dimension order is
            acquisition/read/phase/slice/repetition/channel, with optional
            waterfat before repetition and inversion before channel. Coordinate
            values, directions and units follow the handler. Spatial-domain
            storage dtype follows ``data_precision``; k-space retains calculation
            precision.

        Raises
        ------
        ValueError
            If ``PVM_EncMatrix`` is missing while ``reshape_fid=True`` (checked
            before FID preprocessing), if the handler has no FID data, if the
            k-space data are missing, or if building the k-space array, applying
            the FOV shift or the spatial reconstruction fails. Stage-failure
            messages name the original error, which is chained as the cause. A failed
            FOV shift is raised for ``db=True`` and ``db=False`` alike; pass
            ``shift_xarray=False`` to reconstruct without it. An explicit shift
            requires a finite three-vector, 3D k-space and compatible options.
            ``set_kspace_xarray=False`` without ``kspace_xarray`` is rejected.

        Notes
        -----
        The deterministic, eager stages pass intermediate arrays directly. No
        reconstruction arrays or acquisition metadata are written to the handler.
        FOV shifts increment the result's ``num_image_fov_shift`` attribute once.
        An explicit vector is retained in ``fov_shift_vox`` (read/phase/slice
        voxels); it does not imply scanner metadata validation.
        Acquired samples, explicitly supplied arrays and their metadata are kept.

        Examples
        --------
        >>> result = strategy.reco_rawdata(
        ...     handler, values=sorted_fid, postprocess_fid=False,
        ...     reshape_fid=False, shift_xarray=False
        ... )  # doctest: +SKIP
        """
        return self._reconstruct_rawdata(
            handler,
            postprocess_fid,
            reshape_fid,
            set_kspace_xarray,
            shift_xarray,
            fov_shift_sign,
            set_data,
            db,
            data_precision,
            mge=False,
            data=values,
            kspace_xarray=kspace_xarray,
            fov_shift_vox=fov_shift_vox,
        )

    def _label_raw_kspace(self, handler, kspace):
        """Build raw k-space labels; MGE additionally prepares echo metadata."""
        n_inversion_time = getattr(handler, "n_inversion_time", 1)
        is_fair_rare = n_inversion_time > 1
        is_fat_water = self._is_fat_water_enabled(handler)

        dims = [
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
        ]
        if is_fat_water:
            dims.append("waterfat_idx")
        if is_fair_rare:
            dims.append("inversion_idx")
        dims.extend(["channel_idx", "read_idx"])

        if len(dims) != kspace.ndim:
            raise ValueError(
                f"Dimension mismatch for k-space data: shape {kspace.shape} "
                f"has {kspace.ndim} axes, but inferred dims are {dims}. "
                "Check fat/water and FAIR_RARE axis handling."
            )

        coords = {
            "acquisition_idx": (
                "acquisition_idx",
                (
                    handler.sampling_time_axis_s
                    if handler.sampling_time_axis_s is not None
                    and len(np.atleast_1d(handler.sampling_time_axis_s))
                    == kspace.shape[0]
                    else np.arange(kspace.shape[0])
                ),
                {
                    "unit": (
                        "s"
                        if handler.sampling_time_axis_s is not None
                        and len(np.atleast_1d(handler.sampling_time_axis_s))
                        == kspace.shape[0]
                        else "index"
                    )
                },
            ),
            "repetition_idx": (
                "repetition_idx",
                np.array(np.arange(kspace.shape[1])),
                {"unit": "index"},
                # repetition_idx is count of repetition
            ),
            "channel_idx": (
                "channel_idx",
                np.array(np.arange(handler.n_receivers)),
                {"unit": "index"},  # channel_idx is count of receiver
            ),
        }
        coords.update(self._spatial_coordinates(handler, dims, kspace.shape))

        if is_fat_water:
            n_waterfat = kspace.shape[dims.index("waterfat_idx")]
            waterfat_coords = (
                np.array(["Water", "Fat"], dtype=object)
                if n_waterfat == 2
                else np.arange(n_waterfat)
            )
            coords["waterfat_idx"] = (
                "waterfat_idx",
                waterfat_coords,
                {"unit": "type"},
            )

        # Add inversion_idx coordinate for FAIR_RARE
        if is_fair_rare:
            # Get number of slices from kspace shape
            n_slices = kspace.shape[2]  # slice_idx is 3rd dimension in kspace

            # Use helper function to create slice-dependent inversion coordinate
            coords["inversion_idx"] = _create_inversion_idx_coordinate(
                handler,
                n_inversion_time,
                n_slices,
                {"slice_idx": n_slices, "inversion_idx": n_inversion_time},
            )

        kspace_xarray = self.set_kspace_xarray(
            handler, data=kspace, dims=dims, coords=coords
        )

        return kspace_xarray

    def _reconstruct_rawdata(
        self,
        handler,
        postprocess_fid,
        reshape_fid,
        label_kspace,
        shift_xarray,
        fov_shift_sign,
        spatial_reconstruction,
        db,
        data_precision,
        *,
        mge,
        data=None,
        kspace_xarray=None,
        fov_shift_vox=None,
    ):
        """Pass fresh samples between stages; never cache intermediates on a handler."""
        if fov_shift_vox is not None:
            shift = np.asarray(fov_shift_vox, dtype=float)
            if shift.shape != (3,) or not np.all(np.isfinite(shift)):
                raise ValueError(
                    "fov_shift_vox requires three finite read/phase/slice voxel shifts."
                )
            if not shift_xarray or fov_shift_sign != 1.0:
                raise ValueError(
                    "fov_shift_vox requires shift_xarray=True and fov_shift_sign=1."
                )
            if getattr(handler, "spatial_dimension", None) != "3D":
                raise ValueError("fov_shift_vox requires 3D k-space.")
            fov_shift_vox = tuple(float(value) for value in shift)
        if reshape_fid and getattr(handler, "enc_matrix", None) is None:
            raise ValueError(
                "Bruker raw k-space reshaping requires PVM_EncMatrix in the "
                "method parameters (handler.enc_matrix). Export the original "
                "acquisition encoding metadata; if the method has no documented "
                "Cartesian encoding layout, use the 'seq2d' reconstruction "
                "(scan.scanner_data) explicitly."
            )
        fid = self._fid_for_calculation(handler, postprocess_fid, data_precision, data)
        if fid is None:
            raise ValueError("No FID data available for reconstruction")
        fid_pp = fid
        if postprocess_fid:
            try:
                fid_pp = self.postprocess_fid_file(values=fid, handler=handler)
            except Exception as exc:
                raise ValueError(
                    _raw_stage_message(handler.expnum, "postprocess_fid_file", exc, db)
                ) from exc
        kspace = (
            self.reshape_fid_to_kspace(values=fid_pp, handler=handler)
            if reshape_fid
            else fid_pp
        )
        if label_kspace:
            try:
                kspace_xarray = self._label_raw_kspace(handler, kspace)
            except Exception as exc:
                raise ValueError(
                    _raw_stage_message(handler.expnum, "set_kspace_xarray", exc, db)
                ) from exc
        elif kspace_xarray is None:
            raise ValueError(
                "reco_rawdata with set_kspace_xarray=False requires an explicit "
                "kspace_xarray; pass the labelled k-space as kspace_xarray=... or "
                "use set_kspace_xarray=True to build it from the FID."
            )
        # Metadata updates and phase shifts must not mutate supplied arrays.
        kspace_xarray = kspace_xarray.copy(deep=True)
        if fov_shift_vox is not None and (
            kspace_xarray.attrs.get("spatial_dimension") != "3D"
            or not all(
                dim in kspace_xarray.dims
                for dim in ("read_idx", "phase_idx", "slice_idx")
            )
            or kspace_xarray.attrs.get("num_image_fov_shift", 0) != 0
        ):
            raise ValueError("fov_shift_vox requires unshifted labelled 3D k-space.")
        if shift_xarray:
            try:
                kspace_xarray = shift_image(
                    data=kspace_xarray,
                    shift_method="phase",
                    scan=handler,
                    domain="kspace",
                    fov_shift_sign=fov_shift_sign,
                    shift_vox=fov_shift_vox,
                )
            except Exception as exc:
                message = _raw_stage_message(handler.expnum, "shift_image", exc, db)
                message += "Correct the FOV offset parameters of the scan or pass shift_xarray=False to reconstruct without the FOV shift.\n"
                raise ValueError(message) from exc
        reco_data = kspace_xarray
        domain = "kspace"
        if spatial_reconstruction:
            try:
                kspace_xarray = _transpose_raw_spatial_axes(
                    kspace_xarray, getattr(handler, "n_inversion_time", 1), mge
                )
                reco_data = self.set_data(
                    kspace_xarray=kspace_xarray,
                    axes_names=_raw_fourier_axes(handler.spatial_dimension),
                    handler=handler,
                    write_to_object=False,
                )
                reco_data.data = convert_for_storage(reco_data.data, data_precision)
                domain = "spatial"
            except Exception as exc:
                raise ValueError(
                    _raw_stage_message(handler.expnum, "set_data", exc, db)
                ) from exc
        reco_data.attrs["domain"] = domain
        if fov_shift_vox is not None:
            reco_data.attrs["fov_shift_vox"] = fov_shift_vox
        if (
            not mge
            and "repetition_idx" in reco_data.dims
            and getattr(handler, "frame_time_s", None) is not None
        ):
            reco_data = _attach_raw_frame_time(reco_data, handler.frame_time_s)
        return reco_data

    def set_kspace_xarray(
        self,
        handler,
        data: np.ndarray = None,
        dims: list = None,
        coords: dict = None,
        write_to_object: bool = False,
    ):
        """Label raw k-space samples and attach acquisition metadata.

        Parameters
        ----------
        handler : BrukerHandler
            Metadata source with slice order, directions, acquisition times,
            matrix, RPS geometry and physical sizes. Multiple packages have
            no RPS axes or common ``slice_spacing_mm``.
        data : numpy.ndarray, optional
            Required sample array with one axis per entry of ``dims``.
            The dtype is preserved; acquired input samples are not modified.
        dims : sequence of str, optional
            Required labels in the order of ``data``. Spatial axes are named
            ``read_idx``, ``phase_idx`` and ``slice_idx``; acquisition,
            repetition and channel axes may also be present.
        coords : dict, optional
            xarray coordinate definitions. Missing coordinates default to
            integer indices. Time metadata use seconds and spatial size
            attributes use mm; coordinate values keep their input precision.
        write_to_object : bool, optional
            Store the result as ``handler.kspace_xarray`` when True.
            Defaults to False, which returns the array without storing it.

        Returns
        -------
        xarray.DataArray or None
            Labelled k-space with unchanged dimension order and sample dtype.
            Slices are reordered by the acquisition object order. The slice
            coordinate and array attributes include ``slice_spacing_mm`` and
            ``slice_thickness_mm`` for one spatial package. Thickness means
            nominal 2D slice thickness or 3D partition size, not excitation
            slab thickness. Array attributes survive dropping ``slice_idx``.
            Returns None when ``write_to_object`` is True.

        Raises
        ------
        ValueError
            If dimensions, coordinates or slice order do not match the data.
        AttributeError
            If required handler metadata are unavailable.

        Notes
        -----
        This eager, deterministic metadata operation does not read acquired
        files or apply Fourier transforms. It may share sample storage with
        the input when no slice reordering is needed. Missing optional
        coordinates and RPS attributes retain their documented fallbacks.

        Examples
        --------
        >>> labelled = strategy.set_kspace_xarray(
        ...     handler, data=kspace, dims=dimension_names
        ... )  # doctest: +SKIP
        """
        logger.debug(f"set_kspace_xarray called with data shape: {data.shape}")
        kspace_xarray = _create_labelled_kspace(data=data, dims=dims, coords=coords)
        sampling = getattr(handler, "sampling_time_axis_s", None)
        if (
            "acquisition_idx" in dims
            and sampling is not None
            and np.ndim(sampling) == 1
            and len(sampling) == kspace_xarray.sizes["acquisition_idx"]
        ):
            kspace_xarray = kspace_xarray.assign_coords(
                acquisition_time_s=("acquisition_idx", sampling, {"unit": "s"})
            )
        if _has_spectral_axis(handler):
            kspace_xarray = _attach_kspace_frequency_axes(
                freq_axis_hz=handler.get("freq_axis_hz", None),
                freq_axis_ppm=handler.get("freq_axis_ppm", None),
                kspace_xarray=kspace_xarray,
            )
        try:
            if handler.n_slices > 1:
                kspace_xarray = _reorder_kspace_slices(
                    acquisition_slice_order=handler.slice_order,
                    kspace_xarray=kspace_xarray,
                )

        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"\n Exp: {handler.expnum}: Could not set slice order {handler.slice_order} for xarray DataArray, shape data: {data.shape}, dims: {dims}, coords: {coords}\n"
            ) from exc

        kspace_xarray.attrs["patient_pos"] = handler.patient_pos
        kspace_xarray.attrs["slice_orientation"] = handler.slice_orientation
        kspace_xarray.attrs["read_orientation"] = handler.read_orientation
        kspace_xarray.attrs["num_image_fov_shift"] = 0
        kspace_xarray.attrs["signal_domain"] = "temporal"  # or "spectral"
        kspace_xarray.attrs["domain"] = "kspace"
        kspace_xarray.attrs["read_direction"] = handler.acquisition_matrix[0]
        kspace_xarray.attrs["phase_direction"] = handler.acquisition_matrix[1]
        kspace_xarray.attrs["slice_direction"] = handler.acquisition_matrix[2]
        kspace_xarray.attrs["slice_thickness_mm"] = handler.slice_thickness_mm
        slice_geometry = getattr(handler, "_slice_geometry_attributes", None)
        if slice_geometry is not None:
            attributes = slice_geometry()
            kspace_xarray.attrs.update(attributes)
            if "slice_idx" in kspace_xarray.coords:
                kspace_xarray["slice_idx"].attrs.update(attributes)
        kspace_xarray.attrs["n_slices"] = handler.n_slices
        kspace_xarray.attrs["git_commit_hash"] = handler.git_commit_hash
        kspace_xarray.attrs["spatial_dimension"] = handler.spatial_dimension  # 2D or 3D

        # those cases need to be tested because purley spectroscopy data does
        # not have those attributes.  Scans with several slice packages have
        # no RPS positions but keep their matrix and voxel size.
        rps_extent_mm = getattr(handler, "rps_extent_mm", None)
        rps_axes_mm = getattr(handler, "rps_axes_mm", None)
        has_rps_positions = rps_extent_mm is not None and rps_axes_mm is not None
        kspace_xarray = _attach_kspace_axis_geometry(
            has_rps_positions=has_rps_positions,
            kspace_xarray=kspace_xarray,
            matrix=getattr(handler, "matrix", None),
            n_slices=handler.n_slices,
            resolution_mm=getattr(handler, "resolution_mm", None),
            rps_axes_mm=rps_axes_mm,
            rps_extent_mm=rps_extent_mm,
        )

        if write_to_object:
            handler.kspace_xarray = kspace_xarray
        else:
            return kspace_xarray

    def set_data(
        self,
        kspace_xarray: xr.DataArray,
        axes_names: list,
        handler=None,
        write_to_object: bool = False,
    ):
        """Fourier-transform k-space into the image in ``2dseq`` voxel order.

        ``read_idx`` and ``phase_idx`` take the centred inverse DFT
        :func:`~hypermri.reconstruction.centred_dft.centered_ifft` with the
        scale of the forward FFT, which gives the ``2dseq`` voxel order and
        voxel positions, also for odd matrices, for every patient position
        and orientation without a reversal or roll of the data.  The 3D
        ``slice_idx`` takes
        :func:`~hypermri.reconstruction.centred_dft.slice_encoding_to_image`,
        the forward FFT with the ``2dseq`` slice centre at ``(n - 1) / 2``.
        Other axes, such as the spectroscopic ``acquisition_idx``, are not
        transformed.  :meth:`rearrange_data_xarray` then
        applies the slice entry of the flip table, and
        ``assign_xyz_coordinates`` attaches the ``VisuCore`` positions.
        Uniform single-package stacks use those XYZ positions for the slice
        labels, so their origin agrees with the physical voxel centres.

        Parameters
        ----------
        kspace_xarray : xarray.DataArray
            Complex k-space with the k-space centre in the centred order of
            the raw reconstruction. Dimensions are kept. Read and phase retain
            their acquired labels; uniform single-package slice labels use
            physical XYZ positions when available, otherwise acquisition labels
            remain. The input is not modified.
        axes_names : list of str
            Spatial dimensions to transform, a subset of ``read_idx``,
            ``phase_idx`` and ``slice_idx``.  ``read_idx`` and ``phase_idx`` are left
            untouched when they are not listed, as in a spectroscopic
            reconstruction whose ``phase_idx`` may hold echoes.
        handler : VendorHandler, optional
            Bruker handler with the orientation attributes of
            ``get_orientation_flip_multipliers`` and the ``VisuCore`` geometry
            of ``assign_xyz_coordinates``.  Without ``VisuCore`` geometry no
            ``x``, ``y`` and ``z`` are attached.
        write_to_object : bool, optional
            If True, store the image as ``handler.data`` and return
            None.  Default is False.

        Returns
        -------
        xarray.DataArray or None
            New complex image with the dimensions of ``kspace_xarray``, the
            ``domain`` attribute ``"spatial"`` and the scale of the forward
            FFT, or None when ``write_to_object`` is True.

        Raises
        ------
        ValueError
            If ``axes_names`` lists a dimension other than ``read_idx``,
            ``phase_idx`` and ``slice_idx``.

        Notes
        -----
        Evaluation is eager and deterministic. Input samples and metadata remain
        unchanged; only ``write_to_object=True`` stores the result on the handler.
        NumPy determines transform precision; the caller sets storage precision.

        Examples
        --------
        >>> image = strategy.set_data(
        ...     kspace, ["read_idx", "phase_idx", "slice_idx"], handler=handler
        ... )  # doctest: +SKIP
        """
        other_axes = [name for name in axes_names if name not in _SPATIAL_AXES]
        if other_axes:
            raise ValueError(
                "set_data transforms only read_idx, phase_idx and slice_idx, "
                f"but axes_names lists {other_axes}. Remove these dimensions "
                "from axes_names."
            )
        spatial_axes = list(axes_names)
        reco_data = kspace_xarray.copy(deep=True)
        for name in spatial_axes:
            if name not in reco_data.dims:
                continue
            axis = (reco_data.get_axis_num(name),)
            if name == "slice_idx":
                reco_data.data = slice_encoding_to_image(reco_data.data, axis)
            else:
                reco_data.data = centered_ifft(reco_data.data, axis, norm="forward")
        if (
            "read_idx" in axes_names
            or "phase_idx" in axes_names
            or "slice_idx" in axes_names
        ):
            reco_data.attrs["domain"] = "spatial"
        if _has_spectral_axis(handler):
            for unit, name in (("ppm", "freq_axis_ppm"), ("Hz", "freq_axis_hz")):
                axis = getattr(handler, name, None)
                if axis is not None:
                    reco_data = assign_frequency_axis(reco_data, axis, unit)
        reco_data.attrs["git_commit_hash"] = getattr(handler, "git_commit_hash", None)
        reco_data = self.rearrange_data_xarray(data=reco_data, handler=handler)
        reco_data = _attach_image_geometry(reco_data, handler)

        reco_data.attrs["domain"] = "spatial"

        # write the xarray to the object or return it:
        if write_to_object:
            handler.data = reco_data
        else:
            return reco_data

    def rearrange_data_xarray(
        self, data=None, handler=None, force_orient: bool = False
    ):
        """Apply the slice entry of the flip table to a raw reconstruction.

        Reverses ``slice_idx`` when ``get_orientation_flip_multipliers``
        gives -1 for it.  The table gives +1 for every reviewed orientation,
        so the call checks that the orientation is listed and leaves the data
        in the ``2dseq`` voxel order that :meth:`set_data` produces.
        Coordinates are not reversed, so each index coordinate keeps its
        position in mm.

        Parameters
        ----------
        data : xarray.DataArray, optional
            Reconstructed image; it is modified in place and returned.  If
            None, a copy of ``handler.data`` is used.
        handler : VendorHandler
            Bruker handler with the attributes ``patient_pos``,
            ``slice_orientation``, ``read_orientation`` and ``spatial_dimension``.
        force_orient : bool, optional
            If True, apply the slice entry again although the attribute
            ``num_rearrange`` records an earlier call.  Default is False.

        Returns
        -------
        xarray.DataArray or None
            The image with ``num_rearrange`` incremented by one, or None when
            ``num_rearrange`` is already positive and ``force_orient`` is
            False.

        Raises
        ------
        ValueError
            If the patient position or the orientation of the handler is not
            in the flip table (see ``get_orientation_flip_multipliers``).
            A handler without ``slice_orientation`` and ``read_orientation``
            is not looked up.
        """
        if data is None:
            reco_data = handler.data.copy()
        else:
            reco_data = data

        num_rearrange = reco_data.attrs.get("num_rearrange", 0)
        if num_rearrange > 0 and not force_orient:
            return None

        slice_multiplier = get_orientation_flip_multipliers(handler)[2]
        if slice_multiplier < 0 and "slice_idx" in reco_data.dims:
            axis_idx = reco_data.get_axis_num("slice_idx")
            logger.debug(
                "rearrange_data_xarray: reversing slice_idx (axis %d) for "
                "patient_pos=%s, slice_orientation=%s, read_orientation=%s",
                axis_idx,
                getattr(handler, "patient_pos", None),
                getattr(handler, "slice_orientation", None),
                getattr(handler, "read_orientation", None),
            )
            reco_data.data = np.flip(reco_data.data, axis=axis_idx)
        reco_data.attrs["num_rearrange"] = num_rearrange + 1
        return reco_data

    def postprocess_fid_file(
        self,
        values: np.ndarray = None,
        handler=None,
    ):
        """Remove acquisition-line block padding from a Bruker FID.

        Parameters
        ----------
        values : numpy.ndarray, optional
            Acquired complex samples in scanner storage order, including line
            padding. Defaults to handler.fid. Samples and dtype are not modified.
        handler : BrukerHandler
            Metadata source. ACQ_size counts real samples, GO_raw_data_format
            defines bytes per sample, and GO_block_size defines 1024-byte padding.
            NI, NR and PVM_EncNReceivers define line and receiver counts.

        Returns
        -------
        numpy.ndarray
            Unpadded line samples in (acquisition lines, complex read samples times
            receivers) order when padding removal applies. SingleExperiment with
            multiple receivers retains its acquired layout. The result can be a view.

        Raises
        ------
        KeyError
            If required acquisition metadata are absent, or the method file
            lacks ``PVM_EncNReceivers``; the latter message names the
            parameter and the scan.
        TypeError
            If required acquisition counts are missing or invalid.
        ValueError
            If the acquired samples do not fit the specified storage layout.

        Notes
        -----
        This eager, deterministic operation neither reads files nor writes handler
        state. Padding-removal failures propagate instead of using padded data.
        Storage trimming delegates to the MIT-licensed brukerapi adaptation
        documented in ``NOTICE``. It does not invoke the full Dataset reader.

        Examples
        --------
        >>> lines = strategy.postprocess_fid_file(scan.handler.fid, handler=scan.handler)
        """
        complex_file = handler.fid if values is None else values
        n_receivers = handler.method_parameter("PVM_EncNReceivers")
        if handler.acqp["ACQ_experiment_mode"] == "SingleExperiment":
            if n_receivers == 1:
                complex_file = _strip_block_padding(complex_file, handler.acqp, 1)
        else:
            complex_file = _strip_block_padding(complex_file, handler.acqp, n_receivers)
        return complex_file

    def reshape_fid_to_kspace(self, values: np.ndarray = None, handler=None):
        """
        Reshape and sort acquired Bruker samples into centred k-space.

        Parameters
        ----------
        values : np.ndarray, optional
            Complex acquired or processed samples in scanner storage order.
            If None, uses the acquired ``handler.fid``. Neither input values
            nor calculation dtype are changed.
        handler : VendorHandler
            The vendor handler containing necessary attributes for processing.

        Returns
        -------
        kspace : np.ndarray
            K-space with shape ``(1, repetitions, slices or phase-2 steps,
            phase steps, receivers, read samples)`` sized by
            ``handler.matrix``.  Phase lines are sorted by ``PVM_EncSteps1``
            (and ``PVM_EncSteps2`` for 3D) and placed at each signed step plus
            half the full target length. Missing phase positions are zero.
            Readout zero filling remains symmetric; partial-echo placement is
            not validated. Sample units and calculation dtype are preserved.

        Raises
        ------
        ValueError
            If sample count, phase factor, encoding order or matrix sizes are
            inconsistent.
        AttributeError
            If required acquired samples or encoding metadata are absent.

        Notes
        -----
        This eager, deterministic operation reads no files and writes no handler
        state. Returned samples occupy new storage; the acquired FID is unchanged.

        Examples
        --------
        >>> kspace = strategy.reshape_fid_to_kspace(scan.handler.fid, handler=scan.handler)
        """
        fid = handler.fid if values is None else values

        if getattr(handler, "spatial_dimension", None) == "2D":
            kspace = _reshape_2d_multislice_lines(
                fid,
                handler,
                line_shape=(handler.n_receivers, handler.enc_matrix[0]),
                label="generic Cartesian data",
            )
        else:
            kspace = np.reshape(
                fid,
                (
                    1,
                    handler.n_repetitions,  # Number of repetitions
                    handler.enc_matrix[2],  # Number of 2nd phase encoding steps
                    handler.enc_matrix[1],  # Number of phase encoding steps
                    handler.n_receivers,  # Number of receiver channels
                    handler.enc_matrix[0],  # Number of readout points
                ),
            )

        return _fill_encoded_kspace(kspace, handler, read_axis=5)

    def _get_phase_encode_order(self, handler):
        """
        Get the phase encoding order from the handler.

        Parameters
        ----------
        handler : VendorHandler
            The handler containing the phase encoding order.

        Returns
        -------
        list
            The phase encoding order.

        Raises
        ------
        KeyError
            If the method file lacks ``PVM_EncSteps1``, or ``PVM_EncSteps2``
            for a 3D scan; the message names the parameter and the scan.
        """
        order_ind = handler.method_parameter("PVM_EncSteps1")
        order = np.argsort(order_ind)
        if handler.spatial_dimension == "3D":
            order_ind_2 = handler.method_parameter("PVM_EncSteps2")
            order_2 = np.argsort(order_ind_2)
        else:
            order_2 = None

        return order, order_2


class BrukerMGERawRecoStrategy(BrukerRawRecoStrategy):
    """
    Reconstruction strategy for Bruker MGE raw data.
    """

    accepted_options: ClassVar[frozenset[str]] = (
        BrukerRawRecoStrategy.accepted_options - {"fov_shift_source"}
    )

    def reco_rawdata(
        self,
        handler=None,
        postprocess_fid=True,
        reshape_fid=True,
        set_kspace_xarray=True,
        shift_xarray=True,
        fov_shift_sign=1.0,
        set_data=True,
        db=True,
        data_precision=RAW_RECONSTRUCTION_PRECISION,
        *,
        values: np.ndarray | None = None,
        kspace_xarray: xr.DataArray | None = None,
    ):
        """Reconstruct raw MGE fid data into k-space and spatial domain arrays.

        Specialized for MGE: the echoes form the ``acquisition_idx`` dimension.
        The stages and the arguments are those of
        :meth:`BrukerRawRecoStrategy.reco_rawdata`.

        Parameters
        ----------
        handler : VendorHandler
            Metadata source for acquired samples, geometry and units. With
            ``reshape_fid=True``, ``handler.enc_matrix`` must contain the
            Cartesian lengths derived from ``PVM_EncMatrix``, as for
            :meth:`BrukerRawRecoStrategy.reco_rawdata`.
        postprocess_fid : bool
            If True, postprocess the fid data before reconstruction.
        reshape_fid : bool
            If True, reshape the fid data into k-space.
        set_kspace_xarray : bool
            If True, create an xarray DataArray for the k-space data.
        shift_xarray : bool
            If True, shift the k-space data with a linear phase ramp by the
            FOV offsets of the scan (``shift_image``).  A failed shift raises
            ``ValueError``; the reconstruction never continues unshifted.
        fov_shift_sign : float
            Multiplier applied to the shift derived from the FOV offsets.
        set_data : bool
            If True, create an xarray DataArray for the spatial-domain data.
            If False, return the k-space xarray.
        db : bool
            If True, add the full traceback to the message of every
            ``ValueError`` raised for a failed stage.  It does not change
            whether a stage fails.
        data_precision : str, numpy type or numpy.dtype
            Precision of the calculation and of the stored result, as for
            :meth:`BrukerRawRecoStrategy.reco_rawdata`.

        values : numpy.ndarray, optional
            Explicit complex FID or sorted k-space samples; shape follows the
            selected reshaping stage. Defaults to acquired ``handler.fid``,
            also when ``postprocess_fid=False``. Inputs are never mutated.
        kspace_xarray : xarray.DataArray, optional
            Labelled k-space; required when ``set_kspace_xarray=False``. The
            array and its metadata are copied before phase shifting or spatial
            reconstruction.

        Returns
        -------
        xr.DataArray
            The spatial-domain reconstruction when ``set_data`` is True;
            otherwise the k-space xarray. Dimensions follow
            acquisition/read/phase/slice/repetition/channel, with echoes on
            acquisition and absent axes omitted. Coordinate units and direction
            follow the handler; spatial storage dtype follows ``data_precision``.

        Raises
        ------
        ValueError
            If ``PVM_EncMatrix`` is missing while ``reshape_fid=True`` (checked
            before FID preprocessing), if the handler has no FID data, if the
            k-space data are missing,
            if building the k-space array, applying the FOV shift or the
            spatial reconstruction fails.  A failed FOV shift is raised for
            ``db=True`` and ``db=False`` alike; pass ``shift_xarray=False`` to
            reconstruct without it. ``set_kspace_xarray=False`` without
            ``kspace_xarray`` is rejected.

        Notes
        -----
        The deterministic, eager stages pass intermediate arrays directly. No
        reconstruction arrays or acquisition metadata are written to the handler.
        FOV shifts increment the result's ``num_image_fov_shift`` attribute once.
        Acquired samples, explicitly supplied arrays and their metadata are kept.

        Examples
        --------
        >>> result = strategy.reco_rawdata(
        ...     handler, values=sorted_fid, postprocess_fid=False,
        ...     reshape_fid=False, shift_xarray=False
        ... )  # doctest: +SKIP
        """
        return self._reconstruct_rawdata(
            handler,
            postprocess_fid,
            reshape_fid,
            set_kspace_xarray,
            shift_xarray,
            fov_shift_sign,
            set_data,
            db,
            data_precision,
            mge=True,
            data=values,
            kspace_xarray=kspace_xarray,
        )

    def _echo_times_ms(self, handler):
        """Return the time of each acquired echo in milliseconds.

        ``EffectiveTE`` is used when it has one value per echo. Otherwise the
        echoes start at its first value (or ``handler.echo_time_ms``) and are
        spaced by ``handler.echo_spacing_ms`` or by the first ``EffectiveTE``
        difference. Without any timing metadata the echo indices are returned.
        """
        echo_spacing_ms = getattr(handler, "echo_spacing_ms", 0.0)

        # Robustly retrieve EffectiveTE (support dict and object access)
        if isinstance(handler.method, dict) or hasattr(handler.method, "get"):
            effective_te = handler.method.get("EffectiveTE", None)
        else:
            effective_te = getattr(handler.method, "EffectiveTE", None)

        if effective_te is None:
            effective_te = np.arange(handler.n_echoes)
        effective_te = np.atleast_1d(effective_te)

        if echo_spacing_ms == 0.0 and len(effective_te) > 1:
            echo_spacing_ms = float(effective_te[1] - effective_te[0])

        # Echo sampling is local to this result, independent of the readout FID.
        if len(effective_te) == handler.n_echoes:
            echo_times_ms = np.asarray(effective_te)
        else:
            te1 = (
                effective_te[0]
                if len(effective_te)
                else getattr(handler, "echo_time_ms", 0.0)
            )
            echo_times_ms = te1 + np.arange(handler.n_echoes) * echo_spacing_ms
        return echo_times_ms

    def _label_raw_kspace(self, handler, kspace):
        """Build raw k-space labels; MGE additionally prepares echo metadata."""
        dims = [
            "unused_idx",  # The '1' dimension
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "acquisition_idx",  # Mapping echo to acquisition
            "channel_idx",
            "read_idx",
        ]

        echo_times_ms = self._echo_times_ms(handler)
        echo_axes = _echo_frequency_axes(handler, echo_times_ms)

        # Create coordinates
        coords = {
            "unused_idx": np.array([0]),
            "repetition_idx": (
                "repetition_idx",
                np.arange(kspace.shape[1]),
                {"unit": "index"},
            ),
            "slice_idx": (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[2],
                {"unit": "mm"},
            ),
            "phase_idx": (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[1],
                {"unit": "mm"},
            ),
            "acquisition_idx": (
                "acquisition_idx",
                echo_times_ms,
                {"unit": "ms"},
            ),
            "channel_idx": (
                "channel_idx",
                np.arange(handler.n_receivers),
                {"unit": "index"},
            ),
            "read_idx": (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[0],
                {"unit": "mm"},
            ),
        }

        kspace_xarray = self.set_kspace_xarray(
            handler, data=kspace, dims=dims, coords=coords
        )

        kspace_xarray = kspace_xarray.assign_coords(
            acquisition_time_s=("acquisition_idx", echo_times_ms / 1000.0)
        )
        kspace_xarray["acquisition_time_s"].attrs["unit"] = "s"
        kspace_xarray.attrs["sampling_time_axis_s"] = echo_times_ms / 1000.0
        for unit, axis in echo_axes.items():
            kspace_xarray = assign_frequency_axis(kspace_xarray, axis, unit)

        # Squeeze out the unused dimension
        if "unused_idx" in kspace_xarray.dims:
            kspace_xarray = kspace_xarray.squeeze("unused_idx")

        return kspace_xarray

    def reshape_fid_to_kspace(self, values=None, handler=None):
        """Reshape an MGE FID into k-space with an echo dimension.

        2D multislice MGE acquires, from the outermost loop inwards, phase
        segments, slices, the ``ACQ_phase_factor`` phase lines of a segment,
        echoes, receivers and read samples; the lines are sorted into
        slice-major phase order. 3D MGE acquires partitions, phase lines,
        echoes, receivers and read samples. For ``EchoAcqMode=allEchoes``
        (bipolar readout) the read order of echoes 1, 3, 5, ... is reversed
        so that every echo has the read direction of the first echo.

        Parameters
        ----------
        values : np.ndarray, optional
            The fid data to be reshaped. If None, uses the acquired
            ``handler.fid``.
        handler : VendorHandler
            The vendor handler with the encoding and reconstruction matrices,
            the number of echoes, receivers and repetitions, and
            ``PVM_EncSteps1``.

        Returns
        -------
        np.ndarray
            K-space with shape ``(1, repetitions, slices or phase-2 steps,
            phase steps, echoes, receivers, read samples)`` sized by
            ``handler.matrix``. Phase lines and encoded 3D partitions occupy
            each signed encoding step plus half the full target length.
            Missing phase positions are zero. Readout zero filling remains
            symmetric; partial-echo placement is not validated. Signal units
            and calculation dtype are preserved; echoes are not transformed.

        Raises
        ------
        ValueError
            If the fid does not match the encoding matrix and echo count, or
            if the reconstruction matrix is smaller than the encoding matrix.
        AttributeError
            If acquired samples or required encoding metadata are absent.

        Notes
        -----
        The eager, deterministic conversion preserves sample dtype, reads no
        files and writes no handler state. Acquired samples are not modified.

        Examples
        --------
        >>> kspace = strategy.reshape_fid_to_kspace(scan.handler.fid, handler=scan.handler)
        """
        fid = handler.fid if values is None else values
        n_echoes = int(handler.n_echoes)
        if getattr(handler, "spatial_dimension", None) == "2D":
            kspace = _reshape_2d_multislice_lines(
                fid,
                handler,
                line_shape=(n_echoes, handler.n_receivers, handler.enc_matrix[0]),
                label=f"MGE with n_echoes={n_echoes}",
            )
        else:
            try:
                kspace = np.reshape(
                    fid,
                    (
                        1,
                        handler.n_repetitions,
                        handler.enc_matrix[2],
                        handler.enc_matrix[1],
                        n_echoes,
                        handler.n_receivers,
                        handler.enc_matrix[0],
                    ),
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to reshape MGE data with n_echoes={n_echoes}: {e}"
                ) from e

        if _mge_readout_is_bipolar(handler):
            kspace = kspace.copy()
            kspace[:, :, :, :, 1::2] = kspace[:, :, :, :, 1::2, :, ::-1]
        return _fill_encoded_kspace(kspace, handler, read_axis=6)


__all__ = ["BrukerRawRecoStrategy", "BrukerMGERawRecoStrategy"]
