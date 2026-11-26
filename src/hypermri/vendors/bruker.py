"""
Base class used to read in one Bruker experiment/<expno> directory.

This module provides functionality for reading and processing Bruker MRI data.
It handles the specific directory structure and file formats used by Bruker
ParaVision systems.

Notes
-----
Have a look at the ParaVision Manual for more information on the scan-data folder
structure.

Includes parts of functions adapted from brukerMRI package
(https://github.com/jdoepfert/brukerMRI).
The parameter readers are modified by HyperMRI contributors and now delegate
to ``vendors._jcamp``. See NOTICE and LICENSES/brukerMRI-Apache-2.0.txt for
the upstream attribution and licence.
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import xarray as xr
from brukerapi.dataset import Dataset

from hypermri.exceptions import DataLoadingError
from hypermri.mixins import StandardizedMRIMixin
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.reconstruction.precision import convert_for_storage, parse_data_precision
from hypermri.reconstruction.strategies.bruker._storage import _raw_storage_dtype
from hypermri.utils.coordinates.grids import _slice_centre_span_mm
from hypermri.utils.system.environment import get_git_commit_hash
from hypermri.utils.system.logging import LOG_MODES, debug_log, init_default_logger
from hypermri.utils.time.temporal import calc_per_image_timeaxis_s
from hypermri.vendors._jcamp import (
    _parse_paravision_array,
    _read_paravision_parameters,
)
from hypermri.vendors.base import VendorHandler
from hypermri.vendors.bruker_acquisition import bruker_to_csi_acquisition

# Set a global variable to control info message visibility
DB_MSG = True  # Change this to False to hide detailed error messages


DEBUG_FILE = "/tmp/bruker_exp_debug.log"
DEBUG_LOG = False


def debug_log_brukerexp(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


def _has_samples(data) -> bool:
    """Return whether loaded scanner data holds at least one sample.

    ``None`` holds none. An object without a length, such as a scalar or a
    0-d array (whose ``len`` raises ``TypeError``), counts as one sample.
    """
    if data is None:
        return False
    try:
        return len(data) > 0
    except TypeError:
        return True


class BrukerHandler(VendorHandler, StandardizedMRIMixin):
    """Vendor handler for Bruker ParaVision scan directories.

    The handler expects the path to a single experiment directory (the
    ``<expno>`` folder), not a processed ``pdata/<procno>`` directory. During
    initialization it reads the Bruker parameter files (typically ``method``,
    ``acqp``, ``visu_pars``, and ``pdata/1/reco``), maps their contents onto the
    canonical HyperMRI parameter schema, and optionally loads the corresponding
    raw or reconstructed data arrays.

    Parameters
    ----------
    source : str or pathlib.Path or BrukerHandler
        Single scan directory, or a handler whose directory is read again.
    load_data : bool, optional
        Load acquired arrays eagerly when True. The default reads metadata
        only and does not reconstruct data or initialize visualization.
    log_mode : str, optional
        Logging level name, for example ``"error"`` or ``"critical"``.

    Attributes
    ----------
    reco_parameter_source : str
        Scan-relative path of the active ``reco`` parameters, initially
        ``pdata/1/reco``. Successful explicit selection changes both together.
    inversion_time_ms : numpy.ndarray or int
        Inversion times in ms from ``PVM_FairTIR_Arr`` or T1 EPI ``IRTime``;
        zero when neither source is available. T1 EPI accepts a scalar, a
        nonempty vector or a single row of shape (1, N), normalized to a
        one-dimensional float array. Multi-row and higher-dimensional tables
        are rejected; the original method parameter retains its parsed shape.
    inversion_time_source : str or None
        Parameter key owning the inversion times. T1 EPI ``IRTime`` values
        are not adjusted by FAIR slice offsets.
    slice_spacing_mm : float or None
        Derived centre-to-centre slice sampling distance of a single
        spatial stack; see the property for singleton and 3D conventions.
    resolution_mm : numpy.ndarray or None
        Read, phase and slice sizes in mm, in that order. Read and phase
        are the field of view divided by the acquisition matrix. For
        single-package 2D scans, the third entry is nominal slice thickness,
        excluding gaps. Single-package 3D scans use partition size. With
        multiple packages, the third entry is NaN (unavailable); read/phase
        values and declared nominal slice thickness remain available.
        Missing thickness retains the field-of-view/matrix fallback.
        Use ``slice_spacing_mm`` for the sampling distance between centres.
    fov_mm : numpy.ndarray or None
        Read, phase and slice field of view in mm. For multiple slice packages
        the third entry is NaN: their combined slice count does not define one
        spatial stack. Select an individual package for uniform slice geometry.
        Single-voxel spectroscopy without ``PVM_Fov`` uses the voxel size
        (``PVM_VoxArrSize``) and has ``matrix`` ``[1, 1, 1]``, so ``resolution_mm``
        equals the voxel size.
    read_bandwidth_hz : float or None
        Effective readout bandwidth in Hz from ``PVM_EffSWh``. This is separate
        from the spectral bandwidth, ``bandwidth_hz``.
    PARAMETER_SOURCES : dict of str to str
        Class attribute: the ParaVision parameter each canonical parameter is
        read from, for example ``"matrix"`` to ``"PVM_Matrix"``; ``"[i]"``
        marks entry i of an array parameter. Shown next to the parameter
        names by the browser viewer. Parameters that are not read from one
        ParaVision parameter (folder number, derived axes) have no entry.

    Raises
    ------
    FileNotFoundError
        If the source is not an existing ParaVision experiment directory.
    DataLoadingError
        If parameter parsing or requested binary loading fails, including
        invalid T1 EPI inversion times. The original exception is chained.
    KeyError
        If ``log_mode`` is not one of the supported logging levels.

    Notes
    -----
    ``timestamp`` records the measurement or edit time reported by the scan
    metadata and is used for chronological comparison between experiments.
    Metadata are read eagerly and stored on the handler. Initialization
    failures propagate as loading errors. Coordinate
    properties are evaluated lazily; reading slice spacing never loads
    samples. Repeated metadata reads are deterministic for unchanged files.

    Examples
    --------
    >>> handler = BrukerHandler("study/20", load_data=False)  # doctest: +SKIP
    >>> handler.inversion_time_ms  # doctest: +SKIP
    """

    # ParaVision parameter that each canonical parameter is read from, for
    # readers who know the scanner names (for example "matrix" is
    # "PVM_Matrix"). Display only: _parse_parameters owns how each value is
    # read and converted. Several canonical names can share one source; a
    # value that only one source fills sometimes (n_echoes of EPSI from
    # SpecSize, inversion_time_ms of T1 EPI from IRTime) names its usual one.
    # "[i]" is the entry i of an array parameter.
    PARAMETER_SOURCES: ClassVar[dict[str, str]] = {
        "timestamp": "MetaTimeStamp",
        "nucleus": "PVM_Nucleus1",
        "method_type": "Method",
        "sequence_name": "Method",
        "is_spectroscopic": "PVM_EncSpectroscopy",
        "subject_name": "VisuSubjectName",
        "subject_id": "VisuSubjectId",
        "subject_position": "VisuSubjectPosition",
        "subject_type": "VisuSubjectType",
        "patient_position": "ACQ_patient_pos",
        "spatial_dimension": "PVM_SpatDimEnum",
        "n_slice_packs": "PVM_NSPacks",
        "n_slices": "PVM_SPackArrNSlices",
        "slice_thickness_mm": "PVM_SliceThick",
        "slice_gap_mm": "PVM_SPackArrSliceGap",
        "slice_orientation": "PVM_SPackArrSliceOrient",
        "slice_order": "PVM_ObjOrderList",
        "slice_offset_mm": "PVM_SPackArrSliceOffset",
        "slice_offset_mm_array": "PVM_SliceOffset",
        "read_orientation": "PVM_SPackArrReadOrient",
        "read_offset_mm": "PVM_SPackArrReadOffset",
        "phase_offset_mm": "PVM_SPackArrPhase1Offset",
        "phase_0_offset_mm": "PVM_SPackArrPhase0Offset",
        "phase_1_offset_mm": "PVM_SPackArrPhase1Offset",
        "position_matrix": "ACQ_patient_pos",
        "fov_mm": "PVM_Fov",
        "matrix": "PVM_Matrix",
        "encoding_matrix": "PVM_EncMatrix",
        "resolution_mm": "PVM_Fov / PVM_Matrix",
        "measurement_time": "ACQ_time",
        "center_ppm": "PVM_FrqWorkPpm",
        "n_receivers": "PVM_EncNReceivers",
        "n_repetitions": "PVM_NRepetitions",
        "n_averages": "PVM_NAverages",
        "n_echoes": "PVM_NEchoImages",
        "n_acquisition": "PVM_SpecMatrix",
        "n_voxels": "PVM_NVoxels",
        "inversion_time_ms": "PVM_FairTIR_Arr",
        "rare_factor": "PVM_RareFactor",
        "scan_duration_s": "PVM_ScanTime",
        "read_bandwidth_hz": "PVM_EffSWh",
        "bandwidth_hz": "PVM_SpecSWH",
        "bandwidth_ppm": "PVM_SpecSW",
        "echo_time_ms": "PVM_EchoTime",
        "repetition_time_ms": "PVM_RepetitionTime",
        "dwell_time_s": "PVM_DigDw",
        "fat_water_separation": "PVM_FwsOnOff",
        "b_values": "PVM_DwEffBval",
        "echo_spacing_ms": "EchoSpacing",
        "effective_te_ms": "EffectiveTE",
        "reference_frequency_mhz": "PVM_FrqRef",
        "b0_tesla": "PVM_FrqRef",
        "coil_receive": "VisuCoilReceiveName",
        "coil_transmit": "VisuCoilTransmitName",
        "receiver_gain": "RG",
        "acquisition_matrix": "ACQ_grad_matrix",
        "rf_pulse_params": "ExcPulse1",
        "rfpulse_duration_ms": "ExcPulse1[0]",
        "rfpulse_bandwidth_hz": "ExcPulse1[1]",
        "flipangle_deg": "ExcPulse1[2]",
        "rfpulse_is_calculated": "ExcPulse1[3]",
        "rfpulse_sharpness": "ExcPulse1[4]",
        "rfpulse_bwfac_hzms": "ExcPulse1[5]",
        "rfpulse_sint": "ExcPulse1[6]",
        "rfpulse_pint": "ExcPulse1[7]",
        "rfpulse_type": "ExcPulse1[8]",
        "rfpulse_rpfac_percent": "ExcPulse1[9]",
        "rfpulse_peakpow_w": "ExcPulse1[10]",
        "rfpulse_shape": "ExcPulse1Enum",
        "rfpulse_refpow_w": "PVM_RefPowCh1",
        "ref_pulse_params": "RefPulse1",
        "refpulse_duration_ms": "RefPulse1[0]",
        "refpulse_bandwidth_hz": "RefPulse1[1]",
        "refpulse_flipangle_deg": "RefPulse1[2]",
        "refpulse_is_calculated": "RefPulse1[3]",
        "refpulse_sharpness": "RefPulse1[4]",
        "refpulse_bwfac_hzms": "RefPulse1[5]",
        "refpulse_sint": "RefPulse1[6]",
        "refpulse_pint": "RefPulse1[7]",
        "refpulse_type": "RefPulse1[8]",
        "refpulse_rpfac_percent": "RefPulse1[9]",
        "refpulse_peakpow_w": "RefPulse1[10]",
        "refpulse_shape": "RefPulse1Enum",
        "refpulse_refpow_w": "PVM_RefPowCh1",
    }

    @property
    def slice_spacing_mm(self) -> float | None:
        """Return the sampling distance of one spatial slice stack in mm.

        Returns
        -------
        float or None
            Positive distance between consecutive 2D slice centres: nominal
            thickness plus gap for multiple slices, thickness for one slice.
            For 3D acquisitions this is the partition spacing, the slice FOV
            divided by the third matrix entry, not the excitation slab
            thickness. None for multiple packages, non-spatial data or
            missing or nonpositive spatial geometry.

        Notes
        -----
        Uses the same centre-span calculation as the RPS axes. The scalar
        is computed on access without caching, input mutation, sample reads
        or reconstruction. Multiple packages have no shared scalar.

        Examples
        --------
        A single-package scan with five 1 mm slices and 4 mm gaps:

        >>> scan.handler.slice_spacing_mm  # doctest: +SKIP
        5.0
        """
        if self.spatial_dimension not in ("2D", "3D") or self.n_slice_packs != 1:
            return None
        if self.fov_mm is None or self.matrix is None:
            return None
        fov = np.atleast_1d(self.fov_mm)
        matrix = np.atleast_1d(self.matrix)
        if fov.size < 3 or matrix.size < 3 or matrix[2] <= 0:
            return None
        spacing = _slice_centre_span_mm(self, float(fov[2]), matrix) / matrix[2]
        return float(spacing) if np.isfinite(spacing) and spacing > 0 else None

    def _slice_geometry_attributes(self) -> dict[str, float]:
        """Return scalar slice geometry for reconstructed arrays in mm.

        Array thickness denotes a voxel's support, so a 3D partition uses
        its sampling distance rather than the acquisition slab thickness.
        Multiple 2D packages retain a supported nominal thickness but no
        scalar centre spacing or global slice resolution.
        """
        spacing = self.slice_spacing_mm
        attributes = {} if spacing is None else {"slice_spacing_mm": spacing}
        thickness = (
            self.slice_thickness_mm if self.spatial_dimension == "2D" else spacing
        )
        if thickness is not None and np.isfinite(thickness) and thickness > 0:
            attributes["slice_thickness_mm"] = float(thickness)
        return attributes

    @property
    def id(self) -> str:
        return self.expnum

    @staticmethod
    def _is_spectroscopic_acquisition(
        sequence_name: Any,
        encoding_flag: Any,
    ) -> bool:
        """Recognize spectroscopy methods whose ParaVision flag is incomplete.

        Stock Bruker CSI reports ``PVM_EncSpectroscopy=No`` because it uses
        spatial phase encoding, even though every encoded voxel contains an
        FID. Treat CSI method names consistently with the sequence registry.
        """
        return encoding_flag != "No" or "csi" in str(sequence_name).casefold()

    @property
    def type(self) -> str:
        return self.sequence_name

    @property
    def has_dicom_reco(self) -> bool:
        """Check if any reconstruction has DICOM files available."""
        if not hasattr(self, "recon_names") or not self.recon_names:
            return False

        for recon in self.recon_names:
            pdata_dir = self.path / "pdata" / str(recon)
            if not pdata_dir.is_dir():
                continue
            dicom_dirs = [pdata_dir / "dicom"]
            dicom_dirs.extend(
                path
                for path in pdata_dir.rglob("*")
                if path.is_dir()
                and "dicom" in path.name.lower()
                and path != pdata_dir / "dicom"
            )
            for dicom_dir in dicom_dirs:
                if not dicom_dir.is_dir():
                    continue
                try:
                    if any(
                        f.is_file() and not f.name.startswith(".")
                        for f in dicom_dir.iterdir()
                    ):
                        return True
                except (OSError, PermissionError):
                    continue
        return False

    @property
    def has_fid_data(self) -> bool:
        """Return whether a non-empty ``fid``/``ser`` source is available."""
        data = getattr(self, "fid", None)
        if data is not None and getattr(data, "size", 0) > 0:
            return True

        scan_path = Path(self.path)
        for filename in ("fid", "ser"):
            file_path = scan_path / filename
            try:
                if file_path.is_file() and file_path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    @property
    def has_raw_data(self) -> bool:
        """Return whether any supported Bruker raw-data source is available."""
        if self.has_fid_data:
            return True

        rawdata = getattr(self, "rawdatajob0", None)
        if rawdata is not None and getattr(rawdata, "size", 0) > 0:
            return True

        rawdata_path = Path(self.path) / "rawdata.job0"
        try:
            return rawdata_path.is_file() and rawdata_path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _slice_pack_counts(value: Any) -> tuple[int, ...]:
        """Return the positive slice counts declared for all Bruker packs.

        ``PVM_SPackArrNSlices`` is a scalar for a single slice pack and an
        array for a multi-pack acquisition.  ``n_slices`` describes the full
        acquisition, so selecting only the first array element loses raw-data
        objects and makes the reconstruction shape inconsistent with
        ``ACQ_size``/``NI``.
        """
        if value is None:
            return ()

        try:
            values = np.asarray(value).reshape(-1)
        except (TypeError, ValueError):
            values = np.asarray([value])

        counts = []
        for item in values:
            try:
                count = int(item)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts.append(count)
        return tuple(counts)

    @staticmethod
    def _first_parameter_value(value: Any, default: Any = 0) -> Any:
        """Return the first scalar value without coupling unrelated fields."""
        if value is None:
            return default
        try:
            values = np.asarray(value).reshape(-1)
        except (TypeError, ValueError):
            return value
        return values[0] if values.size else default

    @staticmethod
    def _reco_transposition_frame_axes(
        reco_transposition: Any, frame_dim_descriptions: Any
    ) -> tuple[int, int] | None:
        """Return the ``2dseq`` frame axes that ``RECO_transposition`` swaps.

        ParaVision stores the dimensions of each ``2dseq`` frame in
        ``VisuCoreDimDesc`` order, fastest first.  ``RECO_transposition``
        states per frame which two of them were swapped before storage:
        0 none, 1 the first and second, 2 the second and third, and 3 the
        first and third.  Only the swap of the first two spatial frame
        dimensions, read and phase, is supported.  When some frames are
        transposed and others are not, the returned axes are those of the
        transposed frames; :meth:`undo_reco_transposition` swaps them only
        in these frames.

        Parameters
        ----------
        reco_transposition : int, sequence of int or None
            ``RECO_transposition`` of the ``reco`` file, one value per frame.
            ``None`` means that no frame is transposed.
        frame_dim_descriptions : str, sequence of str or None
            ``VisuCoreDimDesc`` of ``visu_pars``: ``"spatial"`` or
            ``"spectroscopic"`` for each frame dimension in storage order.

        Returns
        -------
        tuple of int or None
            The zero-based array axes of the two swapped frame dimensions in
            ascending order, or ``None`` when no frame is transposed.

        Raises
        ------
        ValueError
            If ``RECO_transposition`` is not an integer array, uses more than
            one nonzero value or a value outside 0 to 3, or swaps frame
            dimensions other than the first two spatial ones of
            ``VisuCoreDimDesc``, or if ``VisuCoreDimDesc`` is missing for a
            transposed scan.

        Examples
        --------
        A 2D image with phase stored before read, and a CSI scan with the
        spectral dimension first:

        >>> BrukerHandler._reco_transposition_frame_axes(
        ...     [1, 1], ["spatial", "spatial"]
        ... )
        (0, 1)
        >>> BrukerHandler._reco_transposition_frame_axes(
        ...     [2], ["spectroscopic", "spatial", "spatial"]
        ... )
        (1, 2)
        """
        if reco_transposition is None:
            return None
        try:
            values = np.asarray(reco_transposition, dtype=int).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"RECO_transposition {reco_transposition!r} in the reco file is "
                "not an integer array; repair or re-export the reco file."
            ) from error
        nonzero = sorted({int(value) for value in values if value != 0})
        if not nonzero:
            return None
        if len(nonzero) > 1 or nonzero[0] not in (1, 2, 3):
            raise ValueError(
                f"RECO_transposition values {nonzero} are not supported; only "
                "one value from 1 to 3 is allowed for the transposed frames. "
                "Use the raw reconstruction for this scan."
            )
        value = nonzero[0]
        swapped_axes = {1: (0, 1), 2: (1, 2), 3: (0, 2)}[value]

        if frame_dim_descriptions is None:
            raise ValueError(
                f"RECO_transposition {value} needs VisuCoreDimDesc in visu_pars "
                "to identify the swapped 2dseq frame dimensions; add it from "
                "the ParaVision export or use the raw reconstruction."
            )
        if isinstance(frame_dim_descriptions, str):
            frame_dim_descriptions = frame_dim_descriptions.split()
        descriptions = tuple(str(item) for item in frame_dim_descriptions)
        spatial_axes = tuple(
            axis
            for axis, description in enumerate(descriptions)
            if description == "spatial"
        )
        if swapped_axes != spatial_axes[:2]:
            raise ValueError(
                f"RECO_transposition {value} swaps the 2dseq frame dimensions "
                f"{swapped_axes} of VisuCoreDimDesc {descriptions}; only the swap "
                "of the first two spatial frame dimensions (read and phase) is "
                "supported. Use the raw reconstruction for this scan."
            )
        return swapped_axes

    @property
    def transpose_read_phase(self) -> bool:
        """Whether the ``2dseq`` frames of ``pdata/1`` store phase before read.

        Derived from ``RECO_transposition`` of ``pdata/1/reco`` and
        ``VisuCoreDimDesc`` of the scan-level ``visu_pars`` as read by
        :meth:`load_parameters`.  :meth:`undo_reco_transposition` swaps the
        two axes back before the ``2dseq`` loaders and the ``2dseq``
        reconstruction label them; for another reconstruction it reads the
        parameters of that ``pdata`` folder instead (see
        :meth:`load_reconstruction_parameters`).

        Returns
        -------
        bool
            True when ParaVision swapped the first two spatial frame
            dimensions of at least one frame before storage.

        Raises
        ------
        ValueError
            If ``RECO_transposition`` swaps other frame dimensions, which the
            ``2dseq`` loaders do not support.
        """
        return (
            self._reco_transposition_frame_axes(*self._reco_transposition_parameters)
            is not None
        )

    @property
    def reco_transposition(self) -> np.ndarray | None:
        """``RECO_transposition`` of ``pdata/1/reco``, one value per frame.

        ParaVision states for each ``2dseq`` frame which two frame dimensions
        it swapped before storage (see :meth:`_reco_transposition_frame_axes`).
        For 2D images, 0 stores read along the DICOM columns and 1 along the
        rows.  :func:`~hypermri.utils.coordinates.grids.get_xyz_grid_from_visu`
        reads these values to place read and phase of each frame, also when
        another reconstruction is loaded.

        Returns
        -------
        numpy.ndarray or None
            One-dimensional array with one value per ``2dseq`` frame in
            storage order, as read by :meth:`load_parameters`, or ``None``
            when the reco file has no ``RECO_transposition``.  Scans with
            echoes or repetitions have more frames than slice positions.
        """
        parameters = getattr(self, "_reco_transposition_parameters", (None, None))
        if parameters[0] is None:
            return None
        return np.asarray(parameters[0]).reshape(-1)

    @property
    def slice_pack_offsets_mm(self) -> np.ndarray:
        """Read, phase and slice offsets of every slice package in mm.

        ParaVision positions each slice package on its own, as in the
        three-plane localizers.  ``PVM_SPackArrReadOffset``,
        ``PVM_SPackArrPhase1Offset`` and ``PVM_SPackArrSliceOffset`` of the
        method file hold one offset per package along the read, phase and
        slice direction of that package.  Row 0 holds the values of
        ``read_offset_mm``, ``phase_offset_mm`` and ``slice_offset_mm``.
        :func:`~hypermri.utils.coordinates.image_shifts.shift_image` shifts
        the slices of each package of a raw reconstruction by its row.

        Returns
        -------
        numpy.ndarray
            ``float64`` array of shape ``(n_packages, 3)``: one row per slice
            package in the order of ``slices_per_pack``, with the read, phase
            and slice offset in this order.  ``n_packages`` is the length of
            ``slices_per_pack``, or 1 when it is empty.  A missing parameter
            is read as 0 for every package, as for ``read_offset_mm``.

        Raises
        ------
        ValueError
            If a parameter holds another number of values than there are
            slice packages.
        """
        method = getattr(self, "method", None) or {}
        n_packages = max(len(getattr(self, "slices_per_pack", ())), 1)
        columns = []
        for name in (
            "PVM_SPackArrReadOffset",
            "PVM_SPackArrPhase1Offset",
            "PVM_SPackArrSliceOffset",
        ):
            value = method.get(name)
            if value is None:
                columns.append(np.zeros(n_packages))
                continue
            values = np.asarray(value, dtype=float).reshape(-1)
            if values.size != n_packages:
                raise ValueError(
                    f"{name} in the method file holds {values.size} values for "
                    f"{n_packages} slice packages (PVM_SPackArrNSlices); repair "
                    "or re-export the method file with one value per package."
                )
            columns.append(values)
        return np.stack(columns, axis=1)

    def load_reconstruction_parameters(self, recon_num: int = 1) -> tuple[dict, dict]:
        """Return the ``reco`` and ``visu_pars`` parameters of one reconstruction.

        Every ``pdata/<recon_num>`` folder holds the ``reco`` and
        ``visu_pars`` files that describe its own ``2dseq``: for example
        ``RECO_transposition``, ``RecoNumOutputChan``, ``VisuCoreDimDesc``
        and ``VisuFGOrderDesc``.  A parameter map computed by ParaVision in
        ``pdata/2`` can store its frames in another order than the images
        in ``pdata/1``.  :meth:`load_2dseq_file`,
        :meth:`undo_reco_transposition` and the ``2dseq`` reconstruction
        read these parameters here instead of the ``reco`` and ``visu``
        attributes, which hold ``pdata/1/reco`` and the scan-level
        ``visu_pars``.

        Parameters
        ----------
        recon_num : int, optional
            Number of the reconstruction folder ``pdata/<recon_num>``,
            starting at 1.  Default is 1.

        Returns
        -------
        reco : dict
            Parameters of ``pdata/<recon_num>/reco`` by JCAMP name, or an
            empty dict when the file is missing or empty.
        visu_pars : dict
            Parameters of ``pdata/<recon_num>/visu_pars`` by JCAMP name, or
            an empty dict when the file is missing or empty.

        Notes
        -----
        The files of each reconstruction are parsed once per handler and
        cached.  Each call returns new shallow copies of the cached dicts,
        so adding or replacing keys does not change the cache.  The values,
        such as numpy arrays, are shared and must not be modified in place.
        """
        recon_num = int(recon_num)
        parameters = self._reconstruction_parameters_cache.get(recon_num)
        if parameters is None:
            pdata = os.path.join("pdata", str(recon_num))
            parameters = (
                self._read_param_file(os.path.join(pdata, "reco")),
                self._read_param_file(os.path.join(pdata, "visu_pars")),
            )
            self._reconstruction_parameters_cache[recon_num] = parameters
        return dict(parameters[0]), dict(parameters[1])

    def undo_reco_transposition(
        self, seq2d_data: np.ndarray | None, recon_num: int = 1
    ) -> np.ndarray | None:
        """Return ``2dseq`` data with the ``RECO_transposition`` swap undone.

        :meth:`load_data`, the fallback of :meth:`load_reconstruction` and
        the ``2dseq`` reconstruction call this method on the array of
        :meth:`load_2dseq_file` before they label the axes, so that the
        spatial frame axes are ordered read, phase as ParaVision acquired
        them.  The swap follows ``RECO_transposition`` of
        ``pdata/<recon_num>/reco`` and ``VisuCoreDimDesc`` of
        ``pdata/<recon_num>/visu_pars`` (see
        :meth:`load_reconstruction_parameters`), the files that describe
        this ``2dseq``.
        One value, or one nonzero value for every frame, swaps the axes of
        all frames.  When the values differ between frames, as for the
        slice packages of a localizer, only the frames with the nonzero
        value are swapped.  Frame ``k`` is the ``k``-th entry of the frame
        groups in storage order, fastest frame group first, and takes the
        ``k``-th value.

        Parameters
        ----------
        seq2d_data : numpy.ndarray or None
            ``2dseq`` array in storage order: the frame dimensions in
            ``VisuCoreDimDesc`` order, followed by the frame groups.  It is
            not modified. None and empty arrays represent unavailable scanner
            data and are returned without reading or validating axis metadata.
        recon_num : int, optional
            Reconstruction folder ``pdata/<recon_num>`` that ``seq2d_data``
            was loaded from.  Default is 1.

        Returns
        -------
        numpy.ndarray or None
            ``seq2d_data`` with the two frame axes of
            :meth:`_reco_transposition_frame_axes` exchanged: a view when
            every frame is transposed, a new array of the same shape and
            dtype when only some frames are, and ``seq2d_data`` itself when
            no frame is transposed, it is None, or it is empty. Present data
            retain their dtype and scanner intensity units.

        Raises
        ------
        ValueError
            If ``RECO_transposition`` swaps frame dimensions other than read
            and phase (see :meth:`_reco_transposition_frame_axes`), or if
            its values differ between frames and either their number differs
            from the number of frames of ``seq2d_data`` or the swapped axes
            differ in size, so that the frames would not fit one array.

        Examples
        --------
        A localizer whose first package of 7 frames is not transposed and
        whose other 14 frames are, with 128 by 128 voxels per frame:

        >>> handler.reco_transposition  # doctest: +SKIP
        array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> handler.undo_reco_transposition(
        ...     handler.load_2dseq_file()
        ... ).shape  # doctest: +SKIP
        (128, 128, 21)

        A second reconstruction is undone with its own parameters:

        >>> handler.undo_reco_transposition(
        ...     handler.load_2dseq_file(recon_num=2), recon_num=2
        ... )  # doctest: +SKIP
        """
        if seq2d_data is None or seq2d_data.size == 0:
            return seq2d_data
        reco, visu_pars = self.load_reconstruction_parameters(recon_num)
        parameters = (
            reco.get("RECO_transposition"),
            visu_pars.get("VisuCoreDimDesc"),
        )
        swapped_axes = self._reco_transposition_frame_axes(*parameters)
        if swapped_axes is None:
            return seq2d_data
        transposed = np.asarray(parameters[0], dtype=int).reshape(-1) != 0
        if transposed.all():
            return np.swapaxes(seq2d_data, *swapped_axes)
        return self._undo_reco_transposition_per_frame(
            seq2d_data, swapped_axes, transposed, parameters[1]
        )

    @staticmethod
    def _undo_reco_transposition_per_frame(
        seq2d_data: np.ndarray,
        swapped_axes: tuple[int, int],
        transposed: np.ndarray,
        frame_dim_descriptions: Any,
    ) -> np.ndarray:
        """Swap ``swapped_axes`` in the frames where ``transposed`` is True.

        The frame groups of ``seq2d_data`` are flattened in storage order,
        fastest first, so that frame ``k`` takes ``transposed[k]``.  Returns
        a new array of the shape of ``seq2d_data``; raises ``ValueError``
        when ``transposed`` has not one entry per frame or the swapped axes
        differ in size.
        """
        if isinstance(frame_dim_descriptions, str):
            frame_dim_descriptions = frame_dim_descriptions.split()
        frame_shape = seq2d_data.shape[: len(frame_dim_descriptions)]
        n_frames = int(np.prod(seq2d_data.shape[len(frame_shape) :]))
        if transposed.size != n_frames:
            raise ValueError(
                f"RECO_transposition has {transposed.size} values that differ "
                f"between frames, but the 2dseq data has {n_frames} frames; "
                "undoing the transposition per frame needs one value per "
                "frame. Use the raw reconstruction for this scan."
            )
        sizes = tuple(frame_shape[axis] for axis in swapped_axes)
        if sizes[0] != sizes[1]:
            raise ValueError(
                "RECO_transposition swaps read and phase only in some frames, "
                f"and the swapped 2dseq frame dimensions {swapped_axes} have "
                f"the sizes {sizes}, so the frames do not fit one array. Use "
                "the raw reconstruction for this scan."
            )
        frames = seq2d_data.reshape(frame_shape + (n_frames,), order="F").copy()
        frames[..., transposed] = np.swapaxes(frames[..., transposed], *swapped_axes)
        return frames.reshape(seq2d_data.shape, order="F")

    def __init__(self, source, load_data: bool = False, log_mode: str = "error"):
        """Read the parameter files of one ParaVision scan folder.

        Parameters
        ----------
        source : str, pathlib.Path or BrukerHandler
            The scan folder that holds ``method`` and ``acqp``, or a handler
            whose folder is reused.
        load_data : bool
            If true, also read the ``fid`` and ``2dseq`` payloads during
            construction.  If false, only the parameter files are read and the
            payloads are read on demand.
        log_mode : str
            Level of the module logger, one of the keys of
            ``hypermri.utils.system.logging.LOG_MODES`` such as ``"error"``
            or ``"debug"``.

        Raises
        ------
        FileNotFoundError
            If ``source`` does not exist or is not a ParaVision scan folder.
        DataLoadingError
            If the parameter files or, with ``load_data``, the payloads cannot
            be read.  The original exception is chained as ``__cause__``.
        """
        # Initialize mixins first
        super().__init__()
        self._dicom_data_cache: dict[tuple[int, str | None], xr.DataArray] = {}
        self._reconstruction_parameters_cache: dict[int, tuple[dict, dict]] = {}

        logger.setLevel(LOG_MODES[log_mode])
        exp_folder_path = (
            source if not isinstance(source, BrukerHandler) else str(source.path)
        )
        if not self.is_valid_scan_dir(exp_folder_path):
            raise FileNotFoundError(
                f"Folder {exp_folder_path} does not exist or is not a bruker experiment."
            )

        # Use standardized path handling
        self.path = exp_folder_path

        try:
            self.load_parameters(source)
        except Exception as error:
            raise DataLoadingError(
                f"Reading the parameter files of the Bruker scan "
                f"{exp_folder_path} failed ({type(error).__name__}: {error}). "
                "Check that its method and acqp files are complete ParaVision "
                "parameter files.",
                path=str(exp_folder_path),
                vendor="bruker",
            ) from error
        self._full_load_2dseq = None
        if load_data:
            try:
                self.load_data(source)
            except Exception as error:
                raise DataLoadingError(
                    f"Reading the data of the Bruker scan {exp_folder_path} "
                    f"failed ({type(error).__name__}: {error}). Load the scan "
                    "with mode='lazy' to inspect its parameters.",
                    path=str(exp_folder_path),
                    vendor="bruker",
                ) from error

        # Initialize cached frequency axes as None - will be computed on first access
        self._freq_axis_ppm = None
        self._freq_axis_hz = None

    def _parse_parameters(self, source) -> dict[str, Any]:
        if hasattr(source, "path"):
            path = source.path
        else:
            path = source

        self.path = path

        # ---------- Scan parameter files: ----------
        self.method = self._read_param_file("method")
        self.acqp = self._read_param_file("acqp")
        self.visu = self._read_param_file("visu_pars")
        self.reco = self._read_param_file(os.path.join("pdata", "1", "reco"))
        self.reco_parameter_source = "pdata/1/reco"

        # Explicitly store VisuCoreFrameType from visu_pars
        self.visu_core_frame_type = self.visu.get("VisuCoreFrameType", [])

        experiment_number = self._get_visu_experiment_number()
        self.expnum = experiment_number
        self.experiment_number = experiment_number
        self.vendor = "bruker"

        sequence_name = self._get_measurement_type()
        self.sequence_name = sequence_name

        nucleus = self.method.get("PVM_Nucleus1", None)
        if nucleus is not None and len(nucleus) >= 2:
            nucleus = nucleus[1:-1]
        self.nucleus = nucleus

        if sequence_name is not None and len(sequence_name) >= 7:
            self.method_type = "stock" if sequence_name[1:7] == "Bruker" else "custom"
        else:
            self.method_type = None

        self.is_spectroscopic = self._is_spectroscopic_acquisition(
            sequence_name,
            self.method.get("PVM_EncSpectroscopy", None),
        )
        self.n_slice_packs = self.method.get("PVM_NSPacks", 1)
        # Stock CPMG records short echo windows (PVM_DigNp of a few points,
        # PVM_DigShift of about 3) that the 70-point FID cut-off would exceed.
        is_cpmg = str(sequence_name).strip("<>").lower().split(":")[-1] == "cpmg"
        if self.is_spectroscopic and self.method_type == "stock" and not is_cpmg:
            self.cut_off = 70
        else:
            self.cut_off = 0

        self.timestamp = self.method.get("MetaTimeStamp")

        def _strip_braces(value):
            if value is None:
                return None
            if (
                isinstance(value, str)
                and len(value) >= 2
                and value[0] == value[-1] == '"'
            ):
                return value[1:-1]
            return value

        self.subject_name = _strip_braces(self.visu.get("VisuSubjectName", None))
        self.subject_id = _strip_braces(self.visu.get("VisuSubjectId", None))
        self.subject_position = self.visu.get("VisuSubjectPosition", None)
        self.subject_type = self.visu.get("VisuSubjectType", None)
        if self.subject_type == "Quadruped":
            self.subject_type = "Rodent"

        spatial_dimension = self.method.get("PVM_SpatDimEnum", None)
        if spatial_dimension is not None:
            spatial_dimension = spatial_dimension[1:-1]
        self.spatial_dimension = spatial_dimension

        self.recon_names = self._find_recons()
        self.n_receivers = self.method.get("PVM_EncNReceivers", 1)

        self.slices_per_pack = self._slice_pack_counts(
            self.method.get("PVM_SPackArrNSlices")
        )
        if self.slices_per_pack:
            self.n_slices = sum(self.slices_per_pack)
        else:
            self.n_slices = self.method.get("SliceNum", 1)

        self.slice_thickness_mm = self.method.get("PVM_SliceThick", 0)
        # ParaVision stores one gap per slice package; a scalar is one package.
        slice_gap_mm = self.method.get("PVM_SPackArrSliceGap", 0)
        try:
            self.slice_gap_mm = slice_gap_mm[0]
        except (TypeError, IndexError):
            self.slice_gap_mm = slice_gap_mm

        acquisition_matrix_raw = self.acqp.get("ACQ_grad_matrix", None)
        if acquisition_matrix_raw is not None:
            try:
                self.acquisition_matrix = acquisition_matrix_raw[0]
            except Exception:
                self.acquisition_matrix = acquisition_matrix_raw
        self.acquisition_matrix = self.acquisition_matrix

        self.patient_position = self.acqp.get("ACQ_patient_pos", None)
        self.patient_pos = self.patient_position

        self.read_orientation = self.method.get("PVM_SPackArrReadOrient", None)
        self.slice_orientation = self.method.get("PVM_SPackArrSliceOrient", None)
        self.slice_order = np.atleast_1d(self.method.get("PVM_ObjOrderList", 0))
        # PVM_FrqRef holds one 0-ppm frequency per transmit channel in MHz;
        # the first belongs to the observed nucleus PVM_Nucleus1.
        reference_frequencies_mhz = self.method.get("PVM_FrqRef", None)
        self.reference_frequency_mhz = (
            None
            if reference_frequencies_mhz is None
            else float(np.atleast_1d(reference_frequencies_mhz)[0])
        )
        self.b0_tesla = self._derive_b0_tesla()

        self.fov_mm = self.method.get("PVM_Fov", None)
        fov_is_voxel_size = False
        if self.is_spectroscopic and self.fov_mm is None:
            vox_arr_size = self.method.get("PVM_VoxArrSize", None)
            if vox_arr_size is not None:
                fov_is_voxel_size = True
                try:
                    self.fov_mm = np.array([float(f) for f in vox_arr_size[0]])
                except Exception:
                    self.fov_mm = vox_arr_size

        self.matrix = self.method.get("PVM_Matrix", None)
        if fov_is_voxel_size:
            # Single-voxel methods keep an unused imaging PVM_Matrix; the voxel
            # size is the whole field of view, so it is one voxel.
            self.matrix = np.array([1, 1, 1])
        elif self.matrix is not None:
            self.matrix = np.atleast_1d(self.matrix)
            if spatial_dimension != "3D" and len(self.matrix) == 2:
                self.matrix = np.append(self.matrix, self.n_slices)
        elif self.is_spectroscopic:
            self.matrix = np.array([1, 1, 1])

        self.encoding_matrix = self.method.get("PVM_EncMatrix", None)
        self.enc_matrix = self.encoding_matrix

        if self.fov_mm is not None:
            self.fov_mm = np.atleast_1d(self.fov_mm).astype(float, copy=True)
            if spatial_dimension != "3D" and len(self.fov_mm) == 2:
                slice_fov_mm = (
                    np.nan
                    if self.n_slice_packs != 1
                    else self.slice_thickness_mm * self.n_slices
                    + self.slice_gap_mm * (self.n_slices - 1)
                )
                self.fov_mm = np.append(self.fov_mm, slice_fov_mm)
            if self.n_slice_packs != 1 and len(self.fov_mm) >= 3:
                self.fov_mm[2] = np.nan

        self.center_ppm = self.method.get("PVM_FrqWorkPpm", [0.0, 0.0])[0]

        if spatial_dimension == "2D" and self.encoding_matrix is not None:
            self.encoding_matrix = np.append(self.encoding_matrix, self.n_slices)
            self.enc_matrix = self.encoding_matrix

        self.measurement_time = self._get_measurement_time()

        self.n_repetitions = self.method.get("PVM_NRepetitions", 1)
        self.n_averages = self.method.get("PVM_NAverages", 1)
        if sequence_name == "<Bruker:EPSI>":
            self.n_echoes = self.method.get("SpecSize", 1)
        else:
            self.n_echoes = self.method.get("PVM_NEchoImages", 1)
        self.n_acquisition = self.method.get("PVM_SpecMatrix", 1)

        self.inversion_time_ms = self.method.get("PVM_FairTIR_Arr", None)
        self.inversion_time_source = (
            "PVM_FairTIR_Arr" if self.inversion_time_ms is not None else None
        )
        if self.inversion_time_ms is None and "t1_epi" in sequence_name.lower():
            ir_time = self.method.get("IRTime")
            if ir_time is not None:
                message = (
                    "T1 EPI IRTime must be a scalar, nonempty vector or single "
                    "row (1, N) of finite, nonnegative inversion times in ms; "
                    "multi-row and higher-dimensional tables are unsupported"
                )
                try:
                    values = np.atleast_1d(np.asarray(ir_time, dtype=float))
                except (TypeError, ValueError) as exc:
                    raise ValueError(message) from exc
                if values.ndim == 2 and values.shape[0] == 1:
                    values = values[0]
                if (
                    values.ndim != 1
                    or not values.size
                    or not np.all(np.isfinite(values))
                    or np.any(values < 0)
                ):
                    raise ValueError(message)
                self.inversion_time_ms = values
                self.inversion_time_source = "IRTime"
        if self.inversion_time_ms is not None:
            self.n_inversion_time = np.size(self.inversion_time_ms)
        else:
            self.n_inversion_time = 1
            self.inversion_time_ms = 0

        if self.fov_mm is None or self.matrix is None:
            self.resolution_mm = None
        else:
            try:
                self.resolution_mm = self.fov_mm / self.matrix
                if (
                    self.spatial_dimension == "2D"
                    and self.n_slice_packs == 1
                    and len(self.resolution_mm) == 3
                    and self.slice_thickness_mm is not None
                    and np.isfinite(self.slice_thickness_mm)
                    and self.slice_thickness_mm > 0
                ):
                    self.resolution_mm[2] = self.slice_thickness_mm
            except Exception:
                self.resolution_mm = None

        self.read_offset_mm = self._first_parameter_value(
            self.method.get("PVM_SPackArrReadOffset")
        )
        self.phase_offset_mm = self._first_parameter_value(
            self.method.get("PVM_SPackArrPhase1Offset")
        )
        self.phase_0_offset_mm = self._first_parameter_value(
            self.method.get("PVM_SPackArrPhase0Offset")
        )
        self.phase_1_offset_mm = self._first_parameter_value(
            self.method.get("PVM_SPackArrPhase1Offset")
        )
        self.slice_offset_mm = self._first_parameter_value(
            self.method.get("PVM_SPackArrSliceOffset")
        )

        self.phase_0offset_mm = self.phase_0_offset_mm
        self.phase_1offset_mm = self.phase_1_offset_mm

        self.slice_offset_mm_array = self.method.get("PVM_SliceOffset", 0)

        self.coil_receive = self.visu.get("VisuCoilReceiveName", None)
        self.coil_transmit = self.visu.get("VisuCoilTransmitName", None)
        self.receiver_gain = self.acqp.get("RG", None)

        # Validated on use by transpose_read_phase, so that an unsupported
        # transposition fails the 2dseq workflows instead of the handler.
        self._reco_transposition_parameters = (
            self.reco.get("RECO_transposition", None),
            self.visu.get("VisuCoreDimDesc", None),
        )

        self.visucore_extent = self.visu.get("VisuCoreExtent", None)
        self.visucore_thickness = self.visu.get("VisuCoreFrameThickness", None)
        self.visucore_size = self.visu.get("VisuCoreSize", None)
        self.visucore_position = self.visu.get("VisuCorePosition", None)
        self.visucore_orientation = self.visu.get("VisuCoreOrientation", None)
        self.spat_resol = self.method.get("PVM_SpatResol", None)

        position_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if self.subject_type == "Material":
            # Mentioned in Bruker manual
            position_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:
            match self.patient_position:
                # Mentioned in Bruker manual
                case "Foot_Supine":
                    pass
                # Taken from ParaVision 7
                case "Head_Supine":
                    position_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                # Taken from ParaVision 7
                case "Foot_Prone":
                    position_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                # Taken from ParaVision 7
                case "Head_Prone":
                    position_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                case None:
                    pass
                case _:
                    logger.warning(
                        "Bruker scan %s: unknown ACQ_patient_pos %r; "
                        "position_matrix is the identity. Expected Head_Prone, "
                        "Head_Supine, Foot_Prone or Foot_Supine.",
                        self.path,
                        self.patient_position,
                    )

        self.position_matrix = position_matrix

        self.fat_water_separation = self.method.get("PVM_FwsOnOff", "Off")
        self.b_values = self.method.get("PVM_DwEffBval", 0.0)
        self.n_voxels = self.method.get("PVM_NVoxels", 1)
        self.rare_factor = self.method.get("PVM_RareFactor", 1)

        self.echo_spacing_ms = self.method.get("EchoSpacing", 0.0)
        self.effective_te_ms = self.method.get("EffectiveTE", 0.0)

        self.scan_duration_s = self.method.get("PVM_ScanTime", 0.0) * 1e-3
        self.read_bandwidth_hz = self.method.get("PVM_EffSWh", None)
        self.bandwidth_hz = self.method.get("PVM_SpecSWH", None)
        self.bandwidth_ppm = self.method.get("PVM_SpecSW", None)
        self.echo_time_ms = self.method.get("PVM_EchoTime", None)
        self.repetition_time_ms = self.method.get("PVM_RepetitionTime", None)

        self.frame_time_s = calc_per_image_timeaxis_s(scan=self, start_with_0=False)
        self.dwell_time_s = self.method.get("PVM_DigDw", None)
        if self.dwell_time_s is not None:
            self.dwell_time_s = self.dwell_time_s / 1000.0
        try:
            from hypermri.utils.time.temporal import calc_sampling_time_axis

            self.sampling_time_axis_s = calc_sampling_time_axis(
                dwell_time_s=self.dwell_time_s, npoints=self.n_acquisition
            )
        except Exception:
            self.sampling_time_axis_s = None

        # Excitation pulse parameters
        self.rf_pulse_params = self.method.get("ExcPulse1", None)
        if self.rf_pulse_params is not None:
            self.rfpulse_duration_ms = self.rf_pulse_params[0]
            self.rfpulse_bandwidth_hz = self.rf_pulse_params[1]
            self.flipangle_deg = self.rf_pulse_params[2]
            rfpulse_calc = self.rf_pulse_params[3]
            self.rfpulse_is_calculated = rfpulse_calc != "No"
            self.rfpulse_sharpness = self.rf_pulse_params[4]
            self.rfpulse_bwfac_hzms = self.rf_pulse_params[5]
            self.rfpulse_sint = self.rf_pulse_params[6]
            self.rfpulse_pint = self.rf_pulse_params[7]
            self.rfpulse_type = self.rf_pulse_params[8]
            self.rfpulse_rpfac_percent = self.rf_pulse_params[9]
            self.rfpulse_peakpow_w = self.rf_pulse_params[10]
            self.rfpulse_shape = (
                self.method.get("ExcPulse1Enum", None).replace("<", "").replace(">", "")
            )
            self.rfpulse_refpow_w = self.method.get("PVM_RefPowCh1", 0.0)

        # Refocusing pulse parameters
        self.ref_pulse_params = self.method.get("RefPulse1", None)
        if self.ref_pulse_params is not None:
            self.refpulse_duration_ms = self.ref_pulse_params[0]
            self.refpulse_bandwidth_hz = self.ref_pulse_params[1]
            self.refpulse_flipangle_deg = self.ref_pulse_params[2]
            refpulse_calc = self.ref_pulse_params[3]
            self.refpulse_is_calculated = refpulse_calc != "No"
            self.refpulse_sharpness = self.ref_pulse_params[4]
            self.refpulse_bwfac_hzms = self.ref_pulse_params[5]
            self.refpulse_sint = self.ref_pulse_params[6]
            self.refpulse_pint = self.ref_pulse_params[7]
            self.refpulse_type = self.ref_pulse_params[8]
            self.refpulse_rpfac_percent = self.ref_pulse_params[9]
            self.refpulse_peakpow_w = self.ref_pulse_params[10]
            self.refpulse_shape = (
                self.method.get("RefPulse1Enum", None).replace("<", "").replace(">", "")
            )
            self.refpulse_refpow_w = self.method.get("PVM_RefPowCh1", 0.0)

        self.git_commit_hash = self._get_git_commit_hash()

        return self.build_parameter_snapshot()

    def load_data(self, source):
        """
        Load the raw files, ``pdata/1/2dseq`` and DICOM files of the scan.

        Reads from :attr:`path`, eagerly, and stores the results on the
        handler: ``fid``, ``rawdatajob0``, ``seq2d``, ``dicom`` and the
        ``2dseq`` data as primary scanner data.  No reconstruction runs.

        Missing or empty ``2dseq`` files produce a warning naming the scan
        and file, leave ``seq2d`` empty, and do not require image-axis metadata.
        Independent raw-data loading continues; malformed present image data
        still raise an error.

        ``seq2d`` is the unlabelled ``2dseq`` array as ParaVision stores it,
        frame dimensions first in ``VisuCoreDimDesc`` order and frame groups
        after them, except that the swap of read and phase recorded in
        ``RECO_transposition`` is undone, so read precedes phase.

        ``2dseq`` is read once per call: the next :meth:`load_2dseq_file`
        call for ``pdata/1`` at the default ``float32`` precision, normally
        the ``2dseq`` reconstruction of a full load, receives a copy of the
        array read here instead of reading the file again.  Later calls
        read the file.

        Parameters
        ----------
        source : str, pathlib.Path or BrukerHandler
            Scan source of the :class:`~hypermri.vendors.base.VendorHandler`
            interface.  Unused; the files are read from :attr:`path`.

        Returns
        -------
        None
            Loaded arrays and scanner-data state are stored on the handler.

        Raises
        ------
        ValueError
            If ``RECO_transposition`` cannot be undone: it swaps frame
            dimensions other than read and phase, or its values differ
            between frames that do not fit one array (see
            :meth:`undo_reco_transposition`).
        OSError
            If an available raw or scanner data file cannot be read.
        """
        total_start_time = time.time()
        logger.info(f"[TIMING] Starting data loading for scan {self.expnum}")

        # load raw data into the instance
        start_time = time.time()
        self.fid = self.load_fid_file()
        fid_time = time.time() - start_time
        logger.info(f"[TIMING] FID loading took {fid_time:.3f}s")
        logger.debug(
            f"BrukerHandler.load_data: after loading FID, self.fid type: {type(self.fid)}"
        )
        logger.debug(f"BrukerHandler.load_data: handler id: {id(self)}")
        logger.debug(
            f"BrukerHandler.load_data: hasattr(self, 'fid'): {hasattr(self, 'fid')}"
        )

        start_time = time.time()
        self.rawdatajob0 = self.load_rawdatajob0_file()
        rawdata_time = time.time() - start_time
        logger.info(f"[TIMING] rawdata.job0 loading took {rawdata_time:.3f}s")

        # load reconstructed data stored in '.../pdata/1' and store using
        # standardized naming
        start_time = time.time()
        self._full_load_2dseq = None
        stored_2dseq = self.load_2dseq_file()
        self._full_load_2dseq = stored_2dseq
        seq2d_data = self.undo_reco_transposition(stored_2dseq)
        self.seq2d = seq2d_data
        seq2d_time = time.time() - start_time
        logger.info(f"[TIMING] 2dseq loading took {seq2d_time:.3f}s")

        # Check if we have valid seq2d data
        start_time = time.time()
        if _has_samples(seq2d_data):
            # Convert to xarray if it's numpy array
            if isinstance(seq2d_data, np.ndarray):
                seq2d_data = xr.DataArray(seq2d_data)
            self.set_primary_data(seq2d_data, "scanner_data")
            # Only set legacy if we have valid data
            self._legacy_seq2d = seq2d_data
        else:
            # No valid seq2d data available
            self._legacy_seq2d = None

        # kspace_array is no longer stored permanently - used only during
        # reconstruction

        # Load DICOM files if available
        self.dicom = self.load_dicom_files()

        processing_time = time.time() - start_time
        logger.info(f"[TIMING] Data processing took {processing_time:.3f}s")

        total_time = time.time() - total_start_time
        logger.info(
            f"[TIMING] Total data loading took {total_time: .3f}s for scan {self.expnum}"
        )

    def discover_dicom_series(self, recon_num: int = 1) -> list:
        """Return DICOM series found below one ``pdata`` reconstruction."""
        from hypermri.utils.io.dicom import discover_dicom_series

        pdata_dir = self.path / "pdata" / str(recon_num)
        if not pdata_dir.is_dir():
            return []
        return discover_dicom_series(pdata_dir)

    @staticmethod
    def _select_dicom_series(series: list, series_uid: str | None = None):
        if series_uid is not None:
            for item in series:
                if item.series_instance_uid == series_uid:
                    return item
            raise ValueError(f"DICOM series not found: {series_uid}")
        if not series:
            return None
        return max(
            series,
            key=lambda item: (item.number_of_instances, item.series_instance_uid),
        )

    def load_dicom_data(
        self,
        recon_num: int = 1,
        series_uid: str | None = None,
        *,
        use_cache: bool = True,
    ) -> xr.DataArray | None:
        """Load one DICOM series as an xarray DataArray."""
        from hypermri.utils.io.dicom import load_dicom_series

        cache_key = (int(recon_num), series_uid)
        if use_cache and cache_key in self._dicom_data_cache:
            return self._dicom_data_cache[cache_key]

        selected = self._select_dicom_series(
            self.discover_dicom_series(recon_num), series_uid
        )
        if selected is None:
            return None
        data = load_dicom_series(selected)
        if use_cache:
            self._dicom_data_cache[cache_key] = data
        return data

    @property
    def dicom_data(self) -> xr.DataArray | None:
        """Lazily load the largest DICOM series from ``pdata/1`` as xarray."""
        return self.load_dicom_data()

    def load_dicom_files(
        self, recon_num: int = 1, series_uid: str | None = None
    ) -> list:
        """
        Load DICOM files from pdata directory if available.

        Parameters
        ----------
        recon_num : int, optional
            Reconstruction number to load DICOMs from, by default 1.

        Returns
        -------
        list
            List of loaded pydicom datasets, sorted by instance number.
        """
        try:
            import pydicom
        except ImportError:
            logger.warning("pydicom not installed. Cannot load DICOM files.")
            return []

        selected = self._select_dicom_series(
            self.discover_dicom_series(recon_num), series_uid
        )
        if selected is None:
            logger.warning(
                "No DICOM series found below %s",
                self.path / "pdata" / str(recon_num),
            )
            return []

        dicom_files = []
        try:
            files = selected.paths
            logger.info(
                "[TIMING] Loading %d DICOM files from series %s",
                len(files),
                selected.series_instance_uid,
            )
            start_time = time.time()

            for f in files:
                try:
                    ds = pydicom.dcmread(f)
                    dicom_files.append(ds)
                except Exception as e:
                    logger.debug(f"Failed to load DICOM file {f}: {e}")

            # Sort by Instance Number if available
            dicom_files.sort(key=lambda x: getattr(x, "InstanceNumber", 0))

            logger.info(f"[TIMING] DICOM loading took {time.time() - start_time:.3f}s")

        except Exception as e:
            logger.warning(f"Error loading DICOM files: {e}")
            return []

        return dicom_files

    def load_reconstruction(self, recon_num: int = 1, set_as_primary: bool = True):
        """Load one ``pdata`` reconstruction and optionally make it the primary data.

        The ``2dseq`` of ``pdata/<recon_num>`` is labelled by the ``2dseq``
        reconstruction strategy with the ``reco`` and ``visu_pars``
        parameters of that folder (see
        :meth:`load_reconstruction_parameters`), so that a parameter map in
        ``pdata/2`` is labelled from its own frame groups and
        ``RECO_transposition``.  Use it to compare reconstructions or to
        load maps computed by ParaVision after the handler was created.

        Parameters
        ----------
        recon_num : int, optional
            Number of the reconstruction folder ``pdata/<recon_num>``,
            starting at 1.  Default is 1.
        set_as_primary : bool, optional
            If True, the result becomes the scanner data of the handler
            (``scanner_data`` and ``seq2d``) and the DICOM files of the same
            folder are loaded into ``dicom``.  If False, only the result is
            returned.  Default is True.

        Returns
        -------
        xarray.DataArray or None
            The labelled ``2dseq`` values as returned by
            :meth:`~hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy.reconstruct`,
            with ``read_idx``, ``phase_idx`` and ``slice_idx`` in this
            order. None if the ``2dseq`` file is empty or has no usable
            reconstruction layout. Scanner loading and layout errors propagate.

        Raises
        ------
        FileNotFoundError
            If ``pdata/<recon_num>`` or its ``2dseq`` file does not exist.
        ValueError
            If reconstruction parameters or frame-group metadata are invalid.
        OSError
            If scanner samples or parameter files cannot be read.

        Notes
        -----
        The handler is mutated: ``reco`` is replaced by the parameters of
        ``pdata/<recon_num>/reco`` (``set_reco_file=True``), and with
        ``set_as_primary`` also ``scanner_data``, ``seq2d`` and ``dicom``.
        ``reco_parameter_source`` records this selection. Returned scanner
        arrays snapshot it as ``bruker_reco_parameter_source`` and record the
        data folder as ``bruker_reconstruction_number``, even when full
        parameter attachment is disabled. Later raw spectroscopy also snapshots
        the selected parameters it uses. Existing ``data`` is not changed.

        Examples
        --------
        >>> from hypermri import Study  # doctest: +SKIP
        >>> scan = Study("/path/to/study")[5]  # doctest: +SKIP
        >>> scan.load_reconstruction(recon_num=2)  # doctest: +SKIP
        >>> pdata3 = scan.load_reconstruction(
        ...     recon_num=3, set_as_primary=False
        ... )  # doctest: +SKIP

        After these calls ``scan.scanner_data`` holds ``pdata/2`` and
        ``pdata3`` holds ``pdata/3``.
        """
        logger.info(f"Loading reconstruction pdata/{recon_num} for scan {self.expnum}")

        # Check if the pdata directory and 2dseq file exist before attempting to load
        pdata_dir = self.path / "pdata" / str(recon_num)
        seq2d_file = pdata_dir / "2dseq"

        if not pdata_dir.exists():
            error_msg = (
                f"WARNING: Reconstruction directory 'pdata/{recon_num}' does not exist "
                f"for scan {self.expnum} at path: {self.path}\n"
                f"Available pdata directories: {[p.name for p in (self.path / 'pdata').glob('*') if p.is_dir()]}"
            )
            logger.warning(error_msg)
            raise FileNotFoundError(error_msg)

        if not seq2d_file.exists():
            error_msg = (
                f"WARNING: Reconstruction file 'pdata/{recon_num}/2dseq' does not exist "
                f"for scan {self.expnum} at path: {seq2d_file}"
            )
            logger.warning(error_msg)
            raise FileNotFoundError(error_msg)

        if os.path.getsize(seq2d_file) == 0:
            error_msg = (
                f"WARNING: Reconstruction file 'pdata/{recon_num}/2dseq' is empty (0 bytes) "
                f"for scan {self.expnum} at path: {seq2d_file}"
            )
            logger.warning(error_msg)
            return None

        # Use the reconstruction strategy to properly load and format the data
        # This ensures coordinates and complex data are handled correctly
        try:
            from hypermri.reconstruction.strategies.bruker.seq2d import (
                Brukerseq2dRecoStrategy,
            )

            strategy = Brukerseq2dRecoStrategy()

            # Reconstruct with the specified recon_num
            # This will use _reconstruct_internal which handles all coordinate assignment
            seq2d_data = strategy.reconstruct(
                self, recon_num=recon_num, set_reco_file=True, no_reconstruction=False
            )

            if seq2d_data is None:
                error_msg = (
                    f"WARNING: No valid data found in pdata/{recon_num} for scan {self.expnum}. "
                    f"The file exists but contains no data."
                )
                logger.warning(error_msg)
                return None

        except Exception as e:
            logger.error(f"Failed to load reconstruction using strategy: {e}")
            logger.info("Falling back to basic loading method")

            # Fallback to basic loading if strategy fails
            seq2d_data = self.undo_reco_transposition(
                self.load_2dseq_file(recon_num=recon_num, set_reco_file=True),
                recon_num=recon_num,
            )

            if not _has_samples(seq2d_data):
                error_msg = (
                    f"WARNING: No valid data found in pdata/{recon_num} for scan {self.expnum}. "
                    f"The file exists but contains no data."
                )
                logger.warning(error_msg)
                return None

            # Convert to xarray if it's a numpy array (basic fallback)
            if isinstance(seq2d_data, np.ndarray):
                seq2d_data = xr.DataArray(seq2d_data)

        if (
            not self.is_spectroscopic
            and "map_idx" not in seq2d_data.dims
            and all(
                dim in seq2d_data.dims for dim in ("read_idx", "phase_idx", "slice_idx")
            )
        ):
            from hypermri.reconstruction.layout import to_imaging_layout

            seq2d_data = to_imaging_layout(
                seq2d_data,
                sampling_time_s=self.sampling_time_axis_s,
                frame_time_s=self.frame_time_s,
            )

        # Optionally set as primary data (scanner_data, not data)
        if set_as_primary:
            # Use "scanner_data" type to set scanner_data
            self.set_primary_data(seq2d_data, "scanner_data")
            self._legacy_seq2d = seq2d_data
            self.seq2d = seq2d_data

            # Load associated DICOM files
            self.dicom = self.load_dicom_files(recon_num=recon_num)

            logger.info(
                f"Set pdata/{recon_num} as primary scanner_data for scan {self.expnum}"
            )

        return seq2d_data

    def __str__(self):
        return f"{self.acqp['ACQ_scan_name'][1:-1]}"  # drop encasing arrows <>

    def __gt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp < other.timestamp

    def get(self, attribute_name: str, default=None):
        """
        Gets an attribute value, returning a default if the attribute
        does not exist. This mimics dictionary .get() behavior.

        Parameters
        ----------
        attribute_name: str
            The name of the attribute to retrieve (e.g., 'name').
        default: str
            The value to return if the attribute is not found.

        Returns
        -------
        The attribute's value or the default.

        """
        # getattr is a built-in Python function that gets an attribute from an
        # object, with an optional default value.
        return getattr(self, attribute_name, default)

    def method_parameter(self, name: str) -> Any:
        """Return one parameter of the ParaVision ``method`` file.

        Sequence and analysis code reads acquisition parameters that have no
        named handler attribute through this method, so that a missing
        parameter fails with a message that names it and the scan instead of
        a bare ``KeyError`` or a silent default.

        Parameters
        ----------
        name : str
            ParaVision parameter name without the ``##$`` prefix, for example
            ``"EchoSpacing"``.

        Returns
        -------
        Any
            The parsed value as stored in ``self.method``, unchanged; units
            are those of ParaVision for this parameter (for example ms for
            ``EchoSpacing``).

        Raises
        ------
        KeyError
            If the ``method`` file of this scan has no parameter ``name``.
            The message names the parameter, the experiment number and the
            scan folder.

        Examples
        --------
        >>> scan.handler.method_parameter("EchoSpacing")  # doctest: +SKIP
        5.0
        """
        try:
            return self.method[name]
        except KeyError:
            raise KeyError(
                f"Scan {getattr(self, 'expnum', None)} ({getattr(self, 'path', None)}): "
                f"the method file has no parameter {name!r}, which this "
                "operation needs. Check that the scan was acquired with the "
                "expected sequence."
            ) from None

    def to_csi_acquisition(self) -> CSIAcquisition:
        """Describe this CSI scan as a vendor-neutral :class:`CSIAcquisition`.

        Code that holds a handler but must not import ``vendors`` (sequences,
        visualization) reaches the description through this method. It
        delegates once to
        :func:`hypermri.vendors.bruker_acquisition.bruker_to_csi_acquisition`.

        Returns
        -------
        CSIAcquisition
            A new description built from the current parameters on every call;
            nothing is cached.

        Raises
        ------
        KeyError
            If an encoding parameter is missing from ``method``.
        ValueError
            If the method or its encoding order is not a supported CSI
            acquisition, or a value fails the :class:`CSIAcquisition`
            validation.

        Examples
        --------
        >>> acquisition = scan.handler.to_csi_acquisition()
        >>> acquisition.read_positions.size == acquisition.n_fids
        True
        """
        return bruker_to_csi_acquisition(self)

    @staticmethod
    def is_valid_scan_dir(path) -> bool:
        """Check if directory provided is a valid bruker scan directory."""
        if not os.path.isdir(path):
            return False

        content = os.listdir(path)

        if len(set(content).intersection(["pdata", "fid", "acqp", "method"])) > 0:
            if "acqp" not in content:
                return False
            if "method" not in content:
                return False
            return True
        return False

    @staticmethod
    def __is_valid_recon_dir(path) -> bool:
        """Check if directory provided is a 'pdata' reconstruction directory."""
        if os.path.isdir(path):
            content = os.listdir(path)
            if len(set(content).intersection(["2dseq"])) > 0:
                return True
        return False

    def _get_visu_experiment_number(self) -> int:
        """Try to get experiment number. Returns -1 when fails.

        First try to convert folder name into experiment number.

        In case that fails, falls back to extracting the number from the
        ACQ_scan_name (see self.__str__) using regular expressions.

        In case no experiment number is found inside the name, return -1.
        """
        # try to get ExpNum from scan folder name
        if self.path.name.isdigit():
            return int(self.path.name)

        # try to match last (E...) in scan name using regex
        match = re.match(r".*\(E(\d+)\)", str(self))

        return int(match.group(1)) if match else -1

    def _get_measurement_time(self) -> datetime | None:
        """Return the acquisition time from ``ACQ_time``, or None without it.

        A value in another format, such as that of older ParaVision versions,
        also gives None and logs a warning that names the scan and value.
        """
        date_string = self.acqp.get("ACQ_time")
        if date_string is None:
            return None
        cleaned = str(date_string).strip("<>").replace(",", ".")
        try:
            return datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            logger.warning(
                "Scan %s: ACQ_time %r is not of the form "
                "'<2024-03-01T12:00:00,000+0100>'; no measurement time is set.",
                self.path,
                date_string,
            )
            return None

    @staticmethod
    def _parse_array(current_file, line):
        """Delegate to the shared JCAMP parser, retaining the vendor dialect."""
        return _parse_paravision_array(current_file, line)

    def _read_param_file(self, file_name: str):
        """Delegate to the shared JCAMP parser, retaining the vendor dialect."""
        return _read_paravision_parameters(self.path / file_name)

    def _get_measurement_type(self) -> str:
        """Returns measurement type from method or acqp file as string"""
        # 1. Try Method from method file (most specific)
        method = self.method.get("Method")
        if method and method != "unknown":
            return method

        # 2. Try ACQ_method from acqp file (fallback for some ParaVision versions)
        acq_method = self.acqp.get("ACQ_method")
        if acq_method and acq_method != "unknown":
            return acq_method

        # 3. Try ACQ_scan_name from acqp file (least specific, but better than nothing)
        scan_name = self.acqp.get("ACQ_scan_name")
        if scan_name and scan_name != "unknown":
            return scan_name

        return "unknown"

    def _find_recons(self):
        """Find all recon-folder in pdata and return their names in a list."""
        pdata_path = self.path / "pdata"
        pdata_content = os.listdir(pdata_path)

        recon_folder_names = []

        for recon_name in pdata_content:
            if self.__is_valid_recon_dir(pdata_path / recon_name):
                recon_folder_names.append(recon_name)

        recon_folder_names.sort()

        return recon_folder_names

    # ---------------------------------------------------
    # ------------ LOAD DATA AND PARAMETERS -------------
    # ---------------------------------------------------
    def _read_raw_samples(self, file_path: Path, *, rawdata_job: bool = False):
        """Read declared scalar samples, retaining all storage-block padding."""
        acqp = self.acqp
        dtype = _raw_storage_dtype(acqp, rawdata_job=rawdata_job)
        mode = str(acqp.get("AQ_mod", "")).lower()
        if rawdata_job and mode == "qf":
            raise ValueError(
                "Real-only rawdata.job0 storage is unsupported; its complex job layout must be declared and validated."
            )
        if mode not in {"qf", "qdig", "dqd", "qsim"}:
            raise ValueError(
                f"Unsupported Bruker AQ_mod={mode!r} for {file_path}; expected qf or interleaved quadrature samples."
            )
        if file_path.stat().st_size % dtype.itemsize:
            raise ValueError(f"Incomplete {dtype} scalar sample in {file_path}.")
        raw = np.fromfile(file_path, dtype=dtype).astype(np.float64)
        if mode == "qf":
            return raw
        if raw.size % 2:
            raise ValueError(
                f"Incomplete real/imaginary pair in {file_path}: {raw.size} scalar samples."
            )
        return raw[0::2] + 1j * raw[1::2]

    def load_fid_file(self):
        """Decode the acquired ``fid`` or nonempty ``ser`` in storage order.

        Returns
        -------
        numpy.ndarray
            One-dimensional complex128 quadrature samples, or float64 real
            samples for ``AQ_mod=qf``, in unscaled scanner units. Storage
            padding is retained for the reconstruction stage. An empty array
            is returned when neither acquisition file is present and nonempty.

        Raises
        ------
        ValueError
            If format, byte order or acquisition mode is missing/unsupported,
            or the file contains an incomplete scalar or complex pair.
        OSError
            If the selected file cannot be read.

        Notes
        -----
        ``GO_raw_data_format`` supports signed int16/int32 and float32 with
        ``BYTORDA=little`` or ``big``. ParaVision 360 can instead declare
        ``ACQ_word_size``. ``qdig``, ``dqd`` and ``qsim`` are interleaved
        quadrature modes; sequential quadrature is unsupported. Each call is
        eager and deterministic, does not cache, and never mutates files or
        handler arrays. Calculation/storage precision conversion belongs to
        reconstruction, separately from this byte decoding.
        """
        for name in ("fid", "ser"):
            file_path = self.path / name
            if file_path.is_file() and file_path.stat().st_size > 0:
                return self._read_raw_samples(file_path)
        return np.array([])

    def load_rawdatajob0_file(self):
        """Decode ``rawdata.job0`` using its declared job storage format.

        Returns
        -------
        numpy.ndarray
            Flat complex128 quadrature samples in unscaled scanner units,
            retaining storage padding. Real-only job storage is unsupported. An empty
            array is returned when the file is absent or empty.

        Raises
        ------
        ValueError
            If storage declarations are unsupported or samples are incomplete.
        OSError
            If the file cannot be read.

        Notes
        -----
        The ``ACQ_ScanPipeJobSettings`` record matching ``job0`` overrides the
        FID format with ``STORE_32bit_signed`` or ``STORE_64bit_float``. Nine-field
        ``ACQ_jobs`` records select it by name; older eight-field records and
        absent job lists use index zero. Ambiguous declarations raise an error.
        Without job settings the FID format contract of :meth:`load_fid_file`
        applies. Byte order and
        acquisition mode use ``BYTORDA`` and ``AQ_mod``. Reads are eager,
        deterministic, uncached and do not mutate files or handler arrays.
        """
        file_path = self.path / "rawdata.job0"
        if file_path.is_file() and file_path.stat().st_size > 0:
            return self._read_raw_samples(file_path, rawdata_job=True)
        return np.array([])

    def load_2dseq_file(
        self, recon_num: int = 1, set_reco_file: bool = False, data_precision=np.float32
    ):
        """Load the ``2dseq`` file of one reconstruction in storage order.

        The array is read with :class:`brukerapi.dataset.Dataset`, which
        shapes it from ``pdata/<recon_num>/visu_pars`` and applies
        ``VisuCoreDataSlope`` and ``VisuCoreDataOffs``.  When
        ``VisuFGOrderDesc`` declares ``FG_COMPLEX``, the reader combines
        its real and imaginary frames along that declared axis exactly once,
        removing only the complex frame group.  The
        ``RECO_transposition`` swap is not undone; pass the result to
        :meth:`undo_reco_transposition` with the same ``recon_num``.

        Parameters
        ----------
        recon_num : int, optional
            Number of the reconstruction folder ``pdata/<recon_num>``,
            starting at 1.  Default is 1.
        set_reco_file : bool, optional
            If True, replace the ``reco`` attribute of the handler with the
            parameters of ``pdata/<recon_num>/reco``, so that later readers
            of ``reco`` see this reconstruction. Also update
            ``reco_parameter_source`` to its scan-relative path. Default False.
        data_precision : str, numpy type or numpy.dtype, optional
            Floating storage precision of the returned values:
            ``float16``, ``float32`` or ``float64``, as a name or NumPy
            type; complex data keep a complex dtype (see
            :func:`~hypermri.reconstruction.precision.convert_for_storage`).  Default
            is ``np.float32``, which holds the full finite range of common
            ParaVision image intensities.

        Returns
        -------
        numpy.ndarray
            ``2dseq`` values with the frame dimensions in
            ``VisuCoreDimDesc`` order followed by the frame groups in
            ``VisuFGOrderDesc`` order, fastest first, with ``FG_COMPLEX``
            removed. Values retain the scanner intensity units after scaling.
            An empty array when
            ``pdata/<recon_num>/2dseq`` is missing or empty.

        Raises
        ------
        ValueError
            If ``data_precision`` is not supported, or reconstruction
            parameters requested with ``set_reco_file=True`` are malformed.
        OSError
            If the data or requested reconstruction parameters cannot be read.

        Notes
        -----
        Repeated reads are deterministic and never modify the acquired file.
        A missing or empty file logs ``No seq2d data available`` at warning
        level with the scan number and requested file path. Axis metadata are
        required only when image data are present.
        Parameter files requested with ``set_reco_file=True`` are read through
        :meth:`load_reconstruction_parameters` and cached per
        reconstruction.  ``set_reco_file=True`` mutates the handler.  The
        assignment occurs only after successful parameter loading; a failure
        propagates without returning data using the previous ``reco``.
        Successful selection remains visible to subsequent readers of
        ``reco``, including raw spectroscopy and reconstruction timestamps.
        Labelled scanner and spectroscopy results snapshot the parameter
        source, ``RECO_time`` and ``RecoObjectsPerRepetition`` when available;
        later selections do not change those existing result attributes.
        The first call for ``recon_num=1`` and a ``float32`` ``data_precision``
        after :meth:`load_data` returns a copy of the array that
        :meth:`load_data` read, with the same values and dtype, instead of
        reading the file again; every other call reads the file.
        """
        data_precision = parse_data_precision(data_precision)
        start_time = time.time()

        file_path = self.path / "pdata" / str(recon_num) / "2dseq"

        if file_path.exists() and os.path.getsize(file_path) > 0:
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            logger.info(f"[TIMING] Loading 2dseq file ({file_size_mb:.1f} MB)")

            if set_reco_file:
                reco_start = time.time()
                self.reco = self.load_reconstruction_parameters(recon_num)[0]
                self.reco_parameter_source = f"pdata/{int(recon_num)}/reco"
                reco_time = time.time() - reco_start
                logger.info(f"[TIMING] Reco file loading took {reco_time: .3f}s")

            full_load_2dseq = self._take_full_load_2dseq(recon_num, data_precision)
            if full_load_2dseq is not None:
                return full_load_2dseq

            data_start = time.time()
            data = Dataset(file_path, combine_complex=True).data

            data_time = time.time() - data_start
            logger.info(f"[TIMING] Dataset loading took {data_time:.3f}s")

            # Apply data precision conversion
            precision_start = time.time()
            data = convert_for_storage(data, data_precision)
            precision_time = time.time() - precision_start
            logger.info(
                f"[TIMING] Data precision conversion took {precision_time:.3f}s"
            )

            total_time = time.time() - start_time
            logger.info(f"[TIMING] load_2dseq_file total: {total_time:.3f}s")

            return data
        else:
            logger.warning(
                "No seq2d data available for scan %s: %s is missing or empty.",
                self.expnum,
                file_path,
            )
            return np.array([])

    def _take_full_load_2dseq(self, recon_num: int, data_precision):
        """Return a copy of the ``2dseq`` array of :meth:`load_data` once.

        :meth:`load_data` reads ``pdata/1`` at the default ``float32``
        precision.  The first request for the same reconstruction and
        precision takes that array, so that a full load reads ``2dseq``
        once; the copy keeps the result independent of the handler's
        ``seq2d`` and ``scanner_data``, which may be views of it.  Returns None
        for every other request and after the array has been taken.
        """
        stored = getattr(self, "_full_load_2dseq", None)
        if stored is None or recon_num != 1 or data_precision is not np.float32:
            return None
        self._full_load_2dseq = None
        return stored.copy()

    def _get_git_commit_hash(
        self,
    ):
        """
        Get the current git commit hash of the repository containing this script.

        Returns
        -------
        git commit has: str
            The git commit hash, or an error message if not found.

        """
        try:
            script_path = Path(__file__).resolve().parent
        except NameError:
            return get_git_commit_hash(
                missing="Not a git repository (interactive mode)."
            )
        return get_git_commit_hash(
            script_path, missing="Could not find a git repository."
        )

    # def open(self, paramfile: str = "method"):
    #     """Open method file in default text editor for '.json' files."""
    #     import pprint
    #     import subprocess
    #     import sys
    #     import tempfile
    #     import time
    #
    #     tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
    #
    #     def open_file_platfrom_independent(filename):
    #         if sys.platform == "win32":
    #             os.startfile(filename)
    #         else:
    #             opener = "open" if sys.platform == "darwin" else "xdg-open"
    #             subprocess.call([opener, filename])
    #
    #     try:
    #         # print(f"Dumping temporary file here:", tmp.name)
    #         pretty_str = pprint.pformat(getattr(self, paramfile))
    #         tmp.write(pretty_str.encode("ascii"))
    #     finally:
    #         tmp.close()
    #         open_file_platfrom_independent(tmp.name)
    #         time.sleep(2)
    #         os.unlink(tmp.name)
