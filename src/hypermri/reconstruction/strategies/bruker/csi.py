"""Bruker CSI reconstruction strategies."""

import json
from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.reconstruction.precision import convert_for_storage, parse_data_precision
from hypermri.utils.coordinates.grids import assign_xyz_coordinates
from hypermri.utils.coordinates.image_shifts import shift_image

from ...axes import (
    apply_processed_axis_metadata,
    processed_sampling_axis,
    resolve_cut_off,
)
from ...centred_dft import centered_ifft
from ...csi_noise import reconstruct_csi_with_noise
from ...csi_raw import (
    DENSITY_WEIGHTED_METHODS,
    PAIRED_ORDER_METHODS,
    PVM_ENCODING_METHODS,
    PVM_ENCODING_ORDERS,
    REPEATED_POSITION_RULES,
    SCANNER_SPATIAL_METHODS,
    SUPPORTED_ENCODINGS,
    attach_csi_raw_provenance,
    sort_csi_encoding_grid,
    strip_csi_storage_padding,
)
from .._common import logger
from ._encoding import _encoding_step_positions
from .raw import BrukerRawRecoStrategy


def _encoding_order_name(value):
    """Return ``PVM_EncOrder`` as one text such as ``"LINEAR_ENC LINEAR_ENC"``.

    ParaVision stores one order per phase-encoded axis. Depending on the
    parameter reader the two entries arrive as a list or as one text.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(item) for item in np.asarray(value).ravel())
    return value


def _scanner_encoding_positions(handler):
    """Return the scanner order name and one read/phase grid position per FID.

    ``PVM_EncSteps0`` (read) and ``PVM_EncSteps1`` (phase) list the signed
    steps of one axis in the order of acquisition, for a linear order and for
    a centric order alike. Every phase step is paired with all read steps, so
    read is the inner loop. A step is relative to the matrix centre ``n // 2``.

    Raises ``ValueError`` when ``PVM_EncOrder`` is not one of
    ``PVM_ENCODING_ORDERS``; a different order on the two axes has no reference
    scan and is therefore not reconstructed. The parameters are read through
    ``handler.method_parameter``, so a missing one raises a ``KeyError`` that
    names it and the scan.
    """
    parameter = handler.method_parameter
    order = _encoding_order_name(parameter("PVM_EncOrder"))
    if order not in PVM_ENCODING_ORDERS:
        supported = ", ".join(repr(name) for name in PVM_ENCODING_ORDERS)
        raise ValueError(
            f"Unsupported CSI encoding order PVM_EncOrder={order!r} of method "
            f"{parameter('Method')!r}. Supported orders: {supported}. Acquire "
            "the scan with the same supported order on both axes."
        )
    n_read, n_phase = parameter("PVM_Matrix")[0], parameter("PVM_Matrix")[1]
    read_steps = _encoding_step_positions(
        parameter("PVM_EncSteps0"), n_read, "PVM_EncSteps0"
    )
    phase_steps = _encoding_step_positions(
        parameter("PVM_EncSteps1"), n_phase, "PVM_EncSteps1"
    )
    read_positions = np.tile(read_steps, phase_steps.size)
    phase_positions = np.repeat(phase_steps, read_steps.size)
    return order, read_positions, phase_positions


def _repeat_by_average_list(handler, read_positions, phase_positions):
    """Repeat each scanner position as often as ``AverageList`` acquired it.

    Density-weighted methods store one count per k-space position, in the
    order of acquisition, and acquire the repeats of a position one after the
    other. The returned arrays therefore name one position per acquired FID.

    Raises ``ValueError`` when ``AverageList`` does not hold one positive
    integer count per position; the message names the scan's method.
    """
    parameter = handler.method_parameter
    counts = np.asarray(parameter("AverageList"), dtype=float).reshape(-1)
    if (
        counts.size != read_positions.size
        or not np.all(np.isfinite(counts))
        or np.any(counts != np.rint(counts))
        or np.any(counts < 1)
    ):
        raise ValueError(
            f"AverageList of method {parameter('Method')!r} must hold one "
            f"positive integer count per k-space position "
            f"({read_positions.size}); it holds {counts.size} values."
        )
    counts = counts.astype(np.int64)
    return np.repeat(read_positions, counts), np.repeat(phase_positions, counts)


def _stored_pair_positions(read, phase, n_read, n_phase):
    """Return the read/phase pairs a custom method stored as grid positions.

    Arrays without a negative value are grid positions already and are
    returned unchanged. Otherwise both arrays hold signed steps relative to the
    matrix centre ``n // 2``, not to the smallest position that was acquired.
    """
    if np.min(phase) < 0 or np.min(read) < 0:
        phase = _encoding_step_positions(phase, n_phase, "CSI phase encoding")
        read = _encoding_step_positions(read, n_read, "CSI read encoding")
    return read, phase


def _csi_acquisition_plan(
    n_acquisition, n_receivers, n_repetitions, n_slices, phase_idx, read_idx
):
    """Validate paired CSI encoding positions and calculate the raw sample count.

    Encoding indices are 1D paired arrays. Each position carries acquisition
    samples for every repetition, slice and channel; invalid pairs raise ValueError.
    """
    read_idx, phase_idx = np.asarray(read_idx), np.asarray(phase_idx)
    if (
        read_idx.ndim != 1
        or phase_idx.ndim != 1
        or read_idx.size != phase_idx.size
        or not read_idx.size
    ):
        raise ValueError(
            "CSI encoding requires paired read/phase positions for every acquired FID."
        )
    n_positions = read_idx.size
    expected_fid_size = (
        n_repetitions * n_slices * n_positions * n_receivers * n_acquisition
    )

    return read_idx, phase_idx, n_positions, expected_fid_size


def _bruker_storage_record_samples(acqp, *, samples, channels):
    """Return the complex samples one padded ParaVision FID record occupies.

    Only the confirmed layout is recognised: one receiver,
    ``GO_block_size`` ``Standard_KBlock_Format``, ``GO_raw_data_format``
    ``GO_32BIT_SGN_INT``, ``BYTORDA`` ``little`` and ``ACQ_size[0]`` equal to
    twice ``samples`` (real and imaginary parts). Each record then fills whole
    1024-byte blocks of 8-byte complex samples. Every other layout returns
    None, so no padding is removed and the exact size check decides.
    """
    acquisition_size = np.atleast_1d(acqp.get("ACQ_size", []))
    if (
        channels != 1
        or acqp.get("GO_block_size") != "Standard_KBlock_Format"
        or acqp.get("GO_raw_data_format") != "GO_32BIT_SGN_INT"
        or acqp.get("BYTORDA") != "little"
        or not acquisition_size.size
        or acquisition_size[0] != 2 * samples
    ):
        return None
    return ((samples * 8 + 1023) // 1024) * 1024 // 8


def _select_csi_payload(
    data_type,
    expected_fid_size,
    data,
    n_acquisition,
    n_positions,
    n_receivers,
    n_repetitions,
    n_slices,
    read_payload,
    scan_path,
    storage_record_samples,
):
    """Choose explicit or acquired CSI samples that match the acquisition plan.

    read_payload lazily supplies fid or rawdatajob0. Storage padding is removed
    from FID records without mutating the payload. Missing/mismatched samples
    raise ValueError; returns samples, selected source and complex padding count.
    """
    storage_padding = 0
    if data is not None:
        fid, source = np.asarray(data), "input_data"
        if fid.size != expected_fid_size:
            raise ValueError(
                "Supplied CSI payload size does not match the acquisition plan; no silent fallback is allowed."
            )
    else:
        preferred = "fid" if data_type == "fid" else "job"
        candidates = (preferred, "job" if preferred == "fid" else "fid")
        fid = None
        n_raw_samples = 0
        for source in candidates:
            candidate = read_payload("fid" if source == "fid" else "rawdatajob0")
            n_raw_samples += 0 if candidate is None else np.size(candidate)
            candidate_padding = 0
            if candidate is not None and source == "fid":
                candidate, candidate_padding = strip_csi_storage_padding(
                    candidate,
                    samples=n_acquisition,
                    channels=n_receivers,
                    records=n_repetitions * n_slices * n_positions,
                    record_samples=storage_record_samples,
                )
            if candidate is not None and np.size(candidate) == expected_fid_size:
                fid = np.asarray(candidate)
                storage_padding = candidate_padding
                break
        if fid is None and n_raw_samples == 0:
            raise ValueError(
                f"Scan {scan_path} has no CSI raw data: "
                "its 'fid' (or 'ser') and 'rawdata.job0' files are missing "
                "or empty, so there is nothing to reconstruct.  A scan "
                "that was set up but not acquired has only parameter files."
            )
        if fid is None:
            raise ValueError(
                f"No CSI payload matches {expected_fid_size} samples from the explicit acquisition plan."
            )

    return fid, source, storage_padding


def _sort_csi_payload(
    enc_order,
    fid,
    n_acquisition,
    n_phase,
    n_positions,
    n_read,
    n_receivers,
    n_repetitions,
    n_slices,
    phase_idx,
    read_idx,
    repeated_positions="mean",
):
    """Reshape CSI samples and sort paired encoding positions into the grid.

    FID order is repetition/slice/position/channel/acquisition. Output uses
    acquisition/slice/read/phase/repetition/channel; duplicate positions are
    averaged or added (``repeated_positions``) and missing bins are tracked by
    acquisition_counts.
    """
    # Keep every acquired position; missing grid bins are represented by the
    # occupancy map, never by adding synthetic observations to the payload.
    data = fid.reshape(n_repetitions, n_slices, n_positions, n_receivers, n_acquisition)
    data = data.transpose(4, 1, 2, 0, 3)

    if enc_order not in SUPPORTED_ENCODINGS:
        raise NotImplementedError(
            f"CSI encoding {enc_order!r} has no verified acquisition-grid sorter."
        )
    kspace_array, acquisition_counts = sort_csi_encoding_grid(
        data,
        read_idx,
        phase_idx,
        (n_read, n_phase),
        repeated_positions=repeated_positions,
    )

    return kspace_array, acquisition_counts


def _label_csi_grid_coordinates(
    kspace_array, n_receivers, n_repetitions, rps_grid_mm, sampling_axis, sampling_unit
):
    """Order the CSI grid as acquisition/repetition/slice/phase/channel/read.

    Sampling coordinates carry sampling_unit; spatial grids use RPS mm values.
    Transposition preserves the samples and calculation precision.
    """
    kspace_array = np.transpose(kspace_array, [0, 4, 1, 3, 5, 2])
    dims = [
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    ]
    coords = {
        "acquisition_idx": (
            "acquisition_idx",
            sampling_axis,
            {"unit": sampling_unit},
        ),
        "repetition_idx": np.array(np.arange(n_repetitions)),
        "slice_idx": (
            [
                "read_idx",
                "phase_idx",
                "slice_idx",
            ],
            rps_grid_mm[2],
        ),
        "phase_idx": (
            [
                "read_idx",
                "phase_idx",
                "slice_idx",
            ],
            rps_grid_mm[1],
        ),
        "channel_idx": np.array(np.arange(n_receivers)),
        "read_idx": (
            [
                "read_idx",
                "phase_idx",
                "slice_idx",
            ],
            rps_grid_mm[0],
        ),
    }

    return kspace_array, dims, coords


def _transpose_csi_spatial_axes(kspace_xarray):
    """Copy CSI samples into acquisition/read/phase/slice/repetition/channel order.

    Additional labelled dimensions are retained at the end. Coordinates, units
    and sample dtype are preserved; the returned data do not share storage.
    """
    preferred_order = [
        "acquisition_idx",
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    ]
    transpose_order = [dim for dim in preferred_order if dim in kspace_xarray.dims] + [
        dim for dim in kspace_xarray.dims if dim not in preferred_order
    ]

    reco_data = kspace_xarray.transpose(*transpose_order).copy(deep=True)

    return reco_data


def _csi_spatial_fft_sizes(grid):
    """Return the read/phase axes of ``grid`` that the spatial transform uses.

    Axes of length 1 are left untouched; the mapping is recorded in the
    reconstruction trace.
    """
    return {
        dim: grid.sizes[dim]
        for dim in ("read_idx", "phase_idx")
        if dim in grid.dims and grid.sizes[dim] > 1
    }


def _shift_csi_fov(grid, phase_shift):
    """Return ``grid`` with the FOV phase operator applied, as labelled data.

    ``phase_shift`` multiplies a copy of the sorted k-space by the scan's
    FOV phase ramp; the input is not modified.
    """
    shifted = phase_shift(grid.copy(deep=True))
    if not isinstance(shifted, xr.DataArray):
        raise ValueError("FOV phase shift did not return labelled CSI data.")
    return shifted


def _csi_spatial_ifft(grid):
    """Return the centred inverse DFT of ``grid`` along read and phase.

    The result is a NumPy array in the dimension order of ``grid``; the
    spectroscopic ``acquisition_idx`` axis stays temporal.
    """
    values = grid.to_numpy().copy()
    for dim in _csi_spatial_fft_sizes(grid):
        values = centered_ifft(values, (grid.get_axis_num(dim),))
    return values


def _csi_fov_phase_weights(grid, phase_shift):
    """Return the read x phase FOV phase weights and the attributes they set.

    The phase operator is applied once to ones, so the same weights enter the
    signal and the noise propagation. The weights must not depend on the
    other axes (slice, repetition, channel); otherwise each group would need
    its own reconstruction operator.
    """
    shifted = _shift_csi_fov(xr.ones_like(grid, dtype=np.complex128), phase_shift)
    fixed = {dim: 0 for dim in shifted.dims if dim not in ("read_idx", "phase_idx")}
    weights = shifted.isel(fixed, drop=True).transpose("read_idx", "phase_idx")
    if not np.allclose(
        shifted.values,
        weights.broadcast_like(shifted).values,
        rtol=1e-13,
        atol=1e-13,
    ):
        raise ValueError(
            "Group-dependent FOV phase shifts require separate group reconstruction operators."
        )
    return weights.values, shifted.attrs.copy()


def _finalize_csi_spatial_result(
    data_precision,
    fft_sizes,
    fov_phase_shift,
    propagated,
    reco_data,
    spatial_mode,
    values,
):
    """Store CSI spatial values and transfer raw-grid provenance to source fields.

    Complex storage follows data_precision. Acquisition-bin masks are removed
    because they do not describe reconstructed voxels. Remaining coordinates
    and units are preserved, and the trace records FFT sizes and phase status.
    """
    reco_data = reco_data.copy(data=values)
    reco_data.data = convert_for_storage(reco_data.data, data_precision)
    reco_data.attrs["domain"] = "spatial"
    reco_data.attrs["signal_domain"] = "temporal"
    reco_data.attrs["csi_spatial_reconstruction"] = spatial_mode
    reco_data.attrs["csi_grid_domain"] = "spatial"
    reco_data.attrs["csi_domain_evidence"] = "spatial_reconstruction"
    reco_data.attrs["csi_reconstruction_trace"] = json.dumps(
        {
            "mode": spatial_mode,
            "spatial_ifft_sizes": fft_sizes,
            "fft_normalization": "numpy_backward_1_over_axis_length",
            "fft_shifts": (
                "ifftshift_before_fftshift_after_each_spatial_ifft;"
                "odd_length_half_voxel_phase_ramp_before_ifft"
            ),
            "fov_phase_shift": fov_phase_shift,
            "window": "none",
            "channel_combination": "none",
            "channel_weights": "identity",
            "output_dtype": str(reco_data.dtype),
        },
        sort_keys=True,
    )
    if propagated is not None:
        reco_data.attrs["csi_noise_id"] = propagated.attrs["noise_id"]
        reco_data.attrs["csi_noise_reconstruction_id"] = propagated.attrs[
            "reconstruction_id"
        ]
    # These masks describe raw encoding bins, not reconstructed voxels.
    reco_data = reco_data.drop_vars(
        [
            name
            for name in ("csi_acquired", "csi_acquisition_count")
            if name in reco_data.coords
        ]
    )
    for key in ("csi_raw_fingerprint", "csi_raw_trace", "csi_raw_contract"):
        if key in reco_data.attrs:
            reco_data.attrs[key.replace("csi_raw", "csi_source_raw")] = (
                reco_data.attrs.pop(key)
            )

    return reco_data


class BrukerCsiRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruction strategy for Bruker CSI data."""

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "data_type",
            "cut_off",
            "data_precision",
            "set_data",
            "append_parameters",
            "repeated_positions",
        }
    )

    @staticmethod
    def _method_name(handler) -> str:
        method = getattr(handler, "method", {}) or {}
        if isinstance(method, dict):
            value = method.get("Method")
        else:
            value = None
        if value is None:
            value = getattr(handler, "sequence_name", None) or getattr(
                handler, "type", ""
            )
        return str(value)

    @classmethod
    def _uses_scanner_spatial_grid(cls, handler) -> bool:
        """Return True for CSI methods whose sorted raw grid is already spatial."""
        return cls._method_name(handler) in SCANNER_SPATIAL_METHODS

    def reconstruct(self, handler, **reco_options) -> xr.DataArray | None:
        """Reconstruct CSI from the acquired encoding plan and temporal samples.

        Parameters
        ----------
        handler : BrukerHandler
            Acquired FID, encoding positions, receiver counts, dwell time in
            seconds and spatial geometry in mm. A missing FID is loaded once.
        **reco_options
            ``no_reconstruction`` skips work (default False); ``data_type``
            selects ``"job"`` or ``"fid"`` (default ``"job"``); ``cut_off``
            removes leading temporal points (default scan metadata); the
            sampling-time axis then starts at 0 s at the first kept sample,
            not at the delay since excitation;
            ``data_precision`` selects storage precision (default float16);
            ``set_data`` enables spatial reconstruction (default True);
            ``append_parameters`` adds acquisition provenance (default True);
            ``repeated_positions`` combines FIDs acquired more than once at
            one k-space position: ``"mean"`` (default, as the scanner's
            ``2dseq``) or ``"sum"``, which keeps the acquisition counts of a
            density-weighted method such as ``<User:ssfpCSI>`` as k-space
            weighting.
            Propagated noise is not an option here; use
            :meth:`reconstruct_with_noise`.

        Returns
        -------
        xarray.DataArray or None
            Temporal signal in acquisition/read/phase/slice/repetition/channel
            order with arbitrary scanner amplitudes, physical geometry and time
            coordinates. Float16/float32 storage is complex64; float64 storage
            is complex128. With spatial reconstruction disabled, returns sorted
            k-space. None means the caller explicitly skipped reconstruction.

        Raises
        ------
        ValueError
            If payload, precision, encoding or geometry metadata are invalid
            or inconsistent, or if a ``noise_estimate`` option is passed.
            Phase and transform failures propagate.
        NotImplementedError
            If no verified sorter or reconstruction operator exists.
        OSError
            If acquired samples cannot be read.

        Notes
        -----
        This eager, deterministic operation preserves acquired samples and
        acquisition metadata and does not store intermediate or final arrays
        on the handler. Use ``reconstruct_with_noise`` to receive the signal
        and its propagated noise together.

        Examples
        --------
        >>> signal = strategy.reconstruct(handler, data_type="fid", cut_off=0)  # doctest: +SKIP
        """
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerCsiRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        if "noise_estimate" in reco_options:
            raise ValueError(
                "BrukerCsiRecoStrategy.reconstruct does not accept "
                "noise_estimate; call reconstruct_with_noise(handler, "
                "noise_estimate) to receive the signal and its propagated noise."
            )
        return self._reconstruct(handler, reco_options)[0]

    def reconstruct_with_noise(
        self,
        handler,
        noise_estimate,
        *,
        data_type="job",
        cut_off=None,
        data_precision=np.float64,
        append_parameters=True,
    ) -> tuple[xr.DataArray, xr.Dataset]:
        """Return the CSI signal and propagated noise without handler caches.

        Parameters
        ----------
        handler : BrukerHandler
            Source of acquired samples, encoding positions, geometry in mm,
            dwell time in seconds and receiver metadata. A missing FID is
            read once and cached as acquired data.
        noise_estimate : xarray.Dataset
            Noise estimate from ``estimate_csi_outer_kspace_noise``. Its
            receiver covariance and acquisition metadata must match the scan.
        data_type : {"job", "fid"}, optional
            Preferred acquired payload; defaults to the job file.
        cut_off : int, optional
            Leading acquisition samples to remove; defaults to scan metadata.
        data_precision : str, numpy type or numpy.dtype, optional
            Requested storage precision, validated before reading data. Noise
            propagation always stores complex128 signal and float64 variance
            to preserve its linear reconstruction contract.
        append_parameters : bool, optional
            Add acquisition metadata and provenance to the signal.

        Returns
        -------
        signal : xarray.DataArray
            Temporal signal in acquisition/read/phase/slice/repetition/channel
            order, retaining physical geometry, time coordinates and units.
        noise : xarray.Dataset
            Signal, propagated covariance, variance and reconstruction/noise
            identifiers from the same phase operator and spatial transform.

        Raises
        ------
        ValueError
            If precision, payload, encoding, phase shift or noise metadata are
            invalid, or no noise estimate was supplied.
        NotImplementedError
            If the encoding has no verified sorter or reconstruction operator.

        Notes
        -----
        The deterministic calculation does not mutate acquired samples or
        store intermediate/final reconstruction arrays on the handler.
        Only loading a missing acquired FID may update the handler.

        Examples
        --------
        >>> signal, noise = strategy.reconstruct_with_noise(
        ...     handler, estimate, data_type="fid"
        ... )  # doctest: +SKIP
        """
        if noise_estimate is None:
            raise ValueError("reconstruct_with_noise requires a noise_estimate.")
        return self._reconstruct(
            handler,
            {
                "data_type": data_type,
                "cut_off": cut_off,
                "data_precision": data_precision,
                "append_parameters": append_parameters,
            },
            noise_estimate=noise_estimate,
        )

    def _reconstruct(self, handler, reco_options, noise_estimate=None):
        """Return both final results through one eager CSI reconstruction path."""
        DATA_PRECISION = parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        append_parameters = reco_options.get("append_parameters", True)

        if not hasattr(handler, "fid") or handler.fid is None:
            handler.load_data(source=None)

        set_data = reco_options.get("set_data", True)
        if noise_estimate is not None and not set_data:
            raise ValueError("noise_estimate requires set_data=True.")
        data_type = reco_options.get("data_type", "job")
        requested_cut_off = reco_options.get("cut_off", None)

        kspace_xarray = self.sort_kspace(
            handler=handler,
            data_type=data_type,
            cut_off=requested_cut_off,
            db=False,
            write_to_object=False,
            repeated_positions=reco_options.get("repeated_positions", "mean"),
        )
        propagated = None
        if not set_data:
            reco_data = kspace_xarray
        else:
            reco_data, propagated = self._reconstruct_sorted_csi(
                kspace_xarray=kspace_xarray,
                handler=handler,
                data_precision=DATA_PRECISION,
                noise_estimate=noise_estimate,
            )

        if append_parameters:
            reco_data = self._append_parameters(array=reco_data, handler=handler)
            reco_data = apply_processed_axis_metadata(
                data=reco_data,
                handler=handler,
                cut_off=int(kspace_xarray.attrs["cut_off"]),
                sampling_axis=kspace_xarray.coords["acquisition_idx"].values,
                sampling_unit=kspace_xarray.coords["acquisition_idx"].attrs.get(
                    "unit", "index"
                ),
            )
        return reco_data, propagated

    def _sorted_csi_to_data(
        self,
        kspace_xarray: xr.DataArray,
        handler,
        data_precision=np.float16,
        noise_estimate=None,
    ) -> xr.DataArray:
        """Return Bruker CSI data in display-ready spatial order.

        The Bruker CSI ``fid`` payload is first sorted into read/phase order. For
        methods that do not use the scanner spatial grid, the sorted read/phase
        grid is still k-space-like, so apply the Bruker FOV phase shift before
        inverse-transforming it to the localized voxel grid. Stock
        ``<Bruker:CSI>`` stores the sorted grid in the same spatial convention as
        the scanner ``2dseq`` output, so no additional spatial Fourier transform
        is applied there. The spectroscopic ``acquisition_idx`` dimension remains
        temporal.
        """
        return self._reconstruct_sorted_csi(
            kspace_xarray, handler, data_precision, noise_estimate
        )[0]

    def _reconstruct_sorted_csi(
        self,
        kspace_xarray,
        handler,
        data_precision=np.float16,
        noise_estimate=None,
    ):
        """Return spatial signal and its optional propagated noise explicitly."""
        grid = _transpose_csi_spatial_axes(kspace_xarray=kspace_xarray)
        propagated = None
        if self._uses_scanner_spatial_grid(handler):
            if noise_estimate is not None:
                raise ValueError(
                    "Automatic outer-kspace noise propagation requires a verified Fourier-encoded method."
                )
            reco_data, values = grid, grid.to_numpy().copy()
            spatial_mode, fft_sizes, fov_phase_shift = (
                "scanner_grid",
                {},
                "not_requested",
            )
        else:

            def phase_shift(array):
                return shift_image(
                    data=array,
                    shift_method="phase",
                    force_shift=True,
                    scan=handler,
                    domain="kspace",
                )

            spatial_mode, fov_phase_shift = "ifft", "applied"
            fft_sizes = _csi_spatial_fft_sizes(grid)
            if noise_estimate is None:
                reco_data = _shift_csi_fov(grid, phase_shift)
                values = _csi_spatial_ifft(reco_data)
            else:
                weights, shifted_attrs = _csi_fov_phase_weights(grid, phase_shift)
                reco_data = grid
                reco_data.attrs = shifted_attrs
                propagated = reconstruct_csi_with_noise(
                    kspace_xarray, noise_estimate, spatial_weights=weights
                )
                values = propagated.signal.transpose(*grid.dims).values
                # Quantization is nonlinear and outside linear propagation.
                data_precision = parse_data_precision(np.float64)
        reco_data = _finalize_csi_spatial_result(
            data_precision=data_precision,
            fft_sizes=fft_sizes,
            fov_phase_shift=fov_phase_shift,
            propagated=propagated,
            reco_data=reco_data,
            spatial_mode=spatial_mode,
            values=values,
        )
        try:
            reco_data.attrs["git_commit_hash"] = handler.git_commit_hash
        except AttributeError:
            reco_data.attrs["git_commit_hash"] = None
        reco_data = assign_xyz_coordinates(reco_data, scan=handler)
        return reco_data, propagated

    def sort_kspace(
        self,
        handler,
        data_type="job",
        cut_off=None,
        db=False,
        data=None,
        write_to_object=False,
        repeated_positions="mean",
    ):
        """Sort acquired CSI samples into a labelled encoding grid.

        Paired linear and centric positions share one sorter. Duplicate
        positions are averaged or added (``repeated_positions``) and missing
        bins contain NaN. The source and
        method determine whether the spatial axes represent k-space, scanner
        positions or an unknown domain. The acquired payload is not modified.

        Parameters
        ----------
        handler
            Bruker handler. The encoding order, read/phase matrix, one grid
            position per FID and the samples per FID come from
            ``handler.to_csi_acquisition()``; receiver, repetition and slice
            counts, sampling times, RPS grids in mm and source metadata in
            ``method`` and ``acqp`` come from the handler itself. FID or job
            samples are read lazily when no input array is supplied.
        data_type : str, optional
            Preferred source: ``"job"`` (default) or ``"fid"``. Other values
            retain the legacy job preference. The alternate source is tried
            only when the preferred source does not match the acquisition plan.
        cut_off : int, optional
            Leading temporal samples to remove once. None uses
            ``handler.cut_off``; zero keeps the complete FID. Must be smaller
            than the acquisition length and nonnegative.
        db : bool, optional
            Legacy diagnostic flag; no plots are constructed.
        data : numpy.ndarray, optional
            Complex samples in repetition/slice/encoding-position/channel/
            acquisition order. The sample count must match the acquisition
            plan; supplied data bypass file reads and are not modified.
        write_to_object : bool, optional
            Store the result on ``handler.kspace_xarray`` when True. The
            default is False; no handler arrays are overwritten in that case.
        repeated_positions : {"mean", "sum"}, optional
            How FIDs acquired more than once at one read/phase position are
            combined. ``"mean"`` (default) matches the scanner's ``2dseq``.
            ``"sum"`` keeps the acquisition counts as k-space weighting, the
            intended density weighting of ``<User:ssfpCSI>``; noise
            propagation then refuses the data, since it assumes averaged
            repeats.

        Returns
        -------
        xarray.DataArray
            Complex samples with dimensions ``acquisition_idx``,
            ``repetition_idx``, ``slice_idx``, ``phase_idx``, ``channel_idx``,
            ``read_idx`` in that order. Sampling coordinates use seconds or
            indices and spatial coordinates retain the handler's RPS mm
            orientation. Sorting retains calculation precision and adds
            encoding counts, source policy and provenance attributes.

        Raises
        ------
        ValueError
            If the scan cannot be described as a ``CSIAcquisition`` (the
            message names the field or scanner parameter),
            ``repeated_positions`` is neither ``"mean"`` nor ``"sum"``,
            samples are missing, their count or paired encoding positions are
            inconsistent, or the cutoff is outside its valid range.
        NotImplementedError
            If the encoding scheme has no verified grid sorter.

        Examples
        --------
        >>> sorted_fid = strategy.sort_kspace(handler, data_type="fid", cut_off=0)
        >>> sorted_fid.dims
        ('acquisition_idx', 'repetition_idx', 'slice_idx', 'phase_idx', 'channel_idx', 'read_idx')
        """
        if repeated_positions not in REPEATED_POSITION_RULES:
            raise ValueError(
                f"CSI reconstruction option repeated_positions="
                f"{repeated_positions!r} is not supported; use 'mean' (as the "
                "scanner's 2dseq) or 'sum' (keep acquisition counts as k-space "
                "weighting)."
            )
        acquisition = handler.to_csi_acquisition()
        n_repetitions = handler.n_repetitions
        n_read, n_phase = acquisition.matrix
        n_slices = handler.n_slices
        n_acquisition = acquisition.n_points
        n_receivers = handler.n_receivers
        cut_off = resolve_cut_off(handler, cut_off, n_acquisition, context="CSI")
        enc_order = acquisition.encoding_order
        read_idx = acquisition.read_positions
        phase_idx = acquisition.phase_positions
        read_idx, phase_idx, n_positions, expected_fid_size = _csi_acquisition_plan(
            n_acquisition=n_acquisition,
            n_receivers=n_receivers,
            n_repetitions=n_repetitions,
            n_slices=n_slices,
            phase_idx=phase_idx,
            read_idx=read_idx,
        )
        acqp = getattr(handler, "acqp", {}) or {}
        fid, source, storage_padding = _select_csi_payload(
            data_type=data_type,
            expected_fid_size=expected_fid_size,
            data=data,
            n_acquisition=n_acquisition,
            n_positions=n_positions,
            n_receivers=n_receivers,
            n_repetitions=n_repetitions,
            n_slices=n_slices,
            read_payload=lambda name: getattr(handler, name, None),
            scan_path=getattr(handler, "path", "(no path)"),
            storage_record_samples=_bruker_storage_record_samples(
                acqp, samples=n_acquisition, channels=n_receivers
            ),
        )
        kspace_array, acquisition_counts = _sort_csi_payload(
            enc_order=enc_order,
            fid=fid,
            n_acquisition=n_acquisition,
            n_phase=n_phase,
            n_positions=n_positions,
            n_read=n_read,
            n_receivers=n_receivers,
            n_repetitions=n_repetitions,
            n_slices=n_slices,
            phase_idx=phase_idx,
            read_idx=read_idx,
            repeated_positions=repeated_positions,
        )
        logger.debug(f"kspace_array.shape after sorting: {kspace_array.shape}")
        logger.debug(f"handler.sampling_time_axis_s={handler.sampling_time_axis_s}")

        # Remove the leading digital-filter/transient points exactly once.
        kspace_array = kspace_array[cut_off:, ...]
        sampling_axis, sampling_unit = processed_sampling_axis(
            handler=handler,
            cut_off=cut_off,
            npoints=kspace_array.shape[0],
        )
        kspace_array, dims, coords = _label_csi_grid_coordinates(
            kspace_array=kspace_array,
            n_receivers=n_receivers,
            n_repetitions=n_repetitions,
            rps_grid_mm=handler.rps_grid_mm,
            sampling_axis=sampling_axis,
            sampling_unit=sampling_unit,
        )
        # generate "empty" xarray:
        kspace_xarray = self.set_kspace_xarray(
            handler=handler,
            data=kspace_array,
            dims=dims,
            coords=coords,
            write_to_object=False,
        )
        kspace_xarray = apply_processed_axis_metadata(
            data=kspace_xarray,
            handler=handler,
            cut_off=cut_off,
            sampling_axis=sampling_axis,
            sampling_unit=sampling_unit,
        )

        kspace_xarray = attach_csi_raw_provenance(
            kspace_xarray,
            counts=acquisition_counts,
            method=self._method_name(handler),
            encoding=enc_order,
            source=source,
            requested_source=data_type,
            storage_padding_complex=storage_padding,
            declared_storage_format=acqp.get("GO_raw_data_format", "unknown"),
            declared_byte_order=acqp.get("BYTORDA", "unknown"),
            repeated_positions=repeated_positions,
        )
        if write_to_object:
            handler.kspace_xarray = kspace_xarray
        return kspace_xarray

    def get_phase_encoding_order(self, handler):
        """Return the encoding order of a CSI scan and its k-space positions.

        The two position arrays name, for every acquired FID in the order of
        acquisition, the read and phase position on the full ``PVM_Matrix``
        grid. Stock methods (``<Bruker:CSI>``, ``<User:csi_fix>``,
        ``<User:geoffCSI>``) take them from ``PVM_EncSteps0`` and
        ``PVM_EncSteps1``. The density-weighted balanced-SSFP method
        ``<User:ssfpCSI>`` takes the same scanner steps and repeats each
        position as often as its entry in ``AverageList``; the repeats of one
        position are acquired one after the other. The custom methods ``<User:lucaCSI>`` to
        ``<User:lucaCSI4>`` take them from ``CentricEncOrderMatrixx/y`` or
        ``SpiralEncOrderMatrixx/y`` when ``CentricEncOrder_OnOff`` or
        ``SpiralEncOrder_OnOff`` is ``"On"``, and from the scanner steps
        otherwise.

        Parameters
        ----------
        handler : VendorHandler
            Bruker acquisition whose ``method`` mapping holds ``Method``,
            ``PVM_Matrix`` (read, phase) and the encoding parameters named
            above. Signed steps are relative to the matrix centre ``n // 2``,
            also for odd and unequal matrix sizes.

        Returns
        -------
        enc_order : str
            ``"LINEAR_ENC LINEAR_ENC"`` or ``"CENTRIC_ENC CENTRIC_ENC"`` for
            scanner steps, ``"luca_centric"`` or ``"luca_spiral"`` for stored
            pairs. The name does not change the positions; the reconstruction
            uses it to refuse orders without a verified sorter.
        read_positions : numpy.ndarray
            1D integer read positions, zero-based and dimensionless, one per
            acquired FID. For scanner steps the length is ``n_read * n_phase``
            with read as the inner loop (``sum(AverageList)`` for
            ``<User:ssfpCSI>``).
        phase_positions : numpy.ndarray
            Phase position of the same FID, same length. Stored pairs keep
            their order; stored values without a negative entry are returned
            unchanged.

        Raises
        ------
        KeyError
            If a required ``method`` parameter is missing; the message names
            the parameter and the scan.
        ValueError
            If ``Method`` is not a supported CSI method, if ``PVM_EncOrder``
            is not a supported order (for example a different order on the
            two axes), if ``AverageList`` does not hold one positive count per
            position, or if signed steps are not finite integers inside the
            matrix. The message names the parameter and its value.

        Notes
        -----
        Deterministic and eager. Neither acquired samples nor parameters are
        modified, and no file is read. A custom method without stored pairs
        logs a warning that the scanner steps are used.

        Examples
        --------
        A 3 x 2 matrix with the read steps ``[0, -1, 1]`` and the phase steps
        ``[0, -1]`` gives the read positions ``[1, 0, 2, 1, 0, 2]`` and the
        phase positions ``[1, 1, 1, 0, 0, 0]``:

        >>> order, read, phase = strategy.get_phase_encoding_order(scan.handler)
        """
        parameter = handler.method_parameter
        method_name = parameter("Method")
        if method_name in DENSITY_WEIGHTED_METHODS:
            order, read, phase = _scanner_encoding_positions(handler)
            return (order, *_repeat_by_average_list(handler, read, phase))
        if method_name in PVM_ENCODING_METHODS:
            return _scanner_encoding_positions(handler)
        if method_name not in PAIRED_ORDER_METHODS:
            supported = ", ".join(
                repr(name)
                for name in sorted(PVM_ENCODING_METHODS | PAIRED_ORDER_METHODS)
            )
            raise ValueError(
                f"The CSI encoding order is not defined for Method={method_name!r}. "
                f"Supported methods: {supported}."
            )

        if parameter("CentricEncOrder_OnOff") == "On":
            order, stored = "luca_centric", "CentricEncOrderMatrix"
        elif parameter("SpiralEncOrder_OnOff") == "On":
            order, stored = "luca_spiral", "SpiralEncOrderMatrix"
        else:
            order, read, phase = _scanner_encoding_positions(handler)
            logger.warning(
                f"CSI method {method_name} stores no centric or spiral order; "
                f"using the scanner steps with PVM_EncOrder={order!r}."
            )
            return order, read, phase
        read, phase = _stored_pair_positions(
            parameter(f"{stored}x"),
            parameter(f"{stored}y"),
            n_read=parameter("PVM_Matrix")[0],
            n_phase=parameter("PVM_Matrix")[1],
        )
        return order, read, phase


# ---------------------------------------------------------------
# ----------------       SIEMENS RECO     -----------------------
# ---------------------------------------------------------------

__all__ = ["BrukerCsiRecoStrategy"]
