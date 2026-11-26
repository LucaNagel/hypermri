"""Bruker RARE-family raw-data reconstruction strategies."""

from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.reconstruction.precision import (
    convert_for_calculation,
    parse_data_precision,
)
from hypermri.utils.system.logging import init_default_logger

from ._encoding import _encoding_step_positions
from .raw import RAW_RECONSTRUCTION_PRECISION, BrukerRawRecoStrategy

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


def _reshape_fair_loop(
    data, method, n_acquisition_loops, n_repetitions, n_inversion_time
):
    """Separate FAIR mode/TI loops, retaining tail dimensions and dtype.

    Input begins with acquisition*repetition*inversion iterations. Output
    begins with acquisition/repetition/inversion; method selects the verified
    INTERLEAVED or INTERLEAVED2 storage order. Invalid mode raises ValueError.
    """
    tail_shape = data.shape[1:]
    method = method
    fair_mode = str(method.get("PVM_FairMode", "")).strip(" <>\t\r\n").upper()
    if fair_mode == "INTERLEAVED2":
        reshaped = np.reshape(
            data,
            (
                n_repetitions,
                n_inversion_time,
                n_acquisition_loops,
                *tail_shape,
            ),
        )
        return np.transpose(
            reshaped,
            (2, 0, 1, *range(3, reshaped.ndim)),
        )

    if fair_mode == "INTERLEAVED":
        return np.reshape(
            data,
            (
                n_acquisition_loops,
                n_repetitions,
                n_inversion_time,
                *tail_shape,
            ),
        )

    raise ValueError(
        "Cannot determine Bruker FAIR loop order: PVM_FairMode must be "
        "INTERLEAVED or INTERLEAVED2 when acquisition and inversion axes "
        "are both present."
    )


def _sort_rare_acquisition_iterations(
    all_repetitions_data,
    fid_process,
    final_read_size,
    is_fat_water,
    n_channels,
    n_phase_encoded,
    n_read_encoded,
    n_slice_encoded,
    n_waterfat,
    pe2_order,
    pe_order,
    spatial_dimension,
    total_iterations,
):
    """Sort 2D slice/echo-train loops or 3D phase loops into acquisition iterations.

    fid_process contains echo-train/channel/read blocks. Each returned array
    is slice/(optional waterfat)/channel/phase/read with unchanged sample dtype.
    Encoding permutations do not interpolate or modify the acquired FID.
    """
    # --- 2. Process based on spatial dimension (3D or 2D Multi-Slice) ---
    if spatial_dimension == "3D":
        # --- 3D Acquisition Logic ---
        if is_fat_water:
            fid_process_by_rep = np.reshape(
                fid_process,
                (
                    total_iterations,
                    -1,
                    n_waterfat,
                    n_channels,
                    n_read_encoded,
                ),
            )
        else:
            fid_process_by_rep = np.reshape(
                fid_process,
                (total_iterations, -1, n_channels, n_read_encoded),
            )

        for iteration_idx in range(total_iterations):
            single_iteration_data = fid_process_by_rep[iteration_idx, ...]
            if is_fat_water:
                fid_3d_reshaped = np.reshape(
                    single_iteration_data,
                    (
                        n_slice_encoded,
                        n_phase_encoded,
                        n_waterfat,
                        n_channels,
                        n_read_encoded,
                    ),
                )
            else:
                fid_3d_reshaped = np.reshape(
                    single_iteration_data,
                    (n_slice_encoded, n_phase_encoded, n_channels, n_read_encoded),
                )
                fid_3d_reshaped = fid_3d_reshaped[:, :, np.newaxis, :, :]

            if pe2_order is not None:
                fid_3d_reshaped = fid_3d_reshaped[pe2_order, ...]

            processed_single_iteration = []
            for slice_idx in range(n_slice_encoded):
                for waterfat_idx in range(n_waterfat):
                    for channel_idx in range(n_channels):
                        temp_slice_channel_data = fid_3d_reshaped[
                            slice_idx, :, waterfat_idx, channel_idx, :
                        ]
                        temp_reordered = temp_slice_channel_data[pe_order, :]
                        processed_single_iteration.append(temp_reordered)

            iteration_array = np.array(processed_single_iteration)
            iteration_array = np.reshape(
                iteration_array,
                (
                    n_slice_encoded,
                    n_waterfat,
                    n_channels,
                    n_phase_encoded,
                    n_read_encoded,
                ),
            )
            if is_fat_water:
                all_repetitions_data.append(iteration_array)
            else:
                all_repetitions_data.append(iteration_array[:, 0, ...])

    else:
        # --- 2D Multi-Slice Acquisition Logic ---
        if fid_process.shape[0] % total_iterations != 0:
            raise ValueError(
                "Cannot split 2D RARE data into acquisition iterations: "
                f"{fid_process.shape[0]} blocks are not divisible by "
                f"{total_iterations} iterations."
            )
        num_blocks_per_iteration = fid_process.shape[0] // total_iterations
        image_loop_size = n_slice_encoded * n_waterfat
        if num_blocks_per_iteration % image_loop_size != 0:
            raise ValueError(
                "Cannot split 2D RARE image loops into slice and fat/water axes: "
                f"{num_blocks_per_iteration} blocks per iteration is not divisible "
                f"by {image_loop_size} ({n_slice_encoded} slices * "
                f"{n_waterfat} fat/water loops)."
            )
        for iteration_idx in range(total_iterations):
            start_idx = iteration_idx * num_blocks_per_iteration
            end_idx = (iteration_idx + 1) * num_blocks_per_iteration
            single_iteration_data = fid_process[start_idx:end_idx, ...]

            processed_single_iteration = []
            for slice_idx in range(n_slice_encoded):
                for waterfat_idx in range(n_waterfat):
                    for channel_idx in range(n_channels):
                        if is_fat_water:
                            # In RARE.ppg, PVM_FwsLoop repeats the full slice loop.
                            image_loop_idx = waterfat_idx * n_slice_encoded + slice_idx
                            temp_slice_channel_data = single_iteration_data[
                                image_loop_idx::image_loop_size,
                                :,
                                channel_idx,
                                :,
                            ]
                        else:
                            temp_slice_channel_data = single_iteration_data[
                                slice_idx::n_slice_encoded, :, channel_idx, :
                            ]
                        temp_reshaped = np.reshape(
                            temp_slice_channel_data, (-1, final_read_size)
                        )
                        temp_reordered = temp_reshaped[pe_order, ...]
                        processed_single_iteration.append(temp_reordered)

            iteration_array = np.array(processed_single_iteration)
            iteration_array = np.reshape(
                iteration_array,
                (
                    n_slice_encoded,
                    n_waterfat,
                    n_channels,
                    n_phase_encoded,
                    n_read_encoded,
                ),
            )
            if is_fat_water:
                all_repetitions_data.append(iteration_array)
            else:
                all_repetitions_data.append(iteration_array[:, 0, ...])

    return all_repetitions_data


def _reshape_rare_output_axes(
    fid_reordered,
    is_fair_rare,
    is_fat_water,
    method,
    n_acquisition_loops,
    n_channels,
    n_inversion_time,
    n_phase_encoded,
    n_read_encoded,
    n_reps,
    n_slice_encoded,
    n_waterfat,
):
    """Separate acquisition, repetition and FAIR/fat-water axes in sorted RARE data.

    Input stacks acquisition iterations; output is acquisition/repetition/slice/
    phase/(waterfat)/(inversion)/channel/read. Values and dtype are preserved.
    """
    if is_fair_rare and is_fat_water:
        fid_reordered = _reshape_fair_loop(
            fid_reordered,
            method,
            n_acquisition_loops,
            n_reps,
            n_inversion_time,
        )
        logger.debug(
            f"FAIR_RARE fat/water fid_reordered after reshape: shape={fid_reordered.shape}"
        )

        # Target order before zero fill: acquisition, repetition, slice,
        # phase, waterfat, inversion, channel, read.
        fid_reordered = np.transpose(fid_reordered, (0, 1, 3, 6, 4, 2, 5, 7))
        logger.debug(
            f"FAIR_RARE fat/water fid_reordered final shape: {fid_reordered.shape}"
        )
    elif is_fair_rare:
        # For FAIR_RARE: reshape to separate inversion time from repetitions
        # Current shape: (n_reps * n_inversion_time, n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded)
        # Target shape:  (n_reps, n_inversion_time, n_slice_encoded,
        # n_channels, n_phase_encoded, n_read_encoded)
        fid_reordered = _reshape_fair_loop(
            fid_reordered,
            method,
            n_acquisition_loops,
            n_reps,
            n_inversion_time,
        )
        logger.debug(
            f"FAIR_RARE fid_reordered after reshape: shape={fid_reordered.shape}"
        )

        # Transpose to match expected dimension order: (1, n_reps,
        # n_slice_encoded, n_phase_encoded, n_inversion_time, n_channels,
        # n_read_encoded)
        fid_reordered = np.transpose(fid_reordered, (0, 1, 3, 5, 2, 4, 6))
        logger.debug(f"FAIR_RARE fid_reordered final shape: {fid_reordered.shape}")
    elif is_fat_water:
        fid_reordered = np.reshape(
            fid_reordered,
            (
                n_acquisition_loops,
                n_reps,
                n_slice_encoded,
                n_waterfat,
                n_channels,
                n_phase_encoded,
                n_read_encoded,
            ),
        )
        # Target order before zero fill: acquisition, repetition, slice,
        # phase, waterfat, channel, read.
        fid_reordered = np.transpose(fid_reordered, (0, 1, 2, 5, 3, 4, 6))
        logger.debug(
            f"fat/water fid_reordered after transpose: type={type(fid_reordered)}, shape={getattr(fid_reordered, 'shape', 'no shape')}"
        )

    else:
        # Regular RARE processing
        fid_reordered = np.reshape(
            fid_reordered,
            (
                n_acquisition_loops,
                n_reps,
                n_slice_encoded,
                n_channels,
                n_phase_encoded,
                n_read_encoded,
            ),
        )
        fid_reordered = np.transpose(fid_reordered, (0, 1, 2, 4, 3, 5))
        logger.debug(
            f"fid_reordered after transpose: type={type(fid_reordered)}, shape={getattr(fid_reordered, 'shape', 'no shape')}"
        )

    return fid_reordered


def _zero_fill_rare_encoding(
    fid_reordered,
    final_phase_size,
    final_read_size,
    final_slice_size,
    is_fair_rare,
    is_fat_water,
    n_acquisition_loops,
    n_channels,
    n_inversion_time,
    n_phase_encoded,
    n_read_encoded,
    n_reps,
    n_slice_encoded,
    n_waterfat,
    phase_target_indices,
):
    """Place sorted RARE lines into the final slice/phase/read matrix.

    Leading acquisition/repetition and optional waterfat/inversion axes are
    retained. Phase indices follow measured partial-Fourier steps, read and
    slice blocks are centred, and new samples are zeros of the input dtype.
    """
    # --- 4. Zero-Padding to Final Matrix Size with Centering ---
    if is_fair_rare and is_fat_water:
        final_shape = (
            n_acquisition_loops,  # acquisition_idx
            n_reps,  # repetition_idx
            final_slice_size,  # slice_idx
            final_phase_size,  # phase_idx
            n_waterfat,  # waterfat_idx
            n_inversion_time,  # inversion_idx
            n_channels,  # channel_idx
            final_read_size,  # read_idx
        )
    elif is_fair_rare:
        # For FAIR_RARE: include inversion_idx dimension
        final_shape = (
            n_acquisition_loops,  # acquisition_idx
            n_reps,  # repetition_idx
            final_slice_size,  # slice_idx
            final_phase_size,  # phase_idx
            n_inversion_time,  # inversion_idx - NEW (key for FAIR_RARE)
            n_channels,  # channel_idx
            final_read_size,  # read_idx
        )
    elif is_fat_water:
        final_shape = (
            n_acquisition_loops,  # acquisition_idx
            n_reps,  # repetition_idx
            final_slice_size,  # slice_idx
            final_phase_size,  # phase_idx
            n_waterfat,  # waterfat_idx
            n_channels,  # channel_idx
            final_read_size,  # read_idx
        )
    else:
        # Regular RARE: original shape
        final_shape = (
            n_acquisition_loops,  # acquisition_idx
            n_reps,  # repetition_idx
            final_slice_size,  # slice_idx
            final_phase_size,  # phase_idx
            n_channels,  # channel_idx
            final_read_size,  # read_idx
        )
    logger.debug(f"About to create zeros with final_shape: {final_shape}")
    logger.debug(f"About to access fid_reordered.dtype: {fid_reordered.dtype}")
    fid_reordered_zeros = np.zeros(final_shape, dtype=fid_reordered.dtype)

    # Calculate padding and starting indices for centering the data
    slice_pad = final_slice_size - n_slice_encoded
    slice_start = slice_pad // 2

    phase_target_indices = phase_target_indices(
        final_phase_size,
        n_phase_encoded,
    )

    read_pad = final_read_size - n_read_encoded
    read_start = read_pad // 2

    # Calculate ending indices
    slice_end = slice_start + n_slice_encoded
    read_end = read_start + n_read_encoded

    # Every RARE variant keeps phase at axis 3 and read at the last axis.
    # Place partial-Fourier lines by their actual encoding step rather than
    # centering the acquired block, which would shift asymmetric sampling.
    for source_phase_idx, target_phase_idx in enumerate(phase_target_indices):
        fid_reordered_zeros[
            :,
            :,
            slice_start:slice_end,
            target_phase_idx,
            ...,
            read_start:read_end,
        ] = fid_reordered[:, :, :, source_phase_idx, ..., :]

    return fid_reordered_zeros


class BrukerRareRawRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruct Bruker RARE, FAIR-RARE and RAREVTR raw data.

    Notes
    -----
    Construction takes no arguments and performs no reads or reconstruction.
    :meth:`reconstruct` sorts the echo trains into k-space and then uses the
    stages of :class:`BrukerRawRecoStrategy`.
    """

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "postprocess_fid",
            "reshape_fid",
            "set_kspace_xarray",
            "shift_xarray",
            "set_data",
            "db",
            "data_precision",
            "append_parameters",
        }
    )

    @staticmethod
    def _acquisition_loop_count(handler, n_repetitions, n_inversion_time) -> int:
        """Return Bruker ``NR`` frames not represented by public dimensions.

        ParaVision stores RAREVTR movie frames, and the selective/non-selective
        FAIR pair, in ``ACQ NR``. ``PVM_NRepetitions`` and FAIR inversion times
        already have dedicated axes, so only the remaining factor belongs on
        ``acquisition_idx``.
        """
        n_repetitions = BrukerRareRawRecoStrategy._positive_integer_parameter(
            n_repetitions,
            "PVM_NRepetitions",
        )
        n_inversion_time = BrukerRareRawRecoStrategy._positive_integer_parameter(
            n_inversion_time,
            "inversion-time count",
        )
        logical_iterations = n_repetitions * n_inversion_time
        acqp = getattr(handler, "acqp", None) or {}
        acquisition_repetitions = BrukerRareRawRecoStrategy._positive_integer_parameter(
            acqp.get("NR", logical_iterations),
            "ACQ NR",
        )

        if (
            acquisition_repetitions < logical_iterations
            or acquisition_repetitions % logical_iterations != 0
        ):
            raise ValueError(
                "Cannot map Bruker ACQ NR onto RARE dimensions: "
                f"NR={acquisition_repetitions}, repetitions={n_repetitions}, "
                f"inversion times={n_inversion_time}."
            )
        return acquisition_repetitions // logical_iterations

    @staticmethod
    def _reshape_fair_iterations(
        data,
        handler,
        n_acquisition_loops,
        n_repetitions,
        n_inversion_time,
    ):
        """Separate FAIR mode/TI loops according to ParaVision ordering."""
        return _reshape_fair_loop(
            data,
            getattr(handler, "method", None) or {},
            n_acquisition_loops,
            n_repetitions,
            n_inversion_time,
        )

    @staticmethod
    def _phase_target_indices(handler, final_phase_size, encoded_phase_size):
        """Map sorted Bruker encoding steps onto the final phase matrix."""
        final_phase_size = BrukerRareRawRecoStrategy._positive_integer_parameter(
            final_phase_size,
            "final phase-matrix size",
        )
        encoded_phase_size = BrukerRareRawRecoStrategy._positive_integer_parameter(
            encoded_phase_size,
            "encoded phase-matrix size",
        )
        if encoded_phase_size > final_phase_size:
            raise ValueError(
                "Invalid Bruker phase dimensions: encoded phase-matrix size "
                f"{encoded_phase_size} exceeds final size {final_phase_size}."
            )

        method = getattr(handler, "method", None) or {}
        raw_encoding_steps = method.get("PVM_EncSteps1")
        if raw_encoding_steps is None:
            # Historical compatibility for acquisitions that do not publish
            # encoding steps: place the measured block in the matrix centre.
            phase_start = final_phase_size // 2 - encoded_phase_size // 2
            logger.warning(
                "PVM_EncSteps1 is missing; using centred phase-line placement."
            )
            return np.arange(phase_start, phase_start + encoded_phase_size)

        target_indices = np.sort(
            _encoding_step_positions(
                raw_encoding_steps, final_phase_size, "PVM_EncSteps1"
            )
        )
        if target_indices.size != encoded_phase_size:
            raise ValueError(
                "Invalid Bruker PVM_EncSteps1 length: "
                f"expected {encoded_phase_size}, got {target_indices.size}."
            )
        if np.unique(target_indices).size != encoded_phase_size:
            raise ValueError("Invalid Bruker PVM_EncSteps1: encoding steps repeat.")
        return target_indices

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        """Reconstruct the raw FID of a Bruker RARE scan.

        The FID is first sorted into the phase-encoding order of the echo
        train with :meth:`postprocess_fid_rare`, then reconstructed with
        :meth:`reco_rawdata`.

        Parameters
        ----------
        handler : BrukerHandler
            Handler of the scan.  A missing FID is read first.
        **reco_options
            The names in :attr:`accepted_options`: ``no_reconstruction``,
            ``postprocess_fid``, ``reshape_fid``, ``set_kspace_xarray``,
            ``shift_xarray``, ``set_data``, ``db`` and
            ``data_precision`` as for :class:`BrukerRawRecoStrategy`, with
            ``postprocess_fid`` and ``reshape_fid`` defaulting to False here,
            and ``append_parameters`` (bool, default True), which adds the
            scan parameters to the attributes of the result. The FOV phase
            ramp always uses the acquisition offsets with sign +1.

        Returns
        -------
        xarray.DataArray or None
            The reconstruction, or ``None`` for ``no_reconstruction=True``.

        Raises
        ------
        ValueError
            If ``data_precision`` is not supported, if the FID is missing or
            empty, or if a stage of the reconstruction fails.

        Notes
        -----
        ``handler.fid`` is not modified.  The sorted k-space is passed directly between stages; no
        intermediate arrays or acquisition metadata are stored on the handler.
        """
        logger.debug(
            f"BrukerRareRawRecoStrategy.reconstruct starting for scan {handler.expnum}"
        )
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerRareRawRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_data = reco_options.get("set_data", True)
        db = reco_options.get("db", True)
        DATA_PRECISION = parse_data_precision(
            reco_options.get("data_precision", RAW_RECONSTRUCTION_PRECISION)
        )
        append_parameters = reco_options.get("append_parameters", True)

        self._load_fid_on_demand(handler)

        logger.debug("About to call postprocess_fid_rare ->")
        # ---------------------------------------------
        # try:
        # Postprocess the FID data for RARE
        # ---------------------------------------------
        fid_pp = self.postprocess_fid_rare(
            handler=handler,
            data_precision=DATA_PRECISION,
        )
        logger.debug(
            f"-> postprocess_fid_rare completed, processed FID type: {type(fid_pp)}"
        )

        # Reconstruct the raw data into an xarray DataArray
        logger.debug("About to call reco_rawdata ->")
        reco_data = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=set_kspace_xarray,
            shift_xarray=shift_xarray,
            set_data=set_data,
            db=db,
            data_precision=DATA_PRECISION,
            values=fid_pp,
        )
        logger.debug(f"-> reco_rawdata returned: {type(reco_data)}")
        logger.debug(f"reco_data is None: {reco_data is None}")

        if append_parameters:
            logger.debug("About to append parameters ->")
            reco_data = self._append_parameters(array=reco_data, handler=handler)
            logger.debug("_append_parameters completed")

        logger.debug(f"-> BrukerRareRawRecoStrategy returning: {type(reco_data)}")
        return reco_data

    def _sort_rare_preprocessed_fid(self, fid_pp, handler):
        """Read RARE layout metadata and call sorting, axis and zero-fill stages.

        fid_pp contains read-line samples after storage padding removal. Returns
        acquisition/repetition/slice/phase/(waterfat)/(inversion)/channel/read
        k-space with its calculation dtype; handler state is not written.
        """
        fid_process = fid_pp.copy()

        pe_order, pe2_order = self._get_phase_encode_order(handler=handler)

        # --- 1. Get essential parameters from the handler ---
        n_reps = handler.n_repetitions
        n_channels = handler.n_receivers
        n_read_encoded = handler.enc_matrix[0]
        n_phase_encoded = handler.enc_matrix[1]
        n_slice_encoded = handler.enc_matrix[2]  # slices or 2nd phase encode

        final_read_size = handler.matrix[0]
        final_phase_size = handler.matrix[1]
        final_slice_size = handler.matrix[2]

        # --- FAIR_RARE Detection and Parameters ---
        n_inversion_time = getattr(handler, "n_inversion_time", 1)
        is_fair_rare = n_inversion_time > 1
        logger.debug(
            f"FAIR_RARE detection: is_fair_rare={is_fair_rare}, n_inversion_time={n_inversion_time}"
        )

        if is_fair_rare:
            logger.info(
                f"FAIR_RARE sequence detected with {n_inversion_time} inversion time points"
            )

        # --- Fat/Water Separation Detection ---
        is_fat_water = self._is_fat_water_enabled(handler)
        n_waterfat = (
            int(getattr(handler, "method", {}).get("PVM_FwsLoop", 2))
            if is_fat_water
            else 1
        )
        logger.debug(f"scan is fat water separated: {is_fat_water}")

        if is_fat_water and handler.spatial_dimension == "3D":
            fid_process = np.reshape(
                fid_process,
                (
                    -1,
                    handler.rare_factor,
                    n_waterfat,
                    handler.n_receivers,
                    handler.matrix[0],
                ),
            )
        else:
            fid_process = np.reshape(
                fid_process,
                (-1, handler.rare_factor, handler.n_receivers, handler.matrix[0]),
            )

        logger.debug(f"fid_process was reshaped to {fid_process.shape}")

        all_repetitions_data = []
        logical_inversion_count = n_inversion_time if is_fair_rare else 1
        n_acquisition_loops = self._acquisition_loop_count(
            handler,
            n_reps,
            logical_inversion_count,
        )
        total_iterations = n_acquisition_loops * n_reps * logical_inversion_count

        all_repetitions_data = _sort_rare_acquisition_iterations(
            all_repetitions_data=all_repetitions_data,
            fid_process=fid_process,
            final_read_size=final_read_size,
            is_fat_water=is_fat_water,
            n_channels=n_channels,
            n_phase_encoded=n_phase_encoded,
            n_read_encoded=n_read_encoded,
            n_slice_encoded=n_slice_encoded,
            n_waterfat=n_waterfat,
            pe2_order=pe2_order,
            pe_order=pe_order,
            spatial_dimension=handler.spatial_dimension,
            total_iterations=total_iterations,
        )
        # --- 3. Finalize Data Structure ---
        logger.debug(f"all_repetitions_data length: {len(all_repetitions_data)}")
        if len(all_repetitions_data) > 0:
            logger.debug(
                f"all_repetitions_data[0] type: {type(all_repetitions_data[0])}"
            )
            logger.debug(
                f"all_repetitions_data[0] shape: {getattr(all_repetitions_data[0], 'shape', 'no shape')}"
            )

        fid_reordered = np.stack(all_repetitions_data, axis=0)
        logger.debug(
            f"fid_reordered after stack: type={type(fid_reordered)}, shape={getattr(fid_reordered, 'shape', 'no shape')}"
        )

        fid_reordered = _reshape_rare_output_axes(
            fid_reordered=fid_reordered,
            is_fair_rare=is_fair_rare,
            is_fat_water=is_fat_water,
            method=(getattr(handler, "method", None) or {}) if is_fair_rare else {},
            n_acquisition_loops=n_acquisition_loops,
            n_channels=n_channels,
            n_inversion_time=n_inversion_time,
            n_phase_encoded=n_phase_encoded,
            n_read_encoded=n_read_encoded,
            n_reps=n_reps,
            n_slice_encoded=n_slice_encoded,
            n_waterfat=n_waterfat,
        )
        logger.debug(f"fid_reordered is None: {fid_reordered is None}")
        logger.debug(
            f"fid_reordered.dtype: {getattr(fid_reordered, 'dtype', 'no dtype attr')}"
        )

        fid_reordered_zeros = _zero_fill_rare_encoding(
            fid_reordered=fid_reordered,
            final_phase_size=final_phase_size,
            final_read_size=final_read_size,
            final_slice_size=final_slice_size,
            is_fair_rare=is_fair_rare,
            is_fat_water=is_fat_water,
            n_acquisition_loops=n_acquisition_loops,
            n_channels=n_channels,
            n_inversion_time=n_inversion_time,
            n_phase_encoded=n_phase_encoded,
            n_read_encoded=n_read_encoded,
            n_reps=n_reps,
            n_slice_encoded=n_slice_encoded,
            n_waterfat=n_waterfat,
            phase_target_indices=lambda final_size, encoded_size: (
                self._phase_target_indices(handler, final_size, encoded_size)
            ),
        )

        return fid_reordered_zeros

    def postprocess_fid_rare(
        self, handler=None, db=True, data_precision=RAW_RECONSTRUCTION_PRECISION
    ):
        """
        Reorder and zero-pad the raw FID data of a RARE acquisition.

        This function handles both 2D multi-slice and 3D acquisitions, sorts
        the k-space data according to the phase-encoding order, and centers
        the acquired data within the final zero-padded matrix.

        Parameters
        ----------
        handler : VendorHandler
            Bruker handler of the scan with the acquired ``fid`` and the
            method parameters.  The FID is loaded if the attribute is missing.
        db : bool, optional
            Unused; kept for compatibility.
        data_precision : str, numpy type or numpy.dtype, optional
            Calculation precision of the FID; see
            :func:`~hypermri.reconstruction.precision.convert_for_calculation`.  A
            complex FID is only reduced to ``complex64`` for ``float16``.
            Defaults to ``RAW_RECONSTRUCTION_PRECISION`` (``np.float32``).

        Returns
        -------
        numpy.ndarray
            Reordered and zero-filled complex k-space with dimensions
            acquisition/repetition/slice/phase/(waterfat)/(inversion)/channel/read.
            Optional axes are present only for fat-water and FAIR acquisitions;
            the calculation dtype is preserved.

        Raises
        ------
        ValueError
            If the FID cannot be loaded or is empty, or if ``data_precision``
            is not supported.

        Notes
        -----
        ``handler.fid`` is not modified; the conversion works on a copy.
        """
        # Check if FID data is available
        if not hasattr(handler, "fid"):
            logger.info(
                f"Handler for scan {handler.expnum} missing FID data - loading automatically"
            )
            # Try to load the data if it's missing
            try:
                handler.load_data(source=handler.path)
                logger.info(f"Successfully loaded FID data for scan {handler.expnum}")
            except Exception as e:
                logger.error(f"Failed to load FID data for scan {handler.expnum}: {e}")
                raise ValueError(
                    f"FID data attribute missing for scan {handler.expnum} and could not load data. Cannot perform raw RARE reconstruction."
                ) from e

        if handler.fid is None or handler.fid.size == 0:
            logger.error(
                f"Handler for scan {handler.expnum} has fid attribute but it's None or empty"
            )
            raise ValueError(
                f"FID data is not available for scan {handler.expnum}. Cannot perform raw RARE reconstruction."
            )

        # process rawdata:
        logger.debug("About to call postprocess_fid_file with handler.fid")
        fid = convert_for_calculation(handler.fid, data_precision)
        fid_pp = self.postprocess_fid_file(values=fid, handler=handler)
        logger.debug(
            f"postprocess_fid_file returned: {type(fid_pp)}, shape: {getattr(fid_pp, 'shape', 'no shape')}"
        )

        fid_reordered_zeros = self._sort_rare_preprocessed_fid(fid_pp, handler)
        return fid_reordered_zeros


__all__ = ["BrukerRareRawRecoStrategy"]
