"""Raw reconstruction for Bruker 2D and 3D EPI acquisitions."""

from pathlib import Path
from typing import ClassVar

import numpy as np
from brukerapi.dataset import Dataset
from scipy.interpolate import interp1d

from hypermri.reconstruction.centred_dft import centered_fft, centered_ifft
from hypermri.reconstruction.precision import parse_data_precision
from hypermri.utils.system.logging import init_default_logger

from ._storage import _raw_storage_dtype
from .cartesian import BrukerCartesianRawRecoStrategy

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BrukerEpiRawRecoStrategy(BrukerCartesianRawRecoStrategy):
    """Reconstruct Bruker EPI while respecting shot-local echo polarity."""

    accepted_options: ClassVar[frozenset[str]] = (
        BrukerCartesianRawRecoStrategy.accepted_options
        | {
            "apply_trajectory_regridding",
            "trajectory_interpolation",
            "apply_phase_correction",
        }
    )

    @staticmethod
    def _enabled_parameter(value) -> bool:
        return str(value).strip().lower() in {"yes", "on", "true", "1"}

    def _apply_trajectory_regridding(
        self,
        values: np.ndarray,
        method: dict,
        reco_options: dict,
        *,
        axis: int,
    ) -> tuple[np.ndarray, dict]:
        """Interpolate a measured nonuniform EPI read trajectory to a unit grid."""
        requested = reco_options.get("apply_trajectory_regridding")
        advertised = self._enabled_parameter(method.get("PVM_EpiTrajAdjYesNo"))
        measured = self._enabled_parameter(method.get("PVM_EpiTrajAdjMeasured"))
        should_apply = advertised and measured if requested is None else bool(requested)
        b0_trajectory = np.asarray(
            method.get("PVM_EpiTrajAdjb0", []), dtype=float
        ).reshape(-1)
        attrs = {
            "epi_trajectory_regridding": False,
            "epi_trajectory_source": None,
            "epi_b0_trajectory_available": bool(
                b0_trajectory.size == values.shape[axis]
                and np.all(np.isfinite(b0_trajectory))
            ),
            "epi_b0_trajectory_correction": False,
        }
        if not should_apply:
            return values, attrs

        read_size = int(values.shape[axis])
        try:
            trajectory = np.asarray(
                method.get("PVM_EpiTrajAdjkx"), dtype=float
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid Bruker PVM_EpiTrajAdjkx: expected one finite value "
                "per read sample."
            ) from exc
        if trajectory.size != read_size or not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "Invalid Bruker PVM_EpiTrajAdjkx: expected "
                f"{read_size} finite values, got {trajectory.size}."
            )

        matrix0 = self._positive_integer_parameter(
            method.get("PVM_EpiTrajAdjMatrix0", read_size),
            "PVM_EpiTrajAdjMatrix0",
        )
        if matrix0 != read_size:
            raise ValueError(
                "PVM_EpiTrajAdjMatrix0 does not match the decoded EPI readout: "
                f"{matrix0} != {read_size}."
            )

        interpolation = str(
            reco_options.get("trajectory_interpolation", "cubic")
        ).lower()
        if interpolation not in {"linear", "cubic"}:
            raise ValueError(
                "trajectory_interpolation must be either 'linear' or 'cubic'."
            )

        # A few measured trajectories turn around slightly at the final ramp
        # sample. Sorting by measured kx preserves its physical grid location.
        order = np.argsort(trajectory, kind="stable")
        source_grid = trajectory[order]
        source_values = np.take(values, order, axis=axis)
        unique = np.concatenate(([True], np.diff(source_grid) > 1e-10))
        source_grid = source_grid[unique]
        source_values = np.take(source_values, np.flatnonzero(unique), axis=axis)
        minimum_points = 4 if interpolation == "cubic" else 2
        if source_grid.size < minimum_points:
            raise ValueError(
                "PVM_EpiTrajAdjkx does not contain enough unique trajectory "
                f"positions for {interpolation} interpolation."
            )

        # ParaVision's stored trajectory uses unit-spaced Cartesian k-grid
        # coordinates. Samples outside its measured support are zero-filled.
        target_grid = np.arange(read_size, dtype=float)
        interpolator = interp1d(
            source_grid,
            source_values,
            axis=axis,
            kind=interpolation,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )
        regridded = interpolator(target_grid).astype(values.dtype, copy=False)
        attrs.update(
            {
                "epi_trajectory_regridding": True,
                "epi_trajectory_source": "PVM_EpiTrajAdjkx",
                "epi_trajectory_interpolation": interpolation,
                "epi_trajectory_points": int(trajectory.size),
            }
        )
        return regridded, attrs

    @staticmethod
    def _phase_correction_coefficients(method, object_count: int):
        try:
            order = int(method.get("PVM_EpiMaxOrder", 0))
            values = np.asarray(
                method.get("PVM_EpiPhaseCorrection"), dtype=float
            ).reshape(-1)
        except (TypeError, ValueError):
            return None
        if order < 0 or values.size == 0 or values.size % (order + 1):
            return None
        coefficients = values.reshape(-1, order + 1)
        if coefficients.shape[0] not in (1, object_count):
            return None
        return coefficients

    def _apply_phase_correction(
        self,
        kspace: np.ndarray,
        odd_phase_lines: np.ndarray,
        coefficients: np.ndarray | None,
    ) -> np.ndarray:
        """Apply ParaVision's stored odd/even phase polynomial in read image space."""
        if coefficients is None or odd_phase_lines.size == 0:
            return kspace

        hybrid = centered_ifft(kspace, (0,))
        read_coordinate = (
            np.arange(kspace.shape[0], dtype=float) - kspace.shape[0] / 2.0
        )
        for object_index in range(kspace.shape[2]):
            row = coefficients[object_index if coefficients.shape[0] > 1 else 0]
            phase_cycles = np.polynomial.polynomial.polyval(read_coordinate, row)
            correction = np.exp(2j * np.pi * phase_cycles)
            hybrid[:, odd_phase_lines, object_index, :, :] *= correction[
                :, None, None, None
            ]
        return centered_fft(hybrid, (0,))

    @staticmethod
    def _semantic_2d_layout(handler, kspace: np.ndarray):
        """Map brukerapi's object/repetition axes to MRI semantic dimensions."""
        read_size, phase_size, object_count, raw_repetitions, channels = kspace.shape
        method = getattr(handler, "method", None) or {}
        method_name = str(method.get("Method", "")).lower()
        n_slices = max(1, int(getattr(handler, "n_slices", 1)))
        n_echoes = max(1, int(getattr(handler, "n_echoes", 1)))
        n_inversions = max(1, int(getattr(handler, "n_inversion_time", 1)))
        b_values = np.asarray(getattr(handler, "b_values", [])).reshape(-1)

        if n_slices > 1 and n_echoes > 1:
            raise ValueError(
                "Unsupported simultaneous slice/echo loops in 2D EPI: "
                f"{object_count} objects, {n_slices} slices, {n_echoes} echoes. "
                "Their raw object order needs validation; use scan.scanner_data "
                "or scan.reconstruct('seq2d') for the labelled scanner images."
            )

        if object_count == n_slices:
            object_kind = "slice"
            slice_count, echo_count, object_acquisitions = n_slices, 1, 1
        elif object_count == n_echoes and n_echoes > 1:
            object_kind = "echo"
            slice_count, echo_count, object_acquisitions = 1, n_echoes, 1
        elif (
            "t1_epi" in method_name
            and n_slices == n_echoes == 1
            and object_count == n_inversions
            and n_inversions > 1
        ):
            object_kind = "inversion"
            slice_count, echo_count, object_acquisitions = 1, 1, 1
        else:
            if n_slices > 1 or n_echoes > 1:
                raise ValueError(
                    "2D EPI object count does not match slice/echo metadata: "
                    f"{object_count} objects, {n_slices} slices, {n_echoes} echoes. "
                    "Use scan.scanner_data for the scanner reconstruction."
                )
            object_kind = "acquisition"
            slice_count, echo_count, object_acquisitions = 1, 1, object_count

        if object_kind == "inversion":
            repetition_kind = "acquisition"
            inversion_count, bvalue_count, repetition_acquisitions = (
                n_inversions,
                1,
                raw_repetitions,
            )
        elif "dtiepi" in method_name and b_values.size == raw_repetitions:
            repetition_kind = "bvalue"
            inversion_count, bvalue_count, repetition_acquisitions = (
                1,
                raw_repetitions,
                1,
            )
        elif n_inversions > 1 and raw_repetitions % n_inversions == 0:
            repetition_kind = "inversion"
            inversion_count, bvalue_count, repetition_acquisitions = (
                n_inversions,
                1,
                raw_repetitions // n_inversions,
            )
        else:
            repetition_kind = "acquisition"
            inversion_count, bvalue_count, repetition_acquisitions = (
                1,
                1,
                raw_repetitions,
            )

        acquisition_count = object_acquisitions * repetition_acquisitions
        full = np.zeros(
            (
                read_size,
                phase_size,
                slice_count,
                inversion_count,
                echo_count,
                bvalue_count,
                acquisition_count,
                channels,
            ),
            dtype=kspace.dtype,
        )
        for object_index in range(object_count):
            slice_index = object_index if object_kind == "slice" else 0
            echo_index = object_index if object_kind == "echo" else 0
            object_acquisition = object_index if object_kind == "acquisition" else 0
            for repetition_index in range(raw_repetitions):
                inversion_index = 0
                if object_kind == "inversion":
                    inversion_index = object_index
                elif repetition_kind == "inversion":
                    inversion_index = repetition_index % n_inversions
                bvalue_index = repetition_index if repetition_kind == "bvalue" else 0
                repetition_acquisition = (
                    repetition_index // n_inversions
                    if repetition_kind == "inversion"
                    else (repetition_index if repetition_kind == "acquisition" else 0)
                )
                acquisition_index = (
                    object_acquisition * repetition_acquisitions
                    + repetition_acquisition
                )
                full[
                    :,
                    :,
                    slice_index,
                    inversion_index,
                    echo_index,
                    bvalue_index,
                    acquisition_index,
                    :,
                ] = kspace[:, :, object_index, repetition_index, :]

        all_dims = [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "inversion_idx",
            "echo_idx",
            "bvalue_idx",
            "acquisition_idx",
            "channel_idx",
        ]
        keep = [0, 1, 2]
        keep.extend(axis for axis in range(3, 7) if full.shape[axis] > 1)
        keep.append(7)
        squeezed = np.squeeze(
            full, axis=tuple(axis for axis in range(full.ndim) if axis not in keep)
        )
        return squeezed, [all_dims[axis] for axis in keep]

    def _reconstruct_2d(self, handler, reco_options):
        fid_path = Path(handler.path) / "fid"
        dataset = Dataset(fid_path)
        raw = np.asarray(dataset.data)
        if (
            raw.ndim == 6
            and raw.shape[-1] == 1
            and getattr(dataset, "scheme_id", None) == "dEPI"
            and getattr(dataset, "dim_type", None)
            == [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "k_space_encode_step_2",
            ]
        ):
            # The dEPI reader keeps an explicit unused 2D encoding axis.
            # Object, repetition and receiver axes retain their positions.
            raw = raw[..., 0]
        if raw.ndim != 5:
            raise ValueError(
                "Unsupported Bruker 2D EPI fid layout: expected five axes "
                "(read, phase, object, repetition, channel), or the labelled "
                "dEPI layout with a trailing singleton k_space_encode_step_2; "
                f"got {raw.shape}, dim_type={getattr(dataset, 'dim_type', None)!r}."
            )

        matrix = self._integer_values(handler.matrix, "PVM_Matrix")
        read_size, phase_size = int(matrix[0]), int(matrix[1])
        if raw.shape[0] != read_size:
            raise ValueError(
                f"EPI raw readout length {raw.shape[0]} != matrix size {read_size}."
            )
        method = getattr(handler, "method", None) or {}
        phase_order, phase_positions = self._phase_positions(
            method.get("PVM_EncSteps1"), phase_size, "PVM_EncSteps1"
        )
        if phase_order.size != raw.shape[1]:
            raise ValueError(
                "EPI PVM_EncSteps1 length does not match the decoded fid: "
                f"{phase_order.size} != {raw.shape[1]}."
            )

        n_echoes = self._positive_integer_parameter(
            method.get("PVM_EpiNEchoes", raw.shape[1]), "PVM_EpiNEchoes"
        )
        acquired_echo_index = phase_order % n_echoes
        odd_phase_lines = np.flatnonzero(acquired_echo_index % 2)

        # Undo only the transform the pinned reader actually applied. Its
        # schema resolves RECO_inp_order or the scheme default. Shot-local
        # polarity and sorted-row mirroring differ for interleaved shots.
        raw = raw.copy()
        if dataset._schema.mirror_odd_lines:
            raw[:, 1::2, :, :, :] = raw[::-1, 1::2, :, :, :]
        raw[:, odd_phase_lines, :, :, :] = raw[::-1, odd_phase_lines, :, :, :]

        object_count = int(raw.shape[2])
        # Dataset owns ACQ_obj_order sorting.
        self._object_order(handler, object_count)
        canonical = raw

        canonical, trajectory_attrs = self._apply_trajectory_regridding(
            canonical,
            method,
            reco_options,
            axis=0,
        )

        coefficients = self._phase_correction_coefficients(method, object_count)
        n_shots = int(method.get("PVM_EpiNShots", 1))
        requested_phase_correction = reco_options.get("apply_phase_correction")
        apply_phase_correction = (
            n_shots == 1
            if requested_phase_correction is None
            else bool(requested_phase_correction)
        )
        if apply_phase_correction:
            canonical = self._apply_phase_correction(
                canonical, odd_phase_lines, coefficients
            )

        padded = np.zeros(
            (read_size, phase_size, *canonical.shape[2:]), dtype=canonical.dtype
        )
        padded[:, phase_positions, :, :, :] = canonical
        semantic, dims = self._semantic_2d_layout(handler, padded)
        return (
            semantic,
            dims,
            {
                "raw_source": "fid",
                "epi_shots": n_shots,
                "epi_phase_correction": bool(
                    apply_phase_correction and coefficients is not None
                ),
                "phase_encode_partial_fourier": raw.shape[1] != phase_size,
                **trajectory_attrs,
            },
        )

    def _reconstruct_3d(self, handler, reco_options):
        method = getattr(handler, "method", None) or {}
        acqp = getattr(handler, "acqp", None) or {}
        matrix = self._integer_values(handler.matrix, "PVM_Matrix")
        enc_matrix = self._integer_values(method.get("PVM_EncMatrix"), "PVM_EncMatrix")
        if matrix.size < 3 or enc_matrix.size < 3:
            raise ValueError("3D EPI requires three-dimensional encoding metadata.")
        read_size, phase_size, phase2_size = map(int, matrix[:3])
        acquired_phase, acquired_phase2 = int(enc_matrix[1]), int(enc_matrix[2])
        channels = self._positive_integer_parameter(
            method.get("PVM_EncNReceivers", 1), "PVM_EncNReceivers"
        )
        repetitions = self._positive_integer_parameter(acqp.get("NR", 1), "NR")

        fid = getattr(handler, "fid", None)
        if fid is None or getattr(fid, "size", 0) == 0:
            fid = handler.load_fid_file()
        fid = np.asarray(fid)
        shots = self._positive_integer_parameter(
            method.get("PVM_EpiNShots", 1), "PVM_EpiNShots"
        )
        if acquired_phase % shots:
            raise ValueError(
                "3D EPI acquired phase rows must be divisible by PVM_EpiNShots."
            )
        echoes = acquired_phase // shots
        declared_echoes = self._positive_integer_parameter(
            method.get("PVM_EpiNEchoes", echoes), "PVM_EpiNEchoes"
        )
        if declared_echoes != echoes or int(acqp.get("NI", 1)) != 1:
            raise ValueError(
                "Unsupported 3D EPI echo/object loops; use scan.scanner_data for scanner reconstruction."
            )
        mode = str(acqp.get("AQ_mod", "")).lower()
        if mode not in {"qdig", "dqd", "qsim"}:
            raise ValueError("3D EPI requires declared interleaved quadrature AQ_mod.")
        size = self._integer_values(acqp.get("ACQ_size"), "ACQ_size")
        scan_size = acqp.get("ACQ_scan_size")
        train_echoes = echoes if scan_size == "ACQ_phase_factor_scans" else 1
        if (
            scan_size not in {"One_scan", "ACQ_phase_factor_scans"}
            or size.size != 3
            or size[0] != 2 * read_size * train_echoes
            or size[1] * train_echoes != acquired_phase
            or size[2] != acquired_phase2
        ):
            raise ValueError(
                "3D EPI ACQ_size/ACQ_scan_size does not declare the supported readout train; use scan.scanner_data."
            )
        scalar_dtype = _raw_storage_dtype(acqp)
        logical_samples = read_size * train_echoes * channels
        storage_samples = logical_samples
        block_format = acqp.get("GO_block_size")
        if block_format == "Standard_KBlock_Format":
            block_bytes = logical_samples * 2 * scalar_dtype.itemsize
            storage_samples = (
                ((block_bytes + 1023) // 1024) * 1024 // (2 * scalar_dtype.itemsize)
            )
        elif block_format != "continuous":
            raise ValueError(f"Unsupported 3D EPI GO_block_size={block_format!r}.")
        block_count = repetitions * acquired_phase2 * acquired_phase // train_echoes
        expected = block_count * storage_samples
        if fid.size != expected:
            raise ValueError(
                f"3D EPI fid size does not match declared storage blocks: {fid.size} != {expected}."
            )
        logical = fid.reshape(block_count, storage_samples)[:, :logical_samples]
        if train_echoes > 1:
            acquired = logical.reshape(
                repetitions, acquired_phase2, shots, channels, echoes, read_size
            )
            acquired = acquired.transpose(0, 1, 2, 4, 3, 5).reshape(
                repetitions, acquired_phase2, acquired_phase, channels, read_size
            )
        else:
            acquired = logical.reshape(
                repetitions, acquired_phase2, acquired_phase, channels, read_size
            )
        acquired = acquired.copy()
        odd_rows = np.flatnonzero((np.arange(acquired_phase) % echoes) % 2)
        acquired[:, :, odd_rows, :, :] = acquired[:, :, odd_rows, :, ::-1]
        acquired, trajectory_attrs = self._apply_trajectory_regridding(
            acquired,
            method,
            reco_options,
            axis=-1,
        )

        phase_order, phase_positions = self._phase_positions(
            method.get("PVM_EncSteps1"), phase_size, "PVM_EncSteps1"
        )
        phase2_order, phase2_positions = self._phase_positions(
            method.get("PVM_EncSteps2"), phase2_size, "PVM_EncSteps2"
        )
        if phase_order.size != acquired_phase or phase2_order.size != acquired_phase2:
            raise ValueError("3D EPI encoding-step lengths do not match PVM_EncMatrix.")

        sorted_data = acquired[:, phase2_order, :, :, :]
        sorted_data = sorted_data[:, :, phase_order, :, :]
        sorted_data = sorted_data.transpose(4, 2, 1, 0, 3)
        padded = np.zeros(
            (read_size, phase_size, phase2_size, repetitions, channels),
            dtype=sorted_data.dtype,
        )
        padded[
            np.ix_(
                np.arange(read_size),
                phase_positions,
                phase2_positions,
                np.arange(repetitions),
                np.arange(channels),
            )
        ] = sorted_data

        b_values = np.asarray(getattr(handler, "b_values", [])).reshape(-1)
        fourth_dim = "bvalue_idx" if b_values.size == repetitions else "acquisition_idx"
        return (
            padded,
            [
                "read_idx",
                "phase_idx",
                "slice_idx",
                fourth_dim,
                "channel_idx",
            ],
            {
                "raw_source": "fid",
                "epi_shots": int(method.get("PVM_EpiNShots", 1)),
                "epi_phase_correction": False,
                "phase_encode_partial_fourier": acquired_phase != phase_size,
                "phase2_partial_fourier": acquired_phase2 != phase2_size,
                **trajectory_attrs,
            },
        )

    def reconstruct(self, handler, **reco_options):
        """Reconstruct supported Cartesian 2D/3D EPI acquisitions.

        The 2D reader owns phase/object sorting and reports its sorted-row
        mirroring, which is undone before shot-local echo polarity is applied.
        The dEPI decoder's labelled trailing ``k_space_encode_step_2`` axis is
        removed only when its length is one and its other five axis labels
        match read, phase, object, repetition and channel, in that order.
        Single-slice, single-echo T1 EPI may store its declared inversion loop
        as objects; repetitions then remain separate acquisitions. Other
        supported inversion loops are split from the repetition axis.
        The 3D path decodes declared readout-train padding and retains shot,
        receiver and repetition loops before sorting both phase axes. Slice/echo
        product loops in 2D and extra 3D object loops require validated encoding
        contracts and are rejected with an explicit scanner-data alternative.

        Parameters
        ----------
        handler : BrukerHandler
            A nonempty fid and matrix, encoding steps, acquisition loops and
            geometry. The 3D path requires ACQ_size, ACQ_scan_size, binary format
            and block declarations matching the supported readout trains.
        **reco_options : dict
            ``no_reconstruction`` returns None. ``data_precision`` chooses
            float16/float32 (complex64) or float64 (complex128) storage.
            ``shift_xarray`` applies FOV-offset correction (default True),
            ``fov_shift_sign`` sets its sign (default 1), and
            ``append_parameters`` attaches metadata (default True).
            ``apply_phase_correction`` enables supplied 2D polynomial correction
            (default: single shot). ``apply_trajectory_regridding`` selects
            measured-trajectory interpolation (default: scanner declaration);
            ``trajectory_interpolation`` is linear or cubic (default cubic).
            B0 trajectory correction is not applied. Defaults are resolved by
            the experiment API before this call.

        Returns
        -------
        xarray.DataArray or None
            Complex image intensities in scanner units, ordered read, phase,
            slice, retained inversion/echo/diffusion/acquisition loops, channel.
            Indices and spatial coordinates in mm are supplied by the Cartesian
            finisher. Only encoded spatial axes are inverse transformed.
            Reads and calculations are eager and deterministic; acquired
            samples and handler arrays are not modified or cached here.

        Raises
        ------
        ValueError
            If declared storage, matrix, encoding steps or acquisition loops are
            inconsistent or unsupported. Select scan.scanner_data explicitly to
            use the labelled scanner reconstruction.
        OSError
            If fid cannot be read.
        brukerapi.exceptions.InvalidDataset
            If the 2D reader cannot decode the declared acquisition file.

        Examples
        --------
        >>> image = scan.reconstruct("raw", apply_phase_correction=False)  # doctest: +SKIP
        """
        if reco_options.get("no_reconstruction", False):
            return None
        precision = parse_data_precision(reco_options.get("data_precision", np.float32))
        fid_path = Path(handler.path) / "fid"
        if not fid_path.is_file() or fid_path.stat().st_size <= 0:
            raise ValueError("EPI raw reconstruction requires a non-empty fid file.")

        if str(getattr(handler, "spatial_dimension", "")).upper() == "3D":
            kspace, dims, attrs = self._reconstruct_3d(handler, reco_options)
            spatial_dims = ("read_idx", "phase_idx", "slice_idx")
        else:
            kspace, dims, attrs = self._reconstruct_2d(handler, reco_options)
            spatial_dims = ("read_idx", "phase_idx")

        return self._finish_reconstruction(
            handler,
            kspace,
            dims,
            spatial_dims=spatial_dims,
            shift_xarray=reco_options.get("shift_xarray", True),
            fov_shift_sign=float(reco_options.get("fov_shift_sign", 1.0)),
            data_precision=precision,
            append_parameters=reco_options.get("append_parameters", True),
            description="Bruker EPI reconstruction from fid raw data",
            extra_attrs=attrs,
        )
