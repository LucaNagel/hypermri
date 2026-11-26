"""Raw reconstruction for Bruker IntraGate FLASH ``rawdata.job0`` data."""

from pathlib import Path

import numpy as np

from hypermri.reconstruction.precision import (
    convert_complex_for_storage,
    parse_data_precision,
)
from hypermri.utils.system.logging import init_default_logger

from .cartesian import BrukerCartesianRawRecoStrategy

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BrukerIgflashRawRecoStrategy(BrukerCartesianRawRecoStrategy):
    """Reconstruct repeated segmented IgFLASH profiles from ``rawdata.job0``.

    ParaVision's physiological IntraGate bin order is generated dynamically
    from ``rawdata.job1`` and is not persisted in the experiment. The raw
    reconstruction therefore returns reproducible oversample-modulo movie
    bins. This preserves every acquired profile and yields the requested movie
    shape without pretending that retrospective physiological gating was
    available.
    """

    def reconstruct(self, handler, **reco_options):
        """Reconstruct deterministic IgFLASH movie bins from acquired profiles.

        Parameters
        ----------
        handler : BrukerHandler
            Experiment with a non-empty ``rawdata.job0``, matrix, slice order,
            segment count, encoding positions and requested movie frame count.
            Spatial geometry is expressed in mm.
        **reco_options
            Existing reconstruction options: ``no_reconstruction`` skips work;
            ``data_precision`` defaults to float32 storage; ``shift_xarray``
            defaults to True; ``fov_shift_sign`` defaults to +1; and
            ``append_parameters`` defaults to True. Acquired profiles are first
            represented as complex64, preserving the established calculation
            policy independently of the requested output storage precision.

        Returns
        -------
        xarray.DataArray or None
            Complex image in read/phase/acquisition/slice/channel order, where
            acquisition indexes movie bins. Values have arbitrary scanner units
            and physical coordinates use mm. Float16/float32 store complex64;
            float64 stores complex128. None is returned only when work is skipped.

        Raises
        ------
        ValueError
            If the payload, segmentation, matrix, movie count, geometry or
            precision is invalid or inconsistent.
        OSError
            If acquired profiles cannot be read.

        Notes
        -----
        The eager reconstruction averages oversamples into reproducible modulo
        bins without physiological gating. Acquired samples are not mutated and
        intermediate arrays are not stored on the handler. A missing acquired
        payload is loaded through the handler.

        Examples
        --------
        >>> image = strategy.reconstruct(handler, data_precision="float64")  # doctest: +SKIP
        """
        if reco_options.get("no_reconstruction", False):
            return None
        precision = parse_data_precision(reco_options.get("data_precision", np.float32))

        raw_path = Path(handler.path) / "rawdata.job0"
        if not raw_path.is_file() or raw_path.stat().st_size <= 0:
            raise ValueError(
                "IgFLASH raw reconstruction requires a non-empty rawdata.job0 file."
            )

        method = getattr(handler, "method", None) or {}
        matrix = self._integer_values(handler.matrix, "PVM_Matrix")
        if matrix.size < 3:
            raise ValueError("IgFLASH requires a three-value PVM_Matrix.")
        read_size, phase_size = int(matrix[0]), int(matrix[1])
        n_slices = self._positive_integer_parameter(
            getattr(handler, "n_slices", matrix[2]), "PVM_SPackArrNSlices"
        )
        n_segments = self._positive_integer_parameter(
            method.get("Nsegments", 1), "Nsegments"
        )
        movie_frames = self._positive_integer_parameter(
            method.get("PVM_NMovieFrames", 1), "PVM_NMovieFrames"
        )

        steps = self._integer_values(method.get("PVM_EncSteps1"), "PVM_EncSteps1")
        if steps.size != phase_size or phase_size % n_segments:
            raise ValueError(
                "IgFLASH phase encoding does not match PVM_Matrix/Nsegments: "
                f"{steps.size} steps, matrix {phase_size}, segments {n_segments}."
            )
        phase_positions = steps + phase_size // 2
        if (
            np.unique(phase_positions).size != phase_size
            or np.any(phase_positions < 0)
            or np.any(phase_positions >= phase_size)
        ):
            raise ValueError("Invalid IgFLASH PVM_EncSteps1 phase positions.")
        lines_per_segment = phase_size // n_segments

        raw = getattr(handler, "rawdatajob0", None)
        if raw is None or getattr(raw, "size", 0) == 0:
            raw = handler.load_rawdatajob0_file()
        raw = np.asarray(raw)
        if raw.size % read_size:
            raise ValueError(
                f"IgFLASH rawdata.job0 size {raw.size} is not divisible by {read_size}."
            )
        profiles = raw.size // read_size
        profiles_per_oversample = n_segments * lines_per_segment * n_slices
        if profiles % profiles_per_oversample:
            raise ValueError(
                "IgFLASH profile count does not match segment/phase/slice metadata: "
                f"{profiles} is not divisible by {profiles_per_oversample}."
            )
        oversamples = profiles // profiles_per_oversample
        if oversamples < movie_frames or oversamples % movie_frames:
            raise ValueError(
                "IgFLASH repetitions cannot be divided into movie frames: "
                f"{oversamples} repetitions for {movie_frames} frames."
            )

        # rawdata.job0 stores each read profile contiguously, followed by
        # segment, oversample, line, and acquired-slice loops (G-O-L-S).
        acquired = convert_complex_for_storage(raw.reshape(profiles, read_size).T)
        acquired = acquired.reshape(
            read_size,
            n_segments,
            oversamples,
            lines_per_segment,
            n_slices,
        )
        object_order = self._object_order(handler, n_slices)
        canonical = np.empty_like(acquired)
        canonical[..., object_order] = acquired

        averages_per_frame = oversamples // movie_frames
        framed = canonical.reshape(
            read_size,
            n_segments,
            averages_per_frame,
            movie_frames,
            lines_per_segment,
            n_slices,
        ).mean(axis=2)

        kspace = np.zeros(
            (read_size, phase_size, movie_frames, n_slices, 1),
            dtype=framed.dtype,
        )
        positions = phase_positions.reshape(n_segments, lines_per_segment)
        for segment in range(n_segments):
            for line_index, phase_index in enumerate(positions[segment]):
                kspace[:, phase_index, :, :, 0] = framed[:, segment, :, line_index, :]

        return self._finish_reconstruction(
            handler,
            kspace,
            [
                "read_idx",
                "phase_idx",
                "acquisition_idx",
                "slice_idx",
                "channel_idx",
            ],
            spatial_dims=("read_idx", "phase_idx"),
            shift_xarray=reco_options.get("shift_xarray", True),
            fov_shift_sign=float(reco_options.get("fov_shift_sign", 1.0)),
            data_precision=precision,
            append_parameters=reco_options.get("append_parameters", True),
            description="Bruker IgFLASH reconstruction from rawdata.job0",
            extra_attrs={
                "raw_source": "rawdata.job0",
                "raw_frame_binning": "oversample_modulo",
                "physiological_gating_applied": False,
                "raw_oversamples": int(oversamples),
                "raw_averages_per_frame": int(averages_per_frame),
            },
        )
