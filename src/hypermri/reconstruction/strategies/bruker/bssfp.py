"""Raw reconstruction for Bruker FISP/BSSFP movie acquisitions."""

from pathlib import Path

import numpy as np
from brukerapi.dataset import Dataset

from hypermri.reconstruction.precision import parse_data_precision
from hypermri.utils.system.logging import init_default_logger

from .cartesian import BrukerCartesianRawRecoStrategy

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)

# Multi-echo balanced-SSFP methods whose acquisition objects are echoes. The
# method parameter DelayArrayForTE lists one echo time in ms per object.
MULTI_ECHO_METHODS = frozenset({"<User:nmrsuFISP_H2_Fb>"})


def multi_echo_times_s(method, n_echoes):
    """Return the echo times of a multi-echo bSSFP scan in seconds.

    Parameters
    ----------
    method : mapping
        Parsed Bruker ``method`` file. ``DelayArrayForTE`` holds one echo time
        in ms per acquired echo, in acquisition order.
    n_echoes : int
        Number of echoes the raw data hold (acquisition objects per slice).

    Returns
    -------
    numpy.ndarray
        Float64 echo times in s, shape ``(n_echoes,)``, strictly increasing.

    Raises
    ------
    ValueError
        If ``DelayArrayForTE`` is missing, has another length than
        ``n_echoes``, or is not finite, positive and strictly increasing.
    """
    values = method.get("DelayArrayForTE")
    times_ms = np.asarray([] if values is None else values, dtype=float).reshape(-1)
    if (
        times_ms.size != n_echoes
        or not np.all(np.isfinite(times_ms))
        or np.any(times_ms <= 0)
        or np.any(np.diff(times_ms) <= 0)
    ):
        raise ValueError(
            f"Multi-echo bSSFP method {method.get('Method')!r} needs "
            f"DelayArrayForTE with {n_echoes} positive, increasing echo times "
            f"in ms (one per acquired echo); got {times_ms.tolist()}."
        )
    return times_ms / 1000.0


class BrukerBssfpRawRecoStrategy(BrukerCartesianRawRecoStrategy):
    """Reconstruct Cartesian FISP/BSSFP data with movie and slice loops."""

    def reconstruct(self, handler, **reco_options):
        """Reconstruct Cartesian 2D BSSFP/FISP movie acquisitions.

        The Bruker reader decodes padding and binary format and sorts phase
        steps and acquisition objects once. Movie frames vary fastest inside
        slices, followed by repetitions and receivers. For multi-echo methods
        (``<User:nmrsuFISP_H2_Fb>``) the acquisition objects are echoes
        instead: they become ``echo_idx`` with echo times in s from
        ``DelayArrayForTE``, and one slice is required.
        Only read and phase axes are inverse transformed; missing phase rows
        are zero filled. The acquired arrays are never modified.

        Parameters
        ----------
        handler : BrukerHandler
            Scan metadata and a nonempty ``fid`` file. Matrix, encoding steps,
            object counts and physical geometry must describe the acquisition.
        **reco_options : dict
            ``no_reconstruction`` returns None. ``data_precision`` chooses
            float16/float32 (complex64) or float64 (complex128) storage.
            ``shift_xarray`` enables FOV-offset correction (default True);
            ``fov_shift_sign`` is its sign (default 1). ``append_parameters``
            attaches scanner metadata (default True). Strategy defaults are
            resolved by the experiment API before this call.

        Returns
        -------
        xarray.DataArray or None
            Complex image intensities in scanner units. Dimensions follow
            read, phase and the documented acquisition loops (``echo_idx``
            in place of ``acquisition_idx`` for multi-echo methods), with index and
            spatial coordinates in mm provided by the Cartesian finisher.
            Reconstruction is eager, deterministic and does not cache raw data.

        Raises
        ------
        ValueError
            If geometry, encoding, object order or raw layout is inconsistent.
        OSError
            If the acquisition file cannot be read.
        brukerapi.exceptions.InvalidDataset
            If the declared storage layout does not match the file.

        Examples
        --------
        >>> image = scan.reconstruct("raw", data_precision="float64")  # doctest: +SKIP
        """
        if reco_options.get("no_reconstruction", False):
            return None
        precision = parse_data_precision(reco_options.get("data_precision", np.float32))

        if str(getattr(handler, "spatial_dimension", "")).upper() == "3D":
            raise ValueError(
                "3D BSSFP raw reconstruction requires a validated phase/echo "
                "loop order and three-dimensional Fourier convention. "
                "Use scan.reconstruct('seq2d') or scan.plot(scan.scanner_data) "
                "for the labelled scanner reconstruction."
            )

        fid_path = Path(handler.path) / "fid"
        if not fid_path.is_file() or fid_path.stat().st_size <= 0:
            raise ValueError("BSSFP raw reconstruction requires a non-empty fid file.")

        dataset = Dataset(fid_path)
        raw = np.asarray(dataset.data)
        if raw.ndim != 5:
            raise ValueError(
                "Unsupported Bruker BSSFP fid layout: expected five axes "
                f"(read, phase, object, repetition, channel), got {raw.shape}."
            )

        matrix = self._integer_values(handler.matrix, "PVM_Matrix")
        if matrix.size < 3:
            raise ValueError(
                f"Invalid Bruker PVM_Matrix for BSSFP reconstruction: {matrix}."
            )
        read_size, phase_size = int(matrix[0]), int(matrix[1])
        if raw.shape[0] != read_size:
            raise ValueError(
                "BSSFP raw readout length does not match PVM_Matrix: "
                f"{raw.shape[0]} != {read_size}."
            )

        method = getattr(handler, "method", None) or {}
        phase_order, phase_positions = self._phase_positions(
            method.get("PVM_EncSteps1"), phase_size, "PVM_EncSteps1"
        )
        if phase_order.size != raw.shape[1]:
            raise ValueError(
                "BSSFP PVM_EncSteps1 length does not match the decoded fid: "
                f"{phase_order.size} != {raw.shape[1]}."
            )

        movie_frames = self._positive_integer_parameter(
            method.get("PVM_NMovieFrames", 1), "PVM_NMovieFrames"
        )
        n_slices = self._positive_integer_parameter(
            getattr(handler, "n_slices", 1), "PVM_SPackArrNSlices"
        )
        n_repetitions = int(raw.shape[3])
        n_channels = int(raw.shape[4])
        object_count = int(raw.shape[2])
        loop_dim, echo_times_s = "acquisition_idx", None
        if method.get("Method") in MULTI_ECHO_METHODS:
            if n_slices != 1 or movie_frames != 1:
                raise ValueError(
                    f"Multi-echo bSSFP raw reconstruction of {method.get('Method')!r} "
                    "is verified for one slice without movie frames; got "
                    f"{n_slices} slice(s) and {movie_frames} movie frame(s). Use "
                    "scan.reconstruct('seq2d') for this scan."
                )
            # One object per echo; the echo is the loop that replaces movie frames.
            echo_times_s = multi_echo_times_s(method, object_count)
            loop_dim, movie_frames = "echo_idx", object_count
        if object_count != movie_frames * n_slices:
            raise ValueError(
                "BSSFP object count does not match movie/slice metadata: "
                f"{object_count} != {movie_frames} * {n_slices}."
            )

        # Dataset owns phase and ACQ_obj_order sorting. Validate the declared
        # permutation without applying it to the already-canonical object axis.
        self._object_order(handler, object_count)
        canonical = raw

        kspace = np.zeros(
            (
                read_size,
                phase_size,
                movie_frames,
                n_slices,
                n_repetitions,
                n_channels,
            ),
            dtype=canonical.dtype,
        )
        for object_index in range(object_count):
            movie_index = object_index % movie_frames
            slice_index = object_index // movie_frames
            kspace[:, phase_positions, movie_index, slice_index, :, :] = canonical[
                :, :, object_index, :, :
            ]

        image = self._finish_reconstruction(
            handler,
            kspace,
            [
                "read_idx",
                "phase_idx",
                loop_dim,
                "slice_idx",
                "repetition_idx",
                "channel_idx",
            ],
            spatial_dims=("read_idx", "phase_idx"),
            shift_xarray=reco_options.get("shift_xarray", True),
            fov_shift_sign=float(reco_options.get("fov_shift_sign", 1.0)),
            data_precision=precision,
            append_parameters=reco_options.get("append_parameters", True),
            description="Bruker FISP/BSSFP reconstruction from fid raw data",
            extra_attrs={
                "raw_source": "fid",
                "phase_encode_partial_fourier": raw.shape[1] != phase_size,
            },
        )
        if echo_times_s is not None:
            image = image.assign_coords(
                echo_idx=("echo_idx", echo_times_s, {"unit": "s"})
            )
        return image
