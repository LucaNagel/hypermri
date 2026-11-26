"""Vendor-neutral description of one chemical shift imaging (CSI) acquisition.

A vendor reader fills :class:`CSIAcquisition` once from its scanner
parameters, for Bruker :func:`hypermri.vendors.bruker_acquisition.bruker_to_csi_acquisition`.
Code outside the vendor readers then works with these named physical values
instead of reading scanner parameter names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _read_only_positions(values, name: str) -> np.ndarray:
    """Return a read-only one-dimensional int64 copy of grid positions."""
    array = np.asarray(values)
    if (
        array.ndim != 1
        or not array.size
        or np.iscomplexobj(array)
        or not np.all(np.isfinite(array))
        or np.any(array != np.floor(array))
    ):
        raise ValueError(
            f"CSIAcquisition.{name} must be a non-empty one-dimensional array "
            f"of integer grid positions; got shape {array.shape}."
        )
    array = array.astype(np.int64, copy=True)
    array.flags.writeable = False
    return array


def _positive_pair(values, name: str, kind: type) -> tuple:
    """Return two positive finite values as a tuple of ``kind``."""
    pair = tuple(np.asarray(values, dtype=float).reshape(-1))
    if len(pair) != 2 or not all(np.isfinite(pair)) or min(pair) <= 0:
        raise ValueError(
            f"CSIAcquisition.{name} needs two positive finite values "
            f"(read, phase); got {values!r}."
        )
    if kind is int and any(value != int(value) for value in pair):
        raise ValueError(f"CSIAcquisition.{name} must hold integers; got {values!r}.")
    return tuple(kind(value) for value in pair)


@dataclass(frozen=True)
class CSIAcquisition:
    """Acquisition plan and physical scales of one 2D CSI scan.

    The object describes how the acquired FIDs map onto the read/phase
    encoding grid and which physical values label that grid and the spectral
    axis. It holds no signal samples. All values are validated and the
    position arrays are read-only copies, so an instance cannot change after
    construction.

    Parameters
    ----------
    encoding_order : str
        Name of the order in which the scanner visited the encoding grid,
        for example ``"LINEAR_ENC LINEAR_ENC"``, ``"CENTRIC_ENC CENTRIC_ENC"``
        or ``"luca_centric"``. The positions below already encode the order;
        the name lets a reconstruction refuse orders without a verified
        sorter.
    matrix : tuple of int
        Number of encoding steps ``(read, phase)``; both at least 1.
    fov_mm : tuple of float or None
        Field of view ``(read, phase)`` in millimetres; both positive.
        ``None`` when the scan does not record it. Readers that need physical
        distances must then take them from elsewhere (for example an explicit
        voxel spacing) or fail with an error naming ``fov_mm``.
    fov_offset_mm : tuple of float
        Offset of the field-of-view centre from the scanner isocentre
        ``(read, phase, slice)`` in millimetres, in the scanner's
        read/phase/slice frame.
    read_positions : numpy.ndarray
        One zero-based read grid position per acquired FID, in acquisition
        order, shape ``(n_fids,)``. Stored as read-only int64.
    phase_positions : numpy.ndarray
        Phase grid position of the same FID, same shape and storage as
        ``read_positions``.
    n_points : int
        Number of complex samples per acquired FID before any cut-off; at
        least 1.
    dwell_time_s : float or None
        Time between two samples of one FID in seconds; ``None`` when the
        scan does not record it, in which case the sampling axis falls back to
        sample indices.
    reference_frequency_mhz : float or None
        Spectrometer reference frequency in MHz used to convert Hz to ppm;
        ``None`` when the scan does not record it.
    center_ppm : float
        Chemical shift in ppm at the centre of the spectral window.

    Raises
    ------
    ValueError
        If a position is outside the grid, the two position arrays differ in
        length, a count or size is not positive, or a physical value is not
        finite. The message names the field.

    Examples
    --------
    A 3 x 2 grid acquired with read as the inner loop:

    >>> acquisition = CSIAcquisition(
    ...     encoding_order="LINEAR_ENC LINEAR_ENC",
    ...     matrix=(3, 2),
    ...     fov_mm=(30.0, 20.0),
    ...     fov_offset_mm=(0.0, 0.0, 0.0),
    ...     read_positions=[0, 1, 2, 0, 1, 2],
    ...     phase_positions=[0, 0, 0, 1, 1, 1],
    ...     n_points=256,
    ...     dwell_time_s=1e-4,
    ...     reference_frequency_mhz=75.4,
    ...     center_ppm=171.0,
    ... )
    >>> acquisition.n_fids
    6
    """

    encoding_order: str
    matrix: tuple[int, int]
    fov_mm: tuple[float, float] | None
    fov_offset_mm: tuple[float, float, float]
    read_positions: np.ndarray
    phase_positions: np.ndarray
    n_points: int
    dwell_time_s: float | None
    reference_frequency_mhz: float | None
    center_ppm: float

    def __post_init__(self) -> None:
        """Validate every field and store normalized, immutable copies."""
        if not isinstance(self.encoding_order, str) or not self.encoding_order:
            raise ValueError(
                "CSIAcquisition.encoding_order must be a non-empty text; "
                f"got {self.encoding_order!r}."
            )
        matrix = _positive_pair(self.matrix, "matrix", int)
        fov_mm = (
            None
            if self.fov_mm is None
            else _positive_pair(self.fov_mm, "fov_mm", float)
        )
        offset = tuple(float(v) for v in np.asarray(self.fov_offset_mm).reshape(-1))
        if len(offset) != 3 or not all(np.isfinite(offset)):
            raise ValueError(
                "CSIAcquisition.fov_offset_mm needs three finite values "
                f"(read, phase, slice); got {self.fov_offset_mm!r}."
            )
        read = _read_only_positions(self.read_positions, "read_positions")
        phase = _read_only_positions(self.phase_positions, "phase_positions")
        if read.shape != phase.shape:
            raise ValueError(
                "CSIAcquisition.read_positions and phase_positions must have "
                f"one entry per FID each; got {read.size} and {phase.size}."
            )
        for name, positions, size in (
            ("read_positions", read, matrix[0]),
            ("phase_positions", phase, matrix[1]),
        ):
            if np.any((positions < 0) | (positions >= size)):
                raise ValueError(
                    f"CSIAcquisition.{name} must lie in 0..{size - 1} for the "
                    f"matrix {matrix}."
                )
        n_points = int(self.n_points)
        if n_points != self.n_points or n_points < 1:
            raise ValueError(
                f"CSIAcquisition.n_points must be a positive integer; got {self.n_points!r}."
            )
        optional = {}
        for name in ("dwell_time_s", "reference_frequency_mhz"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"CSIAcquisition.{name} must be positive and finite or None; "
                        f"got {getattr(self, name)!r}."
                    )
            optional[name] = value
        center_ppm = float(self.center_ppm)
        if not np.isfinite(center_ppm):
            raise ValueError(
                f"CSIAcquisition.center_ppm must be finite; got {self.center_ppm!r}."
            )
        for name, value in (
            ("matrix", matrix),
            ("fov_mm", fov_mm),
            ("fov_offset_mm", offset),
            ("read_positions", read),
            ("phase_positions", phase),
            ("n_points", n_points),
            ("center_ppm", center_ppm),
            *optional.items(),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_fids(self) -> int:
        """Number of acquired FIDs per slice, repetition and channel.

        Returns
        -------
        int
            Length of ``read_positions``; positions visited twice count twice.
        """
        return int(self.read_positions.size)


__all__ = ["CSIAcquisition"]
