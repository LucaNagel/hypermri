"""Convert signed scanner encoding steps to centred reconstruction indices."""

from __future__ import annotations

import numpy as np


def _encoding_step_positions(
    steps, matrix_size: int, parameter_name: str
) -> np.ndarray:
    """Map finite integer steps onto the full matrix, retaining acquired order.

    Each signed step is relative to matrix index ``matrix_size // 2``.
    Repeated positions are permitted here for paired CSI grids; Cartesian
    callers validate uniqueness. Invalid steps or positions raise ValueError.
    The input is not modified and returned positions are integer indices.
    """
    try:
        values = np.asarray(steps, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid Bruker {parameter_name}: expected finite integer values."
        ) from error
    if (
        not values.size
        or not np.all(np.isfinite(values))
        or not np.all(values == np.rint(values))
    ):
        raise ValueError(
            f"Invalid Bruker {parameter_name}: expected finite integer values."
        )
    centre = int(matrix_size) // 2
    if np.any(values < -centre) or np.any(values >= int(matrix_size) - centre):
        raise ValueError(
            f"Invalid Bruker {parameter_name}: encoded positions fall outside "
            f"matrix size {matrix_size}."
        )
    return values.astype(int) + centre
