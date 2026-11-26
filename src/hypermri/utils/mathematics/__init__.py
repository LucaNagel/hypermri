"""Re-export existing mathematical utilities through explicit imports.

Implementations, units, precision and numerical behavior belong to calculations.
The existing __all__ and additional directly accessible names are preserved;
importing performs no numerical operation or data read.
"""

from .calculations import (
    apply_gaussian_smoothing,
    calc_sampling_time_axis,
    calculate_fwhm,
    compute_weighted_mean_and_std,
    find_max_and_index,
    gaussian,
    get_histogram,
    init_default_logger,
    logger,
    moving_average,
    np,
    part_func,
    polynomial_fit,
    xr,
)

__all__ = [
    # Mathematical calculations
    "gaussian",
    "get_histogram",
    "compute_weighted_mean_and_std",
    "find_max_and_index",
    "moving_average",
    "polynomial_fit",
    "apply_gaussian_smoothing",
]
