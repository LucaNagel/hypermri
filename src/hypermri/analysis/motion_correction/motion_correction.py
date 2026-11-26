from __future__ import annotations

from collections.abc import Callable, Mapping

import xarray as xr

from .simpleitk import simpleitk_dce_motion_correction
from .skimage import skimage_dce_motion_correction


def dce_motion_correction(
    data: xr.DataArray,
    *,
    correction_method: str = "skimage",
    return_metadata: bool = False,
    **kwargs,
) -> xr.DataArray | tuple[xr.DataArray, dict[str, object]]:
    """
    Dispatch DCE motion correction to a configured backend.

    Supported backends are ``"skimage"`` and ``"simpleitk"``.
    Backends with optional third-party dependencies raise a clear
    ``ImportError`` when selected but not installed.
    """
    method = str(correction_method).lower()
    if method == "skimage":
        return skimage_dce_motion_correction(
            data,
            return_metadata=return_metadata,
            **kwargs,
        )
    if method == "simpleitk":
        return simpleitk_dce_motion_correction(
            data,
            return_metadata=return_metadata,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported correction_method '{correction_method}'. "
        "Currently supported: 'skimage', 'simpleitk'."
    )


def apply_motion_corrector(
    data: xr.DataArray,
    motion_corrector,
) -> xr.DataArray:
    """
    Apply a user-supplied motion-correction specification to a DCE series.

    Parameters
    ----------
    data
        Dynamic DCE data stored as an ``xarray.DataArray``.
    motion_corrector
        Either a callable that accepts ``data``, a backend name such as
        ``"skimage"`` or ``"simpleitk"``, or a mapping containing
        ``dce_motion_correction`` keyword arguments.
    """
    if motion_corrector is None:
        return data
    if callable(motion_corrector):
        return motion_corrector(data)
    if isinstance(motion_corrector, str):
        return dce_motion_correction(
            data,
            correction_method=motion_corrector,
        )
    if isinstance(motion_corrector, Mapping):
        parameters = dict(motion_corrector)
        correction_method = parameters.pop("correction_method", "skimage")
        return dce_motion_correction(
            data,
            correction_method=correction_method,
            **parameters,
        )

    raise TypeError(
        "motion_corrector must be callable, a correction method string, or a mapping "
        "of dce_motion_correction keyword arguments."
    )


motion_corrector = apply_motion_corrector
