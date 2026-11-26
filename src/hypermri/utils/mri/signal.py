"""Signal processing utilities for MRI data."""

from typing import Optional, Union

import numpy as np
import xarray as xr


def zero_fill_temporal(
    values: np.ndarray, additional_points: int, axis: int = -1
) -> np.ndarray:
    """Append zero samples along one temporal axis without changing precision.

    Parameters
    ----------
    values : numpy.ndarray
        Temporal samples with any independent leading or trailing dimensions.
    additional_points : int
        Nonnegative number of zero samples appended after the acquired samples.
    axis : int, optional
        Temporal sampling axis, defaulting to the final axis.

    Returns
    -------
    numpy.ndarray
        New padded array, preserving dtype and sample order. Other axes retain
        their sizes. No units, coordinate labels or timing metadata are inferred.

    Raises
    ------
    ValueError
        If the padding count is negative.
    IndexError
        If the axis is outside the input dimensions.

    Notes
    -----
    Evaluation is eager, deterministic and does not mutate the input.
    """
    padding = [(0, 0)] * values.ndim
    padding[axis] = (0, additional_points)
    return np.pad(values, padding, mode="constant")


def rotate_phase(
    values: np.ndarray | xr.DataArray, phase_rad: float | np.ndarray | xr.DataArray
) -> np.ndarray | xr.DataArray:
    """Multiply complex signals by a positive phase rotation.

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Real or complex samples in arbitrary signal units.
    phase_rad : float, numpy.ndarray or xarray.DataArray
        Rotation in radians. NumPy shapes broadcast positionally; labelled
        xarray inputs align by dimension names using normal xarray arithmetic.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        ``values * exp(1j * phase_rad)`` with unchanged signal units, under
        NumPy/xarray's dtype and metadata policy. No phase is estimated here.
        Evaluation is eager and deterministic, without input mutation.

    Raises
    ------
    ValueError
        If input shapes or labelled coordinates cannot align.
    """
    return values * np.exp(1j * phase_rad)


def phase_correct_zero_order(
    data: xr.DataArray,
    dynamic_dim: str = "inversion_idx",
    ref_idx: int = -1,
    use_magnitude_max: bool = False,
) -> xr.DataArray:
    """Align a complex dynamic signal to a selected reference phase.

    Parameters
    ----------
    data : xarray.DataArray
        Complex samples with dynamic_dim and any independent labelled dimensions,
        in arbitrary signal units. Noncomplex inputs return unchanged.
    dynamic_dim : str, optional
        Dimension along which the reference is selected, such as inversion_idx.
    ref_idx : int, optional
        Reference sample index when use_magnitude_max=False; -1 selects the last.
    use_magnitude_max : bool, optional
        Select the largest magnitude separately for each independent signal instead
        of using one fixed dynamic index.

    Returns
    -------
    xarray.DataArray
        Samples multiplied by the negative reference phase, preserving dimension
        order, coordinates and signal units under NumPy/xarray precision and attrs
        policy. Reference samples become positive real up to numerical roundoff.
        Global attrs follow ordinary xarray arithmetic and may be omitted.

    Raises
    ------
    ValueError, KeyError
        If a complex input lacks the requested dynamic dimension.
    IndexError
        If the fixed reference index is outside the dynamic axis.

    Notes
    -----
    Input values are materialized eagerly to check complex dtype. Phase extraction
    and rotation are deterministic and uncached; no acquired values or coordinates
    are mutated. Correction-factor coordinates are dropped before broadcasting so
    multidimensional input coordinates remain attached to the result.
    """
    if not np.iscomplexobj(data.values):
        return data

    if use_magnitude_max:
        # Find index of maximum magnitude along the dynamic dimension
        max_idx = np.abs(data).argmax(dim=dynamic_dim)
        ref_data = data.isel({dynamic_dim: max_idx})
    else:
        # Use a fixed index (e.g., the last point) as reference
        ref_data = data.isel({dynamic_dim: ref_idx})

    # Use xr.apply_ufunc to ensure we get a DataArray back with correct dimensions
    # np.angle can return a numpy array if called directly on a DataArray
    phase_map = xr.apply_ufunc(np.angle, ref_data)

    # Calculate the complex correction factor
    # We use exp(1j * -phase) to rotate the signal to the real axis
    correction = xr.apply_ufunc(lambda phase: rotate_phase(1.0, -phase), phase_map)

    # Drop coordinates from the correction factor to avoid conflicts
    # with the original data's coordinates during broadcasting.
    # This ensures that coordinates like 'inversion_idx' remain 2D in the output.
    correction = correction.drop_vars(correction.coords.keys())

    # xarray now aligns by dimension NAME and broadcasts the correction
    # across the missing dynamic_dim while preserving original coordinates.
    return data * correction
