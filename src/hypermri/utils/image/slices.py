"""Match acquired slice positions without plotting or constructing a viewer.

Matching uses the existing slice_idx axis_mm metadata and nearest-position
policy; no geometry/orientation correction or interpolation is performed.
"""

import numpy as np


def get_overlapping_slices(
    data_obj_1: object | None = None,
    data_obj_2: object | None = None,
    tol_mm: float | int = 0.1,
) -> tuple[list[int], list[int]]:
    """Find nearest overlapping slice centres using existing axis_mm metadata.

    Parameters
    ----------
    data_obj_1 : object, optional
        First xarray DataArray/Dataset with slice_idx.attrs['axis_mm'], or a
        scan exposing it through data_array or reco_data. axis_mm contains a
        nonempty one-dimensional sequence of slice-centre positions in mm.
        A scan prefers valid scanner_data metadata and falls back to data;
        accessing those properties retains their existing lazy-loading effects.
    data_obj_2 : object, optional
        Second source with the same metadata contract and coordinate orientation.
        Both sources are required despite the historical None defaults.
    tol_mm : float or int, optional
        Strict absolute distance threshold in mm. The existing algorithm uses
        its absolute value; zero gives no matches, including identical centres.

    Returns
    -------
    tuple of list of int
        First-source indices in input order and their nearest second-source
        indices. Ties select the first second-source index; multiple first
        indices may map to one second index. No input arrays or metadata are
        modified, and no plotting or UI is initialized.

    Raises
    ------
    ValueError
        If the tolerance is not a Python float/int or nonempty slice-position
        metadata cannot be obtained from either source.

    Notes
    -----
    The caller must establish compatible physical coordinate orientation.
    This deterministic matcher does not check orientation, reorient images or
    infer positions from voxel-index coordinates. Existing matching semantics
    and metadata precedence are preserved.

    Examples
    --------
    >>> import xarray as xr
    >>> first = xr.DataArray([0, 0], dims="slice_idx", coords={"slice_idx": [0, 1]})
    >>> first.slice_idx.attrs["axis_mm"] = [0.0, 5.0]
    >>> second = first.copy(deep=True)
    >>> get_overlapping_slices(first, second)
    ([0, 1], [0, 1])
    """
    import xarray as xr

    def _extract_axis_mm(scan: object | None, obj_name: str) -> np.ndarray:
        def _axis_from_xarray(
            xr_obj: object | None,
            source_name: str,
        ) -> np.ndarray | None:
            if xr_obj is None:
                return None
            if not isinstance(xr_obj, (xr.DataArray, xr.Dataset)):
                return None
            if "slice_idx" not in xr_obj.coords:
                return None
            axis_mm = xr_obj.slice_idx.attrs.get("axis_mm")
            if axis_mm is None:
                return None
            axis_mm = np.asarray(axis_mm)
            if axis_mm.size == 0:
                location = f"{obj_name}.{source_name}" if source_name else obj_name
                msg = f"{location}.slice_idx.attrs['axis_mm'] exists but is empty."
                raise ValueError(msg)
            return axis_mm

        # Direct xarray input
        axis_mm = _axis_from_xarray(scan, "")
        if axis_mm is not None:
            return axis_mm

        # Common MRIExperiment-like objects may expose both; prefer a valid scanner_data,
        # but robustly fall back to data when scanner_data is missing/incomplete.
        for attr_name in ("scanner_data", "data"):
            axis_mm = _axis_from_xarray(getattr(scan, attr_name, None), attr_name)
            if axis_mm is not None:
                return axis_mm

        msg = (
            f"Could not extract non-empty slice coordinates from {obj_name}. "
            "Expected an xarray object with coord 'slice_idx' and "
            "slice_idx.attrs['axis_mm'], or an object exposing this via "
            "'scanner_data' or 'data'."
        )
        raise ValueError(msg)

    if not (isinstance(tol_mm, float) or isinstance(tol_mm, int)):
        raise ValueError("diff_mm has to be float or integer!")
    tol_mm = np.abs(tol_mm)
    # --- 1. Extract axis_mm robustly from xarray objects or MRIExperiment-like wrappers ----
    data_obj_1_slice_coords = _extract_axis_mm(data_obj_1, "data_obj_1")
    data_obj_2_slice_coords = _extract_axis_mm(data_obj_2, "data_obj_2")

    # --- 2. Compare the axis_mm to find overlapping slices ---
    slice_map = {}  # dce_index -> t1_index
    for i_1, c_1 in enumerate(data_obj_1_slice_coords):
        diffs = np.abs(data_obj_2_slice_coords - c_1)
        i_2 = np.argmin(diffs)
        if diffs[i_2] < tol_mm:
            # slice_map keys are dce index, slice_map values are t1 index
            slice_map[i_1] = int(i_2)

    # --- 3. Extract the indices to keep, ensuring they stay in the same relative order ---
    data_obj_1_indices = sorted(slice_map.keys())
    data_obj_2_indices = [slice_map[idx] for idx in data_obj_1_indices]
    return data_obj_1_indices, data_obj_2_indices
