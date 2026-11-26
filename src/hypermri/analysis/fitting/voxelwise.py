"""Execute deterministic voxel fits and assemble labelled parameter maps.

Model callbacks own scientific calculations and units. The shared loop retains
non-dynamic dimension order and compatible coordinates, with no result cache.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import product

import numpy as np
import xarray as xr

from .execution import ExecutionConfig, run_indexed_tasks


class VoxelwiseExecutor:
    """Run a model callback once per voxel and assemble labelled fit results.

    The executor has no configuration or cached state. All non-dynamic input
    dimensions identify voxels, in their original order. Serial and joblib
    execution use the same callback and result assembly. Numerical assumptions,
    units, success criteria and uncertainty belong to the supplied model.

    Notes
    -----
    Calls are eager. Determinism requires a deterministic callback. Acquired
    input values are preserved; callbacks must also preserve their inputs.

    Examples
    --------
    >>> executor = VoxelwiseExecutor()
    """

    def execute(
        self,
        data: xr.DataArray,
        model_id: str,
        fit_params: dict | xr.Dataset,
        single_voxel_fit_func: callable,
        output_param_names: Sequence[str],
        dynamic_dim: str,
        model_name: str | None = None,
        model_equation: str | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """Fit each voxel and attach the existing model reconstruction metadata.

        Parameters
        ----------
        data : xarray.DataArray
            Acquired signal with one named dynamic dimension and any number of
            voxel dimensions. Units and dynamic coordinate units are defined by
            the callback. Real floating input is promoted to float64 for fitting;
            complex input retains its storage dtype. Values are not modified.
        model_id : str
            Existing model identifier. It selects compatibility coordinate
            injection, output dtypes and reconstruction metadata, not a fitter.
        fit_params : dict or xarray.Dataset
            Scientific and execution options. Dataset attributes are copied.
            Floating NumPy arrays in a supplied dictionary may be replaced by
            float64 copies. ``use_multiprocessing`` and ``number_of_cpu_cores``
            configure the shared dispatcher; model options go to the callback.
        single_voxel_fit_func : callable
            Called with ``signal``, ``dynamic_coords``, ``fit_params`` and
            ``indices`` keywords. Return a result dictionary (or None) and an
            index tuple. Signal and dynamic coordinates have shape ``(n_time,)``;
            indices follow the non-dynamic input dimension order. Unknown/bulky
            result fields are discarded. Callback exceptions propagate.
        output_param_names : Sequence[str]
            Ordered parameter names retained in the dataset. Missing numeric
            results remain NaN. The callback owns parameter units and meanings.
        dynamic_dim : str
            Input dimension and coordinate indexing the fitted samples.
        model_name : str or None, optional
            Model name recorded in result attributes.
        model_equation : str or None, optional
            Human-readable equation recorded in result attributes.
        **kwargs
            Ignored legacy compatibility arguments; not forwarded to the model.

        Returns
        -------
        xarray.Dataset
            Parameter maps in the original non-dynamic dimension order, with
            compatible coordinates, input attributes and existing model
            reconstruction metadata. Numeric outputs are float64, except
            complex T1 A/B or B1 amplitude outputs retain the input complex dtype.
            Sourbron, empirical and 2C-IRF success maps are boolean; sample counts
            are int32 for Sourbron and int64 for empirical/2C-IRF models.
            Uncertainty is included only when supplied as a named model output.
            This call is eager and creates no cache.

        Raises
        ------
        KeyError
            If the requested dynamic dimension coordinate is absent.
        ValueError
            If execution settings, callback outputs or model options are invalid.
            Scientific exceptions from the callback are propagated unchanged.

        Examples
        --------
        >>> fitted = VoxelwiseExecutor().execute(
        ...     data=data, model_id="patlak", fit_params=params,
        ...     single_voxel_fit_func=fit_voxel, output_param_names=("Ki", "vp"),
        ...     dynamic_dim="repetition_idx", model_name="patlak"
        ... )  # doctest: +SKIP
        """
        # --- Ensure data is in a supported precision for linear algebra ---
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float64:
            data = data.astype(np.float64)

        fit_params_dict = (
            dict(fit_params.attrs) if isinstance(fit_params, xr.Dataset) else fit_params
        )
        output_dataset = self._execute_fits(
            data=data,
            model_id=model_id,
            fit_params=fit_params_dict,
            single_voxel_fit_func=single_voxel_fit_func,
            output_param_names=output_param_names,
            dynamic_dim=dynamic_dim,
        )

        from .result_metadata import attach_fit_metadata

        return attach_fit_metadata(
            output_dataset=output_dataset,
            data=data,
            fit_type=model_id,
            fit_params=fit_params_dict,
            model_name=model_name,
            model_equation=model_equation,
            dynamic_dim=dynamic_dim,
        )

    def _execute_fits(
        self,
        *,
        data: xr.DataArray,
        model_id: str,
        fit_params: dict,
        single_voxel_fit_func: callable,
        output_param_names: Sequence[str],
        dynamic_dim: str,
        dynamic_coords: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Assemble fits using labels or explicit dynamics; callers own metadata.

        Explicit dynamics supply the numerical sample axis without changing
        input labels. They are already validated in the caller's domain units.
        """
        fit_params_dict = fit_params

        # Handle precision in fit_params
        for key, val in fit_params_dict.items():
            if (
                isinstance(val, np.ndarray)
                and np.issubdtype(val.dtype, np.floating)
                and val.dtype != np.float64
            ):
                fit_params_dict[key] = val.astype(np.float64)

        execution_config = ExecutionConfig.from_mapping(fit_params_dict)

        non_dynamic_dims = [dim for dim in data.dims if dim != dynamic_dim]
        shape_to_iterate = tuple(data.sizes[dim] for dim in non_dynamic_dims)

        index_iterator = list(
            product(*(range(data.sizes[dim]) for dim in non_dynamic_dims))
        )
        dim_to_index_pos = {dim: i for i, dim in enumerate(non_dynamic_dims)}
        dyn_coord_array = (
            data.coords[dynamic_dim]
            if dynamic_coords is None
            else xr.DataArray(dynamic_coords, dims=[dynamic_dim])
        )

        def run_single_task(indices: tuple):
            current_selection_isel = dict(zip(non_dynamic_dims, indices, strict=False))
            signal_1d_da = data.isel(current_selection_isel)

            # Extract dynamic coordinates
            coord_selection = {}
            for coord_dim in dyn_coord_array.dims:
                if coord_dim in non_dynamic_dims:
                    coord_dim_pos = dim_to_index_pos[coord_dim]
                    coord_selection[coord_dim] = indices[coord_dim_pos]
                elif coord_dim == dynamic_dim:
                    coord_selection[coord_dim] = slice(None)
                else:
                    coord_selection[coord_dim] = 0

            dynamic_coords_for_fit = dyn_coord_array.isel(
                coord_selection
            ).values.astype(float)

            # This is a bit model-specific currently, we might want to normalize this
            # but for now we follow the fit_maps.py logic.
            local_fit_params = dict(fit_params_dict)

            # Map dynamic coords to expected names if needed
            if model_id == "t1":
                local_fit_params["inversion_times_ms"] = dynamic_coords_for_fit
            elif model_id == "adc":
                local_fit_params["b_values"] = dynamic_coords_for_fit

            fit_result, _ = single_voxel_fit_func(
                signal=signal_1d_da.values,
                dynamic_coords=dynamic_coords_for_fit,
                fit_params=local_fit_params,
                indices=indices,
            )

            return _compact_fit_result_for_storage(
                fit_result, output_param_names=output_param_names
            ), indices

        # --- Execute Tasks ---
        results_list = run_indexed_tasks(
            index_iterator,
            run_single_task,
            execution_config=execution_config,
            parallel_desc=f"Fitting voxels ({model_id.upper()}, parallel)",
            sequential_desc=f"Fitting voxels ({model_id.upper()}, sequential)",
        )

        # --- Assemble Results ---
        result_coords = {}
        non_dyn_dims_set = set(non_dynamic_dims)
        for dim in non_dynamic_dims:
            sanitized = _sanitize_dim_coordinate(
                data, dim, non_dyn_dims_set, dynamic_dim
            )
            if sanitized is not None:
                result_coords[dim] = sanitized

        for coord_name, coord_data in data.coords.items():
            if coord_name not in result_coords:
                coord_dims_set = set(coord_data.dims)
                if (
                    coord_dims_set.issubset(non_dyn_dims_set)
                    or len(coord_dims_set) == 0
                ):
                    result_coords[coord_name] = coord_data

        # Determine dtypes
        complex_param_dtype = data.dtype if np.iscomplexobj(data.values) else np.float64

        from hypermri.analysis.dce.model_selection import _DCE_MODELS

        model = _DCE_MODELS.get(model_id)
        output_dtypes = model.module._OUTPUT_DTYPES if model is not None else {}
        fitted_param_arrays = {}
        for param_name in output_param_names:
            dtype = output_dtypes.get(param_name, np.float64)
            if model_id == "t1" and param_name in ["A", "B"]:
                dtype = complex_param_dtype
            elif model_id == "b1" and param_name == "amplitude":
                dtype = complex_param_dtype

            if dtype is bool:
                fill_value = False
            elif np.issubdtype(np.dtype(dtype), np.integer):
                fill_value = 0
            else:
                fill_value = np.nan
            fitted_param_arrays[param_name] = np.full(
                shape_to_iterate, fill_value, dtype=dtype
            )

        for fit_result_dict, indices in results_list:
            if fit_result_dict is not None:
                for param_name in output_param_names:
                    fitted_param_arrays[param_name][indices] = fit_result_dict.get(
                        param_name, np.nan
                    )

        output_dataset_dict = {}
        for param_name, data_array in fitted_param_arrays.items():
            output_dataset_dict[param_name] = xr.DataArray(
                data_array, coords=result_coords, dims=non_dynamic_dims, name=param_name
            )

        return xr.Dataset(output_dataset_dict)


def _sanitize_dim_coordinate(
    data: xr.DataArray,
    dim_name: str,
    non_dynamic_dims_set: set[str],
    dynamic_dim: str,
) -> xr.DataArray | None:
    """Sanitize coordinates that might depend on the dynamic dimension."""
    if dim_name not in data.coords:
        return None

    coord_da = data.coords[dim_name]
    coord_dims_set = set(coord_da.dims)

    if coord_dims_set.issubset(non_dynamic_dims_set):
        return coord_da

    if dynamic_dim in coord_dims_set:
        try:
            dyn_axis = coord_da.dims.index(dynamic_dim)
            values = np.asarray(coord_da.values)
            reference = np.take(values, 0, axis=dyn_axis)
            reference = np.expand_dims(reference, axis=dyn_axis)
            reference = np.broadcast_to(reference, values.shape)

            if np.issubdtype(values.dtype, np.number) or np.issubdtype(
                values.dtype, np.bool_
            ):
                identical = np.allclose(values, reference, equal_nan=True)
            else:
                identical = np.array_equal(values, reference)

            if identical:
                reduced = coord_da.isel({dynamic_dim: 0})
                if dynamic_dim in reduced.dims:
                    reduced = reduced.squeeze(dim=dynamic_dim)
                return reduced
        except Exception:
            pass

    warnings.warn(
        f"Dropping coordinate '{dim_name}' from fit outputs because it depends on the "
        f"dynamic dimension '{dynamic_dim}'.",
        UserWarning,
        stacklevel=2,
    )

    return xr.DataArray(
        np.arange(data.sizes[dim_name]),
        dims=(dim_name,),
        name=dim_name,
    )


def _compact_fit_result_for_storage(
    fit_result_dict: dict | None,
    *,
    output_param_names: Sequence[str],
) -> dict | None:
    """Drop bulky auxiliary fields before aggregating voxel fit results."""
    if fit_result_dict is None:
        return None

    keys_to_keep = set(output_param_names)
    return {key: fit_result_dict[key] for key in keys_to_keep if key in fit_result_dict}
