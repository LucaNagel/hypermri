"""Preserve fitted-result labels, provenance and model reconstruction inputs.

Model modules own scientific schemas. These helpers transfer existing metadata
without fitting, changing physical units or constructing visualization objects.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import xarray as xr

from hypermri.analysis.dce.conversions import DEFAULT_R1_PER_MM_S


def dataset_reconstruction_array(
    dataset: xr.Dataset | xr.DataArray,
    name: str,
) -> xr.DataArray | np.ndarray | None:
    """Find reconstruction metadata without converting its values or labels.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Result searched in data variables, then coordinates, then legacy attrs.
        DataArray inputs have no data-variable lookup. The input is preserved.
    name : str
        Reconstruction field, for example ``"aif_concentration"`` (mM) or
        ``"time_points_s"`` (s). Units and shape belong to the stored field.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or object or None
        Original object, including legacy attribute values of other types;
        None when absent. No copy, dtype conversion, loading or cache is added.
    """
    if isinstance(dataset, xr.Dataset):
        if name in dataset.data_vars:
            return dataset[name]
        if name in dataset.coords:
            return dataset.coords[name]
    elif isinstance(dataset, xr.DataArray) and name in dataset.coords:
        return dataset.coords[name]
    return getattr(dataset, "attrs", {}).get(name)


def dataset_reconstruction_values(
    dataset: xr.Dataset | xr.DataArray,
    name: str,
) -> np.ndarray | None:
    """Read reconstruction metadata as an unlabelled NumPy array.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Result searched using :func:`dataset_reconstruction_array` precedence.
        Inputs and their metadata remain unchanged.
    name : str
        Reconstruction field whose existing shape, units and dtype are retained.

    Returns
    -------
    numpy.ndarray or None
        Values without labels, or None when missing. Access is eager for lazy
        xarray values, may share storage with the input, and creates no cache.

    Raises
    ------
    TypeError, ValueError
        If a legacy attribute cannot be converted with ``numpy.asarray``.
    """
    values = dataset_reconstruction_array(dataset, name)
    if values is None:
        return None
    if isinstance(values, xr.DataArray):
        return np.asarray(values.values)
    return np.asarray(values)


def select_reconstruction_array(
    dataset: xr.Dataset | xr.DataArray,
    name: str,
    voxel_coords: dict | None = None,
) -> xr.DataArray | np.ndarray | None:
    """Select positional voxel entries from labelled reconstruction metadata.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Result searched using :func:`dataset_reconstruction_array` precedence.
        Source values, units and metadata are preserved.
    name : str
        Reconstruction field with model-owned units, shape and dimension order.
    voxel_coords : dict or None, optional
        Dimension-to-positional-index mapping for xarray ``isel``. Keys absent
        from the field are ignored. Unlabelled legacy values are not selected.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or object or None
        Selected labelled view, unchanged legacy object, or None when absent.
        Surviving dimensions retain order, coordinates, attrs and storage dtype.
        No scientific calculation, cache or input mutation is introduced.

    Raises
    ------
    IndexError, TypeError
        If an applicable positional selector is invalid for xarray ``isel``.

    Examples
    --------
    >>> result = xr.Dataset({"baseline": ("read_idx", [10.0, 20.0])})
    >>> select_reconstruction_array(result, "baseline", {"read_idx": 1}).item()
    20.0
    """
    values = dataset_reconstruction_array(dataset, name)
    if isinstance(values, xr.DataArray) and voxel_coords:
        selectors = {
            dim: idx for dim, idx in voxel_coords.items() if dim in values.dims
        }
        if selectors:
            values = values.isel(selectors)
    return values


def set_reconstruction_variable(
    output_dataset: xr.Dataset,
    name: str,
    values: object,
    *,
    dims: list[str],
    coords: dict[str, object] | None = None,
    attrs: dict[str, object] | None = None,
) -> None:
    """Store reconstruction inputs as a labelled result variable in place.

    Parameters
    ----------
    output_dataset : xarray.Dataset
        Caller-owned destination. Existing variables with ``name`` are replaced.
    name : str
        Reconstruction variable name chosen by the model's schema.
    values : object
        Array-like values converted eagerly with ``numpy.asarray`` without an
        explicit dtype cast. Shape must match ``dims`` and coordinate lengths.
        None leaves the dataset untouched.
    dims : list of str
        Ordered dimension names for the stored values; no axes are reordered.
    coords : dict or None, optional
        Compatible coordinates. xarray assignment aligns with existing dataset
        indexes; callers must supply matching labels to preserve every value.
    attrs : dict or None, optional
        Variable metadata, including units defined by the caller (e.g. s or mM).
        Input DataArray attrs are not implicitly copied through NumPy conversion.

    Returns
    -------
    None
        Updates the supplied dataset deterministically and creates no cache.

    Raises
    ------
    ValueError
        If values, dimensions or coordinates have incompatible shapes.
    """
    if values is None:
        return
    data_array = xr.DataArray(
        np.asarray(values),
        dims=dims,
        coords=coords or None,
    )
    if attrs:
        data_array.attrs.update(attrs)
    output_dataset[name] = data_array


def attach_reconstruction_schema(
    output_dataset: xr.Dataset,
    *,
    fit_type: str,
    fit_params: dict,
    dynamic_dim_name: str,
    data: xr.DataArray,
) -> None:
    """Attach the existing model reconstruction inputs as labelled variables.

    Parameters
    ----------
    output_dataset : xarray.Dataset
        Caller-owned fitted result updated in place; parameter values are retained.
    fit_type : str
        Canonical model key. DCE schemas come from its dictionary-selected module;
        ADC, T1 and B1 retain their existing legacy schemas.
    fit_params : dict
        Effective scientific options, preserved. DCE time/AIF values use seconds/mM.
    dynamic_dim_name : str
        Dynamic dimension whose acquisition labels are retained when lengths match.
    data : xarray.DataArray
        Acquired samples and coordinates, preserved. Time arrays use their existing
        float64 conversion and positional fallback; T1 inversion times use ms.

    Returns
    -------
    None
        Reconstruction variables/attributes are attached eagerly and deterministically
        without fitting or caching. Unsupported schemas retain their no-op behavior.

    Raises
    ------
    ValueError
        If supplied reconstruction arrays and labelled dimensions cannot be aligned.
    """
    from hypermri.analysis.dce.model_selection import _DCE_MODELS

    model = _DCE_MODELS.get(fit_type)
    if model is not None:
        attach = getattr(model.module, "_attach_reconstruction_arrays", None)
        if attach is not None:
            attach(output_dataset, fit_params, dynamic_dim_name, data)
        return

    if fit_type == "adc":
        b_values_used = fit_params.get("b_values")
        if b_values_used is not None:
            output_dataset.attrs["b_values"] = list(b_values_used)
        return

    if fit_type == "t1":
        inversion_times_coord = data.coords.get(dynamic_dim_name)
        if inversion_times_coord is not None:
            output_dataset["inversion_times_ms"] = inversion_times_coord
            output_dataset["inversion_times_ms"].attrs.update(
                {
                    "unit": "ms",
                    "units": "ms",
                    "long_name": "inversion time",
                }
            )
        return

    if fit_type == "b1":
        refpower_W = fit_params.get("refpower_W")
        if refpower_W is not None:
            output_dataset.attrs["refpower_W"] = list(refpower_W)
        output_dataset.attrs["fa_deg"] = fit_params.get("fa_deg", 90)
        output_dataset.attrs["trf_ms"] = fit_params.get("trf_ms", 1)
        output_dataset.attrs["sint"] = fit_params.get("sint", 1)


def inherit_input_xarray_metadata(
    output_dataset: xr.Dataset,
    data: xr.DataArray,
) -> xr.Dataset:
    """Copy input attributes and compatible coordinates to fitted results.

    Parameters
    ----------
    output_dataset : xarray.Dataset
        Destination. Its existing attrs and coordinate attrs take precedence
        over input entries with the same key and are updated in place.
    data : xarray.DataArray
        Preserved source with arbitrary scientific units, dimension order and
        coordinate direction. Missing coordinates are added only when all their
        dimensions occur in the output. Existing coordinate values are retained.

    Returns
    -------
    xarray.Dataset
        Output with deep-copied merged attrs. When coordinates are added,
        ``assign_coords`` returns a new dataset using shallow coordinate copies;
        otherwise the original output is returned. Coordinate values and dtypes
        are not converted. No fitting, caching or eager sample read occurs.

    Raises
    ------
    ValueError
        If a new coordinate has a dimension length incompatible with the result.

    Examples
    --------
    >>> signal = xr.DataArray([1.0], dims="voxel", attrs={"units": "signal"})
    >>> fitted = xr.Dataset({"amplitude": ("voxel", [1.0])})
    >>> inherit_input_xarray_metadata(fitted, signal).attrs["units"]
    'signal'
    """
    source_attrs = deepcopy(dict(getattr(data, "attrs", {})))
    if source_attrs:
        existing_attrs = deepcopy(dict(getattr(output_dataset, "attrs", {})))
        output_dataset.attrs = {**source_attrs, **existing_attrs}

    coord_updates = {}
    for coord_name, source_coord in data.coords.items():
        if coord_name in output_dataset.coords:
            source_coord_attrs = deepcopy(dict(getattr(source_coord, "attrs", {})))
            existing_coord_attrs = deepcopy(
                dict(getattr(output_dataset.coords[coord_name], "attrs", {}))
            )
            output_dataset.coords[coord_name].attrs = {
                **source_coord_attrs,
                **existing_coord_attrs,
            }
            continue

        if all(dim in output_dataset.dims for dim in source_coord.dims):
            coord_updates[coord_name] = source_coord.copy(deep=False)

    if coord_updates:
        output_dataset = output_dataset.assign_coords(coord_updates)

    return output_dataset


def attach_fit_metadata(
    output_dataset: xr.Dataset,
    data: xr.DataArray,
    fit_type: str,
    fit_params: dict,
    model_name: str,
    model_equation: str,
    dynamic_dim: str | None = None,
) -> xr.Dataset:
    """Attach generic and model-owned metadata to fitted results.

    Copies the input attributes and compatible coordinates, records the
    model, the fit options needed to interpret the result and the arrays
    needed by ``Dataset.hp.reconstruct``.  For ``fit_type="tofts"`` the
    attribute ``"r1"`` holds the relaxivity in 1/(mM s), by default
    :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.

    Parameters
    ----------
    output_dataset : xarray.Dataset
        Fit result.  Its attributes are updated in place.
    data : xarray.DataArray
        Fitted data whose attributes and coordinates are inherited.
    fit_type : str
        Registered fit type, for example ``"t1"``, ``"tofts"`` or
        ``"sourbron"``; selects the recorded options.
    fit_params : dict
        Effective fit options; not modified.
    model_name : str
        Registered result-model name stored in ``attrs["model_name"]``.
    model_equation : str
        Human-readable model equation stored in ``attrs["model_equation"]``.
    dynamic_dim : str or None, optional
        Dynamic dimension of the fit; ``"repetition_idx"`` is used for the
        reconstruction arrays when omitted.

    Returns
    -------
    xarray.Dataset
        The annotated result.  A new object when inherited coordinates were
        added, else ``output_dataset`` itself. Updates attrs/variables eagerly
        and deterministically without fitting, changing sample precision or
        caching. Coordinate values and direction retain their existing contract.

    Raises
    ------
    ValueError
        If inherited coordinates or model reconstruction arrays have incompatible
        dimensions. Model-owned schema errors propagate unchanged.
    """
    output_dataset = inherit_input_xarray_metadata(output_dataset, data)

    output_dataset.attrs["model_name"] = model_name
    output_dataset.attrs["model_equation"] = model_equation
    if dynamic_dim:
        output_dataset.attrs["dynamic_dim"] = dynamic_dim

    if fit_type == "t1":
        # Public T1 grids are supplied in seconds, while both model time axes and
        # fitted T1 maps use milliseconds. Keep the contract machine-readable on
        # the result rather than relying on parameter names or documentation.
        output_dataset.attrs.update(
            {
                "t1_grid_input_units": "s",
                "t1_output_units": "ms",
                "inversion_time_units": "ms",
            }
        )
        if "t1" in output_dataset:
            output_dataset["t1"].attrs.update(
                {
                    "unit": "ms",
                    "units": "ms",
                    "long_name": "longitudinal relaxation time",
                }
            )

    if "git_commit_hash" not in output_dataset.attrs:
        try:
            output_dataset.attrs["git_commit_hash"] = data.attrs.get(
                "git_commit_hash", "unknown"
            )
        except (AttributeError, KeyError):
            output_dataset.attrs["git_commit_hash"] = "unknown"

    dynamic_dim_name = dynamic_dim or "repetition_idx"
    from hypermri.analysis.dce.model_selection import _DCE_MODELS

    model = _DCE_MODELS.get(fit_type)
    if model is not None:
        attach = getattr(model.module, "_attach_fit_options", None)
        if attach is not None:
            attach(output_dataset, fit_params)

    attach_reconstruction_schema(
        output_dataset,
        fit_type=fit_type,
        fit_params=fit_params,
        dynamic_dim_name=dynamic_dim_name,
        data=data,
    )

    return output_dataset
