"""Xarray accessors for fitted datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr

from hypermri.analysis.dce.conversions import DEFAULT_R1_PER_MM_S

from .registry import get_registry
from .result_metadata import select_reconstruction_array


@xr.register_dataset_accessor("hp")
class FitAccessor:
    """Reconstruct fitted datasets through their scientific model functions.

    Parameters
    ----------
    xarray_obj : xarray.Dataset
        Fitted parameter maps and reconstruction inputs with model_name metadata.
        Scientific units, precision and sampling requirements belong to the model.

    Notes
    -----
    Construction retains a reference without reconstructing or reading acquired data.
    Reconstruction is eager, preserves inputs and creates no result cache. DCE
    selection uses its explicit domain dictionary; other models use legacy adapters.

    Examples
    --------
    >>> prediction = fitted.hp.reconstruct(voxel_coords={"read_idx": 0})  # doctest: +SKIP
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def reconstruct(self, voxel_coords: dict = None, **kwargs) -> xr.DataArray:
        """Reconstruct fitted signals for whole volumes or individual voxels.

        The model is selected from ``attrs['model_name']`` through the
        domain dictionary for DCE, or the legacy registry for other models,
        and evaluated eagerly from the fitted parameter variables.
        FID results of a ``lineshape='voigt'`` fit are reconstructed with
        their shared Gaussian FWHM ``fit_gauss_fwhm_hz``; Lorentz results are
        evaluated unchanged.

        Parameters
        ----------
        voxel_coords : dict, optional
            Positional selection passed to :meth:`xarray.Dataset.isel`, for
            example ``{"x": 3, "y": 4}``. ``None`` reconstructs all voxels.
        **kwargs
            Model-specific sampling inputs. ``dynamic_dim`` overrides the
            model's sampling axis; for FID results ``sampling_time_axis_s``
            (or legacy ``t``) gives the times in seconds and otherwise
            ``attrs['sampling_time_axis_s']`` is used. The legacy ``voxel``
            keyword is an alias of ``voxel_coords``. Tofts results read
            ``reco_space``, ``model_mode``, ``tr_s``, ``fa_deg`` and ``r1``
            (1/(mM s)); missing values come from the dataset attributes, and
            DCE time/AIF arrays use seconds/mM; the existing keyword-over-variable
            precedence is retained. Inputs and fitted maps are preserved, and
            no result cache is created. ``r1`` finally defaults to
            :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.
            Quantitative T2 reads ``echo_times_s`` in seconds from a keyword or
            the result variable and uses fitted ``t2`` in ms. A numerical
            ``dynamic_dim`` array overrides echo times in seconds, supporting
            dense reconstruction; a string names the axis of an unlabelled
            echo-time array.

        Returns
        -------
        xr.DataArray
            Reconstructed signal as returned by the model's reconstructor;
            for FID results the complex metabolite components with a
            ``metabolite`` dimension. DCE concentration predictions are float64
            mM, with the model's labelled spatial/dynamic dimensions and coordinate
            direction. Tofts signal reconstruction uses the acquired signal scale.

        Raises
        ------
        NotImplementedError
            If the model is unknown or has no reconstructor.
        KeyError
            If a required fitted parameter variable is missing.
        ValueError
            If a required sampling axis is missing, or a Voigt FID result
            lacks ``fit_gauss_fwhm_hz``.

        Examples
        --------
        >>> fitted.hp.reconstruct(reco_space="concentration")  # doctest: +SKIP
        """
        legacy_voxel = kwargs.pop("voxel", None)
        if voxel_coords is None and legacy_voxel is not None:
            voxel_coords = legacy_voxel

        from hypermri.analysis.dce.model_selection import _find_dce_result_model

        dce_model = _find_dce_result_model(self._obj.attrs.get("model_name"))
        if dce_model is not None:
            data = self._obj.isel(voxel_coords) if voxel_coords else self._obj
            params = dce_model.module._reconstruction_parameters(
                data, kwargs, voxel_coords
            )
            return dce_model.module._reconstruct(**params)

        try:
            spec = get_registry().resolve_result_model(self._obj)
        except KeyError:
            model_name = self._obj.attrs.get("model_name")
            raise NotImplementedError(
                f"Reconstruction for model '{model_name}' is not implemented or model_name is missing, available models: {list(get_registry().list_models())}"
            )
        model_name = self._obj.attrs.get("model_name")
        reconstructor = spec.reconstruct
        if reconstructor is None:
            raise NotImplementedError(
                f"Reconstruction for model '{model_name}' is not implemented."
            )
        data_to_reconstruct = (
            self._obj.isel(voxel_coords) if voxel_coords else self._obj
        )

        params = {}
        dynamic_dim_attr = data_to_reconstruct.attrs.get("dynamic_dim")
        dynamic_dim_override = kwargs.get("dynamic_dim")
        if model_name == "monoexponential_adc":
            b_values = dynamic_dim_override
            if b_values is None:
                b_values = kwargs.get(
                    "b_values", data_to_reconstruct.attrs.get("b_values")
                )
            if b_values is None:
                dim_name = dynamic_dim_attr or "bvalue_idx"
                if dim_name in data_to_reconstruct.coords:
                    b_values = data_to_reconstruct.coords[dim_name]
            if b_values is None:
                raise ValueError(
                    f"b-values must be provided via `dynamic_dim` or `b_values` kwarg, exist in dataset attrs, or be available as a coordinate for '{dim_name}'."
                )
            params = {
                "s0": data_to_reconstruct["s0"],
                "adc": data_to_reconstruct["adc"],
                "b_values": b_values,
            }
        elif model_name == "t1_inversion_recovery":
            full_inversion_times = dynamic_dim_override
            if full_inversion_times is None:
                full_inversion_times = kwargs.get(
                    "inversion_times_ms",
                    select_reconstruction_array(self._obj, "inversion_times_ms"),
                )
            if full_inversion_times is None:
                dim_name = dynamic_dim_attr or "inversion_idx"
                if dim_name in self._obj.coords:
                    full_inversion_times = self._obj.coords[dim_name]
            if full_inversion_times is None:
                raise ValueError(
                    f"Inversion times must be provided via `dynamic_dim` or `inversion_times_ms` kwarg, exist in dataset attrs, or be available as a coordinate for '{dim_name}'."
                )
            if voxel_coords and hasattr(full_inversion_times, "dims"):
                slice_selector = {
                    dim: voxel_coords[dim]
                    for dim in full_inversion_times.dims
                    if dim in voxel_coords
                }
                inversion_times_for_model = full_inversion_times.isel(slice_selector)
            else:
                inversion_times_for_model = full_inversion_times
            params = {
                "a": data_to_reconstruct["A"],
                "b": data_to_reconstruct["B"],
                "t1_ms": data_to_reconstruct["t1"],
                "inversion_times_ms": inversion_times_for_model,
            }
        elif model_name == "monoexponential_t2":
            echo_dim = dynamic_dim_attr or "echo_idx"
            if isinstance(dynamic_dim_override, str):
                echo_dim = dynamic_dim_override or echo_dim
                times = kwargs.get("echo_times_s", self._obj.get("echo_times_s"))
            elif dynamic_dim_override is not None:
                times = dynamic_dim_override
            else:
                times = kwargs.get("echo_times_s", self._obj.get("echo_times_s"))
            if times is None:
                raise ValueError("T2 reconstruction requires echo_times_s in seconds.")
            params = {
                "echo_times_s": times,
                "s0": data_to_reconstruct["s0"],
                "t2": data_to_reconstruct["t2"],
                "dynamic_dim": echo_dim,
            }
        elif model_name == "FID":
            sampling_time_axis = dynamic_dim_override
            if sampling_time_axis is None:
                sampling_time_axis = kwargs.get(
                    "sampling_time_axis_s",
                    kwargs.get(
                        "t", data_to_reconstruct.attrs.get("sampling_time_axis_s")
                    ),
                )
            if sampling_time_axis is None:
                raise ValueError(
                    "FID reconstruction requires `dynamic_dim` or `sampling_time_axis_s` via kwargs or dataset attrs"
                )
            params = {
                "amplitude": data_to_reconstruct["fit_amps"],
                "freq_hz": data_to_reconstruct["fit_freqs"],
                "t2_star_s": data_to_reconstruct["fit_t2s"],
                "time_axis_s": np.asarray(sampling_time_axis),
                "db_mode": False,
            }
            from .models.spectroscopy.fid_model import _uses_voigt_lineshape

            if "fit_gauss_fwhm_hz" in data_to_reconstruct:
                params["gauss_fwhm_hz"] = data_to_reconstruct["fit_gauss_fwhm_hz"]
            elif _uses_voigt_lineshape(data_to_reconstruct.attrs):
                raise ValueError(
                    "FID reconstruction of a lineshape='voigt' fit requires the 'fit_gauss_fwhm_hz' variable; keep it when selecting or subsetting the fit results."
                )
        elif model_name == "b1_sinusoidal":
            refpower_W = dynamic_dim_override
            if refpower_W is None:
                refpower_W = kwargs.get(
                    "refpower_W", data_to_reconstruct.attrs.get("refpower_W")
                )
            if refpower_W is None:
                raise ValueError(
                    "B1 reconstruction requires 'dynamic_dim' or 'refpower_W' via kwargs or dataset attrs"
                )
            params = {
                "refpower_W": np.asarray(refpower_W),
                "amplitude": data_to_reconstruct["amplitude"],
                "period": data_to_reconstruct["period"],
                "db_mode": kwargs.get("db_mode", False),
            }

        return reconstructor(**params)


@xr.register_dataset_accessor("reconstruct")
class ReconstructAliasAccessor:
    """Callable alias so fitted datasets support ``fit_results.reconstruct()``."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, *args, **kwargs) -> xr.DataArray:
        return self._obj.hp.reconstruct(*args, **kwargs)
