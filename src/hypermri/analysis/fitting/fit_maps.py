"""Model-key entry points for voxel-wise quantitative fits.

:func:`fit_quantitative_maps` and :func:`def_fit_params` select a fit model by
its registry key (``fit_type``) and delegate once to
:func:`hypermri.analysis.fitting.api.fit` and
:func:`hypermri.analysis.fitting.api.prepare_fit_params`.  The named domain
functions such as :func:`hypermri.analysis.quantitative.fit_t1` run the same
operations.  Importing this module registers the ``hp`` xarray accessor; it
fits no data and initializes no plotting.
"""

from __future__ import annotations

import xarray as xr

from . import accessors as _accessors  # noqa: F401
from .api import fit as _api_fit
from .api import prepare_fit_params as _api_prepare_fit_params
from .registry import get_registry


def def_fit_params(
    data: xr.DataArray = None,
    fit_type: str = "t1",
    fit_params: dict = None,
    return_params_info: bool = False,
) -> xr.Dataset:
    """Prepare the parameter dataset of a quantitative fit by its model key.

    The call delegates once to
    :func:`hypermri.analysis.fitting.api.prepare_fit_params`; see there for the
    model-specific schema.

    Parameters
    ----------
    data : xarray.DataArray or None, optional
        Labelled acquired samples with the dimensions, coordinates and units
        that the selected model requires.  ``None`` is accepted only by models
        that can describe their options without data.
    fit_type : str, optional
        Registry key or supported alias of the model, such as ``"t1"``,
        ``"adc"``, ``"b1"`` or ``"ideal"``.  The default is ``"t1"``.
    fit_params : dict or None, optional
        Scientific and execution settings in the model's documented units.
        The mapping is not modified.
    return_params_info : bool, optional
        Request the model's description instead of a parameter dataset.

    Returns
    -------
    xarray.Dataset or None
        Prepared parameter values and attributes in the model's schema and
        precision, or ``None`` for the description-only request.  Preparation
        is eager and creates no cache.

    Raises
    ------
    KeyError
        If ``fit_type`` is not a known model key or alias.
    ValueError
        If the scientific inputs are invalid for the selected model.
    """
    return _api_prepare_fit_params(
        data=data,
        model=fit_type,
        fit_params=fit_params,
        return_params_info=return_params_info,
    )


def _print_params_info(fit_type: str):
    """Print parameter information for a given fit_type."""
    spec = get_registry().get(fit_type)
    if spec.describe_params:
        info = spec.describe_params()
        # Print parameter information in a nice format (simplified mirror of legacy)
        print("=" * 80)
        print(f"Parameter Information for fit_type='{fit_type}'")
        print("=" * 80)
        if "model_equation" in info:
            print(f"\nModel Equation:\n  {info['model_equation']}")
        print(f"\n{'Required Parameters in fit_params:':-<80}")
        for param, desc in info.get("required_fit_params", {}).items():
            print(f"  • {param}\n      {desc}")
        print(f"\n{'Optional Parameters in fit_params:':-<80}")
        for param, desc in info.get("optional_fit_params", {}).items():
            print(f"  • {param}\n      {desc}")
        print(f"\n{'Output Parameters (in returned Dataset):':-<80}")
        for param, desc in info.get("output_parameters", {}).items():
            print(f"  • {param}\n      {desc}")
        print("=" * 80)
    else:
        print(f"No parameter information available for '{fit_type}'.")


def fit_quantitative_maps(
    data: xr.DataArray = None,
    fit_type: str = "t1",
    fit_params: dict = None,
    return_params_info: bool = False,
):
    """Fit a quantitative model voxel-wise, selected by its model key.

    The call delegates once to :func:`hypermri.analysis.fitting.api.fit`, so
    results equal those of that function and of the named domain functions
    such as :func:`hypermri.analysis.quantitative.fit_t1`.

    Parameters
    ----------
    data : xarray.DataArray or None, optional
        Labelled acquired samples with the dimensions, coordinates and units
        that the selected model requires.  The input is not modified.
    fit_type : str, optional
        Registry key or supported alias of the model, such as ``"t1"``,
        ``"adc"``, ``"b1"``, ``"ideal"`` or a DCE model key.  The default is
        ``"t1"``.
    fit_params : dict or None, optional
        Scientific and execution settings in the model's documented units.
    return_params_info : bool, optional
        If true, print the model equation and its required, optional and
        output parameters instead of fitting, and return ``None``.

    Returns
    -------
    xarray.Dataset or None
        Parameter, status and uncertainty maps with labelled spatial
        dimensions and reconstruction and provenance metadata, as returned by
        :func:`hypermri.analysis.fitting.api.fit`, or ``None`` if
        ``return_params_info`` is true.  The fit is eager.

    Raises
    ------
    ValueError
        If ``fit_type`` is not a known model key or alias, or the scientific
        inputs or options are inconsistent.

    Examples
    --------
    >>> maps = fit_quantitative_maps(ir_data, fit_type="t1")  # doctest: +SKIP
    >>> fit_quantitative_maps(fit_type="adc", return_params_info=True)  # doctest: +SKIP
    """
    if return_params_info:
        _print_params_info(fit_type)
        return None

    try:
        return _api_fit(
            data=data,
            model=fit_type,
            fit_params=fit_params,
            return_params_info=return_params_info,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported fit_type '{fit_type}'.") from exc
