"""Simple public spectroscopy fitting workflow.

This domain entry delegates numerical estimation to the established
pseudo-inverse fitter and optionally adds supplied-noise diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _require_lorentz_lineshape,
)
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
)
from hypermri.analysis.spectroscopy.frequency import add_fit_freqs_ppm

from .fit_diagnostics import (
    NoiseInput,
    calculate_reduced_chi_squared,
    calculate_spectroscopy_fit_diagnostics,
    noise_sigma_for_fit,
)


def _prepare_fit_params(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
) -> xr.Dataset:
    """Return the established labelled fit-parameter contract."""
    if isinstance(fit_params, xr.Dataset):
        return fit_params
    if isinstance(fit_params, Mapping):
        return def_fit_params(data, dict(fit_params))
    raise TypeError("fit_params must be an xarray.Dataset or a mapping.")


def _noise_template(
    data: xr.DataArray,
    fit_params: xr.Dataset,
) -> xr.DataArray:
    """Build the expected per-spectrum/metabolite layout without fitting."""
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            f"Input data is missing the spectral dimension '{spectral_dim}'."
        )
    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    loop_template = data.isel({spectral_dim: 0}, drop=True)
    return loop_template.expand_dims(
        metabolite=fit_params.coords["metabolite"]
    ).transpose(*loop_dims, "metabolite")


def fit_spectrum(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
    *,
    noise: NoiseInput | None = None,
    show_tqdm: bool | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Fit spectroscopy data and optionally attach cheap noise diagnostics.

    ``noise=None`` delegates to ``fit_data_pseudo_inv``; its result is returned
    unchanged except for the added ``fit_freqs_ppm`` (see Returns).  A supplied scalar or labelled per-spectrum map is a
    real/imaginary quadrature sigma in the temporal FID used by the fit.  It
    does not weight the ordinary least-squares fit or change point estimates.

    Full bootstrap intervals are intentionally a separate operation because
    they require ``B`` complete refits per selected spectrum.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs with the sampling dimension used by
        :func:`~hypermri.analysis.fitting.spectroscopy.fit_data_pseudo_inv`.
    fit_params : xr.Dataset or Mapping
        Prepared fit parameters, or a mapping passed to ``def_fit_params``,
        which rejects unknown keys with ``ValueError``.
        Either lineshape is fitted; supplied-noise diagnostics require
        ``lineshape='lorentz'``.
    noise : NoiseInput, optional
        Real/imaginary quadrature sigma of the temporal FID, see
        :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.noise_sigma_for_fit`.
        ``None`` skips the diagnostics.
    show_tqdm, n_jobs, batch_size, parallel_backend
        Execution options forwarded unchanged to ``fit_data_pseudo_inv``.

    Returns
    -------
    xr.Dataset
        Fit results. ``fit_freqs_ppm`` (unit ppm, same dimensions as
        ``fit_freqs``) is added when the result carries both the Hz and ppm
        calibration axes; it is the linear Hz-to-ppm conversion of
        ``fit_freqs`` via :func:`~hypermri.analysis.spectroscopy.frequency.add_fit_freqs_ppm`
        and is omitted for uncalibrated data. With ``noise`` also
        ``noise_sigma_quadrature``, the variables of
        ``calculate_spectroscopy_fit_diagnostics`` and
        ``fit_reduced_chi_squared`` computed with the supplied sigma (see
        :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.calculate_reduced_chi_squared`).

    Raises
    ------
    ValueError
        If ``noise`` is combined with ``detection_target``, with a
        ``lineshape='voigt'`` fit or with input carrying a
        ``noise_sigma_quadrature`` coordinate (checked before fitting), or if the fit
        or the noise input is invalid.
    """
    prepared = _prepare_fit_params(data, fit_params)
    sigma = None
    if noise is not None:
        if "noise_sigma_quadrature" in data.coords:
            raise ValueError(
                "fit_spectrum: data carries the coordinate "
                "'noise_sigma_quadrature' and noise was also supplied; drop the "
                "coordinate or omit noise so that one noise source is used."
            )
        _require_lorentz_lineshape(
            "fit_spectrum with supplied noise diagnostics", prepared
        )
        if prepared.attrs.get("detection_target") is not None:
            raise ValueError(
                "Simple supplied-noise diagnostics cannot be combined with "
                "detection_target in fit_spectrum. Run the ordinary fit here and "
                "use the explicit detection workflow separately."
            )
        sigma = noise_sigma_for_fit(
            noise,
            _noise_template(data, prepared),
            require_valid=True,
        )

    result = fit_data_pseudo_inv(
        data,
        prepared,
        show_tqdm=show_tqdm,
        n_jobs=n_jobs,
        batch_size=batch_size,
        parallel_backend=parallel_backend,
    )
    result = add_fit_freqs_ppm(result)
    if sigma is None:
        return result

    diagnostics = calculate_spectroscopy_fit_diagnostics(result, prepared, sigma)
    result = result.assign(noise_sigma_quadrature=sigma)
    result["fit_reduced_chi_squared"] = calculate_reduced_chi_squared(result, sigma)
    for name, values in diagnostics.data_vars.items():
        result[name] = values
    result.attrs["supplied_noise_diagnostics"] = True
    result.attrs["noise_sigma_convention"] = (
        "SD per temporal real/imaginary quadrature; E[abs(noise)**2]=2*sigma**2"
    )
    return result


__all__ = ["fit_spectrum"]
