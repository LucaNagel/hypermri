"""Prepare CSI decay diagnostics and open them in the browser viewer.

Acquisition sorting, filtering and spatial reconstruction delegate to their
existing sequence/Bruker owners. This module builds labelled previews without
constructing widgets or mutating acquired samples or handler caches.
:func:`plot_signal_decay` shows them in
:class:`~hypermri.visualization.scan_view.ScanView` with the "signal_decay"
tool (:mod:`hypermri.visualization.signal_decay_tool`).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.noise_estimation import suggest_signal_end
from hypermri.analysis.spectroscopy.signal_decay import csi_order_indices
from hypermri.reconstruction.centred_dft import centered_fft
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy
from hypermri.visualization.signal_decay_tool import new_signal_decay_state

if TYPE_CHECKING:
    from hypermri.visualization.scan_view import ScanView

__all__ = ["plot_signal_decay"]

#: Settings context of the view settings stored on the scan.
SIGNAL_DECAY_SETTINGS_CONTEXT = "signal_decay"


def plot_signal_decay(
    scan: Any,
    data: np.ndarray | xr.DataArray | None = None,
    *,
    spatial_spacing_mm: tuple[float, float] | None = None,
    bands_hz: Sequence[tuple[float, float]] | None = None,
) -> ScanView:
    """Show the decay of CSI raw data and choose its filter in the browser viewer.

    Experimental browser counterpart of
    :meth:`hypermri.sequences.CSI.CSI.plot_signal_decay`. The view shows
    the CSI data reconstructed with the chosen filter (map and voxel
    spectrum) and opens the "signal_decay" tool: a plot of the decay over
    the acquired k-space points, the filter inputs and "Confirm", which
    stores ``scan.signal_end`` and ``scan.filter_params``. See
    :mod:`hypermri.visualization.signal_decay_tool` for the curves.

    Parameters
    ----------
    scan : CSI
        CSI scan with loaded raw data (``rawdatajob0`` or ``fid``) and a
        CSI acquisition description (encoding order, dwell time).
    data : numpy.ndarray, xarray.DataArray or None, optional
        Raw complex FIDs as a vector or (FID sample, acquired point) matrix,
        or labelled CSI samples with ``acquisition_idx``, ``read_idx`` and
        ``phase_idx`` (``attrs["domain"]`` ``"spatial"`` or ``"kspace"``,
        spatial when missing). None uses the loaded raw data. The input is
        copied and never changed.
    spatial_spacing_mm : tuple of float or None, optional
        Read and phase voxel spacing in mm for the spatial k-space filter,
        for labelled input without calibrated axes; None uses the field of
        view of the scan.
    bands_hz : sequence of (float, float) or None, optional
        Frequency bands in Hz, for example ``[(-330, -150), (200, 420)]``.
        When given, the tool starts with the band power in these bands;
        otherwise with the mean magnitude.

    Returns
    -------
    ScanView
        The view, with the tool chosen. Display it as the last expression of
        a notebook cell.

    Raises
    ------
    ValueError
        If raw data are missing or incomplete, labelled input lacks a
        dimension, or ``bands_hz`` is not a list of increasing pairs.

    Notes
    -----
    The filter is applied with ``scan.filter_signal`` along the acquired
    points (``filter_axis=1``) to a copy, whenever a filter input changes.
    Acquired data and reconstructions stored on the scan stay unchanged.
    The view starts with signal end and window centre at half the acquired
    points and no filter, so it shows the data truncated at the signal end.

    Examples
    --------
    >>> from hypermri.visualization.signal_decay import plot_signal_decay
    >>> view = plot_signal_decay(csi, bands_hz=[(-330, -150), (200, 420)])  # doctest: +SKIP
    >>> view  # doctest: +SKIP
    >>> csi.filter_signal(**csi.filter_params)  # after "Confirm"  # doctest: +SKIP
    """
    from hypermri.visualization.scan_view import plot_scan

    prepared = _SignalDecayData(scan, data, spatial_spacing_mm)
    state = new_signal_decay_state(prepared, bands_hz)
    view = plot_scan(
        prepared.original,
        scan=scan,
        settings_context=SIGNAL_DECAY_SETTINGS_CONTEXT,
    )
    view._offer_signal_decay(prepared, state)
    return view


class _SignalDecayData:
    def __init__(self, sequence, data, spatial_spacing_mm=None):
        self.sequence = sequence
        self.spatial_spacing_mm = spatial_spacing_mm
        self.handler = sequence.handler
        self.strategy = BrukerCsiRecoStrategy()
        provided = data is not None
        if not provided:
            data = getattr(self.handler, "rawdatajob0", None)
            if data is None:
                data = getattr(self.handler, "fid", None)
            if data is None:
                raise ValueError(
                    "CSI.plot_signal_decay requires loaded rawdatajob0/fid or explicit labelled CSI data."
                )
        self.input = (
            data.copy(deep=True)
            if isinstance(data, xr.DataArray)
            else np.asarray(data).copy()
        )
        self.n_acquisition = int(sequence.n_acquisition)
        self.n_repetitions = int(getattr(sequence, "n_repetitions", 1))
        self.signal_domain = "temporal"
        if isinstance(self.input, xr.DataArray):
            self.signal_domain = str(
                self.input.attrs.get("signal_domain", "temporal")
            ).lower()
            required = {"acquisition_idx", "read_idx", "phase_idx"}
            if not required.issubset(self.input.dims):
                raise ValueError(
                    "CSI.plot_signal_decay labelled input requires acquisition_idx, read_idx and phase_idx."
                )
            if "domain" not in self.input.attrs:
                self.input.attrs["domain"] = "spatial"
            self.n_acquisition = self.input.sizes["acquisition_idx"]
            self.n_repetitions = self.input.sizes.get("repetition_idx", 1)
            kspace = self.input
        else:
            if self.input.ndim not in (1, 2) or self.input.size % self.n_acquisition:
                raise ValueError(
                    "CSI.plot_signal_decay raw input must contain complete FIDs in a vector or (FID sample, acquired point) matrix."
                )
            raw = self._matrix(self.input).T.reshape(-1)
            kspace = (
                self.strategy.sort_kspace(
                    handler=self.handler, write_to_object=False, data=raw
                )
                if provided
                else self.strategy.sort_kspace(
                    handler=self.handler, write_to_object=False
                )
            )
        self.original = self._spatial(kspace)
        self.original_matrix = self._matrix(self.input)
        self.filtered = self.original
        self.filtered_matrix = self.original_matrix
        self.n_points = self.original_matrix.shape[1]
        if self.n_points % self.n_repetitions == 0:
            self.n_points //= self.n_repetitions
        self._suggestions = {}

    def suggested_signal_end(self, repetition_index=0, bands_hz=None):
        """Return the noise-onset ``signal_end`` suggestion of one repetition.

        Parameters
        ----------
        repetition_index : int, optional
            Zero-based repetition of the unfiltered acquisition; default 0.
        bands_hz : sequence of (float, float) or None, optional
            Metabolite frequency ranges in Hz; None uses the whole spectrum.

        Returns
        -------
        int
            Suggested number of kept acquired points from
            :func:`~hypermri.analysis.spectroscopy.suggest_signal_end` with
            margin 0. Cached per repetition and band selection, because the
            unfiltered acquisition does not change during a viewer session.

        Raises
        ------
        ValueError
            If the input holds spectra, the dwell time is missing, or the
            acquisition has no detectable noise-only remainder.
        """
        key = (
            int(repetition_index),
            None if bands_hz is None else tuple(map(tuple, bands_hz)),
        )
        if key not in self._suggestions:
            try:
                self._suggestions[key] = suggest_signal_end(
                    self.original_matrix,
                    dwell_time_s=self.band_power_dwell_time_s(),
                    bands_hz=bands_hz,
                    n_repetitions=self.n_repetitions,
                    repetition_index=int(repetition_index),
                )
            except ValueError as exc:
                self._suggestions[key] = exc
        result = self._suggestions[key]
        if isinstance(result, ValueError):
            raise result
        return result

    def band_power_dwell_time_s(self):
        """Return the FID sampling interval that band power needs.

        Returns
        -------
        float
            Positive dwell time in seconds from the CSI acquisition description.

        Raises
        ------
        ValueError
            If the input holds spectra instead of FIDs, or the acquisition
            description has no positive dwell time.
        """
        if self.signal_domain == "spectral":
            raise ValueError(
                "CSI.plot_signal_decay band power needs FIDs; the labelled input "
                "has signal_domain='spectral'. Pass time-domain CSI data."
            )
        dwell = self.handler.to_csi_acquisition().dwell_time_s
        if dwell is None or not (np.isfinite(dwell) and dwell > 0):
            raise ValueError(
                "CSI.plot_signal_decay band power needs the acquisition dwell "
                f"time, but the CSI acquisition description has {dwell!r}."
            )
        return float(dwell)

    def _spatial(self, data):
        if str(data.attrs.get("domain", "")).lower() == "spatial":
            return data.copy(deep=True)
        return self.strategy._sorted_csi_to_data(
            kspace_xarray=data, handler=self.handler, data_precision=np.float64
        )

    def _matrix(self, data):
        if not isinstance(data, xr.DataArray):
            values = np.asarray(data)
            if values.ndim == 1:
                return values.reshape(-1, self.n_acquisition).T
            if (
                values.shape[0] != self.n_acquisition
                and values.shape[-1] == self.n_acquisition
            ):
                values = np.moveaxis(values, -1, 0)
            return values.reshape(self.n_acquisition, -1)
        dims = ["acquisition_idx"]
        if "repetition_idx" in data.dims:
            dims.append("repetition_idx")
        dims += ["phase_idx", "read_idx"]
        dims += [d for d in data.dims if d not in dims]
        working = data.transpose(*dims)
        values = np.asarray(working.data).copy()
        if str(data.attrs.get("domain", "spatial")).lower() != "kspace":
            values = centered_fft(
                values,
                [working.get_axis_num(dim) for dim in ("phase_idx", "read_idx")],
            )
        acquisition = self.handler.to_csi_acquisition()
        order = csi_order_indices(
            working.copy(data=values),
            acquisition.read_positions,
            acquisition.phase_positions,
        )
        n_fids = data.sizes["phase_idx"] * data.sizes["read_idx"]
        if "repetition_idx" in dims:
            flat = values.reshape(self.n_acquisition, self.n_repetitions, n_fids, -1)
            return np.take(flat, order, axis=2).reshape(self.n_acquisition, -1)
        return np.take(
            values.reshape(self.n_acquisition, n_fids, -1), order, axis=1
        ).reshape(self.n_acquisition, -1)

    def update(self, params):
        source = (
            self.input.copy(deep=True)
            if isinstance(self.input, xr.DataArray)
            else self.original_matrix.copy()
        )
        filtered = self.sequence.filter_signal(
            data=source,
            filter_type=params["filter_type"],
            signal_end=params.get("signal_end"),
            decay_constant=params.get("decay_constant", 0.01),
            alpha=params.get("alpha", 0.54),
            center_position=params.get("center_position"),
            remove_lowest_signal=params["remove_lowest_signal"],
            apply_flipangle_correction=params["apply_flipangle_correction"],
            flipangle_deg=params.get("flipangle_deg"),
            filter_axis=1,
            spatial_filter=params["spatial_filter"],
            spatial_spacing_mm=params.get("spatial_spacing_mm"),
        )
        matrix = self._matrix(filtered)
        if isinstance(filtered, xr.DataArray):
            spatial = self._spatial(filtered)
        else:
            handler = copy.copy(self.handler)
            if getattr(handler, "rawdatajob0", None) is not None:
                handler.rawdatajob0 = filtered
            else:
                handler.fid = filtered
            spatial = self.strategy.reconstruct(handler=handler)
        self.filtered, self.filtered_matrix = spatial, matrix
        self.filtered_input = filtered


def _mean_preview(
    data, *, image_space=True, spectral_mode="mean", state=None, scan=None
):
    """Prepare the diagnostic mean map through the shared preprocessing owner."""
    from hypermri.utils.plot.data_preprocessor import DataPreprocessor
    from hypermri.visualization.render_data import _display_preprocessing_options

    preprocessor = DataPreprocessor(scan)
    working = data.copy(deep=True)
    if not image_space:
        # The inverse of the raw reconstruction: zero frequency at n // 2.
        axes = tuple(working.get_axis_num(dim) for dim in ("read_idx", "phase_idx"))
        values = centered_fft(np.asarray(working.data), axes)
        working = working.copy(data=values)
        working.attrs["domain"] = "kspace"
        for dim in ("read_idx", "phase_idx"):
            axis, unit = preprocessor._spatial_frequency_axis(data, dim)
            working = working.assign_coords(
                {
                    dim: xr.DataArray(
                        axis,
                        dims=dim,
                        attrs={"units": unit, "unit": unit, "domain": "kspace"},
                    )
                }
            )
    if state is None:
        if str(working.attrs.get("signal_domain", "")).lower() != "spectral":
            working = preprocessor._apply_fft(working)
        working = preprocessor.apply_data_part_transform(working, "Magnitude")
    else:
        options = _display_preprocessing_options(state, "ds0")
        options["ds0_show_image_space"] = True
        working = preprocessor.apply_preprocessing(
            working, options, widget_prefix="ds0"
        )
    if not image_space and spectral_mode == "first":
        working = working.isel(acquisition_idx=0, drop=True)
    else:
        working = working.mean("acquisition_idx", keep_attrs=True)
    return working
