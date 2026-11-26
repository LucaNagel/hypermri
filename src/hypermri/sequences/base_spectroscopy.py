"""Provide shared spectroscopy operations and direct viewer convenience.

Logging inherits the package threshold unless explicitly configured by the caller.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import (
    add_fit_freqs_ppm,
    freq_ppm_to_hz,
    get_freq_axis,
)
from hypermri.loading.protocols import VendorHandlerProtocol

from ..analysis.spatial_filtering import SpatialFilterOptions, apply_spatial_filter
from ..analysis.spectroscopy.schema import _sampling_time_axis_from_data
from ..compatibility import resolve_frequency_axis
from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import debug_log, init_default_logger
from ..utils.time.temporal import calc_sampling_time_axis
from .base_sequence import BaseSequence

logger = init_default_logger(__name__)

DEBUG_LOG = False  # Enable debug logging for widget debugging
DEBUG_FILE = "/tmp/base_spectroscopy_debug.log"  # Use widget-specific debug file


def debug_log_base_spec(message, file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    debug_log(message=message, file_name=file_name, write_to_file=write_to_file)


class BaseSpectroscopy(BaseSequence):
    """Provide lazy spectroscopy data access, analysis helpers and direct fit review.

    Parameters
    ----------
    handler : VendorHandlerProtocol
        Vendor handler of the scan. Temporal data use acquisition_idx with
        sampling times in seconds; spatial labels, coordinates, units and
        storage precision remain owned by reconstruction.
    **kwargs
        BaseSequence loading, logging, reconstruction, path and vendor settings.
        Unknown keywords are rejected.

    Raises
    ------
    TypeError
        If ``handler`` is an ``MRIExperiment`` or a keyword is unknown.
    ValueError
        If a supplied handler uses an unsupported load_data construction setting.

    Notes
    -----
    Construction marks this sequence as spectroscopic and updates handler logging
    settings without reconstructing or initializing a viewer. Analysis and
    numerical precision retain their domain owners. Plotting and fit review use
    the direct viewer on request and do not create a plotter cache.

    Examples
    --------
    >>> scan = BaseSpectroscopy(handler, load_data="lazy")  # doctest: +SKIP
    >>> widget = scan.plot(plot_height=480)  # doctest: +SKIP
    """

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("raw",),
        "full": ("raw",),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from its handler and mark it as spectroscopic.

        Parameters
        ----------
        handler : VendorHandlerProtocol
            Vendor handler of the scan; it is used without rereading its
            acquired arrays, coordinates, units or metadata.
        **kwargs
            BaseSequence construction settings for loading, logging, reconstruction,
            path and vendor context. Unknown names are rejected.

        Raises
        ------
        TypeError
            If ``handler`` is an ``MRIExperiment`` or a keyword is unknown.
        ValueError
            If a supplied handler uses an unsupported load_data setting.

        Notes
        -----
        No reconstruction or viewer initialization occurs. Handler logging settings
        are updated and already-loaded arrays retain their storage dtype.
        """
        super().__init__(handler, **kwargs)

        logger.debug(
            f"BaseSpectroscopy: Initialized with handler type {type(self.handler)}"
        )
        logger.debug(
            f"BaseSpectroscopy: Handler has fid: {hasattr(self.handler, 'fid')}"
        )
        if hasattr(self.handler, "fid") and self.handler.fid is not None:
            logger.debug(f"BaseSpectroscopy: Fid shape: {self.handler.fid.shape}")

        # Spectroscopy sequences must advertise spectroscopic behavior regardless
        # of vendor-handler heuristics/metadata (which can be false for some CSI methods).
        self.is_spectroscopic = True

        # Initialize standardized data arrays (inherited from StandardizedMRIMixin via MRIExperiment)
        # self.scanner_data is already initialized via mixin (kspace_array no
        # longer stored)

        # self.reco_rawdata()

    def _load_scanner_data(self):
        """Lazy loading implementation for scanner reconstructed data (seq2d)."""
        if hasattr(self, "handler") and self.handler is not None:
            handler_data = getattr(self.handler, "scanner_data", None)
            if handler_data is not None:
                return handler_data
            return self.reconstruct(reco_type="seq2d")
        return None

    def _load_data(self):
        """Delegate on-demand raw loading and reconstruction to the strategy."""
        if getattr(self, "handler", None) is not None:
            return self.reconstruct(reco_type="raw")
        return None

    def _loaded_spectroscopy_data(self) -> xr.DataArray | None:
        """Return already-loaded spectroscopy data without triggering reconstruction."""
        for attr_name in ("_data", "_scanner_data"):
            candidate = self.__dict__.get(attr_name)
            if (
                isinstance(candidate, xr.DataArray)
                and "acquisition_idx" in candidate.dims
            ):
                return candidate
        return None

    def _frequency_axis_for_data(
        self,
        unit: str,
        data: xr.DataArray | None = None,
    ) -> np.ndarray | None:
        """Return a frequency axis whose length matches the selected data."""
        selected_data = data if data is not None else self._loaded_spectroscopy_data()
        npoints = None
        if (
            isinstance(selected_data, xr.DataArray)
            and "acquisition_idx" in selected_data.dims
        ):
            npoints = int(selected_data.sizes["acquisition_idx"])
            stored_axis = resolve_frequency_axis(selected_data, unit)
            if stored_axis is not None:
                try:
                    stored_axis = np.asarray(stored_axis, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    stored_axis = None
                if stored_axis is not None and stored_axis.size == npoints:
                    return stored_axis

        return get_freq_axis(
            scan=self,
            unit=unit,
            cut_off=int(getattr(self, "cut_off", 0) or 0),
            npoints=npoints,
        )

    @property
    def sampling_time_axis(self):
        loaded_data = self._loaded_spectroscopy_data()
        data_axis = _sampling_time_axis_from_data(loaded_data)
        if data_axis is not None:
            return data_axis

        # Prefer pre-calculated axis from handler if no aligned array axis exists
        # (e.g. for MGE/EPSI or before reconstruction).
        handler_axis = getattr(self.handler, "sampling_time_axis_s", None)
        if handler_axis is not None:
            return handler_axis

        return calc_sampling_time_axis(
            scan=self, npoints=self.n_acquisition - self.cut_off
        )

    @property
    def time_axis(self):
        sampling_axis = self.sampling_time_axis
        if sampling_axis is not None:
            try:
                sampling_axis = np.asarray(sampling_axis, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                sampling_axis = np.array([], dtype=float)
            if sampling_axis.size:
                return sampling_axis - sampling_axis[0]
        return calc_sampling_time_axis(scan=self, start_with_0=True)

    @property
    def ppm(self):
        """
        Chemical-shift axis in ppm that matches the loaded spectral data.

        Unlike :attr:`freq_axis_ppm`, which follows the acquisition
        parameters, this axis has the length of the already loaded data's
        ``acquisition_idx`` dimension, so it fits spectra whose first points
        were cut off.

        Returns
        -------
        numpy.ndarray, list or None
            The ``freq_axis_ppm`` coordinate of the loaded data if its length
            matches; otherwise
            :func:`hypermri.reconstruction.axes.get_freq_axis`
            with ``unit="ppm"`` for that length (or for ``n_acquisition``
            minus ``cut_off`` when no data are loaded). Nothing is loaded,
            reconstructed or cached.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        return self._frequency_axis_for_data("ppm")

    @property
    def hz(self):
        """
        Frequency axis in Hz that matches the loaded spectral data.

        Returns
        -------
        numpy.ndarray, list or None
            The Hz counterpart of :attr:`ppm`, relative to the centre
            frequency, with the same length rule and absent cases. Nothing is
            loaded, reconstructed or cached.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        return self._frequency_axis_for_data("Hz")

    def filter_signal(
        self,
        data: np.ndarray | xr.DataArray | None = None,
        filter_type: str = "None",
        decay_constant: float = 0.01,
        alpha: float = 0.54,
        center_position: int | None = None,
        signal_end: int | None = None,
        remove_lowest_signal: bool = False,
        apply_flipangle_correction: bool = False,
        flipangle_deg: float | None = None,
        filter_axis: int = 1,
        return_reconstructed: bool = False,
        *,
        spatial_filter: SpatialFilterOptions | None = None,
        spatial_spacing_mm: tuple[float, float] | None = None,
        **kwargs,
    ) -> np.ndarray | xr.DataArray:
        """
        Filter CSI k-space samples in acquisition order.

        By default, the filter is applied across k-space points in the order in
        which they were acquired. Filtering along the FID sampling dimension is
        available as an explicit opt-in via ``filter_axis=0`` or, for xarray
        input, ``filter_dim="acquisition_idx"``.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            Input data to filter. For legacy CSI raw workflow this is typically a 1D
            flattened FID vector in acquired-point/FID-sample order; raw matrices
            have (FID sample, acquired point) order. ND arrays and labelled arrays
            are also supported. Labelled CSI input includes acquisition_idx,
            phase_idx and read_idx, with optional channel, repetition and slice
            dimensions in any order. domain='spatial' requests the existing inverse
            spatial FFT before filtering and returns to that domain afterwards.
            Missing domain retains native k-space handling. If omitted, loaded
            rawdatajob0, then fid is copied from the handler.
        spatial_filter : SpatialFilterOptions or None, optional
            Explicit physical Gaussian/Fermi Read/Phase window, applied once
            before the acquired-order/FID window and lowest-signal selection.
            None retains the existing result. Labelled inputs delegate to
            apply_spatial_filter; raw CSI uses its declared Fourier/scanner-grid
            method and paired acquisition positions. Other dimensions are not
            transformed or combined. Legacy normalized spatial_width settings
            are rejected rather than reinterpreted in cycles/mm.
        spatial_spacing_mm : tuple of float or None, optional
            Explicit (Read, Phase) image sampling distances in mm. For labelled
            data, supplies missing coordinate calibration. Raw CSI otherwise
            uses fov_mm divided by its declared matrix counts. Unknown raw
            methods/layouts fail; supply calibrated labelled data instead.
        filter_type : str, optional
            Type of filter to apply: "None", "Exponential", "Hamming" or "Hann".
            Default is "None". Other names raise ``ValueError``.
        decay_constant : float, optional
            Nonnegative exponential attenuation per selected sample index, not Hz
            or seconds. Default is 0.01; the existing API does not clip this value.
        alpha : float, optional
            Dimensionless Hamming coefficient, normally in [0, 1]; default 0.54.
            Existing out-of-range arithmetic is retained without clipping.
        center_position : int, optional
            Center position for Hamming and Hann filters. If None, uses middle of signal.
        signal_end : int, optional
            End index of the significant signal segment. By default this is the
            number of k-space points to keep in acquisition order.
            For raw 1D/2D legacy input, this follows the selected filter axis.
            For ND CSI xarray input (with ``acquisition_idx`` + ``read_idx`` + ``phase_idx``),
            this is interpreted as the number of FIDs to keep in acquired k-space order.
        remove_lowest_signal : bool, optional
            If True, ``signal_end`` is interpreted as the number of points to keep and
            the lowest-mean-signal points along the filter axis are zeroed instead of
            zeroing the tail after ``signal_end``. Default is False.
        apply_flipangle_correction : bool, optional
            Whether to apply flipangle correction for hyperpolarized experiments. Default is False.
        flipangle_deg : float, optional
            Flip angle in degrees for correction. None uses the sequence's value,
            or the existing 90-degree fallback if unavailable. The correction
            operates along the selected filter axis; its physical limitations
            and singularities are retained.
        filter_axis : int, optional
            Axis along which to apply the filter for NumPy input. Default is 1,
            the k-space acquisition-order dimension after raw FID data are
            reshaped. Set to 0 to explicitly filter the FID
            acquisition/time dimension. For CSI xarray input, 1 selects ordered
            k-space filtering and 0 selects ``acquisition_idx`` filtering.
        return_reconstructed : bool, optional
            If True, reconstruct the filtered FID and return an xarray.DataArray ordered
            like ``self.data`` when available. For xarray input, filtering is
            applied directly and the filtered xarray is returned. Default is False.
        **kwargs : dict
            Additional options:

            ``filter_dim`` : str, optional
                xarray dimension name to filter along. Setting this to
                ``"acquisition_idx"`` explicitly enables FID-dimension
                filtering; it is off by default.
            ``spatial_order_mode`` : {"auto", "native", "csi_acquisition_order"}
                In ``"auto"``, xarray input with ``read_idx``/``phase_idx``
                uses the CSI acquisition order from
                ``self.handler.to_csi_acquisition()``. If that description
                cannot be built, native flattened k-space order
                is preserved as a fallback. ``"csi_acquisition_order"`` raises
                instead when metadata are unavailable.

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            Raw vectors/matrices return a flattened acquired-point/FID vector;
            higher-dimensional arrays retain shape. Labelled arrays retain input
            dimension order, coordinate values/direction, units and attributes,
            adding the effective k-space order mode for ordered CSI filtering.
            Window/FFT/correction arithmetic may promote calculation precision;
            acquired storage is unchanged. Reconstructed raw output uses the
            canonical CSI strategy and the available data dimension order.

        Raises
        ------
        ValueError
            If ``filter_type`` is unknown (checked before any data are read),
            raw data are absent, filter axis/dimension/order mode is invalid,
            explicitly requested encoding order cannot be resolved, or raw
            reconstruction lacks a handler/source, or physical spatial filtering
            lacks calibrated coordinates, verified raw metadata or valid options.
            Active legacy radial settings require migration to physical widths.
        TypeError
            If an option other than ``filter_dim`` and ``spatial_order_mode``
            (or the rejected legacy ``spatial_filter_type``/``spatial_width``)
            is passed through ``**kwargs``, for example the former keyword
            ``input_data``.
            ND NumPy reconstruction is
            unsupported; use a raw vector/matrix or labelled CSI data.
        TypeError
            If spatial_filter is not SpatialFilterOptions or None.
        Exception
            Scientific transform or reconstruction failures propagate without
            selecting another sequence or silently truncating the input.

        Notes
        -----
        Eager and deterministic for unchanged data/metadata. Inputs, source
        coordinates, scanner metadata and handler caches are not mutated. Ordered
        CSI filtering and decay diagnostics share one measured-centre permutation
        owner. NumPy raw repetition blocks retain their established per-block
        truncation, while labelled input filters each trailing group explicitly.
        Remaining legacy keyword names are
        ignored, except unsupported legacy spatial-filter keys now fail with
        migration guidance. Physical windows and viewer previews use the same
        analysis owner. A labelled result records spatial_filter provenance;
        bare raw output retains its existing NumPy layout without attributes.

        For typical workflow:
        1. Use plot_signal_decay() to determine optimal filter parameters
        2. Call filter_signal() with selected parameters on raw data
        3. Use the filtered data with reconstruct() for final results

        Examples
        --------
        >>> filtered = csi.filter_signal(data=raw_fids, signal_end=20)  # doctest: +SKIP
        >>> fid_filtered = csi.filter_signal(  # doctest: +SKIP
        ...     data=labelled_csi, filter_dim="acquisition_idx", signal_end=32
        ... )
        """
        from ._csi_kspace_filter import (
            SPATIAL_ORDER_MODES,
            SampleWindow,
            filter_along_dim,
            filter_kspace_in_acquired_order,
            filter_raw_fids,
            normalize_axis,
            raw_samples_from_handler,
            reconstruct_filtered_raw,
            resolve_filter_dim,
            window_and_truncate,
        )

        filter_dim = kwargs.pop("filter_dim", None)
        spatial_order_mode = kwargs.pop("spatial_order_mode", "auto")
        legacy_spatial = kwargs.pop("spatial_filter_type", "None")
        kwargs.pop("spatial_width", None)
        if kwargs:
            raise TypeError(
                "filter_signal got unknown options "
                f"{sorted(kwargs)}; pass the samples as data=... and use only "
                "filter_dim or spatial_order_mode as extra options."
            )
        if legacy_spatial not in (None, "None"):
            raise ValueError(
                "Legacy spatial_filter_type/spatial_width used a normalized grid radius; use spatial_filter=SpatialFilterOptions with physical widths in 1/mm."
            )
        if spatial_filter is not None and not isinstance(
            spatial_filter, SpatialFilterOptions
        ):
            raise TypeError("spatial_filter must be SpatialFilterOptions or None.")
        if spatial_spacing_mm is not None and spatial_filter is None:
            raise ValueError("spatial_spacing_mm requires an explicit spatial_filter.")

        if apply_flipangle_correction and flipangle_deg is None:
            flipangle_deg = getattr(self, "flipangle_deg", 90.0)
        window = SampleWindow(
            filter_type=filter_type,
            decay_constant=decay_constant,
            alpha=alpha,
            center_position=center_position,
            signal_end=signal_end,
            remove_lowest_signal=remove_lowest_signal,
            apply_flipangle_correction=apply_flipangle_correction,
            flipangle_deg=flipangle_deg,
        )

        if data is None:
            data = raw_samples_from_handler(self.handler)

        raw_spatial_provenance = None
        if spatial_filter is not None:
            if isinstance(data, xr.DataArray):
                data = apply_spatial_filter(
                    data,
                    spatial_filter,
                    spatial_spacing_mm=spatial_spacing_mm,
                    domain=data.attrs.get("domain", "kspace"),
                )
            else:
                from ._csi_spatial_filter import _filter_raw_csi_spatial

                data, raw_spatial_provenance = _filter_raw_csi_spatial(
                    data, self, spatial_filter, spatial_spacing_mm
                )

        if isinstance(data, xr.DataArray):
            if spatial_order_mode not in SPATIAL_ORDER_MODES:
                raise ValueError(
                    "spatial_order_mode must be one of {'auto', 'native', 'csi_acquisition_order'}"
                )
            if (
                filter_dim is None
                and filter_axis == 1
                and {"acquisition_idx", "read_idx", "phase_idx"} <= set(data.dims)
            ):
                filtered = filter_kspace_in_acquired_order(
                    data, self, window, spatial_order_mode
                )
            else:
                dim = resolve_filter_dim(data, filter_dim, filter_axis)
                filtered = filter_along_dim(data, dim, window)
            if return_reconstructed:
                debug_log_base_spec(
                    message=(
                        "return_reconstructed=True with xarray input: returning filtered xarray "
                        "without raw FID reconstruction."
                    )
                )
            return filtered

        values = np.asarray(data)
        if values.ndim > 2:
            filtered = window_and_truncate(
                values, normalize_axis(filter_axis, values.ndim), window
            )
            if return_reconstructed:
                raise ValueError(
                    "return_reconstructed=True is only supported for raw 1D/2D FID input."
                )
            return filtered

        n_repetitions = max(int(getattr(self, "n_repetitions", 1) or 1), 1)
        filtered = filter_raw_fids(
            values,
            self.n_acquisition if values.ndim == 1 else None,
            filter_axis,
            window,
            n_repetitions,
        )
        if return_reconstructed:
            return reconstruct_filtered_raw(self, filtered, raw_spatial_provenance)
        return filtered
