"""Anatomical sequence data access and direct viewer convenience."""

import xarray as xr

from hypermri.analysis.quantitative import ADCFitOptions, T2FitOptions
from hypermri.utils.system.logging import init_default_logger

from .base_sequence import BaseSequence

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BaseAnatomy(BaseSequence):
    """Provide lazy anatomical data access and direct viewer plotting.

    Parameters
    ----------
    handler : VendorHandlerProtocol
        Vendor handler of the scan. Acquired arrays, spatial labels,
        coordinates, units and storage precision are preserved.
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
    Construction creates no viewer and performs no reconstruction. The loading
    owner applies the anatomical reconstruction policy; metadata and scientific
    defaults retain their domain owners. Handler logging settings are updated.

    Examples
    --------
    >>> scan = BaseAnatomy(handler, load_data="lazy")  # doctest: +SKIP
    >>> widget = scan.plot(plot_height=480)  # doctest: +SKIP
    """

    RAW_SOURCE_CAPABILITY = "has_fid_data"

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    @property
    def fit_results(self) -> dict[str, xr.Dataset]:
        """Return this scan's stored quantitative datasets without fitting.

        Returns
        -------
        dict of str to xarray.Dataset
            Mutable per-instance results, initially empty. ADC contains ``adc``
            in mm²/s, ``s0`` in input signal units and ``rss_adc`` in squared
            log-magnitude units, retaining all non-dynamic dimensions and labels.
            T2 contains ``t2`` in ms, ``s0`` in signal units, ``rss_t2`` in
            squared signal units and ``echo_times_s`` for reconstruction.
            Access creates only an empty dictionary if needed. Datasets are not
            copied and remain snapshots after scanner images are replaced; call
            :meth:`fit_maps` explicitly to recompute. No samples are loaded.
        """
        return self.__dict__.setdefault("_fit_results", {})

    @property
    def fit_status(self) -> dict[str, dict]:
        """Return recorded quantitative execution outcomes without calculation.

        Returns
        -------
        dict of str to dict
            Mutable per-instance entries with ``state``, ``reason``,
            ``valid_voxels`` and ``total_voxels``. States are ``computed`` (all
            map values finite), ``partial`` (some finite), ``failed`` (exception
            or none finite) and ``skipped`` (unsuitable input). Counts may be
            None without a numerical result. Finiteness is a technical indicator;
            no statistical quality or uncertainty is inferred. First access
            creates only an empty dictionary; samples and fits remain lazy.
        """
        return self.__dict__.setdefault("_fit_status", {})

    def fit_maps(
        self,
        models: list[str] | tuple[str, ...] | None = None,
        *,
        adc_options: ADCFitOptions | None = None,
        t2_options: T2FitOptions | None = None,
    ) -> dict[str, xr.Dataset]:
        """Fit supported scanner image series and store their maps on this scan.

        Parameters
        ----------
        models : list of str or tuple of str or None, optional
            Requested models; None selects all released models (ADC and T2).
            Empty selections do nothing; duplicates are collapsed in order.
            ADC supports Bruker DTI/DTIEPI original ``seq2d`` image series with
            exactly matched acquisition b-values and one documented direction.
        adc_options : ADCFitOptions or None, optional
            Finite nonnegative threshold in input signal units and existing
            voxel worker settings.
            None preserves the ADC model's scientific defaults. The scan path
            owns the physical axis: leave ``b_values_s_per_mm2`` unset and
            ``dynamic_dim`` unset or ``"bvalue_idx"``. Use ``fit_adc`` directly
            for custom input axes. b-values are s/mm²; output ADC is mm²/s.
        t2_options : T2FitOptions or None, optional
            Finite nonnegative signal threshold and worker settings for MSME
            magnitude echo decays.
            Echo coordinates must match actual scanner/acquisition times in
            seconds; dynamic_dim must be unset or echo_idx. Output T2 is ms.
            None uses the approved model's defaults, including threshold 1e-5.
            New T2 fits have a fixed upper bound of 5 s (5000 ms).

        Returns
        -------
        dict of str to xarray.Dataset
            Datasets computed in this call, including all-NaN diagnostic results.
            Objects are identical to entries in :attr:`fit_results`. All other
            dimensions (including channels/repetitions), their order, spatial
            coordinates and metadata remain; no averaging or input mutation.

        Raises
        ------
        TypeError, ValueError
            Invalid selections/options fail before data access or result changes.
        Exception
            Reconstruction or fitting exceptions are recorded in ``fit_status``
            with context and re-raised. User interrupts propagate immediately.
            Unsuitable data instead produce ``skipped`` with a specific reason.

        Notes
        -----
        Eager, deterministic fitting uses the named numerical analysis once.
        Reuses stored labelled ``seq2d`` or loads it once through reconstruction;
        never requests raw reconstruction or viewers. Each requested model loses
        its old result/status before a new attempt. Other models are preserved.
        Replaced images require an explicit new fit; accessors never recompute.

        Examples
        --------
        >>> result = scan.fit_maps(["adc"], adc_options=ADCFitOptions(
        ...     signal_threshold=1e-5))  # doctest: +SKIP
        >>> adc_map = scan.fit_results["adc"]["adc"]  # doctest: +SKIP
        """
        from ._map_fitting import _fit_scan_maps

        return _fit_scan_maps(self, models, adc_options, t2_options)

    def _load_scanner_data(self):
        """Lazily load scanner-reconstructed data."""
        if hasattr(self, "handler") and self.handler is not None:
            return self.reconstruct(reco_type="seq2d")
        return None

    def _load_data(self):
        """Lazily reconstruct raw data."""
        if hasattr(self, "handler") and self.handler is not None:
            return self.reconstruct(reco_type="raw")
        return None
