"""Store one scan's metadata and reconstructions through MRIExperiment.

Notes
-----
Vendor handlers own acquisition metadata and file reads. Registered strategies
own numerical reconstruction; the experiment stores successful labelled arrays
after checking their spatial geometry. Arrays retain their strategy's units,
coordinates and precision policy. Reconstruction is explicit and eager;
metadata access does not trigger it. See the class and method contracts for
loading, caching and input-preservation behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import numpy as np
import xarray as xr

from hypermri.exceptions import ReconstructionError
from hypermri.loading.contracts import ExperimentMetadata, SequenceCategory, VendorType
from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.mixins import StandardizedMRIMixin
from hypermri.reconstruction.consistency import check_reconstruction_geometry
from hypermri.reconstruction.layout import to_imaging_layout
from hypermri.utils.system.logging import debug_log, init_default_logger
from hypermri.vendors import get_vendor_handler

if TYPE_CHECKING:
    from hypermri.visualization.api import PlotOptions

logger = init_default_logger(__name__)

# Global configuration
DB_MSG = True  # Change this to False to hide detailed error messages


DEBUG_FILE = "/tmp/experiment_debug.log"
DEBUG_LOG = False


def debug_log_experiment(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


def _handler_vendor_name(handler: VendorHandlerProtocol | None) -> str:
    """Return the lowercase registry vendor key that a handler belongs to.

    ``"unknown"`` is returned for ``None``.  Handlers without a ``vendor``
    attribute are named after their class.
    """
    if not handler:
        return "unknown"

    handler_vendor = getattr(handler, "vendor", None)
    if isinstance(handler_vendor, VendorType):
        return handler_vendor.value
    if handler_vendor:
        return str(handler_vendor).strip().lower()

    class_name = handler.__class__.__name__
    class_name_aliases = {
        "BrukerTopSpinHandler": "bruker_topspin",
    }
    if class_name in class_name_aliases:
        return class_name_aliases[class_name]

    # Compatibility fallback for third-party handlers that predate the
    # standardized ``vendor`` attribute.
    return class_name.removesuffix("Handler").lower()


class MRIExperiment(StandardizedMRIMixin):
    """Vendor-independent view of one MRI scan and its reconstructions.

    The experiment wraps one vendor handler, which parses the scan directory,
    and stores the reconstructions computed from it.  Sequence classes derive
    from it.

    Attributes
    ----------
    handler : VendorHandlerProtocol or None
        Vendor handler of the scan.  Vendor-specific parameter files and raw
        data, for example ``handler.method`` or ``handler.fid`` of a Bruker
        scan, are read from it.
    params : dict
        Canonical parameters reported by the handler.  Each canonical
        parameter is also readable as an attribute, for example
        ``scan.repetition_time_ms``; it is ``None`` when the handler did not
        report it.
    reconstructions : dict of str to xarray.DataArray
        Stored reconstructions keyed by reconstruction type.

    Notes
    -----
    Reading any other vendor-handler attribute through the experiment is
    deprecated; see :meth:`__getattr__`.
    """

    _PARAMETERS_AS_ATTRIBUTES = False

    def __init__(
        self,
        path: str | Path,
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: str | None = None,
        handler_options: dict[str, Any] | None = None,
        *,
        handler: VendorHandlerProtocol | None = None,
    ):
        """Initialise a registry-backed experiment wrapper.

        Parameters
        ----------
        path
            Root directory that stores the vendor-specific measurement files.
        load_data
            If ``True``, the vendor handler immediately loads the raw data
            payload when instantiated and the experiment mirrors it as its
            primary data. Defers loading otherwise.
        log_mode
            Logging verbosity forwarded to vendor handlers and reconstruction
            strategies. Typical values are ``"error"`` (default) or ``"debug"``.
        force_vendor
            Optional vendor name that bypasses automatic vendor detection. The
            value is normalised to lowercase before being stored.
        handler_options
            Extra keyword arguments forwarded to the vendor handler factory to
            customise loading behaviour.
        handler
            Vendor handler already constructed for ``path``, for example by
            :func:`hypermri.vendors.get_vendor_handler`. When given, no vendor
            is detected and no handler is constructed; ``load_data`` must then
            match the ``load_data`` the handler was constructed with.
            ``force_vendor`` and ``handler_options`` are only stored.

        Raises
        ------
        Exception
            Errors of the vendor handler's ``get_parameters`` propagate
            unchanged; no empty parameter set is substituted.

        Notes
        -----
        The handler's ``log_mode`` attribute is set to ``log_mode``; this is
        the only change made to a given handler.
        """
        super().__init__()

        # Store log_mode for passing to reconstruction functions
        self.log_mode = log_mode
        self._forced_vendor = force_vendor.lower() if force_vendor else None
        self._handler_options = handler_options or {}

        # Use standardized path handling
        self.path = path

        if handler is None:
            handler = self._get_vendor_handler(
                str(self.path),
                load_data,
                log_mode,
                self._forced_vendor,
                self._handler_options,
            )
        self.handler = handler

        # Store log_mode on handler if it exists
        if self.handler and hasattr(self.handler, "__dict__"):
            self.handler.log_mode = log_mode
            handler_params = self.handler.get_parameters()
            if handler_params:
                self.set_parameters(handler_params)
            if load_data:
                handler_reco = getattr(self.handler, "data", None)
                handler_scanner = getattr(self.handler, "scanner_data", None)
                if handler_reco is not None:
                    self.set_primary_data(handler_reco, "data")
                elif handler_scanner is not None:
                    self.set_primary_data(handler_scanner, "scanner_data")
        # Note: Unlike the strict version, we allow handler to be None for backward compatibility
        # This allows Study class to filter out invalid paths later via scan.handler is not None

        # Initialize metadata
        self._metadata = None
        self.reconstructions = {}
        self._initialize_masks()

    @property
    def id(self) -> str:
        """Return the experiment identifier supplied by the vendor handler.

        Returns
        -------
        str
            Identifier emitted by :py:attr:`~hypermri.loading.protocols.VendorHandlerProtocol.id`.
            ``"unknown"`` is returned if no handler was initialised.
        """
        if not self.handler:
            return "unknown"
        return self.handler.id

    @property
    def type(self) -> str:
        """Return the sequence name associated with the experiment.

        Returns
        -------
        str
            Sequence identifier reported by the vendor handler. ``"unknown"``
            is returned when the experiment lacks a handler.
        """
        if not self.handler:
            return "unknown"
        return self.handler.type

    @property
    def metadata(self) -> ExperimentMetadata:
        """Generate and cache experiment metadata.

        Returns
        -------
        ExperimentMetadata
            Dataclass describing vendor, sequence, and parameter snapshot for
            the experiment.
        """
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata

    def reconstruct(self, reco_type: str, **kwargs) -> xr.DataArray | None:
        """Run the reconstruction strategy selected for this scan and store the result.

        The strategy class and its default options come from
        :func:`hypermri.reconstruction.selection.find_reconstruction` for the
        scan's vendor, sequence and ``reco_type``.

        The result replaces ``reconstructions[reco_type]`` and becomes
        ``scanner_data`` for ``"seq2d"`` or ``data`` for ``"raw"``.  It is
        stored only if it describes the same voxels as every stored
        reconstruction of another type, as checked by
        :func:`hypermri.reconstruction.consistency.check_reconstruction_geometry`.

        The raw reconstruction of a non-spectroscopic scan is stored in the
        dimension layout of :func:`hypermri.reconstruction.layout.to_imaging_layout`:
        the dimensions are transposed to the order of the ``2dseq``
        reconstruction and the size-1 dimensions that the strategy omitted are
        added.  Values and attributes are unchanged.

        Parameters
        ----------
        reco_type
            Name of the reconstruction workflow to execute, such as ``"raw"``
            or ``"seq2d"``.
        **kwargs
            Keyword arguments that override the strategy's default options.
            Each name must be in the strategy's ``accepted_options``; the
            strategy's ``reconstruct`` method documents meaning, units and
            defaults. Generic Bruker raw reconstruction also accepts
            ``fov_shift_source="scanner"`` for the bounded FLASH geometry
            documented by
            :meth:`hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy.reconstruct`;
            the other strategies do not accept it.

        Returns
        -------
        xarray.DataArray or None
            Reconstructed dataset as stored, in the layout described above, or
            ``None`` if the strategy explicitly returns nothing.

        Raises
        ------
        ReconstructionError
            If no handler is available, if no strategy is selected for the
            vendor, sequence and ``reco_type``, if an option is not accepted
            by the selected strategy (checked before any data are read), if
            the strategy itself raises an exception, or if the
            result differs from a stored reconstruction of another type in
            its spatial order, sizes, mm coordinates or orientation
            attributes.  Nothing is stored in these cases.

        Notes
        -----
        This eager operation records the latest attempt's outcome for Study
        summaries, including failures even when an earlier result is cached.
        An explicit ``no_reconstruction=True`` returning ``None`` is recorded
        as not run; another ``None`` result is recorded as failed. A successful
        return records completion of the strategy and geometry checks, not
        scientific validation of the image or completeness of the acquisition.

        Examples
        --------
        >>> scan = load_scan("path/to/scan", mode="lazy")  # doctest: +SKIP
        >>> image = scan.reconstruct("raw")  # doctest: +SKIP
        """
        outcomes = self.__dict__.setdefault("_reconstruction_status", {})
        if not self.handler:
            outcomes[reco_type] = "Failed"
            raise ReconstructionError("No handler available for this experiment")

        try:
            from hypermri.reconstruction.selection import find_reconstruction

            selection = find_reconstruction(
                self._get_vendor_name(), self.type, reco_type
            )
            if selection is None:
                raise ReconstructionError(
                    f"{self.id}: No reconstruction strategy found for vendor "
                    f"'{self._get_vendor_name()}', sequence '{self.type}', "
                    f"reconstruction type '{reco_type}'",
                    strategy=f"{self._get_vendor_name()}.{self.type}.{reco_type}",
                    sequence=self.type,
                )
            strategy_class, default_options = selection

            unknown = sorted(set(kwargs) - strategy_class.accepted_options)
            if unknown:
                raise ValueError(
                    f"reconstruct({reco_type!r}) does not accept the option(s) "
                    f"{', '.join(unknown)} for {strategy_class.__name__}; "
                    "accepted options are "
                    f"{', '.join(sorted(strategy_class.accepted_options)) or 'none'}."
                )

            final_options = {**default_options, **kwargs}
            strategy = strategy_class()
            logger.debug(
                f"{self.id} ({self.type}): Using strategy: {strategy_class.__name__}"
            )
            logger.debug(f"Reconstruction options: {final_options}")

            # Perform reconstruction
            reconstructed_data = strategy.reconstruct(self.handler, **final_options)

            # Validate reconstruction result
            if reconstructed_data is None:
                outcomes[reco_type] = (
                    "Not run" if final_options.get("no_reconstruction") else "Failed"
                )
                logger.warning(
                    f"Reconstruction failed for {reco_type} - strategy returned None"
                )
                return None

            if (
                reco_type in {"raw", "seq2d"}
                and self._guess_sequence_category() is not SequenceCategory.SPECTROSCOPY
                and "map_idx" not in reconstructed_data.dims
                and all(
                    dim in reconstructed_data.dims
                    for dim in ("read_idx", "phase_idx", "slice_idx")
                )
            ):
                reconstructed_data = to_imaging_layout(
                    reconstructed_data,
                    sampling_time_s=getattr(self.handler, "sampling_time_axis_s", None),
                    frame_time_s=getattr(self.handler, "frame_time_s", None),
                )

            # Store reconstruction using standardized data handling
            self._store_reconstruction(reco_type, reconstructed_data)
            outcomes[reco_type] = "Success"

            return reconstructed_data

        except Exception as e:
            outcomes[reco_type] = "Failed"
            raise ReconstructionError(
                f"Reconstruction failed: {e}",
                strategy=f"{self._get_vendor_name()}.{self.type}.{reco_type}",
                sequence=self.type,
            ) from e

    def load_reconstruction(self, recon_num: int = 1, set_as_primary: bool = True):
        """
        Load a specific pdata reconstruction and optionally set it as the primary data.

        This method delegates to the vendor-specific handler to load different
        reconstructions (e.g., pdata/1, pdata/2, etc.) after the scan object has
        been initialized.

        Parameters
        ----------
        recon_num : int, optional
            The pdata reconstruction number to load (e.g., 1 for pdata/1, 2 for pdata/2).
            Default is 1.
        set_as_primary : bool, optional
            If True, sets the loaded reconstruction as the primary data (accessible via
            scan.scanner_data). If False, only returns the data without modifying the scan's
            primary data. Default is True.

        Returns
        -------
        xr.DataArray or None
            The loaded reconstruction as an xarray DataArray, or None if loading failed.

        Examples
        --------
        >>> # Load a study and get a scan
        >>> study = Study('/path/to/study')
        >>> scan = study[5]
        >>>
        >>> # Load alternative reconstruction (pdata/2) and set as primary
        >>> scan.load_reconstruction(recon_num=2)
        >>> scan.scanner_data  # Now contains pdata/2 data
        >>>
        >>> # Load reconstruction without changing primary data (for comparison)
        >>> pdata3 = scan.load_reconstruction(recon_num=3, set_as_primary=False)
        >>> # Compare: scan.scanner_data still has pdata/2, pdata3 has pdata/3
        """
        if hasattr(self.handler, "load_reconstruction"):
            # Load from handler (this sets the handler's _data_array)
            result = self.handler.load_reconstruction(
                recon_num=recon_num,
                set_as_primary=False,  # Don't set on handler, we'll set on self instead
            )

            # Also set as primary on the MRIExperiment/sequence object if requested
            if set_as_primary and result is not None:
                self.set_primary_data(result, "scanner_data")

            return result
        else:
            raise NotImplementedError(
                f"load_reconstruction is not implemented for {type(self.handler).__name__}"
            )

    def get_available_reconstructions(self) -> dict[str, Any]:
        """Describe the reconstructions available for this experiment.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of ``"raw"`` and ``"seq2d"``, where a strategy is selected
            by :func:`hypermri.reconstruction.selection.find_reconstruction`,
            to a dictionary with the strategy class name
            (``"strategy_class"``) and a copy of its default options
            (``"default_options"``).

        Notes
        -----
        Constructs no strategy, reads no data and starts no reconstruction.
        """
        from hypermri.reconstruction.selection import find_reconstruction

        vendor_name = self._get_vendor_name()
        available = {}
        for reco_type in ("raw", "seq2d"):
            selection = find_reconstruction(vendor_name, self.type, reco_type)
            if selection is not None:
                strategy_class, options = selection
                available[reco_type] = {
                    "strategy_class": strategy_class.__name__,
                    "default_options": options,
                }

        return available

    def plot(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        options: "PlotOptions | None" = None,
        **option_values: Any,
    ):
        """Show this scan, or data belonging to it, in the browser viewer.

        Same as :func:`hypermri.visualization.plot` with this scan as data
        (or as ``scan`` for explicit ``data``).

        Parameters
        ----------
        data : xarray.DataArray, xarray.Dataset or None, optional
            Labelled samples or maps of this scan, for example
            ``scan.scanner_data`` (scanner reconstruction) or
            ``scan.fit_results``. None shows ``self.primary_data``. Not
            modified.
        options : PlotOptions or None, optional
            Display settings.
        **option_values
            PlotOptions fields given as keywords, for example
            ``anatomy=t2_scan`` or ``overlay=scan.fit_results``. Use these or
            ``options``, not both.

        Returns
        -------
        ScanView
            The viewer; display it as the last expression of a notebook cell.

        Raises
        ------
        TypeError
            For unknown options, options combined with keywords, or data
            that are not labelled.
        ValueError
            For removed options (the message names the replacement) and data
            the viewer cannot show.
        ImportError
            If the optional visualization dependencies are unavailable.

        Examples
        --------
        >>> view = scan.plot()  # doctest: +SKIP
        >>> view = scan.plot(anatomy=t2_scan, overlay=scan.fit_results)  # doctest: +SKIP
        >>> view = scan.plot(scan.scanner_data, plot_height=500)  # doctest: +SKIP
        """
        from hypermri.visualization.api import _scan_plot_options, plot

        options = _scan_plot_options(options, option_values, "scan.plot")
        if data is None:
            return plot(self, options)
        return plot(data, options, scan=self)

    def plot_dicom(
        self,
        recon_num: int = 1,
        series_uid: str | None = None,
        options: "PlotOptions | None" = None,
        **option_values: Any,
    ):
        """Load one DICOM series of this scan and show it in the browser viewer.

        Parameters
        ----------
        recon_num : int, optional
            Scanner reconstruction number containing the DICOM series; default 1.
        series_uid : str or None, optional
            Series to load; None uses the vendor loader's selection.
        options : PlotOptions or None, optional
            Display settings. Without ``coordinate_mode`` the series is shown in
            scanner coordinates (``"xyz"``).
        **option_values
            PlotOptions fields given as keywords, for example
            ``plot_height=500``. Use these or ``options``, not both.

        Returns
        -------
        ScanView
            The viewer of the loaded series. The vendor loader owns dimension
            order, coordinate direction and units; the series is loaded once and
            not modified.

        Raises
        ------
        NotImplementedError
            If the vendor handler does not support DICOM loading.
        FileNotFoundError
            If the requested reconstruction contains no DICOM series.
        TypeError, ValueError
            For unknown or removed options and data the viewer cannot show.
        ImportError
            If the optional visualization dependencies are unavailable.

        Examples
        --------
        >>> view = scan.plot_dicom(recon_num=1)  # doctest: +SKIP
        """
        from dataclasses import replace

        from hypermri.visualization.api import _scan_plot_options, plot

        options = _scan_plot_options(options, option_values, "plot_dicom")
        if options.coordinate_mode is None:
            options = replace(options, coordinate_mode="xyz")
        if not hasattr(self.handler, "load_dicom_data"):
            raise NotImplementedError(
                f"DICOM loading is not implemented for {type(self.handler).__name__}"
            )
        data = self.handler.load_dicom_data(
            recon_num=recon_num,
            series_uid=series_uid,
        )
        if data is None:
            raise FileNotFoundError(
                f"No DICOM series found for reconstruction {recon_num}"
            )
        return plot(data, options, scan=self)

    def _get_vendor_handler(
        self,
        path: str,
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: str | None = None,
        handler_options: dict[str, Any] | None = None,
    ) -> VendorHandlerProtocol | None:
        """Instantiate a vendor handler resolved via the registry.

        Parameters
        ----------
        path
            Filesystem path that contains the scan data.
        load_data
            If ``True``, the handler eagerly loads the raw data payload.
        log_mode
            Logging level hint forwarded to handler creation.
        force_vendor
            Optional vendor override. When ``None`` the detection heuristic is
            executed.
        handler_options
            Additional keyword arguments forwarded to the handler factory.

        Returns
        -------
        VendorHandlerProtocol or None
            Concrete handler instance if detection and construction succeed,
            otherwise ``None``.
        """
        return get_vendor_handler(
            source=path,
            load_data=load_data,
            log_mode=log_mode,
            force_vendor=force_vendor,
            handler_options=handler_options,
        )

    def _get_vendor_name(self) -> str:
        """Return the registry-compatible vendor identifier for the handler.

        Returns
        -------
        str
            Lowercase vendor key (for example ``"bruker"`` or ``"siemens"``).
        """
        forced_vendor = getattr(self, "_forced_vendor", None)
        if forced_vendor:
            return forced_vendor
        return _handler_vendor_name(self.handler)

    def _initialize_masks(self, masks: xr.Dataset | None = None) -> None:
        """Ensure every experiment exposes an xarray-backed mask container."""
        if isinstance(masks, xr.Dataset):
            self.masks = masks.copy(deep=True)
        else:
            self.masks = xr.Dataset()

    def get_finalized_masks(self) -> xr.Dataset:
        """Return the masks persisted by the modern interactive mask plugin."""
        masks = getattr(self, "masks", None)
        if isinstance(masks, xr.Dataset):
            return masks.copy(deep=True)
        return xr.Dataset()

    def create_mask(self, options: "PlotOptions | None" = None, **option_values: Any):
        """Show this scan in the browser viewer with the mask tool open.

        Shortcut for ``view = plot(scan, options); view.tool = "masks"``.
        Masks drawn and saved in the tool are stored in ``self.masks`` on the
        scan's voxel grid, as float32 voxel coverage in [0, 1] with the
        weighted voxel count and, with known spacing, the volume in mL as
        attributes ``weighted_voxel_count`` and ``volume_ml``.

        Parameters
        ----------
        options : PlotOptions or None, optional
            Display settings, for example an ``anatomy`` to draw on.
        **option_values
            PlotOptions fields given as keywords, for example
            ``anatomy=t2_scan``. Use these or ``options``, not both.

        Returns
        -------
        ScanView
            The viewer. The acquired data are not modified.

        Raises
        ------
        TypeError, ValueError
            For unknown or removed options and data the viewer cannot show.
        ImportError
            If the optional visualization dependencies are unavailable.

        Examples
        --------
        >>> view = scan.create_mask(anatomy=t2_scan)  # doctest: +SKIP
        """
        from hypermri.visualization.api import _scan_plot_options, plot

        view = plot(self, _scan_plot_options(options, option_values, "create_mask"))
        view.tool = "masks"
        return view

    def project_masks_to(
        self,
        target: xr.DataArray | xr.Dataset | "MRIExperiment",
        names: str | list[str] | tuple[str, ...] | None = None,
        *,
        store: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Project masks from this experiment onto a target data grid.

        Projection uses scanner-space ``x/y/z`` coordinates, so target datasets
        may have different position, FOV, orientation, or resolution.

        Parameters
        ----------
        target : xr.DataArray, xr.Dataset, or MRIExperiment
            Data or experiment whose spatial grid receives the projected masks.
            If an xarray Dataset contains multiple variables, pass
            ``target_variable`` through ``kwargs``.
        names : str, sequence of str, optional
            Mask name or names to project. If omitted, all finalized masks on
            this experiment are projected.
        store : bool, default False
            If True, merge the projected masks into ``target.masks``. This
            requires an object target with a writable ``masks`` attribute.
            Existing target masks with the same names are replaced.
        **kwargs
            Projection options forwarded to
            :func:`hypermri.analysis.anatomical.masks.project_mask_to_data_grid`.
            Supported keywords are ``target_variable``, ``source_variable``,
            ``spatial_dims``, ``method``, ``coverage_samples``, ``threshold``,
            ``plane_tolerance_mm``, and ``batch_size``. ``source_data`` defaults
            to ``self.primary_data`` and ``source_obj`` defaults to ``self``;
            both may be overridden. ``target_obj`` may be supplied when target
            geometry must come from a separate scan object.

        Returns
        -------
        xr.Dataset
            One fractional target-grid DataArray per selected mask. Variable
            names and stored color/opacity attributes are preserved.

        Raises
        ------
        ValueError
            If no masks exist, a requested mask is missing, or ``store=True``
            is used with a target that cannot store masks.
        """
        from hypermri.analysis.anatomical.masks import project_mask_to_data_grid

        masks = self.get_finalized_masks()
        if not isinstance(masks, xr.Dataset) or not masks.data_vars:
            raise ValueError("No masks available on this experiment.")

        if names is None:
            selected_names = list(masks.data_vars)
        elif isinstance(names, str):
            selected_names = [names]
        else:
            selected_names = list(names)

        missing = [name for name in selected_names if name not in masks]
        if missing:
            raise ValueError(f"Mask(s) not found: {missing}")

        explicit_target_obj = kwargs.pop("target_obj", None)
        target_obj = (
            explicit_target_obj
            if explicit_target_obj is not None
            else target
            if not isinstance(target, xr.DataArray | xr.Dataset)
            else None
        )
        source_data = kwargs.pop("source_data", self.primary_data)
        source_obj = kwargs.pop("source_obj", self)
        projected = xr.Dataset(
            {
                name: project_mask_to_data_grid(
                    masks[name],
                    target,
                    source_data=source_data,
                    source_obj=source_obj,
                    target_obj=target_obj,
                    name=name,
                    **kwargs,
                )
                for name in selected_names
            }
        )

        if store:
            if not hasattr(target, "masks"):
                raise ValueError("store=True requires a target object with masks.")
            existing = getattr(target, "masks", xr.Dataset())
            if not isinstance(existing, xr.Dataset):
                existing = xr.Dataset()
            for name in selected_names:
                if name in existing:
                    existing = existing.drop_vars(name)
            target.masks = xr.merge([existing, projected])

        return projected

    def apply_mask_to(
        self,
        target: xr.DataArray | xr.Dataset | "MRIExperiment",
        mask_name: str | None = None,
        *,
        reduction: Literal["none", "sum", "mean"] | None = None,
        signal_part: Literal["complex", "magnitude"] = "complex",
        **kwargs,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Apply one mask from this experiment to a target dataset.

        Parameters
        ----------
        target : xr.DataArray, xr.Dataset, or MRIExperiment
            Data or experiment to mask. Scanner-space coordinates are read from
            the xarray data or derived from the target experiment geometry.
        mask_name : str, optional
            Name of the finalized mask stored on this experiment. It may be
            omitted only when exactly one mask is available.
        reduction : {"none", "sum", "mean"} or None, optional
            Return the complete weighted grid, spatial sum or spatial mean.
            None preserves the existing ``mean_along_space`` option (False by
            default). True ``mean_along_space`` only permits None or ``"mean"``.
            Non-finite samples are excluded from reductions; empty sums are zero
            and empty means are NaN. Non-spatial dimensions retain their order.
        signal_part : {"complex", "magnitude"}, default "complex"
            Retain input values and phase, or take each voxel's magnitude before
            weighting/reduction. Opposite phases can cancel in complex sums or
            means. This method does not Fourier transform or preprocess signals;
            pass the desired signal domain as ``target``.
        **kwargs
            Options forwarded to
            :func:`hypermri.analysis.anatomical.masks.apply_projected_mask`.
            Supported keywords are ``target_variable``, ``source_variable``,
            ``spatial_dims``, ``method``, ``coverage_samples``, ``threshold``,
            ``fill_value``, ``mean_along_space``, ``return_mask``,
            ``plane_tolerance_mm``, and ``batch_size``. ``source_data`` defaults
            to ``self.primary_data`` and ``source_obj`` defaults to ``self``;
            both may be overridden. ``target_obj`` may also be supplied.

        Returns
        -------
        xr.DataArray or tuple[xr.DataArray, xr.DataArray]
            Masked target data, or a spatially weighted mean when
            ``mean_along_space=True``, or an explicitly requested sum/mean.
            Coordinates/direction and units retain the target convention; only
            reduced spatial dimensions/coordinates are removed. Float64 weights
            promote float32/complex64 signals to float64/complex128. Explicit
            reduction/magnitude requests retain target name/attributes and add
            ``applied_mask``, ``mask_reduction`` and ``mask_signal_part`` metadata.
            If ``return_mask=True``, the projected
            target-grid mask is returned as the second tuple element.

        Raises
        ------
        ValueError
            If no masks exist, ``mask_name`` is ambiguous, or the requested mask
            does not exist, options conflict, or target/projection geometry is
            inconsistent. Unknown projection options raise TypeError.
        TypeError
            If data types or forwarded projection options are unsupported.

        Notes
        -----
        Resolves the finalized mask and delegates once to the anatomical mask
        operation. Projection is eager and signal arithmetic retains supported
        xarray laziness. Acquired values and stored masks are not mutated; no
        viewer, new reconstruction cache or mask persistence is created.
        Existing calls retain their original numerical and metadata behavior.

        Examples
        --------
        Keep the target shape and fill outside-mask voxels with ``NaN``::

            masked = source_scan.apply_mask_to(
                target_scan,
                mask_name="mask_1",
                fill_value=np.nan,
            )

        Calculate a weighted curve while retaining non-spatial dimensions::

            curve = source_scan.apply_mask_to(
                target_scan,
                mask_name="mask_1",
                reduction="mean",
                signal_part="magnitude",
            )
        """
        from hypermri.analysis.anatomical.masks import apply_projected_mask

        masks = self.get_finalized_masks()
        if not isinstance(masks, xr.Dataset) or not masks.data_vars:
            raise ValueError("No masks available on this experiment.")

        if mask_name is None:
            if len(masks.data_vars) != 1:
                raise ValueError(
                    "mask_name is required when multiple masks are available."
                )
            mask_name = next(iter(masks.data_vars))
        if mask_name not in masks:
            raise ValueError(f"Mask {mask_name!r} not found.")

        explicit_target_obj = kwargs.pop("target_obj", None)
        target_obj = (
            explicit_target_obj
            if explicit_target_obj is not None
            else target
            if not isinstance(target, xr.DataArray | xr.Dataset)
            else None
        )
        source_data = kwargs.pop("source_data", self.primary_data)
        source_obj = kwargs.pop("source_obj", self)
        return apply_projected_mask(
            target,
            masks[mask_name],
            source_data=source_data,
            source_obj=source_obj,
            target_obj=target_obj,
            name=mask_name,
            reduction=reduction,
            signal_part=signal_part,
            **kwargs,
        )

    def interactive_fit(
        self,
        data: Optional[xr.DataArray] = None,
        options: "PlotOptions | None" = None,
        *,
        fit_results: xr.Dataset | None = None,
        fit_params: xr.Dataset | None = None,
        noise: xr.DataArray | None = None,
        **option_values: Any,
    ):
        """Show these spectra in the browser viewer with the fit tool open.

        In the fit tool the lines are chosen in the spectrum, all voxels are
        fitted, and single voxels are refitted or reviewed. Fits are stored on
        the scan (``fit_results_review`` and ``fit_params_review``, see
        :func:`~hypermri.visualization.fit_tool.store_fit_on_scan`).

        Parameters
        ----------
        data : xarray.DataArray or None, optional
            Labelled FID samples to show and fit, with ``acquisition_idx`` and
            sampling times in seconds. None uses ``self.primary_data``. Not
            modified.
        options : PlotOptions or None, optional
            Display settings.
        fit_results, fit_params : xarray.Dataset or None, optional
            An earlier fit to continue from, given together and of the same
            lineshape (Lorentz or Voigt). They are stored on the scan with
            :func:`~hypermri.visualization.fit_tool.store_fit_on_scan` before
            the view opens, so the tool shows them and takes its start values
            from them.
        noise : xarray.DataArray or None, optional
            Per-voxel acquisition-tail noise of
            :func:`hypermri.analysis.spectroscopy.propagate_acquisition_tail_noise`,
            per real or imaginary quadrature in the signal units of the data.
            It is attached as the coordinate ``noise_sigma_quadrature`` so that
            fits report a finite ``fit_reduced_chi_squared``; without it, and
            without that coordinate on ``data``, the value is NaN.
        **option_values
            PlotOptions fields given as keywords. Use these or ``options``,
            not both.

        Returns
        -------
        ScanView
            The viewer. Nothing is fitted until the fit tool is run.

        Raises
        ------
        ValueError
            If only one of ``fit_results`` and ``fit_params`` is given, they
            describe different lineshapes, ``noise`` is invalid, or an option
            is removed; all checked before the view is built.
        TypeError
            For unknown options or data that are not labelled.
        ImportError
            If the optional visualization dependencies are unavailable.

        Examples
        --------
        >>> view = scan.interactive_fit(noise=noise)  # doctest: +SKIP
        >>> scan.fit_results_review  # after running the fit  # doctest: +SKIP
        """
        from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
            _require_same_lineshape,
        )
        from hypermri.visualization.api import _scan_plot_options, plot
        from hypermri.visualization.fit_tool import store_fit_on_scan

        options = _scan_plot_options(options, option_values, "interactive_fit")
        if (fit_results is None) != (fit_params is None):
            raise ValueError(
                "interactive_fit: give fit_results and fit_params together to "
                "continue an earlier fit."
            )
        if fit_params is not None:
            _require_same_lineshape("interactive_fit", fit_params, fit_results)
        if data is None:
            data = self.primary_data
        if noise is not None:
            from hypermri.analysis.spectroscopy.noise_estimation import (
                attach_acquisition_tail_noise,
            )

            data = attach_acquisition_tail_noise(data, noise=noise)
        if fit_params is not None:
            store_fit_on_scan(self, fit_params, fit_results, data)
        view = plot(data, options, scan=self)
        view.tool = "fit"
        return view

    def _create_metadata(self) -> ExperimentMetadata:
        """Assemble :class:`ExperimentMetadata` from handler information.

        Returns
        -------
        ExperimentMetadata
            Metadata populated with handler parameters and inferred sequence
            category.
        """
        vendor_name = self._get_vendor_name()

        # Determine vendor type
        try:
            vendor_type = VendorType(vendor_name)
        except ValueError:
            vendor_type = VendorType.UNKNOWN

        parameters = self.handler.get_parameters()

        return ExperimentMetadata(
            id=self.id,
            sequence_name=self.type,
            vendor=vendor_type,
            category=self._guess_sequence_category(),
            parameters=parameters,
            custom_fields=(
                {"vendor_name": vendor_name}
                if vendor_type is VendorType.UNKNOWN
                else {}
            ),
        )

    def _guess_sequence_category(self):
        """Resolve the category from the vendor-specific sequence mapping.

        Returns
        -------
        SequenceCategory
            The mapped category. Unregistered methods retain the handler
            spectroscopy flag or the anatomical fallback.
        """
        from hypermri.loading.contracts import SequenceCategory
        from hypermri.sequences.registry import get_sequence_registry

        vendor = _handler_vendor_name(self.handler)
        entry = get_sequence_registry().resolve_sequence(
            self.type, None if vendor == "unknown" else vendor
        )
        if entry is not None:
            return entry["category"]
        if getattr(self.handler, "is_spectroscopic", False):
            return SequenceCategory.SPECTROSCOPY
        return SequenceCategory.ANATOMICAL

    def _store_reconstruction(
        self, reco_type: str, reconstructed_data: xr.DataArray
    ) -> None:
        """Store reconstructed data in the standard experiment registry.

        Before anything is stored, ``reconstructed_data`` is checked against
        every stored reconstruction of another type with
        :func:`hypermri.reconstruction.consistency.check_reconstruction_geometry`.
        A reconstruction of the same type is replaced without a check.

        Parameters
        ----------
        reco_type
            Name of the reconstruction that produced ``reconstructed_data``.
        reconstructed_data
            Resulting :class:`xarray.DataArray` emitted by the reconstruction
            strategy.

        Raises
        ------
        ValueError
            If ``reco_type`` is ``"raw"`` and ``reconstructed_data`` encodes
            k-space rather than image space data, which is no longer supported
            for persistent storage, or if its spatial order, sizes, mm
            coordinates or orientation attributes differ from a stored
            reconstruction of another type.  Nothing is stored in either case.
        """
        if (
            reco_type == "raw"
            and hasattr(reconstructed_data, "attrs")
            and reconstructed_data.attrs.get("domain") == "kspace"
        ):
            raise ValueError(
                "kspace_array is no longer supported for permanent storage. Use during reconstruction only."
            )
        for stored_type, stored_data in self.reconstructions.items():
            if stored_type != reco_type:
                check_reconstruction_geometry(
                    reconstructed_data,
                    stored_data,
                    candidate_name=reco_type,
                    reference_name=stored_type,
                )

        if reco_type == "seq2d":
            self.set_primary_data(reconstructed_data, "scanner_data")
        elif reco_type == "raw":
            self.set_primary_data(reconstructed_data, "data")

        # Always store in reconstructions dict for reference
        self.reconstructions[reco_type] = reconstructed_data

    def get(self, name: str, default: Any = None) -> Any:
        """Return an attribute of the vendor handler, or ``default``.

        This mirrors ``dict.get`` for the duck-typed geometry and timing
        helpers that accept either a scan or a vendor handler, and delegates
        to the ``get`` method of the vendor handler.

        Parameters
        ----------
        name : str
            Attribute name, for example a canonical parameter such as
            ``"fov_mm"`` or a vendor-specific attribute such as ``"method"``.
        default : Any, optional
            Value returned when the handler has no attribute ``name``.

        Returns
        -------
        Any
            The current value of the handler attribute, or ``default``.

        Raises
        ------
        AttributeError
            If the experiment has no handler, or the handler has no ``get``
            method.

        Notes
        -----
        The value is read from the handler at call time.  It can differ from
        the canonical parameter in :attr:`params`, which the experiment copied
        when it was constructed, if the handler attribute was changed later.
        """
        return self.handler.get(name, default)

    def __getattr__(self, name: str) -> Any:
        """Resolve a canonical acquisition parameter.

        Python calls this method only for names that the experiment does not
        define itself.  A canonical parameter (see
        :meth:`~hypermri.mixins.StandardizedMRIMixin.parameter_keys`) is read
        from :attr:`params`; a canonical parameter that the vendor handler did
        not provide is ``None``.  Vendor-handler attributes such as the Bruker
        ``method`` or ``fid`` are not forwarded; read them from
        :attr:`handler`.

        Parameters
        ----------
        name : str
            Attribute name requested by Python's attribute lookup.

        Returns
        -------
        Any
            The parameter value from :attr:`params`, or ``None`` for a
            canonical parameter without a value.

        Raises
        ------
        AttributeError
            If ``name`` is not a canonical parameter.  When the vendor handler
            has an attribute or parameter of that name, the message names
            ``scan.handler.<name>`` as the replacement.

        Examples
        --------
        >>> scan = load_scan("path/to/scan")  # doctest: +SKIP
        >>> scan.repetition_time_ms  # doctest: +SKIP
        >>> scan.handler.method["PVM_EncSteps1"]  # doctest: +SKIP
        """
        state = self.__dict__
        params = state.get("params") or {}
        if name in params:
            return params[name]
        if name in self.parameter_keys():
            return None

        handler = state.get("handler")
        if handler is not None:
            # Membership checks only: evaluating a lazy handler property such
            # as ``fid`` here would read raw data merely to build an error.
            handler_params = vars(handler).get("params") or {}
            if (
                name in vars(handler)
                or hasattr(type(handler), name)
                or name in handler_params
            ):
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'. "
                    f"'{name}' belongs to the vendor handler; read it as "
                    f"'scan.handler.{name}'."
                )

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __lt__(self, other):
        """Compare experiments by deferring to handler ordering.

        Parameters
        ----------
        other
            Another :class:`MRIExperiment` instance.

        Returns
        -------
        bool
            ``True`` when both experiments have handlers and the current handler
            is strictly less than ``other.handler``.
        """
        if not self.handler or not other.handler:
            return False
        debug_log_experiment(message=f"Comparing {self.handler} < {other.handler}")
        return self.handler < other.handler

    def __gt__(self, other):
        """Check if this experiment sorts higher than another.

        Parameters
        ----------
        other
            Another :class:`MRIExperiment` instance.

        Returns
        -------
        bool
            ``True`` when both experiments provide handlers and the current
            handler compares greater than ``other.handler``.
        """
        if not self.handler or not other.handler:
            return False
        return self.handler > other.handler

    @property
    def display_name(self) -> str:
        """Return a human-readable experiment description.

        Returns
        -------
        str
            String representation of the underlying handler, or a fallback that
            marks the experiment invalid when no handler is available.
        """
        if not self.handler:
            return f"Invalid experiment: {self.path}"
        return str(self.handler)

    def paper_summary(
        self, *, display_output: bool = False, include_missing: bool = False
    ):
        """Return a paper-ready summary table for this experiment.

        Parameters
        ----------
        display_output
            If ``True`` and IPython is available, explicitly display the styled
            table. The default is ``False`` because notebooks already render
            the returned Styler when it is the last expression in a cell.
        include_missing
            If ``True``, include unavailable parameters as ``"MISSING"`` rows.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled summary table. The underlying DataFrame is available via
            ``styler.data``.
        """
        import pandas as pd

        missing = "MISSING"

        def is_missing(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return value == ""
            if isinstance(value, float) and np.isnan(value):
                return True
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    return len(value) == 0
                except TypeError:
                    return False
            return False

        def get_value(attribute: str) -> Any:
            try:
                return getattr(self, attribute)
            except Exception:
                handler = getattr(self, "handler", None)
                if handler is None:
                    return None
                try:
                    return getattr(handler, attribute)
                except Exception:
                    return None

        def format_scalar(value: Any) -> str:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        def format_value(value: Any, unit: str | None = None) -> str:
            if is_missing(value):
                return missing
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, str):
                # Otherwise will not be printed:
                if value.startswith("<") and value.endswith(">"):
                    value = value[1:-1]
            if isinstance(value, (list, tuple)):
                rendered = " x ".join(format_scalar(item) for item in value)
            else:
                rendered = format_scalar(value)
            if unit:
                return f"{rendered} {unit}"
            return rendered

        parameter_rows = [
            ("Scan", "Experiment number", "experiment_number", None),
            ("Scan", "Sequence", "sequence_name", None),
            ("Scan", "Method type", "method_type", None),
            ("Scan", "Vendor", "vendor", None),
            ("Scan", "Nucleus", "nucleus", None),
            ("Scan", "B0", "b0_tesla", "T"),
            ("Subject", "Subject", "subject_name", None),
            ("Subject", "Subject type", "subject_type", None),
            ("Subject", "Position", "patient_position", None),
            ("Geometry", "Spatial dimension", "spatial_dimension", None),
            ("Geometry", "Matrix", "matrix", None),
            ("Geometry", "Encoding matrix", "encoding_matrix", None),
            ("Geometry", "FOV", "fov_mm", "mm"),
            ("Geometry", "Resolution", "resolution_mm", "mm"),
            ("Geometry", "Number of slices", "n_slices", None),
            ("Geometry", "Slice thickness", "slice_thickness_mm", "mm"),
            ("Geometry", "Slice gap", "slice_gap_mm", "mm"),
            ("Geometry", "Slice orientation", "slice_orientation", None),
            ("Geometry", "Read orientation", "read_orientation", None),
            ("Geometry", "Read offset", "read_offset_mm", "mm"),
            ("Geometry", "Phase offset", "phase_offset_mm", "mm"),
            ("Geometry", "Slice offset", "slice_offset_mm", "mm"),
            ("Acquisition", "Repetition time", "repetition_time_ms", "ms"),
            ("Acquisition", "Echo time", "echo_time_ms", "ms"),
            ("Acquisition", "Effective TE", "effective_te_ms", "ms"),
            ("Acquisition", "Echo spacing", "echo_spacing_ms", "ms"),
            ("Acquisition", "Flip angle", "flipangle_deg", "deg"),
            ("Acquisition", "Repetitions", "n_repetitions", None),
            ("Acquisition", "Averages", "n_averages", None),
            ("Acquisition", "Acquisition points", "n_acquisition", None),
            ("Acquisition", "Scan duration", "scan_duration_s", "s"),
            ("Acquisition", "Read bandwidth", "read_bandwidth_hz", "Hz"),
            ("Acquisition", "Bandwidth", "bandwidth_hz", "Hz"),
            ("Acquisition", "Bandwidth", "bandwidth_ppm", "ppm"),
            ("Acquisition", "Dwell time", "dwell_time_s", "s"),
            ("Acquisition", "RARE factor", "rare_factor", None),
            ("Acquisition", "b-values", "b_values", "s/mm^2"),
            ("Hardware", "Receive coil", "coil_receive", None),
            ("Hardware", "Transmit coil", "coil_transmit", None),
            ("Hardware", "Receiver gain", "receiver_gain", None),
            ("RF pulse", "Pulse shape", "rfpulse_shape", None),
            ("RF pulse", "Pulse type", "rfpulse_type", None),
            ("RF pulse", "Pulse duration", "rfpulse_duration_ms", "ms"),
            ("RF pulse", "Pulse bandwidth", "rfpulse_bandwidth_hz", "Hz"),
            ("RF pulse", "Bandwidth factor", "rfpulse_bwfac_hzms", "Hz ms"),
            ("RF pulse", "Reference power", "rfpulse_refpow_w", "W"),
            ("RF pulse", "Peak power", "rfpulse_peakpow_w", "W"),
            ("RF pulse", "Calculated pulse", "rfpulse_is_calculated", None),
            ("Refocusing pulse", "Pulse shape", "refpulse_shape", None),
            ("Refocusing pulse", "Pulse type", "refpulse_type", None),
            ("Refocusing pulse", "Pulse duration", "refpulse_duration_ms", "ms"),
            ("Refocusing pulse", "Pulse bandwidth", "refpulse_bandwidth_hz", "Hz"),
            ("Refocusing pulse", "Flip angle", "refpulse_flipangle_deg", "deg"),
        ]

        rows: list[dict[str, str]] = []
        for section, label, attribute, unit in parameter_rows:
            value = get_value(attribute)
            if is_missing(value) and not include_missing:
                continue
            rows.append(
                {
                    "Section": section,
                    "Parameter": label,
                    "Value": format_value(value, unit),
                    "Attribute": attribute,
                }
            )

        cached_data = self.__dict__.get("_data")
        cached_scanner_data = self.__dict__.get("_scanner_data")
        cached_default_data = (
            cached_data if cached_data is not None else cached_scanner_data
        )
        for label, data_array in (
            ("Default data", cached_default_data),
            ("Reco data", cached_data),
            ("Scanner data", cached_scanner_data),
        ):
            if data_array is None:
                if include_missing:
                    rows.append(
                        {
                            "Section": "Data",
                            "Parameter": label,
                            "Value": missing,
                            "Attribute": "",
                        }
                    )
                continue
            dims = " x ".join(str(dim) for dim in getattr(data_array, "dims", ()))
            shape = format_value(getattr(data_array, "shape", None))
            value = shape if not dims else f"{shape} ({dims})"
            rows.append(
                {
                    "Section": "Data",
                    "Parameter": label,
                    "Value": value,
                    "Attribute": "",
                }
            )

        summary = pd.DataFrame(
            rows, columns=["Section", "Parameter", "Value", "Attribute"]
        )

        styles = [
            {
                "selector": "caption",
                "props": [("caption-side", "top"), ("font-size", "115%")],
            },
            {"selector": "th", "props": [("font-size", "109%")]},
            {"selector": "td", "props": [("font-size", "107%")]},
        ]
        styler = summary.style.set_table_styles(styles).hide(axis="index")

        section_styles = {
            "Scan": "background-color: #eef4ff;",
            "Subject": "background-color: #f6eef8;",
            "Geometry": "background-color: #eef8f2;",
            "Acquisition": "background-color: #fff6e6;",
            "Hardware": "background-color: #f1f3f5;",
            "RF pulse": "background-color: #fff0f0;",
            "Refocusing pulse": "background-color: #f0f3ff;",
            "Data": "background-color: #edf7f8;",
        }

        def color_section(row):
            section = row.get("Section")
            base_style = section_styles.get(section, "")
            styles = [base_style] * len(row)
            if base_style and "Section" in row.index:
                section_idx = row.index.get_loc("Section")
                styles[section_idx] = (
                    f"{base_style} font-weight: 600; border-left: 4px solid #5f6f89;"
                )
            return styles

        styler = styler.apply(color_section, axis=1)

        caption_parts = [self.display_name]
        timestamp = get_value("timestamp")
        if timestamp is not None:
            caption_parts.append(f"Timestamp: {timestamp}")
        styler = styler.set_caption(" | ".join(str(part) for part in caption_parts))

        def color_missing(cell_value):
            if isinstance(cell_value, str) and cell_value == missing:
                return "color: red"
            return ""

        try:
            styler = styler.map(color_missing)
        except AttributeError:
            styler = styler.applymap(color_missing)

        if display_output:
            try:
                from IPython.display import display

                display(styler)
            except ImportError:
                pass

        return styler
