"""
Mixins for standardizing variable access patterns across the HyperMRI codebase.

This module provides base classes and mixins to ensure consistent handling of:
- Data arrays (standardized naming)
- Path objects (always pathlib.Path)
- Canonical parameter snapshots and shared derived properties
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Set, Union

import xarray as xr


class StandardizedDataArrayMixin:
    """Hold the scanner and the raw reconstruction of a scan with lazy loading.

    Construction loads nothing.  A subclass provides the loaders
    ``_load_scanner_data`` and ``_load_data``; each runs on the first
    access of its property unless an array has been assigned before.

    Attributes
    ----------
    scanner_data : xarray.DataArray or None
        Scanner reconstruction, for Bruker the ``2dseq`` data.
    data : xarray.DataArray or None
        Reconstruction from the raw ``fid`` or ``ser`` data.
    primary_data : xarray.DataArray or None
        ``data``, or ``scanner_data`` if there is no raw reconstruction;
        see :attr:`primary_data` for when the first access starts a
        reconstruction.
    """

    def __init__(self, *args, **kwargs):
        """Initialize empty reconstruction storage without loading samples.

        Parameters
        ----------
        *args
            Positional arguments for the next cooperative base constructor.
        **kwargs
            Named arguments for the next cooperative base constructor.

        Notes
        -----
        Each object owns its lazy-load flags and cached raw failure. No array,
        reconstruction or interactive resource is created by this mixin.
        """
        super().__init__(*args, **kwargs)
        # Initialize private attributes for lazy loading
        self._scanner_data: Optional[xr.DataArray] = (
            None  # Scanner reconstructed (e.g., seq2d)
        )
        # Raw-reconstructed data
        self._data: Optional[xr.DataArray] = None

        # Lazy loading flags
        self._scanner_data_loaded = False
        self._data_loaded = False
        self._data_error: Exception | None = None

        # Private attributes for compatibility with older scanner-data naming
        self._legacy_seq2d: Optional[xr.DataArray] = None

    @property
    def scanner_data(self) -> Optional[xr.DataArray]:
        """Lazy loading property for scanner reconstructed data (e.g., seq2d)."""
        if not self._scanner_data_loaded and self._scanner_data is None:
            self._scanner_data = self._load_scanner_data()
            self._scanner_data_loaded = True
        return self._scanner_data

    @scanner_data.setter
    def scanner_data(self, value: Optional[xr.DataArray]):
        """Set the scanner reconstruction."""
        self._scanner_data = value
        self._scanner_data_loaded = True

    @property
    def data(self) -> Optional[xr.DataArray]:
        """Return the raw reconstruction, loading it once on first access.

        Returns
        -------
        xarray.DataArray or None
            Stored result itself, with its labelled dimensions, units, dtype
            and provenance, or None when raw reconstruction is unsupported.

        Raises
        ------
        Exception
            A lazy reconstruction failure. Failed attempts are cached and
            raised again on subsequent access; no scanner fallback occurs.
            Assigning a result or explicitly reconstructing successfully
            clears the failure.
        """
        if self._data_error is not None:
            raise self._data_error
        if not self._data_loaded and self._data is None:
            try:
                self._data = self._load_data()
            except Exception as exc:
                self._data_error = exc
                raise
            finally:
                self._data_loaded = True
        return self._data

    @data.setter
    def data(self, value: Optional[xr.DataArray]):
        """Store a raw result explicitly and clear any cached loading failure.

        Parameters
        ----------
        value : xarray.DataArray or None
            Result to store without copying or changing dtype, coordinates,
            units or provenance. None marks raw reconstruction as unavailable.

        Returns
        -------
        None
            Later property access returns this value without invoking a loader.
        """
        self._data = value
        self._data_error = None
        self._data_loaded = True

    def _load_scanner_data(self) -> Optional[xr.DataArray]:
        """Override in subclasses to implement lazy loading of scanner data."""
        return None

    def _load_data(self) -> Optional[xr.DataArray]:
        """Override in subclasses to implement lazy loading of reconstructed data."""
        return None

    @property
    def primary_data(self) -> Optional[xr.DataArray]:
        """Return the raw reconstruction, or the scanner reconstruction without one.

        The choice has two steps, in this order:

        1. :attr:`data`, the reconstruction from the raw ``fid`` or ``ser``
           data.  If it has not been stored yet, the first access starts it
           through the lazy loader of the sequence, also after
           ``load_scan(mode="lazy")`` or ``mode="minimal"``, whose loading
           policies do not include it.  It can read the raw data file.
        2. :attr:`scanner_data`, the scanner reconstruction (Bruker
           ``2dseq``), if step 1 gives ``None``.  The lazy loader of the
           sequence returns ``None`` when the sequence has no raw
           reconstruction for the scan. Reconstruction failures propagate and
           remain failures on repeated access. If :attr:`scanner_data` has not
           been stored yet, this access starts it.

        Each lazy loader runs at most once per object, also when it gives
        ``None``, so later accesses return the same array without new work.
        Use :attr:`data` or :attr:`scanner_data` to select one reconstruction
        explicitly.

        Returns
        -------
        xarray.DataArray or None
            The stored array itself, not a copy, with the dimensions,
            coordinates and attributes of that reconstruction, or ``None``
            if neither reconstruction is available.

        Raises
        ------
        Exception
            A failure of the lazy reconstruction. Access ``scanner_data``
            explicitly to choose the scanner reconstruction after a raw error.
        """
        data = self.data
        if data is not None:
            return data
        return self.scanner_data

    def set_primary_data(self, data: xr.DataArray, data_type: str = "data"):
        """
        Set the primary data array with proper type designation.

        Parameters
        ----------
        data : xr.DataArray
            The data array to store
        data_type : str
            Type of data: "data" (raw-reconstructed), "scanner_data" (seq2d), or "legacy"
        """
        # Ensure unit are set to "a.u." if not already present
        if "unit" not in data.attrs:
            data.attrs["unit"] = "a.u."

        if (
            data_type == "data" or data_type == "reconstructed"
        ):  # Support old name for compatibility
            self.data = data
        elif data_type == "scanner_data" or data_type == "seq2d":
            self.scanner_data = data
        elif data_type == "kspace":
            raise ValueError(
                "kspace_array is no longer supported for permanent storage. Use during reconstruction only."
            )
        elif data_type == "legacy":
            self._legacy_seq2d = data
        else:
            raise ValueError(
                f"Unknown data_type: {data_type}. Use 'data', 'scanner_data', or 'legacy'"
            )


class StandardizedPathMixin:
    """
    Mixin providing standardized path handling using pathlib.Path objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        """Returns the path as a pathlib.Path object."""
        return self._path

    @path.setter
    def path(self, value: Union[str, Path, None]):
        """Sets the path, converting to pathlib.Path if needed."""
        if value is None:
            self._path = None
        elif isinstance(value, (str, Path)):
            self._path = Path(value).resolve()
        else:
            raise TypeError(f"Path must be str, Path, or None, got {type(value)}")


class UniformDataPropertiesMixin:
    """
    Coordinate and frequency axes shared by vendor handlers and scans.

    The six coordinate properties evaluate the metadata-only functions of
    :mod:`hypermri.utils.coordinates.grids` on first access and cache a
    successful result on the instance.  They return None only when the scan
    declares no geometry of the requested kind, which the grid functions
    signal with :class:`~hypermri.utils.coordinates.grids.GeometryUnavailableError`
    (absent ``fov_mm`` or ``matrix``, several slice packages, or no
    ``VisuCore`` geometry).  Any other error, such as incomplete or invalid
    geometry, propagates.  An unavailable result is not cached, so later
    metadata changes are seen on the next access.

    ``freq_axis_ppm`` and ``freq_axis_hz`` return
    :func:`~hypermri.reconstruction.axes.get_freq_axis` unchanged
    on every access, including its None or empty result for absent bandwidth
    metadata; its errors propagate. ``ppm`` and ``hz`` are short names for
    them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize caches for expensive computations
        self._rps_axes_mm = None
        self._rps_extent_mm = None
        self._rps_grid_mm = None
        self._xyz_axes_mm = None
        self._xyz_extent_mm = None
        self._xyz_grid_mm = None

    def _cached_geometry(self, cache_name: str, function_name: str):
        """Return a cached grid result, or None for unavailable geometry.

        Parameters
        ----------
        cache_name : str
            Instance attribute that caches the result.
        function_name : str
            Name of the function in :mod:`hypermri.utils.coordinates.grids`
            that is called with ``scan=self``.

        Returns
        -------
        object or None
            The cached or newly computed result, or None if the function
            raises ``GeometryUnavailableError``.
        """
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached
        import hypermri.utils.coordinates.grids as grids

        try:
            result = getattr(grids, function_name)(scan=self)
        except grids.GeometryUnavailableError:
            return None
        setattr(self, cache_name, result)
        return result

    @property
    def rps_extent_mm(self):
        """
        Limits of the voxel grid along read, phase and slice in mm.

        Returns
        -------
        list of float or None
            ``[read_min, read_max, phase_min, phase_max, slice_min,
            slice_max]`` from :func:`hypermri.utils.coordinates.grids.get_rps_extent`,
            or None if the scan declares no read, phase and slice geometry.

        Raises
        ------
        ValueError
            If the declared geometry is incomplete or invalid.
        """
        return self._cached_geometry("_rps_extent_mm", "get_rps_extent")

    @property
    def rps_axes_mm(self):
        """
        Voxel-centre positions along read, phase and slice in mm.

        Returns
        -------
        tuple of numpy.ndarray or None
            ``(read_axis, phase_axis, slice_axis)`` from
            :func:`hypermri.utils.coordinates.grids.get_rps_axes`, or None if
            the scan declares no read, phase and slice geometry.

        Raises
        ------
        ValueError
            If the declared geometry is incomplete or invalid.
        """
        return self._cached_geometry("_rps_axes_mm", "get_rps_axes")

    @property
    def rps_grid_mm(self):
        """
        Read, phase and slice positions of every voxel in mm.

        Returns
        -------
        tuple of numpy.ndarray or None
            ``(read_grid, phase_grid, slice_grid)``, each with shape
            ``(read, phase, slice)``, from
            :func:`hypermri.utils.coordinates.grids.get_rps_grid`, or None if
            the scan declares no read, phase and slice geometry.

        Raises
        ------
        ValueError
            If the declared geometry is incomplete or invalid.
        """
        return self._cached_geometry("_rps_grid_mm", "get_rps_grid")

    @property
    def xyz_extent_mm(self):
        """
        Limits of the scanner positions of the voxels along x, y and z in mm.

        Returns
        -------
        list of float or None
            Limits from :func:`hypermri.utils.coordinates.grids.get_xyz_extent`
            in the DICOM patient coordinate system (LPS), or None if the scan
            has no ``VisuCore`` geometry.

        Raises
        ------
        ValueError
            If the ``VisuCore`` geometry is incomplete or cannot be evaluated.
        """
        return self._cached_geometry("_xyz_extent_mm", "get_xyz_extent")

    @property
    def xyz_axes_mm(self):
        """
        Sorted distinct scanner coordinates along x, y and z in mm.

        Coordinates that differ by no more than the geometry position tolerance
        of 1e-3 mm along an axis-aligned image axis are one position (see
        :func:`hypermri.utils.coordinates.grids.get_xyz_axes`).

        Returns
        -------
        tuple of numpy.ndarray or None
            ``(x_axis, y_axis, z_axis)`` from
            :func:`hypermri.utils.coordinates.grids.get_xyz_axes` in the DICOM
            patient coordinate system (LPS), cached after the first successful
            call, or None if the scan has no ``VisuCore`` geometry.

        Raises
        ------
        ValueError
            If the ``VisuCore`` geometry is incomplete or cannot be evaluated,
            for example when positions chain beyond the tolerance.
        """
        return self._cached_geometry("_xyz_axes_mm", "get_xyz_axes")

    @property
    def xyz_grid_mm(self):
        """
        Scanner coordinates of the voxel centres in mm.

        Returns
        -------
        tuple of numpy.ndarray or None
            ``(x_grid, y_grid, z_grid)`` in the DICOM patient coordinate system
            (LPS), each with shape ``(read, phase, slice)``, from
            :func:`hypermri.utils.coordinates.grids.get_xyz_grid`, cached after
            the first successful call, or None if the scan has no
            ``VisuCore`` geometry.

        Raises
        ------
        ValueError
            If the ``VisuCore`` geometry is incomplete or cannot be evaluated.
        """
        return self._cached_geometry("_xyz_grid_mm", "get_xyz_grid")

    @property
    def freq_axis_ppm(self):
        """
        Chemical-shift axis of the spectral dimension in ppm.

        Returns
        -------
        numpy.ndarray, list or None
            The result of
            :func:`hypermri.reconstruction.axes.get_freq_axis`
            with ``unit="ppm"``: one value per retained spectral point
            (``n_acquisition`` minus ``cut_off``) in FFT-shifted order, offset
            by ``center_ppm``; an empty list if ``bandwidth_hz`` is None, or
            None if the object has no ``bandwidth_hz`` attribute. Recomputed
            on every access; nothing is cached or changed and no signal
            samples are read.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        from hypermri.reconstruction.axes import get_freq_axis

        return get_freq_axis(scan=self, unit="ppm")

    @property
    def freq_axis_hz(self):
        """
        Frequency axis of the spectral dimension in Hz.

        Returns
        -------
        numpy.ndarray, list or None
            The result of
            :func:`hypermri.reconstruction.axes.get_freq_axis`
            with ``unit="Hz"``: one value per retained spectral point in
            FFT-shifted order, relative to the centre frequency, with the
            same absent cases as :attr:`freq_axis_ppm`. Recomputed on every
            access.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        from hypermri.reconstruction.axes import get_freq_axis

        return get_freq_axis(scan=self, unit="Hz")

    @property
    def ppm(self):
        """
        Short name for :attr:`freq_axis_ppm`.

        Spectroscopy sequences replace it with an axis that matches the
        length of their loaded data.

        Returns
        -------
        numpy.ndarray, list or None
            The value of :attr:`freq_axis_ppm`.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        return self.freq_axis_ppm

    @property
    def hz(self):
        """
        Short name for :attr:`freq_axis_hz`.

        Spectroscopy sequences replace it with an axis that matches the
        length of their loaded data.

        Returns
        -------
        numpy.ndarray, list or None
            The value of :attr:`freq_axis_hz`.

        Raises
        ------
        ValueError
            If the acquisition length or the sampling metadata are invalid.
        """
        return self.freq_axis_hz


class StandardizedMRIMixin(
    StandardizedDataArrayMixin,
    StandardizedPathMixin,
    UniformDataPropertiesMixin,
):
    """Combine the standardized data, path and parameter patterns of MRI classes.

    The mixin owns the canonical parameter schema shared by all vendors and
    stores a validated parameter mapping in :attr:`params`.  Vendor handlers
    additionally keep each canonical parameter as an attribute, because their
    parsers assign the attributes and :meth:`build_parameter_snapshot` reads
    them back.  An :class:`~hypermri.loading.experiment.MRIExperiment` keeps no
    such attribute copies and reads a parameter from :attr:`params` instead.

    Attributes
    ----------
    PARAMETER_SCHEMA : dict of str to iterable of str
        Canonical parameter names grouped by category.
    OPTIONAL_PARAMETERS : set of str
        Canonical parameters that :meth:`set_parameters` does not require.
    params : dict
        Validated parameter mapping stored by :meth:`set_parameters`; empty
        until then.
    """

    # Whether each canonical parameter is also kept as an attribute.  Vendor
    # handlers need the attributes; experiments read ``params`` only.
    _PARAMETERS_AS_ATTRIBUTES: ClassVar[bool] = True

    # Canonical parameter definitions shared across vendors. Keys grouped by logical
    # category to keep the schema human-readable and maintainable.
    PARAMETER_SCHEMA: ClassVar[Dict[str, Iterable[str]]] = {
        "core": (
            "experiment_number",
            "timestamp",
            "vendor",
            "nucleus",
            "method_type",
            "is_spectroscopic",
            "cut_off",
            "git_commit_hash",
        ),
        "subject": (
            "subject_name",
            "subject_id",
            "subject_position",
            "subject_type",
            "patient_position",
        ),
        "geometry": (
            "spatial_dimension",
            "n_slice_packs",
            "n_slices",
            "slice_thickness_mm",
            "slice_gap_mm",
            "slice_orientation",
            "slice_order",
            "slice_offset_mm",
            "slice_offset_mm_array",
            "read_orientation",
            "read_offset_mm",
            "phase_offset_mm",
            "phase_0_offset_mm",
            "phase_1_offset_mm",
            "position_matrix",
            "fov_mm",
            "matrix",
            "encoding_matrix",
            "resolution_mm",
            "measurement_time",
            "center_ppm",
        ),
        "acquisition": (
            "n_receivers",
            "n_repetitions",
            "n_averages",
            "n_echoes",
            "n_acquisition",
            "n_voxels",
            "n_inversion_time",
            "inversion_time_ms",
            "rare_factor",
            "scan_duration_s",
            "read_bandwidth_hz",
            "bandwidth_hz",
            "bandwidth_ppm",
            "echo_time_ms",
            "repetition_time_ms",
            "frame_time_s",
            "dwell_time_s",
            "sampling_time_axis_s",
            "fat_water_separation",
            "b_values",
            "sequence_name",
            "echo_spacing_ms",
            "effective_te_ms",
        ),
        "field": (
            "reference_frequency_mhz",
            "b0_tesla",
        ),
        "hardware": (
            "coil_receive",
            "coil_transmit",
            "receiver_gain",
        ),
        "rf": (
            "rf_pulse_params",
            "rfpulse_duration_ms",
            "rfpulse_bandwidth_hz",
            "rfpulse_is_calculated",
            "rfpulse_sharpness",
            "rfpulse_bwfac_hzms",
            "rfpulse_sint",
            "rfpulse_pint",
            "rfpulse_type",
            "rfpulse_rpfac_percent",
            "rfpulse_peakpow_w",
            "rfpulse_shape",
            "rfpulse_refpow_w",
            "flipangle_deg",
            "ref_pulse_params",
            "refpulse_duration_ms",
            "refpulse_bandwidth_hz",
            "refpulse_flipangle_deg",
            "refpulse_is_calculated",
            "refpulse_sharpness",
            "refpulse_bwfac_hzms",
            "refpulse_sint",
            "refpulse_pint",
            "refpulse_type",
            "refpulse_rpfac_percent",
            "refpulse_peakpow_w",
            "refpulse_shape",
            "refpulse_refpow_w",
        ),
        "reconstruction": (
            "recon_names",
            "acquisition_matrix",
        ),
    }

    OPTIONAL_PARAMETERS: ClassVar[Set[str]] = {
        "resolution_mm",
        "sampling_time_axis_s",
        "fat_water_separation",
        "b_values",
        "rf_pulse_params",
        "ref_pulse_params",
        "rfpulse_duration_ms",
        "rfpulse_bandwidth_hz",
        "rfpulse_is_calculated",
        "rfpulse_sharpness",
        "rfpulse_bwfac_hzms",
        "rfpulse_sint",
        "rfpulse_pint",
        "rfpulse_type",
        "rfpulse_rpfac_percent",
        "rfpulse_peakpow_w",
        "rfpulse_shape",
        "rfpulse_refpow_w",
        "flipangle_deg",
        "refpulse_duration_ms",
        "refpulse_bandwidth_hz",
        "refpulse_flipangle_deg",
        "refpulse_is_calculated",
        "refpulse_sharpness",
        "refpulse_bwfac_hzms",
        "refpulse_sint",
        "refpulse_pint",
        "refpulse_type",
        "refpulse_rpfac_percent",
        "refpulse_peakpow_w",
        "refpulse_shape",
        "refpulse_refpow_w",
        "reference_frequency_mhz",
        "b0_tesla",
        "coil_receive",
        "coil_transmit",
        "receiver_gain",
        "recon_names",
        "acquisition_matrix",
        "read_bandwidth_hz",
        "echo_spacing_ms",
        "effective_te_ms",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize common attributes with standard types
        self.params: dict = {}

        # Vendor handlers rely on a ``None`` placeholder for every canonical field.
        if self._PARAMETERS_AS_ATTRIBUTES:
            for key in self.parameter_keys():
                setattr(self, key, None)

    @classmethod
    def parameter_keys(cls) -> Set[str]:
        keys: Set[str] = set()
        for group_keys in cls.PARAMETER_SCHEMA.values():
            keys.update(group_keys)
        return keys

    @classmethod
    def required_parameter_keys(cls) -> Set[str]:
        return cls.parameter_keys() - cls.OPTIONAL_PARAMETERS

    def set_parameters(
        self, parameters: Mapping[str, Any], *, allow_extra: bool = False
    ) -> None:
        """Store standardized parameters after validating against the schema.

        Parameters
        ----------
        parameters : Mapping[str, Any]
            Parameter values keyed by canonical name.  Every name in
            :meth:`required_parameter_keys` must be present; values keep the
            units named by their keys (for example ``_ms`` or ``_mm``).
        allow_extra : bool
            If False (default), raise ValueError on unknown keys.
            If True, unknown keys are accepted with a warning.

        Raises
        ------
        ValueError
            If a required parameter is missing, or if an unknown key is given
            while ``allow_extra`` is False.

        Warns
        -----
        UserWarning
            If unknown keys are accepted because ``allow_extra`` is True.

        Notes
        -----
        A shallow copy of ``parameters`` replaces :attr:`params`.  Vendor
        handlers also assign each value to the attribute of the same name; an
        experiment does not, so its parameters have one store.
        """
        provided_keys = set(parameters.keys())
        missing = self.required_parameter_keys() - provided_keys
        if missing:
            raise ValueError(
                f"Missing required parameters: {', '.join(sorted(missing))}"
            )

        allowed_keys = self.parameter_keys()
        extra = provided_keys - allowed_keys
        if extra and not allow_extra:
            raise ValueError(
                f"Unexpected parameters not in canonical schema: "
                f"{', '.join(sorted(extra))}. "
                f"Pass allow_extra=True to accept non-standard keys."
            )
        elif extra:
            import warnings

            warnings.warn(
                (
                    "Received additional parameters not in the canonical schema: "
                    + ", ".join(sorted(extra))
                ),
                UserWarning,
                stacklevel=2,
            )

        self.params = dict(parameters)

        if self._PARAMETERS_AS_ATTRIBUTES:
            for key, value in self.params.items():
                setattr(self, key, value)

    def get_parameters(self) -> Dict[str, Any]:
        """Return the standardized parameter mapping for metadata consumers."""
        return dict(self.params)

    def build_parameter_snapshot(self) -> Dict[str, Any]:
        """Collect the canonical parameter snapshot from handler attributes."""
        return {key: getattr(self, key, None) for key in self.parameter_keys()}

    def __getattr__(self, name: str):
        """Fallback attribute access to standardized parameters."""
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def get_primary_data_info(self) -> dict:
        """
        Returns information about available data arrays.

        Returns
        -------
        dict
            Dictionary with data array information
        """
        info = {
            "has_scanner_data": self.scanner_data is not None,
            "has_kspace_array": False,  # kspace_array no longer stored permanently
            "has_legacy_data": self._legacy_seq2d is not None,
            "primary_data_shape": None,
            "primary_data_dims": None,
        }

        primary = self.primary_data
        if primary is not None:
            info["primary_data_shape"] = primary.shape
            info["primary_data_dims"] = primary.dims

        return info
