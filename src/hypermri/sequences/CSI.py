"""Chemical shift imaging workflows with shared analysis and visualization.

Interactive decay review delegates to the common viewer; reconstruction and
filtering retain their existing owners and labelled scientific data contracts.
Logging inherits the package threshold unless explicitly configured by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.reconstruction.axes import get_freq_axis

from ..analysis.spatial_filtering import SpatialFilterOptions
from ..utils.coordinates.grids import define_xyz_grid
from ..utils.coordinates.transformations import define_imagematrix_parameters
from ..utils.image.transforms import define_imageFOV_parameters
from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import debug_log, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__)
DEBUG_FILE = "/tmp/csi_debug.log"
DEBUG_LOG = False

# Indices of the read and phase lines in the legacy z/x/y grid.
_LEGACY_PLOTTING_AXES = {
    "axial": {"L_R": (1, 2), "A_P": (2, 1)},
    "sagittal": {"H_F": (0, 2), "A_P": (2, 0)},
    "coronal": {"H_F": (0, 1), "L_R": (1, 0)},
}


def debug_log_csi(message, file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    debug_log(message=message, file_name=file_name, write_to_file=write_to_file)


def _signal_end_provenance(end_index, *, filter_axis, remove_lowest_signal):
    """Return the truncation settings of ``create_filtered_data`` as attrs.

    The values are plain strings and integers so that NetCDF can store them;
    ``"none"`` marks a reconstruction without truncation.
    """
    return {
        "signal_end": "none" if end_index is None else int(end_index),
        "signal_end_axis": "fid_samples"
        if filter_axis in (0, -2)
        else "acquired_points",
        "signal_end_selection": "highest_mean_signal"
        if remove_lowest_signal
        else "first",
    }


class CSI(BaseSpectroscopy):
    """Chemical shift imaging sequence with spectroscopy utilities.

    The class extends :class:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy`
    with interactive signal filtering and voxel-wise fitting visualisations used in
    the spectroscopy tutorials. Channel combination is provided by
    :func:`hypermri.utils.mri.channels.combine_channels`.
    """

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        super().__init__(handler, **kwargs)

        # Check if FID data is available for CSI reconstruction
        # In minimal/lazy loading modes, we allow CSI to load even without FID data
        if not hasattr(self.handler, "fid") or self.handler.fid is None:
            logger.warning(
                "CSI: No FID data available - this may limit functionality but allowing initialization to continue"
            )

    def plot_signal_decay(
        self,
        data: np.ndarray | xr.DataArray | None = None,
        *,
        spatial_spacing_mm: tuple[float, float] | None = None,
        bands_hz: Sequence[tuple[float, float]] | None = None,
    ):
        """Show the decay of the raw data and choose the FID filter in the viewer.

        Same as
        :func:`hypermri.visualization.signal_decay.plot_signal_decay` with
        this scan: the view shows the data reconstructed with the chosen
        filter and opens the "signal_decay" tool, whose "Confirm" stores
        ``self.signal_end`` and ``self.filter_params``.

        Parameters
        ----------
        data : numpy.ndarray, xarray.DataArray or None, optional
            Raw complex FIDs as a vector or (FID sample, acquired point)
            matrix, or labelled CSI samples with ``acquisition_idx``,
            ``read_idx`` and ``phase_idx``. None uses the loaded raw data.
            Copied, never changed.
        spatial_spacing_mm : tuple of float or None, optional
            Read and phase voxel spacing in mm for the spatial k-space filter
            of labelled input without calibrated axes; None uses the field
            of view of the scan.
        bands_hz : sequence of (float, float) or None, optional
            Frequency bands in Hz, for example ``[(-330, -150), (200, 420)]``;
            the tool then starts with the band power in these bands.

        Returns
        -------
        ScanView
            The view with the "signal_decay" tool open.

        Raises
        ------
        ValueError
            If raw data are missing or incomplete, labelled input lacks a
            dimension, or ``bands_hz`` is not a list of increasing pairs.

        Examples
        --------
        >>> view = csi.plot_signal_decay(bands_hz=[(-330, -150), (200, 420)])  # doctest: +SKIP
        >>> csi.filter_signal(**csi.filter_params)  # after "Confirm"  # doctest: +SKIP
        """
        from ..visualization.signal_decay import plot_signal_decay

        return plot_signal_decay(
            self,
            data,
            spatial_spacing_mm=spatial_spacing_mm,
            bands_hz=bands_hz,
        )

    def create_filtered_data(
        self,
        end_index: int = None,
        *,
        filter_type: str = "None",
        decay_constant: float = 0.01,
        alpha: float = 0.54,
        center_position: int = None,
        remove_lowest_signal: bool = False,
        apply_flipangle_correction: bool = False,
        flipangle_deg: float = None,
        filter_axis: int = 1,
        source_data: str = "data",
        update_attributes: bool = True,
        filter_params: dict = None,
        spatial_filter: SpatialFilterOptions | None = None,
        spatial_spacing_mm: tuple[float, float] | None = None,
    ) -> xr.DataArray:
        """
        Create a filtered version of spectroscopic data using signal filtering and reconstruction.

        This method applies filtering to the underlying FID data (like plot_signal_decay workflow)
        and then reconstructs to create a filtered spectroscopic data array with the same
        structure as data but with applied filtering effects.

        Parameters
        ----------
        end_index : int, optional
            Count of acquired points to keep when filter_type="None" (or FID
            samples with filter_axis=0). A count of 3 keeps indices 0, 1 and 2.
            None keeps every point; ``self.signal_end`` is not read, pass it
            explicitly or through ``filter_params``.
        filter_type : str, optional
            Type of filter to apply: "None", "Exponential", "Hamming" or "Hann",
            by default "None". Other names raise ``ValueError``.
        decay_constant : float, optional
            Exponential attenuation per selected sample index, not Hz or seconds;
            default 0.01. Existing arithmetic does not clip this value.
        alpha : float, optional
            Dimensionless Hamming coefficient, normally in [0, 1]; default 0.54.
        center_position : int, optional
            Centre index along the selected window axis for Hamming/Hann. None
            uses its midpoint; no physical coordinate is inferred.
        remove_lowest_signal : bool, optional
            If True, remove the lowest-mean-signal points instead of truncating the
            tail after ``end_index``.
        apply_flipangle_correction : bool, optional
            Whether to apply flipangle correction, by default False
        flipangle_deg : float, optional
            Flip angle in degrees; None uses the sequence's value or the existing
            90-degree fallback. Existing correction singularities are retained.
        filter_axis : int, optional
            Axis along which to apply the filter. Default is 1 (k-space points
            in acquisition order). Set to 0 to explicitly filter the FID
            acquisition dimension.
        source_data : str, optional
            Source data to use as template. Options: "data" or "scanner_data",
            by default "data"
        update_attributes : bool, optional
            Whether to update time and frequency attributes, by default True
        filter_params : dict, optional
            Dictionary of filter parameters (alternative to individual parameters).
            Confirmed plot_signal_decay options override individual parameters,
            including spatial_filter and spatial_spacing_mm. Active legacy
            spatial_filter_type/spatial_width settings require migration.
        spatial_filter : SpatialFilterOptions or None, optional
            Physical Gaussian/Fermi window on Read and Phase, applied once to
            the unchanged raw samples before the acquired-order window. Widths
            use cycles/mm. None retains the existing unweighted result.
        spatial_spacing_mm : tuple of float or None, optional
            Explicit positive (Read, Phase) image sampling distances in mm.
            Otherwise the verified raw CSI adapter uses fov_mm/matrix. This
            setting requires spatial_filter; missing calibration raises.

        Returns
        -------
        xr.DataArray
            Canonically reconstructed complex CSI with labelled dimensions,
            coordinates and units; available data sets dimension order.
            Spatial-filter provenance records effective physical widths and
            calibration. With ``update_attributes`` the truncation is recorded
            in attrs that fit results inherit: ``signal_end`` (int count of
            kept points, or ``"none"`` without truncation), ``signal_end_axis``
            (``"acquired_points"`` or ``"fid_samples"``) and
            ``signal_end_selection`` (``"first"`` or ``"highest_mean_signal"``
            for ``remove_lowest_signal``).

        Raises
        ------
        ValueError
            If the source name or ``filter_type`` is invalid or a requested
            physical filter has invalid options, calibration, raw method or
            encoding layout.
        TypeError
            If spatial_filter has the wrong type.
        AttributeError
            If required methods or data are not available
        Exception
            Every error of ``filter_signal``, including reconstruction
            failures, propagates; no truncated substitute is returned.

        Notes
        -----
        Eager and deterministic. Copies loaded rawdatajob0 (then fid), filters
        with filter_signal and reconstructs with the canonical CSI strategy;
        source_data selects only the template, not the samples to filter. The
        temporal/acquired-order window, its defaults and precision are those of
        filter_signal. Source arrays and handler caches are not mutated.
        update_attributes changes only the returned array's descriptive metadata.
        To filter a specific labelled dataset, call filter_signal(data=...)
        or analysis.spatial_filtering.apply_spatial_filter directly.

        Examples
        --------
        >>> options = SpatialFilterOptions(sigma_per_mm=0.1)
        >>> filtered = csi.create_filtered_data(  # doctest: +SKIP
        ...     end_index=20, spatial_filter=options
        ... )
        >>> filtered = csi.create_filtered_data(  # doctest: +SKIP
        ...     filter_params=csi.filter_params
        ... )
        """
        # Use filter_params dict if provided (from plot_signal_decay workflow)
        if filter_params is not None:
            if filter_params.get("spatial_filter_type") not in (None, "None"):
                raise ValueError(
                    "Legacy spatial_filter_type/spatial_width require spatial_filter=SpatialFilterOptions with physical widths in 1/mm."
                )
            spatial_filter = filter_params.get("spatial_filter", spatial_filter)
            spatial_spacing_mm = filter_params.get(
                "spatial_spacing_mm", spatial_spacing_mm
            )
            filter_type = filter_params.get("filter_type", filter_type)
            end_index = filter_params.get("signal_end", end_index)
            decay_constant = filter_params.get("decay_constant", decay_constant)
            alpha = filter_params.get("alpha", alpha)
            center_position = filter_params.get("center_position", center_position)
            remove_lowest_signal = filter_params.get(
                "remove_lowest_signal", remove_lowest_signal
            )
            apply_flipangle_correction = filter_params.get(
                "apply_flipangle_correction", apply_flipangle_correction
            )
            flipangle_deg = filter_params.get("flipangle_deg", flipangle_deg)
            filter_axis = filter_params.get("filter_axis", filter_axis)
        # Get source data for structure reference
        if source_data == "data":
            if not hasattr(self, "data") or self.data is None:
                if hasattr(self, "scanner_data") and self.scanner_data is not None:
                    logger.warning("data not available, falling back to scanner_data")
                    source = self.scanner_data
                else:
                    raise AttributeError("Neither data nor scanner_data is available")
            else:
                source = self.data
        elif source_data == "scanner_data":
            if not hasattr(self, "scanner_data") or self.scanner_data is None:
                raise AttributeError("scanner_data is not available")
            source = self.scanner_data
        else:
            raise ValueError(
                f"Invalid source_data: {source_data}. Must be 'data' or 'scanner_data'"
            )

        # Validate source data structure
        if "acquisition_idx" not in source.dims:
            raise AttributeError(
                "Source data must have 'acquisition_idx' dimension for spectroscopic filtering"
            )

        logger.info(
            f"Creating filtered data with filter_type='{filter_type}', end_index={end_index}"
        )

        filtered_data = self.filter_signal(
            filter_type=filter_type,
            signal_end=end_index,
            decay_constant=decay_constant,
            alpha=alpha,
            center_position=center_position,
            remove_lowest_signal=remove_lowest_signal,
            apply_flipangle_correction=apply_flipangle_correction,
            flipangle_deg=flipangle_deg,
            filter_axis=filter_axis,
            spatial_filter=spatial_filter,
            spatial_spacing_mm=spatial_spacing_mm,
            return_reconstructed=True,
        )

        if update_attributes:
            # Update time-related attributes
            # This is necessary because filtering could change the effective time axis
            new_attrs = filtered_data.attrs.copy()

            # Add filtering metadata
            new_attrs["filter_type"] = filter_type
            new_attrs.update(
                _signal_end_provenance(
                    end_index,
                    filter_axis=filter_axis,
                    remove_lowest_signal=remove_lowest_signal,
                )
            )
            new_attrs["original_length"] = source.sizes["acquisition_idx"]

            if filter_type == "Exponential":
                new_attrs["decay_constant"] = decay_constant
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {decay_constant}"
            elif filter_type == "Hamming":
                new_attrs["alpha"] = alpha
                new_attrs["center_position"] = center_position
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {alpha}, {center_position}"
            elif filter_type == "Hann":
                new_attrs["center_position"] = center_position
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {center_position}"
            else:
                if remove_lowest_signal:
                    description_update = (
                        f"Data kept {end_index} highest-mean-signal points "
                        "without additional filtering"
                    )
                else:
                    description_update = f"Data was truncated to first {end_index} points without additional filtering"

            # Apply updated attributes
            filtered_data.attrs = new_attrs

            # Update description
            if "description" in filtered_data.attrs:
                filtered_data.attrs["description"] += f" | {description_update}"
            else:
                filtered_data.attrs["description"] = description_update

        return filtered_data

    def intensity_map(self):
        return np.sum(self.primary_data, axis=0)

    def find_peaks(
        self, range_unit="ppm", searchrange=None, sr_index=0, height=0, **kwargs
    ):
        """Detect peaks through the legacy per-spectrum iterator.

        Use BaseSpectroscopy.single_spec_find_peaks on an explicitly selected spectrum.
        This method checks its prerequisites before loading, mutating data or creating plots.

        Parameters
        ----------
        range_unit : str, optional
            Spectral interval unit: ppm, Hz or sample indices.
        searchrange : sequence of float, optional
            Two interval endpoints in range_unit; None selects the full spectrum.
        sr_index : int, optional
            Index of the legacy search-range and peak dictionaries.
        height : float or array_like, optional
            Minimum peak height forwarded to scipy.signal.find_peaks.
        **kwargs
            Peak detection options forwarded to scipy.signal.find_peaks.

        Returns
        -------
        None
            Stores peaks and ranges in the legacy dictionaries.

        Raises
        ------
        NotImplementedError
            If the iterator or peak/search-range dictionaries are unavailable.
        ValueError
            If no spectral axis matches the effective acquisition length or
            SciPy peak options are invalid.
        AssertionError
            If the selected spectral interval is not increasing.

        Notes
        -----
        Peak indices refer to the effective spectral array after cutoff.
        This eager operation updates the legacy peak and search-range
        dictionaries without changing acquired spectra; peak heights use
        the signal units supplied by the iterator.

        Examples
        --------
        After an extension supplies the legacy iterator and dictionaries::

            scan.find_peaks(range_unit="ppm", searchrange=[170.0, 180.0])
        """
        missing = [
            name
            for name in [
                "peaks",
                "peakproperties",
                "searchranges",
                "searchranges_idx",
                "apply_to_every_spec",
            ]
            if not hasattr(self, name)
        ]
        if missing:
            raise NotImplementedError(
                "CSI.find_peaks requires unavailable legacy inputs: "
                + ", ".join(missing)
                + ". Use BaseSpectroscopy.single_spec_find_peaks on an explicitly selected spectrum."
            )
        if searchrange is None:
            searchrange = []
        self.peaks[sr_index] = dict()
        self.peakproperties[sr_index] = dict()

        spectral_data = self._loaded_spectroscopy_data()
        if spectral_data is not None:
            npoints = int(spectral_data.sizes["acquisition_idx"])
        else:
            legacy_data = getattr(self, "seq2d", None)
            if legacy_data is not None and np.ndim(legacy_data) > 0:
                npoints = int(np.shape(legacy_data)[0])
            else:
                npoints = max(
                    0,
                    int(getattr(self, "n_acquisition", 0) or 0)
                    - int(getattr(self, "cut_off", 0) or 0),
                )

        # Resolve the axis from the actual data when possible.  A raw
        # reconstruction may already have discarded the configured leading FID
        # points, while scanner-reconstructed data can retain a different size.
        if range_unit == "ppm":
            xaxis = self._frequency_axis_for_data("ppm", spectral_data)
        elif range_unit == "Hz":
            xaxis = self._frequency_axis_for_data("Hz", spectral_data)
        else:
            xaxis = np.arange(npoints)

        if xaxis is None or len(xaxis) != npoints:
            raise ValueError(
                "Could not resolve a frequency axis aligned with the CSI data "
                f"({npoints} acquisition points)."
            )

        if len(searchrange):

            def find_closest_index(x):
                return np.argmin(np.abs(x - xaxis))

            a = find_closest_index(searchrange[0])
            b = find_closest_index(searchrange[1])

            assert a < b, f"Illegal range! {searchrange}-->i_min={a} > i_max={b}."

        else:
            # that is, x[0, -1] == x[:]
            a, b = 0, -1

        # store info on searchrange and range unit

        self.searchrangeunit = range_unit
        self.searchranges[sr_index] = searchrange
        self.searchranges_idx[sr_index] = [a, b]

        def custom_find_peaks(spec):
            peaks, info = find_peaks(spec[a:b], height=height, **kwargs)
            peaks += a
            return peaks, info

        for idx, res in self.apply_to_every_spec(custom_find_peaks):
            tmp_peaks, tmp_properties = res

            # store peak location and peak properties
            self.peaks[sr_index][idx] = tmp_peaks
            self.peakproperties[sr_index][idx] = tmp_properties

    def get_plotting_grid(self, csi_interpolation_matrix=None):
        """
        Return two legacy in-plane CSI plotting lines in millimetres.

        The lines are :func:`~hypermri.utils.coordinates.grids.define_xyz_grid`
        of :func:`~hypermri.utils.image.transforms.define_imageFOV_parameters`
        and
        :func:`~hypermri.utils.coordinates.transformations.define_imagematrix_parameters`,
        both in z, x and y order.  For Head_Prone and Foot_Prone scans the
        read and phase offsets are added to the lines of the read and phase
        axes, and the two in-plane lines are returned in read, phase order.
        Head_Supine axial L_R scans ignore offsets; other supine pairs are
        unavailable because either the matrix or this orientation table is
        incomplete. These lines do not represent ``VisuCore`` voxel positions.

        Parameters
        ----------
        csi_interpolation_matrix : sequence of int, optional
            At least four positive integer factors. Elements 1, 2 and 3
            multiply the voxel counts along z, x and y; element 0 is ignored.
            Without it the acquired matrix is used. This changes sampling of
            the lines only; no signal or image array is interpolated.

        Returns
        -------
        list of numpy.ndarray
            Two one-dimensional float64 arrays in read, phase order, in mm.
            Each line is spaced by its assigned field of view divided by its
            voxel count, starting at minus half that field of view before
            adding the supported offsets. Its midpoint is half a voxel below
            the offset. Read and phase lengths are paired with their
            respective voxel counts for all six supported prone orientations.

        Raises
        ------
        ValueError
            If required geometry is absent or unsupported, or the resulting
            counts and field-of-view lengths are not finite and positive.
        TypeError
            If interpolation factors are applied to incomplete Head_Supine
            geometry with None voxel counts.
        IndexError
            If fewer than four interpolation factors are supplied.
        Exception
            If the patient position or the orientation pair is not in the
            table.

        Warns
        -----
        UserWarning
            For Head_Supine geometry, whose legacy matrix table is incomplete.

        Notes
        -----
        This deterministic metadata-only operation does not reconstruct data,
        initialize plotting, cache results or mutate input metadata or factors.
        Slice offsets, oblique directions and scanner positions are ignored.
        For scanner voxel positions use
        :func:`~hypermri.utils.coordinates.grids.get_xyz_grid`.

        Examples
        --------
        For an axial prone scan, double the two in-plane grid counts while
        retaining the slice count and field of view:

        >>> read_mm, phase_mm = scan.get_plotting_grid([1, 1, 2, 2])
        """
        patient_pos = self.patient_position
        read_orient = self.read_orientation
        slice_orient = self.slice_orientation

        mm_z_csi, mm_x_csi, mm_y_csi = define_imageFOV_parameters(scan=self)
        dim_z_csi, dim_x_csi, dim_y_csi = define_imagematrix_parameters(scan=self)
        if csi_interpolation_matrix is not None:
            dim_z_csi, dim_x_csi, dim_y_csi = (
                dim_z_csi * csi_interpolation_matrix[1],
                dim_x_csi * csi_interpolation_matrix[2],
                dim_y_csi * csi_interpolation_matrix[3],
            )
            debug_log_csi(
                "Updated plotting grid according to interpolation matrix: "
                + str(csi_interpolation_matrix)
            )
        csi_grid = define_xyz_grid(
            mat=np.array((dim_z_csi, dim_x_csi, dim_y_csi)),
            fov=np.array((mm_z_csi, mm_x_csi, mm_y_csi)),
        )

        # offsets [mm]
        read_offset = getattr(self, "read_offset_mm", 0.0)
        phase_offset = getattr(self, "phase_1_offset_mm", None)
        if phase_offset is None:
            phase_offset = getattr(self, "phase_offset_mm", 0.0)

        if patient_pos not in ("Head_Prone", "Foot_Prone", "Head_Supine"):
            raise Exception("unknown patient_position: " + patient_pos)
        if patient_pos == "Head_Supine":
            debug_log_csi(message="Head_Supine is not fully implemented yet.")
        if slice_orient not in _LEGACY_PLOTTING_AXES:
            raise Exception("unknown slice orientation: " + slice_orient)
        orientation_axes = _LEGACY_PLOTTING_AXES[slice_orient]
        supine_read = "L_R" if slice_orient == "axial" else "H_F"
        if read_orient not in orientation_axes or (
            patient_pos == "Head_Supine" and read_orient != supine_read
        ):
            raise Exception(
                "unknown read orientation: "
                + read_orient
                + "for slice_orient: "
                + slice_orient
            )
        read_axis, phase_axis = orientation_axes[read_orient]
        lines = [csi_grid[read_axis], csi_grid[phase_axis]]
        if patient_pos != "Head_Supine":
            lines = [lines[0] + read_offset, lines[1] + phase_offset]
        return lines
