"""Data preprocessing module for unified plotting system.

This module consolidates all data preprocessing logic that was previously
scattered across AnatomyPlotter, SpectroscopyPlotter, and UnifiedPlotter.
"""

import warnings
from typing import Optional

import numpy as np
import xarray as xr

from hypermri.analysis.denoising.algorithms import pca_denoise
from hypermri.analysis.spectroscopy.processing import multi_dim_linebroadening
from hypermri.reconstruction.centred_dft import centered_fft
from hypermri.utils.mathematics.calculations import part_func
from hypermri.utils.mri.fourier import (
    fid_is_recoverable,
    spectral_fft,
    spectral_ifft,
)


class DataPreprocessor:
    """Handles all data preprocessing operations for plotting.

    This class provides a unified interface for common preprocessing operations
    including denoising, domain transforms, channel combination, and more.
    """

    def __init__(self, scan=None):
        """Initialize the DataPreprocessor.

        Parameters
        ----------
        scan : object, optional
            The data object (anatomy or spectroscopy) that provides context
            like sampling_time_axis_s, cut_off, etc.
        """
        self.data_obj = scan

    def apply_preprocessing(
        self, data: xr.DataArray, kwargs: dict, widget_prefix: str = ""
    ) -> xr.DataArray:
        """Apply full preprocessing pipeline to data array.

        Parameters
        ----------
        data : xr.DataArray
            Input data array to preprocess
        kwargs : dict
            Dictionary of widget parameters
        widget_prefix : str
            Prefix for widget names (e.g., 'ds0', 'ds1')

        Returns
        -------
        xr.DataArray
            Preprocessed data array
        """
        # Make a copy to avoid modifying original (shallow copy to preserve lazy loading)
        processed_data = data.copy(deep=False)

        # 1. Standardize dimension ordering for consistent overlay alignment
        processed_data = self.standardize_dimension_order(processed_data)

        # 2. Apply denoising if requested
        denoise_components = kwargs.get(f"{widget_prefix}_denoise_components", 0)
        if denoise_components > 0:
            processed_data = self.apply_denoising(processed_data, denoise_components)

        # 3. Transform to k-space if requested (anatomical data)
        show_image_space = kwargs.get(f"{widget_prefix}_show_image_space", True)
        if not show_image_space:
            processed_data = self.apply_kspace_transform(processed_data)

        # 4. Apply line broadening if requested (spectroscopic data only)
        lb_value = kwargs.get(f"{widget_prefix}_linebroadening", 0.0)
        if lb_value > 0 and "acquisition_idx" in data.dims:
            processed_data = self.apply_linebroadening(
                processed_data, lb_value, widget_prefix
            )

        # 4.5. Calculate and apply zeropadding from npoints (spectroscopic data only, before FFT)
        # Get original data length
        original_npoints = data.sizes.get("acquisition_idx", 0)
        # Get requested npoints (which may include zeropadding)
        npoints = kwargs.get(
            f"{widget_prefix}_npoints",
            (
                original_npoints
                if original_npoints > 0
                else len(data.attrs.get("sampling_time_axis_s", []))
            ),
        )
        # Calculate zeropadding as the difference
        zeropadding = max(0, npoints - original_npoints)

        if zeropadding > 0 and "acquisition_idx" in processed_data.dims:
            processed_data = self.apply_zeropadding(processed_data, zeropadding)

        # 5. Repetition reduction/accumulation controls (optionally over a sub-range)
        cum_sum_reps = kwargs.get(f"{widget_prefix}_cum_sum_repetitions", False)
        average_reps = kwargs.get(f"{widget_prefix}_average_repetitions", False)
        if cum_sum_reps and "repetition_idx" in processed_data.dims:
            rep_start = kwargs.get(f"{widget_prefix}_rep_avg_start", 0)
            rep_end = kwargs.get(f"{widget_prefix}_rep_avg_end", None)
            processed_data = self.cumulative_sum_repetitions_range(
                processed_data, start=rep_start, end=rep_end
            )
        elif average_reps and "repetition_idx" in processed_data.dims:
            rep_start = kwargs.get(f"{widget_prefix}_rep_avg_start", 0)
            rep_end = kwargs.get(f"{widget_prefix}_rep_avg_end", None)
            processed_data = self.average_repetitions_range(
                processed_data, start=rep_start, end=rep_end
            )

        # 6. Apply domain transformation (temporal ↔ spectral) if needed
        is_fid = processed_data.attrs.get("signal_domain", "temporal") == "temporal"
        show_temporal = kwargs.get(f"{widget_prefix}_view_toggle", False)

        if "acquisition_idx" in processed_data.dims:
            processed_data = self.apply_domain_transform(
                processed_data,
                is_fid=is_fid,
                show_temporal=show_temporal,
                npoints=npoints,
            )

        # 7. Apply channel combination if requested
        combine_channels_active = kwargs.get(f"{widget_prefix}_combine_channels", False)
        if combine_channels_active and "channel_idx" in processed_data.dims:
            combine_method = kwargs.get(f"{widget_prefix}_combine_method", "rss")
            processed_data = self.apply_channel_combination(
                processed_data, method=combine_method
            )

        # 8. Apply data part transformation (Magnitude/Phase/Real/Imaginary)
        spec_part = kwargs.get(f"{widget_prefix}_spec_part")
        if spec_part and isinstance(spec_part, str):
            processed_data = self.apply_data_part_transform(processed_data, spec_part)

        return processed_data

    def standardize_dimension_order(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize dimension ordering for consistent overlay alignment.

        Parameters
        ----------
        data : xr.DataArray
            Input data array

        Returns
        -------
        xr.DataArray
            Data array with standardized dimension order
        """
        if data is None:
            return None

        # Define the preferred dimension order
        preferred_order = [
            "read_idx",
            "phase_idx",
            "slice_idx",  # Spatial dims first
            "waterfat_idx",
            "channel_idx",
            "inversion_idx",
            "repetition_idx",  # Common non-spatial
            "echo_idx",
            "frame_idx",
            "acquisition_idx",  # Temporal/acquisition
        ]

        # Get current dimensions
        current_dims = list(data.dims)

        # Create new dimension order: preferred dims first, then remaining dims
        new_dim_order = []
        remaining_dims = current_dims.copy()

        # Add preferred dimensions in order (if they exist)
        for dim in preferred_order:
            if dim in current_dims:
                new_dim_order.append(dim)
                remaining_dims.remove(dim)

        # Add any remaining dimensions at the end
        new_dim_order.extend(remaining_dims)

        # Only transpose if order actually changed
        if new_dim_order != current_dims:
            data = data.transpose(*new_dim_order)

        # Apply spatial orientation standardization
        data = self._standardize_spatial_orientation(data)

        return data

    def _standardize_spatial_orientation(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize spatial orientation for consistent display.

        Parameters
        ----------
        data : xr.DataArray
            Input data array

        Returns
        -------
        xr.DataArray
            Data array with standardized spatial orientation
        """
        if data is None:
            return None

        has_read_idx = "read_idx" in data.dims
        has_slice_idx = "slice_idx" in data.dims

        if not (has_read_idx or has_slice_idx):
            return data

        # Apply orientation rules
        if has_read_idx:
            # read_idx should go left→right
            try:
                if hasattr(data, "coords") and "read_idx" in data.coords:
                    read_coords = data.coords["read_idx"]
                    if len(read_coords) > 1:
                        first_val = float(read_coords.values[0])
                        last_val = float(read_coords.values[-1])
                        if first_val > last_val:
                            data = data.isel(read_idx=slice(None, None, -1))
            except (ValueError, TypeError, IndexError):
                pass

        elif has_slice_idx and not has_read_idx:
            # If no read_idx, slice_idx should go up→down
            try:
                if hasattr(data, "coords") and "slice_idx" in data.coords:
                    slice_coords = data.coords["slice_idx"]
                    if len(slice_coords) > 1:
                        first_val = float(slice_coords.values[0])
                        last_val = float(slice_coords.values[-1])
                        if first_val < last_val:
                            data = data.isel(slice_idx=slice(None, None, -1))
            except (ValueError, TypeError, IndexError):
                pass

        return data

    def apply_denoising(self, data: xr.DataArray, components: int) -> xr.DataArray:
        """Apply PCA denoising to data array.

        Parameters
        ----------
        data : xr.DataArray
            Input data array
        components : int
            Number of PCA components to keep

        Returns
        -------
        xr.DataArray
            Denoised data array
        """
        try:
            return pca_denoise(data, components=components)
        except Exception as e:
            warnings.warn(
                f"PCA denoising failed: {e}. Returning original data.", stacklevel=2
            )
            return data

    def apply_kspace_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Transform data from image space to k-space.

        Parameters
        ----------
        data : xr.DataArray
            Input data array in image space

        Returns
        -------
        xr.DataArray
            Data array in k-space. Data already declared as k-space
            (``attrs["domain"] == "kspace"``, for example from
            :func:`hypermri.analysis.spatial_filtering.spatial_to_kspace`) are
            returned unchanged, so they are never transformed twice.
        """
        if data.attrs.get("domain") == "kspace":
            return data
        try:
            spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
            transform_dims = [
                dim for dim in spatial_dims if dim in data.dims and data.sizes[dim] > 1
            ]

            if not transform_dims:
                return data

            # Apply FFT along spatial dimensions
            axes = [data.get_axis_num(dim) for dim in transform_dims]

            # Transform from image space to k-space (forward FFT). Bruker's CSI
            # scanner grid carries the centered-frequency modulation in its complex
            # voxel phases already; applying a second output shift puts k=0 at the
            # corners even though magnitude image space looks conventional. Every
            # other image follows the reconstruction's convention, so its exact
            # inverse returns the acquired k-space, also for odd matrix sizes.
            if data.attrs.get("csi_spatial_reconstruction") == "scanner_grid":
                centered_input = np.fft.ifftshift(data.values, axes=axes)
                transformed_data = np.fft.fftn(centered_input, axes=axes)
            else:
                transformed_data = centered_fft(data.values, axes)

            # Create new DataArray with transformed data. Replace image-space
            # coordinates with the matching centered spatial-frequency axes.
            result = data.copy(data=transformed_data)
            for dim in transform_dims:
                frequency_axis, frequency_unit = self._spatial_frequency_axis(data, dim)
                result = result.assign_coords(
                    {
                        dim: xr.DataArray(
                            frequency_axis,
                            dims=(dim,),
                            attrs={
                                "unit": frequency_unit,
                                "units": frequency_unit,
                                "domain": "kspace",
                            },
                        )
                    }
                )
            result.attrs["domain"] = "kspace"
            return result

        except Exception as e:
            warnings.warn(
                f"k-space transformation failed: {e}. Using original data.",
                stacklevel=2,
            )
            return data

    @staticmethod
    def _spatial_frequency_axis(data: xr.DataArray, dim: str) -> tuple[np.ndarray, str]:
        """Return a centered Fourier axis using sampling distance, not thickness.

        ``slice_spacing_mm`` on the slice coordinate takes precedence over
        the legacy ``resolution_mm`` attribute. Other dimensions and arrays
        without explicit slice spacing retain their previous interpretation.
        """
        size = int(data.sizes[dim])
        spacing = None
        source_unit = None
        coord = data.coords.get(dim)

        if coord is not None:
            source_unit = coord.attrs.get("unit") or coord.attrs.get("units")
            resolution_mm = (
                coord.attrs.get("slice_spacing_mm") if dim == "slice_idx" else None
            )
            if resolution_mm is not None:
                source_unit = "mm"
            if resolution_mm is None:
                resolution_mm = coord.attrs.get("resolution_mm")
            if resolution_mm is not None:
                try:
                    spacing = abs(float(resolution_mm))
                    source_unit = source_unit or "mm"
                except (TypeError, ValueError):
                    spacing = None

            if spacing is None:
                try:
                    selectors = {name: 0 for name in coord.dims if name != dim}
                    values = np.asarray(coord.isel(selectors).values, dtype=float)
                    values = values.reshape(-1)
                    diffs = np.abs(np.diff(values))
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size:
                        spacing = float(np.median(diffs))
                except (TypeError, ValueError, IndexError):
                    spacing = None

        if spacing is None or not np.isfinite(spacing) or spacing <= 0:
            spacing = 1.0
            source_unit = source_unit or "voxel"

        frequency_axis = np.fft.fftshift(np.fft.fftfreq(size, d=spacing))
        frequency_unit = f"1/{source_unit}" if source_unit else "1/voxel"
        return frequency_axis, frequency_unit

    def apply_linebroadening(
        self, data: xr.DataArray, lb_hz: float, widget_prefix: str = ""
    ) -> xr.DataArray:
        """Apply line broadening to spectroscopic data.

        Parameters
        ----------
        data : xr.DataArray
            Input spectroscopic data
        lb_hz : float
            Line broadening in Hz
        widget_prefix : str
            Widget prefix for accessing data_obj attributes

        Returns
        -------
        xr.DataArray
            Data with applied line broadening
        """
        if not hasattr(self.data_obj, "sampling_time_axis_s"):
            warnings.warn(
                "Cannot apply line broadening: no sampling_time_axis_s", stacklevel=2
            )
            return data

        dwell_time_s = (
            self.data_obj.sampling_time_axis_s[2]
            - self.data_obj.sampling_time_axis_s[1]
        )

        signal_domain = (
            "spectral"
            if data.attrs.get("signal_domain", "temporal") != "temporal"
            else "temporal"
        )

        axis = data.dims.index("acquisition_idx")

        # Apply line broadening
        result = data.copy()
        result.values = multi_dim_linebroadening(
            data.values,
            lb_hz=lb_hz,
            dwell_time_s=dwell_time_s,
            signal_domain=signal_domain,
            axis=axis,
        )

        return result

    def apply_zeropadding(self, data: xr.DataArray, num_zp: int) -> xr.DataArray:
        """Apply zeropadding to spectroscopic data.

        Adds zeros to the end of the FID to improve frequency resolution
        after FFT. Automatically converts from spectral to temporal domain
        if needed, applies zeropadding, then converts back.

        Parameters
        ----------
        data : xr.DataArray
            Input data array (can be in temporal or spectral domain)
        num_zp : int
            Number of zeros to pad

        Returns
        -------
        xr.DataArray
            Zeropadded data array (in the same domain as input)

        Notes
        -----
        Zeropadding must be applied in temporal domain before FFT. This method
        handles domain conversion automatically. It does not add new information
        but provides better interpolation in the frequency domain.
        """
        from hypermri.analysis.spectroscopy.processing import zeropad_data

        if num_zp <= 0:
            return data

        # Only apply to data with acquisition dimension
        if "acquisition_idx" not in data.dims:
            return data

        # Check current domain
        is_temporal = data.attrs.get("signal_domain", "temporal") == "temporal"

        # Convert to temporal domain if needed
        working_data = data.copy(deep=True)
        if not is_temporal:
            # Convert from spectral to temporal domain
            working_data = self._apply_inverse_fft(working_data)
            working_data.attrs["signal_domain"] = "temporal"

        try:
            # Get the acquisition axis index
            acq_axis = working_data.dims.index("acquisition_idx")

            # Apply zeropadding along the acquisition axis
            zeropadded_values = np.apply_along_axis(
                lambda arr: zeropad_data(
                    values=arr,
                    num_zp=num_zp,
                    fid_start=0,
                    fid_end=len(arr),
                    signal_domain="temporal",
                ),
                axis=acq_axis,
                arr=working_data.values,
            )

            # Create new acquisition index coordinates
            old_acq_coords = working_data.coords["acquisition_idx"]
            if hasattr(old_acq_coords, "values"):
                old_acq_values = old_acq_coords.values
            else:
                old_acq_values = old_acq_coords

            # Extend the acquisition coordinates
            new_length = len(old_acq_values) + num_zp
            if len(old_acq_values) > 1:
                dt = old_acq_values[1] - old_acq_values[0]
                new_acq_coords = np.arange(new_length) * dt
            else:
                new_acq_coords = np.arange(new_length)

            # Create new DataArray with extended coordinates
            # Copy all coordinates except those that depend on acquisition_idx dimension
            new_coords = {}
            for coord_name, coord_data in working_data.coords.items():
                if coord_name == "acquisition_idx":
                    # Use the extended acquisition coordinates
                    new_coords[coord_name] = new_acq_coords
                elif "acquisition_idx" in getattr(coord_data, "dims", []):
                    # Extend coordinates that have acquisition_idx dimension
                    old_coord_values = coord_data.values
                    if len(old_coord_values) > 1:
                        dt_coord = old_coord_values[1] - old_coord_values[0]
                        new_coord_values = np.arange(new_length) * dt_coord
                    else:
                        new_coord_values = np.arange(new_length)
                    new_coords[coord_name] = (coord_data.dims, new_coord_values)
                else:
                    # Copy other coordinates unchanged
                    new_coords[coord_name] = coord_data

            result = xr.DataArray(
                zeropadded_values,
                dims=working_data.dims,
                coords=new_coords,
                attrs=working_data.attrs.copy(),
            )

            # Convert back to spectral domain if input was spectral
            if not is_temporal:
                result = self._apply_fft(result)
                result.attrs["signal_domain"] = "spectral"

            return result

        except Exception as e:
            import warnings

            warnings.warn(
                f"Zeropadding failed: {e}. Returning original data.", stacklevel=2
            )
            return data

    def _apply_fft(self, data: xr.DataArray) -> xr.DataArray:
        """Apply FFT transform along acquisition dimension (temporal → spectral).

        Parameters
        ----------
        data : xr.DataArray
            Input data array in temporal domain

        Returns
        -------
        xr.DataArray
            Data array in spectral domain
        """
        if "acquisition_idx" not in data.dims:
            return data

        acq_axis = data.dims.index("acquisition_idx")
        spectral_values = spectral_fft(data.values, axis=acq_axis)

        result = data.copy(data=spectral_values)
        result.attrs["signal_domain"] = "spectral"
        return result

    def _apply_inverse_fft(self, data: xr.DataArray) -> xr.DataArray:
        """Apply inverse FFT transform along acquisition dimension (spectral → temporal).

        Parameters
        ----------
        data : xr.DataArray
            Input data array in spectral domain

        Returns
        -------
        xr.DataArray
            Data array in temporal domain
        """
        if "acquisition_idx" not in data.dims:
            return data

        acq_axis = data.dims.index("acquisition_idx")
        temporal_values = spectral_ifft(data.values, axis=acq_axis)

        result = data.copy(data=temporal_values)
        result.attrs["signal_domain"] = "temporal"
        return result

    def apply_domain_transform(
        self, data: xr.DataArray, is_fid: bool, show_temporal: bool, npoints: int
    ) -> xr.DataArray:
        """Transform between temporal (FID) and spectral domains.

        Parameters
        ----------
        data : xr.DataArray
            Signal with the spectral or temporal axis ``acquisition_idx``.
        is_fid : bool
            Whether ``data`` holds temporal samples (FIDs).
        show_temporal : bool
            Whether the FID view is requested; otherwise the spectrum.
        npoints : int
            Number of samples kept along ``acquisition_idx``.

        Returns
        -------
        xr.DataArray
            New array in the requested domain with ``signal_domain`` set.
            Spectra follow :func:`~hypermri.utils.mri.fourier.spectral_fft`.

        Raises
        ------
        ValueError
            If the FID view is requested for real-valued spectra, whose
            phase is lost (see
            :func:`~hypermri.utils.mri.fourier.fid_is_recoverable`).
        """
        if "acquisition_idx" not in data.dims:
            return data

        dim_name = "acquisition_idx"
        current_size = data.sizes[dim_name]
        actual_npoints = min(npoints, current_size)

        # Case 1: Temporal data, want temporal view
        if show_temporal and is_fid:
            slicer = {dim_name: slice(0, actual_npoints)}
            return data.isel(**slicer)

        # Case 2: Spectral data, want temporal view (IFFT)
        elif show_temporal and not is_fid:
            if not fid_is_recoverable(data):
                raise ValueError(
                    "apply_domain_transform: the FID view needs complex spectra, "
                    "but these spectra are real (for example a scanner 2dseq "
                    "stored as magnitude), so their phase is lost. Show the "
                    "spectrum, or plot the raw reconstruction scan.data."
                )
            slicer = {dim_name: slice(0, current_size)}
            processed = data.isel(**slicer)
            processed = xr.apply_ufunc(
                lambda x: spectral_ifft(x, axis=-1),
                processed,
                input_core_dims=[[dim_name]],
                output_core_dims=[[dim_name]],
                keep_attrs=True,
            )
            slicer = {dim_name: slice(0, npoints)}
            processed = processed.isel(**slicer)
            processed.attrs["signal_domain"] = "temporal"
            return processed

        # Case 3: Temporal data, want spectral view (FFT)
        elif not show_temporal and is_fid:
            slicer = {dim_name: slice(0, npoints)}
            processed = data.isel(**slicer)
            processed = xr.apply_ufunc(
                lambda x: spectral_fft(x, axis=-1),
                processed,
                input_core_dims=[[dim_name]],
                output_core_dims=[[dim_name]],
                keep_attrs=True,
            )
            processed.attrs["signal_domain"] = "spectral"
            return processed

        # Case 4: Spectral data, want spectral view
        else:
            # Apply spec -> FID -> spec transformation to get correct npoints
            slicer = {dim_name: slice(0, current_size)}
            processed = data.isel(**slicer)
            processed = xr.apply_ufunc(
                lambda x: spectral_ifft(x, axis=-1),
                processed,
                input_core_dims=[[dim_name]],
                output_core_dims=[[dim_name]],
                keep_attrs=True,
            )
            slicer = {dim_name: slice(0, npoints)}
            processed = processed.isel(**slicer)
            processed = xr.apply_ufunc(
                lambda x: spectral_fft(x, axis=-1),
                processed,
                input_core_dims=[[dim_name]],
                output_core_dims=[[dim_name]],
                keep_attrs=True,
            )
            processed.attrs["signal_domain"] = "spectral"
            return processed

    def apply_channel_combination(
        self, data: xr.DataArray, method: str = "rss"
    ) -> xr.DataArray:
        """Combine multiple channels using requested method.

        Parameters
        ----------
        data : xr.DataArray
            Input data with channel_idx dimension
        method : str, optional
            Combination method ('rss', 'sum', 'mean', 'adaptive'). Default is 'rss'.

        Returns
        -------
        xr.DataArray
            Data with channels combined. The channel_idx dimension has size 1.

        Notes
        -----
        This method now delegates to the general-purpose channel combination utility.
        """
        if "channel_idx" not in data.dims:
            return data

        # Use the general-purpose channel combination utility
        # This will keep channel_idx dimension with size 1
        from hypermri.utils.mri.channels import combine_channels

        return combine_channels(data, method=method)

    def apply_data_part_transform(
        self, data: xr.DataArray, spec_part: str
    ) -> xr.DataArray:
        """Apply data part transformation (Magnitude/Phase/Real/Imaginary).

        Parameters
        ----------
        data : xr.DataArray
            Input data array
        spec_part : str
            Type of transformation: 'Magnitude', 'Phase', 'Real', or 'Imaginary'

        Returns
        -------
        xr.DataArray
            Transformed data array
        """
        transformed_values = part_func(data.values, spec_part)

        result = data.copy(data=transformed_values)
        result.attrs["spec_part_applied"] = spec_part

        return result

    def average_repetitions(self, data: xr.DataArray) -> xr.DataArray:
        """Average data along the repetition dimension.

        Parameters
        ----------
        data : xr.DataArray
            Input data with repetition_idx dimension

        Returns
        -------
        xr.DataArray
            Data averaged along repetition_idx
        """
        if "repetition_idx" not in data.dims:
            return data
        # Preserve metadata such as `signal_domain` so downstream logic keeps the
        # correct spectral/FID interpretation after averaging.
        averaged = data.mean(dim="repetition_idx", keep_attrs=True)
        if averaged.attrs is data.attrs:
            averaged.attrs = dict(data.attrs)
        elif data.attrs:
            for key, value in data.attrs.items():
                averaged.attrs.setdefault(key, value)
        return averaged

    def average_repetitions_range(
        self, data: xr.DataArray, start: int = 0, end: Optional[int] = None
    ) -> xr.DataArray:
        """Average data along repetition_idx over an inclusive index range [start, end].

        Parameters
        ----------
        data : xr.DataArray
            Input data with repetition_idx dimension.
        start : int
            First repetition index to include (0-based).
        end : int or None
            Last repetition index to include (inclusive). None means last rep.
        """
        if "repetition_idx" not in data.dims:
            return data
        n = data.sizes["repetition_idx"]
        s = max(0, int(start))
        e = (n - 1) if end is None else min(int(end), n - 1)
        e = max(e, s)  # ensure at least one repetition
        sliced = data.isel(repetition_idx=slice(s, e + 1))
        averaged = sliced.mean(dim="repetition_idx", keep_attrs=True)
        if averaged.attrs is data.attrs:
            averaged.attrs = dict(data.attrs)
        elif data.attrs:
            for key, value in data.attrs.items():
                averaged.attrs.setdefault(key, value)
        return averaged

    def cumulative_sum_repetitions_range(
        self, data: xr.DataArray, start: int = 0, end: Optional[int] = None
    ) -> xr.DataArray:
        """Apply cumulative sum along repetition_idx over an inclusive range [start, end].

        The output keeps the original repetition dimension length. Values outside the
        selected range are preserved unchanged.
        """
        if "repetition_idx" not in data.dims:
            return data

        n = data.sizes["repetition_idx"]
        s = max(0, int(start))
        e = (n - 1) if end is None else min(int(end), n - 1)
        e = max(e, s)

        result = data.copy(deep=True)
        subset = data.isel(repetition_idx=slice(s, e + 1))
        subset_cumsum = subset.cumsum(dim="repetition_idx", keep_attrs=True)
        result.loc[{"repetition_idx": subset["repetition_idx"]}] = subset_cumsum

        if result.attrs is data.attrs:
            result.attrs = dict(data.attrs)
        elif data.attrs:
            for key, value in data.attrs.items():
                result.attrs.setdefault(key, value)
        return result
