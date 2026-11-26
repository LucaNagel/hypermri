"""Shared reconstruction helpers for FID-based spectroscopy vendors."""

from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.reconstruction.precision import convert_complex_for_storage
from hypermri.utils.mri.channels import combine_channels as combine_mri_channels
from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft
from hypermri.utils.mri.signal import rotate_phase, zero_fill_temporal

from ._common import debug_log_recon_strategy
from .base import BaseRecoStrategy


class FidSpectroRecoStrategy(BaseRecoStrategy):
    """Base reconstruction for vendor handlers exposing FID spectroscopy stacks."""

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "append_parameters",
            "digital_filter_points",
            "apply_digital_filter",
            "remove_dc",
            "apodization",
            "line_broadening_hz",
            "gaussian_broadening_hz",
            "zero_fill_to",
            "zero_fill_factor",
            "combine_channels",
            "retain_channel_axis",
            "combine_method",
            "apply_phasing",
            "phase0_deg",
            "phase1_deg",
        }
    )

    spectrum_name = "spectrum"
    missing_fid_message = "Handler has no FID loaded"

    def reconstruct(self, handler, **kwargs) -> xr.DataArray:
        """Fourier transform independent FIDs along the acquisition axis.

        Labelled input retains repetition, receiver and all extra export axes.
        Unlabelled arrays support acquisition, channel/acquisition, or
        repetition/channel/acquisition order only. No extra axis is flattened.
        Processing works on derived arrays without modifying acquired samples;
        the returned spectrum is stored on the handler as primary raw data.

        Parameters
        ----------
        handler : VendorHandler
            FID samples or a lazily loadable vendor export, sampling interval
            in seconds, frequency coordinates in Hz/ppm and optional defaults.
        **kwargs : dict
            Processing settings: ``digital_filter_points`` removes a nonnegative
            number of leading samples when ``apply_digital_filter=True``;
            ``remove_dc`` subtracts the individual FID mean (default True).
            ``apodization`` selects the window; ``line_broadening_hz`` and
            ``gaussian_broadening_hz`` set its widths in Hz. ``zero_fill_to`` or
            ``zero_fill_factor`` (at least 1) pads each FID. ``apply_phasing``
            enables ``phase0_deg``/``phase1_deg`` in degrees. ``combine_channels``
            (default True), ``combine_method`` (default rss) and
            ``retain_channel_axis`` (default False) control receiver reduction;
            all other independent loops are retained. ``append_parameters``
            (default True) attaches scanner metadata. Unspecified scientific
            settings use the corresponding handler values.

        Returns
        -------
        xarray.DataArray
            Spectrum in unscaled scanner intensity units, ordered repetition,
            channel, retained extra axes, acquisition. Channel reduction may
            remove its dimension. Frequency coordinates are rebuilt if filtering
            or zero filling changes the number of samples. Complex64 input
            storage is transformed with NumPy's FFT precision policy; the output
            retains that calculation dtype. The operation is eager and
            deterministic and only loads a missing FID once per invocation.

        Raises
        ------
        ValueError
            If samples are missing, extra axes lack labels, filtering removes
            all samples, or processing settings/coordinates are inconsistent.
        OSError
            If a lazily loaded vendor export cannot be read.

        Examples
        --------
        >>> spectrum = scan.reconstruct("raw", combine_channels=False)  # doctest: +SKIP
        """
        fid = self._ensure_handler_fid(handler)
        if fid is None:
            raise ValueError(self.missing_fid_message)

        retained_coords = {}
        inserted_channel = False
        if isinstance(fid, xr.DataArray):
            if "acquisition_idx" not in fid.dims:
                raise ValueError("Labelled spectroscopy FID requires acquisition_idx.")
            for dim in ("repetition_idx", "channel_idx"):
                if dim not in fid.dims:
                    inserted_channel |= dim == "channel_idx"
                    fid = fid.expand_dims({dim: [0]})
            extra_dims = [
                dim
                for dim in fid.dims
                if dim not in {"repetition_idx", "channel_idx", "acquisition_idx"}
            ]
            fid = fid.transpose(
                "repetition_idx", "channel_idx", *extra_dims, "acquisition_idx"
            )
            dims = list(fid.dims)
            retained_coords = {
                name: coord
                for name, coord in fid.coords.items()
                if "acquisition_idx" not in coord.dims
            }
            fid = convert_complex_for_storage(fid.data)
        else:
            fid = self._ensure_fid_shape(np.asarray(fid))
            dims = ["repetition_idx", "channel_idx", "acquisition_idx"]
        debug_log_recon_strategy(message=f"kwargs: {kwargs}")
        debug_log_recon_strategy(message=f"fid.shape: {fid.shape}")
        append_parameters = kwargs.get("append_parameters", True)
        processing_log = []

        digital_filter_points = kwargs.get(
            "digital_filter_points",
            getattr(handler, "digital_filter_points", 0) or 0,
        )
        apply_digital_filter = kwargs.get("apply_digital_filter", True)
        remove_dc = kwargs.get("remove_dc", True)

        apodization = kwargs.get("apodization")
        lb_hz = kwargs.get(
            "line_broadening_hz",
            getattr(handler, "line_broadening_hz", 0.0) or 0.0,
        )
        gb_hz = kwargs.get(
            "gaussian_broadening_hz",
            getattr(handler, "gaussian_broadening_hz", 0.0) or 0.0,
        )

        zero_fill_to = kwargs.get("zero_fill_to")
        zero_fill_factor = kwargs.get("zero_fill_factor")
        if zero_fill_to is None and zero_fill_factor is None:
            default_zf = getattr(handler, "zero_fill_points", None)
            if default_zf:
                zero_fill_to = int(default_zf)

        combine_channels = kwargs.get("combine_channels", True)
        retain_channel_axis = kwargs.get("retain_channel_axis", False)
        combine_method = kwargs.get("combine_method", "rss")

        apply_phasing = kwargs.get("apply_phasing", False)
        phase0_deg = kwargs.get(
            "phase0_deg", getattr(handler, "phase0_deg", 0.0) or 0.0
        )
        phase1_deg = kwargs.get(
            "phase1_deg", getattr(handler, "phase1_deg", 0.0) or 0.0
        )

        dwell_time = getattr(handler, "dwell_time_s", None)
        if dwell_time is None or dwell_time == 0:
            bandwidth = getattr(handler, "bandwidth_hz", None)
            dwell_time = 1.0 / bandwidth if bandwidth else 1.0

        fid, sampling_time_axis_s = self._apply_digital_filter(
            fid,
            dwell_time,
            digital_filter_points,
            apply_digital_filter,
            processing_log,
        )

        fid = self._maybe_remove_dc(fid, remove_dc, processing_log)
        fid = self._apply_apodization(
            fid,
            dwell_time,
            apodization,
            lb_hz,
            gb_hz,
            processing_log,
        )
        fid, sampling_time_axis_s = self._apply_zero_fill(
            fid,
            sampling_time_axis_s,
            dwell_time,
            zero_fill_to,
            zero_fill_factor,
            processing_log,
        )

        spectral = spectral_fft(fid, axis=-1)

        if apply_phasing and (phase0_deg or phase1_deg):
            spectral = self._apply_phase_correction(
                spectral, phase0_deg=phase0_deg, phase1_deg=phase1_deg
            )
            processing_log.append(
                f"phase_correction: rp={phase0_deg:.2f}deg lp={phase1_deg:.2f}deg"
            )

        freq_axis_hz = getattr(handler, "freq_axis_hz", None)
        freq_axis_ppm = getattr(handler, "freq_axis_ppm", None)
        if (freq_axis_hz is None and freq_axis_ppm is None) or any(
            axis is not None and np.asarray(axis).size != spectral.shape[-1]
            for axis in (freq_axis_hz, freq_axis_ppm)
        ):
            freq_axis_hz, freq_axis_ppm = self._build_frequency_axes(
                spectral.shape[-1],
                dwell_time,
                handler,
            )

        coords = {
            "repetition_idx": (
                "repetition_idx",
                np.arange(spectral.shape[0]),
                {"unit": "index"},
            ),
            "channel_idx": (
                "channel_idx",
                np.arange(spectral.shape[1]),
                {"unit": "index"},
            ),
            "acquisition_idx": (
                "acquisition_idx",
                np.arange(spectral.shape[-1]),
                {"unit": "index"},
            ),
        }

        for axis, dim in enumerate(dims):
            if dim not in coords:
                coords[dim] = (dim, np.arange(spectral.shape[axis]), {"unit": "index"})
        coords.update(retained_coords)

        if freq_axis_hz is not None:
            coords["freq_axis_hz"] = (
                "acquisition_idx",
                freq_axis_hz,
                {"unit": "Hz"},
            )

        if freq_axis_ppm is not None:
            coords["freq_axis_ppm"] = (
                "acquisition_idx",
                freq_axis_ppm,
                {"unit": "ppm"},
            )

        if sampling_time_axis_s is not None:
            coords["acquisition_time_s"] = (
                "acquisition_idx",
                sampling_time_axis_s,
                {"unit": "s"},
            )

        spectral_attrs = {
            "domain": "spectral",
            "signal_domain": "spectral",
            "bandwidth_hz": getattr(handler, "bandwidth_hz", None),
            "dwell_time_s": dwell_time,
            "b0_tesla": getattr(handler, "b0_tesla", None),
            "nucleus": getattr(handler, "nucleus", None),
            "channel_combination": "none",
            "processing_log": processing_log,
            "sequence_variant": getattr(handler, "sequence_variant", None),
        }

        if sampling_time_axis_s is not None:
            spectral_attrs["time_axis_s"] = sampling_time_axis_s

        reco_data = xr.DataArray(
            spectral,
            dims=tuple(dims),
            coords=coords,
            attrs=spectral_attrs,
            name=self.spectrum_name,
        )

        combine_method_normalized = (combine_method or "rss").lower()
        if (
            combine_channels
            and combine_method_normalized not in {"none", "off"}
            and reco_data.sizes["channel_idx"] > 1
        ):
            reco_data = combine_mri_channels(
                reco_data,
                method=combine_method_normalized,
            )
            processing_log.append(f"coil_combination:{combine_method_normalized}")
            reco_data.attrs["channel_combination"] = combine_method_normalized

            if not retain_channel_axis:
                reco_data = reco_data.isel(channel_idx=0, drop=True)

        if (
            inserted_channel
            and combine_channels
            and combine_method_normalized not in {"none", "off"}
            and not retain_channel_axis
            and "channel_idx" in reco_data.dims
        ):
            reco_data = reco_data.isel(channel_idx=0, drop=True)

        if append_parameters:
            reco_data = self._append_parameters(array=reco_data, handler=handler)

        handler.set_primary_data(reco_data, "data")
        return reco_data

    def _ensure_handler_fid(self, handler):
        if getattr(handler, "fid", None) is None:
            handler.load_data(getattr(handler, "path", None))
        return getattr(handler, "fid", None)

    @staticmethod
    def _ensure_fid_shape(fid: np.ndarray) -> np.ndarray:
        if fid.ndim == 1:
            fid = fid[np.newaxis, np.newaxis, :]
        elif fid.ndim == 2:
            fid = fid[np.newaxis, :, :]
        elif fid.ndim > 3:
            raise ValueError(
                "Spectroscopy FIDs with extra axes require a labelled DataArray; independent axes cannot be flattened into acquisition_idx."
            )
        return convert_complex_for_storage(fid)

    def _apply_digital_filter(
        self,
        fid: np.ndarray,
        dwell_time: float,
        lsfid_points: int,
        apply_filter: bool,
        processing_log,
    ):
        if apply_filter and lsfid_points:
            if lsfid_points >= fid.shape[-1]:
                raise ValueError("Digital filter removal would drop all FID points")
            fid = fid[..., lsfid_points:]
            processing_log.append(
                f"digital_filter: removed first {lsfid_points} points"
            )

        time_axis = np.arange(fid.shape[-1]) * dwell_time
        return fid, time_axis

    @staticmethod
    def _maybe_remove_dc(
        fid: np.ndarray, remove_dc: bool, processing_log
    ) -> np.ndarray:
        if remove_dc:
            fid = fid - fid.mean(axis=-1, keepdims=True)
            processing_log.append("dc_offset_removed")
        return fid

    def _apply_apodization(
        self,
        fid: np.ndarray,
        dwell_time: float,
        apodization,
        lb_hz: float,
        gb_hz: float,
        processing_log,
    ) -> np.ndarray:
        window, mode = self._build_apodization_window(
            fid.shape[-1], dwell_time, apodization, lb_hz, gb_hz
        )
        if window is not None:
            fid = fid * window
            processing_log.append(f"apodization:{mode}")
        return fid

    def _apply_zero_fill(
        self,
        fid: np.ndarray,
        time_axis: np.ndarray,
        dwell_time: float,
        zero_fill_to,
        zero_fill_factor,
        processing_log,
    ):
        target_points = fid.shape[-1]
        if zero_fill_to is not None:
            target_points = max(target_points, int(zero_fill_to))
        elif zero_fill_factor is not None:
            factor = float(zero_fill_factor)
            if factor < 1.0:
                raise ValueError("zero_fill_factor must be >= 1.0")
            target_points = int(np.ceil(fid.shape[-1] * factor))

        if target_points > fid.shape[-1]:
            pad = target_points - fid.shape[-1]
            fid = zero_fill_temporal(fid, pad)
            time_axis = np.arange(fid.shape[-1]) * dwell_time
            processing_log.append(f"zero_fill:{fid.shape[-1]}")

        return fid, time_axis

    @staticmethod
    def _build_apodization_window(
        n_points: int,
        dwell_time: float,
        mode,
        lb_hz: float,
        gb_hz: float,
    ) -> tuple[np.ndarray | None, str | None]:
        if not mode or str(mode).lower() in {"none", "", "off"}:
            return None, None

        mode = str(mode).lower()
        t = np.arange(n_points) * dwell_time

        if mode in {"auto", "default"}:
            if lb_hz > 0:
                mode = "exponential"
            elif gb_hz > 0:
                mode = "gaussian"
            else:
                return None, None

        if mode == "exponential":
            lb = lb_hz if lb_hz > 0 else 1.0
            window = np.exp(-np.pi * lb * t)
        elif mode == "gaussian":
            gb = gb_hz if gb_hz > 0 else 1.0
            window = np.exp(-((gb * t) ** 2))
        elif mode == "hamming":
            window = np.hamming(n_points)
        elif mode == "hann":
            window = np.hanning(n_points)
        elif mode == "blackman":
            window = np.blackman(n_points)
        else:
            raise ValueError(f"Unsupported apodization mode '{mode}'")

        return window.astype(np.float32, copy=False), mode

    @staticmethod
    def _apply_phase_correction(
        spectral: np.ndarray, phase0_deg: float, phase1_deg: float
    ) -> np.ndarray:
        phase0 = np.deg2rad(phase0_deg)
        phase1 = np.deg2rad(phase1_deg)
        n_points = spectral.shape[-1]
        phase_axis = np.linspace(-0.5, 0.5, n_points, endpoint=False)
        phase_ramp = phase0 + phase1 * phase_axis
        return rotate_phase(spectral, phase_ramp)

    @staticmethod
    def _build_frequency_axes(
        n_points: int, dwell_time: float, handler
    ) -> tuple[np.ndarray, np.ndarray | None]:
        freq_hz = np.fft.fftshift(np.fft.fftfreq(n_points, d=dwell_time))

        freq_ppm = None
        spectrometer_freq = getattr(handler, "spectrometer_frequency_mhz", None)
        if spectrometer_freq:
            freq_ppm = freq_hz / spectrometer_freq
            ref_freq = getattr(handler, "reference_frequency_mhz", None)
            if ref_freq:
                offset_ppm = (spectrometer_freq - ref_freq) / spectrometer_freq
                freq_ppm = freq_ppm + offset_ppm

        return freq_hz, freq_ppm


__all__ = ["FidSpectroRecoStrategy"]
