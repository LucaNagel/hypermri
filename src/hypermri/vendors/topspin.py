"""Handler for Bruker TopSpin NMR data.

This module provides a vendor handler for Bruker TopSpin NMR datasets,
which use a different directory structure than ParaVision MRI datasets.

TopSpin scan directories contain:
- ``acqu`` / ``acqus`` – acquisition parameter files (JCAMP-DX format)
- ``fid`` – raw free-induction decay data (binary)
- ``pdata/`` – processed data (optional)

These are distinct from ParaVision datasets which use ``acqp`` / ``method``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hypermri.mixins import StandardizedMRIMixin
from hypermri.utils.system.logging import LOG_MODES, init_default_logger
from hypermri.vendors._jcamp import (
    _parse_topspin_array,
    _read_topspin_parameters,
)
from hypermri.vendors.base import VendorHandler

logger = init_default_logger(
    # Preserve the established log category through the import migration.
    "hypermri.topspin_exp",
    fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s ",
)


class BrukerTopSpinHandler(VendorHandler, StandardizedMRIMixin):
    """Vendor handler for Bruker TopSpin NMR datasets.

    Attribute names mirror those of :class:`~hypermri.vendors.bruker.BrukerHandler`
    so that downstream spectroscopy utilities (``get_freq_axis``, etc.) work
    without modification.

    Parameters
    ----------
    source : str or Path
        Path to a single TopSpin experiment directory (e.g. ``study/10/``).
    load_data : bool, optional
        If ``True`` the FID binary is read immediately. Default ``False``.
    log_mode : str, optional
        Logging verbosity from LOG_MODES. Unrecognized names retain the
        historical error-level fallback; changes this module's shared logger.
    **kwargs : Any
        Legacy constructor keywords, accepted and ignored as before. New
        callers should use only the named parameters.

    Raises
    ------
    FileNotFoundError
        If the directory does not contain a TopSpin fid and acqu/acqus pair,
        or if an eagerly requested binary is absent.
    OSError
        If metadata or requested binary data cannot be read.
    ValueError
        If numeric metadata or the acquired binary layout cannot be interpreted.

    Attributes
    ----------
    reference_frequency_mhz : float or None
        Frequency of 0 ppm in MHz, read from ``BF1`` in ``acqus``. None when
        ``BF1`` is missing or not positive. ``SFO1`` (transmitter frequency)
        and ``SF`` in ``procs`` are not used.
    b0_tesla : float or None
        Static field in T derived from ``reference_frequency_mhz`` and the
        nucleus; None without a reference frequency.
    center_ppm : float
        Transmitter offset in ppm, ``O1 / BF1`` with ``O1`` in Hz and ``BF1``
        in MHz; it is the ppm value of 0 Hz on the frequency axes. 0.0 when
        ``BF1`` is missing.
    bandwidth_ppm : float or None
        Spectral width in ppm, ``SW_h / BF1`` with ``SW_h`` in Hz and ``BF1``
        in MHz, so the ppm spacing of the frequency axes is ``Hz / BF1``.
        ``SW`` in ``acqus`` (``SW_h / SFO1``) is not used. None without
        ``BF1``; the ppm axis is then absent.

    Notes
    -----
    Construction reads acquisition metadata, stores the canonical snapshot and
    prepares frequency coordinates. It reads no binary samples unless load_data
    is True and initializes no viewer. Acquired files are never modified.
    The existing 70-complex-sample cutoff and constructor-time frequency-axis
    preparation are preserved. This relocation does not validate or revise
    those scientific choices.

    Examples
    --------
    >>> from hypermri.vendors.topspin import BrukerTopSpinHandler
    >>> handler = BrukerTopSpinHandler("study/10")  # doctest: +SKIP
    >>> fid = handler.load_data()  # doctest: +SKIP
    """

    def __init__(
        self,
        source: str | Path,
        load_data: bool = False,
        log_mode: str = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        logger.setLevel(LOG_MODES.get(log_mode, LOG_MODES["error"]))
        self.log_mode = log_mode

        self.path = Path(source).resolve()

        if not self.is_valid_scan_dir(self.path):
            raise FileNotFoundError(
                f"{self.path} is not a valid TopSpin experiment directory."
            )

        self.load_parameters(source)
        self.has_dicom_reco = False

        # Compute and cache frequency axes immediately
        self._freq_axis_ppm = None
        self._freq_axis_hz = None
        self._freq_axis_ppm = self._compute_freq_axis("ppm")
        self._freq_axis_hz = self._compute_freq_axis("hz")

        if load_data:
            self.load_data(source)

    # ------------------------------------------------------------------
    # VendorHandler abstract interface
    # ------------------------------------------------------------------

    @property
    def id(self) -> str | int:
        """Return the experiment directory identifier without reading data.

        Returns
        -------
        str or int
            Directory name converted to an integer when entirely numeric.
        """
        return self.expnum

    @property
    def type(self) -> str:
        """Return the pulse-program name parsed from acquisition metadata.

        Returns
        -------
        str
            PULPROG value, or unknown when missing. No data is loaded.
        """
        return self.sequence_name

    @staticmethod
    def is_valid_scan_dir(path: str | Path) -> bool:
        """Return ``True`` for a TopSpin experiment directory.

        Parameters
        ----------
        path : str or Path
            Candidate experiment directory. Checks entry names only; metadata
            and binary contents are not read or changed.

        Returns
        -------
        bool
            Whether the directory contains fid and at least one of acqu/acqus,
            and contains neither acqp nor method (ParaVision markers).

        Raises
        ------
        OSError
            If an existing directory cannot be listed.
        """
        path = Path(path)
        if not path.is_dir():
            return False
        contents = set(os.listdir(path))
        has_fid = "fid" in contents
        has_acqu = bool(contents & {"acqu", "acqus"})
        is_paravision = bool(contents & {"acqp", "method"})
        return has_fid and has_acqu and not is_paravision

    def get(self, attribute_name: str, default=None):
        """Read an attribute with the historical dictionary-style fallback.

        Parameters
        ----------
        attribute_name : str
            Handler attribute name, including standardized acquisition metadata.
        default : object, optional
            Value returned when the attribute does not exist.

        Returns
        -------
        object
            Attribute itself or default, without copying or changing its units
            or dtype. Attribute properties retain their own loading behavior.
        """
        return getattr(self, attribute_name, default)

    def _parse_parameters(self, source) -> dict[str, Any]:
        input_path = source.path if hasattr(source, "path") else source
        self.path = Path(input_path).resolve()

        # Parse parameters from acqus (status file), fall back to acqu.
        self.acqu = self._read_param_file("acqus") or self._read_param_file("acqu")

        self.vendor = "bruker_topspin"
        self.expnum = (
            int(self.path.name) if self.path.name.isdigit() else self.path.name
        )
        self.experiment_number = self.expnum
        self.sequence_name = self._extract_str_param("PULPROG") or "unknown"
        self.nucleus = self._extract_str_param("NUC1") or "unknown"
        self.method_type = "stock"
        self.is_spectroscopic = True
        self.cut_off = 70
        self.git_commit_hash = None

        date_unix = self.acqu.get("DATE", None)
        self.timestamp = (
            datetime.fromtimestamp(int(date_unix), tz=timezone.utc)
            if date_unix is not None and int(date_unix) > 0
            else None
        )
        self.measurement_time = self.timestamp

        self.subject_name = None
        self.subject_id = None
        self.subject_position = None
        self.subject_type = None
        self.patient_position = None

        self.spatial_dimension = "spectroscopy"
        self.n_slice_packs = 1
        self.n_slices = 1
        self.slice_thickness_mm = 0.0
        self.slice_gap_mm = 0.0
        self.slice_orientation = None
        self.slice_order = np.array([0])
        self.slice_offset_mm = 0.0
        self.slice_offset_mm_array = np.array([0.0])
        self.read_orientation = None
        self.read_offset_mm = 0.0
        self.phase_offset_mm = 0.0
        self.phase_0_offset_mm = 0.0
        self.phase_1_offset_mm = 0.0
        self.position_matrix = np.eye(3)
        self.fov_mm = np.array([1.0, 1.0, 1.0])
        self.matrix = np.array([1, 1, 1])
        self.encoding_matrix = np.array([1, 1, 1])
        self.resolution_mm = None

        # BF1 is the basic frequency of the observe channel, i.e. the frequency
        # of 0 ppm. SFO1 = BF1 + O1 is the transmitter frequency and is not the
        # ppm reference; procs/SF is deliberately not read.
        bf1_mhz = float(self.acqu.get("BF1", 0.0))
        self.reference_frequency_mhz = bf1_mhz if bf1_mhz > 0 else None
        self.b0_tesla = (
            self._derive_b0_tesla() if self.reference_frequency_mhz else None
        )
        self.bandwidth_hz = float(self.acqu.get("SW_h", 0.0))
        # The ppm spacing uses the same 0-ppm frequency BF1 as center_ppm, so
        # one Hz step is 1 / BF1 ppm everywhere on the axis. TopSpin's own SW
        # divides by SFO1 instead and is not read.
        self.bandwidth_ppm = (
            self.bandwidth_hz / self.reference_frequency_mhz
            if self.reference_frequency_mhz
            else None
        )
        self.dwell_time_s = 1.0 / self.bandwidth_hz if self.bandwidth_hz else None

        td_raw = self.acqu.get("TD", 0)
        self.n_acquisition = int(td_raw) // 2
        # Transmitter offset O1 (Hz) relative to the 0-ppm frequency BF1 (MHz).
        o1_hz = float(self.acqu.get("O1", 0.0))
        self.center_ppm = (
            o1_hz / self.reference_frequency_mhz
            if self.reference_frequency_mhz
            else 0.0
        )

        self.n_receivers = 1
        self.n_repetitions = 1
        self.n_averages = 1
        self.n_echoes = 1
        self.n_voxels = 1
        self.n_inversion_time = 1
        self.inversion_time_ms = 0
        self.rare_factor = None
        self.scan_duration_s = None
        self.echo_time_ms = None
        self.repetition_time_ms = None
        self.frame_time_s = None
        self.sampling_time_axis_s = (
            np.arange(self.n_acquisition, dtype=float) * self.dwell_time_s
            if self.dwell_time_s is not None and self.n_acquisition > 0
            else None
        )
        self.fat_water_separation = None
        self.b_values = None
        self.echo_spacing_ms = None
        self.effective_te_ms = None

        self.coil_receive = None
        self.coil_transmit = None
        self.rf_pulse_params = None
        self.rfpulse_duration_ms = None
        self.rfpulse_bandwidth_hz = None
        self.rfpulse_is_calculated = None
        self.rfpulse_sharpness = None
        self.rfpulse_bwfac_hzms = None
        self.rfpulse_sint = None
        self.rfpulse_pint = None
        self.rfpulse_type = None
        self.rfpulse_rpfac_percent = None
        self.rfpulse_peakpow_w = None
        self.rfpulse_shape = None
        self.rfpulse_refpow_w = None
        self.flipangle_deg = None
        self.ref_pulse_params = None
        self.refpulse_duration_ms = None
        self.refpulse_bandwidth_hz = None
        self.refpulse_flipangle_deg = None
        self.refpulse_is_calculated = None
        self.refpulse_sharpness = None
        self.refpulse_bwfac_hzms = None
        self.refpulse_sint = None
        self.refpulse_pint = None
        self.refpulse_type = None
        self.refpulse_rpfac_percent = None
        self.refpulse_peakpow_w = None
        self.refpulse_shape = None
        self.refpulse_refpow_w = None
        self.recon_names = []
        self.acquisition_matrix = np.array([self.n_acquisition, 1, 1])

        return self.build_parameter_snapshot()

    def load_data(self, source=None) -> xr.DataArray:
        """Read and store the acquired TopSpin FID using the parsed metadata.

        Parameters
        ----------
        source : object, optional
            Retained compatibility argument; ignored. Reads fid from the path
            already validated and stored by the constructor.

        Returns
        -------
        xarray.DataArray
            Eager complex128 signal in arbitrary units, with one acquisition_idx
            dimension. Interleaved real/imaginary integer32 or float64 samples
            use the declared byte order and are sliced from cut_off (70) to
            n_acquisition (TD/2). The index starts at zero after the cutoff;
            acquisition_time_s uses the declared dwell time in seconds, with
            the historical index fallback when dwell time is absent. Available
            shifted frequency coordinates are in Hz and ppm. Attributes retain
            nucleus, sequence, bandwidth, centre, cutoff, reference frequency
            in MHz and absolute source path.

        Raises
        ------
        FileNotFoundError
            If fid is absent from the stored directory.
        OSError
            If the acquired binary cannot be read.
        ValueError
            If metadata, unequal real/imaginary lengths or prepared coordinate
            lengths are inconsistent with the binary signal.

        Notes
        -----
        Each call rereads the file deterministically and replaces the stored
        scanner_data array; there is no preprocessing of earlier results and
        no modification of acquired bytes or arrays returned by earlier calls.
        The constructor-prepared frequency axes retain their original behavior.
        """
        fid_path = self.path / "fid"
        if not fid_path.exists():
            raise FileNotFoundError(f"FID file not found: {fid_path}")

        dtypa = int(self.acqu.get("DTYPA", 0))
        bytorda = int(self.acqu.get("BYTORDA", 0))
        endian = "<" if bytorda == 0 else ">"

        dtype = np.dtype(f"{endian}f8") if dtypa == 2 else np.dtype(f"{endian}i4")

        raw = np.fromfile(fid_path, dtype=dtype)

        # Interleaved real / imag → complex; apply cut_off to match freq axes
        fid = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
        fid = fid[self.cut_off : self.n_acquisition]

        n_points = len(fid)
        idx = np.arange(n_points)
        time_s = idx * self.dwell_time_s if self.dwell_time_s else idx

        coords = {
            "acquisition_idx": idx,
            "acquisition_time_s": ("acquisition_idx", time_s),
        }
        if self._freq_axis_hz is not None:
            coords["freq_axis_hz"] = ("acquisition_idx", self._freq_axis_hz)
        if self._freq_axis_ppm is not None:
            coords["freq_axis_ppm"] = ("acquisition_idx", self._freq_axis_ppm)

        da = xr.DataArray(
            fid,
            dims=["acquisition_idx"],
            coords=coords,
            attrs={
                "unit": "a.u.",
                "domain": "time",
                "nucleus": self.nucleus,
                "sequence_name": self.sequence_name,
                "bandwidth_hz": self.bandwidth_hz,
                "bandwidth_ppm": self.bandwidth_ppm,
                "center_ppm": self.center_ppm,
                "cut_off": self.cut_off,
                "reference_frequency_mhz": self.reference_frequency_mhz,
                "source_path": str(self.path),
            },
        )

        self.set_primary_data(da, data_type="scanner_data")
        return da

    # ------------------------------------------------------------------
    # Comparison operators (used by Study sort="time")
    # ------------------------------------------------------------------

    def __lt__(self, other: object) -> bool:
        if (
            self.timestamp is None
            or not hasattr(other, "timestamp")
            or other.timestamp is None
        ):
            return False
        return self.timestamp < other.timestamp

    def __gt__(self, other: object) -> bool:
        if (
            self.timestamp is None
            or not hasattr(other, "timestamp")
            or other.timestamp is None
        ):
            return False
        return self.timestamp > other.timestamp

    def __str__(self) -> str:
        reference = (
            f"{self.reference_frequency_mhz:.3f} MHz"
            if self.reference_frequency_mhz is not None
            else "unknown"
        )
        return (
            f"TopSpin [{self.expnum}] {self.sequence_name} ({self.nucleus}) "
            f"BF1={reference}"
        )

    def __repr__(self) -> str:
        return f"BrukerTopSpinHandler(path={self.path})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_param_file(self, file_name: str):
        """Delegate to the shared JCAMP parser, retaining the vendor dialect."""
        return _read_topspin_parameters(self.path / file_name)

    @staticmethod
    def _parse_array(current_file, line):
        """Delegate to the shared JCAMP parser, retaining the vendor dialect."""
        return _parse_topspin_array(current_file, line)

    def _extract_str_param(self, key: str) -> str | None:
        """Return a parameter as a clean string (strip angle brackets)."""
        val = self.acqu.get(key, None)
        if val is None:
            return None
        if isinstance(val, str):
            return val.strip("<>").strip()
        return str(val)

    def _compute_freq_axis(self, unit: str) -> np.ndarray | None:
        """Compute frequency axis using the same logic as ``get_freq_axis``.

        The cut_off is applied: the returned axis has ``n_acquisition - cut_off``
        points, matching what downstream spectroscopy functions expect.
        """
        from hypermri.reconstruction.axes import get_freq_axis

        return get_freq_axis(scan=self, unit=unit)
