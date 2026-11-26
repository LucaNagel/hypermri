"""Magritek Spinsolve / Spinsolve Expert vendor handler."""

from __future__ import annotations

import re
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.mixins import StandardizedMRIMixin
from hypermri.reconstruction.precision import convert_complex_for_storage
from hypermri.utils.system.logging import LOG_MODES, init_default_logger
from hypermri.vendors.base import VendorHandler

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class MagritekHandler(VendorHandler, StandardizedMRIMixin):
    """Handler for Magritek Spinsolve / Spinsolve Expert spectroscopy exports."""

    HEADER_MAGIC = b"SORPATAD1.1V"
    HEADER_SIZE = 32

    def __init__(
        self,
        source: str | Path,
        load_data: bool = False,
        log_mode: str = "error",
    ) -> None:
        super().__init__()

        self.path = source
        if not self.is_valid_scan_dir(self.path):
            raise FileNotFoundError(
                f"Folder {self.path} does not look like a Magritek Spinsolve export"
            )

        logger.setLevel(LOG_MODES.get(log_mode, LOG_MODES["error"]))
        self.log_mode = log_mode

        self.acqu: dict[str, Any] = {}
        self.proc: dict[str, Any] = {}
        self._data_header: dict[str, Any] = {}
        self._freq_axis_ppm = None
        self._freq_axis_hz = None
        self.spectra: np.ndarray | None = None

        self.load_parameters(source)
        if load_data:
            self.load_data(source)

    @property
    def id(self) -> str:
        return str(self.experiment_number)

    @property
    def type(self) -> str:
        return str(self.sequence_name)

    def get(self, attribute_name: str, default=None):
        """Provide dict-like attribute access used by helper utilities."""
        return getattr(self, attribute_name, default)

    def _load_scanner_data(self) -> xr.DataArray | None:
        try:
            return self.load_data(self.path)
        except Exception:
            return None

    def _parse_parameters(self, source) -> dict[str, Any]:
        input_path = source.path if hasattr(source, "path") else source
        self.path = input_path

        self.acqu = self._read_param_file("acqu.par")
        self.proc = self._read_param_file("proc.par")
        self._data_header = self._read_data_header()

        self.spinsolve_experiment_name = str(
            self.acqu.get("expName") or self.path.name or "Magritek experiment"
        )
        self.spinsolve_experiment = self._optional_str(self.acqu.get("experiment"))
        self.spinsolve_spec_type = self._optional_str(self.acqu.get("specType"))
        self.spinsolve_spec_id = self._optional_str(self.acqu.get("specID"))
        self.spinsolve_software_version = self._optional_str(
            self.acqu.get("softwareVersion")
        )
        self.spinsolve_rx_channel = self._optional_str(self.acqu.get("rxChannel"))
        self.spinsolve_processing_script = self._read_processing_script()
        self.spinsolve_pulse_program_name = self._derive_pulse_program_name()

        measurement_time = self._parse_timestamp()
        experiment_number = self._derive_experiment_id()

        n_acquisition = int(
            self.acqu.get("nrPnts", self._data_header.get("n_points", 0)) or 0
        )
        n_repetitions = int(
            self.acqu.get("nrSpectra", self._data_header.get("n_spectra", 1)) or 1
        )
        n_averages = int(self.acqu.get("nrScans", 1) or 1)
        n_receivers = int(self._data_header.get("dimension_1", 1) or 1)

        dwell_time_s = self._derive_dwell_time_s()
        bandwidth_hz = self._derive_bandwidth_hz(dwell_time_s)
        b1_freq_mhz = self._to_float(
            self.acqu.get("b1Freq13C", self.acqu.get("b1Freq"))
        )
        bandwidth_ppm = (
            float(bandwidth_hz) / float(b1_freq_mhz)
            if bandwidth_hz not in (None, 0) and b1_freq_mhz not in (None, 0)
            else None
        )
        repetition_time_ms = self._to_float(self.acqu.get("repTime"))
        frame_time_s = (
            float(repetition_time_ms) / 1000.0
            if repetition_time_ms not in (None, 0)
            else None
        )
        sampling_time_axis_s = (
            np.arange(n_acquisition, dtype=float) * dwell_time_s
            if dwell_time_s is not None and n_acquisition > 0
            else None
        )

        self.vendor = "magritek"
        self.sequence_name = self._derive_sequence_name()
        self.experiment_number = experiment_number
        self.timestamp = measurement_time
        self.measurement_time = measurement_time
        self.nucleus = str(self.acqu.get("nucleus", "unknown"))
        # b1Freq is the spectrometer frequency in MHz that also sets the Hz/ppm
        # slope of bandwidth_ppm above.
        self.reference_frequency_mhz = b1_freq_mhz or None
        self.b0_tesla = self._derive_b0_tesla()
        self.method_type = "stock"
        self.is_spectroscopic = True
        self.cut_off = 0
        self.git_commit_hash = None
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
        self.center_ppm = self._to_float(
            self.acqu.get("centerFreqPPM", self.proc.get("ppmOffset", 0.0))
        )
        self.n_receivers = n_receivers
        self.n_repetitions = n_repetitions
        self.n_averages = n_averages
        self.n_echoes = 1
        self.n_acquisition = n_acquisition
        self.n_voxels = 1
        self.n_inversion_time = 1
        self.inversion_time_ms = 0
        self.rare_factor = None
        self.scan_duration_s = self._to_float(self.acqu.get("duration"))
        self.bandwidth_hz = bandwidth_hz
        self.bandwidth_ppm = bandwidth_ppm
        self.echo_time_ms = None
        self.repetition_time_ms = repetition_time_ms
        self.frame_time_s = frame_time_s
        self.dwell_time_s = dwell_time_s
        self.sampling_time_axis_s = sampling_time_axis_s
        self.fat_water_separation = None
        self.b_values = None
        self.echo_spacing_ms = None
        self.effective_te_ms = None
        self.coil_receive = self.spinsolve_rx_channel
        self.coil_transmit = None
        self.rf_pulse_params = None
        self.rfpulse_duration_ms = self._derive_pulse_duration_ms()
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
        self.flipangle_deg = self._to_float(self.acqu.get("tipAngle"))
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
        self.acquisition_matrix = np.array([n_acquisition, 1, 1])

        return self.build_parameter_snapshot()

    def load_data(self, source=None) -> xr.DataArray:  # type: ignore[override]
        """Read Spinsolve temporal samples and store the labelled scanner data.

        Parameters
        ----------
        source : object, optional
            Legacy argument accepted for handler compatibility. The reader
            always uses ``data.2d`` in this handler's experiment directory.

        Returns
        -------
        xarray.DataArray
            Complex64 amplitudes in arbitrary scanner units. Dimensions are
            repetition/acquisition, or repetition/channel/average/acquisition
            when either auxiliary header dimension is non-singleton. Available
            sampling and repetition times are in seconds; frequency coordinates
            are in Hz and ppm. Source and protocol attributes are retained.

        Raises
        ------
        OSError
            If the payload cannot be read.
        ValueError
            If the header, payload size or coordinate lengths are inconsistent.

        Notes
        -----
        Each call eagerly reads the little-endian float32 interleaved payload.
        It stores the returned array as scanner data and shares its complex
        sample storage with ``fid``. No spectral reconstruction is performed.
        """
        data_path = self.path / "data.2d"
        header = self._data_header or self._read_data_header()
        n_points = int(header["n_points"])
        n_spectra = int(header["n_spectra"])
        dim_1 = int(header["dimension_1"])
        dim_2 = int(header["dimension_2"])

        raw = np.fromfile(data_path, dtype="<f4", offset=self.HEADER_SIZE)
        expected_float_count = n_points * n_spectra * dim_1 * dim_2 * 2
        if raw.size != expected_float_count:
            raise ValueError(
                f"Unexpected Magritek payload size in {data_path}: "
                f"expected {expected_float_count} float32 values, found {raw.size}"
            )

        complex_data = raw[0::2] + 1j * raw[1::2]
        fid = complex_data.reshape(n_spectra, dim_1, dim_2, n_points)

        if dim_1 == 1 and dim_2 == 1:
            data = fid[:, 0, 0, :]
            dims = ["repetition_idx", "acquisition_idx"]
            coords: dict[str, Any] = {
                "repetition_idx": np.arange(n_spectra),
                "acquisition_idx": np.arange(n_points),
            }
        else:
            data = fid
            dims = ["repetition_idx", "channel_idx", "average_idx", "acquisition_idx"]
            coords = {
                "repetition_idx": np.arange(n_spectra),
                "channel_idx": np.arange(dim_1),
                "average_idx": np.arange(dim_2),
                "acquisition_idx": np.arange(n_points),
            }

        if self.frame_time_s is not None:
            coords["time"] = (
                "repetition_idx",
                np.arange(n_spectra) * self.frame_time_s,
            )
        if self.sampling_time_axis_s is not None:
            coords["sampling_time_axis_s"] = (
                "acquisition_idx",
                np.asarray(self.sampling_time_axis_s),
            )

        data_array = xr.DataArray(
            convert_complex_for_storage(data),
            dims=dims,
            coords=coords,
        )
        data_array.attrs.update(
            {
                "unit": "a.u.",
                "domain": "time",
                "signal_domain": "temporal",
                "vendor": self.vendor,
                "sequence_name": self.sequence_name,
                "source_path": str(self.path),
                "spinsolve_experiment_name": self.spinsolve_experiment_name,
                "spinsolve_experiment": self.spinsolve_experiment,
                "spinsolve_spec_type": self.spinsolve_spec_type,
                "spinsolve_spec_id": self.spinsolve_spec_id,
                "spinsolve_software_version": self.spinsolve_software_version,
            }
        )
        data_array = assign_frequency_axis(data_array, self.freq_axis_hz, "Hz")
        data_array = assign_frequency_axis(data_array, self.freq_axis_ppm, "ppm")

        self.fid = data_array.data
        self.set_primary_data(data_array, data_type="scanner_data")
        return data_array

    @staticmethod
    def is_valid_scan_dir(path: str | Path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False

        required = ("acqu.par", "proc.par", "data.2d")
        if not all((path / name).exists() for name in required):
            return False

        try:
            with (path / "data.2d").open("rb") as handle:
                magic = handle.read(len(MagritekHandler.HEADER_MAGIC))
            return magic == MagritekHandler.HEADER_MAGIC
        except OSError:
            return False

    def _read_param_file(self, filename: str) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        file_path = self.path / filename
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                parameters[key.strip()] = self._parse_value(raw_value.strip())
        return parameters

    def _read_data_header(self) -> dict[str, Any]:
        data_path = self.path / "data.2d"
        with data_path.open("rb") as handle:
            header = handle.read(self.HEADER_SIZE)

        if len(header) != self.HEADER_SIZE:
            raise ValueError(f"Incomplete Magritek header in {data_path}")

        magic = header[: len(self.HEADER_MAGIC)]
        if magic != self.HEADER_MAGIC:
            raise ValueError(
                f"Unexpected Magritek header magic in {data_path}: {magic!r}"
            )

        version, n_points, n_spectra, dim_1, dim_2 = struct.unpack(
            "<5I", header[len(self.HEADER_MAGIC) : self.HEADER_SIZE]
        )
        return {
            "magic": magic.decode("ascii", errors="ignore"),
            "version": int(version),
            "n_points": int(n_points),
            "n_spectra": int(n_spectra),
            "dimension_1": int(dim_1),
            "dimension_2": int(dim_2),
        }

    def _derive_experiment_id(self) -> str:
        timestamp = self._parse_timestamp()
        if timestamp is not None:
            return timestamp.strftime("%Y%m%d-%H%M%S")
        return self.path.name

    def _parse_timestamp(self) -> datetime | None:
        match = re.match(r"(?P<stamp>\d{6}-\d{6})", self.path.name)
        if not match:
            return None
        try:
            return datetime.strptime(match.group("stamp"), "%y%m%d-%H%M%S")
        except ValueError:
            return None

    def _derive_dwell_time_s(self) -> float | None:
        dwell_time_us = self._to_float(self.acqu.get("dwellTime"))
        if dwell_time_us is None:
            return None
        return dwell_time_us * 1e-6

    def _derive_bandwidth_hz(self, dwell_time_s: float | None) -> float | None:
        if dwell_time_s not in (None, 0):
            return 1.0 / float(dwell_time_s)

        bandwidth_khz = self._to_float(self.acqu.get("bandwidth"))
        if bandwidth_khz is None:
            return None
        return bandwidth_khz * 1000.0

    def _derive_pulse_duration_ms(self) -> float | None:
        pulse_length_us = self._to_float(
            self.acqu.get("pulseLength13C", self.acqu.get("pulseLength"))
        )
        if pulse_length_us is None:
            return None
        return pulse_length_us / 1000.0

    def _read_processing_script(self) -> str | None:
        script_path = self.path / "processing.script"
        if not script_path.exists():
            return None
        try:
            return script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

    def _derive_pulse_program_name(self) -> str | None:
        pp_dir = self.path / "ppCode"
        if not pp_dir.exists() or not pp_dir.is_dir():
            return None

        names = []
        for file_path in pp_dir.iterdir():
            if not file_path.is_file() or file_path.suffix not in {".mac", ".par"}:
                continue
            stem = file_path.stem
            for suffix in ("Default", "_interface", "_pp"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            if stem:
                names.append(stem)

        unique_names = sorted(set(names))
        if len(unique_names) == 1:
            return unique_names[0]
        if unique_names:
            return unique_names[0]
        return None

    def _derive_sequence_name(self) -> str:
        if self.spinsolve_pulse_program_name:
            return self.spinsolve_pulse_program_name

        exp_name = self.spinsolve_experiment_name or ""
        match = re.match(
            r"^\d{6}-\d{6}\s+(?P<name>.+?)(?:\s+\([^)]*\))?$",
            exp_name,
        )
        if match:
            cleaned = match.group("name").strip()
            if cleaned:
                return cleaned

        if self.spinsolve_spec_type:
            return self.spinsolve_spec_type
        if self.spinsolve_experiment:
            return self.spinsolve_experiment
        return "spinsolve_expert"

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_value(raw_value: str) -> Any:
        value = raw_value.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            return value[1:-1]

        lowered = value.lower()
        if lowered in {"yes", "true"}:
            return True
        if lowered in {"no", "false"}:
            return False

        try:
            if any(char in value for char in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value

    def __str__(self) -> str:
        label = self.spinsolve_experiment_name or self.sequence_name
        return f"Magritek experiment {self.experiment_number} ({label})"

    def __lt__(self, other: Any) -> bool:
        other_time = getattr(other, "timestamp", None)
        other_path = getattr(other, "path", "")
        if self.timestamp is None or other_time is None:
            return str(self.path) < str(other_path)
        return self.timestamp < other_time

    def __gt__(self, other: Any) -> bool:
        other_time = getattr(other, "timestamp", None)
        other_path = getattr(other, "path", "")
        if self.timestamp is None or other_time is None:
            return str(self.path) > str(other_path)
        return self.timestamp > other_time
