"""Agilent/Varian vendor handler.

This module provides a minimal spectroscopy-focused handler for Agilent data
using ``nmrglue``. The implementation targets single-voxel spectroscopy data
and populates the standardized attributes expected by HyperMRI reconstruction
strategies.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from nmrglue import agilent as ng_agilent

from hypermri.reconstruction.precision import convert_complex_for_storage

try:
    from nmrglue.fileio import varian as ng_varian
except ImportError:  # pragma: no cover - optional dependency alias
    ng_varian = None

from hypermri.mixins import StandardizedMRIMixin
from hypermri.utils.system.logging import LOG_MODES, debug_log, init_default_logger
from hypermri.vendors.base import VendorHandler

DEBUG_FILE = "/tmp/agilentexp.log"
DEBUG_LOG = False


def debug_log_agilentexp(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


def _coerce_value(value: Any) -> Any:
    """Best-effort conversion from procpar value representation to Python."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")

    if isinstance(value, str):
        # Remove surrounding quotes used in procpar for strings
        stripped = value.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            return stripped[1:-1]
        return stripped

    return value


class AgilentHandler(VendorHandler, StandardizedMRIMixin):
    """Handler for Agilent/Varian spectroscopy datasets."""

    is_spectroscopic = True

    def __init__(
        self,
        source: str | Path,
        load_data: bool = False,
        log_mode: str = "error",
    ) -> None:
        debug_log_agilentexp(
            f"AgilentHandler.__init__ starting with source={source}, load_data={load_data}, log_mode={log_mode}"
        )
        super().__init__()

        self.path = Path(source)
        if not self.is_valid_scan_dir(self.path):
            raise FileNotFoundError(
                f"Folder {self.path} does not look like an Agilent .fid directory"
            )
        debug_log_agilentexp(
            f"AgilentHandler: Valid scan directory confirmed: {self.path}"
        )

        logger.setLevel(LOG_MODES.get(log_mode, LOG_MODES["error"]))
        self.log_mode = log_mode

        self._procpar: dict[str, dict[str, Any]] = {}
        self._fid_header: dict[str, Any] = {}
        self._parameters: dict[str, Any] = {}

        self.sequence_variant: str | None = None
        # Initialize sequence_name - this will be set by _parse_parameters()
        self.sequence_name: str | None = None

        self.fid: np.ndarray | None = None
        self.dwell_time_s: float | None = None
        self.sampling_time_axis_s: np.ndarray | None = None
        # Note: freq_axis_hz and freq_axis_ppm are implemented as @property methods

        self.digital_filter_points: int = 0
        self.zero_fill_points: int | None = None
        self.line_broadening_hz: float = 0.0
        self.gaussian_broadening_hz: float = 0.0
        self.phase0_deg: float = 0.0
        self.phase1_deg: float = 0.0
        self.reference_frequency_mhz: float | None = None

        self.n_receivers: int = 1
        self.n_repetitions: int = 1
        self.n_acquisition: int = 0
        self.n_averages: int = 1
        self.matrix = np.array([1, 1, 1])
        self.fov_mm = np.array([1.0, 1.0, 1.0])
        self.spatial_dimension = "spectroscopy"

        # Populate standardized parameter mapping
        debug_log_agilentexp("AgilentHandler: Loading parameters...")
        self.load_parameters(source)
        self._parameters = {**getattr(self, "_parameters", {}), **self.params}
        debug_log_agilentexp("AgilentHandler: Parameters loaded successfully")

        if load_data:
            debug_log_agilentexp("AgilentHandler: Loading data...")
            self.load_data(source)
            debug_log_agilentexp("AgilentHandler: Data loaded successfully")

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.experiment_id

    @property
    def type(self) -> str:
        return self.sequence_name

    # ----------------------------------------------------------------------
    def get_parameters(self) -> dict[str, Any]:
        """Return cached parameter snapshot."""
        return super().get_parameters()

    def _parse_parameters(self, source) -> dict[str, Any]:
        """Translate Agilent metadata into the standardized schema."""
        if hasattr(source, "path"):
            input_path = source.path
        else:
            input_path = source

        self.path = Path(input_path)

        # Ensure procpar metadata is loaded before deriving any fields
        self._load_procpar()

        # Vendor/core identifiers
        self.vendor = "agilent"
        self.experiment_number = self.experiment_id
        self.method_type = "stock"
        self.is_spectroscopic = True
        self.cut_off = 0
        self.git_commit_hash = None
        self.timestamp = None

        # Geometry defaults for single-voxel spectroscopy
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
        self.fov_mm = np.asarray(self.fov_mm)
        self.matrix = np.asarray(self.matrix)
        self.encoding_matrix = np.asarray(self.matrix)
        self.center_ppm = self._derive_center_ppm()
        self.measurement_time = self._get_measurement_time()
        # Acquisition parameters pulled from procpar metadata
        if self.bandwidth_hz not in (None, 0):
            with np.errstate(divide="ignore"):
                self.dwell_time_s = 1.0 / float(self.bandwidth_hz)
        else:
            self.dwell_time_s = None

        self.sampling_time_axis_s = None
        if self.bandwidth_hz not in (
            None,
            0,
        ) and self.spectrometer_frequency_mhz not in (None, 0):
            self.bandwidth_ppm = float(self.bandwidth_hz) / float(
                self.spectrometer_frequency_mhz
            )
        else:
            self.bandwidth_ppm = None

        self.scan_duration_s = self.acquisition_duration_s
        self.n_voxels = 1
        self.n_inversion_time = 1
        self.inversion_time_ms = 0
        self.rare_factor = None

        # Reconstruction metadata
        self.recon_names = []
        self.acquisition_matrix = np.asarray(self.matrix)

        return self.build_parameter_snapshot()

    def load_data(self, source: str | None = None) -> None:  # type: ignore[override]
        """Read Agilent FIDs and update acquired-sample metadata eagerly.

        Parameters
        ----------
        source : str, optional
            Agilent experiment directory; defaults to this handler's path.

        Returns
        -------
        None
            Samples are stored in ``fid`` as complex64 in repetition/channel/
            acquisition order, with amplitudes in arbitrary scanner units.

        Raises
        ------
        RuntimeError
            If the Agilent reader cannot read the experiment.
        ValueError
            If the reader returns a scalar instead of an FID array.

        Notes
        -----
        Each call reads the source again and replaces the stored FID and header.
        Sample counts, dwell time in seconds, sampling times in seconds and
        bandwidth metadata are refreshed in the handler's parameter snapshot.
        The reader's array is not mutated and no reconstruction is performed.
        """
        if source is None:
            source = str(self.path)
        else:
            pass
        try:
            header, data = ng_agilent.read(source)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read Agilent fid from {self.path}: {exc}"
            ) from exc

        self._fid_header = header
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[np.newaxis, np.newaxis, :]
        elif data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim >= 3:
            # Keep first three dimensions as repetitions/channels/points
            data = data.reshape(data.shape[0], data.shape[1], -1)
        else:
            raise ValueError("Unexpected dimensionality for Agilent FID array")

        self.fid = convert_complex_for_storage(data)
        self.n_repetitions = int(self.fid.shape[0])
        self.n_receivers = int(self.fid.shape[1])
        self.n_acquisition = int(self.fid.shape[2])

        # Update sampling characteristics
        if self.bandwidth_hz:
            self.dwell_time_s = 1.0 / self.bandwidth_hz
        elif self.acquisition_duration_s and self.n_acquisition:
            self.dwell_time_s = self.acquisition_duration_s / self.n_acquisition
        else:
            self.dwell_time_s = None

        if self.dwell_time_s is not None and self.n_acquisition:
            from hypermri.utils.time.temporal import calc_sampling_time_axis

            self.sampling_time_axis_s = calc_sampling_time_axis(
                dwell_time_s=self.dwell_time_s, npoints=self.n_acquisition
            )
            debug_log_agilentexp(
                message=f"self.sampling_time_axis_s: {self.sampling_time_axis_s}"
            )
        else:
            self.sampling_time_axis_s = np.arange(self.n_acquisition)

        self._parameters.update(
            {
                "n_repetitions": self.n_repetitions,
                "n_channels": self.n_receivers,
                "n_points": self.n_acquisition,
            }
        )

        bandwidth_ppm = self.params.get("bandwidth_ppm")
        if (
            bandwidth_ppm is None
            and self.bandwidth_hz not in (None, 0)
            and self.spectrometer_frequency_mhz not in (None, 0)
        ):
            bandwidth_ppm = float(self.bandwidth_hz) / float(
                self.spectrometer_frequency_mhz
            )

        self.params.update(
            {
                "n_repetitions": self.n_repetitions,
                "n_receivers": self.n_receivers,
                "n_acquisition": self.n_acquisition,
                "dwell_time_s": self.dwell_time_s,
                "sampling_time_axis_s": self.sampling_time_axis_s,
                "bandwidth_hz": self.bandwidth_hz,
                "bandwidth_ppm": bandwidth_ppm,
            }
        )
        self._parameters.update(self.params)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _load_procpar(self) -> None:
        procpar_path = self.path / "procpar"
        read_procpar = getattr(ng_agilent, "read_procpar", None)
        if not read_procpar and ng_varian is not None:
            read_procpar = getattr(ng_varian, "read_procpar", None)

        try:
            if read_procpar is not None:
                self._procpar = read_procpar(str(procpar_path))
            else:
                header, _ = ng_agilent.read(str(self.path))
                self._procpar = header.get("procpar", {})
        except Exception as exc:
            logger.warning(f"Failed to read procpar in {self.path}: {exc}")
            self._procpar = {}

        self.experiment_id = self._get_procpar_value(
            "experiment", default=self.path.name
        )
        raw_sequence = self._get_procpar_value("seqfil", default=None)
        if raw_sequence:
            self.sequence_variant = str(raw_sequence).strip().lower()
        else:
            self.sequence_variant = None

        # Normalise to a canonical spectroscopy identifier understood by the
        # reconstruction table while keeping the raw sequence variant for
        # metadata/inspection purposes.
        self.sequence_name = "nspect"
        self.acquisition_time = self._get_procpar_value("time_run", default="")

        self.bandwidth_hz = self._get_procpar_value("sw", default=None, as_float=True)
        self.bandwidth_ppm = self._get_procpar_value(
            "swppm", default=None, as_float=True
        )
        self.acquisition_duration_s = self._get_procpar_value(
            "at", default=None, as_float=True
        )
        self.spectrometer_frequency_mhz = self._get_procpar_value(
            "sfrq", default=None, as_float=True
        )
        self.nucleus = self._get_procpar_value("tn", default="unknown")
        self.coil_receive = self._get_procpar_value("rfcoil", default=None)

        self.reference_frequency_mhz = self._get_procpar_value(
            "reffrq", default=None, as_float=True
        )
        self.b0_tesla = self._derive_b0_tesla()
        self.digital_filter_points = int(
            self._get_procpar_value("lsfid", default=0, as_float=True) or 0
        )
        self.zero_fill_points = (
            int(self._get_procpar_value("zf", default=0, as_float=True) or 0) or None
        )
        self.line_broadening_hz = float(
            self._get_procpar_value("lb", default=0.0, as_float=True) or 0.0
        )
        self.gaussian_broadening_hz = float(
            self._get_procpar_value("gb", default=0.0, as_float=True) or 0.0
        )
        self.phase0_deg = float(
            self._get_procpar_value("rp", default=0.0, as_float=True) or 0.0
        )
        self.phase1_deg = float(
            self._get_procpar_value("lp", default=0.0, as_float=True) or 0.0
        )
        self.repetition_time_ms = self._get_procpar_value(
            "tr", default=None, as_float=True
        )
        self.object_position = self._get_procpar_value("position1", default=None)
        arraydim = self._get_procpar_value("arraydim", default=None, as_float=True)
        if arraydim:
            self.n_repetitions = int(arraydim)

        self.n_averages = int(
            self._get_procpar_value("averages", default=1, as_float=True) or 1
        )
        self.slice_order = int(
            self._get_procpar_value("sliceorder", default=1, as_float=False) or 0
        )
        self.subject_weight = self._get_procpar_value(
            "weight", default=None, as_float=True
        )
        self.study_id = self._get_procpar_value(
            "studyid_", default=None, as_float=False
        )

        np_points = self._get_procpar_value("np", default=None, as_float=True)
        if np_points:
            try:
                self.n_acquisition = int(np_points)
            except (TypeError, ValueError):
                self.n_acquisition = 0

        self._parameters = {
            "vendor": "agilent",
            "sequence_name": self.sequence_name,
            "sequence_variant": self.sequence_variant,
            "experiment_id": self.experiment_id,
            "acquisition_time": self.acquisition_time,
            "bandwidth_hz": self.bandwidth_hz,
            "acquisition_duration_s": self.acquisition_duration_s,
            "b0_tesla": self.b0_tesla,
            "spectrometer_frequency_mhz": self.spectrometer_frequency_mhz,
            "nucleus": self.nucleus,
            "n_repetitions": self.n_repetitions,
            "n_averages": self.n_averages,
            "digital_filter_points": self.digital_filter_points,
            "zero_fill_points": self.zero_fill_points,
            "line_broadening_hz": self.line_broadening_hz,
            "gaussian_broadening_hz": self.gaussian_broadening_hz,
            "phase0_deg": self.phase0_deg,
            "phase1_deg": self.phase1_deg,
            "reference_frequency_mhz": self.reference_frequency_mhz,
        }

    def _get_procpar_value(
        self,
        key: str,
        default: Any | None = None,
        *,
        as_float: bool = False,
    ) -> Any | None:
        entry = self._procpar.get(key)
        if not entry:
            return default
        values = entry.get("values", [])
        if not values:
            return default
        value = _coerce_value(values[0])
        if value in (None, ""):
            return default
        if as_float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        return value

    def _derive_center_ppm(self) -> float:
        ppmref = self._get_procpar_value("ppmref", default=None, as_float=True)
        if ppmref is not None:
            return float(ppmref)

        reference_ppm = self._get_procpar_value("refppm", default=None, as_float=True)
        if reference_ppm is not None:
            return float(reference_ppm)

        if self.reference_frequency_mhz not in (
            None,
            0,
        ) and self.spectrometer_frequency_mhz not in (None, 0):
            try:
                delta_ppm = (
                    (self.reference_frequency_mhz - self.spectrometer_frequency_mhz)
                    / self.spectrometer_frequency_mhz
                ) * 1e6
                return float(delta_ppm)
            except ZeroDivisionError:
                pass

        # Default to 0 ppm when no reference information is available
        return 0.0

    def _get_measurement_time(self) -> datetime | None:
        try:
            date_string = self._get_procpar_value(
                "time_submitted_local", default=None, as_float=False
            )

            # Extract the clear date and time parts
            date_part = date_string[:8]  # "20241021"
            time_part = date_string[9:15]  # "145857"

            # Combine and parse
            datetime_str = date_part + time_part
            return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def is_valid_scan_dir(path: str | Path) -> bool:
        """Check if directory provided is a valid agilent scan directory."""
        path = Path(path)
        if not path.is_dir():
            return False
        required = ["fid", "procpar"]
        return all((path / req).exists() for req in required)

    def to_json(self) -> str:
        """Serialize key parameters for debugging purposes."""
        payload = {
            "path": str(self.path),
            "parameters": self._parameters,
            "n_points": self.n_acquisition,
            "n_repetitions": self.n_repetitions,
        }
        return json.dumps(payload, indent=2)

    def __str__(self) -> str:
        return f"Agilent experiment {self.experiment_id} ({self.sequence_name})"

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, AgilentHandler):
            return NotImplemented

        # Get acquisition times, defaulting to empty string if None
        self_time = self.acquisition_time or ""
        other_time = other.acquisition_time or ""

        # Ensure both are strings for comparison
        if not isinstance(self_time, str):
            self_time = str(self_time)
        if not isinstance(other_time, str):
            other_time = str(other_time)

        return self_time < other_time

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, AgilentHandler):
            return NotImplemented

        # Get acquisition times, defaulting to empty string if None
        self_time = self.acquisition_time or ""
        other_time = other.acquisition_time or ""

        # Ensure both are strings for comparison
        if not isinstance(self_time, str):
            self_time = str(self_time)
        if not isinstance(other_time, str):
            other_time = str(other_time)

        return self_time > other_time

    def get(self, attribute_name: str, default=None):
        """
        Gets an attribute value, returning a default if the attribute does not exist. This mimics dictionary .get() behavior.

        Parameters
        ----------
        attribute_name: str
            The name of the attribute to retrieve (e.g., 'name').
        default: str
            The value to return if the attribute is not found.

        Returns
        -------
            The attribute's value or the default.

        """
        # getattr is a built-in Python function that gets an attribute from an
        # object, with an optional default value.
        return getattr(self, attribute_name, default)
