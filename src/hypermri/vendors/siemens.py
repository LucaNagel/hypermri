"""
Base class used to read in one Siemens experiment.
"""

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.mixins import StandardizedMRIMixin
from hypermri.reconstruction.precision import convert_for_storage
from hypermri.utils.system.environment import get_git_commit_hash
from hypermri.utils.system.logging import LOG_MODES, init_default_logger
from hypermri.vendors.base import VendorHandler

# for paper_summary() method only


# Set a global variable to control debug message visibility
DB_MSG = True  # Change this to False to hide detailed error messages


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


_MISSING = object()


def _import_twixtools():
    from hypermri._optional import _import_optional

    return _import_optional("twixtools", "siemens")


class _TwixtoolsHeaderSection(Mapping):
    """Dict adapter that supports legacy tuple lookups on twixtools headers."""

    def __init__(self, data: Mapping | None):
        self._data = data or {}

    def __getitem__(self, key):
        value = self._resolve(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        value = self._resolve(key, _MISSING)
        return default if value is _MISSING else value

    def _resolve(self, key, default):
        if key in self._data:
            return self._data[key]

        if isinstance(key, tuple):
            if len(key) == 1 and key[0] in self._data:
                return self._data[key[0]]
            path = key
        else:
            path = (key,)

        value: Any = self._data
        for part in path:
            if isinstance(value, Mapping):
                if part in value:
                    value = value[part]
                    continue
                part_str = str(part)
                if part_str in value:
                    value = value[part_str]
                    continue
                return default

            if isinstance(value, (list, tuple)):
                try:
                    index = int(part)
                except (TypeError, ValueError):
                    return default
                if index < 0 or index >= len(value):
                    return default
                value = value[index]
                continue

            return default

        return value


class SiemensHandler(VendorHandler, StandardizedMRIMixin):
    """Base class for experiments conducted on Siemens systems."""

    @property
    def id(self) -> str:
        return self.expnum

    def get(self, attribute_name: str, default=None):
        """Dict-style attribute access used by shared spectroscopy helpers."""
        return getattr(self, attribute_name, default)

    def __init__(self, source, load_data=True, log_mode="warning"):
        """
        Initialize the siemensexp class.

        Parameters
        ----------
        source : str or SiemensExp-object
            The complete path to measurement/experiment folder
            In case 'source' is a SiemensExp object the path is automatically
            extracted form the object. That makes reloading a SiemensExp easier.

        Raises
        ------
        FileNotFoundError
            If path provided does not lead to a directory containing a .dat file.

        """
        super().__init__()
        logger.setLevel(LOG_MODES[log_mode])

        input_path = source.path if isinstance(source, SiemensHandler) else source

        dat_file_path = self.__find_dat_file(input_path)

        if dat_file_path:
            self.path = Path(dat_file_path)
            self._set_header_from_twixtools()

            # Initialize properties to None or default values
            self._rps_axes_mm = None
            self._rps_extent_mm = None
            self._rps_grid_mm = None
            self._xyz_axes_mm = None
            self._xyz_extent_mm = None
            self._xyz_grid_mm = None
            self.scanner_source = "Siemens"

            # ---------- Scan parameter files: ----------
            # expnum: Derived from MeasUID, or a placeholder
            self.expnum = (
                int(self.visu.get("MeasUid", 0)) if self.visu.get("MeasUid", 0) else -1
            )
            logger.debug("Siemens experiment number set to %s", self.expnum)

            # define the measurement type (placeholder for now)
            self.type = self.method.get(("tSequenceFileName",), None)
            if self.type is not None:
                self.type = self.type.strip('"')
                if "Siemens" in self.type:  # Placeholder logic
                    self.method_type = "stock"
                else:
                    self.method_type = "custom"
            else:
                self.method_type = "unknown"

            # define Nucleus:
            self.nucleus = self.method.get(
                ("sTXSPEC", "asNucleusInfo", "0", "tNucleus"), "1H"
            ).strip('"')

            self.is_spectroscopic = (
                self._is_enabled(self.method.get(("ucCSI",), None))
                or self._is_enabled(self.method.get(("ucSVSSeq",), None))
                or self._is_spectroscopy_sequence_name(self.type)
            )

            # Number of slice packs:
            self.n_slice_packs = self.method.get(("sSliceArray", "lSize"), 1)

            self.cut_off = 0

            # the time the SiemensExp object was last edited on the scanner
            self.timestamp = datetime.fromtimestamp(self.acqp.get("tTimeStamp", 0))

            # ---------- Scan subject ----------
            self.subject_name = self.visu.get("tPatientName", None)
            if self.subject_name is not None:
                self.subject_name = self.subject_name.strip('"')
            self.subject_id = self.visu.get("PatientID", None)
            if self.subject_id is not None:
                self.subject_id = self.subject_id.strip('"')
            self.subject_position = self.visu.get("tPatientPosition", None)
            if self.subject_position is not None:
                self.subject_position = self.subject_position.strip('"')
            self.subject_type = None  # Not directly available in provided headers
            self.patient_position = self.subject_position

            # ---------- Scan Geometry ----------
            self.spatial_dimension = self.method.get(("sKSpace", "ucDimension"), None)
            if self.spatial_dimension == 2:
                self.spatial_dimension = "2D"
            elif self.spatial_dimension == 3:
                self.spatial_dimension = "3D"
            else:
                self.spatial_dimension = None

            # list of names of all recon folders inside 'pdata' (not applicable
            # for Siemens)
            self.recon_names = []
            # add number of receiver channels
            self.n_receivers = self.method.get(
                ("sProtConsistencyInfo", "lMaximumNofRxReceiverChannels"), 1
            )

            # add number of slices
            self.n_slices = self.method.get(("sSliceArray", "lSize"), 1)
            # slice thickess:
            self.slice_thickness_mm = self.method.get(
                ("sSliceArray", "asSlice", "0", "dThickness"), 0
            )
            # gap between slices:
            self.slice_gap_mm = 0  # Placeholder, couldn't find a clear equivalent
            self.acquisition_matrix = [
                self.visu.get("iNoOfFourierColumns", 0),
                self.visu.get("iNoOfFourierLines", 0),
                self.visu.get("iNoOfFourierPartitions", 0),
            ]
            self.patient_pos = self.visu.get("tPatientPosition", "HFS").strip('"')
            self.read_orientation = None  # Not directly available
            self.slice_orientation = None  # Not directly available
            self.slice_order = np.atleast_1d(0)  # Placeholder

            self.fov_mm = [
                self.method.get(("sSliceArray", "asSlice", "0", "dReadoutFOV"), 0),
                self.method.get(("sSliceArray", "asSlice", "0", "dPhaseFOV"), 0),
            ]
            # For PRESS, CSI etc.
            if self.is_spectroscopic and self.fov_mm[0] == 0:
                self.fov_mm = [
                    self.method.get(("sSpecPara", "sVoI", "dReadoutFOV"), 0),
                    self.method.get(("sSpecPara", "sVoI", "dPhaseFOV"), 0),
                    self.method.get(("sSpecPara", "sVoI", "dThickness"), 0),
                ]
            # add matrix:
            self.matrix = [
                self.method.get(("sKSpace", "lBaseResolution"), 1),
                self.method.get(("sKSpace", "lPhaseEncodingLines"), 1),
                self.n_slices,
            ]
            if self.is_spectroscopic:
                self.matrix = np.array([1, 1, 1])

            # add encoding matrix (actual acquired points, can be affected by
            # Partial Fourier e.g.):
            self.enc_matrix = [
                self.method.get(("sKSpace", "lBaseResolution"), 1),
                self.method.get(("sKSpace", "lPhaseEncodingLines"), 1),
                self.method.get(("sKSpace", "lPartitions"), 1),
            ]
            self.encoding_matrix = self.enc_matrix

            # add number of repetitions:
            self.n_repetitions = self.method.get(("lRepetitions",), 1)
            # add number of averages:
            self.n_averages = self.method.get(("lAverages",), 1)
            # add number of echoes:
            self.n_echoes = self.method.get(("lContrasts",), 1)
            # add number of spectral points:
            self.n_acquisition = self.method.get(("sSpecPara", "lVectorSize"), 1)
            logger.debug("Siemens acquisition points set to %s", self.n_acquisition)
            # add number of variable TRs:
            self.inversion_time_ms = self.method.get(("alTI", "0"), 0)
            if isinstance(
                self.inversion_time_ms, str
            ):  # Handle cases where it's a string of values
                self.inversion_time_ms = np.array(
                    [float(x) for x in self.inversion_time_ms.split()]
                )
            else:
                self.inversion_time_ms = np.atleast_1d(self.inversion_time_ms)

            self.n_inversion_time = (
                len(self.inversion_time_ms) if self.inversion_time_ms is not None else 1
            )

            # add resolution:
            if self.fov_mm is None or self.matrix is None or 0 in self.matrix:
                self.resolution_mm = None
            else:
                try:
                    self.resolution_mm = np.array(self.fov_mm) / np.array(self.matrix)
                except (TypeError, ValueError):
                    # Non-numeric or differently sized FOV and matrix.
                    logger.warning(
                        "Scan %s: cannot divide fov_mm %r by matrix %r; "
                        "resolution_mm is None.",
                        self.path,
                        self.fov_mm,
                        self.matrix,
                    )
                    self.resolution_mm = None

            self.read_offset_mm = None
            self.phase_offset_mm = None
            self.phase_0_offset_mm = None
            self.phase_1_offset_mm = None
            self.slice_offset_mm = None
            self.slice_offset_mm_array = None

            # ---------- Scan RF Coil ----------
            self.coil_receive = self.visu.get(
                "tTransmittingCoil", None
            )  # Placeholder, often combined in Siemens
            self.coil_transmit = self.visu.get(
                "tTransmittingCoil", None
            )  # Placeholder, often combined in Siemens

            # ---------- More Sequence parameters ----------
            position_matrix = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )  # Default identity
            # Add logic to adjust position_matrix based on self.patient_pos if
            # needed
            self.position_matrix = position_matrix

            self.b_values = self.method.get(("sDiffusion", "alBValue", "0"), 0)
            if isinstance(self.b_values, str):
                self.b_values = np.array([float(x) for x in self.b_values.split()])
            else:
                self.b_values = np.atleast_1d(self.b_values)

            self.n_voxels = self.method.get(
                ("sSpecPara", "lVectorSize"), 1
            )  # Using vector size as a proxy
            self.rare_factor = self.method.get(("sFastImaging", "lTurboFactor"), 1)

            self.scan_duration_s = self.method.get(("lTotalScanTimeSec",), 0)

            dwell_time = self.method.get(("sRXSPEC", "alDwellTime", "0"), 0)
            if dwell_time > 0:
                self.bandwidth_hz = (1 / dwell_time) * 1e9 / 2
                self.dwell_time_s = 1 / self.bandwidth_hz
            else:
                self.bandwidth_hz = None
                self.dwell_time_s = None
            self.reference_frequency_hz = self._get_reference_frequency_hz()
            self.reference_frequency_mhz = (
                self.reference_frequency_hz / 1e6
                if self.reference_frequency_hz
                else None
            )
            self.b0_tesla = self._derive_b0_tesla()
            self.center_ppm = 0.0
            if self.bandwidth_hz and self.reference_frequency_mhz:
                self.bandwidth_ppm = self.bandwidth_hz / self.reference_frequency_mhz
            else:
                self.bandwidth_ppm = None

            self.echo_time_ms = self.method.get(("alTE", "0"), 0) / 1000
            self.repetition_time_ms = self.method.get(("alTR", "0"), 0) / 1000
            self.frame_time_s = None  # Requires calculation
            if self.dwell_time_s and self.n_acquisition:
                self.sampling_time_axis_s = (
                    np.arange(int(self.n_acquisition)) * self.dwell_time_s
                )
            else:
                self.sampling_time_axis_s = None

            # ---------- RF Pulse parameters ----------
            # Siemens headers are complex for RF pulses, mapping basic ones
            self.rf_pulse_params = self.method.get(("sTXSPEC", "aRFPULSE", "0"), None)
            if self.rf_pulse_params is not None:
                self.rfpulse_duration_ms = self.rf_pulse_params.get(
                    "flDuration", 0
                )  # Placeholder
                self.rfpulse_bandwidth_hz = 0  # Placeholder
                self.flipangle_deg = self.method.get(("adFlipAngleDegree", "0"), 0)
                self.rfpulse_is_calculated = False  # Placeholder
                self.rfpulse_sharpness = 0  # Placeholder
                self.rfpulse_bwfac_hzms = 0  # Placeholder
                self.rfpulse_sint = 0  # Placeholder
                self.rfpulse_pint = 0  # Placeholder
                self.rfpulse_type = self.rf_pulse_params.get("tName", "").strip(
                    '"'
                )  # Placeholder
                self.rfpulse_rpfac_percent = 0  # Placeholder
                self.rfpulse_peakpow_w = 0  # Placeholder
                self.rfpulse_shape = self.rf_pulse_params.get("tName", "").strip(
                    '"'
                )  # Placeholder
                self.rfpulse_refpow_w = 0  # Placeholder
            else:
                self.rfpulse_duration_ms = 0
                self.rfpulse_bandwidth_hz = 0
                self.flipangle_deg = self.method.get(("adFlipAngleDegree", "0"), 0)
                self.rfpulse_is_calculated = False
                self.rfpulse_sharpness = 0
                self.rfpulse_bwfac_hzms = 0
                self.rfpulse_sint = 0
                self.rfpulse_pint = 0
                self.rfpulse_type = ""
                self.rfpulse_rpfac_percent = 0
                self.rfpulse_peakpow_w = 0
                self.rfpulse_shape = ""
                self.rfpulse_refpow_w = 0

            self.ref_pulse_params = None  # Not directly available
            self.refpulse_duration_ms = 0
            self.git_commit_hash = self.__get_git_commit_hash()
            self.refpulse_bandwidth_hz = 0
            self.refpulse_flipangle_deg = 0
            self.refpulse_is_calculated = False
            self.refpulse_sharpness = 0
            self.refpulse_bwfac_hzms = 0
            self.refpulse_sint = 0
            self.refpulse_pint = 0
            self.refpulse_type = ""
            self.refpulse_rpfac_percent = 0
            self.refpulse_peakpow_w = 0
            self.refpulse_shape = ""
            self.refpulse_refpow_w = 0

            self.vendor = "siemens"
            self.experiment_number = self.expnum
            self.sequence_name = self.type
            self.measurement_time = self.timestamp
            self.receiver_gain = None
            self.echo_spacing_ms = None
            self.effective_te_ms = None
            self.fat_water_separation = None
            self.load_parameters(self.path)

            if load_data:
                self.load_data(self.path)

        else:
            raise FileNotFoundError(
                f"Path {input_path} is not a valid Siemens experiment path (no .dat file found)."
            )

    def __repr__(self):
        return f"< {__package__}.{self.__class__.__name__} name = {self.path.name} >"

    def __str__(self):
        return f"{self.path.name}"

    def __gt__(self, other):
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __find_dat_file(self, path_str) -> str | None:
        """
        Find a .dat file from a given path.

        Parameters
        ----------
        path_str: str
            A string that is either a path to a directory or a .dat file.

        Returns
        -------
        path: str
            The path to the .dat file as a string, or None if not found.

        """
        path = Path(path_str)
        if path.is_file() and path.suffix == ".dat":
            return str(path)
        if path.is_dir():
            for f in path.iterdir():
                if f.is_file() and f.suffix == ".dat":
                    return str(f)  # Return the first one found
        return None

    def _read_twix_measurement(self, *, parse_data: bool = False):
        """Read the last TWIX measurement with twixtools."""
        twixtools = _import_twixtools()

        measurements = twixtools.read_twix(
            str(self.path),
            parse_data=parse_data,
            parse_pmu=False,
            parse_geometry=False,
            verbose=False,
        )
        return measurements[-1] if isinstance(measurements, list) else measurements

    def _set_header_from_twixtools(self):
        """Load Siemens TWIX headers and expose tuple-addressable sections."""
        self.twix = self._read_twix_measurement(parse_data=False)
        raw_header = self.twix.get("hdr", {}) if isinstance(self.twix, Mapping) else {}
        self.hdr = {
            key: _TwixtoolsHeaderSection(value)
            for key, value in raw_header.items()
            if isinstance(value, Mapping)
        }
        self.hdr_raw = raw_header
        self.method = self.hdr.get("MeasYaps", _TwixtoolsHeaderSection({}))
        self.acqp = self.hdr.get("Config", _TwixtoolsHeaderSection({}))
        self.visu = self.hdr.get("Meas", _TwixtoolsHeaderSection({}))

    @staticmethod
    def _is_enabled(value) -> bool:
        """Return True for Siemens flag values encoded as bool/int/hex/string."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        if isinstance(value, str):
            normalized = value.strip().strip('"').lower()
            if normalized.startswith("0x"):
                try:
                    return int(normalized, 16) != 0
                except ValueError:
                    return False
            return normalized in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _is_spectroscopy_sequence_name(sequence_name) -> bool:
        """Recognize common Siemens spectroscopy sequence file names."""
        if not sequence_name:
            return False

        basename = (
            str(sequence_name)
            .strip()
            .strip('"')
            .replace("\\", "/")
            .rsplit("/", 1)[-1]
            .lower()
        )
        return any(
            token in basename
            for token in ("svs", "csi", "press", "steam", "laser", "spect")
        )

    def load_fid_file(self):
        """Read fid file and return it as np.array."""
        return self.fid

    def _get_reference_frequency_hz(self):
        """Return the Siemens transmitter/reference frequency in Hz."""
        for lookup, key in (
            (self.method, ("sTXSPEC", "asNucleusInfo", "0", "lFrequency")),
            (self.visu, "Frequency"),
            (self.visu, "lFrequency"),
            (self.acqp, "Frequency"),
            (self.acqp, "lFrequency"),
        ):
            value = lookup.get(key) if lookup is not None else None
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _parse_parameters(self, source):
        """Return the Siemens metadata mapped into HyperMRI's parameter schema."""
        self.vendor = "siemens"
        self.experiment_number = self.expnum
        self.sequence_name = self.type
        self.encoding_matrix = self.enc_matrix
        self.measurement_time = self.timestamp
        return self.build_parameter_snapshot()

    def load_data(self, source=None):
        """Load acquired TWIX samples and construct a supported preview.

        Parameters
        ----------
        source : str or pathlib.Path, optional
            TWIX file or directory to select instead of this handler's current
            source. A changed file refreshes its header before data loading.

        Returns
        -------
        xarray.DataArray
            Cartesian 2D image or spectroscopy preview in unscaled scanner
            intensity units. Canonical read/phase/slice, repetition, channel and
            acquisition axes retain additional TWIX counters. Spatial
            coordinates use mm when FOV metadata permits their construction.

        Raises
        ------
        ImportError
            If twixtools is unavailable.
        OSError
            If the selected TWIX file cannot be read.
        ValueError
            If acquired image data or its dimension mapping is inconsistent.
        NotImplementedError
            If imaging includes a non-singleton Fourier-encoded Par partition.
            Its 3D preview needs a validated scanner reference; acquired labelled
            k-space remains available on the handler for explicit processing.

        Notes
        -----
        Each call is eager and deterministic and stores labelled k-space,
        preview, fid and reconstruction names on this handler. Receivers and
        independent slice counters are retained; no coil reduction is applied.
        Spectroscopy transforms acquisition only. Imaging transforms read and
        phase only for supported 2D data. Encoded Par and independent Sli
        counters are distinguished by the fourier_encoded_dims attribute.
        """
        if source is not None:
            dat_file_path = self.__find_dat_file(source)
            if dat_file_path and Path(dat_file_path) != self.path:
                self.path = Path(dat_file_path)
                self._set_header_from_twixtools()

        data, source_dims = self._load_data_with_twixtools()
        self.raw_loader = "twixtools"
        if self.is_spectroscopic:
            self.kspace_xarray = self._to_standard_kspace_xarray(data, source_dims)
            self.data = self._spectral_preview(self.kspace_xarray)
        else:
            self.kspace_xarray = self._to_standard_imaging_kspace_xarray(
                data, source_dims
            )
            self.data = self._image_preview(self.kspace_xarray)
        self.fid = self.kspace_xarray.data
        self.recon_names = ["raw"]
        if hasattr(self, "params"):
            self.params["recon_names"] = self.recon_names.copy()
        return self.data

    def _load_data_with_twixtools(self):
        """Load a TWIX image array with twixtools."""
        twixtools = _import_twixtools()

        # Mapping a filename makes twixtools parse PMU data by default. Some VD
        # files contain legacy PMU blocks that twixtools cannot decode, although
        # their image MDBs are valid. Reuse our PMU-disabled measurement reader.
        raw_measurement = self._read_twix_measurement(parse_data=True)
        mapped = twixtools.map_twix(raw_measurement, verbose=False)
        measurement = mapped[-1] if isinstance(mapped, list) else mapped
        image = measurement.get("image") if isinstance(measurement, Mapping) else None
        if image is None:
            raise ValueError(f"No image data found in Siemens TWIX file: {self.path}")
        image.flags["remove_os"] = True
        image.flags["regrid"] = False
        image.flags["squeeze_singletons"] = True
        average_flags = image.flags.get("average")
        if isinstance(average_flags, dict):
            for dim in average_flags:
                average_flags[dim] = False
        data = np.asarray(image[:])
        return data, list(image.dims)

    def _to_standard_kspace_xarray(self, data, source_dims):
        """Convert Siemens TWIX dimension names to HyperMRI spectroscopy names."""
        da = xr.DataArray(
            data,
            dims=self._standard_dim_names(source_dims),
            attrs={
                "domain": "kspace",
                "signal_domain": "temporal",
                "source_format": "siemens_twix",
                "raw_loader": self.raw_loader,
            },
        )

        for dim in (
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ):
            if dim not in da.dims:
                da = da.expand_dims({dim: [0]})

        standard_dims = (
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        )
        extra_dims = tuple(dim for dim in da.dims if dim not in standard_dims)
        da = da.transpose(*standard_dims, *extra_dims)

        coords = {}
        for dim in da.dims:
            coords[dim] = np.arange(da.sizes[dim])
        if self.sampling_time_axis_s is not None and da.sizes["acquisition_idx"] == len(
            self.sampling_time_axis_s
        ):
            coords["acquisition_idx"] = self.sampling_time_axis_s
        da = da.assign_coords(coords)
        da.coords["acquisition_idx"].attrs["unit"] = "s"
        da.coords["acquisition_idx"].attrs["axis_s"] = np.asarray(
            da.coords["acquisition_idx"].values
        )
        return self._assign_spatial_axis_coords(da)

    def _to_standard_imaging_kspace_xarray(self, data, source_dims):
        """Convert Siemens imaging dimensions to HyperMRI's canonical layout."""
        da = xr.DataArray(
            data,
            dims=self._standard_imaging_dim_names(source_dims),
            attrs={
                "domain": "kspace",
                "signal_domain": "spatial",
                "source_format": "siemens_twix",
                "raw_loader": self.raw_loader,
                "fourier_encoded_dims": [
                    name
                    for source, name in zip(
                        source_dims,
                        self._standard_imaging_dim_names(source_dims),
                        strict=True,
                    )
                    if source in {"Col", "Lin", "Par"}
                ],
            },
        )

        standard_dims = (
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        )
        for dim in standard_dims:
            if dim not in da.dims:
                da = da.expand_dims({dim: [0]})

        extra_dims = tuple(dim for dim in da.dims if dim not in standard_dims)
        da = da.transpose(*standard_dims, *extra_dims)
        da = da.assign_coords(
            {dim: np.arange(da.sizes[dim], dtype=int) for dim in da.dims}
        )
        return self._assign_spatial_axis_coords(da)

    def _assign_spatial_axis_coords(self, da):
        """Attach simple read/phase/slice physical axes in millimetres."""
        fov = np.asarray(self.fov_mm, dtype=float).reshape(-1)
        if fov.size < 2 or not np.all(np.isfinite(fov[:2])) or np.any(fov[:2] <= 0):
            return da

        spatial_specs = (
            ("read_idx", fov[0]),
            ("phase_idx", fov[1]),
            ("slice_idx", fov[2] if fov.size > 2 and fov[2] > 0 else 0.0),
        )
        for dim, extent_mm in spatial_specs:
            if dim not in da.dims:
                continue
            size = int(da.sizes[dim])
            if size <= 1 or extent_mm <= 0:
                axis_mm = np.arange(size, dtype=float)
            else:
                resolution_mm = extent_mm / size
                axis_mm = np.linspace(
                    -extent_mm / 2 + resolution_mm / 2,
                    extent_mm / 2 - resolution_mm / 2,
                    size,
                )
            da = da.assign_coords({dim: axis_mm})
            da.coords[dim].attrs["unit"] = "mm" if extent_mm > 0 else "idx"
            da.coords[dim].attrs["axis_mm"] = axis_mm
        return da

    @staticmethod
    def _standard_dim_names(source_dims):
        dim_names = []
        used = set()
        has_ave = "Ave" in source_dims
        has_repetition_dim = any(dim in source_dims for dim in ("Ave", "Rep"))
        for dim in source_dims:
            if dim == "Col":
                name = "acquisition_idx"
            elif dim == "Cha":
                name = "channel_idx"
            elif dim == "Lin":
                name = "phase_idx"
            elif dim in {"Par", "Sli"}:
                name = "slice_idx"
            elif dim == "Seg":
                name = "read_idx"
            elif dim == "Ave":
                name = "repetition_idx"
            elif dim == "Rep" and not has_ave:
                name = "repetition_idx"
            elif dim == "Set" and not has_repetition_dim:
                # Older Siemens spectroscopy stores averages in the Set counter.
                name = "repetition_idx"
            else:
                name = f"siemens_{dim.lower()}_idx"

            if name in used:
                name = f"siemens_{dim.lower()}_idx"
            dim_names.append(name)
            used.add(name)
        return dim_names

    @staticmethod
    def _standard_imaging_dim_names(source_dims):
        """Map active Siemens imaging counters to canonical dimensions."""
        dim_names = []
        used = set()
        has_ave = "Ave" in source_dims
        has_repetition_dim = any(dim in source_dims for dim in ("Ave", "Rep"))
        for dim in source_dims:
            if dim == "Col":
                name = "read_idx"
            elif dim == "Cha":
                name = "channel_idx"
            elif dim == "Lin":
                name = "phase_idx"
            elif dim in {"Par", "Sli"}:
                name = "slice_idx"
            elif dim == "Eco":
                name = "acquisition_idx"
            elif dim == "Ave":
                name = "repetition_idx"
            elif dim == "Rep" and not has_ave:
                name = "repetition_idx"
            elif dim == "Set" and not has_repetition_dim:
                name = "repetition_idx"
            else:
                name = f"siemens_{dim.lower()}_idx"

            if name in used:
                name = f"siemens_{dim.lower()}_idx"
            dim_names.append(name)
            used.add(name)
        return dim_names

    @staticmethod
    def _image_preview(kspace_xarray):
        """Reconstruct a Cartesian image while retaining individual coils."""
        unsupported = [
            dim
            for dim in kspace_xarray.attrs.get("fourier_encoded_dims", [])
            if dim not in {"phase_idx", "read_idx"}
            and kspace_xarray.sizes.get(dim, 1) > 1
        ]
        if unsupported:
            raise NotImplementedError(
                "Siemens 3D partition image preview requires a validated TWIX/scanner "
                f"Fourier reference for {unsupported}. The acquired array is retained "
                "as handler.kspace_xarray; 2D preview cannot reconstruct encoded Par."
            )
        reco_data = kspace_xarray.copy()
        spatial_dims = [
            dim for dim in ("phase_idx", "read_idx") if reco_data.sizes.get(dim, 1) > 1
        ]
        spatial_axes = tuple(reco_data.get_axis_num(dim) for dim in spatial_dims)
        if spatial_axes:
            shifted = np.fft.ifftshift(reco_data.data, axes=spatial_axes)
            reconstructed = np.fft.ifftn(shifted, axes=spatial_axes)
            reco_data.data = np.fft.fftshift(reconstructed, axes=spatial_axes)
        reco_data.attrs["domain"] = "spatial"
        reco_data.attrs["signal_domain"] = "spatial"
        return reco_data

    def _spectral_preview(self, kspace_xarray):
        """Create a lightweight spectral preview without doing spatial CSI reco."""
        reco_data = kspace_xarray.copy()
        acq_axis = reco_data.get_axis_num("acquisition_idx")
        reco_data.data = np.fft.fftshift(
            np.fft.fft(reco_data.data, axis=acq_axis),
            axes=acq_axis,
        )
        reco_data.attrs["signal_domain"] = "spectral"
        npoints = int(kspace_xarray.sizes["acquisition_idx"])
        try:
            from hypermri.reconstruction.axes import get_freq_axis

            freq_axis_ppm = get_freq_axis(
                scan=self,
                unit="ppm",
                cut_off=0,
                npoints=npoints,
            )
        except Exception:
            freq_axis_ppm = None
        try:
            from hypermri.reconstruction.axes import get_freq_axis

            freq_axis_hz = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=npoints,
            )
        except Exception:
            freq_axis_hz = None
        reco_data = assign_frequency_axis(reco_data, freq_axis_ppm, "ppm")
        return assign_frequency_axis(reco_data, freq_axis_hz, "Hz")

    def open(self, paramfile="method"):
        """Open method file in default text editor for '.json' files."""
        import pprint
        import subprocess
        import sys
        import tempfile
        import time

        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)

        def open_file_platfrom_independent(filename):
            if sys.platform == "win32":
                os.startfile(filename)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, filename])

        try:
            # print(f"Dumping temporary file here:", tmp.name)
            pretty_str = pprint.pformat(getattr(self, paramfile))
            tmp.write(pretty_str.encode("ascii"))
        finally:
            tmp.close()
            open_file_platfrom_independent(tmp.name)
            time.sleep(2)
            os.unlink(tmp.name)

    def __get_git_commit_hash(
        self,
    ):
        """
        Get the current git commit hash of the repository containing this script.

        Returns
        -------
        str
            The git commit hash, or an error message if not found.
        """
        try:
            script_path = Path(__file__).resolve().parent
        except NameError:
            return get_git_commit_hash(
                missing="Not a git repository (interactive mode)."
            )
        return get_git_commit_hash(
            script_path, missing="Could not find a git repository."
        )

    def set_kspace_xarray(self, data, dims, coords=None, write_to_object=True):
        """
        Initialize or update the kspace_xarray with given data, dimensions, and coordinates.

        This method creates a new xarray DataArray if one does not exist, or updates
        the existing one if it does. It enables flexible handling of k-space data
        with various shapes and dimensions.

        Parameters
        ----------
        data : numpy.ndarray
            The k-space data to be stored in the xarray.
        dims : list of str
            List of dimension names.
        coords : dict, optional
            Dictionary of coordinates for each dimension. If None, default coordinates
            are created with np.arange() for each dimension based on the shape of the data.
        write_to_object : bool, optional
            If True (default), the generated k-space xarray is written to self.kspace_xarray,
            otherwise the generated k-space xarray is returned.

        Raises
        ------
        ValueError
            If the number of dimensions does not match the data shape.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the k-space data, dimensions, and coordinates.

        Notes
        -----
        - If coords is not specified, it is automatically generated with np.arange() based on the shape of the data.
        - This method can be used for both the initial creation and subsequent updates of the xarray.

        """
        # Validate dimensions
        if len(data.shape) != len(dims):
            raise ValueError(
                f"Number of dimensions({len(dims)}) must match the data shape({len(data.shape)})"
            )

        # Create default coordinates if not specified
        if coords is None:
            coords = {
                dim: np.arange(size)
                for dim, size in zip(dims, data.shape, strict=False)
            }

        # Create or update xarray DataArray
        try:
            kspace_xarray = xr.DataArray(data, dims=dims, coords=coords)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"""\nCould not create xarray DataArray, shape of: \nData: {data.shape},
                \nDimensions: {dims}, \nCoordinates: {coords}\n
                """
            ) from error

        if write_to_object:
            self.kspace_xarray = kspace_xarray
        else:
            return kspace_xarray

    def ft(
        self,
        xarray: xr.DataArray,
        axes_names: str | list[str] | None = None,
        ift: bool = False,
    ) -> xr.DataArray:
        """Apply shifted Fourier transforms along named Siemens data dimensions.

        Parameters
        ----------
        xarray
            Labelled acquired array. Values can be real or complex; dimension
            order, coordinate values/direction, units and attributes are retained
            in the result. No scanner metadata is required for this transform.
        axes_names
            Ordered dimensions to transform. None uses read_idx then phase_idx.
            Every name must be present in xarray.dims.
        ift
            False applies fftshift(fft(fftshift(values))) per dimension; True
            applies ifftshift(ifft(ifftshift(values))). Forward transforms use
            NumPy's unnormalized convention; inverse transforms normalize by
            each transformed axis length.

        Returns
        -------
        xarray.DataArray
            New labelled array with transformed values, unchanged shape and
            metadata. Calculation precision follows the installed NumPy FFT;
            no storage downcast is applied.

        Raises
        ------
        ValueError
            If an axis name is absent or a transformed dimension has zero length.
        TypeError
            If values cannot be interpreted by NumPy's numerical FFT.

        Notes
        -----
        The operation is eager and deterministic. Input samples are copied
        before transforming; the handler, raw input and metadata are not mutated
        and no result is cached. FFT errors and interrupts propagate rather than
        returning an untransformed or partially transformed array.

        Examples
        --------
        >>> transformed = handler.ft(data, axes_names=["read_idx"])  # doctest: +SKIP
        """
        if axes_names is None:
            axes_names = ["read_idx", "phase_idx"]
        axes_names = np.atleast_1d(axes_names)
        axes_idx = [xarray.get_axis_num(axis_name) for axis_name in axes_names]

        ft_array = xarray.to_numpy().copy()

        if ift:
            for d in range(len(axes_idx)):
                ft_array = np.fft.ifftshift(
                    np.fft.ifft(
                        np.fft.ifftshift(
                            ft_array,
                            axes=(axes_idx[d],),
                        ),
                        axis=axes_idx[d],
                    ),
                    axes=(axes_idx[d],),
                )
                logger.debug("ift along " + str(axes_idx[d]))
        else:
            for d in range(len(axes_idx)):
                ft_array = np.fft.fftshift(
                    np.fft.fft(
                        np.fft.fftshift(
                            ft_array,
                            axes=(axes_idx[d],),
                        ),
                        axis=axes_idx[d],
                    ),
                    axes=(axes_idx[d],),
                )
                logger.debug("ft along " + str(axes_idx[d]))

        ft_xarray = xarray.copy()
        ft_xarray.data = ft_array

        return ft_xarray

    def set_data(self, kspace_xarray, axes_names, ift=False, write_to_object=True):
        """
        Calculate the reco_data with given kspace_xarray.

        Parameters
        ----------
        kspace_xarray : xarray
            The k-space data to be Fourier-transformed.
        ft_axis_name_list : list of str
            List of dimension names to be Fourier-transformed.
        ift: bool
            whether to calculate the inverse Fourier transform.
        write_to_object: bool
            whether to write the reco_data to self

        Returns
        -------
        xarray dataarray containing the spatial domain signal
        """
        # generate a copy of the kspace xarray:
        reco_data = kspace_xarray.copy()

        # Keep track of the signal domain:
        if "acquisition_idx" in axes_names:
            if kspace_xarray.attrs["signal_domain"] == "temporal":
                reco_data.attrs["signal_domain"] = "spectral"
            elif kspace_xarray.attrs["signal_domain"] == "spectral":
                reco_data.attrs["signal_domain"] = "temporal"
            else:
                reco_data.attrs["signal_domain"] = None

        # perform the Fourier transform along the specified axes
        reco_data.data = self.ft(xarray=kspace_xarray, axes_names=axes_names, ift=ift)
        if (
            "read_idx" in axes_names
            or "phase_idx" in axes_names
            or "slice_idx" in axes_names
        ):
            reco_data.attrs["domain"] = "spatial"
        reco_data = assign_frequency_axis(
            reco_data, getattr(self, "freq_axis_ppm", None), "ppm"
        )
        reco_data = assign_frequency_axis(
            reco_data, getattr(self, "freq_axis_hz", None), "Hz"
        )

        # write the xarray to the object or return it:
        if write_to_object:
            self.data = reco_data
        else:
            return reco_data

    def set_seq2d_rps_xarray(
        self, data, dims, coords=None, *, data_precision="float32"
    ):
        """Store labelled scanner samples with explicit storage precision.

        Parameters
        ----------
        data : numpy.ndarray
            Real or complex scanner samples of any shape. Samples are not mutated.
        dims : sequence of str
            Dimension labels in sample storage order, one per axis.
        coords : dict, optional
            xarray coordinate definitions with physical units supplied by the caller.
            Defaults to integer indices. Coordinates are not converted or reordered.
        data_precision : str, numpy type or numpy.dtype, optional
            float16, float32 or float64; defaults to float32. Real storage uses the
            requested dtype; complex storage uses complex64 or complex128 according
            to the shared policy in hypermri.reconstruction.precision.

        Returns
        -------
        None
            Sets self.seq2d_rps_xarray to the newly constructed DataArray.

        Raises
        ------
        ValueError
            If precision, dimensions or coordinates are invalid.

        Notes
        -----
        This eager, deterministic metadata operation copies sample storage, preserves
        coordinate units and leaves the acquired array unchanged. It reads no files.
        """
        # Validate dimensions
        if len(data.shape) != len(dims):
            raise ValueError(
                f"Number of dimensions({len(dims)}) must match the length of the data shape(len={len(data.shape)}, shape={data.shape})"
            )

        data = convert_for_storage(data, data_precision)

        # Create default coordinates if not provided
        if coords is None:
            coords = {
                dim: np.arange(size)
                for dim, size in zip(dims, data.shape, strict=False)
            }

        # Create or update the xarray DataArray
        try:
            self.seq2d_rps_xarray = xr.DataArray(data, dims=dims, coords=coords)
        except ValueError as exc:
            raise ValueError(
                f"""\nKonnte kein xarray DataArray erstellen, Form der: \nDaten: {
                    data.shape
                },
                \nDimensionen: {dims}, \nKoordinaten: {coords}\n
                """
            ) from exc

    def rearrange_data_xarray(self, data=None, force_orient=False):
        """Rearrange the reco_data xarray according to the patient position and slice orientation to match the seq2d."""
        if data is None:
            reco_data = self.data.copy()
        else:
            reco_data = data

        # make sure to onlt rearange data once unless it's explicitly
        # requested:
        num_rearrange = reco_data.attrs.get("num_rearrange", 0)
        if num_rearrange == 0:
            reco_data.attrs["num_rearrange"] = 0
        elif num_rearrange > 0 and force_orient is False:
            return
        elif num_rearrange > 0 and force_orient is True:
            pass

        return reco_data

    # def set_seq2d_xyz_xarray(self, seq2d_rps_xarray, return_seq2d_xyz=False):
    #     seq2d_rps_xarray.attrs["patient_pos"]
    #     seq2d_rps_xarray.attrs["slice_orientation"]

    #     np.round(seq2d_rps_xarray.attrs["read_direction"])
    #     np.round(seq2d_rps_xarray.attrs["phase_direction"])
    #     np.round(seq2d_rps_xarray.attrs["slice_direction"])

    #     seq2d_xyz_xarray = seq2d_rps_xarray.copy()

    #     if return_seq2d_xyz:
    #         return seq2d_xyz_xarray
    #     else:
    #         self.seq2d_xyz_xarray = seq2d_xyz_xarray

    def postprocess_fid_file(self, values=None):
        """
        Applies standard Siemens post-processing to the raw data.
        This includes oversampling removal and ramp sampling regridding.
        """
        data, _source_dims = self._load_data_with_twixtools()
        return data
