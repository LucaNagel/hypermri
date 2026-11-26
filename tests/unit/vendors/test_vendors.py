from __future__ import annotations

import logging
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hypermri.mixins import UniformDataPropertiesMixin
from hypermri.vendors.base import VendorHandler


class _DummyVendorHandler(VendorHandler, UniformDataPropertiesMixin):
    """Smallest handler with the frequency accessors every real handler has."""

    @property
    def id(self) -> str:
        return "dummy"

    def load_data(self, source):
        return xr.DataArray(np.zeros(1), dims=("acquisition_idx",))


def _write_topspin_dataset(root: Path) -> Path:
    scan_dir = root / "10"
    scan_dir.mkdir()

    (scan_dir / "acqus").write_text(
        "\n".join(
            [
                "##$PULPROG= <zg>",
                "##$NUC1= <13C>",
                "##$BF1= 100.0",
                "##$SFO1= 100.001",
                "##$SW_h= 2000.0",
                "##$SW= 20.0",
                "##$TD= 256",
                "##$O1= 1000.0",
                "##$DATE= 1700000000",
                "##$DTYPA= 0",
                "##$BYTORDA= 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fid_payload = struct.pack("<8i", 1, 0, 2, 0, 3, 0, 4, 0)
    (scan_dir / "fid").write_bytes(fid_payload)
    return scan_dir


def test_topspin_summary_returns_canonical_parameter_snapshot(tmp_path):
    from hypermri.vendors.topspin import BrukerTopSpinHandler

    scan_dir = _write_topspin_dataset(tmp_path)
    handler = BrukerTopSpinHandler(scan_dir, load_data=False)

    params = handler.get_parameters()

    assert handler.summary() == params
    assert set(params) == handler.parameter_keys()
    assert params["vendor"] == "bruker_topspin"
    assert params["sequence_name"] == "zg"
    assert params["nucleus"] == "13C"
    assert params["n_acquisition"] == 128


def test_topspin_reference_frequency_is_bf1_and_absent_without_it(tmp_path):
    """0 ppm is BF1 in acqus, not the transmitter frequency SFO1 = BF1 + O1.

    Synthetic acqus: BF1 = 100.0 MHz, O1 = 1000 Hz, SFO1 = 100.001 MHz. The
    expected transmitter offset is O1 / BF1 = 1000 Hz / 100 MHz = 10 ppm.
    """
    from hypermri.utils.physics import get_gmr
    from hypermri.vendors.topspin import BrukerTopSpinHandler

    scan_dir = _write_topspin_dataset(tmp_path)
    handler = BrukerTopSpinHandler(scan_dir, load_data=False)

    assert handler.reference_frequency_mhz == 100.0
    assert handler.center_ppm == pytest.approx(10.0, rel=1e-12)
    assert handler.b0_tesla == pytest.approx(100.0 / get_gmr("13C", unit="MHz_T"))
    assert handler.params["b0_tesla"] == handler.b0_tesla
    assert handler.params["reference_frequency_mhz"] == 100.0

    # TD = 256 gives 128 complex points; the cutoff of 70 leaves 58. The
    # shifted FFT puts 0 Hz at index 29, which the ppm axis maps to center_ppm.
    ppm_axis = handler._freq_axis_ppm
    assert ppm_axis.shape == (58,)
    assert handler._freq_axis_hz[29] == 0.0
    assert ppm_axis[29] == pytest.approx(10.0, rel=1e-12)

    acqus = scan_dir / "acqus"
    acqus.write_text(acqus.read_text().replace("##$BF1= 100.0\n", ""))
    without_bf1 = BrukerTopSpinHandler(scan_dir, load_data=False)
    assert without_bf1.reference_frequency_mhz is None
    assert without_bf1.bandwidth_ppm is None
    assert without_bf1._freq_axis_ppm is None
    assert without_bf1.b0_tesla is None
    assert without_bf1.center_ppm == 0.0
    assert "BF1=unknown" in str(without_bf1)


def test_vendor_handler_provides_shared_frequency_and_dwell_accessors(monkeypatch):
    from hypermri.reconstruction import axes as spectroscopy_core

    calls = []

    def fake_get_freq_axis(scan, unit):
        calls.append(unit)
        if unit == "ppm":
            return np.array([1.0, 2.0])
        return np.array([10.0, 20.0])

    monkeypatch.setattr(spectroscopy_core, "get_freq_axis", fake_get_freq_axis)

    handler = _DummyVendorHandler()
    handler.dwell_time_s = 2.5e-4

    np.testing.assert_array_equal(handler.freq_axis_ppm, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(handler.freq_axis_hz, np.array([10.0, 20.0]))
    assert handler.dwell_times_s == 2.5e-4
    assert calls == ["ppm", "Hz"]
    np.testing.assert_array_equal(handler.freq_axis_ppm, np.array([1.0, 2.0]))
    assert calls == ["ppm", "Hz", "ppm"]


def test_siemens_handler_satisfies_vendor_handler_interface():
    from hypermri.vendors.siemens import SiemensHandler

    assert SiemensHandler.__abstractmethods__ == frozenset()


def test_shared_frequency_axes_follow_current_acquisition_parameters():
    """No first-access cache may freeze an obsolete acquisition length."""
    handler = _DummyVendorHandler()
    handler.get = lambda name, default=None: getattr(handler, name, default)
    handler.bandwidth_hz = 8.0
    handler.bandwidth_ppm = 4.0
    handler.dwell_time_s = 0.125
    handler.n_acquisition = 2
    np.testing.assert_array_equal(handler.freq_axis_hz, [-4.0, 0.0])
    handler.n_acquisition = 4
    np.testing.assert_array_equal(handler.freq_axis_hz, [-4.0, -2.0, 0.0, 2.0])
    np.testing.assert_array_equal(handler.freq_axis_ppm, [-2.0, -1.0, 0.0, 1.0])
    assert not hasattr(handler, "_freq_axis_hz")


def test_siemens_handler_preloads_twixtools_data_as_standard_xarray(
    tmp_path,
    monkeypatch,
):
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"twix")
    raw = np.zeros((4, 2, 3, 5), dtype=np.complex64)

    class FakeTwixImage:
        def __init__(self):
            self.dims = ["Col", "Lin", "Ave", "Seg"]
            self.flags = {
                "average": {
                    "Seg": True,
                    "Ave": True,
                    "Rep": False,
                }
            }

        def __getitem__(self, key):
            assert isinstance(key, slice)
            return raw

    fake_twix_image = FakeTwixImage()
    fake_header = _siemens_csi_header()
    fake_twixtools = types.SimpleNamespace(
        read_twix=lambda path, **kwargs: [{"hdr": fake_header}],
        map_twix=lambda path, verbose=False: [{"image": fake_twix_image}],
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    handler = SiemensHandler(dat_file, load_data=True, log_mode="error")

    assert handler.get_parameters()["vendor"] == "siemens"
    assert handler.cut_off == 0
    assert handler.raw_loader == "twixtools"
    assert handler.recon_names == ["raw"]
    assert handler.get_parameters()["recon_names"] == ["raw"]
    assert handler.method.get(("sTXSPEC", "asNucleusInfo", "0", "tNucleus")) == "13C"
    assert handler.is_spectroscopic is True
    assert fake_twix_image.flags["average"] == {
        "Seg": False,
        "Ave": False,
        "Rep": False,
    }
    assert fake_twix_image.flags["remove_os"] is True
    assert fake_twix_image.flags["regrid"] is False
    assert fake_twix_image.flags["squeeze_singletons"] is True
    assert handler.kspace_xarray.dims == (
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    )
    assert handler.kspace_xarray.shape == (4, 3, 1, 2, 1, 5)
    np.testing.assert_allclose(
        handler.kspace_xarray.coords["acquisition_idx"].values,
        np.array([0.0, 0.001, 0.002, 0.003]),
    )
    assert handler.kspace_xarray.coords["acquisition_idx"].attrs["unit"] == "s"
    np.testing.assert_allclose(
        handler.kspace_xarray.coords["read_idx"].values,
        np.array([-120.0, -60.0, 0.0, 60.0, 120.0]),
    )
    assert handler.kspace_xarray.coords["read_idx"].attrs["unit"] == "mm"
    np.testing.assert_allclose(
        handler.kspace_xarray.coords["phase_idx"].values,
        np.array([-75.0, 75.0]),
    )
    assert handler.kspace_xarray.coords["phase_idx"].attrs["unit"] == "mm"
    assert handler.data.shape == (4, 3, 1, 2, 1, 5)
    assert handler.data.attrs["signal_domain"] == "spectral"
    np.testing.assert_allclose(
        handler.data["freq_axis_hz"],
        np.array([-500.0, -250.0, 0.0, 250.0]),
    )
    assert handler.data["freq_axis_ppm"].dims == ("acquisition_idx",)
    assert not {"freq_axis_hz", "freq_axis_ppm"} & set(handler.data.attrs)


def _siemens_csi_header() -> dict:
    """Return an invented twix header of a 13C CSI scan at 30.98754 MHz."""
    return {
        "MeasYaps": {
            "tSequenceFileName": "%CustomerSeq%\\csi_fid_10mm",
            "sTXSPEC": {
                "asNucleusInfo": [
                    {
                        "tNucleus": "13C",
                        "lFrequency": 30_987_540.0,
                    }
                ],
                "aRFPULSE": [
                    {
                        "flDuration": 1.0,
                        "tName": "pulse",
                    }
                ],
            },
            "ucCSI": 1,
            "sSliceArray": {
                "lSize": 1,
                "asSlice": [
                    {
                        "dThickness": 30.0,
                        "dReadoutFOV": 300.0,
                        "dPhaseFOV": 300.0,
                    }
                ],
            },
            "sKSpace": {
                "ucDimension": 2,
                "lBaseResolution": 5,
                "lPhaseEncodingLines": 2,
                "lPartitions": 1,
            },
            "lRepetitions": 1,
            "lAverages": 3,
            "lContrasts": 1,
            "sSpecPara": {"lVectorSize": 4},
            "sRXSPEC": {"alDwellTime": [500000.0]},
            "alTE": [2300.0],
            "alTR": [10000000.0],
        },
        "Config": {"tTimeStamp": 0},
        "Meas": {
            "MeasUid": 104,
            "tPatientPosition": "HFS",
            "iNoOfFourierColumns": 4,
            "iNoOfFourierLines": 2,
            "iNoOfFourierPartitions": 1,
        },
    }


def test_siemens_handler_accepts_hex_spectroscopy_flags(tmp_path, monkeypatch):
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"twix")

    fake_header = {
        "MeasYaps": {
            "tSequenceFileName": "%CustomerSeq%\\csi_fid_10mm",
            "sTXSPEC": {
                "asNucleusInfo": [{"tNucleus": "13C", "lFrequency": 30_987_540.0}]
            },
            "ucCSI": "0x1",
            "sSliceArray": {
                "lSize": 1,
                "asSlice": [
                    {
                        "dThickness": 30.0,
                        "dReadoutFOV": 300.0,
                        "dPhaseFOV": 300.0,
                    }
                ],
            },
            "sKSpace": {
                "ucDimension": 2,
                "lBaseResolution": 5,
                "lPhaseEncodingLines": 2,
                "lPartitions": 1,
            },
            "lRepetitions": 1,
            "lAverages": 1,
            "lContrasts": 1,
            "sSpecPara": {"lVectorSize": 4},
            "sRXSPEC": {"alDwellTime": [500000.0]},
            "alTE": [2300.0],
            "alTR": [10000000.0],
        },
        "Config": {"tTimeStamp": 0},
        "Meas": {
            "MeasUid": 104,
            "tPatientPosition": "HFS",
            "iNoOfFourierColumns": 4,
            "iNoOfFourierLines": 2,
            "iNoOfFourierPartitions": 1,
        },
    }
    fake_twixtools = types.SimpleNamespace(
        read_twix=lambda path, **kwargs: [{"hdr": fake_header}],
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    handler = SiemensHandler(dat_file, load_data=False, log_mode="error")

    assert handler.is_spectroscopic is True
    assert handler.method.get(("sRXSPEC", "alDwellTime", "0")) == 500000.0
    assert handler.method.get(("alTE", "0")) == 2300.0


def test_siemens_field_follows_the_nucleus_frequency(tmp_path, monkeypatch):
    from hypermri.utils.physics import get_gmr
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"twix")
    fake_header = _siemens_csi_header()
    fake_twixtools = types.SimpleNamespace(
        read_twix=lambda path, **kwargs: [{"hdr": fake_header}],
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    handler = SiemensHandler(dat_file, load_data=False, log_mode="error")

    assert handler.reference_frequency_mhz == pytest.approx(30.98754)
    assert handler.b0_tesla == pytest.approx(30.98754 / get_gmr("13C", unit="MHz_T"))


def test_siemens_handler_recognizes_svs_name_and_maps_set_to_repetitions(
    tmp_path,
    monkeypatch,
):
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "twix_vb.dat"
    dat_file.write_bytes(b"twix")
    raw = np.zeros((3, 2, 4), dtype=np.complex64)

    class FakeTwixImage:
        def __init__(self):
            self.dims = ["Set", "Cha", "Col"]
            self.flags = {"average": {"Set": False}}

        def __getitem__(self, key):
            assert isinstance(key, slice)
            return raw

    fake_header = {
        "MeasYaps": {
            "tSequenceFileName": r"%SiemensSeq%\svs_se",
            "sTXSPEC": {
                "asNucleusInfo": [{"tNucleus": "1H", "lFrequency": 123_000_000.0}]
            },
            "sSpecPara": {
                "lVectorSize": 4,
                "sVoI": {
                    "dReadoutFOV": 20.0,
                    "dPhaseFOV": 20.0,
                    "dThickness": 20.0,
                },
            },
            "sRXSPEC": {"alDwellTime": [500000.0]},
            "lAverages": 3,
            "lRepetitions": 1,
        },
        "Config": {"tTimeStamp": 0},
        "Meas": {"MeasUid": 178, "tPatientPosition": "HFS"},
    }
    fake_image = FakeTwixImage()
    read_calls = []

    def fake_read_twix(path, **kwargs):
        read_calls.append(kwargs)
        return [{"hdr": fake_header, "mdb": []}]

    def fake_map_twix(measurement, verbose=False):
        assert isinstance(measurement, dict)
        assert verbose is False
        return {"image": fake_image}

    fake_twixtools = types.SimpleNamespace(
        read_twix=fake_read_twix,
        map_twix=fake_map_twix,
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    handler = SiemensHandler(dat_file, load_data=True, log_mode="error")

    assert handler.is_spectroscopic is True
    assert [call["parse_data"] for call in read_calls] == [False, True]
    assert all(call["parse_pmu"] is False for call in read_calls)
    assert all(call["parse_geometry"] is False for call in read_calls)
    assert handler.kspace_xarray.dims == (
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    )
    assert handler.kspace_xarray.shape == (4, 3, 1, 1, 2, 1)


def test_siemens_handler_reconstructs_cartesian_imaging_preview(tmp_path, monkeypatch):
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "se.dat"
    dat_file.write_bytes(b"twix")
    raw = np.zeros((3, 2, 4), dtype=np.complex64)
    raw[1, :, 2] = 1.0

    class FakeTwixImage:
        def __init__(self):
            self.dims = ["Lin", "Cha", "Col"]
            self.flags = {"average": {}}

        def __getitem__(self, key):
            assert isinstance(key, slice)
            return raw

    fake_header = {
        "MeasYaps": {
            "tSequenceFileName": r"%SiemensSeq%\se",
            "sTXSPEC": {
                "asNucleusInfo": [{"tNucleus": "1H", "lFrequency": 123_000_000.0}]
            },
            "sSliceArray": {
                "lSize": 1,
                "asSlice": [
                    {
                        "dThickness": 1.0,
                        "dReadoutFOV": 40.0,
                        "dPhaseFOV": 30.0,
                    }
                ],
            },
            "sKSpace": {
                "ucDimension": 2,
                "lBaseResolution": 4,
                "lPhaseEncodingLines": 3,
                "lPartitions": 1,
            },
            "sRXSPEC": {"alDwellTime": [500000.0]},
            "lAverages": 1,
        },
        "Config": {"tTimeStamp": 0},
        "Meas": {"MeasUid": 3485, "tPatientPosition": "HFS"},
    }
    fake_image = FakeTwixImage()
    fake_twixtools = types.SimpleNamespace(
        read_twix=lambda path, **kwargs: [{"hdr": fake_header, "mdb": []}],
        map_twix=lambda measurement, verbose=False: {"image": fake_image},
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    handler = SiemensHandler(dat_file, load_data=True, log_mode="error")

    assert handler.is_spectroscopic is False
    assert handler.kspace_xarray.dims == (
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    )
    assert handler.kspace_xarray.shape == (1, 1, 1, 3, 2, 4)
    assert handler.kspace_xarray.attrs["domain"] == "kspace"
    assert handler.data.shape == (1, 1, 1, 3, 2, 4)
    assert handler.data.attrs["domain"] == "spatial"
    assert np.iscomplexobj(handler.data.values)


def test_siemens_csi_without_matching_fov_and_matrix_has_no_resolution(
    tmp_path, monkeypatch, caplog
):
    # The CSI header gives a two-entry slice FOV and a three-entry matrix.
    from hypermri.vendors.siemens import SiemensHandler

    dat_file = tmp_path / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"twix")
    fake_header = _siemens_csi_header()
    fake_twixtools = types.SimpleNamespace(
        read_twix=lambda path, **kwargs: [{"hdr": fake_header}],
    )
    monkeypatch.setitem(sys.modules, "twixtools", fake_twixtools)

    with caplog.at_level(logging.WARNING):
        handler = SiemensHandler(dat_file, load_data=False, log_mode="warning")

    assert handler.resolution_mm is None
    assert "fov_mm" in caplog.text
    assert str(dat_file) in caplog.text


def test_siemens_kspace_array_error_keeps_its_cause():
    from hypermri.vendors.siemens import SiemensHandler

    handler = SiemensHandler.__new__(SiemensHandler)

    with pytest.raises(ValueError, match="Could not create xarray") as caught:
        handler.set_kspace_xarray(np.zeros(2), ["x"], coords={"x": [0, 1, 2]})

    assert isinstance(caught.value.__cause__, ValueError)


def test_topspin_ppm_spacing_is_hz_over_bf1_not_sw(tmp_path):
    """The ppm step is Hz / BF1, not SW = SW_h / SFO1 from acqus.

    Synthetic acqus: SW_h = 2000 Hz, BF1 = 100.0 MHz, SFO1 = 100.001 MHz and
    a deliberately inconsistent SW = 20.0 ppm (SW_h / SFO1 would be 19.9998).
    The expected spectral width is 2000 Hz / 100 MHz = 20 ppm, and every Hz
    step of the axis maps to Hz / 100 ppm. Tolerance: relative 1e-12, the
    double-precision rounding of the axis arithmetic.
    """
    from hypermri.vendors.topspin import BrukerTopSpinHandler

    scan_dir = _write_topspin_dataset(tmp_path)
    acqus = scan_dir / "acqus"
    acqus.write_text(acqus.read_text().replace("##$SW= 20.0", "##$SW= 19.0"))
    handler = BrukerTopSpinHandler(scan_dir, load_data=False)

    assert handler.bandwidth_ppm == pytest.approx(20.0, rel=1e-12)
    assert handler.params["bandwidth_ppm"] == handler.bandwidth_ppm
    np.testing.assert_allclose(
        handler._freq_axis_ppm - handler.center_ppm,
        handler._freq_axis_hz / 100.0,
        rtol=1e-12,
        atol=1e-12,
    )
