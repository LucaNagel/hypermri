from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _write_magritek_dataset(
    root: Path,
    *,
    n_points: int = 8,
    n_spectra: int = 4,
    dim_1: int = 1,
    dim_2: int = 1,
) -> Path:
    scan_dir = root / "260312-171441 PolarizationMonitoring13C (Magritek_Test)"
    scan_dir.mkdir()

    (scan_dir / "acqu.par").write_text(
        "\n".join(
            [
                'expName = "260312-171441 PolarizationMonitoring13C (Magritek_Test)"',
                'experiment = "Proton"',
                'specType = "C43Grad"',
                'specID = "SPA312"',
                'softwareVersion = "2.01.18"',
                'nucleus = "13C"',
                "nrPnts = 8",
                "nrSpectra = 4",
                "nrScans = 1",
                "dwellTime = 200",
                "bandwidth = 5",
                "centerFreqPPM = 110",
                "repTime = 5000",
                "duration = 20",
                "b1Freq13C = 10.8417577",
                'rxChannel = "13C"',
                "pulseLength13C = 64.56",
                "tipAngle = 90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (scan_dir / "proc.par").write_text(
        'ppmOffset = 110\nphaseMethod = "p0, p1 fixed phase"\n',
        encoding="utf-8",
    )
    (scan_dir / "processing.script").write_text("Phase(0,0);\nZoom(-20,200);\n")
    pp_dir = scan_dir / "ppCode"
    pp_dir.mkdir()
    (pp_dir / "PolarizationMonitoring13C.mac").write_text(
        "procedure(PolarizationMonitoring13C, parameters=null)\nendproc()\n",
        encoding="utf-8",
    )
    (pp_dir / "PolarizationMonitoring13CDefault.par").write_text(
        'expName = "PolarizationMonitoring13C"\n',
        encoding="utf-8",
    )

    header = b"SORPATAD1.1V" + struct.pack(
        "<5I", 501, n_points, n_spectra, dim_1, dim_2
    )
    count = n_spectra * dim_1 * dim_2 * n_points
    shape = (n_spectra, dim_1, dim_2, n_points)
    complex_data = (
        np.arange(count, dtype=np.float32).reshape(shape)
        + 1j * np.linspace(0.5, 1.5, count, dtype=np.float32).reshape(shape)
    ).astype(np.complex64)
    interleaved = np.empty(complex_data.size * 2, dtype=np.float32)
    interleaved[0::2] = complex_data.real.ravel()
    interleaved[1::2] = complex_data.imag.ravel()
    with (scan_dir / "data.2d").open("wb") as handle:
        handle.write(header)
        interleaved.tofile(handle)

    return scan_dir


@pytest.mark.parametrize("dim_1,dim_2", [(1, 1), (2, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize("filtered", [False, True])
def test_magritek_spectral_fft_preserves_independent_export_axes(
    tmp_path, dim_1, dim_2, filtered
):
    """Invented exports use separate Fourier references for each exported FID.

    The existing loader labels are retained without reducing either counter.
    These tests establish sample/axis preservation, not scanner-specific
    physical meanings of the two optional counters. Complex tolerance is 1e-6.
    """
    from hypermri.reconstruction.strategies.magritek import MagritekSpectroRecoStrategy
    from hypermri.vendors.magritek import MagritekHandler

    handler = MagritekHandler(
        _write_magritek_dataset(tmp_path, dim_1=dim_1, dim_2=dim_2), load_data=True
    )
    source = handler.scanner_data.copy(deep=True)
    fid_before = handler.fid.copy()
    options = dict(
        append_parameters=True,
        combine_channels=False,
        remove_dc=False,
        apply_digital_filter=False,
        apodization="off",
    )
    expected_fid = source.data
    if filtered:
        options.update(
            apply_digital_filter=True,
            digital_filter_points=2,
            zero_fill_to=16,
            apodization="hann",
        )
        expected_fid = source.data[..., 2:] * np.hanning(6).astype(np.float32)
        expected_fid = np.pad(
            expected_fid, [(0, 0)] * (expected_fid.ndim - 1) + [(0, 10)]
        )
    strategy = MagritekSpectroRecoStrategy()
    actual = strategy.reconstruct(handler, **options)
    expected = np.fft.fftshift(np.fft.fft(expected_fid, axis=-1), axes=-1)
    if "channel_idx" not in source.dims:
        expected = expected[:, None, :]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)
    assert actual.sizes["repetition_idx"] == 4
    assert actual.sizes["acquisition_idx"] == (16 if filtered else 8)
    assert actual.freq_axis_hz.size == actual.sizes["acquisition_idx"]
    expected_hz = np.fft.fftshift(
        np.fft.fftfreq(expected.shape[-1], d=handler.dwell_time_s)
    )
    np.testing.assert_allclose(actual.freq_axis_hz, expected_hz, rtol=1e-12, atol=1e-12)
    expected_ppm = (
        handler.center_ppm + expected_hz * handler.bandwidth_ppm / handler.bandwidth_hz
    )
    np.testing.assert_allclose(
        actual.freq_axis_ppm, expected_ppm, rtol=1e-12, atol=1e-12
    )
    for name in ("freq_axis_hz", "freq_axis_ppm"):
        assert actual[name].dims == ("acquisition_idx",)
        assert name not in actual.attrs
    if dim_1 > 1 or dim_2 > 1:
        assert actual.sizes["channel_idx"] == dim_1
        assert actual.sizes["average_idx"] == dim_2
    np.testing.assert_array_equal(handler.fid, fid_before)
    xr.testing.assert_identical(handler.scanner_data, source)
    xr.testing.assert_identical(strategy.reconstruct(handler, **options), actual)


def test_detect_vendor_for_magritek_export(tmp_path):
    from hypermri.vendors import detect_vendor

    scan_dir = _write_magritek_dataset(tmp_path)

    assert detect_vendor(scan_dir) == "magritek"


def test_magritek_handler_loads_spectral_data(tmp_path):
    from hypermri.vendors.magritek import MagritekHandler

    scan_dir = _write_magritek_dataset(tmp_path)
    handler = MagritekHandler(scan_dir, load_data=True)

    assert handler.vendor == "magritek"
    assert handler.sequence_name == "PolarizationMonitoring13C"
    assert handler.spinsolve_experiment_name.endswith("(Magritek_Test)")
    assert handler.spinsolve_experiment == "Proton"
    assert handler.spinsolve_spec_type == "C43Grad"
    assert handler.spinsolve_pulse_program_name == "PolarizationMonitoring13C"
    assert handler.repetition_time_ms == 5000.0
    assert handler.dwell_time_s == pytest.approx(200e-6)
    assert handler.rfpulse_duration_ms == pytest.approx(0.06456)
    assert handler.n_acquisition == 8
    assert handler.n_repetitions == 4
    assert handler.scanner_data is not None
    assert handler.scanner_data.dims == ("repetition_idx", "acquisition_idx")
    assert handler.scanner_data.shape == (4, 8)
    assert "time" in handler.scanner_data.coords
    assert handler.scanner_data.attrs["signal_domain"] == "temporal"
    assert handler.freq_axis_ppm is not None
    assert handler.freq_axis_hz is not None


def test_magritek_field_follows_the_13c_spectrometer_frequency(tmp_path):
    from hypermri.utils.physics import get_gmr
    from hypermri.vendors.magritek import MagritekHandler

    handler = MagritekHandler(_write_magritek_dataset(tmp_path))

    assert handler.reference_frequency_mhz == pytest.approx(10.8417577)
    assert handler.b0_tesla == pytest.approx(10.8417577 / get_gmr("13C", unit="MHz_T"))
    assert handler.params["b0_tesla"] == handler.b0_tesla


def test_magritek_summary_returns_canonical_parameter_snapshot(tmp_path):
    from hypermri.vendors.magritek import MagritekHandler

    scan_dir = _write_magritek_dataset(tmp_path)
    handler = MagritekHandler(scan_dir, load_data=False)

    params = handler.get_parameters()

    assert handler.summary() == params
    assert set(params) == handler.parameter_keys()
    assert params["vendor"] == "magritek"
    assert params["sequence_name"] == "PolarizationMonitoring13C"
    assert params["n_acquisition"] == 8


def test_study_accepts_direct_magritek_scan_directory(tmp_path):
    from hypermri.study import Study

    scan_dir = _write_magritek_dataset(tmp_path)
    study = Study(scan_dir, verbose=False, smart_load=False, load_data=True)

    assert len(study) == 1
    scan = next(iter(study))
    assert scan.handler is not None
    assert scan.handler.vendor == "magritek"
    assert scan.type == "PolarizationMonitoring13C"
    assert scan.scanner_data is not None
    assert scan.scanner_data.shape == (4, 8)
    assert scan.repetition_time_ms == 5000.0
    assert scan.dwell_time_s == pytest.approx(200e-6)
    assert scan.bandwidth_hz == pytest.approx(5000.0)
    assert scan.bandwidth_ppm is not None
    assert scan.n_acquisition == 8
    assert scan.center_ppm == 110.0
    assert scan.sampling_time_axis_s is not None
    assert len(scan.sampling_time_axis_s) == 8
    assert scan.sampling_time_axis_s[1] == pytest.approx(200e-6)


def test_def_fit_params_can_convert_ppm_after_repetition_mean(tmp_path):
    from hypermri.analysis.fitting.spectroscopy import def_fit_params
    from hypermri.study import Study

    scan_dir = _write_magritek_dataset(tmp_path)
    study = Study(scan_dir, verbose=False, smart_load=False, load_data=True)
    scan = next(iter(study))

    data_mean = scan.scanner_data.mean(dim="repetition_idx")
    fit_params = def_fit_params(
        fit_params={"metabs": "unknown", "metabs_freqs_ppm": 50},
        data=data_mean,
    )

    assert fit_params.attrs["freq_axis_ppm"] is not None
    assert fit_params.attrs["freq_axis_hz"] is not None
    assert fit_params.attrs["metabs_freqs_hz"][0] != 0.0
    assert fit_params["initial_freqs_hz"].values[0] != 0.0


def test_magritek_sequence_smart_loads_as_singlepulse(tmp_path):
    from hypermri.study import Study

    scan_dir = _write_magritek_dataset(tmp_path)
    study = Study(scan_dir, verbose=False, log_mode="error")
    scan = next(iter(study))

    assert type(scan).__name__ == "Singlepulse"
    assert scan.handler.vendor == "magritek"
    assert scan.type == "PolarizationMonitoring13C"
    assert scan.data is not None
    assert scan.scanner_data.attrs["signal_domain"] == "temporal"
    assert scan.data.attrs["signal_domain"] == "spectral"


@pytest.mark.parametrize(
    ("mode", "expected_data_loads", "expected_reconstructions"),
    [
        ("lazy", 0, 0),
        ("minimal", 1, 1),
        ("full", 1, 1),
    ],
)
def test_smart_load_constructs_and_processes_magritek_scan_once(
    tmp_path,
    monkeypatch,
    mode,
    expected_data_loads,
    expected_reconstructions,
):
    """Characterize handler and reconstruction work in the Phase 2B path."""
    from hypermri import load_scan
    from hypermri.reconstruction.strategies.magritek import (
        MagritekSpectroRecoStrategy,
    )
    from hypermri.vendors.magritek import MagritekHandler

    scan_dir = _write_magritek_dataset(tmp_path)
    calls = {"handler": 0, "load_data": 0, "reconstruct": 0}
    constructed_handlers = []

    original_handler_init = MagritekHandler.__init__
    original_load_data = MagritekHandler.load_data
    original_reconstruct = MagritekSpectroRecoStrategy.reconstruct

    def counted_handler_init(self, *args, **kwargs):
        calls["handler"] += 1
        constructed_handlers.append(self)
        original_handler_init(self, *args, **kwargs)

    def counted_load_data(self, *args, **kwargs):
        calls["load_data"] += 1
        return original_load_data(self, *args, **kwargs)

    def counted_reconstruct(self, *args, **kwargs):
        calls["reconstruct"] += 1
        return original_reconstruct(self, *args, **kwargs)

    monkeypatch.setattr(MagritekHandler, "__init__", counted_handler_init)
    monkeypatch.setattr(MagritekHandler, "load_data", counted_load_data)
    monkeypatch.setattr(
        MagritekSpectroRecoStrategy,
        "reconstruct",
        counted_reconstruct,
    )

    scan = load_scan(scan_dir, mode=mode)

    assert type(scan).__name__ == "Singlepulse"
    assert scan.handler is constructed_handlers[0]
    assert calls == {
        "handler": 1,
        "load_data": expected_data_loads,
        "reconstruct": expected_reconstructions,
    }
    if mode == "lazy":
        assert scan._scanner_data is None
        assert scan._data is None
    else:
        assert scan._scanner_data.attrs["signal_domain"] == "temporal"
        assert scan._data.attrs["signal_domain"] == "spectral"
        assert scan._data.dims == ("repetition_idx", "acquisition_idx")

    # Reapplying the automatic policy must reuse the registered result rather
    # than processing the same raw data a second time.
    scan.run_default_reconstructions(load_data=mode)
    assert calls["reconstruct"] == expected_reconstructions


def test_registered_load_constructs_no_generic_wrapper(
    tmp_path,
    monkeypatch,
):
    """The sequence is built from the one handler, not from an experiment."""
    from hypermri import load_scan
    from hypermri.loading.experiment import MRIExperiment

    scan_dir = _write_magritek_dataset(tmp_path)
    generic_initializations = []
    original_experiment_init = MRIExperiment.__init__

    def counted_experiment_init(self, *args, **kwargs):
        if type(self) is MRIExperiment:
            generic_initializations.append(self)
        original_experiment_init(self, *args, **kwargs)

    monkeypatch.setattr(MRIExperiment, "__init__", counted_experiment_init)

    scan = load_scan(scan_dir, mode="minimal")

    assert generic_initializations == []
    assert type(scan).__name__ == "Singlepulse"
    assert scan.handler.vendor == "magritek"
    assert scan.data.attrs["signal_domain"] == "spectral"


def test_direct_and_study_minimal_loading_are_equivalent(tmp_path):
    from hypermri import Study, load_scan

    scan_dir = _write_magritek_dataset(tmp_path)

    direct_scan = load_scan(scan_dir, mode="minimal")
    study_scan = next(
        iter(
            Study(
                scan_dir,
                verbose=False,
                smart_load=True,
                load_data="minimal",
                sort=None,
            )
        )
    )

    assert type(study_scan) is type(direct_scan)
    assert study_scan.handler.vendor == direct_scan.handler.vendor == "magritek"
    assert study_scan.type == direct_scan.type
    assert study_scan.recon_names == direct_scan.recon_names
    xr.testing.assert_identical(study_scan.data, direct_scan.data)
