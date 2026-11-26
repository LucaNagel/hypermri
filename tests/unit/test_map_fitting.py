"""Quantitative scan/Study contracts on invented diffusion images only.

Route comparisons use rtol=1e-12, atol=1e-14 to detect structural differences.
Recovery from noiseless float64 S=2 exp(-b*0.0012) uses atol=1e-12 mm²/s
(rounding allowance far below the synthetic ADC). Scanner float32 storage
uses rtol=1e-6 for its quantization. Neither is a clinical accuracy claim.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri import Study, load_scan
from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.quantitative import ADCFitOptions, T2FitOptions, fit_adc, fit_t2
from hypermri.sequences import _map_fitting
from hypermri.sequences.base_anatomy import BaseAnatomy
from hypermri.sequences.DTI import DTI
from hypermri.sequences.DTIEPI import DTIEPI
from hypermri.sequences.MSME import MSME

B_VALUES = [0.0, 100.0, 300.0, 700.0, 1000.0]
OPTIONS = ADCFitOptions(execution=ExecutionConfig(False, 1))


def _scan(cls=DTI, *, invalid_voxels=0):
    scan = cls.__new__(cls)
    signal = np.broadcast_to(2 * np.exp(-np.array(B_VALUES) * 0.0012), (2, 5)).copy()
    signal[:invalid_voxels] = 0
    data = xr.DataArray(
        signal,
        dims=("read_idx", "bvalue_idx"),
        coords={"read_idx": [4.0, -2.0], "bvalue_idx": B_VALUES},
        attrs={"unit": "a.u.", "bruker_reconstruction_number": 2},
    )
    data.bvalue_idx.attrs["unit"] = "s/mm2"
    visu = {
        "VisuCoreFrameType": ["MAGNITUDE_IMAGE"],
        "VisuFGOrderDesc": "(5, <FG_DIFFUSION>, <>, 0, 0)",
    }
    method = {
        "PVM_DwEffBval": B_VALUES.copy(),
        "PVM_DwNDiffExp": 5,
        "PVM_DwNDiffDir": 1,
        "PVM_DwDir": [1.0, 0.0, 0.0],
    }
    scan.handler = SimpleNamespace(
        method=method,
        vendor="bruker",
        load_reconstruction_parameters=lambda number: ({}, visu),
    )
    scan.reconstructions = {"seq2d": data}
    return scan, data, visu


def _study(*scans):
    import pandas as pd

    study = Study.__new__(Study)
    study.dataframe = pd.DataFrame(
        {"ScanObject": list(scans)}, index=range(7, 7 + len(scans))
    )
    return study


def _write_diffusion_scan(path, write_parameters, method_name="<Bruker:DtiStandard>"):
    """Use existing public geometry and entirely invented float32 diffusion frames."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "paravision_water_phantom_geometry.json"
    )
    series = json.loads(fixture.read_text())["series"]["head_prone/10"]["paravision"]
    method = {
        **series["method"],
        "Method": method_name,
        "PVM_Matrix": [4, 4],
        "PVM_EncMatrix": [4, 4],
        "PVM_EncNReceivers": 1,
        "PVM_EncSpectroscopy": "No",
        "PVM_NEchoImages": 1,
        "PVM_DwEffBval": B_VALUES,
        "PVM_DwNDiffExp": 5,
        "PVM_DwNDiffDir": 1,
        "PVM_DwDir": [1.0, 0.0, 0.0],
    }
    visu = {
        **series["visu_pars"],
        "VisuCoreSize": [4, 4],
        "VisuCoreDimDesc": ["spatial", "spatial"],
        "VisuCreatorVersion": "7.0.0",
        "VisuCoreWordType": "_32BIT_FLOAT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreFrameCount": 25,
        "VisuFGOrderDescDim": 2,
        "VisuFGOrderDesc": [
            "(5, <FG_SLICE>, <>, 0, 0)",
            "(5, <FG_DIFFUSION>, <>, 0, 0)",
        ],
        "VisuCoreFrameType": ["MAGNITUDE_IMAGE"],
        "VisuCoreDataSlope": [1] * 25,
        "VisuCoreDataOffs": [0] * 25,
    }
    write_parameters(path / "method", method)
    write_parameters(
        path / "acqp",
        {
            **series["acqp"],
            "ACQ_size": [8, 4],
            "NI": 25,
            "NR": 1,
            "ACQ_scan_name": "<synthetic diffusion>",
        },
    )
    write_parameters(path / "visu_pars", visu)
    write_parameters(path / "pdata" / "1" / "visu_pars", visu)
    write_parameters(path / "pdata" / "1" / "reco", series["reco"])
    frames = np.broadcast_to(2 * np.exp(-np.array(B_VALUES) * 0.0012), (4, 4, 5, 5))
    (path / "pdata" / "1" / "2dseq").write_bytes(
        frames.astype("<f4").tobytes(order="F")
    )
    return method, visu


def _write_echo_scan(path, write_parameters, *, reconstruction=1):
    method, visu = _write_diffusion_scan(path, write_parameters, "<Bruker:MSME>")
    method = {
        key: value for key, value in method.items() if not key.startswith("PVM_Dw")
    }
    method.update(PVM_NEchoImages=5, EffectiveTE=[10, 20, 40, 80, 160])
    write_parameters(path / "method", method)
    times_ms = [12, 24, 48, 96, 192] if reconstruction == 2 else method["EffectiveTE"]
    visu["VisuFGOrderDesc"] = ["(5, <FG_SLICE>, <>, 0, 0)", "(5, <FG_ECHO>, <>, 0, 1)"]
    visu["VisuGroupDepVals"] = ["(<VisuAcqEchoTime>, 0)"]
    visu["VisuAcqEchoTime"] = times_ms
    folder = path / "pdata" / str(reconstruction)
    write_parameters(folder / "visu_pars", visu)
    if reconstruction == 2:
        (folder / "reco").write_bytes((path / "pdata" / "1" / "reco").read_bytes())
    else:
        write_parameters(path / "visu_pars", visu)
    signal = 2 * np.exp(-np.array(times_ms) / 80)
    (folder / "2dseq").write_bytes(
        np.broadcast_to(signal, (4, 4, 5, 5)).astype("<f4").tobytes(order="F")
    )


def _echo_scan(invalid_voxels=0):
    scan, data, visu = _scan(MSME, invalid_voxels=invalid_voxels)
    data = data.rename(bvalue_idx="echo_idx").assign_coords(
        echo_idx=("echo_idx", [0.01, 0.02, 0.04, 0.08, 0.16], {"unit": "s"})
    )
    data.values[:] = 2 * np.exp(-data.echo_idx.values / 0.08)
    data.values[:invalid_voxels] = 0
    visu["VisuFGOrderDesc"] = "(5, <FG_ECHO>, <>, 0, 0)"
    scan.handler.method = {"EffectiveTE": [10, 20, 40, 80, 160]}
    scan.reconstructions["seq2d"] = data
    return scan, data, visu


@pytest.mark.parametrize(
    "method_name,cls", [("<Bruker:DtiStandard>", DTI), ("<Bruker:DTIEPI>", DTIEPI)]
)
def test_actual_loading_automatic_manual_and_direct_adc_agree(
    tmp_path, write_paravision_parameters, method_name, cls, monkeypatch
):
    path = tmp_path / "7"
    _write_diffusion_scan(path, write_paravision_parameters, method_name)
    raw_bytes = (path / "pdata" / "1" / "2dseq").read_bytes()
    calls = []
    original_reconstruct = BaseAnatomy.reconstruct

    def only_scanner(self, reco_type, **options):
        assert reco_type == "seq2d"
        calls.append(reco_type)
        return original_reconstruct(self, reco_type, **options)

    monkeypatch.setattr(BaseAnatomy, "reconstruct", only_scanner)
    monkeypatch.setattr(
        BaseAnatomy, "plot", lambda *a, **k: pytest.fail("viewer opened during fitting")
    )
    fits = []

    def counted(data, options):
        fits.append(data)
        return fit_adc(data, options)

    monkeypatch.setattr(_map_fitting, "fit_adc", counted)
    # Thread backend exercises unchanged default worker settings without processes.
    with parallel_config(backend="threading", n_jobs=2):
        automatic = Study(path, auto_fit=["adc", "adc"], verbose=False, sort=None)
        manual = Study(path, load_data="lazy", verbose=False, sort=None)
        assert "Fits" not in manual.dataframe
        assert manual[7].reconstructions == {}
        assert len(fits) == 1
        table = manual.fit_maps(["adc"], adc_options=OPTIONS)
        assert manual.dataframe.columns[-1] == "Measurement Time"
        direct = load_scan(path, mode="minimal")
        returned = direct.fit_maps(["adc"], adc_options=OPTIONS)
    assert isinstance(direct, cls)
    assert len(fits) == 3
    assert len(calls) == 3
    assert returned["adc"] is direct.fit_results["adc"]
    expected = fit_adc(direct.reconstructions["seq2d"], OPTIONS)
    for scan in (automatic[7], manual[7], direct):
        result = scan.fit_results["adc"]
        xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.adc, 0.0012, rtol=1e-6, atol=1e-12)
        assert scan.fit_status["adc"]["state"] == "computed"
        assert result.attrs["fit_reconstruction_type"] == "seq2d"
        assert result.attrs["bruker_reconstruction_number"] == 1
        assert result.attrs["fit_package_version"]
    assert table.iloc[0]["scan_key"] == 7
    assert automatic.dataframe.loc[7, "Fits"] == "ADC computed"
    assert (path / "pdata" / "1" / "2dseq").read_bytes() == raw_bytes


@pytest.mark.parametrize("selection", [False, [], ()])
def test_inactive_automation_keeps_lazy_loading_and_summary(
    tmp_path, write_paravision_parameters, monkeypatch, selection
):
    path = tmp_path / "7"
    _write_diffusion_scan(path, write_paravision_parameters)
    monkeypatch.setattr(
        _map_fitting, "fit_adc", lambda *args: pytest.fail("unexpected fit")
    )
    study = Study(path, auto_fit=selection, load_data=False, verbose=False)
    assert study[7].reconstructions == {}
    assert "Fits" not in study.dataframe
    assert study[7].fit_results == study[7].fit_status == {}


@pytest.mark.parametrize(
    "selection,kwargs,error",
    [
        ("adc", {}, TypeError),
        (None, {}, TypeError),
        (["unknown"], {}, ValueError),
        (["adc", 4], {}, TypeError),
        (True, {"load_data": "lazy"}, ValueError),
        (True, {"load_data": False}, ValueError),
        (True, {"smart_load": False}, ValueError),
    ],
)
def test_constructor_rejects_invalid_settings_before_scan_access(
    monkeypatch, selection, kwargs, error
):
    monkeypatch.setattr(
        Study, "_read_in_measurements", lambda *a, **k: pytest.fail("read scans")
    )
    with pytest.raises(error):
        Study("/synthetic/unread", auto_fit=selection, verbose=False, **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        "bad",
        ADCFitOptions(signal_threshold=np.nan),
        ADCFitOptions(signal_threshold=-1),
        ADCFitOptions(dynamic_dim="other"),
        ADCFitOptions(b_values_s_per_mm2=(0, 1)),
        ADCFitOptions(execution=ExecutionConfig(True, 0)),
    ],
)
def test_invalid_manual_options_preserve_old_results_and_access_no_data(options):
    scan, _, _ = _scan()
    old = xr.Dataset()
    scan.fit_results["adc"] = old
    scan.fit_status["adc"] = {"state": "computed"}
    for caller in (scan, _study(scan)):
        with pytest.raises((TypeError, ValueError)):
            caller.fit_maps(["adc"], adc_options=options)
        assert scan.fit_results["adc"] is old
        assert scan.fit_status["adc"] == {"state": "computed"}


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing_values", "PVM_DwEffBval"),
        ("short_values", "PVM_DwEffBval"),
        ("negative_values", "PVM_DwEffBval"),
        ("nonfinite_values", "PVM_DwEffBval"),
        ("one_value", "one distinct"),
        ("indices", "index"),
        ("reordered", "order"),
        ("missing_dim", "dimension"),
        ("multi_direction", "Multiple diffusion"),
        ("missing_direction", "unambiguously"),
        ("zero_direction", "unambiguously"),
        ("wrong_count", "PVM_DwNDiffExp"),
        ("map", "parameter maps"),
        ("map_frame", "original image"),
        ("missing_source", "identity"),
        ("wrong_source", "disagree"),
        ("wrong_group", "FG_DIFFUSION"),
    ],
)
def test_unsuitable_diffusion_metadata_are_skipped(change, reason, monkeypatch):
    scan, data, visu = _scan()
    method = scan.handler.method
    if change == "missing_values":
        method.pop("PVM_DwEffBval")
    elif change == "short_values":
        method["PVM_DwEffBval"] = [0, 100]
    elif change == "negative_values":
        method["PVM_DwEffBval"] = [0, 100, -3, 700, 1000]
    elif change == "nonfinite_values":
        method["PVM_DwEffBval"] = [0, 100, np.nan, 700, 1000]
    elif change == "one_value":
        method["PVM_DwEffBval"] = [100] * 5
    elif change == "indices":
        scan.reconstructions["seq2d"] = data.assign_coords(bvalue_idx=np.arange(5))
    elif change == "reordered":
        scan.reconstructions["seq2d"] = data.isel(bvalue_idx=slice(None, None, -1))
    elif change == "missing_dim":
        scan.reconstructions["seq2d"] = data.rename(bvalue_idx="other")
    elif change == "multi_direction":
        method["PVM_DwNDiffDir"] = 3
    elif change == "missing_direction":
        method.pop("PVM_DwNDiffDir")
    elif change == "zero_direction":
        method["PVM_DwDir"] = [0, 0, 0]
    elif change == "wrong_count":
        method["PVM_DwNDiffExp"] = 6
    elif change == "map":
        visu["VisuFGOrderDesc"] = "(5, <FG_DTI>, <>, 0, 0)"
    elif change == "map_frame":
        visu["VisuCoreFrameType"] = ["PARAMETER_MAP"]
    elif change == "missing_source":
        data.attrs.pop("bruker_reconstruction_number")
    elif change == "wrong_source":
        data.attrs["bruker_visu_parameter_source"] = "pdata/1/visu_pars"
    elif change == "wrong_group":
        visu["VisuFGOrderDesc"] = "(4, <FG_DIFFUSION>, <>, 0, 0)"
    monkeypatch.setattr(
        _map_fitting, "fit_adc", lambda *a: pytest.fail("unfit data reached fitter")
    )
    assert scan.fit_maps(["adc"]) == {}
    assert scan.fit_status["adc"]["state"] == "skipped"
    assert reason in scan.fit_status["adc"]["reason"]


def test_unsupported_sequence_is_skipped_without_data_access(monkeypatch):
    scan = MSME.__new__(MSME)
    monkeypatch.setattr(
        BaseAnatomy, "reconstruct", lambda *a, **k: pytest.fail("data accessed")
    )
    assert scan.fit_maps(["adc"]) == {}
    assert scan.fit_status["adc"]["state"] == "skipped"


def test_refit_failure_clears_stale_result_and_study_continues(monkeypatch):
    first, _, _ = _scan()
    second, _, _ = _scan(DTIEPI)
    first.fit_maps(["adc"], adc_options=OPTIONS)
    other = xr.Dataset()
    first.fit_results["other"] = other
    real_fit = _map_fitting.fit_adc

    def fail_first(data, options):
        if data is first.reconstructions["seq2d"]:
            raise RuntimeError("invented failure")
        return real_fit(data, options)

    monkeypatch.setattr(_map_fitting, "fit_adc", fail_first)
    with pytest.raises(RuntimeError, match="invented failure") as info:
        first.fit_maps(["adc"], adc_options=OPTIONS)
    assert "seq2d" in info.value.__notes__[0]
    assert "adc" not in first.fit_results
    assert first.fit_results["other"] is other
    study = _study(first, second)
    table = study.fit_maps(["adc"], adc_options=OPTIONS)
    assert len(study) == 2
    assert table.state.tolist() == ["failed", "computed"]
    assert first.fit_status["adc"]["valid_voxels"] is None
    assert study.dataframe.loc[7, "Fits"] == ""
    assert "invented failure" in first.fit_status["adc"]["reason"]


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_user_interrupt_escapes_scan_and_study(monkeypatch, exception):
    scan, _, _ = _scan()

    def interrupt(*args):
        raise exception()

    monkeypatch.setattr(_map_fitting, "fit_adc", interrupt)
    for caller in (scan, _study(scan)):
        with pytest.raises(exception):
            caller.fit_maps(["adc"])


@pytest.mark.parametrize(
    "invalid,expected", [(0, "computed"), (1, "partial"), (2, "failed")]
)
def test_finite_status_and_numerical_recovery_preserve_inputs(invalid, expected):
    scan, data, _ = _scan(invalid_voxels=invalid)
    data = data.expand_dims(channel_idx=[2, 1], repetition_idx=[0, 1]).transpose(
        "bvalue_idx", "read_idx", "repetition_idx", "channel_idx"
    )
    scan.reconstructions["seq2d"] = data
    before = data.copy(deep=True)
    result = scan.fit_maps(["adc"], adc_options=OPTIONS)["adc"]
    xr.testing.assert_allclose(result, fit_adc(data, OPTIONS), rtol=1e-12, atol=1e-14)
    assert result.adc.dims == ("read_idx", "repetition_idx", "channel_idx")
    assert result.attrs["bruker_reconstruction_number"] == 2
    assert result.attrs["fit_prepared_options"]["signal_threshold"] == 1e-5
    assert scan.fit_status["adc"]["state"] == expected
    assert scan.fit_status["adc"]["valid_voxels"] == (2 - invalid) * 4
    xr.testing.assert_identical(data, before)
    if invalid < 2:
        np.testing.assert_allclose(
            result.adc.isel(read_idx=-1), 0.0012, rtol=0, atol=1e-12
        )
    other, _, _ = _scan()
    assert other.fit_results == other.fit_status == {}
    assert other.fit_results is not scan.fit_results


def test_constructor_fit_error_keeps_loaded_scans(
    tmp_path, write_paravision_parameters, monkeypatch
):
    for key in (7, 8):
        _write_diffusion_scan(tmp_path / str(key), write_paravision_parameters)

    def fail(*args):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(_map_fitting, "fit_adc", fail)
    study = Study(tmp_path, auto_fit=True, verbose=False, sort=None)
    assert len(study) == 2
    assert all(scan.fit_status["adc"]["state"] == "failed" for scan in study)


def test_actual_msme_automatic_manual_direct_and_numerical_t2_agree(
    tmp_path, write_paravision_parameters, monkeypatch
):
    path = tmp_path / "7"
    _write_echo_scan(path, write_paravision_parameters)
    calls = []
    original = BaseAnatomy.reconstruct

    def only_scanner(self, reco_type, **kwargs):
        assert reco_type == "seq2d"
        calls.append(reco_type)
        return original(self, reco_type, **kwargs)

    monkeypatch.setattr(BaseAnatomy, "reconstruct", only_scanner)
    monkeypatch.setattr(
        BaseAnatomy, "plot", lambda *a, **k: pytest.fail("viewer opened during fitting")
    )
    options = T2FitOptions(execution=ExecutionConfig(False, 1))
    fits = []

    def counted(data, options):
        fits.append(data)
        return fit_t2(data, options)

    monkeypatch.setattr(_map_fitting, "fit_t2", counted)
    with parallel_config(backend="threading", n_jobs=2):
        automatic = Study(path, auto_fit=["t2", "t2"], verbose=False)
        manual = Study(path, load_data="lazy", verbose=False)
        table = manual.fit_maps(["t2"], t2_options=options)
        direct = load_scan(path, mode="minimal")
        result = direct.fit_maps(["t2"], t2_options=options)["t2"]
    expected = fit_t2(direct.reconstructions["seq2d"], options)
    for scan in (automatic[7], manual[7], direct):
        xr.testing.assert_allclose(
            scan.fit_results["t2"], expected, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(scan.fit_results["t2"].t2, 80, rtol=1e-6, atol=1e-6)
    assert result is direct.fit_results["t2"]
    assert table.state.tolist() == ["computed"]
    assert len(calls) == len(fits) == 3
    assert automatic.dataframe.loc[7, "Fits"] == "T2 computed"


def test_selected_scanner_echo_times_override_method_without_guessed_source(
    tmp_path, write_paravision_parameters
):
    path = tmp_path / "7"
    _write_echo_scan(path, write_paravision_parameters, reconstruction=2)
    scan = load_scan(path, mode="lazy")
    data = scan.reconstruct("seq2d", recon_num=2)
    result = scan.fit_maps(
        ["t2"], t2_options=T2FitOptions(execution=ExecutionConfig(False, 1))
    )["t2"]
    assert result.attrs["bruker_reconstruction_number"] == 2
    assert result.attrs["bruker_visu_parameter_source"] == "pdata/2/visu_pars"
    np.testing.assert_allclose(
        result.echo_times_s, [0.012, 0.024, 0.048, 0.096, 0.192], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(result.t2, 80, rtol=1e-6)
    assert scan.reconstructions["seq2d"] is data


@pytest.mark.parametrize(
    "change,reason",
    [
        ("single", "three"),
        ("missing", "EffectiveTE"),
        ("wrong", "differ"),
        ("indices", "unit"),
        ("map", "parameter maps"),
        ("missing_group", "FG_ECHO"),
        ("nonfinite", "finite"),
        ("short_times", "EffectiveTE"),
    ],
)
def test_unsuitable_msme_echo_metadata_are_skipped(change, reason, monkeypatch):
    scan, data, visu = _echo_scan()
    if change == "single":
        scan.reconstructions["seq2d"] = data.isel(echo_idx=slice(1))
        visu["VisuFGOrderDesc"] = "(1, <FG_ECHO>, <>, 0, 0)"
    elif change == "missing":
        scan.handler.method = {}
    elif change == "wrong":
        scan.handler.method["EffectiveTE"] = [11, 22, 44, 88, 176]
    elif change == "indices":
        data.echo_idx.attrs["unit"] = "index"
    elif change == "map":
        visu["VisuFGOrderDesc"] = "(5, <FG_ISA>, <>, 0, 0)"
    elif change == "missing_group":
        visu["VisuFGOrderDesc"] = ""
    elif change == "nonfinite":
        scan.reconstructions["seq2d"] = data.assign_coords(
            echo_idx=("echo_idx", [0.01, np.nan, 0.04, 0.08, 0.16], {"unit": "s"})
        )
    elif change == "short_times":
        scan.handler.method["EffectiveTE"] = [10, 20]
    monkeypatch.setattr(
        _map_fitting, "fit_t2", lambda *a: pytest.fail("unsuitable data fitted")
    )
    assert scan.fit_maps(["t2"]) == {}
    assert scan.fit_status["t2"]["state"] == "skipped"
    assert reason in scan.fit_status["t2"]["reason"]


@pytest.mark.parametrize(
    "invalid,state", [(0, "computed"), (1, "partial"), (2, "failed")]
)
def test_t2_status_input_and_dimension_preservation(invalid, state):
    scan, data, _ = _echo_scan(invalid)
    data = data.expand_dims(channel_idx=[2, 1], repetition_idx=[0, 1]).transpose(
        "echo_idx", "read_idx", "repetition_idx", "channel_idx"
    )
    scan.reconstructions["seq2d"] = data
    before = data.copy(deep=True)
    result = scan.fit_maps(
        ["t2"], t2_options=T2FitOptions(execution=ExecutionConfig(False, 1))
    )["t2"]
    assert result.t2.dims == ("read_idx", "repetition_idx", "channel_idx")
    assert scan.fit_status["t2"]["state"] == state
    assert scan.fit_status["t2"]["valid_voxels"] == (2 - invalid) * 4
    xr.testing.assert_identical(data, before)
    study = _study(scan)
    study.fit_maps(["t2"], t2_options=T2FitOptions(execution=ExecutionConfig(False, 1)))
    if invalid == 1:
        assert "4/8" in study.dataframe.loc[7, "Fits"]


def test_t2_failure_reaches_direct_caller_and_study_continues(monkeypatch):
    first, _, _ = _echo_scan()
    second, _, _ = _echo_scan()
    options = T2FitOptions(execution=ExecutionConfig(False, 1))
    first.fit_maps(["t2"], t2_options=options)
    original = _map_fitting.fit_t2

    def fail(data, options):
        if data is first.reconstructions["seq2d"]:
            raise RuntimeError("synthetic T2 failure")
        return original(data, options)

    monkeypatch.setattr(_map_fitting, "fit_t2", fail)
    with pytest.raises(RuntimeError):
        first.fit_maps(["t2"], t2_options=options)
    assert "t2" not in first.fit_results
    assert first.fit_status["t2"]["state"] == "failed"
    assert _study(first, second).fit_maps(
        ["t2"], t2_options=options
    ).state.tolist() == ["failed", "computed"]


def test_selection_all_models_and_invalid_t2_options_validate_before_changes():
    scan, _, _ = _echo_scan()
    assert (
        _map_fitting._select_models(None)
        == _map_fitting._select_models(True, automatic=True)
        == ("adc", "t2")
    )
    for options in (
        "invalid",
        T2FitOptions(signal_threshold=-1),
        T2FitOptions(dynamic_dim="other"),
        T2FitOptions(execution=ExecutionConfig(True, 0)),
    ):
        old = xr.Dataset()
        scan.fit_results["t2"] = old
        for caller in (scan, _study(scan)):
            with pytest.raises((TypeError, ValueError)):
                caller.fit_maps(["t2"], t2_options=options)
            assert scan.fit_results["t2"] is old


def test_mixed_study_fits_only_each_suitable_model_once(
    tmp_path, write_paravision_parameters
):
    _write_diffusion_scan(tmp_path / "7", write_paravision_parameters)
    _write_echo_scan(tmp_path / "8", write_paravision_parameters)
    with parallel_config(backend="threading", n_jobs=2):
        study = Study(tmp_path, auto_fit=True, sort=None, verbose=False)
    assert set(study[7].fit_results) == {"adc"}
    assert set(study[8].fit_results) == {"t2"}
    assert (
        study[7].fit_status["t2"]["state"]
        == study[8].fit_status["adc"]["state"]
        == "skipped"
    )
    assert study.dataframe["Fits"].tolist() == ["ADC computed", "T2 computed"]
    study._update_summary_columns()
    assert study.dataframe["Fits"].tolist() == ["ADC computed", "T2 computed"]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (
            {"adc": _map_fitting._status("computed", valid=128, total=128)},
            "ADC computed",
        ),
        (
            {"adc": _map_fitting._status("partial", valid=120, total=128)},
            "ADC partially computed: 120/128 voxels finite",
        ),
        (
            {
                "adc": _map_fitting._status("skipped", "Unsuitable series"),
                "t2": _map_fitting._status("failed", "No finite values"),
            },
            "",
        ),
        (
            {
                "adc": _map_fitting._status("computed"),
                "t2": _map_fitting._status("failed", "Failed"),
            },
            "ADC computed",
        ),
        ({}, ""),
    ],
)
def test_fit_summary_lists_only_computed_maps_in_english_without_data_access(
    statuses, expected
):
    scan = BaseAnatomy.__new__(BaseAnatomy)
    scan._fit_status = statuses
    before = dict(vars(scan))
    assert _map_fitting._fit_summary(scan) == expected
    assert vars(scan) == before


def test_scan_and_study_t2_maps_use_five_second_optimizer_bound():
    scan, data, _ = _echo_scan()
    data.values[:] = 2
    options = T2FitOptions(execution=ExecutionConfig(False, 1))
    direct = scan.fit_maps(["t2"], t2_options=options)["t2"]
    assert np.isfinite(direct.t2).all()
    assert float(direct.t2.max()) <= 5000.0
    assert direct.attrs["maximum_t2_s"] == 5.0
    table = _study(scan).fit_maps(["t2"], t2_options=options)
    assert table.state.tolist() == ["computed"]
    xr.testing.assert_allclose(scan.fit_results["t2"], direct, rtol=1e-12, atol=1e-14)
