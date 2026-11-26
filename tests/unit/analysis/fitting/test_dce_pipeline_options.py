"""Public pipeline option contracts on invented labels and stubbed orchestration.

Actual model numerics are covered by the synthetic renal/DCE workflow tests.
This module protects forwarding, precedence and early unknown-name rejection.
"""

import inspect

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.dce import pipeline


def _capture_pipeline(monkeypatch):
    calls = []
    output = xr.Dataset(attrs={"invented_fixture": True})

    def capture(data, **kwargs):
        calls.append((data, kwargs))
        return output

    monkeypatch.setattr(pipeline, "_run_dce_pipeline_core", capture)
    data = xr.DataArray(
        [1.0, 2.0], dims="repetition_idx", coords={"repetition_idx": [5, 2]}
    )
    return data, calls, output


def test_auxiliary_options_reach_existing_pipeline_once_and_preserve_inputs(
    monkeypatch,
):
    data, calls, output = _capture_pipeline(monkeypatch)
    t1 = xr.DataArray([1000.0], dims="read_idx")
    masks = {"invented_roi": np.array([True])}
    conversion = {"gfr_factor": 1.0}
    motion = {"correction_method": "skimage"}
    fit_options = {"n_precontrast": 1}
    before = data.copy(deep=True)
    actual = pipeline.run_dce_pipeline(
        data,
        fit_type="patlak",
        fitted_t1=t1,
        motion_correct_dce=motion,
        renal_masks=masks,
        gfr_conversion=conversion,
        aif_t1_ms=1500.0,
        aif_t1_s=1.5,
        fit_params_pk=fit_options,
    )
    assert actual is output and len(calls) == 1
    forwarded = calls[0][1]
    assert forwarded["fitted_t1"] is t1
    assert forwarded["motion_corrector"] is motion
    assert forwarded["renal_masks"] is masks
    assert forwarded["gfr_conversion"] is conversion
    assert forwarded["aif_t1_ms"] == 1500.0
    assert forwarded["aif_t1_s"] == 1.5
    assert fit_options == {"n_precontrast": 1}
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("canonical", [None, False, True])
def test_blood_alias_only_fills_an_unspecified_canonical_option(monkeypatch, canonical):
    data, calls, _ = _capture_pipeline(monkeypatch)
    pipeline.run_dce_pipeline(
        data,
        fit_type="patlak",
        aif_convert_blood_to_plasma=canonical,
        blood_to_plasma=True,
    )
    assert calls[0][1]["fit_params_pk"]["aif_convert_blood_to_plasma"] is (
        True if canonical is None else canonical
    )


@pytest.mark.parametrize("canonical", [None, 0.37])
def test_hematocrit_alias_preserves_canonical_precedence(monkeypatch, canonical):
    data, calls, _ = _capture_pipeline(monkeypatch)
    pipeline.run_dce_pipeline(
        data,
        fit_type="patlak",
        aif_hematocrit=canonical,
        hematocrit=0.42,
    )
    assert calls[0][1]["fit_params_pk"]["aif_hematocrit"] == (
        0.42 if canonical is None else canonical
    )


@pytest.mark.parametrize("explicit", [None, False, True])
def test_quality_skip_precedence_is_preserved(monkeypatch, explicit):
    data, calls, _ = _capture_pipeline(monkeypatch)
    pipeline.run_dce_pipeline(
        data,
        fit_type="patlak",
        fit_params_pk={"skip_fit_quality_metrics": True},
        skip_fit_quality_metrics=explicit,
    )
    assert calls[0][1]["skip_fit_quality_metrics"] is (
        True if explicit is None else explicit
    )


@pytest.mark.parametrize(
    "name", ["fited_t1", "aif_t1", "motion_corrector", "gfr_converison"]
)
def test_unknown_pipeline_option_is_rejected_before_processing(monkeypatch, name):
    data, calls, _ = _capture_pipeline(monkeypatch)
    with pytest.raises(TypeError, match=name):
        pipeline.run_dce_pipeline(data, fit_type="patlak", **{name: 1})
    assert calls == []


def test_pipeline_signature_exposes_options_without_variadic_keywords():
    params = inspect.signature(pipeline.run_dce_pipeline).parameters
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())
    for name in (
        "fitted_t1",
        "motion_correct_dce",
        "renal_masks",
        "gfr_conversion",
        "aif_t1_ms",
        "aif_t1_s",
        "blood_to_plasma",
        "hematocrit",
    ):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
