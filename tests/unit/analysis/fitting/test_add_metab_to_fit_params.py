import warnings

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import add_metab_to_fit_params


def _make_fit_datasets():
    repetitions = np.arange(4, dtype=int)
    metabolites = ["pyruvate"]
    coords = {"repetition_idx": repetitions, "metabolite": metabolites}
    template_shape = (len(repetitions), len(metabolites))

    fit_params = xr.Dataset(
        {
            "initial_freqs_hz": xr.DataArray(
                np.full(template_shape, 100.0, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "initial_t2s_s": xr.DataArray(
                np.full(template_shape, 0.03, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "metabs_freq_ranges": xr.DataArray(
                np.full(template_shape, 40.0, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "metabs_t2_ranges": xr.DataArray(
                np.full(template_shape, 0.02, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "min_t2_s": xr.DataArray(
                np.full(template_shape, 0.005, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "max_t2_s": xr.DataArray(
                np.full(template_shape, 0.1, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "freq_range_lower_bound_hz": xr.DataArray(
                np.full(template_shape, 80.0, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "freq_range_upper_bound_hz": xr.DataArray(
                np.full(template_shape, 120.0, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
        },
        attrs={"metabs": metabolites},
    )

    fit_results = xr.Dataset(
        {
            "fit_amps": xr.DataArray(
                np.array([[1.0 + 0.0j], [8.0 + 0.0j], [3.0 + 0.0j], [2.0 + 0.0j]]),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "fit_freqs": xr.DataArray(
                np.array([[100.0], [200.0], [300.0], [800.0]], dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
            "fit_t2s": xr.DataArray(
                np.full(template_shape, 0.025, dtype=float),
                dims=("repetition_idx", "metabolite"),
                coords=coords,
            ),
        },
        attrs={"metabs": metabolites},
    )

    return fit_params, fit_results


def test_add_metab_to_fit_params_rejects_frequency_and_reference_together():
    fit_params, _ = _make_fit_datasets()

    with pytest.raises(ValueError, match="mutually exclusive"):
        add_metab_to_fit_params(
            new_metab_name="lactate",
            new_freq=183.2,
            reference_metabolite="pyruvate",
            fit_params=fit_params,
        )


@pytest.mark.parametrize(
    ("extra_kwargs", "error_match"),
    [
        ({"freq_offset": 20.0}, "freq_offset requires reference_metabolite"),
        (
            {"reference_repetitions": 1},
            "reference_repetitions requires reference_metabolite",
        ),
    ],
)
def test_add_metab_to_fit_params_rejects_reference_options_without_reference(
    extra_kwargs,
    error_match,
):
    fit_params, _ = _make_fit_datasets()

    with pytest.raises(ValueError, match=error_match):
        add_metab_to_fit_params(
            new_metab_name="lactate",
            new_freq=120.0,
            fit_params=fit_params,
            **extra_kwargs,
        )


def test_add_metab_to_fit_params_rejects_duplicate_metabolite_name():
    fit_params, _ = _make_fit_datasets()

    with pytest.raises(ValueError, match="already present"):
        add_metab_to_fit_params(
            new_metab_name="pyruvate",
            new_freq=120.0,
            fit_params=fit_params,
        )


@pytest.mark.parametrize(
    ("invalid_kwargs", "error_match"),
    [
        ({"new_freq_range": -1.0}, "new_freq_range must be non-negative"),
        ({"new_t2_range_s": -0.01}, "new_t2_range_s must be non-negative"),
        ({"new_min_t2_s": 0.0}, "new_min_t2_s must be positive"),
        (
            {"new_min_t2_s": 0.05, "new_max_t2_s": 0.04},
            "new_max_t2_s must be greater",
        ),
        (
            {"new_t2_s": 0.2, "new_max_t2_s": 0.1},
            "new_t2_s must lie within",
        ),
        ({"new_freq_range": np.inf}, "new_freq_range must be a finite scalar"),
    ],
)
def test_add_metab_to_fit_params_rejects_invalid_ranges_and_bounds(
    invalid_kwargs,
    error_match,
):
    fit_params, _ = _make_fit_datasets()

    with pytest.raises(ValueError, match=error_match):
        add_metab_to_fit_params(
            new_metab_name="lactate",
            new_freq=120.0,
            fit_params=fit_params,
            **invalid_kwargs,
        )


def test_add_metab_to_fit_params_keeps_default_per_repetition_reference():
    fit_params, fit_results = _make_fit_datasets()

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params,
        fit_results=fit_results,
        freq_offset=20.0,
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    lactate_freqs = updated["initial_freqs_hz"].sel(metabolite="lactate").values
    lactate_t2_range = updated["metabs_t2_ranges"].sel(metabolite="lactate")
    assert np.allclose(lactate_freqs, [120.0, 220.0, 320.0, 820.0])
    assert np.allclose(lactate_t2_range, 0.04)


def test_add_metab_to_fit_params_can_use_single_reference_repetition():
    fit_params, fit_results = _make_fit_datasets()

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params,
        fit_results=fit_results,
        freq_offset=20.0,
        reference_repetitions=1,
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    lactate_freqs = updated["initial_freqs_hz"].sel(metabolite="lactate").values
    assert np.allclose(lactate_freqs, [220.0, 220.0, 220.0, 220.0])


def test_add_metab_to_fit_params_can_use_reference_repetition_range():
    fit_params, fit_results = _make_fit_datasets()

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params,
        fit_results=fit_results,
        freq_offset=20.0,
        reference_repetitions=range(1, 3),
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    lactate_freqs = updated["initial_freqs_hz"].sel(metabolite="lactate").values
    assert np.allclose(lactate_freqs, [270.0, 270.0, 270.0, 270.0])


def test_add_metab_to_fit_params_can_use_max_fit_amp_reference_repetition():
    fit_params, fit_results = _make_fit_datasets()

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params,
        fit_results=fit_results,
        freq_offset=20.0,
        reference_repetitions="max_fit_amp",
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    lactate_freqs = updated["initial_freqs_hz"].sel(metabolite="lactate").values
    assert np.allclose(lactate_freqs, [220.0, 220.0, 220.0, 220.0])


def test_add_metab_to_fit_params_warns_for_large_ppm_freq_range():
    fit_params, fit_results = _make_fit_datasets()
    fit_params["freq_axis_hz"] = xr.DataArray(
        np.array([-1000.0, 0.0, 1000.0], dtype=float),
        dims=("spectral_idx",),
    )
    fit_params["freq_axis_ppm"] = xr.DataArray(
        np.array([-10.0, 0.0, 10.0], dtype=float),
        dims=("spectral_idx",),
    )

    with pytest.warns(UserWarning, match="larger than 1 ppm"):
        add_metab_to_fit_params(
            new_metab_name="lactate",
            reference_metabolite="pyruvate",
            new_freq=None,
            new_t2_s=0.02,
            fit_params=fit_params,
            fit_results=fit_results,
            new_freq_range=2.0,
            new_freq_unit="ppm",
            freq_offset=0.1,
            freq_offset_unit="ppm",
            freeze_other_freqs=True,
            freeze_other_t2s=False,
        )


def test_add_metab_to_fit_params_keeps_variables_without_metabolite_unchanged():
    """A variable without ``metabolite`` keeps its dimensions and values.

    Here a frequency axis stored as a data variable stands for any shared
    variable; the Gaussian start map of a Voigt fit is the case in use.
    """
    fit_params, _ = _make_fit_datasets()
    axis_hz = np.array([-1000.0, 0.0, 1000.0])
    fit_params["freq_axis_hz"] = xr.DataArray(axis_hz, dims=("spectral_idx",))
    fit_params["initial_gauss_fwhm_hz"] = xr.DataArray(
        np.array([4.0, 5.0, 6.0, 7.0]),
        dims=("repetition_idx",),
        coords={"repetition_idx": fit_params.repetition_idx},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        updated = add_metab_to_fit_params(
            new_metab_name="lactate",
            new_freq=120.0,
            fit_params=fit_params,
            new_freq_range=200.0,
        )

    xr.testing.assert_identical(updated["freq_axis_hz"], fit_params["freq_axis_hz"])
    xr.testing.assert_identical(
        updated["initial_gauss_fwhm_hz"], fit_params["initial_gauss_fwhm_hz"]
    )


def test_add_metab_to_fit_params_extends_every_metabolite_variable():
    """Per-line variables keep the old lines and gain the new line.

    The old lines are not frozen, so their search ranges keep their values.

    The expected values of the new line are the ones the concatenation of
    every variable gave before shared variables were left out of it.
    """
    fit_params, _ = _make_fit_datasets()
    fit_params["initial_amplitudes"] = xr.full_like(fit_params["initial_freqs_hz"], 3.0)

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        new_freq=120.0,
        fit_params=fit_params,
        new_freq_range=200.0,
        freeze_other_freqs=False,
        freeze_other_t2s=False,
    )

    assert list(updated.metabolite.values) == ["pyruvate", "lactate"]
    for name, variable in fit_params.data_vars.items():
        assert updated[name].dims == variable.dims
        xr.testing.assert_identical(
            updated[name].sel(metabolite=["pyruvate"]), variable
        )
    new_line = updated.sel(metabolite="lactate")
    np.testing.assert_allclose(new_line["initial_freqs_hz"], 120.0)
    np.testing.assert_allclose(new_line["metabs_freq_ranges"], 200.0)
    np.testing.assert_allclose(new_line["freq_range_lower_bound_hz"], 20.0)
    np.testing.assert_allclose(new_line["freq_range_upper_bound_hz"], 220.0)
    assert np.isnan(new_line["initial_amplitudes"]).all()


def test_add_metab_to_fit_params_freezing_keeps_the_range_dimension_order():
    """Frozen old lines get zero ranges in the input's dimension order."""
    fit_params, _ = _make_fit_datasets()

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        new_freq=120.0,
        fit_params=fit_params,
        new_freq_range=200.0,
        new_t2_range_s=0.04,
    )

    for name, new_range in (("metabs_freq_ranges", 200.0), ("metabs_t2_ranges", 0.04)):
        assert updated[name].dims == ("repetition_idx", "metabolite")
        np.testing.assert_array_equal(updated[name].sel(metabolite="pyruvate"), 0.0)
        np.testing.assert_allclose(updated[name].sel(metabolite="lactate"), new_range)


def test_add_metab_to_fit_params_promotes_scalar_global_constraint_to_attribute():
    """Keep a global peak-separation constraint out of metabolite parameter maps."""
    fit_params, _ = _make_fit_datasets()
    fit_params.attrs["min_peak_separation_hz"] = 5.0
    fit_params["min_peak_separation_hz"] = 15.0
    original = fit_params.copy(deep=True)

    updated = add_metab_to_fit_params(
        new_metab_name="lactate",
        new_freq=120.0,
        fit_params=fit_params,
        new_freq_range=200.0,
    )

    assert "min_peak_separation_hz" not in updated.data_vars
    assert updated.attrs["min_peak_separation_hz"] == pytest.approx(15.0)
    xr.testing.assert_identical(fit_params, original)


def test_add_metab_to_fit_params_preserves_existing_added_metabs_with_older_results():
    fit_params, fit_results = _make_fit_datasets()

    fit_params_lactate = add_metab_to_fit_params(
        new_metab_name="lactate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params,
        fit_results=fit_results,
        freq_offset=20.0,
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    fit_params_lactate_pyruvate_hydrate = add_metab_to_fit_params(
        new_metab_name="pyruvatehydrate",
        reference_metabolite="pyruvate",
        new_freq=None,
        new_t2_s=0.02,
        fit_params=fit_params_lactate,
        fit_results=fit_results,
        freq_offset=5.0,
        freeze_other_freqs=True,
        freeze_other_t2s=False,
    )

    lactate_freqs = fit_params_lactate_pyruvate_hydrate["initial_freqs_hz"].sel(
        metabolite="lactate"
    )
    pyruvatehydrate_freqs = fit_params_lactate_pyruvate_hydrate["initial_freqs_hz"].sel(
        metabolite="pyruvatehydrate"
    )
    lactate_t2s = fit_params_lactate_pyruvate_hydrate["initial_t2s_s"].sel(
        metabolite="lactate"
    )

    assert np.allclose(lactate_freqs.values, [120.0, 220.0, 320.0, 820.0])
    assert np.allclose(pyruvatehydrate_freqs.values, [105.0, 205.0, 305.0, 805.0])
    assert np.allclose(lactate_t2s.values, 0.02)


def test_add_metab_to_fit_params_rejects_start_closer_than_peak_separation():
    """Reject a new start the fit would reject, using the default 5 Hz limit."""
    fit_params, _ = _make_fit_datasets()
    original = fit_params.copy(deep=True)
    with pytest.raises(
        ValueError,
        match=r"add_metab_to_fit_params: .*'pyruvate' \(100 Hz\) and 'lactate' "
        r"\(103 Hz\) are 3 Hz apart.*min_peak_separation_hz=5 Hz "
        r"\(4 of 4 spectra affected\).*Move the start frequencies apart",
    ):
        add_metab_to_fit_params(
            new_metab_name="lactate", new_freq=103.0, fit_params=fit_params
        )
    xr.testing.assert_identical(fit_params, original)


def test_add_metab_to_fit_params_checks_refreshed_starts_per_spectrum():
    """Compare the new start with the fitted frequencies of every repetition."""
    fit_params, fit_results = _make_fit_datasets()
    with pytest.raises(
        ValueError,
        match=r"'pyruvate' \(200 Hz\) and 'lactate' \(203 Hz\).*"
        r"\{'repetition_idx': 1\}.*\(1 of 4 spectra affected\)",
    ):
        add_metab_to_fit_params(
            new_metab_name="lactate",
            new_freq=203.0,
            fit_params=fit_params,
            fit_results=fit_results,
        )


def test_add_metab_to_fit_params_ignores_skipped_lines_for_separation():
    """A skipped line is not fitted, so its start cannot block a new line."""
    fit_params, _ = _make_fit_datasets()
    fit_params["skip_fitting"] = xr.full_like(
        fit_params["initial_freqs_hz"], True, dtype=bool
    )
    updated = add_metab_to_fit_params(
        new_metab_name="lactate", new_freq=103.0, fit_params=fit_params
    )
    np.testing.assert_allclose(
        updated["initial_freqs_hz"].sel(metabolite="lactate"), 103.0
    )


# Adding a line through the single owner. The fixture is a synthetic
# three-spectrum FID template (3 kHz bandwidth, 64 points, no ppm axis); only
# its sampling coordinates are used, so the comparisons are exact.
def _template_data():
    t = np.arange(64) / 3000.0
    return xr.DataArray(
        np.zeros((3, 64), dtype=complex),
        dims=("x", "acquisition_idx"),
        coords={
            "x": np.arange(3),
            "acquisition_idx": np.arange(64),
            "sampling_time_axis_s": ("acquisition_idx", t),
        },
        attrs={"signal_domain": "temporal"},
    )


def _prepared(metabs, freqs_hz, **options):
    from hypermri.analysis.fitting.spectroscopy import def_fit_params

    return def_fit_params(
        _template_data(),
        {"metabs": metabs, "metabs_freqs_hz": freqs_hz, **options},
    )


@pytest.mark.parametrize(
    "options",
    [{}, {"lineshape": "voigt", "gauss_initial_fwhm_hz": 10.0}],
    ids=["lorentz", "voigt"],
)
def test_adding_a_line_equals_preparing_the_extended_set(options):
    base = _prepared(["a", "b"], [100.0, -200.0], t2_coupling={"b": "a"}, **options)
    original = base.copy(deep=True)

    added = add_metab_to_fit_params(
        400.0,
        new_metab_name="c",
        fit_params=base,
        freeze_other_freqs=False,
        freeze_other_t2s=False,
        t2_coupling=("a", (0.5, 2.0)),
    )
    expected = _prepared(
        ["a", "b", "c"],
        [100.0, -200.0, 400.0],
        t2_coupling={"b": "a", "c": ("a", (0.5, 2.0))},
        **options,
    )

    xr.testing.assert_equal(added, expected)
    # The fit options include arrays (time and frequency axes).
    np.testing.assert_equal(added.attrs, expected.attrs)
    xr.testing.assert_equal(base, original)
    np.testing.assert_equal(base.attrs, original.attrs)


def test_added_line_takes_the_def_fit_params_defaults():
    base = _prepared(["a"], [100.0])

    added = add_metab_to_fit_params(-300.0, new_metab_name="b", fit_params=base)

    line = added.sel(metabolite="b")
    np.testing.assert_allclose(line["metabs_freq_ranges"], 70.0)
    np.testing.assert_allclose(line["initial_t2s_s"], 0.025)
    np.testing.assert_allclose(line["metabs_t2_ranges"], 0.04)
    np.testing.assert_allclose(line["min_t2_s"], 0.005)
    np.testing.assert_allclose(line["max_t2_s"], 0.07)

    voigt = _prepared(["a"], [100.0], lineshape="voigt", gauss_initial_fwhm_hz=10.0)
    voigt_added = add_metab_to_fit_params(-300.0, new_metab_name="b", fit_params=voigt)
    np.testing.assert_allclose(voigt_added["max_t2_s"].sel(metabolite="b"), 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"new_freq": 100.0, "new_metab_name": "a"}, "already present"),
        ({"new_freq": 1600.0}, "outside the frequency axis"),
        ({"new_freq": 0.0, "new_freq_range": 3100.0}, "wider than the spectral"),
        ({"new_freq": 0.0, "t2_coupling": "unknown"}, "not fitted metabolites"),
        ({"new_freq": 0.0, "t2_coupling": "b"}, "both leader and dependent"),
        ({"new_freq": 0.0, "t2_coupling": ("a", -1.0)}, "must be positive"),
    ],
)
def test_adding_a_line_fails_early_on_inconsistent_input(kwargs, message):
    base = _prepared(["a", "b"], [100.0, -200.0], t2_coupling={"b": "a"})
    original = base.copy(deep=True)
    options = {"new_metab_name": "c", **kwargs}

    with pytest.raises(ValueError, match=message):
        add_metab_to_fit_params(fit_params=base, **options)
    xr.testing.assert_equal(base, original)
    assert base.attrs["t2_coupling"] == {"b": ("a", 1.0)}
