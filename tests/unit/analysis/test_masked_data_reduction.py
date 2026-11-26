"""Invented phase/coverage references for spatial ROI extraction.

The exact quadratures 3+4j and -(3+4j) have magnitude 5 a.u. Fractional
weights 0.5 and 1 give a coherent sum -(3+4j)/2 and mean -(3+4j)/3,
or a magnitude sum 7.5 and mean 5. Float64 accumulation tolerances are
1e-12 a.u.; no scanner data or fitted tolerances are used.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical import masks

SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _signal(dtype=np.complex64):
    scale = np.array([[1, 2], [2, 4], [3, 6]], dtype=np.float64)
    quadrature = np.array([3 + 4j, -3 - 4j, 15 + 20j])
    if np.issubdtype(dtype, np.floating):
        quadrature = quadrature.real
    values = (quadrature[:, None, None, None, None] * scale[None, None, None]).astype(
        dtype
    )
    data = xr.DataArray(
        values,
        dims=(*SPATIAL_DIMS, "acquisition_idx", "repetition_idx"),
        coords={
            "read_idx": [2, 1, 0],
            "phase_idx": [0],
            "slice_idx": [4],
            "acquisition_idx": [0.0, 0.001, 0.002],
            "repetition_idx": [0, 1],
        },
        name="invented_fid",
        attrs={"units": "a.u.", "signal_domain": "temporal", "provenance": "invented"},
    )
    data.acquisition_idx.attrs["units"] = "s"
    for name, dim in zip(("x", "y", "z"), SPATIAL_DIMS, strict=True):
        data = data.assign_coords({name: (dim, data[dim].values.astype(float))})
    weight = xr.DataArray(
        np.array([0.5, 1.0, 0.0], dtype=np.float32).reshape(3, 1, 1),
        dims=SPATIAL_DIMS,
        coords={dim: data.coords[dim] for dim in SPATIAL_DIMS},
        name="roi",
    )
    return data, weight, scale


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("mean", [False, True])
def test_legacy_projected_mask_retains_weighting_precision_and_inputs(dtype, mean):
    data, weight, scale = _signal(dtype)
    before, weight_before = data.copy(deep=True), weight.copy(deep=True)
    result, projected = masks.apply_projected_mask(
        data,
        weight,
        source_data=data,
        coverage_samples=1,
        mean_along_space=mean,
        return_mask=True,
    )
    xr.testing.assert_identical(
        projected.drop_vars(["x", "y", "z"]), weight.assign_attrs(projected.attrs)
    )
    quadrature = 3 + 4j if np.issubdtype(dtype, np.complexfloating) else 3
    expected = (
        -quadrature / 3 * scale
        if mean
        else np.array([quadrature / 2, -quadrature, 0])[:, None, None, None, None]
        * scale[None, None, None]
    )
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12)
    assert result.dtype == (np.complex128 if np.iscomplexobj(data) else np.float64)
    assert result.dims == (("acquisition_idx", "repetition_idx") if mean else data.dims)
    assert result.attrs == ({} if mean else {**data.attrs, "applied_mask": "roi"})
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(weight, weight_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
@pytest.mark.parametrize("permuted", [False, True])
def test_explicit_reductions_preserve_labels_units_precision_and_inputs(
    dtype, reduction, signal_part, permuted
):
    data, weight, scale = _signal(dtype)
    if permuted:
        data = data.transpose(
            "repetition_idx", "phase_idx", "acquisition_idx", "slice_idx", "read_idx"
        )
        weight = weight.transpose("slice_idx", "read_idx", "phase_idx")
    before, weight_before = data.copy(deep=True), weight.copy(deep=True)
    result = masks.extract_masked_data(
        data, weight, reduction=reduction, signal_part=signal_part
    )
    quadrature = 3 + 4j if np.iscomplexobj(data) else 3
    if reduction == "none":
        vector = np.array([quadrature / 2, -quadrature, 0])
        if signal_part == "magnitude":
            vector = abs(vector)
        expected = xr.DataArray(
            vector[:, None, None, None, None] * scale[None, None, None],
            dims=(*SPATIAL_DIMS, "acquisition_idx", "repetition_idx"),
        ).transpose(*data.dims)
        assert result.dims == data.dims
        for coord_name in data.coords:
            xr.testing.assert_identical(
                result.coords[coord_name], data.coords[coord_name]
            )
    else:
        factor = (
            (1.5 if reduction == "sum" else 1) * abs(quadrature)
            if signal_part == "magnitude"
            else (-0.5 if reduction == "sum" else -1 / 3) * quadrature
        )
        expected = xr.DataArray(
            factor * scale, dims=("acquisition_idx", "repetition_idx")
        )
        expected = expected.transpose(
            *[dim for dim in data.dims if dim not in SPATIAL_DIMS]
        )
        assert result.dims == expected.dims
        for dim in result.dims:
            xr.testing.assert_identical(result.coords[dim], data.coords[dim])
        assert not set(SPATIAL_DIMS) & set(result.coords)
        assert not {"x", "y", "z"} & set(result.coords)
    np.testing.assert_allclose(result.values, expected.values, rtol=0, atol=1e-12)
    assert result.dtype == (
        np.complex128
        if np.iscomplexobj(data) and signal_part == "complex"
        else np.float64
    )
    assert result.name == data.name
    assert result.attrs == {
        **data.attrs,
        "applied_mask": "roi",
        "mask_reduction": reduction,
        "mask_signal_part": signal_part,
    }
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(weight, weight_before)
    xr.testing.assert_identical(
        result,
        masks.extract_masked_data(
            data, weight, reduction=reduction, signal_part=signal_part
        ),
    )


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_projected_options_delegate_once_and_match_native_extraction(
    monkeypatch, reduction, signal_part
):
    data, weight, _ = _signal()
    projection = masks.project_mask_to_data_grid
    calls = []

    def record_projection(*args, **kwargs):
        calls.append((args, kwargs))
        return projection(*args, **kwargs)

    monkeypatch.setattr(masks, "project_mask_to_data_grid", record_projection)
    result, projected = masks.apply_projected_mask(
        data,
        weight,
        source_data=data,
        coverage_samples=1,
        reduction=reduction,
        signal_part=signal_part,
        return_mask=True,
    )
    assert len(calls) == 1
    xr.testing.assert_identical(
        result,
        masks.extract_masked_data(
            data, projected, reduction=reduction, signal_part=signal_part
        ),
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_opposite_phases_cancellation_is_an_explicit_choice(reduction, signal_part):
    signal = xr.DataArray([100j, -100j], dims="voxel", attrs={"units": "a.u."})
    result = masks.extract_masked_data(
        signal,
        np.ones(2),
        spatial_dims=("voxel",),
        reduction=reduction,
        signal_part=signal_part,
    )
    expected = (
        (200 if reduction == "sum" else 100) if signal_part == "magnitude" else 0j
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_empty_invalid_and_fractional_rois_have_per_sample_effective_weights(
    reduction, signal_part
):
    data = xr.DataArray(
        [[3 + 4j, np.nan, 3 + 4j], [-3 - 4j, np.inf, np.inf]], dims=("voxel", "time")
    )
    mask = xr.DataArray([0.5, 1.0], dims="voxel")
    result = masks.extract_masked_data(
        data,
        mask,
        spatial_dims=("voxel",),
        reduction=reduction,
        signal_part=signal_part,
    )
    if reduction == "none":
        assert result.dims == data.dims
        assert not np.isfinite(result[:, 1]).any()
        assert np.isinf(result[1, 2])
    else:
        expected = (
            ([7.5, 0, 2.5] if reduction == "sum" else [5, np.nan, 5])
            if signal_part == "magnitude"
            else (
                [-1.5 - 2j, 0j, 1.5 + 2j]
                if reduction == "sum"
                else [-1 - 4j / 3, np.nan, 3 + 4j]
            )
        )
        np.testing.assert_allclose(
            result.values, expected, rtol=0, atol=1e-12, equal_nan=True
        )
    empty = masks.extract_masked_data(
        data,
        xr.zeros_like(mask),
        spatial_dims=("voxel",),
        reduction=reduction,
        signal_part=signal_part,
        fill_value=np.nan,
    )
    if reduction == "sum":
        np.testing.assert_array_equal(empty, np.zeros(3))
    else:
        assert np.isnan(empty).all()


def test_boolean_unreduced_mask_retains_individual_complex_samples_and_nan_fill():
    data, weight, _ = _signal()
    mask = weight > 0
    result = masks.extract_masked_data(data, mask, fill_value=np.nan)
    np.testing.assert_array_equal(result.values[:2], data.values[:2])
    assert np.isnan(result.values[2]).all()
    assert result.dims == data.dims


def test_labelled_partial_mask_broadcasts_only_along_declared_spatial_dimensions():
    data, weight, _ = _signal()
    mask = weight.isel(phase_idx=0, slice_idx=0, drop=True)
    result = masks.extract_masked_data(data, mask, reduction="mean")
    np.testing.assert_allclose(
        result,
        masks.extract_masked_data(data, weight, reduction="mean"),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("invalid", ["reversed", "length", "nonspatial", "xyz"])
def test_native_mask_rejects_inconsistent_grid_without_silent_alignment(invalid):
    data, weight, _ = _signal()
    if invalid == "reversed":
        weight = weight.assign_coords(read_idx=[0, 1, 2])
    elif invalid == "length":
        weight = weight.isel(read_idx=slice(0, 2))
    elif invalid == "nonspatial":
        weight = xr.DataArray(np.ones(3), dims="acquisition_idx")
    else:
        weight = weight.assign_coords(x=("read_idx", [100.0, 101.0, 102.0]))
    with pytest.raises(
        ValueError, match="extract_masked_data.*(grid|dimension|coordinate)"
    ):
        masks.extract_masked_data(data, weight)


@pytest.mark.parametrize(
    "options",
    [
        {"reduction": "median"},
        {"signal_part": "phase"},
        {"mean_along_space": True, "reduction": "sum"},
        {"mean_along_space": True, "reduction": "none"},
    ],
)
def test_projected_invalid_reduction_options_fail_before_geometry(monkeypatch, options):
    def unexpected_projection(*args, **kwargs):
        pytest.fail("Invalid reduction must be rejected before projection")

    monkeypatch.setattr(masks, "project_mask_to_data_grid", unexpected_projection)
    with pytest.raises(ValueError, match="(reduction|signal_part)"):
        masks.apply_projected_mask(xr.DataArray([1]), np.ones(1), **options)


@pytest.mark.parametrize(
    "options",
    [
        {"reduction": "median"},
        {"signal_part": "phase"},
        {"spatial_dims": ("missing",)},
        {"spatial_dims": ()},
        {"spatial_dims": ("read_idx", "read_idx")},
    ],
)
def test_native_invalid_options_are_actionable(options):
    data, weight, _ = _signal()
    with pytest.raises(ValueError, match="(reduction|signal_part|[Ss]patial|spatial)"):
        masks.extract_masked_data(data, weight, **options)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.ones((1, 3)),
        np.array([1j, 0j, 0j]).reshape(3, 1, 1),
        np.array(["a", "b", "c"]).reshape(3, 1, 1),
    ],
)
def test_native_invalid_mask_shape_and_dtype_are_actionable(bad_mask):
    data, _, _ = _signal()
    with pytest.raises(
        (ValueError, TypeError), match="extract_masked_data.*(shape|weights)"
    ):
        masks.extract_masked_data(data, bad_mask)


def test_native_data_type_and_export_contract():
    from hypermri.analysis.anatomical import extract_masked_data

    assert extract_masked_data is masks.extract_masked_data
    assert "extract_masked_data" in masks.__all__
    with pytest.raises(TypeError, match="numeric xarray DataArray"):
        extract_masked_data(np.ones(3), np.ones(3))


def test_native_coverage_clipping_matches_projection_policy_and_preserves_mask():
    data = xr.DataArray([10.0] * 6, dims="voxel")
    mask = xr.DataArray([np.nan, -np.inf, np.inf, -1.0, 0.5, 2.0], dims="voxel")
    before = mask.copy(deep=True)
    result = masks.extract_masked_data(data, mask, spatial_dims=("voxel",))
    np.testing.assert_array_equal(result, [0, 0, 10, 0, 5, 10])
    xr.testing.assert_identical(mask, before)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_experiment_options_delegate_once_and_preserve_stored_masks(
    monkeypatch, reduction, signal_part
):
    from hypermri.loading.experiment import MRIExperiment

    data, weight, _ = _signal()
    source = MRIExperiment.__new__(MRIExperiment)
    source._initialize_masks(xr.Dataset({"roi": weight}))
    source.data = data
    source.scanner_data = None
    source._data_loaded = True
    source._scanner_data_loaded = True
    before = source.masks.copy(deep=True)
    operation = masks.apply_projected_mask
    calls = []

    def record_operation(*args, **kwargs):
        calls.append(kwargs)
        return operation(*args, **kwargs)

    monkeypatch.setattr(masks, "apply_projected_mask", record_operation)
    result = source.apply_mask_to(
        data, "roi", reduction=reduction, signal_part=signal_part, coverage_samples=1
    )
    assert len(calls) == 1
    assert calls[0]["reduction"] == reduction
    assert calls[0]["signal_part"] == signal_part
    assert calls[0]["source_obj"] is source
    xr.testing.assert_identical(
        result,
        masks.extract_masked_data(
            data, weight, reduction=reduction, signal_part=signal_part
        ),
    )
    xr.testing.assert_identical(source.masks, before)


@pytest.mark.parametrize("opposite_phase", [False, True])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_mask_extraction_uses_explicit_input_domain_and_retains_its_axis(
    opposite_phase, reduction, signal_part
):
    # Invented eight-point tone of magnitude 20 a.u., with exact DFT amplitude
    # 160 a.u. at +250 Hz for a 1 ms dwell. No reconstruction/scanner fixture.
    tone = 20 * np.array([1, 1j, -1, -1j] * 2, dtype=np.complex128)
    values = np.stack([tone, -tone if opposite_phase else tone])
    raw = xr.DataArray(
        values,
        dims=("voxel", "time"),
        coords={"time": np.arange(8) * 0.001},
        attrs={"units": "a.u.", "signal_domain": "temporal"},
    )
    raw.time.attrs["units"] = "s"
    transformed = xr.DataArray(
        np.fft.fftshift(np.fft.fft(values, axis=1), axes=1),
        dims=("voxel", "frequency"),
        coords={"frequency": np.arange(-500, 500, 125)},
        attrs={
            "units": "a.u.",
            "signal_domain": "frequency",
            "provenance": "invented DFT",
        },
    )
    transformed.frequency.attrs["units"] = "Hz"
    mask = xr.DataArray([1.0, 1.0], dims="voxel")
    factor = 2 if reduction == "sum" else 1
    for domain in (raw, transformed):
        before = domain.copy(deep=True)
        result = masks.extract_masked_data(
            domain,
            mask,
            spatial_dims=("voxel",),
            reduction=reduction,
            signal_part=signal_part,
        )
        if signal_part == "magnitude":
            expected = np.abs(domain.values[0]) * factor
        else:
            expected = (
                np.zeros(8, dtype=complex)
                if opposite_phase
                else domain.values[0] * factor
            )
        np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12)
        xr.testing.assert_identical(
            result.coords[result.dims[0]], domain.coords[domain.dims[1]]
        )
        assert result.attrs["signal_domain"] == domain.attrs["signal_domain"]
        assert result.attrs["units"] == "a.u."
        xr.testing.assert_identical(domain, before)


def test_matching_coordinate_values_with_conflicting_units_are_rejected():
    data, mask, _ = _signal()
    data.x.attrs["units"] = "mm"
    mask = mask.assign_coords(x=("read_idx", data.x.values))
    mask.x.attrs["units"] = "m"
    with pytest.raises(ValueError, match="coordinate 'x'.*input grid"):
        masks.extract_masked_data(data, mask)


def test_legacy_mean_option_accepts_explicit_mean_and_magnitude_without_changing_default():
    data, mask, _ = _signal()
    default = masks.apply_projected_mask(
        data, mask, source_data=data, coverage_samples=1, mean_along_space=True
    )
    explicit = masks.apply_projected_mask(
        data,
        mask,
        source_data=data,
        coverage_samples=1,
        mean_along_space=True,
        reduction="mean",
    )
    magnitude = masks.apply_projected_mask(
        data,
        mask,
        source_data=data,
        coverage_samples=1,
        mean_along_space=True,
        signal_part="magnitude",
    )
    np.testing.assert_allclose(default, explicit, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        magnitude,
        masks.extract_masked_data(
            data, mask, reduction="mean", signal_part="magnitude"
        ),
        rtol=0,
        atol=1e-12,
    )
    assert default.attrs == {}
    assert explicit.attrs["mask_reduction"] == "mean"


@pytest.mark.parametrize("keep_attrs", ["default", False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_xarray_attribute_setting_is_respected_for_legacy_calls(keep_attrs, mean):
    data, mask, _ = _signal()
    mask.attrs = {"units": "dimensionless", "mask_only": "invented ROI"}
    before, mask_before = data.copy(deep=True), mask.copy(deep=True)
    with xr.set_options(keep_attrs=keep_attrs):
        result, projected = masks.apply_projected_mask(
            data,
            mask,
            source_data=data,
            coverage_samples=1,
            mean_along_space=mean,
            return_mask=True,
        )
    expected_attrs = (
        (data.attrs if keep_attrs is True else {})
        if mean
        else {**data.attrs, "applied_mask": "roi"}
    )
    assert result.attrs == expected_attrs
    assert projected.attrs["mask_projection"] == "xyz"
    assert projected.attrs["mask_only"] == "invented ROI"
    assert projected.attrs["units"] == "dimensionless"
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(mask, mask_before)


@pytest.mark.parametrize("keep_attrs", ["default", False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("signal_part", ["complex", "magnitude"])
def test_explicit_metadata_has_one_source_for_native_and_projected_calls(
    keep_attrs, reduction, signal_part
):
    data, mask, scale = _signal()
    data.attrs["mask_projection"] = "input provenance"
    mask.attrs = {
        "units": "dimensionless",
        "provenance": "mask provenance",
        "mask_only": "invented ROI",
        "mask_reduction": "mask metadata",
    }
    before, mask_before = data.copy(deep=True), mask.copy(deep=True)
    with xr.set_options(keep_attrs=keep_attrs):
        native = masks.extract_masked_data(
            data, mask, reduction=reduction, signal_part=signal_part
        )
        projected_result, projected = masks.apply_projected_mask(
            data,
            mask,
            source_data=data,
            coverage_samples=1,
            reduction=reduction,
            signal_part=signal_part,
            return_mask=True,
        )
        extracted_projection = masks.extract_masked_data(
            data, projected, reduction=reduction, signal_part=signal_part
        )
    assert native.attrs == {
        **data.attrs,
        "applied_mask": "roi",
        "mask_reduction": reduction,
        "mask_signal_part": signal_part,
    }
    xr.testing.assert_identical(projected_result, native)
    xr.testing.assert_identical(extracted_projection, native)
    assert native.name == data.name
    for coord_name in native.coords:
        xr.testing.assert_identical(native.coords[coord_name], data.coords[coord_name])
    if reduction == "none":
        vector = np.array([1.5 + 2j, -3 - 4j, 0j])
        if signal_part == "magnitude":
            vector = abs(vector)
        expected = vector[:, None, None, None, None] * scale[None, None, None]
    else:
        factor = (
            (7.5 if reduction == "sum" else 5)
            if signal_part == "magnitude"
            else (-1.5 - 2j if reduction == "sum" else -1 - 4j / 3)
        )
        expected = factor * scale
    np.testing.assert_allclose(native.values, expected, rtol=0, atol=1e-12)
    assert native.dtype == (np.float64 if signal_part == "magnitude" else np.complex128)
    assert projected.attrs["mask_projection"] == "xyz"
    assert projected.attrs["mask_only"] == "invented ROI"
    assert projected.attrs["provenance"] == "mask provenance"
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(mask, mask_before)
