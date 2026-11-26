import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from hypermri.utils.image.slices import get_overlapping_slices
from hypermri.utils.plot.matplotlib_helpers import (
    _fixed_decimal_tick_formatter,
    _scaled_colorbar_tick_formatter,
    _scientific_colorbar_tick_formatter,
    add_colorbar,
)
from hypermri.utils.plot.matplotlib_safety import (
    disable_usetex_if_unavailable,
    install_matplotlib_usetex_fallback,
)


def test_scientific_colorbar_tick_formatter_formats_mantissa_precision():
    formatter = _scientific_colorbar_tick_formatter(significant_digits=3)

    assert formatter(0.013, 0) == r"$1.30\times 10^{-2}$"
    assert formatter(-0.013, 0) == r"$-1.30\times 10^{-2}$"
    assert formatter(0.0, 0) == "0"


def test_add_colorbar_applies_scientific_tick_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[0.0, 0.013], [0.0065, 0.0100]]), vmin=0.0, vmax=0.013)

    cbar = add_colorbar(im, scientific_notation=True, scientific_digits=2)
    cbar.set_ticks([0.0, 0.013])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]

    assert labels == ["0", r"$1.3\times 10^{-2}$"]

    plt.close(fig)


def test_scientific_colorbar_tick_formatter_without_latex_uses_plain_text():
    formatter = _scientific_colorbar_tick_formatter(
        significant_digits=3, use_latex=False
    )

    assert formatter(0.013, 0) == "1.30×10^-2"
    assert formatter(-0.013, 0) == "-1.30×10^-2"
    assert formatter(0.0, 0) == "0"


def test_add_colorbar_without_latex_uses_hyphen_minus_for_linear_ticks():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[-6.0, -2.0], [0.0, 0.0]]), vmin=-6.0, vmax=0.0)

    cbar = add_colorbar(im, scientific_notation=False, use_latex=False)
    cbar.set_ticks([-6.0, -4.0, -2.0, 0.0])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]

    assert labels == ["-6", "-4", "-2", "0"]

    plt.close(fig)


def test_add_colorbar_without_latex_uses_plain_scientific_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[0.0, 0.013], [0.0065, 0.0100]]), vmin=0.0, vmax=0.013)

    cbar = add_colorbar(im, scientific_notation=True, use_latex=False)
    cbar.set_ticks([0.0, 0.013])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]

    assert labels == ["0", "1.3×10^-2"]

    plt.close(fig)


def test_add_colorbar_linear_ticks_use_hyphen_minus_by_default():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[-6.0, -2.0], [0.0, 0.0]]), vmin=-6.0, vmax=0.0)

    cbar = add_colorbar(im, scientific_notation=False)
    cbar.set_ticks([-6.0, -4.0, -2.0, 0.0])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]

    assert labels == ["-6", "-4", "-2", "0"]

    plt.close(fig)


def test_add_colorbar_default_keeps_latex_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[0.0, 0.013], [0.0065, 0.0100]]), vmin=0.0, vmax=0.013)

    cbar = add_colorbar(im, scientific_notation=True)
    cbar.set_ticks([0.0, 0.013])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]

    assert labels == ["0", r"$1.3\times 10^{-2}$"]

    plt.close(fig)


def test_add_colorbar_can_show_shared_scientific_exponent():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[0.0, 0.013], [0.0065, 0.0100]]), vmin=0.0, vmax=0.013)

    cbar = add_colorbar(
        im,
        scientific_notation=True,
        scientific_digits=2,
        scientific_shared_exponent=True,
    )
    cbar.set_ticks([0.0, 0.013])
    cbar.update_ticks()
    fig.canvas.draw()

    labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]
    exponent_labels = [text.get_text() for text in cbar.ax.texts]

    assert labels == ["0", "1.3"]
    assert exponent_labels == [r"$\times 10^{-2}$"]

    plt.close(fig)


def _colorbar_tick_labels(cbar, ticks):
    cbar.set_ticks(ticks)
    cbar.update_ticks()
    cbar.ax.figure.canvas.draw()
    return [tick.get_text() for tick in cbar.ax.get_yticklabels()]


def test_scientific_formatter_decimal_places_overrides_digits_and_pads_zero():
    formatter = _scientific_colorbar_tick_formatter(
        significant_digits=5, decimal_places=1
    )

    assert formatter(0.013, 0) == r"$1.3\times 10^{-2}$"
    assert formatter(0.0, 0) == "0.0"


def test_scientific_formatters_reject_invalid_precision():
    with pytest.raises(ValueError, match="decimal_places must be at least 0"):
        _scientific_colorbar_tick_formatter(decimal_places=-1)
    with pytest.raises(ValueError, match="decimal_places must be at least 0"):
        _scaled_colorbar_tick_formatter(exponent=-2, decimal_places=-1)
    with pytest.raises(ValueError, match="scientific_digits must be at least 1"):
        _scaled_colorbar_tick_formatter(exponent=-2, significant_digits=0)

    # An explicit decimal_places makes significant_digits irrelevant.
    formatter = _scaled_colorbar_tick_formatter(
        exponent=-2, significant_digits=0, decimal_places=0
    )
    assert formatter(0.013, 0) == "1"


def test_fixed_decimal_formatter_writes_no_negative_zero():
    formatter = _fixed_decimal_tick_formatter(2)

    assert formatter(-1e-17, 0) == "0.00"
    assert formatter(-0.004, 0) == "0.00"
    assert formatter(-1.5, 0) == "-1.50"


def test_add_colorbar_decimal_places_pads_shared_exponent_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[0.0, 0.013], [0.0065, 0.0100]]), vmin=0.0, vmax=0.013)

    cbar = add_colorbar(
        im,
        scientific_notation=True,
        scientific_shared_exponent=True,
        decimal_places=2,
    )

    assert _colorbar_tick_labels(cbar, [0.0, 0.013]) == ["0.00", "1.30"]

    plt.close(fig)


def test_add_colorbar_decimal_places_fixes_linear_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[-6.0, -2.0], [0.0, 0.0]]), vmin=-6.0, vmax=0.0)

    cbar = add_colorbar(im, decimal_places=1)

    assert _colorbar_tick_labels(cbar, [-6.0, -3.0, 0.0]) == ["-6.0", "-3.0", "0.0"]

    plt.close(fig)


def test_add_colorbar_defaults_to_linear_labels():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[-6.0, -2.0], [0.0, 0.0]]), vmin=-6.0, vmax=0.0)

    cbar = add_colorbar(im)

    labels = _colorbar_tick_labels(cbar, [-6.0, -4.0, -2.0, 0.0])
    assert labels == ["-6", "-4", "-2", "0"]

    plt.close(fig)


def test_add_colorbar_rejects_negative_decimal_places():
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(4).reshape(2, 2))

    with pytest.raises(ValueError, match="decimal_places must be at least 0"):
        add_colorbar(im, decimal_places=-1)

    plt.close(fig)


def test_add_colorbar_can_clip_outline_to_colorbar_axes():
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(4).reshape(2, 2))

    cbar = add_colorbar(im, clip_outline=True)

    # Matplotlib optimizes a rectangular clip path into a clip box.
    assert cbar.outline.get_clip_box() is not None

    plt.close(fig)


def test_add_colorbar_keeps_ticks_above_solids_and_can_hide_labels_and_outline():
    with matplotlib.rc_context({"axes.axisbelow": True}):
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(4).reshape(2, 2))

        cbar = add_colorbar(
            im,
            show_ticklabels=False,
            show_outline=False,
        )
        fig.canvas.draw()
        major_ticks = cbar.ax.yaxis.get_major_ticks()

        assert cbar.ax.yaxis.get_zorder() > cbar.solids.get_zorder()
        assert any(tick.tick2line.get_visible() for tick in major_ticks)
        assert not any(
            label.get_visible()
            for tick in major_ticks
            for label in (tick.label1, tick.label2)
        )
        assert cbar.outline.get_visible() is False

        plt.close(fig)


def test_add_colorbar_can_fit_height_and_add_image_outline():
    panel_width = 2.0
    image_aspect = 3 / 4
    fig = plt.figure(figsize=(panel_width, panel_width * image_aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(np.arange(12).reshape(3, 4))
    ax.axis("off")

    cbar = add_colorbar(
        im,
        aspect=15,
        pad_fraction=0.1,
        image_outline=True,
        image_outline_kwargs={"color": "black", "linewidth": 1.0},
        fit_figure_height=True,
    )
    fig.canvas.draw()

    assert fig.get_figwidth() == pytest.approx(panel_width)
    assert ax.get_position().y0 == pytest.approx(0)
    assert ax.get_position().y1 == pytest.approx(1)
    assert cbar.ax.get_position().y0 == pytest.approx(0)
    assert cbar.ax.get_position().y1 == pytest.approx(1)
    assert len(ax._hypermri_image_outline) == 4
    assert all(line.get_color() == "black" for line in ax._hypermri_image_outline)

    plt.close(fig)


def test_disable_usetex_if_unavailable_disables_failed_tex_render(monkeypatch):
    original_rcparams = matplotlib.rcParams.copy()

    def fail_draw(self):
        raise FileNotFoundError("phvr7t.tfm")

    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{sfmath}"
        monkeypatch.setattr(
            "matplotlib.backends.backend_agg.FigureCanvasAgg.draw", fail_draw
        )

        with pytest.warns(RuntimeWarning, match="text.usetex was disabled"):
            disabled = disable_usetex_if_unavailable()

        assert disabled is True
        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["text.latex.preamble"] == ""
    finally:
        matplotlib.rcParams.update(original_rcparams)


def test_global_usetex_fallback_retries_figure_draw(monkeypatch):
    original_rcparams = matplotlib.rcParams.copy()
    original_text_draw = matplotlib.text.Text.draw

    def fail_when_usetex_enabled(self, renderer):
        if matplotlib.rcParams["text.usetex"]:
            raise FileNotFoundError("Matplotlib TeX could not find phvr7t.tfm")
        return original_text_draw(self, renderer)

    try:
        install_matplotlib_usetex_fallback()
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{sfmath}"
        monkeypatch.setattr("matplotlib.text.Text.draw", fail_when_usetex_enabled)

        fig, ax = plt.subplots()
        ax.set_title("TeX fallback")
        with pytest.warns(RuntimeWarning, match="text.usetex was disabled"):
            fig.canvas.draw()

        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["text.latex.preamble"] == ""
        assert ax.title.get_usetex() is False
        plt.close(fig)
    finally:
        matplotlib.rcParams.update(original_rcparams)


def _make_slice_data_array(axis_mm):
    da = xr.DataArray(
        np.zeros(len(axis_mm)),
        dims=("slice_idx",),
        coords={"slice_idx": np.arange(len(axis_mm))},
    )
    da.slice_idx.attrs["axis_mm"] = np.asarray(axis_mm)
    return da


def test_get_overlapping_slices_falls_back_to_data_when_scanner_data_invalid():
    class DummyExperiment:
        pass

    exp = DummyExperiment()
    # Invalid for extraction: missing slice_idx coord and axis_mm attrs
    exp.scanner_data = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    exp.data = _make_slice_data_array([0.0, 5.0, 10.0])

    reference = _make_slice_data_array([0.0, 5.0, 10.0])
    idx_exp, idx_ref = get_overlapping_slices(
        data_obj_1=exp,
        data_obj_2=reference,
        tol_mm=0.1,
    )

    assert idx_exp == [0, 1, 2]
    assert idx_ref == [0, 1, 2]


def test_get_overlapping_slices_raises_for_missing_axis_mm():
    broken = xr.DataArray(
        np.zeros(3),
        dims=("slice_idx",),
        coords={"slice_idx": [0, 1, 2]},
    )
    reference = _make_slice_data_array([0.0, 5.0, 10.0])

    with pytest.raises(
        ValueError, match="Could not extract non-empty slice coordinates"
    ):
        get_overlapping_slices(data_obj_1=broken, data_obj_2=reference, tol_mm=0.1)
