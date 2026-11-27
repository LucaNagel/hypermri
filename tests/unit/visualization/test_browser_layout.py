"""Tests for the browser viewer layout and the joining of its browser parts."""

import re
from importlib.resources import files as resource_files

import pytest

from hypermri.visualization.layout import (
    BELOW_PLOTS,
    BROWSER_PARTS,
    CONTROL_GROUPS,
    OPTION_SECTIONS,
    PANEL_DESCRIPTIONS,
    PANELS,
    TOOLS,
    TOP_BAR,
    browser_layout,
    browser_script,
)


def _placed(layout):
    """Control group names of a layout outside the tool menu, in page order."""
    sections = [name for _, names in layout["option_sections"] for name in names]
    return layout["top_bar"] + sections + layout["below_plots"]


def test_default_layout_lists_every_group_once():
    layout = browser_layout()

    assert layout["panels"] == list(PANELS) == ["image", "curve", "time"]
    assert sorted(_placed(layout) + layout["tools"]) == sorted(CONTROL_GROUPS)
    # Slices stay visible; the tool menu sits below the plots.
    assert layout["top_bar"] == list(TOP_BAR) == ["slices", "layers"]
    assert layout["below_plots"] == list(BELOW_PLOTS) == ["tool", "parameters"]
    assert (
        layout["tools"]
        == list(TOOLS)
        == ["masks", "fit", "curve_fit", "ideal", "signal_decay"]
    )
    # Everything else is in titled sections of the collapsible plot options.
    titles = [title for title, _ in layout["option_sections"]]
    assert titles == [title for title, _ in OPTION_SECTIONS]
    assert dict(layout["option_sections"])["Map"] == ["colour_range", "display"]


def test_layout_accepts_other_places_and_drops_empty_sections():
    layout = browser_layout(
        panels=["curve", "image"],
        top_bar=["overlay"],
        option_sections=[("Empty", []), ("Main", ("slices", "masks"))],
        below_plots=[],
        tools=[],
    )

    assert layout == {
        "panels": ["curve", "image"],
        "top_bar": ["overlay"],
        "option_sections": [["Main", ["slices", "masks"]]],
        "below_plots": [],
        "tools": [],
    }


def test_tools_need_the_tool_menu_and_are_not_also_placed():
    with pytest.raises(ValueError, match="need the 'tool' control group"):
        browser_layout(below_plots=[], tools=["masks"])
    with pytest.raises(ValueError, match="control group .*masks.* more than once"):
        browser_layout(below_plots=["tool", "masks"], tools=["masks"])
    with pytest.raises(ValueError, match="unknown control group"):
        browser_layout(tools=["peaks"])
    # The menu may also sit in the top bar.
    layout = browser_layout(top_bar=["slices", "tool"], below_plots=[])
    assert layout["top_bar"] == ["slices", "tool"]


@pytest.mark.parametrize(
    ("panels", "top_bar", "message"),
    [
        (["image", "histogram"], TOP_BAR, "unknown panel"),
        (["image", "image"], TOP_BAR, "panel .* more than once"),
        ([], TOP_BAR, "at least one panel"),
        (PANELS, ["slices", "slices"], "control group .* more than once"),
        (PANELS, ["spectrum"], "control group .* more than once"),
        (PANELS, ["slice"], "unknown control group"),
    ],
)
def test_layout_rejects_wrong_names(panels, top_bar, message):
    with pytest.raises(ValueError, match=message):
        browser_layout(panels=panels, top_bar=top_bar)


def test_browser_script_joins_parts_in_order():
    script = browser_script()

    markers = re.findall(r"^// ---- browser/(\S+) ----$", script, flags=re.MULTILINE)
    assert markers == list(BROWSER_PARTS)
    assert script.count("export default") == 1
    assert script.rstrip().endswith("export default { render };")


def test_browser_script_builds_every_layout_name():
    """Each panel and control group of layout.py has a builder in viewer.js."""
    script = browser_script()
    builders = re.search(r"CONTROL_GROUP_BUILDERS = \{(.*?)\};", script, re.DOTALL)
    panels = re.search(r"PANEL_CLASSES = \{(.*?)\};", script, re.DOTALL)

    assert set(re.findall(r"(\w+):", builders.group(1))) == set(CONTROL_GROUPS)
    assert set(re.findall(r"(\w+):", panels.group(1))) == set(PANEL_DESCRIPTIONS)


def test_browser_parts_define_each_function_once():
    """The parts share one module scope, so a name defined twice would clash."""
    names = re.findall(
        r"^(?:function|const)\s+(\w+)", browser_script(), flags=re.MULTILINE
    )

    assert len(names) == len(set(names)), sorted(
        {name for name in names if names.count(name) > 1}
    )


def test_plots_of_views_side_by_side_share_one_baseline():
    # The panels stand in one row on its bottom, so that the maps of views
    # side by side line up (checked in the browser pane with the real
    # stylesheet at 0 and 90 degrees, 26 November 2026).
    css = (
        resource_files("hypermri.visualization")
        .joinpath("browser/viewer.css")
        .read_text(encoding="utf-8")
    )
    rule = re.search(r"\.hm-csi-plots \{(.*?)\}", css, re.S).group(1)
    assert "flex-wrap: nowrap;" in rule
    assert "align-items: flex-end;" in rule
    # A lowered map lowers the curve beside it, too.
    panels = re.search(
        r"\.hm-csi-map,\s*\.hm-csi-spectrum,\s*\.hm-csi-time \{(.*?)\}", css, re.S
    )
    assert "var(--hm-panel-height, var(--hm-plot-height))" in panels.group(1)
    # A curve keeps a share of a narrow view; the map is lowered instead.
    curves = re.findall(r"\.hm-csi-spectrum,\s*\.hm-csi-time \{(.*?)\}", css, re.S)
    assert "min-width: max(200px, 40%);" in curves[-1]


def test_map_is_sized_before_plotly_lays_it_out():
    # Plotly keeps the size of its first draw on a redraw; drawn into the box
    # of the last plane (a rotation), it would widen the axis ranges.
    script = browser_script()
    for name in ("drawImagePanel", "drawOutlinePanel"):
        body = re.search(rf"function {name}\(v\) \{{(.*?)\n\}}", script, re.S).group(1)
        assert body.index("sizeImagePanel(v)") < body.index("Plotly.react(")
        assert "...panelSize(v.panels.image)" in body
    observer = re.search(
        r"const resizer = new ResizeObserver\((.*?)\}\);", script, re.S
    ).group(1)
    assert "drawImagePanel(v)" in observer


def test_groups_left_out_of_rebuilds_are_built_once_at_the_start():
    # buildControls skips some groups so their inputs survive a new block;
    # each of them must then be built once when the page is first drawn.
    script = (
        resource_files("hypermri.visualization")
        .joinpath("browser", "viewer.js")
        .read_text(encoding="utf-8")
    )
    kept = re.search(r"const kept = \[([^\]]*)\]", script).group(1)
    kept_names = set(re.findall(r'"(\w+)"', kept)) - {"slices"}
    first = re.search(
        r"buildControls\(v\);\n\s*buildGroups\(v, \[([^\]]*)\]\);", script
    )
    assert first, "render() must build the kept groups right after buildControls"
    assert kept_names <= set(re.findall(r'"(\w+)"', first.group(1)))


def test_tooltips_switch_also_hides_curve_and_time_hover_labels():
    # Off by default, the switch "tooltips" hides the hover labels of the
    # spectrum and the time course too; the fit tool always hides them.
    script = browser_script()
    curve = re.search(r"function drawCurvePanel\(v\) \{(.*?)\n\}", script, re.S)
    time = re.search(r"function drawTimePanel\(v\) \{(.*?)\n\}", script, re.S)
    for body in (curve.group(1), time.group(1)):
        hidden = re.search(
            r"if \((.*?)\)\s*\{?\s*hideHoverLabels\(traces\)", body, re.S
        )
        assert hidden and "show_tooltips" in hidden.group(1)
    assert 'v.model.get("tool") === "fit"' in curve.group(1)
    helper = re.search(r"function hideHoverLabels\(traces\) \{(.*?)\n\}", script, re.S)
    # "none" keeps the click events; a hovertemplate would still show a label.
    assert 'trace.hoverinfo = "none";' in helper.group(1)
    assert "delete trace.hovertemplate;" in helper.group(1)


def test_colour_bars_sit_next_to_the_image_at_its_height():
    # The bars stand a few pixels right of the map, as tall as it, and are
    # placed from the panel width: Plotly's own plot width shrinks with the
    # bars it holds, which squeezed narrow maps to a few pixels.
    script = browser_script()
    place = re.search(r"function colourBarPlace\(v, index\) \{(.*?)\n\}", script, re.S)
    body = place.group(1)
    assert "clientWidth" in body and "_size" not in body
    assert "xDomain[1] + offset" in body
    assert "ypad: 0" in body and "xpad: 0" in body
    # A bar of a very low map keeps 80 px; Plotly stops drawing below that.
    assert "80 / height" in body
    assert re.search(r"const COLOUR_BAR_GAP_PX = [1-8];", script)
    # The margins hold the bars; Plotly must not widen them on its own.
    assert "autoexpand: false" in script


def _function_body(script, name):
    match = re.search(rf"function {name}\((.*?)\) \{{(.*?)\n\}}", script, re.S)
    assert match, name
    return match.group(2)


def test_anatomy_and_overlay_share_one_slice_slider():
    # One helper builds the slider with "follow map" for both layers; the
    # anatomy slider sends {"anatomy": {"slice": ...}}, which ScanView reads.
    script = browser_script()
    assert script.count("function layerSliceSlider(") == 1
    assert "function layerSliceSelect(" not in script
    layers = _function_body(script, "layerControls")
    overlay = _function_body(script, "overlayControls")
    assert "layerSliceSlider(" in layers
    assert "anatomy: { slice: value }" in layers
    assert "layerSliceSlider(" in overlay


def test_anatomy_slice_and_opacities_sit_in_the_top_bar():
    # The "layers" group of the top bar is visible without "Plot options";
    # the opacity sliders live only there.
    assert "layers" in TOP_BAR
    for _, names in OPTION_SECTIONS:
        assert "layers" not in names
    script = browser_script()
    layers = _function_body(script, "layerControls")
    assert '"map opacity"' in layers and '"overlay opacity"' in layers
    assert "outlineSliceControls(v)" in layers
    assert "opacitySlider(" not in _function_body(script, "anatomyControls")
    assert "opacitySlider(" not in _function_body(script, "overlayControls")


def test_overlay_labels_drop_the_overlay_prefix():
    # The section title "Overlay" already names them.
    script = browser_script()
    body = _function_body(script, "overlayControls") + _function_body(
        script, "overlayRangeControls"
    )
    labels = re.findall(r'labelled\("([^"]+)"', body)
    assert labels and not [label for label in labels if label.startswith("overlay")]


def test_two_choices_are_split_buttons():
    script = browser_script()
    assert 'toggle([["FID", "fid"], ["Spec", "spectrum"]]' in script
    assert 'toggle([["k-space", "kspace"], ["image", "image"]]' in script
    assert "(units.length === 2 ? toggle : select)" in script
    assert "toggle(\n    options.range_modes" in script
    assert 'labelled("FID"' not in script


def test_map_and_overlay_colour_ranges_use_one_double_slider():
    script = browser_script()
    assert script.count("function doubleRangeSlider(") == 1
    assert "doubleRangeSlider(" in _function_body(script, "colourRangeControls")
    assert "doubleRangeSlider(" in _function_body(script, "overlayRangeControls")
    assert 'labelled("from"' not in script and 'labelled("min' not in script


def test_spectrum_numbers_line_up_at_the_end_of_the_section():
    script = browser_script()
    for label in ("line broadening [Hz]", "points", "denoise"):
        assert f'numberEntry(labelled("{label}"' in script
    assert "items.push(numberEntry(group));" in _function_body(
        script, "curveScaleControls"
    )
    css = (
        resource_files("hypermri.visualization")
        .joinpath("browser/viewer.css")
        .read_text(encoding="utf-8")
    )
    assert re.search(r"\.hm-csi-number-entry \{\s*order: 1;", css)
    assert 'labelled("y-axis range"' in script


def test_anatomy_colour_range_is_a_double_slider_redrawn_in_the_browser():
    script = browser_script()
    assert "doubleRangeSlider(" in _function_body(script, "anatomyControls")
    assert '"change:_anatomy_range": onAnatomyRange' in script
    limits = _function_body(script, "anatomyColourLimits")
    assert "value_range" in limits and "_anatomy_range" in limits


def test_masks_tool_offers_fit_statistics_per_mask():
    script = browser_script()
    assert '"change:_mask_statistics": onMaskStatistics' in script
    assert "maskStatisticsItems(v)" in _function_body(script, "maskControls")
    table = _function_body(script, "maskStatisticsTable")
    assert "t2_ms" in table and "freq_hz" in table and "n_voxels" in table
    assert "maskShowStatistics: true" in script


def test_fit_result_table_colours_values_by_bound_status():
    script = browser_script()
    assert '"change:_fit_table": onFitTable' in script
    table = _function_body(script, "fitResultTable")
    assert "row.freq_status" in table and "row.t2_status" in table
    assert "gauss_fwhm_hz" in table
    css = (
        resource_files("hypermri.visualization")
        .joinpath("browser/viewer.css")
        .read_text(encoding="utf-8")
    )
    for status in ("estimated", "at_lower_bound", "at_upper_bound", "fixed"):
        assert f".hm-csi-fit-status-{status}" in css


def test_plot_options_and_fit_number_rows_place_inputs_in_columns():
    script = browser_script()
    assert '"hm-csi-fit-settings-items hm-csi-grid-items"' in _function_body(
        script, "buildPage"
    )
    assert re.search(r'FIT_GRID_KINDS = \["detect", "options"\]', script)
    assert "FIT_GRID_KINDS.includes(kind)" in _function_body(script, "settingsTable")
    # Every rebuild measures the inputs again; the observer is detached at
    # cleanup.
    assert "watchGridColumns(v);" in _function_body(script, "buildGroups")
    assert "v.gridResizer?.disconnect();" in script
    span = _function_body(script, "spanWideItems")
    assert "Math.ceil(" in span and "columns.length" in span
    css = (
        resource_files("hypermri.visualization")
        .joinpath("browser/viewer.css")
        .read_text(encoding="utf-8")
    )
    assert re.search(
        r"\.hm-csi-grid-items \{\s*display: grid;\s*grid-template-columns: "
        r"repeat\(auto-fill,",
        css,
    )


def test_result_table_is_shown_and_residual_range_proposed_after_a_fit():
    script = browser_script()
    assert "fitShowResults: true," in script
    assert "const RESIDUAL_HALF_WIDTH_HZ = 10;" in script
    proposal = re.search(
        r"const firstFreq = Number\(peaks\[0\]\?\.freq_hz\);\n\s*if \(settings\.fitted",
        script,
    )
    assert proposal
    # Typed values are kept: only an empty range is replaced.
    assert "!v.fitResidualRange.some(Number.isFinite)" in script


def test_plot_options_table_does_not_cover_the_header_buttons():
    # With the header's flex basis of 100 % the table took the whole header
    # height and lay over the wrapped top bar, so clicks on the plot-options
    # toggle reached the denoise button underneath.
    css = (
        resource_files("hypermri.visualization")
        .joinpath("browser/viewer.css")
        .read_text(encoding="utf-8")
    )
    rule = re.search(r"\.hm-csi-header > \.hm-csi-options \{(.*?)\}", css, re.S)
    assert "flex: none" in rule.group(1)
