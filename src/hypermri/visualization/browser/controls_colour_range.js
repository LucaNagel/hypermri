// Control group "colour_range": range mode and limits of the map colours.
//
// The colour range only redraws: the browser sends it to Python, which stores
// it in state.spatial without recomputing. One slider with two handles sets
// the lower and upper colour limit; its scale is percent or the values of
// the plane. Percentages refer to the shown map
// ("map") or the whole plane ("plane"); the whole-plane reference exists only
// in the browser.

// Shown names of the range modes; the values stay the names Python knows.
const RANGE_MODE_LABELS = {
  "Min/Max": "full range of the data",
  Absolute: "fixed values",
};

function colourRangeControls(v) {
  if (v.view.single_voxel) return [];
  const choices = [];
  for (const mode of v.view.range_modes) {
    if (mode === "Percentage") {
      choices.push(["% of this map", "Percentage map"], ["% of whole plane", "Percentage plane"]);
    } else {
      choices.push([RANGE_MODE_LABELS[mode] ?? mode, mode]);
    }
  }
  const choiceOf = (settings) =>
    settings.range_mode === "Percentage" ? `Percentage ${v.percentReference}` : settings.range_mode;
  const settings = v.model.get("_map_range");
  const mode = select(choices, choiceOf(settings));
  mode.title =
    "What the sliders refer to: percent of the values of the shown map or of the whole " +
    "plane, or the values themselves. The full range follows the data when the complex " +
    "part or line broadening changes; fixed values keep their limits.";
  mode.addEventListener("change", () => {
    const [rangeMode, reference] = mode.value.split(" ");
    const current = v.model.get("_map_range");
    const z = planeValues(v, v.selection.curve_index);
    // Keep the shown colours: convert the current limits to the new mode.
    const [low, high] = mapColourLimits(v, z, current);
    let limits = [low, high];
    if (rangeMode === "Percentage") {
      v.percentReference = reference;
      if (current.range_mode === "Percentage") {
        limits = [current.vmin, current.vmax];
      } else {
        const [start, end] = reference === "map" ? finiteRange(z.flat()) : planeRange(v);
        const span = end - start;
        // Percentages stay within 0-100 as in plot; limits outside the
        // reference range would otherwise show as values such as -300 %.
        const percent = (value) => Math.min(100, Math.max(0, ((value - start) / span) * 100));
        limits = [percent(low), percent(high)];
        if (!(limits[0] < limits[1])) limits = [0, 100];
      }
    }
    setMapRange(v, { range_mode: rangeMode, vmin: limits[0], vmax: limits[1] });
    // A new percentage reference with the same limits leaves _map_range
    // unchanged, so the listener does not redraw by itself.
    if (v.Plotly) drawImagePanel(v);
  });
  const [bottom, top, step] = sliderBounds(v, settings);
  const percent = settings.range_mode === "Percentage";
  const format = (value) => (percent ? `${Number(value).toFixed(1)} %` : `${Number(Number(value).toPrecision(4))}`);
  // While a handle moves, the map follows in this browser; Python gets the
  // limits when the handle is let go.
  const limits = doubleRangeSlider(settings.vmin, settings.vmax, bottom, top, step, format, (low, high, done) => {
    setMapRange(v, { vmin: low, vmax: high }, done);
    return [low, high];
  });
  limits.title = "Values at or below the lower limit get the first colour, at or above the upper the last";
  v.mapRangeInputs = { mode, limits, choiceOf };
  return [labelled("colours scaled to", mode), labelled("range", limits)];
}

// Slider limits and step: 0-100 % for percentages, otherwise the values of
// the whole plane, widened to include the current limits.
function sliderBounds(v, settings) {
  if (settings.range_mode === "Percentage") return [0, 100, 0.5];
  const [start, end] = planeRange(v);
  const bottom = Math.min(start, settings.vmin);
  const top = Math.max(end, settings.vmax);
  const span = top - bottom;
  return [bottom, top, span > 0 ? span / 500 : 1];
}

// Set the colour range; `send` false only redraws in this browser.
function setMapRange(v, update, send = true) {
  v.model.set("_map_range", { ...v.model.get("_map_range"), ...update });
  if (send) v.model.save_changes();
}

// Python or another view changed the range: show it and redraw.
function onMapRange(v) {
  const settings = v.model.get("_map_range");
  if (v.mapRangeInputs) {
    const { mode, limits, choiceOf } = v.mapRangeInputs;
    const unitChanged = mode.value.startsWith("Percentage") !== (settings.range_mode === "Percentage");
    // A new slider scale when the unit changes or a limit lies beyond it.
    if (unitChanged || !limits.fits(settings.vmin, settings.vmax)) {
      buildGroups(v, ["colour_range"]);
    } else {
      mode.value = choiceOf(settings);
      limits.show(settings.vmin, settings.vmax);
    }
  }
  if (!v.Plotly) return;
  drawImagePanel(v);
  // The "map colours" curve scale follows the colour limits.
  if (v.model.get("_curve_axes").scale === "map") drawCurvePanel(v);
}
