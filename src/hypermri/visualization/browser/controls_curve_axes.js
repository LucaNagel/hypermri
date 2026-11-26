// Control group "curve_scale": y axis and frequency window of the curve
// panel, and whether the fitted spectrum and its residual are drawn. They
// only change the curve; they are the synchronised `_curve_axes` trait
// (ScanView.set_curve_scale, set_frequency_window and show_fit_curves),
// which Python stores in state.spectral (couple_ylims, display_range,
// show_fit, show_fit_residual). Python sends new fit curves after the
// fit choices.

const CURVE_SCALES = [
  ["fit to curves", "auto"],
  ["whole plane", "plane"],
  ["map colours", "map"],
];

function setCurveAxes(v, update) {
  v.model.set("_curve_axes", { ...v.model.get("_curve_axes"), ...update });
  v.model.save_changes();
  if (v.Plotly) drawCurvePanel(v);
}

function curveScaleControls(v) {
  if (v.view.curve_dim === null) return [];
  const axes = v.model.get("_curve_axes");
  const scale = select(CURVE_SCALES, axes.scale);
  scale.title =
    "y axis of the curve: fitted to the shown curves, the value range of the " +
    "whole plane, or the colour limits of the map (Couple Y-Limits in plot)";
  scale.addEventListener("change", () => setCurveAxes(v, { scale: scale.value }));
  const items = [labelled("y-axis range", scale)];
  if (!v.view.spectroscopic) return items;

  // Frequency window ("Limit" in plot), in the unit shown now. A window in
  // the other unit is kept in Python but not shown.
  const unit = v.model.get("_controls").spec_unit;
  const shown = frequencyWindow(v);
  const [low, high] = shown ?? finiteRange(v.view.curve_x);
  const limit = element("input");
  limit.type = "checkbox";
  limit.checked = shown !== null;
  limit.disabled = v.model.get("_controls").processing?.show_fid === true;
  limit.title = "Show only this frequency range; it stays for every voxel";
  const number = (value) => {
    const input = element("input");
    input.type = "number";
    input.step = "any";
    input.value = Number(value.toPrecision(6));
    input.style.width = "6em";
    return input;
  };
  const from = number(low);
  const to = number(high);
  const send = () => {
    const limits = [Number(from.value), Number(to.value)];
    if (!limit.checked) {
      setCurveAxes(v, { window: null });
    } else if (from.value !== "" && to.value !== "" && limits[0] !== limits[1]) {
      setCurveAxes(v, { window: limits, window_unit: unit });
    } else {
      v.status.textContent = "Enter two different numbers for the frequency window.";
    }
  };
  for (const input of [limit, from, to]) input.addEventListener("change", send);
  const group = labelled(`limit [${unit}]`, limit);
  group.append(from, to);
  items.push(numberEntry(group));

  // Fitted spectrum and residual (data − fit) of the selected voxel, as
  // "Show Fit" and "Show Residual" in plot (ScanView.show_fit_curves).
  const fit = element("input");
  fit.type = "checkbox";
  fit.checked = axes.fit !== false;
  fit.title = "Draw the fitted spectrum of the selected voxel when a fit is shown";
  fit.addEventListener("change", () => setCurveAxes(v, { fit: fit.checked }));
  const residual = element("input");
  residual.type = "checkbox";
  residual.checked = axes.residual !== false;
  residual.disabled = !fit.checked;
  residual.title = "Draw data − fit below the curves, shifted, with a dashed zero line";
  residual.addEventListener("change", () => setCurveAxes(v, { residual: residual.checked }));
  items.push(labelled("fit", needsPython(v, fit)), labelled("residual", needsPython(v, residual)));
  return items;
}

// The frequency window [low, high] that applies to the shown curve, or null:
// it applies to spectra shown in the unit the window was set in.
function frequencyWindow(v) {
  const axes = v.model.get("_curve_axes");
  const controls = v.model.get("_controls");
  if (!v.view.spectroscopic || !axes.window || controls.processing?.show_fid) return null;
  if (axes.window_unit !== controls.spec_unit) return null;
  return [Math.min(...axes.window), Math.max(...axes.window)];
}

// y range of the "map colours" scale, as couple_ylims in plot: from zero or
// above for the magnitude, symmetric around zero for the other parts (at
// most pi for the phase).
function mapCurveRange(v) {
  const [low, high] = mapColourLimits(v, planeValues(v, v.selection.curve_index));
  const part = v.model.get("_controls").spec_part;
  if (part === "Magnitude") return [Math.max(0, low), Math.max(0, high)];
  let maxAbs = Math.max(Math.abs(low), Math.abs(high));
  if (part === "Phase") maxAbs = Math.min(maxAbs, Math.PI);
  return [-maxAbs, maxAbs];
}

// y range of the traces inside the window [low, high] with a 5 % margin, as
// plot fits the y axis to a limited frequency range; null without values.
function windowCurveRange(traces, [low, high]) {
  const values = [];
  for (const trace of traces) {
    if (!trace.x || !trace.y) continue;
    for (let i = 0; i < trace.y.length; i += 1) {
      const x = trace.x[i];
      const y = trace.y[i];
      if (x >= low && x <= high && Number.isFinite(y)) values.push(y);
    }
  }
  if (!values.length) return null;
  const [yMin, yMax] = finiteRange(values);
  const margin = 0.05 * (yMax - yMin);
  return [yMin - margin, yMax + margin];
}

// x and y axis settings of the curve panel for the shown traces.
function curveAxes(v, traces) {
  const window = frequencyWindow(v);
  const xaxis = window
    ? { range: v.view.curve_invert_x ? [window[1], window[0]] : window, autorange: false }
    : { autorange: v.view.curve_invert_x ? "reversed" : true };
  const scale = v.model.get("_curve_axes").scale;
  let yRange = null;
  if (scale === "plane") yRange = planeRange(v);
  else if (scale === "map") yRange = mapCurveRange(v);
  else if (window) yRange = windowCurveRange(traces, window);
  const yaxis = yRange && yRange[0] < yRange[1] ? { range: yRange, autorange: false } : { autorange: true };
  return { xaxis, yaxis };
}

// Set from Python (ScanView.set_curve_scale, set_frequency_window).
function onCurveAxes(v) {
  buildGroups(v, ["curve_scale"]);
  if (v.Plotly) drawCurvePanel(v);
}
