// Control group "signal_decay": decay of CSI raw data over the acquired
// k-space points (in acquisition order, not FID time), with a time-domain
// filter along these points, a spatial k-space filter and the signal end.
// Offered only in views opened by plot_signal_decay.
//
// A small plot below the inputs shows the decay curve of the unfiltered
// ("Original") and filtered data, the straight line fitted to the
// unfiltered curve from the signal end on ("Tail fit"), the signal end
// (red) and the suggested signal end (grey, dashed). A click on the plot
// sets the signal end. "Confirm" stores signal end and filter on the scan.
//
// Python owns the choices and computes the curves (`_signal_decay_tool`,
// see hypermri.visualization.signal_decay_tool). Every input goes to Python
// as a `{type: "signal_decay"}` message; Python answers with a new
// `_signal_decay_tool`, and with a new block when the filter changed.

function sendSignalDecay(v, action, content = {}) {
  v.model.send({ type: "signal_decay", action, ...content });
}

const DECAY_CURVE_CHOICES = [
  ["magnitude", "magnitude"],
  ["cumulative magnitude", "cumulative"],
  ["band power", "band_power"],
  ["cumulative band power", "cumulative_band_power"],
];
const SPATIAL_FILTER_CHOICES = [
  ["none", "none"],
  ["Gaussian", "gaussian"],
  ["Fermi", "fermi"],
];
const DECAY_CURVE_COLOURS = { Original: "#1f77b4", Filtered: "#ff7f0e", "Tail fit": "#7f7f7f" };

function signalDecayControls(v) {
  const settings = v.model.get("_signal_decay_tool") || {};
  if (!settings.offered) return [];
  const option = (name) => (value) => sendSignalDecay(v, "option", { name, value });
  const number = (name, label, title, width = 5) =>
    labelled(label, fitNumber(settings[name], width, title, option(name)));
  const choice = (name, options, title) => {
    const node = select(options, settings[name]);
    node.title = title;
    node.addEventListener("change", () => option(name)(node.value));
    return node;
  };
  const checkbox = (name, label, title) =>
    labelled(label, fitCheckbox(settings[name], title, option(name)));

  const last = settings.n_points - 1;
  const filterItems = [
    labelled("filter", choice("filter_type", settings.filter_types, "Time-domain filter along the acquired points")),
  ];
  if (settings.filter_type === "Exponential") {
    filterItems.push(number("decay_constant", "decay", "Exponential attenuation per acquired point (not Hz)"));
  }
  if (settings.filter_type === "Hamming") {
    filterItems.push(number("alpha", "alpha", "Hamming coefficient, normally 0 to 1", 4));
  }
  if (settings.filter_type === "Hamming" || settings.filter_type === "Hann") {
    filterItems.push(number("center_position", "centre", `Centre of the window in acquired points, 0 to ${last}`, 4));
  }
  filterItems.push(
    checkbox("remove_lowest_signal", "remove lowest", "Remove the lowest signal (filter_signal)"),
    checkbox("apply_flipangle_correction", "flip angle corr.", "Correct the signal loss of repeated excitations"),
  );
  if (settings.apply_flipangle_correction) {
    filterItems.push(number("flipangle_deg", "flip angle °", "Flip angle in degrees, 0 to 90", 4));
  }

  const spatialItems = [
    labelled("k-space", choice("spatial_filter", SPATIAL_FILTER_CHOICES, "Spatial filter of the k-space")),
  ];
  if (settings.spatial_filter === "gaussian") {
    spatialItems.push(number("sigma_per_mm", "sigma 1/mm", "Width of the Gaussian in cycles/mm"));
  }
  if (settings.spatial_filter === "fermi") {
    spatialItems.push(
      number("radius_per_mm", "radius 1/mm", "Radius of the Fermi filter in cycles/mm"),
      number("transition_per_mm", "edge 1/mm", "Edge width of the Fermi filter in cycles/mm"),
    );
  }

  const curveItems = [
    labelled("curve", choice("curve", DECAY_CURVE_CHOICES, "Decay curve over the acquired points")),
  ];
  if (settings.curve.includes("band_power")) {
    curveItems.push(
      labelled(
        "bands Hz",
        idealText(
          settings.bands_hz, 12, "whole spectrum",
          "Frequency bands as low:high in Hz, separated by commas, e.g. -330:-150, 200:420; empty is the whole spectrum",
          option("bands_hz"),
        ),
      ),
    );
  }
  curveItems.push(
    checkbox("show_tail", "tail fit", "Draw the line fitted to the unfiltered curve from the signal end on"),
    checkbox("subtract_tail", "subtract tail", "Subtract that line from the curves (display only)"),
    checkbox("show_filtered", "show filtered", "Show the data reconstructed with the filter in the map and spectrum"),
  );

  const end = number("signal_end", "signal end", `Kept acquired points (filter "None") and start of the tail fit, 0 to ${last}; or click the plot`, 5);
  const confirm = button("Confirm", "Store signal_end and filter_params on the scan; the acquired data are not changed");
  confirm.classList.add("hm-csi-primary");
  confirm.disabled = !settings.can_confirm;
  confirm.addEventListener("click", () => sendSignalDecay(v, "confirm"));

  const message = element("span", "hm-csi-mask-info", settings.curve_error || settings.message || "");
  const messageRow = element("div", "hm-csi-tool-row");
  messageRow.append(message);

  if (!v.decayPlot) {
    v.decayPlot = element("div", "hm-csi-decay-plot");
  }
  return [
    settingsTable([
      ["Filter", "options", filterItems],
      ["Spatial", "options", spatialItems],
      ["Curve", "detect", curveItems],
      ["Signal end", "run", [end, confirm]],
    ]),
    messageRow,
    v.decayPlot,
  ];
}

// Draw the decay curves into the plot of the tool. Called when the tool's
// inputs are built and once Plotly.js is loaded.
function drawSignalDecay(v) {
  const settings = v.model.get("_signal_decay_tool") || {};
  if (!v.Plotly || !v.decayPlot || !settings.offered) return;
  const curves = settings.curves || [];
  const x = curves.length ? curves[0].y.map((_, i) => i) : [];
  const traces = curves.map((curve) => ({
    type: "scatter",
    mode: "lines",
    name: curve.name,
    x,
    y: curve.y,
    line: { width: curve.name === "Tail fit" ? 1 : 1.5, color: DECAY_CURVE_COLOURS[curve.name], dash: curve.name === "Tail fit" ? "dot" : "solid" },
    hovertemplate: `${curve.name}: %{y:.4g} at point %{x}<extra></extra>`,
  }));
  const marker = (position, colour, dash) => ({
    type: "line",
    xref: "x",
    yref: "paper",
    x0: position,
    x1: position,
    y0: 0,
    y1: 1,
    line: { color: colour, width: 1.5, dash },
  });
  const shapes = [marker(settings.signal_end, "#d62728", "solid")];
  if (settings.suggested_end != null) shapes.push(marker(settings.suggested_end, "#555", "dash"));
  const layout = {
    height: 260,
    margin: { l: 60, r: 10, t: 10, b: 40 },
    xaxis: { title: { text: "acquired k-space point" } },
    yaxis: { title: { text: settings.y_label } },
    shapes,
    showlegend: true,
    legend: { orientation: "h", y: 1.12 },
    hovermode: "x",
  };
  v.Plotly.react(v.decayPlot, traces, layout, { displaylogo: false, responsive: true });
  if (!v.decayPlotClicks) {
    v.decayPlotClicks = true;
    v.decayPlot.on("plotly_click", (event) => {
      const point = event.points?.[0];
      if (!point) return;
      const value = Math.min(Math.max(Math.round(point.x), 0), settings.n_points - 1);
      sendSignalDecay(v, "option", { name: "signal_end", value });
    });
  }
}

function onSignalDecayTool(v) {
  buildGroups(v, ["tool", "signal_decay"]);
  showChosenTool(v);
  drawSignalDecay(v);
}
