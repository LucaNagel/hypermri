// Curve panel: the curve of the selected voxel (its spectrum, or for example
// its time course over repetitions), or the means inside a mask when "ROI
// mean" is on. A red line marks the curve position shown in the image
// panel; clicking the curve moves it. The panel is hidden for an image
// without a curve dimension (image only). With fit results, the fitted
// spectrum of the selected voxel (`_fit_curves`, computed by Python as in
// plot) is drawn over the measured one.
//
// With a time panel, dragging along the spectrum chooses the spectral range
// averaged for the time course (shown as a red band); a double-click returns
// to the value at the selected point.
//
// While the map shows averaged repetitions, the curve keeps every repetition
// and a red band marks the averaged ones (as in plot); the map does not follow
// a curve position, so no position line is drawn.

// "ROI mean": the mean of the shown curves (for example the spectra in the
// chosen part and unit) and of the fitted spectra over the mask voxels of the
// shown plane, drawn like the voxel curve. The mean magnitude of the supplied
// samples over the whole mask (for FID data the mean FID) has its own axes at
// the top and right, as its axis is the acquisition axis.
function drawRoiMean(v) {
  const roiMean = v.roiMean;
  const view = v.view;
  const colour = v.maskPlanes.find((m) => m.name === roiMean.mask)?.color || "#1f77b4";
  const traces = [];
  if (roiMean.shown) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: `mean in ${roiMean.mask}, shown slice`,
      x: view.curve_x,
      y: roiMean.shown,
      line: { width: 1.3, color: "#1f77b4" },
      hovertemplate: `%{x:.4g} ${view.curve_label}<br>%{y:.4g}<extra></extra>`,
    });
  }
  for (const fit of roiMean.fits || []) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: fit.name,
      x: roiMean.fit_x,
      y: fit.y,
      line: { width: 1.3, color: fit.color },
      hovertemplate: `${fit.name}: %{y:.4g}<extra></extra>`,
    });
  }
  const own = traces.length === 0;
  traces.push({
    type: "scatter",
    mode: "lines",
    name: `mean magnitude in ${roiMean.mask}, supplied samples`,
    x: roiMean.x,
    y: roiMean.y,
    xaxis: own ? "x" : "x2",
    yaxis: own ? "y" : "y2",
    line: { width: 1, color: own ? colour : "#8c8c8c" },
    hovertemplate: `%{x:.4g} ${roiMean.x_label}<br>%{y:.4g}<extra></extra>`,
  });
  const rawX = {
    title: { text: roiMean.x_label },
    autorange: roiMean.invert_x ? "reversed" : true,
    showgrid: showGrid(v),
  };
  const rawY = {
    title: { text: roiMean.y_label },
    zeroline: false,
    autorange: true,
    showgrid: showGrid(v),
  };
  const layout = {
    // With the top axis, the legend names the mask instead of a title.
    title: own
      ? { text: `ROI mean magnitude: ${roiMean.mask}`, font: { size: 13 } }
      : undefined,
    margin: { l: 60, r: own ? 10 : 60, t: own ? 36 : 50, b: 45 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: textColour(v), size: 11 },
    hovermode: "x",
    showlegend: !own,
    legend: { orientation: "h", y: -0.25, yanchor: "top", x: 0, xanchor: "left" },
    uirevision: "roi-mean",
  };
  if (own) {
    layout.xaxis = rawX;
    layout.yaxis = rawY;
  } else {
    layout.xaxis = {
      title: { text: view.curve_label },
      autorange: view.curve_invert_x ? "reversed" : true,
      showgrid: showGrid(v),
    };
    layout.yaxis = {
      title: { text: view.value_label },
      zeroline: false,
      autorange: true,
      showgrid: showGrid(v),
    };
    layout.xaxis2 = { ...rawX, overlaying: "x", side: "top", showgrid: false };
    layout.yaxis2 = { ...rawY, overlaying: "y", side: "right", showgrid: false };
  }
  v.Plotly.react(v.panels.curve, traces, layout, { displaylogo: false, responsive: true });
}

// Fitted curves of the selected voxel, and the residual below them; none
// while Python still computes them for a newly selected voxel. A curve with
// its own `x` (the residual zero line) is drawn on that axis.
function fitTraces(v, row, column) {
  const fit = v.fitCurves;
  if (!fit.curves || fit.row !== row || fit.column !== column) return [];
  return fit.curves.map((curve) => ({
    type: "scatter",
    mode: "lines",
    name: curve.name,
    x: curve.x || fit.x,
    y: curve.y,
    line: { width: curve.dash ? 1 : 1.3, color: curve.color, dash: curve.dash || "solid" },
    hovertemplate: `${curve.name}: %{y:.4g}<extra></extra>`,
  }));
}

// While the fit tool is open: the frequency search range of every table
// line as a band in the line's colour (fitLineColour), in the shown unit.
function searchBandShapes(v) {
  if (v.model.get("tool") !== "fit" || !v.view.spectroscopic) return [];
  const controls = v.model.get("_controls");
  if (controls.processing?.show_fid) return [];
  const settings = v.model.get("_fit_tool") || {};
  const peaks = settings.peaks || [];
  return (settings.search_bands || []).flatMap((band, row) => {
    const limits = band[controls.spec_unit];
    if (!limits || peaks[row]?.fit === false) return [];
    return [
      {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: limits[0],
        x1: limits[1],
        y0: 0,
        y1: 1,
        fillcolor: fitLineColour(row),
        opacity: 0.13,
        line: { width: 0 },
        layer: "below",
      },
    ];
  });
}

// With the interleaved metabolites split, the curve of the metabolite the
// slider does not select at the same voxel (`_other_metabolite`, as in plot).
function otherMetaboliteTraces(v, row, column) {
  const other = v.otherMetabolite;
  if (!other.y || other.row !== row || other.column !== column) return [];
  return [
    {
      type: "scatter",
      mode: "lines",
      name: other.name,
      x: v.view.curve_x,
      y: other.y,
      line: { width: 1.3, color: "#2ca02c", dash: "dot" },
      hovertemplate: `${other.name}: %{y:.4g}<extra></extra>`,
    },
  ];
}

function onOtherMetabolite(v) {
  v.otherMetabolite = v.model.get("_other_metabolite") || {};
  if (v.Plotly) drawCurvePanel(v);
}

function onFitCurves(v) {
  v.fitCurves = v.model.get("_fit_curves") || {};
  if (v.Plotly) drawCurvePanel(v);
}

function drawCurvePanel(v) {
  const panel = v.panels.curve;
  if (!panel) return;
  const shown = v.view.curve_dim !== null;
  panel.style.display = shown ? "" : "none";
  if (!shown) return;
  if (v.roiMean.mask) {
    drawRoiMean(v);
    return;
  }
  const view = v.view;
  const { row, column, curve_index: k } = v.selection;
  const position = view.curve_x[k];
  const course = v.timeCourse;
  const band = (x0, x1) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0,
    x1,
    y0: 0,
    y1: 1,
    fillcolor: "rgba(214, 39, 40, 0.12)",
    line: { width: 0 },
    layer: "below",
  });
  const rangeShapes = [];
  if (view.time && course.range && !course.follows_point) {
    rangeShapes.push(band(course.range_values[0], course.range_values[1]));
  }
  if (view.average_range) rangeShapes.push(band(...view.average_range));
  const positionLine = view.map_fixed
    ? []
    : [
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: position,
          x1: position,
          y0: 0,
          y1: 1,
          line: { color: "#d62728", width: 1.5 },
        },
      ];
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: view.curve_x,
      y: curveValues(v, row, column),
      // Next to the other metabolite, name the one the slider selects.
      name: v.otherMetabolite.y
        ? `metabolite_idx ${v.model.get("_controls").indices?.metabolite_idx ?? 0}`
        : "data",
      line: { width: 1.3, color: "#1f77b4" },
      hovertemplate: `%{x:.4g} ${view.curve_label}<br>%{y:.4g}<extra></extra>`,
    },
    ...fitTraces(v, row, column),
    ...otherMetaboliteTraces(v, row, column),
  ];
  // Hover labels follow the display switch "tooltips"; the fit tool reads values
  // from its table instead.
  if (!v.model.get("_display").show_tooltips || v.model.get("tool") === "fit") {
    hideHoverLabels(traces);
  }
  const axes = curveAxes(v, traces);
  v.Plotly.react(
    v.panels.curve,
    traces,
    {
      title: {
        text: view.single_voxel
          ? "single voxel"
          : `${view.row_dim} ${view.row_indices[row]}, ` +
            `${view.column_dim} ${view.column_indices[column]}`,
        font: { size: 13 },
        // On the left, so that the legend on the right never covers it.
        x: 0,
        xref: "paper",
        xanchor: "left",
      },
      margin: { l: 60, r: 10, t: 30, b: 42 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColour(v), size: 11 },
      hovermode: "x",
      showlegend: v.fitCurves.curves !== undefined || v.otherMetabolite.y !== undefined,
      legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 1, xanchor: "right" },
      xaxis: { title: { text: view.curve_label }, showgrid: showGrid(v), ...axes.xaxis },
      yaxis: {
        title: { text: view.value_label },
        zeroline: false,
        showgrid: showGrid(v),
        ...axes.yaxis,
      },
      shapes: [
        ...positionLine,
        ...rangeShapes,
        ...searchBandShapes(v),
      ],
      // Dragging chooses the time course range; the drawn box is not kept.
      dragmode: view.time ? "select" : "zoom",
      selectdirection: "h",
      selections: [],
      // A new window or curve scale replaces a zoom made with the mouse.
      uirevision: `curve ${JSON.stringify(v.model.get("_curve_axes"))}`,
    },
    { displaylogo: false, responsive: true },
  );
}

// Hide the hover labels of *traces* in place. "none" (not "skip") keeps the
// click events that move the curve position; a hovertemplate would show a
// label even with "none", so it is removed.
function hideHoverLabels(traces) {
  for (const trace of traces) {
    trace.hoverinfo = "none";
    delete trace.hovertemplate;
  }
}

function attachCurvePanelEvents(v) {
  if (!v.panels.curve) return;
  v.panels.curve.on("plotly_click", (event) => {
    if (v.roiMean.mask || v.view.map_fixed) return;
    choose(v, { k: event.points[0].pointIndex });
    v.el.focus({ preventScroll: true });
  });
  v.panels.curve.on("plotly_selected", (event) => {
    const range = event?.range?.x;
    if (!v.view.time || v.roiMean.mask || !range) return;
    sendTimeCourse(v, { range: [Math.min(...range), Math.max(...range)] });
    // The red band shows the range; Plotly's selection box would keep its handles.
    v.Plotly.relayout(v.panels.curve, { selections: [] });
  });
  v.panels.curve.on("plotly_deselect", () => {
    if (v.view.time && !v.timeCourse.follows_point) sendTimeCourse(v, { range: null });
  });
}
