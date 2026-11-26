// Time panel: how the selected voxel changes along repetitions (or another
// dimension, `view.time.dim`), and optionally the mean inside a mask.
//
// Each point is the mean of the displayed spectrum over a spectral range
// (dragged in the curve panel), or its value at the selected spectral point.
// Python computes these means for every voxel and sends them in
// `_time_block` ([row][column][time], float32); the voxel curve therefore
// changes without asking Python. A red line marks the shown repetition;
// clicking a point shows that repetition in the map and spectrum.
//
// The panel is hidden when there is no time course (a single repetition, or
// a curve panel that does not show spectra).

function timeTitle(v) {
  const course = v.timeCourse;
  const [first, last] = course.range_values.map(formatPosition);
  const unit = v.view.curve_label;
  return course.follows_point
    ? `at ${first} ${unit}`
    : `mean over ${first} to ${last} ${unit}`;
}

function drawTimePanel(v) {
  const panel = v.panels.time;
  if (!panel) return;
  const time = v.view.time;
  const course = v.timeCourse;
  const shown = Boolean(time && course.range);
  panel.style.display = shown ? "" : "none";
  if (!shown) return;
  const view = v.view;
  const { row, column } = v.selection;
  const voxelName = `${view.row_dim} ${view.row_indices[row]}, ${view.column_dim} ${view.column_indices[column]}`;
  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      x: time.x,
      y: timeValues(v, row, column),
      name: voxelName,
      line: { width: 1.3, color: "#1f77b4" },
      marker: { size: 5 },
      hovertemplate: `%{x:.4g} ${time.label}<br>%{y:.4g}<extra>voxel</extra>`,
    },
  ];
  if (course.mask_y) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: time.x,
      y: course.mask_y,
      name: `mask ${course.mask}`,
      line: { width: 1.6, color: course.mask_color },
      marker: { size: 5 },
      hovertemplate: `%{x:.4g} ${time.label}<br>%{y:.4g}<extra>${course.mask}</extra>`,
    });
  }
  if (!v.model.get("_display").show_tooltips) hideHoverLabels(traces);
  const position = time.x[time.index];
  v.Plotly.react(
    panel,
    traces,
    {
      title: { text: timeTitle(v), font: { size: 13 } },
      margin: { l: 60, r: 10, t: 30, b: 42 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColour(v), size: 11 },
      hovermode: "closest",
      showlegend: true,
      legend: { orientation: "h", x: 0, y: 1.0, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)" },
      xaxis: { title: { text: time.label }, showgrid: showGrid(v) },
      yaxis: {
        title: { text: view.value_label },
        zeroline: false,
        autorange: true,
        showgrid: showGrid(v),
      },
      shapes: [
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
      ],
      uirevision: "time",
    },
    { displaylogo: false, responsive: true },
  );
}

function attachTimePanelEvents(v) {
  const panel = v.panels.time;
  if (!panel) return;
  panel.on("plotly_click", (event) => {
    const time = v.view.time;
    if (!time) return;
    const indices = { ...v.model.get("_controls").indices, [time.dim]: event.points[0].pointIndex };
    sendControls(v, { indices });
    v.el.focus({ preventScroll: true });
  });
}

// Python sent new time courses (new range, mask or mask edit).
function onTimeCourse(v) {
  readTimeCourse(v);
  buildGroups(v, ["time_course"]);
  if (v.Plotly) draw(v, { image: false });
}
