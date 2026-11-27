// Image panel: the map of the displayed plane at the chosen curve position.
//
// Drawn from bottom to top: anatomy, CSI map, overlay, masks. A crosshair and
// a box mark the selected voxel. Clicking selects a voxel, or adds a polygon
// corner while the polygon mask tool is on.

// Colour limits of the map from state.spatial, as _apply_spatial_clim in
// render_data.py: percentages are fractions of the finite range of the
// reference, the other range modes are limits in signal units.
// Curve position for the title: whole numbers (indices) without decimals.
function formatPosition(value) {
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}

function mapColourLimits(v, z, settings = v.model.get("_map_range")) {
  if (settings.range_mode !== "Percentage") return [settings.vmin, settings.vmax];
  const [low, high] = v.percentReference === "map" ? finiteRange(z.flat()) : planeRange(v);
  const span = high - low;
  return [low + (settings.vmin / 100) * span, low + (settings.vmax / 100) * span];
}

function textColour(v) {
  return getComputedStyle(v.el).color || "#333";
}

// Scalebar from Python (ScanView.set_display) as a line and a label, placed
// as in plot: 5 % from the side and 6 % of the height towards the centre.
// The ranges run from the bottom to the top of the screen.
function scalebarTraces(v) {
  const bar = v.view.scalebar;
  if (!bar) return [];
  const [left, right] = [Math.min(...v.view.x_range), Math.max(...v.view.x_range)];
  const [bottom, top] = v.view.y_range;
  const height = top - bottom;
  const length = Math.min(bar.length, right - left);
  const pad = (right - left) * 0.05;
  const x0 = bar.location.includes("left") ? left + pad : right - pad - length;
  const upper = bar.location.includes("upper");
  const y = upper ? top - height * 0.06 : bottom + height * 0.06;
  const textY = upper ? y - height * 0.04 : y + height * 0.04;
  const common = { type: "scatter", hoverinfo: "skip", showlegend: false, cliponaxis: false };
  return [
    { ...common, mode: "lines", x: [x0, x0 + length], y: [y, y], line: { color: bar.color, width: 4 }, name: "scalebar" },
    {
      ...common,
      mode: "text",
      x: [x0 + length / 2],
      y: [textY],
      text: [bar.label],
      textfont: { color: bar.color, size: bar.fontsize },
      name: "scalebar label",
    },
  ];
}

// With "min -> NaN" voxels below the colour minimum are not drawn.
function hideBelowMinimum(z, zmin) {
  return z.map((line) => line.map((value) => (value < zmin ? null : value)));
}

// Hover text per voxel: source indices and the coverage of every mask.
function hoverIndices(v) {
  return v.view.row_indices.map((rowIndex, row) =>
    v.view.column_indices.map((columnIndex, column) => {
      let text = "";
      for (const mask of v.maskPlanes) {
        const value = mask.plane?.[row]?.[column] ?? 0;
        if (value > 0) text += `<br>${mask.name} ${(value * 100).toFixed(0)} %`;
      }
      return [rowIndex, columnIndex, text];
    }),
  );
}

function anatomyShown(v) {
  return Boolean(v.anatomyZ) && v.showAnatomy;
}

// Colour limits of the anatomy: the percentages of "_anatomy_range" applied
// to the value range of the shown slice, as Python computes them for the
// range mode "Percentage"; otherwise the limits Python sent.
function anatomyColourLimits(v) {
  const layer = v.view.anatomy;
  const percent = v.model.get("_anatomy_range");
  const range = layer.value_range;
  if (!percent?.length || !range) return [layer.zmin ?? undefined, layer.zmax ?? undefined];
  return percent.map((share) => range[0] + (share / 100) * (range[1] - range[0]));
}

// Drawn first, fully opaque and without hover, under the CSI map (as in plot).
function anatomyTraces(v) {
  if (!anatomyShown(v)) return [];
  const layer = v.view.anatomy;
  const [zmin, zmax] = anatomyColourLimits(v);
  return [
    {
      type: "heatmap",
      z: v.anatomyZ,
      x: layer.x,
      y: layer.y,
      zmin,
      zmax,
      colorscale: layer.colorscale,
      zsmooth: layer.zsmooth,
      showscale: false,
      hoverinfo: "skip",
      name: "anatomy",
    },
  ];
}

// Drawn above the CSI map without hover, so clicks still select CSI voxels.
function overlayTraces(v) {
  if (!v.overlayZ) return [];
  const layer = v.view.overlay;
  return [
    {
      type: "heatmap",
      z: v.overlayZ,
      x: layer.x,
      y: layer.y,
      zmin: layer.zmin ?? undefined,
      zmax: layer.zmax ?? undefined,
      colorscale: layer.colorscale,
      opacity: v.model.get("_overlay_opacity"),
      hoverinfo: "skip",
      name: "overlay",
      colorbar: {
        ...colourBarPlace(v, v.model.get("_display").show_colorbar ? 1 : 0),
        title: { text: layer.label, side: "right" },
      },
    },
  ];
}

function maskTraces(v) {
  return v.maskPlanes
    .filter((mask) => mask.plane)
    .map((mask) => ({
      type: "heatmap",
      z: mask.plane.map((line) => line.map((value) => (value > 0 ? value : null))),
      x: v.view.x,
      y: v.view.y,
      zmin: 0,
      zmax: 1,
      colorscale: [
        // Partly covered edge voxels stay visible: coverage 0+ starts at 35 % colour.
        [0, hexToRgba(mask.color, 0.35)],
        [1, hexToRgba(mask.color, 1)],
      ],
      opacity: mask.alpha,
      showscale: false,
      hoverinfo: "skip",
      name: mask.name,
    }));
}

// Outlines drawn into the masks, as stored at full precision
// (ScanView.mask_outlines); the heatmap above only shows voxel coverage.
// Python sends the boundary of the area left after erasing; outlines of three
// or more points are closed.
function maskOutlineTraces(v) {
  return v.maskPlanes.flatMap((mask) =>
    (mask.outlines || []).map((outline) => {
      const closed = outline.x.length > 2;
      return {
        type: "scatter",
        mode: "lines",
        x: closed ? [...outline.x, outline.x[0]] : outline.x,
        y: closed ? [...outline.y, outline.y[0]] : outline.y,
        line: { color: mask.color, width: 1.5 },
        hoverinfo: "skip",
        showlegend: false,
        name: `${mask.name} outline`,
      };
    }),
  );
}

// Where the shown slices of other views cross this image
// (ScanView.show_slices_of): a semi-transparent band with a solid edge.
// "skip" lets clicks through to the map, so masks can be drawn under it.
function sliceLineTraces(v) {
  if (v.model.get("_display").show_slice_lines !== true) return [];
  return (v.model.get("_slice_lines") || []).map((line) => ({
    type: "scatter",
    mode: "lines",
    x: line.x,
    y: line.y,
    fill: "toself",
    fillcolor: hexToRgba(line.colour, 0.25),
    line: { color: line.colour, width: 1.5 },
    hoverinfo: "skip",
    showlegend: false,
    name: line.name,
  }));
}

// Set from Python when a linked view shows another slice. The display
// controls offer the switch for the bands only while there are bands.
function onSliceLines(v) {
  buildGroups(v, ["display"]);
  if (v.Plotly) drawImagePanel(v);
}

// Mouse pointer shared by linked views (ScanView.link_pointer). The views of
// one page talk over this event target, without Python: each sends the LPS
// position under its mouse, and the others mark it with a ring at shapes[0].
const POINTER_EVENTS = (globalThis.hypermriPointerEvents ??= new EventTarget());

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// _pointer_plane: LPS = origin_mm + x * x_step_mm + y * y_step_mm (mm).
function axesToLps(plane, x, y) {
  return [0, 1, 2].map((i) => plane.origin_mm[i] + x * plane.x_step_mm[i] + y * plane.y_step_mm[i]);
}

// The point of the plane closest to an LPS position, in plot axes, and how
// far the position lies from the plane.
function lpsToAxes(plane, point) {
  const d = point.map((value, i) => value - plane.origin_mm[i]);
  const a = plane.x_step_mm;
  const b = plane.y_step_mm;
  const [aa, bb, ab, da, db] = [dot(a, a), dot(b, b), dot(a, b), dot(d, a), dot(d, b)];
  const det = aa * bb - ab * ab;
  let x = 0;
  let y = 0;
  if (det > 1e-9 * aa * bb && det > 0) {
    x = (da * bb - db * ab) / det;
    y = (db * aa - da * ab) / det;
  } else if (aa > 0) {
    x = da / aa;
  } else if (bb > 0) {
    y = db / bb;
  }
  const away = d.map((value, i) => value - x * a[i] - y * b[i]);
  return { x, y, distance: Math.sqrt(dot(away, away)) };
}

function inRange(value, range) {
  return value >= Math.min(...range) && value <= Math.max(...range);
}

function sendPointer(v, lps) {
  const link = v.model.get("_pointer_link");
  if (!link) return;
  POINTER_EVENTS.dispatchEvent(new CustomEvent("pointer", { detail: { link, source: v, lps } }));
}

function onLinkedPointer(v, event) {
  const { link, source, lps } = event.detail;
  if (source === v || !link || link !== v.model.get("_pointer_link")) return;
  const plane = v.model.get("_pointer_plane");
  v.linkedPointer = lps && plane?.origin_mm ? lpsToAxes(plane, lps) : null;
  const panel = v.panels.image;
  if (v.Plotly && panel?._fullLayout && panel.style.display !== "none") {
    // Single attributes: a whole object at "shapes[0]" would insert a shape.
    const update = {};
    for (const [key, value] of Object.entries(linkedPointerShape(v))) {
      update[`shapes[0].${key}`] = value;
    }
    v.Plotly.relayout(panel, update);
  }
}

// A filled ring inside the shown slice, an open ring in front of or behind it.
function linkedPointerShape(v) {
  const point = v.linkedPointer;
  const view = v.view;
  if (!point || !inRange(point.x, view.x_range) || !inRange(point.y, view.y_range)) {
    return { type: "circle", visible: false, x0: 0, x1: 0, y0: 0, y1: 0, fillcolor: "rgba(0,0,0,0)" };
  }
  const span = Math.max(
    Math.abs(view.x_range[1] - view.x_range[0]),
    Math.abs(view.y_range[1] - view.y_range[0]),
  );
  const r = 0.02 * span;
  const thickness = v.model.get("_pointer_plane").thickness_mm || 0;
  const inside = point.distance <= Math.max(thickness / 2, 1e-3);
  return {
    type: "circle",
    visible: true,
    x0: point.x - r,
    x1: point.x + r,
    y0: point.y - r,
    y1: point.y + r,
    line: { color: "#ffffff", width: 2 },
    fillcolor: inside ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0)",
  };
}

// Mouse moves are sent at most once per frame.
function attachPointerEvents(v, panel) {
  let frame = null;
  let lastEvent = null;
  const send = () => {
    frame = null;
    const plane = v.model.get("_pointer_plane");
    const layout = panel._fullLayout;
    if (!plane?.origin_mm || !layout?.xaxis?.p2d || v.view.single_voxel) return;
    const box = panel.getBoundingClientRect();
    const x = layout.xaxis.p2d(lastEvent.clientX - box.left - layout.xaxis._offset);
    const y = layout.yaxis.p2d(lastEvent.clientY - box.top - layout.yaxis._offset);
    const inside = inRange(x, layout.xaxis.range) && inRange(y, layout.yaxis.range);
    sendPointer(v, inside ? axesToLps(plane, x, y) : null);
  };
  panel.addEventListener("mousemove", (event) => {
    lastEvent = event;
    frame ??= requestAnimationFrame(send);
  });
  panel.addEventListener("mouseleave", () => sendPointer(v, null));
}

function outlineColour(v) {
  if (v.erase) return "#ffffff";
  return v.maskPlanes.find((mask) => mask.name === v.activeMask)?.color || "#ff0000";
}

// The polygon being clicked, as a dashed outline.
function pendingShape(v) {
  if (!v.pending.length) return [];
  const path =
    v.pending.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" ") +
    (v.pending.length > 2 ? " Z" : "");
  return [
    {
      type: "path",
      path,
      line: { color: outlineColour(v), width: 2, dash: "dash" },
    },
  ];
}

// Horizontal room of one colour bar with its tick labels and title.
const COLOUR_BAR_STEP_PX = 66;
// Gap between the map (or the colour bar before) and a colour bar.
const COLOUR_BAR_GAP_PX = 4;

// Names of the image traces that show a colour bar, left to right.
function colourBarTraces(v) {
  return [
    ...(v.model.get("_display").show_colorbar ? ["csi"] : []),
    ...(v.overlayZ ? ["overlay"] : []),
  ];
}

// Colour bar number `index` (0 next to the map) placed just right of the
// map and as tall as it (at least 80 px). Plotly places colour bars in fractions of the plot
// area, but an equal-scale map can fill less than that area, so the place
// comes from the map's domain in the last drawn layout; fitColourBars
// corrects it once the new layout is known. The plot-area width comes from
// the panel and the margins kept for it, not from Plotly: a colour bar
// placed from Plotly's width, which it narrows to make room for that bar,
// would push itself further out on every draw until the map vanished.
function colourBarPlace(v, index) {
  const full = v.panels.image?._fullLayout;
  const xDomain = full?.xaxis?.domain ?? [0, 1];
  const yDomain = full?.yaxis?.domain ?? [0, 1];
  const margin = v.imageFit?.margin ?? IMAGE_MARGIN;
  const width = (v.panels.image?.clientWidth || 0) - margin.l - margin.r;
  const offset = COLOUR_BAR_GAP_PX + index * COLOUR_BAR_STEP_PX;
  // At least 80 px tall: a bar of a few pixels (a map drawn into a box that
  // was still narrow) would stop Plotly with "Something went wrong with axis
  // scaling" and keep the map at that size.
  const height = (v.panels.image?.clientHeight || 0) - margin.t - margin.b;
  const len = Math.min(1, Math.max(yDomain[1] - yDomain[0], height > 0 ? 80 / height : 0.5));
  return {
    thickness: 12,
    xpad: 0,
    ypad: 0,
    xanchor: "left",
    x: xDomain[1] + offset / (width > 0 ? width : 300),
    yanchor: "middle",
    y: (yDomain[0] + yDomain[1]) / 2,
    len,
  };
}

// Move the colour bars next to the map after Plotly has fitted the map into
// the panel (a draw or a resize); nothing is redrawn when they are in place.
function fitColourBars(v) {
  const panel = v.panels.image;
  if (!v.Plotly || !panel?._fullLayout?.xaxis || !Array.isArray(panel.data)) return;
  const names = colourBarTraces(v);
  const update = { "colorbar.x": [], "colorbar.y": [], "colorbar.len": [] };
  const indices = [];
  panel.data.forEach((trace, i) => {
    const index = names.indexOf(trace.name);
    if (index < 0 || !trace.colorbar) return;
    const place = colourBarPlace(v, index);
    const moved = ["x", "y", "len"].some(
      (key) => Math.abs((trace.colorbar[key] ?? 0) - place[key]) > 1e-3,
    );
    if (!moved) return;
    indices.push(i);
    update["colorbar.x"].push(place.x);
    update["colorbar.y"].push(place.y);
    update["colorbar.len"].push(place.len);
  });
  if (indices.length) v.Plotly.restyle(panel, update, indices);
}

// Keep what fixes the shape of the map (axis ranges at equal scale, margins)
// for sizeImagePanel; returns the margins for the Plotly layout.
function rememberImageFit(v, xRange, yRange, margin) {
  v.imageFit = { xRange, yRange, margin };
  return margin;
}

// Margins of the map: the axis tick labels and the title, no axis titles
// (the axes are named in the tooltip and by the slice sliders).
const IMAGE_MARGIN = { l: 40, r: 10, t: 30, b: 24 };

// Give the map panel the width at which the map fills the plot height, so
// that no empty band remains above and below it; the curve and time panels
// share the rest. When the row leaves less width than that, the map becomes
// lower instead, and so do the curve and time panels beside it
// (--hm-panel-height), so that no empty band remains above any of them.
function sizeImagePanel(v) {
  const panel = v.panels.image;
  if (!panel) return;
  if (!v.imageFit || panel.style.display === "none") {
    v.el.style.removeProperty("--hm-panel-height");
    return;
  }
  const { xRange, yRange, margin } = v.imageFit;
  const spanX = Math.abs(xRange[xRange.length - 1] - xRange[0]);
  const spanY = Math.abs(yRange[yRange.length - 1] - yRange[0]);
  if (!(spanX > 0 && spanY > 0)) return;
  const fullHeight =
    parseFloat(getComputedStyle(panel).getPropertyValue("--hm-plot-height")) || 400;
  const plotHeight = fullHeight - margin.t - margin.b;
  const width = Math.round((plotHeight * spanX) / spanY + margin.l + margin.r);
  // Never wider than that: Plotly fits a wider box by widening the axis
  // ranges on a redraw instead of narrowing the map.
  panel.style.flexBasis = `${width}px`;
  const given = panel.clientWidth;
  const lower =
    given > 0 && given < width - 1
      ? Math.round(((given - margin.l - margin.r) * spanY) / spanX + margin.t + margin.b)
      : fullHeight;
  v.el.style.setProperty("--hm-panel-height", `${Math.min(Math.max(lower, 160), fullHeight)}px`);
}

// Width and height of a panel for its Plotly layout. Plotly takes the size
// of its box only on the first draw and in Plotly.Plots.resize, so a map
// drawn again after sizeImagePanel changed the box would keep the old size
// and widen its axis ranges; with the size given, every draw fits the box.
function panelSize(panel) {
  const width = panel.clientWidth;
  const height = panel.clientHeight;
  return width > 0 && height > 0 ? { width, height, autosize: false } : {};
}

// Single voxel or slab with an anatomy: the anatomy slice and, on it, the cut
// of the excited volume as a closed line. Nothing here can be clicked.
function drawOutlinePanel(v) {
  const panel = v.panels.image;
  const outline = v.view.outline;
  const shape = outline.kind === "slab" ? "slab" : "voxel";
  const margin = rememberImageFit(v, outline.x_range, outline.y_range, { ...IMAGE_MARGIN });
  sizeImagePanel(v);
  v.Plotly.react(
    panel,
    [
      ...anatomyTraces(v),
      {
        type: "scatter",
        mode: "lines",
        x: outline.x,
        y: outline.y,
        line: { color: "#ff3b30", width: 2 },
        hoverinfo: "skip",
        name: shape,
      },
    ],
    {
      title: {
        text:
          `${shape} on anatomy` +
          (outline.slice_dim ? ` ${outline.slice_dim} ${outline.slice_index}` : ""),
        font: { size: 13 },
      },
      showlegend: false,
      margin,
      ...panelSize(v.panels.image),
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColour(v), size: 11 },
      xaxis: { range: outline.x_range, constrain: "domain" },
      yaxis: {
        range: outline.y_range,
        scaleanchor: "x",
        scaleratio: 1,
        constrain: "domain",
        constraintoward: "bottom",
      },
      shapes: [],
      dragmode: "zoom",
      uirevision: `outline ${outline.x_range} ${outline.y_range}`,
    },
    // The ResizeObserver in viewer.js draws the map again on a new box.
    { displaylogo: false, responsive: false },
  );
}

function drawImagePanel(v) {
  const panel = v.panels.image;
  if (!panel) return;
  // A single voxel has no map; with an anatomy it shows the voxel outline.
  const outlined = Boolean(v.view.outline?.kind);
  panel.style.display = v.view.single_voxel && !outlined ? "none" : "";
  if (v.view.single_voxel) {
    if (outlined) drawOutlinePanel(v);
    return;
  }
  const view = v.view;
  const { row, column, curve_index: k } = v.selection;
  const display = v.model.get("_display");
  const values = planeValues(v, k);
  const [zmin, zmax] = mapColourLimits(v, values);
  const z = display.min_to_nan ? hideBelowMinimum(values, zmin) : values;
  const x = view.x[column];
  const y = view.y[row];
  const dx = view.x.length > 1 ? Math.abs(view.x[1] - view.x[0]) / 2 : 0.5;
  const dy = view.y.length > 1 ? Math.abs(view.y[1] - view.y[0]) / 2 : 0.5;
  const line = { color: "white", width: 1, dash: "dot" };
  const curvePosition = view.curve_x[k];
  const margin = rememberImageFit(v, view.x_range, view.y_range, {
    ...IMAGE_MARGIN,
    r: IMAGE_MARGIN.r + COLOUR_BAR_STEP_PX * colourBarTraces(v).length,
    // The margins hold the colour bars; Plotly must not widen them.
    autoexpand: false,
  });
  // The panel gets its size before Plotly lays out the map: drawn into the
  // box of the previous plane (before a rotation, say), Plotly would widen
  // the axis ranges instead of the margins on the next resize.
  sizeImagePanel(v);
  v.Plotly.react(
    panel,
    [
      ...anatomyTraces(v),
      {
        type: "heatmap",
        name: "csi",
        z,
        x: view.x,
        y: view.y,
        zmin,
        zmax,
        opacity: anatomyShown(v) ? v.model.get("_anatomy_opacity") : 1,
        colorscale: view.colorscale,
        zsmooth: display.smoothing === "off" ? false : display.smoothing,
        customdata: v.cachedHoverIndices,
        // The value of voxels hidden by "min -> NaN" too.
        text: values,
        // "none" (not "skip") keeps the click events that select a voxel.
        // A hovertemplate would show a label even with "none", so it is
        // only set while tooltips are on.
        hoverinfo: display.show_tooltips ? undefined : "none",
        hovertemplate: display.show_tooltips
          ? `${view.column_dim} %{customdata[1]}, ${view.row_dim} %{customdata[0]}` +
            "<br>x %{x:.4g}, y %{y:.4g}<br>value %{text:.4g}%{customdata[2]}<extra></extra>"
          : undefined,
        showscale: Boolean(display.show_colorbar),
        colorbar: colourBarPlace(v, 0),
      },
      ...overlayTraces(v),
      ...(display.show_masks === false ? [] : maskTraces(v)),
      ...(display.show_masks === false || display.show_mask_outlines === false
        ? []
        : maskOutlineTraces(v)),
      ...sliceLineTraces(v),
      ...scalebarTraces(v),
    ],
    {
      title: {
        text:
          view.curve_dim === null || view.map_fixed
            ? view.title || ""
            : `${view.title || ""} at ${formatPosition(curvePosition)} ${view.curve_label}`,
        font: { size: 13 },
      },
      margin,
      ...panelSize(v.panels.image),
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColour(v), size: 11 },
      xaxis: {
        range: view.x_range,
        constrain: "domain",
        showgrid: display.show_grid_lines,
        // Above the map; below it the grid would only show in empty voxels.
        layer: "above traces",
      },
      yaxis: {
        range: view.y_range,
        scaleanchor: "x",
        scaleratio: 1,
        constrain: "domain",
        // A map narrower than the panel sits on the bottom, level with the
        // x axis of the curve panel beside it.
        constraintoward: "bottom",
        showgrid: display.show_grid_lines,
        layer: "above traces",
      },
      shapes: [
        linkedPointerShape(v),
        ...(display.show_crosshair
          ? [
              { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line },
              { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: y, y1: y, line },
            ]
          : []),
        {
          type: "rect",
          x0: x - dx,
          x1: x + dx,
          y0: y - dy,
          y1: y + dy,
          line: { color: "white", width: 2 },
        },
        ...pendingShape(v),
      ],
      dragmode: v.tool === "freehand" ? "drawclosedpath" : "zoom",
      newshape: {
        line: { color: outlineColour(v), width: 2, dash: "dash" },
        fillcolor: "rgba(0,0,0,0)",
      },
      // A new plane or rotation starts from its own ranges: Plotly keeps
      // the fitted axes of the last plane as interface state otherwise.
      uirevision: `map ${view.x_range} ${view.y_range}`,
      // Plotly keeps drawn shapes as user state; a new revision drops a committed outline.
      editrevision: v.outlineRevision,
    },
    // The ResizeObserver in viewer.js draws the map again on a new box.
    { displaylogo: false, responsive: false },
  ).then(() => fitColourBars(v));
}

// The polygon corner is where the user clicked, not the voxel centre, so outlines can
// cut voxels partly. Falls back to the voxel centre if Plotly gives no mouse event.
function clickedDataPoint(v, mouse, row, column) {
  const layout = v.panels.image._fullLayout;
  if (!mouse || !layout?.xaxis?.p2d) return [v.view.x[column], v.view.y[row]];
  const box = v.panels.image.getBoundingClientRect();
  return [
    layout.xaxis.p2d(mouse.clientX - box.left - layout.xaxis._offset),
    layout.yaxis.p2d(mouse.clientY - box.top - layout.yaxis._offset),
  ];
}

function attachImagePanelEvents(v) {
  const panel = v.panels.image;
  if (!panel) return;
  attachPointerEvents(v, panel);
  panel.on("plotly_click", (event) => {
    const point = event.points.find((item) => item.data.name === "csi");
    if (!point) return;
    const [row, column] = point.pointIndex;
    if (v.tool === "polygon") {
      v.pending.push(clickedDataPoint(v, event.event, row, column));
      drawImagePanel(v);
    } else {
      choose(v, { row, column });
    }
    v.el.focus({ preventScroll: true });
  });
  panel.on("plotly_relayout", (event) => {
    if (v.tool !== "freehand" || !event.shapes) return;
    const drawn = event.shapes.filter((shape) => shape.type === "path").pop();
    v.outlineRevision += 1;
    drawImagePanel(v);
    if (drawn) commitOutline(v, parsePath(drawn.path));
  });
}
