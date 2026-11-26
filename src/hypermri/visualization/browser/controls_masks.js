// Control group "masks": add, choose, style, draw and delete masks, and show
// the mean spectrum inside a mask.
//
// Python sends each mask's coverage of the shown plane in `_mask_planes`.
// Drawn outlines (in map axis units) go to Python as `{type: "mask"}`
// messages; Python rasterizes and stores them and answers with new
// `_mask_planes`.
//
// Views linked with ScanView.link_masks (same `_mask_link`) share the chosen
// mask, the drawing tool and the add/erase mode through MASK_EVENTS, so one
// tool draws on all of them; Python copies the drawn masks between the scans.
// A linked view with `_mask_tool_shown` false (the second and later images of
// plot_side_by_side) shows only the chosen mask, its volume and the ROI mean.

const MASK_EVENTS = (globalThis.hypermriMaskEvents ??= new EventTarget());

function shareMaskTool(v) {
  const link = v.model.get("_mask_link");
  if (!link) return;
  const detail = { link, source: v, activeMask: v.activeMask, tool: v.tool, erase: v.erase };
  MASK_EVENTS.dispatchEvent(new CustomEvent("masktool", { detail }));
}

function onLinkedMaskTool(v, event) {
  const { link, source, activeMask, tool, erase } = event.detail;
  if (source === v || !link || link !== v.model.get("_mask_link")) return;
  const toolChanged = tool !== v.tool;
  v.activeMask = activeMask;
  v.tool = tool;
  v.erase = erase;
  if (toolChanged) v.pending = [];
  // A mask added in the other view arrives later with `_mask_planes`;
  // rebuilding the controls now would replace the chosen name by an
  // existing one. The image is redrawn anyway, so this view draws with the
  // new tool (freehand needs Plotly's drawing drag mode).
  if (v.maskPlanes.some((mask) => mask.name === activeMask)) buildMaskControls(v);
  if (v.Plotly) draw(v, { curve: false });
}

function sendMask(v, action, content = {}) {
  v.maskMessage = "";
  v.model.send({ type: "mask", action, mask: v.activeMask, ...content });
}

function commitOutline(v, points) {
  if (points.length < 2) return;
  if (!v.activeMask) {
    v.maskMessage = "Add a mask first.";
    buildMaskControls(v);
    return;
  }
  sendMask(v, "draw", { vertices: points, erase: v.erase });
}

function finishPolygon(v) {
  const points = v.pending;
  v.pending = [];
  commitOutline(v, points);
  if (v.Plotly) drawImagePanel(v);
}

// "Close outline" (or Enter) commits the polygon clicked in this view or in a
// mask-linked one: the button sits in the first image only, but corners may
// have been clicked in another. Drawing then stops in all linked views.
function closeOutline(v) {
  finishPolygon(v);
  const link = v.model.get("_mask_link");
  if (link) {
    MASK_EVENTS.dispatchEvent(new CustomEvent("closeoutline", { detail: { link, source: v } }));
  }
  v.tool = "off";
  shareMaskTool(v);
  buildMaskControls(v);
  if (v.Plotly) draw(v, { curve: false });
}

function onLinkedCloseOutline(v, event) {
  const { link, source } = event.detail;
  if (source === v || link !== v.model.get("_mask_link") || !v.pending.length) return;
  finishPolygon(v);
}

function maskControls(v) {
  if (v.view.single_voxel) return [];
  if (v.view.mask_unavailable_reason) {
    return [element("span", "hm-csi-control-label", `Masks: ${v.view.mask_unavailable_reason}`)];
  }
  const names = v.maskPlanes.map((mask) => mask.name);
  if (!names.includes(v.activeMask)) v.activeMask = names.length ? names[names.length - 1] : "";
  const active = v.maskPlanes.find((mask) => mask.name === v.activeMask);

  const newName = element("input");
  newName.type = "text";
  newName.placeholder = "new mask name";
  newName.size = 12;
  const newColour = element("input");
  newColour.type = "color";
  newColour.value = "#ff0000";
  const add = button("Add", "Create an empty mask");
  add.addEventListener("click", () => {
    const name = newName.value.trim() || `mask${names.length + 1}`;
    v.activeMask = name;
    shareMaskTool(v);
    sendMask(v, "add", { mask: name, color: newColour.value, alpha: 0.4 });
  });

  const choice = select(names.length ? names : ["(none)"], v.activeMask || "(none)");
  choice.disabled = !names.length;
  choice.addEventListener("change", () => {
    v.activeMask = choice.value;
    shareMaskTool(v);
    if (v.roiMean.mask) v.model.send({ type: "roi_mean", mask: v.activeMask });
    if (v.timeCourse.mask) sendTimeCourse(v, { mask: v.activeMask });
    buildMaskControls(v);
    if (v.Plotly) drawImagePanel(v);
  });
  const colour = element("input");
  colour.type = "color";
  colour.value = active?.color || "#ff0000";
  colour.disabled = !active;
  colour.addEventListener("change", () => sendMask(v, "style", { color: colour.value }));
  const opacity = opacitySlider(active?.alpha ?? 0.4);
  opacity.disabled = !active;
  opacity.addEventListener("change", () => sendMask(v, "style", { alpha: Number(opacity.value) }));

  const toolChoice = select(["off", "polygon", "freehand"], v.tool);
  toolChoice.addEventListener("change", () => {
    v.tool = toolChoice.value;
    v.pending = [];
    shareMaskTool(v);
    buildMaskControls(v);
    if (v.Plotly) draw(v, { curve: false });
    v.el.focus({ preventScroll: true });
  });
  const mode = toggle(["add", "erase"], v.erase ? "erase" : "add");
  mode.addEventListener("change", () => {
    v.erase = mode.value === "erase";
    shareMaskTool(v);
    if (v.Plotly) drawImagePanel(v);
  });
  const close = button("Close outline", "Commit the clicked polygon (Enter)");
  close.disabled = v.tool !== "polygon";
  close.addEventListener("click", () => closeOutline(v));
  const undo = button("Undo", "Undo the last outline");
  undo.addEventListener("click", () => sendMask(v, "undo"));
  const remove = button("Delete", "Delete the selected mask");
  remove.disabled = !active;
  remove.addEventListener("click", () => {
    if (active && window.confirm(`Delete mask ${active.name}?`)) sendMask(v, "remove");
  });

  const roiBox = element("input");
  roiBox.type = "checkbox";
  roiBox.checked = Boolean(v.roiMean.mask);
  roiBox.disabled = !active;
  roiBox.title =
    "Show the mean magnitude inside the mask (supplied samples, no FFT) instead of the voxel spectrum";
  roiBox.addEventListener("change", () => {
    v.model.send({ type: "roi_mean", mask: roiBox.checked ? v.activeMask : null });
  });

  const info = element("span", "hm-csi-mask-info");
  if (active) {
    const volume =
      active.volume_ml === null || active.volume_ml === undefined
        ? ""
        : `, ${active.volume_ml.toPrecision(3)} mL`;
    info.textContent = `${active.weighted_voxel_count.toFixed(2)} voxels${volume}`;
  }
  const message = element("span", "hm-csi-mask-error", v.maskMessage);
  // An image without a curve panel has no curve to average.
  const roi = v.view.curve_dim === null ? [] : [labelled("ROI mean", roiBox)];
  const statistics = maskStatisticsItems(v);

  if (!v.model.get("_mask_tool_shown")) {
    const name = element("span", "hm-csi-mask-info", active ? active.name : "(none)");
    if (active) name.style.color = active.color;
    return [labelled("mask", name), info, ...roi, message, ...statistics];
  }

  const group = labelled("mask", choice);
  group.append(colour);
  return [
    labelled("new", newName),
    newColour,
    add,
    group,
    labelled("opacity", opacity),
    labelled("draw", toolChoice),
    labelled("mode", mode),
    close,
    undo,
    remove,
    ...roi,
    info,
    message,
    ...statistics,
  ];
}

// Mean ± standard deviation of the fitted values inside each mask
// (`_mask_statistics`, from Python's mask_fit_table). Offered once a fit and
// a mask exist; the table follows the selected position of further
// dimensions without rebuilding the tool.
function maskStatisticsItems(v) {
  v.maskStatisticsOffered = maskStatisticsOffered(v);
  if (!v.maskStatisticsOffered) return [];
  const toggle = element("input");
  toggle.type = "checkbox";
  toggle.checked = v.maskShowStatistics;
  toggle.title = "Show mean ± standard deviation of the fitted values inside each mask";
  toggle.addEventListener("change", () => {
    v.maskShowStatistics = toggle.checked;
    buildMaskControls(v);
  });
  v.maskStatisticsHolder = element("div", "hm-csi-fit-results-holder");
  v.maskStatisticsHolder.append(maskStatisticsTable(v));
  return [labelled("fit statistics", toggle), ...(v.maskShowStatistics ? [v.maskStatisticsHolder] : [])];
}

function maskStatisticsTable(v) {
  const result = v.model.get("_mask_statistics") || {};
  const rows = result.rows || [];
  if (!rows.length) return element("span", "hm-csi-mask-error", result.message || "");
  const has = (key) => rows.some((row) => row[key] && Number.isFinite(row[key][0]));
  const showPpm = has("freq_ppm");
  const showGauss = has("gauss_fwhm_hz");
  const spread = (pair, format) =>
    pair && Number.isFinite(pair[0]) ? `${format(pair[0])} ± ${format(pair[1])}` : "–";
  const fixed = (digits) => (value) => resultNumber(value, digits);
  const columns = [
    ["amplitude", "amplitude", resultAmplitude],
    ["freq_hz", "Hz", fixed(1)],
    ...(showPpm ? [["freq_ppm", "ppm", fixed(3)]] : []),
    ["t2_ms", "T2* ms", fixed(2)],
    ...(showGauss ? [["gauss_fwhm_hz", "Gauss FWHM Hz", fixed(1)]] : []),
  ];
  const colours = Object.fromEntries(v.maskPlanes.map((mask) => [mask.name, mask.color]));
  const table = element("table", "hm-csi-fit-table hm-csi-fit-results hm-csi-mask-statistics");
  const head = element("tr");
  head.append(element("th"));
  for (const heading of ["mask", "line", "voxels", ...columns.map((column) => column[1])]) {
    head.append(element("th", "", heading));
  }
  table.append(head);
  for (const row of rows) {
    const line = element("tr");
    const swatch = element("td", "hm-csi-fit-swatch");
    swatch.style.background = colours[row.mask] || "transparent";
    line.append(
      swatch,
      element("td", "", row.mask),
      element("td", "", row.line),
      element("td", "", String(row.n_voxels)),
      ...columns.map(([key, , format]) => element("td", "", spread(row[key], format))),
    );
    table.append(line);
  }
  const box = element("div", "hm-csi-fit-results-box");
  const note = element(
    "div",
    "hm-csi-fit-results-legend",
    "mean ± standard deviation over the voxels of each mask, weighted by coverage; failed fits left out",
  );
  box.append(table, note);
  return box;
}

function maskStatisticsOffered(v) {
  const result = v.model.get("_mask_statistics") || {};
  return Boolean((result.rows || []).length || result.message);
}

function onMaskStatistics(v) {
  // The toggle appears or disappears with the statistics: rebuild then.
  if (maskStatisticsOffered(v) !== Boolean(v.maskStatisticsOffered)) {
    buildMaskControls(v);
    return;
  }
  v.maskStatisticsHolder?.replaceChildren(maskStatisticsTable(v));
}

function onMaskPlanes(v) {
  v.maskPlanes = v.model.get("_mask_planes") || [];
  v.cachedHoverIndices = hoverIndices(v);
  buildMaskControls(v);
  buildGroups(v, ["time_course"]);
  if (v.Plotly) drawImagePanel(v);
}

function onMaskToolShown(v) {
  buildGroups(v, ["tool", "masks"]);
}

function onRoiMean(v) {
  v.roiMean = v.model.get("_roi_mean") || {};
  buildMaskControls(v);
  if (v.Plotly) drawCurvePanel(v);
}
