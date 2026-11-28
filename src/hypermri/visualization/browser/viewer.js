// Viewer: builds the page from the layout Python sends in `_layout`, keeps
// the selection, and connects Python messages, mouse and keyboard to the
// panels and control groups of the other parts.
//
// Everything that only changes which part of the block is shown (voxel,
// curve position, hover, crosshair, colour scaling) happens here without
// asking Python. Processing settings go back to Python through the
// `_controls` trait, which answers with a new block.

// Control group name (as in layout.py) -> function that builds its inputs.
const CONTROL_GROUP_BUILDERS = {
  slices: sliceControls,
  spectrum: spectrumControls,
  colour_range: colourRangeControls,
  curve_scale: curveScaleControls,
  display: displayControls,
  kspace: kspaceControls,
  save: saveControls,
  layers: layerControls,
  anatomy: anatomyControls,
  overlay: overlayControls,
  tool: toolControls,
  masks: maskControls,
  fit: fitControls,
  curve_fit: curveFitControls,
  ideal: idealControls,
  signal_decay: signalDecayControls,
  time_course: timeCourseControls,
  parameters: parameterControls,
};

// Panel name (as in layout.py) -> CSS class of its plot area.
const PANEL_CLASSES = { image: "hm-csi-map", curve: "hm-csi-spectrum", time: "hm-csi-time" };

// Control groups that only change what Python computes. In an HTML snapshot,
// which has no Python behind it, their inputs are shown but switched off.
const PYTHON_GROUPS = [
  "slices",
  "spectrum",
  "kspace",
  "save",
  "anatomy",
  "overlay",
  "tool",
  "masks",
  "fit",
  "curve_fit",
  "ideal",
  "signal_decay",
  "time_course",
];

function buildGroups(v, names) {
  for (const name of names) {
    const group = v.groups[name];
    if (!group) continue;
    group.replaceChildren(...CONTROL_GROUP_BUILDERS[name](v));
    if (v.isSnapshot && PYTHON_GROUPS.includes(name)) switchOffGroup(group);
  }
  hideEmptySections(v);
  watchGridColumns(v);
}

// Sections with the class hm-csi-grid-items place their inputs in equal
// columns (viewer.css). An input wider than one column spans as many columns
// as it needs. The observer measures again whenever a section changes its
// size: a new width, or a hidden section that is shown.
function watchGridColumns(v) {
  v.gridResizer ??= new ResizeObserver((entries) => {
    for (const entry of entries) spanWideItems(entry.target);
  });
  v.gridResizer.disconnect();
  for (const grid of v.el.querySelectorAll(".hm-csi-grid-items")) v.gridResizer.observe(grid);
}

function spanWideItems(grid) {
  const style = getComputedStyle(grid);
  const columns = style.gridTemplateColumns.split(" ").map(parseFloat);
  const width = columns[0];
  const gap = parseFloat(style.columnGap) || 0;
  if (!(width > 0)) return;
  for (const item of gridItems(grid)) {
    const itemWidth = item.getBoundingClientRect().width;
    if (itemWidth <= 0) continue;
    const span = Math.min(columns.length, Math.ceil((itemWidth + gap) / (width + gap)));
    item.style.gridColumn = span > 1 ? `span ${span}` : "";
  }
}

// The children of a grid that take a cell; a control group (display:
// contents) passes its own children on to the grid.
function gridItems(node) {
  return [...node.children].flatMap((child) =>
    getComputedStyle(child).display === "contents" ? gridItems(child) : [child],
  );
}

const NEEDS_PYTHON = "Needs Python: open the notebook with a running kernel to change this.";

function switchOffGroup(group) {
  group.classList.add("hm-csi-needs-python");
  group.title = NEEDS_PYTHON;
  for (const input of group.querySelectorAll("input, select, button, textarea")) {
    input.disabled = true;
  }
}

// Switch off one input of a browser-side group whose change Python answers
// (a colour scale, the scalebar length, the fit curves) in an HTML snapshot.
function needsPython(v, input) {
  if (!v.isSnapshot) return input;
  input.disabled = true;
  input.title = NEEDS_PYTHON;
  input.classList.add("hm-csi-needs-python");
  return input;
}

// A plot options section whose groups offer nothing for the shown data (no
// anatomy, no overlay, no time course) is hidden, and so is the button when
// no section is left.
function hideEmptySections(v) {
  for (const section of v.sections) {
    section.node.hidden = section.names.every((name) => !v.groups[name]?.childElementCount);
  }
  if (v.optionsToggle) {
    v.optionsToggle.hidden = v.sections.every((section) => section.node.hidden);
  }
}

// Rebuild every control group except the masks, which keep their own state,
// the tools, which change only with `_fit_tool`, `_curve_fit_tool`, `_ideal_tool` and
// `_signal_decay_tool` (so typing in the table
// survives a new block), the parameters, which never change, and the group
// whose slider is being dragged (sendWhileDragged).
function buildControls(v) {
  const kept = ["masks", "fit", "curve_fit", "ideal", "signal_decay", "parameters"];
  if (v.draggedGroup) kept.push(v.draggedGroup);
  buildGroups(v, Object.keys(v.groups).filter((name) => !kept.includes(name)));
}

function buildMaskControls(v) {
  buildGroups(v, ["masks"]);
}

function updateStatus(v) {
  // Timings only when show_timings is set; the snapshot note always.
  const parts = [];
  const timings = Boolean(v.model.get("show_timings"));
  if (timings && v.lastDrawMs !== null) parts.push(`browser redraw ${v.lastDrawMs.toFixed(1)} ms`);
  if (v.isSnapshot) {
    parts.push(`HTML snapshot without Python (${v.view.shape.join(" x ")})`);
  } else if (timings) {
    parts.push(
      `last Python block ${v.view.compute_ms} ms (${v.view.block_megabytes} MB, ` +
        `${v.view.shape.join(" x ")})`,
    );
  }
  if (v.tool === "polygon") {
    parts.push("mask: click corners, Enter closes, Backspace removes, Esc cancels");
  } else if (v.tool === "freehand") {
    parts.push("mask: drag an outline on the map");
  } else {
    parts.push("click or arrow keys: voxel, , and . : curve position");
  }
  v.status.textContent = parts.join(" · ");
}

function draw(v, { image = true, curve = true, time = true } = {}) {
  const start = performance.now();
  if (image) drawImagePanel(v);
  if (curve) drawCurvePanel(v);
  if (time) drawTimePanel(v);
  sizeImagePanel(v);
  v.lastDrawMs = performance.now() - start;
  updateStatus(v);
}

function saveSelection(v) {
  clearTimeout(v.saveTimer);
  v.saveTimer = setTimeout(() => {
    v.model.set("selection", { ...v.selection });
    v.model.save_changes();
  }, 80);
}

// Select a voxel (row, column) and curve index k, clipped to the block.
function choose(v, { row = v.selection.row, column = v.selection.column, k = v.selection.curve_index }) {
  row = Math.min(Math.max(row, 0), nRows(v) - 1);
  column = Math.min(Math.max(column, 0), nColumns(v) - 1);
  k = Math.min(Math.max(k, 0), nCurve(v) - 1);
  const voxelChanged = row !== v.selection.row || column !== v.selection.column;
  const curveChanged = k !== v.selection.curve_index;
  if (!voxelChanged && !curveChanged) return;
  v.selection = { row, column, curve_index: k };
  draw(v);
  saveSelection(v);
}

// Set from Python, for example by a view linked with link_frames: show the
// new voxel and curve position from the block already here. The browser's
// own choice comes back here too and changes nothing.
function onSelection(v) {
  const s = v.model.get("selection");
  if (!s || !v.view.shape) return;
  const row = Math.min(Math.max(s.row, 0), nRows(v) - 1);
  const column = Math.min(Math.max(s.column, 0), nColumns(v) - 1);
  const k = Math.min(Math.max(s.curve_index, 0), nCurve(v) - 1);
  const same =
    row === v.selection.row && column === v.selection.column && k === v.selection.curve_index;
  if (same) return;
  v.selection = { row, column, curve_index: k };
  if (v.Plotly) draw(v);
}

// In a snapshot only the anatomy slice has values saved for every choice.
function snapshotAnswers(v, update) {
  const names = Object.keys(update);
  return names.length === 1 && names[0] === "anatomy" && Boolean(v.model.get("_anatomy_slices"));
}

function sendControls(v, update) {
  if (v.isSnapshot && !snapshotAnswers(v, update)) {
    v.status.textContent = "This snapshot has no Python behind it; open the notebook to change this.";
    return;
  }
  // The new block clears the clicked corners: store the polygon first, on
  // the plane it was clicked on (Python handles the messages in order).
  if (v.tool === "polygon" && v.pending.length > 2) finishPolygon(v);
  const controls = structuredClone(v.model.get("_controls"));
  Object.assign(controls, update);
  if (!v.isSnapshot) {
    v.requestSentAt = performance.now();
    v.status.textContent = "Python is preparing a new block ...";
  }
  v.model.set("_controls", controls);
  v.model.save_changes();
}

// Send the value of a slider in control group `group` to Python while it is
// dragged, so the image follows. update(value) returns the `_controls`
// update for a value, or null when Python already shows it. At most one
// request is on its way to Python: a newer value waits in v.waitingUpdate
// and is sent when the block of the previous one arrives (onNewBlock), so a
// fast drag skips intermediate values instead of queueing them. A request
// older than two seconds counts as lost. The dragged slider's group is not
// rebuilt by the blocks that arrive meanwhile (buildControls).
function sendWhileDragged(v, slider, group, update) {
  const send = () => {
    v.waitingUpdate = () => update(Number(slider.value));
    const busy = v.requestSentAt !== null && performance.now() - v.requestSentAt < 2000;
    if (!busy) sendWaitingUpdate(v);
  };
  slider.addEventListener("input", () => {
    v.draggedGroup = group;
    send();
  });
  slider.addEventListener("change", () => {
    v.draggedGroup = null;
    send();
  });
}

function sendWaitingUpdate(v) {
  const update = v.waitingUpdate?.();
  v.waitingUpdate = null;
  if (update) sendControls(v, update);
}

function onNewBlock(v) {
  readBlock(v);
  v.cachedHoverIndices = hoverIndices(v);
  v.pending = [];
  buildControls(v);
  buildMaskControls(v);
  if (v.Plotly) draw(v);
  if (v.requestSentAt !== null) {
    const roundTrip = performance.now() - v.requestSentAt;
    v.requestSentAt = null;
    v.status.textContent += ` · settings round trip ${roundTrip.toFixed(0)} ms`;
  }
  sendWaitingUpdate(v);
}

function onCustomMessage(v, message) {
  if (message?.type === "settings_saved") {
    v.status.textContent = message.ok
      ? "View settings saved on the scan; the next plot_scan starts from them."
      : message.message;
    return;
  }
  if (message?.type === "time_course_error") {
    v.timeMessage = message.message;
    buildGroups(v, ["time_course"]);
    return;
  }
  if (message?.type !== "mask_error") return;
  v.maskMessage = message.message;
  buildMaskControls(v);
}

function onKey(v, event) {
  if (event.target.closest("input, select, button")) return;
  if (v.tool === "polygon" && v.pending.length) {
    const edits = {
      Enter: () => finishRoi(v),
      Escape: () => (v.pending = []),
      Backspace: () => v.pending.pop(),
    };
    if (edits[event.key]) {
      event.preventDefault();
      edits[event.key]();
      if (v.Plotly) drawImagePanel(v);
      return;
    }
  }
  // Arrow keys and , . move in the direction seen on screen, whatever the
  // axis orientation of the map or a reversed ppm axis.
  const view = v.view;
  const up = screenStep(view.y, view.y_range);
  const right = screenStep(view.x, view.x_range);
  const later = screenStep(view.curve_x, view.curve_invert_x ? [1, 0] : [0, 1]);
  const moves = {
    ArrowUp: { row: v.selection.row + up },
    ArrowDown: { row: v.selection.row - up },
    ArrowLeft: { column: v.selection.column - right },
    ArrowRight: { column: v.selection.column + right },
    ",": { k: v.selection.curve_index - later },
    ".": { k: v.selection.curve_index + later },
  };
  const move = moves[event.key];
  if (!move) return;
  event.preventDefault();
  choose(v, move);
}

// Place the control groups `names` in `parent`. The groups of the tools
// follow the "tool" menu.
function placeGroups(v, parent, names) {
  for (const name of names) {
    const groupNames = name === "tool" ? [name, ...v.layout.tools] : [name];
    for (const groupName of groupNames) {
      v.groups[groupName] = element("span", `hm-csi-group hm-csi-group-${groupName}`);
      parent.append(v.groups[groupName]);
    }
  }
}

// The page has three parts: a header (top bar and the collapsible plot
// options), the panels side by side, and a footer (controls below the plots
// and the status line). In a grid of views (plot_side_by_side) the three
// parts of every view line up in rows, so the plots of all views start and
// end at the same height.
function buildPage(v, layout) {
  const header = element("div", "hm-csi-header");
  const top = element("div", "hm-csi-bar hm-csi-top");
  placeGroups(v, top, layout.top_bar);
  const toggle = button("Plot options", "Show or hide the display and processing settings");
  toggle.classList.add("hm-csi-options-toggle");
  toggle.setAttribute("aria-expanded", "false");
  top.append(toggle);
  // The sections are rows of a table with a coloured title, as in the fit
  // tool; the colour class comes from the section title.
  const options = element("table", "hm-csi-fit-settings hm-csi-options");
  options.hidden = true;
  toggle.addEventListener("click", () => {
    options.hidden = !options.hidden;
    toggle.setAttribute("aria-expanded", String(!options.hidden));
  });
  for (const [title, names] of layout.option_sections) {
    const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const node = element("tr", `hm-csi-section hm-csi-section-${slug}`);
    const cell = element("td");
    const body = element("div", "hm-csi-fit-settings-items hm-csi-grid-items");
    placeGroups(v, body, names);
    cell.append(body);
    node.append(element("th", "", title), cell);
    options.append(node);
    v.sections.push({ node, names });
  }
  v.optionsToggle = toggle;
  // The name sits last in the header, right above the plots it names.
  v.nameLine = element("div", "hm-csi-name");
  showName(v);
  header.append(top, options, v.nameLine);

  const plots = element("div", "hm-csi-plots");
  for (const name of layout.panels) {
    v.panels[name] = element("div", PANEL_CLASSES[name]);
    plots.append(v.panels[name]);
  }

  const footer = element("div", "hm-csi-footer");
  if (layout.below_plots.length) {
    const below = element("div", "hm-csi-bar hm-csi-below");
    placeGroups(v, below, layout.below_plots);
    footer.append(below);
  }
  footer.append(v.status);
  v.el.append(header, plots, footer);
}

// The plot height is a CSS variable that the panels and sizeImagePanel read;
// the ResizeObserver then fits the Plotly plots to their new boxes.
function showPlotHeight(v) {
  v.el.style.setProperty("--hm-plot-height", `${v.model.get("plot_height")}px`);
  if (v.panels.image?._fullLayout) sizeImagePanel(v);
}

function showName(v) {
  const name = v.model.get("name");
  v.nameLine.textContent = name;
  v.nameLine.hidden = !name;
}

function render({ model, el }) {
  el.classList.add("hm-csi");
  el.tabIndex = 0;
  // Everything the parts share: data from Python, what the user chose, and
  // the HTML elements of the page.
  const v = {
    model,
    el,
    status: element("div", "hm-csi-status", "Loading Plotly.js ..."),
    panels: {},
    groups: {},
    sections: [],
    optionsToggle: null,
    nameLine: null,
    imageFit: null,
    Plotly: null,
    showAnatomy: true,
    anatomyOpacitySlider: null,
    overlayOpacitySlider: null,
    percentReference: "map",
    mapRangeInputs: null,
    requestSentAt: null,
    waitingUpdate: null,
    draggedGroup: null,
    linkedPointer: null,
    lastDrawMs: null,
    saveTimer: null,
    tool: "off",
    erase: false,
    pending: [],
    outlineRevision: 0,
    maskMessage: "",
    fitAdvanced: false,
    fitShowResults: true,
    showTutorial: false,
    maskShowStatistics: true,
  };
  showPlotHeight(v);
  readBlock(v);
  v.activeMask = v.maskPlanes.length ? v.maskPlanes[v.maskPlanes.length - 1].name : "";
  v.cachedHoverIndices = hoverIndices(v);
  v.layout = model.get("_layout");
  v.isSnapshot = Boolean(v.layout.snapshot);
  buildPage(v, v.layout);

  const cleanups = [];
  const listeners = {
    "change:_view": onNewBlock,
    "change:_anatomy_opacity": onAnatomyOpacity,
    "change:_anatomy_range": onAnatomyRange,
    "change:_overlay_opacity": onOverlayOpacity,
    "change:_map_range": onMapRange,
    "change:_display": onDisplay,
    "change:_curve_axes": onCurveAxes,
    "change:_mask_planes": onMaskPlanes,
    "change:_mask_tool_shown": onMaskToolShown,
    "change:_fit_tool": onFitTool,
    "change:_curve_fit_tool": onCurveFitTool,
    "change:_ideal_tool": onIdealTool,
    "change:_signal_decay_tool": onSignalDecayTool,
    "change:_slice_lines": onSliceLines,
    "change:selection": onSelection,
    "change:_roi_mean": onRoiMean,
    "change:_fit_curves": onFitCurves,
    "change:_fit_table": onFitTable,
    "change:_mask_statistics": onMaskStatistics,
    "change:_other_metabolite": onOtherMetabolite,
    "change:tool": onTool,
    "change:name": showName,
    "change:plot_height": showPlotHeight,
    "change:show_timings": updateStatus,
    "change:_time_course": onTimeCourse,
    "change:_time_block": onTimeCourse,
    "msg:custom": onCustomMessage,
  };
  for (const [name, listener] of Object.entries(listeners)) {
    const handler = (...args) => listener(v, ...args);
    model.on(name, handler);
    cleanups.push(() => model.off(name, handler));
  }
  const pointerHandler = (event) => onLinkedPointer(v, event);
  POINTER_EVENTS.addEventListener("pointer", pointerHandler);
  cleanups.push(() => POINTER_EVENTS.removeEventListener("pointer", pointerHandler));
  const maskToolHandler = (event) => onLinkedMaskTool(v, event);
  MASK_EVENTS.addEventListener("masktool", maskToolHandler);
  cleanups.push(() => MASK_EVENTS.removeEventListener("masktool", maskToolHandler));
  const closeOutlineHandler = (event) => onLinkedCloseOutline(v, event);
  MASK_EVENTS.addEventListener("closeoutline", closeOutlineHandler);
  cleanups.push(() => MASK_EVENTS.removeEventListener("closeoutline", closeOutlineHandler));
  const keyHandler = (event) => onKey(v, event);
  el.addEventListener("keydown", keyHandler);
  cleanups.push(() => el.removeEventListener("keydown", keyHandler));
  buildControls(v);
  buildGroups(v, ["masks", "fit", "curve_fit", "ideal", "signal_decay", "parameters"]);
  showChosenTool(v);

  loadPlotly(model, (note) => {
    v.status.textContent = note;
  })
    .then((plotly) => {
      v.Plotly = plotly;
      // Every panel becomes a Plotly plot, even one hidden at the start
      // (image only, single voxel, no time course), so that its events can
      // be attached now and work once a new block shows it.
      for (const panel of Object.values(v.panels)) {
        plotly.newPlot(panel, [], {}, { displaylogo: false, responsive: true });
      }
      draw(v);
      drawSignalDecay(v);
      attachImagePanelEvents(v);
      attachCurvePanelEvents(v);
      attachTimePanelEvents(v);
      // Plotly sizes a plot once when it is drawn. A panel that had no width
      // yet (output not attached, hidden time panel) or that wraps to a new
      // row is fitted to its box again whenever that box changes.
      const resizer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (!entry.target._fullLayout || entry.contentRect.width <= 0) continue;
          // The map is drawn again instead of resized: Plotly.Plots.resize
          // keeps the axis ranges of the old box and widens them to the new
          // aspect ratio, so that a rotated map ends up with empty margins.
          if (entry.target === v.panels.image) drawImagePanel(v);
          else plotly.Plots.resize(entry.target);
        }
      });
      for (const panel of Object.values(v.panels)) resizer.observe(panel);
      cleanups.push(() => resizer.disconnect());
    })
    .catch((error) => {
      v.status.textContent = `Could not load Plotly.js: ${error}`;
    });

  return () => {
    clearTimeout(v.saveTimer);
    v.gridResizer?.disconnect();
    for (const cleanup of cleanups) cleanup();
    if (v.Plotly) {
      for (const panel of Object.values(v.panels)) v.Plotly.purge(panel);
      if (v.decayPlot) v.Plotly.purge(v.decayPlot);
    }
  };
}

export default { render };
