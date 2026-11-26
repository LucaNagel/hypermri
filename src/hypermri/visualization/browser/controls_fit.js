// Control group "fit": find spectral lines or add them at the spectral
// cursor, edit them in a table, fit all voxels and, after a fit, align the
// voxel spectra to the weighted mean fitted frequency (or undo that).
// "Advanced" shows the remaining detection and fit settings and the coupling
// columns of the table (T2* leader and ratio, phase band); whether it is open
// is kept in the browser only. "Lines from fit" fills the table from an
// earlier fit (stored on the scan or run here); the "start from earlier fit"
// columns take each line's frequency and T2* per voxel from it. After a fit,
// "Review" refits the selected voxel with the table, jumps to the next
// problem voxel (a line at a bound first, then the worst fit) and searches
// the selected voxel's residual for one more line in a range typed in Hz
// (kept in the browser only). "result table" shows the fitted values of the
// selected voxel, coloured by their bound status. Settings
// and table are tables with coloured groups; each line has a colour, also
// used for its frequency search range drawn in the spectrum.
//
// Python owns the table and the settings (`_fit_tool`, see
// hypermri.visualization.fit_tool). Every edit goes to Python as a
// `{type: "fit"}` message; Python answers with a new `_fit_tool`, and after
// a fit with the result as overlay "fit" and the fitted spectrum.

function sendFit(v, action, content = {}) {
  v.model.send({ type: "fit", action, ...content });
}

// A number input that sends its value when it is changed.
function fitNumber(value, width, title, onChange) {
  const input = element("input");
  input.type = "number";
  input.step = "any";
  input.value = Number.isFinite(value) ? String(Number(value.toPrecision(6))) : "";
  input.style.width = `${width}em`;
  if (title) input.title = title;
  input.addEventListener("change", () => onChange(Number(input.value)));
  return input;
}

function fitCheckbox(checked, title, onChange) {
  const box = element("input");
  box.type = "checkbox";
  box.checked = Boolean(checked);
  box.title = title;
  box.addEventListener("change", () => onChange(box.checked));
  return box;
}

function fitOption(v, settings, name, label, title, width = 4) {
  const input = fitNumber(settings[name], width, title, (value) =>
    sendFit(v, "option", { name, value }),
  );
  return labelled(label, input);
}

function fitOptionCheckbox(v, settings, name, label, title) {
  const box = fitCheckbox(settings[name], title, (value) =>
    sendFit(v, "option", { name, value }),
  );
  return labelled(label, box);
}

// Advanced settings: [setting, label, title, width].
const FIT_DETECTION_OPTIONS = [
  ["prominence_snr", "prominence SNR", "Smallest peak prominence in noise units"],
  ["max_peaks_per_spectrum", "peaks/voxel", "Most peaks taken from one voxel spectrum", 3],
  ["min_peak_distance_hz", "peak dist Hz", "Smallest distance of two peaks in one spectrum"],
  ["b0_shift_tolerance_hz", "B0 tol Hz", "Largest shift in Hz of a peak from its line centre between voxels"],
  ["cluster_min_separation_hz", "line sep Hz", "Drop weaker lines closer than this to a stronger one; 0 keeps all"],
  ["snr_threshold_factor", "fit SNR factor", "Fit amplitude threshold in multiples of the FID tail noise; 0 is off"],
];
const FIT_FIT_OPTIONS = [
  ["min_peak_separation_hz", "fit sep Hz", "Smallest distance in Hz of fitted line frequencies"],
  ["rep_fitting", "repeats", "Repetitions of the start-value search; 0 chooses automatically", 3],
  ["npoints", "grid points", "Grid points per parameter of the start-value search; 0 chooses automatically", 3],
  ["nsamplepoints", "FID points", "Leading FID samples fitted; 0 fits all", 4],
  ["phase_band_tau_ms", "phase tau ms", "Common delay of the phase band in ms; 0 derives it from the scan timing"],
];

// Colours of the table lines; a line's search range in the spectrum has the
// same colour. Blue (data) and red (fit, cursor) are left out.
const FIT_LINE_COLOURS = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];

// Name field and "Save fit" button of a fit tool. `type` is the message
// type of the tool ("fit", "curve_fit" or "ideal"); Python copies the last
// fit into saved_fits of the plotted scan or DataArray. The typed name is
// kept per tool while the controls are rebuilt.
function saveFitItems(v, type, fitted) {
  v.savedFitNames = v.savedFitNames || {};
  const name = element("input");
  name.type = "text";
  name.value = v.savedFitNames[type] || "";
  name.placeholder = "automatic";
  name.title = "Name of the saved fit: a letter, then letters, digits, _ or -; empty chooses one";
  name.style.width = "8em";
  name.addEventListener("input", () => {
    v.savedFitNames[type] = name.value;
  });
  const save = button(
    "Save fit",
    "Store a copy of the last fit on the plotted scan (scan.saved_fits) or DataArray; " +
      "it stays an overlay under its name and is shown again when the object is plotted",
  );
  save.disabled = !fitted;
  save.addEventListener("click", () => {
    v.model.send({ type, action: "save", name: name.value });
    v.savedFitNames[type] = "";
  });
  return [labelled("save as", name), save];
}

function fitLineColour(row) {
  return FIT_LINE_COLOURS[row % FIT_LINE_COLOURS.length];
}

// Half width in Hz of the residual search range proposed after a fit.
const RESIDUAL_HALF_WIDTH_HZ = 10;

// Rows of short number inputs only; their inputs sit in equal columns
// (hm-csi-grid-items in viewer.css). Rows with buttons and notes flow freely.
const FIT_GRID_KINDS = ["detect", "options"];

// The settings of the tool as a table: one row per titled group, its title
// in a coloured first cell (`kind` chooses the colour, see viewer.css).
function settingsTable(rows) {
  const table = element("table", "hm-csi-fit-settings");
  for (const [title, kind, items] of rows) {
    const line = element("tr", `hm-csi-fit-kind-${kind}`);
    line.append(element("th", "", title));
    const cell = element("td");
    const content = element("div", "hm-csi-fit-settings-items");
    if (FIT_GRID_KINDS.includes(kind)) content.classList.add("hm-csi-grid-items");
    content.append(...items);
    cell.append(content);
    line.append(cell);
    table.append(line);
  }
  return table;
}

function fitAdvancedRows(v, settings) {
  const options = (list) =>
    list.map(([name, label, help, width]) => fitOption(v, settings, name, label, help, width));
  const fitItems = options(FIT_FIT_OPTIONS);
  fitItems.push(
    fitOptionCheckbox(
      v,
      settings,
      "t2_coupling_clip_start",
      "clip coupled T2*",
      "Move T2* starts of coupled lines into their ratio band instead of failing",
    ),
  );
  return [
    ["Detection", "detect", options(FIT_DETECTION_OPTIONS)],
    ["Fit options", "options", fitItems],
  ];
}

// Columns of the peak table: [heading, column, kind, title, group, width].
// Columns of one group share a tinted background and a heading above them;
// `width` (em) of a number input defaults to NUMBER_WIDTH.
// Width in em of a number input in the peak table.
const NUMBER_WIDTH = 5;

const FIT_COLUMNS = [
  ["line", "name", "text", "Name of the line (metabolite)", "line"],
  ["Hz", "freq_hz", "number", "Start frequency in Hz", "freq", 7],
  ["range Hz", "freq_range_hz", "number", "Full frequency search width in Hz; shown as band in the spectrum", "freq", 4.5],
  ["fix", "fixed_freq", "check", "Keep the frequency at its start value", "freq"],
  ["ms", "initial_t2_ms", "number", "Start T2* in ms", "t2", 6],
  ["min", "min_t2_ms", "number", "Smallest T2* in ms", "t2", 4],
  ["max", "max_t2_ms", "number", "Largest T2* in ms", "t2", 4],
  ["fix", "fixed_t2", "check", "Keep T2* at its start value", "t2"],
  ["freq", "freq_from_fit", "check", "Start frequency per voxel from the earlier fit", "seed"],
  ["T2*", "t2_from_fit", "check", "Start T2* per voxel from the earlier fit", "seed"],
  ["win ms", "t2_window_ms", "number", "T2* bounds: start ± half this width per voxel (with T2* from fit); 0 keeps min/max", "seed"],
  ["fit", "fit", "check", "Fit this line; off leaves it out", "line"],
];

// Coupling columns, shown with the advanced settings.
const FIT_COUPLING_COLUMNS = [
  ["T2* leader", "t2_leader", "leader", "Line whose T2* this line follows", "coupling"],
  ["ratio min", "t2_ratio_low", "number", "Smallest ratio T2*(line) / T2*(leader)", "coupling"],
  ["max", "t2_ratio_high", "number", "Largest ratio; equal to min fixes the ratio", "coupling"],
  ["phase", "phase_band", "check", "Join the common phase law of the phase band", "coupling"],
  ["tol °", "phase_tol_deg", "number", "Allowed phase deviation in degrees", "coupling"],
];

const FIT_COLUMN_GROUPS = {
  line: "",
  freq: "frequency",
  t2: "T2*",
  seed: "start from earlier fit",
  coupling: "coupling",
};

function fitTable(v, peaks, advanced) {
  const columns = advanced ? [...FIT_COLUMNS, ...FIT_COUPLING_COLUMNS] : FIT_COLUMNS;
  const table = element("table", "hm-csi-fit-table");
  // Group headings over neighbouring columns of one group.
  const groups = element("tr", "hm-csi-fit-groups");
  groups.append(element("th"));
  columns.forEach(([, , , , group], index) => {
    if (index > 0 && columns[index - 1][4] === group) {
      groups.lastChild.colSpan += 1;
      return;
    }
    const cell = element("th", `hm-csi-fit-col-${group}`, FIT_COLUMN_GROUPS[group]);
    cell.colSpan = 1;
    groups.append(cell);
  });
  groups.append(element("th"));
  const head = element("tr");
  head.append(element("th"));
  for (const [heading, , , , group] of columns) {
    head.append(element("th", `hm-csi-fit-col-${group}`, heading));
  }
  head.append(element("th"));
  table.append(groups, head);
  peaks.forEach((peak, row) => {
    const line = element("tr");
    if (peak.fit === false) line.classList.add("hm-csi-fit-off");
    const swatch = element("td", "hm-csi-fit-swatch");
    swatch.style.background = fitLineColour(row);
    swatch.title = "Colour of this line's search range in the spectrum";
    line.append(swatch);
    for (const [, column, kind, title, group, width = NUMBER_WIDTH] of columns) {
      const edit = (value) => sendFit(v, "edit", { row, column, value });
      let input;
      if (kind === "check") {
        input = fitCheckbox(peak[column], title, edit);
      } else if (kind === "leader") {
        const others = peaks.filter((_, other) => other !== row).map((p) => [p.name, p.name]);
        input = select([["none", ""], ...others], peak[column]);
        input.title = title;
        input.addEventListener("change", () => edit(input.value));
      } else if (kind === "number") {
        input = fitNumber(peak[column], width, title, edit);
      } else {
        input = element("input");
        input.type = "text";
        input.size = 8;
        input.value = peak[column];
        input.title = title;
        input.addEventListener("change", () => edit(input.value));
      }
      // A window only acts on a line whose T2* starts from the earlier fit.
      if (column === "t2_window_ms") input.disabled = !peak.t2_from_fit || peak.fixed_t2;
      const cell = element("td", `hm-csi-fit-col-${group}`);
      cell.append(input);
      line.append(cell);
    }
    const remove = button("×", `Remove ${peak.name}`);
    remove.addEventListener("click", () => sendFit(v, "remove", { row }));
    const cell = element("td");
    cell.append(remove);
    line.append(cell);
    table.append(line);
  });
  // A wide table (advanced columns) scrolls instead of being cut off.
  const scroller = element("div", "hm-csi-fit-table-scroll");
  scroller.append(table);
  return scroller;
}

// Result table of the selected voxel (`_fit_table`, from Python's
// voxel_fit_table). A cell of a value with bounds is coloured by its bound
// status: within the bounds green, at the lower bound orange, at the upper
// bound red; fixed values are grey.
const FIT_STATUS_TEXT = {
  estimated: "within the bounds",
  at_lower_bound: "at the lower bound",
  at_upper_bound: "at the upper bound",
  fixed: "fixed",
  not_fitted: "not fitted",
  neighbour_median: "fixed to the median of the neighbours",
};

function resultNumber(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : "–";
}

function resultCell(text, status = "") {
  const cell = element("td", status ? `hm-csi-fit-status-${status}` : "", text);
  if (status) cell.title = FIT_STATUS_TEXT[status] || status;
  return cell;
}

// Amplitudes in a.u. with three significant digits, as 1.23e+4 when large.
function resultAmplitude(value) {
  if (!Number.isFinite(value)) return "–";
  return value !== 0 && (Math.abs(value) >= 1e4 || Math.abs(value) < 1e-2)
    ? value.toExponential(2)
    : value.toPrecision(3);
}

function fitResultTable(v) {
  const result = v.model.get("_fit_table") || {};
  const rows = result.rows || [];
  if (!rows.length) {
    return element("span", "hm-csi-mask-info", "Run a fit to see the values of the selected voxel.");
  }
  const showPpm = rows.some((row) => Number.isFinite(row.freq_ppm));
  const table = element("table", "hm-csi-fit-table hm-csi-fit-results");
  const head = element("tr");
  head.append(element("th"));
  const headings = ["line", "amplitude", "phase °", "Hz", ...(showPpm ? ["ppm"] : []), "T2* ms"];
  for (const heading of headings) head.append(element("th", "", heading));
  table.append(head);
  rows.forEach((row, index) => {
    const line = element("tr");
    const swatch = element("td", "hm-csi-fit-swatch");
    swatch.style.background = fitLineColour(index);
    line.append(
      swatch,
      element("td", "", row.line),
      resultCell(resultAmplitude(row.amplitude)),
      resultCell(resultNumber(row.phase_deg, 1)),
      resultCell(resultNumber(row.freq_hz, 1), row.freq_status),
      ...(showPpm ? [resultCell(resultNumber(row.freq_ppm, 3), row.freq_status)] : []),
      resultCell(resultNumber(row.t2_ms, 2), row.t2_status),
    );
    table.append(line);
  });
  const box = element("div", "hm-csi-fit-results-box");
  box.append(table);
  if (Number.isFinite(result.gauss_fwhm_hz)) {
    const gauss = element("div", "hm-csi-fit-results-gauss");
    const value = element("span", `hm-csi-fit-status-${result.gauss_status}`, `${resultNumber(result.gauss_fwhm_hz, 1)} Hz`);
    value.title = FIT_STATUS_TEXT[result.gauss_status] || result.gauss_status || "";
    gauss.append("Gauss FWHM (all lines): ", value);
    box.append(gauss);
  }
  const legend = element("div", "hm-csi-fit-results-legend");
  for (const status of ["estimated", "at_lower_bound", "at_upper_bound", "fixed"]) {
    legend.append(element("span", `hm-csi-fit-status-${status}`, FIT_STATUS_TEXT[status]));
  }
  box.append(legend);
  return box;
}

// The result table follows the selected voxel without rebuilding the tool.
function onFitTable(v) {
  if (!v.fitResultHolder) return;
  v.fitResultHolder.replaceChildren(fitResultTable(v));
}

function fitControls(v) {
  const settings = v.model.get("_fit_tool") || {};
  if (!settings.offered) return [];
  const peaks = settings.peaks || [];

  const detect = button("Detect lines", "Find lines automatically; replaces the table");
  detect.addEventListener("click", () => sendFit(v, "detect"));
  const add = button("Add at cursor", "Add a line at the spectral cursor (click the spectrum)");
  add.addEventListener("click", () => sendFit(v, "add_at_cursor"));
  const advanced = button(
    v.fitAdvanced ? "Advanced ▾" : "Advanced ▸",
    "Show or hide the remaining detection and fit settings and the coupling columns",
  );
  advanced.addEventListener("click", () => {
    v.fitAdvanced = !v.fitAdvanced;
    buildGroups(v, ["fit"]);
  });

  const lineshape = select(
    [
      ["Lorentz", "lorentz"],
      ["Voigt", "voigt"],
    ],
    settings.lineshape,
  );
  lineshape.title = "Lineshape of the fit";
  lineshape.addEventListener("change", () =>
    sendFit(v, "option", { name: "lineshape", value: lineshape.value }),
  );
  const gauss = fitOption(
    v,
    settings,
    "gauss_max_fwhm_hz",
    "Gauss max Hz",
    "Largest Gaussian width (FWHM) in Hz of a Voigt fit",
  );
  gauss.querySelector("input").disabled = settings.lineshape !== "voigt";

  const message = element("span", "hm-csi-mask-info", settings.message || "");
  const messageRow = element("div", "hm-csi-tool-row");
  messageRow.append(message);
  const run = button("Run fit", "Fit all voxels with the lines of the table");
  run.classList.add("hm-csi-primary");
  run.disabled = !peaks.length;
  run.addEventListener("click", () => {
    run.disabled = true;
    message.textContent = "Fitting all voxels ...";
    sendFit(v, "run");
  });

  const align = button(
    "Align spectra",
    "Shift every voxel spectrum to the weighted mean fitted frequency; fit again afterwards",
  );
  align.disabled = !settings.fitted;
  align.addEventListener("click", () => {
    align.disabled = true;
    message.textContent = "Aligning spectra ...";
    sendFit(v, "align");
  });
  const undoAlign = button("Undo align", "Show the spectra before alignment and their fit again");
  undoAlign.disabled = !settings.aligned;
  undoAlign.addEventListener("click", () => sendFit(v, "undo_align"));

  const refit = button(
    "Refit voxel",
    "Fit the selected voxel again with the start values and bounds of the table",
  );
  refit.disabled = !settings.fitted;
  refit.addEventListener("click", () => {
    refit.disabled = true;
    message.textContent = "Refitting the selected voxel ...";
    sendFit(v, "refit_voxel");
  });
  const nextProblem = button(
    "Next problem voxel",
    "Select the next voxel with a line at a bound or a poor fit (highest chi2/N first)",
  );
  nextProblem.disabled = !settings.fitted;
  nextProblem.addEventListener("click", () => sendFit(v, "next_problem"));
  // Once a fit exists, an empty search range starts at ±10 Hz around the
  // start frequency of the first line; typed values are kept.
  v.fitResidualRange = v.fitResidualRange || [NaN, NaN];
  const firstFreq = Number(peaks[0]?.freq_hz);
  if (settings.fitted && Number.isFinite(firstFreq) && !v.fitResidualRange.some(Number.isFinite)) {
    v.fitResidualRange = [firstFreq - RESIDUAL_HALF_WIDTH_HZ, firstFreq + RESIDUAL_HALF_WIDTH_HZ];
  }
  const residualInputs = [0, 1].map((side) =>
    fitNumber(
      v.fitResidualRange[side],
      5,
      side ? "Upper end in Hz of the search range" : "Lower end in Hz of the search range",
      (value) => {
        v.fitResidualRange[side] = value;
      },
    ),
  );
  const residual = button(
    "Line from residual",
    "Search the selected voxel's fit residual in the range for one more line; " +
      "it is added only when the data need it",
  );
  residual.disabled = !settings.fitted;
  residual.addEventListener("click", () => {
    const [low, high] = v.fitResidualRange;
    if (!Number.isFinite(low) || !Number.isFinite(high)) {
      message.textContent = "Type the search range in Hz first.";
      return;
    }
    message.textContent = "Searching the residual ...";
    sendFit(v, "residual_line", { low_hz: low, high_hz: high });
  });

  const showResults = fitCheckbox(
    v.fitShowResults,
    "Show amplitude, phase, frequency and T2* of the selected voxel below the tool",
    (checked) => {
      v.fitShowResults = checked;
      buildGroups(v, ["fit"]);
    },
  );
  v.fitResultHolder = element("div", "hm-csi-fit-results-holder");
  v.fitResultHolder.append(fitResultTable(v));

  const earlier = settings.earlier_fit || "";
  const fromFit = button(
    "Use lines of earlier fit",
    earlier
      ? `Replace the table by the lines of the earlier fit (${earlier}); every voxel starts ` +
          "from its own fitted frequency and T2*"
      : "Needs an earlier fit: a fit stored on the scan or a fit run here",
  );
  fromFit.disabled = !earlier;
  fromFit.addEventListener("click", () => sendFit(v, "lines_from_fit"));
  const earlierInfo = element(
    "span",
    "hm-csi-mask-info",
    earlier ? `earlier fit: ${earlier}` : "no earlier fit yet",
  );

  return [
    settingsTable([
      [
        "New lines",
        "new",
        [
          fitOption(v, settings, "peak_height_snr", "height SNR", "Smallest peak height in noise units"),
          fitOption(v, settings, "max_peaks", "max lines", "Keep the strongest lines; 0 keeps all", 3),
          fitOption(v, settings, "freq_search_range_hz", "range Hz", "Smallest frequency search width in Hz"),
          fitOption(v, settings, "min_t2_ms", "T2* min ms", "Smallest T2* in ms of new lines"),
          fitOption(v, settings, "max_t2_ms", "max ms", "Largest T2* in ms of new lines"),
          detect,
          add,
          fromFit,
          earlierInfo,
          advanced,
        ],
      ],
      ...(v.fitAdvanced ? fitAdvancedRows(v, settings) : []),
    ]),
    ...(peaks.length ? [fitTable(v, peaks, v.fitAdvanced)] : []),
    settingsTable([
      ["Fit", "run", [labelled("lineshape", lineshape), gauss, fitOption(v, settings, "n_jobs", "workers", "Worker processes; -1 uses all CPU cores", 3), run, align, undoAlign]],
      [
        "Review",
        "run",
        [
          refit,
          nextProblem,
          labelled("residual Hz", residualInputs[0]),
          labelled("to", residualInputs[1]),
          residual,
          labelled("result table", showResults),
        ],
      ],
      ["Save", "run", saveFitItems(v, "fit", settings.fitted)],
    ]),
    messageRow,
    ...(v.fitShowResults ? [v.fitResultHolder] : []),
  ];
}

function onFitTool(v) {
  buildGroups(v, ["tool", "fit"]);
  showChosenTool(v);
}
