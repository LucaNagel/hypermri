// Control groups for the processing settings that Python recomputes
// ("slices", "spectrum") and the "save" button. Each function returns the HTML elements of its group.

// One slider per dimension that is neither image nor curve; Python answers
// a released slider with a new block.
function sliceControls(v) {
  const controls = v.model.get("_controls");
  const items = [];
  if ((v.view.variable_options ?? []).length > 1) {
    // A fit result: choose the variable shown as map.
    const variable = select(v.view.variable_options, controls.variable);
    variable.addEventListener("change", () => sendControls(v, { variable: variable.value }));
    items.push(labelled("variable", variable));
  }
  if (controls.split_metabolites !== undefined) {
    // Interleaved metabolites: split the repetitions into metabolite_idx.
    const split = element("input");
    split.type = "checkbox";
    split.checked = controls.split_metabolites;
    split.title = "Split alternating repetitions into two metabolites";
    split.addEventListener("change", () => sendControls(v, { split_metabolites: split.checked }));
    items.push(labelled("split metabolites", split));
    if (controls.split_metabolites) {
      const other = element("input");
      other.type = "checkbox";
      other.checked = controls.other_metabolite;
      other.title = "Also draw the curve of the other metabolite at the selected voxel";
      other.addEventListener("change", () => sendControls(v, { other_metabolite: other.checked }));
      items.push(labelled("other metabolite", other));
    }
  }
  if (v.view.view_plane_options.length > 1) {
    // Three-dimensional images: the dimension perpendicular to the image.
    const plane = select(v.view.view_plane_options, controls.view_plane);
    plane.title = "Show the image perpendicular to this dimension";
    plane.addEventListener("change", () => sendControls(v, { view_plane: plane.value }));
    items.push(labelled("view plane", plane));
  }
  if (!v.view.single_voxel) {
    const rotate = button(`rotate ${controls.rotation}°`, "Turn the image by a further 90°, as the rotate button of plot");
    rotate.addEventListener("click", () =>
      sendControls(v, { rotation: (controls.rotation + 90) % 360 })
    );
    items.push(rotate);
  }
  for (const [dim, size] of Object.entries(v.view.other_dims)) {
    if (size < 2) continue;
    // Text coordinates such as metabolite names are shown next to the index.
    const names = v.view.other_values?.[dim];
    const shown = (index) => (names ? `${index} ${names[index]}` : `${index}`);
    const slider = element("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = size - 1;
    slider.value = controls.indices?.[dim] ?? 0;
    const value = element("span", "hm-csi-value", shown(slider.value));
    // While the slider is dragged, every new index is sent (see
    // sendSliceIndex), so the image and the slice bands of linked views follow.
    slider.addEventListener("input", () => {
      value.textContent = shown(slider.value);
      v.draggingSlice = true;
      sendSliceIndex(v, dim, Number(slider.value));
    });
    slider.addEventListener("change", () => {
      v.draggingSlice = false;
      sendSliceIndex(v, dim, Number(slider.value));
    });
    const group = labelled(dim, slider);
    group.append(value);
    items.push(group);
  }
  return items;
}

// At most one slice request is on its way to Python. A newer index waits in
// v.sliceRequest and is sent when the block of the previous one arrives
// (onNewBlock), so a fast drag skips intermediate slices instead of queueing
// them. A request older than two seconds counts as lost.
function sendSliceIndex(v, dim, index) {
  v.sliceRequest = { dim, index };
  const busy = v.requestSentAt !== null && performance.now() - v.requestSentAt < 2000;
  if (!busy) sendWaitingSliceIndex(v);
}

function sendWaitingSliceIndex(v) {
  const request = v.sliceRequest;
  v.sliceRequest = null;
  if (!request) return;
  const indices = v.model.get("_controls").indices ?? {};
  if (indices[request.dim] === request.index) return;
  sendControls(v, { indices: { ...indices, [request.dim]: request.index } });
}

// Images get only the complex part; unit and line broadening act on spectra.
// The processing controls (FID, points, denoising, repetitions, channels)
// follow; Python sends only those that apply to the data. Inputs that take
// a number are marked with numberEntry, so they line up at the end of the
// section.
function spectrumControls(v) {
  const controls = v.model.get("_controls");
  const part = select(v.view.spec_part_options, controls.spec_part);
  part.addEventListener("change", () => sendControls(v, { spec_part: part.value }));
  if (!v.view.spectroscopic) return [labelled("part", part), ...processingControls(v)];
  const units = v.view.spec_unit_options;
  const unit = (units.length === 2 ? toggle : select)(units, controls.spec_unit);
  unit.title = "Unit of the frequency axis";
  unit.addEventListener("change", () => sendControls(v, { spec_unit: unit.value }));
  const broadening = element("input");
  broadening.type = "number";
  broadening.min = 0;
  broadening.step = 1;
  broadening.value = controls.line_broadening_hz;
  // A few digits; narrower than the default number input.
  broadening.style.width = "4em";
  broadening.addEventListener("change", () =>
    sendControls(v, { line_broadening_hz: Number(broadening.value) || 0 }),
  );
  return [
    labelled("part", part),
    labelled("unit", unit),
    numberEntry(labelled("line broadening [Hz]", broadening)),
    ...processingControls(v),
  ];
}

// Mark a labelled number input: in the plot options it moves behind the
// other inputs of its section (CSS order), so all numbers sit together.
function numberEntry(node) {
  node.classList.add("hm-csi-number-entry");
  return node;
}

// Processing as in plot. Each change sends the whole "processing" setting;
// Python checks it, processes the data again and sends a new block.
function processingControls(v) {
  const processing = v.model.get("_controls").processing ?? {};
  const send = (change) => sendControls(v, { processing: { ...processing, ...change } });
  const number = (value, min, title, width = "5em") => {
    const input = element("input");
    input.type = "number";
    input.min = min;
    input.step = 1;
    input.value = value;
    input.title = title;
    input.style.width = width;
    return input;
  };
  const items = [];
  if (processing.show_fid !== undefined) {
    const shown = toggle([["FID", "fid"], ["Spec", "spectrum"]], processing.show_fid ? "fid" : "spectrum");
    shown.title = "Show the acquired time signal (FID) or the spectrum";
    shown.addEventListener("change", () => send({ show_fid: shown.value === "fid" }));
    items.push(labelled("show", shown));
    const points = number(processing.npoints, 2, "Spectral points; more than acquired adds zeros to the FID (zero filling)");
    points.addEventListener("change", () => {
      const value = Math.round(Number(points.value));
      if (value >= 2) send({ npoints: value });
      else points.value = processing.npoints;
    });
    items.push(numberEntry(labelled("points", points)));
  }
  const denoise = number(processing.denoise_components, 0, "Principal components kept by the PCA denoising; 0 is off", "3.5em");
  denoise.addEventListener("change", () =>
    send({ denoise_components: Math.max(0, Math.round(Number(denoise.value)) || 0) }),
  );
  items.push(numberEntry(labelled("denoise", denoise)));
  if (processing.repetitions !== undefined) {
    const mode = select(v.view.repetition_modes, processing.repetitions);
    mode.title = "Mean or running sum over the repetition range";
    mode.addEventListener("change", () => send({ repetitions: mode.value }));
    const [first, last] = processing.repetition_range;
    const from = number(first, 0, "First repetition of the range");
    const to = number(last, 0, "Last repetition of the range");
    const sendRange = () => {
      const range = [Math.round(Number(from.value)) || 0, Math.round(Number(to.value)) || 0];
      send({ repetition_range: range });
    };
    from.addEventListener("change", sendRange);
    to.addEventListener("change", sendRange);
    const group = labelled("repetitions", mode);
    group.append(from, to);
    items.push(numberEntry(group));
  }
  if (processing.combine_channels !== undefined) {
    const combine = element("input");
    combine.type = "checkbox";
    combine.checked = processing.combine_channels;
    combine.title = "Combine the receive channels";
    combine.addEventListener("change", () => send({ combine_channels: combine.checked }));
    const method = select(v.view.combine_methods, processing.combine_method);
    method.disabled = !processing.combine_channels;
    method.addEventListener("change", () => send({ combine_method: method.value }));
    const group = labelled("combine channels", combine);
    group.append(method);
    items.push(group);
  }
  return items;
}

function saveControls(v) {
  if (!v.view.can_save_settings) return [];
  const save = button("save view", "Save these view settings on the scan for the next plot_scan");
  save.addEventListener("click", () => {
    v.status.textContent = "Saving view settings ...";
    v.model.send({ type: "save_settings" });
  });
  return [save];
}
