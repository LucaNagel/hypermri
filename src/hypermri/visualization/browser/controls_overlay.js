// Control group "overlay": a parameter map or other image above the CSI map.
// Empty when no overlay was given.
//
// Layer, metabolite, data source, slice, visibility, colormap and colour range
// are recomputed in Python; opacity, smoothing and tooltips only redraw.

function overlayControls(v) {
  const settings = v.model.get("_controls").overlay;
  if (!settings || v.view.single_voxel) return [];
  const options = v.view.overlay_options;
  const send = (update) =>
    sendControls(v, { overlay: { ...v.model.get("_controls").overlay, ...update } });
  // A new layer, metabolite or range mode starts from its full colour
  // range, which Python sets when no limits are sent.
  const sendSelection = (update) => send({ ...update, vmin: null, vmax: null });
  const visible = element("input");
  visible.type = "checkbox";
  visible.checked = settings.visible;
  visible.addEventListener("change", () => send({ visible: visible.checked }));
  const layer = select(options.variables, settings.variable);
  layer.addEventListener("change", () => sendSelection({ variable: layer.value, metabolite: null }));
  const items = [labelled("show", visible), labelled("layer", layer)];
  if (options.metabolites.length) {
    const metabolite = select(options.metabolites, settings.metabolite);
    metabolite.addEventListener("change", () => sendSelection({ metabolite: metabolite.value }));
    items.push(labelled("metabolite", metabolite));
  }
  if (options.data_sources?.length) {
    // Python chooses the layer, range and slice of the new source.
    const source = select(options.data_sources, settings.data_source);
    source.addEventListener("change", () => send({ data_source: source.value }));
    items.push(labelled("source", source));
  }
  if (options.slice_count > 1) {
    const slice = layerSliceSlider(v, "overlay", options.slice_count, settings.slice, v.view.overlay?.slice_index, (value) => {
      const overlay = v.model.get("_controls").overlay;
      return overlay.slice === value ? null : { overlay: { ...overlay, slice: value } };
    });
    items.push(labelled(options.slice_dim, slice));
  }
  const cmap = select(options.cmaps, settings.cmap);
  cmap.addEventListener("change", () => send({ cmap: cmap.value }));
  items.push(labelled("colours", cmap));
  items.push(...overlayRangeControls(v, settings, options, send, sendSelection));
  items.push(...overlayDisplayControls(v));
  if (v.view.overlay?.warning) {
    items.push(element("span", "hm-csi-mask-error", v.view.overlay.warning));
  }
  return items;
}

// Smoothing and tooltips of the overlay are display settings
// (ScanView.set_display): they only redraw the map.
function overlayDisplayControls(v) {
  const display = v.model.get("_display");
  const smoothing = select(v.view.smoothing_options, display.overlay_smoothing);
  smoothing.title = "Smoothing of the overlay between its voxel centres (the values stay the same)";
  smoothing.addEventListener("change", () => setDisplay(v, { overlay_smoothing: smoothing.value }));
  const tooltips = element("input");
  tooltips.type = "checkbox";
  tooltips.checked = display.show_overlay_tooltips === true;
  tooltips.title = "Overlay value in the label under the mouse (of the overlay voxel nearest to the map voxel centre)";
  tooltips.addEventListener("change", () => setDisplay(v, { show_overlay_tooltips: tooltips.checked }));
  return [labelled("smoothing", smoothing), labelled("tooltips", tooltips)];
}

// Range mode (percent of the layer or its values) and one slider with two
// handles for the colour limits; Python recomputes the overlay when a
// handle is let go.
function overlayRangeControls(v, settings, options, send, sendSelection) {
  const percent = settings.range_mode === "Percentage";
  const mode = toggle(
    options.range_modes.map((name) => [name === "Percentage" ? "%" : "values", name]),
    settings.range_mode,
  );
  mode.title = "Colour limits in percent of the layer's value range or in its values";
  mode.addEventListener("change", () => sendSelection({ range_mode: mode.value }));
  const [start, end] = percent ? [0, 100] : options.value_range ?? [settings.vmin, settings.vmax];
  const bottom = Math.min(start, settings.vmin);
  const top = Math.max(end, settings.vmax);
  const step = percent ? 0.5 : top > bottom ? (top - bottom) / 500 : 1;
  const format = (value) =>
    percent ? `${Number(value).toFixed(1)} %` : `${Number(Number(value).toPrecision(4))}`;
  const limits = doubleRangeSlider(settings.vmin, settings.vmax, bottom, top, step, format, (low, high, done) => {
    if (done) send({ vmin: low, vmax: high });
    return [low, high];
  });
  limits.title = "Values at or below the lower limit get the first colour, at or above the upper the last";
  return [labelled("range", mode), limits];
}

function onOverlayOpacity(v) {
  const value = v.model.get("_overlay_opacity");
  const slider = v.overlayOpacitySlider;
  if (slider && Number(slider.value) !== value) slider.value = value;
  if (v.Plotly) drawImagePanel(v);
}
