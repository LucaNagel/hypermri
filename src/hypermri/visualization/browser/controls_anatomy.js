// Control group "anatomy": show the anatomy under the map and choose its
// colour scale and colour range in percent of the shown slice; all three
// only redraw, and the range is saved to Python when a handle is released. Empty when no anatomy was given. The
// anatomy slice and the map opacity sit in the top bar (controls_layers.js),
// and so does the slider for the anatomy slice under the outline of a
// single voxel or slab.

// Under a voxel or slab outline: a slider for the anatomy slice (Python
// draws the new slice and outline). Shown in the top bar (layerControls).
function outlineSliceControls(v) {
  const outline = v.view.outline;
  const items = [];
  if (outline.n_slices > 1) {
    const slider = element("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = outline.n_slices - 1;
    slider.value = outline.slice_index;
    const value = element("span", "hm-csi-value", slider.value);
    slider.addEventListener("input", () => (value.textContent = slider.value));
    slider.addEventListener("change", () =>
      sendControls(v, { anatomy_slice: Number(slider.value) }),
    );
    needsPython(v, slider);
    const group = labelled(`anatomy ${outline.slice_dim}`, slider);
    group.append(value);
    items.push(group);
  }
  if (v.view.anatomy.warning) {
    items.push(element("span", "hm-csi-mask-error", v.view.anatomy.warning));
  }
  return items;
}

function anatomyControls(v) {
  if (!v.view.anatomy?.shape || v.view.single_voxel) return [];
  const visible = element("input");
  visible.type = "checkbox";
  visible.checked = v.showAnatomy;
  visible.addEventListener("change", () => {
    v.showAnatomy = visible.checked;
    if (v.anatomyOpacitySlider) v.anatomyOpacitySlider.disabled = !v.showAnatomy;
    drawImagePanel(v);
  });
  // A colour scale other than the map's tells two overlaid grey images apart.
  const cmap = select(v.view.anatomy_cmaps, v.model.get("_display").anatomy_cmap);
  cmap.title = "Colour scale of the anatomy";
  cmap.addEventListener("change", () => setDisplay(v, { anatomy_cmap: cmap.value }));
  const shown = v.model.get("_anatomy_range");
  const range = doubleRangeSlider(
    ...(shown.length ? shown : [0, 100]),
    0,
    100,
    0.5,
    (share) => `${share.toFixed(1)} %`,
    (low, high, done) => {
      // Redrawn by onAnatomyRange.
      v.model.set("_anatomy_range", [low, high]);
      if (done) v.model.save_changes();
      return [low, high];
    },
  );
  range.title = "Colour range of the anatomy in percent of the shown slice";
  v.anatomyRangeSlider = range;
  const items = [labelled("show", visible), labelled("colours", cmap), labelled("range", range)];
  if (v.view.anatomy.warning) {
    items.push(element("span", "hm-csi-mask-error", v.view.anatomy.warning));
  }
  return items;
}

function onAnatomyRange(v) {
  const shown = v.model.get("_anatomy_range");
  v.anatomyRangeSlider?.show(...(shown.length ? shown : [0, 100]));
  if (v.Plotly) drawImagePanel(v);
}

function onAnatomyOpacity(v) {
  const value = v.model.get("_anatomy_opacity");
  const slider = v.anatomyOpacitySlider;
  if (slider && Number(slider.value) !== value) slider.value = value;
  if (v.Plotly) drawImagePanel(v);
}
