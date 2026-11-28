// Control group "display": how the map is drawn, as the display checkboxes
// of plot, the masks and the bands of the slices of other views. The settings are the
// synchronised `_display` trait
// (ScanView.set_display); Python stores them in the state. Everything only
// redraws the map, except the scalebar, whose length Python chooses, and the
// colour scales of map and anatomy, which Python sends with the view.

const DISPLAY_SWITCHES = [
  ["min_to_nan", "min → NaN", "Leave voxels below the colour minimum transparent"],
  ["show_scalebar", "scalebar", "Scalebar in the corner (only with a physical voxel spacing)"],
  ["show_colorbar", "colorbar", "Colour bar beside the map"],
  ["show_grid_lines", "grid", "Axis grid lines over the map"],
  ["show_tooltips", "tooltips", "Values under the mouse: voxel, position and mask coverage on the map, values on the spectrum and time course"],
  ["show_crosshair", "crosshair", "Dotted lines through the selected voxel"],
  ["show_masks", "masks", "Masks over the map (hidden masks are kept)"],
  ["show_mask_outlines", "outlines", "Drawn mask outlines at full precision"],
];

// Grid lines of every panel, as show_grid_lines in plot.
function showGrid(v) {
  return Boolean(v.model.get("_display").show_grid_lines);
}

function setDisplay(v, update) {
  v.model.set("_display", { ...v.model.get("_display"), ...update });
  v.model.save_changes();
  draw(v);
}

function displayControls(v) {
  if (v.view.single_voxel) return [];
  const display = v.model.get("_display");
  const smoothing = select(v.view.smoothing_options, display.smoothing);
  smoothing.title = "Smoothing between voxel centres (the voxel values stay the same)";
  smoothing.addEventListener("change", () => setDisplay(v, { smoothing: smoothing.value }));
  const cmap = select(v.view.map_cmaps, display.map_cmap);
  cmap.title = "Colour scale of the map (Python sends the new scale)";
  cmap.addEventListener("change", () => setDisplay(v, { map_cmap: cmap.value }));
  needsPython(v, cmap);
  const items = [labelled("map colours", cmap), labelled("smoothing", smoothing)];
  for (const [name, text, title] of DISPLAY_SWITCHES) {
    const box = element("input");
    box.type = "checkbox";
    box.checked = display[name];
    box.title = title;
    box.addEventListener("change", () => setDisplay(v, { [name]: box.checked }));
    if (name === "show_scalebar") needsPython(v, box);
    items.push(labelled(text, box));
  }
  if ((v.model.get("_slice_lines") || []).length) {
    const box = element("input");
    box.type = "checkbox";
    box.checked = display.show_slice_lines === true;
    box.title = "Bands where the slices shown in the other views cross this image";
    box.addEventListener("change", () => setDisplay(v, { show_slice_lines: box.checked }));
    items.push(labelled("other slices", box));
  }
  return items;
}

// Set from Python (ScanView.set_display): show the new settings and redraw.
// The overlay controls hold the overlay's smoothing and tooltips.
function onDisplay(v) {
  buildGroups(v, ["display", "overlay"]);
  if (v.Plotly) draw(v);
}
