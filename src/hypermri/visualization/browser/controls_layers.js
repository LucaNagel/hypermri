// Control group "layers" in the top bar: what is needed all the time when
// the map is drawn over an anatomy or under an overlay. A slider chooses the
// anatomy slice (Python draws it and, for a map with several slices, moves
// the map to the nearest one, also while the slider is dragged); "follow map" lets the anatomy follow the map
// again. The opacity sliders of the map above the anatomy and of the
// overlay only redraw. For a single voxel or slab, a slider chooses the
// anatomy slice under its outline. Empty without anatomy and overlay.

function layerControls(v) {
  if (v.view.single_voxel) return v.view.outline?.kind ? outlineSliceControls(v) : [];
  const items = [];
  const anatomy = v.view.anatomy;
  if (anatomy?.shape) {
    const settings = v.model.get("_controls").anatomy;
    if (settings && anatomy.slice_count > 1) {
      const slice = layerSliceSlider(v, "layers", anatomy.slice_count, settings.slice, anatomy.slice_index, (value) =>
        v.model.get("_controls").anatomy?.slice === value ? null : { anatomy: { slice: value } },
      );
      if (!v.model.get("_anatomy_slices")) {
        for (const input of slice.querySelectorAll("input, button")) needsPython(v, input);
      }
      items.push(labelled(`anatomy ${anatomy.slice_dim}`, slice));
    }
    const opacity = opacitySlider(
      v.model.get("_anatomy_opacity"),
      "Opacity of the map above the anatomy (0 shows only the anatomy)",
    );
    opacity.disabled = !v.showAnatomy;
    // Redrawn by onAnatomyOpacity; saved to Python when the slider is released.
    opacity.addEventListener("input", () => v.model.set("_anatomy_opacity", Number(opacity.value)));
    opacity.addEventListener("change", () => v.model.save_changes());
    v.anatomyOpacitySlider = opacity;
    items.push(labelled("map opacity", opacity));
  }
  if (v.model.get("_controls").overlay) {
    const opacity = opacitySlider(v.model.get("_overlay_opacity"), "Opacity of the overlay above the map");
    // Redrawn by onOverlayOpacity; saved to Python when the slider is released.
    opacity.addEventListener("input", () => v.model.set("_overlay_opacity", Number(opacity.value)));
    opacity.addEventListener("change", () => v.model.save_changes());
    v.overlayOpacitySlider = opacity;
    items.push(labelled("overlay opacity", opacity));
  }
  return items;
}
