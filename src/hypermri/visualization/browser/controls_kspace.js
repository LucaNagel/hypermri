// Control group "kspace": shows the map and curves of the image or of its
// spatial k-space (along Read and Phase), optionally with a spatial filter
// (Gaussian or Fermi window in cycles/mm). Python computes the shown copy
// (ScanView.set_kspace) and answers with a new block; the data of the view
// are not changed. In k-space anatomy, overlays, masks and fits are off.
// The choices arrive as `view.kspace`; every input sends a
// `{type: "kspace"}` message.

function sendKspace(v, update) {
  v.model.send({ type: "kspace", ...update });
}

function kspaceControls(v) {
  const settings = v.view.kspace;
  if (!settings || !settings.available || v.view.single_voxel) return [];
  const space = toggle([["k-space", "kspace"], ["image", "image"]], settings.shown ? "kspace" : "image");
  space.title = "Show the image or its spatial k-space (Read and Phase, each slice on its own)";
  space.addEventListener("change", () => sendKspace(v, { shown: space.value === "kspace" }));
  const filter = select(SPATIAL_FILTER_CHOICES, settings.filter);
  filter.title = "Spatial filter of the k-space, applied to the image and the k-space shown";
  filter.addEventListener("change", () => sendKspace(v, { filter: filter.value }));
  const width = (name, label, title) =>
    labelled(label, fitNumber(settings[name], 5, title, (value) => sendKspace(v, { [name]: value })));
  const items = [labelled("show", space), labelled("filter", filter)];
  if (settings.filter === "gaussian") {
    items.push(width("sigma_per_mm", "sigma 1/mm", "Width of the Gaussian in cycles/mm"));
  }
  if (settings.filter === "fermi") {
    items.push(
      width("radius_per_mm", "radius 1/mm", "Radius of the Fermi filter in cycles/mm"),
      width("transition_per_mm", "edge 1/mm", "Edge width of the Fermi filter in cycles/mm"),
    );
  }
  if (settings.error) items.push(element("span", "hm-csi-mask-info", settings.error));
  return items;
}
