// Control group "curve_fit": fit a relaxation or diffusion model along the
// curve axis in all voxels. The model follows the curve axis: ADC for
// b-values, T2 for echo times, T1 for inversion times. After the fit, the
// result map is the overlay "curve_fit" and the curve panel draws the fit
// and its residual for the selected voxel.
//
// Python owns model and settings (`_curve_fit_tool`, see
// hypermri.visualization.curve_fit_tool). Every input goes to Python as a
// `{type: "curve_fit"}` message; Python answers with a new `_curve_fit_tool`.

function sendCurveFit(v, action, content = {}) {
  v.model.send({ type: "curve_fit", action, ...content });
}

function curveFitControls(v) {
  const settings = v.model.get("_curve_fit_tool") || {};
  if (!settings.offered) return [];
  const message = element("span", "hm-csi-mask-info", settings.message || "");
  const messageRow = element("div", "hm-csi-tool-row");
  messageRow.append(message);

  const items = [
    element("span", "hm-csi-control-label", `${settings.label}, axis in ${settings.axis_unit}`),
  ];
  if (settings.model === "t1") {
    const baseline = fitCheckbox(
      settings.fit_baseline,
      "Fit a constant offset B besides the amplitude A",
      (value) => sendCurveFit(v, "option", { name: "fit_baseline", value }),
    );
    items.push(labelled("baseline", baseline));
  }
  const run = button(
    settings.fitted ? "Fit again" : "Run fit",
    `Fit all voxels; the map (${settings.result_unit}) becomes the overlay "curve_fit"`,
  );
  run.classList.add("hm-csi-primary");
  run.addEventListener("click", () => {
    run.disabled = true;
    message.textContent = "Fitting all voxels ...";
    sendCurveFit(v, "run");
  });
  items.push(run);
  return [
    settingsTable([
      ["Curve fit", "run", items],
      ["Save", "run", saveFitItems(v, "curve_fit", settings.fitted)],
    ]),
    messageRow,
  ];
}

function onCurveFitTool(v) {
  buildGroups(v, ["tool", "curve_fit"]);
  showChosenTool(v);
}
