// Control group "tool": a menu that chooses which tool (plugin) shows its
// inputs. The tools are the control groups listed under `tools` in the
// layout (TOOLS in layout.py). Their inputs sit right after this menu and
// only the chosen one is visible. The choice is the synchronised `tool`
// trait (ScanView.tool), "" for none.

// Tools with nothing to offer for the shown data are left out of the menu:
// the mask tool of a view whose linked neighbour holds its inputs, and the
// fit tool for data that are not spectra and the curve fit for a curve axis
// without a model (b-values, echoes, inversion times), IDEAL unless the
// curve axis is the echoes, and the signal decay outside plot_signal_decay.
function availableTools(v) {
  const offered = {
    masks: !v.view.single_voxel && v.model.get("_mask_tool_shown"),
    fit: Boolean(v.model.get("_fit_tool")?.offered),
    curve_fit: Boolean(v.model.get("_curve_fit_tool")?.offered),
    ideal: Boolean(v.model.get("_ideal_tool")?.offered),
    signal_decay: Boolean(v.model.get("_signal_decay_tool")?.offered),
  };
  return v.layout.tools.filter((name) => offered[name] ?? true);
}

function toolControls(v) {
  const tools = availableTools(v);
  if (!tools.length) return [];
  const chosen = v.model.get("tool");
  const choice = select([["none", ""], ...tools], tools.includes(chosen) ? chosen : "");
  choice.title = "Show the inputs of one tool";
  choice.addEventListener("change", () => {
    v.model.set("tool", choice.value);
    v.model.save_changes();
  });
  return [labelled("tool", choice)];
}

// Show only the chosen tool's inputs. Leaving the masks stops drawing.
function showChosenTool(v) {
  const chosen = v.model.get("tool");
  for (const name of v.layout.tools) {
    if (v.groups[name]) v.groups[name].hidden = name !== chosen;
  }
  if (chosen !== "masks" && v.tool !== "off") {
    v.tool = "off";
    v.pending = [];
    buildMaskControls(v);
    if (v.Plotly) draw(v, { curve: false });
  }
  // The search ranges of the fit lines show only while the fit tool is open.
  if (v.Plotly) drawCurvePanel(v);
}

function onTool(v) {
  buildGroups(v, ["tool"]);
  showChosenTool(v);
}
