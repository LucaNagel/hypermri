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

// Step-by-step instructions of a tool, shown above its inputs while the
// "Tutorial" box is ticked. Tools without an entry offer no tutorial.
const TOOL_TUTORIALS = {
  masks: [
    'Type a name and pick a colour under "new", then click "Add".',
    'Choose "polygon" or "freehand" under "draw".',
    'Polygon: click the corners on the map. Backspace removes the last corner, Esc starts again.',
    "Freehand: hold the mouse button and draw the outline; it is stored when you let go.",
    'To remove part of the mask, switch "mode" to "erase" and draw over it. Switch back to "add" to add.',
    "Change the slice and draw on the next one. An unfinished polygon is stored first.",
    'Click "Finish ROI" (or press Enter) to store the polygon and stop drawing.',
    'Tick "ROI mean" to show the mean spectrum inside the mask.',
  ],
  fit: [
    "Click a voxel with strong signal on the map.",
    'Click "Detect lines": peaks higher than "min height (SNR)" times the noise become lines ("max lines": keep only the strongest, 0 keeps all).',
    'To add a line by hand, click its peak in the spectrum and then "Add at cursor".',
    'Check the line table: rename lines, correct the start frequencies, remove wrong lines with "×".',
    'New lines start with the "Defaults for new lines" (search range, T2* limits).',
    'Choose the lineshape and click "Run fit" to fit all voxels.',
    'Look at the fitted curve in the spectrum; "result table" shows the values of the selected voxel.',
  ],
};

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
  const steps = tools.includes(chosen) ? TOOL_TUTORIALS[chosen] : undefined;
  if (!steps) return [labelled("tool", choice)];
  const box = element("input");
  box.type = "checkbox";
  box.checked = v.showTutorial;
  box.title = "Show step by step how to use this tool";
  const tutorial = element("ol", "hm-csi-tutorial");
  for (const step of steps) tutorial.append(element("li", "", step));
  tutorial.hidden = !box.checked;
  box.addEventListener("change", () => {
    v.showTutorial = box.checked;
    tutorial.hidden = !box.checked;
  });
  const label = labelled("Tutorial", box);
  label.classList.add("hm-csi-tutorial-toggle");
  return [labelled("tool", choice), label, tutorial];
}

// Show only the chosen tool's inputs. Leaving the masks stops drawing.
// While a tool is open, the controls after it (show parameters) start a new
// line below its inputs.
function showChosenTool(v) {
  const chosen = v.model.get("tool");
  v.groups.tool?.parentElement?.classList.toggle("hm-csi-tool-open", Boolean(chosen));
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
