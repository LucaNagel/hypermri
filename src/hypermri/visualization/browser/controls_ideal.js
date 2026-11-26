// Control group "ideal": separate chemical components (for example pyruvate
// and lactate) in complex multi-echo images with IDEAL, offered when the
// curve axis is the echoes (echo_idx). The user enters component names and
// frequencies in Hz; the echo spacing comes from the echo axis unless it is
// entered in ms. After the run, the overlay "ideal" holds one magnitude map
// per component and the field offset map in Hz.
//
// Python owns the settings and checks them (`_ideal_tool`, see
// hypermri.visualization.ideal_tool). Every input goes to Python as a
// `{type: "ideal"}` message; Python answers with a new `_ideal_tool`.

function sendIdeal(v, action, content = {}) {
  v.model.send({ type: "ideal", action, ...content });
}

// A text input that sends its text when it is changed.
function idealText(value, width, placeholder, title, onChange) {
  const input = element("input");
  input.type = "text";
  input.value = value || "";
  input.placeholder = placeholder;
  input.title = title;
  input.style.width = `${width}em`;
  input.addEventListener("change", () => onChange(input.value));
  return input;
}

function idealControls(v) {
  const settings = v.model.get("_ideal_tool") || {};
  if (!settings.offered) return [];
  const message = element("span", "hm-csi-mask-info", settings.message || "");
  if (!settings.complex) {
    message.textContent = "The shown echoes are real; IDEAL needs complex images.";
  }
  const messageRow = element("div", "hm-csi-tool-row");
  messageRow.append(message);
  const option = (name) => (value) => sendIdeal(v, "option", { name, value });

  const names = idealText(
    settings.component_names, 12, "pyruvate, lactate",
    "Names of the components, separated by commas; they name the maps",
    option("component_names"),
  );
  const frequencies = idealText(
    settings.frequencies_hz, 9, "0, 390",
    "Frequency of each component in Hz relative to the reference frequency, in the order of the names",
    option("frequencies_hz"),
  );
  const axis = settings.axis_spacing_ms;
  const spacing = idealText(
    settings.echo_spacing_ms == null ? "" : String(settings.echo_spacing_ms), 5,
    axis == null ? "needed" : `${Number(axis.toPrecision(4))} (axis)`,
    "Time between two echoes in ms; empty takes it from the echo axis",
    option("echo_spacing_ms"),
  );
  const offset = fitCheckbox(
    settings.fit_field_offset, "Fit the field offset (Hz) of each voxel",
    option("fit_field_offset"),
  );
  const theta = fitCheckbox(
    settings.fit_theta, "Fit the alternating-echo modulation Theta (bipolar read-out)",
    option("fit_theta"),
  );
  const run = button(
    settings.fitted ? "Fit again" : "Run IDEAL",
    `Separate the components in all voxels of the ${settings.n_echoes} echoes; the maps become the overlay "ideal"`,
  );
  run.classList.add("hm-csi-primary");
  run.addEventListener("click", () => {
    run.disabled = true;
    message.textContent = "Separating components in all voxels ...";
    sendIdeal(v, "run");
  });
  return [
    settingsTable([
      ["Components", "names", [labelled("names", names), labelled("Hz", frequencies)]],
      [
        "Fit",
        "run",
        [labelled("echo spacing ms", spacing), labelled("field offset", offset), labelled("Theta", theta), run],
      ],
      ["Save", "run", saveFitItems(v, "ideal", settings.fitted)],
    ]),
    messageRow,
  ];
}

function onIdealTool(v) {
  buildGroups(v, ["tool", "ideal"]);
  showChosenTool(v);
}
