// Small helpers shared by all parts: loading Plotly.js, reading the binary
// arrays Python sends, and building HTML inputs.

const PLOTLY_PROMISE_KEY = "__hypermriScanViewPlotly";

// How long the browser waits for Python to send Plotly.js before it falls
// back to the public package (a notebook opened without a running kernel).
const PLOTLY_KERNEL_WAIT_MS = 5000;

// Plotly.js comes from the first source that has it: a page that already
// loaded it (the HTML snapshot), the running Python kernel, which sends the
// copy of the installed plotly package, or, when Python has not answered
// within PLOTLY_KERNEL_WAIT_MS (a notebook opened without a kernel), the same
// version from the public npm package. A late answer from Python still
// counts. `onNote` receives a line for the status bar while it waits.
function loadPlotly(model, onNote = () => {}) {
  if (globalThis.Plotly?.newPlot) return Promise.resolve(globalThis.Plotly);
  if (!globalThis[PLOTLY_PROMISE_KEY]) {
    globalThis[PLOTLY_PROMISE_KEY] = new Promise((resolve) => {
      const onMessage = (message, buffers) => {
        if (message?.type !== "plotly") return;
        clearTimeout(timer);
        model.off("msg:custom", onMessage);
        const source = new TextDecoder().decode(buffers[0]);
        const module = { exports: {} };
        new Function("module", "exports", source)(module, module.exports);
        resolve(module.exports);
      };
      const fromPackage = () => {
        onNote("Python does not answer; loading Plotly.js from the web ...");
        plotlyFromPackage(model.get("_plotly_version")).then(resolve, (error) =>
          onNote(`Waiting for Python to send Plotly.js (web copy failed: ${error.message})`),
        );
      };
      const timer = setTimeout(fromPackage, PLOTLY_KERNEL_WAIT_MS);
      model.on("msg:custom", onMessage);
      try {
        model.send({ type: "request_plotly" });
      } catch {
        clearTimeout(timer);
        fromPackage();
      }
    });
  }
  return globalThis[PLOTLY_PROMISE_KEY];
}

async function plotlyFromPackage(version) {
  if (!version) throw new Error("the Plotly.js version is unknown");
  const module = await import(`https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/+esm`);
  return module.default ?? module;
}

function readFloat32(model, name) {
  const bytes = model.get(name);
  const start = bytes.byteOffset;
  return new Float32Array(bytes.buffer.slice(start, start + bytes.byteLength));
}

// Rows of an anatomy or overlay plane for Plotly; missing values (NaN) become gaps.
function layerRows(model, name, layer) {
  if (!layer?.shape) return null;
  const values = readFloat32(model, name);
  const [rows, columns] = layer.shape;
  const z = [];
  for (let row = 0; row < rows; row += 1) {
    const line = new Array(columns);
    for (let column = 0; column < columns; column += 1) {
      const value = values[row * columns + column];
      line[column] = Number.isFinite(value) ? value : null;
    }
    z.push(line);
  }
  return z;
}

function finiteRange(values) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (Number.isFinite(value)) {
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  if (low === Infinity) return [0, 1];
  if (low === high) return [low - 0.5, high + 0.5];
  return [low, high];
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function labelled(text, control) {
  const wrapper = element("label", "hm-csi-control");
  wrapper.append(element("span", "hm-csi-control-label", text), control);
  return wrapper;
}

// Options are values, or [label, value] pairs when the shown text differs.
function select(options, value) {
  const node = element("select");
  for (const option of options) {
    const [label, optionValue] = Array.isArray(option) ? option : [option, option];
    const item = element("option", "", label);
    item.value = optionValue;
    node.append(item);
  }
  node.value = value;
  return node;
}

// A switch between two choices, drawn as one split button: the shown
// choice is highlighted and a click anywhere switches to the other. It acts
// like a <select>: it has `value` and `disabled`, and fires "change" when the
// value changes. Options are values or [label, value] pairs.
function toggle(options, value) {
  const node = element("span", "hm-csi-toggle");
  node.setAttribute("role", "radiogroup");
  const pairs = options.map((option) => (Array.isArray(option) ? option : [option, option]));
  const parts = pairs.map(([label, optionValue]) => {
    const part = element("button", "hm-csi-toggle-part", label);
    part.type = "button";
    part.dataset.value = optionValue;
    part.setAttribute("role", "radio");
    node.append(part);
    return part;
  });
  let current = String(value);
  const show = () => {
    for (const part of parts) {
      const on = part.dataset.value === current;
      part.classList.toggle("hm-csi-toggle-on", on);
      part.setAttribute("aria-checked", String(on));
    }
  };
  for (const part of parts) {
    part.addEventListener("click", () => {
      // A click on the shown choice switches to the other one.
      const next = part.dataset.value === current ? parts.find((other) => other !== part) : part;
      current = next.dataset.value;
      show();
      node.dispatchEvent(new Event("change"));
    });
  }
  Object.defineProperty(node, "value", {
    get: () => current,
    set: (newValue) => {
      current = String(newValue);
      show();
    },
  });
  Object.defineProperty(node, "disabled", {
    get: () => parts[0].disabled,
    set: (off) => {
      for (const part of parts) part.disabled = off;
    },
  });
  show();
  return node;
}

// A slider for the slice of an anatomy or overlay layer. While the layer
// follows the map, the slider shows the slice drawn now and the button
// "follow map" is pressed; moving the slider fixes the slice. The slider
// sits in control group `group` and is sent while it is dragged
// (sendWhileDragged); update(slice) returns the `_controls` update for a
// slice, or for null from the button, and null when Python already shows it.
function layerSliceSlider(v, group, count, chosen, drawn, update) {
  const wrapper = element("span", "hm-csi-range-slider");
  const slider = element("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = count - 1;
  slider.value = chosen ?? drawn ?? 0;
  const value = element("span", "hm-csi-value", slider.value);
  slider.addEventListener("input", () => (value.textContent = slider.value));
  sendWhileDragged(v, slider, group, update);
  const follow = button("follow map", "Let the slice follow the shown map again");
  follow.classList.toggle("hm-csi-primary", chosen == null);
  follow.disabled = chosen == null;
  follow.addEventListener("click", () => {
    const followed = update(null);
    if (followed) sendControls(v, followed);
  });
  wrapper.append(slider, value, follow);
  return wrapper;
}

function hexToRgba(hex, alpha) {
  const value = parseInt(hex.slice(1), 16);
  return `rgba(${(value >> 16) & 255},${(value >> 8) & 255},${value & 255},${alpha})`;
}

function parsePath(path) {
  const numbers = (path.match(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi) || []).map(Number);
  const points = [];
  for (let i = 0; i + 1 < numbers.length; i += 2) points.push([numbers[i], numbers[i + 1]]);
  return points;
}

function button(text, title) {
  const node = element("button", "hm-csi-button", text);
  node.type = "button";
  if (title) node.title = title;
  return node;
}

// +1 when the next index lies further right (x) or higher up (y) on screen.
function screenStep(centres, range) {
  if (centres.length < 2) return 1;
  const along = Math.sign(centres[1] - centres[0]) * Math.sign(range[range.length - 1] - range[0]);
  return along < 0 ? -1 : 1;
}

// A number input for a colour limit. accept(value) returns false for a
// limit that does not lie on the right side of the other one, which is
// then reverted.
// `hint` is shown in the status line when the entry is refused.
function limitInput(v, value, accept, hint = "Enter a number; the colour minimum must be below the maximum.") {
  const input = element("input");
  input.type = "number";
  input.step = "any";
  input.style.width = "7em";
  const show = (number) => (input.value = number == null ? "" : Number(Number(number).toPrecision(6)));
  show(value);
  input.addEventListener("change", () => {
    const number = Number(input.value);
    if (input.value !== "" && Number.isFinite(number) && accept(number)) return;
    v.status.textContent = hint;
    show(value);
  });
  input.show = (number) => {
    value = number;
    show(number);
  };
  return input;
}

// A 0-1 slider for an opacity.
function opacitySlider(value, title) {
  const slider = element("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = 1;
  slider.step = 0.05;
  slider.value = value;
  if (title) slider.title = title;
  return slider;
}

// One slider with two handles for a lower and an upper limit, with both
// values shown next to it. `change(low, high, done)` returns the accepted
// [low, high]; `done` is true when a handle is let go. `show(low, high)`
// moves the handles, `fits(low, high)` tells whether both lie on the scale.
function doubleRangeSlider(low, high, bottom, top, step, format, change) {
  const wrapper = element("span", "hm-csi-range-slider");
  const track = element("span", "hm-csi-double-range");
  const fill = element("span", "hm-csi-double-range-fill");
  const handle = () => {
    const slider = element("input");
    slider.type = "range";
    slider.min = bottom;
    slider.max = top;
    slider.step = step;
    return slider;
  };
  const lower = handle();
  const upper = handle();
  lower.title = "Lower limit";
  upper.title = "Upper limit";
  const readout = element("span", "hm-csi-range-value");
  const position = (number) => (top > bottom ? ((number - bottom) / (top - bottom)) * 100 : 0);
  wrapper.show = (lowValue, highValue) => {
    lower.value = lowValue;
    upper.value = highValue;
    fill.style.left = `${position(Number(lower.value))}%`;
    fill.style.right = `${100 - position(Number(upper.value))}%`;
    readout.textContent = `${format(lowValue)} – ${format(highValue)}`;
  };
  wrapper.fits = (lowValue, highValue) => lowValue >= bottom && highValue <= top;
  // A handle stops one step before the other one.
  const send = (moved, done) => {
    let lowValue = Number(lower.value);
    let highValue = Number(upper.value);
    if (moved === lower && lowValue > highValue - step) lowValue = highValue - step;
    if (moved === upper && highValue < lowValue + step) highValue = lowValue + step;
    wrapper.show(...change(lowValue, highValue, done));
  };
  for (const slider of [lower, upper]) {
    slider.addEventListener("input", () => send(slider, false));
    slider.addEventListener("change", () => send(slider, true));
  }
  track.append(fill, lower, upper);
  wrapper.append(track, readout);
  wrapper.show(low, high);
  return wrapper;
}
