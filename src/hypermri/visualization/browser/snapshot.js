// Start of an HTML snapshot (ScanView.to_html): a stand-in for the widget
// model that serves the values saved from Python and keeps the browser's own
// changes (voxel, curve position, colour range) on the page. Nothing is sent
// anywhere. A chosen anatomy slice is answered from the values Python saved
// for every slice (_anatomy_slices). This file is appended to the viewer
// parts only in a snapshot.

function decodeBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function decodeValue(value) {
  return value && typeof value === "object" && "base64" in value ? decodeBytes(value.base64) : value;
}

function snapshotModel(state) {
  const values = {};
  for (const [name, value] of Object.entries(state)) values[name] = decodeValue(value);
  const listeners = new Map();
  const change = (name, value) => {
    values[name] = value;
    for (const listener of [...(listeners.get(`change:${name}`) ?? [])]) listener();
  };
  // The saved values of one anatomy choice; _view goes last because its
  // listener reads the other values to draw the new plane.
  const showAnatomySlice = (slice) => {
    const saved = values._anatomy_slices[slice == null ? "follow" : String(slice)];
    if (!saved) return;
    const names = Object.keys(saved).sort((a, b) => (a === "_view") - (b === "_view"));
    for (const name of names) change(name, decodeValue(saved[name]));
  };
  return {
    get: (name) => values[name],
    set(name, value) {
      if (name === "_controls" && values._anatomy_slices) {
        showAnatomySlice(value.anatomy?.slice);
        return;
      }
      change(name, value);
    },
    save_changes() {},
    send() {},
    on(name, listener) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(listener);
    },
    off(name, listener) {
      listeners.get(name)?.delete(listener);
    },
  };
}

render({
  model: snapshotModel(JSON.parse(document.getElementById("hypermri-state").textContent)),
  el: document.getElementById("hypermri-view"),
});
