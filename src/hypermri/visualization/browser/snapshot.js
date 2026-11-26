// Start of an HTML snapshot (ScanView.to_html): a stand-in for the widget
// model that serves the values saved from Python and keeps the browser's own
// changes (voxel, curve position, colour range) on the page. Nothing is sent
// anywhere. This file is appended to the viewer parts only in a snapshot.

function decodeBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function snapshotModel(state) {
  const values = {};
  for (const [name, value] of Object.entries(state)) {
    values[name] = value && typeof value === "object" && "base64" in value ? decodeBytes(value.base64) : value;
  }
  const listeners = new Map();
  return {
    get: (name) => values[name],
    set(name, value) {
      values[name] = value;
      for (const listener of [...(listeners.get(`change:${name}`) ?? [])]) listener();
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
