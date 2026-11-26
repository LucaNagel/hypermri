// What Python sends and how to read it.
//
// Python sends one float32 block with the processed curves of every voxel in
// the displayed plane, laid out as [row][column][curve] (C order), plus a
// view description in `_view` (axes, labels, colours, option lists).
// Anatomy (under the map) and overlay (above it) arrive as float32 planes in
// `_anatomy` and `_overlay`. Mask coverage arrives in `_mask_planes`, the mean
// curve inside a mask in `_roi_mean`, the fitted spectrum of the selected
// voxel in `_fit_curves` and, with the metabolites split, the curve of the
// other metabolite at the selected voxel in `_other_metabolite`. The curve is
// a spectrum or runs along another dimension (`view.curve_dim`), for example
// repetitions. With
// spectra and several repetitions, `_time_block` holds every voxel's time
// course ([row][column][time]) and `_time_course` the chosen spectral range
// and the mask time course. While repetitions are averaged (or channels
// combined) along the curve, `_map_plane` holds the processed map
// ([row][column]) and the curves keep every entry (`view.map_fixed`).
//
// All parts share one object `v` ("viewer") that holds this data, the current
// selection and the HTML elements; readBlock fills its data fields.

function readBlock(v) {
  const model = v.model;
  v.view = model.get("_view");
  v.block = readFloat32(model, "_block");
  v.mapPlane = v.view.map_fixed ? readFloat32(model, "_map_plane") : null;
  v.anatomyZ = layerRows(model, "_anatomy", v.view.anatomy);
  v.overlayZ = layerRows(model, "_overlay", v.view.overlay);
  v.maskPlanes = model.get("_mask_planes") || [];
  v.roiMean = model.get("_roi_mean") || {};
  v.fitCurves = model.get("_fit_curves") || {};
  v.otherMetabolite = model.get("_other_metabolite") || {};
  v.selection = { ...model.get("selection") };
  readTimeCourse(v);
}

function readTimeCourse(v) {
  v.timeBlock = readFloat32(v.model, "_time_block");
  v.timeCourse = v.model.get("_time_course") || {};
  v.timeMessage = "";
}

const nRows = (v) => v.view.shape[0];
const nColumns = (v) => v.view.shape[1];
const nCurve = (v) => v.view.shape[2];

// Map of the plane at curve index k, as rows of values; the processed map
// when it does not follow the curve position.
function planeValues(v, k) {
  const z = [];
  for (let row = 0; row < nRows(v); row += 1) {
    const line = new Array(nColumns(v));
    for (let column = 0; column < nColumns(v); column += 1) {
      const voxel = row * nColumns(v) + column;
      line[column] = v.mapPlane ? v.mapPlane[voxel] : v.block[voxel * nCurve(v) + k];
    }
    z.push(line);
  }
  return z;
}

// Curve (spectrum or other) of one voxel.
function curveValues(v, row, column) {
  const start = (row * nColumns(v) + column) * nCurve(v);
  return Array.from(v.block.subarray(start, start + nCurve(v)));
}

// Time course of one voxel; empty while block and view do not match yet.
function timeValues(v, row, column) {
  const n = v.view.time ? v.view.time.x.length : 0;
  if (v.timeBlock.length !== nRows(v) * nColumns(v) * n) return [];
  const start = (row * nColumns(v) + column) * n;
  return Array.from(v.timeBlock.subarray(start, start + n));
}

// Finite range of every value in the plane, computed by Python.
function planeRange(v) {
  return v.view.plane_range;
}
