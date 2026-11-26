"""Spatial labels from the geometry belonging to a Bruker reconstruction.

Scanner images use their selected ``pdata`` metadata. Raw images use acquisition
geometry and the default VisuCore positions. Neither path changes the handler.

Notes
-----
Spatial labels use millimetres and retain three-dimensional coordinate grids.
Uniform single-package slice origins follow physical XYZ voxel centres; acquired
samples, read/phase labels and storage precision are preserved. Missing physical
geometry retains acquisition labels, and multiple packages retain index axes.
"""

import numpy as np

from hypermri.utils.coordinates.grids import assign_xyz_coordinates, get_rps_axes

_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
_SLICE_AXIS_ATOL_MM = 1e-6
_VISU_FIELDS = {
    "visucore_position": "VisuCorePosition",
    "visucore_orientation": "VisuCoreOrientation",
    "visucore_size": "VisuCoreSize",
    "visucore_extent": "VisuCoreExtent",
    "visucore_thickness": "VisuCoreFrameThickness",
}


def _selected_geometry(handler, recon_num):
    """Snapshot selected VisuCore fields without mixing reconstruction folders."""
    reco, visu = handler.load_reconstruction_parameters(recon_num)
    geometry = {name: visu.get(key) for name, key in _VISU_FIELDS.items()}
    geometry["reco_transposition"] = reco.get("RECO_transposition")
    for name in (
        "acquisition_matrix",
        "spatial_dimension",
        "n_acquisition",
        "method",
    ):
        geometry[name] = getattr(handler, name, None)
    geometry["fov_mm"] = geometry["visucore_extent"]
    geometry["slice_thickness_mm"] = geometry["visucore_thickness"]
    return geometry


def _same_transposition(first, second):
    """Compare scalar and one-dimensional representations of the same frames."""
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(np.asarray(first).reshape(-1), np.asarray(second).reshape(-1))


def _physical_slice_axis(data):
    """Project a uniform stack onto increasing slice index without moving samples."""
    if data.sizes["slice_idx"] < 2 or not all(
        name in data.coords for name in ("x", "y", "z")
    ):
        return None
    line = np.stack(
        [data[name].isel(read_idx=0, phase_idx=0).values for name in ("x", "y", "z")],
        axis=-1,
    )
    step = line[-1] - line[0]
    norm = np.linalg.norm(step)
    if not np.isfinite(norm) or norm == 0:
        return None
    direction = step / norm
    positions = line @ direction
    if not np.allclose(
        np.diff(positions),
        norm / (len(positions) - 1),
        atol=_SLICE_AXIS_ATOL_MM,
        rtol=0,
    ):
        return None
    projected = sum(
        data[name] * component
        for name, component in zip(("x", "y", "z"), direction, strict=True)
    )
    # An independent slice axis must be constant within each image plane.
    if not np.allclose(
        projected.transpose(*_SPATIAL_DIMS),
        positions[None, None, :],
        atol=_SLICE_AXIS_ATOL_MM,
        rtol=0,
    ):
        return None
    return positions, direction


def _attach_image_geometry(data, handler, *, recon_num=None):
    """Attach XYZ and RPS voxel centres in mm, preserving samples and ordering.

    The RPS grids keep the established three-dimensional coordinate contract.
    Multiple packages retain index axes. Selected scanner RPS directions and
    origins come from its VisuCore geometry; unavailable geometry stays indexed.
    Raw and matching scanner stacks use XYZ projection for their slice origin;
    acquisition read/phase axes and stacks without usable XYZ stay unchanged.
    """
    if not all(dim in data.dims for dim in _SPATIAL_DIMS):
        return data
    geometry = handler if recon_num is None else _selected_geometry(handler, recon_num)
    matching_acquisition = recon_num is None or (
        all(
            np.array_equal(geometry[name], getattr(handler, name, None))
            for name in _VISU_FIELDS
        )
        and _same_transposition(
            geometry["reco_transposition"], getattr(handler, "reco_transposition", None)
        )
    )
    if not matching_acquisition:
        data = data.drop_vars([name for name in ("x", "y", "z") if name in data.coords])
        data = data.assign_coords(
            {
                dim: (dim, np.arange(data.sizes[dim]), {"unit": "index"})
                for dim in _SPATIAL_DIMS
            }
        )
        # Acquisition labels cannot describe another reconstruction's geometry.
        for name in (
            "read_direction",
            "phase_direction",
            "slice_direction",
            "slice_spacing_mm",
            "slice_thickness_mm",
        ):
            data.attrs.pop(name, None)
        orientation_matches = np.array_equal(
            geometry["visucore_orientation"],
            getattr(handler, "visucore_orientation", None),
        ) and _same_transposition(
            geometry["reco_transposition"], getattr(handler, "reco_transposition", None)
        )
        if not orientation_matches:
            for name in ("slice_orientation", "read_orientation"):
                if name in data.attrs:
                    data.attrs[f"acquisition_{name}"] = data.attrs.pop(name)
    data = assign_xyz_coordinates(data, scan=geometry)
    if recon_num is not None:
        data.attrs.update(
            {name: value for name, value in geometry.items() if name in _VISU_FIELDS}
        )
        data.attrs["bruker_visu_parameter_source"] = f"pdata/{recon_num}/visu_pars"
        data.attrs["n_slices"] = data.sizes["slice_idx"]
        thickness = geometry["visucore_thickness"]
        if thickness is not None and np.size(thickness) == 1:
            data.attrs["slice_thickness_mm"] = float(np.asarray(thickness).item())
    shape = tuple(data.sizes[dim] for dim in _SPATIAL_DIMS)
    packages = np.atleast_1d(getattr(handler, "slices_per_pack", [shape[2]]))
    if packages.size > 1:
        return data

    physical_slice = None
    if matching_acquisition:
        try:
            axes = get_rps_axes(handler)
        except (AttributeError, ValueError, TypeError):
            return data
        if any(len(axis) != size for axis, size in zip(axes, shape, strict=True)):
            return data
        physical_slice = _physical_slice_axis(data)
        if physical_slice is not None:
            axes = (*axes[:2], physical_slice[0])
            data.attrs["slice_direction"] = physical_slice[1]
    else:
        if not all(name in data.coords for name in ("x", "y", "z")):
            return data
        xyz = np.stack([data[name].values for name in ("x", "y", "z")], axis=-1)
        orientation = np.asarray(geometry["visucore_orientation"]).reshape(-1, 3, 3)
        # A single RPS frame cannot represent changing orientations.
        if not np.allclose(orientation, orientation[0], atol=1e-6, rtol=0):
            return data
        transposition = np.asarray(geometry["reco_transposition"]).reshape(-1)
        along_rows = transposition.size > 0 and transposition[0] == 1
        directions = orientation[0, [1, 0, 2] if along_rows else [0, 1, 2]].copy()
        for name, direction in zip(
            ("read_direction", "phase_direction", "slice_direction"),
            directions,
            strict=True,
        ):
            data.attrs[name] = direction
        axes = []
        for axis, size in enumerate(shape):
            selection = [0, 0, 0]
            selection[axis] = slice(None)
            line = xyz[tuple(selection)]
            if size > 1:
                step = line[-1] - line[0]
                norm = np.linalg.norm(step)
                if norm == 0:
                    return data
                directions[axis] = step / norm
            axes.append(line @ directions[axis])

    dimensions_to_assign = _SPATIAL_DIMS
    if matching_acquisition and all(
        data[dim].attrs.get("unit") == "mm" for dim in _SPATIAL_DIMS
    ):
        if physical_slice is None or np.allclose(
            data["slice_idx"], physical_slice[0], atol=_SLICE_AXIS_ATOL_MM, rtol=0
        ):
            return data
        dimensions_to_assign = ("slice_idx",)
    grids = np.meshgrid(*axes, indexing="ij")
    coordinate_attrs = {dim: dict(data[dim].attrs) for dim in _SPATIAL_DIMS}
    data = data.assign_coords(
        {
            dim: (_SPATIAL_DIMS, grid, {**coordinate_attrs[dim], "unit": "mm"})
            for dim, grid in zip(_SPATIAL_DIMS, grids, strict=True)
            if dim in dimensions_to_assign
        }
    )
    for axis, (dim, values) in enumerate(zip(_SPATIAL_DIMS, axes, strict=True)):
        if dim not in dimensions_to_assign:
            continue
        spacing = abs(values[1] - values[0]) if len(values) > 1 else None
        if spacing is None:
            if matching_acquisition:
                resolution = getattr(handler, "resolution_mm", None)
                spacing = resolution[axis] if resolution is not None else None
            elif axis == 2:
                thickness = geometry["visucore_thickness"]
                spacing = (
                    float(np.asarray(thickness).item())
                    if np.size(thickness) == 1 and thickness is not None
                    else None
                )
            else:
                source_axis = 1 - axis if along_rows else axis
                spacing = (
                    np.asarray(geometry["visucore_extent"]).ravel()[source_axis]
                    / np.asarray(geometry["visucore_size"]).ravel()[source_axis]
                )
        data[dim].attrs.update(matrix=shape[axis], axis_mm=values)
        if spacing is not None and np.isfinite(spacing):
            resolution = spacing
            if matching_acquisition:
                resolution = getattr(handler, "resolution_mm", [spacing] * 3)[axis]
            elif axis == 2 and geometry["visucore_thickness"] is not None:
                thickness = np.asarray(geometry["visucore_thickness"]).ravel()
                if thickness.size and np.all(thickness == thickness[0]):
                    resolution = float(thickness[0])
            data[dim].attrs.update(
                resolution_mm=resolution,
                extent_mm=[
                    float(values[0] - resolution / 2),
                    float(values[-1] + resolution / 2),
                ],
            )
            if axis == 2 and not matching_acquisition:
                data[dim].attrs["slice_spacing_mm"] = spacing
                data.attrs["slice_spacing_mm"] = spacing
                if geometry["visucore_thickness"] is not None:
                    data[dim].attrs["slice_thickness_mm"] = geometry[
                        "visucore_thickness"
                    ]
    return data
