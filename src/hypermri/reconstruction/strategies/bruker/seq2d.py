"""Bruker scanner-reconstructed 2dseq strategy."""

import re
import time
from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.reconstruction.layout import _assign_time_coordinate
from hypermri.reconstruction.precision import parse_data_precision
from hypermri.sequence_mapping import method_name_to_sequence_id
from hypermri.utils.system.logging import init_default_logger
from hypermri.utils.time.temporal import calc_inversion_time_multi_slice

from ...axes import _echo_frequency_axes, _label_cpmg_echo_train
from .._common import (
    _bruker_reco_provenance,
    _create_inversion_idx_coordinate,
    debug_log_recon_strategy,
)
from ..base import BaseRecoStrategy, _has_spectral_axis
from .geometry import _attach_image_geometry

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)

# Label of the 2dseq axis of each frame-group type of VisuFGOrderDesc.  The
# loops FG_MOVIE, FG_IRMODE, FG_CARDIAC_MOVIE and FG_FLOW take the label of the
# raw reconstruction for loops without a dimension of their own, except that
# FG_MOVIE holds the inversion times of FAIR methods (_movie_dim).  FG_ISA and
# FG_DTI hold parameter maps computed by ParaVision.  Other types keep the
# size-based labelling.
_FRAME_GROUP_DIMS = {
    "FG_SLICE": "slice_idx",
    "FG_ECHO": "echo_idx",
    "FG_CYCLE": "repetition_idx",
    "FG_DIFFUSION": "bvalue_idx",
    "FG_MOVIE": "acquisition_idx",
    "FG_IRMODE": "acquisition_idx",
    "FG_CARDIAC_MOVIE": "acquisition_idx",
    "FG_FLOW": "acquisition_idx",
    "FG_ISA": "map_idx",
    "FG_DTI": "map_idx",
}
# Length and type of one VisuFGOrderDesc entry (length, <type>, <name>, ...).
_FRAME_GROUP_PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*<([^>]*)>\s*,\s*<[^>]*>\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
# brukerapi combines the real and imaginary frames of this frame group into
# complex values, so the loaded 2dseq array has no axis for it.
_COMPLEX_FRAME_GROUP = "FG_COMPLEX"
# Order of the spatial dimensions in every reconstruction, as in the raw
# reconstruction and the legacy ``seq2d`` array.
_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _visu_parameter_text(value):
    """Rejoin the handler's tokenized Visu struct/string arrays losslessly."""
    if value is None:
        return ""
    return (
        value
        if isinstance(value, str)
        else " ".join(str(v) for v in np.asarray(value).ravel())
    )


def _frame_group_entries(visu_pars):
    """Parse length, type and dependency windows once per reconstruction path."""
    value = _visu_parameter_text(visu_pars.get("VisuFGOrderDesc", ""))
    entries = _FRAME_GROUP_PATTERN.findall(value)
    if value and not entries:
        raise ValueError(
            "Cannot parse 2dseq VisuFGOrderDesc; expected (length, <type>, <name>, start, count) records"
        )
    return [(int(n), kind, int(start), int(count)) for n, kind, start, count in entries]


def _frame_group_coordinates(visu_pars, shape, dims, frame_groups=None):
    """Read declared scalar timing/label windows from the selected Visu files.

    Descriptor start/count fields address VisuGroupDepVals; a dependency's
    offset addresses its parameter array. Timing is never assigned by length
    alone. Echo/TR coordinates use seconds and inversion times use ms.
    Non-scalar payloads, including cardiac trigger-time structs, are not
    interpreted as scalar times here. Samples and input metadata are unchanged.
    """

    descriptors = (
        _frame_group_entries(visu_pars) if frame_groups is None else frame_groups
    )
    dependencies = re.findall(
        r"\(\s*<([^>]*)>\s*,\s*(\d+)\s*\)",
        _visu_parameter_text(visu_pars.get("VisuGroupDepVals", "")),
    )
    scalar_times = {
        "VisuAcqEchoTime": ("echo_time_s", "s", 0.001),
        "VisuAcqRepetitionTime": ("repetition_time_s", "s", 0.001),
        "VisuAcqInversionTime": ("inversion_time_ms", "ms", 1.0),
    }
    frame_text = {
        "VisuFGElemComment": ("frame_label", "label"),
        "VisuFGElemId": ("frame_id", "id"),
    }
    coords = {}
    axis = int(np.count_nonzero(np.asarray(visu_pars.get("VisuCoreSize", [])) > 1))
    for length, group, start, count in descriptors:
        length, start, count = int(length), int(start), int(count)
        if length <= 1 or group == _COMPLEX_FRAME_GROUP:
            continue
        dim = dims[axis]
        if shape[axis] != length:
            raise ValueError(
                f"2dseq {group} declares {length} frames for axis {dim} of length {shape[axis]}"
            )
        axis += 1
        if dim != "inversion_idx" and group in {
            "FG_MOVIE",
            "FG_IRMODE",
            "FG_FLOW",
            "FG_CARDIAC_MOVIE",
            "FG_ISA",
            "FG_DTI",
        }:
            coords[dim] = (dim, np.arange(length), {"unit": "index"})
        if count and start + count > len(dependencies):
            raise ValueError(
                f"2dseq {group} dependency window ({start}, {count}) exceeds VisuGroupDepVals"
            )
        for parameter, offset in dependencies[start : start + count]:
            if parameter not in scalar_times and parameter not in frame_text:
                continue
            if parameter not in visu_pars:
                raise ValueError(
                    f"2dseq VisuGroupDepVals declares missing {parameter} for {group}"
                )
            offset = int(offset)
            if parameter in frame_text:
                values = np.asarray(
                    re.findall(r"<([^>]*)>", _visu_parameter_text(visu_pars[parameter]))
                )
                coordinate, unit = frame_text[parameter]
            else:
                coordinate, unit, scale = scalar_times[parameter]
                values = (
                    np.asarray(visu_pars[parameter], dtype=float).reshape(-1) * scale
                )
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(
                        f"2dseq {parameter} must contain finite nonnegative times in ms"
                    )
            if offset + length > values.size:
                raise ValueError(
                    f"2dseq {parameter} window starting at {offset} needs {length} values; found {values.size}"
                )
            values = values[offset : offset + length].copy()
            attrs = {"unit": unit, "source_parameter": parameter, "frame_group": group}
            shared = sum(name == parameter for name, _ in dependencies) > 1
            coords[f"{coordinate}_{dim}" if shared else coordinate] = (
                dim,
                values,
                attrs,
            )
            primary = (
                parameter == "VisuAcqEchoTime"
                and group == "FG_ECHO"
                or parameter == "VisuAcqInversionTime"
                and dim == "inversion_idx"
                or parameter == "VisuAcqRepetitionTime"
                and group in {"FG_MOVIE", "FG_CYCLE"}
                and dim != "inversion_idx"
                or parameter == "VisuFGElemComment"
                and group in {"FG_ISA", "FG_DTI"}
            )
            if primary:
                coords[dim] = (dim, values, attrs.copy())
    return coords


def _movie_dim(handler) -> str:
    """Return the label of the ``FG_MOVIE`` frame group of a Bruker scan.

    FAIR methods store their inversion times ``PVM_FairTIR_Arr`` in
    ``FG_MOVIE``, which is then labelled ``inversion_idx`` as in the raw
    reconstruction.  Every other ``FG_MOVIE`` loop, for example the
    repetition times of RAREVTR or the label and control frames of CASL, is
    labelled ``acquisition_idx``.

    Parameters
    ----------
    handler : BrukerHandler
        Handler of the scan; only ``inversion_time_ms`` is read.

    Returns
    -------
    str
        ``"inversion_idx"`` if the method lists more than one inversion
        time, otherwise ``"acquisition_idx"``.
    """
    if np.size(getattr(handler, "inversion_time_ms", None)) > 1:
        return "inversion_idx"
    return "acquisition_idx"


def _frame_group_axes(
    visu_pars: dict,
    shape: tuple[int, ...],
    echo_dim: str = "echo_idx",
    movie_dim: str = "acquisition_idx",
    *,
    include_unmapped: bool = False,
    frame_groups: list | None = None,
) -> dict[int, str | None]:
    """Return the axes of the labelled ``2dseq`` frame groups.

    ParaVision stores the frames of ``2dseq`` in the order of
    ``VisuFGOrderDesc``, fastest frame group first.  After
    :meth:`~hypermri.vendors.bruker.BrukerHandler.undo_reco_transposition`
    and ``squeeze``, the loaded array holds the frame dimensions of
    ``VisuCoreSize`` with more than one element, followed by one axis for
    every frame group with more than one element in that order.
    ``FG_COMPLEX`` has no axis because the loader combines it into complex
    values.  Axes after the frame groups are not described by
    ``VisuFGOrderDesc``.

    Parameters
    ----------
    visu_pars : dict
        Parameters of ``pdata/<recon_num>/visu_pars`` of the array, with
        ``VisuCoreSize`` and ``VisuFGOrderDesc`` as parsed by
        :class:`~hypermri.vendors.bruker.BrukerHandler`; ``VisuFGOrderDesc``
        has entries ``(length, <type>, <name>, start, count)``.
    shape : tuple of int
        Shape of the squeezed ``2dseq`` array.
    echo_dim : str
        Label of ``FG_ECHO``; ``"acquisition_idx"`` for MGE, whose echoes the
        reconstruction stores along the acquisition axis.
    movie_dim : str
        Label of ``FG_MOVIE``, from :func:`_movie_dim`;
        ``"inversion_idx"`` for the inversion times of FAIR methods.
    include_unmapped : bool
        Include unrecognized frame groups with a None label so the strict
        path can reject an unverified assignment. Default False preserves
        the fallback's size-based labels for unknown groups.
    frame_groups : list of tuple, optional
        Already parsed descriptor entries, shared with coordinate preparation
        to avoid parsing the same records again.

    Returns
    -------
    dict of int to str or None
        Axis of each frame group with more than one element and a type in
        ``_FRAME_GROUP_DIMS``, mapped to its label: ``slice_idx``,
        ``echo_idx``, ``repetition_idx``, ``bvalue_idx``, ``inversion_idx``,
        ``acquisition_idx`` for the other loops, or ``map_idx`` for parameter
        maps. Unknown groups map to None only with ``include_unmapped=True``.
        Empty when ``VisuFGOrderDesc`` is missing.

    Examples
    --------
    >>> visu_pars = {
    ...     "VisuCoreSize": [16, 16],
    ...     "VisuFGOrderDesc": "(2, <FG_IRMODE>, <>, 0, 0) (3, <FG_MOVIE>, <>, 0, 0)",
    ... }
    >>> _frame_group_axes(visu_pars, (16, 16, 2, 3), movie_dim="inversion_idx")
    {2: 'acquisition_idx', 3: 'inversion_idx'}

    Raises
    ------
    ValueError
        If two such frame groups have the same label, or if the axes after
        the frame dimensions do not have the lengths of the frame groups.
    """
    frame_group_order = visu_pars.get("VisuFGOrderDesc")
    if frame_group_order is None:
        return {}
    entries = _frame_group_entries(visu_pars) if frame_groups is None else frame_groups
    frame_groups = [
        (length, group_type)
        for length, group_type, _, _ in entries
        if length > 1 and group_type != _COMPLEX_FRAME_GROUP
    ]
    core_size = np.atleast_1d(visu_pars.get("VisuCoreSize", []))
    first_axis = int(np.count_nonzero(core_size.astype(int) > 1))
    stored_lengths = tuple(shape[first_axis : first_axis + len(frame_groups)])
    if stored_lengths != tuple(length for length, _ in frame_groups):
        raise ValueError(
            f"The 2dseq array of shape {tuple(shape)} does not hold the frame "
            f"groups of VisuFGOrderDesc {frame_group_order} after its "
            f"{first_axis} frame dimensions (VisuCoreSize {core_size.tolist()}). "
            "The 2dseq axes cannot be labelled; use the raw reconstruction for "
            "this scan."
        )
    axes = {}
    for offset, (_, group_type) in enumerate(frame_groups):
        dim = _FRAME_GROUP_DIMS.get(group_type)
        if dim is None and not include_unmapped:
            continue
        if group_type == "FG_ECHO":
            dim = echo_dim
        elif group_type == "FG_MOVIE":
            dim = movie_dim
        if dim is not None and dim in axes.values():
            raise ValueError(
                f"VisuFGOrderDesc has more than one frame group labelled {dim}: "
                f"{frame_group_order}. The 2dseq axes cannot be labelled; use "
                "the raw reconstruction for this scan."
            )
        axes[first_axis + offset] = dim
    return axes


def _gradient_directions(handler):
    """Return the read, phase and slice rows of the gradient matrix.

    ``acquisition_matrix`` is the first frame of ``ACQ_grad_matrix`` from
    ``acqp``; its rows are the unit vectors of the read, phase and slice
    directions in the scanner frame. Every ParaVision ``acqp`` contains it,
    so its absence means an incomplete export or synthetic parameters.

    Raises
    ------
    ValueError
        If the handler has no ``acquisition_matrix``. The message names the
        scan and ``ACQ_grad_matrix``.
    """
    gradient_matrix = getattr(handler, "acquisition_matrix", None)
    if gradient_matrix is None:
        raise ValueError(
            f"Scan {getattr(handler, 'path', 'unknown')}: acqp has no "
            "ACQ_grad_matrix, so the read, phase and slice directions of the "
            "scanner images are unknown. Every ParaVision acqp contains this "
            "parameter; export the scan again with its complete acqp file."
        )
    return gradient_matrix


def _read_phase_slice_order(dims) -> list[str]:
    """Return ``dims`` with ``read_idx``, ``phase_idx`` and ``slice_idx`` in this order.

    The spatial dimensions keep the positions they have in ``dims`` but take
    the order read, phase, slice of the raw reconstruction; all other
    dimensions keep their places.  ``dims`` is not modified.
    """
    spatial = iter([dim for dim in _SPATIAL_DIMS if dim in dims])
    return [next(spatial) if dim in _SPATIAL_DIMS else dim for dim in dims]


def _seq2d_expected_sizes(
    b_values,
    is_fieldmap_sequence,
    is_mge_sequence,
    n_echoes,
    n_inversion,
    n_receivers,
    n_repetitions,
    n_waterfat,
    phase_size,
    read_size,
    sampling_time_axis_s,
    slice_size,
):
    """Plan read/phase/slice and acquisition-loop sizes before squeezing.

    Spatial sizes come from the RPS grid. Echo, waterfat, inversion, diffusion
    and acquisition dimensions retain the current sequence-specific counts.
    No data or coordinates are changed.
    """
    # Map all potential dimensions to their expected sizes
    full_shape_map = {
        "read_idx": read_size,
        "phase_idx": phase_size,
        "slice_idx": slice_size,
        "repetition_idx": n_repetitions,
        "channel_idx": n_receivers,
        "acquisition_idx": (
            n_echoes
            if is_mge_sequence
            else (len(sampling_time_axis_s) if sampling_time_axis_s is not None else 0)
        ),
        "echo_idx": (
            0 if (is_mge_sequence or is_fieldmap_sequence) else n_echoes
        ),  # For MGE, don't use echo_idx
        "bvalue_idx": np.size(b_values),
        "inversion_idx": n_inversion,
        "waterfat_idx": n_waterfat,
    }

    return full_shape_map


def _seq2d_input_dimensions(full_shape_map):
    """Determine input labels from unsqueezed sizes in scanner frame order.

    Spatial axes lead read/phase/slice; only sizes greater than one survive
    squeezing. Returns the full convention and the present dimension names.
    """
    # undo_reco_transposition has put read before phase.
    spatial_dims_input = ("read_idx", "phase_idx", "slice_idx")

    # Define the full expected input order by combining spatial and non-spatial dimensions
    expected_input_order_unsqueezed = list(
        spatial_dims_input
    ) + [  # Convert tuple to list to allow concatenation
        "waterfat_idx",
        "inversion_idx",
        "echo_idx",
        "bvalue_idx",
        "channel_idx",
        "acquisition_idx",
        "repetition_idx",
    ]

    # Determine the actual dimensions present in seq2d by filtering the expected input order
    # Only include dimensions whose expected size is greater than 1
    actual_dims_order = [
        dim for dim in expected_input_order_unsqueezed if full_shape_map.get(dim, 0) > 1
    ]

    return expected_input_order_unsqueezed, actual_dims_order


def _expand_seq2d_dimensions(
    actual_dims_order, full_shape_map, seq2d, slice_orientation
):
    """Expand singleton dimensions and transpose scanner samples into target order.

    seq2d follows actual_dims_order; full_shape_map owns unsqueezed sizes.
    Output leads read/phase/slice and retains present nonspatial axes and dtype.
    No interpolation or Fourier transform occurs; invalid labels raise ValueError.
    """
    # --- Define the output order, independent of the input order ---
    if slice_orientation not in ("axial", "sagittal", "coronal"):
        raise ValueError(
            f"Unknown slice_orientation for determining target order: {slice_orientation}"
        )
    target_dims_potential = list(_SPATIAL_DIMS) + [
        "waterfat_idx",
        # Keep desired non-spatial order here
        "inversion_idx",
        "echo_idx",
        "bvalue_idx",
        "channel_idx",
        "acquisition_idx",
        "repetition_idx",
    ]
    # Filter target_dims based on available dimensions (size >= 1)
    target_dims = [
        dim for dim in target_dims_potential if full_shape_map.get(dim, 0) >= 1
    ]

    # --- Reshape/Transpose using xarray ---

    # 1. Create a temporary DataArray with the squeezed data and its
    # *correctly ordered* dimensions
    try:
        # The dims provided here MUST match the order of axes in seq2d
        temp_da = xr.DataArray(seq2d, dims=actual_dims_order, name="mri_data")
    except ValueError as e:
        print(f"Error creating temp_da: {e}")
        print(f"seq2d shape: {seq2d.shape}")
        print(f"actual_dims_order: {actual_dims_order}")
        raise

    # 2. Identify missing dimensions (present in target_dims but not in
    # actual_dims_order)
    missing_dims = [d for d in target_dims if d not in temp_da.dims]

    # 3. Expand the temporary DataArray to include the missing dimensions
    # (size 1)
    if missing_dims:
        # Ensure we only try to expand dimensions that should actually
        # exist (size >= 1)
        dims_to_expand = {d: 1 for d in missing_dims if full_shape_map.get(d, 0) >= 1}
        if dims_to_expand:
            expanded_da = temp_da.expand_dims(dim=dims_to_expand)
        else:
            expanded_da = temp_da

    else:
        expanded_da = temp_da

    # 4. Transpose the expanded DataArray to match the final target dimension order
    # Ensure all target_dims actually exist in the expanded_da before
    # transposing
    final_transpose_dims = [d for d in target_dims if d in expanded_da.dims]
    final_da = expanded_da.transpose(*final_transpose_dims)

    return final_da


def _seq2d_coordinates(
    b_values,
    effective_te_values,
    final_da,
    is_mge_sequence,
    rps_grid_mm,
    sampling_time_axis_s,
    scan_id,
    slice_size,
):
    """Build RPS and acquisition-loop coordinates for the final seq2d shape.

    RPS grids are read/phase/slice in mm, acquisition times are seconds and
    diffusion b-values are s/mm^2. Index axes and fat/water labels retain their
    existing fallbacks. Coordinate creation does not modify sample values.
    """
    coords_dict = {}
    shape_dict = dict(
        zip(final_da.dims, final_da.shape, strict=False)
    )  # Map dim names to sizes in the *final* array

    # Spatial coordinates (using sizes derived earlier)
    if "read_idx" in shape_dict:
        # Use read_size determined earlier for the coordinate vector length
        coords_dict["read_idx"] = (
            ["read_idx", "phase_idx", "slice_idx"],
            rps_grid_mm[0],
            {"unit": "mm"},
        )
    if "phase_idx" in shape_dict:
        # Use phase_size determined earlier

        coords_dict["phase_idx"] = (
            ["read_idx", "phase_idx", "slice_idx"],
            rps_grid_mm[1],
            {"unit": "mm"},
        )
    if "slice_idx" in shape_dict:
        # Use slice_size determined earlier
        np.linspace(-slice_size / 2, slice_size / 2, slice_size)
        coords_dict["slice_idx"] = (
            ["read_idx", "phase_idx", "slice_idx"],
            rps_grid_mm[2],
            {"unit": "mm"},
        )

    # Add water/fat coordinates if present
    if "waterfat_idx" in shape_dict:
        coords_dict["waterfat_idx"] = (
            "waterfat_idx",
            ["Water", "Fat"]
            if shape_dict["waterfat_idx"] == 2
            else np.arange(shape_dict["waterfat_idx"]),
            {"unit": "type" if shape_dict["waterfat_idx"] == 2 else "index"},
        )

    # Non-spatial coordinates (adjusting for potential squeezing reflected
    # in final_da shape)
    if "repetition_idx" in shape_dict:
        # Always use indices for repetition_idx to maintain backward compatibility
        rep_size = shape_dict["repetition_idx"]
        coords_dict["repetition_idx"] = (
            "repetition_idx",
            np.arange(rep_size),
            {"unit": "index"},
        )
    if "channel_idx" in shape_dict:
        # Use the actual size from the final array
        coords_dict["channel_idx"] = (
            "channel_idx",
            np.arange(shape_dict["channel_idx"]),
            {"unit": "index"},
        )

    if "acquisition_idx" in shape_dict:
        acq_size = shape_dict["acquisition_idx"]  # Actual size in final_da
        acq_unit = "index"

        # For MGE sequences, use EffectiveTE values as coordinates
        if is_mge_sequence and effective_te_values is not None:
            effective_te_values = np.atleast_1d(effective_te_values)
            if len(effective_te_values) >= acq_size:
                # EffectiveTE is typically in milliseconds, convert to seconds
                acq_coords = np.array(effective_te_values[:acq_size]) / 1000.0
                acq_unit = "s"
                logger.info(
                    f"Using EffectiveTE for acquisition_idx coordinates: {acq_coords}"
                )
            else:
                logger.warning(
                    f"EffectiveTE has {len(effective_te_values)} values but need {acq_size}, using indices"
                )
                acq_coords = np.arange(acq_size)
        elif (
            acq_size > 0
            and sampling_time_axis_s is not None
            and len(sampling_time_axis_s) >= acq_size
        ):
            # Select the correct number of time points
            acq_coords = sampling_time_axis_s[:acq_size]
            acq_unit = "s"
        elif acq_size > 0:  # Fallback if time axis is unexpectedly short
            acq_coords = np.arange(acq_size)
        else:
            acq_coords = np.array([])
        if acq_coords.size > 0:
            coords_dict["acquisition_idx"] = (
                "acquisition_idx",
                acq_coords,
                {"unit": acq_unit},
            )

    if "echo_idx" in shape_dict:
        # Use the actual size from the final array
        coords_dict["echo_idx"] = (
            "echo_idx",
            np.arange(shape_dict["echo_idx"]),
            {"unit": "index"},
        )

    if "bvalue_idx" in shape_dict:
        bval_size = shape_dict["bvalue_idx"]  # Actual size in final_da
        bval_unit = "index"
        if bval_size > 0 and np.size(b_values) >= bval_size:
            # Select the correct number of b-values
            bval_coords = np.asarray(b_values).reshape(-1)[:bval_size]
            bval_unit = "s/mm2"
            # Ensure it's an array even if scalar after slicing
            if np.isscalar(bval_coords):
                bval_coords = np.array([bval_coords])
        elif bval_size > 0:  # Fallback if b_values array is unexpectedly short
            bval_coords = np.arange(bval_size)
        else:
            bval_coords = np.array([])
        if bval_coords.size > 0:
            coords_dict["bvalue_idx"] = (
                "bvalue_idx",
                bval_coords,
                {"unit": bval_unit},
            )

    return coords_dict, shape_dict


def _assign_seq2d_coordinates(coords_dict, final_da):
    """Assign coordinate tuples compatible with the present seq2d dimensions.

    Each 1D coordinate must match its axis length; multidimensional grids
    must use existing dimensions. A wrong dimension-coordinate length raises
    ValueError instead of silently discarding physical labels. Optional grids
    over missing dimensions are omitted. Sample values and dtype are unchanged.
    """
    # Ensure coordinates match the dimensions *present* in final_da
    # before assigning
    coords_to_assign = {}
    for name, coord_data in coords_dict.items():
        # Check if it's a dimension coordinate (name matches a dim)
        if name in final_da.dims:
            # Check if the coordinate length matches the dimension size
            if isinstance(coord_data, tuple):  # Standard (dims, data, attrs) format
                coord_values = coord_data[1]
                coord_dims = coord_data[0]
                if isinstance(coord_dims, str):  # 1D coord
                    if len(coord_values) == final_da.sizes[name]:
                        coords_to_assign[name] = coord_data
                    else:
                        raise ValueError(
                            f"2dseq coordinate {name} has {len(coord_values)} values "
                            f"for an axis of length {final_da.sizes[name]}"
                        )
                else:  # Multi-dimensional coord (like spatial grids)
                    # Basic check if specified dims exist, shape check
                    # might be more complex
                    if all(d in final_da.dims for d in coord_dims):
                        coords_to_assign[name] = coord_data
                    else:
                        pass

            # Simple case: data array only (assume dim name matches)
            else:
                if len(coord_data) == final_da.sizes[name]:
                    coords_to_assign[name] = coord_data
                else:
                    raise ValueError(
                        f"2dseq coordinate {name} has {len(coord_data)} values "
                        f"for an axis of length {final_da.sizes[name]}"
                    )

        # Check if it's a non-dimension coordinate
        elif isinstance(coord_data, tuple) and isinstance(
            coord_data[0], (str, tuple, list)
        ):
            coord_dims = (
                (coord_data[0],)
                if isinstance(coord_data[0], str)
                else tuple(coord_data[0])
            )
            # Check if all dimensions it depends on exist in the final
            # DataArray
            if all(dim in final_da.dims for dim in coord_dims):
                # Optional: Add shape check against final_da.sizes for
                # those dims
                expected_shape = tuple(final_da.sizes[d] for d in coord_dims)
                actual_shape = np.shape(
                    coord_data[1]
                )  # Shape of the coordinate data array
                if actual_shape == expected_shape:
                    coords_to_assign[name] = coord_data
                else:
                    pass
            else:
                pass
        else:
            pass

    final_da = final_da.assign_coords(coords_to_assign)

    return final_da, coords_to_assign


def _attach_seq2d_frame_time(data, frame_time_s):
    """Delegate seconds-based labels to the shared reconstruction time owner."""
    return _assign_time_coordinate(data, "time", "repetition_idx", frame_time_s)


def _attach_seq2d_acquisition_time(data, sampling_time_axis_s):
    """Delegate seconds-based labels to the shared reconstruction time owner."""
    return _assign_time_coordinate(
        data, "acquisition_time_s", "acquisition_idx", sampling_time_axis_s
    )


def _spectra_in_ascending_frequency(data):
    """Return ``2dseq`` spectra with frequency ascending along ``acquisition_idx``.

    ParaVision stores the spectra of a spectroscopic ``2dseq`` with frequency
    descending: sample ``j`` holds the frequency that the centred FFT of the
    raw reconstruction puts at sample ``(N - j) mod N``.  This mirrors the
    axis about its centre sample ``N // 2`` so that the samples match the
    ascending ``freq_axis_hz`` and ``freq_axis_ppm`` of the handler.  For a
    local Bruker CSI scan (512 spectral points, ``MAGNITUDE_IMAGE``) the
    summed ``2dseq`` spectrum then correlates with the magnitude of the raw
    reconstruction at 1.000, and pyruvate and lactate lie at 171 and 183 ppm
    instead of 139 and 127 ppm.

    Parameters
    ----------
    data : xarray.DataArray
        Scanner spectra with the dimension ``acquisition_idx`` of length N.

    Returns
    -------
    xarray.DataArray
        New array with the same dimensions, coordinates, attributes and dtype;
        only the sample order along ``acquisition_idx`` changes.  The input
        is not modified.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> spectra = xr.DataArray(np.arange(4), dims=["acquisition_idx"])
    >>> _spectra_in_ascending_frequency(spectra).values
    array([0, 3, 2, 1])
    """
    axis = data.get_axis_num("acquisition_idx")
    size = data.sizes["acquisition_idx"]
    order = (-np.arange(size)) % size
    return data.copy(data=np.take(data.values, order, axis=axis))


def _attach_seq2d_axis_geometry(
    data,
    frame_time_s,
    has_rps_positions,
    matrix,
    n_repetitions,
    resolution_mm,
    rps_axes_mm,
    rps_extent_mm,
):
    """Attach read/phase/slice sizes and mm geometry plus repetition seconds.

    Geometry follows the RPS axis order. This stage changes only axis attributes;
    sample values, coordinate values and storage dtype are preserved.
    """
    data["read_idx"].attrs["matrix"] = matrix[0]
    data["read_idx"].attrs["resolution_mm"] = resolution_mm[0]
    if has_rps_positions:
        data["read_idx"].attrs["extent_mm"] = rps_extent_mm[0:2]
        data["read_idx"].attrs["axis_mm"] = rps_axes_mm[0]

    data["phase_idx"].attrs["matrix"] = matrix[1]
    data["phase_idx"].attrs["resolution_mm"] = resolution_mm[1]
    if has_rps_positions:
        data["phase_idx"].attrs["extent_mm"] = rps_extent_mm[2:4]
        data["phase_idx"].attrs["axis_mm"] = rps_axes_mm[1]

    data["slice_idx"].attrs["matrix"] = matrix[2]
    data["slice_idx"].attrs["resolution_mm"] = resolution_mm[2]
    if has_rps_positions:
        data["slice_idx"].attrs["extent_mm"] = rps_extent_mm[4:6]
        data["slice_idx"].attrs["axis_mm"] = rps_axes_mm[2]

    data["repetition_idx"].attrs["matrix"] = n_repetitions
    # Keep axis_s for backward compatibility, but note that the coordinate values
    # themselves now contain the time values in seconds (if available)
    if frame_time_s is not None and frame_time_s is not None:
        data["repetition_idx"].attrs["axis_s"] = frame_time_s
    return data


class Brukerseq2dRecoStrategy(BaseRecoStrategy):
    """
    Reconstruction strategy for Bruker seq2d data.

    Parameters
    ----------
    handler: VendorHandler
        The handler object containing the MRI data and metadata.
    reco_options: dict
        Reconstruction options, such as data precision and whether to skip reconstruction.

    Returns
    -------
    xr.DataArray
        The reconstructed data as an xarray DataArray.

    """

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "data_precision",
            "append_parameters",
            "recon_num",
            "set_reco_file",
        }
    )

    @staticmethod
    def _numeric_values(value) -> list[float]:
        """Return numeric values from Bruker scalar, array, or bracket-string fields."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip().strip("[]").replace(",", " ")
            if not text:
                return []
            values = np.fromstring(text, sep=" ")
        else:
            try:
                values = np.asarray(value).astype(float).ravel()
            except (TypeError, ValueError):
                return []
        return [float(v) for v in values if np.isfinite(v)]

    @classmethod
    def _int_values(cls, value) -> list[int]:
        return [int(round(v)) for v in cls._numeric_values(value)]

    @classmethod
    def _first_int(cls, value, default: int = 1) -> int:
        values = cls._int_values(value)
        return values[0] if values else default

    @classmethod
    def _method_ints(cls, handler, key: str) -> list[int]:
        method = getattr(handler, "method", None) or {}
        return cls._int_values(method.get(key))

    @classmethod
    def _method_first_int(cls, handler, key: str, default: int = 1) -> int:
        values = cls._method_ints(handler, key)
        return values[0] if values else default

    @classmethod
    def _reco_first_int(cls, reco: dict, key: str, default: int = 1) -> int:
        return cls._first_int(reco.get(key), default)

    @staticmethod
    def _is_spectral_seq2d(handler, data) -> bool:
        """Return True when Bruker 2dseq data stores spectra along acquisition_idx."""
        if "acquisition_idx" not in getattr(data, "dims", ()):
            return False
        if getattr(handler, "is_spectroscopic", False):
            return True

        method = getattr(handler, "method", None) or {}
        labels = (
            method.get("Method"),
            getattr(handler, "type", None),
            getattr(handler, "sequence_name", None),
        )
        spectral_labels = ("csi", "epsi")
        if any(
            any(name in str(label).lower() for name in spectral_labels)
            for label in labels
            if label is not None
        ):
            return True

        spec_matrix = method.get("PVM_SpecMatrix", None)
        if spec_matrix is None:
            spec_matrix = method.get("SpecSize", None)
        try:
            spec_size = int(np.atleast_1d(spec_matrix)[0])
        except (TypeError, ValueError, IndexError):
            return False
        return spec_size > 1 and spec_size == data.sizes["acquisition_idx"]

    @classmethod
    def _fallback_shape_info(cls, handler, reco, seq2d_shape, canonical_sequence):
        matrix = cls._int_values(getattr(handler, "matrix", None))
        read_size = matrix[0] if len(matrix) > 0 and matrix[0] > 0 else seq2d_shape[0]
        phase_size = (
            matrix[1]
            if len(matrix) > 1 and matrix[1] > 0
            else (seq2d_shape[1] if len(seq2d_shape) > 1 else 1)
        )

        pack_slices = cls._method_ints(handler, "PVM_SPackArrNSlices")
        slice_size = sum(pack_slices) if pack_slices else 1
        if slice_size <= 1:
            slice_size = cls._first_int(getattr(handler, "n_slices", None), 1)
        if slice_size <= 1 and len(matrix) > 2:
            slice_size = matrix[2]

        n_echoes = cls._first_int(getattr(handler, "n_echoes", None), 1)
        effective_te = cls._numeric_values(
            (getattr(handler, "method", None) or {}).get("EffectiveTE")
        )
        if canonical_sequence in {"mge", "msme"} and effective_te:
            n_echoes = len(effective_te)

        movie_frames = cls._method_first_int(handler, "PVM_NMovieFrames", 1)
        averages = cls._method_first_int(handler, "PVM_NAverages", 1)
        reco_objects = cls._reco_first_int(reco, "RecoObjectsPerRepetition", 1)
        object_axis = 1
        if slice_size > 0 and reco_objects % slice_size == 0:
            object_axis = max(1, reco_objects // slice_size)
        elif reco_objects > 1 and reco_objects in seq2d_shape:
            object_axis = reco_objects

        sampling_axis = 1
        try:
            sampling = getattr(handler, "sampling_time_axis_s", None)
            sampling_axis = len(sampling) if sampling is not None else 1
        except (TypeError, ValueError):
            sampling_axis = 1

        acquisition_size = max(movie_frames, object_axis, sampling_axis)
        if canonical_sequence == "mge":
            acquisition_size = max(acquisition_size, n_echoes)

        b_values = getattr(handler, "b_values", 1)
        bvalue_size = int(np.size(b_values)) if b_values is not None else 1
        inversion_size = int(np.size(getattr(handler, "inversion_time_ms", 1)))
        waterfat_size = (
            2 if getattr(handler, "fat_water_separation", "Off") == "On" else 1
        )

        return {
            "read_idx": int(read_size),
            "phase_idx": int(phase_size),
            "slice_idx": int(slice_size),
            "waterfat_idx": int(waterfat_size),
            "inversion_idx": int(max(1, inversion_size)),
            "echo_idx": int(max(1, n_echoes)),
            "bvalue_idx": int(max(1, bvalue_size)),
            "acquisition_idx": int(max(1, acquisition_size)),
            "average_idx": int(max(1, averages)),
            "repetition_idx": int(
                max(1, cls._first_int(getattr(handler, "n_repetitions", None), 1))
            ),
            "channel_idx": int(
                max(1, cls._reco_first_int(reco, "RecoNumOutputChan", 1))
            ),
        }

    @staticmethod
    def _fallback_expected_order(canonical_sequence: str | None, method_name: str):
        read_phase = ["read_idx", "phase_idx"]
        method_lower = str(method_name or "").lower()
        if canonical_sequence == "mge" and "epsi" in method_lower:
            return ["read_idx", "acquisition_idx", "phase_idx", "slice_idx"]
        if canonical_sequence == "mge":
            return read_phase + ["acquisition_idx", "slice_idx"]
        if canonical_sequence == "msme":
            return read_phase + ["echo_idx", "slice_idx"]
        if canonical_sequence in {"flash", "bssfp", "flowmap"}:
            return read_phase + ["acquisition_idx", "slice_idx"]
        if canonical_sequence == "epi":
            return read_phase + [
                "slice_idx",
                "inversion_idx",
                "acquisition_idx",
                "average_idx",
            ]
        return read_phase + [
            "slice_idx",
            "waterfat_idx",
            "inversion_idx",
            "echo_idx",
            "bvalue_idx",
            "acquisition_idx",
            "average_idx",
            "repetition_idx",
            "channel_idx",
        ]

    @classmethod
    def _infer_fallback_dims(
        cls,
        handler,
        reco,
        visu_pars,
        seq2d,
        canonical_sequence,
        method_name,
        frame_groups=None,
    ):
        shape = tuple(int(s) for s in seq2d.shape)
        sizes = cls._fallback_shape_info(handler, reco, shape, canonical_sequence)
        if not getattr(handler, "is_spectroscopic", False):
            selected_size = cls._int_values(visu_pars.get("VisuCoreSize"))
            if len(selected_size) >= 2:
                swap = cls._first_int(reco.get("RECO_transposition"), 0) == 1
                sizes["read_idx"] = selected_size[1 if swap else 0]
                sizes["phase_idx"] = selected_size[0 if swap else 1]
        expected_order = cls._fallback_expected_order(canonical_sequence, method_name)
        tail_order = [
            dim
            for dim in (
                "slice_idx",
                "waterfat_idx",
                "inversion_idx",
                "echo_idx",
                "bvalue_idx",
                "acquisition_idx",
                "average_idx",
                "repetition_idx",
                "channel_idx",
            )
            if dim not in expected_order
        ]
        # VisuFGOrderDesc labels its frame-group axes; the other axes are
        # labelled by their size.
        frame_group_axes = _frame_group_axes(
            visu_pars,
            shape,
            echo_dim="acquisition_idx" if canonical_sequence == "mge" else "echo_idx",
            movie_dim=_movie_dim(handler),
            frame_groups=frame_groups,
        )
        search_order = expected_order + tail_order

        dims = []
        used = set(frame_group_axes.values())
        extra_count = 0
        for axis, axis_size in enumerate(shape):
            chosen = frame_group_axes.get(axis)
            if chosen is None:
                chosen = next(
                    (
                        dim
                        for dim in search_order
                        if dim not in used and sizes.get(dim, 1) == axis_size
                    ),
                    None,
                )
            if chosen is None:
                generic = "acquisition_idx"
                if generic not in used:
                    chosen = generic
                else:
                    extra_count += 1
                    chosen = f"extra_dim_{extra_count}_idx"
            dims.append(chosen)
            used.add(chosen)
        return dims, sizes

    @classmethod
    def _build_fallback_coords(cls, handler, dims, shape, sizes, canonical_sequence):
        coords = {}
        rps_grid = None
        try:
            rps_grid = getattr(handler, "rps_grid_mm", None)
        except AttributeError:
            rps_grid = None

        shape_by_dim = dict(zip(dims, shape, strict=False))
        can_use_grid = (
            rps_grid is not None
            and all(
                dim in shape_by_dim for dim in ("read_idx", "phase_idx", "slice_idx")
            )
            and all(
                np.shape(rps_grid[i])
                == (
                    shape_by_dim["read_idx"],
                    shape_by_dim["phase_idx"],
                    shape_by_dim["slice_idx"],
                )
                for i in range(3)
            )
        )

        for dim, size in zip(dims, shape, strict=False):
            if dim == "read_idx" and can_use_grid:
                coords[dim] = (
                    ["read_idx", "phase_idx", "slice_idx"],
                    rps_grid[0],
                    {"unit": "mm"},
                )
            elif dim == "phase_idx" and can_use_grid:
                coords[dim] = (
                    ["read_idx", "phase_idx", "slice_idx"],
                    rps_grid[1],
                    {"unit": "mm"},
                )
            elif dim == "slice_idx" and can_use_grid:
                coords[dim] = (
                    ["read_idx", "phase_idx", "slice_idx"],
                    rps_grid[2],
                    {"unit": "mm"},
                )
            elif dim == "waterfat_idx":
                labels = ["Water", "Fat"] if size == 2 else list(range(size))
                coords[dim] = (dim, labels, {"unit": "type"})
            elif dim == "acquisition_idx":
                method = getattr(handler, "method", None) or {}
                effective_te = cls._numeric_values(method.get("EffectiveTE"))
                if canonical_sequence == "mge" and len(effective_te) >= size:
                    values = np.asarray(effective_te[:size], dtype=float) / 1000.0
                    unit = "s"
                else:
                    try:
                        sampling = np.asarray(handler.sampling_time_axis_s)
                    except AttributeError:
                        sampling = np.asarray([])
                    if _has_spectral_axis(handler) and sampling.size >= size:
                        values = sampling[:size]
                        unit = "s"
                    else:
                        values = np.arange(size)
                        unit = "index"
                coords[dim] = (dim, values, {"unit": unit})
            elif dim == "echo_idx":
                method = getattr(handler, "method", None) or {}
                effective_te = cls._numeric_values(method.get("EffectiveTE"))
                if len(effective_te) >= size:
                    coords[dim] = (
                        dim,
                        np.asarray(effective_te[:size], dtype=float) / 1000.0,
                        {"unit": "s"},
                    )
                else:
                    coords[dim] = (dim, np.arange(size), {"unit": "index"})
            elif dim == "bvalue_idx":
                b_values = np.atleast_1d(getattr(handler, "b_values", []))
                has_values = b_values.size >= size
                values = b_values[:size] if has_values else np.arange(size)
                coords[dim] = (
                    dim,
                    values,
                    {"unit": "s/mm2" if has_values else "index"},
                )
            elif dim == "inversion_idx":
                values = np.atleast_1d(getattr(handler, "inversion_time_ms", []))
                has_values = values.size >= size
                values = values[:size] if has_values else np.arange(size)
                coords[dim] = (dim, values, {"unit": "ms" if has_values else "index"})
            else:
                coords[dim] = (dim, np.arange(size), {"unit": "index"})
        return coords

    def _fallback_seq2d_scanner_data(
        self,
        handler,
        seq2d,
        reco_options,
        append_parameters=True,
        reason="metadata mismatch",
        frame_groups=None,
    ):
        raw_sequence_name = getattr(handler, "sequence_name", None)
        method_name = None
        if hasattr(handler, "method"):
            method_name = (getattr(handler, "method", None) or {}).get("Method")
        canonical_sequence = method_name_to_sequence_id(raw_sequence_name)
        if canonical_sequence is None and method_name is not None:
            canonical_sequence = method_name_to_sequence_id(method_name)

        reco, visu_pars = handler.load_reconstruction_parameters(
            reco_options.get("recon_num", 1)
        )
        if frame_groups is None:
            frame_groups = _frame_group_entries(visu_pars)
        dims, sizes = self._infer_fallback_dims(
            handler,
            reco,
            visu_pars,
            seq2d,
            canonical_sequence,
            method_name,
            frame_groups,
        )
        coords = self._build_fallback_coords(
            handler, dims, seq2d.shape, sizes, canonical_sequence
        )
        coords.update(
            _frame_group_coordinates(visu_pars, seq2d.shape, dims, frame_groups)
        )
        if "read_idx" in dims and "phase_idx" in dims and "slice_idx" not in dims:
            seq2d = np.expand_dims(seq2d, axis=-1)
            dims.append("slice_idx")
            coords["slice_idx"] = ("slice_idx", [0], {"unit": "index"})
        data_array = self.set_scanner_data(
            data=seq2d,
            dims=dims,
            coords=coords,
            handler=handler,
            reco_options=reco_options,
        )
        # The raw reconstruction orders the spatial dimensions read, phase,
        # slice.
        data_array = data_array.transpose(*_read_phase_slice_order(data_array.dims))
        data_array.attrs["description"] = "MRI data loaded from Bruker 2dseq fallback"
        data_array.attrs["seq2d_fallback_reason"] = reason
        data_array.attrs["original_seq2d_shape"] = str(seq2d.shape)
        data_array.attrs["determined_actual_dims_order"] = str(dims)
        if append_parameters:
            data_array = self._append_parameters(array=data_array, handler=handler)
        return data_array

    def reconstruct(self, handler, **reco_options):
        """Load the scanner-reconstructed ``2dseq`` images of a Bruker scan.

        The ``2dseq`` frames are labelled from the method parameters after
        :meth:`~hypermri.vendors.bruker.BrukerHandler.undo_reco_transposition`
        has undone the ``RECO_transposition`` swap: read, phase, slice and the
        non-spatial dimensions with more than one element.  The axes of the
        frame groups of ``VisuFGOrderDesc`` in ``pdata/<recon_num>/visu_pars``
        must be labelled at the positions where they are stored: after the
        frame dimensions, fastest frame group first, counting every frame
        group with more than one element except ``FG_COMPLEX``.  Slices,
        echoes, cycles and diffusion weightings are labelled ``slice_idx``,
        ``echo_idx``, ``repetition_idx`` and ``bvalue_idx``; ``FG_MOVIE`` is
        labelled ``inversion_idx`` when it holds the inversion times of a
        FAIR or T1 EPI method; the other loops (``FG_MOVIE``, ``FG_IRMODE``,
        ``FG_CARDIAC_MOVIE``, ``FG_FLOW``) are labelled ``acquisition_idx`` as
        in the raw reconstruction; and the parameter maps of ``FG_ISA`` and
        ``FG_DTI`` are labelled ``map_idx`` with available frame labels. Other
        frame-group types are labelled by their size.  The transposition,
        frame groups and number of output channels are read from
        ``pdata/<recon_num>`` (see
        :meth:`~hypermri.vendors.bruker.BrukerHandler.load_reconstruction_parameters`),
        so that a parameter map in ``pdata/2`` is labelled from its own
        files.  When the method parameters do not explain the array, the
        strategy labels these axes from their frame groups and every other
        axis by its size in a sequence-specific search order, and records
        the reason in the ``seq2d_fallback_reason`` attribute.
        The strict path validates every non-spatial axis length. Declared
        ``VisuGroupDepVals`` windows supply scalar echo/TR times in seconds,
        inversion times in ms and parameter-map labels. ``VisuFGElemComment``
        supplies readable ``frame_label`` values and the existing ``map_idx``
        labels. Declared ``VisuFGElemId`` dependencies additionally supply
        machine identifiers as ``frame_id``; IDs are optional and never replace
        display labels. Undeclared loop times
        retain index units; readout sampling is not a loop clock. Compound
        cardiac timing records are not interpreted as scalar times.

        Parameters
        ----------
        handler : BrukerHandler
            Handler of the scan.  Arrays held by the handler are converted
            to ``data_precision``; the parameter files and ``2dseq`` are only
            read.
        **reco_options
            Supported options: ``no_reconstruction`` (bool, default False)
            returns None without reading data; ``data_precision`` (str or
            numpy floating type, default ``np.float32``) is the storage
            precision of the returned values; ``append_parameters`` (bool,
            default True) attaches the acquisition parameters as attributes;
            ``recon_num`` (int, default 1) selects ``pdata/<recon_num>``
            and its parameter files; ``set_reco_file`` (bool, default
            False) also replaces the handler's ``reco`` attribute with
            ``pdata/<recon_num>/reco`` for later readers, but the labelling
            does not depend on it.

        Returns
        -------
        xarray.DataArray or None
            Image intensities of ``2dseq`` after ``VisuCoreDataSlope`` and
            ``VisuCoreDataOffs`` scaling.  The spatial dimensions
            ``read_idx``, ``phase_idx`` and ``slice_idx`` are in this order
            for every slice orientation and ``RECO_transposition``, as in
            the raw reconstruction; the strict path puts them first,
            followed by the non-spatial dimensions such as ``echo_idx``,
            and the fallback keeps them at the positions of the stored
            axes.  None if no reconstruction
            was requested, or the scan has no usable ``2dseq`` payload or RPS
            grid. Reconstruction errors propagate. Verified layout fallbacks
            record their reason and report coordinate failures as warnings.
            ``bruker_reconstruction_number`` identifies the data folder;
            ``bruker_reco_parameter_source`` and available timestamp/object-count
            attributes snapshot active handler parameters independently of
            ``append_parameters``.
            Timing, label and ID auxiliary coordinates retain their source
            parameter and frame group. A parameter shared by several groups
            appends the dimension name to its auxiliary coordinate name.
            Uniform single-package slice labels project selected XYZ voxel
            centres onto increasing slice index, including when scalar and
            one-element RECO_transposition representations are equivalent.

        Raises
        ------
        ValueError
            If precision or scanner layout is invalid, including ambiguous
            frame-group labels or inconsistent dimension lengths. Precision is
            checked before any data are read.
        OSError
            If a scanner payload or reconstruction parameter file cannot be read.

        Notes
        -----
        Reconstruction is eager and deterministic. Scanner loading may populate
        acquired-file caches. This strategy preserves acquired sample counts,
        bandwidth and readout timing and stores no processed intermediates.
        ``set_reco_file=True`` explicitly changes the handler's active ``reco``
        parameters and ``reco_parameter_source`` after successful loading.

        Examples
        --------
        >>> images = strategy.reconstruct(handler, recon_num=2)  # doctest: +SKIP
        """
        logger.info(f"reconstructing data ... for scan {handler.expnum}")
        reco_options["data_precision"] = parse_data_precision(
            reco_options.get("data_precision", np.float32)
        )
        return self._reconstruct_internal(handler, **reco_options)

    def _load_scanner_samples(self, handler, recon_num, set_reco_file, DATA_PRECISION):
        """Read the selected 2dseq, undo storage transposition and squeeze axes.

        Returns the acquired scanner samples in read/phase/frame order, retaining
        requested storage dtype, or None for absent/empty data. Handler loading
        keeps its existing set_reco_file behavior; no numerical transforms occur.
        """
        data_load_start = time.time()
        logger.debug(f"seq2d reconstruction starting for scan {handler.expnum}")

        seq2d_raw = handler.load_2dseq_file(
            recon_num=recon_num,
            set_reco_file=set_reco_file,
            data_precision=DATA_PRECISION,
        )
        logger.debug(f"seq2d_raw type: {type(seq2d_raw)}")
        logger.debug(f"seq2d_raw shape: {getattr(seq2d_raw, 'shape', 'no shape')}")
        logger.debug(f"seq2d_raw size: {getattr(seq2d_raw, 'size', 'no size')}")

        if seq2d_raw is None or getattr(seq2d_raw, "size", 0) == 0:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: No 2dseq data available; skipping seq2d reconstruction."
            )
            return None

        seq2d = handler.undo_reco_transposition(
            seq2d_raw, recon_num=recon_num
        ).squeeze()
        logger.debug(f"seq2d after squeeze type: {type(seq2d)}")
        logger.debug(
            f"seq2d after squeeze shape: {getattr(seq2d, 'shape', 'no shape')}"
        )
        logger.debug(f"seq2d after squeeze size: {getattr(seq2d, 'size', 'no size')}")

        data_load_time = time.time() - data_load_start
        logger.debug(f"[TIMING] seq2d data loading took {data_load_time:.3f}s")

        # Check if seq2d data is available
        if seq2d.size == 0:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: Empty 2dseq data encountered; skipping seq2d reconstruction."
            )
            return None

        return seq2d

    def _read_scanner_axes(self, handler, recon_num, rps_grid_mm):
        """Read dimension counts and coordinate inputs of the selected reconstruction.

        Counts follow read/phase/slice geometry; sampling is in seconds, echo
        times in ms and b-values in s/mm^2. Returns layout and coordinate inputs
        or None for an invalid RPS grid. Inversion-time preparation is local;
        acquired sample metadata are not changed.
        """
        n_repetitions = handler.n_repetitions
        # The files of pdata/<recon_num> describe this 2dseq; handler.visu is
        # the scan-level visu_pars and handler.reco depends on set_reco_file.
        reco, visu_pars = handler.load_reconstruction_parameters(recon_num)
        n_receivers = reco.get("RecoNumOutputChan", 1)
        sampling_time_axis_s = handler.sampling_time_axis_s
        n_echoes = handler.n_echoes
        b_values = handler.b_values
        inversion_time_ms = handler.inversion_time_ms
        np.atleast_1d(handler.inversion_time_ms)  # Ensure it's an array

        n_inversion = np.size(inversion_time_ms)
        if (
            n_inversion > 1
            and getattr(handler, "inversion_time_source", None) != "IRTime"
        ):
            calc_inversion_time_multi_slice(scan=handler)

        # --- Determine expected full shape ---
        # !!! IMPORTANT: Logic to derive read/phase/slice sizes from rps_grid_mm
        # needs to be correct based on YOUR data structure.
        # This assumes rps_grid_mm[0] always provides the read/phase/slice grid
        # and its shape reflects those dimensions directly.
        try:
            # Assuming grid shapes are consistently (read, phase, slice) in
            # rps_grid_mm[0]
            read_size = rps_grid_mm[0].shape[0]
            phase_size = rps_grid_mm[0].shape[1]
            slice_size = rps_grid_mm[0].shape[2]

        except (IndexError, AttributeError) as e:
            logger.warning(
                f"{getattr(handler, 'id', 'unknown')}: Invalid rps_grid_mm structure ({e}); skipping seq2d reconstruction."
            )
            return None

        # Check for PVM_FwsOnOff to determine water/fat separation
        pvm_fws_on_off = handler.fat_water_separation
        if pvm_fws_on_off == "On":
            n_waterfat = 2  # Water and Fat
        else:
            n_waterfat = 1  # No water/fat separation

        # Detect canonical sequence via alias mapping to determine sequence-specific handling
        raw_sequence_name = getattr(handler, "sequence_name", None)
        canonical_sequence = method_name_to_sequence_id(raw_sequence_name)
        if canonical_sequence is None and hasattr(handler, "method"):
            canonical_sequence = method_name_to_sequence_id(
                handler.method.get("Method")
            )

        # Check if this is an MGE sequence - use EffectiveTE for acquisition_idx instead of echo_idx
        is_mge_sequence = canonical_sequence == "mge"
        effective_te_values = None
        if is_mge_sequence:
            method_params = getattr(handler, "method", None) or {}
            effective_te_values = method_params.get("EffectiveTE")
            if effective_te_values is None:
                logger.warning(
                    f"{handler.id}: Sequence resolved to 'mge' via alias but EffectiveTE is missing; "
                    "falling back to standard echo-based handling."
                )
                is_mge_sequence = False
            else:
                logger.info(
                    f"{handler.id}: Detected MGE sequence with EffectiveTE: {effective_te_values}"
                )

        # Check if this is an FieldMap sequence - do not use echo_idx or acquisition_idx
        is_fieldmap_sequence = canonical_sequence == "fieldmap"
        if is_fieldmap_sequence:
            method_params = getattr(handler, "method", None) or {}
            effective_te_values = method_params.get("EffectiveTE")
            if effective_te_values is None:
                logger.warning(
                    f"{handler.id}: Sequence resolved to 'mge' via alias but EffectiveTE is missing; "
                    "falling back to standard echo-based handling."
                )
                is_fieldmap_sequence = False
            else:
                logger.info(
                    f"{handler.id}: Detected MGE sequence with EffectiveTE: {effective_te_values}"
                )

        full_shape_map = _seq2d_expected_sizes(
            b_values=b_values,
            is_fieldmap_sequence=is_fieldmap_sequence,
            is_mge_sequence=is_mge_sequence,
            n_echoes=n_echoes,
            n_inversion=n_inversion,
            n_receivers=n_receivers,
            n_repetitions=n_repetitions,
            n_waterfat=n_waterfat,
            phase_size=phase_size,
            read_size=read_size,
            sampling_time_axis_s=sampling_time_axis_s,
            slice_size=slice_size,
        )

        return (
            full_shape_map,
            visu_pars,
            sampling_time_axis_s,
            b_values,
            is_mge_sequence,
            effective_te_values,
            slice_size,
        )

    def _build_strict_scanner_data(
        self,
        handler,
        seq2d,
        axes,
        rps_grid_mm,
        slice_orientation,
        reco_options,
        append_parameters,
    ):
        """Assemble strict scanner labels or use the existing frame-layout fallback.

        axes contains expected sizes, reconstruction parameters and sampling,
        diffusion and echo-coordinate inputs. seq2d retains acquired scanner
        dtype. The numerical expansion and coordinate stages have explicit
        inputs; existing inversion metadata and fallback behavior are retained.
        """
        (
            full_shape_map,
            visu_pars,
            sampling_time_axis_s,
            b_values,
            is_mge_sequence,
            effective_te_values,
            slice_size,
        ) = axes
        # --- Determine actual dimension order based on known expected input order and sizes ---

        # Define the expected input order *before* squeezing, based on orientation
        if slice_orientation is None:
            # If slice_orientation is None, we assume the input order is
            # already correct
            return None

        expected_input_order_unsqueezed, actual_dims_order = _seq2d_input_dimensions(
            full_shape_map=full_shape_map
        )
        # Validation check: Does the number of determined dimensions match the
        # input array's dimensions?
        if len(actual_dims_order) != seq2d.ndim:
            logger.debug(
                "Dimension mismatch in strict seq2d reconstruction; using fallback. "
                f"Input shape={seq2d.shape}, expected_dims={actual_dims_order}, "
                f"shape_map={full_shape_map}"
            )
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason="strict dimension mismatch",
            )
        frame_groups = _frame_group_entries(visu_pars)
        frame_group_axes = _frame_group_axes(
            visu_pars,
            seq2d.shape,
            echo_dim="acquisition_idx" if is_mge_sequence else "echo_idx",
            movie_dim=_movie_dim(handler),
            include_unmapped=True,
            frame_groups=frame_groups,
        )
        if any(dim is None for dim in frame_group_axes.values()):
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason="unmapped frame group in VisuFGOrderDesc",
                frame_groups=frame_groups,
            )
        if any(
            actual_dims_order[axis] != dim for axis, dim in frame_group_axes.items()
        ):
            logger.debug(
                "The strict seq2d labels differ from the frame groups of "
                f"VisuFGOrderDesc; using fallback. Labels={actual_dims_order}, "
                f"frame groups by axis={frame_group_axes}"
            )
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason="frame group order differs from VisuFGOrderDesc",
                frame_groups=frame_groups,
            )
        mismatches = [
            f"{dim}: stored {size}, expected {full_shape_map[dim]}"
            for dim, size in zip(actual_dims_order, seq2d.shape, strict=True)
            if dim not in _SPATIAL_DIMS and size != full_shape_map[dim]
        ]
        if mismatches:
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason="strict size mismatch: " + "; ".join(mismatches),
                frame_groups=frame_groups,
            )
        # --- End of dimension order determination ---

        final_da = _expand_seq2d_dimensions(
            actual_dims_order=actual_dims_order,
            full_shape_map=full_shape_map,
            seq2d=seq2d,
            slice_orientation=slice_orientation,
        )
        # --- Define and Assign Coordinates ---
        coords_dict, shape_dict = _seq2d_coordinates(
            b_values=b_values,
            effective_te_values=effective_te_values,
            final_da=final_da,
            is_mge_sequence=is_mge_sequence,
            rps_grid_mm=rps_grid_mm,
            sampling_time_axis_s=sampling_time_axis_s,
            scan_id=getattr(handler, "id", "unknown"),
            slice_size=slice_size,
        )
        visu_coords = _frame_group_coordinates(
            visu_pars, seq2d.shape, actual_dims_order, frame_groups
        )
        coords_dict.update(visu_coords)
        # FAIR effective times retain the established slice offsets, using
        # the selected reconstruction's declared base times when available.
        if "inversion_idx" in shape_dict:
            n_slices_final = shape_dict["slice_idx"]
            n_inversions_final = shape_dict["inversion_idx"]

            # Use helper function to create slice-dependent inversion coordinate
            coords_dict["inversion_idx"] = _create_inversion_idx_coordinate(
                handler,
                n_inversions_final,
                n_slices_final,
                shape_dict,
                inversion_times_ms=visu_coords.get("inversion_idx", (None, None))[1],
            )
            if "inversion_idx" in visu_coords:
                coords_dict["inversion_idx"][2]["source_parameter"] = (
                    "VisuAcqInversionTime"
                )

        # 5. Assign the coordinates to the final DataArray
        # --- Assign Coordinates ---
        if "inversion_idx" in coords_dict:
            required_dims_for_eff_ti = coords_dict["inversion_idx"][0]
            if isinstance(required_dims_for_eff_ti, str):
                required_dims_for_eff_ti = (required_dims_for_eff_ti,)
            if not all(dim in final_da.dims for dim in required_dims_for_eff_ti):
                del coords_dict["inversion_idx"]

        # Assign all prepared coordinates (both dimension and non-dimension)
        # xarray will handle matching them correctly.
        try:
            final_da, coords_to_assign = _assign_seq2d_coordinates(
                coords_dict=coords_dict, final_da=final_da
            )

        except ValueError as e:
            logger.warning(
                "Coordinate assignment failed in strict seq2d reconstruction; "
                f"using fallback. Error: {e}"
            )
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason=f"coordinate assignment failed: {e}",
                frame_groups=frame_groups,
            )

        # Add attributes if needed
        final_da.attrs["slice_orientation"] = slice_orientation
        final_da.attrs["description"] = "MRI data reshaped and transposed"
        final_da.attrs["original_seq2d_shape"] = str(seq2d.shape)
        final_da.attrs["determined_actual_dims_order"] = str(actual_dims_order)
        final_da.attrs["expected_input_order_convention"] = str(
            expected_input_order_unsqueezed
        )
        data_array = self.set_scanner_data(
            data=final_da.data,
            dims=final_da.dims,
            coords=coords_to_assign,
            handler=handler,
            reco_options=reco_options,
        )
        if append_parameters:
            data_array = self._append_parameters(array=data_array, handler=handler)

        return data_array

    def _reconstruct_internal(self, handler, **reco_options):
        """Load scanner samples and build labels through explicit geometry stages.

        The handler supplies lazy 2dseq samples, frame groups, orientation,
        matrix sizes, mm grids and temporal/spectral coordinates. Options
        contain normalized storage precision and reconstruction selection.
        Missing geometry retains the existing frame-group fallback; invalid
        strict dimensions raise ValueError. Returns a labelled DataArray or
        None under the existing no-reconstruction/missing-input policy.
        Scanner loading may populate handler caches; optional parameter
        attachment and calculation/storage conversions preserve current policy.
        """
        time.time()
        logger.info(
            f"[TIMING] Starting Brukerseq2dRecoStrategy reconstruction for scan {handler.expnum}"
        )
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[Brukerseq2dRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        # Define the potential dimensions in a standard, unsqueezed order (used for reference and target)
        DATA_PRECISION = reco_options["data_precision"]
        append_parameters = reco_options.get("append_parameters", True)

        recon_num = reco_options.get("recon_num", 1)
        set_reco_file = reco_options.get("set_reco_file", False)

        # Get data and parameters from handler
        seq2d = self._load_scanner_samples(
            handler, recon_num, set_reco_file, DATA_PRECISION
        )
        if seq2d is None:
            return None
        time.time()
        slice_orientation = handler.slice_orientation
        try:
            rps_grid_mm = handler.rps_grid_mm
        except AttributeError as exc:
            logger.warning(
                f"{getattr(handler, 'id', 'unknown')}: Failed to access rps_grid_mm ({exc}); using fallback seq2d reconstruction."
            )
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason=f"rps_grid_mm unavailable: {exc}",
            )

        if (
            rps_grid_mm is None
            or slice_orientation is None
            or isinstance(slice_orientation, (list, tuple))
        ):
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: coordinate metadata incomplete; using fallback seq2d reconstruction."
            )
            return self._fallback_seq2d_scanner_data(
                handler,
                seq2d,
                reco_options,
                append_parameters=append_parameters,
                reason="coordinate metadata incomplete",
            )
        axes = self._read_scanner_axes(handler, recon_num, rps_grid_mm)
        if axes is None:
            return None
        return self._build_strict_scanner_data(
            handler,
            seq2d,
            axes,
            rps_grid_mm,
            slice_orientation,
            reco_options,
            append_parameters,
        )

    def set_scanner_data(
        self,
        data: np.ndarray,
        dims: list,
        coords: dict = None,
        handler=None,
        reco_options=None,
    ):
        """Label scanner-reconstructed samples with acquisition geometry.

        Parameters
        ----------
        data : numpy.ndarray
            Scanner-reconstructed samples with one axis per entry of ``dims``.
            Values and dtype are preserved; the input is not modified.
        dims : sequence of str
            Labels in the order of ``data``. Spatial dimensions are named
            ``read_idx``, ``phase_idx`` and ``slice_idx`` when present.
        coords : dict, optional
            xarray coordinates, defaulting to integer dimension indices.
            Compatible acquisition and frame times are attached in seconds.
        handler : BrukerHandler, optional
            Required source of directions, size metadata and VisuCore geometry.
            Physical size attributes are in mm. Multiple packages have no
            common RPS axes or ``slice_spacing_mm``.
        reco_options : dict, optional
            ``recon_num`` records the selected data folder, default 1. This
            metadata operation does not transform samples or change precision.

        Returns
        -------
        xarray.DataArray
            New labelled array, potentially sharing sample storage with data.
            Dimension order and dtype are unchanged. DICOM-LPS ``x``, ``y``
            and ``z`` coordinates are attached in mm when VisuCore geometry
            applies. For one spatial package, ``slice_spacing_mm`` and
            ``slice_thickness_mm`` are stored on the slice coordinate and in
            array attributes, which survive dropping ``slice_idx``. Thickness
            means nominal 2D slice thickness or 3D partition size, not
            excitation slab thickness.
            ``bruker_reconstruction_number`` identifies the selected data folder.
            ``bruker_reco_parameter_source``, ``bruker_reco_time`` and
            ``bruker_reco_objects_per_repetition`` snapshot active handler
            parameters when available, independently of full parameter attachment.

        Raises
        ------
        ValueError
            If data and coordinate shapes are incompatible.
        ValueError
            If the handler has no ``acquisition_matrix`` because ``acqp``
            lacks ``ACQ_grad_matrix``.

        Notes
        -----
        No reconstruction is run and no acquired sample file is read. The
        result is deterministic for unchanged inputs; it is returned without
        storing it on the handler. Missing optional geometry or timing leaves
        the corresponding coordinates or attributes absent.

        Examples
        --------
        >>> labelled = strategy.set_scanner_data(
        ...     samples, dimension_names, handler=scan.handler
        ... )  # doctest: +SKIP
        """
        if reco_options is None:
            reco_options = {}
        start_time = time.time()

        # Create default coordinates if not provided
        if coords is None:
            coords = {
                dim: np.arange(size)
                for dim, size in zip(dims, data.shape, strict=False)
            }

        # Create or update the xarray DataArray
        try:
            data_array = xr.DataArray(data, dims=dims, coords=coords)
        except ValueError as exc:
            raise ValueError(
                f"""
                    Could not create xarray DataArray, shape of:
                    Data: {data.shape},
                    Dimensions: {dims},
                    Coordinates: {coords.sizes}

                """
            ) from exc

        # Add time coordinate if frame_time_s is available
        if (
            "repetition_idx" in data_array.dims
            and hasattr(handler, "frame_time_s")
            and handler.frame_time_s is not None
        ):
            try:
                data_array = _attach_seq2d_frame_time(
                    data=data_array, frame_time_s=handler.frame_time_s
                )

            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to add 'time' coordinate from frame_time_s: {e}"
                )

        # MGE echo coordinates declare ms or s explicitly.
        # Both describe echo timing independently of acquired readout sampling.
        acquisition = data_array.coords.get("acquisition_idx")
        is_echo_axis = (
            method_name_to_sequence_id(getattr(handler, "type", None)) == "mge"
            and acquisition is not None
            and acquisition.attrs.get("unit") in {"ms", "s"}
        )
        echo_times_ms = None
        if is_echo_axis:
            echo_times_ms = acquisition.values * (
                1.0 if acquisition.attrs["unit"] == "ms" else 1000.0
            )
        sampling = (
            echo_times_ms / 1000.0
            if is_echo_axis
            else getattr(handler, "sampling_time_axis_s", None)
            if _has_spectral_axis(handler)
            else None
        )
        if "acquisition_idx" in dims and sampling is not None:
            data_array = _attach_seq2d_acquisition_time(data_array, sampling)
        if is_echo_axis:
            for unit, axis in _echo_frequency_axes(handler, echo_times_ms).items():
                data_array = assign_frequency_axis(data_array, axis, unit)

        data_array.attrs["patient_pos"] = handler.patient_pos
        data_array.attrs["bruker_reconstruction_number"] = int(
            reco_options.get("recon_num", 1)
        )
        data_array.attrs.update(_bruker_reco_provenance(handler))
        data_array.attrs["slice_orientation"] = handler.slice_orientation
        data_array.attrs["read_orientation"] = handler.read_orientation
        data_array.attrs["num_image_fov_shift"] = 0
        gradient_directions = _gradient_directions(handler)
        data_array.attrs["read_direction"] = gradient_directions[0]
        data_array.attrs["phase_direction"] = gradient_directions[1]
        data_array.attrs["slice_direction"] = gradient_directions[2]
        data_array.attrs["domain"] = "spatial"
        if self._is_spectral_seq2d(handler, data_array):
            data_array = _spectra_in_ascending_frequency(data_array)
            data_array.attrs["signal_domain"] = "spectral"
            acq_size = data_array.sizes["acquisition_idx"]
            for unit, attr_name in (("ppm", "freq_axis_ppm"), ("Hz", "freq_axis_hz")):
                freq_axis = getattr(handler, attr_name, None)
                if freq_axis is not None and len(np.atleast_1d(freq_axis)) == acq_size:
                    data_array = assign_frequency_axis(data_array, freq_axis, unit)
        data_array.attrs["n_slices"] = handler.n_slices
        data_array.attrs["git_commit_hash"] = handler.git_commit_hash
        slice_geometry = getattr(handler, "_slice_geometry_attributes", None)
        if slice_geometry is not None:
            attributes = slice_geometry()
            data_array.attrs.update(attributes)
            if "slice_idx" in data_array.coords:
                data_array["slice_idx"].attrs.update(attributes)

        # Scans with several slice packages have no RPS positions but
        # keep their matrix and voxel size.
        rps_extent_mm = getattr(handler, "rps_extent_mm", None)
        rps_axes_mm = getattr(handler, "rps_axes_mm", None)
        has_rps_positions = rps_extent_mm is not None and rps_axes_mm is not None
        try:
            data_array = _attach_seq2d_axis_geometry(
                data=data_array,
                frame_time_s=getattr(handler, "frame_time_s", None),
                has_rps_positions=has_rps_positions,
                matrix=handler.matrix,
                n_repetitions=handler.n_repetitions,
                resolution_mm=handler.resolution_mm,
                rps_axes_mm=rps_axes_mm,
                rps_extent_mm=rps_extent_mm,
            )

        except (KeyError, IndexError, AttributeError) as exc:
            logger.warning("Optional seq2d axis geometry is incomplete: %s", exc)

        data_array = _attach_image_geometry(
            data_array, handler, recon_num=reco_options.get("recon_num", 1)
        )

        total_time = time.time() - start_time
        logger.warning(
            f"[TIMING] Brukerseq2dRecoStrategy total reconstruction took {total_time: .3f}s for scan {handler.expnum}"
        )

        logger.debug(
            f"Brukerseq2dRecoStrategy returning data_array: {type(data_array)}"
        )
        logger.debug(f"data_array is None: {data_array is None}")
        if data_array is not None:
            logger.debug(
                f"data_array.shape: {getattr(data_array, 'shape', 'no shape')}"
            )

        return _label_cpmg_echo_train(data_array, handler)


# -----------       BRUKER RAW RECO CLASSES     -----------------

__all__ = ["Brukerseq2dRecoStrategy"]
