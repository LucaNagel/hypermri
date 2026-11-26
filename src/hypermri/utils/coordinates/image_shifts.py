"""
Image shift and transformation utilities.

This module contains functions for shifting and transforming medical images,
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.
"""

import logging
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from hypermri.reconstruction.centred_dft import (
    centered_fft,
    centered_ifft,
    image_to_slice_encoding,
    slice_encoding_to_image,
)

# Initialize logger without dependencies
logger = logging.getLogger(__name__)

_PATIENT_POSITIONS = ("Head_Prone", "Head_Supine", "Foot_Prone", "Foot_Supine")


def get_orientation_flip_multipliers(scan) -> np.ndarray:
    """Return the orientation multipliers of a Bruker scan for read, phase and slice.

    The multipliers come from a table over patient position, slice
    orientation and read orientation.  :func:`determine_FOV_shift`
    multiplies the read, phase and slice offsets by them, and the Bruker raw
    reconstruction reverses the slice axis when the slice entry is -1.  Every
    table entry is +1: :func:`shift_image` defines its shift in the
    reconstructed image, where positive Bruker read and phase offsets move the
    object towards higher indices, and 2D multi-slice data and the 3D slice
    axis already follow the ``2dseq`` slice order (the raw reconstruction of
    a 3D Head_Prone coronal H_F bSSFP scan correlates with its ``2dseq``
    slices at 0.895 without reversal and 0.044 with the slices reversed).
    The table therefore only checks that the orientation of the scan has been
    reviewed; an orientation outside it raises an error.

    Parameters
    ----------
    scan : object
        Bruker handler or sequence object with the attributes ``patient_pos``
        (ParaVision patient position ``ACQ_patient_pos``: ``"Head_Prone"``,
        ``"Head_Supine"``, ``"Foot_Prone"`` or ``"Foot_Supine"``),
        ``slice_orientation`` (``"axial"``, ``"sagittal"`` or ``"coronal"``),
        ``read_orientation`` (``"L_R"``, ``"A_P"`` or ``"H_F"``) and
        ``spatial_dimension`` (``"2D"`` or ``"3D"``).
        ``slice_orientation`` and ``read_orientation`` may hold one entry per
        slice package.  A missing attribute is read as None.  For a sequence
        object, ``patient_pos`` and ``spatial_dimension`` are read from its vendor
        handler ``scan.handler``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(3,)`` with +1 or -1 for the ``read_idx``,
        ``phase_idx`` and ``slice_idx`` axes, in this order.  With several
        slice packages the multipliers common to all packages are returned.
        An object without ``slice_orientation`` and ``read_orientation``
        (non-imaging data) gives +1 for every axis.

    Raises
    ------
    ValueError
        If ``patient_pos`` is not one of the four positions above, if the
        slice orientation or read orientation of a Head_Prone or Head_Supine
        scan is not in the table (this includes a missing one when the other
        is given), if ``slice_orientation`` and ``read_orientation`` hold
        different numbers of entries (more than one), or if the slice
        packages need different multipliers.  The message names the values
        found.

    Notes
    -----
    The read and phase sign was checked for the 2D single-package
    Head_Prone and Head_Supine entries against ``2dseq`` in
    ``tests/unit/reconstruction/test_bruker_raw_versus_2dseq.py``.  The 3D
    slice entry of Head_Prone coronal scans has one scanner reference (a
    bSSFP scan without slice offset, so the sign of the 3D slice offset stays
    unverified); the other 3D entries and the Foot_Prone and Foot_Supine
    entries have none.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> scan = SimpleNamespace(
    ...     patient_pos="Head_Supine",
    ...     slice_orientation="axial",
    ...     read_orientation="A_P",
    ...     spatial_dimension="2D",
    ... )
    >>> get_orientation_flip_multipliers(scan).tolist()
    [1, 1, 1]
    """
    # ``patient_pos`` and ``spatial_dimension`` belong to the vendor handler of a scan.
    vendor_source = getattr(scan, "handler", scan)
    patient_pos = getattr(vendor_source, "patient_pos", None)
    slice_orientation = getattr(scan, "slice_orientation", None)
    read_orientation = getattr(scan, "read_orientation", None)
    spatial_dimension = getattr(vendor_source, "spatial_dimension", None)

    if patient_pos is not None and patient_pos not in _PATIENT_POSITIONS:
        raise ValueError(
            f"get_orientation_flip_multipliers: unknown patient_pos "
            f"'{patient_pos}'. Use one of {', '.join(_PATIENT_POSITIONS)} or "
            "correct ACQ_patient_pos of the scan."
        )
    if slice_orientation is None and read_orientation is None:
        return np.ones(3, dtype=int)

    slice_orientations = np.atleast_1d(slice_orientation).reshape(-1)
    read_orientations = np.atleast_1d(read_orientation).reshape(-1)
    if slice_orientations.size > 1 or read_orientations.size > 1:
        count = max(slice_orientations.size, read_orientations.size)
        if slice_orientations.size not in (1, count) or read_orientations.size not in (
            1,
            count,
        ):
            raise ValueError(
                "get_orientation_flip_multipliers: slice_orientation has "
                f"{slice_orientations.size} entries and read_orientation has "
                f"{read_orientations.size}; both must hold one entry per slice "
                "package."
            )

        slice_orientations = np.resize(slice_orientations, count)
        read_orientations = np.resize(read_orientations, count)
        per_pack = [
            get_orientation_flip_multipliers(
                _OrientationMetadata(
                    patient_pos=patient_pos,
                    slice_orientation=str(pack_slice_orientation),
                    read_orientation=str(pack_read_orientation),
                    spatial_dimension=spatial_dimension,
                )
            )
            for pack_slice_orientation, pack_read_orientation in zip(
                slice_orientations, read_orientations, strict=True
            )
        ]
        if all(
            np.array_equal(per_pack[0], multipliers) for multipliers in per_pack[1:]
        ):
            return per_pack[0]
        raise ValueError(
            "get_orientation_flip_multipliers: the slice packages need "
            "different orientation multipliers "
            f"({[multipliers.tolist() for multipliers in per_pack]}) and have "
            "no common flip. Query the packages separately with the "
            "slice_pack argument of determine_FOV_shift."
        )

    slice_orientation = str(slice_orientations[0])
    read_orientation = str(read_orientations[0])

    if patient_pos == "Head_Prone":
        listed = (
            (slice_orientation == "axial" and read_orientation in ("L_R", "A_P"))
            or (slice_orientation == "sagittal" and read_orientation in ("A_P", "H_F"))
            or slice_orientation == "coronal"
        )
    elif patient_pos == "Head_Supine":
        listed = (
            (slice_orientation == "axial" and read_orientation in ("L_R", "A_P"))
            or (slice_orientation == "sagittal" and read_orientation in ("H_F", "A_P"))
            or (slice_orientation == "coronal" and read_orientation in ("H_F", "L_R"))
        )
    else:
        # Foot_Prone and Foot_Supine have no scanner reference; they take the
        # entries of the verified Head positions for every orientation.
        listed = patient_pos in ("Foot_Prone", "Foot_Supine")

    if not listed:
        raise ValueError(
            "get_orientation_flip_multipliers: no flip table entry for "
            f"patient_pos '{patient_pos}', slice_orientation "
            f"'{slice_orientation}' and read_orientation '{read_orientation}'. "
            "Correct the orientation parameters of the scan or add a verified "
            "entry to the table."
        )

    return np.ones(3, dtype=int)


class _OrientationMetadata:
    """Small attribute container used to evaluate each Bruker slice pack."""

    def __init__(self, **values):
        self.__dict__.update(values)


def calc_mat_origin_diff(
    res_metab=None,
    fov_metab=None,
    mat_metab=None,
    res_anat=None,
    fov_anat=None,
    mat_anat=None,
):
    """Calculate voxel offsets required to align metabolite and anatomical datasets.

    Parameters mirror the legacy implementation to remain backward compatible. Either
    the explicit resolutions or the combination of field-of-view and matrix sizes must
    be provided for both datasets. Returned offsets follow the order ``[z, x, y]`` and
    are expressed in voxel units of the high-resolution anatomical acquisition.
    """
    if res_metab is None:
        if fov_metab is not None and mat_metab is not None:
            res_metab = [a / b for a, b in zip(fov_metab, mat_metab, strict=False)]
        else:
            return None, None, None

    if res_anat is None:
        if fov_anat is not None and mat_anat is not None:
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat, strict=False)]
        else:
            return None, None, None

    if fov_anat is None or mat_anat is None:
        return None, None, None

    shift_vox_list = [
        ma * (rm - ra) / 2 / fa
        for rm, ra, fa, ma in zip(res_metab, res_anat, fov_anat, mat_anat, strict=False)
    ]
    return shift_vox_list


def shift_image(
    data=None,
    shift_vox=None,
    shift_method="phase",
    force_shift=False,
    scan=None,
    domain="kspace",
    fov_shift_sign=1.0,
):
    """
    Shift image content by a number of voxels along the spatial axes.

    The shift is applied either as a linear phase ramp in k-space
    (``shift_method="phase"``) or as a periodic spline shift in the image
    domain with :func:`scipy.ndimage.shift` (``shift_method="scipy"``).  Data
    given in the other domain are transformed with the centred discrete
    Fourier transform, shifted once, and transformed back.

    The shift is defined in the reconstructed image: k-space and image are
    related by the transforms of the Bruker raw reconstruction,
    :func:`~hypermri.reconstruction.centred_dft.centered_ifft` for read and
    phase and
    :func:`~hypermri.reconstruction.centred_dft.slice_encoding_to_image` for
    the slice axis of 3D data, with the k-space centre at index ``n // 2``.
    A positive shift moves image content towards higher indices for k-space
    and image input alike, and an integer shift adds no constant phase to
    the image.  Shifts wrap around periodically.

    Parameters
    ----------
    data : ndarray or xarray.DataArray, optional
        Data to shift.  If None, ``scan.kspace_xarray`` is used.  An
        xarray input must have the dimensions ``read_idx`` and ``phase_idx``
        and, if its ``spatial_dimension`` attribute is ``"3D"``, ``slice_idx``; these
        are the spatial axes.  For an ndarray, the last ``len(shift_vox)``
        axes are the spatial axes in (read, phase, slice) order.
    shift_vox : array_like of float, optional
        Shift in voxels per spatial axis, in (read, phase, slice) order.
        Fractional values are allowed.  If None, the shift is derived from the
        FOV offsets of ``scan`` with :func:`determine_FOV_shift`; this is
        only possible for xarray input.  When ``scan`` has several slice
        packages (``slices_per_pack``), the input must be 2D with one
        ``slice_idx`` entry per slice, ordered by package, and the slices of
        each package are shifted by the shift of that package.  Required for
        ndarray input, where its length must equal ``data.ndim``.
    shift_method : {'phase', 'scipy'}, default 'phase'
        ``"phase"`` multiplies k-space by ``exp(-2j * pi * k * shift / n)``
        along read and phase and by ``exp(2j * pi * k * shift / n)`` along the
        3D slice axis, with the centred k index
        ``k = -(n // 2), ..., n - 1 - n // 2``.
        ``"scipy"`` shifts the real and imaginary parts with cubic spline
        interpolation and the periodic boundary mode ``"grid-wrap"`` of
        :func:`scipy.ndimage.shift`, along the spatial axes only; the other
        axes of the input are not shifted.  Content shifted across a border
        re-enters at the opposite border.
    force_shift : bool, default False
        Shift an xarray input even if its ``num_image_fov_shift`` attribute
        shows that it was shifted before.  Without it such an input is
        returned unchanged.
    scan : object, optional
        Sequence object that provides ``kspace_xarray`` when ``data``
        is None and the FOV offsets when ``shift_vox`` is None, with the
        attributes read by :func:`determine_FOV_shift` and, optionally,
        ``slices_per_pack`` (number of slices of each slice package).  A
        vendor handler works as well; for a sequence object,
        ``kspace_xarray`` and ``slices_per_pack`` are read from its handler
        ``scan.handler``.
    domain : {'kspace', 'image'}, default 'kspace'
        Domain of the input data.  The ``domain`` attribute of an xarray input
        takes precedence over this argument.
    fov_shift_sign : float, default 1.0
        Multiplier applied to a shift derived from ``scan``.  It is not
        applied to an explicit ``shift_vox``.

    Returns
    -------
    ndarray or xarray.DataArray
        Shifted data in the input domain with the input shape.  The
        calculation uses at least ``complex128``.  A complex input keeps its
        dtype; a real input returns the magnitude of the shifted data as
        ``float64``.  An xarray input is modified in place and returned, and
        its ``num_image_fov_shift`` attribute is incremented by one, starting
        from 0 when it is missing.

    Raises
    ------
    ValueError
        If ``shift_method`` is not supported, if neither ``data`` nor
        ``scan`` is given, if ``data`` has an unsupported type, if
        ``shift_vox`` is missing or has the wrong length for ndarray input, or
        if the FOV shift cannot be determined from ``scan``, which
        includes a ``scan`` with several slice packages for 3D input or
        for input whose ``slice_idx`` size differs from the sum of
        ``slices_per_pack``.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.zeros(8, dtype=complex)
    >>> image[3] = 1.0
    >>> shifted = shift_image(image, [1.0], shift_method="phase", domain="image")
    >>> int(np.argmax(np.abs(shifted)))
    4
    """
    if data is None:
        if scan is None:
            raise ValueError("Either data or scan must be provided.")
        vendor_source = getattr(scan, "handler", scan)
        if hasattr(vendor_source, "kspace_xarray"):
            kspace_xarray = vendor_source.kspace_xarray
            input_is_xarray = True
            data_np = kspace_xarray.to_numpy()
        else:
            raise ValueError("scan does not have the attribute kspace_xarray.")
    elif isinstance(data, xr.DataArray):
        kspace_xarray = data
        input_is_xarray = True
        data_np = kspace_xarray.to_numpy()
    elif isinstance(data, np.ndarray):
        data_np = data
        input_is_xarray = False
        if scan is None:
            if shift_vox is None:
                raise ValueError(
                    "shift_vox must be provided when data is a numpy array and scan is None."
                )
            if len(shift_vox) != data_np.ndim:
                raise ValueError(
                    "The number of dimensions of shift_vox must match the number of dimensions of data when scan is None."
                )
    else:
        raise ValueError(
            "Unsupported type for data. Must be ndarray or xarray.DataArray."
        )

    if input_is_xarray and "domain" in kspace_xarray.attrs:
        domain = kspace_xarray.attrs["domain"]

    apply_fov_shift_sign = False
    if input_is_xarray:
        read_axis = kspace_xarray.get_axis_num("read_idx")
        phase_axis = kspace_xarray.get_axis_num("phase_idx")
        if kspace_xarray.attrs.get("spatial_dimension") == "2D":
            slice_axis = None
        elif kspace_xarray.attrs.get("spatial_dimension") == "3D":
            slice_axis = kspace_xarray.get_axis_num("slice_idx")
        else:
            slice_axis = None

        if slice_axis is not None:
            spatial_axes = (read_axis, phase_axis, slice_axis)
        else:
            spatial_axes = (read_axis, phase_axis)
        is_complex = np.iscomplexobj(data_np)

        num_image_fov_shift = kspace_xarray.attrs.get("num_image_fov_shift", 0)
        if num_image_fov_shift > 0 and not force_shift:
            return kspace_xarray

        if shift_vox is None:
            try:
                slice_pack_indices = _slice_pack_indices(scan, kspace_xarray)
                if slice_pack_indices is None:
                    shift_rps_vox, _ = determine_FOV_shift(scan=scan)
                    shifts = [(None, shift_rps_vox)]
                else:
                    shifts = [
                        (index, determine_FOV_shift(scan, slice_pack=pack)[0])
                        for pack, index in enumerate(slice_pack_indices)
                    ]
                apply_fov_shift_sign = True

            except Exception as e:
                raise ValueError(
                    "Error determining FOV shift: "
                    f"{str(e)}. Please provide shift_vox or ensure scan is valid."
                )
        else:
            shifts = [(None, shift_vox)]

    else:  # numpy array
        if shift_vox is None:
            raise ValueError(
                "shift_vox must be provided for numpy array input when scan is None."
            )
        if len(shift_vox) != data_np.ndim:
            raise ValueError(
                "The length of shift_vox must match the number of dimensions of data."
            )
        spatial_axes = tuple(range(data_np.ndim - len(shift_vox), data_np.ndim))
        is_complex = np.iscomplexobj(data_np)
        shifts = [(None, shift_vox)]

    # Determine if this is 3D data for debugging
    is_3d = input_is_xarray and kspace_xarray.attrs.get("spatial_dimension") == "3D"
    if is_3d:
        logger.debug(f"3D spatial axes: {spatial_axes} (read, phase, slice)")
    else:
        logger.debug(f"2D spatial axes: {spatial_axes} (read, phase)")
    # The shift is calculated in at least complex128; the result keeps the
    # complex dtype of the input (see the Returns section).
    shifted_data = data_np.astype(np.result_type(data_np.dtype, np.complex128))
    # Each entry of shifts is the index of the slices of one slice package,
    # or None for all data, and its voxel shift.
    for index, shift_rps_vox in shifts:
        shift_rps_vox = np.asarray(shift_rps_vox, dtype=float)
        if apply_fov_shift_sign:
            shift_rps_vox = shift_rps_vox * float(fov_shift_sign)
        logger.debug(
            f"Shifting {'3D' if is_3d else '2D'} data with shape {data_np.shape}, "
            f"slices {index}, shift_rps_vox: {shift_rps_vox}, "
            f"shift_method: {shift_method}, domain: {domain}"
        )
        if index is None:
            shifted_data = _shift_spatial_axes(
                shifted_data, spatial_axes, shift_rps_vox, shift_method, domain
            )
        else:
            shifted_data[index] = _shift_spatial_axes(
                shifted_data[index], spatial_axes, shift_rps_vox, shift_method, domain
            )

    if is_complex:
        shifted_data = shifted_data.astype(data_np.dtype, copy=False)
    else:
        shifted_data = np.abs(shifted_data)

    if input_is_xarray:
        kspace_xarray.data = shifted_data
        kspace_xarray.attrs["num_image_fov_shift"] = (
            kspace_xarray.attrs.get("num_image_fov_shift", 0) + 1
        )
        return kspace_xarray
    return shifted_data


def _slice_pack_indices(scan, kspace_xarray):
    """Return the index of the slices of each slice package of ``kspace_xarray``.

    The Bruker raw reconstructions sort the slices of a 2D scan by
    ``PVM_ObjOrderList``, so the ``slice_idx`` entries follow the packages
    in the order of ``slices_per_pack``.  Each index selects the slices of
    one package along ``slice_idx`` and all entries of the other
    dimensions.  None is returned when ``scan`` has at most one slice
    package or no ``slices_per_pack``.

    Raises
    ------
    ValueError
        If ``scan`` has several slice packages and ``kspace_xarray`` is
        not 2D or its ``slice_idx`` size differs from the number of slices
        of the packages.
    """
    counts = getattr(getattr(scan, "handler", scan), "slices_per_pack", None)
    counts = np.atleast_1d(() if counts is None else counts).astype(int)
    if counts.size <= 1:
        return None
    if kspace_xarray.attrs.get("spatial_dimension") != "2D":
        raise ValueError(
            f"shift_image: the scan has {counts.size} slice packages "
            f"{tuple(counts.tolist())}, but field of view shifts per slice "
            "package are implemented for 2D data only; pass shift_vox to shift "
            "the data by one explicit voxel shift."
        )
    n_slices = kspace_xarray.sizes.get("slice_idx")
    if n_slices != counts.sum():
        raise ValueError(
            f"shift_image: slice_idx has {n_slices} slices, but the "
            f"{counts.size} slice packages of the scan (slices_per_pack "
            f"{tuple(counts.tolist())}) hold {counts.sum()}; pass the data of "
            "all slices, ordered by slice package."
        )
    slice_axis = kspace_xarray.get_axis_num("slice_idx")
    stops = np.cumsum(counts)
    indices = []
    for start, stop in zip(stops - counts, stops, strict=True):
        index = [slice(None)] * kspace_xarray.ndim
        index[slice_axis] = slice(int(start), int(stop))
        indices.append(tuple(index))
    return indices


def _shift_spatial_axes(data, spatial_axes, shift_rps_vox, shift_method, domain):
    """Return ``data`` shifted by ``shift_rps_vox`` voxels along ``spatial_axes``.

    This is the numerical part of :func:`shift_image`, which documents the
    methods, the transform convention and the sign of the shift.  ``data``
    must be complex; the phase method multiplies k-space input in place.  The
    result is in the input ``domain``.
    """
    if shift_method == "scipy":
        if domain == "kspace":
            data = _kspace_to_image(data, spatial_axes)
        # ndimage.shift takes one shift per array axis; the axes outside
        # spatial_axes keep a zero shift.
        axis_shifts = np.zeros(data.ndim)
        for i, axis in enumerate(spatial_axes):
            axis_shifts[axis] = shift_rps_vox[i]
        # "grid-wrap" is periodic with the array length; scipy's "wrap" is not.
        real_shifted = shift(np.real(data), axis_shifts, mode="grid-wrap")
        imag_shifted = shift(np.imag(data), axis_shifts, mode="grid-wrap")
        data = real_shifted + 1j * imag_shifted
        if domain == "kspace":
            data = _image_to_kspace(data, spatial_axes)

    elif shift_method == "phase":
        if domain != "kspace":
            data = _image_to_kspace(data, spatial_axes)
        axis_names = ["read", "phase", "slice"]
        for i, axis in enumerate(spatial_axes):
            axis_shift = shift_rps_vox[i]
            if axis_shift == 0:
                continue
            logger.debug(
                f"Applying phase shift on {axis_names[i]} axis {axis} with shift "
                f"{axis_shift} voxels"
            )
            data *= _centred_phase_ramp(
                data.shape, axis, axis_shift, slice_encoding=i == 2
            )
        if domain != "kspace":
            data = _kspace_to_image(data, spatial_axes)
    else:
        raise ValueError(
            f"Unsupported shift method: {shift_method}, use either 'phase' or 'scipy'."
        )
    return data


def _kspace_to_image(data, spatial_axes):
    """Transform the spatial axes (read, phase, slice order) to the image grid.

    Read and phase take :func:`centered_ifft`, the slice axis of a 3D scan
    :func:`slice_encoding_to_image`, as the Bruker raw reconstruction does.
    """
    data = centered_ifft(data, spatial_axes[:2])
    return slice_encoding_to_image(data, spatial_axes[2:])


def _image_to_kspace(data, spatial_axes):
    """Return the inverse of :func:`_kspace_to_image` along ``spatial_axes``."""
    data = centered_fft(data, spatial_axes[:2])
    return image_to_slice_encoding(data, spatial_axes[2:])


def _centred_phase_ramp(shape, axis, shift_vox, *, slice_encoding):
    """Return the k-space phase ramp that shifts the image by ``shift_vox`` voxels.

    The ramp uses the centred k index ``k = -(n // 2), ..., n - 1 - n // 2``,
    so it is 1 at the k-space centre and adds no constant phase to the image.
    Read and phase are reconstructed with an inverse DFT and take
    ``exp(-2j * pi * k * shift_vox / n)``; the slice-encoding axis is
    reconstructed with a forward DFT and takes the opposite sign.  Both move
    the reconstructed image towards higher indices for a positive shift.  The
    ramp has length ``n`` along ``axis`` and length 1 along all other axes for
    broadcasting.
    """
    n = shape[axis]
    k = np.arange(n) - n // 2
    sign = 1.0 if slice_encoding else -1.0
    ramp_shape = [1] * len(shape)
    ramp_shape[axis] = n
    return np.exp(sign * 2j * np.pi * k * shift_vox / n).reshape(ramp_shape)


def determine_FOV_shift(
    scan=None,
    slice_pack=None,
):
    """Return the field of view shift of a scan from its slice package offsets.

    The read, phase and slice offsets of the scan are converted to voxels
    with the voxel size and multiplied by the orientation multipliers of
    :func:`get_orientation_flip_multipliers`.  The phase offset is always
    converted.  The read offset is converted only for CSI scans; imaging
    scans keep the read entry of the voxel shift at 0.  :func:`shift_image`
    applies the voxel shift when it is not given explicitly, for a scan with
    several slice packages the shift of each package to its slices.

    Parameters
    ----------
    scan : object, optional
        Bruker handler or sequence object with the attributes
        ``read_offset_mm``, ``phase_offset_mm`` and, optionally,
        ``slice_offset_mm`` (offsets in mm along read, phase and slice;
        a missing slice offset is read as 0), ``resolution_mm`` (voxel size
        in mm along read, phase and, optionally, slice), ``type`` (sequence
        type; a scan is CSI when it contains ``"csi"``, ignoring case, or,
        when it is empty or None, when the class name does), ``spatial_dimension``
        and the attributes read by :func:`get_orientation_flip_multipliers`.
        The Bruker handler provides the offsets of the first slice package
        in these attributes.  With ``slice_pack``, ``slice_pack_offsets_mm``
        replaces the three offsets.  For a sequence object,
        ``slice_pack_offsets_mm``, ``spatial_dimension`` and ``patient_pos`` are read
        from its vendor handler ``scan.handler``.
    slice_pack : int, optional
        Index of a slice package, counted from 0 in the order of
        ``slices_per_pack``.  If given, the offsets are row ``slice_pack``
        of ``scan.slice_pack_offsets_mm`` (array of shape
        ``(n_packages, 3)`` with the read, phase and slice offset in mm),
        and the multipliers are those of the slice and read orientation of
        that package; a single orientation applies to every package.  If
        None (default), the offsets of the first package are used, with the
        multipliers that all packages share (see
        :func:`get_orientation_flip_multipliers`).

    Returns
    -------
    shift_rps_vox : numpy.ndarray
        Shift in voxels along read, phase and slice, ``float64``.  It has
        three entries when ``spatial_dimension`` is ``"3D"`` and the read and phase
        entries otherwise.  The slice entry is 0 when ``resolution_mm`` has
        no slice entry.
    shift_rps_mm : numpy.ndarray
        Offsets in mm times the orientation multipliers, ``float64``, with
        the same entries as ``shift_rps_vox``.

    Raises
    ------
    AttributeError
        If ``scan`` has no ``type`` attribute.
    TypeError, IndexError
        If ``resolution_mm`` is None or has fewer than two entries, or if
        ``slice_pack`` is not a row of ``slice_pack_offsets_mm`` or an entry
        of an orientation with one entry per package.
    ValueError
        If the patient position or orientation of the scan, or of the
        requested package, is not in the table, or if ``slice_pack`` is None
        and the packages have different multipliers (see
        :func:`get_orientation_flip_multipliers`).

    Notes
    -----
    When ``scan`` is None (logged as critical) or lacks an offset, the
    ``slice_pack_offsets_mm`` of a requested package or the resolution
    (logged as a warning), both results are zeros of length 2, the voxel
    shift as integers and the mm shift as ``float64``.  The voxel and mm
    shifts are logged at the info level.

    Examples
    --------
    A CSI scan in the Head_Prone axial orientation:

    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> scan = SimpleNamespace(
    ...     type="CSI",
    ...     read_offset_mm=2.0,
    ...     phase_offset_mm=3.0,
    ...     resolution_mm=np.array([0.5, 0.25, 1.0]),
    ...     patient_pos="Head_Prone",
    ...     slice_orientation="axial",
    ...     read_orientation="L_R",
    ...     spatial_dimension="2D",
    ... )
    >>> shift_rps_vox, shift_rps_mm = determine_FOV_shift(scan)
    >>> shift_rps_vox.tolist(), shift_rps_mm.tolist()
    ([4.0, 12.0], [2.0, 3.0])
    """
    if scan is None:
        logger.critical("scan is None, returning zero shifts")
        return np.array([0, 0]), np.array([0.0, 0.0])

    vendor_source = getattr(scan, "handler", scan)
    try:
        if slice_pack is None:
            offsets_rps_mm = (
                scan.read_offset_mm,
                scan.phase_offset_mm,
                getattr(scan, "slice_offset_mm", 0),
            )
        else:
            offsets_rps_mm = np.asarray(
                vendor_source.slice_pack_offsets_mm, dtype=float
            )[slice_pack]
        resolution_mm = scan.resolution_mm
    except AttributeError as e:
        logger.warning(
            f"Missing required attribute in scan: {e}, returning zero shifts"
        )
        return np.array([0, 0]), np.array([0.0, 0.0])

    sequence_type = scan.type
    if sequence_type:
        is_csi = "csi" in str(sequence_type).lower()
    else:
        is_csi = "csi" in scan.__class__.__name__.lower()
    logger.debug(f"Sequence type: {sequence_type}, is_csi: {is_csi}")

    if slice_pack is None:
        flip_multipliers = get_orientation_flip_multipliers(scan)
    else:
        flip_multipliers = get_orientation_flip_multipliers(
            _slice_pack_orientation(scan, slice_pack)
        )
    shift_rps_vox, shift_rps_mm = _fov_shift_from_offsets(
        offsets_rps_mm, resolution_mm, flip_multipliers, shift_read=is_csi
    )

    logger.info(f"shift_rps_vox: {shift_rps_vox}")
    logger.info(f"shift_rps_mm: {shift_rps_mm}")

    if getattr(vendor_source, "spatial_dimension", None) == "3D":
        return shift_rps_vox, shift_rps_mm
    return shift_rps_vox[:2], shift_rps_mm[:2]


def _slice_pack_orientation(scan, slice_pack):
    """Return the orientation attributes of one slice package of ``scan``.

    ``slice_orientation`` and ``read_orientation`` hold one entry per slice
    package or one entry for all packages.  The result has the attributes
    that :func:`get_orientation_flip_multipliers` reads.
    """
    orientations = {}
    for name in ("slice_orientation", "read_orientation"):
        values = np.atleast_1d(getattr(scan, name, None)).reshape(-1)
        orientations[name] = str(values[slice_pack] if values.size > 1 else values[0])
    vendor_source = getattr(scan, "handler", scan)
    return _OrientationMetadata(
        patient_pos=getattr(vendor_source, "patient_pos", None),
        spatial_dimension=getattr(vendor_source, "spatial_dimension", None),
        **orientations,
    )


def _fov_shift_from_offsets(
    offsets_rps_mm, resolution_mm, flip_multipliers, *, shift_read
):
    """Return the field of view shift of one slice package.

    Parameters
    ----------
    offsets_rps_mm : sequence of float
        Read, phase and slice offsets of the package in mm
        (``PVM_SPackArrReadOffset``, ``PVM_SPackArrPhase1Offset`` and
        ``PVM_SPackArrSliceOffset`` for Bruker scans).
    resolution_mm : sequence of float
        Voxel size in mm along read, phase and, optionally, slice.
    flip_multipliers : sequence of int
        +1 or -1 for read, phase and slice from
        :func:`get_orientation_flip_multipliers`.
    shift_read : bool
        Convert the read offset to voxels; otherwise the read entry of the
        voxel shift is 0 (CSI scans convert it, imaging scans do not).

    Returns
    -------
    shift_rps_vox : numpy.ndarray
        ``float64`` array of shape ``(3,)``: the offsets divided by the voxel
        size and multiplied by ``flip_multipliers``.  The slice entry is 0
        when ``resolution_mm`` has two entries.
    shift_rps_mm : numpy.ndarray
        ``float64`` array of shape ``(3,)``: the offsets multiplied by
        ``flip_multipliers``.

    Raises
    ------
    TypeError, IndexError
        If ``resolution_mm`` is None or lacks a required entry.
    """
    shift_rps_mm = list(offsets_rps_mm)
    shift_rps_vox = [
        shift_rps_mm[0] / resolution_mm[0] if shift_read else 0,
        shift_rps_mm[1] / resolution_mm[1],
        shift_rps_mm[2] / resolution_mm[2] if len(resolution_mm) > 2 else 0,
    ]
    shift_rps_vox = np.multiply(shift_rps_vox, flip_multipliers).astype(float)
    shift_rps_mm = np.multiply(shift_rps_mm, flip_multipliers).astype(float)
    return shift_rps_vox, shift_rps_mm


def _ensure_dataarray(
    data: Union[xr.DataArray, np.ndarray],
    dims_hint: Optional[Sequence[str]] = None,
) -> xr.DataArray:
    """Return an xarray.DataArray, inferring dimensions when possible."""
    if isinstance(data, xr.DataArray):
        return data

    if dims_hint is None:
        raise ValueError(
            "dims_hint is required when providing numpy arrays to alignment utilities."
        )

    array = np.asarray(data)
    if len(dims_hint) != array.ndim:
        raise ValueError(
            f"dims_hint has length {len(dims_hint)}, but data has {array.ndim} dimensions."
        )

    return xr.DataArray(array, dims=tuple(dims_hint))


def _normalize_for_ssim(values: np.ndarray) -> np.ndarray:
    """Normalize values into the [0, 1] interval for SSIM computation."""
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val > 0:
        arr /= max_val
    return arr


def _shift_complex_nd(
    data: np.ndarray,
    shift_amounts: Sequence[float],
    order: int = 3,
    mode: str = "constant",
) -> np.ndarray:
    """Apply a sub-pixel shift to complex data."""
    real_shifted = shift(np.real(data), shift_amounts, order=order, mode=mode)
    imag_shifted = shift(np.imag(data), shift_amounts, order=order, mode=mode)
    return real_shifted + 1j * imag_shifted


def _average_repetition_dimensions(
    data: xr.DataArray,
    plane_dims: Sequence[str],
    repetition_dim_keywords: Sequence[str],
) -> xr.DataArray:
    """Average dimensions likely representing repetitions before projection."""
    lowered_keywords = [kw.lower() for kw in repetition_dim_keywords]
    rep_dims = [
        dim
        for dim in data.dims
        if dim not in plane_dims
        and any(keyword in dim.lower() for keyword in lowered_keywords)
    ]
    if not rep_dims:
        return data
    return data.mean(dim=rep_dims)


def _prepare_projection(
    data: xr.DataArray,
    plane_dims: Sequence[str],
    reduce_dims: Union[str, Sequence[str], None],
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
) -> xr.DataArray:
    """
    Reduce an N-D xarray.DataArray down to the requested imaging plane.

    Parameters
    ----------
    data
        Source array with at least the dimensions listed in ``plane_dims``.
    plane_dims
        Ordered tuple of dimensions that describe the 2D imaging plane
        (e.g., ``(\"read_idx\", \"phase_idx\")``). The returned DataArray is
        transposed to match this order.
    reduce_dims
        Dimensions to collapse before extracting the plane. ``\"auto\"`` (default)
        removes every dimension not present in ``plane_dims``. ``None`` skips the
        reduction step. A string is treated as a single dimension name.
    reduce_func
        Aggregation to apply when reducing. Supported values are ``\"mean\"``,
        ``\"median\"``, or a callable accepting ``(data, dims_to_reduce)`` and
        returning an xarray.DataArray.

    Returns
    -------
    xr.DataArray
        2D plane ordered according to ``plane_dims``. All requested reductions are
        applied and singleton dimensions squeezed.

    Raises
    ------
    ValueError
        If ``plane_dims`` contain missing dimensions or ``reduce_func`` is not one
        of the supported values/callables.
    """
    missing = [dim for dim in plane_dims if dim not in data.dims]
    if missing:
        raise ValueError(f"Data is missing required dimensions: {missing}")

    if reduce_dims == "auto" or reduce_dims is None:
        reduce_dims = [dim for dim in data.dims if dim not in plane_dims]
    else:
        reduce_dims = list(reduce_dims)

    reduced = data
    if reduce_dims:
        if callable(reduce_func):
            reduced = reduce_func(reduced, reduce_dims)
        elif reduce_func == "mean":
            reduced = reduced.mean(dim=reduce_dims)
        elif reduce_func == "median":
            reduced = reduced.median(dim=reduce_dims)
        else:
            raise ValueError(
                f"Unsupported reduce_func '{reduce_func}'. Provide 'mean', 'median', or a callable."
            )

    reduced = reduced.squeeze()
    reduced = reduced.transpose(*plane_dims)
    return reduced


def estimate_image_alignment(
    source: Union[xr.DataArray, np.ndarray],
    reference: Union[xr.DataArray, np.ndarray],
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    reduce_dims: Union[str, Sequence[str], None] = "auto",
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
    flip_mode: str = "auto",
    flip_dim: Optional[str] = "phase_idx",
    upsample_factor: int = 10,
    shift_order: int = 3,
    average_repetitions: bool = True,
    repetition_dim_keywords: Sequence[str] = (
        "rep",
        "repetition",
        "average",
        "repetition_idx",
    ),
) -> Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]:
    """
    Estimate the flip + sub-voxel translation required to align ``source`` to ``reference``.

    Parameters
    ----------
    source, reference
        Complex-valued data defined over at least the dimensions listed in ``image_dims``.
        ``numpy`` arrays must be ordered according to ``image_dims``.
    image_dims
        Size-2 iterable naming the imaging axes (typically ``(\"read_idx\", \"phase_idx\")``).
    reduce_dims
        Dimensions to average/aggregate away prior to registration. ``\"auto\"`` removes
        every dimension not present in ``image_dims``. ``None`` disables reduction.
    reduce_func
        Aggregation used when collapsing ``reduce_dims``. Accepts ``\"mean\"``, ``\"median\"``,
        or a callable ``f(data, dims)`` returning an xarray.DataArray.
    flip_mode
        Controls whether a flip along ``flip_dim`` is considered: ``\"auto\"`` tests both
        orientations, ``\"always\"`` forces a flip, and ``\"never\"`` skips it.
    flip_dim
        Dimension inside ``image_dims`` to flip when testing mirrored alignments. ``None``
        disables flipping entirely.
    upsample_factor
        Sub-pixel refinement factor passed to ``phase_cross_correlation``.
    shift_order
        Interpolation order used when previewing the shift (0 = nearest neighbor, 3 = cubic).
    average_repetitions
        When ``True`` (default), automatically average dimensions whose name contains any
        of ``repetition_dim_keywords`` before extracting the imaging plane. This reduces
        noise from repeated acquisitions while leaving the original data untouched.
    repetition_dim_keywords
        Keywords used to detect repetition-like dimensions. Matching ignores case.

    Returns
    -------
    dict
        Dictionary containing:

        - ``\"shifts\"``: mapping of each image dimension to the detected translation in voxels.
        - ``\"shift_vector\"``: tuple sorted according to ``image_dims``.
        - ``\"flip_applied\"`` / ``\"flip_dim\"``: indicates whether the best match required a flip.
        - ``\"ssim\"`` plus raw ``phase_cross_correlation`` metrics.
        - ``\"preview_source\"``, ``\"preview_reference\"``, ``\"preview_aligned\"``: 2D projections
          useful for quick visualization.

    Raises
    ------
    ValueError
        If inputs are missing required dimensions or invalid flip/reduce settings are supplied.
    RuntimeError
        If no alignment scenario can be evaluated (should only occur with empty inputs).
    """
    plane_dims = tuple(image_dims)
    source_da = _ensure_dataarray(source, dims_hint=plane_dims)
    reference_da = _ensure_dataarray(reference, dims_hint=plane_dims)

    if average_repetitions:
        source_da = _average_repetition_dimensions(
            source_da,
            plane_dims=plane_dims,
            repetition_dim_keywords=repetition_dim_keywords,
        )
        reference_da = _average_repetition_dimensions(
            reference_da,
            plane_dims=plane_dims,
            repetition_dim_keywords=repetition_dim_keywords,
        )

    source_plane = _prepare_projection(
        source_da,
        plane_dims=plane_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
    )
    reference_plane = _prepare_projection(
        reference_da,
        plane_dims=plane_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
    )

    if flip_dim and flip_dim not in plane_dims:
        raise ValueError(
            f"flip_dim '{flip_dim}' is not part of the image plane {plane_dims}."
        )

    flip_axis = (
        plane_dims.index(flip_dim) if flip_dim and flip_dim in plane_dims else None
    )
    if flip_mode not in {"auto", "never", "always"}:
        raise ValueError("flip_mode must be 'auto', 'never', or 'always'.")

    flip_options = [False]
    if flip_axis is not None:
        if flip_mode == "always":
            flip_options = [True]
        elif flip_mode == "auto":
            flip_options = [False, True]
    elif flip_mode in {"always"}:
        raise ValueError(
            "flip_mode 'always' requires flip_dim to be part of the plane."
        )

    best_result: Optional[
        Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]
    ] = None

    reference_values = reference_plane.data
    mag_reference = np.abs(reference_values)

    for flip in flip_options:
        candidate = source_plane
        if flip and flip_axis is not None:
            candidate = candidate.isel({plane_dims[flip_axis]: slice(None, None, -1)})

        candidate_values = candidate.data
        mag_source = np.abs(candidate_values)

        shifts, error, phasediff = phase_cross_correlation(
            mag_reference, mag_source, upsample_factor=upsample_factor
        )
        aligned_preview = _shift_complex_nd(
            candidate_values, shifts, order=shift_order, mode="constant"
        )

        ssim_score = float(
            structural_similarity(
                _normalize_for_ssim(mag_reference),
                _normalize_for_ssim(np.abs(aligned_preview)),
                data_range=1.0,
            )
        )

        preview_coords = {
            dim: candidate.coords[dim] for dim in plane_dims if dim in candidate.coords
        }

        result: Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]] = {
            "shifts": {
                dim: float(shift_val)
                for dim, shift_val in zip(plane_dims, shifts, strict=False)
            },
            "shift_vector": tuple(float(val) for val in shifts),
            "flip_applied": bool(flip),
            "flip_dim": flip_dim if (flip and flip_axis is not None) else None,
            "ssim": ssim_score,
            "phase_correlation_error": float(error),
            "phase_correlation_phasediff": float(phasediff),
            "preview_source": candidate,
            "preview_reference": reference_plane,
            "preview_aligned": xr.DataArray(
                aligned_preview,
                dims=plane_dims,
                coords=preview_coords,
            ),
        }

        if best_result is None or ssim_score > best_result["ssim"]:  # type: ignore[index]
            best_result = result

    if best_result is None:
        raise RuntimeError("Alignment estimation failed to evaluate any scenarios.")

    return best_result


def apply_image_alignment(
    data: Union[xr.DataArray, np.ndarray],
    alignment: Mapping[str, Union[float, Sequence[float], Mapping[str, float], bool]],
    *,
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    order: int = 3,
    mode: str = "constant",
    wrap_edges: bool = True,
) -> xr.DataArray:
    """
    Apply previously estimated flip/shift parameters to an entire dataset.

    Parameters
    ----------
    data
        Data to transform. Accepts an xarray.DataArray (preferred) or a numpy ndarray
        ordered according to ``image_dims``.
    alignment
        Mapping produced by :func:`estimate_image_alignment`. At minimum it should
        contain ``\"shifts\"`` and ``\"flip_applied\"``. Extra metadata is preserved.
    image_dims
        Ordered imaging axes that describe the plane over which the shift is applied.
    order
        Interpolation order forwarded to ``scipy.ndimage.shift`` (0=nearest, 1=linear, ...).
    mode
        Padding mode used during interpolation. Ignored when ``wrap_edges=True``.
    wrap_edges
        When ``True``, perform circular wrapping at the image boundaries (``mode=\"wrap\"``).
        ``False`` keeps the previous zero-padding behaviour.

    Returns
    -------
    xr.DataArray
        Copy of ``data`` with shifts applied across the specified axes and an
        ``alignment_parameters`` attribute describing what was executed.

    Raises
    ------
    ValueError
        If ``data`` lacks required dimensions or mismatched hints are provided.
    """
    plane_dims = tuple(image_dims)
    dims_hint = plane_dims if isinstance(data, np.ndarray) else None
    data_da = _ensure_dataarray(data, dims_hint=dims_hint)

    for dim in plane_dims:
        if dim not in data_da.dims:
            raise ValueError(f"Data is missing required dimension '{dim}'.")

    result = data_da
    original_dim_order = tuple(result.dims)

    flip_dim = alignment.get("flip_dim")
    flip_applied = alignment.get("flip_applied", False)
    if flip_applied and isinstance(flip_dim, str) and flip_dim in result.dims:
        result = result.isel({flip_dim: slice(None, None, -1)})

    shift_mapping = alignment.get("shifts", {})
    shift_vector = tuple(
        (
            float(shift_mapping.get(dim, 0.0))
            if isinstance(shift_mapping, Mapping)
            else 0.0
        )
        for dim in plane_dims
    )

    shift_mode = "wrap" if wrap_edges else mode
    if any(abs(val) > 1e-6 for val in shift_vector):
        result = xr.apply_ufunc(
            _shift_complex_nd,
            result,
            input_core_dims=[list(plane_dims)],
            output_core_dims=[list(plane_dims)],
            kwargs={"shift_amounts": shift_vector, "order": order, "mode": shift_mode},
            vectorize=True,
            dask="parallelized",
        )
        # xarray moves the broadcast dimensions ahead of the core dims inside
        # apply_ufunc(vectorize=True); transpose back so callers see the original layout.
        result = result.transpose(*original_dim_order)

    attrs = dict(result.attrs)
    attrs["alignment_parameters"] = {
        "image_dims": plane_dims,
        "shift_vector": shift_vector,
        "flip_dim": flip_dim if flip_applied else None,
        "ssim": alignment.get("ssim"),
        "shift_mode": shift_mode,
        "wrap_edges": bool(shift_mode == "wrap"),
    }
    result = result.assign_attrs(attrs)
    return result


def align_image_data(
    source: xr.DataArray,
    reference: xr.DataArray,
    *,
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    reduce_dims: Union[str, Sequence[str], None] = "auto",
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
    flip_mode: str = "auto",
    flip_dim: Optional[str] = "phase_idx",
    upsample_factor: int = 10,
    shift_order: int = 3,
    order: int = 3,
    mode: str = "constant",
    wrap_edges: bool = False,
    average_repetitions: bool = True,
    repetition_dim_keywords: Sequence[str] = ("rep", "repetition", "average"),
) -> Tuple[xr.DataArray, Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]]:
    """
    Estimate alignment parameters and immediately apply them to ``source``.

    Parameters
    ----------
    source
        Xarray object describing the moving dataset.
    reference
        Xarray object describing the fixed dataset.
    image_dims
        Forwarded verbatim to :func:`estimate_image_alignment`.
    reduce_dims
        Forwarded verbatim to :func:`estimate_image_alignment`.
    reduce_func
        Forwarded verbatim to :func:`estimate_image_alignment`.
    flip_mode
        Forwarded verbatim to :func:`estimate_image_alignment`.
    flip_dim
        Forwarded verbatim to :func:`estimate_image_alignment`.
    upsample_factor
        Forwarded verbatim to :func:`estimate_image_alignment`.
    shift_order
        Forwarded verbatim to :func:`estimate_image_alignment`.
    average_repetitions
        Forwarded verbatim to :func:`estimate_image_alignment`.
    repetition_dim_keywords
        Forwarded verbatim to :func:`estimate_image_alignment`.
    order
        Forwarded to :func:`apply_image_alignment` when applying the detected shift.
    mode
        Forwarded to :func:`apply_image_alignment` when applying the detected shift.
    wrap_edges
        Forwarded to :func:`apply_image_alignment` when applying the detected shift.

    Returns
    -------
    tuple
        ``(aligned_source, alignment_info)`` where ``aligned_source`` is the shifted
        DataArray and ``alignment_info`` matches the structure produced by
        :func:`estimate_image_alignment`.

    """
    alignment = estimate_image_alignment(
        source,
        reference,
        image_dims=image_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
        flip_mode=flip_mode,
        flip_dim=flip_dim,
        upsample_factor=upsample_factor,
        shift_order=shift_order,
        average_repetitions=average_repetitions,
        repetition_dim_keywords=repetition_dim_keywords,
    )
    aligned = apply_image_alignment(
        source,
        alignment,
        image_dims=image_dims,
        order=order,
        mode=mode,
        wrap_edges=wrap_edges,
    )
    return aligned, alignment


def plot_alignment_overview(
    source_preview: xr.DataArray,
    reference_preview: xr.DataArray,
    aligned_preview: xr.DataArray,
    alignment_info: Mapping[str, Union[float, bool, Sequence[float]]],
):
    """
    Create a compact diagnostic figure summarizing the alignment result.

    Parameters
    ----------
    source_preview, reference_preview, aligned_preview
        2D magnitude images (typically produced by :func:`estimate_image_alignment`)
        that illustrate the original moving data, the fixed reference, and the
        aligned output.
    alignment_info
        Mapping containing SSIM/shift metadata used for figure annotations. Passing
        the dict returned by :func:`estimate_image_alignment` is recommended.

    Returns
    -------
    matplotlib.figure.Figure
        Handle to the constructed figure for further customization or saving.
    """
    import matplotlib.pyplot as plt  # Local import to avoid mandatory dependency at import time

    source_vals = np.abs(source_preview.values)
    reference_vals = np.abs(reference_preview.values)
    aligned_vals = np.abs(aligned_preview.values)

    vmax = max(
        np.nanmax(source_vals),
        np.nanmax(reference_vals),
        np.nanmax(aligned_vals),
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    axes[0, 0].imshow(source_vals, cmap="viridis", vmax=vmax)
    axes[0, 0].set_title("Original Source")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reference_vals, cmap="viridis", vmax=vmax)
    axes[0, 1].set_title("Reference")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(aligned_vals, cmap="viridis", vmax=vmax)
    ssim_val = alignment_info.get("ssim")
    if isinstance(ssim_val, (float, int)):
        axes[0, 2].set_title(f"Aligned (SSIM={ssim_val:.3f})")
    else:
        axes[0, 2].set_title("Aligned")
    axes[0, 2].axis("off")

    overlay_before = np.zeros((*source_vals.shape, 3))
    overlay_before[..., 0] = reference_vals / vmax
    overlay_before[..., 1] = source_vals / vmax
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title("Overlay Before\n(Red=Reference, Green=Source)")
    axes[1, 0].axis("off")

    overlay_after = np.zeros((*source_vals.shape, 3))
    overlay_after[..., 0] = reference_vals / vmax
    overlay_after[..., 1] = aligned_vals / vmax
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title("Overlay After\n(Red=Reference, Green=Aligned)")
    axes[1, 1].axis("off")

    diff = np.abs(reference_vals - aligned_vals)
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Absolute Difference")
    axes[1, 2].axis("off")

    mid = source_vals.shape[0] // 2
    axes[2, 0].plot(np.abs(source_vals[mid, :]))
    axes[2, 0].set_title("Source Profile")

    axes[2, 1].plot(np.abs(reference_vals[mid, :]))
    axes[2, 1].set_title("Reference Profile")

    axes[2, 2].plot(np.abs(aligned_vals[mid, :]), label="Aligned")
    axes[2, 2].plot(np.abs(reference_vals[mid, :]), label="Reference")
    axes[2, 2].legend()
    axes[2, 2].set_title("Profile Comparison")

    shift_vector = alignment_info.get("shift_vector")
    axes[2, 2].set_xlabel(f"Shift: {shift_vector}")

    plt.suptitle(
        f"Alignment preview (flip={alignment_info.get('flip_applied')}, shift={shift_vector})"
    )
    plt.tight_layout()
    return fig
