"""Frame groups of Bruker ``2dseq`` reconstructions.

ParaVision stores the frames of ``2dseq`` in the order of ``VisuFGOrderDesc``,
fastest frame group first, after the frame dimensions.  The seq2d strategy
labels the axes of the frame groups at the positions where they are stored:
the strict path falls back when its labels contradict them, and the fallback
labels these axes from their frame groups and the other axes by their size
.  Every frame-group type has one label, except that ``FG_MOVIE`` holds
the inversion times of FAIR methods.  Every reconstruction folder
``pdata/<n>`` has its own
``visu_pars`` and ``reco``; the strategy reads those of the selected
reconstruction.

Each test scan is synthetic: the ParaVision parameters of the zero-offset axial
series ``head_prone/10`` of the ParaVision reference fixture
``tests/reference/paravision_water_phantom_geometry.json`` (five slices), with
a 16 x 16 matrix, the method name and echo count of the test,
``PVM_EncSpectroscopy`` ``No`` and a ``2dseq`` of 16-bit integers.  Every frame
has the constant value ``1000 + 100 * echo + 10 * slice``, so a swapped echo
and slice axis changes the labelled values; the read and phase axes are
covered by ``test_bruker_raw_versus_2dseq.py``.  The frame groups of the
frame-group position tests, such as the five fit parameters of a parameter map, take the
place of the echoes in this formula, and a third frame group adds its index
to the slice index.  The values are no scanner data.
Integers below 2**15 survive the float32 conversion of the loader exactly,
so the labelled frames are compared with exact equality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from brukerapi.dataset import Dataset

from hypermri.loading.scans import load_scan
from hypermri.reconstruction.strategies.bruker.seq2d import (
    Brukerseq2dRecoStrategy,
    _frame_group_axes,
)
from hypermri.vendors.bruker import BrukerHandler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "reference" / "paravision_water_phantom_geometry.json"
)
SERIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["series"][
    "head_prone/10"
]["paravision"]
N_SLICES = SERIES["method"]["PVM_SPackArrNSlices"]
MATRIX_SIZE = 16

# MGE stores its echoes along acquisition_idx; the other methods along echo_idx.
ECHO_DIMS = {
    "<Bruker:MGE>": "acquisition_idx",
    "<Bruker:MSME>": "echo_idx",
    "<Bruker:RARE>": "echo_idx",
    "<Bruker:FLASH>": "echo_idx",
    "<Bruker:T2_EPI>": "echo_idx",
}
FRAME_GROUP_TYPES = {"echo": "FG_ECHO", "slice": "FG_SLICE"}
FRAME_GROUP_ORDER_REASON = "frame group order differs from VisuFGOrderDesc"


@pytest.mark.parametrize("complex_axis", [0, 1, 2])
@pytest.mark.parametrize("cycles", [1, 2, 3])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_scanner_complex_frames_are_combined_once(
    tmp_path, write_paravision_parameters, complex_axis, cycles, precision
):
    """Invented integer frames preserve pairing, scaling and every cycle.

    The reference is the uncombined, scaled actual-reader array. Integer
    slopes and offsets make the comparison exact at both storage precisions.
    The complex group may precede, separate or follow echo and cycle groups.
    """
    groups = [(2, "FG_ECHO"), (cycles, "FG_CYCLE")]
    groups.insert(complex_axis, (2, "FG_COMPLEX"))
    shape = tuple(size for size, _ in groups)
    frames = np.arange(np.prod(shape)).reshape(shape) + 1
    pdata = tmp_path / "pdata" / "2"
    visu = _write_frames(
        pdata,
        write_paravision_parameters,
        frames,
        groups,
        frame_type=["REAL_IMAGE", "IMAGINARY_IMAGE"],
    )
    visu["VisuCoreDataSlope"] = list(range(1, frames.size + 1))
    visu["VisuCoreDataOffs"] = list(range(10, frames.size + 10))
    write_paravision_parameters(pdata / "visu_pars", visu)
    stored_bytes = (pdata / "2dseq").read_bytes()
    uncombined = Dataset(pdata / "2dseq", combine_complex=False).data
    axis = complex_axis + 2
    expected = np.take(uncombined, 0, axis=axis) + 1j * np.take(
        uncombined, 1, axis=axis
    )
    handler = BrukerHandler.__new__(BrukerHandler)
    handler.path = tmp_path
    handler._reconstruction_parameters_cache = {}
    actual = handler.load_2dseq_file(recon_num=2, data_precision=precision)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == (np.complex64 if precision is np.float32 else np.complex128)
    np.testing.assert_array_equal(
        handler.load_2dseq_file(recon_num=2, data_precision=precision), actual
    )
    assert (pdata / "2dseq").read_bytes() == stored_bytes


@pytest.mark.parametrize("n_echoes", [1, 3])
def test_mge_reconstruction_order_preserves_acquired_metadata(
    tmp_path, write_paravision_parameters, n_echoes
):
    """Scanner-first and raw-first MGE reconstructions have identical labels.

    Small synthetic complex samples use no FOV offset or block padding.
    Identical deterministic calculations must compare exactly .
    """
    import xarray as xr

    scan_dir = tmp_path / "1"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MGE>",
        n_echoes=n_echoes,
        frame_group_order=("echo", "slice"),
        method_parameters={
            "PVM_EncSteps1": list(range(-MATRIX_SIZE // 2, MATRIX_SIZE // 2))
        },
    )

    def load():
        scan = load_scan(scan_dir, mode="lazy", log_mode="critical")
        scan.handler.fid = np.arange(MATRIX_SIZE**2 * N_SLICES * n_echoes).astype(
            np.complex128
        )
        return scan

    first, second = load(), load()
    metadata = {
        name: np.array(getattr(first.handler, name), copy=True)
        for name in (
            "n_acquisition",
            "dwell_time_s",
            "bandwidth_hz",
            "sampling_time_axis_s",
        )
    }
    second.reconstruct("seq2d")
    options = {"postprocess_fid": False, "shift_xarray": False}
    direct = first.reconstruct("raw", **options)
    after_scanner = second.reconstruct("raw", **options)
    second.reconstruct("raw", data_precision="float64", **options)
    repeated = second.reconstruct("raw", **options)
    xr.testing.assert_identical(direct, after_scanner)
    xr.testing.assert_identical(direct, repeated)
    np.testing.assert_array_equal(
        direct.acquisition_time_s, np.arange(1, n_echoes + 1) * 0.01
    )
    if n_echoes > 1:
        fallback = _reconstruct_with_fallback(scan_dir)
        np.testing.assert_array_equal(
            fallback.acquisition_time_s, np.arange(1, n_echoes + 1) * 0.01
        )
    for scan in (first, second):
        for name, value in metadata.items():
            np.testing.assert_array_equal(getattr(scan.handler, name), value)
        assert not hasattr(scan.handler, "kspace_xarray")


def _expected_frames(n_echoes: int) -> np.ndarray:
    """Return the frame values with shape ``(echo, slice)``."""
    echo = np.arange(n_echoes)[:, None]
    slice_ = np.arange(N_SLICES)[None, :]
    return 1000 + 100 * echo + 10 * slice_


def _write_frames(
    pdata_dir: Path,
    write_parameters,
    frames: np.ndarray,
    frame_groups: list[tuple[int, str]],
    *,
    reco: dict | None = None,
    frame_type: str | list[str] = "MAGNITUDE_IMAGE",
) -> dict:
    """Write ``visu_pars``, ``reco`` and ``2dseq`` of one ``pdata`` folder.

    ``frame_groups`` lists the length and type of each ``VisuFGOrderDesc``
    entry, fastest first, and ``frames`` holds the constant value of each
    frame with one axis per entry in the same order.  ``reco`` defaults to
    the ``reco`` parameters of the fixture series.  Returns the written
    ``visu_pars``.
    """
    n_frames = frames.size
    # brukerapi reads the frame layout and scaling of 2dseq from these fields.
    visu_pars = {
        **SERIES["visu_pars"],
        "VisuCoreSize": [MATRIX_SIZE, MATRIX_SIZE],
        "VisuCoreDimDesc": ["spatial", "spatial"],
        "VisuCreatorVersion": "7.0.0",
        "VisuCoreWordType": "_16BIT_SGN_INT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreFrameCount": n_frames,
        "VisuFGOrderDescDim": len(frame_groups),
        "VisuFGOrderDesc": [
            f"({length}, <{group_type}>, <>, 0, 0)"
            for length, group_type in frame_groups
        ],
        "VisuCoreDataSlope": [1] * n_frames,
        "VisuCoreDataOffs": [0] * n_frames,
        "VisuCoreFrameType": frame_type,
    }
    write_parameters(pdata_dir / "visu_pars", visu_pars)
    write_parameters(pdata_dir / "reco", SERIES["reco"] if reco is None else reco)

    stored = np.broadcast_to(frames, (MATRIX_SIZE, MATRIX_SIZE, *frames.shape))
    # ParaVision stores the fastest axis first, which is Fortran order here.
    (pdata_dir / "2dseq").write_bytes(stored.astype("<i2").tobytes(order="F"))
    return visu_pars


def _write_reconstruction(
    pdata_dir: Path,
    write_parameters,
    *,
    n_echoes: int,
    frame_group_order: tuple[str, str],
    reco: dict | None = None,
) -> dict:
    """Write a ``pdata`` folder with echoes and slices.

    The frame groups are stored in ``frame_group_order``, fastest first.
    ``reco`` defaults to the ``reco`` parameters of the fixture series.
    Returns the written ``visu_pars``.
    """
    sizes = {"echo": n_echoes, "slice": N_SLICES}
    frames = _expected_frames(n_echoes)
    if frame_group_order == ("slice", "echo"):
        frames = frames.T
    return _write_frames(
        pdata_dir,
        write_parameters,
        frames,
        [(sizes[group], FRAME_GROUP_TYPES[group]) for group in frame_group_order],
        reco=reco,
    )


def _write_scan(
    scan_dir: Path,
    write_parameters,
    *,
    method_name: str,
    n_echoes: int,
    frame_group_order: tuple[str, str],
    method_parameters: dict | None = None,
) -> None:
    """Write a scan whose frame groups are stored in ``frame_group_order``.

    The scan-level ``visu_pars`` is a copy of ``pdata/1/visu_pars``, as in
    every local scan.  ``method_parameters`` replace or add ``method``
    parameters.
    """
    n_frames = n_echoes * N_SLICES
    method = {
        **SERIES["method"],
        "Method": method_name,
        "PVM_Matrix": [MATRIX_SIZE, MATRIX_SIZE],
        "PVM_EncMatrix": [MATRIX_SIZE, MATRIX_SIZE],
        "PVM_EncNReceivers": 1,
        "PVM_EncSpectroscopy": "No",
        "PVM_NEchoImages": n_echoes,
        "EffectiveTE": [10.0 * (echo + 1) for echo in range(n_echoes)],
        **(method_parameters or {}),
    }
    acqp = {
        **SERIES["acqp"],
        "ACQ_size": [2 * MATRIX_SIZE, MATRIX_SIZE],
        "NI": n_frames,
        "NR": 1,
    }
    write_parameters(scan_dir / "method", method)
    write_parameters(scan_dir / "acqp", acqp)
    visu_pars = _write_reconstruction(
        scan_dir / "pdata" / "1",
        write_parameters,
        n_echoes=n_echoes,
        frame_group_order=frame_group_order,
    )
    write_parameters(scan_dir / "visu_pars", visu_pars)


def _frames_along(data_array, dims: tuple[str, ...]) -> np.ndarray:
    """Return the frame values with one axis per label of ``dims``."""
    frame_dims = [dim for dim in data_array.dims if dim not in dims]
    high = data_array.max(dim=frame_dims).transpose(*dims)
    low = data_array.min(dim=frame_dims).transpose(*dims)
    np.testing.assert_array_equal(high.values, low.values)
    return high.values


def _labelled_frames(data_array, echo_dim: str) -> np.ndarray:
    """Return the frame values with shape ``(echo, slice)`` from the labels."""
    return _frames_along(data_array, (echo_dim, "slice_idx"))


def _reconstruct(scan_dir: Path, **reco_options):
    return load_scan(scan_dir, mode="lazy", log_mode="critical").reconstruct(
        "seq2d", **reco_options
    )


def _reconstruct_with_fallback(scan_dir: Path, recon_num: int = 1):
    handler = BrukerHandler(scan_dir)
    seq2d = handler.undo_reco_transposition(
        handler.load_2dseq_file(recon_num=recon_num), recon_num=recon_num
    ).squeeze()
    return Brukerseq2dRecoStrategy()._fallback_seq2d_scanner_data(
        handler, seq2d, {"recon_num": recon_num}, append_parameters=False
    )


METHOD_PARAMS = [
    pytest.param(name, id=name.strip("<>").split(":")[1]) for name in ECHO_DIMS
]


# --------------------------------------------------------------------------
# Frame groups of VisuFGOrderDesc
# --------------------------------------------------------------------------


ECHO_THEN_SLICE = "(5, <FG_ECHO>, <>, 0, 0) (4, <FG_SLICE>, <>, 0, 0)"
IMAGE_SIZE = [MATRIX_SIZE, MATRIX_SIZE]


@pytest.mark.parametrize(
    ("core_size", "frame_group_order", "shape", "echo_dim", "expected"),
    [
        (IMAGE_SIZE, None, (16, 16, 5), "echo_idx", {}),
        # Whitespace tokens, as BrukerHandler parses the visu_pars field.
        (
            IMAGE_SIZE,
            ["(5,", "<FG_ECHO>,", "<>,", "0,", "0)"]
            + ["(4,", "<FG_SLICE>,", "<>,", "0,", "0)"],
            (16, 16, 5, 4),
            "echo_idx",
            {2: "echo_idx", 3: "slice_idx"},
        ),
        (
            IMAGE_SIZE,
            "(4, <FG_SLICE>, <>, 0, 0) (3, <FG_CYCLE>, <>, 0, 0)",
            (16, 16, 4, 3),
            "echo_idx",
            {2: "slice_idx", 3: "repetition_idx"},
        ),
        (
            IMAGE_SIZE,
            ECHO_THEN_SLICE,
            (16, 16, 5, 4),
            "acquisition_idx",
            {2: "acquisition_idx", 3: "slice_idx"},
        ),
        # Frame groups with one element have no 2dseq axis.
        (
            IMAGE_SIZE,
            "(1, <FG_SLICE>, <>, 0, 0) (6, <FG_DIFFUSION>, <>, 0, 0)",
            (16, 16, 6),
            "echo_idx",
            {2: "bvalue_idx"},
        ),
        (
            IMAGE_SIZE,
            "(8, <FG_CARDIAC_MOVIE>, <>, 0, 2) (4, <FG_SLICE>, <>, 2, 2)",
            (16, 16, 8, 4),
            "echo_idx",
            {2: "acquisition_idx", 3: "slice_idx"},
        ),
        (
            IMAGE_SIZE,
            "(5, <FG_ISA>, <>, 0, 1) (5, <FG_SLICE>, <>, 1, 0)",
            (16, 16, 5, 5),
            "echo_idx",
            {2: "map_idx", 3: "slice_idx"},
        ),
        # Types without a label keep their axis for the size-based labelling.
        (
            IMAGE_SIZE,
            "(3, <FG_UNKNOWN>, <>, 0, 0) (4, <FG_SLICE>, <>, 0, 0)",
            (16, 16, 3, 4),
            "echo_idx",
            {3: "slice_idx"},
        ),
        # The loader combines FG_COMPLEX into complex values.
        (
            IMAGE_SIZE,
            "(2, <FG_COMPLEX>, <>, 0, 1) (5, <FG_SLICE>, <>, 1, 0)",
            (16, 16, 5),
            "echo_idx",
            {2: "slice_idx"},
        ),
        # Frame dimensions with one element are squeezed.
        (
            [256, 1],
            "(3, <FG_CYCLE>, <>, 0, 0)",
            (256, 3),
            "echo_idx",
            {1: "repetition_idx"},
        ),
        (
            [16, 16, 8],
            "(3, <FG_CYCLE>, <>, 0, 0)",
            (16, 16, 8, 3),
            "echo_idx",
            {3: "repetition_idx"},
        ),
        # Axes after the frame groups, such as appended channels, stay free.
        (
            IMAGE_SIZE,
            ECHO_THEN_SLICE,
            (16, 16, 5, 4, 2),
            "echo_idx",
            {2: "echo_idx", 3: "slice_idx"},
        ),
    ],
)
def test_frame_group_axes_follow_the_storage_position(
    core_size, frame_group_order, shape, echo_dim, expected
):
    visu_pars = {"VisuCoreSize": core_size, "VisuFGOrderDesc": frame_group_order}

    assert _frame_group_axes(visu_pars, shape, echo_dim=echo_dim) == expected


@pytest.mark.parametrize(
    ("group_type", "movie_dim", "dim"),
    [
        ("FG_MOVIE", "acquisition_idx", "acquisition_idx"),
        ("FG_MOVIE", "inversion_idx", "inversion_idx"),
        ("FG_IRMODE", "inversion_idx", "acquisition_idx"),
        ("FG_CARDIAC_MOVIE", "acquisition_idx", "acquisition_idx"),
        ("FG_FLOW", "acquisition_idx", "acquisition_idx"),
        ("FG_ISA", "acquisition_idx", "map_idx"),
        ("FG_DTI", "acquisition_idx", "map_idx"),
    ],
)
def test_frame_group_axes_label_loops_and_maps_by_their_type(
    group_type, movie_dim, dim
):
    """Regression test: every frame-group type of ParaVision has a label.

    Previously these types had no label and were labelled by their size.
    Only ``FG_MOVIE`` depends on the method, through ``movie_dim``.
    """
    visu_pars = {
        "VisuCoreSize": IMAGE_SIZE,
        "VisuFGOrderDesc": f"(3, <{group_type}>, <>, 0, 0) (4, <FG_SLICE>, <>, 0, 0)",
    }

    axes = _frame_group_axes(visu_pars, (16, 16, 3, 4), movie_dim=movie_dim)

    assert axes == {2: dim, 3: "slice_idx"}


@pytest.mark.parametrize(
    "frame_group_order",
    [
        "(2, <FG_SLICE>, <>, 0, 0) (3, <FG_SLICE>, <>, 0, 0)",
        "(2, <FG_MOVIE>, <>, 0, 0) (3, <FG_IRMODE>, <>, 0, 0)",
        "(2, <FG_ISA>, <>, 0, 0) (3, <FG_DTI>, <>, 0, 0)",
    ],
    ids=["slices", "movie-and-inversion-mode", "two-maps"],
)
def test_frame_group_axes_reject_two_groups_with_one_label(frame_group_order):
    visu_pars = {"VisuCoreSize": IMAGE_SIZE, "VisuFGOrderDesc": frame_group_order}

    with pytest.raises(ValueError, match="more than one frame group labelled"):
        _frame_group_axes(visu_pars, (16, 16, 2, 3))


@pytest.mark.parametrize(
    "shape",
    [(16, 16, 4, 5), (16, 16, 5), (16, 5, 4)],
    ids=["swapped", "missing-group", "missing-frame-dim"],
)
def test_frame_group_axes_reject_axes_without_the_frame_group_lengths(shape):
    visu_pars = {"VisuCoreSize": IMAGE_SIZE, "VisuFGOrderDesc": ECHO_THEN_SLICE}

    with pytest.raises(ValueError, match="does not hold the frame groups"):
        _frame_group_axes(visu_pars, shape)


# --------------------------------------------------------------------------
# Labelled 2dseq reconstructions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_slice_before_echo_stays_on_the_strict_path(
    tmp_path, write_paravision_parameters, method_name
):
    """Characterization: the strict order matches, as before."""
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        n_echoes=5,
        frame_group_order=("slice", "echo"),
    )

    data_array = _reconstruct(scan_dir)

    assert "seq2d_fallback_reason" not in data_array.attrs
    np.testing.assert_array_equal(
        _labelled_frames(data_array, ECHO_DIMS[method_name]), _expected_frames(5)
    )


@pytest.mark.parametrize("group_type", ["FG_ECHO", "FG_UNLISTED"])
def test_strict_path_rejects_unverified_nonspatial_sizes(
    tmp_path, write_paravision_parameters, group_type
):
    """Equal axis counts alone cannot establish the strict label contract."""
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=6 if group_type == "FG_ECHO" else 4,
        frame_group_order=("slice", "echo"),
    )
    _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        _expected_frames(4).T,
        [(N_SLICES, "FG_SLICE"), (4, group_type)],
    )

    result = _reconstruct(scan_dir)

    assert "seq2d_fallback_reason" in result.attrs
    assert (
        "strict size mismatch" if group_type == "FG_ECHO" else "unmapped frame group"
    ) in result.attrs["seq2d_fallback_reason"]
    np.testing.assert_array_equal(
        _labelled_frames(result, "echo_idx"), _expected_frames(4)
    )


@pytest.mark.parametrize(
    "group, dim, parameters",
    [
        ("FG_CYCLE", "repetition_idx", {"PVM_NRepetitions": 6}),
        ("FG_DIFFUSION", "bvalue_idx", {"PVM_DwEffBval": [0, 10, 20, 30, 40, 50]}),
        ("FG_MOVIE", "inversion_idx", {"PVM_FairTIR_Arr": [20, 40, 60, 80, 100, 120]}),
    ],
)
def test_strict_path_checks_every_nonspatial_axis_length(
    tmp_path, write_paravision_parameters, group, dim, parameters
):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters=parameters,
    )
    _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        _expected_frames(4).T,
        [(N_SLICES, "FG_SLICE"), (4, group)],
    )
    result = _reconstruct(scan_dir)
    assert f"{dim}: stored 4, expected 6" in result.attrs["seq2d_fallback_reason"]
    np.testing.assert_array_equal(_labelled_frames(result, dim), _expected_frames(4))


def test_strict_coordinate_length_mismatch_is_reported():
    from hypermri.reconstruction.strategies.bruker.seq2d import (
        _assign_seq2d_coordinates,
    )

    data = xr.DataArray(np.ones(4), dims="echo_idx")
    with pytest.raises(ValueError, match="echo_idx.*6.*4"):
        _assign_seq2d_coordinates(
            {"echo_idx": ("echo_idx", np.arange(6), {"unit": "ms"})}, data
        )


def test_scanner_result_records_both_data_folder_and_parameter_selection(
    tmp_path, write_paravision_parameters
):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=4,
        frame_group_order=("slice", "echo"),
    )
    handler = BrukerHandler(scan_dir)
    strategy = Brukerseq2dRecoStrategy()
    result = strategy.reconstruct(
        handler, recon_num=1, set_reco_file=False, append_parameters=False
    )
    assert result.attrs["bruker_reconstruction_number"] == 1
    assert result.attrs["bruker_reco_parameter_source"] == "pdata/1/reco"
    handler.reco_parameter_source = "pdata/2/reco"
    repeated = strategy.reconstruct(
        handler, recon_num=1, set_reco_file=False, append_parameters=False
    )
    assert repeated.attrs["bruker_reconstruction_number"] == 1
    assert repeated.attrs["bruker_reco_parameter_source"] == "pdata/2/reco"
    assert result.attrs["bruker_reco_parameter_source"] == "pdata/1/reco"
    np.testing.assert_array_equal(result.values, repeated.values)


@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_echo_before_slice_with_different_sizes(
    tmp_path, write_paravision_parameters, method_name
):
    """Characterization: different sizes label the axes, as before."""
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        n_echoes=6,
        frame_group_order=("echo", "slice"),
    )

    data_array = _reconstruct(scan_dir)

    np.testing.assert_array_equal(
        _labelled_frames(data_array, ECHO_DIMS[method_name]), _expected_frames(6)
    )


@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_echo_before_slice_with_equal_sizes(
    tmp_path, write_paravision_parameters, method_name
):
    """Regression: equal echo and slice counts follow VisuFGOrderDesc.

    Previously the strict path labelled the stored echo axis as slices for
    every method here: echo 1 over the slices read 1010, 1110, 1210, 1310,
    1410 instead of 1100, 1110, 1120, 1130, 1140.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        n_echoes=5,
        frame_group_order=("echo", "slice"),
    )

    data_array = _reconstruct(scan_dir)

    np.testing.assert_array_equal(
        _labelled_frames(data_array, ECHO_DIMS[method_name]), _expected_frames(5)
    )
    assert data_array.attrs["seq2d_fallback_reason"] == FRAME_GROUP_ORDER_REASON


@pytest.mark.parametrize(
    "frame_group_order",
    [("echo", "slice"), ("slice", "echo")],
    ids=["echo-first", "slice-first"],
)
@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_fallback_follows_visu_frame_group_order(
    tmp_path, write_paravision_parameters, method_name, frame_group_order
):
    """Regression: the fallback labels the echo and slice axes where stored.

    Previously its fixed search order swapped equal echo and slice counts for
    MGE and MSME with slices stored first and for RARE, FLASH and T2_EPI with
    echoes stored first.  An intermediate version searched the labels
    in the stored order; now it labels the stored axes from their frame
    groups.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        n_echoes=5,
        frame_group_order=frame_group_order,
    )

    data_array = _reconstruct_with_fallback(scan_dir)

    np.testing.assert_array_equal(
        _labelled_frames(data_array, ECHO_DIMS[method_name]), _expected_frames(5)
    )


# --------------------------------------------------------------------------
# Parameters of each reconstruction
# --------------------------------------------------------------------------

FRAME_GROUP_ORDERS = [
    pytest.param(("slice", "echo"), ("echo", "slice"), id="pdata2-echo-first"),
    pytest.param(("echo", "slice"), ("slice", "echo"), id="pdata2-slice-first"),
]


def _write_two_reconstructions(
    scan_dir: Path,
    write_parameters,
    *,
    method_name: str,
    pdata1_order: tuple[str, str],
    pdata2_order: tuple[str, str],
    pdata2_reco: dict | None = None,
) -> None:
    """Write ``pdata/1`` and a ``pdata/2`` with other frame groups.

    Both reconstructions have five echoes and five slices, so only the
    frame-group order of their own ``visu_pars`` tells the axes apart.
    """
    _write_scan(
        scan_dir,
        write_parameters,
        method_name=method_name,
        n_echoes=N_SLICES,
        frame_group_order=pdata1_order,
    )
    _write_reconstruction(
        scan_dir / "pdata" / "2",
        write_parameters,
        n_echoes=N_SLICES,
        frame_group_order=pdata2_order,
        reco=pdata2_reco,
    )


@pytest.mark.parametrize(("pdata1_order", "pdata2_order"), FRAME_GROUP_ORDERS)
@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_frame_groups_follow_the_selected_reconstruction(
    tmp_path, write_paravision_parameters, method_name, pdata1_order, pdata2_order
):
    """Regression test: ``pdata/2`` is labelled from its own frame groups.

    Previously the strategy read ``VisuFGOrderDesc`` of the scan-level
    ``visu_pars``, which describes ``pdata/1``.  With equal echo and slice
    counts it swapped the echo and slice labels of ``pdata/2``: echo 1 over
    the slices read 1010, 1110, 1210, 1310, 1410 instead of 1100, 1110,
    1120, 1130, 1140.
    """
    scan_dir = tmp_path / "7"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        pdata1_order=pdata1_order,
        pdata2_order=pdata2_order,
    )
    echo_dim = ECHO_DIMS[method_name]

    pdata1 = _reconstruct(scan_dir)
    pdata2 = _reconstruct(scan_dir, recon_num=2)
    loaded = BrukerHandler(scan_dir).load_reconstruction(
        recon_num=2, set_as_primary=False
    )

    for data_array in (pdata1, pdata2, loaded):
        np.testing.assert_array_equal(
            _labelled_frames(data_array, echo_dim), _expected_frames(N_SLICES)
        )
    # The strict labels put slices before echoes; the other order falls back.
    for data_array, order in ((pdata1, pdata1_order), (pdata2, pdata2_order)):
        if order == ("slice", "echo"):
            assert "seq2d_fallback_reason" not in data_array.attrs
        else:
            assert data_array.attrs["seq2d_fallback_reason"] == (
                FRAME_GROUP_ORDER_REASON
            )


@pytest.mark.parametrize(("pdata1_order", "pdata2_order"), FRAME_GROUP_ORDERS)
@pytest.mark.parametrize("method_name", METHOD_PARAMS)
def test_fallback_follows_the_selected_reconstruction(
    tmp_path, write_paravision_parameters, method_name, pdata1_order, pdata2_order
):
    """Regression test: the fallback reads ``recon_num`` of the options.

    Previously it searched the labels in the frame-group order of the
    scan-level ``visu_pars`` for every reconstruction.
    """
    scan_dir = tmp_path / "7"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        method_name=method_name,
        pdata1_order=pdata1_order,
        pdata2_order=pdata2_order,
    )

    data_array = _reconstruct_with_fallback(scan_dir, recon_num=2)

    np.testing.assert_array_equal(
        _labelled_frames(data_array, ECHO_DIMS[method_name]),
        _expected_frames(N_SLICES),
    )


def test_fallback_reads_the_output_channels_of_the_selected_reconstruction(
    tmp_path, write_paravision_parameters
):
    """Regression test: ``RecoNumOutputChan`` of ``pdata/2`` labels channels.

    No local scan has more than one output channel and ``VisuFGOrderDesc``
    has no channel frame group, so the two channels are appended to the
    loaded array instead of being written to ``2dseq``.  Previously the
    fallback read ``RecoNumOutputChan`` from the ``reco`` attribute of the
    handler, which holds ``pdata/1/reco`` without the parameter, and
    labelled the channel axis ``acquisition_idx``.
    """
    scan_dir = tmp_path / "7"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        pdata1_order=("slice", "echo"),
        pdata2_order=("slice", "echo"),
        pdata2_reco={**SERIES["reco"], "RecoNumOutputChan": 2},
    )
    handler = BrukerHandler(scan_dir)
    seq2d = handler.undo_reco_transposition(
        handler.load_2dseq_file(recon_num=2), recon_num=2
    ).squeeze()
    channels = np.stack([seq2d, seq2d + 1], axis=-1)

    data_array = Brukerseq2dRecoStrategy()._fallback_seq2d_scanner_data(
        handler, channels, {"recon_num": 2}, append_parameters=False
    )

    assert "RecoNumOutputChan" not in handler.reco
    assert data_array.sizes["channel_idx"] == 2
    assert "acquisition_idx" not in data_array.dims
    np.testing.assert_array_equal(
        _labelled_frames(data_array.isel(channel_idx=0), "echo_idx"),
        _expected_frames(N_SLICES),
    )


def test_transposition_follows_the_selected_reconstruction(
    tmp_path, write_paravision_parameters
):
    """Regression test: ``pdata/2`` is swapped by its own ``reco``.

    The fixture series stores ``pdata/1`` without a swap.  ``pdata/2`` has
    ``RECO_transposition`` 1 in every frame and stores each frame with read
    and phase swapped.  Previously both paths used the
    ``RECO_transposition`` of ``pdata/1/reco`` that the handler read on
    construction and returned every frame of ``pdata/2`` transposed.
    """
    scan_dir = tmp_path / "7"
    n_frames = N_SLICES * N_SLICES
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        pdata1_order=("slice", "echo"),
        pdata2_order=("slice", "echo"),
        pdata2_reco={"RECO_transposition": [1] * n_frames},
    )
    # Distinct values in every voxel of a frame, (read, phase) order.
    in_plane = np.arange(MATRIX_SIZE**2).reshape(MATRIX_SIZE, MATRIX_SIZE)
    frames = _expected_frames(N_SLICES).T  # (slice, echo), slices fastest
    stored = in_plane.T[:, :, None, None] + frames
    (scan_dir / "pdata" / "2" / "2dseq").write_bytes(
        stored.astype("<i2").tobytes(order="F")
    )
    expected = in_plane[:, :, None, None] + frames

    reconstructed = _reconstruct(scan_dir, recon_num=2)
    loaded = BrukerHandler(scan_dir).load_reconstruction(
        recon_num=2, set_as_primary=False
    )

    for data_array in (reconstructed, loaded):
        labelled = data_array.squeeze(drop=True).transpose(
            "read_idx", "phase_idx", "slice_idx", "echo_idx"
        )
        np.testing.assert_array_equal(labelled.values, expected)


@pytest.mark.parametrize("fallback", [False, True])
def test_geometry_follows_selected_reconstruction_without_handler_mutation(
    tmp_path, write_paravision_parameters, fallback
):
    """Selected positions, extent, orientation and swap own voxel geometry.

    Invented pdata/2 geometry has 2 mm read spacing and 3 mm phase spacing,
    with read along DICOM rows. The 1e-9 mm tolerance covers roundoff only.
    """
    scan_dir = tmp_path / "7"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        pdata1_order=("slice", "echo"),
        pdata2_order=("slice", "echo"),
        pdata2_reco={"RECO_transposition": [1] * (N_SLICES**2)},
    )
    path = scan_dir / "pdata" / "2" / "visu_pars"
    from hypermri.vendors.bruker import BrukerHandler

    handler = BrukerHandler(scan_dir)
    _, selected = handler.load_reconstruction_parameters(2)
    selected = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in selected.items()
    }
    selected.update(
        {
            "VisuCorePosition": [[10.0, 20.0, 30.0 + s] for s in range(N_SLICES)],
            "VisuCoreOrientation": [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * N_SLICES,
            "VisuCoreExtent": [3.0 * MATRIX_SIZE, 2.0 * MATRIX_SIZE],
            "VisuCoreFrameThickness": 1.0,
        }
    )
    write_paravision_parameters(path, selected)
    handler = BrukerHandler(scan_dir)
    before = np.array(handler.visucore_position, copy=True)
    if fallback:
        stored = handler.undo_reco_transposition(
            handler.load_2dseq_file(recon_num=2), recon_num=2
        ).squeeze()
        result = Brukerseq2dRecoStrategy()._fallback_seq2d_scanner_data(
            handler, stored, {"recon_num": 2}, append_parameters=False
        )
    else:
        result = Brukerseq2dRecoStrategy().reconstruct(
            handler, recon_num=2, append_parameters=False
        )
    np.testing.assert_allclose(result.x[:, 0, 0], 10.0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        result.y[:, 0, 0], 20.0 + 2.0 * np.arange(MATRIX_SIZE), atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        result.x[0, :, 0], 10.0 + 3.0 * np.arange(MATRIX_SIZE), atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        result.z[0, 0, :], 30.0 + np.arange(N_SLICES), atol=1e-9, rtol=0
    )
    np.testing.assert_array_equal(handler.visucore_position, before)
    assert not handler.transpose_read_phase


# --------------------------------------------------------------------------
# Positions of the frame groups
# --------------------------------------------------------------------------


def _write_parameter_map(scan_dir: Path, write_parameters) -> None:
    """Write an MSME scan with a ParaVision parameter map in ``pdata/2``.

    ``pdata/1`` stores the five slices and then six echoes.  ``pdata/2``
    stores five fit parameters in ``FG_ISA`` and then the five slices, like
    the T2 map of a local MSME scan.  The frame of fit parameter ``p`` and
    slice ``s`` has the value ``1000 + 100 * p + 10 * s``.
    """
    _write_scan(
        scan_dir,
        write_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=6,
        frame_group_order=("slice", "echo"),
    )
    _write_frames(
        scan_dir / "pdata" / "2",
        write_parameters,
        _expected_frames(N_SLICES),
        [(N_SLICES, "FG_ISA"), (N_SLICES, "FG_SLICE")],
    )


def test_parameter_map_labels_its_slices_at_their_position(
    tmp_path, write_paravision_parameters
):
    """Regression test: a parameter map before the slices.

    Previously the strict path compared only the order of the mapped frame
    groups, ``FG_SLICE`` alone, and labelled the fit parameters
    ``slice_idx`` and the slices ``echo_idx``; the fallback labelled the
    fit parameters ``slice_idx`` and the slices ``acquisition_idx``.  In an
    intermediate version the fit parameters had the size-based label
    ``acquisition_idx``.
    """
    scan_dir = tmp_path / "7"
    _write_parameter_map(scan_dir, write_paravision_parameters)

    reconstructed = _reconstruct(scan_dir, recon_num=2)
    loaded = BrukerHandler(scan_dir).load_reconstruction(
        recon_num=2, set_as_primary=False
    )
    fallback = _reconstruct_with_fallback(scan_dir, recon_num=2)

    for data_array in (reconstructed, loaded, fallback):
        assert data_array.dims == ("read_idx", "phase_idx", "map_idx", "slice_idx")
        np.testing.assert_array_equal(
            _labelled_frames(data_array, "map_idx"), _expected_frames(N_SLICES)
        )
    for data_array in (reconstructed, loaded):
        assert data_array.attrs["seq2d_fallback_reason"] == FRAME_GROUP_ORDER_REASON


@pytest.mark.parametrize(
    ("group_type", "dim"),
    [("FG_ECHO", "echo_idx"), ("FG_CYCLE", "repetition_idx")],
)
def test_fallback_labels_frame_groups_that_the_method_does_not_size(
    tmp_path, write_paravision_parameters, group_type, dim
):
    """Regression test: a frame group labels its axis by its type.

    The method and ``acqp`` describe one echo and one repetition, but
    ``pdata/1`` stores the five slices and then three frames of
    ``group_type``, so the strict labels lack a dimension and the fallback
    labels the array.  Previously the fallback searched a label of size 3
    and, finding none, labelled the axis ``acquisition_idx``.  Two local
    spectroscopic scans store their repetitions or echoes in such a frame
    group.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
    )
    visu_pars = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        _expected_frames(3).T,
        [(N_SLICES, "FG_SLICE"), (3, group_type)],
    )
    write_paravision_parameters(scan_dir / "visu_pars", visu_pars)

    data_array = _reconstruct(scan_dir)

    assert data_array.attrs["seq2d_fallback_reason"] == "strict dimension mismatch"
    np.testing.assert_array_equal(
        _labelled_frames(data_array, dim), _expected_frames(3)
    )


def test_complex_frames_take_no_frame_group_position(
    tmp_path, write_paravision_parameters
):
    """Characterization: ``FG_COMPLEX`` has no axis in the array.

    The loader combines the real and imaginary frames into complex values,
    so the echo and slice axes follow the frame dimensions directly.  The
    imaginary part of each frame is the real part plus 1000.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=N_SLICES,
        frame_group_order=("echo", "slice"),
    )
    frames = _expected_frames(N_SLICES)
    visu_pars = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        np.stack([frames, frames + 1000]),
        [(2, "FG_COMPLEX"), (N_SLICES, "FG_ECHO"), (N_SLICES, "FG_SLICE")],
        frame_type=["REAL_IMAGE", "IMAGINARY_IMAGE"],
    )
    write_paravision_parameters(scan_dir / "visu_pars", visu_pars)

    data_array = _reconstruct(scan_dir)

    assert np.iscomplexobj(data_array.values)
    np.testing.assert_array_equal(_labelled_frames(data_array.real, "echo_idx"), frames)
    np.testing.assert_array_equal(
        _labelled_frames(data_array.imag, "echo_idx"), frames + 1000
    )


# --------------------------------------------------------------------------
# Loops and maps of ParaVision frame groups
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group_type", "dim"),
    [
        ("FG_MOVIE", "acquisition_idx"),
        ("FG_IRMODE", "acquisition_idx"),
        ("FG_CARDIAC_MOVIE", "acquisition_idx"),
        ("FG_FLOW", "acquisition_idx"),
        ("FG_ISA", "map_idx"),
        ("FG_DTI", "map_idx"),
    ],
)
def test_fallback_labels_loops_and_maps_by_their_frame_group(
    tmp_path, write_paravision_parameters, group_type, dim
):
    """Regression test: a loop or map is not labelled by its size.

    The method describes three averages and ``pdata/1`` stores the five
    slices and then three frames of ``group_type``, so the strict labels
    lack a dimension and the fallback labels the array.  Previously the
    fallback labelled the three frames ``average_idx`` by their size, as it
    labelled the selective and non-selective inversions of a local FAIR EPI
    scan and the label and control frames of a local CASL EPI scan, which
    the raw reconstruction labels ``acquisition_idx``.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"PVM_NAverages": 3},
    )
    visu_pars = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        _expected_frames(3).T,
        [(N_SLICES, "FG_SLICE"), (3, group_type)],
    )
    write_paravision_parameters(scan_dir / "visu_pars", visu_pars)

    data_array = _reconstruct(scan_dir)

    assert data_array.attrs["seq2d_fallback_reason"] == "strict dimension mismatch"
    if dim == "map_idx":
        assert data_array.dims == ("read_idx", "phase_idx", "slice_idx", dim)
    else:
        from hypermri.reconstruction.layout import IMAGING_DIM_ORDER

        assert data_array.dims == IMAGING_DIM_ORDER
    np.testing.assert_array_equal(
        _labelled_frames(data_array, dim), _expected_frames(3)
    )


FAIR_INVERSION_TIMES_MS = [100.0, 1000.0]


def _write_fair_scan(scan_dir: Path, write_parameters, frame_groups) -> None:
    """Write a FAIR RARE scan with two inversion times.

    ``frame_groups`` lists the length and type of each ``VisuFGOrderDesc``
    entry, fastest first, from ``FG_IRMODE`` (the selective and the
    non-selective inversion), ``FG_MOVIE`` (the inversion times) and
    ``FG_SLICE``.  The frame of inversion mode ``m``, inversion time ``t``
    and slice ``s`` has the value ``1000 + 100 * m + 10 * t + s``; without
    ``FG_IRMODE`` only the inversion mode ``m = 0`` is stored.
    """
    _write_scan(
        scan_dir,
        write_parameters,
        method_name="<Bruker:FAIR_RARE>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"PVM_FairTIR_Arr": FAIR_INVERSION_TIMES_MS},
    )
    group_types = [group_type for _, group_type in frame_groups]
    frames = _fair_frames()
    stored_types = ["FG_IRMODE", "FG_MOVIE", "FG_SLICE"]
    if "FG_IRMODE" not in group_types:
        frames = frames[0]
        stored_types.remove("FG_IRMODE")
    frames = np.transpose(frames, [stored_types.index(t) for t in group_types])
    visu_pars = _write_frames(
        scan_dir / "pdata" / "1", write_parameters, frames, frame_groups
    )
    write_parameters(scan_dir / "visu_pars", visu_pars)


def _fair_frames() -> np.ndarray:
    """Return the FAIR frame values with shape ``(mode, inversion, slice)``."""
    mode = np.arange(2)[:, None, None]
    inversion = np.arange(len(FAIR_INVERSION_TIMES_MS))[None, :, None]
    slice_ = np.arange(N_SLICES)[None, None, :]
    return 1000 + 100 * mode + 10 * inversion + slice_


def test_fair_inversion_mode_and_inversion_times_are_not_swapped(
    tmp_path, write_paravision_parameters
):
    """Regression test: equal FAIR loop sizes follow the frame groups.

    The inversion modes and the two inversion times both have two frames,
    stored in this order as in a local FAIR RARE scan.  Previously the
    fallback labelled the inversion modes ``inversion_idx`` by their size
    and the inversion times ``acquisition_idx``.  The raw reconstruction
    labels the inversion modes ``acquisition_idx``.
    """
    scan_dir = tmp_path / "7"
    _write_fair_scan(
        scan_dir,
        write_paravision_parameters,
        [(2, "FG_IRMODE"), (2, "FG_MOVIE"), (N_SLICES, "FG_SLICE")],
    )

    data_array = _reconstruct(scan_dir)

    from hypermri.reconstruction.layout import IMAGING_DIM_ORDER

    assert data_array.dims == IMAGING_DIM_ORDER
    np.testing.assert_array_equal(
        _frames_along(data_array, ("acquisition_idx", "inversion_idx", "slice_idx")),
        _fair_frames(),
    )
    np.testing.assert_array_equal(
        data_array["inversion_idx"].values, FAIR_INVERSION_TIMES_MS
    )


def test_fair_inversion_times_stay_on_the_strict_path(
    tmp_path, write_paravision_parameters
):
    """Characterization: FAIR inversion times without inversion modes.

    The slices and then the inversion times are stored, as in a local FAIR
    RARE scan with non-selective inversions only, so the strict labels match
    the frame groups before and after the change.
    """
    scan_dir = tmp_path / "7"
    _write_fair_scan(
        scan_dir,
        write_paravision_parameters,
        [(N_SLICES, "FG_SLICE"), (2, "FG_MOVIE")],
    )

    data_array = _reconstruct(scan_dir)

    assert "seq2d_fallback_reason" not in data_array.attrs
    np.testing.assert_array_equal(
        _frames_along(data_array, ("inversion_idx", "slice_idx")),
        _fair_frames()[0],
    )


@pytest.mark.parametrize("declared_times", [None, [25.0, 200.0, 800.0]])
@pytest.mark.parametrize("single_row", [False, True])
def test_t1_epi_inversion_times_use_ir_time_without_fair_slice_offsets(
    tmp_path, write_paravision_parameters, declared_times, single_row
):
    scan_dir = tmp_path / "7"
    times = [20.0, 180.0, 720.0]
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:T1_EPI>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"IRTime": [times] if single_row else times},
    )
    frames = 1000 + np.arange(N_SLICES)[:, None] + 100 * np.arange(3)[None, :]
    visu = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (3, "FG_MOVIE")],
    )
    if declared_times is not None:
        visu.update(
            {
                "VisuFGOrderDesc": [
                    f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                    "(3, <FG_MOVIE>, <>, 0, 1)",
                ],
                "VisuGroupDepVals": ["(<VisuAcqInversionTime>, 0)"],
                "VisuAcqInversionTime": declared_times,
            }
        )
        write_paravision_parameters(scan_dir / "pdata" / "1" / "visu_pars", visu)
    scan = load_scan(scan_dir, mode="lazy", log_mode="critical")
    assert scan.handler.n_inversion_time == 3
    np.testing.assert_array_equal(scan.handler.inversion_time_ms, times)
    result = scan.reconstruct("seq2d")
    np.testing.assert_array_equal(
        result.inversion_idx, times if declared_times is None else declared_times
    )
    assert result.inversion_idx.dims == ("inversion_idx",)
    assert result.inversion_idx.attrs["unit"] == "ms"
    np.testing.assert_array_equal(
        _frames_along(result, ("slice_idx", "inversion_idx")), frames
    )

    from hypermri.reconstruction.strategies.bruker.cartesian import (
        BrukerCartesianRawRecoStrategy,
    )
    from hypermri.reconstruction.strategies.bruker.epi import BrukerEpiRawRecoStrategy

    raw = np.broadcast_to(
        np.arange(3)[None, None, None, :, None], (2, 3, N_SLICES, 3, 1)
    ).copy()
    labelled, dims = BrukerEpiRawRecoStrategy._semantic_2d_layout(scan.handler, raw)
    coords = BrukerCartesianRawRecoStrategy._coords(scan.handler, dims, labelled.shape)
    np.testing.assert_array_equal(coords["inversion_idx"][1], times)
    np.testing.assert_array_equal(labelled, raw)
    assert "acquisition_idx" not in dims


@pytest.mark.parametrize("force_fallback", [False, True])
def test_echo_coordinates_use_selected_reconstruction_dependency_window(
    tmp_path, write_paravision_parameters, force_fallback
):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=4,
        frame_group_order=("slice", "echo"),
    )
    frames = _expected_frames(4).T
    visu = _write_frames(
        scan_dir / "pdata" / "2",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (4, "FG_ECHO")],
    )
    visu.update(
        {
            "VisuFGOrderDesc": [
                f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                "(4, <FG_ECHO>, <>, 0, 1)",
            ],
            "VisuGroupDepVals": ["(<VisuAcqEchoTime>, 1)"],
            "VisuAcqEchoTime": [999.0, 12.0, 24.0, 48.0, 96.0],
        }
    )
    write_paravision_parameters(scan_dir / "pdata" / "2" / "visu_pars", visu)
    handler = BrukerHandler(scan_dir)
    strategy = Brukerseq2dRecoStrategy()
    if force_fallback:
        stored = handler.undo_reco_transposition(
            handler.load_2dseq_file(recon_num=2), recon_num=2
        ).squeeze()
        result = strategy._fallback_seq2d_scanner_data(
            handler, stored, {"recon_num": 2}
        )
    else:
        result = strategy.reconstruct(handler, recon_num=2)
    np.testing.assert_allclose(
        result.echo_idx, [0.012, 0.024, 0.048, 0.096], rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(result.echo_time_s, result.echo_idx)
    assert result.echo_idx.attrs["source_parameter"] == "VisuAcqEchoTime"
    assert result.echo_idx.attrs["unit"] == "s"
    np.testing.assert_array_equal(
        _labelled_frames(result, "echo_idx"), _expected_frames(4)
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_map_and_cycle_labels_use_their_own_parameter_offsets(
    tmp_path, write_paravision_parameters, wrapped
):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
    )
    frames = np.arange(N_SLICES * 2 * 3).reshape(N_SLICES, 2, 3)
    visu = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (2, "FG_CYCLE"), (3, "FG_ISA")],
    )
    visu.update(
        {
            "VisuFGOrderDesc": [
                f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                "(2, <FG_CYCLE>, <>, 0, 1)",
                "(3, <FG_ISA>, <>, 1, 1)",
            ],
            "VisuGroupDepVals": [
                "(<VisuFGElemComment>, 0)",
                "(<VisuFGElemComment>, 2)",
            ],
            "VisuFGElemComment": [
                "<cycle one>",
                "<cycle two>",
                "<T1>",
                "<M0>",
                "<Fit chi squared>",
            ],
        }
    )
    write_paravision_parameters(scan_dir / "pdata" / "1" / "visu_pars", visu)
    if wrapped:
        path = scan_dir / "pdata" / "1" / "visu_pars"
        text = path.read_text().replace(
            "(<VisuFGElemComment>, 0) (<VisuFGElemComment>, 2)",
            "(<VisuFGElemComment>, 0) \n(<VisuFGElemComment>, 2)",
        )
        text = text.replace(
            "<cycle one> <cycle two> <T1> <M0> <Fit chi squared>",
            "<cycle one> <cycle two> \n<T1> <M0> <Fit chi squared>",
        )
        path.write_text(text)
    result = _reconstruct(scan_dir)
    assert result.map_idx.values.tolist() == ["T1", "M0", "Fit chi squared"]
    assert result.frame_label_repetition_idx.values.tolist() == [
        "cycle one",
        "cycle two",
    ]
    np.testing.assert_array_equal(
        _frames_along(result, ("slice_idx", "repetition_idx", "map_idx")), frames
    )


@pytest.mark.parametrize("force_fallback", [False, True])
def test_selected_map_ids_preserve_display_labels_and_dependency_offsets(
    tmp_path, write_paravision_parameters, force_fallback
):
    """Machine IDs supplement readable labels without changing frames."""
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
    )
    frames = np.arange(N_SLICES * 2 * 3).reshape(N_SLICES, 2, 3)
    visu = _write_frames(
        scan_dir / "pdata" / "2",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (2, "FG_CYCLE"), (3, "FG_DTI")],
    )
    visu.update(
        {
            "VisuFGOrderDesc": [
                f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                "(2, <FG_CYCLE>, <>, 0, 1)",
                "(3, <FG_DTI>, <>, 1, 2)",
            ],
            "VisuGroupDepVals": [
                "(<VisuFGElemComment>, 0)",
                "(<VisuFGElemId>, 1)",
                "(<VisuFGElemComment>, 2)",
            ],
            "VisuFGElemId": ["<unused>", "<DTI_FA>", "<DTI_TRACE>", "<DTI_A0>"],
            "VisuFGElemComment": [
                "<cycle one>",
                "<cycle two>",
                "<Fractional Anisotropy>",
                "<Trace>",
                "<Intensity>",
            ],
        }
    )
    path = scan_dir / "pdata" / "2" / "visu_pars"
    write_paravision_parameters(path, visu)
    path.write_text(
        path.read_text().replace(
            "<Fractional Anisotropy>", "<Fractional \nAnisotropy>"
        ),
        encoding="utf-8",
    )
    original_bytes = (path.parent / "2dseq").read_bytes()
    result = (
        _reconstruct_with_fallback(scan_dir, recon_num=2)
        if force_fallback
        else _reconstruct(scan_dir, recon_num=2)
    )
    assert result.map_idx.values.tolist() == [
        "Fractional Anisotropy",
        "Trace",
        "Intensity",
    ]
    assert result.frame_id.values.tolist() == ["DTI_FA", "DTI_TRACE", "DTI_A0"]
    assert result.frame_id.dims == ("map_idx",)
    assert result.frame_id.attrs == {
        "unit": "id",
        "source_parameter": "VisuFGElemId",
        "frame_group": "FG_DTI",
    }
    selected = result.isel(map_idx=result.frame_id == "DTI_FA")
    assert selected.map_idx.item() == "Fractional Anisotropy"
    np.testing.assert_array_equal(
        _frames_along(selected, ("slice_idx", "repetition_idx", "map_idx")),
        frames[..., :1],
    )
    assert result.frame_label_repetition_idx.values.tolist() == [
        "cycle one",
        "cycle two",
    ]
    np.testing.assert_array_equal(
        _frames_along(result, ("slice_idx", "repetition_idx", "map_idx")), frames
    )
    assert (path.parent / "2dseq").read_bytes() == original_bytes


def test_loop_times_are_declared_repetition_times_not_readout_sampling(
    tmp_path, write_paravision_parameters
):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:RAREVTR>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
    )
    frames = np.arange(N_SLICES * 3).reshape(N_SLICES, 3)
    visu = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (3, "FG_MOVIE")],
    )
    visu.update(
        {
            "VisuFGOrderDesc": [
                f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                "(3, <FG_MOVIE>, <>, 0, 1)",
            ],
            "VisuGroupDepVals": ["(<VisuAcqRepetitionTime>, 0)"],
            "VisuAcqRepetitionTime": [100.0, 400.0, 1600.0],
        }
    )
    write_paravision_parameters(scan_dir / "pdata" / "1" / "visu_pars", visu)
    handler = BrukerHandler(scan_dir)
    # This acquired readout sampling belongs to neither the loop nor its TRs.
    handler.sampling_time_axis_s = np.arange(32) * 0.0001
    result = Brukerseq2dRecoStrategy().reconstruct(handler)
    np.testing.assert_allclose(
        result.acquisition_idx, [0.1, 0.4, 1.6], rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(result.repetition_time_s, result.acquisition_idx)
    assert "acquisition_time_s" not in result.coords


@pytest.mark.parametrize(
    "dim, attribute",
    [("inversion_idx", "inversion_time_ms"), ("bvalue_idx", "b_values")],
)
def test_incomplete_method_coordinates_keep_index_units(dim, attribute):
    from types import SimpleNamespace

    handler = SimpleNamespace(**{attribute: [10.0], "rps_grid_mm": None})
    coords = Brukerseq2dRecoStrategy._build_fallback_coords(
        handler, [dim], [3], {}, None
    )
    np.testing.assert_array_equal(coords[dim][1], [0, 1, 2])
    assert coords[dim][2]["unit"] == "index"


@pytest.mark.parametrize("values", [[10.0], [10.0, np.nan], None])
def test_declared_frame_times_reject_missing_or_invalid_values(values):
    from hypermri.reconstruction.strategies.bruker.seq2d import (
        _frame_group_coordinates,
    )

    visu = {
        "VisuCoreSize": [2, 2],
        "VisuFGOrderDesc": "(2, <FG_ECHO>, <>, 0, 1)",
        "VisuGroupDepVals": "(<VisuAcqEchoTime>, 0)",
    }
    if values is not None:
        visu["VisuAcqEchoTime"] = values
    with pytest.raises(ValueError, match="VisuAcqEchoTime"):
        _frame_group_coordinates(visu, (2, 2, 2), ("read_idx", "phase_idx", "echo_idx"))


def test_ir_time_never_uses_nonzero_fair_slice_offsets():
    from types import SimpleNamespace

    from hypermri.reconstruction.strategies._common import (
        _create_inversion_idx_coordinate,
    )

    handler = SimpleNamespace(
        inversion_time_ms=np.array([20.0, 720.0]),
        inversion_time_source="IRTime",
        slice_order=np.array([0, 1, 2]),
        refocusing_pulse_length_ms=2.0,
        rf_pulse_length_ms=1.0,
        rare_factor=8,
        echo_time_ms=3.0,
        method={},
    )
    handler.get = lambda key, default=None: getattr(handler, key, default)
    coords = _create_inversion_idx_coordinate(
        handler, 2, 3, {"slice_idx": 3, "inversion_idx": 2}
    )
    assert coords[0] == "inversion_idx"
    np.testing.assert_array_equal(coords[1], [20.0, 720.0])
    assert coords[2]["unit"] == "ms"


def test_diffusion_tensor_maps_are_not_labelled_as_b_values(
    tmp_path, write_paravision_parameters
):
    """Regression test: ``FG_DTI`` maps leave the strict path.

    The method describes five b-values and ``pdata/1`` stores the five
    slices and then three diffusion tensor maps.  The strict path compares
    only the number of dimensions, so previously it labelled the maps
    ``bvalue_idx``, as it labelled the 22 maps of a local DTI scan with
    the first 22 of its 45 b-values.
    """
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"PVM_DwEffBval": [0.0, 250.0, 500.0, 750.0, 1000.0]},
    )
    visu_pars = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        _expected_frames(3).T,
        [(N_SLICES, "FG_SLICE"), (3, "FG_DTI")],
    )
    write_paravision_parameters(scan_dir / "visu_pars", visu_pars)

    data_array = _reconstruct(scan_dir)

    assert data_array.attrs["seq2d_fallback_reason"] == FRAME_GROUP_ORDER_REASON
    assert data_array.dims == ("read_idx", "phase_idx", "slice_idx", "map_idx")
    np.testing.assert_array_equal(
        _labelled_frames(data_array, "map_idx"), _expected_frames(3)
    )


# Recorded from the checkout before the reconstruction rewrite on 6 October 2026 with NumPy 2.2.6,
# brukerapi 0.4.6 and xarray 2025.6.1; only synthetic ParaVision parameters.
_SEQ2D_STRUCTURAL_REFERENCES = {
    (
        "<Bruker:MSME>",
        ("echo", "slice"),
    ): "f49914ced9a357677cf00cb4066982ef67a40db7535cda533662d537ca966191",
    (
        "<Bruker:MSME>",
        ("slice", "echo"),
    ): "5ff47bd1ec749dd813df2a1041dbc928ae68e369d3607a31a65a70e4442a5121",
    (
        "<Bruker:MGE>",
        ("echo", "slice"),
    ): "e8580ca2e3ae6bf44246b7707c2ce581b07fa0b30fc4b89a79536e31be311aad",
    (
        "<Bruker:MGE>",
        ("slice", "echo"),
    ): "8277d6cf0f141466d13d9e9ccf859e934f9018d3b9bed7901be56f86391deba8",
}


def _seq2d_structural_result(scan_dir, write_parameters, method, order):
    _write_scan(
        scan_dir,
        write_parameters,
        method_name=method,
        n_echoes=3,
        frame_group_order=order,
    )
    handler = BrukerHandler(scan_dir, load_data=False)
    handler.git_commit_hash = None
    result = Brukerseq2dRecoStrategy().reconstruct(
        handler, data_precision="float32", append_parameters=False
    )
    return result


@pytest.mark.parametrize("dwell_ms", [None, 0.0, -1.0, 0.25])
def test_bruker_sampling_axis_requires_valid_acquired_dwell(
    tmp_path, write_paravision_parameters, dwell_ms
):
    """Missing/invalid synthetic dwell never acquires the legacy 1-ms default."""
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:MSME>",
        n_echoes=3,
        frame_group_order=("slice", "echo"),
        method_parameters={"PVM_DigDw": dwell_ms},
    )
    handler = BrukerHandler(scan_dir, load_data=False)
    if dwell_ms is None or dwell_ms <= 0:
        assert handler.sampling_time_axis_s is None
    else:
        np.testing.assert_allclose(
            handler.sampling_time_axis_s,
            np.arange(handler.n_acquisition) * dwell_ms / 1000.0,
            rtol=0,
            atol=1e-15,
        )


@pytest.mark.parametrize("method", ["<Bruker:MSME>", "<Bruker:MGE>"])
@pytest.mark.parametrize("order", [("echo", "slice"), ("slice", "echo")])
def test_seq2d_structural_reference(
    tmp_path, write_paravision_parameters, method, order
):
    """Trace strict and frame-order fallback paths against unchanged code."""
    import hashlib

    result = _seq2d_structural_result(
        tmp_path / "7", write_paravision_parameters, method, order
    )
    # Values and labelled coordinates retain exact binary identity. Paths and
    # commit-dependent provenance are intentionally excluded from this fixture.
    payload = result.values.tobytes() + repr(result.dims).encode()
    if method == "<Bruker:MGE>":
        np.testing.assert_array_equal(result.acquisition_time_s, result.acquisition_idx)
        np.testing.assert_allclose(
            result.acquisition_idx, [0.01, 0.02, 0.03], rtol=0, atol=1e-15
        )
    for key, coord in sorted(result.coords.items()):
        # Explicit echo times were added to direct-handler MGE labelling. Keep
        # the earlier reference for all values and previously present axes.
        if method == "<Bruker:MGE>" and key == "acquisition_time_s":
            continue
        if key in ("freq_axis_hz", "freq_axis_ppm"):
            # The frequency axes are now stored as coordinates instead of
            # attributes, which this fingerprint never covered.
            assert key not in result.attrs
            continue
        values = np.asarray(coord.values)
        if (
            method == "<Bruker:MGE>"
            and order == ("slice", "echo")
            and key == "acquisition_idx"
        ):
            # A later fix corrected the strict path's discarded EffectiveTE values. The
            # independent assertion above checks the corrected times; retain
            # the old digest for every other value, label and coordinate.
            values = np.arange(3)
        legacy_attrs = dict(coord.attrs)
        if key == "waterfat_idx" and coord.size == 1:
            # Both imaging paths now use the same singleton index. Assert
            # the correction explicitly and retain the old digest elsewhere.
            np.testing.assert_array_equal(values, [0])
            assert coord.attrs["unit"] == "index"
            values = np.array(["Unknown"])
            legacy_attrs["unit"] = "type"
        if (
            key in ("read_idx", "phase_idx", "slice_idx")
            and "extent_mm" in legacy_attrs
        ):
            axis = ("read_idx", "phase_idx", "slice_idx").index(key)
            # A limit pair replaced the former wrong scalar with a limit pair. The
            # original fingerprint continues checking all unaffected fields.
            extent = []
            for dim in ("read_idx", "phase_idx", "slice_idx"):
                extent.extend(result[dim].attrs["extent_mm"])
            legacy_attrs["extent_mm"] = extent[axis]
        payload += (
            key.encode()
            + values.tobytes()
            + repr(sorted(legacy_attrs.items())).encode()
        )
    assert (
        hashlib.sha256(payload).hexdigest()
        == _SEQ2D_STRUCTURAL_REFERENCES[(method, order)]
    )


@pytest.mark.parametrize("load_data", [False, True])
@pytest.mark.parametrize("time_shape", ["vector", "row"])
def test_t1_epi_parsed_time_table_direct_and_study_reconstruction(
    tmp_path, write_paravision_parameters, load_data, time_shape
):
    """Invented single-row timings retain ms labels through both loading paths."""
    from hypermri import Study

    scan_dir = tmp_path / "20"
    times = [20.0, 180.0, 720.0]
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:T1_EPI>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={
            "IRTime": [times] if time_shape == "row" else times,
            "PVM_EncSteps1": list(range(-MATRIX_SIZE // 2, MATRIX_SIZE // 2)),
            "PVM_DigNp": MATRIX_SIZE**2,
            "PVM_EpiNEchoes": MATRIX_SIZE,
            "PVM_EpiNShots": 1,
            "PVM_NRepetitions": len(times),
        },
    )
    acqp = {
        **SERIES["acqp"],
        "ACQ_size": [2 * MATRIX_SIZE**2, 1],
        "ACQ_scan_name": "<synthetic T1 EPI>",
        "ACQ_dim": 2,
        "NI": N_SLICES,
        "NR": len(times),
        "ACQ_phase_factor": MATRIX_SIZE,
        "NSegments": 1,
        "ACQ_obj_order": list(range(N_SLICES)),
        "PULPROG": "<T1_EPI.ppg>",
        "AQ_mod": "qdig",
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "GO_block_size": "continuous",
        "ACQ_scan_size": "ACQ_phase_factor_scans",
    }
    write_paravision_parameters(scan_dir / "acqp", acqp)
    scalar = np.zeros((len(times), N_SLICES, MATRIX_SIZE, MATRIX_SIZE, 2), dtype="<i4")
    scalar[..., 0] = np.arange(scalar[..., 0].size).reshape(scalar.shape[:-1]) + 1
    (scan_dir / "fid").write_bytes(scalar.tobytes())
    frames = 1000 + np.arange(N_SLICES)[:, None] + 100 * np.arange(len(times))[None, :]
    visu = _write_frames(
        scan_dir / "pdata" / "1",
        write_paravision_parameters,
        frames,
        [(N_SLICES, "FG_SLICE"), (len(times), "FG_MOVIE")],
    )
    visu.update(
        {
            "VisuFGOrderDesc": [
                f"({N_SLICES}, <FG_SLICE>, <>, 0, 0)",
                f"({len(times)}, <FG_MOVIE>, <>, 0, 1)",
            ],
            "VisuGroupDepVals": ["(<VisuAcqInversionTime>, 0)"],
            "VisuAcqInversionTime": times,
        }
    )
    write_paravision_parameters(scan_dir / "pdata" / "1" / "visu_pars", visu)
    originals = {
        path: path.read_bytes() for path in scan_dir.rglob("*") if path.is_file()
    }
    direct = load_scan(
        scan_dir, mode="full" if load_data else "lazy", log_mode="critical"
    )
    study = Study(tmp_path, load_data=load_data, verbose=False, log_mode="critical")
    loaded = study[20]
    for scan in (direct, loaded):
        assert scan.handler.inversion_time_source == "IRTime"
        assert scan.handler.n_inversion_time == len(times)
        np.testing.assert_array_equal(scan.handler.inversion_time_ms, times)
        assert np.asarray(scan.handler.method["IRTime"]).shape == (
            (1, len(times)) if time_shape == "row" else (len(times),)
        )
    for strategy in ("seq2d", "raw"):
        options = (
            {}
            if strategy == "seq2d"
            else {"shift_xarray": False, "apply_phase_correction": False}
        )
        actual = direct.reconstruct(strategy, **options)
        xr.testing.assert_identical(actual, loaded.reconstruct(strategy, **options))
        xr.testing.assert_identical(actual, direct.reconstruct(strategy, **options))
        np.testing.assert_array_equal(actual.inversion_idx, times)
        assert actual.inversion_idx.attrs["unit"] == "ms"
        if strategy == "seq2d":
            np.testing.assert_array_equal(
                _frames_along(actual, ("slice_idx", "inversion_idx")), frames
            )
    for path, original in originals.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize("times", [20.0, [20.0], [[20.0]], [[20.0, 180.0, 720.0]]])
def test_t1_epi_valid_time_forms(tmp_path, write_paravision_parameters, times):
    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:T1_EPI>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"IRTime": times},
    )
    handler = load_scan(scan_dir, mode="lazy", log_mode="critical").handler
    np.testing.assert_array_equal(
        handler.inversion_time_ms, np.asarray(times).reshape(-1)
    )


@pytest.mark.parametrize(
    "times",
    [
        [],
        [[20.0], [180.0]],
        [[20.0, 180.0], [300.0, 720.0]],
        [[[20.0, 180.0]]],
        [-1.0, 20.0],
        [float("nan")],
        [float("inf")],
        ["invalid"],
    ],
)
def test_t1_epi_invalid_time_forms_fail_early(
    tmp_path, write_paravision_parameters, times
):
    from hypermri.exceptions import DataLoadingError

    scan_dir = tmp_path / "7"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:T1_EPI>",
        n_echoes=1,
        frame_group_order=("slice", "echo"),
        method_parameters={"IRTime": times},
    )
    with pytest.raises(DataLoadingError, match="T1 EPI IRTime"):
        BrukerHandler(scan_dir, load_data=False, log_mode="critical")
