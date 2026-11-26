"""``RECO_transposition`` handling of the Bruker ``2dseq`` loaders.

ParaVision stores the dimensions of each ``2dseq`` frame in ``VisuCoreDimDesc``
order, fastest first, and ``RECO_transposition`` states which two of them it
swapped before storage: 1 the first and second, 2 the second and third, 3 the
first and third.  The local scans use 1 for 2D and 3D images (read and phase)
and 2 for one CSI scan whose frames are ``(spectroscopic, spatial, spatial)``.
Seven local multi-package localizers mix 0 and 1 between their slice packages,
all with square frames and one value per frame.  Each ``pdata/<n>``
folder has its own ``reco`` and ``visu_pars``, and the swap of its ``2dseq``
follows those files.

Each test scan is synthetic: parameter files with only the fields that
``BrukerHandler`` and brukerapi need, and a ``2dseq`` of 16-bit integers that
count up through the array, written in ParaVision storage order with a slope
of 1.  The values are no scanner data.  Integer values below 2**15 survive the
float32 conversion of the loader exactly, so the loaded arrays are compared
with exact equality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler

CSI_FRAME_DIMS = ["spectroscopic", "spatial", "spatial"]
IMAGE_FRAME_DIMS = ["spatial", "spatial"]
VOLUME_FRAME_DIMS = ["spatial", "spatial", "spatial"]


def _write_reconstruction(
    pdata_dir: Path,
    write_parameters,
    *,
    frame_dim_descriptions: list[str],
    reco_transposition: list[int] | None,
    stored: np.ndarray,
) -> dict:
    """Write ``visu_pars``, ``reco`` and ``2dseq`` of one ``pdata`` folder.

    The leading ``len(frame_dim_descriptions)`` axes of ``stored`` are the
    frame dimensions; the remaining axes form one frame group.  Returns the
    written ``visu_pars``.
    """
    frame_ndim = len(frame_dim_descriptions)
    n_frames = int(np.prod(stored.shape[frame_ndim:]))
    # brukerapi reads the frame layout and scaling of 2dseq from these fields.
    visu_pars = {
        "VisuCoreSize": list(stored.shape[:frame_ndim]),
        "VisuCoreDimDesc": frame_dim_descriptions,
        "VisuCreatorVersion": "7.0.0",
        "VisuCoreWordType": "_16BIT_SGN_INT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreFrameCount": n_frames,
        "VisuFGOrderDescDim": 1,
        "VisuFGOrderDesc": [f"({n_frames}, <FG_CYCLE>, <>, 0, 0)"],
        "VisuCoreDataSlope": [1] * n_frames,
        "VisuCoreDataOffs": [0] * n_frames,
        "VisuCoreFrameType": "MAGNITUDE_IMAGE",
    }
    write_parameters(pdata_dir / "visu_pars", visu_pars)
    write_parameters(pdata_dir / "reco", {"RECO_transposition": reco_transposition})
    # ParaVision stores the fastest axis first, which is Fortran order here.
    (pdata_dir / "2dseq").write_bytes(stored.astype("<i2").tobytes(order="F"))
    return visu_pars


def _write_scan(
    scan_dir: Path,
    write_parameters,
    *,
    method_name: str,
    frame_dim_descriptions: list[str],
    reco_transposition: list[int] | None,
    stored: np.ndarray,
) -> None:
    """Write a scan whose ``pdata/1/2dseq`` holds ``stored`` in ParaVision order.

    The scan-level ``visu_pars`` is a copy of ``pdata/1/visu_pars``, as in
    every local scan.
    """
    write_parameters(scan_dir / "method", {"Method": method_name})
    write_parameters(scan_dir / "acqp", {"ACQ_patient_pos": "Head_Supine"})
    visu_pars = _write_reconstruction(
        scan_dir / "pdata" / "1",
        write_parameters,
        frame_dim_descriptions=frame_dim_descriptions,
        reco_transposition=reco_transposition,
        stored=stored,
    )
    write_parameters(scan_dir / "visu_pars", visu_pars)


def _counting_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(int(np.prod(shape))).reshape(shape)


def _load_handler(scan_dir: Path) -> BrukerHandler:
    handler = BrukerHandler(scan_dir)
    assert "patient_position" in handler.params
    return handler


# --------------------------------------------------------------------------
# Frame axes of RECO_transposition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reco_transposition", "frame_dims", "expected"),
    [
        (None, IMAGE_FRAME_DIMS, None),
        ([0, 0], IMAGE_FRAME_DIMS, None),
        (0, None, None),
        ([1, 1], IMAGE_FRAME_DIMS, (0, 1)),
        ([1], VOLUME_FRAME_DIMS, (0, 1)),
        ([2], CSI_FRAME_DIMS, (1, 2)),
        # Multi-package localizers transpose only some packages.
        ([0, 1, 1], IMAGE_FRAME_DIMS, (0, 1)),
        ([1, 1], "spatial spatial", (0, 1)),
    ],
)
def test_reco_transposition_frame_axes_of_supported_layouts(
    reco_transposition, frame_dims, expected
):
    assert (
        BrukerHandler._reco_transposition_frame_axes(reco_transposition, frame_dims)
        == expected
    )


@pytest.mark.parametrize(
    ("reco_transposition", "frame_dims", "message"),
    [
        ([2], IMAGE_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(1, 2\)"),
        ([3], IMAGE_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(0, 2\)"),
        ([2], VOLUME_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(1, 2\)"),
        ([3], VOLUME_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(0, 2\)"),
        ([1], CSI_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(0, 1\)"),
        ([3], CSI_FRAME_DIMS, r"swaps the 2dseq frame dimensions \(0, 2\)"),
        ([1], ["spatial", "spectroscopic", "spatial"], r"\(0, 1\)"),
        ([1, 2], IMAGE_FRAME_DIMS, r"values \[1, 2\] are not supported"),
        ([4], IMAGE_FRAME_DIMS, r"values \[4\] are not supported"),
        ([1], None, "needs VisuCoreDimDesc"),
        (["a"], IMAGE_FRAME_DIMS, "not an integer array"),
    ],
)
def test_reco_transposition_frame_axes_rejects_other_swaps(
    reco_transposition, frame_dims, message
):
    with pytest.raises(ValueError, match=message):
        BrukerHandler._reco_transposition_frame_axes(reco_transposition, frame_dims)


@pytest.mark.parametrize(
    ("reco_transposition", "expected"), [([0, 1], [0, 1]), ([1], [1]), (None, None)]
)
def test_reco_transposition_has_one_value_per_frame(
    tmp_path, write_paravision_parameters, reco_transposition, expected
):
    """The handler exposes the values that place read per frame."""
    n_frames = 1 if reco_transposition is None else len(reco_transposition)
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=reco_transposition,
        stored=_counting_array((4, 4, n_frames)),
    )
    handler = _load_handler(scan_dir)

    if expected is None:
        assert handler.reco_transposition is None
    else:
        assert handler.reco_transposition.shape == (len(expected),)
        np.testing.assert_array_equal(handler.reco_transposition, expected)
        # The xyz grid reads metadata through the get method of the handler.
        np.testing.assert_array_equal(handler.get("reco_transposition"), expected)


# --------------------------------------------------------------------------
# Loaded 2dseq arrays
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reco_transposition", "transposed"), [([0, 0], False), ([1, 1], True)]
)
def test_image_seq2d_has_read_before_phase(
    tmp_path, write_paravision_parameters, reco_transposition, transposed
):
    """Characterization: 2D images swap the first two axes, as before."""
    canonical = _counting_array((5, 4, 2))  # read, phase, slice
    stored = np.swapaxes(canonical, 0, 1) if transposed else canonical
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=reco_transposition,
        stored=stored,
    )
    handler = _load_handler(scan_dir)

    handler.load_data(scan_dir)

    assert handler.transpose_read_phase is transposed
    assert handler.seq2d.shape == (5, 4, 2)
    np.testing.assert_array_equal(handler.seq2d, canonical)


@pytest.mark.parametrize(
    ("reco_transposition", "transposed"),
    [(None, False), ([0, 0], False), ([1, 1], True)],
)
def test_undo_reco_transposition_puts_read_before_phase(
    tmp_path, write_paravision_parameters, reco_transposition, transposed
):
    """The swap that every ``2dseq`` consumer applies leaves its input as is."""
    canonical = _counting_array((5, 4, 2))  # read, phase, slice
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=reco_transposition,
        stored=np.swapaxes(canonical, 0, 1) if transposed else canonical,
    )
    handler = _load_handler(scan_dir)
    stored = handler.load_2dseq_file()
    stored_before = stored.copy()

    undone = handler.undo_reco_transposition(stored)

    np.testing.assert_array_equal(undone, canonical)
    np.testing.assert_array_equal(stored, stored_before)
    assert handler.undo_reco_transposition(None) is None


@pytest.mark.parametrize("data", [None, np.empty(0), np.empty((3, 0))])
def test_absent_seq2d_needs_no_transposition_metadata(
    tmp_path, write_paravision_parameters, data
):
    """Absent images do not require metadata describing image axes."""
    scan_dir = tmp_path / "5"
    write_paravision_parameters(scan_dir / "method", {"Method": "<Bruker:FLASH>"})
    write_paravision_parameters(scan_dir / "acqp", {"ACQ_patient_pos": "Head_Supine"})
    write_paravision_parameters(
        scan_dir / "pdata" / "1" / "reco", {"RECO_transposition": [1]}
    )
    handler = _load_handler(scan_dir)

    assert handler.undo_reco_transposition(data) is data


def test_csi_seq2d_swaps_the_spatial_frame_axes(tmp_path, write_paravision_parameters):
    """Regression: value 2 swaps the two spatial axes, not spectrum and phase.

    Previously every nonzero value swapped the first two array axes, which gave
    the shape ``(4, 6, 5, 3)`` here and ``(24, 100, 24, 4)`` for the local CSI
    scan with ``RECO_transposition`` 2.
    """
    canonical = _counting_array((6, 5, 4, 3))  # spectrum, read, phase, cycle
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:CSI>",
        frame_dim_descriptions=CSI_FRAME_DIMS,
        reco_transposition=[2, 2, 2],
        stored=np.swapaxes(canonical, 1, 2),
    )
    handler = _load_handler(scan_dir)

    handler.load_data(scan_dir)

    assert handler.transpose_read_phase is True
    assert handler.seq2d.shape == (6, 5, 4, 3)
    np.testing.assert_array_equal(handler.seq2d, canonical)


def test_unsupported_transposition_fails_on_use_not_on_construction(
    tmp_path, write_paravision_parameters
):
    """Regression: a phase and slice swap of a 3D scan raises when loaded.

    Previously value 2 was read as a read and phase swap without an error.
    """
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=VOLUME_FRAME_DIMS,
        reco_transposition=[2],
        stored=_counting_array((5, 4, 3)),
    )
    handler = _load_handler(scan_dir)

    with pytest.raises(ValueError, match="RECO_transposition 2"):
        _ = handler.transpose_read_phase
    with pytest.raises(ValueError, match="Use the raw reconstruction"):
        handler.load_data(scan_dir)


def test_mixed_transposition_is_undone_per_frame(tmp_path, write_paravision_parameters):
    """Regression test: only the frames with value 1 are swapped back.

    Previously one nonzero value swapped read and phase in every frame, so
    the first frame, stored with value 0, came out transposed.
    """
    canonical = _counting_array((4, 4, 3))  # read, phase, slice
    stored = canonical.copy()
    stored[:, :, 1:] = np.swapaxes(canonical[:, :, 1:], 0, 1)
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=[0, 1, 1],
        stored=stored,
    )
    handler = _load_handler(scan_dir)

    handler.load_data(scan_dir)

    assert handler.transpose_read_phase is True
    assert handler.seq2d.shape == (4, 4, 3)
    np.testing.assert_array_equal(handler.seq2d, canonical)


def test_undo_reco_transposition_counts_frames_fastest_group_first(
    tmp_path, write_paravision_parameters
):
    """Frame ``(i, j)`` of two frame groups of sizes 3 and 2 is frame ``i + 3j``.

    ParaVision stores the frame groups fastest first, so the ``k``-th value
    of ``RECO_transposition`` belongs to the ``k``-th frame in Fortran
    order.  The input is not modified and the result does not share its
    memory.
    """
    reco_transposition = [0, 1, 1, 0, 0, 1]
    canonical = _counting_array((4, 4, 3, 2))  # read, phase, two frame groups
    stored = canonical.copy()
    for i in range(3):
        for j in range(2):
            if reco_transposition[i + 3 * j]:
                stored[:, :, i, j] = canonical[:, :, i, j].T
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=reco_transposition,
        stored=stored.reshape((4, 4, 6), order="F"),
    )
    handler = _load_handler(scan_dir)
    stored_before = stored.copy()

    undone = handler.undo_reco_transposition(stored)

    np.testing.assert_array_equal(undone, canonical)
    np.testing.assert_array_equal(stored, stored_before)
    assert not np.shares_memory(undone, stored)


@pytest.mark.parametrize(
    ("reco_transposition", "stored_shape", "message"),
    [
        (
            [0, 1],
            (4, 4, 3),
            "has 2 values that differ between frames, but the 2dseq data has 3 frames",
        ),
        ([0, 1], (5, 4, 2), r"have the sizes \(5, 4\), so the frames do not fit"),
    ],
    ids=["value-count", "non-square-frames"],
)
def test_mixed_transposition_fails_when_the_frames_do_not_match(
    tmp_path, write_paravision_parameters, reco_transposition, stored_shape, message
):
    """Per-frame values need one value per frame and square swapped axes."""
    scan_dir = tmp_path / "5"
    _write_scan(
        scan_dir,
        write_paravision_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=reco_transposition,
        stored=_counting_array(stored_shape),
    )
    handler = _load_handler(scan_dir)

    with pytest.raises(ValueError, match=message):
        handler.load_data(scan_dir)


# --------------------------------------------------------------------------
# Parameters of each reconstruction
# --------------------------------------------------------------------------


def _write_two_reconstructions(
    scan_dir: Path,
    write_parameters,
    *,
    pdata1_transposition: list[int],
    pdata2_transposition: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Write a FLASH scan with two reconstructions of non-square frames.

    ``pdata/2`` holds other values than ``pdata/1`` so that the folder of
    each loaded array is visible.  Returns the canonical ``(read, phase,
    slice)`` arrays of ``pdata/1`` and ``pdata/2``.
    """
    canonical = (_counting_array((5, 4, 2)), _counting_array((5, 4, 2)) + 100)
    _write_scan(
        scan_dir,
        write_parameters,
        method_name="<Bruker:FLASH>",
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=pdata1_transposition,
        stored=np.swapaxes(canonical[0], 0, 1)
        if pdata1_transposition[0]
        else canonical[0],
    )
    _write_reconstruction(
        scan_dir / "pdata" / "2",
        write_parameters,
        frame_dim_descriptions=IMAGE_FRAME_DIMS,
        reco_transposition=pdata2_transposition,
        stored=np.swapaxes(canonical[1], 0, 1)
        if pdata2_transposition[0]
        else canonical[1],
    )
    return canonical


TRANSPOSITIONS = [
    pytest.param([0, 0], [1, 1], id="pdata2-transposed"),
    pytest.param([1, 1], [0, 0], id="pdata1-transposed"),
]


@pytest.mark.parametrize(
    ("pdata1_transposition", "pdata2_transposition"), TRANSPOSITIONS
)
def test_undo_reco_transposition_follows_the_selected_reconstruction(
    tmp_path, write_paravision_parameters, pdata1_transposition, pdata2_transposition
):
    """Regression test: ``pdata/2`` is swapped by its own ``reco``.

    Previously the swap used the ``RECO_transposition`` of ``pdata/1/reco``
    that the handler read on construction, also after ``set_reco_file``.
    ``pdata/2`` then came out with the shape ``(4, 5, 2)`` instead of
    ``(5, 4, 2)``.  The ``2dseq``
    reconstruction is tested in ``test_bruker_seq2d_frame_group_order``.
    """
    scan_dir = tmp_path / "5"
    canonical = _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        pdata1_transposition=pdata1_transposition,
        pdata2_transposition=pdata2_transposition,
    )
    handler = _load_handler(scan_dir)

    for recon_num, expected in enumerate(canonical, start=1):
        stored = handler.load_2dseq_file(recon_num=recon_num)
        undone = handler.undo_reco_transposition(stored, recon_num=recon_num)
        np.testing.assert_array_equal(undone, expected)
    np.testing.assert_array_equal(
        handler.reco["RECO_transposition"], pdata1_transposition
    )


def test_reconstruction_parameters_are_read_from_each_pdata_folder(
    tmp_path, write_paravision_parameters
):
    """Each ``pdata`` folder has its own ``reco`` and ``visu_pars``.

    The ``reco`` and ``visu`` attributes keep ``pdata/1/reco`` and the
    scan-level ``visu_pars``.  A missing folder gives empty dicts.
    """
    scan_dir = tmp_path / "5"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        pdata1_transposition=[0, 0],
        pdata2_transposition=[1, 1],
    )
    write_parameters = write_paravision_parameters
    write_parameters(
        scan_dir / "pdata" / "2" / "visu_pars",
        {"VisuCoreDimDesc": IMAGE_FRAME_DIMS, "VisuCoreFrameType": "T2_MAP"},
    )
    handler = _load_handler(scan_dir)

    reco, visu_pars = handler.load_reconstruction_parameters(recon_num=2)

    np.testing.assert_array_equal(reco["RECO_transposition"], [1, 1])
    assert visu_pars["VisuCoreFrameType"] == "T2_MAP"
    np.testing.assert_array_equal(handler.reco["RECO_transposition"], [0, 0])
    assert handler.visu["VisuCoreFrameType"] == "MAGNITUDE_IMAGE"
    assert handler.load_reconstruction_parameters(recon_num=3) == ({}, {})


@pytest.mark.parametrize(
    "failure", [ValueError("invalid parameter"), KeyboardInterrupt(), SystemExit(2)]
)
def test_selecting_reco_parameters_propagates_failure_without_stale_result(
    tmp_path, monkeypatch, failure
):
    """A failed selection must not return data with the previous reco."""
    folder = tmp_path / "pdata" / "2"
    folder.mkdir(parents=True)
    (folder / "2dseq").write_bytes(b"synthetic")
    handler = object.__new__(BrukerHandler)
    handler.path = tmp_path
    original_reco = {"RecoObjectsPerRepetition": 1}
    handler.reco = original_reco
    handler.reco_parameter_source = "pdata/1/reco"

    def fail_selection(recon_num):
        assert recon_num == 2
        raise failure

    monkeypatch.setattr(handler, "load_reconstruction_parameters", fail_selection)
    monkeypatch.setattr(handler, "_take_full_load_2dseq", lambda *args: np.ones((2, 2)))

    with pytest.raises(type(failure)) as error:
        handler.load_2dseq_file(recon_num=2, set_reco_file=True)

    assert error.value is failure
    assert handler.reco is original_reco
    assert handler.reco_parameter_source == "pdata/1/reco"


@pytest.mark.parametrize("set_reco_file", [False, True])
def test_reco_selection_mutates_only_when_requested(
    tmp_path, monkeypatch, set_reco_file
):
    """Keep the documented successful mutation policy."""
    folder = tmp_path / "pdata" / "2"
    folder.mkdir(parents=True)
    (folder / "2dseq").write_bytes(b"synthetic")
    handler = object.__new__(BrukerHandler)
    handler.path = tmp_path
    original_reco = {"RecoObjectsPerRepetition": 1}
    selected_reco = {"RecoObjectsPerRepetition": 2}
    handler.reco = original_reco
    handler.reco_parameter_source = "pdata/1/reco"
    calls = []

    def select_parameters(recon_num):
        calls.append(recon_num)
        return selected_reco, {}

    monkeypatch.setattr(handler, "load_reconstruction_parameters", select_parameters)
    values = np.arange(4).reshape(2, 2)
    monkeypatch.setattr(handler, "_take_full_load_2dseq", lambda *args: values.copy())

    actual = handler.load_2dseq_file(recon_num=2, set_reco_file=set_reco_file)

    np.testing.assert_array_equal(actual, values)
    assert calls == ([2] if set_reco_file else [])
    assert handler.reco is (selected_reco if set_reco_file else original_reco)
    assert handler.reco_parameter_source == (
        "pdata/2/reco" if set_reco_file else "pdata/1/reco"
    )


def test_malformed_selected_reco_file_does_not_return_data(
    tmp_path, write_paravision_parameters
):
    """Exercise the actual parameter reader, rather than only mocked failures."""
    scan_dir = tmp_path / "5"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        pdata1_transposition=[0, 0],
        pdata2_transposition=[1, 1],
    )
    handler = _load_handler(scan_dir)
    original_reco = handler.reco
    (scan_dir / "pdata" / "2" / "reco").write_text(
        "##$RECO_transposition=invalid=parameter\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        handler.load_2dseq_file(recon_num=2, set_reco_file=True)

    assert handler.reco is original_reco


def test_reconstruction_parameters_are_cached_and_returned_as_copies(
    tmp_path, write_paravision_parameters
):
    """The files are parsed once; callers cannot change the cached dicts."""
    scan_dir = tmp_path / "5"
    _write_two_reconstructions(
        scan_dir,
        write_paravision_parameters,
        pdata1_transposition=[0, 0],
        pdata2_transposition=[1, 1],
    )
    handler = _load_handler(scan_dir)
    reco, visu_pars = handler.load_reconstruction_parameters(recon_num=2)
    reco["RECO_transposition"] = [0, 0]
    visu_pars.clear()
    (scan_dir / "pdata" / "2" / "reco").unlink()

    cached_reco, cached_visu_pars = handler.load_reconstruction_parameters(2)

    np.testing.assert_array_equal(cached_reco["RECO_transposition"], [1, 1])
    assert cached_visu_pars["VisuCoreDimDesc"] == IMAGE_FRAME_DIMS
