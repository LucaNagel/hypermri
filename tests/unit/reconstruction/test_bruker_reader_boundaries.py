"""Actual-reader regressions using invented integer acquisition labels.

Small constant readouts make object-order comparisons exact. No scanner
samples or identifying metadata are used.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from brukerapi.dataset import Dataset

from hypermri.reconstruction.strategies.bruker._block_padding import (
    _strip_block_padding,
)
from hypermri.reconstruction.strategies.bruker.bssfp import BrukerBssfpRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.dti import BrukerDtiRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.epi import BrukerEpiRawRecoStrategy


@pytest.mark.parametrize(
    "order", [list(range(6)), [1, 0, 2, 3, 4, 5], [2, 0, 1, 5, 3, 4]]
)
@pytest.mark.parametrize("sequence", ["bssfp", "dti", "epi"])
def test_actual_reader_object_order_is_applied_once(
    tmp_path, write_paravision_parameters, monkeypatch, order, sequence
):
    """Physical object labels survive identity, swap and three-cycle orders."""
    acqp = {
        "ACQ_dim": 2,
        "ACQ_size": [8, 2],
        "NI": 6,
        "NR": 1,
        "ACQ_phase_factor": 1,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "GO_block_size": "continuous",
        "BYTORDA": "little",
        "PULPROG": "<EPI.ppg>" if sequence == "epi" else "<FISP.ppg>",
        "AQ_mod": "qdig",
        "ACQ_scan_size": "ACQ_phase_factor_scans" if sequence == "epi" else "One_scan",
        "ACQ_obj_order": order,
        "NSegments": 2,
    }
    method = {
        "Method": "<Bruker:EPI>" if sequence == "epi" else "<Bruker:FISP>",
        "PVM_EncNReceivers": 1,
        "PVM_DigNp": 4,
        "PVM_EncMatrix": [4, 2],
        "PVM_EncSteps1": [-1, 0],
        "PVM_NMovieFrames": 2,
        "PVM_DwNDiffExp": 2,
        "PVM_EpiNEchoes": 2,
        "PVM_EpiNShots": 1,
    }
    write_paravision_parameters(tmp_path / "acqp", acqp)
    write_paravision_parameters(tmp_path / "method", method)
    samples = np.zeros((2, 6, 4, 2), dtype="<i4")
    samples[..., 0] = np.array(order)[None, :, None] + 1
    if sequence == "epi":
        samples = samples.transpose(1, 0, 2, 3)
    (tmp_path / "fid").write_bytes(samples.tobytes())
    original_bytes = (tmp_path / "fid").read_bytes()
    decoded = Dataset(tmp_path / "fid").data
    np.testing.assert_array_equal(decoded[0, 0, :, 0, 0], np.arange(1, 7))
    original_decoded = decoded.copy()
    handler = SimpleNamespace(
        path=tmp_path,
        matrix=np.array([4, 2, 3]),
        n_slices=6 if sequence == "epi" else 3,
        spatial_dimension="2D",
        n_echoes=1,
        n_inversion_time=1,
        b_values=[],
        method=method,
        acqp=acqp,
    )
    if sequence == "epi":
        actual, dims, _ = BrukerEpiRawRecoStrategy()._reconstruct_2d(
            handler, {"apply_phase_correction": False}
        )
        labels = actual[0, 0, :, 0]
        expected = np.arange(1, 7)
    else:
        strategy = (
            BrukerBssfpRawRecoStrategy
            if sequence == "bssfp"
            else BrukerDtiRawRecoStrategy
        )()
        monkeypatch.setattr(
            strategy, "_finish_reconstruction", lambda h, k, d, **kw: (k, d)
        )
        actual, dims = strategy.reconstruct(handler)
        labels = actual[0, 0, :, :, 0, 0]
        expected = (
            np.arange(1, 7).reshape(3, 2).T
            if sequence == "bssfp"
            else np.arange(1, 7).reshape(2, 3).T
        )
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(decoded, original_decoded)
    assert (tmp_path / "fid").read_bytes() == original_bytes


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("shots", [1, 2])
@pytest.mark.parametrize("channels,repetitions", [(1, 1), (2, 2)])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize(
    "format_name,code", [("GO_32BIT_SGN_INT", "i4"), ("GO_16BIT_SGN_INT", "i2")]
)
def test_epi_3d_declared_train_padding_preserves_loops(
    tmp_path, padded, shots, channels, repetitions, train, format_name, code
):
    """Invented read trains retain receivers/repetitions and shot polarity.

    ACQ_size declares scalar points per train. Each train stores all echoes
    of one shot for each receiver and has one optional 1024-byte tail.
    Labels encode every independent loop; all comparisons are exact.
    """
    from hypermri.vendors.bruker import BrukerHandler

    read, phase, phase2 = 4, 6, 2
    echoes = phase // shots
    acquired = (
        np.arange(repetitions * phase2 * shots * channels * echoes * read).reshape(
            repetitions, phase2, shots, channels, echoes, read
        )
        + 1
    )
    train_echoes = echoes if train else 1
    logical = (
        acquired.reshape(-1, channels * echoes * read)
        if train
        else acquired.transpose(0, 1, 2, 4, 3, 5).reshape(-1, channels * read)
    )
    dtype = np.dtype("<" + code)
    scalar = np.zeros((len(logical), 2 * logical.shape[1]), dtype=dtype)
    scalar[:, 0::2] = logical
    if padded:
        blocks = np.full((len(logical), 1024 // dtype.itemsize), 9999, dtype=dtype)
        blocks[:, : scalar.shape[1]] = scalar
        scalar = blocks
    (tmp_path / "fid").write_bytes(scalar.tobytes())
    acqp = {
        "NR": repetitions,
        "NI": 1,
        "ACQ_size": [2 * train_echoes * read, phase // train_echoes, phase2],
        "ACQ_scan_size": "ACQ_phase_factor_scans" if train else "One_scan",
        "ACQ_phase_factor": train_echoes,
        "GO_block_size": "Standard_KBlock_Format" if padded else "continuous",
        "GO_raw_data_format": format_name,
        "BYTORDA": "little",
        "AQ_mod": "qdig",
    }
    handler = BrukerHandler.__new__(BrukerHandler)
    handler.path = tmp_path
    handler.acqp = acqp
    handler.matrix = np.array([read, phase, phase2])
    handler.method = {
        "PVM_EncMatrix": [read, phase, phase2],
        "PVM_EncNReceivers": channels,
        "PVM_EncSteps1": list(range(-phase // 2, phase // 2)),
        "PVM_EncSteps2": [-1, 0],
        "PVM_EpiNShots": shots,
        "PVM_EpiNEchoes": echoes,
    }
    handler.b_values = []
    handler.fid = handler.load_fid_file()
    original = handler.fid.copy()
    expected = acquired.copy()
    expected[:, :, :, :, 1::2, :] = expected[:, :, :, :, 1::2, ::-1]
    expected = (
        expected.transpose(0, 1, 2, 4, 3, 5)
        .reshape(repetitions, phase2, phase, channels, read)
        .transpose(4, 2, 1, 0, 3)
    )
    strategy = BrukerEpiRawRecoStrategy()
    actual, dims, _ = strategy._reconstruct_3d(handler, {})
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(strategy._reconstruct_3d(handler, {})[0], actual)
    np.testing.assert_array_equal(handler.fid, original)
    assert dims == [
        "read_idx",
        "phase_idx",
        "slice_idx",
        "acquisition_idx",
        "channel_idx",
    ]


def test_epi_slice_echo_product_is_rejected_before_losing_labels():
    """Unvalidated product loops must not become a single acquisition axis."""
    handler = SimpleNamespace(
        n_slices=2,
        n_echoes=3,
        n_inversion_time=1,
        b_values=[],
        method={"Method": "<Bruker:T2_EPI>"},
    )
    source = np.ones((4, 2, 6, 1, 1), dtype=complex)
    before = source.copy()
    with pytest.raises(ValueError, match="simultaneous slice/echo"):
        BrukerEpiRawRecoStrategy()._semantic_2d_layout(handler, source)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize(
    ("block_size", "actual_samples", "expected"),
    [
        ("continuous", 7, 8),
        ("continuous", 9, 8),
        ("Standard_KBlock_Format", 255, 256),
        ("Standard_KBlock_Format", 257, 256),
    ],
)
def test_block_padding_rejects_mismatched_raw_sample_count_with_metadata(
    block_size, actual_samples, expected
):
    """Unsupported synthetic counts fail before reshape with layout context."""
    # ACQ_size[0]=8 real samples means four complex samples per line; two
    # lines follow from ACQ_size[1], NI and NR. Standard blocks store 128
    # complex values per line for this 32-bit fixture; continuous stores 8.
    acqp = {
        "ACQ_size": [8, 2],
        "NI": 1,
        "NR": 1,
        "GO_block_size": block_size,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
    }
    with pytest.raises(
        ValueError,
        match=(
            rf"block-padding removal.*got {actual_samples} complex samples, "
            rf"expected {expected}.*ACQ_size=.*NI=1.*NR=1.*receivers=1.*"
            r"Check that the FID and its acquisition parameter files"
        ),
    ):
        _strip_block_padding(np.zeros(actual_samples, dtype=np.complex64), acqp, 1)


@pytest.mark.parametrize("block_size", ["continuous", "Standard_KBlock_Format"])
def test_block_padding_valid_layouts_preserve_samples_and_dtype(block_size):
    """Valid padded and unpadded synthetic lines keep exact samples and dtype."""
    acqp = {
        "ACQ_size": [8, 2],
        "NI": 1,
        "NR": 1,
        "GO_block_size": block_size,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
    }
    row_samples = 128 if block_size == "Standard_KBlock_Format" else 4
    raw = np.full((2, row_samples), -9 + 4j, dtype=np.complex64)
    raw[:, :4] = np.arange(8, dtype=np.float32).reshape(2, 4) + 1j
    before = raw.copy()

    actual = _strip_block_padding(raw, acqp, 1)

    assert actual.dtype == raw.dtype
    np.testing.assert_array_equal(actual, before[:, :4])
    np.testing.assert_array_equal(raw, before)


def test_unsupported_bssfp_3d_fails_before_binary_reader(monkeypatch):
    """Unsupported 3D encoding must name the explicit scanner-data path."""

    def unexpected_read(path):
        pytest.fail("Unsupported geometry must be rejected before loading fid")

    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.bssfp.Dataset", unexpected_read
    )
    handler = SimpleNamespace(spatial_dimension="3D")
    with pytest.raises(ValueError, match=r"scan.plot\(scan.scanner_data\)"):
        BrukerBssfpRawRecoStrategy().reconstruct(handler)


@pytest.mark.parametrize("input_order", ["REV_ALT_ROWS", "NO_REORDERING"])
@pytest.mark.parametrize("diffusion", [False, True])
def test_epi_mirroring_uses_actual_reader_state(
    tmp_path, write_paravision_parameters, monkeypatch, input_order, diffusion
):
    """Loaded RECO declarations produce the same shot-corrected k-space.

    The binary contains invented integer ramps; the reference reverses odd
    acquired echoes before sorting interleaved shot lines. Equality is exact.
    """
    read, phase, shots, objects = 4, 6, 2, 2
    steps = [-3, -1, 1, -2, 0, 2]
    acquired = (
        np.arange(objects * shots * 3 * read).reshape(objects, shots, 3, read) + 1
    )
    scalar = np.zeros((*acquired.shape, 2), dtype="<i4")
    scalar[..., 0] = acquired
    (tmp_path / "fid").write_bytes(scalar.tobytes())
    acqp = {
        "ACQ_dim": 2,
        "ACQ_size": [24, 2],
        "NI": objects,
        "NR": 1,
        "ACQ_phase_factor": 3,
        "NSegments": shots,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "GO_block_size": "continuous",
        "AQ_mod": "qdig",
        "ACQ_scan_size": "ACQ_phase_factor_scans",
        "ACQ_obj_order": [0, 1],
        "PULPROG": "<DtiEpi.ppg>" if diffusion else "<EPI.ppg>",
    }
    method = {
        "Method": "<Bruker:DtiEpi>" if diffusion else "<Bruker:EPI>",
        "PVM_EncNReceivers": 1,
        "PVM_DigNp": 12,
        "PVM_EncMatrix": [read, phase],
        "PVM_EncSteps1": steps,
        "PVM_EpiNShots": shots,
        "PVM_EpiNEchoes": 3,
    }
    write_paravision_parameters(tmp_path / "acqp", acqp)
    write_paravision_parameters(tmp_path / "method", method)
    write_paravision_parameters(tmp_path / "reco", {"RECO_inp_order": input_order})
    dataset = Dataset(tmp_path / "fid", load=False)
    dataset.load_parameters()
    dataset.add_parameter_file(tmp_path / "reco")
    dataset.load_properties()
    dataset.load_schema()
    context = (
        pytest.warns(RuntimeWarning, match="RECO_inp_order")
        if input_order == "NO_REORDERING"
        else nullcontext()
    )
    with context:
        dataset.load_data()
    before = dataset.data.copy()
    original_bytes = (tmp_path / "fid").read_bytes()
    if diffusion:
        assert dataset.data.shape == (read, phase, objects, 1, 1, 1)
        assert dataset.dim_type[-1] == "k_space_encode_step_2"
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.epi.Dataset", lambda p: dataset
    )
    handler = SimpleNamespace(
        path=tmp_path,
        matrix=[read, phase, objects],
        n_slices=objects,
        n_echoes=1,
        n_inversion_time=1,
        b_values=[],
        method=method,
        acqp=acqp,
    )
    actual, _, _ = BrukerEpiRawRecoStrategy()._reconstruct_2d(
        handler, {"apply_phase_correction": False}
    )
    expected = acquired.copy()
    expected[:, :, 1::2, :] = expected[:, :, 1::2, ::-1]
    expected = expected.reshape(objects, phase, read)[:, np.argsort(steps)].transpose(
        2, 1, 0
    )
    np.testing.assert_array_equal(actual[..., 0], expected)
    np.testing.assert_array_equal(dataset.data, before)
    repeated, _, _ = BrukerEpiRawRecoStrategy()._reconstruct_2d(
        handler, {"apply_phase_correction": False}
    )
    np.testing.assert_array_equal(repeated, actual)
    assert (tmp_path / "fid").read_bytes() == original_bytes


@pytest.mark.parametrize(
    "extra_size,labels",
    [
        (
            2,
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "k_space_encode_step_2",
            ],
        ),
        (
            1,
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "echo",
            ],
        ),
        (1, []),
        (
            1,
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "repetition",
                "object",
                "channel",
                "k_space_encode_step_2",
            ],
        ),
    ],
)
def test_epi_rejects_unverified_extra_axes(monkeypatch, tmp_path, extra_size, labels):
    """An extra acquisition loop must not be mistaken for singleton encoding."""
    raw = np.zeros((4, 6, 1, 2, 2, extra_size), dtype=np.complex64)
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.epi.Dataset",
        lambda path: SimpleNamespace(data=raw, dim_type=labels),
    )
    with pytest.raises(ValueError, match="Unsupported Bruker 2D EPI fid layout"):
        BrukerEpiRawRecoStrategy()._reconstruct_2d(SimpleNamespace(path=tmp_path), {})


@pytest.mark.parametrize("shots", [1, 2])
@pytest.mark.parametrize("objects,repetitions,channels", [(1, 1, 1), (3, 3, 2)])
def test_depi_singleton_encoding_axis_preserves_labelled_loops(
    tmp_path,
    write_paravision_parameters,
    monkeypatch,
    shots,
    objects,
    repetitions,
    channels,
):
    """The real dEPI reader retains all loops and one explicit unused encoding axis.

    Integer ramps encode every acquired loop. The k-space reference reverses
    odd echoes within each shot and independently sorts phase/object labels.
    The image reference uses explicit inverse DFT matrices (absolute tolerance
    1e-10 in invented scanner units, complex128). Geometry is isolated here;
    no measured samples or scanner/raw agreement claim is involved.
    """
    read, phase = 4, 6
    echoes = phase // shots
    order = [2, 0, 1] if objects == 3 else [0]
    steps = list(range(-3, 3)) if shots == 1 else [-3, -1, 1, -2, 0, 2]
    acquired = (
        np.arange(repetitions * objects * shots * channels * echoes * read).reshape(
            repetitions, objects, shots, channels, echoes, read
        )
        + 1
    )
    scalar = np.zeros((*acquired.shape, 2), dtype="<i4")
    scalar[..., 0] = acquired
    scalar[..., 1] = -acquired
    (tmp_path / "fid").write_bytes(scalar.tobytes())
    acqp = {
        "ACQ_dim": 2,
        "ACQ_size": [2 * read * echoes, shots],
        "NI": objects,
        "NR": repetitions,
        "NSegments": shots,
        "ACQ_phase_factor": echoes,
        "ACQ_obj_order": order,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "GO_block_size": "continuous",
        "PULPROG": "<DtiEpi.ppg>",
        "AQ_mod": "qdig",
        "ACQ_scan_size": "ACQ_phase_factor_scans",
    }
    method = {
        "Method": "<Bruker:DtiEpi>",
        "PVM_EncNReceivers": channels,
        "PVM_DigNp": read * echoes,
        "PVM_EncMatrix": [read, phase],
        "PVM_EncSteps1": steps,
        "PVM_EpiNEchoes": echoes,
        "PVM_EpiNShots": shots,
    }
    write_paravision_parameters(tmp_path / "acqp", acqp)
    write_paravision_parameters(tmp_path / "method", method)
    dataset = Dataset(tmp_path / "fid")
    assert dataset.scheme_id == "dEPI"
    assert dataset.data.shape == (read, phase, objects, repetitions, channels, 1)
    assert dataset.dim_type == [
        "k_space_encode_step_0",
        "k_space_encode_step_1",
        "object",
        "repetition",
        "channel",
        "k_space_encode_step_2",
    ]
    original = dataset.data.copy()
    stored = (tmp_path / "fid").read_bytes()
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.epi.Dataset", lambda path: dataset
    )
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.cartesian._attach_image_geometry",
        lambda array, handler: array,
    )
    handler = SimpleNamespace(
        path=tmp_path,
        matrix=np.array([read, phase, objects]),
        n_slices=objects,
        n_echoes=1,
        n_inversion_time=1,
        b_values=np.arange(repetitions) * 500,
        method=method,
        acqp=acqp,
        spatial_dimension="2D",
    )
    expected = acquired.astype(complex) * (1 - 1j)
    expected[..., 1::2, :] = expected[..., 1::2, ::-1]
    expected = expected.transpose(5, 2, 4, 1, 0, 3).reshape(
        read, phase, objects, repetitions, channels
    )
    expected = expected[:, np.argsort(steps), :, :, :][:, :, np.argsort(order), :, :]
    if repetitions == 1:
        expected = expected[:, :, :, 0, :]
    strategy = BrukerEpiRawRecoStrategy()
    actual, dims, _ = strategy._reconstruct_2d(
        handler, {"apply_phase_correction": False}
    )
    np.testing.assert_array_equal(actual, expected)
    assert dims == ["read_idx", "phase_idx", "slice_idx"] + (
        ["bvalue_idx"] if repetitions > 1 else []
    ) + ["channel_idx"]
    options = dict(
        shift_xarray=False,
        append_parameters=False,
        apply_phase_correction=False,
        data_precision="float64",
    )
    image = strategy.reconstruct(handler, **options)
    centered = np.fft.ifftshift(expected, axes=(0, 1))
    matrices = [
        np.exp(2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) / n
        for n in (read, phase)
    ]
    reference = np.fft.fftshift(
        np.einsum("ij,kl,jl...->ik...", *matrices, centered), axes=(0, 1)
    )
    np.testing.assert_allclose(image.values, reference, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(strategy.reconstruct(handler, **options), image)
    assert image.dims == tuple(dims)
    assert image.sizes["slice_idx"] == objects
    assert image.sizes["channel_idx"] == channels
    if repetitions > 1:
        np.testing.assert_array_equal(image.bvalue_idx, handler.b_values)
        assert image.bvalue_idx.attrs["unit"] == "s/mm2"
    assert image.dtype == np.complex128
    np.testing.assert_array_equal(dataset.data, original)
    assert (tmp_path / "fid").read_bytes() == stored


@pytest.mark.parametrize(
    "shape,labels,scheme",
    [
        (
            (4, 6, 1, 1, 1, 2),
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "k_space_encode_step_2",
            ],
            "dEPI",
        ),
        ((4, 6, 1, 1, 1, 1), None, "dEPI"),
        (
            (4, 6, 1, 1, 1, 1),
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "channel",
                "repetition",
                "k_space_encode_step_2",
            ],
            "dEPI",
        ),
        (
            (4, 6, 1, 1, 1, 1),
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "unknown",
            ],
            "dEPI",
        ),
        (
            (4, 6, 1, 1, 1, 1),
            [
                "k_space_encode_step_0",
                "k_space_encode_step_1",
                "object",
                "repetition",
                "channel",
                "k_space_encode_step_2",
            ],
            "EPI",
        ),
        ((4, 6, 1, 1, 1, 1, 1), [], "dEPI"),
    ],
)
def test_epi_rejects_unverified_decoder_contract(
    tmp_path, monkeypatch, shape, labels, scheme
):
    """An extra singleton is accepted only with the verified decoder contract."""
    source = np.ones(shape, dtype=complex)
    dataset = SimpleNamespace(data=source, dim_type=labels, scheme_id=scheme)
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.epi.Dataset", lambda path: dataset
    )
    with pytest.raises(ValueError, match="2D EPI fid layout"):
        BrukerEpiRawRecoStrategy()._reconstruct_2d(SimpleNamespace(path=tmp_path), {})
    np.testing.assert_array_equal(source, np.ones(shape))
