"""Tests for the order in which a Bruker CSI scan visits its k-space positions.

All handlers are synthetic: a 3 x 2 matrix (read x phase) with invented
encoding steps.  A signed step is relative to the matrix centre ``n // 2``, so
the read steps ``[0, -1, 1]`` are the positions ``[1, 0, 2]`` and the phase
steps ``[0, -1]`` are the positions ``[1, 0]``.  The phase steps form the outer
loop of the acquisition, which gives the expected pairs below.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy

STOCK_METHODS = ["<Bruker:CSI>", "<User:csi_fix>", "<User:geoffCSI>"]
CUSTOM_METHODS = [
    "<User:lucaCSI>",
    "<User:lucaCSI2>",
    "<User:lucaCSI3>",
    "<User:lucaCSI4>",
]

STOCK_READ = [1, 0, 2, 1, 0, 2]
STOCK_PHASE = [1, 1, 1, 0, 0, 0]

# One invented read/phase pair per acquired FID, as a custom method stores it.
PAIRED_READ = np.array([1, 1, 0, 2, 0, 2])
PAIRED_PHASE = np.array([1, 0, 1, 1, 0, 0])


def _handler(method: str, **parameters) -> SimpleNamespace:
    """Return a handler that has only the ``method`` parameters of a 3 x 2 CSI."""
    values = {
        "Method": method,
        "PVM_Matrix": np.array([3, 2]),
        "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
        "PVM_EncSteps0": np.array([0, -1, 1]),
        "PVM_EncSteps1": np.array([0, -1]),
        "CentricEncOrder_OnOff": "Off",
        "SpiralEncOrder_OnOff": "Off",
    }
    values.update(parameters)
    return SimpleNamespace(method=values, method_parameter=values.__getitem__)


def _encoding(handler):
    return BrukerCsiRecoStrategy().get_phase_encoding_order(handler)


def _assert_integer_positions(read, phase, expected_read, expected_phase):
    read, phase = np.asarray(read), np.asarray(phase)
    assert np.issubdtype(read.dtype, np.integer)
    assert np.issubdtype(phase.dtype, np.integer)
    np.testing.assert_array_equal(read, expected_read)
    np.testing.assert_array_equal(phase, expected_phase)


@pytest.mark.parametrize("method", STOCK_METHODS)
@pytest.mark.parametrize("axis_order", ["LINEAR_ENC", "CENTRIC_ENC"])
@pytest.mark.parametrize("as_text", [False, True], ids=["list", "text"])
def test_stock_methods_pair_every_phase_step_with_all_read_steps(
    method, axis_order, as_text
):
    order = f"{axis_order} {axis_order}" if as_text else [axis_order, axis_order]

    name, read, phase = _encoding(_handler(method, PVM_EncOrder=order))

    assert name == f"{axis_order} {axis_order}"
    _assert_integer_positions(read, phase, STOCK_READ, STOCK_PHASE)


@pytest.mark.parametrize("method", CUSTOM_METHODS)
@pytest.mark.parametrize(
    ("switch", "prefix", "expected_name"),
    [
        ("CentricEncOrder_OnOff", "CentricEncOrderMatrix", "luca_centric"),
        ("SpiralEncOrder_OnOff", "SpiralEncOrderMatrix", "luca_spiral"),
    ],
)
@pytest.mark.parametrize("signed", [False, True], ids=["absolute", "signed"])
def test_custom_methods_keep_their_stored_pairs(
    method, switch, prefix, expected_name, signed
):
    """Stored positions without a negative value are grid positions already.

    With a negative value they are relative to the matrix centre (1, 1).
    """
    offset = 1 if signed else 0
    handler = _handler(
        method,
        **{
            switch: "On",
            f"{prefix}x": PAIRED_READ - offset,
            f"{prefix}y": PAIRED_PHASE - offset,
        },
    )

    name, read, phase = _encoding(handler)

    assert name == expected_name
    _assert_integer_positions(read, phase, PAIRED_READ, PAIRED_PHASE)


@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_custom_method_without_stored_pairs_uses_the_scanner_steps(method):
    name, read, phase = _encoding(
        _handler(method, PVM_EncOrder=["CENTRIC_ENC", "CENTRIC_ENC"])
    )

    assert name == "CENTRIC_ENC CENTRIC_ENC"
    _assert_integer_positions(read, phase, STOCK_READ, STOCK_PHASE)


def test_encoding_order_is_repeatable_and_leaves_the_parameters_unchanged():
    handler = _handler(
        "<User:lucaCSI4>",
        CentricEncOrder_OnOff="On",
        CentricEncOrderMatrixx=PAIRED_READ - 1,
        CentricEncOrderMatrixy=PAIRED_PHASE - 1,
    )
    before = {key: np.array(value) for key, value in handler.method.items()}

    first = _encoding(handler)
    second = _encoding(handler)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    for key, value in before.items():
        np.testing.assert_array_equal(handler.method[key], value)


@pytest.mark.parametrize("method", ["<Bruker:CSI>", "<User:lucaCSI4>"])
@pytest.mark.parametrize(
    "order",
    [["LINEAR_ENC", "CENTRIC_ENC"], ["SPIRAL_ENC", "SPIRAL_ENC"], "LINEAR_ENC"],
    ids=["mixed", "unknown", "one-axis"],
)
def test_unsupported_scanner_order_names_the_parameter_and_its_value(method, order):
    """Regression: these orders ended in an ``UnboundLocalError``."""
    text = order if isinstance(order, str) else " ".join(order)

    with pytest.raises(ValueError, match="PVM_EncOrder") as error:
        _encoding(_handler(method, PVM_EncOrder=order))

    assert repr(text) in str(error.value)
    assert "'LINEAR_ENC LINEAR_ENC'" in str(error.value)  # supported orders listed


def test_unknown_method_names_the_method_instead_of_guessing_a_linear_order():
    """Regression: the result was ``None`` with unpaired floating-point axes."""
    with pytest.raises(ValueError, match="Method") as error:
        _encoding(_handler("<User:otherCSI>"))

    assert "'<User:otherCSI>'" in str(error.value)
    assert "'<Bruker:CSI>'" in str(error.value)  # supported methods listed
