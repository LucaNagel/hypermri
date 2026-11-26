"""Check that the Bruker raw reconstruction names a missing parameter and scan.

The scan is the synthetic series of ``synthetic_flash_scan`` with an invented
``method`` file that lacks the parameter under test.
"""

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.raw import BrukerRawRecoStrategy
from hypermri.vendors.bruker import BrukerHandler


def _handler_without_parameters(scan, write_paravision_parameters):
    write_paravision_parameters(scan / "method", {"Method": "<Bruker:FLASH>"})
    return BrukerHandler(scan, log_mode="error")


def test_missing_receiver_count_names_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    handler = _handler_without_parameters(
        synthetic_flash_scan, write_paravision_parameters
    )

    with pytest.raises(KeyError) as error:
        BrukerRawRecoStrategy().postprocess_fid_file(
            np.zeros(8, dtype=np.complex128), handler=handler
        )

    message = str(error.value)
    assert "'PVM_EncNReceivers'" in message
    assert str(synthetic_flash_scan) in message


def test_missing_phase_steps_name_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    handler = _handler_without_parameters(
        synthetic_flash_scan, write_paravision_parameters
    )

    with pytest.raises(KeyError) as error:
        BrukerRawRecoStrategy()._get_phase_encode_order(handler)

    message = str(error.value)
    assert "'PVM_EncSteps1'" in message
    assert str(synthetic_flash_scan) in message
