"""Check that the CSI encoding order names a missing parameter and its scan.

The scan is the synthetic series of ``synthetic_flash_scan`` with an invented
CSI ``method`` file that lacks ``PVM_EncSteps1``.
"""

import pytest

from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy
from hypermri.vendors.bruker import BrukerHandler


def test_missing_phase_steps_name_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {
            "Method": "<Bruker:CSI>",
            "PVM_Matrix": "( 2 )\n3 2",
            "PVM_EncOrder": "( 2 )\nLINEAR_ENC LINEAR_ENC",
            "PVM_EncSteps0": "( 3 )\n0 -1 1",
        },
    )
    handler = BrukerHandler(synthetic_flash_scan, log_mode="error")

    with pytest.raises(KeyError) as error:
        BrukerCsiRecoStrategy().get_phase_encoding_order(handler)

    # KeyError.__str__ quotes its argument and doubles Windows backslashes.
    message = error.value.args[0]
    assert "'PVM_EncSteps1'" in message
    assert str(synthetic_flash_scan) in message
