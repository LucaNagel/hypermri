"""Check how ``b1_calibration`` reads its RF-power range.

The scan is the synthetic series of ``synthetic_flash_scan`` with an invented
calibration ``method`` file that lacks the end of the RF-power range.
"""

import pytest

from hypermri.analysis.anatomical.b1_mapping import b1_calibration
from hypermri.vendors.bruker import BrukerHandler


def test_missing_power_range_names_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {"Method": "<User:invented>", "PulseVectorStart": 0.01},
    )
    handler = BrukerHandler(synthetic_flash_scan, log_mode="error")

    with pytest.raises(KeyError) as error:
        b1_calibration(handler, plot=False, db=False)

    # KeyError.__str__ quotes its argument and doubles Windows backslashes.
    message = error.value.args[0]
    assert "'PulseVectorEnd'" in message
    assert str(synthetic_flash_scan) in message
