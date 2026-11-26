"""Check how ``IVRecovery`` reads its inversion count and times.

The scans are the synthetic series of ``synthetic_flash_scan`` with an
invented inversion-recovery ``method`` file; the inversion times in
milliseconds are arbitrary distinct values.
"""

import numpy as np
import pytest

from hypermri.sequences.IVRecovery import IVRecovery
from hypermri.vendors.bruker import BrukerHandler


def _ivrecovery(scan, write_paravision_parameters, parameters):
    write_paravision_parameters(
        scan / "method", {"Method": "<Bruker:IVRecovery>", **parameters}
    )
    return IVRecovery(BrukerHandler(scan, log_mode="error"), load_data="lazy")


def test_inversion_count_and_times_come_from_the_method_file(
    synthetic_flash_scan, write_paravision_parameters
):
    times = np.array([10.0, 250.0, 1200.0])
    ivrecovery = _ivrecovery(
        synthetic_flash_scan,
        write_paravision_parameters,
        {"NoIT": 3, "InversionTime": "( 3 )\n10 250 1200"},
    )

    assert ivrecovery.NI == 3
    np.testing.assert_array_equal(ivrecovery.IVTimes, times)


def test_missing_inversion_count_names_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    with pytest.raises(KeyError) as error:
        _ivrecovery(
            synthetic_flash_scan,
            write_paravision_parameters,
            {"InversionTime": "( 2 )\n10 250"},
        )

    # KeyError.__str__ quotes its argument and doubles Windows backslashes.
    message = error.value.args[0]
    assert "'NoIT'" in message
    assert str(synthetic_flash_scan) in message
