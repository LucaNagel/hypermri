"""Check how ``MGE`` reads its echo timing and reference frequency.

The scans are the synthetic series of ``synthetic_flash_scan`` with an
invented MGE ``method`` file: three echoes at 2, 4 and 6 ms and a 0-ppm
reference frequency of 75.4 MHz (13C at 7 T). The expected values follow
from these numbers, not from the implementation.
"""

import numpy as np
import pytest

from hypermri.sequences.MGE import MGE
from hypermri.vendors.bruker import BrukerHandler

ECHO_TIMES_MS = [2.0, 4.0, 6.0]


def _mge(scan, write_paravision_parameters, parameters):
    write_paravision_parameters(
        scan / "method", {"Method": "<Bruker:MGE>", **parameters}
    )
    return MGE(BrukerHandler(scan, log_mode="error"), load_data="lazy")


def test_echo_times_and_ppm_bandwidth_follow_the_method_file(
    synthetic_flash_scan, write_paravision_parameters
):
    mge = _mge(
        synthetic_flash_scan,
        write_paravision_parameters,
        {"EffectiveTE": ECHO_TIMES_MS, "PVM_FrqRef": [75.4, 0.0]},
    )

    np.testing.assert_allclose(mge.TEs, ECHO_TIMES_MS)
    # Echo spacing 2 ms gives 500 Hz; 500 Hz / 75.4 MHz = 6.631 ppm.
    assert mge.bandwidth_hz == pytest.approx(500.0)
    assert mge.bandwidth_ppm == pytest.approx(500.0 / 75.4)


def test_ppm_bandwidth_is_unknown_without_reference_frequency(
    synthetic_flash_scan, write_paravision_parameters
):
    mge = _mge(
        synthetic_flash_scan,
        write_paravision_parameters,
        {"EffectiveTE": ECHO_TIMES_MS},
    )

    assert mge.bandwidth_ppm is None


def test_missing_echo_times_name_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    with pytest.raises(KeyError) as error:
        _mge(synthetic_flash_scan, write_paravision_parameters, {})

    message = str(error.value)
    assert "'EffectiveTE'" in message
    assert str(synthetic_flash_scan) in message
