"""Check how ``NSPECT.get_spec`` reads the spectroscopic acquisition time.

The scans are the synthetic series of ``synthetic_flash_scan`` with an
invented NSPECT ``method`` file and a replaced FID of four points.
"""

import numpy as np
import pytest

from hypermri.sequences.NSPECT import NSPECT
from hypermri.vendors.bruker import BrukerHandler


def _nspect(scan, write_paravision_parameters, parameters):
    write_paravision_parameters(
        scan / "method", {"Method": "<Bruker:NSPECT>", **parameters}
    )
    nspect = NSPECT(BrukerHandler(scan, log_mode="error"), load_data="lazy")
    nspect.handler.rawdatajob0 = np.ones(4, dtype=complex)
    nspect.n_repetitions = 1
    nspect.n_acquisition = 4
    return nspect


def test_spectrum_uses_the_acquisition_time_of_the_method_file(
    synthetic_flash_scan, write_paravision_parameters
):
    nspect = _nspect(
        synthetic_flash_scan,
        write_paravision_parameters,
        {"PVM_SpecAcquisitionTime": 300.0},
    )

    _, fids, _ = nspect.get_spec(LB=1.0)

    # Line broadening exp(-2 pi LB t) on the time axis 0..0.3 s.
    time_s = np.linspace(0.0, 0.3, 4)
    np.testing.assert_allclose(fids, np.exp(-2.0 * np.pi * time_s), rtol=1e-12)


def test_missing_acquisition_time_names_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    nspect = _nspect(synthetic_flash_scan, write_paravision_parameters, {})

    with pytest.raises(KeyError) as error:
        nspect.get_spec()

    # KeyError.__str__ quotes its argument and doubles Windows backslashes.
    message = error.value.args[0]
    assert "'PVM_SpecAcquisitionTime'" in message
    assert str(synthetic_flash_scan) in message
