"""Check how the IDEAL chemical-shift correction reads its bSSFP parameters.

The scan is the synthetic series of ``synthetic_flash_scan`` with an invented
bSSFP ``method`` file that lacks the read bandwidth.
"""

import numpy as np
import pytest

from hypermri.analysis.fitting.ideal import IDEAL_Solver
from hypermri.vendors.bruker import BrukerHandler


def test_missing_bandwidth_names_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {"Method": "<Bruker:FISP>", "PVM_Matrix": "( 2 )\n4 2", "ReadFreqList": 0},
    )
    solver = IDEAL_Solver.__new__(IDEAL_Solver)
    solver.bssfp_object = BrukerHandler(synthetic_flash_scan, log_mode="error")
    solver.solution_matrix = np.zeros((1, 4, 2, 1, 1), dtype=complex)
    solver.relative_frequencies = [0.0]
    solver.num_transients = 0

    with pytest.raises(KeyError) as error:
        solver.correct_chemical_shift()

    message = str(error.value)
    assert "'PVM_EffSWh'" in message
    assert str(synthetic_flash_scan) in message
