"""Check how ``BSSFP`` reads its echo or movie-frame count.

The scans are the synthetic series of ``synthetic_flash_scan`` with an
invented bSSFP ``method`` file; the counts are arbitrary distinct integers.
"""

import pytest

from hypermri.sequences.BSSFP import BSSFP
from hypermri.vendors.bruker import BrukerHandler


def _bssfp(scan, write_paravision_parameters, parameters):
    write_paravision_parameters(
        scan / "method", {"Method": "<Bruker:FISP>", **parameters}
    )
    return BSSFP(BrukerHandler(scan, log_mode="error"), load_data="lazy")


@pytest.mark.parametrize(
    ("parameters", "n_echoes"),
    [
        ({"PVM_NEchoImages": 3, "PVM_NMovieFrames": 5}, 3),
        ({"PVM_NMovieFrames": 5}, 5),
    ],
)
def test_echo_count_prefers_echo_images_over_movie_frames(
    synthetic_flash_scan, write_paravision_parameters, parameters, n_echoes
):
    bssfp = _bssfp(synthetic_flash_scan, write_paravision_parameters, parameters)

    assert bssfp.n_echoes == n_echoes


def test_missing_counts_name_the_parameter_and_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    with pytest.raises(KeyError) as error:
        _bssfp(synthetic_flash_scan, write_paravision_parameters, {})

    message = str(error.value)
    assert "'PVM_NMovieFrames'" in message
    assert str(synthetic_flash_scan) in message
