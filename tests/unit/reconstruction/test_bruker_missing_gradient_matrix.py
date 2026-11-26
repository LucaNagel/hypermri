"""Name ``ACQ_grad_matrix`` when the scanner-image reconstruction lacks it.

The scan is the synthetic two-slice FLASH series of ``synthetic_flash_scan``
(invented 2 x 2 integer samples, parameters only, no ``fid``). Its ``acqp``
holds an identity ``ACQ_grad_matrix`` per slice; removing it imitates an
incomplete export. Every ParaVision ``acqp`` has the parameter.
"""

import numpy as np
import pytest

from hypermri import load_scan
from hypermri.exceptions import ReconstructionError


def test_scanner_images_name_the_missing_gradient_matrix(
    synthetic_flash_scan, write_paravision_parameters
):
    write_paravision_parameters(
        synthetic_flash_scan / "acqp", {"ACQ_scan_name": "<synthetic>"}
    )
    with pytest.raises(ReconstructionError, match="ACQ_grad_matrix") as caught:
        load_scan(synthetic_flash_scan, mode="minimal", log_mode="critical")

    message = str(caught.value)
    assert str(synthetic_flash_scan) in message
    assert "acqp" in message
    assert isinstance(caught.value.__cause__, ValueError)


def test_scanner_images_keep_the_gradient_directions(synthetic_flash_scan):
    scan = load_scan(synthetic_flash_scan, mode="minimal", log_mode="critical")

    attrs = scan.scanner_data.attrs

    for name, row in zip(
        ("read_direction", "phase_direction", "slice_direction"),
        np.eye(3, dtype=int),
        strict=True,
    ):
        np.testing.assert_array_equal(attrs[name], row)
