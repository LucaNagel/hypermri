"""Characterize the geometry of single-voxel Bruker spectroscopy (PRESS).

All parameter files are synthetic. The voxel edge lengths (4.5, 3.5, 7.5 mm)
are invented; the unused ``PVM_Matrix = 128`` mimics the leftover imaging
parameter that ParaVision writes into PRESS ``method`` files. A single-voxel
acquisition has one voxel, so matrix and resolution follow from the voxel size
exactly (1e-12 rounding allowed).
"""

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler

VOXEL_SIZE_MM = [4.5, 3.5, 7.5]


def _press_handler(tmp_path, write_parameters, **method_overrides):
    """Load a synthetic PRESS scan through the real parameter parser."""
    scan_dir = tmp_path / "1"
    method = {
        "Method": "<Bruker:PRESS>",
        "PVM_EncSpectroscopy": "Yes",
        "PVM_Matrix": [128],
        "PVM_VoxArrSize": [VOXEL_SIZE_MM],
        "PVM_NVoxels": 1,
        "PVM_SpecMatrix": [2048],
        "PVM_SpecSWH": [6000.0],
        "PVM_SpecSW": [20.0],
        **method_overrides,
    }
    write_parameters(scan_dir / "method", method)
    write_parameters(
        scan_dir / "acqp",
        {"ACQ_grad_matrix": [np.eye(3).tolist()], "ACQ_patient_pos": "Head_Prone"},
    )
    write_parameters(scan_dir / "pdata" / "1" / "reco", {})
    return BrukerHandler(scan_dir, log_mode="critical")


def test_single_voxel_matrix_is_one_voxel_despite_unused_imaging_matrix(
    tmp_path, write_paravision_parameters
):
    handler = _press_handler(tmp_path, write_paravision_parameters)

    assert handler.is_spectroscopic
    np.testing.assert_array_equal(handler.matrix, [1, 1, 1])


def test_single_voxel_resolution_is_the_voxel_size(
    tmp_path, write_paravision_parameters
):
    handler = _press_handler(tmp_path, write_paravision_parameters)

    np.testing.assert_allclose(handler.fov_mm, VOXEL_SIZE_MM, rtol=1e-12)
    np.testing.assert_allclose(handler.resolution_mm, VOXEL_SIZE_MM, rtol=1e-12)


def test_spectroscopic_scan_with_a_real_fov_keeps_its_imaging_matrix(
    tmp_path, write_paravision_parameters
):
    """Only a field of view taken from the voxel size implies a 1x1x1 matrix."""
    handler = _press_handler(
        tmp_path,
        write_paravision_parameters,
        Method="<Bruker:CSI>",
        PVM_SpatDimEnum="<2D>",
        PVM_Matrix=[8, 6],
        PVM_Fov=[32.0, 24.0],
    )

    np.testing.assert_array_equal(handler.matrix, [8, 6, handler.n_slices])
    np.testing.assert_allclose(handler.fov_mm[:2], [32.0, 24.0], rtol=1e-12)
