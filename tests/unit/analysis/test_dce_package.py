from hypermri.analysis.dce.fitting import run_dce_pipeline
from hypermri.analysis.dce.motion_correction import dce_motion_correction
from hypermri.analysis.dce.preprocessing import compute_s0_map
from hypermri.analysis.dce.renal import (
    run_renal_dce_pipeline,
)


def test_dce_package_exports_expected_entry_points():
    assert callable(compute_s0_map)
    assert callable(dce_motion_correction)
    assert callable(run_dce_pipeline)
    assert callable(run_renal_dce_pipeline)
