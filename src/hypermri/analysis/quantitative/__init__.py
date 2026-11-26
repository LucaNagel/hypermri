"""Named quantitative MRI fits and immutable supplied scientific settings.

Use fit_t1 for inversion recovery, fit_adc for diffusion, fit_b1 for acquired
RF-power maps, fit_b1_calibration for iterative single-voxel calibration with
search histories, fit_ideal for echo decomposition and fit_t2 for magnitude
echo-decay maps (echo times in seconds, output T2 in ms). Each uses its
model's scientific units, calculation precision, labelled axes and defaults.
Calls are eager and preserve inputs; importing this package fits no signal,
initializes no plotting and retains no handler cache. The generic model-key
fitting API delegates to these same operations.
"""

from .b1_calibration import fit_b1_calibration
from .fitting import fit_adc, fit_b1, fit_ideal, fit_t1, fit_t2
from .options import (
    ADCFitOptions,
    B1FitOptions,
    IDEALFitOptions,
    T1FitOptions,
    T2FitOptions,
)

__all__ = [
    "fit_t1",
    "fit_adc",
    "fit_b1",
    "fit_b1_calibration",
    "fit_ideal",
    "T1FitOptions",
    "ADCFitOptions",
    "B1FitOptions",
    "IDEALFitOptions",
    "fit_t2",
    "T2FitOptions",
]
