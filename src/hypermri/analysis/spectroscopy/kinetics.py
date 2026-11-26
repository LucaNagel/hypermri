"""Discrete metabolite time-curve ratios."""

import numpy as np

__all__ = ["calculate_pyruvate_to_lactate_auc"]


def calculate_pyruvate_to_lactate_auc(lactate_timecurve, pyruvate_timecurve):
    """Calculate the ratio of discrete lactate and pyruvate signal sums.

    Parameters
    ----------
    lactate_timecurve, pyruvate_timecurve : array_like
        Signals in compatible amplitude units, sampled at the same uniform
        intervals. Each array is summed over all axes; no time integration or
        baseline subtraction is performed.

    Returns
    -------
    numpy scalar
        Dimensionless lactate/pyruvate ratio in NumPy accumulation precision.
        Zero pyruvate sum retains NumPy's inf/NaN behavior. Evaluation is eager,
        deterministic and does not mutate either curve.
    """
    pyruvate_auc = np.sum(pyruvate_timecurve)
    lactate_auc = np.sum(lactate_timecurve)
    lac_pyr = lactate_auc / pyruvate_auc

    return lac_pyr
