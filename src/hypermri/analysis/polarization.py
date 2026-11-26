"""Deterministic magnetization evolution in hyperpolarized pulse trains."""

import numpy as np

__all__ = ["simulate_hyperpolarized_pulse_train"]


def simulate_hyperpolarized_pulse_train(
    initial_magnetization: float,
    flip_angles_rad: np.ndarray,
    relaxation_per_interval: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate longitudinal and transverse signals for a pulse train.

    Parameters
    ----------
    initial_magnetization : float
        Initial longitudinal magnetization in arbitrary signal units.
    flip_angles_rad : numpy.ndarray
        One-dimensional flip angles in radians, in excitation order.
    relaxation_per_interval : float, optional
        Longitudinal retention between pulses. Use ``exp(-TR_s / T1_s)`` for
        constant repetition and relaxation times, or one for RF depletion alone.
        Equilibrium recovery is excluded from this hyperpolarized model.

    Returns
    -------
    longitudinal : numpy.ndarray
        Float64 magnetization immediately before each pulse, shape ``(pulse,)``.
    transverse : numpy.ndarray
        Float64 transverse signal immediately after each pulse, same shape and
        signal units. Each pulse depletes longitudinal signal by its cosine.

    Raises
    ------
    ValueError
        If flip angles are not one-dimensional.

    Notes
    -----
    Evaluation is eager, deterministic and uncached. Inputs are not mutated;
    no plotting, metadata access, noise or RF spoiling model is introduced.

    Examples
    --------
    >>> mz, mxy = simulate_hyperpolarized_pulse_train(2., np.array([0., 0.]), .5)
    >>> mz.tolist()
    [2.0, 1.0]
    """
    angles = np.asarray(flip_angles_rad)
    if angles.ndim != 1:
        raise ValueError("flip_angles_rad must be one-dimensional in pulse order")
    longitudinal = np.zeros(angles.size)
    transverse = np.zeros(angles.size)
    current = initial_magnetization
    for index, angle in enumerate(angles):
        longitudinal[index] = current
        transverse[index] = current * np.sin(angle)
        current = current * np.cos(angle) * relaxation_per_interval
    return longitudinal, transverse
