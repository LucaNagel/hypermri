"""Effective NSA and chemical-shift design calculations for spectral imaging.

These deterministic NumPy operations use the existing chemical-shift table and
legacy phantom offsets. They construct no figures or notebook controls, do not
modify input arrays, and assume independent equal-variance complex echo noise.
"""

import numpy as np

from hypermri.analysis.spectroscopy.metabolites import (
    METABOLITE_CS_PPM as _METABOLITE_CS_PPM,
)
from hypermri.utils.physics import get_gmr as _get_gmr

__all__ = ["get_freq_hz", "calculate_nsa"]


def get_freq_hz(
    name: str, b0: float, use_reeder_phantom: bool, center_name: str
) -> float:
    """Return a carbon-13 metabolite's frequency relative to a selected centre.

    Parameters
    ----------
    name : str
        Lowercase key in the shared metabolite chemical-shift table. Unknown
        names retain the legacy zero-ppm fallback; this is not name validation.
    b0 : float
        Magnetic field strength in Tesla.
    use_reeder_phantom : bool
        Use stored phantom offsets (-602, -242 and -210 Hz at 3 T, relative to
        lactate) for matching names, scaled linearly with field strength.
        Otherwise use the shared ppm table and carbon-13 gyromagnetic ratio.
    center_name : str
        Centre metabolite key, subtracted using the selected calibration. A
        missing phantom centre uses zero Hz; a missing ppm centre uses zero ppm.

    Returns
    -------
    float
        Frequency offset in Hz, with the sign of the retained calibration.

    Notes
    -----
    Metadata and constants are read without mutation or caching. This is an
    eager deterministic scalar calculation, without a plotting dependency.
    """
    # Exact shifts from Reeder 2007 Paper (Fig 1b) at 3.0T
    # Relative to Lactate (0 Hz)
    reeder_offsets_3T = {
        "lactate": 0.0,
        "alanine": -210.0,
        "pyruvate": -602.0,
        "pyruvatehydrate": -242.0,  # The "Ester" peak
        "none": 0.0,
    }

    # 1. Determine the absolute frequency/offset of the target
    if use_reeder_phantom and name in reeder_offsets_3T:
        # Scale Hz from 3T to current B0
        # freq = val_at_3T * (Current_B0 / 3.0)
        abs_freq = reeder_offsets_3T[name] * (b0 / 3.0)

        # Determine center offset
        if center_name in reeder_offsets_3T:
            center_freq = reeder_offsets_3T[center_name] * (b0 / 3.0)
        else:
            center_freq = 0.0

        return abs_freq - center_freq

    else:
        # Use PPM logic
        larmor = _get_gmr(nucleus="13c", unit="MHz_T") * b0

        val_ppm = _METABOLITE_CS_PPM.get(name, 0.0)
        center_ppm = _METABOLITE_CS_PPM.get(center_name, 0.0)

        # ppm to Hz: (val_ppm - center_ppm) * Larmor_MHz
        # Larmor is in MHz/T * T = MHz
        # ppm diff * MHz = Hz
        return (val_ppm - center_ppm) * larmor


def calculate_nsa(
    metab_slots: list[str],
    center_name: str,
    b0: float,
    echo_times: np.ndarray,
    t2_stars: list[float],
    couple_pyr: bool,
    use_reeder: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Calculate effective NSA for the existing spectral design matrix.

    Parameters
    ----------
    metab_slots : list[str]
        Ordered lowercase metabolite names; 'none' slots are omitted. Duplicate
        independent names retain duplicate columns and can make the design
        singular. Unknown names retain get_freq_hz's zero-ppm fallback.
    center_name : str
        Reference metabolite used for relative frequency offsets.
    b0 : float
        Magnetic field strength in Tesla.
    echo_times : numpy.ndarray
        One-dimensional echo times in seconds, with one design row per echo.
    t2_stars : list[float]
        One T2-star in seconds per input slot. Nonpositive values disable
        decay; positive values multiply that column by exp(-echo_times/T2-star).
    couple_pyr : bool
        Combine pyruvate and hydrate oscillations with the retained weights
        0.61 and 0.39. Repeated pyruvate slots produce one coupled column;
        standalone hydrate slots are omitted when coupling is enabled.
    use_reeder : bool
        Use the retained phantom frequency calibration where available.

    Returns
    -------
    nsa : numpy.ndarray
        Float64 dimensionless effective averages, one per output design column,
        calculated as the reciprocal real diagonal of inverse(A.conj().T @ A).
        Negative values or values above 1e6 are set to zero. Singular inversion
        returns zeros; an empty design returns an empty array.
    names : list[str]
        Display names in output-column order, including 'Pyruvate (Cpl)'.
    frequencies : list[str]
        Hz offsets formatted to one decimal place, or the coupled main/hydrate
        description. These are display strings, not a numeric calibration axis.

    Raises
    ------
    IndexError
        If t2_stars has no entry for an active input slot.
    AttributeError, ValueError
        If echo_times cannot form the numeric design matrix.

    Notes
    -----
    Assumes independent equal-variance noise across echoes and a fixed frequency
    design. These values do not estimate measured noise or include frequency-fit
    uncertainty. Calculations are eager, deterministic and uncached; input
    arrays/lists are not modified. The established inverse and failure policy
    are retained rather than replaced by a pseudoinverse or regularization.

    Examples
    --------
    >>> calculate_nsa(['lactate'], 'lactate', 3.0,
    ...               np.array([0., 0.001, 0.002]), [-1.], False, True)[0]
    array([3.])
    """
    # Filter out "none"
    active_indices = [i for i, m in enumerate(metab_slots) if m != "none"]
    selected_names = [metab_slots[i] for i in active_indices]
    selected_t2s = [t2_stars[i] for i in active_indices]

    cols = []
    out_names = []
    out_freqs = []  # For display

    t = echo_times.reshape(-1, 1)  # (N_echoes, 1)

    # Reeder Coupling Weights
    r_main = 0.61
    r_ester = 0.39

    # Track if we already processed coupled pyruvate to avoid duplicates
    processed_coupled = False

    for i, name in enumerate(selected_names):
        # Calculate Decay
        t2 = selected_t2s[i]
        decay = 1.0 if t2 <= 0 else np.exp(-t / t2)

        # Logic for Coupled Pyruvate
        if couple_pyr and name == "pyruvate":
            if processed_coupled:
                continue  # Skip if we already added the coupled column

            # Freq A: Pyruvate Main
            f_a = get_freq_hz("pyruvate", b0, use_reeder, center_name)
            term_a = r_main * np.exp(1j * 2 * np.pi * f_a * t)

            # Freq B: Pyruvate Hydrate (Implicit)
            f_b = get_freq_hz("pyruvatehydrate", b0, use_reeder, center_name)
            term_b = r_ester * np.exp(1j * 2 * np.pi * f_b * t)

            cols.append(decay * (term_a + term_b))
            out_names.append("Pyruvate (Cpl)")
            out_freqs.append(f"Main: {f_a:.1f}, Hyd: {f_b:.1f}")
            processed_coupled = True

        elif couple_pyr and name == "pyruvatehydrate":
            # If coupling is on, 'hydrate' is absorbed into 'pyruvate'.
            # If 'pyruvate' wasn't selected but 'hydrate' was, we technically
            # shouldn't simulate the couple unless the user explicitly wants Pyruvate.
            # For simplicity, we skip standalone hydrate if coupling is ON
            # (assuming the user selects Pyruvate to get the couple).
            pass

        else:
            # Standard Independent Species
            freq = get_freq_hz(name, b0, use_reeder, center_name)
            osc = np.exp(1j * 2 * np.pi * freq * t)
            cols.append(decay * osc)
            out_names.append(name.capitalize())
            out_freqs.append(f"{freq:.1f}")

    if not cols:
        return np.zeros(0), [], []

    A = np.hstack(cols)

    # NSA Calculation
    AtA = np.dot(A.conj().T, A)
    try:
        cov = np.linalg.inv(AtA)
        nsa = 1.0 / np.real(np.diag(cov))
        nsa[nsa < 0] = 0
        nsa[nsa > 1e6] = 0
    except np.linalg.LinAlgError:
        nsa = np.zeros(len(out_names))

    return nsa, out_names, out_freqs
