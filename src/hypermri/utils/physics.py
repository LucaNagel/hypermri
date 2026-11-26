"""
Physics-related functions for MRI calculations.

This module contains functions for:

- Magnetic field conversions (gauss_to_kHz, kHz_to_gauss, etc.)
- Gyromagnetic ratios and the field of a reference frequency (get_gmr,
  b0_tesla_from_reference_frequency)
- Nuclear physics (larmor frequency, zeeman energy, thermal polarization)
- Hyperpolarization (hyperpolarized_Mz_T1, etc.)
- Angle conversions (rad_to_deg, deg_to_rad, etc.)
- Flip angle corrections (flipangle_corr, etc.)
- Temperature conversions (celsius_to_kelvin, kelvin_to_celsius)
- RF power calculations (calc_B1_amp_G, calc_fa_deg, rfpower_from_refpower)
"""

import re

import numpy as np

from ..utils.system.logging import init_default_logger

# Physics constants
GAMMA_RATIOS = {
    "1H": 267522187,
    "1H (in H2O)": 267522187,
    "2H": 41066225,
    "3H": 285349779,
    "3He": 203789473,
    "7Li": 103977013,
    "13C": 67283474,
    "14N": 19337792,
    "15N": 27126030,
    "17O": 36281518,
    "19F": 251815948,
    "23Na": 70808493,
    "27Al": 69763060,
    "29Si": 53190356,
    "31P": 108394000,
    "57Fe": 8680624,
    "89Y": 13162791,
    "129Xe": 74521644,
    "195Pt": 57715900,
}


logger = init_default_logger(__name__)


def flipangle_corr(T1_obs: float, flipangle: float, TR: float):
    """
    Corrects the observed flipangle for low flipangle experiments with a certain repetition time.

    Parameters
    ----------
    T1_obs : float
        Observed T1 decay constant in seconds, usually around 50 for
        [1-13C]pyruvate.
    flipangle : float
        Flipangle in degree.
    TR : float
        Repetition time in seconds.

    Returns
    -------
    T1 : float
        corrected T1 decay constant

    """
    T1 = 1 / ((1 / T1_obs) + (np.log(np.cos(flipangle * np.pi / 180)) / TR))
    return T1


def gauss_to_kHz(B1_G: float = 0.0, nucleus: str = "13c"):
    """
    Calculate B1_G (Gauss) in kHz.

    Parameters
    ----------
    B1_G: float
        The B1 amplitude in Gauss
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_G * gmr


def kHz_to_gauss(B1_kHz: float = 0.0, nucleus: str = "13c"):
    """
    Calculate B1_kHz in Gauss.

    Parameters
    ----------
    B1_kHz: float
        The B1 amplitude in kHz
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_kHz / gmr


def deg_to_rad(angle_deg: float = 0.0):
    """
    Convert an angle from degrees to radians.

    Parameters
    ----------
    angle_deg : float, optional
        Angle in degrees. Default is 0.

    Returns
    -------
    float
        The angle in radians.
    """
    return angle_deg / 180 * np.pi


def rad_to_deg(angle_rad: float = 0.0):
    """
    Convert an angle from radians to degrees.

    Parameters
    ----------
    angle_rad : float, optional
        Angle in radians. Default is 0.

    Returns
    -------
    float
        The angle in degrees.
    """
    return angle_rad * 180 / np.pi


def calc_B1_amp_G(
    fa_deg: float = 90.0, trf_ms: float = 1.0, nucleus: str = "13c", sint: float = 1.0
):
    """
    Calculate the B1 amplitude in Gauss.

    Parameters
    ----------
    fa_deg : float, optional
        The flip angle in degrees. Default is 90.
    trf_ms : float, optional
        The pulse duration in milliseconds. Default is 1.
    nucleus : str, optional
        The nucleus type, e.g., '1h', '2h', '13c'. Default is '13c'.
    sint : float, optional
        Integration factor (0-1). Represents the area the RF pulse takes up compared to a block pulse of the same length.
        Default is 1 for a block pulse. For example, sint = 0.762 for a Gaussian of 1ms and fwhm=1/e.

    Returns
    -------
    float
        The B1 amplitude in Gauss [G]. Use `Gauss_to_kHz` to convert the amplitude to kHz.

    """
    # check if integration factor is 0:
    if sint == 0:
        logger.critical("Can't divide by 0, please change integration factor sint!")
        return None
    # check if pulse duration in [ms] is 0:
    if trf_ms == 0:
        logger.warning("Can't divide by 0, please change pulse duration trf_ms!")
        return None

    # gyromagnetic ratio in radian / Tesla / second:
    gmr_rad_Ts = get_gmr(nucleus=nucleus, unit="rad_Ts", db=False)

    # flip-angle  in radian:
    fa_rad = deg_to_rad(angle_deg=fa_deg)

    return (
        fa_rad / (gmr_rad_Ts * trf_ms * 1e-3 * sint) * 1e4
    )  # 1e4 to convert Tesla to Gauss


def calc_fa_deg(
    rfpow_W: float = 1.0,
    reference_pow_W: float = 1.0,
    trf_ms: float = 1.0,
    sint: float = 1.0,
    fa_deg: float = 90.0,
):
    """
    Calculate the flip angle in degrees based on the RF pulse amplitude, reference power, pulse duration, and integration factor.

    Parameters
    ----------
    rfpow_W : float
        RF pulse amplitude in watts.
    reference_pow_W : float
        Reference power for a 1ms pulse with sint=1 and block pulse, resulting in a 90-degree flip angle.
    trf_ms : float
        Pulse duration in milliseconds.
    sint : float
        Integration factor.
    fa_deg: float
        set flipangle in degrees.

    Returns
    -------
    float
        The flip angle in degrees.
    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"
    # print(f"calc_fa_deg: rfpow_W={rfpow_W}, reference_pow_W={reference_pow_W}, trf_ms={trf_ms}, sint={sint}, fa_deg={fa_deg}")
    # calculate flipangle from input parameters:
    return fa_deg * trf_ms * sint * np.sqrt(rfpow_W / reference_pow_W)


def calc_fa_deg_from_B1(b1_G=0.0, trf_ms=1.0, nucleus="13c", sint=1.0):
    """
    Calculate the flip angle in degrees based on the B1 amplitude, pulse duration, and integration factor.

    Parameters
    ----------
    B1_G : float
        B1 amplitude in Gauss.
    trf_ms : float
        Pulse duration in milliseconds.
    nucleus : str
        Nucleus type, e.g., '1h', '2h', '13c'.
    sint : float
        Integration factor (0-1). Represents the area the RF pulse takes up compared to a block pulse of the same length.
        Default is 1 for a block pulse. For example, sint = 0.762 for a Gaussian of 1ms and fwhm=1/e.

    Returns
    -------
    float
        The flip angle in degrees.
    """
    # gyromagnetic ratio in radian / Tesla / second:
    gmr_rad_Ts = get_gmr(nucleus=nucleus, unit="rad_Ts", db=False)

    # Calculate flip angle in radians
    fa_rad = (
        b1_G * gmr_rad_Ts * trf_ms * 1e-3 * sint / 1e4
    )  # 1e4 to convert Gauss to Tesla

    # Convert to degrees
    fa_deg = rad_to_deg(angle_rad=fa_rad)

    return fa_deg


def get_gmr(nucleus: str = "13c", unit: str = "MHz_T", db: bool = False):
    """
    Get the gyromagnetic ratio of nearly any nucleus.

    Parameters
    ----------
    nucleus: str
        can be 1h, 2h, 13c, ...
    unit: str
        Either MHz/T (MHz_T, "gamma_bar") or 10e6 rad/T/s (rad_Ts)

    Returns
    -------
    The gyromagnetic ratio in standard unit

    """
    # valid nuclei:
    nuclei = [
        "1H",
        "1H (in H2O)",
        "2H",
        "3H",
        "3He",
        "7Li",
        "13C",
        "14N",
        "15N",
        "17O",
        "19F",
        "23Na",
        "27Al",
        "29Si",
        "31P",
        "57Fe",
        "63Cu",
        "67Zn",
        "129Xe",
    ]

    nucleus = nucleus.lower()

    # Checking for valid nucleus and unit
    valid_nuclei = [nuc.lower() for nuc in nuclei]
    valid_units = ["rad_Ts", "MHz_T"]

    # Dictionary with gmr values (taken from wikipedia)
    # https://en.wikipedia.org/wiki/Gyromagnetic_ratio
    gmr_values = {
        "1h_rad_Ts": 267.52218744 * 1e6,
        "1h_MHz_T": 42.577478518,
        "1h_in_h2o_rad_Ts": 267.5153151 * 1e6,
        "1h_in_h2o_MHz_T": 42.57638474,
        "2h_rad_Ts": 41.065 * 1e6,
        "2h_MHz_T": 6.536,
        "3h_rad_Ts": 285.3508 * 1e6,
        "3h_MHz_T": 45.415,
        "3he_rad_Ts": -203.7894569 * 1e6,
        "3he_MHz_T": -32.43409942,
        "7li_rad_Ts": 103.962 * 1e6,
        "7li_MHz_T": 16.546,
        "13c_rad_Ts": 67.2828 * 1e6,
        "13c_MHz_T": 10.7084,
        "14n_rad_Ts": 19.331 * 1e6,
        "14n_MHz_T": 3.077,
        "15n_rad_Ts": -27.116 * 1e6,
        "15n_MHz_T": -4.316,
        "17o_rad_Ts": -36.264 * 1e6,
        "17o_MHz_T": -5.772,
        "19f_rad_Ts": 251.815 * 1e6,
        "19f_MHz_T": 40.078,
        "23na_rad_Ts": 70.761 * 1e6,
        "23na_MHz_T": 11.262,
        "27al_rad_Ts": 69.763 * 1e6,
        "27al_MHz_T": 11.103,
        "29si_rad_Ts": -53.19 * 1e6,
        "29si_MHz_T": -8.465,
        "31p_rad_Ts": 108.291 * 1e6,
        "31p_MHz_T": 17.235,
        "57fe_rad_Ts": 8.681 * 1e6,
        "57fe_MHz_T": 1.382,
        "63cu_rad_Ts": 71.118 * 1e6,
        "63cu_MHz_T": 11.319,
        "67zn_rad_Ts": 16.767 * 1e6,
        "67zn_MHz_T": 2.669,
        "129xe_rad_Ts": -73.997 * 1e6,
        "129xe_MHz_T": -11.777,
    }

    # Constructing the key for the dictionary lookup
    key = f"{nucleus.lower()}_{unit}"

    # check if passed input is valid:
    if nucleus.lower() not in valid_nuclei:
        return f"Unknown nucleus: {nucleus}. Please provide a valid nucleus."
    elif unit not in valid_units:
        return f"Unknown unit: {unit}. Please provide a valid unit (either 'rad_Ts' or 'MHz_T')."
    else:
        pass

    # print which unit is passed:
    if db:
        if unit == "MHz_T":
            print("in [MHz/T]")
        elif unit == "rad_Ts":
            print("in [10e6 rad/T/s]")
        else:
            print("unknown unit")

    # Returning the value if the key exists, otherwise return None
    return gmr_values.get(key, None)


_ELEMENT_FIRST_NUCLEUS = re.compile(r"^([A-Za-z]{1,2})(\d{1,3})$")


def b0_tesla_from_reference_frequency(
    reference_frequency_mhz: float | None, nucleus: str | None
) -> float | None:
    """
    Return the static field that makes the reference frequency the Larmor frequency.

    ``B0 = reference_frequency_mhz / gamma_bar(nucleus)``, with ``gamma_bar``
    from :func:`get_gmr` in MHz/T. The reference frequency is the frequency
    of 0 ppm of the observed nucleus (Bruker ``PVM_FrqRef``). A frequency
    offset of the receiver comes from demodulation, not from the field, so
    it does not enter. For the 13C example of 75.4901665 MHz this gives
    75.4901665 / 10.7084 = 7.0496 T.

    Parameters
    ----------
    reference_frequency_mhz : float or None
        Frequency of 0 ppm of the observed nucleus in MHz. None means the
        scan does not report it.
    nucleus : str or None
        Observed nucleus with the mass number first, as in ``"13C"``, or
        with the element first, as in Agilent's ``"C13"``; case does not
        matter. None, ``""`` and ``"unknown"`` mean the scan does not report
        it.

    Returns
    -------
    float or None
        Static field in T, or None if the reference frequency or the nucleus
        is not reported.

    Raises
    ------
    ValueError
        If ``nucleus`` is not one of the nuclei of :func:`get_gmr`, or if
        ``reference_frequency_mhz`` is not a positive finite number.

    Examples
    --------
    >>> round(b0_tesla_from_reference_frequency(75.4901665, "13C"), 4)
    7.0496
    >>> b0_tesla_from_reference_frequency(None, "13C") is None
    True
    """
    if reference_frequency_mhz is None or nucleus in (None, "", "unknown"):
        return None
    reference_frequency_mhz = float(reference_frequency_mhz)
    if not np.isfinite(reference_frequency_mhz) or reference_frequency_mhz <= 0:
        raise ValueError(
            "b0_tesla_from_reference_frequency: reference_frequency_mhz must be "
            f"a positive finite frequency in MHz, got {reference_frequency_mhz!r}."
        )
    name = str(nucleus).strip()
    element_first = _ELEMENT_FIRST_NUCLEUS.match(name)
    if element_first is not None:
        name = element_first.group(2) + element_first.group(1)
    gmr_mhz_per_t = get_gmr(nucleus=name, unit="MHz_T")
    if not isinstance(gmr_mhz_per_t, float):
        raise ValueError(
            f"b0_tesla_from_reference_frequency: nucleus {nucleus!r} has no "
            "gyromagnetic ratio in hypermri.utils.physics.get_gmr; use a "
            "nucleus such as '1H' or '13C'."
        )
    return reference_frequency_mhz / gmr_mhz_per_t


def rfpower_from_refpower(refpower_W=1, fa_deg=90, trf_ms=1, sint=1):
    """
    Calculate the RF power necessary for a pulse.

    Parameters
    ----------
    refpower_W : float, optional
        Reference power for a 90-degree flip angle, 1ms pulse duration, and block pulse (sint=1). Default is 1.
    fa_deg : float, optional
        Flip angle in degrees. Default is 90.
    trf_ms : float, optional
        Pulse duration in milliseconds. Default is 1.
    sint : float, optional
        Integration factor (0-1, with 1 representing a block pulse). Default is 1.

    Returns
    -------
    float
        The RF power in watts.

    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"
    return refpower_W * (fa_deg / (90 * trf_ms * sint)) ** 2


def celsius_to_kelvin(celsius=0):
    """
    Convert a temperature from Celsius to Kelvin.

    Parameters
    ----------
    celsius : float, optional
        Temperature in Celsius. Default is 0.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    assert np.all(celsius >= -273.15), "temperature must be greater -273.15C"
    kelvin = celsius + 273.15
    return kelvin


def kelvin_to_celsius(kelvin=273.15):
    """
    Convert a temperature from Kelvin to Celsius.

    Parameters
    ----------
    kelvin : float, optional
        Temperature in Kelvin. Default is 273.15.

    Returns
    -------
    float
        Temperature in Celsius.

    """
    # assert np.all(kelvin >= 0), "temperature must be greater/equal 0K"
    celsius = kelvin - 273.15
    return celsius


def get_zeeman_energy_split(nucleus="13c", fieldstrength_T=7):
    """
    Calculate the energy difference between E_alpha and E_beta of a spin of a nucleus at a given field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.

    Returns
    -------
    float
        The energy difference in Joules.

    """
    assert fieldstrength_T > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    # Larmor frequency in Hz
    larmor_freq_hz = get_larmor_freq(
        nucleus=nucleus, fieldstrength_T=fieldstrength_T, unit="Hz"
    )
    return planck_constant * larmor_freq_hz


def get_larmor_freq(nucleus="13c", fieldstrength_T=7, unit="Hz"):
    """
    Calculate the Larmor frequency for a given nucleus at a given field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.
    unit : str, optional
        Unit for the frequency, either "Hz" or "rad". Default is "Hz".

    Returns
    -------
    float
        The Larmor frequency in the specified unit.

    """
    if unit == "Hz":
        gmr = get_gmr(nucleus=nucleus, unit="MHz_T") * 1e6
    elif unit == "rad":
        gmr = get_gmr(nucleus=nucleus, unit="rad_Ts")
    else:
        raise Exception(f"Unknown unit = {unit}, has to be either Hz or rad!")

    return gmr * fieldstrength_T


def get_thermal_polarisation_ratio(
    nucleus: str = "13c",
    temperature: float | np.ndarray = 230,
    fieldstrength_T: float = 7,
    temperature_unit: str = "K",
) -> float | np.ndarray:
    """Return the thermal upper-to-lower Zeeman population ratio.

    Parameters
    ----------
    nucleus : str, optional
        Nuclear identifier understood by get_gmr, for example '13c' or '1h'.
        The two-level interpretation assumes a spin-half nucleus.
    temperature : float or numpy.ndarray, optional
        Positive absolute temperature, expressed in temperature_unit. Arrays
        retain their shape and are not modified. Default is 230 K.
    fieldstrength_T : float, optional
        Positive magnetic field strength in Tesla. Default is 7 T.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float or numpy.ndarray
        Dimensionless upper-to-lower population ratio exp(-Delta E / (k_B T)),
        with the temperature shape and NumPy floating-point calculation
        precision. This ratio is not the normalized spin polarization.

    Raises
    ------
    AssertionError
        If fieldstrength_T is nonpositive or Celsius is below absolute zero.
    ValueError
        If temperature_unit is not K or C.
    TypeError
        If a nucleus has no gyromagnetic-ratio lookup or inputs are nonnumeric.

    Notes
    -----
    Uses the existing positive-frequency Zeeman splitting and Boltzmann
    constants. Temperatures must be positive in Kelvin; invalid zero/negative
    Kelvin inputs retain NumPy's legacy arithmetic behavior. The operation is
    eager, deterministic and uncached, and does not modify inputs or read data.

    Examples
    --------
    >>> ratio = get_thermal_polarisation_ratio('13c', 230, 7)
    >>> 0 < ratio < 1
    True

    """
    assert fieldstrength_T > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    boltzmann_constant = 1.380649e-23  # Boltzmann constant in J/K
    planck_constant / (2 * np.pi)  # Reduced Planck constant

    if temperature_unit == "K":
        pass
    elif temperature_unit == "C":
        temperature = celsius_to_kelvin(celsius=temperature)
    else:
        raise ValueError("temperature unit must be either K or C!")

    # calculate energy of zeeman splitting at fieldstrength:
    dE = get_zeeman_energy_split(nucleus=nucleus, fieldstrength_T=fieldstrength_T)

    population_ratio = np.exp(-dE / (boltzmann_constant * temperature))

    return population_ratio


def get_thermal_polarisation_ppm(
    nucleus: str = "13c",
    temperature: float | np.ndarray = 230,
    fieldstrength_T: float = 7,
    temperature_unit: str = "K",
) -> float | np.ndarray:
    """Return thermal spin-half polarization normalized to all spins, in ppm.

    Parameters
    ----------
    nucleus : str, optional
        Nuclear identifier understood by get_gmr, for example '13c' or '1h'.
        The polarization formula assumes a spin-half, two-level ensemble.
    temperature : float or numpy.ndarray, optional
        Positive absolute temperature expressed in temperature_unit. Arrays
        retain their shape and are not modified. Default is 230 K.
    fieldstrength_T : float, optional
        Positive magnetic field strength in Tesla. Default is 7 T.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float or numpy.ndarray
        1e6 * (N_lower - N_upper) / (N_lower + N_upper), using the temperature
        shape and NumPy floating-point calculation precision. For positive
        splitting and temperature the result is between zero and 1e6 ppm.

    Raises
    ------
    AssertionError, ValueError, TypeError
        As raised by get_thermal_polarisation_ratio for invalid field, units,
        nucleus lookup or nonnumeric inputs.

    Notes
    -----
    With population ratio r, polarization is (1-r)/(1+r) [1]_. The denominator was
    added in the owner-approved scientific correction of 10 October 2026;
    earlier versions returned 1e6*(1-r), approximately twice the normalized
    polarization at ordinary MRI fields and temperatures. This is a theoretical
    equilibrium reference, not a measured hyperpolarized-signal calibration.
    The operation is eager, deterministic and uncached, without input mutation.

    References
    ----------
    .. [1] Tateishi et al., PNAS 111 (2014), 7527-7530, Eq. 1,
       doi:10.1073/pnas.1315778111.

    Examples
    --------
    >>> round(float(get_thermal_polarisation_ppm('13c', 230, 7)), 3)
    7.821

    """
    population_ratio = get_thermal_polarisation_ratio(
        nucleus=nucleus,
        temperature=temperature,
        fieldstrength_T=fieldstrength_T,
        temperature_unit=temperature_unit,
    )

    polarisation_ppm = (1 - population_ratio) / (1 + population_ratio) * 1e6

    return polarisation_ppm


def hyperpolarized_Mz_flipangle(
    M0_au=1,
    n_excitations=100,
    flipangle_deg=10,
):
    """Calculate hyperpolarized magnetization depletion by RF pulses alone.

    Parameters
    ----------
    M0_au : float, optional
        Initial longitudinal magnetization in arbitrary signal units.
    n_excitations : int, optional
        Nonnegative number of RF pulses in excitation order.
    flipangle_deg : float or numpy.ndarray, optional
        Constant angle in degrees or one-dimensional angles of length n_excitations.
        RF-only calculations require exact array length and do not interpolate.

    Returns
    -------
    Mz : numpy.ndarray
        Float64 longitudinal magnetization immediately before each pulse, shape
        (n_excitations,), in the initial signal units.
    Mxy : numpy.ndarray
        Float64 transverse signal immediately after each pulse, same shape and units.

    Raises
    ------
    ValueError
        If angle-array length differs from n_excitations, angles are not a vector
        or the pulse count is negative.
    TypeError
        If pulse count or angles cannot be interpreted numerically.

    Notes
    -----
    The shared pulse-train model uses cosine RF depletion without relaxation,
    equilibrium recovery, noise or spoiling. Evaluation is eager, deterministic
    and uncached; inputs are preserved and no figure or widget is created.

    Examples
    --------
    >>> mz, mxy = hyperpolarized_Mz_flipangle(2., 2, 0.)
    >>> mz.tolist()
    [2.0, 2.0]
    """
    # Handle the flip angle input
    if np.isscalar(flipangle_deg):
        flip_angles_rad = np.deg2rad(np.full(n_excitations, flipangle_deg))
    else:
        if len(flipangle_deg) != n_excitations:
            raise ValueError("Length of flipangle_deg array must equal n_excitations.")
        flip_angles_rad = np.deg2rad(np.array(flipangle_deg))

    from hypermri.analysis.polarization import simulate_hyperpolarized_pulse_train

    return simulate_hyperpolarized_pulse_train(M0_au, flip_angles_rad)


def hyperpolarized_Mz_T1(M0_au, t_s, T1_s):
    """
    Calculate the hyperpolarized magnetization decay due to T1 relaxation alone.

    Parameters
    ----------
    M0_au : float
        Initial magnetization in arbitrary unit.
    t_s : array_like
        Time points in seconds.
    T1_s : float
        T1 relaxation time in seconds.

    Returns
    -------
    Mz : np.ndarray
        Calculated longitudinal magnetization (Mz) values.
    Mxy : np.ndarray
        Transverse magnetization (Mxy), which is zero for T1 decay alone.

    """
    Mz = M0_au * np.exp(-np.array(t_s) / T1_s)
    # T1 relaxation affects Mz only; it does not generate transverse
    # magnetization.
    Mxy = np.zeros_like(Mz)
    return Mz, Mxy


def hyperpolarized_Mz_flipangle_T1(
    M0_au: float = 1e5,
    n_excitations: int = 100,
    TR_s: float = 0.20,
    T1_s: float = 30.0,
    flipangle_deg: float = None,
    plot: bool = True,
    interactive: bool = False,
    scan=None,
    plot_params: dict = None,
):
    """Calculate and optionally display RF depletion with T1 relaxation.

    Parameters
    ----------
    M0_au : float, optional
        Initial longitudinal magnetization in arbitrary signal units.
    n_excitations : int, optional
        Nonnegative pulse count; a supplied scan overrides this value.
    TR_s : float, optional
        Constant repetition time in seconds; overridden by supplied scan.
    T1_s : float, optional
        Positive longitudinal relaxation time in seconds; equilibrium recovery is
        excluded from this hyperpolarized model.
    flipangle_deg : float or numpy.ndarray, optional
        Constant angle or one-dimensional angle train in degrees. None uses the
        supplied source's flip angle, otherwise 12 degrees. Unequal arrays retain
        legacy linear resampling over linspace(0, n_excitations, len(angles)).
    plot : bool, optional
        Create a magnitude/time figure when True.
    interactive : bool, optional
        Use notebook controls when plot=True. Changing controls redraws the plot;
        returned arrays describe the initial settings.
    scan : object, optional
        Source supplying repetition_time_ms, matrix and n_repetitions. Pulse count
        becomes prod(matrix) times n_repetitions. Its flipangle_deg is used only
        when no explicit angle is supplied. No acquired array is read or changed.
    plot_params : dict, optional
        Display options; figsize is read in inches. Other legacy keys are ignored.

    Returns
    -------
    Mz : numpy.ndarray
        Float64 longitudinal signal immediately before each pulse, shape
        (n_excitations,), in initial magnetization units.
    Mxy : numpy.ndarray
        Float64 transverse signal immediately after each pulse, same shape and units.
    t_s : numpy.ndarray
        Excitation times from zero with TR_s spacing, same shape, in seconds.

    Raises
    ------
    ValueError, TypeError
        If pulse counts, angle arrays or interpolation settings are invalid.
    ZeroDivisionError
        If a zero scalar T1 is supplied.
    AttributeError
        If requested source metadata are absent.
    ImportError
        If interactive plotting is requested without notebook dependencies.

    Notes
    -----
    Numerical calculation is eager, deterministic and uncached, delegating RF/T1
    products to the shared pulse-train helper. Plotting initializes Matplotlib,
    changes its interactive state and may construct notebook controls. Inputs
    remain unchanged. This is a forward signal model, not an absolute experimental
    polarization determination.

    Examples
    --------
    >>> mz, mxy, time_s = hyperpolarized_Mz_flipangle_T1(
    ...     n_excitations=3, flipangle_deg=10., plot=False
    ... )
    >>> time_s.tolist()
    [0.0, 0.2, 0.4]
    """
    import matplotlib.pyplot as plt

    # Parameter handling from scan or defaults
    if plot_params is None:
        plot_params = {}
    if scan:
        TR_s = scan.repetition_time_ms / 1000.0
        n_excitations = np.prod(scan.matrix) * scan.n_repetitions
        if flipangle_deg is None:
            flipangle_deg = scan.flipangle_deg

    if flipangle_deg is None:
        flipangle_deg = 12.0  # Default value if still None

    # This is the core calculation function, now separate for clarity
    def _calculate_signal(m0, t1, tr, fa_deg, n_exc):
        # Handle scalar or array flip angle input
        if np.isscalar(fa_deg):
            flip_angles_rad = np.deg2rad(np.full(n_exc, fa_deg))
        else:
            if len(fa_deg) != n_exc:  # Ensure array has correct length
                fa_deg_resampled = np.interp(
                    np.arange(n_exc), np.linspace(0, n_exc, len(fa_deg)), fa_deg
                )
            else:
                fa_deg_resampled = fa_deg
            flip_angles_rad = np.deg2rad(np.array(fa_deg_resampled))

        # Initialize arrays
        t = np.arange(n_exc) * tr
        from hypermri.analysis.polarization import (
            simulate_hyperpolarized_pulse_train,
        )

        mz, mxy = simulate_hyperpolarized_pulse_train(
            m0, flip_angles_rad, np.exp(-tr / t1)
        )

        return mz, mxy, t

    # Perform initial calculation
    Mz, Mxy, t_s = _calculate_signal(M0_au, T1_s, TR_s, flipangle_deg, n_excitations)

    if plot:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
        ]
        figsize = plot_params.get("figsize", (8, 5))

        if interactive:
            from IPython.display import display
            from ipywidgets import (
                Checkbox,
                FloatSlider,
                HBox,
                IntSlider,
                Layout,
                Output,
                VBox,
            )

            # Widgets
            w_m0 = FloatSlider(
                min=1e4, max=1e6, step=1e4, value=M0_au, description="M0"
            )
            w_exc = IntSlider(
                min=10,
                max=500,
                step=10,
                value=n_excitations,
                description="# Excitations",
            )
            w_tr = FloatSlider(
                min=0.01, max=2, step=0.001, value=TR_s, description="TR (s)"
            )
            w_t1 = FloatSlider(min=1, max=100, step=1, value=T1_s, description="T1 (s)")
            w_fa = FloatSlider(
                min=0,
                max=90,
                step=1,
                value=flipangle_deg if np.isscalar(flipangle_deg) else 12,
                description="Flip Angle (°)",
            )
            w_comp = Checkbox(value=True, description="Show components")

            plot_out = Output()

            plt.ioff()
            fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)
            ax2 = ax1.twiny()
            plt.ion()

            with plot_out:
                display(fig.canvas)

            def update_plot(change=None):
                M0_au_w = w_m0.value
                n_excitations_w = w_exc.value
                TR_s_w = w_tr.value
                T1_s_w = w_t1.value
                flipangle_deg_w = w_fa.value
                show_components = w_comp.value

                mz_plot, mxy_plot, t_plot = _calculate_signal(
                    M0_au_w, T1_s_w, TR_s_w, flipangle_deg_w, n_excitations_w
                )

                ax1.clear()
                ax2.clear()

                # Plot Mz and Mxy
                ax1.plot(t_plot, mz_plot, label="$M_z$ (pre-pulse)", color=colors[0])
                ax1.plot(
                    t_plot,
                    mxy_plot,
                    label="$M_{xy}$ (signal)",
                    color=colors[-1],
                    linestyle="--",
                    marker="o",
                    markersize=4,
                )

                # Optional: show the individual decay components
                if show_components:
                    mz_t1_only, _ = hyperpolarized_Mz_T1(M0_au_w, t_plot, T1_s_w)
                    ax1.plot(
                        t_plot,
                        mz_t1_only,
                        label=f"T1 Decay Only (T1={T1_s_w:.1f}s)",
                        color=colors[1],
                        linestyle=":",
                        alpha=0.8,
                    )
                    mz_fa_only, _ = hyperpolarized_Mz_flipangle(
                        M0_au_w, n_excitations_w, flipangle_deg_w
                    )
                    ax1.plot(
                        t_plot,
                        mz_fa_only,
                        label=f"Flip Angle Decay Only ({flipangle_deg_w:.1f}°)",
                        color=colors[2],
                        linestyle=":",
                        alpha=0.8,
                    )

                eff_t1_val_at_m0e = M0_au_w / np.e
                eff_T1_s_idx = np.argmin(np.abs(mz_plot - eff_t1_val_at_m0e))
                eff_T1_s = t_plot[eff_T1_s_idx]

                ax1.vlines(
                    eff_T1_s,
                    0,
                    eff_t1_val_at_m0e,
                    color="r",
                    linestyle="--",
                    label=rf"$M_0/e$ (Effective T1 $\approx$ {eff_T1_s:.1f}s)",
                )

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Magnetization (a.u.)")
                ax1.set_title("Hyperpolarized Magnetization Simulation")
                ax1.legend()
                ax1.set_ylim(0, M0_au_w * 1.1)
                ax1.set_xlim(0, t_plot[-1] if len(t_plot) > 1 else 1)

                ax2.set_xlim(ax1.get_xlim())
                tick_indices = np.linspace(0, n_excitations_w - 1, 11, dtype=int)
                ax2.set_xticks(t_plot[tick_indices])
                ax2.set_xticklabels(tick_indices)
                ax2.set_xlabel("Excitation Number")

                fig.canvas.draw_idle()

            # Observe changes
            for w in [w_m0, w_exc, w_tr, w_t1, w_fa, w_comp]:
                w.observe(update_plot, names="value")

            # Initial Plot
            update_plot()

            # Layout
            sidebar = VBox(
                [w_m0, w_exc, w_tr, w_t1, w_fa, w_comp],
                layout=Layout(width="300px", margin="0 20px 0 0"),
            )
            ui = HBox([sidebar, plot_out])
            display(ui)

        else:
            # Non-interactive plot
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.ion()
            ax2 = ax1.twiny()

            flipangle_deg if np.isscalar(flipangle_deg) else "VFA"

            # Use the inner function for logic, but need to adapt args slightly since update_plot expects widgets
            # Actually easier to just copy the plotting logic briefly or adapt the call
            mz_plot, mxy_plot, t_plot = Mz, Mxy, t_s  # Already calculated above

            ax1.plot(t_plot, mz_plot, label="$M_z$ (pre-pulse)", color=colors[0])
            ax1.plot(
                t_plot,
                mxy_plot,
                label="$M_{xy}$ (signal)",
                color=colors[-1],
                linestyle="--",
                marker="o",
                markersize=4,
            )

            # Components (always on for non-interactive default?) - let's keep it simple as before
            # The original code passed show_components=True for non-interactive
            mz_t1_only, _ = hyperpolarized_Mz_T1(M0_au, t_plot, T1_s)
            ax1.plot(
                t_plot,
                mz_t1_only,
                label=f"T1 Decay Only (T1={T1_s:.1f}s)",
                color=colors[1],
                linestyle=":",
                alpha=0.8,
            )

            mz_fa_only, _ = hyperpolarized_Mz_flipangle(
                M0_au,
                n_excitations,
                flipangle_deg if np.isscalar(flipangle_deg) else flipangle_deg[0],
            )  # Simplified for VFA array handling in this call
            ax1.plot(
                t_plot,
                mz_fa_only,
                label="Flip Angle Decay Only",
                color=colors[2],
                linestyle=":",
                alpha=0.8,
            )

            eff_t1_val_at_m0e = M0_au / np.e
            eff_T1_s_idx = np.argmin(np.abs(mz_plot - eff_t1_val_at_m0e))
            eff_T1_s = t_plot[eff_T1_s_idx]

            ax1.vlines(
                eff_T1_s,
                0,
                eff_t1_val_at_m0e,
                color="r",
                linestyle="--",
                label=rf"$M_0/e$ (Effective T1 $\approx$ {eff_T1_s:.1f}s)",
            )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Magnetization (a.u.)")
            ax1.set_title("Hyperpolarized Magnetization Simulation")
            ax1.legend()
            ax1.set_ylim(0, M0_au * 1.1)
            ax1.set_xlim(0, t_plot[-1] if len(t_plot) > 1 else 1)

            ax2.set_xlim(ax1.get_xlim())
            tick_indices = np.linspace(0, n_excitations - 1, 11, dtype=int)
            ax2.set_xticks(t_plot[tick_indices])
            ax2.set_xticklabels(tick_indices)
            ax2.set_xlabel("Excitation Number")

            plt.show()

    if interactive:
        return
    else:
        return Mz, Mxy, t_s
