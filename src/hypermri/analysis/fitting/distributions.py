"""Probability-density fitting for acquired signal magnitude distributions.

This module owns the existing histogram-based Rayleigh fit. It preserves its
scientific defaults, unconstrained least-squares fit and optional static plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats


def rayleigh_pdf_fit(x, mu, sigma):
    """Evaluate a Rayleigh probability density function.

    Parameters
    ----------
    x : array_like
        Magnitude sample locations, in the same signal units as mu and sigma.
    mu : float
        Location parameter in signal units.
    sigma : float
        Positive scale parameter in signal units.

    Returns
    -------
    numpy.ndarray
        Rayleigh density in inverse signal units, with the broadcast input
        shape. SciPy determines precision and behavior for invalid parameters.
        Does not mutate inputs, read data, create figures or cache values.

    Notes
    -----
    This function is tailored for use with :func:`scipy.optimize.curve_fit`.
    """
    return stats.rayleigh.pdf(x, loc=mu, scale=sigma)


def fit_rayleigh(data=None, plot=True, **kwargs):
    """Fit a Rayleigh distribution to the magnitude of complex data.

    Parameters
    ----------
    data : array_like, optional
        Real or complex acquired samples of any shape, in consistent signal
        units. Fitting uses a flattened magnitude copy; inputs remain unchanged.
        Labelled arrays are treated as unlabelled samples. None raises.
    plot : bool, optional
        If ``True`` the sample magnitude and fitted PDF are plotted. Default is
        ``True``.
    **kwargs
        Retained legacy options read from this mapping; unknown keys retain
        their historical ignored behavior. This function does not forward
        arbitrary keywords to the solver.

        ``bins`` : int, optional
            Number of histogram bins. Default is 30.
        ``density`` : bool, optional
            If ``True`` the histogram is normalised. Default is ``True``.
        ``p0`` : tuple of float, optional
            Initial guesses for (mu, sigma) in signal units. Defaults to the
            sample magnitude mean and standard deviation.

    Returns
    -------
    popt : numpy.ndarray
        Two fitted parameters (mu, sigma) in input signal units. The existing
        histogram least-squares solver is unconstrained, with SciPy's returned
        floating precision; it does not impose new scientific bounds.
    pcov : numpy.ndarray
        Estimated 2 by 2 covariance matrix in squared signal units. SciPy may
        report unidentifiable parameters through infinite covariance/warnings.

    Raises
    ------
    ValueError
        If data is None, histogram settings are invalid or the solver
        rejects the sample values or parameter initialization.
    RuntimeError
        If the least-squares solver fails to converge.

    Notes
    -----
    Histograms are computed on the absolute value of ``data``. When
    plotting is enabled the function creates two subplots: one for the signal
    magnitude and one for the histogram with the fitted PDF. It does not show,
    save or close the figure; the caller owns its lifecycle. With plot=False
    no figure is constructed. Repeated calls with the same data/options are
    deterministic, consume no RNG and do not cache fitted values.

    Examples
    --------
    >>> from scipy.stats import rayleigh
    >>> probability = (np.arange(20000) + 0.5) / 20000
    >>> signal = rayleigh.ppf(probability, loc=0.35, scale=1.75)
    >>> params, covariance = fit_rayleigh(signal, plot=False, bins=40, p0=[0.3, 1.7])
    >>> np.allclose(params, [0.35, 1.75], atol=0.01)
    True
    """
    if data is None:
        raise ValueError("data has to be passed!")

    data = np.abs(data).flatten()  # Ensure input is flattened and magnitude is taken

    # Extract or set default keyword arguments
    bins = kwargs.get("bins", 30)
    plot_density = kwargs.get(
        "density", True
    )  # Renamed to avoid confusion with histogram output
    p0 = kwargs.get("p0", [np.mean(data), np.std(data)])

    # Compute histogram density and bins
    hist_density, bins = np.histogram(data, bins=bins, density=plot_density)
    centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    # Fit Rayleigh distribution
    popt, pcov = optimize.curve_fit(rayleigh_pdf_fit, centers, hist_density, p0=p0)

    if plot:
        fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
        ax[0].plot(data)
        ax[0].set_title("Signal Magnitude")
        ax[1].hist(data, bins=bins, density=plot_density)
        ax[1].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[1].axvline(np.mean(data), color="k", label="Mean")
        ax[1].axvline(popt[0], color="r", label="Fitted Mu")
        ax[1].axvline(popt[1], color="g", label="Fitted Sigma")
        ax[1].set_title("Histogram Magnitude")
        ax[1].legend()

    return popt, pcov
