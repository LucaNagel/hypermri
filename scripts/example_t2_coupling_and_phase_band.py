#!/usr/bin/env python3
"""Show what ``t2_coupling`` and ``phase_band_tolerance_deg`` do on synthetic FIDs.

Examples
--------
Run from a checkout with HyperMRI installed (preferably editable)::

    python scripts/example_t2_coupling_and_phase_band.py
    python scripts/example_t2_coupling_and_phase_band.py --repetitions 100
    python scripts/example_t2_coupling_and_phase_band.py --figure tmp/phase_band.png

Notes
-----
Every example fits many noise realizations of the same noiseless signal at
once, so the printed mean and standard deviation (SD) show both the bias and
the scatter of a fit setting. The signals are sums of Lorentzian lines made by
:func:`hypermri.analysis.spectroscopy.generate_fid`: 256 complex samples at a
dwell time of 0.25 ms (4 kHz bandwidth), complex white noise with the stated
SD per real and imaginary quadrature, and seeded random numbers, so every run
prints the same numbers. All lines share one phase ``PHI0_RAD`` at a signal
origin ``TAU_S`` = 1.5 ms before the first sample, so line ``k`` starts at
``PHI0_RAD + 2 pi f_k TAU_S``. This is the phase law the phase band assumes.
Frequencies are in Hz relative to the carrier, T2* values in seconds and
amplitudes in arbitrary signal units.

The values are illustrative, not biological references. Import has no data
generation, fitting, plotting or file-system side effects; ``--figure`` writes
one PNG file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy import generate_fid

DWELL_TIME_S = 2.5e-4
NPOINTS = 256
TIME_AXIS_S = np.arange(NPOINTS) * DWELL_TIME_S
TAU_S = 1.5e-3
PHI0_RAD = 0.4


def synthetic_signals(
    freqs_hz, t2s_s, amplitudes, *, noise_sd, n_repetitions, seed=0
) -> xr.DataArray:
    """Return noisy FIDs of Lorentzian lines that follow one phase law.

    Parameters
    ----------
    freqs_hz, t2s_s, amplitudes : sequence of float
        One frequency in Hz, T2* in seconds and real, positive amplitude per
        line.
    noise_sd : float
        Noise SD per real and imaginary quadrature, in signal units.
    n_repetitions : int
        Number of independent noise realizations.
    seed : int, optional
        Seed of the noise generator.

    Returns
    -------
    xarray.DataArray
        Complex FIDs with dims ``("repetition", "acquisition_idx")`` and the
        attribute ``sampling_time_axis_s`` that the fit needs.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    noiseless = generate_fid(
        time_axis_s=TIME_AXIS_S,
        amplitude=amplitudes,
        t2_s=t2s_s,
        freq_hz=freqs_hz,
        phase_rad=PHI0_RAD + 2 * np.pi * freqs_hz * TAU_S,
    )
    rng = np.random.default_rng(seed)
    noise = noise_sd * (
        rng.standard_normal((n_repetitions, NPOINTS))
        + 1j * rng.standard_normal((n_repetitions, NPOINTS))
    )
    signals = xr.DataArray(
        noiseless[None, :] + noise,
        dims=["repetition", "acquisition_idx"],
        coords={"repetition": np.arange(n_repetitions)},
    )
    signals.attrs["sampling_time_axis_s"] = TIME_AXIS_S
    return signals


def fit(signals, names, freqs_hz, **options) -> xr.Dataset:
    """Fit ``signals`` with the FID pseudo-inverse fit and extra ``options``.

    The start T2* is 0.03 s for every line, the T2* bounds are 0.005-0.15 s and
    each frequency may move by +/- 3 Hz. ``options`` are added to, or replace,
    these settings, for example ``t2_coupling`` or ``initial_t2s_s``.
    """
    fit_params = def_fit_params(
        data=signals,
        fit_params={
            "metabs": names,
            "metabs_freqs_hz": freqs_hz,
            "metabs_freq_ranges": 6.0,
            "initial_t2s_s": 0.03,
            "min_t2_s": 0.005,
            "max_t2_s": 0.15,
            "show_tqdm": False,
            **options,
        },
    )
    return fit_data_pseudo_inv(signals, fit_params)


def phase_deviation_deg(result: xr.Dataset, reference: str) -> np.ndarray:
    """Return each line's phase minus the phase law, relative to ``reference``.

    The phase law predicts ``phi0 + 2 pi f_k tau``. Removing ``2 pi f_k tau``
    with the fitted frequency leaves ``phi0 + delta_k``; subtracting the same
    value of the ``reference`` line removes the unknown ``phi0``. The result
    has shape ``(repetition, metabolite)`` in degrees within (-180, 180].
    """
    amplitudes = result.fit_amps.transpose("repetition", "metabolite").values
    freqs_hz = result.fit_freqs.transpose("repetition", "metabolite").values
    on_law = amplitudes * np.exp(-2j * np.pi * freqs_hz * TAU_S)
    index = list(result.metabolite.values).index(reference)
    return np.degrees(np.angle(on_law * np.conj(on_law[:, index : index + 1])))


def print_t2_table(title, results, truth) -> None:
    """Print mean and SD of the fitted T2* of 'b' and 'c' per fit setting."""
    print(f"\n{title}")
    print(f"  true T2* [ms]: b {1e3 * truth[0]:.1f}, c {1e3 * truth[1]:.1f}")
    print(f"  {'setting':<34}{'b mean (SD)':>16}{'c mean (SD)':>16}{'c/b':>7}")
    for label, result in results.items():
        t2s_ms = 1e3 * result.fit_t2s.transpose("repetition", "metabolite").values
        cells = [f"{t.mean():.1f} ({t.std():.1f})" for t in t2s_ms.T]
        ratio = (t2s_ms[:, 1] / t2s_ms[:, 0]).mean()
        print(f"  {label:<34}{cells[0]:>16}{cells[1]:>16}{ratio:>7.2f}")


def t2_coupling_examples(n_repetitions: int) -> dict[str, dict[str, xr.Dataset]]:
    """Fit a 0/12 Hz doublet with independent, fixed and banded T2* coupling.

    ``t2_coupling={"c": ("b", r)}`` means ``T2*(c) = r * T2*(b)``: only 'b'
    is searched and 'c' follows. ``{"c": ("b", (low, high))}`` means 'c' is
    searched itself but ``low <= T2*(c) / T2*(b) <= high`` always holds. The
    two lines overlap (FWHM 10.6 Hz at T2* 30 ms), so free T2* values scatter.

    Returns
    -------
    dict
        ``{"equal": ..., "half": ...}`` with one fit result per setting label.
    """
    names, freqs_hz, amplitudes = ["b", "c"], [0.0, 12.0], [10.0, 6.0]
    common = {"noise_sd": 0.6, "n_repetitions": n_repetitions, "seed": 1}

    # 1a: both lines decay equally fast, so 1:1 coupling is correct.
    signals = synthetic_signals(freqs_hz, [0.03, 0.03], amplitudes, **common)
    equal = {
        "independent (no coupling)": fit(signals, names, freqs_hz),
        "fixed 1:1  {'c': 'b'}": fit(signals, names, freqs_hz, t2_coupling={"c": "b"}),
        "band  {'c': ('b', (0.75, 1.25))}": fit(
            signals, names, freqs_hz, t2_coupling={"c": ("b", (0.75, 1.25))}
        ),
    }
    print_t2_table("1a  T2* coupling, true ratio c/b = 1.0", equal, [0.03, 0.03])

    # 1b: 'c' decays twice as fast. A wrong fixed ratio biases both lines; a
    # band that excludes the truth ends at its edge.
    signals = synthetic_signals(freqs_hz, [0.03, 0.015], amplitudes, **common)
    half = {
        "independent (no coupling)": fit(signals, names, freqs_hz),
        "fixed 1:1 (wrong)": fit(signals, names, freqs_hz, t2_coupling={"c": "b"}),
        "fixed 0.5  {'c': ('b', 0.5)}": fit(
            signals, names, freqs_hz, t2_coupling={"c": ("b", 0.5)}
        ),
        "band (0.75, 1.25) (excludes 0.5)": fit(
            signals, names, freqs_hz, t2_coupling={"c": ("b", (0.75, 1.25))}
        ),
        # A band needs a start ratio inside it: 0.015 / 0.03 = 0.5.
        "band (0.4, 0.6), start 30/15 ms": fit(
            signals,
            names,
            freqs_hz,
            initial_t2s_s=[0.03, 0.015],
            t2_coupling={"c": ("b", (0.4, 0.6))},
        ),
    }
    print_t2_table("1b  T2* coupling, true ratio c/b = 0.5", half, [0.03, 0.015])
    try:
        fit(signals, names, freqs_hz, t2_coupling={"c": ("b", (0.4, 0.6))})
    except ValueError as error:
        print(f"  start ratio 1.0 outside (0.4, 0.6) is rejected:\n    {error}")
    return {"equal": equal, "half": half}


def print_phase_table(title, results) -> None:
    """Print amplitudes and phase deviations of 'b' and 'c' per fit setting."""
    print(f"\n{title}")
    print("  true |a|: ref 8.0, b 10.0, c 4.0; true phase deviation 0 deg")
    print(
        f"  {'setting':<30}{'|a_ref|':>8}{'|a_b| (SD)':>13}{'|a_c| (SD)':>13}"
        f"{'max|dphi_b|':>12}{'max|dphi_c|':>12}"
    )
    for label, result in results.items():
        amplitudes = np.abs(result.fit_amps.transpose("repetition", "metabolite"))
        mean, sd = amplitudes.mean("repetition").values, amplitudes.std("repetition")
        deviation = np.abs(phase_deviation_deg(result, "ref")).max(axis=0)
        # Without the reference line there is no phase to compare with.
        cells = (
            [f"{deviation[1]:.1f}", f"{deviation[2]:.1f}"]
            if mean[0] > 1e-6
            else ["n/a", "n/a"]
        )
        print(
            f"  {label:<30}{mean[0]:>8.2f}"
            f"{f'{mean[1]:.2f} ({sd.values[1]:.2f})':>13}"
            f"{f'{mean[2]:.2f} ({sd.values[2]:.2f})':>13}"
            f"{cells[0]:>12}{cells[1]:>12}"
        )


def phase_band_examples(
    n_repetitions: int,
) -> tuple[xr.DataArray, dict[str, xr.Dataset]]:
    """Fit a strongly overlapping 0/5 Hz pair with and without the phase band.

    A separated reference line at -250 Hz makes the shared phase ``phi0``
    well determined. ``phase_band_tolerance_deg={"b": 5.0}`` forces
    ``a_b = A_b exp(1j (phi0 + 2 pi f_b tau + delta_b))`` with ``A_b >= 0``
    and ``|delta_b| <= 5 deg``; ``phase_band_tau_s`` is ``tau``. Free complex
    amplitudes of overlapping lines can instead take nearly opposite phases.

    Returns
    -------
    signals : xarray.DataArray
        The fitted noisy FIDs.
    results : dict
        One fit result per setting label.
    """
    names, freqs_hz = ["ref", "b", "c"], [-250.0, 0.0, 5.0]
    signals = synthetic_signals(
        freqs_hz,
        [0.03, 0.03, 0.03],
        [8.0, 10.0, 4.0],
        noise_sd=1.0,
        n_repetitions=n_repetitions,
        seed=2,
    )

    def band(tolerance_deg, tau_s=TAU_S):
        return fit(
            signals,
            names,
            freqs_hz,
            phase_band_tolerance_deg=dict.fromkeys(names, tolerance_deg),
            phase_band_tau_s=tau_s,
        )

    results = {
        "free complex amplitudes": fit(signals, names, freqs_hz),
        "band 90 deg (only A_k >= 0)": band(90.0),
        "band 30 deg": band(30.0),
        "band 5 deg": band(5.0),
        "band 0 deg (exactly on law)": band(0.0),
        "band 5 deg, wrong tau = 0": band(5.0, tau_s=0.0),
    }
    print_phase_table("2   Phase band, lines at -250 / 0 / 5 Hz", results)
    print(
        "  dphi = phase of a_k minus phi0 + 2 pi f_k tau, relative to 'ref'.\n"
        "  'ref' itself may deviate by up to the tolerance, so |dphi| can reach\n"
        "  twice the tolerance. A 90 deg band only forbids opposite phases, so it\n"
        "  matches the free fit unless that needs them.\n"
        "  With tau = 0 the law misses the phase 2 pi 250 Hz 1.5 ms = 135 deg of\n"
        "  'ref'; its amplitude collapses because A_ref >= 0 cannot reach it."
    )
    return signals, results


def plot_phase_band(
    signals: xr.DataArray, results: dict[str, xr.Dataset], path: Path
) -> None:
    """Save measured and fitted spectra, free versus the 5 deg phase band.

    The figure uses the noise realization in which the free fit puts line 'c'
    furthest from the phase law. Each fitted line is rebuilt with
    :func:`generate_fid` from its fitted complex amplitude, frequency and T2*.
    All FIDs are zero-filled to 4096 points for a smooth display, and every
    spectrum is phased with the true ``PHI0_RAD`` and ``TAU_S``, so a line on
    the phase law appears as a positive absorption peak in the real part.
    """
    import matplotlib.pyplot as plt

    n_display = 4096
    freqs_axis_hz = np.fft.fftshift(np.fft.fftfreq(n_display, DWELL_TIME_S))
    phase_correction = np.exp(-1j * (PHI0_RAD + 2 * np.pi * freqs_axis_hz * TAU_S))
    window = np.abs(freqs_axis_hz) < 40

    def phased_spectrum(fid):
        spectrum = np.fft.fftshift(np.fft.fft(fid, n=n_display)) * phase_correction
        return spectrum.real[window]

    free = results["free complex amplitudes"]
    repetition = int(np.abs(phase_deviation_deg(free, "ref")[:, 2]).argmax())
    measured = phased_spectrum(signals.isel(repetition=repetition).values)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, label in zip(
        axes, ["free complex amplitudes", "band 5 deg"], strict=True
    ):
        result = results[label].isel(repetition=repetition)
        axis.plot(freqs_axis_hz[window], measured, color="0.6", label="data")
        for name in ("b", "c"):
            line = result.sel(metabolite=name)
            amplitude = complex(line.fit_amps)
            fid = generate_fid(
                time_axis_s=TIME_AXIS_S,
                amplitude=abs(amplitude),
                t2_s=float(line.fit_t2s),
                freq_hz=float(line.fit_freqs),
                phase_rad=np.angle(amplitude),
            )
            axis.plot(freqs_axis_hz[window], phased_spectrum(fid), label=f"fit {name}")
        axis.axhline(0.0, color="0.85", linewidth=0.8)
        axis.set_title(f"{label} (realization {repetition})")
        axis.set_xlabel("frequency [Hz]")
        axis.legend()
    axes[0].set_ylabel("real part of phased spectrum [a.u.]")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"\nFigure written to {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="noise realizations per example (default: 20, about 2 minutes)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="optional PNG path for the free versus phase-band components",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run both examples and print their tables; optionally save the figure."""
    arguments = _parser().parse_args(argv)
    t2_coupling_examples(arguments.repetitions)
    signals, results = phase_band_examples(arguments.repetitions)
    if arguments.figure is not None:
        plot_phase_band(signals, results, arguments.figure)


if __name__ == "__main__":
    main()
