"""Predict the phase of FID lines at the first sample from sequence parameters.

The phase band of the hypermri FID fit assumes
``angle(a_k) = phi0 + 2*pi*f_k*tau``. Given the excitation pulse, the slice
selection and ``tau`` (for example from
``hypermri.analysis.spectroscopy.phase_band_tau_s_from_scan``), this script
prints for every requested line

- the phase that this law predicts, ``360 * f * tau`` degrees;
- the phase from a Bloch simulation of the pulse, the slice-select and
  rephasing gradients and the free precession up to ``tau``;
- the difference of both.

All phases are relative to a line on the carrier (``f = 0``), because
``phi0`` contains the unknown receiver phase. Frequencies are offsets from the
RF carrier in Hz, with the sign of the hypermri FID model
``exp(+i 2 pi f t)``.

The simulation counts ``tau`` from the pulse isodelay (``rephasing percent``
of the pulse length before its end). Everything after the pulse, including
receiver and digital-filter delays, is treated as gradient-free precession,
except a rephasing lobe right after the pulse. The excited object is a
uniform slab twice as thick as the slice, centred on the slice. Phase
encoding and relaxation are not simulated. The functions come from
``simulate_fid_phase_timing.py`` in the same directory.

Examples
--------
A slice-selective CSI with a 0.84 ms SLR pulse (Bruker bandwidth 5000 Hz,
sharpness 3, 8 degrees), a 3 mm slice 1.445 mm off centre and
``phase_band_tau_s_from_scan`` = 1.59027 ms, on a grid over the spectral
width::

    python scripts/predict_fid_line_phase.py --pulse slr --duration-ms 0.84 \\
        --bandwidth-hz 5000 --slr-sharpness 3 --flip-deg 8 --slice-mm 3 \\
        --slice-offset-mm 1.44547 --tau-ms 1.59027 --span-hz 3000 --step-hz 250

Selected lines in ppm (carrier and Larmor frequency from the scan, for example
``PVM_FrqWorkPpm[0]`` and ``SFO1``)::

    python scripts/predict_fid_line_phase.py ... --tau-ms 1.59027 \\
        --ppm 171.0 183.0 --carrier-ppm 174.5 --larmor-mhz 75.47

Requires the ``blochsimulator`` package; hypermri itself is not needed.
"""

from __future__ import annotations

import argparse

import numpy as np
from simulate_fid_phase_timing import (
    Metabolite,
    PulseSettings,
    SequenceSettings,
    fit_phase_law,
    transverse_signal_at_acquisition,
    uniform_slab,
)


def wrap_deg(phase_deg: np.ndarray) -> np.ndarray:
    """Wrap phases in degrees to the interval ``(-180, 180]``."""
    return -((-np.asarray(phase_deg) + 180.0) % 360.0 - 180.0)


def predict_line_phase_deg(
    pulse: PulseSettings,
    slice_mm: float | None,
    slice_offset_mm: float,
    rephaser_fraction: float,
    rephaser_duration_s: float,
    tau_s: float,
    frequency_hz: np.ndarray,
) -> dict:
    """Simulate the line phases at ``tau`` after the pulse isodelay.

    Parameters
    ----------
    pulse : PulseSettings
        Excitation pulse, including the flip angle and the rephasing percent
        that places its isodelay.
    slice_mm : float or None
        Slice thickness in mm, or ``None`` for a non-selective pulse.
    slice_offset_mm : float
        Slice centre along the slice axis in mm; must be 0 without slice.
    rephaser_fraction : float
        Area of the rephasing lobe relative to the slice-select area (0.5
        refocuses a symmetric pulse).
    rephaser_duration_s : float
        Length of the rephasing lobe in s; it must fit between the pulse end
        and ``tau``.
    tau_s : float
        Time from the pulse isodelay to the first sample in s.
    frequency_hz : numpy.ndarray
        Line offsets from the carrier in Hz, shape ``(n,)``.

    Returns
    -------
    dict
        ``rule_deg``: ``360 f tau`` wrapped to ``(-180, 180]``;
        ``bloch_deg``: simulated phase relative to the carrier line, wrapped;
        ``difference_deg``: ``bloch_deg - rule_deg``, wrapped; all shape
        ``(n,)``. ``tau_fit_s``: ``tau`` of a magnitude-weighted fit of the
        phase law to the simulated lines (``nan`` for fewer than two distinct
        lines). ``dead_time_s``: simulated time from pulse end to the first
        sample. Deterministic.

    Raises
    ------
    ValueError
        If ``tau_s`` ends before the pulse or the rephasing lobe does not fit.
    """
    isodelay_to_end_s = pulse.duration_s * pulse.rephasing_percent / 100.0
    dead_time_s = tau_s - isodelay_to_end_s
    if dead_time_s <= 0:
        raise ValueError(
            f"predict_line_phase_deg: tau_s {tau_s * 1e3:.4f} ms ends before the "
            f"pulse ({isodelay_to_end_s * 1e3:.4f} ms after its isodelay); "
            "check tau and the rephasing percent."
        )
    dt_s = pulse.duration_s / pulse.n_points
    simulated_dead_time_s = round(dead_time_s / dt_s) * dt_s
    sequence = SequenceSettings(
        dead_time_s=simulated_dead_time_s,
        slice_mm=slice_mm,
        rephaser_fraction=rephaser_fraction,
        rephaser_duration_s=rephaser_duration_s,
        slice_offset_mm=slice_offset_mm,
    )
    if slice_mm is None:
        z_cm, weights = np.zeros(1), np.ones(1)
    else:
        z_cm, weights = uniform_slab(2 * slice_mm)
        z_cm = z_cm + slice_offset_mm / 10.0
    lines = [
        Metabolite(f"{f:+.1f} Hz", float(f), z_cm, weights)
        for f in np.concatenate([[0.0], frequency_hz])
    ]
    signal = transverse_signal_at_acquisition(pulse, sequence, lines)
    # The raster rounding of the dead time is pure free precession.
    all_hz = np.array([line.frequency_hz for line in lines])
    signal = signal * np.exp(
        2j * np.pi * all_hz * (dead_time_s - simulated_dead_time_s)
    )
    carrier, lines_signal = signal[0], signal[1:]
    bloch_deg = np.rad2deg(np.angle(lines_signal * np.conj(carrier)))
    rule_deg = 360.0 * frequency_hz * tau_s
    tau_fit_s = np.nan
    if np.unique(frequency_hz).size >= 2:
        tau_fit_s = fit_phase_law(frequency_hz, lines_signal, tau_s)["tau_s"]
    return {
        "rule_deg": wrap_deg(rule_deg),
        "bloch_deg": wrap_deg(bloch_deg),
        "difference_deg": wrap_deg(bloch_deg - rule_deg),
        "tau_fit_s": tau_fit_s,
        "dead_time_s": dead_time_s,
    }


def main() -> None:
    """Parse the command line, simulate and print the line phases."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pulse_group = parser.add_argument_group("excitation pulse")
    pulse_group.add_argument(
        "--pulse", default="slr", help="rect, sinc, slr, gaussian, hermite or .exc"
    )
    pulse_group.add_argument("--duration-ms", type=float, required=True)
    pulse_group.add_argument("--flip-deg", type=float, required=True)
    width = pulse_group.add_mutually_exclusive_group()
    width.add_argument(
        "--bandwidth-hz", type=float, help="Bruker ExcPulse1 bandwidth (field 1)"
    )
    width.add_argument("--tbw", type=float, help="time-bandwidth product")
    pulse_group.add_argument("--slr-sharpness", type=int, default=1)
    pulse_group.add_argument(
        "--rephasing-percent", type=float, default=50.0, help="Bruker Rpfac"
    )
    pulse_group.add_argument("--n-points", type=int, default=168)
    slice_group = parser.add_argument_group("slice selection")
    slice_group.add_argument("--slice-mm", type=float, help="omit for no slice")
    slice_group.add_argument("--slice-offset-mm", type=float, default=0.0)
    slice_group.add_argument("--rephaser-fraction", type=float, default=0.5)
    slice_group.add_argument("--rephaser-ms", type=float, default=0.2)
    parser.add_argument(
        "--tau-ms",
        type=float,
        required=True,
        help="pulse isodelay to first sample, e.g. phase_band_tau_s_from_scan",
    )
    lines_group = parser.add_argument_group("lines (default: grid over span)")
    lines_group.add_argument("--freq-hz", type=float, nargs="+")
    lines_group.add_argument("--ppm", type=float, nargs="+")
    lines_group.add_argument("--carrier-ppm", type=float)
    lines_group.add_argument("--larmor-mhz", type=float)
    lines_group.add_argument("--span-hz", type=float, default=3000.0)
    lines_group.add_argument("--step-hz", type=float, default=250.0)
    args = parser.parse_args()

    if args.tbw is not None:
        tbw = args.tbw
    elif args.bandwidth_hz is not None:
        tbw = args.bandwidth_hz * args.duration_ms * 1e-3
    else:
        parser.error("give --bandwidth-hz or --tbw")
    ppm = None
    if args.ppm is not None:
        if args.carrier_ppm is None or args.larmor_mhz is None:
            parser.error("--ppm needs --carrier-ppm and --larmor-mhz")
        ppm = np.asarray(args.ppm)
        frequency_hz = (ppm - args.carrier_ppm) * args.larmor_mhz
    elif args.freq_hz is not None:
        frequency_hz = np.asarray(args.freq_hz)
    else:
        half = args.span_hz / 2
        frequency_hz = np.arange(-half, half + args.step_hz / 2, args.step_hz)

    pulse = PulseSettings(
        shape=args.pulse,
        duration_s=args.duration_ms * 1e-3,
        flip_deg=args.flip_deg,
        time_bandwidth=tbw,
        n_points=args.n_points,
        rephasing_percent=args.rephasing_percent,
        slr_sharpness=args.slr_sharpness,
    )
    tau_s = args.tau_ms * 1e-3
    result = predict_line_phase_deg(
        pulse,
        args.slice_mm,
        args.slice_offset_mm,
        args.rephaser_fraction,
        args.rephaser_ms * 1e-3,
        tau_s,
        frequency_hz,
    )

    print(
        f"{args.pulse} {args.duration_ms} ms, TBW {tbw:.2f}, {args.flip_deg} deg, "
        f"slice {args.slice_mm} mm at {args.slice_offset_mm:+.3f} mm; "
        f"tau {args.tau_ms:.5f} ms (pulse end to first sample "
        f"{result['dead_time_s'] * 1e3:.5f} ms)"
    )
    if np.isfinite(result["tau_fit_s"]):
        print(
            f"tau fitted to the simulated lines: {result['tau_fit_s'] * 1e3:.5f} ms "
            f"({(result['tau_fit_s'] - tau_s) * 1e6:+.2f} us)"
        )
    print("phases relative to the carrier line, wrapped to (-180, 180] deg")
    header = f"{'f/Hz':>9}" + (f"{'ppm':>9}" if ppm is not None else "")
    print(header + f"{'law/deg':>10}{'Bloch/deg':>11}{'diff/deg':>10}")
    for index, f in enumerate(frequency_hz):
        row = f"{f:9.1f}" + (f"{ppm[index]:9.2f}" if ppm is not None else "")
        print(
            row
            + f"{result['rule_deg'][index]:10.2f}"
            + f"{result['bloch_deg'][index]:11.2f}"
            + f"{result['difference_deg'][index]:10.2f}"
        )


if __name__ == "__main__":
    main()
