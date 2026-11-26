"""Simulate the phase of FID lines after an excitation pulse with Bloch equations.

The phase band of the hypermri FID fit assumes the phase law
``angle(a_k) = phi0 + 2*pi*f_k*tau``: all lines share one phase at the
excitation isodelay, and ``tau`` is the time from there to the first sample.
This script checks that assumption for a chosen pulse, sequence timing and
metabolite distribution over a range of flip angles. It simulates the
excitation (and, for slice-selective excitation, the slice and rephasing
gradients) with ``blochsimulator`` (https://github.com/LucaNagel/bloch_sim_gui),
takes the transverse magnetization at the start of the acquisition on a fine
frequency grid, and fits the phase law to it for every flip angle.

The rule ``pulse length * rephasing factor + dead time`` (the rule of
``hypermri.analysis.spectroscopy.phase_band_tau_s_from_scan`` without the
receiver terms) is compared with the simulated ``tau``. The simulation covers
spins only: the digital filter, the receiver delay and the empirical extra
dwell of ``phase_band_tau_s_from_scan`` are not part of it and must be added
to the simulated ``tau``.

For slice-selective scenarios, two small compartments at +-3 mm with the same
frequency show whether the phase also depends on the position in the slice,
which no single ``tau`` can describe.

All frequencies are offsets from the RF carrier in Hz, so the nucleus enters
only through the gradient strength in Hz/cm. ``blochsimulator`` uses the
proton gyromagnetic ratio for B1 and gradients internally; this script
converts its Hz-based settings with that same ratio, so results hold for any
nucleus.

Examples
--------
Built-in scenarios, each with a flip-angle sweep from 5 to 90 degrees::

    python scripts/simulate_fid_phase_timing.py --scenario singlepulse --plot sp.png
    python scripts/simulate_fid_phase_timing.py --scenario all --plot phase

A custom setting with selected flip angles and a wider band::

    python scripts/simulate_fid_phase_timing.py --pulse sinc --duration-ms 0.84 \\
        --tbw 4 --slice-mm 10 --dead-time-ms 0.5 --flip-deg 5 10 20 45 90 \\
        --span-hz 4000 --plot sinc.png

Requires the ``blochsimulator`` package (``pip install -e`` of the
bloch_sim_gui checkout) and, for ``--plot``, matplotlib; hypermri itself does
not depend on them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

import numpy as np

PROTON_GAMMA_HZ_PER_GAUSS = 4258.0
"""Gyromagnetic ratio that blochsimulator uses for B1 and gradients."""

DEFAULT_FLIPS_DEG = tuple(float(f) for f in range(5, 95, 5))
"""Flip angles of the default sweep in degrees (5, 10, ..., 90)."""


@dataclass(frozen=True)
class PulseSettings:
    """Excitation pulse of the simulated sequence.

    Parameters
    ----------
    shape : str
        ``blochsimulator.design_rf_pulse`` type (``"rect"``, ``"sinc"``,
        ``"gaussian"``, ``"slr"``, ``"hermite"``) or a path to a Bruker
        ``.exc`` shape file.
    duration_s : float
        Pulse length in s.
    flip_deg : float
        Nominal flip angle on resonance in degrees.
    time_bandwidth : float
        Time-bandwidth product of ``sinc``, ``gaussian`` and ``slr`` pulses;
        also sets the excited bandwidth ``time_bandwidth / duration_s`` for
        slice selection.
    n_points : int
        Number of RF samples.
    rephasing_percent : float
        Fraction of the pulse after its isodelay, in percent, used for the
        predicted ``tau`` (Bruker ``Rpfac``; 50 for symmetric pulses).
    slr_sharpness : int
        Design order of ``slr`` pulses (``blochsimulator`` ``slr_sharpness``);
        1 gives a single central lobe, higher values more side lobes.
    """

    shape: str = "rect"
    duration_s: float = 0.128e-3
    flip_deg: float = 90.0
    time_bandwidth: float = 4.0
    n_points: int = 128
    rephasing_percent: float = 50.0
    slr_sharpness: int = 1


@dataclass(frozen=True)
class Metabolite:
    """One spectral line and its spatial distribution along the slice axis.

    Parameters
    ----------
    name : str
        Label used in the output.
    frequency_hz : float
        Offset from the RF carrier in Hz.
    z_cm : numpy.ndarray
        Positions along the slice axis in cm, shape ``(n_positions,)``.
    weights : numpy.ndarray
        Spin density at ``z_cm``, same shape, non-negative.
    """

    name: str
    frequency_hz: float
    z_cm: np.ndarray = field(default_factory=lambda: np.zeros(1))
    weights: np.ndarray = field(default_factory=lambda: np.ones(1))


@dataclass(frozen=True)
class SequenceSettings:
    """Timing and gradients between the pulse and the first sample.

    Parameters
    ----------
    dead_time_s : float
        Time from the end of the pulse to the first acquired sample in s
        (Bruker ``DeadTime`` for SINGLEPULSE).
    slice_mm : float or None
        Slice thickness in mm. ``None`` excites without gradients.
    rephaser_fraction : float
        Area of the rephasing lobe relative to the slice-select area during
        the whole pulse. 0.5 refocuses a symmetric pulse exactly.
    rephaser_duration_s : float
        Duration of the rephasing lobe in s; it starts right after the pulse
        and must fit into ``dead_time_s``.
    gradient_polarity : int
        +1 or -1. Multiplies slice-select and rephasing gradients; -1 mirrors
        the slice axis.
    slice_offset_mm : float
        Centre of the excited slice along the slice axis in mm. The RF is
        frequency-modulated to excite it, with its phase referenced to the
        pulse isodelay, so the slice centre keeps the phase of an unshifted
        slice.
    """

    dead_time_s: float = 0.05e-3
    slice_mm: float | None = None
    rephaser_fraction: float = 0.5
    rephaser_duration_s: float = 0.2e-3
    gradient_polarity: int = 1
    slice_offset_mm: float = 0.0


@dataclass(frozen=True)
class SweepResult:
    """Simulated signals and phase-law fits for a flip-angle sweep.

    Parameters
    ----------
    flips_deg : numpy.ndarray
        Flip angles in degrees, shape ``(n_flips,)``.
    frequency_hz : numpy.ndarray
        Grid offsets from the carrier in Hz, shape ``(n_lines,)``, ascending.
    grid_signal : numpy.ndarray
        Complex transverse magnetization on the grid at the first sample,
        shape ``(n_flips, n_lines)``.
    rule_tau_s : float
        ``pulse length * rephasing factor + dead time`` in s.
    tau_s : numpy.ndarray
        Fitted ``tau`` per flip angle in s, shape ``(n_flips,)``.
    nonlinear_deg : numpy.ndarray
        Deviation of every grid line from its best linear phase law in
        degrees, shape ``(n_flips, n_lines)``.
    rule_error_deg : numpy.ndarray
        Phase of every grid line after removing ``2 pi f rule_tau_s``,
        relative to the line closest to the carrier, in degrees, shape
        ``(n_flips, n_lines)``. This is the phase error left when the fit
        uses the rule.
    compartments : list of Metabolite
        Extra lines that are not part of the grid.
    compartment_rule_error_deg : numpy.ndarray
        Their phase error relative to the rule and to the grid line closest
        to the carrier, in degrees, shape ``(n_flips, n_compartments)``.
    """

    flips_deg: np.ndarray
    frequency_hz: np.ndarray
    grid_signal: np.ndarray
    rule_tau_s: float
    tau_s: np.ndarray
    nonlinear_deg: np.ndarray
    rule_error_deg: np.ndarray
    compartments: list
    compartment_rule_error_deg: np.ndarray


def pulse_waveform(pulse: PulseSettings) -> tuple[np.ndarray, float]:
    """Return the complex B1 samples in Gauss and the raster step in s.

    Parameters
    ----------
    pulse : PulseSettings
        Pulse to build. A Bruker shape file is rescaled so that its integral
        gives ``pulse.flip_deg`` on resonance; this is exact only for
        amplitude-modulated shapes.

    Returns
    -------
    b1_gauss : numpy.ndarray
        Complex B1 in Gauss, shape ``(n_points,)``.
    dt_s : float
        Raster step in s.
    """
    from blochsimulator import design_rf_pulse

    if pulse.shape.endswith(".exc"):
        from blochsimulator.pulse_loader import load_pulse

        b1, _, _ = load_pulse(pulse.shape)
        b1 = np.asarray(b1, dtype=complex)
        source_t = np.linspace(0.0, 1.0, b1.size)
        target_t = np.linspace(0.0, 1.0, pulse.n_points)
        b1 = np.interp(target_t, source_t, b1.real) + 1j * np.interp(
            target_t, source_t, b1.imag
        )
        dt_s = pulse.duration_s / pulse.n_points
        area = abs(b1.sum()) * dt_s
        target = np.deg2rad(pulse.flip_deg) / (2 * np.pi * PROTON_GAMMA_HZ_PER_GAUSS)
        return b1 * (target / area), dt_s
    b1, time_s = design_rf_pulse(
        pulse_type=pulse.shape,
        duration=pulse.duration_s,
        flip_angle=pulse.flip_deg,
        time_bw_product=pulse.time_bandwidth,
        npoints=pulse.n_points,
        slr_sharpness=pulse.slr_sharpness,
    )
    return np.asarray(b1, dtype=complex), float(time_s[1] - time_s[0])


def build_sequence(
    pulse: PulseSettings, sequence: SequenceSettings, precession_sign: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return B1, gradients and sample times from pulse start to acquisition.

    Parameters
    ----------
    pulse : PulseSettings
        Excitation pulse.
    sequence : SequenceSettings
        Dead time and optional slice selection.
    precession_sign : int
        +1 if the simulator lets a positive offset precess as
        ``exp(+i 2 pi f t)``, else -1. Sets the direction of the RF frequency
        modulation for ``sequence.slice_offset_mm``: in ``blochsimulator`` the
        B1 phase that excites spins precessing as ``exp(i s 2 pi f t)`` turns
        as ``exp(-i s 2 pi f t)``, so the excited profile centres on the
        offset for both gradient polarities.

    Returns
    -------
    b1_gauss : numpy.ndarray
        Complex B1 in Gauss, shape ``(n,)``.
    gradients_gauss_per_cm : numpy.ndarray
        Gradients in G/cm, shape ``(n, 3)``; only z is used.
    time_s : numpy.ndarray
        Sample centres in s, shape ``(n,)``. The simulated end ``n * dt``
        is the first acquired sample.

    Raises
    ------
    ValueError
        If the rephasing lobe does not fit into the dead time, or the slice
        offset is not zero without slice selection.
    """
    b1_pulse, dt_s = pulse_waveform(pulse)
    if sequence.slice_mm is None and sequence.slice_offset_mm != 0.0:
        raise ValueError(
            "build_sequence: slice_offset_mm needs slice_mm; set slice_mm or use "
            "slice_offset_mm=0."
        )
    n_pulse = b1_pulse.size
    n_dead = int(round(sequence.dead_time_s / dt_s))
    b1 = np.concatenate([b1_pulse, np.zeros(n_dead, dtype=complex)])
    gradients = np.zeros((b1.size, 3))
    if sequence.slice_mm is not None:
        bandwidth_hz = pulse.time_bandwidth / pulse.duration_s
        slice_hz_per_cm = bandwidth_hz / (sequence.slice_mm / 10.0)
        g_slice = (
            sequence.gradient_polarity * slice_hz_per_cm / PROTON_GAMMA_HZ_PER_GAUSS
        )
        n_reph = int(round(sequence.rephaser_duration_s / dt_s))
        if n_reph > n_dead or n_reph < 1:
            raise ValueError(
                "build_sequence: rephaser_duration_s must lie between one raster "
                f"step ({dt_s:.2e} s) and dead_time_s ({sequence.dead_time_s:.2e} s)."
            )
        offset_hz = (
            sequence.gradient_polarity
            * slice_hz_per_cm
            * sequence.slice_offset_mm
            / 10.0
        )
        isodelay_s = pulse.duration_s * (1.0 - pulse.rephasing_percent / 100.0)
        pulse_time_s = (np.arange(n_pulse) + 0.5) * dt_s
        b1[:n_pulse] *= np.exp(
            -1j * precession_sign * 2 * np.pi * offset_hz * (pulse_time_s - isodelay_s)
        )
        gradients[:n_pulse, 2] = g_slice
        reph_area = sequence.rephaser_fraction * g_slice * n_pulse
        gradients[n_pulse : n_pulse + n_reph, 2] = -reph_area / n_reph
    time_s = (np.arange(b1.size) + 0.5) * dt_s
    return b1, gradients, time_s


def transverse_signal_at_acquisition(
    pulse: PulseSettings,
    sequence: SequenceSettings,
    metabolites: list[Metabolite],
) -> np.ndarray:
    """Simulate each metabolite and return its signal at the first sample.

    The sign follows the hypermri FID model, in which a line at ``f`` evolves
    as ``exp(+i 2 pi f t)``. The simulator's own sign is measured with a
    short free precession and conjugated if necessary. Metabolites with the
    same spatial distribution are simulated in one call.

    Parameters
    ----------
    pulse : PulseSettings
        Excitation pulse.
    sequence : SequenceSettings
        Dead time and optional slice selection.
    metabolites : list of Metabolite
        Lines with their spatial distribution. Relaxation is neglected
        (T1 = T2 = 100 s), because it changes amplitudes, not phases.

    Returns
    -------
    numpy.ndarray
        Complex transverse magnetization summed over each metabolite's
        positions with its weights, shape ``(n_metabolites,)``. Deterministic.
    """
    from blochsimulator import BlochSimulator, TissueParameters

    simulator = BlochSimulator()
    tissue = TissueParameters("phase", t1=100.0, t2=100.0)
    sign = _simulator_precession_sign(
        simulator, tissue, pulse.duration_s / pulse.n_points
    )
    b1, gradients, time_s = build_sequence(pulse, sequence, sign)
    dt_s = time_s[1] - time_s[0]
    groups: dict[tuple, list[int]] = {}
    for index, metabolite in enumerate(metabolites):
        key = (metabolite.z_cm.tobytes(), metabolite.weights.tobytes())
        groups.setdefault(key, []).append(index)
    signals = np.empty(len(metabolites), dtype=complex)
    for indices in groups.values():
        first = metabolites[indices[0]]
        positions_m = np.zeros((first.z_cm.size, 3))
        positions_m[:, 2] = first.z_cm / 100.0
        result = simulator.simulate(
            (b1, gradients, time_s),
            tissue,
            positions=positions_m,
            frequencies=np.array([metabolites[i].frequency_hz for i in indices]),
            mode=0,
            dt=dt_s,
        )
        mxy = np.asarray(result["mx"]) + 1j * np.asarray(result["my"])
        total = first.weights @ mxy.reshape(first.z_cm.size, len(indices))
        signals[indices] = total if sign > 0 else np.conj(total)
    return signals


def _simulator_precession_sign(simulator, tissue, dt_s: float) -> int:
    """Return +1 if a positive offset precesses as exp(+i 2 pi f t), else -1."""
    n_free = 200
    b1 = np.zeros(n_free + 1, dtype=complex)
    time_s = (np.arange(b1.size) + 0.5) * dt_s
    result = simulator.simulate(
        (b1, np.zeros((b1.size, 3)), time_s),
        tissue,
        frequencies=np.array([100.0]),
        initial_magnetization=np.array([1.0, 0.0, 0.0]),
        mode=0,
        dt=dt_s,
    )
    mxy = complex(np.ravel(result["mx"])[-1] + 1j * np.ravel(result["my"])[-1])
    return 1 if np.angle(mxy) > 0 else -1


def fit_phase_law(
    frequency_hz: np.ndarray, signal: np.ndarray, reference_tau_s: float = 0.0
) -> dict:
    """Fit ``angle(signal) = phi0 + 2 pi f tau`` with magnitude weights.

    Parameters
    ----------
    frequency_hz : numpy.ndarray
        Line offsets in Hz, shape ``(n,)``, at least two distinct values.
    signal : numpy.ndarray
        Complex line amplitudes, same shape.
    reference_tau_s : float
        Expected ``tau`` in s. Its linear phase is removed before unwrapping
        and added back afterwards, so widely spaced lines are unwrapped
        correctly as long as the true ``tau`` is within ``1 / (2 df)`` of it.

    Returns
    -------
    dict
        ``tau_s`` (s), ``phi0_deg`` (degrees) and ``residual_deg``
        (deviation of every line from the law in degrees, shape ``(n,)``).
        The remaining phase is unwrapped along increasing frequency.
    """
    order = np.argsort(frequency_hz)
    f = frequency_hz[order]
    reference = np.exp(-2j * np.pi * f * reference_tau_s)
    phase = np.unwrap(np.angle(signal[order] * reference)) + 2 * np.pi * f * (
        reference_tau_s
    )
    weights = np.abs(signal[order])
    design = np.column_stack([np.ones_like(f), 2 * np.pi * f])
    sqrt_w = np.sqrt(weights)[:, None]
    (phi0, tau_s), *_ = np.linalg.lstsq(design * sqrt_w, phase * sqrt_w[:, 0])
    residual = np.empty_like(phase)
    residual[order] = np.angle(np.exp(1j * (phase - design @ [phi0, tau_s])))
    return {
        "tau_s": float(tau_s),
        "phi0_deg": float(np.rad2deg(phi0)),
        "residual_deg": np.rad2deg(residual),
    }


def predicted_tau_s(pulse: PulseSettings, sequence: SequenceSettings) -> float:
    """Return ``pulse length * rephasing factor + dead time`` in s."""
    return pulse.duration_s * pulse.rephasing_percent / 100.0 + sequence.dead_time_s


def uniform_slab(thickness_mm: float, n_positions: int = 101) -> tuple:
    """Return z positions in cm and equal weights over ``thickness_mm``."""
    z_cm = np.linspace(-thickness_mm / 20.0, thickness_mm / 20.0, n_positions)
    return z_cm, np.ones_like(z_cm)


def compartment(name: str, frequency_hz: float, centre_mm: float) -> Metabolite:
    """Return a line confined to a 2 mm compartment centred at ``centre_mm``."""
    z_cm, weights = uniform_slab(2.0, 21)
    return Metabolite(name, frequency_hz, z_cm + centre_mm / 10.0, weights)


def frequency_grid(
    span_hz: float, n_lines: int, slab_mm: float | None, centre_mm: float = 0.0
) -> list[Metabolite]:
    """Return evenly spaced lines over ``[-span_hz/2, span_hz/2]``.

    Parameters
    ----------
    span_hz : float
        Total width of the grid in Hz.
    n_lines : int
        Number of lines; odd values place one line on the carrier.
    slab_mm : float or None
        Thickness of a uniform slab that every line fills, or ``None`` for a
        single position at z = 0.
    centre_mm : float
        Centre of the slab along the slice axis in mm.

    Returns
    -------
    list of Metabolite
        Lines in ascending frequency, all with the same distribution.
    """
    if slab_mm is None:
        distribution = (np.zeros(1), np.ones(1))
    else:
        z_cm, weights = uniform_slab(slab_mm)
        distribution = (z_cm + centre_mm / 10.0, weights)
    return [
        Metabolite(f"{f:+.0f} Hz", f, *distribution)
        for f in np.linspace(-span_hz / 2, span_hz / 2, n_lines)
    ]


def simulate_flip_sweep(
    pulse: PulseSettings,
    sequence: SequenceSettings,
    flips_deg: tuple[float, ...],
    span_hz: float,
    n_lines: int,
    compartments: list[Metabolite],
) -> SweepResult:
    """Simulate the frequency grid and compartments for every flip angle.

    Parameters
    ----------
    pulse : PulseSettings
        Excitation pulse; its ``flip_deg`` is replaced by each sweep value.
    sequence : SequenceSettings
        Dead time and optional slice selection. With slice selection the
        grid lines fill a uniform slab twice as thick as the slice, centred
        on the slice.
    flips_deg : tuple of float
        Flip angles in degrees.
    span_hz, n_lines : float, int
        Width in Hz and number of lines of the frequency grid.
    compartments : list of Metabolite
        Extra lines evaluated against the grid's carrier phase.

    Returns
    -------
    SweepResult
        Signals and fits; see its documentation for shapes and units.
    """
    slab_mm = 2 * sequence.slice_mm if sequence.slice_mm is not None else None
    grid = frequency_grid(span_hz, n_lines, slab_mm, sequence.slice_offset_mm)
    frequency_hz = np.array([m.frequency_hz for m in grid])
    compartment_hz = np.array([m.frequency_hz for m in compartments])
    rule_tau_s = predicted_tau_s(pulse, sequence)
    carrier = int(np.argmin(np.abs(frequency_hz)))
    rows = []
    for flip in flips_deg:
        signal = transverse_signal_at_acquisition(
            replace(pulse, flip_deg=flip), sequence, grid + compartments
        )
        grid_signal, compartment_signal = signal[: len(grid)], signal[len(grid) :]
        fit = fit_phase_law(frequency_hz, grid_signal, rule_tau_s)
        rule_removed = grid_signal * np.exp(-2j * np.pi * frequency_hz * rule_tau_s)
        carrier_phase = np.angle(rule_removed[carrier])
        rule_error = np.unwrap(np.angle(rule_removed) - carrier_phase)
        rule_error -= rule_error[carrier]
        compartment_error = np.angle(
            compartment_signal
            * np.exp(-2j * np.pi * compartment_hz * rule_tau_s - 1j * carrier_phase)
        )
        rows.append(
            (
                grid_signal,
                fit["tau_s"],
                fit["residual_deg"],
                np.rad2deg(rule_error),
                np.rad2deg(compartment_error),
            )
        )
    grid_signal, tau_s, nonlinear, rule_error, compartment_error = map(
        np.array, zip(*rows, strict=True)
    )
    return SweepResult(
        flips_deg=np.asarray(flips_deg, dtype=float),
        frequency_hz=frequency_hz,
        grid_signal=grid_signal,
        rule_tau_s=rule_tau_s,
        tau_s=tau_s,
        nonlinear_deg=nonlinear,
        rule_error_deg=rule_error,
        compartments=compartments,
        compartment_rule_error_deg=compartment_error.reshape(
            len(flips_deg), len(compartments)
        ),
    )


def print_sweep(title: str, result: SweepResult) -> None:
    """Print one summary row per flip angle."""
    f_max = result.frequency_hz.max()
    print(f"\n== {title}")
    print(
        f"rule tau {result.rule_tau_s * 1e3:.4f} ms; grid {result.frequency_hz.size} "
        f"lines over +-{f_max:.0f} Hz "
        f"(spacing {np.diff(result.frequency_hz)[0]:.1f} Hz)"
    )
    header = (
        f"{'flip/deg':>8} {'tau/ms':>8} {'tau-rule/us':>11} "
        f"{'nonlin max/deg':>14} {'rule err +-fmax/deg':>20}"
    )
    header += "".join(f" {m.name + '/deg':>14}" for m in result.compartments)
    print(header)
    for i, flip in enumerate(result.flips_deg):
        line = (
            f"{flip:8.0f} {result.tau_s[i] * 1e3:8.4f} "
            f"{(result.tau_s[i] - result.rule_tau_s) * 1e6:+11.2f} "
            f"{np.abs(result.nonlinear_deg[i]).max():14.3f} "
            f"{result.rule_error_deg[i, 0]:+9.2f} {result.rule_error_deg[i, -1]:+9.2f}"
        )
        line += "".join(f" {v:+14.2f}" for v in result.compartment_rule_error_deg[i])
        print(line)


SEQUENTIAL_BLUE = ("#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
"""Light-to-dark blue steps for the flip-angle colour scale."""
COMPARTMENT_COLOURS = ("#eb6834", "#1baf7a")
"""Categorical colours for the two compartments (validated pair)."""
INK = "#33322f"
MUTED_INK = "#77766f"
GRID = "#e4e3df"


def plot_sweep(title: str, result: SweepResult, path: str) -> None:
    """Save a four-panel summary of the sweep as PNG.

    Panels: (a) phase error versus frequency when the rule's ``tau`` is
    used; (b) ``tau`` minus the rule versus flip angle; (c) spectral
    excitation profile ``|M_xy|`` relative to the carrier line; (d) largest deviation from the
    best linear law and, for slice-selective scenarios, the compartment phase
    errors versus flip angle. Colour in (a) and (c) encodes the flip angle.

    Parameters
    ----------
    title : str
        Figure title.
    result : SweepResult
        Sweep to plot.
    path : str
        Output PNG path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED_INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )
    cmap = LinearSegmentedColormap.from_list("flip", SEQUENTIAL_BLUE)
    norm = Normalize(result.flips_deg.min(), result.flips_deg.max())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    (ax_err, ax_tau), (ax_mag, ax_lin) = axes
    f = result.frequency_hz
    carrier = int(np.argmin(np.abs(f)))
    for i, flip in enumerate(result.flips_deg):
        colour = cmap(norm(flip))
        ax_err.plot(f, result.rule_error_deg[i], color=colour, lw=1.5)
        ax_mag.plot(
            f,
            np.abs(result.grid_signal[i]) / np.abs(result.grid_signal[i, carrier]),
            color=colour,
            lw=1.5,
        )
    ax_err.axhline(0, color=MUTED_INK, lw=0.8)
    ax_err.set(
        xlabel="offset from carrier / Hz",
        ylabel="phase error / deg",
        title=f"(a) phase error with rule tau = {result.rule_tau_s * 1e3:.3f} ms",
    )
    ax_mag.set(
        xlabel="offset from carrier / Hz",
        ylabel="|M_xy| relative to carrier",
        title="(c) spectral excitation profile",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax_err, ax_mag],
        label="flip angle / deg",
        location="left",
        shrink=0.6,
        aspect=30,
        pad=0.06,
    )

    tau_diff_us = (result.tau_s - result.rule_tau_s) * 1e6
    ax_tau.plot(result.flips_deg, tau_diff_us, "o-", color=SEQUENTIAL_BLUE[2], ms=5)
    ax_tau.axhline(0, color=MUTED_INK, lw=0.8)
    ax_tau.set(
        xlabel="flip angle / deg",
        ylabel="simulated tau - rule / us",
        title="(b) shift of the fitted tau",
    )

    nonlinear_max = np.abs(result.nonlinear_deg).max(axis=1)
    ax_lin.plot(
        result.flips_deg,
        nonlinear_max,
        "o-",
        color=INK,
        ms=5,
        label="max deviation from best linear law",
    )
    for j, comp in enumerate(result.compartments):
        ax_lin.plot(
            result.flips_deg,
            result.compartment_rule_error_deg[:, j],
            "s--" if j else "^-",
            color=COMPARTMENT_COLOURS[j % 2],
            ms=5,
            label=f"compartment {comp.name} ({comp.frequency_hz:+.0f} Hz), vs rule",
        )
    ax_lin.axhline(0, color=MUTED_INK, lw=0.8)
    ax_lin.set(
        xlabel="flip angle / deg",
        ylabel="phase / deg",
        title="(d) nonlinearity and position dependence",
    )
    ax_lin.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


SCENARIOS = {
    "singlepulse": (
        "Bruker SINGLEPULSE (OMPD titration): 0.128 ms block, DeadTime 0.05 ms",
        PulseSettings("rect", 0.128e-3, 90.0, 4.0, 128, 50.0),
        SequenceSettings(dead_time_s=0.05e-3),
        None,
    ),
    "slice-sinc": (
        "Slice-selective sinc, TBW 4, 0.84 ms, 10 mm, rephaser 0.5, dead time 0.5 ms",
        PulseSettings("sinc", 0.84e-3, 10.0, 4.0, 168, 50.0),
        SequenceSettings(dead_time_s=0.5e-3, slice_mm=10.0),
        -750.0,
    ),
    "slice-sinc-onresonant": (
        "As slice-sinc, compartments on resonance (0 Hz)",
        PulseSettings("sinc", 0.84e-3, 10.0, 4.0, 168, 50.0),
        SequenceSettings(dead_time_s=0.5e-3, slice_mm=10.0),
        0.0,
    ),
    "slice-sinc-misrephased": (
        "As slice-sinc, rephaser area 0.45 instead of 0.5",
        PulseSettings("sinc", 0.84e-3, 10.0, 4.0, 168, 50.0),
        SequenceSettings(dead_time_s=0.5e-3, slice_mm=10.0, rephaser_fraction=0.45),
        -750.0,
    ),
}
"""Built-in settings: title, pulse, sequence and compartment frequency in Hz
(``None`` for no compartments). The flip angle of each pulse is replaced by
the sweep."""


def run(
    title: str,
    pulse: PulseSettings,
    sequence: SequenceSettings,
    args: argparse.Namespace,
    plot_path: str | None,
    compartment_hz: float | None,
) -> SweepResult:
    """Simulate, print and optionally plot one sweep.

    ``compartment_hz`` adds two 2 mm compartments at +-3 mm with that
    frequency; ``None`` adds none.
    """
    compartments = []
    if compartment_hz is not None:
        compartments = [
            compartment("+3 mm", compartment_hz, 3.0),
            compartment("-3 mm", compartment_hz, -3.0),
        ]
    result = simulate_flip_sweep(
        pulse, sequence, tuple(args.flip_deg), args.span_hz, args.n_lines, compartments
    )
    print_sweep(title, result)
    if plot_path:
        plot_sweep(title, result, plot_path)
    return result


def main() -> None:
    """Parse the command line and run the selected settings."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"])
    parser.add_argument("--pulse", default="rect", help="design type or .exc file")
    parser.add_argument("--duration-ms", type=float, default=0.128)
    parser.add_argument(
        "--flip-deg", type=float, nargs="+", default=list(DEFAULT_FLIPS_DEG)
    )
    parser.add_argument("--tbw", type=float, default=4.0)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--rephasing-percent", type=float, default=50.0)
    parser.add_argument("--dead-time-ms", type=float, default=0.05)
    parser.add_argument("--slice-mm", type=float, default=None)
    parser.add_argument("--rephaser-fraction", type=float, default=0.5)
    parser.add_argument("--rephaser-ms", type=float, default=0.2)
    parser.add_argument(
        "--flip-gradient",
        action="store_true",
        help="invert slice-select and rephasing gradients (also for scenarios)",
    )
    parser.add_argument("--span-hz", type=float, default=2000.0)
    parser.add_argument("--n-lines", type=int, default=401)
    parser.add_argument(
        "--compartment-hz",
        type=float,
        default=-750.0,
        help="frequency of the +-3 mm compartments (custom slice-selective only)",
    )
    parser.add_argument("--plot", help="PNG path (with --scenario all: a prefix)")
    args = parser.parse_args()

    if args.scenario:
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for name in names:
            title, pulse, sequence, compartment_hz = SCENARIOS[name]
            if args.flip_gradient:
                sequence = replace(sequence, gradient_polarity=-1)
                title += ", gradients inverted"
            plot = args.plot
            if plot and args.scenario == "all":
                plot = f"{plot}_{name}.png"
            run(f"{name}: {title}", pulse, sequence, args, plot, compartment_hz)
        return

    pulse = PulseSettings(
        args.pulse,
        args.duration_ms * 1e-3,
        args.flip_deg[0],
        args.tbw,
        args.n_points,
        args.rephasing_percent,
    )
    sequence = SequenceSettings(
        args.dead_time_ms * 1e-3,
        args.slice_mm,
        args.rephaser_fraction,
        args.rephaser_ms * 1e-3,
        -1 if args.flip_gradient else 1,
    )
    compartment_hz = args.compartment_hz if args.slice_mm is not None else None
    run("custom setting", pulse, sequence, args, args.plot, compartment_hz)


if __name__ == "__main__":
    main()
