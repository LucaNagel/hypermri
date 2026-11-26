#!/usr/bin/env python3
"""Simulate and fit 256-point five-metabolite 13C FIDs and audit local errors.

Examples
--------
Run from a checkout with HyperMRI installed (preferably editable)::

    python scripts/simulate_13c_spectroscopy.py --output tmp/13c_simulation
    python scripts/simulate_13c_spectroscopy.py --repeats 200 --n-jobs 4
    python scripts/simulate_13c_spectroscopy.py --snr 0.5 5 100 \
        --ratios 1 0.2 0.08 0.06 0.03 --t2-ms 35 25 20 18 12

Notes
-----
SNR means Pyruvate's FID amplitude at t=0 divided by the known noise SD per
real/imaginary quadrature. Each metabolite has its own amplitude/sigma SNR.
Default ratios and T2* profiles are illustrative, not biological reference
values. Chemical shifts come from HyperMRI's metabolite table; their offsets
from Pyruvate are converted to Hz using 7 * get_gmr('13c') by default, without
additional frequency perturbations. Independent
Gaussian noise is added before fitting the temporal complex128 FIDs; no
filtering, baseline, alignment or noise estimation is involved.

Each realization is fitted once by fit_spectrum, followed by
estimate_fit_covariance. Repetitions measure empirical sampling behavior;
there is no bootstrap inside a fit. The chemical-shift frequencies are also
the initial guesses, but remain free fit parameters. T2* starts at 25 ms for
all metabolites. Float64 calculations are eager and seeded. Import
has no data-generation, fitting, plotting or filesystem side effects. Execution
writes CSV, JSON, NPZ, PNG and SVG files to a new or empty output directory.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    reconstruct_fitted_signal,
)
from hypermri.analysis.spectroscopy import estimate_fit_covariance, fit_spectrum
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm
from hypermri.utils.physics import get_gmr

_METABOLITES = ("Pyruvate", "Lactate", "Alanine", "PyruvateHydrate", "Bicarbonate")
_RATIOS = {
    "balanced": (1.0, 0.5, 0.3, 0.25, 0.15),
    "weak_products": (1.0, 0.2, 0.08, 0.06, 0.03),
}
_T2_MS = {
    "mixed": (35.0, 25.0, 20.0, 18.0, 12.0),
    "short": (12.0,) * 5,
    "long": (40.0,) * 5,
}
_PARAMETERS = {"amplitude": "a.u.", "frequency": "Hz", "t2": "ms", "fwhm": "Hz"}
_SAMPLES = 256


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/13c_simulation"))
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=[0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Independent noisy spectra per scenario; >= 2.",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel fitter workers; -1 uses all cores.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        metavar="A",
        help="Replace ratio profiles; order: Pyruvate Lactate Alanine Hydrate Bicarbonate.",
    )
    parser.add_argument(
        "--t2-ms",
        nargs="+",
        type=float,
        metavar="MS",
        help="Replace T2* profiles in the same metabolite order.",
    )
    parser.add_argument("--dwell-ms", type=float, default=0.4)
    parser.add_argument(
        "--larmor-mhz",
        type=float,
        default=None,
        help="Override the default 7 T 13C frequency, 7 * get_gmr('13c'), in MHz.",
    )
    return parser


def _configuration(args):
    if args.repeats < 2 or args.seed < 0 or args.n_jobs == 0:
        raise ValueError("Require repeats >= 2, seed >= 0, and n-jobs != 0.")
    gmr_mhz_per_t = get_gmr(nucleus="13c", unit="MHz_T")
    larmor_mhz = 7 * gmr_mhz_per_t if args.larmor_mhz is None else args.larmor_mhz
    for name, values in (
        ("snr", args.snr),
        ("dwell-ms", [args.dwell_ms]),
        ("larmor-mhz", [larmor_mhz]),
    ):
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError(f"{name} must contain positive finite values.")
    if len(set(args.snr)) != len(args.snr):
        raise ValueError("snr values must be unique.")
    ratios = _RATIOS if args.ratios is None else {"custom": tuple(args.ratios)}
    t2s = _T2_MS if args.t2_ms is None else {"custom": tuple(args.t2_ms)}
    for values in ratios.values():
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError(
                "ratios must be positive finite numbers, including Pyruvate."
            )
    for values in t2s.values():
        if (
            not np.all(np.isfinite(values))
            or np.any(np.asarray(values) <= 5)
            or np.any(np.asarray(values) >= 80)
        ):
            raise ValueError(
                "t2-ms must lie strictly inside the fit bounds (5, 80) ms."
            )
    ppm = np.array([get_metab_cs_ppm(name.lower()) for name in _METABOLITES])
    nominal = larmor_mhz * (ppm - ppm[0])
    if np.max(np.abs(nominal)) + 40 >= 0.5 / (args.dwell_ms * 1e-3):
        raise ValueError(
            "dwell-ms is too large: metabolite search windows reach Nyquist. Reduce dwell-ms."
        )
    if np.min(np.diff(np.sort(nominal))) <= 80:
        raise ValueError("larmor-mhz is too small for disjoint 80 Hz search windows.")
    return {
        "metabolites": _METABOLITES,
        "samples": _SAMPLES,
        "snr_pyruvate": sorted(args.snr),
        "snr_definition": "abs(a_Pyruvate(t=0))/sigma_quadrature; not spectral peak-height SNR",
        "ratio_profiles": {
            key: (np.asarray(value) / value[0]).tolist()
            for key, value in ratios.items()
        },
        "t2_profiles_ms": t2s,
        "chemical_shifts_ppm": ppm.tolist(),
        "nominal_frequencies_hz": nominal.tolist(),
        "frequency_convention": "(metabolite_ppm - pyruvate_ppm) * larmor_mhz; no additional frequency offsets",
        "amplitude_phase_rad": 0.35,
        "dwell_ms": args.dwell_ms,
        "larmor_mhz": larmor_mhz,
        "nucleus": "13c",
        "gmr_mhz_per_t": gmr_mhz_per_t,
        "field_strength_t": larmor_mhz / gmr_mhz_per_t,
        "repeats": args.repeats,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "hypermri_version": version("hypermri"),
        "noise": "Independent Gaussian real and imaginary quadratures; supplied sigma is exact.",
        "fit_options": {
            "metabs": list(_METABOLITES),
            "metabs_freqs_hz": nominal.tolist(),
            "initial_t2s_s": 0.025,
            "metabs_freq_ranges": 80.0,
            "metabs_t2_ranges": 0.15,
            "min_t2_s": 0.005,
            "max_t2_s": 0.080,
            "snr_threshold": None,
            "amp_solver": "pinv",
        },
    }


def _simulate(config, ratios, t2_ms, snr, rng):
    time_s = np.arange(_SAMPLES) * config["dwell_ms"] * 1e-3
    frequencies = np.asarray(config["nominal_frequencies_hz"])
    amplitudes = np.asarray(ratios) * np.exp(1j * config["amplitude_phase_rad"])
    components = amplitudes[:, None] * np.exp(
        (2j * np.pi * frequencies[:, None] - 1000 / np.asarray(t2_ms)[:, None]) * time_s
    )
    clean = components.sum(axis=0)
    sigma = 1.0 / snr
    noise = sigma * (
        rng.normal(size=(config["repeats"], _SAMPLES))
        + 1j * rng.normal(size=(config["repeats"], _SAMPLES))
    )
    data = xr.DataArray(
        clean + noise,
        dims=("realization", "acquisition_idx"),
        coords={
            "realization": np.arange(config["repeats"]),
            "acquisition_idx": np.arange(_SAMPLES),
            "sampling_time_axis_s": ("acquisition_idx", time_s),
        },
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )
    return data, clean, frequencies, sigma


def _trial_table(fit, errors, ratios, t2_ms, frequencies, sigma):
    shape = (fit.sizes["realization"], len(_METABOLITES))

    def values(array):
        return np.asarray(array.transpose("realization", "metabolite")).ravel()

    table = pd.DataFrame(
        {
            "realization": np.repeat(np.arange(shape[0]), shape[1]),
            "metabolite": np.tile(_METABOLITES, shape[0]),
            "snr_metabolite": np.tile(np.asarray(ratios) / sigma, shape[0]),
            "sigma_quadrature": sigma,
            "fit_status": np.repeat(fit.fit_status.values, shape[1]),
            "covariance_status": np.repeat(errors.covariance_status.values, shape[1]),
            "amplitude_abs_status": values(errors.amplitude_abs_status),
            "amplitude_true": np.tile(ratios, shape[0]),
            "amplitude_fit": np.abs(values(fit.fit_amps)),
            "amplitude_std": values(errors.amplitude_abs_std),
            "amplitude_real_fit": values(fit.fit_amps).real,
            "amplitude_imag_fit": values(fit.fit_amps).imag,
            "amplitude_conditional_quadrature_std": values(
                fit.amplitude_std_quadrature_conditional
            ),
            "frequency_true": np.tile(frequencies, shape[0]),
            "frequency_fit": values(fit.fit_freqs),
            "frequency_std": values(errors.frequency_std),
            "t2_true": np.tile(t2_ms, shape[0]),
            "t2_fit": values(fit.fit_t2s) * 1000,
            "t2_std": values(errors.t2_std) * 1000,
            "fwhm_true": np.tile(1000 / (np.pi * np.asarray(t2_ms)), shape[0]),
            "fwhm_fit": 1 / (np.pi * values(fit.fit_t2s)),
            "fwhm_std": values(errors.fwhm_std),
        }
    )
    return table


def _summarize(trials):
    rows = []
    group_names = ["ratio_profile", "t2_profile", "snr_pyruvate", "metabolite"]
    for keys, group in trials.groupby(group_names, sort=False):
        for parameter, unit in _PARAMETERS.items():
            truth = group[f"{parameter}_true"].iloc[0]
            fitted = group[f"{parameter}_fit"].to_numpy()
            std = group[f"{parameter}_std"].to_numpy()
            fit_ok = group.fit_status.eq("success").to_numpy() & np.isfinite(fitted)
            eligible = (
                fit_ok
                & group.covariance_status.eq("success").to_numpy()
                & np.isfinite(std)
                & (std > 0)
            )
            if parameter == "amplitude":
                eligible &= group.amplitude_abs_status.eq("success").to_numpy()
            delta = fitted - truth
            count = int(eligible.sum())
            empirical = float(np.std(fitted[eligible], ddof=1)) if count > 1 else np.nan
            predicted = float(np.sqrt(np.mean(std[eligible] ** 2))) if count else np.nan
            z = delta[eligible] / std[eligible]
            rows.append(
                {
                    **dict(zip(group_names, keys, strict=True)),
                    "parameter": parameter,
                    "unit": unit,
                    "truth": truth,
                    "snr_metabolite": group.snr_metabolite.iloc[0],
                    "n_total": len(group),
                    "n_fit": int(fit_ok.sum()),
                    "n_uncertainty": count,
                    "fit_success_fraction": float(fit_ok.mean()),
                    "uncertainty_available_fraction": float(eligible.mean()),
                    "fit_mean": float(np.mean(fitted[fit_ok]))
                    if fit_ok.any()
                    else np.nan,
                    "bias": float(np.mean(delta[fit_ok])) if fit_ok.any() else np.nan,
                    "rmse": float(np.sqrt(np.mean(delta[fit_ok] ** 2)))
                    if fit_ok.any()
                    else np.nan,
                    "empirical_std_all_fits": float(np.std(fitted[fit_ok], ddof=1))
                    if fit_ok.sum() > 1
                    else np.nan,
                    "empirical_std_eligible": empirical,
                    "predicted_std_rms_eligible": predicted,
                    "predicted_to_empirical_std": predicted / empirical
                    if empirical > 0
                    else np.nan,
                    "coverage_68_eligible": float(np.mean(np.abs(z) <= 1))
                    if count
                    else np.nan,
                    "coverage_95_eligible": float(np.mean(np.abs(z) <= 1.96))
                    if count
                    else np.nan,
                    "coverage_95_mc_se": float(
                        np.sqrt(
                            np.mean(np.abs(z) <= 1.96)
                            * (1 - np.mean(np.abs(z) <= 1.96))
                            / count
                        )
                    )
                    if count
                    else np.nan,
                    "standardized_error_mean": float(np.mean(z)) if count else np.nan,
                    "standardized_error_std": float(np.std(z, ddof=1))
                    if count > 1
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot(summary, examples, config, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True, layout="constrained")
    amp = summary[summary.parameter.eq("amplitude")]
    for column, metab in enumerate(_METABOLITES):
        for (ratio, t2), group in amp[amp.metabolite.eq(metab)].groupby(
            ["ratio_profile", "t2_profile"], sort=False
        ):
            x = group.snr_pyruvate
            label = f"{ratio} / {t2}"
            axes[0, column].plot(
                x, 100 * group.bias / group.truth, "o-", ms=3, label=label
            )
            axes[1, column].plot(x, group.predicted_to_empirical_std, "o-", ms=3)
            axes[2, column].plot(x, group.uncertainty_available_fraction, "o-", ms=3)
        axes[0, column].set_title(metab.replace("PyruvateHydrate", "Pyruvate hydrate"))
        axes[0, column].axhline(0, color="black", lw=0.7)
        axes[0, column].set_yscale("symlog", linthresh=5)
        axes[1, column].axhline(1, color="black", lw=0.7)
        axes[1, column].set_yscale("log")
        axes[2, column].set_ylim(-0.03, 1.03)
        axes[2, column].set_xlabel("Pyruvate FID SNR")
        for ax in axes[:, column]:
            ax.set_xscale("log")
    axes[0, 0].set_ylabel("Amplitude bias (%)\nall successful fits")
    axes[1, 0].set_ylabel("Predicted / empirical SD\nsame eligible subset")
    axes[2, 0].set_ylabel("Fraction with available\namplitude uncertainty")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(
        f"Five-peak amplitude calibration | {config['repeats']} realizations per condition | 256 complex samples",
        fontsize=16,
    )
    save(fig, "amplitude_statistics")

    first_ratio, first_t2 = (
        next(iter(config["ratio_profiles"])),
        next(iter(config["t2_profiles_ms"])),
    )
    selected = summary[
        summary.ratio_profile.eq(first_ratio) & summary.t2_profile.eq(first_t2)
    ]
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharex=True, layout="constrained")
    for column, metab in enumerate(_METABOLITES):
        for row, parameter in enumerate(("frequency", "t2")):
            group = selected[
                selected.metabolite.eq(metab) & selected.parameter.eq(parameter)
            ]
            ax = axes[row, column]
            for field, label, style in (
                ("empirical_std_all_fits", "Empirical: all fits", ":"),
                ("empirical_std_eligible", "Empirical: eligible", "-"),
                ("predicted_std_rms_eligible", "Predicted RMS: eligible", "--"),
            ):
                ax.plot(
                    group.snr_pyruvate,
                    group[field],
                    style,
                    marker="o",
                    ms=3,
                    label=label,
                )
            ax.set(xscale="log", yscale="log")
            ax.set_ylim(1e-3, 1e2)
            if row == 0:
                ax.set_title(metab.replace("PyruvateHydrate", "Pyruvate hydrate"))
            else:
                ax.set_xlabel("Pyruvate FID SNR")
    axes[0, 0].set_ylabel("Frequency SD (Hz)")
    axes[1, 0].set_ylabel("T2* SD (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(f"Parameter uncertainty | {first_ratio} / {first_t2}", fontsize=16)
    save(fig, "parameter_uncertainty")

    fig, axes = plt.subplots(1, 5, figsize=(19, 4), layout="constrained")
    for ax, metab in zip(axes, _METABOLITES, strict=True):
        group = selected[
            selected.metabolite.eq(metab) & selected.parameter.eq("amplitude")
        ]
        ax.plot(
            group.snr_pyruvate, group.coverage_68_eligible, "o-", label="Within ±1 SD"
        )
        ax.plot(
            group.snr_pyruvate,
            group.coverage_95_eligible,
            "o-",
            label="Within ±1.96 SD",
        )
        ax.axhline(0.6827, color="C0", ls=":")
        ax.axhline(0.95, color="C1", ls=":")
        for snr, count in zip(group.snr_pyruvate, group.n_uncertainty, strict=True):
            ax.text(
                snr,
                0.06,
                str(int(count)),
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
        ax.set(
            xscale="log",
            xlim=(min(config["snr_pyruvate"]) / 1.2, max(config["snr_pyruvate"]) * 1.2),
            ylim=(-0.03, 1.03),
            xlabel="Pyruvate FID SNR",
            title=metab.replace("PyruvateHydrate", "Pyruvate hydrate"),
        )
    axes[0].set_ylabel("Fraction containing the truth\namong eligible amplitude errors")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle(
        f"Observed local-interval coverage | {first_ratio} / {first_t2}\n"
        f"Numbers show eligible realizations out of {config['repeats']}; coverage is selection-dependent",
        fontsize=14,
    )
    save(fig, "coverage")

    snrs = np.array(config["snr_pyruvate"])
    display = np.unique(
        [np.argmin(abs(np.log(snrs / target))) for target in (0.5, 2, 10, 100)]
    )
    fig, axes = plt.subplots(
        len(display),
        1,
        figsize=(13, 3 * len(display)),
        squeeze=False,
        layout="constrained",
    )
    hz = np.fft.fftshift(np.fft.fftfreq(_SAMPLES, config["dwell_ms"] * 1e-3))
    ppm = config["chemical_shifts_ppm"][0] + hz / config["larmor_mhz"]
    for ax, idx in zip(axes[:, 0], display, strict=True):
        for array, label, color in (
            (examples["noisy"][idx], "Noisy spectrum", "0.70"),
            (examples["clean"][idx], "Ground truth", "black"),
            (examples["fitted"][idx], "Fit", "C1"),
        ):
            spectrum = np.fft.fftshift(np.fft.fft(array)) / _SAMPLES
            ax.plot(ppm, spectrum.real, lw=1.2, label=label, color=color)
        for shift in config["chemical_shifts_ppm"]:
            ax.axvline(shift, color="C0", lw=0.6, alpha=0.3)
        ax.set(
            xlim=(186, 158),
            ylabel="Real FFT / N (a.u.)",
            title=f"Pyruvate FID SNR = {snrs[idx]:g}",
        )
    axes[-1, 0].set_xlabel("Chemical shift (ppm; decreasing)")
    axes[-1, 0].margins(y=0.2)
    for shift, name in zip(config["chemical_shifts_ppm"], _METABOLITES, strict=True):
        axes[-1, 0].text(
            shift,
            0.90,
            name.replace("PyruvateHydrate", "Pyruvate hydrate"),
            transform=axes[-1, 0].get_xaxis_transform(),
            ha="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    axes[0, 0].legend(loc="upper left", ncol=3)
    fig.suptitle(
        f"13C example spectra | {first_ratio} / {first_t2} | first realization, no zero filling",
        fontsize=14,
    )
    save(fig, "spectra")


def _main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _configuration(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(f"Output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    tables, examples = [], {"clean": [], "noisy": [], "fitted": []}
    conditions = list(
        product(
            config["ratio_profiles"].items(),
            config["t2_profiles_ms"].items(),
            config["snr_pyruvate"],
        )
    )
    start, fit_seconds, covariance_seconds = perf_counter(), 0.0, 0.0
    for index, ((ratio_name, ratios), (t2_name, t2_ms), snr) in enumerate(conditions):
        data, clean, frequencies, sigma = _simulate(config, ratios, t2_ms, snr, rng)
        params = def_fit_params(data, config["fit_options"])
        tick = perf_counter()
        fit = fit_spectrum(
            data, params, noise=sigma, show_tqdm=False, n_jobs=args.n_jobs
        )
        fit_seconds += perf_counter() - tick
        tick = perf_counter()
        errors = estimate_fit_covariance(fit, params)
        covariance_seconds += perf_counter() - tick
        table = _trial_table(fit, errors, ratios, t2_ms, frequencies, sigma)
        table["ratio_profile"], table["t2_profile"], table["snr_pyruvate"] = (
            ratio_name,
            t2_name,
            snr,
        )
        tables.append(table)
        if index < len(config["snr_pyruvate"]):
            examples["clean"].append(clean)
            examples["noisy"].append(data.isel(realization=0).values)
            reconstructed = reconstruct_fitted_signal(
                fit.isel(realization=0), data.isel(realization=0)
            )
            examples["fitted"].append(reconstructed.values)
        if index == 0:
            config["effective_fit_options"] = {
                key: params.attrs[key]
                for key in (
                    "npoints",
                    "niter",
                    "rep_fitting",
                    "zoomfactor",
                    "min_peak_separation_hz",
                )
            }
        print(
            f"[{index + 1}/{len(conditions)}] {ratio_name}/{t2_name}, SNR={snr:g}: {len(table) // 5} fits; elapsed {perf_counter() - start:.1f}s",
            flush=True,
        )
    trials = pd.concat(tables, ignore_index=True)
    summary = _summarize(trials)
    trials.to_csv(args.output / "trials.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    trials.groupby(
        [
            "ratio_profile",
            "t2_profile",
            "snr_pyruvate",
            "metabolite",
            "fit_status",
            "covariance_status",
            "amplitude_abs_status",
        ],
        dropna=False,
    ).size().rename("count").to_csv(args.output / "statuses.csv")
    np.savez_compressed(
        args.output / "example_spectra.npz",
        **{key: np.asarray(value) for key, value in examples.items()},
        snr_pyruvate=config["snr_pyruvate"],
        time_s=np.arange(_SAMPLES) * args.dwell_ms * 1e-3,
    )
    _plot(summary, examples, config, args.output)
    config["timings_s"] = {
        "fit": fit_seconds,
        "covariance": covariance_seconds,
        "total": perf_counter() - start,
    }
    config["n_spectra_fitted"] = len(conditions) * args.repeats
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Saved {config['n_spectra_fitted']} fits and four figures to {args.output.resolve()}"
    )


if __name__ == "__main__":
    _main()
