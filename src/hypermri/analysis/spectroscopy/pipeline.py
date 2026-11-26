"""CSI peak detection and staged fitting through existing scientific owners.

Detection is eager, deterministic and independent of widget construction. Its
explicit options preserve spectral-edge noise, clustering and Lorentz linewidth
rules. Fitting, review plotting and quality evaluation remain separate operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.signal import find_peaks, peak_widths

from hypermri.analysis.spectroscopy.frequency import get_freq_axis
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm
from hypermri.analysis.spectroscopy.schema import ensure_standard_spectroscopy_dataarray
from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft

from ...compatibility import resolve_frequency_axis
from ..fitting.models.spectroscopy.fid_model import _require_lorentz_lineshape
from ..fitting.spectroscopy import (
    add_metab_to_fit_params,
    def_fit_params,
    fit_data_pseudo_inv,
    initialise_fit_params_from_results,
)
from .quality import evaluate_spectroscopy_fit_quality

__all__ = [
    "CSIPeakDetectionOptions",
    "detect_csi_peaks",
    "run_csi_auto_fitting_pipeline",
    "run_csi_fitting_pipeline",
]


@dataclass(frozen=True)
class CSIPeakDetectionOptions:
    """Keep the scientific settings of CSI detection independent of its UI.

    Parameters
    ----------
    b0_shift_tolerance_hz : float, optional
        Largest distance from a candidate to a cluster centre in Hz; defaults 15.
        Candidates from one spectrum never join the same cluster.
    peak_height_snr, prominence_snr : float, optional
        Magnitude height and prominence in multiples of robust spectral-edge
        noise; defaults 5 and 3. These are detection thresholds, not fit quality.
    max_peaks_per_spectrum : int, optional
        Candidate limit after ranking by prominence; defaults 4.
    max_peaks : int or None, optional
        Cluster limit ranked by median height SNR times support. None retains
        all clusters. Integer values use the existing Python slice policy;
        interactive callers translate nonpositive limits to None.
    cluster_min_separation_hz : float or None, optional
        Suppress weaker neighbouring clusters closer than this distance in Hz.
        None or a nonpositive distance disables suppression.
    min_peak_distance_hz : float, optional
        Candidate separation within one spectrum in Hz; defaults 8.
    noise_edge_fraction : float, optional
        Fraction of bins at each spectral edge for detection noise; defaults 0.15.
        The existing detector retains its minimum-edge and short-spectrum rules.
    noise_tail_fraction : float, optional
        Fraction of FID samples used by the optional legacy amplitude filter;
        defaults 0.30, with at least eight samples when available.
    min_cluster_support : int or None, optional
        Required number of spectra per cluster. None selects the larger of two
        and the ceiling of 2 percent of all spectra.
    freq_search_range_hz : float, optional
        Minimum derived full frequency search width in Hz; defaults 20.
    min_t2_s, max_t2_s : float, optional
        Lorentz T2-star bounds in seconds; defaults 0.005 and 0.06. Initial T2
        comes from the median linewidth and is clipped to these bounds.
    snr_threshold_factor : float or None, optional
        Optional multiple of temporal tail noise for the legacy amplitude
        threshold, clipped below at zero. None skips temporal noise estimation.
        This threshold is not a trusted fit-quality noise reference.

    Notes
    -----
    The immutable object stores values only. Existing numerical helpers own
    coercion and edge-case behavior; it creates no widgets or data caches.
    Ordinary use requires positive count limits, nonnegative thresholds and
    frequencies, and ordered positive T2 bounds. Wrapper-specific validation and
    legacy warnings remain with their public workflows.

    Examples
    --------
    >>> options = CSIPeakDetectionOptions(b0_shift_tolerance_hz=6, max_peaks=2)
    >>> detected = detect_csi_peaks(data, options)  # doctest: +SKIP
    """

    b0_shift_tolerance_hz: float = 15.0
    peak_height_snr: float = 5.0
    prominence_snr: float = 3.0
    max_peaks_per_spectrum: int = 4
    max_peaks: int | None = None
    cluster_min_separation_hz: float | None = None
    min_peak_distance_hz: float = 8.0
    noise_edge_fraction: float = 0.15
    noise_tail_fraction: float = 0.30
    min_cluster_support: int | None = None
    freq_search_range_hz: float = 20.0
    min_t2_s: float = 0.005
    max_t2_s: float = 0.06
    snr_threshold_factor: float | None = None


_DEFAULT_CSI_PEAK_OPTIONS = CSIPeakDetectionOptions()

_METABOLITE_ATTR_KEYS = (
    "metabs",
    "metabs_freqs_ppm",
    "metabs_freqs_hz",
    "metabs_t2s",
)


def _as_metabolite_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _get_dataset_metabolites(fit_params: xr.Dataset) -> list[str]:
    if "metabolite" not in fit_params.coords:
        raise ValueError("fit_params must contain a 'metabolite' coordinate.")
    return [str(metab) for metab in fit_params.coords["metabolite"].values.tolist()]


def _subset_fit_params_dataset(
    fit_params: xr.Dataset,
    metabolites: Sequence[str],
) -> xr.Dataset:
    source_metabs = _get_dataset_metabolites(fit_params)
    selected_metabs = [str(metab) for metab in metabolites]
    missing_metabs = [metab for metab in selected_metabs if metab not in source_metabs]
    if missing_metabs:
        raise ValueError(
            "Requested metabolites are missing from fit_params: "
            + ", ".join(missing_metabs)
        )

    indices = [source_metabs.index(metab) for metab in selected_metabs]
    subset = fit_params.sel(metabolite=selected_metabs).copy(deep=True)
    subset.attrs["metabs"] = selected_metabs

    for attr_name in _METABOLITE_ATTR_KEYS[1:]:
        attr_value = fit_params.attrs.get(attr_name)
        if attr_value is None:
            continue
        if isinstance(attr_value, np.ndarray):
            attr_value = attr_value.tolist()
        if (
            isinstance(attr_value, Sequence)
            and not isinstance(attr_value, (str, bytes, range))
            and len(attr_value) == len(source_metabs)
        ):
            subset.attrs[attr_name] = [attr_value[index] for index in indices]

    return subset


def _extract_uniform_metabolite_scalar(
    fit_params: xr.Dataset,
    var_name: str,
    metabolite: str,
) -> float:
    if var_name not in fit_params:
        raise KeyError(f"fit_params is missing required variable '{var_name}'.")

    values = np.asarray(
        fit_params[var_name].sel(metabolite=metabolite).values,
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError(
            f"Could not extract '{var_name}' for metabolite '{metabolite}'."
        )

    finite_values = values[np.isfinite(values)]
    reference_value = (
        float(finite_values[0]) if finite_values.size else float(values.reshape(-1)[0])
    )

    if finite_values.size and not np.allclose(finite_values, reference_value):
        raise ValueError(
            f"'{var_name}' for metabolite '{metabolite}' must be spatially uniform "
            "to use run_csi_fitting_pipeline."
        )

    return reference_value


def _resolve_metabolite_scalar(
    *,
    fit_params: xr.Dataset,
    metabolite: str,
    var_name: str,
    fallback: float | dict[str, float],
) -> float:
    dataset_metabolites = _get_dataset_metabolites(fit_params)
    if metabolite in dataset_metabolites:
        return _extract_uniform_metabolite_scalar(fit_params, var_name, metabolite)

    if isinstance(fallback, dict):
        if metabolite in fallback:
            return float(fallback[metabolite])
        if "default" in fallback:
            return float(fallback["default"])
        raise ValueError(
            f"No fallback value provided for metabolite '{metabolite}' and parameter "
            f"'{var_name}'."
        )

    return float(fallback)


def _format_metabolite_fit_ranges(fit_params: xr.Dataset) -> str:
    """Build a compact per-metabolite summary of frequency and T2 ranges."""
    metabs = _get_dataset_metabolites(fit_params)
    parts: list[str] = []
    for metab in metabs:
        freq_range = _extract_uniform_metabolite_scalar(
            fit_params,
            "metabs_freq_ranges",
            metab,
        )
        t2_range = _extract_uniform_metabolite_scalar(
            fit_params,
            "metabs_t2_ranges",
            metab,
        )
        parts.append(
            f"{metab}: freq_range={freq_range:.6g} Hz, t2_range={t2_range:.6g} s"
        )
    return "; ".join(parts)


def _robust_sigma(values: np.ndarray) -> float:
    """Estimate a robust standard deviation via the median absolute deviation."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(mad / 0.6744897501960817) if mad > 0 else 0.0


def _get_frequency_axis_hz(data: xr.DataArray, npoints: int) -> np.ndarray:
    freq_axis_hz = resolve_frequency_axis(data, "Hz")
    if freq_axis_hz is not None:
        freq_axis_hz = np.asarray(freq_axis_hz, dtype=np.float64)
        if freq_axis_hz.shape[0] == npoints:
            return freq_axis_hz

    dwell_time_s = data.attrs.get("dwell_time_s")
    if dwell_time_s is None:
        raise ValueError(
            "Could not determine freq_axis_hz from data. Please provide "
            "a 'freq_axis_hz' coordinate or 'dwell_time_s' in data.attrs."
        )

    return np.asarray(
        get_freq_axis(
            scan=None,
            unit="Hz",
            dwell_time_s=float(dwell_time_s),
            npoints=int(npoints),
        ),
        dtype=np.float64,
    )


def _get_frequency_axis_ppm(
    data: xr.DataArray,
    npoints: int,
) -> np.ndarray | None:
    freq_axis_ppm = resolve_frequency_axis(data, "ppm")
    if freq_axis_ppm is None:
        return None
    freq_axis_ppm = np.asarray(freq_axis_ppm, dtype=np.float64)
    if freq_axis_ppm.shape[0] != npoints:
        return None
    return freq_axis_ppm


def _to_standard_signal_domain(
    data: xr.DataArray,
    target_domain: str,
) -> xr.DataArray:
    standard = ensure_standard_spectroscopy_dataarray(data=data)
    current_domain = str(standard.attrs.get("signal_domain", "temporal")).lower()

    if current_domain == target_domain:
        return standard

    values = np.asarray(standard.values)
    if target_domain == "spectral":
        transformed = spectral_fft(values, axis=0)
    elif target_domain == "temporal":
        transformed = spectral_ifft(values, axis=0)
    else:
        raise ValueError(f"Unsupported target_domain '{target_domain}'.")

    output = standard.copy(data=transformed)
    output.attrs = dict(standard.attrs)
    output.attrs["signal_domain"] = target_domain
    return output


def _flatten_spectra(
    data: xr.DataArray,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    spectral_dim = "acquisition_idx"
    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    aligned = data.transpose(*loop_dims, spectral_dim)
    spectra = np.asarray(aligned.values).reshape(-1, aligned.sizes[spectral_dim])
    spectrum_indices = list(np.ndindex(*(aligned.sizes[dim] for dim in loop_dims)))
    return spectra, spectrum_indices


def _estimate_temporal_noise_std(
    temporal_data: xr.DataArray,
    noise_tail_fraction: float,
) -> float:
    spectra, _ = _flatten_spectra(temporal_data)
    npoints = spectra.shape[-1]
    tail_points = max(8, int(np.ceil(npoints * float(noise_tail_fraction))))
    tail = spectra[:, -tail_points:]
    stacked = np.concatenate([np.real(tail).ravel(), np.imag(tail).ravel()])
    return _robust_sigma(stacked)


def _collect_auto_peak_candidates(
    spectral_data: xr.DataArray,
    *,
    freq_axis_hz: np.ndarray,
    noise_edge_fraction: float,
    peak_height_snr: float,
    prominence_snr: float,
    min_peak_distance_hz: float,
    max_peaks_per_spectrum: int,
) -> tuple[list[dict[str, float | int | tuple[int, ...]]], dict[str, float | int]]:
    spectra, spectrum_indices = _flatten_spectra(spectral_data)
    npoints = spectra.shape[-1]
    edge_points = max(4, int(np.ceil(npoints * float(noise_edge_fraction))))
    if edge_points * 2 >= npoints:
        edge_points = max(2, npoints // 4)

    freq_step_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))
    min_peak_distance_points = max(1, int(np.ceil(min_peak_distance_hz / freq_step_hz)))

    candidates: list[dict[str, float | int | tuple[int, ...]]] = []
    used_spectra = 0
    spectrum_noise_stds: list[float] = []
    for spectrum_id, (spec, spec_index) in enumerate(
        zip(spectra, spectrum_indices, strict=True)
    ):
        spec_mag = np.abs(spec)
        noise_region = np.concatenate([spec_mag[:edge_points], spec_mag[-edge_points:]])
        spectral_noise_std = _robust_sigma(noise_region)
        if spectral_noise_std <= 0:
            continue

        spectrum_noise_stds.append(spectral_noise_std)
        height_threshold = peak_height_snr * spectral_noise_std
        prominence_threshold = prominence_snr * spectral_noise_std
        if np.max(spec_mag) < height_threshold:
            continue

        peak_indices, properties = find_peaks(
            spec_mag,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=min_peak_distance_points,
        )
        if peak_indices.size == 0:
            continue

        order = np.argsort(properties["prominences"])[::-1][:max_peaks_per_spectrum]
        selected_indices = peak_indices[order]
        selected_widths = peak_widths(spec_mag, selected_indices, rel_height=0.5)[0]
        used_spectra += 1

        for peak_index, width_points in zip(
            selected_indices,
            selected_widths,
            strict=True,
        ):
            candidates.append(
                {
                    "spectrum_id": spectrum_id,
                    "spectrum_index": spec_index,
                    "peak_index": int(peak_index),
                    "freq_hz": float(freq_axis_hz[peak_index]),
                    "height": float(spec_mag[peak_index]),
                    "height_snr": float(spec_mag[peak_index] / spectral_noise_std),
                    "width_hz": float(width_points * freq_step_hz),
                }
            )

    diagnostics = {
        "n_spectra_total": int(spectra.shape[0]),
        "n_spectra_with_detected_peaks": int(used_spectra),
        "median_spectral_noise_std": float(np.median(spectrum_noise_stds))
        if spectrum_noise_stds
        else 0.0,
    }
    return candidates, diagnostics


def _cluster_peak_candidates(
    candidates: Sequence[dict[str, float | int | tuple[int, ...]]],
    *,
    tolerance_hz: float,
    min_cluster_support: int,
) -> list[dict[str, object]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: float(item["freq_hz"]))
    clusters: list[dict[str, object]] = []
    for candidate in sorted_candidates:
        candidate_freq = float(candidate["freq_hz"])
        candidate_spectrum_id = int(candidate["spectrum_id"])
        compatible_clusters = []
        for cluster_idx, cluster in enumerate(clusters):
            if abs(candidate_freq - float(cluster["center_hz"])) > tolerance_hz:
                continue
            cluster_spectrum_ids = cluster["spectrum_ids"]
            if candidate_spectrum_id in cluster_spectrum_ids:
                continue
            compatible_clusters.append(
                (cluster_idx, abs(candidate_freq - float(cluster["center_hz"])))
            )

        if compatible_clusters:
            best_cluster_idx = min(compatible_clusters, key=lambda item: item[1])[0]
            cluster = clusters[best_cluster_idx]
            cluster["members"].append(candidate)
            cluster["spectrum_ids"].add(candidate_spectrum_id)
        else:
            clusters.append(
                {
                    "members": [candidate],
                    "spectrum_ids": {candidate_spectrum_id},
                }
            )

        cluster = (
            clusters[-1] if not compatible_clusters else clusters[best_cluster_idx]
        )
        heights = np.asarray(
            [float(member["height_snr"]) for member in cluster["members"]],
            dtype=np.float64,
        )
        freqs = np.asarray(
            [float(member["freq_hz"]) for member in cluster["members"]],
            dtype=np.float64,
        )
        cluster["center_hz"] = float(
            np.average(freqs, weights=np.maximum(heights, 1.0))
        )

    filtered_clusters = []
    for cluster in clusters:
        support = len(cluster["spectrum_ids"])
        if support < min_cluster_support:
            continue
        members = cluster["members"]
        freqs = np.asarray(
            [float(member["freq_hz"]) for member in members], dtype=np.float64
        )
        widths = np.asarray(
            [float(member["width_hz"]) for member in members], dtype=np.float64
        )
        height_snrs = np.asarray(
            [float(member["height_snr"]) for member in members],
            dtype=np.float64,
        )
        filtered_clusters.append(
            {
                "center_hz": float(cluster["center_hz"]),
                "support": int(support),
                "spread_hz": float(np.ptp(freqs)) if freqs.size > 1 else 0.0,
                "median_width_hz": float(np.median(widths)) if widths.size else 0.0,
                "median_height_snr": float(np.median(height_snrs))
                if height_snrs.size
                else 0.0,
                "members": members,
            }
        )

    return sorted(filtered_clusters, key=lambda item: float(item["center_hz"]))


def _cluster_importance_score(cluster: dict[str, object]) -> float:
    return float(cluster["median_height_snr"]) * int(cluster["support"])


def _suppress_close_clusters(
    clusters: Sequence[dict[str, object]],
    *,
    min_separation_hz: float,
) -> list[dict[str, object]]:
    """Keep the strongest cluster when centers are closer than ``min_separation_hz``."""
    if min_separation_hz <= 0 or len(clusters) < 2:
        return list(clusters)

    ranked_indices = sorted(
        range(len(clusters)),
        key=lambda idx: (_cluster_importance_score(clusters[idx]), -idx),
        reverse=True,
    )
    selected_indices: list[int] = []
    for cluster_idx in ranked_indices:
        center_hz = float(clusters[cluster_idx]["center_hz"])
        if any(
            abs(center_hz - float(clusters[kept_idx]["center_hz"])) < min_separation_hz
            for kept_idx in selected_indices
        ):
            continue
        selected_indices.append(cluster_idx)

    return sorted(
        [clusters[idx] for idx in selected_indices],
        key=lambda cluster: float(cluster["center_hz"]),
    )


def _derive_auto_fit_ranges(
    clusters: Sequence[dict[str, object]],
    *,
    min_freq_range_hz: float,
    freq_step_hz: float,
) -> list[float]:
    centers = np.asarray(
        [float(cluster["center_hz"]) for cluster in clusters], dtype=np.float64
    )
    ranges: list[float] = []
    for cluster_idx, cluster in enumerate(clusters):
        observed_spread = float(cluster["spread_hz"])
        base_range = max(min_freq_range_hz, observed_spread * 3.0 + 4.0 * freq_step_hz)

        neighbour_gaps = []
        if cluster_idx > 0:
            neighbour_gaps.append(abs(centers[cluster_idx] - centers[cluster_idx - 1]))
        if cluster_idx < len(clusters) - 1:
            neighbour_gaps.append(abs(centers[cluster_idx + 1] - centers[cluster_idx]))

        if neighbour_gaps:
            base_range = min(
                base_range, max(min_freq_range_hz, 0.8 * min(neighbour_gaps))
            )

        ranges.append(float(base_range))

    return ranges


def _derive_auto_t2_settings(
    clusters: Sequence[dict[str, object]],
    *,
    min_t2_s: float,
    max_t2_s: float,
) -> tuple[list[float], list[float]]:
    initial_t2s: list[float] = []
    t2_ranges: list[float] = []
    for cluster in clusters:
        median_width_hz = max(float(cluster["median_width_hz"]), 1e-6)
        initial_t2 = np.clip(1.0 / (np.pi * median_width_hz), min_t2_s, max_t2_s)
        t2_range = float(
            np.clip(max(0.01, 0.8 * initial_t2), 0.01, max_t2_s - min_t2_s)
        )
        initial_t2s.append(float(initial_t2))
        t2_ranges.append(t2_range)
    return initial_t2s, t2_ranges


def _choose_auto_fit_resolution(
    *,
    max_freq_range_hz: float,
    freq_step_hz: float,
) -> tuple[int, int]:
    bins_spanned = max_freq_range_hz / max(freq_step_hz, 1e-9)
    if bins_spanned <= 8:
        return 2, 31
    if bins_spanned <= 18:
        return 2, 31
    return 2, 51


def _cluster_to_summary(
    clusters: Sequence[dict[str, object]],
    *,
    freq_axis_hz: np.ndarray,
    freq_axis_ppm: np.ndarray | None,
) -> list[dict[str, float | int | str]]:
    summary = []
    for cluster_idx, cluster in enumerate(clusters, start=1):
        center_hz = float(cluster["center_hz"])
        center_ppm = None
        if freq_axis_ppm is not None:
            nearest_idx = int(np.argmin(np.abs(freq_axis_hz - center_hz)))
            center_ppm = float(freq_axis_ppm[nearest_idx])
        summary.append(
            {
                "name": f"peak_{cluster_idx}",
                "center_hz": center_hz,
                "center_ppm": center_ppm if center_ppm is not None else np.nan,
                "support": int(cluster["support"]),
                "spread_hz": float(cluster["spread_hz"]),
                "median_width_hz": float(cluster["median_width_hz"]),
                "median_height_snr": float(cluster["median_height_snr"]),
            }
        )
    return summary


def detect_csi_peaks(
    data: xr.DataArray,
    options: CSIPeakDetectionOptions | None = None,
) -> dict[str, object]:
    """Detect stable CSI peaks and derive initial Lorentz fit ranges without fitting.

    The same operation serves scripting, the automatic fitting pipeline and the
    interactive peak selector. Candidates are detected from magnitude spectra
    and clustered across spectra under the supplied B0 tolerance. Suppression
    and cluster limits precede linewidth and search-range derivation.

    Parameters
    ----------
    data : xarray.DataArray
        Complex temporal or spectral signal in arbitrary dimension order.
        The canonical spectral dimension is acquisition_idx; supported aliases
        and missing singleton dimensions follow the spectroscopy schema.
        signal_domain defaults to temporal. Supply a matching freq_axis_hz
        attribute or dwell_time_s in seconds for shifted FFT bins. Optional
        freq_axis_ppm supplies the corresponding ppm calibration. Signal units
        and labelled coordinates are retained. Acquired storage dtype is not
        changed; domain conversion retains the existing NumPy FFT precision.
    options : CSIPeakDetectionOptions or None, optional
        Explicit scientific settings. None uses the documented domain defaults.
        No fit options, peak selection, manual peaks or widget state are read.

    Returns
    -------
    dict of str to object
        spectral_data is the prepared labelled spectral array in canonical
        dimension order, retaining input coordinates and attributes except its
        signal_domain. It may share samples with spectral input. freq_axis_hz
        and optional freq_axis_ppm are float64 arrays of length acquisition_idx;
        freq_step_hz is the median absolute bin spacing. candidates contains
        spectrum_id, canonical non-spectral spectrum_index, peak_index, freq_hz,
        height in signal units, height_snr and width_hz. peak_detection records
        spectrum counts and median spectral-edge noise in signal units.

        clusters contains the retained centres, support, spread, linewidth, SNR
        and candidate members, ordered by centre Hz. cluster_summary gives names
        peak_1, peak_2, etc., centre Hz/nearest-bin ppm (NaN when unavailable),
        support, spread_hz, median_width_hz and median_height_snr. Parallel lists
        freq_ranges_hz, initial_t2s_s and t2_ranges_s use that same order and Hz
        or seconds. Empty detections return empty lists, allowing callers to
        decide whether manual peaks suffice or absence is an error.

        n_clusters_before_suppression, n_clusters_after_suppression and
        n_clusters_after_filtering describe each stage. temporal_noise_std is
        NaN and snr_threshold is None when tail filtering is disabled; otherwise
        both use the input signal units. options retains the effective immutable
        settings. No fitted parameter, uncertainty or quality decision is made.

    Raises
    ------
    TypeError
        If options is not a CSIPeakDetectionOptions object or None.
    ValueError
        If dimensions cannot be normalized, frequency metadata are unavailable,
        or an underlying detection setting cannot be used by the existing helpers.

    Notes
    -----
    The operation eagerly reads samples and performs the needed domain
    conversions once, without caching or mutating input samples. Frequency
    resolution may copy a frequency axis that exists only as a coordinate into
    the attribute of the same name; existing metadata are retained. It
    creates no widgets or figures and does not reconstruct or fit data. Existing
    spectral-edge and optional FID-tail noise estimates retain their distinct
    roles; neither provides fit-quality eligibility. Prepared spectral samples
    and metadata-derived axis arrays may share input references; callers must
    not mutate them. Candidate/cluster dictionaries and derived lists are new.

    Examples
    --------
    >>> from hypermri.analysis.spectroscopy import (
    ...     CSIPeakDetectionOptions, detect_csi_peaks,
    ... )
    >>> settings = CSIPeakDetectionOptions(max_peaks=2)
    >>> peaks = detect_csi_peaks(csi_data, settings)  # doctest: +SKIP
    >>> centres_hz = [p['center_hz'] for p in peaks['cluster_summary']]  # doctest: +SKIP
    """
    if options is None:
        options = _DEFAULT_CSI_PEAK_OPTIONS
    if not isinstance(options, CSIPeakDetectionOptions):
        raise TypeError(
            "detect_csi_peaks options must be CSIPeakDetectionOptions or None."
        )

    spectral_data, freq_axis_hz, freq_axis_ppm, freq_step_hz = _prepare_csi_peak_data(
        data
    )
    return _detect_csi_peaks_from_spectra(
        data, spectral_data, freq_axis_hz, freq_axis_ppm, freq_step_hz, options
    )


def _prepare_csi_peak_data(
    data: xr.DataArray,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray | None, float]:
    """Prepare the existing spectral domain and its Hz/ppm calibration once."""
    spectral_data = _to_standard_signal_domain(data, "spectral")
    npoints = int(spectral_data.sizes["acquisition_idx"])
    freq_axis_hz = _get_frequency_axis_hz(data, npoints)
    freq_axis_ppm = _get_frequency_axis_ppm(data, npoints)
    freq_step_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))

    return spectral_data, freq_axis_hz, freq_axis_ppm, freq_step_hz


def _detect_csi_peaks_from_spectra(
    data: xr.DataArray,
    spectral_data: xr.DataArray,
    freq_axis_hz: np.ndarray,
    freq_axis_ppm: np.ndarray | None,
    freq_step_hz: float,
    options: CSIPeakDetectionOptions,
) -> dict[str, object]:
    """Run the shared scientific detector on already prepared spectra and axes."""
    temporal_noise_std = np.nan
    snr_threshold = None
    if options.snr_threshold_factor is not None:
        temporal_data = _to_standard_signal_domain(data, "temporal")
        temporal_noise_std = _estimate_temporal_noise_std(
            temporal_data, noise_tail_fraction=options.noise_tail_fraction
        )
        snr_threshold = float(
            max(temporal_noise_std * options.snr_threshold_factor, 0.0)
        )

    candidates, peak_detection_diagnostics = _collect_auto_peak_candidates(
        spectral_data,
        freq_axis_hz=freq_axis_hz,
        noise_edge_fraction=options.noise_edge_fraction,
        peak_height_snr=options.peak_height_snr,
        prominence_snr=options.prominence_snr,
        min_peak_distance_hz=options.min_peak_distance_hz,
        max_peaks_per_spectrum=options.max_peaks_per_spectrum,
    )
    min_cluster_support = options.min_cluster_support
    if min_cluster_support is None:
        min_cluster_support = max(
            2, int(np.ceil(0.02 * peak_detection_diagnostics["n_spectra_total"]))
        )
    clusters = _cluster_peak_candidates(
        candidates,
        tolerance_hz=options.b0_shift_tolerance_hz,
        min_cluster_support=min_cluster_support,
    )
    n_clusters_before_suppression = len(clusters)
    if options.cluster_min_separation_hz is not None:
        clusters = _suppress_close_clusters(
            clusters, min_separation_hz=float(options.cluster_min_separation_hz)
        )
    n_clusters_after_suppression = len(clusters)
    if options.max_peaks is not None:
        clusters = sorted(
            clusters,
            key=lambda item: (
                _cluster_importance_score(item),
                -abs(float(item["center_hz"])),
            ),
            reverse=True,
        )[: int(options.max_peaks)]
        clusters = sorted(clusters, key=lambda item: float(item["center_hz"]))

    freq_ranges_hz = _derive_auto_fit_ranges(
        clusters,
        min_freq_range_hz=options.freq_search_range_hz,
        freq_step_hz=freq_step_hz,
    )
    initial_t2s_s, t2_ranges_s = _derive_auto_t2_settings(
        clusters, min_t2_s=options.min_t2_s, max_t2_s=options.max_t2_s
    )
    cluster_summary = _cluster_to_summary(
        clusters, freq_axis_hz=freq_axis_hz, freq_axis_ppm=freq_axis_ppm
    )
    return {
        "spectral_data": spectral_data,
        "freq_axis_hz": freq_axis_hz,
        "freq_axis_ppm": freq_axis_ppm,
        "freq_step_hz": freq_step_hz,
        "temporal_noise_std": temporal_noise_std,
        "snr_threshold": snr_threshold,
        "candidates": candidates,
        "peak_detection": peak_detection_diagnostics,
        "clusters": clusters,
        "cluster_summary": cluster_summary,
        "freq_ranges_hz": freq_ranges_hz,
        "initial_t2s_s": initial_t2s_s,
        "t2_ranges_s": t2_ranges_s,
        "n_clusters_before_suppression": n_clusters_before_suppression,
        "n_clusters_after_suppression": n_clusters_after_suppression,
        "n_clusters_after_filtering": len(clusters),
        "options": options,
    }


def _resolve_selected_peak_indices(
    selected_peaks: Sequence[int | str] | None,
    *,
    peak_names: Sequence[str],
) -> list[int]:
    if selected_peaks is None:
        return list(range(len(peak_names)))

    if not peak_names:
        raise ValueError("No peaks are available for selection.")

    index_by_name = {
        str(name).strip().lower(): idx for idx, name in enumerate(peak_names)
    }
    resolved_indices: list[int] = []
    for item in selected_peaks:
        from_name = False
        if isinstance(item, (int, np.integer)):
            raw_idx = int(item)
        else:
            token = str(item).strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered in index_by_name:
                raw_idx = index_by_name[lowered]
                from_name = True
            elif token.lstrip("+-").isdigit():
                raw_idx = int(token)
            else:
                raise ValueError(
                    f"Unknown peak selection token '{token}'. "
                    f"Use peak names ({', '.join(peak_names)}) or indices."
                )

        # Accept both 1-based and 0-based integer indexing.
        if from_name:
            idx = raw_idx
        elif 1 <= raw_idx <= len(peak_names):
            idx = raw_idx - 1
        elif 0 <= raw_idx < len(peak_names):
            idx = raw_idx
        else:
            raise ValueError(
                f"Peak index '{raw_idx}' is out of bounds for {len(peak_names)} peaks."
            )

        if idx not in resolved_indices:
            resolved_indices.append(idx)

    if not resolved_indices:
        raise ValueError("Peak selection did not resolve to any detected peaks.")

    return resolved_indices


def _prompt_for_peak_selection(
    cluster_summary: Sequence[dict[str, float | int | str]],
) -> list[int]:
    peak_names = [str(item["name"]) for item in cluster_summary]
    print("CSI auto pipeline: review detected peaks before fitting:")
    for idx, item in enumerate(cluster_summary, start=1):
        ppm_str = (
            f", center_ppm={float(item['center_ppm']):.6g}"
            if np.isfinite(float(item["center_ppm"]))
            else ""
        )
        print(
            f"  [{idx}] {item['name']}: center_hz={float(item['center_hz']):.6g}"
            f"{ppm_str}, support={int(item['support'])}, "
            f"spread_hz={float(item['spread_hz']):.6g}"
        )

    try:
        keep_all = input("CSI auto pipeline: keep all detected peaks? [Y/n]: ")
    except EOFError as exc:
        raise RuntimeError(
            "interactive_peak_selection=True requires an interactive stdin."
        ) from exc
    if keep_all.strip().lower() in {"", "y", "yes"}:
        return list(range(len(cluster_summary)))

    while True:
        try:
            selected = input(
                "CSI auto pipeline: enter peaks to keep (comma-separated indices "
                "or names, e.g. '1,3,peak_4'): "
            )
        except EOFError as exc:
            raise RuntimeError(
                "interactive_peak_selection=True requires an interactive stdin."
            ) from exc

        try:
            return _resolve_selected_peak_indices(
                [token.strip() for token in selected.split(",")],
                peak_names=peak_names,
            )
        except ValueError as exc:
            print(f"CSI auto pipeline: {exc}")


def _choose_peak_review_spectrum_id(
    *,
    candidates: Sequence[dict[str, float | int | tuple[int, ...]]],
    spectra: np.ndarray,
) -> int:
    if spectra.size == 0:
        return 0

    n_spectra = int(spectra.shape[0])
    if candidates:
        scores = np.zeros(n_spectra, dtype=np.float64)
        for candidate in candidates:
            spectrum_id = int(candidate["spectrum_id"])
            if 0 <= spectrum_id < n_spectra:
                scores[spectrum_id] += float(candidate.get("height_snr", 0.0))
        if np.any(scores > 0):
            return int(np.argmax(scores))

    spectrum_max = np.max(np.abs(spectra), axis=1)
    return int(np.argmax(spectrum_max))


def _render_peak_review_plot(
    *,
    spectral_data: xr.DataArray,
    freq_axis_hz: np.ndarray,
    freq_axis_ppm: np.ndarray | None,
    candidates: Sequence[dict[str, float | int | tuple[int, ...]]],
    cluster_summary_all: Sequence[dict[str, float | int | str]],
    selected_peak_indices: Sequence[int],
    spectrum_id: int | None,
    show_plot: bool,
) -> tuple[object | None, dict[str, object]]:
    if not show_plot:
        return None, {}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "show_peak_review_plot=True requested, but matplotlib is not available.",
            stacklevel=2,
        )
        return None, {}

    spectra, spectrum_indices = _flatten_spectra(spectral_data)
    if spectra.size == 0:
        return None, {}

    if spectrum_id is None:
        spectrum_id = _choose_peak_review_spectrum_id(
            candidates=candidates,
            spectra=spectra,
        )
    else:
        spectrum_id = int(spectrum_id)
        if spectrum_id < 0 or spectrum_id >= int(spectra.shape[0]):
            raise ValueError(
                f"peak_review_spectrum_id={spectrum_id} is out of bounds "
                f"for {int(spectra.shape[0])} spectra."
            )

    spectrum_mag = np.abs(spectra[spectrum_id])
    if freq_axis_ppm is not None:
        x_axis = np.asarray(freq_axis_ppm, dtype=np.float64)
        x_label = "Frequency (ppm)"
    else:
        x_axis = np.asarray(freq_axis_hz, dtype=np.float64)
        x_label = "Frequency (Hz)"

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(x_axis, spectrum_mag, color="black", linewidth=1.0, label="Spectrum |S|")

    candidates_in_spectrum = [
        candidate
        for candidate in candidates
        if int(candidate["spectrum_id"]) == spectrum_id
    ]
    if candidates_in_spectrum:
        candidate_indices = np.asarray(
            [int(candidate["peak_index"]) for candidate in candidates_in_spectrum],
            dtype=np.int64,
        )
        ax.scatter(
            x_axis[candidate_indices],
            spectrum_mag[candidate_indices],
            color="0.6",
            s=12,
            marker="o",
            alpha=0.8,
            label="Candidates in shown spectrum",
        )

    selected_set = set(int(idx) for idx in selected_peak_indices)
    y_top = float(np.nanmax(spectrum_mag)) if spectrum_mag.size else 1.0
    for cluster_idx, item in enumerate(cluster_summary_all):
        center_ppm = float(item["center_ppm"])
        center_hz = float(item["center_hz"])
        center_x = (
            center_ppm
            if np.isfinite(center_ppm) and freq_axis_ppm is not None
            else center_hz
        )
        is_selected = cluster_idx in selected_set
        ax.axvline(
            center_x,
            color="tab:green" if is_selected else "tab:orange",
            linewidth=1.8 if is_selected else 1.0,
            linestyle="-" if is_selected else "--",
            alpha=0.9 if is_selected else 0.45,
        )
        ax.text(
            center_x,
            y_top * (0.92 if is_selected else 0.82),
            str(item["name"]),
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
            color="tab:green" if is_selected else "tab:orange",
        )

    ax.set_title("CSI Auto Pipeline Peak Review")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Magnitude [a.u.]")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    plt.show()

    return fig, {
        "spectrum_id": int(spectrum_id),
        "spectrum_index": spectrum_indices[spectrum_id],
        "n_candidates_in_spectrum": int(len(candidates_in_spectrum)),
    }


def run_csi_auto_fitting_pipeline(
    data: xr.DataArray,
    *,
    b0_shift_tolerance_hz: float = _DEFAULT_CSI_PEAK_OPTIONS.b0_shift_tolerance_hz,
    peak_height_snr: float = _DEFAULT_CSI_PEAK_OPTIONS.peak_height_snr,
    prominence_snr: float = _DEFAULT_CSI_PEAK_OPTIONS.prominence_snr,
    max_peaks_per_spectrum: int = _DEFAULT_CSI_PEAK_OPTIONS.max_peaks_per_spectrum,
    max_peaks: int | None = _DEFAULT_CSI_PEAK_OPTIONS.max_peaks,
    cluster_min_separation_hz: float
    | None = _DEFAULT_CSI_PEAK_OPTIONS.cluster_min_separation_hz,
    selected_peaks: Sequence[int | str] | None = None,
    interactive_peak_selection: bool = False,
    show_peak_review_plot: bool = False,
    peak_review_spectrum_id: int | None = None,
    min_peak_distance_hz: float = _DEFAULT_CSI_PEAK_OPTIONS.min_peak_distance_hz,
    noise_edge_fraction: float = _DEFAULT_CSI_PEAK_OPTIONS.noise_edge_fraction,
    noise_tail_fraction: float = _DEFAULT_CSI_PEAK_OPTIONS.noise_tail_fraction,
    min_cluster_support: int | None = _DEFAULT_CSI_PEAK_OPTIONS.min_cluster_support,
    freq_search_range_hz: float = _DEFAULT_CSI_PEAK_OPTIONS.freq_search_range_hz,
    min_t2_s: float = _DEFAULT_CSI_PEAK_OPTIONS.min_t2_s,
    max_t2_s: float = _DEFAULT_CSI_PEAK_OPTIONS.max_t2_s,
    snr_threshold_factor: float | None = _DEFAULT_CSI_PEAK_OPTIONS.snr_threshold_factor,
    final_freq_ranges_hz: float
    | Sequence[float]
    | np.ndarray
    | xr.DataArray
    | None = 5.0,
    final_t2_ranges_s: float
    | Sequence[float]
    | np.ndarray
    | xr.DataArray
    | None = 0.01,
    freeze_other_freqs_during_growth: bool = True,
    freeze_other_t2s_during_growth: bool = False,
    show_tqdm: bool | None = None,
    verbose: bool = True,
    run_quality: bool = False,
    quality_noise_references: xr.Dataset | Sequence[xr.Dataset] | None = None,
    quality_optional_diagnostics: xr.Dataset | None = None,
    quality_config: Mapping[str, float | int] | None = None,
    quality_run_detection: bool = True,
) -> dict[str, object]:
    """Automatically detect CSI peaks, derive fit settings, and run a staged fit.

    The helper estimates noise from the data itself, detects peaks across
    spectra, clusters frequency-shifted instances under a configurable B0
    tolerance, and then uses the resulting cluster centres as initial fit
    frequencies. Peaks that occur in the same spectrum are never merged into the
    same cluster, which helps distinguish true neighbouring metabolites from
    voxel-wise B0 shifts of one underlying peak. Optional post-clustering
    suppression/selection can be used when the automatic detector is too
    permissive for a given dataset. The default ordinary fits use
    ``snr_threshold=None`` and do not estimate quality noise from the FID tail.
    Passing ``snr_threshold_factor`` explicitly enables that legacy amplitude
    filter with a warning. Set ``run_quality=True`` and supply a compatible
    ``quality_noise_references`` contract to append the post-fit product rule.

    The workflow fits the dominant cluster first, adds the remaining clusters
    one at a time and ends with one reduced-range refinement seeded from the
    expanded fit. The input samples are not modified; existing frequency
    resolution can append the documented frequency-range aliases.

    Parameters
    ----------
    data : xr.DataArray
        Complex CSI data in arbitrary labelled dimension order in the temporal
        or spectral domain, normalized to acquisition_idx, slice_idx, read_idx,
        phase_idx, repetition_idx and channel_idx. Supply matching freq_axis_hz
        or dwell_time_s in seconds, and optional corresponding freq_axis_ppm.
        Coordinates, direction, signal units, storage dtype and provenance stay
        with the input; detection uses the existing NumPy FFT precision.
    b0_shift_tolerance_hz : float
        Largest frequency difference, in Hz, between a peak candidate and a
        cluster centre for the candidate to join that cluster.
    peak_height_snr, prominence_snr : float
        Minimum magnitude-spectrum peak height and prominence, in multiples of
        the robust noise standard deviation of the spectrum edges.
    max_peaks_per_spectrum : int
        Largest number of candidates kept per spectrum.
    max_peaks : int, optional
        Keep only this many clusters, ranked by median height SNR times
        support.  All stable clusters are kept when omitted.
    cluster_min_separation_hz : float, optional
        Of two cluster centres closer than this, in Hz, keep only the stronger
        one.  No suppression when omitted.
    selected_peaks : sequence of int or str, optional
        Clusters to fit, given by name (``"peak_1"``, ...) or by 1-based or
        0-based index.  All clusters are fitted when omitted.
    interactive_peak_selection : bool
        Ask on standard input which clusters to fit.
    show_peak_review_plot : bool
        Draw a magnitude spectrum with the candidates and clusters.
    peak_review_spectrum_id : int, optional
        Flattened spectrum index shown in the review plot.  A spectrum with
        many candidates is chosen when omitted.
    min_peak_distance_hz : float
        Minimum distance, in Hz, between two candidates of one spectrum.
    noise_edge_fraction : float
        Fraction of points at each end of a magnitude spectrum used for its
        robust noise estimate during peak detection.
    noise_tail_fraction : float
        Fraction of temporal points at the end of each FID used for the legacy
        noise estimate.  Used only with ``snr_threshold_factor``.
    min_cluster_support : int, optional
        Minimum number of spectra in which a cluster must occur.  Defaults to
        two or 2 % of the spectra, whichever is larger.
    freq_search_range_hz : float
        Minimum frequency search range, in Hz, of each fitted peak.
    min_t2_s, max_t2_s : float
        T2 bounds, in seconds, of every fitted peak.  The initial T2 is derived
        from the median cluster linewidth and clipped to these bounds.
    snr_threshold_factor : float, optional
        Enables the legacy FID-tail amplitude filter with this multiple of the
        tail noise as ``snr_threshold``, with a ``FutureWarning``.  Must be
        finite and non-negative.  Disabled when omitted.
    final_freq_ranges_hz, final_t2_ranges_s : optional
        Search ranges, in Hz and seconds, forwarded to
        :func:`initialise_fit_params_from_results` for the final refinement.
    freeze_other_freqs_during_growth, freeze_other_t2s_during_growth : bool
        Whether already fitted frequencies or T2 values stay fixed while a
        further peak is added.
    show_tqdm : bool, optional
        Forwarded to :func:`fit_data_pseudo_inv`.
    verbose : bool
        Print the detected clusters and short stage messages.
    run_quality : bool
        Append the post-fit Q0/Q1/Q2 evaluation of the final fit, which is
        reused and not repeated.
    quality_noise_references : xr.Dataset or sequence of xr.Dataset, optional
        Trusted noise-contract candidates resolved in scientific priority
        order.  Missing or unusable references produce a blocked Q0 status and
        no M0/M1 simulations.
    quality_optional_diagnostics : xr.Dataset, optional
        Labelled uncertainty, stability, residual or model-order diagnostics.
    quality_config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`~hypermri.analysis.spectroscopy.quality.normalize_spectroscopy_quality_options`.
        Passed unchanged as ``config`` to
        :func:`~hypermri.analysis.spectroscopy.quality.evaluate_spectroscopy_fit_quality`.
    quality_run_detection : bool
        Run still-required full M0/M1 tests when quality evaluation is enabled.

    Returns
    -------
    dict
        ``auto_fit_config`` (derived fit settings), ``auto_diagnostics``
        (detection and clustering summary), the parameters and results of the
        initial, expanded and final stages (``fit_params_initial``,
        ``fit_results_initial``, ``fit_params_expanded``,
        ``fit_results_expanded``, ``fit_params_final``, ``fit_results_final``),
        the aliases ``fit_params`` and ``fit_results`` of the final stage,
        ``peak_review_plot`` (a figure or ``None``) and ``quality_results``
        (``None`` unless ``run_quality=True``).

    Raises
    ------
    ValueError
        If no stable peak cluster is found, ``snr_threshold_factor`` is
        invalid, a peak selection or ``peak_review_spectrum_id`` is invalid,
        or ``quality_config`` contains an unknown or invalid option.  Invalid
        stage options raise the errors of the underlying fits.

    Notes
    -----
    Detection uses the same numerical stage as detect_csi_peaks, once. This workflow eagerly reads
    samples and fits the initial, growing and final models; it does not mutate
    acquired samples or add an input cache. Review figures, terminal prompts and
    quality calculations run only when explicitly requested. No widget controls
    are created. The existing batch defaults, tail-filter warning, selected-peak
    policy and fit-result contracts are unchanged.

    Examples
    --------
    >>> fitted = run_csi_auto_fitting_pipeline(  # doctest: +SKIP
    ...     data, max_peaks=2, show_tqdm=False, verbose=False,
    ... )
    >>> final_results = fitted['fit_results']  # doctest: +SKIP
    """
    spectral_data, freq_axis_hz, freq_axis_ppm, freq_step_hz = _prepare_csi_peak_data(
        data
    )
    if snr_threshold_factor is not None:
        factor = float(snr_threshold_factor)
        if not np.isfinite(factor) or factor < 0:
            raise ValueError("snr_threshold_factor must be finite and non-negative.")
        warnings.warn(
            "snr_threshold_factor enables the legacy FID-tail amplitude filter. "
            "That tail is not a valid fit-quality noise reference; prefer the "
            "trusted-noise quality workflow.",
            FutureWarning,
            stacklevel=2,
        )

    detected = _detect_csi_peaks_from_spectra(
        data,
        spectral_data,
        freq_axis_hz,
        freq_axis_ppm,
        freq_step_hz,
        CSIPeakDetectionOptions(
            b0_shift_tolerance_hz=b0_shift_tolerance_hz,
            peak_height_snr=peak_height_snr,
            prominence_snr=prominence_snr,
            max_peaks_per_spectrum=max_peaks_per_spectrum,
            max_peaks=max_peaks,
            cluster_min_separation_hz=cluster_min_separation_hz,
            min_peak_distance_hz=min_peak_distance_hz,
            noise_edge_fraction=noise_edge_fraction,
            noise_tail_fraction=noise_tail_fraction,
            min_cluster_support=min_cluster_support,
            freq_search_range_hz=freq_search_range_hz,
            min_t2_s=min_t2_s,
            max_t2_s=max_t2_s,
            snr_threshold_factor=(factor if snr_threshold_factor is not None else None),
        ),
    )
    spectral_data = detected["spectral_data"]
    freq_axis_hz = detected["freq_axis_hz"]
    freq_axis_ppm = detected["freq_axis_ppm"]
    freq_step_hz = detected["freq_step_hz"]
    temporal_noise_std = detected["temporal_noise_std"]
    snr_threshold = detected["snr_threshold"]
    candidates = detected["candidates"]
    peak_detection_diagnostics = detected["peak_detection"]
    clusters = detected["clusters"]
    n_clusters_before_suppression = detected["n_clusters_before_suppression"]
    n_clusters_after_suppression = detected["n_clusters_after_suppression"]
    n_clusters_after_filtering = detected["n_clusters_after_filtering"]
    if not clusters:
        raise ValueError(
            "Automatic CSI peak detection did not find any stable peak clusters. "
            "Try lowering peak_height_snr/prominence_snr or increasing the search tolerance."
        )
    freq_ranges_hz = detected["freq_ranges_hz"]
    initial_t2s_s = detected["initial_t2s_s"]
    t2_ranges_s = detected["t2_ranges_s"]
    rep_fitting, npoints_fit = _choose_auto_fit_resolution(
        max_freq_range_hz=max(freq_ranges_hz),
        freq_step_hz=freq_step_hz,
    )
    cluster_summary = detected["cluster_summary"]
    cluster_summary_all = cluster_summary
    peak_names_all = [str(item["name"]) for item in cluster_summary]
    selected_peak_indices = _resolve_selected_peak_indices(
        selected_peaks,
        peak_names=peak_names_all,
    )
    peak_review_plot, peak_review_plot_info = _render_peak_review_plot(
        spectral_data=spectral_data,
        freq_axis_hz=freq_axis_hz,
        freq_axis_ppm=freq_axis_ppm,
        candidates=candidates,
        cluster_summary_all=cluster_summary_all,
        selected_peak_indices=selected_peak_indices,
        spectrum_id=peak_review_spectrum_id,
        show_plot=show_peak_review_plot,
    )
    if interactive_peak_selection:
        selected_peak_indices = _prompt_for_peak_selection(cluster_summary)

    cluster_summary = [cluster_summary[idx] for idx in selected_peak_indices]
    clusters = [clusters[idx] for idx in selected_peak_indices]
    freq_ranges_hz = [freq_ranges_hz[idx] for idx in selected_peak_indices]
    initial_t2s_s = [initial_t2s_s[idx] for idx in selected_peak_indices]
    t2_ranges_s = [t2_ranges_s[idx] for idx in selected_peak_indices]

    peak_names = [str(item["name"]) for item in cluster_summary]
    cluster_centers_hz = [float(item["center_hz"]) for item in cluster_summary]

    auto_fit_params = {
        "metabs": peak_names,
        "metabs_freqs_hz": cluster_centers_hz,
        "metabs_freq_ranges": freq_ranges_hz,
        "initial_t2s_s": initial_t2s_s,
        "metabs_t2_ranges": t2_ranges_s,
        "min_t2_s": [float(min_t2_s)] * len(peak_names),
        "max_t2_s": [float(max_t2_s)] * len(peak_names),
        "rep_fitting": rep_fitting,
        "npoints": npoints_fit,
        "snr_threshold": snr_threshold,
        "show_tqdm": show_tqdm if show_tqdm is not None else False,
    }

    dominant_peak_idx = int(
        np.argmax([_cluster_importance_score(cluster) for cluster in clusters])
    )
    peak_order = [dominant_peak_idx] + [
        idx for idx in range(len(peak_names)) if idx != dominant_peak_idx
    ]

    if verbose:
        if snr_threshold is None:
            print("CSI auto pipeline: legacy FID-tail amplitude filtering is disabled.")
        else:
            print(
                "CSI auto pipeline: legacy estimated temporal noise std="
                f"{temporal_noise_std:.6g}, snr_threshold={snr_threshold:.6g}."
            )
        print(
            "CSI auto pipeline: detected "
            f"{len(candidates)} peak candidates and {n_clusters_after_filtering} stable clusters."
        )
        if cluster_min_separation_hz is not None:
            print(
                "CSI auto pipeline: cluster suppression kept "
                f"{n_clusters_after_suppression} / {n_clusters_before_suppression} clusters "
                f"(min_separation_hz={float(cluster_min_separation_hz):.6g})."
            )
        if selected_peak_indices != list(range(len(peak_names_all))):
            selected_labels = ", ".join(peak_names)
            print(f"CSI auto pipeline: selected peaks: {selected_labels}.")
        if show_peak_review_plot and peak_review_plot_info:
            print(
                "CSI auto pipeline: peak review plot uses spectrum_id="
                f"{peak_review_plot_info['spectrum_id']} "
                f"(candidates_in_spectrum={peak_review_plot_info['n_candidates_in_spectrum']})."
            )
        for item, freq_range_hz, initial_t2_s, t2_range_s in zip(
            cluster_summary,
            freq_ranges_hz,
            initial_t2s_s,
            t2_ranges_s,
            strict=True,
        ):
            ppm_str = (
                f", center_ppm={float(item['center_ppm']):.6g}"
                if np.isfinite(float(item["center_ppm"]))
                else ""
            )
            print(
                "  "
                f"{item['name']}: center_hz={float(item['center_hz']):.6g}{ppm_str}, "
                f"support={int(item['support'])}, spread_hz={float(item['spread_hz']):.6g}, "
                f"freq_range={freq_range_hz:.6g} Hz, initial_t2={initial_t2_s:.6g} s, "
                f"t2_range={t2_range_s:.6g} s"
            )
        print(
            "CSI auto pipeline: selected rep_fitting="
            f"{rep_fitting}, npoints={npoints_fit}."
        )

    initial_idx = peak_order[0]
    fit_params_initial = def_fit_params(
        data=data,
        fit_params={
            "metabs": [peak_names[initial_idx]],
            "metabs_freqs_hz": [cluster_centers_hz[initial_idx]],
            "metabs_freq_ranges": [freq_ranges_hz[initial_idx]],
            "initial_t2s_s": [initial_t2s_s[initial_idx]],
            "metabs_t2_ranges": [t2_ranges_s[initial_idx]],
            "min_t2_s": [float(min_t2_s)],
            "max_t2_s": [float(max_t2_s)],
            "rep_fitting": rep_fitting,
            "npoints": npoints_fit,
            "snr_threshold": snr_threshold,
            "show_tqdm": show_tqdm if show_tqdm is not None else False,
        },
    )

    if verbose:
        print(f"CSI auto pipeline: fitting dominant peak '{peak_names[initial_idx]}'.")
    fit_results_current = fit_data_pseudo_inv(
        data=data,
        fit_params=fit_params_initial,
        show_tqdm=show_tqdm,
    )
    fit_results_initial = fit_results_current
    fit_params_current = fit_params_initial

    for peak_idx in peak_order[1:]:
        if verbose:
            print(
                "CSI auto pipeline: adding peak "
                f"'{peak_names[peak_idx]}' at {cluster_centers_hz[peak_idx]:.6g} Hz "
                f"(freq_range={freq_ranges_hz[peak_idx]:.6g} Hz, "
                f"initial_t2={initial_t2s_s[peak_idx]:.6g} s)."
            )
        fit_params_current = add_metab_to_fit_params(
            new_metab_name=peak_names[peak_idx],
            new_freq=cluster_centers_hz[peak_idx],
            new_t2_s=initial_t2s_s[peak_idx],
            fit_params=fit_params_current,
            fit_results=fit_results_current,
            new_freq_range=freq_ranges_hz[peak_idx],
            new_t2_range_s=t2_ranges_s[peak_idx],
            new_min_t2_s=min_t2_s,
            new_max_t2_s=max_t2_s,
            new_freq_unit="Hz",
            freeze_other_freqs=freeze_other_freqs_during_growth,
            freeze_other_t2s=freeze_other_t2s_during_growth,
        )
        fit_results_current = fit_data_pseudo_inv(
            data=data,
            fit_params=fit_params_current,
            show_tqdm=show_tqdm,
        )

    fit_params_refined = initialise_fit_params_from_results(
        fit_params=fit_params_current,
        fit_results=fit_results_current,
        freq_ranges_hz=final_freq_ranges_hz,
        t2_ranges_s=final_t2_ranges_s,
    )
    if verbose:
        print(
            "CSI auto pipeline: running final refinement "
            f"(freq_ranges_hz={final_freq_ranges_hz}, t2_ranges_s={final_t2_ranges_s})."
        )
    fit_results_final = fit_data_pseudo_inv(
        data=data,
        fit_params=fit_params_refined,
        show_tqdm=show_tqdm,
    )
    if verbose:
        print("CSI auto pipeline: done.")

    quality_results = None
    if run_quality:
        quality_results = evaluate_spectroscopy_fit_quality(
            data,
            fit_params_refined,
            quality_noise_references,
            fit_results=fit_results_final,
            optional_diagnostics=quality_optional_diagnostics,
            config=quality_config,
            run_detection=quality_run_detection,
            show_tqdm=bool(show_tqdm),
        )

    diagnostics = {
        "temporal_noise_std": temporal_noise_std,
        "snr_threshold": snr_threshold,
        "legacy_fid_tail_noise_used": snr_threshold_factor is not None,
        "peak_detection": peak_detection_diagnostics,
        "cluster_summary": cluster_summary,
        "cluster_summary_all": cluster_summary_all,
        "n_clusters_before_suppression": n_clusters_before_suppression,
        "n_clusters_after_suppression": n_clusters_after_suppression,
        "n_clusters_after_filtering": n_clusters_after_filtering,
        "n_clusters_after_selection": int(len(cluster_summary)),
        "selected_peak_indices": [int(idx) for idx in selected_peak_indices],
        "selected_peak_names": [str(name) for name in peak_names],
        "peak_review_plot_info": peak_review_plot_info,
        "rep_fitting": rep_fitting,
        "npoints": npoints_fit,
        "b0_shift_tolerance_hz": float(b0_shift_tolerance_hz),
    }
    return {
        "auto_fit_config": auto_fit_params,
        "auto_diagnostics": diagnostics,
        "fit_params_initial": fit_params_initial,
        "fit_results_initial": fit_results_initial,
        "fit_params_expanded": fit_params_current,
        "fit_results_expanded": fit_results_current,
        "peak_review_plot": peak_review_plot,
        "fit_params_final": fit_params_refined,
        "fit_results_final": fit_results_final,
        "fit_params": fit_params_refined,
        "fit_results": fit_results_final,
        "quality_results": quality_results,
    }


def _normalize_per_metabolite_option(
    value: bool | Sequence[bool],
    metabolites: Sequence[str],
    name: str,
) -> list[bool]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = [bool(item) for item in value]
        if len(normalized) != len(metabolites):
            raise ValueError(
                f"{name} must have length {len(metabolites)} to match metabolites_to_add."
            )
        return normalized

    return [bool(value)] * len(metabolites)


def _get_freq_offset_ppm(
    *,
    metabolite: str,
    reference_metabolite: str,
) -> float:
    metabolite_shift = get_metab_cs_ppm(metabolite)
    reference_shift = get_metab_cs_ppm(reference_metabolite)
    if metabolite_shift is None or reference_shift is None:
        raise ValueError(
            "Could not determine chemical shifts for metabolite addition. "
            f"metabolite='{metabolite}', reference_metabolite='{reference_metabolite}'."
        )
    return float(metabolite_shift - reference_shift)


def run_csi_fitting_pipeline(
    data: xr.DataArray,
    fit_params: dict | xr.Dataset,
    initial_metabolite: str | None = None,
    metabolites_to_add: Sequence[str] | None = None,
    reference_metabolite: str | None = None,
    freeze_other_freqs: bool | Sequence[bool] = True,
    freeze_other_t2s: bool | Sequence[bool] = False,
    reference_repetitions: int | slice | range | Sequence[int] | str | None = None,
    final_freq_ranges_hz: float
    | Sequence[float]
    | np.ndarray
    | xr.DataArray
    | None = 5.0,
    final_t2_ranges_s: float
    | Sequence[float]
    | np.ndarray
    | xr.DataArray
    | None = 0.01,
    added_metabolite_initial_t2_s: float | dict[str, float] = 0.02,
    added_metabolite_freq_range_hz: float | dict[str, float] = 30.0,
    added_metabolite_t2_range_s: float | dict[str, float] = 0.04,
    added_metabolite_min_t2_s: float | dict[str, float] = 0.005,
    added_metabolite_max_t2_s: float | dict[str, float] = 0.1,
    run_final_refinement: bool = True,
    show_tqdm: bool | None = None,
    verbose: bool = True,
    run_quality: bool = False,
    quality_noise_references: xr.Dataset | Sequence[xr.Dataset] | None = None,
    quality_optional_diagnostics: xr.Dataset | None = None,
    quality_config: Mapping[str, float | int] | None = None,
    quality_run_detection: bool = True,
) -> dict[str, xr.Dataset]:
    """Run a CSI fitting workflow with sequential metabolite expansion.

    The workflow is:

    1. fit a single reference metabolite,
    2. add further metabolites using that fit as the reference for initial
       frequencies,
    3. refit the expanded metabolite set,
    4. optionally run one final reduced-range refinement seeded from the
       expanded-fit results.

    Parameters
    ----------
    data : xr.DataArray
        CSI data to fit.
    fit_params : dict or xr.Dataset
        Full metabolite configuration. If a dict is passed it is first converted
        with :func:`def_fit_params`.
    initial_metabolite : str, optional
        Metabolite to fit first. Defaults to the first metabolite in
        ``fit_params``.
    metabolites_to_add : sequence of str, optional
        Remaining metabolites to add after the initial fit. Defaults to all
        metabolites from ``fit_params`` except ``initial_metabolite``.
    reference_metabolite : str, optional
        Metabolite used as the frequency reference when adding further peaks.
        This must be present in the initial fit and therefore defaults to
        ``initial_metabolite``.
    freeze_other_freqs, freeze_other_t2s : bool or sequence of bool, optional
        Flags forwarded to :func:`add_metab_to_fit_params` for each added
        metabolite. Scalars apply to all additions.
    reference_repetitions : int, slice, range, sequence, str, optional
        Forwarded to :func:`add_metab_to_fit_params`.
    final_freq_ranges_hz, final_t2_ranges_s : optional
        Search ranges forwarded to
        :func:`initialise_fit_params_from_results` for the final refinement.
    added_metabolite_initial_t2_s, added_metabolite_freq_range_hz,
    added_metabolite_t2_range_s, added_metabolite_min_t2_s,
    added_metabolite_max_t2_s : float or dict, optional
        Fallback parameters used when a metabolite in ``metabolites_to_add`` is
        not already present in ``fit_params``. Scalars apply to every missing
        metabolite; dicts can provide per-metabolite overrides and may include a
        ``"default"`` entry.
    run_final_refinement : bool, optional
        If True, run the final reduced-range refinement. Default is True.
    show_tqdm : bool, optional
        Forwarded to :func:`fit_data_pseudo_inv`.
    verbose : bool, optional
        If True, print short stage messages before and after each major fitting
        step. Default is True.
    run_quality : bool, optional
        Append the post-fit Q0/Q1/Q2 evaluation. Default is False. The supplied
        final M1 fit is preserved and is not repeated.
    quality_noise_references : xr.Dataset or sequence of xr.Dataset, optional
        Trusted noise-contract candidates resolved in scientific priority order.
        Missing or unusable references produce a blocked Q0 status and no M0/M1
        simulations.
    quality_optional_diagnostics : xr.Dataset, optional
        Labelled uncertainty, stability, residual or model-order diagnostics.
    quality_config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`~hypermri.analysis.spectroscopy.quality.normalize_spectroscopy_quality_options`.
        Passed unchanged as ``config`` to
        :func:`~hypermri.analysis.spectroscopy.quality.evaluate_spectroscopy_fit_quality`.
    quality_run_detection : bool, optional
        Run still-required full M0/M1 tests when quality evaluation is enabled.

    Returns
    -------
    dict
        Dictionary containing the initial, expanded, and final fit parameters
        and fit results. ``fit_params`` and ``fit_results`` are aliases for the
        final stage.

    Raises
    ------
    ValueError
        If ``fit_params`` defines no metabolites or the metabolite selection is
        inconsistent, or if ``run_quality=True`` is combined with a
        ``lineshape='voigt'`` fit, whose quality evaluation is not supported.
        Invalid stage options raise the errors of the underlying fits.
    """
    full_fit_params = (
        fit_params.copy(deep=True)
        if isinstance(fit_params, xr.Dataset)
        else def_fit_params(data=data, fit_params=deepcopy(fit_params))
    )

    all_metabolites = _get_dataset_metabolites(full_fit_params)
    if not all_metabolites:
        raise ValueError("fit_params must define at least one metabolite.")
    if run_quality:
        _require_lorentz_lineshape(
            "run_csi_fitting_pipeline(run_quality=True)", full_fit_params
        )

    initial_metabolite = str(initial_metabolite or all_metabolites[0])
    if initial_metabolite not in all_metabolites:
        raise ValueError(
            f"initial_metabolite '{initial_metabolite}' is not present in fit_params."
        )

    if metabolites_to_add is None:
        metabolites_to_add = [
            metab for metab in all_metabolites if metab != initial_metabolite
        ]
    else:
        metabolites_to_add = _as_metabolite_list(metabolites_to_add)

    if initial_metabolite in metabolites_to_add:
        raise ValueError("initial_metabolite must not appear in metabolites_to_add.")

    duplicate_metabs = {
        metab for metab in metabolites_to_add if metabolites_to_add.count(metab) > 1
    }
    if duplicate_metabs:
        raise ValueError(
            "metabolites_to_add contains duplicates: "
            + ", ".join(sorted(duplicate_metabs))
        )

    reference_metabolite = str(reference_metabolite or initial_metabolite)
    if reference_metabolite != initial_metabolite:
        raise ValueError(
            "reference_metabolite must currently be the initial metabolite, "
            "because additional peaks are seeded from the initial fit results."
        )

    freeze_other_freqs_list = _normalize_per_metabolite_option(
        freeze_other_freqs,
        metabolites_to_add,
        "freeze_other_freqs",
    )
    freeze_other_t2s_list = _normalize_per_metabolite_option(
        freeze_other_t2s,
        metabolites_to_add,
        "freeze_other_t2s",
    )

    fit_params_initial = _subset_fit_params_dataset(
        full_fit_params, [initial_metabolite]
    )
    if verbose:
        print(
            "CSI pipeline: fitting initial metabolite "
            f"'{initial_metabolite}' "
            f"({_format_metabolite_fit_ranges(fit_params_initial)})."
        )
    fit_results_initial = fit_data_pseudo_inv(
        data=data,
        fit_params=fit_params_initial,
        show_tqdm=show_tqdm,
    )
    if verbose:
        print(f"CSI pipeline: initial fit complete for '{initial_metabolite}'.")

    fit_params_expanded = fit_params_initial
    for metab_name, freeze_freqs, freeze_t2s in zip(
        metabolites_to_add,
        freeze_other_freqs_list,
        freeze_other_t2s_list,
        strict=True,
    ):
        if verbose:
            new_t2_s = _resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="initial_t2s_s",
                fallback=added_metabolite_initial_t2_s,
            )
            new_freq_range = _resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="metabs_freq_ranges",
                fallback=added_metabolite_freq_range_hz,
            )
            new_t2_range_s = _resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="metabs_t2_ranges",
                fallback=added_metabolite_t2_range_s,
            )
            new_min_t2 = _resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="min_t2_s",
                fallback=added_metabolite_min_t2_s,
            )
            new_max_t2 = _resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="max_t2_s",
                fallback=added_metabolite_max_t2_s,
            )
            print(
                "CSI pipeline: adding metabolite "
                f"'{metab_name}' relative to '{reference_metabolite}' "
                f"(initial_t2={new_t2_s:.6g} s, "
                f"freq_range={new_freq_range:.6g} Hz, "
                f"t2_range={new_t2_range_s:.6g} s, "
                f"min_t2={new_min_t2:.6g} s, "
                f"max_t2={new_max_t2:.6g} s, "
                f"freq_offset={_get_freq_offset_ppm(metabolite=metab_name, reference_metabolite=reference_metabolite):.6g} ppm, "
                f"(freeze_other_freqs={freeze_freqs}, "
                f"freeze_other_t2s={freeze_t2s}))."
            )
        fit_params_expanded = add_metab_to_fit_params(
            new_metab_name=metab_name,
            reference_metabolite=reference_metabolite,
            new_freq=None,
            new_freq_unit="Hz",
            new_t2_s=_resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="initial_t2s_s",
                fallback=added_metabolite_initial_t2_s,
            ),
            new_freq_range=_resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="metabs_freq_ranges",
                fallback=added_metabolite_freq_range_hz,
            ),
            new_t2_range_s=_resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="metabs_t2_ranges",
                fallback=added_metabolite_t2_range_s,
            ),
            new_min_t2_s=_resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="min_t2_s",
                fallback=added_metabolite_min_t2_s,
            ),
            new_max_t2_s=_resolve_metabolite_scalar(
                fit_params=full_fit_params,
                metabolite=metab_name,
                var_name="max_t2_s",
                fallback=added_metabolite_max_t2_s,
            ),
            fit_params=fit_params_expanded,
            fit_results=fit_results_initial,
            freq_offset=_get_freq_offset_ppm(
                metabolite=metab_name,
                reference_metabolite=reference_metabolite,
            ),
            freq_offset_unit="ppm",
            reference_repetitions=reference_repetitions,
            freeze_other_freqs=freeze_freqs,
            freeze_other_t2s=freeze_t2s,
        )

    if verbose:
        expanded_metabolites = _get_dataset_metabolites(fit_params_expanded)
        print(
            "CSI pipeline: fitting expanded metabolite set "
            f"{expanded_metabolites} "
            f"with ranges: {_format_metabolite_fit_ranges(fit_params_expanded)}."
        )
    fit_results_expanded = fit_data_pseudo_inv(
        data=data,
        fit_params=fit_params_expanded,
        show_tqdm=show_tqdm,
    )
    if verbose:
        print("CSI pipeline: expanded fit complete.")

    fit_params_final = fit_params_expanded
    fit_results_final = fit_results_expanded
    if run_final_refinement:
        if verbose:
            print(
                "CSI pipeline: starting final refinement "
                f"(freq_ranges_hz={final_freq_ranges_hz}, "
                f"t2_ranges_s={final_t2_ranges_s}) "
                f"from: {_format_metabolite_fit_ranges(fit_params_expanded)}."
            )
        fit_params_final = initialise_fit_params_from_results(
            fit_params=fit_params_expanded,
            fit_results=fit_results_expanded,
            freq_ranges_hz=final_freq_ranges_hz,
            t2_ranges_s=final_t2_ranges_s,
        )
        fit_results_final = fit_data_pseudo_inv(
            data=data,
            fit_params=fit_params_final,
            show_tqdm=show_tqdm,
        )
        if verbose:
            print("CSI pipeline: final refinement complete.")
    elif verbose:
        print("CSI pipeline: final refinement skipped.")

    if verbose:
        print("CSI pipeline: done.")
    result = {
        "fit_params_initial": fit_params_initial,
        "fit_results_initial": fit_results_initial,
        "fit_params_expanded": fit_params_expanded,
        "fit_results_expanded": fit_results_expanded,
        "fit_params_final": fit_params_final,
        "fit_results_final": fit_results_final,
        "fit_params": fit_params_final,
        "fit_results": fit_results_final,
    }
    if run_quality:
        result["quality_results"] = evaluate_spectroscopy_fit_quality(
            data,
            fit_params_final,
            quality_noise_references,
            fit_results=fit_results_final,
            optional_diagnostics=quality_optional_diagnostics,
            config=quality_config,
            run_detection=quality_run_detection,
            show_tqdm=bool(show_tqdm),
        )
    return result
