"""Per-spectrum numerical status and optional complete FID optimization records."""

from __future__ import annotations

import numpy as np

FIT_STATUSES = ("success", "zero_signal", "skipped", "invalid_input", "fit_failed")
FIT_FAILURE_REASONS = (
    "none",
    "nonfinite_samples",
    "linear_algebra_error",
    "nonfinite_objective",
    "nonfinite_amplitudes",
    "no_feasible_state",
    "inconsistent_final_state",
)

HISTORY_FIELDS = (
    "event",
    "fit_round",
    "repetition",
    "iteration",
    "metabolite_index",
    "candidate_index",
    "accepted",
    "evaluated",
    "rejection_reason",
)
HISTORY_EVENTS = {
    "initial": 0,
    "frequency_candidate": 1,
    "t2_candidate": 2,
    "frequency_step": 3,
    "t2_step": 4,
    "verified": 5,
    "returned": 6,
    "t2_band_leader_candidate": 7,
    "joint_refinement": 8,
}
HISTORY_REASONS = {
    "none": 0,
    "bounds": 1,
    "peak_separation": 2,
    "linear_algebra_error": 3,
    "nonfinite_objective": 4,
}


class FIDNumericalError(RuntimeError):
    """A failed numerical fit, distinct from invalid user configuration."""

    def __init__(self, reason: str):
        super().__init__(f"FID fit failed: {reason}")
        self.reason = reason


def history_capacity(options: dict, n_metabolites: int) -> int:
    """Bound the event count, including one optional legacy amplitude refit.

    Each repetition records the candidates and outer step of the frequency
    and T2* searches and, with ``t2_coupling``, the extra T2* candidates of
    each leader with ratio bands. Each round adds its initial, joint
    refinement and verified states; the refinement slot stays unused padding
    with ``joint_refinement=False``. ``t2_coupling`` may be raw or normalized;
    every entry whose ratio is a sequence counts as a band, so the bound never
    undercounts.

    Parameters
    ----------
    options : dict
        Fit options of one spectrum; reads ``snr_threshold`` (a positive
        value adds the amplitude-filter refit round), ``niter``, ``npoints``,
        ``rep_fitting`` and ``t2_coupling``, with the fit defaults where a key
        is missing. Not modified.
    n_metabolites : int
        Number of fitted lines, at least 1.

    Returns
    -------
    int
        Upper bound on the number of history events of one spectrum, used to
        size the preallocated history arrays.
    """
    rounds = 2 if (options.get("snr_threshold") or 0) > 0 else 1
    niter = int(options.get("niter", 1))
    npoints = int(options.get("npoints", 31))
    candidates = niter * n_metabolites * npoints
    coupling = options.get("t2_coupling") or {}
    band_leaders = {
        str(value[0])
        for value in coupling.values()
        if isinstance(value, (tuple, list)) and len(value) == 2 and np.ndim(value[1])
    }
    per_repetition = 2 * (candidates + 1) + niter * len(band_leaders) * npoints
    return rounds * (3 + int(options.get("rep_fitting", 11)) * per_repetition) + 1


class FitHistory:
    """One private recorder per spectrum; never shared between worker tasks.

    Every evaluated candidate has all active complex amplitudes and both error
    measures. Invalid candidates have NaN amplitudes/RSS and evaluated=False.
    Candidate acceptance is local to its coordinate grid; *_step acceptance
    records the outer joint-state decision. Unused padded rows have event=-1.
    """

    def __init__(self, frequencies, t2s, options):
        self.initial_frequencies = np.array(frequencies, dtype=float, copy=True)
        self.initial_t2s = np.array(t2s, dtype=float, copy=True)
        n_metabolites = len(frequencies)
        size = history_capacity(options, n_metabolites)
        shape = (size, n_metabolites)
        self.amplitudes = np.full(shape, complex(np.nan, np.nan))
        self.frequencies = np.full(shape, np.nan)
        self.t2s = np.full(shape, np.nan)
        self.active = np.zeros(shape, dtype=bool)
        self.rss = np.full(size, np.nan)
        self.objective = np.full(size, np.nan)
        self.info = np.full((size, len(HISTORY_FIELDS)), -1, dtype=np.int32)
        self.count = 0

    def append(
        self,
        frequencies,
        t2s,
        amplitudes,
        rss,
        objective,
        options,
        *,
        event,
        iteration=-1,
        metabolite=-1,
        candidate=-1,
        accepted=False,
        evaluated=True,
        reason="none",
    ):
        """Copy one complete state into the original metabolite coordinate system."""
        index = self.count
        if index >= len(self.info):
            raise RuntimeError("FID history capacity exceeded.")
        active = np.asarray(options.get("_fit_indices", np.arange(len(frequencies))))
        self.frequencies[index] = self.initial_frequencies
        self.t2s[index] = self.initial_t2s
        self.amplitudes[index] = 0
        self.frequencies[index, active] = frequencies
        self.t2s[index, active] = t2s
        self.amplitudes[index, active] = amplitudes
        self.active[index, active] = True
        self.rss[index] = rss
        self.objective[index] = objective
        self.info[index] = (
            HISTORY_EVENTS[event],
            options.get("_fit_round", 0),
            options.get("_fit_repetition", -1),
            iteration,
            active[metabolite] if metabolite >= 0 else -1,
            candidate,
            int(accepted),
            int(evaluated),
            HISTORY_REASONS[reason],
        )
        self.count += 1
        return index

    def accept(self, index):
        """Mark the selected candidate after comparing the entire coordinate grid."""
        self.info[index, HISTORY_FIELDS.index("accepted")] = 1

    def outputs(self):
        """Return padded, fixed-shape outputs for NumPy, joblib and Dask."""
        return (
            np.int64(self.count),
            self.amplitudes,
            self.frequencies,
            self.t2s,
            self.active,
            self.rss,
            self.objective,
            self.info,
        )
