"""Immutable supplied settings for quantitative map fitting.

None leaves a setting to its existing scientific model owner. These containers
introduce no numerical defaults, perform no fitting and retain no result cache.
T1 settings include supported weighting, bounds and regularization; IDEAL settings
include active transient initialization and convergence options. Use a prepared
Dataset for other existing advanced map-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from hypermri.analysis.fitting.execution import ExecutionConfig


def _freeze_sequences(options, names):
    for name in names:
        value = getattr(options, name)
        if value is not None:
            object.__setattr__(options, name, tuple(value))


@dataclass(frozen=True)
class T1FitOptions:
    """Supply inversion-recovery settings without replacing omitted defaults.

    Parameters
    ----------
    t1_values_s : tuple of float or None, optional
        Trial positive relaxation times in seconds; fitted T1 is reported in ms.
        None retains the model's existing search grid.
    fit_baseline : bool or None, optional
        Whether to fit the additive signal baseline; None retains the model default.
    dynamic_dim : str or None, optional
        Sample dimension whose coordinate supplies inversion times in ms.
    execution : ExecutionConfig or None, optional
        Existing shared worker settings; None retains the executor defaults.
    weight_by_signal : bool or None, optional
        Weight squared residuals by abs(signal)/(max(abs(signal))+1e-8).
        This dimensionless signal-magnitude heuristic is not a noise likelihood.
        None retains unweighted fitting.
    bounds_a, bounds_b : tuple of float or None, optional
        Strictly ordered lower/upper coefficient bounds in input signal units;
        infinities are allowed. Complex real/imaginary coefficients are bounded
        separately. B is fitted only with ``fit_baseline=True``. None retains
        unbounded coefficients unless relative/automatic bounds are supplied.
    rel_bounds_a, rel_bounds_b : float or tuple of float or None, optional
        Dimensionless relative bounds scaled by each voxel's max(abs(signal)).
        A scalar r gives amplitude factors (1-r, 1+r) or baseline factors (-r, r);
        a pair supplies factors directly. Override respective absolute bounds.
    auto_calc_rel_bound : bool or None, optional
        Use the existing inversion-recovery bound heuristic, overriding supplied
        relative bounds. None retains False. The heuristic uses the longest
        inversion time, the largest trial T1 and ``inversion_imperfection``.
    inversion_imperfection : float or None, optional
        Dimensionless imperfection setting for automatic bounds; None retains 0.1.
    regularization_alpha : float or None, optional
        Dimensionless ridge penalty on all fitted coefficients when positive and
        all bounds are unbounded. Any finite bound selects bounded least squares
        instead. Non-positive values retain ordinary least squares; None retains
        0.0. Reported RSS excludes the penalty.

    Notes
    -----
    Fields are immutable; supplied sequences are copied to tuples. Construction
    fits no data and creates no cache. Scientific validation remains in the model.
    The model retains its original grid search and coefficient solver. Preparation
    forwards these supplied settings once and records them in fit provenance.
    Lowercase coefficient fields translate to the established mapping keys
    ``bounds_A``, ``bounds_B``, ``rel_bounds_A`` and ``rel_bounds_B``.

    Examples
    --------
    >>> options = T1FitOptions(t1_values_s=(0.5, 0.7, 0.9))
    """

    t1_values_s: tuple[float, ...] | None = None
    fit_baseline: bool | None = None
    dynamic_dim: str | None = None
    execution: ExecutionConfig | None = None
    weight_by_signal: bool | None = None
    bounds_a: tuple[float, float] | None = None
    bounds_b: tuple[float, float] | None = None
    rel_bounds_a: float | tuple[float, float] | None = None
    rel_bounds_b: float | tuple[float, float] | None = None
    auto_calc_rel_bound: bool | None = None
    inversion_imperfection: float | None = None
    regularization_alpha: float | None = None

    def __post_init__(self):
        _freeze_sequences(self, ("t1_values_s", "bounds_a", "bounds_b"))
        for name in ("rel_bounds_a", "rel_bounds_b"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, int | float):
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class ADCFitOptions:
    """Supply settings for magnitude log-linear diffusion fitting.

    Parameters
    ----------
    b_values_s_per_mm2 : tuple of float or None, optional
        Diffusion weighting in s/mm² for legacy preparation metadata. The sample
        dimension coordinate remains the numerical axis during voxel execution.
    signal_threshold : float or None, optional
        Minimum magnitude in input signal units; samples at/below it are excluded.
        None retains the existing scientific threshold.
    dynamic_dim : str or None, optional
        Sample dimension whose coordinate supplies diffusion weighting in s/mm².
    execution : ExecutionConfig or None, optional
        Existing shared worker settings; None retains the executor defaults.

    Notes
    -----
    Frozen supplied settings, with sequence values copied to tuples. Preparation
    and fitting are separate; construction mutates no input and retains no cache.
    The existing model handles invalid or insufficient samples.

    Examples
    --------
    >>> options = ADCFitOptions(signal_threshold=1e-5)
    """

    b_values_s_per_mm2: tuple[float, ...] | None = None
    signal_threshold: float | None = None
    dynamic_dim: str | None = None
    execution: ExecutionConfig | None = None

    def __post_init__(self):
        _freeze_sequences(self, ("b_values_s_per_mm2",))


@dataclass(frozen=True)
class T2FitOptions:
    """Supply settings for monoexponential magnitude echo-decay fitting.

    Parameters
    ----------
    signal_threshold : float or None, optional
        Include finite magnitudes strictly above this threshold in input signal
        units. None uses the model default 1e-5. Must be finite and nonnegative.
    dynamic_dim : str or None, optional
        Echo dimension with a one-dimensional coordinate in seconds. None uses
        ``"echo_idx"``. Scan fitting requires that standard dimension.
    execution : ExecutionConfig or None, optional
        Shared voxel worker settings; None preserves executor defaults.

    Notes
    -----
    Frozen settings; construction reads no data and creates no cache. Validation
    occurs before fitting. The approved model uses unweighted nonlinear least
    squares in magnitude signal space, no offset, all echoes, at least three
    valid echoes, S0 >= 0 and 0 < T2 <= 5 s. S0 has no finite upper bound.
    The map is ms (maximum 5000 ms); internal time is seconds. No uncertainty
    is calculated. The T2 upper bound is fixed by the numerical model.

    Examples
    --------
    >>> options = T2FitOptions(signal_threshold=1e-5)
    """

    signal_threshold: float | None = None
    dynamic_dim: str | None = None
    execution: ExecutionConfig | None = None


@dataclass(frozen=True)
class B1FitOptions:
    """Supply existing RF-power sweep settings for sinusoidal B1 fitting.

    Parameters
    ----------
    fa_deg : float or None, optional
        Pulse flip angle in degrees used for absolute reference-power calibration.
    trf_ms : float or None, optional
        RF pulse duration in ms for that calibration.
    sint : float or None, optional
        Dimensionless RF pulse integration factor.
    period_min_sqrt_w, period_max_sqrt_w : float or None, optional
        Bounds of the trial sinusoidal period in sqrt(W); omitted bounds retain
        the existing model's fixed/data-derived defaults.
    period_samples : int or None, optional
        Number of trial periods; None retains the existing model default.
    use_region_growing : bool or None, optional
        Select the existing two-dimensional neighbor propagation instead of
        independent voxel fits. None retains the model default.
    seed_voxel : tuple of int or None, optional
        Positional seed indices in the existing spatial-dimension order. None
        retains automatic seed selection. Use legacy mappings for labelled seeds.
    return_debug_info : bool or None, optional
        In region mode, include existing stop/order/region maps when True.
    dynamic_dim : str or None, optional
        Sample dimension with reference-power coordinates in W.
    execution : ExecutionConfig or None, optional
        Existing worker settings. Region growing retains its ordered execution.

    Notes
    -----
    Frozen settings, with seed sequences copied to tuples. Scientific defaults,
    sweep fitting, seed interpretation and pulse calibration remain owned by the
    existing B1 model. Construction reads no data and keeps no cache.

    Examples
    --------
    >>> options = B1FitOptions(fa_deg=90.0, trf_ms=1.0, sint=1.0)
    """

    fa_deg: float | None = None
    trf_ms: float | None = None
    sint: float | None = None
    period_min_sqrt_w: float | None = None
    period_max_sqrt_w: float | None = None
    period_samples: int | None = None
    use_region_growing: bool | None = None
    seed_voxel: tuple[int, ...] | None = None
    return_debug_info: bool | None = None
    dynamic_dim: str | None = None
    execution: ExecutionConfig | None = None

    def __post_init__(self):
        _freeze_sequences(self, ("seed_voxel",))


@dataclass(frozen=True)
class IDEALFitOptions:
    """Supply calibration and active transient settings to the IDEAL solver.

    Parameters
    ----------
    relative_frequencies_hz : tuple of float or None, optional
        Component frequencies relative to the reference in Hz. Required for fitting.
    echo_spacing_s : float or None, optional
        Echo spacing in seconds. Required for fitting; the existing model uses
        centered, equally spaced echo times rather than coordinate values as times.
    metabolite_names : tuple of str or None, optional
        One output label per supplied component frequency, in matching order.
    echo_dim : str or None, optional
        Input echo dimension; None retains the existing model default.
    execution : ExecutionConfig or None, optional
        Existing shared worker settings; None retains solver defaults.
    solve_for : tuple of str or None, optional
        Nonempty sequence of distinct ``Field_offset`` and/or ``Theta`` names.
        None retains optimization of both in the existing solver. Theta models
        alternating-echo modulation; Theta alone omits the field-offset output.
    init_f_offset_hz : float or None, optional
        Initial field-offset guess in Hz; None retains 0.0 Hz.
    init_theta : complex or None, optional
        Initial dimensionless alternating-echo log-amplitude/phase modulation;
        None retains 0.0+0.0j. The solver applies exp(Theta*(-1)**(echo_index+1)).
    stop_f_offset_hz : float or None, optional
        Field-offset increment threshold in Hz; None retains 1.0 Hz.
    stop_theta : complex or None, optional
        Dimensionless real/imaginary increment thresholds for the respective
        Theta components; None retains 0.1+0.1j.
    max_iterations : int or None, optional
        Positive iteration cap per voxel; None retains 25 iterations.

    Notes
    -----
    Frozen settings, with supplied sequences copied to tuples. Construction fits
    no echoes and retains no cache. Supplied active settings are prepared once;
    scientific defaults and calculations remain in the existing solver. Lowercase
    initialization names translate to ``init_f_offset_Hz``/``init_Theta`` mapping
    keys. Validation occurs in the model. T2-star correction/optimization is not
    implemented; quantitative APIs reject its legacy option keys explicitly.

    Examples
    --------
    >>> options = IDEALFitOptions(
    ...     relative_frequencies_hz=(0.0, 120.0), echo_spacing_s=0.001
    ... )
    """

    relative_frequencies_hz: tuple[float, ...] | None = None
    echo_spacing_s: float | None = None
    metabolite_names: tuple[str, ...] | None = None
    echo_dim: str | None = None
    execution: ExecutionConfig | None = None
    solve_for: tuple[str, ...] | None = None
    init_f_offset_hz: float | None = None
    init_theta: complex | None = None
    stop_f_offset_hz: float | None = None
    stop_theta: complex | None = None
    max_iterations: int | None = None

    def __post_init__(self):
        _freeze_sequences(
            self, ("relative_frequencies_hz", "metabolite_names", "solve_for")
        )


def _options_mapping(options):
    renames = {
        "b_values_s_per_mm2": "b_values",
        "period_min_sqrt_w": "period_min",
        "period_max_sqrt_w": "period_max",
        "bounds_a": "bounds_A",
        "bounds_b": "bounds_B",
        "rel_bounds_a": "rel_bounds_A",
        "rel_bounds_b": "rel_bounds_B",
        "init_f_offset_hz": "init_f_offset_Hz",
        "init_theta": "init_Theta",
    }
    values = {
        renames.get(field.name, field.name): (
            list(getattr(options, field.name))
            if isinstance(getattr(options, field.name), tuple)
            else getattr(options, field.name)
        )
        for field in fields(options)
        if field.name != "execution" and getattr(options, field.name) is not None
    }
    if options.execution is not None:
        values.update(options.execution.as_fit_params())
    return values


__all__ = [
    "T1FitOptions",
    "ADCFitOptions",
    "B1FitOptions",
    "IDEALFitOptions",
    "T2FitOptions",
]
