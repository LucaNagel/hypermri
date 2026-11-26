"""Derive the phase-band delay ``phase_band_tau_s`` from scan timing metadata.

The phase band of the FID pseudo-inverse fit (``phase_band_tolerance_deg``)
ties the listed lines to the phase law
``angle(a_k) = phi0 + 2*pi*f_k*tau``. ``tau`` (``phase_band_tau_s``) is the time
from the moment at which all lines share one phase to ``t = 0`` of
``sampling_time_axis_s``. This module owns the one timing rule for it. The rule
was checked against measured data for the methods in
``VALIDATED_PHASE_BAND_SEQUENCES``; other Bruker methods get the same rule with
a warning. Only scan metadata is read; no data are loaded or reconstructed.
"""

from __future__ import annotations

import math
import operator
import warnings
from typing import Any

__all__ = [
    "PULSE_ACQUIRE_DEAD_TIME_SEQUENCES",
    "VALIDATED_PHASE_BAND_SEQUENCES",
    "phase_band_tau_s_from_scan",
]

VALIDATED_PHASE_BAND_SEQUENCES = frozenset({"<User:lucaCSI4>"})
"""Bruker method names for which the timing rule was checked on measured data."""

PULSE_ACQUIRE_DEAD_TIME_SEQUENCES = frozenset({"<Bruker:SINGLEPULSE>"})
"""Bruker methods without an echo time whose pulse program was read.

For them, ``TE`` in the timing rule is the time from the excitation isodelay
to the acquisition start: ``ExcPulse1`` length times its rephasing factor plus
``DeadTime``, as set by ``SINGLEPULSE.ppg`` of ParaVision 6.0.1.
"""


def _positive_finite(value: Any, name: str, unit: str) -> float:
    """Return a positive finite scalar or raise an actionable error."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"phase_band_tau_s_from_scan needs a positive finite {name} in {unit}; "
            f"got {value!r}. Load the scan with its method file or pass "
            "phase_band_tau_s explicitly."
        )
    return number


def _point_count(value: Any, name: str) -> int:
    """Return a non-negative integer number of points or raise."""
    try:
        if isinstance(value, bool):
            raise TypeError
        count = operator.index(value)
    except TypeError:
        count = -1
    if count < 0:
        raise ValueError(
            f"phase_band_tau_s_from_scan needs a non-negative integer {name}; "
            f"got {value!r}."
        )
    return count


def _method_parameters(scan: Any) -> dict:
    """Return the parsed ParaVision method parameters of ``scan`` or ``{}``.

    A loaded scan keeps them on its vendor handler (``scan.handler.method``);
    plain metadata objects may carry them as ``scan.method``.
    """
    source = getattr(scan, "handler", None) or scan
    method = getattr(source, "method", None)
    return method if isinstance(method, dict) else {}


def _excitation_to_acquisition_ms(method: dict) -> float:
    """Return the time from the excitation isodelay to the acquisition start.

    In ``SINGLEPULSE.ppg`` the excitation pulse ``p0`` is followed by
    ``dur1 = d1 - de`` and the receiver delay ``de``; the acquisition starts
    ``d1 = DeadTime`` after the end of the pulse. The isodelay lies
    ``Rpfac / 100`` of the pulse length before its end (``ExcPulse1`` fields
    0, length in ms, and 9, rephasing factor in percent).
    """
    pulse = method.get("ExcPulse1")
    try:
        length_ms, rephasing_percent = pulse[0], pulse[9]
    except (TypeError, IndexError, KeyError):
        length_ms = rephasing_percent = None
    length_ms = _positive_finite(length_ms, "scan.method['ExcPulse1'][0]", "ms")
    rephasing_percent = _positive_finite(
        rephasing_percent, "scan.method['ExcPulse1'][9] (rephasing factor)", "%"
    )
    dead_time_ms = _positive_finite(
        method.get("DeadTime"), "scan.method['DeadTime']", "ms"
    )
    return length_ms * rephasing_percent / 100.0 + dead_time_ms


def _echo_time_ms(scan: Any) -> tuple[float, str]:
    """Return the echo time in ms and the Bruker parameters it was read from.

    ``scan.echo_time_ms`` holds ``PVM_EchoTime``. Methods without that
    parameter, such as ``<Bruker:NSPECT>``, store the echo time as
    ``EchoTime`` in the method file. Pulse-acquire methods without any echo
    time (``PULSE_ACQUIRE_DEAD_TIME_SEQUENCES``) use the time from the
    excitation isodelay to the acquisition start in its place.
    """
    echo_time_ms = getattr(scan, "echo_time_ms", None)
    if echo_time_ms is not None:
        return _positive_finite(echo_time_ms, "scan.echo_time_ms", "ms"), (
            "PVM_EchoTime"
        )
    method = _method_parameters(scan)
    method_echo_time_ms = method.get("EchoTime")
    if method_echo_time_ms is not None:
        return _positive_finite(
            method_echo_time_ms, "scan.method['EchoTime']", "ms"
        ), "EchoTime"
    if getattr(scan, "sequence_name", None) in PULSE_ACQUIRE_DEAD_TIME_SEQUENCES:
        return _excitation_to_acquisition_ms(method), (
            "ExcPulse1 length * rephasing factor + DeadTime"
        )
    raise ValueError(
        "phase_band_tau_s_from_scan needs an echo time: scan.echo_time_ms "
        "(PVM_EchoTime) and scan.method['EchoTime'] are both missing, and "
        f"{getattr(scan, 'sequence_name', None)!r} is not a pulse-acquire method "
        "with a known pulse program "
        f"({', '.join(sorted(PULSE_ACQUIRE_DEAD_TIME_SEQUENCES))}). Pass "
        "phase_band_tau_s explicitly."
    )


def _filter_delay_points(scan: Any) -> int:
    """Return the digital filter delay in points that the loaded FID still holds.

    Stock Bruker methods store the raw FID with the filter delay of
    ``PVM_DigShift`` points in front of the signal. The validated custom method
    delivers an FID that starts after it; other custom methods are assumed to
    do the same.
    """
    method_type = getattr(scan, "method_type", None)
    if method_type == "custom":
        return 0
    if method_type == "stock":
        return _point_count(
            _method_parameters(scan).get("PVM_DigShift"),
            "scan.method['PVM_DigShift'] for a stock Bruker method",
        )
    raise ValueError(
        "phase_band_tau_s_from_scan needs scan.method_type 'stock' or 'custom' "
        f"to locate the digital filter delay; got {method_type!r}. Pass "
        "phase_band_tau_s explicitly."
    )


def phase_band_tau_s_from_scan(scan: Any) -> float:
    """Return ``phase_band_tau_s`` of a Bruker FID scan from its timing metadata.

    The rule was found for ``<User:lucaCSI4>``: there the first loaded sample
    lies one dwell time after the echo time, so ``tau = TE + dwell``. ``TE``
    counts from the isodelay of the excitation pulse, where all lines share one
    phase. In general, the sample ``m`` after the digital filter delay
    (counted from 0) lies at ``TE + (m + 1) * dwell`` after the isodelay, and

    ``tau = TE + (1 + cut_off - filter_delay_points) * dwell``.

    ``cut_off`` is the number of leading points removed at loading, after which
    ``sampling_time_axis_s`` starts at 0. ``filter_delay_points`` is the
    digital filter delay still contained in the raw FID: ``PVM_DigShift`` for
    stock Bruker methods and 0 for custom methods. For ``<User:lucaCSI4>``
    both are 0. A stock method with the default ``cut_off`` of 70 and
    ``PVM_DigShift`` 68 gives ``tau = TE + 3 * dwell``.

    The rule is empirical. Free fits (no phase band) of seven well-resolved
    hyperpolarized [1-13C]pyruvate ``<User:lucaCSI4>`` CSI scans at 7 T follow
    it with an rms deviation of 8 microseconds; a free slope gives
    ``tau = TE + (1.008 +- 0.010) * dwell``. ``tau = TE`` and
    ``tau = TE + dwell / 2`` deviate by about 380 and 190 microseconds and are
    excluded. The checked scans cover ``TE`` 1.10-1.26 ms, dwell 0.19-0.50 ms,
    excitation pulses of 0.525-0.84 ms and the Sharp and Medium digital
    filters.

    Methods outside ``VALIDATED_PHASE_BAND_SEQUENCES`` get the same rule and a
    warning. Their definition of ``TE``, the filter delay in their raw data and
    therefore the returned value have not been checked against measured
    phases.

    ``<Bruker:SINGLEPULSE>`` (``PULSE_ACQUIRE_DEAD_TIME_SEQUENCES``) has no
    echo time. Its pulse program starts the acquisition ``DeadTime`` after the
    end of the excitation pulse, whose isodelay lies the rephasing factor
    times the pulse length before its end. ``TE`` in the rule is then
    ``ExcPulse1`` length * rephasing factor / 100 + ``DeadTime``.

    The rule assumes a pulse-acquire FID. In spin-echo or
    stimulated-echo methods such as PRESS or STEAM, the lines share one phase
    at the echo top, not at the excitation, and the returned value does not
    apply.

    Parameters
    ----------
    scan : object
        Loaded scan, for example from ``hypermri.load_scan``, or any metadata
        object with these attributes:

        - ``vendor``: must be ``"bruker"``;
        - ``sequence_name``: the ParaVision method name, for example
          ``"<User:lucaCSI4>"``;
        - ``method_type``: ``"stock"`` for ``<Bruker:...>`` methods and
          ``"custom"`` otherwise;
        - ``echo_time_ms``: ``PVM_EchoTime`` in ms. If it is ``None``, the
          method parameter ``EchoTime`` in ms is used (``<Bruker:NSPECT>``),
          and for methods in ``PULSE_ACQUIRE_DEAD_TIME_SEQUENCES`` the
          method parameters ``ExcPulse1`` (field 0: length in ms, field 9:
          rephasing factor in percent) and ``DeadTime`` in ms;
        - ``dwell_time_s``: ``PVM_DigDw`` in s;
        - ``cut_off``: number of leading points removed at loading;
        - ``handler.method`` (loaded scans) or ``method``: dictionary of
          method parameters. It must contain ``PVM_DigShift`` (points) for
          stock methods and ``EchoTime``, or ``ExcPulse1`` and ``DeadTime``,
          when ``echo_time_ms`` is ``None``.

        The object is not modified.

    Returns
    -------
    float
        ``phase_band_tau_s`` in seconds: the time from the excitation
        isodelay to ``t = 0`` of ``sampling_time_axis_s``. It holds for the
        data as loaded. If leading samples are removed afterwards and the time
        axis is reset to start at 0, add their duration.

    Raises
    ------
    ValueError
        If the vendor is not Bruker, ``method_type`` is neither ``"stock"`` nor
        ``"custom"``, no positive finite echo time (or pulse length,
        rephasing factor and ``DeadTime`` for SINGLEPULSE) or dwell time is
        available,
        or ``cut_off`` or ``PVM_DigShift`` (stock methods) is not a
        non-negative integer. Then estimate ``tau`` from free fits of several
        well-separated lines and pass ``phase_band_tau_s`` explicitly.

    Warns
    -----
    UserWarning
        If the method is not in ``VALIDATED_PHASE_BAND_SEQUENCES`` or its echo
        time is not read from ``PVM_EchoTime``. The warning gives the value
        and how it was derived; it has not been verified.

    See Also
    --------
    hypermri.analysis.spectroscopy.fit_spectrum : Fit that applies the band.

    Examples
    --------
    For the validated method the value is ``TE + dwell``:

    >>> from types import SimpleNamespace
    >>> from hypermri.analysis.spectroscopy import phase_band_tau_s_from_scan
    >>> scan = SimpleNamespace(
    ...     vendor="bruker",
    ...     sequence_name="<User:lucaCSI4>",
    ...     method_type="custom",
    ...     echo_time_ms=1.258,
    ...     dwell_time_s=0.3323e-3,
    ...     cut_off=0,
    ... )
    >>> round(phase_band_tau_s_from_scan(scan) * 1e3, 4)
    1.5903

    A stock NSPECT scan keeps two points after the filter delay of 68 points,
    so the value is ``TE + 3 * dwell``, with a warning:

    >>> import warnings
    >>> nspect = SimpleNamespace(
    ...     vendor="bruker",
    ...     sequence_name="<Bruker:NSPECT>",
    ...     method_type="stock",
    ...     echo_time_ms=None,
    ...     dwell_time_s=0.3312e-3,
    ...     cut_off=70,
    ...     method={"EchoTime": 0.8225, "PVM_DigShift": 68},
    ... )
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", UserWarning)
    ...     round(phase_band_tau_s_from_scan(nspect) * 1e3, 4)
    1.8161

    A SINGLEPULSE scan with a 0.128 ms block pulse (rephasing factor 50 %)
    and ``DeadTime`` 0.05 ms gives ``0.114 ms + 3 * dwell``, with a warning:

    >>> single = SimpleNamespace(
    ...     vendor="bruker",
    ...     sequence_name="<Bruker:SINGLEPULSE>",
    ...     method_type="stock",
    ...     echo_time_ms=None,
    ...     dwell_time_s=0.5e-3,
    ...     cut_off=70,
    ...     method={
    ...         "ExcPulse1": [0.128, 10000, 90, "No", 3, 1280, 1, 1, 0, 50],
    ...         "DeadTime": 0.05,
    ...         "PVM_DigShift": 68,
    ...     },
    ... )
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", UserWarning)
    ...     round(phase_band_tau_s_from_scan(single) * 1e3, 4)
    1.614

    With a loaded scan the result goes into the fit parameters::

        scan = hypermri.load_scan(path)
        fit_params["phase_band_tau_s"] = phase_band_tau_s_from_scan(scan)
    """
    vendor = getattr(scan, "vendor", None)
    if vendor != "bruker":
        raise ValueError(
            "phase_band_tau_s_from_scan supports only Bruker scans; got vendor "
            f"{vendor!r}. Estimate tau from free fits and pass phase_band_tau_s "
            "explicitly."
        )
    sequence_name = getattr(scan, "sequence_name", None)
    echo_time_ms, echo_time_parameter = _echo_time_ms(scan)
    dwell_time_s = _positive_finite(
        getattr(scan, "dwell_time_s", None), "scan.dwell_time_s", "s"
    )
    cut_off = _point_count(getattr(scan, "cut_off", None), "scan.cut_off")
    filter_delay_points = _filter_delay_points(scan)
    first_sample_dwells = 1 + cut_off - filter_delay_points
    tau_s = echo_time_ms * 1e-3 + first_sample_dwells * dwell_time_s

    if (
        sequence_name not in VALIDATED_PHASE_BAND_SEQUENCES
        or echo_time_parameter != "PVM_EchoTime"
    ):
        validated = ", ".join(sorted(VALIDATED_PHASE_BAND_SEQUENCES))
        warnings.warn(
            f"phase_band_tau_s_from_scan: tau = {tau_s * 1e3:.4f} ms for the "
            f"method {sequence_name!r} has not been verified. It applies the "
            f"rule checked for {validated}: TE ({echo_time_parameter}) + "
            f"(1 + cut_off {cut_off} - filter delay {filter_delay_points} points)"
            " * dwell, for a pulse-acquire FID. Check it against free fits of "
            "several well-separated lines.",
            UserWarning,
            stacklevel=2,
        )
    return tau_s
