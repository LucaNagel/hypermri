"""Test the phase-band delay rule ``tau = TE + (1 + cut_off - D) * dwell``.

Scan metadata are synthetic namespaces with the attributes a loaded Bruker
scan exposes; no scanner files are read. ``D`` is the digital filter delay that
the loaded raw FID still contains. The rule was checked on measured data for
``<User:lucaCSI4>`` (seven resolved pyruvate CSI scans, rms deviation 8 us),
where it reduces to ``TE + dwell``. The NSPECT values (``EchoTime`` 0.8225 ms,
dwell 0.3312 ms, ``PVM_DigShift`` 68, ``cut_off`` 70) follow a stock method
file; the rule is not verified for them, which the helper reports as a
warning. The SINGLEPULSE values (0.128 ms block pulse, rephasing factor 50 %,
``DeadTime`` 0.05 ms, dwell 0.5 ms) follow a ParaVision 6.0.1 stock method file;
its pulse program starts the acquisition ``DeadTime`` after the pulse end. These
tests fix the units, the point bookkeeping, the warning and the
agreement with the phase convention of the FID model.
"""

import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    fid_basefunc,
    solve_for_complex_amplitudes,
)
from hypermri.analysis.spectroscopy import (
    PULSE_ACQUIRE_DEAD_TIME_SEQUENCES,
    VALIDATED_PHASE_BAND_SEQUENCES,
    phase_band_tau_s_from_scan,
)

UNVERIFIED = "has not been verified"


def scan_metadata(**changes):
    """Return lucaCSI4 metadata with TE 1.258 ms and dwell 0.3323 ms."""
    values = {
        "vendor": "bruker",
        "sequence_name": "<User:lucaCSI4>",
        "method_type": "custom",
        "echo_time_ms": 1.258,
        "dwell_time_s": 0.3323e-3,
        "cut_off": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def nspect_metadata(**changes):
    """Return stock NSPECT metadata without PVM_EchoTime and with cut_off 70."""
    values = {
        "sequence_name": "<Bruker:NSPECT>",
        "method_type": "stock",
        "echo_time_ms": None,
        "dwell_time_s": 0.3312e-3,
        "cut_off": 70,
        "method": {"EchoTime": 0.8225, "PVM_DigShift": 68},
    }
    values.update(changes)
    return scan_metadata(**values)


def singlepulse_metadata(**changes):
    """Return stock SINGLEPULSE metadata, which has no echo time at all."""
    values = {
        "sequence_name": "<Bruker:SINGLEPULSE>",
        "method_type": "stock",
        "echo_time_ms": None,
        "dwell_time_s": 0.5e-3,
        "cut_off": 70,
        "method": {
            "ExcPulse1": [0.128, 10000, 90, "No", 3, 1280, 1, 1, 0, 50, 4.88, "<bp>"],
            "DeadTime": 0.05,
            "PVM_DigShift": 68,
        },
    }
    values.update(changes)
    return scan_metadata(**values)


def singlepulse_method(**changes):
    """Return SINGLEPULSE method parameters with ``changes`` applied."""
    method = dict(singlepulse_metadata().method)
    method.update(changes)
    return {key: value for key, value in method.items() if value is not None}


def tau_without_warning(scan):
    """Return the helper's value and fail if it warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return phase_band_tau_s_from_scan(scan)


@pytest.mark.parametrize(
    ("echo_time_ms", "dwell_time_s", "tau_s"),
    [(1.258, 0.3323e-3, 1.5903e-3), (1.1005, 0.5e-3, 1.6005e-3)],
)
def test_validated_tau_is_echo_time_plus_one_dwell_in_seconds(
    echo_time_ms, dwell_time_s, tau_s
):
    """TE in ms and dwell in s give tau in s, without a warning."""
    scan = scan_metadata(echo_time_ms=echo_time_ms, dwell_time_s=dwell_time_s)
    assert tau_without_warning(scan) == pytest.approx(tau_s, rel=1e-12)


def test_removed_leading_points_move_tau_by_their_duration():
    """Each point removed at loading moves t = 0 one dwell later."""
    scan = scan_metadata(cut_off=5)
    assert tau_without_warning(scan) == pytest.approx(
        1.258e-3 + 6 * 0.3323e-3, rel=1e-12
    )


def test_stock_nspect_gets_the_same_rule_with_a_warning():
    """NSPECT: EchoTime + (1 + 70 - 68) dwell, reported as not verified."""
    with pytest.warns(UserWarning, match=UNVERIFIED) as record:
        tau_s = phase_band_tau_s_from_scan(nspect_metadata())
    assert tau_s == pytest.approx(0.8225e-3 + 3 * 0.3312e-3, rel=1e-12)
    assert len(record) == 1
    message = str(record[0].message)
    assert "'<Bruker:NSPECT>'" in message
    assert "1.8161 ms" in message
    assert "TE (EchoTime)" in message
    assert record[0].filename == __file__


def test_singlepulse_uses_pulse_isodelay_plus_dead_time_with_a_warning():
    """SINGLEPULSE: 0.128 ms * 50 % + 0.05 ms + (1 + 70 - 68) dwell."""
    with pytest.warns(UserWarning, match=UNVERIFIED) as record:
        tau_s = phase_band_tau_s_from_scan(singlepulse_metadata())
    assert tau_s == pytest.approx(0.114e-3 + 3 * 0.5e-3, rel=1e-12)
    assert len(record) == 1
    message = str(record[0].message)
    assert "'<Bruker:SINGLEPULSE>'" in message
    assert "1.6140 ms" in message
    assert "ExcPulse1 length * rephasing factor + DeadTime" in message


def test_singlepulse_rephasing_factor_sets_the_isodelay():
    """A rephasing factor of 100 % puts the isodelay at the pulse start."""
    scan = singlepulse_metadata(
        method=singlepulse_method(
            ExcPulse1=[0.2, 0, 0, "", 0, 0, 0, 0, 0, 100], DeadTime=0.03
        )
    )
    with pytest.warns(UserWarning, match=UNVERIFIED):
        tau_s = phase_band_tau_s_from_scan(scan)
    assert tau_s == pytest.approx(0.23e-3 + 3 * 0.5e-3, rel=1e-12)


def test_dead_time_is_used_only_for_listed_pulse_acquire_methods():
    """Another method with DeadTime but no echo time still needs tau."""
    assert PULSE_ACQUIRE_DEAD_TIME_SEQUENCES == frozenset({"<Bruker:SINGLEPULSE>"})
    scan = singlepulse_metadata(sequence_name="<Bruker:PRESS>")
    with pytest.raises(ValueError, match="not a pulse-acquire method"):
        phase_band_tau_s_from_scan(scan)


def test_stock_method_with_pvm_echo_time_uses_it():
    """PVM_EchoTime takes precedence over the method parameter EchoTime."""
    scan = nspect_metadata(sequence_name="<Bruker:CSI>", echo_time_ms=1.0)
    with pytest.warns(UserWarning, match=r"TE \(PVM_EchoTime\)"):
        tau_s = phase_band_tau_s_from_scan(scan)
    assert tau_s == pytest.approx(1.0e-3 + 3 * 0.3312e-3, rel=1e-12)


@pytest.mark.parametrize("sequence_name", ["<User:otherCSI>", None])
def test_unvalidated_custom_method_gets_te_plus_dwell_with_a_warning(sequence_name):
    """Custom methods are assumed to deliver an FID after the filter delay."""
    scan = scan_metadata(sequence_name=sequence_name)
    with pytest.warns(UserWarning, match=UNVERIFIED):
        tau_s = phase_band_tau_s_from_scan(scan)
    assert tau_s == pytest.approx(1.5903e-3, rel=1e-12)


def test_validated_method_without_pvm_echo_time_warns():
    """The validated rule uses PVM_EchoTime; another TE source is unverified."""
    scan = scan_metadata(echo_time_ms=None, method={"EchoTime": 1.258})
    with pytest.warns(UserWarning, match=UNVERIFIED):
        phase_band_tau_s_from_scan(scan)


def test_loaded_scan_method_parameters_come_from_the_handler():
    """A loaded scan's method file is read via scan.handler, not scan.method.

    ``scan.method`` on a loaded scan is a deprecated forward to the handler.
    """

    class LoadedScan(SimpleNamespace):
        @property
        def method(self):
            raise AssertionError("use scan.handler.method")

    metadata = vars(nspect_metadata())
    handler = SimpleNamespace(method=metadata.pop("method"))
    scan = LoadedScan(handler=handler, **metadata)
    with pytest.warns(UserWarning, match=UNVERIFIED):
        tau_s = phase_band_tau_s_from_scan(scan)
    assert tau_s == pytest.approx(0.8225e-3 + 3 * 0.3312e-3, rel=1e-12)


@pytest.mark.parametrize(
    "make_scan", [scan_metadata, nspect_metadata, singlepulse_metadata]
)
def test_only_metadata_is_read_and_left_unchanged(make_scan):
    """The helper neither modifies the scan nor touches its data."""

    class LazyScan(SimpleNamespace):
        @property
        def primary_data(self):
            raise AssertionError("phase_band_tau_s_from_scan must not load data")

    scan = LazyScan(**vars(make_scan()))
    before = {name: value for name, value in vars(scan).items()}
    method_before = dict(getattr(scan, "method", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phase_band_tau_s_from_scan(scan)
    assert vars(scan) == before
    assert getattr(scan, "method", {}) == method_before


def test_validated_methods_are_the_checked_custom_csi():
    """Only the method checked on measured data is listed as validated."""
    assert VALIDATED_PHASE_BAND_SEQUENCES == frozenset({"<User:lucaCSI4>"})


@pytest.mark.parametrize(
    ("make_scan", "changes", "message"),
    [
        (scan_metadata, {"vendor": "siemens"}, "supports only Bruker scans"),
        (scan_metadata, {"vendor": None}, "supports only Bruker scans"),
        (scan_metadata, {"method_type": None}, "method_type 'stock' or 'custom'"),
        (scan_metadata, {"cut_off": -1}, "integer scan.cut_off; got -1"),
        (scan_metadata, {"cut_off": 1.5}, "integer scan.cut_off; got 1.5"),
        (scan_metadata, {"cut_off": None}, "integer scan.cut_off; got None"),
        (scan_metadata, {"cut_off": True}, "integer scan.cut_off; got True"),
        (scan_metadata, {"echo_time_ms": np.nan}, "scan.echo_time_ms in ms"),
        (scan_metadata, {"echo_time_ms": 0.0}, "scan.echo_time_ms in ms"),
        (scan_metadata, {"echo_time_ms": [1.2, 2.4]}, "scan.echo_time_ms in ms"),
        (scan_metadata, {"echo_time_ms": None}, "are both missing"),
        (scan_metadata, {"dwell_time_s": None}, "scan.dwell_time_s in s"),
        (scan_metadata, {"dwell_time_s": -0.3e-3}, "scan.dwell_time_s in s"),
        (scan_metadata, {"dwell_time_s": np.inf}, "scan.dwell_time_s in s"),
        (nspect_metadata, {"method": {"PVM_DigShift": 68}}, "are both missing"),
        (
            nspect_metadata,
            {"method": {"EchoTime": -1.0, "PVM_DigShift": 68}},
            "scan.method['EchoTime'] in ms",
        ),
        (nspect_metadata, {"method": {"EchoTime": 0.8225}}, "PVM_DigShift'] for"),
        (
            nspect_metadata,
            {"method": {"EchoTime": 0.8225, "PVM_DigShift": 67.5}},
            "PVM_DigShift'] for a stock Bruker method; got 67.5",
        ),
        (
            singlepulse_metadata,
            {"method": singlepulse_method(DeadTime=None)},
            "scan.method['DeadTime'] in ms",
        ),
        (
            singlepulse_metadata,
            {"method": singlepulse_method(DeadTime=0.0)},
            "scan.method['DeadTime'] in ms",
        ),
        (
            singlepulse_metadata,
            {"method": singlepulse_method(ExcPulse1=None)},
            "scan.method['ExcPulse1'][0] in ms",
        ),
        (
            singlepulse_metadata,
            {"method": singlepulse_method(ExcPulse1=[0.128])},
            "scan.method['ExcPulse1'][0] in ms",
        ),
        (
            singlepulse_metadata,
            {"method": singlepulse_method(ExcPulse1=[0.128] + [0] * 9)},
            "(rephasing factor) in %",
        ),
    ],
)
def test_incomplete_timing_metadata_is_rejected(make_scan, changes, message):
    """Missing or invalid timing fails early and names the attribute."""
    with pytest.raises(ValueError, match=re.escape(message)):
        phase_band_tau_s_from_scan(make_scan(**changes))


def test_missing_attribute_is_rejected():
    """A metadata object without a dwell time names the missing attribute."""
    scan = scan_metadata()
    del scan.dwell_time_s
    with pytest.raises(ValueError, match="scan.dwell_time_s"):
        phase_band_tau_s_from_scan(scan)


@pytest.mark.parametrize(
    "make_scan", [scan_metadata, nspect_metadata, singlepulse_metadata]
)
def test_signal_timed_by_the_rule_fits_the_phase_law_exactly(make_scan):
    """The returned tau matches the phase convention of the FID model.

    All lines start with one phase at the excitation isodelay. The raw FID
    holds ``D`` filter-delay points (``PVM_DigShift`` for the stock method, 0
    for lucaCSI4); raw sample ``r`` lies at ``TE + (r - D + 1) * dwell`` after
    the isodelay, and loading keeps the samples from ``cut_off`` on. With
    tolerance 0 the banded solve then reproduces the signal; tau = TE leaves
    phase errors of up to 0.6 rad at -300 Hz. The bounded phi0 search stops
    within 1e-4 rad, so the relative residual with the rule is checked to 1e-3.
    """
    scan = make_scan()
    filter_delay_points = getattr(scan, "method", {}).get("PVM_DigShift", 0)
    if scan.method_type == "custom":
        filter_delay_points = 0
    echo_time_ms = scan.echo_time_ms or scan.method.get("EchoTime")
    if echo_time_ms is None:
        pulse = scan.method["ExcPulse1"]
        echo_time_ms = pulse[0] * pulse[9] / 100 + scan.method["DeadTime"]
    freqs_hz = np.array([-300.0, 0.0, 120.0])
    t2s_s = np.full(3, 0.03)
    areas = np.array([10.0, 20.0, 6.0])
    raw_index = np.arange(scan.cut_off, scan.cut_off + 256)
    since_isodelay_s = (
        echo_time_ms * 1e-3 + (raw_index - filter_delay_points + 1) * scan.dwell_time_s
    )
    time_s = np.arange(256) * scan.dwell_time_s
    fid = fid_basefunc(freqs_hz, t2s_s, since_isodelay_s).T @ (areas * np.exp(0.7j))

    def relative_residual(tau_s):
        amplitudes = solve_for_complex_amplitudes(
            fid,
            freqs_hz,
            t2s_s,
            time_s,
            signal_domain="temporal",
            amp_solver="pinv",
            phase_band_tolerance_deg=np.zeros(3),
            phase_band_tau_s=tau_s,
        )[1]
        model = fid_basefunc(freqs_hz, t2s_s, time_s).T @ amplitudes
        return np.linalg.norm(fid - model) / np.linalg.norm(fid)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tau_s = phase_band_tau_s_from_scan(scan)
    assert relative_residual(tau_s) < 1e-3
    assert relative_residual(echo_time_ms * 1e-3) > 0.1
