"""Behavioral tests for the public single-scan loading path."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.exceptions import DataLoadingError
from hypermri.loading import scans as factory
from hypermri.loading.experiment import MRIExperiment
from hypermri.sequences.base_sequence import BaseSequence


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, "lazy"),
        (True, "full"),
        ("lazy", "lazy"),
        (" MINIMAL ", "minimal"),
        ("FULL", "full"),
    ],
)
def test_normalize_load_mode_accepts_compatibility_and_named_modes(value, expected):
    assert factory.normalize_load_mode(value) == expected


@pytest.mark.parametrize("value", [None, 1, object()])
def test_normalize_load_mode_rejects_non_boolean_values(value):
    with pytest.raises(TypeError, match="Loading mode"):
        factory.normalize_load_mode(value)


def test_normalize_load_mode_rejects_unknown_string():
    with pytest.raises(ValueError, match="Unsupported loading mode"):
        factory.normalize_load_mode("eager")


def test_load_scan_rejects_non_boolean_sequence_resolution():
    with pytest.raises(TypeError, match="resolve_sequence"):
        factory.load_scan("scan", resolve_sequence="false")


def _handler(**attributes):
    """Return a synthetic vendor handler with the attributes load_scan reads."""
    defaults = {
        "path": Path("scan"),
        "type": "unknown",
        "vendor": "unknown",
        "get_parameters": lambda: {},
    }
    return SimpleNamespace(**{**defaults, **attributes})


def _patch_handler(monkeypatch, handler, calls=None):
    """Make ``load_scan`` construct ``handler`` and record each construction."""
    calls = [] if calls is None else calls

    def fake_get_vendor_handler(**kwargs):
        calls.append(kwargs)
        return handler

    monkeypatch.setattr(factory, "get_vendor_handler", fake_get_vendor_handler)
    return calls


def _array(value: float) -> xr.DataArray:
    return xr.DataArray(np.full((2, 2), value), dims=("read_idx", "phase_idx"))


class _RecordingSequence(BaseSequence):
    """Registered sequence class whose default policy only records its calls."""

    def run_default_reconstructions(self, **kwargs):
        self.policy_calls = getattr(self, "policy_calls", []) + [kwargs]


@pytest.mark.parametrize(
    ("mode", "expected_handler_flag"),
    [
        (False, False),
        ("lazy", False),
        ("minimal", False),
        (True, True),
        ("full", True),
    ],
)
def test_load_scan_passes_only_a_boolean_to_vendor_handler(
    monkeypatch, mode, expected_handler_flag
):
    handler = _handler(load_2dseq_file=lambda: None)
    calls = _patch_handler(monkeypatch, handler)
    monkeypatch.setattr(MRIExperiment, "reconstruct", lambda self, reco_type: None)

    result = factory.load_scan(
        Path("scan"),
        mode=mode,
        force_vendor=None,
        handler_options={"example": 1},
    )

    assert type(result) is MRIExperiment
    assert result.handler is handler
    assert calls == [
        {
            "source": "scan",
            "load_data": expected_handler_flag,
            "log_mode": "error",
            "force_vendor": None,
            "handler_options": {"example": 1},
        }
    ]
    assert isinstance(calls[0]["load_data"], bool)


def test_load_scan_resolves_sequence_and_runs_normalized_policy(monkeypatch):
    handler = _handler(
        type="Bruker:FLASH", vendor="bruker", load_2dseq_file=lambda: None
    )
    calls = _patch_handler(monkeypatch, handler)
    resolver_calls = []
    category = object()

    def fake_resolve(method_name, vendor):
        resolver_calls.append((method_name, vendor))
        return {"sequence_class": _RecordingSequence, "category": category}

    monkeypatch.setattr(factory, "_resolve_sequence_entry", fake_resolve)

    result = factory.load_scan(Path("scan"), mode="minimal")

    assert isinstance(result, _RecordingSequence)
    assert len(calls) == 1
    assert calls[0]["load_data"] is False
    assert result.handler is handler
    assert result.path == Path("scan").resolve()
    assert result._sequence_requested_load_data == "minimal"
    assert result._sequence_reco_options == {"minimal_loading": True}
    assert resolver_calls == [("Bruker:FLASH", "bruker")]
    assert result.policy_calls == [
        {
            "load_data": "minimal",
            "reco_options": {"minimal_loading": True},
            "category": category,
        }
    ]


def test_load_scan_builds_the_registered_sequence_from_its_one_handler(monkeypatch):
    """No generic experiment is built and adopted for a registered scan."""
    from hypermri.sequences.FLASH import FLASH

    handler = _handler(type="Bruker:FLASH", vendor="bruker")
    calls = _patch_handler(monkeypatch, handler)

    def fail(*_args, **_kwargs):
        raise AssertionError("load_scan must construct no second handler")

    monkeypatch.setattr(MRIExperiment, "_get_vendor_handler", fail)
    monkeypatch.setattr(
        factory,
        "_resolve_sequence_entry",
        lambda *_args: {"sequence_class": FLASH, "category": "anatomical"},
    )

    result = factory.load_scan(Path("scan"), mode="lazy", log_mode="debug")

    assert type(result) is FLASH
    assert len(calls) == 1
    assert result.handler is handler
    assert handler.log_mode == "debug"
    assert result.log_mode == "debug"
    assert "_plotter" not in vars(result)
    assert result.reconstructions == {}


@pytest.mark.parametrize(
    ("mode", "expected_reconstructions"),
    [
        ("lazy", []),
        ("minimal", ["seq2d"]),
        ("full", ["seq2d", "raw"]),
    ],
)
def test_load_scan_bruker_csi_constructs_no_viewer_and_runs_policy_once(
    monkeypatch, mode, expected_reconstructions
):
    """Cover the loading seam used by the Bruker CSI reconstruction pipeline."""
    from hypermri.sequences.CSI import CSI

    handler = _handler(
        fid=SimpleNamespace(shape=(1,)),
        has_raw_data=True,
        id="synthetic-csi",
        type="<Bruker:CSI>",
        vendor="bruker",
        load_2dseq_file=lambda *_args, **_kwargs: None,
    )
    calls = _patch_handler(monkeypatch, handler)
    reconstruction_calls = []

    def fake_reconstruct(self, reco_type, **kwargs):
        reconstruction_calls.append((reco_type, kwargs))
        result = object()
        self.reconstructions[reco_type] = result
        return result

    monkeypatch.setattr(
        factory,
        "_resolve_sequence_entry",
        lambda *_args: {"sequence_class": CSI, "category": "spectroscopy"},
    )
    monkeypatch.setattr(CSI, "reconstruct", fake_reconstruct)

    result = factory.load_scan(Path("scan"), mode=mode)

    assert isinstance(result, CSI)
    assert calls[0]["load_data"] is (mode == "full")
    assert "_plotter" not in vars(result)
    assert [reco_type for reco_type, _kwargs in reconstruction_calls] == (
        expected_reconstructions
    )

    result.run_default_reconstructions(load_data=mode)
    assert [reco_type for reco_type, _kwargs in reconstruction_calls] == (
        expected_reconstructions
    )


def test_minimal_registered_non_paravision_sequence_prepares_vendor_data(
    monkeypatch,
):
    payload = _array(1.0)
    load_calls = []
    handler = _handler(type="csi", vendor="siemens", reco_data=None, data_array=None)

    def load_data(path):
        load_calls.append(path)
        handler.data = payload

    handler.load_data = load_data
    _patch_handler(monkeypatch, handler)
    monkeypatch.setattr(
        factory,
        "_resolve_sequence_entry",
        lambda *_args: {
            "sequence_class": _RecordingSequence,
            "category": "spectroscopy",
        },
    )

    result = factory.load_scan(Path("scan"), mode="minimal")

    assert isinstance(result, _RecordingSequence)
    assert load_calls == [Path("scan")]
    assert result.data is payload
    assert result.reconstructions == {"raw": payload}
    assert len(result.policy_calls) == 1


def test_load_scan_explicit_sequence_skips_registry_resolution(monkeypatch):
    handler = _handler(type="custom")
    _patch_handler(monkeypatch, handler)

    def fail_if_resolved(*_args):
        raise AssertionError("An explicit class must bypass the registry")

    class ExplicitSequence:
        def __init__(self, received_handler):
            assert received_handler is handler
            self.policy_calls = []

        def run_default_reconstructions(self, **kwargs):
            self.policy_calls.append(kwargs)

    monkeypatch.setattr(factory, "_resolve_sequence_entry", fail_if_resolved)

    result = factory.load_scan(
        Path("scan"),
        mode="lazy",
        sequence_class=ExplicitSequence,
    )

    assert isinstance(result, ExplicitSequence)
    assert result.policy_calls == [
        {"load_data": "lazy", "reco_options": {}, "category": None}
    ]


def test_load_scan_can_keep_generic_experiment_without_registry_resolution(
    monkeypatch,
):
    handler = _handler(type="Bruker:FLASH", vendor="bruker")
    _patch_handler(monkeypatch, handler)

    def fail_if_resolved(*_args):
        raise AssertionError("Registry must not be used when resolution is disabled")

    monkeypatch.setattr(factory, "_resolve_sequence_entry", fail_if_resolved)

    result = factory.load_scan(
        Path("scan"),
        mode="lazy",
        resolve_sequence=False,
    )

    assert type(result) is MRIExperiment
    assert result.handler is handler


@pytest.mark.parametrize("mode", ["lazy", "minimal", "full"])
def test_load_scan_raises_when_sequence_construction_fails(monkeypatch, mode):
    """The scan is never returned as a generic experiment in its place."""
    _patch_handler(
        monkeypatch,
        _handler(type="broken", vendor="bruker", load_2dseq_file=lambda: None),
    )

    class BrokenSequence:
        def __init__(self, _handler, **_kwargs):
            raise RuntimeError("cannot wrap")

    monkeypatch.setattr(
        factory,
        "_resolve_sequence_entry",
        lambda *_args: {"sequence_class": BrokenSequence, "category": "anatomical"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="cannot wrap"):
            factory.load_scan(Path("scan"), mode=mode)


@pytest.mark.parametrize(
    ("force_vendor", "expected_vendor"), [(None, "bruker"), (" Bruker ", "bruker")]
)
def test_load_scan_resolves_the_sequence_with_the_handler_vendor(
    monkeypatch, force_vendor, expected_vendor
):
    calls = _patch_handler(monkeypatch, _handler(type="Bruker:FLASH", vendor="bruker"))
    resolver_calls = []
    monkeypatch.setattr(
        factory,
        "_resolve_sequence_entry",
        lambda method, vendor: resolver_calls.append((method, vendor)),
    )

    result = factory.load_scan(Path("scan"), mode="lazy", force_vendor=force_vendor)

    assert type(result) is MRIExperiment
    assert len(calls) == 1
    assert calls[0]["force_vendor"] == force_vendor
    assert resolver_calls == [("Bruker:FLASH", expected_vendor)]


def test_load_scan_skips_the_registry_for_a_handler_of_unknown_vendor(monkeypatch):
    _patch_handler(monkeypatch, _handler(type="Bruker:FLASH", vendor="unknown"))

    def fail_if_resolved(*_args):
        raise AssertionError("An unknown vendor must not be resolved")

    monkeypatch.setattr(factory, "_resolve_sequence_entry", fail_if_resolved)

    assert type(factory.load_scan(Path("scan"))) is MRIExperiment


def test_minimal_load_reconstructs_seq2d_for_unregistered_paravision_scan(
    monkeypatch,
):
    reconstruction_calls = []
    _patch_handler(
        monkeypatch,
        _handler(type="unregistered", vendor="bruker", load_2dseq_file=lambda: None),
    )
    monkeypatch.setattr(
        MRIExperiment,
        "reconstruct",
        lambda self, reco_type: reconstruction_calls.append(reco_type),
    )
    monkeypatch.setattr(factory, "_resolve_sequence_entry", lambda *_args: None)

    result = factory.load_scan(Path("scan"), mode="minimal")

    assert type(result) is MRIExperiment
    assert reconstruction_calls == ["seq2d"]


def test_minimal_load_uses_vendor_data_for_unregistered_non_paravision_scan(
    monkeypatch,
):
    payload = _array(1.0)
    load_calls = []
    handler = _handler(
        type="unregistered", vendor="siemens", reco_data=None, data_array=None
    )

    def load_data(path):
        load_calls.append(path)
        handler.scanner_data = payload

    handler.load_data = load_data
    _patch_handler(monkeypatch, handler)
    monkeypatch.setattr(factory, "_resolve_sequence_entry", lambda *_args: None)

    result = factory.load_scan(Path("scan"), mode="minimal")

    assert type(result) is MRIExperiment
    assert load_calls == [Path("scan")]
    assert result.scanner_data is payload


def test_load_scan_raises_for_a_path_no_vendor_recognizes(monkeypatch, tmp_path):
    _patch_handler(monkeypatch, None)

    with pytest.raises(DataLoadingError, match="no registered vendor recognizes"):
        factory.load_scan(tmp_path)


def test_create_experiment_preserves_historical_arguments(monkeypatch):
    expected = object()
    calls = []

    def fake_load_scan(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    sequence_class = type("Sequence", (), {})
    monkeypatch.setattr(factory, "load_scan", fake_load_scan)

    result = factory.create_experiment(
        "scan",
        load_data="minimal",
        sequence_class=sequence_class,
        force_vendor="bruker",
        handler_options={"log_mode": "debug"},
    )

    assert result is expected
    assert calls == [
        (
            "scan",
            {
                "mode": "minimal",
                "sequence_class": sequence_class,
                "force_vendor": "bruker",
                "handler_options": {"log_mode": "debug"},
                "log_mode": "error",
            },
        )
    ]


def test_create_experiment_forwards_explicit_log_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        factory,
        "load_scan",
        lambda path, **kwargs: calls.append((path, kwargs)) or object(),
    )

    factory.create_experiment("scan", log_mode="debug")

    assert calls[0][1]["log_mode"] == "debug"


def test_load_scan_is_exported_from_package_root():
    import hypermri
    from hypermri.loading.scans import load_scan

    assert hypermri.load_scan is load_scan


def test_unresolved_sequence_logs_method_and_generic_result(monkeypatch, caplog):
    caplog.set_level("WARNING", logger=factory.__name__)
    handler = _handler(type="UnregisteredMethod", vendor="bruker")
    _patch_handler(monkeypatch, handler)
    result = factory.load_scan("scan")
    assert type(result) is MRIExperiment
    assert "UnregisteredMethod" in caplog.text
    assert "MRIExperiment" in caplog.text
    caplog.clear()
    factory.load_scan("scan", resolve_sequence=False)
    assert not caplog.records


def test_sequence_registry_import_error_propagates(monkeypatch):
    import builtins

    original = builtins.__import__

    def broken_registry(name, *args, **kwargs):
        if name == "hypermri.sequences.registry":
            raise ImportError("synthetic registry import defect")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_registry)
    with pytest.raises(ImportError, match="synthetic registry"):
        factory._resolve_sequence_entry("FLASH", "bruker")


@pytest.mark.parametrize(
    "method, category",
    [
        ("FLASH", "anatomical"),
        ("Singlepulse", "spectroscopy"),
        ("IVRecovery", "spectroscopy"),
        ("CPMG", "spectroscopy"),
        ("ISIS", "spectroscopy"),
        ("GeoffSlcSpec", "spectroscopy"),
        ("FEXI", "diffusion"),
    ],
)
def test_experiment_uses_vendor_mapping_category(method, category):

    scan = MRIExperiment("scan", handler=_handler(type=method, vendor="bruker"))
    assert scan._guess_sequence_category().value == category


def test_categories_registered_at_runtime_have_one_owner(monkeypatch):
    from hypermri.loading.contracts import SequenceCategory
    from hypermri.sequences import registry as sequence_registry
    from hypermri.sequences.registry import SequenceRegistry

    # Every category accepted by the schema is resolved without guessing names.
    for category in SequenceCategory:
        registry = SequenceRegistry()
        registry.register_sequence(
            "synthetic", object, category, ["exact:NeutralName"], vendor="bruker"
        )
        assert (
            registry.resolve_sequence("NeutralName", "bruker")["category"] is category
        )
        monkeypatch.setattr(
            sequence_registry,
            "get_sequence_registry",
            lambda registry=registry: registry,
        )
        scan = MRIExperiment(
            "scan", handler=_handler(type="NeutralName", vendor="bruker")
        )
        assert scan._guess_sequence_category() is category
