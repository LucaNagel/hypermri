"""Regression tests for the Study-to-single-scan loading contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _PreliminaryExperiment:
    """Generic experiment returned when sequence resolution is disabled."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.handler = SimpleNamespace(vendor="bruker")
        self.recon_names = []
        self.has_dicom_reco = False
        self.display_name = self.path.name
        self.id = 7
        self.type = "<preliminary_type>"

    def _get_vendor_name(self):
        return self.handler.vendor


class _FinalSequence:
    """Sequence object returned by the canonical single-scan loader."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.handler = SimpleNamespace(vendor="bruker")
        self.recon_names = ["minimal"]
        self.has_dicom_reco = False
        self.display_name = self.path.name
        self.id = 7
        self.type = "<final_type>"


def _patch_direct_scan_discovery(monkeypatch, study_module, scan_path):
    detection_calls = []

    def fake_detect_vendor(path):
        normalized_path = Path(path).resolve()
        detection_calls.append(normalized_path)
        return "bruker" if normalized_path == scan_path.resolve() else None

    monkeypatch.setattr(study_module, "detect_vendor", fake_detect_vendor)
    monkeypatch.setattr(
        study_module,
        "get_vendor_registry",
        lambda: SimpleNamespace(get_vendor_info=lambda _vendor: None),
    )
    return detection_calls


def test_study_rejects_non_boolean_smart_load(tmp_path):
    from hypermri.study import Study

    with pytest.raises(TypeError, match="smart_load"):
        Study(tmp_path, verbose=False, smart_load="false")


@pytest.mark.parametrize(
    ("requested_mode", "canonical_mode"),
    [("lazy", "lazy"), ("minimal", "minimal"), (True, "full")],
)
def test_smart_study_uses_canonical_loader_once_with_requested_mode(
    tmp_path, monkeypatch, requested_mode, canonical_mode
):
    """Study should delegate final smart loading to the public scan workflow."""
    import hypermri.study as study_module

    scan_path = tmp_path / "7"
    scan_path.mkdir()
    detection_calls = _patch_direct_scan_discovery(monkeypatch, study_module, scan_path)

    loader_calls = []

    def fake_load_scan(
        path,
        *,
        mode,
        force_vendor,
        log_mode,
        resolve_sequence,
    ):
        loader_calls.append(
            {
                "path": Path(path).resolve(),
                "mode": mode,
                "force_vendor": force_vendor,
                "log_mode": log_mode,
                "resolve_sequence": resolve_sequence,
            }
        )
        return _FinalSequence(path)

    monkeypatch.setattr(study_module, "load_scan", fake_load_scan)

    study = study_module.Study(
        scan_path,
        verbose=False,
        smart_load=True,
        load_data=requested_mode,
        log_mode="debug",
        sort=None,
    )

    assert loader_calls == [
        {
            "path": scan_path.resolve(),
            "mode": canonical_mode,
            "force_vendor": "bruker",
            "log_mode": "debug",
            "resolve_sequence": True,
        }
    ]
    assert detection_calls == [scan_path.resolve()]

    loaded_scan = next(iter(study))
    assert isinstance(loaded_scan, _FinalSequence)
    assert study.dataframe.loc[7, "SequenceClass"] == "_FinalSequence"
    assert study.dataframe.loc[7, "Sequence Type"] == "final_type"


@pytest.mark.parametrize(
    ("load_data", "canonical_mode"),
    [
        (False, "lazy"),
        (True, "full"),
        ("lazy", "lazy"),
        ("minimal", "minimal"),
        ("full", "full"),
    ],
)
def test_non_smart_study_disables_sequence_resolution_on_shared_loader(
    tmp_path, monkeypatch, load_data, canonical_mode
):
    """Non-smart loading keeps generic objects without leaking string modes."""
    import hypermri.study as study_module

    scan_path = tmp_path / "7"
    scan_path.mkdir()
    detection_calls = _patch_direct_scan_discovery(monkeypatch, study_module, scan_path)

    loader_calls = []

    def fake_load_scan(
        path,
        *,
        mode,
        force_vendor,
        log_mode,
        resolve_sequence,
    ):
        loader_calls.append(
            {
                "path": Path(path).resolve(),
                "mode": mode,
                "log_mode": log_mode,
                "force_vendor": force_vendor,
                "resolve_sequence": resolve_sequence,
            }
        )
        return _PreliminaryExperiment(path)

    monkeypatch.setattr(study_module, "load_scan", fake_load_scan)

    study = study_module.Study(
        scan_path,
        verbose=False,
        smart_load=False,
        load_data=load_data,
        log_mode="warning",
        sort=None,
    )

    assert loader_calls == [
        {
            "path": scan_path.resolve(),
            "mode": canonical_mode,
            "log_mode": "warning",
            "force_vendor": "bruker",
            "resolve_sequence": False,
        }
    ]
    assert detection_calls == [scan_path.resolve()]
    assert isinstance(next(iter(study)), _PreliminaryExperiment)
    assert "SequenceClass" not in study.dataframe
    assert "Sequence Type" not in study.dataframe
