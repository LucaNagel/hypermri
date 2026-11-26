from __future__ import annotations

from pathlib import Path


class _DummyHandler:
    vendor = "siemens"


class _DummyExperiment:
    def __init__(
        self,
        path,
        load_data=False,
        log_mode="error",
        force_vendor=None,
    ):
        self.path = Path(path).resolve()
        self.handler = _DummyHandler()
        self.recon_names = []
        self.has_dicom_reco = False
        self.display_name = self.path.name
        self._id = 104
        self._type = r"%CustomerSeq%\csi_fid_10mm"

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type


def _load_dummy_scan(path, *, mode, force_vendor, log_mode, resolve_sequence):
    assert resolve_sequence is False
    return _DummyExperiment(
        path,
        load_data=mode == "full",
        log_mode=log_mode,
        force_vendor=force_vendor,
    )


def test_study_discovers_top_level_siemens_dat_file(tmp_path, monkeypatch):
    import hypermri.study as study_module

    monkeypatch.setattr(study_module, "load_scan", _load_dummy_scan)

    study_root = tmp_path / "Siemens study"
    study_root.mkdir()
    dat_file = study_root / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"")
    (study_root / "DICOMS").mkdir()
    (study_root / "notes.ipynb").write_text("{}", encoding="utf-8")

    study = study_module.Study(
        study_root,
        verbose=False,
        smart_load=False,
        load_data=False,
        sort=None,
    )

    assert len(study) == 1
    assert next(iter(study)).path == dat_file.resolve()


def test_study_accepts_direct_siemens_dat_file(tmp_path, monkeypatch):
    import hypermri.study as study_module

    monkeypatch.setattr(study_module, "load_scan", _load_dummy_scan)

    dat_file = tmp_path / "meas_MID00104_FID52379_csi.dat"
    dat_file.write_bytes(b"")

    study = study_module.Study(
        dat_file,
        verbose=False,
        smart_load=False,
        load_data=False,
        sort=None,
    )

    assert len(study) == 1
    assert next(iter(study)).path == dat_file.resolve()
