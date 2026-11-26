"""Strategy selection: identity, default options, order, lookup and options.

``_EXPECTED`` is the effective table of the former YAML registry (73 entries,
in file order), so these tests show that the Python table selects the same
strategy classes with the same default options.
"""

import importlib
import subprocess
import sys

import pytest

from hypermri import load_scan
from hypermri.exceptions import ReconstructionError
from hypermri.reconstruction.selection import RECONSTRUCTIONS, find_reconstruction
from hypermri.reconstruction.strategies.base import BaseRecoStrategy
from hypermri.sequence_mapping import method_name_to_sequence_id

_EXPECTED = {
    "bruker.flash.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.flash.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.<bruker:igflash>.raw": (
        "hypermri.reconstruction.strategies.bruker.igflash.BrukerIgflashRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.bssfp.raw": (
        "hypermri.reconstruction.strategies.bruker.bssfp.BrukerBssfpRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.bssfp.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.<user:fisp_grad_1metab_cs>.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"data_precision": "float32", "fov_shift_sign": -1.0},
    ),
    "bruker.b1calibration.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {},
    ),
    "bruker.b1calibration.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.cpmg.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.cpmg.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.dtiepi.raw": (
        "hypermri.reconstruction.strategies.bruker.epi.BrukerEpiRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.dtiepi.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.epi.raw": (
        "hypermri.reconstruction.strategies.bruker.epi.BrukerEpiRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.epi.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.spspepi.raw": (
        "hypermri.reconstruction.strategies.bruker.epi.BrukerEpiRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.spspepi.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.fair_rare.raw": (
        "hypermri.reconstruction.strategies.bruker.rare.BrukerRareRawRecoStrategy",
        {},
    ),
    "bruker.fair_rare.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.fexi.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {},
    ),
    "bruker.fexi.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.rare.raw": (
        "hypermri.reconstruction.strategies.bruker.rare.BrukerRareRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.rare.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.geoffslcspec.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {},
    ),
    "bruker.geoffslcspec.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.gnspect_ms.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.gnspect_ms.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.ivrecovery.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {},
    ),
    "bruker.ivrecovery.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.mge.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerMGERawRecoStrategy",
        {},
    ),
    "bruker.<bruker:epsi>.raw": (
        "hypermri.reconstruction.strategies.bruker.epsi.BrukerEpsiRawRecoStrategy",
        {},
    ),
    "bruker.mge.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.nspect.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.nspect.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.csi.raw": (
        "hypermri.reconstruction.strategies.bruker.csi.BrukerCsiRecoStrategy",
        {},
    ),
    "bruker.csi.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.press.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.press.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.rarevtr.raw": (
        "hypermri.reconstruction.strategies.bruker.rare.BrukerRareRawRecoStrategy",
        {},
    ),
    "bruker.rarevtr.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.semilaser.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {},
    ),
    "bruker.semilaser.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.singlepulse.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.singlepulse.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.steam.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.steam.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {},
    ),
    "bruker.fieldmap.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.fieldmap.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.flowmap.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.flowmap.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.isis.raw": (
        "hypermri.reconstruction.strategies.bruker.spectroscopy.BrukerSpectroRecoStrategy",
        {
            "reshape_fid": False,
            "postprocess_fid": False,
            "set_kspace_xarray": True,
            "shift_xarray": False,
            "set_data": True,
        },
    ),
    "bruker.isis.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.rfprofile.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.rfprofile.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.msme.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.msme.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.mdeft.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.mdeft.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.dti.raw": (
        "hypermri.reconstruction.strategies.bruker.dti.BrukerDtiRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.dti.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.spiral.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.spiral.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.dtispiral.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.dtispiral.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.ute.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.ute.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.zte.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"no_reconstruction": True},
    ),
    "bruker.zte.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"no_reconstruction": False},
    ),
    "bruker.default.raw": (
        "hypermri.reconstruction.strategies.bruker.raw.BrukerRawRecoStrategy",
        {"data_precision": "float32"},
    ),
    "bruker.default.seq2d": (
        "hypermri.reconstruction.strategies.bruker.seq2d.Brukerseq2dRecoStrategy",
        {"data_precision": "float32"},
    ),
    "agilent.nspect.raw": (
        "hypermri.reconstruction.strategies.agilent.AgilentSpectroRecoStrategy",
        {
            "remove_dc": False,
            "apply_digital_filter": False,
            "apodization": "off",
            "combine_channels": True,
            "combine_method": "rss",
            "retain_channel_axis": False,
            "apply_phasing": False,
        },
    ),
    "agilent.default.raw": (
        "hypermri.reconstruction.strategies.agilent.AgilentSpectroRecoStrategy",
        {
            "remove_dc": False,
            "apply_digital_filter": False,
            "apodization": "off",
            "combine_channels": True,
            "combine_method": "rss",
            "retain_channel_axis": False,
            "apply_phasing": False,
        },
    ),
    "magritek.singlepulse.raw": (
        "hypermri.reconstruction.strategies.magritek.MagritekSpectroRecoStrategy",
        {},
    ),
    "magritek.default.raw": (
        "hypermri.reconstruction.strategies.magritek.MagritekSpectroRecoStrategy",
        {},
    ),
    "siemens.default.raw": (
        "hypermri.reconstruction.strategies.siemens.SiemensRawRecoStrategy",
        {},
    ),
}


def _class(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _key(key):
    return tuple(key.split(".", 2))


def test_table_lists_the_expected_entries_in_order():
    assert list(RECONSTRUCTIONS) == [_key(key) for key in _EXPECTED]


@pytest.mark.parametrize("key", _EXPECTED)
def test_each_entry_keeps_strategy_class_and_default_options(key):
    cls, options = _EXPECTED[key]
    strategy_class, defaults = RECONSTRUCTIONS[_key(key)]
    assert strategy_class is _class(cls)
    assert dict(defaults) == options
    found = find_reconstruction(*_key(key))
    assert found == (_class(cls), options)


@pytest.mark.parametrize("key", _EXPECTED)
def test_default_options_are_accepted_by_their_strategy(key):
    strategy_class, defaults = RECONSTRUCTIONS[_key(key)]
    assert issubclass(strategy_class, BaseRecoStrategy)
    assert set(defaults) <= strategy_class.accepted_options


@pytest.mark.parametrize("key", _EXPECTED)
def test_keys_name_a_known_sequence_and_kind(key):
    vendor, sequence, kind = _key(key)
    assert vendor in {"bruker", "agilent", "magritek", "siemens"}
    assert kind in {"raw", "seq2d"}
    assert sequence == "default" or method_name_to_sequence_id(sequence) is not None


def test_default_options_cannot_be_changed_through_the_table_or_a_lookup():
    _, defaults = RECONSTRUCTIONS[("bruker", "flash", "raw")]
    with pytest.raises(TypeError):
        defaults["data_precision"] = "float64"
    _, copy = find_reconstruction("bruker", "flash", "raw")
    copy["data_precision"] = "float64"
    assert find_reconstruction("bruker", "flash", "raw")[1] == {
        "data_precision": "float32"
    }


def test_exact_method_override_alias_vendor_default_and_missing_entry():
    fisp = find_reconstruction("bruker", "<User:fisp_grad_1metab_CS>", "raw")
    alias = find_reconstruction("bruker", "<User:fisp_grad_1metab_CS2>", "raw")
    default = find_reconstruction("bruker", "invented_unknown", "raw")
    assert fisp == (
        _class(_EXPECTED["bruker.<user:fisp_grad_1metab_cs>.raw"][0]),
        {"data_precision": "float32", "fov_shift_sign": -1.0},
    )
    assert alias == (
        _class(_EXPECTED["bruker.bssfp.raw"][0]),
        {"data_precision": "float32"},
    )
    assert default == (
        _class(_EXPECTED["bruker.default.raw"][0]),
        {"data_precision": "float32"},
    )
    assert find_reconstruction("BRUKER", "<Bruker:FLASH>", "RAW") == (
        _class(_EXPECTED["bruker.flash.raw"][0]),
        {"data_precision": "float32"},
    )
    assert find_reconstruction("bruker", None, "raw") == default
    assert find_reconstruction("invented_vendor", "flash", "raw") is None
    assert find_reconstruction("siemens", "flash", "seq2d") is None


def test_lookup_constructs_no_strategy(monkeypatch):
    cls = _class(_EXPECTED["bruker.default.raw"][0])
    calls = []
    monkeypatch.setattr(cls, "__init__", lambda self: calls.append(self))
    find_reconstruction("bruker", "flash", "raw")
    assert calls == []


def test_scanner_fov_shift_is_accepted_only_by_the_generic_raw_strategy():
    accepting = {
        strategy_class.__name__
        for strategy_class, _ in RECONSTRUCTIONS.values()
        if "fov_shift_source" in strategy_class.accepted_options
    }
    assert accepting == {"BrukerRawRecoStrategy"}


def test_package_import_does_not_import_the_strategies():
    script = (
        "import sys, hypermri, hypermri.reconstruction; "
        "assert 'hypermri.reconstruction.selection' not in sys.modules; "
        "assert 'hypermri.reconstruction.strategies.bruker.raw' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_reconstruct_rejects_an_unknown_option_before_reading_data(
    synthetic_flash_scan, monkeypatch
):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    strategy_class, _ = find_reconstruction("bruker", scan.type, "raw")

    def must_not_run(self, handler, **options):
        raise AssertionError("strategy ran despite an unknown option")

    monkeypatch.setattr(strategy_class, "reconstruct", must_not_run)

    with pytest.raises(ReconstructionError) as info:
        scan.reconstruct("raw", data_precison="float64")
    message = str(info.value)
    assert "data_precison" in message
    assert strategy_class.__name__ in message
    assert "data_precision" in message
    assert scan.reconstructions == {}


def test_reconstruct_passes_defaults_and_overrides_to_a_new_strategy(
    synthetic_flash_scan, monkeypatch
):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    strategy_class, defaults = find_reconstruction("bruker", scan.type, "seq2d")
    received = []

    def record(self, handler, **options):
        received.append((self, options))

    monkeypatch.setattr(strategy_class, "reconstruct", record)

    scan.reconstruct("seq2d", recon_num=1)
    scan.reconstruct("seq2d")

    assert received[0][1] == {**defaults, "recon_num": 1}
    assert received[1][1] == defaults
    assert received[0][0] is not received[1][0]
