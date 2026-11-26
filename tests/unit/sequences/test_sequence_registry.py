from hypermri.sequence_mapping import method_name_to_sequence_id
from hypermri.sequences.BSSFP import BSSFP
from hypermri.sequences.fexi import FEXI
from hypermri.sequences.FLASH import FLASH
from hypermri.sequences.gNSPECT_MS import gNSPECT_MS
from hypermri.sequences.MGE import MGE
from hypermri.sequences.NSPECT import NSPECT
from hypermri.sequences.PRESS import PRESS
from hypermri.sequences.RAREVTR import RAREVTR
from hypermri.sequences.registry import get_sequence_registry
from hypermri.sequences.semiLASER import semiLASER
from hypermri.sequences.SpinEcho import SpinEcho


def test_registry_resolves_legacy_smart_load_aliases():
    registry = get_sequence_registry()

    assert registry.get_sequence_class("<User:lucaFLASH_1D>") is FLASH
    assert registry.get_sequence_class("<User:fisp_grad_1metab_CS>") is BSSFP
    assert registry.get_sequence_class("<User:wg_mv_press_mk4>") is PRESS
    assert registry.get_sequence_class("<User:multislice_EPSI>") is MGE
    assert registry.get_sequence_class("<Bruker:FEXI>") is FEXI


def test_registry_prefers_exact_aliases_before_broad_patterns():
    registry = get_sequence_registry()

    assert registry.get_sequence_class("<Bruker:RAREVTR>") is RAREVTR
    assert registry.get_sequence_class("<User:gNSPECT>") is NSPECT
    assert registry.get_sequence_class("<User:gNSPECT_MS>") is gNSPECT_MS
    assert registry.get_sequence_class("<User:wg_press_adiab_mk4>") is semiLASER


def test_registry_resolves_prefixed_siemens_svs_sequence():
    registry = get_sequence_registry()

    entry = registry.resolve_sequence(r"%SiemensSeq%\svs_se", vendor="siemens")

    assert entry is not None
    assert entry["sequence_class"] is PRESS
    assert entry["sequence_name"] == "press"


def test_registry_resolves_prefixed_siemens_spin_echo_sequence():
    registry = get_sequence_registry()

    entry = registry.resolve_sequence(r"%SiemensSeq%\se", vendor="siemens")

    assert entry is not None
    assert entry["sequence_class"] is SpinEcho
    assert entry["sequence_name"] == "spin_echo"


def test_reconstruction_aliases_use_sequence_registry():
    assert method_name_to_sequence_id("<User:sliceprofile_FLASHv3>") == "flash"
    assert method_name_to_sequence_id("<User:wg_press_adiab_mk7>") == "semilaser"
    assert method_name_to_sequence_id("agilent_spectroscopy") == "nspect"
    assert method_name_to_sequence_id("<Bruker:FEXI>") == "fexi"
    assert method_name_to_sequence_id("unregistered_method") is None
