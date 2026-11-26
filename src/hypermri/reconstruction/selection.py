"""Choose the reconstruction strategy and its default options for a scan.

One Python table, :data:`RECONSTRUCTIONS`, owns which strategy class
reconstructs a (vendor, sequence, kind) combination and with which default
options. :func:`find_reconstruction` is the only lookup.
:meth:`hypermri.loading.experiment.MRIExperiment.reconstruct` merges the
defaults with the caller's options, rejects options the strategy does not
accept (see ``accepted_options`` on each strategy class) and runs the strategy.

Notes
-----
Importing this module imports the strategy classes but constructs no strategy,
reads no data and starts no reconstruction. The loading code imports it only
when a reconstruction is requested, so lazy and metadata-only loading do not
import the strategies.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from hypermri.sequence_mapping import method_name_to_sequence_id

from .strategies.agilent import AgilentSpectroRecoStrategy
from .strategies.base import BaseRecoStrategy
from .strategies.bruker.bssfp import BrukerBssfpRawRecoStrategy
from .strategies.bruker.csi import BrukerCsiRecoStrategy
from .strategies.bruker.dti import BrukerDtiRawRecoStrategy
from .strategies.bruker.epi import BrukerEpiRawRecoStrategy
from .strategies.bruker.epsi import BrukerEpsiRawRecoStrategy
from .strategies.bruker.igflash import BrukerIgflashRawRecoStrategy
from .strategies.bruker.rare import BrukerRareRawRecoStrategy
from .strategies.bruker.raw import BrukerMGERawRecoStrategy, BrukerRawRecoStrategy
from .strategies.bruker.seq2d import Brukerseq2dRecoStrategy
from .strategies.bruker.spectroscopy import BrukerSpectroRecoStrategy
from .strategies.magritek import MagritekSpectroRecoStrategy
from .strategies.siemens import SiemensRawRecoStrategy

__all__ = ["RECONSTRUCTIONS", "find_reconstruction"]

_NONE: Mapping[str, Any] = MappingProxyType({})

# Imaging storage precision stays explicit; CSI and
# spectroscopy keep the float16 storage default of their strategies.
_FLOAT32: Mapping[str, Any] = MappingProxyType({"data_precision": "float32"})

# Raw data that HyperMRI does not reconstruct; the scanner images are used.
_SKIP: Mapping[str, Any] = MappingProxyType({"no_reconstruction": True})
_RUN: Mapping[str, Any] = MappingProxyType({"no_reconstruction": False})

# Single-voxel Bruker spectroscopy: the FID is transformed without the
# imaging line-padding removal, k-space sorting or FOV phase ramp.
_BRUKER_SPECTRUM: Mapping[str, Any] = MappingProxyType(
    {
        "reshape_fid": False,
        "postprocess_fid": False,
        "set_kspace_xarray": True,
        "shift_xarray": False,
        "set_data": True,
    }
)

# Agilent spectra are returned unprocessed apart from the receiver
# root-sum-of-squares combination.
_AGILENT_SPECTRUM: Mapping[str, Any] = MappingProxyType(
    {
        "remove_dc": False,
        "apply_digital_filter": False,
        "apodization": "off",
        "combine_channels": True,
        "combine_method": "rss",
        "retain_channel_axis": False,
        "apply_phasing": False,
    }
)

RECONSTRUCTIONS: Mapping[
    tuple[str, str, str], tuple[type[BaseRecoStrategy], Mapping[str, Any]]
] = MappingProxyType(
    {
        ("bruker", "flash", "raw"): (BrukerRawRecoStrategy, _FLOAT32),
        ("bruker", "flash", "seq2d"): (Brukerseq2dRecoStrategy, _FLOAT32),
        # IntraGate FLASH stores imaging profiles in rawdata.job0, not fid.
        ("bruker", "<bruker:igflash>", "raw"): (BrukerIgflashRawRecoStrategy, _FLOAT32),
        ("bruker", "bssfp", "raw"): (BrukerBssfpRawRecoStrategy, _FLOAT32),
        ("bruker", "bssfp", "seq2d"): (Brukerseq2dRecoStrategy, _FLOAT32),
        # This FISP method applies its FOV offsets with the opposite sign.
        ("bruker", "<user:fisp_grad_1metab_cs>", "raw"): (
            BrukerRawRecoStrategy,
            MappingProxyType({"data_precision": "float32", "fov_shift_sign": -1.0}),
        ),
        ("bruker", "b1calibration", "raw"): (BrukerRawRecoStrategy, _NONE),
        ("bruker", "b1calibration", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "cpmg", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "cpmg", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "dtiepi", "raw"): (BrukerEpiRawRecoStrategy, _FLOAT32),
        ("bruker", "dtiepi", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "epi", "raw"): (BrukerEpiRawRecoStrategy, _FLOAT32),
        ("bruker", "epi", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "spspepi", "raw"): (BrukerEpiRawRecoStrategy, _FLOAT32),
        ("bruker", "spspepi", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "fair_rare", "raw"): (BrukerRareRawRecoStrategy, _NONE),
        ("bruker", "fair_rare", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "fexi", "raw"): (BrukerRawRecoStrategy, _NONE),
        ("bruker", "fexi", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "rare", "raw"): (BrukerRareRawRecoStrategy, _FLOAT32),
        ("bruker", "rare", "seq2d"): (Brukerseq2dRecoStrategy, _FLOAT32),
        ("bruker", "geoffslcspec", "raw"): (BrukerRawRecoStrategy, _NONE),
        ("bruker", "geoffslcspec", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "gnspect_ms", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "gnspect_ms", "seq2d"): (Brukerseq2dRecoStrategy, _SKIP),
        ("bruker", "ivrecovery", "raw"): (BrukerRawRecoStrategy, _NONE),
        ("bruker", "ivrecovery", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "mge", "raw"): (BrukerMGERawRecoStrategy, _NONE),
        # EPSI interleaves NSegments echo trains into one spectral FID.
        ("bruker", "<bruker:epsi>", "raw"): (BrukerEpsiRawRecoStrategy, _NONE),
        ("bruker", "mge", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "nspect", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "nspect", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "csi", "raw"): (BrukerCsiRecoStrategy, _NONE),
        ("bruker", "csi", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "press", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "press", "seq2d"): (Brukerseq2dRecoStrategy, _SKIP),
        ("bruker", "rarevtr", "raw"): (BrukerRareRawRecoStrategy, _NONE),
        ("bruker", "rarevtr", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "semilaser", "raw"): (BrukerRawRecoStrategy, _NONE),
        ("bruker", "semilaser", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "singlepulse", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "singlepulse", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        ("bruker", "steam", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "steam", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),
        # The scanner's fieldmap images already contain the B0 calculation.
        ("bruker", "fieldmap", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "fieldmap", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        # Acquisition order: flow encoding, slices, spatial encoding, repetitions.
        ("bruker", "flowmap", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "flowmap", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        # ISIS stores its eight inversion accumulations in rawdata.job0, fid and ser.
        ("bruker", "isis", "raw"): (BrukerSpectroRecoStrategy, _BRUKER_SPECTRUM),
        ("bruker", "isis", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "rfprofile", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "rfprofile", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "msme", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "msme", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "mdeft", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "mdeft", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "dti", "raw"): (BrukerDtiRawRecoStrategy, _FLOAT32),
        ("bruker", "dti", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        # Spiral, UTE and ZTE raw data need non-Cartesian regridding.
        ("bruker", "spiral", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "spiral", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "dtispiral", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "dtispiral", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "ute", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "ute", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "zte", "raw"): (BrukerRawRecoStrategy, _SKIP),
        ("bruker", "zte", "seq2d"): (Brukerseq2dRecoStrategy, _RUN),
        ("bruker", "default", "raw"): (BrukerRawRecoStrategy, _FLOAT32),
        ("bruker", "default", "seq2d"): (Brukerseq2dRecoStrategy, _FLOAT32),
        ("agilent", "nspect", "raw"): (AgilentSpectroRecoStrategy, _AGILENT_SPECTRUM),
        ("agilent", "default", "raw"): (AgilentSpectroRecoStrategy, _AGILENT_SPECTRUM),
        ("magritek", "singlepulse", "raw"): (MagritekSpectroRecoStrategy, _NONE),
        ("magritek", "default", "raw"): (MagritekSpectroRecoStrategy, _NONE),
        ("siemens", "default", "raw"): (SiemensRawRecoStrategy, _NONE),
    }
)
"""Strategy class and default options per (vendor, sequence, kind).

Keys are lower case. ``vendor`` is ``"bruker"``, ``"agilent"``,
``"magritek"`` or ``"siemens"``. ``sequence`` is a canonical sequence ID of
``sequences/config/sequence_mapping.yaml``, an exact raw method name such as
``"<bruker:igflash>"`` that overrides its canonical sequence, or ``"default"``
for every sequence of the vendor without its own entry. ``kind`` is ``"raw"``
(reconstruct the acquired raw data) or ``"seq2d"`` (read the scanner's own
reconstruction). The default options are read-only; units and meaning are
documented by the strategy's ``reconstruct`` method, and each name is in the
strategy's ``accepted_options``.
"""


def find_reconstruction(
    vendor: str, sequence: str | None, kind: str
) -> tuple[type[BaseRecoStrategy], dict[str, Any]] | None:
    """Return the strategy class and default options for one reconstruction.

    The lookup tries, in this order, the entry of ``sequence`` as given (so an
    exact method entry overrides its canonical sequence), the entry of the
    canonical sequence ID that
    :func:`~hypermri.sequence_mapping.method_name_to_sequence_id` returns,
    and the vendor's ``"default"`` entry. No other fallback applies.

    Parameters
    ----------
    vendor : str
        Vendor name, compared case-insensitively.
    sequence : str or None
        Raw method name or canonical sequence ID, compared case-insensitively.
        ``None`` selects the vendor's default entry.
    kind : str
        ``"raw"`` or ``"seq2d"``, compared case-insensitively.

    Returns
    -------
    tuple or None
        The strategy class and a new, mutable copy of its default options, or
        ``None`` if neither the sequence nor the vendor default has an entry.

    Raises
    ------
    ConfigurationError
        If the packaged sequence mapping cannot be read while resolving
        ``sequence``.

    Notes
    -----
    The lookup constructs no strategy and reads no data. Repeated calls give
    the same result.

    Examples
    --------
    >>> strategy_class, defaults = find_reconstruction("bruker", "FLASH", "raw")
    >>> strategy_class.__name__, defaults
    ('BrukerRawRecoStrategy', {'data_precision': 'float32'})
    """
    vendor, kind = vendor.lower(), kind.lower()
    candidates = []
    if sequence is not None:
        candidates.append(sequence.lower())
        canonical = method_name_to_sequence_id(sequence)
        if canonical is not None:
            candidates.append(canonical)
    candidates.append("default")
    for candidate in candidates:
        entry = RECONSTRUCTIONS.get((vendor, candidate, kind))
        if entry is not None:
            strategy_class, defaults = entry
            return strategy_class, dict(defaults)
    return None
