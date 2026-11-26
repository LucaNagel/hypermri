"""Map canonical sequence identifiers to maintained Python sequence types.

Notes
-----
The sequence registry imports this table when loading external settings.
Importing class definitions constructs no sequences, reads no acquired samples,
and starts no reconstruction or viewer. Categories, scanner aliases and vendor
scopes belong to the validated sequence YAML file, which names no classes.
"""

from .B1Calibration import B1Calibration as _B1Calibration
from .BSSFP import BSSFP as _BSSFP
from .CPMG import CPMG as _CPMG
from .CSI import CSI as _CSI
from .DTI import DTI as _DTI
from .DTIEPI import DTIEPI as _DTIEPI
from .DTISPIRAL import DTISPIRAL as _DTISPIRAL
from .EPI import EPI as _EPI
from .FAIR_RARE import FAIR_RARE as _FAIR_RARE
from .fexi import FEXI as _FEXI
from .Fieldmap import Fieldmap as _Fieldmap
from .FLASH import FLASH as _FLASH
from .FLOWMAP import FLOWMAP as _FLOWMAP
from .GeoffSlcSpec import GeoffSlcSpec as _GeoffSlcSpec
from .gNSPECT_MS import gNSPECT_MS as _gNSPECT_MS
from .ISIS import ISIS as _ISIS
from .IVRecovery import IVRecovery as _IVRecovery
from .MDEFT import MDEFT as _MDEFT
from .MGE import MGE as _MGE
from .MSME import MSME as _MSME
from .NSPECT import NSPECT as _NSPECT
from .PRESS import PRESS as _PRESS
from .RARE import RARE as _RARE
from .RAREVTR import RAREVTR as _RAREVTR
from .RFPROFILE import RFPROFILE as _RFPROFILE
from .semiLASER import semiLASER as _semiLASER
from .Singlepulse import Singlepulse as _Singlepulse
from .SpinEcho import SpinEcho as _SpinEcho
from .SPIRAL import SPIRAL as _SPIRAL
from .STEAM import STEAM as _STEAM
from .UTE import UTE as _UTE
from .ZTE import ZTE as _ZTE

_SEQUENCE_TYPES: dict[str, type] = {
    "flash": _FLASH,
    "bssfp": _BSSFP,
    "b1calibration": _B1Calibration,
    "dtiepi": _DTIEPI,
    "epi": _EPI,
    "fair_rare": _FAIR_RARE,
    "fexi": _FEXI,
    "fieldmap": _Fieldmap,
    "mge": _MGE,
    "rarevtr": _RAREVTR,
    "rare": _RARE,
    "spin_echo": _SpinEcho,
    "csi": _CSI,
    "press": _PRESS,
    "steam": _STEAM,
    "singlepulse": _Singlepulse,
    "gnspect_ms": _gNSPECT_MS,
    "nspect": _NSPECT,
    "geoffslcspec": _GeoffSlcSpec,
    "ivrecovery": _IVRecovery,
    "semilaser": _semiLASER,
    "cpmg": _CPMG,
    "msme": _MSME,
    "mdeft": _MDEFT,
    "flowmap": _FLOWMAP,
    "dti": _DTI,
    "spiral": _SPIRAL,
    "dtispiral": _DTISPIRAL,
    "ute": _UTE,
    "rfprofile": _RFPROFILE,
    "isis": _ISIS,
    "zte": _ZTE,
}
