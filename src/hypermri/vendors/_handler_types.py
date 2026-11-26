"""Map canonical vendor identifiers to their maintained Python handler types.

Notes
-----
The vendor registry imports this table during configuration loading. Importing
handler definitions reads no acquired samples and constructs no handlers.
Detection patterns and their priority remain external settings in the validated
vendor YAML file, which names no classes.
"""

from hypermri.vendors.topspin import BrukerTopSpinHandler as _BrukerTopSpinHandler

from .agilent import AgilentHandler as _AgilentHandler
from .bruker import BrukerHandler as _BrukerHandler
from .external import ExternalDataHandler as _ExternalDataHandler
from .magritek import MagritekHandler as _MagritekHandler
from .siemens import SiemensHandler as _SiemensHandler

_VENDOR_HANDLERS: dict[str, type] = {
    "bruker": _BrukerHandler,
    "bruker_topspin": _BrukerTopSpinHandler,
    "siemens": _SiemensHandler,
    "agilent": _AgilentHandler,
    "magritek": _MagritekHandler,
    "external": _ExternalDataHandler,
}
