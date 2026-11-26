"""FLASH gradient-echo sequence wrapper for anatomical imaging."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

DB_MSG = False  # Change this to False to hide detailed error messages


class FLASH(BaseAnatomy):
    """Fast low-angle shot gradient-echo sequence for anatomical imaging."""

    # Standard FLASH uses fid; IntraGate FLASH uses rawdata.job0.
    RAW_SOURCE_CAPABILITY = "has_raw_data"

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
