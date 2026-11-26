"""Shared sequence base class.

This module centralizes the construction of a sequence-specific object from
the vendor handler of one scan. Concrete sequence classes should focus on
sequence-specific helpers and metadata, while reconstruction loading policy is
applied by callers via :meth:`run_default_reconstructions`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from hypermri.loading.contracts import SequenceCategory
from hypermri.loading.experiment import MRIExperiment
from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.loading.scans import normalize_load_mode

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__)


class BaseSequence(MRIExperiment):
    """Base class of the sequence-specific experiment classes.

    A sequence object is an ``MRIExperiment`` built from the vendor handler of
    one scan; it never constructs a second handler for the same path.
    :func:`hypermri.loading.scans.load_scan` constructs the handler once and
    passes it to the registered sequence class.

    Parameters
    ----------
    handler : VendorHandlerProtocol
        Vendor handler already constructed for the scan.
    **kwargs
        Loading, logging, reconstruction, path and vendor settings documented in
        the constructor. Unknown names are rejected.

    Raises
    ------
    TypeError
        If ``handler`` is an ``MRIExperiment`` or a keyword is unknown.
    ValueError
        If a supplied handler uses an unsupported load_data construction setting.

    Notes
    -----
    Construction is lazy and initializes neither a viewer nor plotting state.
    Reconstruction policy is applied explicitly by the loading owner. Handler
    logging settings are updated; acquired samples and input metadata are retained.
    """

    RECONSTRUCTION_POLICY: ClassVar[dict[str, tuple[str, ...]] | None] = None
    RAW_SOURCE_CAPABILITY: ClassVar[str] = "has_raw_data"
    _CONSTRUCTION_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "load_data",
            "log_mode",
            "reco_options",
            "path",
            "force_vendor",
            "handler_options",
        }
    )

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence object from the vendor handler of its scan.

        Parameters
        ----------
        handler : VendorHandlerProtocol
            Vendor handler already constructed for the scan. It is used as it
            is: no vendor is detected and no handler is constructed.
        **kwargs
            Construction options. Only the keys below are accepted.

            ``load_data`` : bool or {"lazy", "minimal", "full"}, default False
                Loading mode requested by the caller and stored as
                ``_sequence_requested_load_data``.  It starts no
                reconstruction; see :meth:`run_default_reconstructions`.  The
                mode ``"full"`` (or ``True``) means that the handler was
                constructed with ``load_data=True``; its loaded payload then
                becomes the primary data.
            ``log_mode`` : str, default "error"
                Logging verbosity.
            ``reco_options`` : dict, optional
                Reconstruction options stored as ``_sequence_reco_options``.
            ``path`` : str or pathlib.Path, optional
                Scan location stored on the object; defaults to the handler's
                ``path``.
            ``force_vendor`` : str, optional
                Vendor name stored on the object; it only affects the reported
                vendor.
            ``handler_options`` : dict, optional
                Handler options stored on the object.

        Raises
        ------
        TypeError
            If ``handler`` is an ``MRIExperiment`` (pass ``scan.handler``
            instead) or if a keyword is not one of the construction options.
        ValueError
            If the handler uses an unsupported load_data setting.

        Notes
        -----
        The handler gets its ``log_mode`` attribute set to the given
        ``log_mode``. No viewer, plotting state or widgets are created.

        Examples
        --------
        >>> from hypermri.vendors import get_vendor_handler
        >>> handler = get_vendor_handler("path/to/scan")  # doctest: +SKIP
        >>> scan = FLASH(handler, load_data="lazy")  # doctest: +SKIP
        """
        if isinstance(handler, MRIExperiment):
            raise TypeError(
                f"{type(self).__name__} is constructed from a vendor handler, "
                "not from an MRIExperiment; pass scan.handler instead, or load "
                "the scan with hypermri.load_scan."
            )
        unknown = sorted(set(kwargs) - self._CONSTRUCTION_KEYWORDS)
        if unknown:
            accepted = ", ".join(sorted(self._CONSTRUCTION_KEYWORDS))
            raise TypeError(
                f"{type(self).__name__} got unknown construction keyword(s) "
                f"{', '.join(map(repr, unknown))}; supported keywords are: "
                f"{accepted}."
            )
        self._init_from_handler(handler, kwargs)

    def _init_from_handler(
        self, handler: VendorHandlerProtocol, options: dict[str, Any]
    ) -> None:
        """Initialize this object as an ``MRIExperiment`` of ``handler``."""
        load_data = options.pop("load_data", False)
        self._sequence_requested_load_data = load_data
        self._sequence_log_mode = options.pop("log_mode", "error")
        self._sequence_reco_options = dict(options.pop("reco_options", {}) or {})
        path = options.pop("path", None)
        MRIExperiment.__init__(
            self,
            getattr(handler, "path", None) if path is None else path,
            load_data=normalize_load_mode(load_data) == "full",
            log_mode=self._sequence_log_mode,
            force_vendor=options.pop("force_vendor", None),
            handler_options=options.pop("handler_options", None),
            handler=handler,
        )

    def run_default_reconstructions(
        self,
        load_data: bool | str = "minimal",
        reco_options: Mapping[str, Any] | None = None,
        category: SequenceCategory | str | None = None,
    ) -> None:
        """Run the central reconstruction policy for this sequence.

        Parameters
        ----------
        load_data
            Study/factory loading mode. ``False`` and ``"lazy"`` do not run
            automatic reconstructions.
        reco_options
            Options merged into each reconstruction call.
        category
            Optional registry category. Used only when the class does not define
            its own ``RECONSTRUCTION_POLICY``.

        Raises
        ------
        ValueError
            If ``load_data`` names an unsupported loading mode.
        ReconstructionError
            If a policy reconstruction fails. Later reconstructions of the
            policy are not attempted, results stored before the failure are
            kept, and the failure is recorded as the latest outcome of that
            reconstruction type.

        Notes
        -----
        Reconstructions already stored on the scan are not repeated. A raw
        reconstruction is skipped, with an info log entry, when the vendor
        reports no non-empty raw data source.
        """
        mode = normalize_load_mode(load_data)
        policy = self.get_reconstruction_policy(category)
        reco_types = policy.get(mode, ())
        if not reco_types:
            return

        options = dict(reco_options or {})
        options.pop("minimal_loading", None)

        for reco_type in reco_types:
            if self._reconstruction_available(reco_type):
                continue
            if reco_type == "raw" and not self._raw_reconstruction_available():
                logger.info(
                    "%s raw reconstruction skipped because no non-empty raw "
                    "data source is available.",
                    self.__class__.__name__,
                )
                continue
            self.reconstruct(reco_type=reco_type, **options)

    @classmethod
    def get_reconstruction_policy(
        cls, category: SequenceCategory | str | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Return default reconstructions by normalized loading mode."""
        if cls.RECONSTRUCTION_POLICY is not None:
            return cls.RECONSTRUCTION_POLICY

        category_value = cls._category_value(category)
        if category_value == SequenceCategory.SPECTROSCOPY.value:
            return {
                "lazy": (),
                "minimal": ("raw",),
                "full": ("raw",),
            }

        return {
            "lazy": (),
            "minimal": ("seq2d",),
            "full": ("seq2d", "raw"),
        }

    @staticmethod
    def _category_value(category: SequenceCategory | str | None) -> str | None:
        if isinstance(category, SequenceCategory):
            return category.value
        if category is None:
            return None
        return str(category).lower()

    def _reconstruction_available(self, reco_type: str) -> bool:
        reco_type = reco_type.lower()
        # Handler-loaded arrays are not necessarily strategy reconstructions;
        # Bruker full loads populate scanner_data from raw 2dseq before the
        # coordinate-aware seq2d strategy has run.
        return reco_type in getattr(self, "reconstructions", {})

    def _raw_reconstruction_available(self) -> bool:
        """Consult an optional vendor capability before automatic raw work."""
        handler = getattr(self, "handler", None)
        if handler is None:
            return False

        capability = self.RAW_SOURCE_CAPABILITY
        try:
            availability = getattr(handler, capability)
        except (AttributeError, OSError):
            # Vendors without an explicit capability retain historical behavior.
            return True
        return bool(availability() if callable(availability) else availability)
