"""Load and organize studies without initializing optional notebook display.

Metadata/loading behavior is unchanged; styled display imports IPython on use.
"""

import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from hypermri.analysis.quantitative import ADCFitOptions, T2FitOptions
from hypermri.loading.scans import load_scan, normalize_load_mode
from hypermri.sequences._map_fitting import (
    _fit_summary,
    _select_models,
    _status,
    _validate_options,
)
from hypermri.sequences.base_anatomy import BaseAnatomy
from hypermri.vendors._jcamp import _open_jcamp_text

from .mixins import StandardizedPathMixin
from .utils.system.logging import LOG_MODES as LOG_MODES
from .utils.system.logging import init_default_logger, set_hypermri_log_level
from .vendors import detect_vendor, get_vendor_registry

logger = init_default_logger(__name__)


class StudyError(Exception):
    pass


def _custom_reconstruction_status(scan):
    """Summarize the latest raw reconstruction without loading its samples."""
    outcome = vars(scan).get("_reconstruction_status", {}).get("raw")
    if outcome is not None:
        return outcome
    if getattr(scan, "reconstructions", {}).get("raw") is not None:
        return "Success"
    handler = getattr(scan, "handler", None)
    capability = getattr(scan, "RAW_SOURCE_CAPABILITY", "has_raw_data")
    if handler is not None and getattr(handler, capability, None) is False:
        return "No raw data"
    return "Not run"


def _measurement_clock_time(scan):
    """Format the vendor's acquisition time in its stored timezone."""
    handler = getattr(scan, "handler", None)
    measurement_time = getattr(handler, "measurement_time", None)
    if isinstance(measurement_time, datetime):
        return measurement_time.strftime("%H:%M:%S")
    return "MISSING"


def _acquisition_time_s(scan) -> float | None:
    """Return the time a scan is ordered by, in seconds since 1 January 1970 UTC.

    The vendor handler's ``timestamp`` is used when it is a ``datetime``,
    otherwise its ``measurement_time``.  A ``datetime`` without a time zone is
    read as local time.  None means that the scan has no usable time.
    """
    handler = getattr(scan, "handler", None)
    for name in ("timestamp", "measurement_time"):
        value = getattr(handler, name, None)
        if isinstance(value, datetime):
            return value.timestamp()
    return None


def _sorted_by_acquisition_time(scans: pd.DataFrame) -> pd.DataFrame:
    """Return the scan table with its rows ordered by acquisition time.

    Scans with a time come first, earliest first; scans without a time
    follow.  Scans with equal times, and scans without a time, keep the order
    they have in ``scans``.  The input table is not modified.
    """
    times = [_acquisition_time_s(scan) for scan in scans["ScanObject"]]
    has_no_time = [time is None for time in times]
    seconds = [0.0 if time is None else time for time in times]
    # ``sorted`` is stable, so equal keys keep their row order.
    order = sorted(range(len(times)), key=lambda row: (has_no_time[row], seconds[row]))
    return scans.iloc[order]


# Marks a name that neither a scan nor its vendor handler provides.
_NOT_FOUND = object()


def _vendor_name(scan) -> str:
    """Return the lowercase vendor name of a scan, or ``"unknown"``."""
    vendor = getattr(scan, "vendor", None)
    vendor = getattr(vendor, "value", vendor)  # an enum member gives its text
    return str(vendor).strip().lower() if vendor else "unknown"


def _scan_or_handler_value(scan, name: str):
    """Return ``scan.<name>``, or ``scan.handler.<name>`` if the scan lacks it.

    The order is that of attribute access on a scan, but the vendor handler is
    read explicitly, so no deprecation warning is emitted:

    1. an attribute, property or method of the scan itself;
    2. a canonical parameter in ``scan.params`` (``None`` without a value);
    3. an attribute of ``scan.handler``, for example ``method`` or ``acqp``;
    4. an entry of ``scan.handler.params``.

    ``_NOT_FOUND`` is returned when none of them has ``name``.
    """
    if name in vars(scan) or hasattr(type(scan), name):
        return getattr(scan, name)
    params = getattr(scan, "params", None) or {}
    if name in params:
        return params[name]
    if name in scan.parameter_keys():
        return None
    handler = getattr(scan, "handler", None)
    value = getattr(handler, name, _NOT_FOUND)
    if value is _NOT_FOUND:
        value = (getattr(handler, "params", None) or {}).get(name, _NOT_FOUND)
    return value


def _value_in_parameter_file(parameters, key):
    """Return ``parameters[key]``, or ``_NOT_FOUND`` if the key is absent.

    A tuple ``(name, index)`` selects ``parameters[name][index]``.
    """
    try:
        if isinstance(key, tuple):
            name, index = key
            if name in parameters and hasattr(parameters[name], "__getitem__"):
                return parameters[name][index]
        elif key in parameters:
            return parameters[key]
    except (KeyError, IndexError, TypeError):
        pass
    return _NOT_FOUND


def _parameter_file_names(scan) -> list[str]:
    """Return the names of the vendor handler's parameter dictionaries."""
    attributes = getattr(getattr(scan, "handler", None), "__dict__", {})
    return sorted(
        name
        for name, value in attributes.items()
        if isinstance(value, Mapping) and not name.startswith("_")
    )


class Study(Mapping, StandardizedPathMixin):
    """Load an ordered scan collection and summarize its metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Study directory or supported single-scan path. Relative paths are
        resolved against the working directory.
    keywords : str, tuple or list, optional
        Additional metadata columns accepted by :meth:`add_column` and
        :meth:`add_columns`. None adds only the standard summary columns.
    verbose : bool, optional
        Display the styled scan table after loading. Default is True.
    smart_load : bool, optional
        Resolve registered sequence classes when True (default); otherwise
        retain generic MRIExperiment objects.
    load_data : bool or str, optional
        False or ``"lazy"`` loads metadata only; ``"minimal"`` (default)
        requests the sequence's minimal reconstruction; True or ``"full"``
        loads available samples and requests its full reconstruction policy.
    log_mode : str or int, optional
        Shared package logging threshold, default ``"error"``. Named logging
        modes and numeric levels are accepted; explicit child levels prevail.
    colorize : bool, optional
        Color metadata columns in the styled display. Default is True.
    sort : str or None, optional
        Row order of the scan table. ``"time"`` (default) orders by
        acquisition time, earliest first: the vendor handler's ``timestamp``
        (for Bruker the time ParaVision finished the ``method`` file) or,
        when that is absent, its ``measurement_time``. Scans without a
        usable time follow at the end. ``"e_number"`` orders by experiment
        key, ``"sequence_class"`` by sequence class name and ``"scan_name"``
        by display name. Scans with equal keys keep discovery order in every
        mode. None retains deterministic discovery order.
    auto_fit : bool or list of str or tuple of str, optional
        Keyword-only opt-in quantitative phase. False (default) or an empty
        selection performs no fits; True selects released automatic models
        (ADC and MSME T2). A list/tuple selects named models. Active automation
        requires ``smart_load=True`` and ``load_data="minimal"`` or ``"full"``.
        Uses original seq2d images and existing scientific defaults; no viewers.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Scans indexed by experiment key and their summary columns. ``Custom
        Reco`` reports the latest HyperMRI raw reconstruction outcome:
        ``"Success"``, ``"Failed"``, ``"Not run"`` or ``"No raw data"``.
        Scanner images alone do not imply a successful custom reconstruction.
        The last column, ``Measurement Time``, is the acquisition clock time
        in ``HH:MM:SS`` format, in the vendor's stored timezone, or ``"MISSING"``.
        For Bruker scans this is derived from ``ACQ_time``, never from the
        parameter-file modification timestamp.
        ``Fits`` appears after a requested fit phase and lists computed maps in
        English, including partial maps with finite voxel counts. Skipped/failed
        fits remain in each scan's fit_status and the fit_maps return table.
        An empty cell means no map was computed; this is not a quality assessment.
    load_errors : dict of str to Exception
        Scans that could not be loaded, keyed by scan path as a string, with
        the exception :func:`hypermri.load_scan` raised for them, for example
        a :class:`hypermri.exceptions.ReconstructionError` of a failed
        reconstruction requested by ``load_data``. These scans are not in
        :attr:`dataframe`. Empty when every detected scan loaded.

    Raises
    ------
    TypeError
        If path, smart_load, loading mode or auto_fit has an unsupported type.
    ValueError
        If loading mode/models are unsupported or automation conflicts with
        lazy loading or disabled sequence resolution.
    AttributeError
        If the logging mode is unknown.
    ImportError
        If styled display requires an unavailable optional notebook runtime.

    Notes
    -----
    Discovery and metadata assembly are eager. Lazy loading and summary display
    do not load samples or run reconstruction. Construction changes the shared
    package logging threshold and may display the table. Acquired files are
    not modified. Per-scan loading and automatic reconstruction errors are
    logged and recorded in :attr:`load_errors` while the remaining scans
    continue to load. A reconstruction's
    success records execution, not scientific image quality or acquisition
    completeness. Summary outcomes are refreshed by :meth:`display`.
    Auto-fitting runs eagerly after loading, before display, through fit_maps.
    Per-scan fit errors preserve loaded scans; interrupts propagate immediately.

    Examples
    --------
    >>> study = Study("/data/study", load_data="lazy", verbose=False)
    """

    def __init__(
        self,
        path,
        keywords=None,
        verbose=True,
        smart_load=True,
        load_data="minimal",
        log_mode="error",
        colorize=True,
        sort="time",
        *,
        auto_fit: bool | list[str] | tuple[str, ...] = False,
    ):
        """
        Load and organize scans using the selected loading mode.

        Parameters
        ----------
        path : str or Path
            Study directory containing scans, or a supported single-scan path.
            Relative paths are resolved against the current working directory.
        keywords : str, tuple, list, optional
            Additional metadata columns, using the names and tuple mappings
            accepted by :meth:`add_column` and :meth:`add_columns`.
        verbose : bool, optional
            Whether to display the DataFrame after loading. Default is True.
        smart_load : bool, optional
            Resolve registered sequence classes when True (default); otherwise
            retain generic MRIExperiment objects. This does not select a data
            loading mode.
        load_data : bool or str, optional
            How to load data:
            - True or "full": Load all data
            - False or "lazy": Load only parameters
            - "minimal": Metadata plus the sequence's minimal reconstruction
        log_mode : str or int, optional
            Shared package threshold, default 'error'. Named modes are
            'critical', 'error', 'warning', 'info', 'debug' and 'notset';
            numeric logging levels are also accepted. Explicit child logger
            levels and application handlers remain authoritative.
        colorize : bool, optional
            Whether to colorize the DataFrame display. Default is True.
        sort : str or None, optional
            Sorting method for the DataFrame. Options are:
            - "time": Sort by timestamp (default)
            - "e_number": Sort by experiment number
            - "sequence_class": Sort by sequence class name
            - "scan_name": Sort by scan display name
            - None: No sorting (keep original order)
        auto_fit : bool or list of str or tuple of str, optional
            Keyword-only quantitative fitting after loading; False/empty skips,
            True selects released models (ADC/T2), a list/tuple selects names.
            Requires smart loading and minimal/full mode. Defaults and seq2d
            source follow :meth:`fit_maps`; adds the ``Fits`` summary column.

        Returns
        -------
        None
            Initializes this collection and its scan metadata in place.

        Raises
        ------
        TypeError
            If ``path`` has an unsupported type, ``smart_load`` is not a boolean
            or ``load_data`` has an unsupported type.
        ValueError
            If loading mode/models are unknown or active automation conflicts
            with lazy loading or ``smart_load=False``.
        AttributeError
            If a string ``log_mode`` is not a member of ``LOG_MODES``.
        ImportError
            If ``verbose=True`` requires an unavailable optional notebook runtime.

        Notes
        -----
        Discovery and metadata assembly are eager. Sample loading and automatic
        reconstruction follow ``load_data``; lazy mode does not load samples.
        Construction changes the process-wide package logging threshold and may
        display a table when ``verbose=True``. Explicit logger levels, handlers
        and propagation settings are preserved. Acquired files are not modified.
        Scan discovery is deterministic; ``sort=None`` retains discovery order.
        ``Custom Reco`` summarizes the latest raw reconstruction attempt;
        ``Measurement Time`` is the last column and uses the acquisition clock
        time in the stored timezone. Neither column starts reconstruction.

        Examples
        --------
        >>> study = Study("/data/study", load_data="lazy", verbose=False)
        """
        selected_models = _select_models(auto_fit, automatic=True)
        if not isinstance(smart_load, bool):
            raise TypeError("smart_load must be a boolean.")
        load_mode = normalize_load_mode(load_data)
        if selected_models and load_mode == "lazy":
            raise ValueError(
                'Active auto_fit needs scanner images; use load_data="minimal" '
                'or "full", or call study.fit_maps(...) later after lazy loading.'
            )
        if selected_models and not smart_load:
            raise ValueError(
                "Active auto_fit requires sequence-specific fitting; use smart_load=True."
            )
        # Initialize mixin first
        super().__init__()

        self.log_mode = log_mode  # Store log_mode as instance variable

        set_hypermri_log_level(log_mode)

        # Use standardized path handling
        self.path = path

        self.load_errors: dict[str, Exception] = {}
        self.__scans = self._read_in_measurements(
            load_data=load_mode,
            smart_load=smart_load,
        )

        self.new_column_from_scanobj(
            "Reconstructions", lambda x: x.recon_names if x.recon_names else "MISSING"
        )
        self.new_column_from_scanobj(
            "DICOM",
            lambda x: (
                "Yes"
                if getattr(getattr(x, "handler", x), "has_dicom_reco", False)
                else "No"
            ),
        )
        self.new_column_from_scanobj("Scan Name", lambda x: x.display_name)

        self.date = self._get_subject_date()
        self.subject = {}
        self.subject["subject_name"] = self._get_subject_name()
        self.subject["subject_dob"] = self._get_subject_dob()
        self.subject["subject_weight"] = self._get_subject_weight()
        self.subject["subject_type"] = self._get_subject_type()
        self.subject["rf_coils"] = self._get_used_coils()

        if isinstance(keywords, (str, tuple)):
            self.add_column(keywords)

        if isinstance(keywords, list):
            self.add_columns(keywords)

        # Every mode is stable: rows with equal keys keep discovery order.
        if sort == "time":
            self.__scans = _sorted_by_acquisition_time(self.__scans)
        elif sort == "e_number":
            self.__scans = self.__scans.sort_index(kind="stable")
        elif sort == "sequence_class" and "SequenceClass" in self.__scans.columns:
            self.__scans = self.__scans.sort_values(by="SequenceClass", kind="stable")
        elif sort == "scan_name" and "Scan Name" in self.__scans.columns:
            self.__scans = self.__scans.sort_values(by="Scan Name", kind="stable")

        if selected_models:
            self.fit_maps(models=selected_models)
        self._update_summary_columns()
        if verbose:
            self.display(colorize=colorize)

    def __getitem__(self, key):
        if key in self.__scans.index:
            return self.__scans.loc[(key, "ScanObject")]
        else:
            raise StudyError(f"{key} is not a valid key.")

    def __iter__(self):
        return iter(self.__scans["ScanObject"])

    def __len__(self):
        return len(self.__scans)

    def __str__(self):
        return str(self.__scans)

    def fit_maps(
        self,
        models: list[str] | tuple[str, ...] | None = None,
        *,
        adc_options: ADCFitOptions | None = None,
        t2_options: T2FitOptions | None = None,
    ) -> pd.DataFrame:
        """Fit quantitative scanner maps sequentially and summarize each outcome.

        Parameters
        ----------
        models : list of str or tuple of str or None, optional
            None selects all released models (ADC/T2); list/tuple selects names.
            Duplicates collapse in order; an empty selection performs no fits.
            ADC supports Bruker DTI/DTIEPI series with one documented direction,
            original seq2d images and exact physical b-value mapping in s/mm².
        adc_options : ADCFitOptions or None, optional
            Finite nonnegative signal threshold and existing voxel execution
            settings.
            None retains scientific defaults. Physical axes are scan-owned:
            leave b-values unset and dynamic_dim unset or ``"bvalue_idx"``.
        t2_options : T2FitOptions or None, optional
            Finite nonnegative MSME magnitude decay threshold and shared voxel
            worker settings.
            Echo times in seconds must be established by the selected scanner
            reconstruction or acquisition metadata; T2 maps use ms. None uses
            the approved no-offset signal-space fit with threshold 1e-5.
            New T2 fits have a fixed upper bound of 5 s (5000 ms).

        Returns
        -------
        pandas.DataFrame
            Columns ``scan_key``, ``model``, ``state``, ``reason``,
            ``valid_voxels``, ``total_voxels``. Counts are None without numerical
            results. Datasets remain solely on each scan in ``fit_results``;
            ADC maps use mm²/s and retain all non-dynamic input dimensions.

        Raises
        ------
        TypeError, ValueError
            Invalid selections/options fail before any scan access/result changes.
            Scan-specific exceptions are recorded/logged and remaining scans
            continue. KeyboardInterrupt and SystemExit are never caught.

        Notes
        -----
        Eager and sequential across scans, including after lazy loading. Each
        scan delegates to its fit_maps method, which may load seq2d once, without
        raw reconstruction, viewers or input mutation. Refit replaces requested
        results; other models remain. The Fits column lists computed/partial maps
        in English and is empty when none were computed. Skips/failures remain
        in scan.fit_status and the returned table.
        Finite values indicate execution only, without a quality/uncertainty claim.
        After replacing images, explicitly fit again. No Study result cache exists.

        Examples
        --------
        >>> outcomes = study.fit_maps(["adc"], adc_options=ADCFitOptions(
        ...     signal_threshold=1e-5))  # doctest: +SKIP
        """
        selected = _select_models(models)
        _validate_options(adc_options, t2_options)
        rows = []
        for key, scan in self.__scans["ScanObject"].items():
            if not selected:
                break
            if isinstance(scan, BaseAnatomy):
                try:
                    scan.fit_maps(
                        models=selected, adc_options=adc_options, t2_options=t2_options
                    )
                except Exception as exc:
                    logger.error("Map fitting failed for scan %s: %s", key, exc)
            else:
                statuses = vars(scan).setdefault("_fit_status", {})
                for model in selected:
                    statuses[model] = _status(
                        "skipped",
                        "Sequence has no supported anatomical map fitting path.",
                    )
            statuses = vars(scan).get("_fit_status", {})
            for model in selected:
                status = statuses.get(
                    model,
                    _status("skipped", "Not attempted after a preceding fit error."),
                )
                rows.append({"scan_key": key, "model": model, **status})
        summaries = self.__scans["ScanObject"].apply(_fit_summary)
        if "Fits" in self.__scans:
            self.__scans["Fits"] = summaries
        else:
            position = (
                self.__scans.columns.get_loc("Measurement Time")
                if "Measurement Time" in self.__scans
                else len(self.__scans.columns)
            )
            self.__scans.insert(position, "Fits", summaries)
        columns = [
            "scan_key",
            "model",
            "state",
            "reason",
            "valid_voxels",
            "total_voxels",
        ]
        return pd.DataFrame(rows, columns=columns, dtype=object)

    @property
    def dataframe(self):
        return self.__scans

    @dataframe.setter
    def dataframe(self, df):
        if isinstance(df, pd.DataFrame):
            self.__scans = df
        else:
            raise StudyError("Study.dataframe must be a pandas DataFrame object.")

    def colorize_col(self, col, df_styler=None, opacity=0.5):
        """Add background colors to one summary column.

        Parameters
        ----------
        col : str
            Summary column to style. ``Custom Reco`` uses green for success,
            violet for missing raw data and red for failure. ``DICOM`` uses
            green for ``Yes`` and red for ``No``. Other columns use a
            categorical rainbow palette.
        df_styler : pandas.io.formats.style.Styler, optional
            Styler to update in place. None creates a styler for the current
            summary DataFrame.
        opacity : float, optional
            Background opacity in the inclusive range [0, 1], default 0.5.

        Returns
        -------
        pandas.io.formats.style.Styler or None
            Styler with deferred cell formatting. Returns None if values in
            an ordinary categorical column cannot be collected as a set.

        Raises
        ------
        AssertionError
            If opacity is outside [0, 1].

        Notes
        -----
        Status colors are independent of which outcomes occur in the study.
        In status columns, ``Not run``, unknown statuses and ``MISSING`` have
        no background color.
        Styling does not change summary values, load samples or reconstruct.
        """
        assert opacity <= 1 and opacity >= 0, "Opacity must be in range [0, 1]"

        if df_styler is None:
            df_styler = self.__scans.style

        status_colors = {
            "Custom Reco": {
                "Success": "#8DEACB",
                "No raw data": "#7f00ff",
                "Failed": "#ff0000",
            },
            "DICOM": {"Yes": "#8DEACB", "No": "#ff0000"},
        }.get(col)
        if status_colors is not None:

            def color_status(value):
                color = status_colors.get(value)
                return (
                    f"background-color: {color}{int(opacity * 255):02x};"
                    if color
                    else ""
                )

            try:
                return df_styler.map(color_status, subset=[col])
            except AttributeError:
                return df_styler.applymap(color_status, subset=[col])

        # if the column item is not hashable, abort
        try:
            values = set(self.__scans[col].values)
        except TypeError:
            return

        # Remove 'MISSING' from values for color assignment
        values = values - {"MISSING"}
        if not values:
            return df_styler

        # Create rainbow colormap with consistent ordering

        try:
            from matplotlib.pyplot import get_cmap
        except ImportError:
            from matplotlib.cm import get_cmap

        # Sort values to ensure consistent color assignment regardless of DataFrame order
        # Use a predefined order for common sequence types to create a logical color progression
        sequence_priority = {
            "RARE": 0,  # Anatomical - violet/blue
            "FLASH": 1,  # Anatomical - blue
            "PRESS": 2,  # Spectroscopy - green/cyan
            "CSI": 3,  # Spectroscopy - yellow/orange
            "Fieldmap": 4,  # Utility - orange
            "Singlepulse": 5,  # Utility - red
        }

        # Sort values using priority order, with alphabetical fallback for unknown sequences
        def get_sort_key(value):
            return (sequence_priority.get(value, 1000), value)

        sorted_values = sorted(values, key=get_sort_key)
        n_colors = len(sorted_values)

        if n_colors == 0:
            return df_styler

        # Generate rainbow colors spanning full spectrum (violet to red)
        rainbow = get_cmap("rainbow")
        # Always use full rainbow range (0.0 to 1.0) regardless of number of colors
        if n_colors == 1:
            colors = [rainbow(0.5)]  # Use middle color for single item
        else:
            # Distribute colors evenly across the full rainbow spectrum
            colors = [rainbow(i / (n_colors - 1)) for i in range(n_colors)]

        # Convert to hex with opacity
        hex_colors = []
        for color in colors:
            # Convert RGBA to hex with opacity
            r, g, b, _ = color
            hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255), int(opacity * 255)
            )
            hex_colors.append(hex_color)

        # generate color look up table with consistent ordering
        color_lut = {k: v for k, v in zip(sorted_values, hex_colors, strict=False)}
        color_lut["MISSING"] = ""

        def f(v):
            return f"background-color: {color_lut.get(v, '')};"

        try:
            return df_styler.map(f, subset=[col])
        except AttributeError:
            return df_styler.applymap(f, subset=[col])

    def remove(self, scan=None):
        """Remove one scan from this study's scan table.

        Parameters
        ----------
        scan : int or None, optional
            Experiment key of the scan, as shown in the index of
            :attr:`dataframe`. None (default) removes nothing.

        Returns
        -------
        None
            The scan table of this study is replaced by one without the
            scan; the order of the remaining scans is unchanged.

        Raises
        ------
        AssertionError
            If ``scan`` is neither None nor an integer.

        Warns
        -----
        UserWarning
            If no scan with this key is in the study; nothing is removed.

        Notes
        -----
        Only the entry in this study is removed. Files on disk and the scan
        object itself are not modified.

        Examples
        --------
        >>> study = Study("/data/study", load_data="lazy", verbose=False)
        >>> study.remove(3)
        """
        if scan is None:
            return None

        assert isinstance(scan, int), "scan_num has to be an integer"

        if scan in self.__scans.index:
            self.__scans = self.__scans.drop(index=scan)
        else:
            warnings.warn(f"scan {scan} does not exist in scans.", stacklevel=2)

    def _update_summary_columns(self):
        """Refresh outcomes and keep acquisition clock time as the last column."""
        scans = self.__scans["ScanObject"]
        if "Fits" in self.__scans:
            self.__scans["Fits"] = scans.apply(_fit_summary)
        outcomes = scans.apply(_custom_reconstruction_status)
        if "Custom Reco" in self.__scans:
            self.__scans["Custom Reco"] = outcomes
        else:
            position = self.__scans.columns.get_loc("Reconstructions") + 1
            self.__scans.insert(position, "Custom Reco", outcomes)
        if "Measurement Time" in self.__scans:
            self.__scans.pop("Measurement Time")
        self.__scans["Measurement Time"] = scans.apply(_measurement_clock_time)

    def display(self, colorize=True):
        """Display scan metadata as a styled table in an IPython frontend.

        Parameters
        ----------
        colorize : bool, optional
            Color metadata columns, default True. Custom reconstruction
            success and available DICOM images are green, failed reconstruction
            and unavailable DICOM images red, and missing raw data violet.

        Returns
        -------
        None
            Refreshes the reconstruction outcome and acquisition clock-time
            columns, then displays an HTML table without loading scan samples.
            ``Measurement Time`` remains last even after adding custom columns.

        Raises
        ------
        ImportError
            If IPython is missing from the environment.
        """
        from ._optional import _import_optional

        display = _import_optional("IPython.display").display
        self._update_summary_columns()
        styles = [
            # tabular headers
            {"selector": "th", "props": [("font-size", "109%")]},
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        # Reset index to make "Key" a regular column, then hide the new numeric index
        df_styler = (
            self.__scans.reset_index().style.set_table_styles(styles).hide(axis="index")
        )

        if self.date:
            date_string = self.date.strftime("%A %d. %B %Y  %H:%M:%S")

            # Add subject information if available
            subject_info = []
            if self.subject.get("subject_name"):
                subject_info.append(f"Subject: {self.subject['subject_name']}")
            if self.subject.get("subject_type"):
                subject_info.append(f"Type: {self.subject['subject_type']}")
            if self.subject.get("subject_weight"):
                subject_info.append(f"Weight: {self.subject['subject_weight']} g")
            if self.subject.get("subject_dob"):
                subject_info.append(f"DOB: {self.subject['subject_dob']}")

            caption = date_string
            if subject_info:
                caption += "<br>" + " | ".join(subject_info)

            df_styler = df_styler.set_caption(caption)

        if colorize:
            # skip first three columns, try to colorize all others
            for col in self.__scans.columns[3:]:
                self.colorize_col(col, df_styler, opacity=0.3)

        def color_MISSING(cell_value):
            # cell_value == "MISSING" raises a warning if cell_value is an
            # array
            if isinstance(cell_value, str) and cell_value == "MISSING":
                return "color: red"
            return ""

        try:
            df_styler = df_styler.map(color_MISSING)
        except AttributeError:
            df_styler = df_styler.applymap(color_MISSING)

        # Format ScanObject to show visible class name instead of hidden <Object> repr
        df_styler = df_styler.format({"ScanObject": lambda x: x.__class__.__name__})

        # Hide ScanObject column as it is redundant with SequenceClass
        try:
            df_styler.hide(axis="columns", subset=["ScanObject"])
        except AttributeError:
            # Fallback for older pandas versions
            df_styler.hide_columns(["ScanObject"])

        display(df_styler)

    def get_available_keys(self):
        keys = []
        for key in self.__scans.index:
            keys.append(key)
        return keys

    def new_column_from_scanobj(self, colname, func):
        self.__scans[colname] = self.__scans["ScanObject"].apply(func)

    def add_column(self, keyword, paramfile="method", column_name=None):
        """Add one column with a parameter of every scan to the study table.

        For each scan the value is looked up in this order; the first value
        that is not ``None`` is used:

        1. If ``keyword`` is a function, its result is the value.
        2. If ``keyword`` is a string: the attribute of that name of the scan
           (for example the canonical parameter ``repetition_time_ms``), or,
           if the scan has no such attribute, of its vendor handler
           ``scan.handler``.
        3. The entry in the vendor's parameter files named by ``paramfile``,
           searched in the given order.

        A scan without a value gets the text ``"MISSING"``. Values are stored
        as found; no unit is converted.

        Parameters
        ----------
        keyword : str, tuple, dict or callable
            The parameter to read:

            - str: an attribute name or a key of a parameter file, for example
              ``"repetition_time_ms"`` or ``"PVM_RepetitionTime"``.
            - tuple ``(key, index)``: one element of an array-valued entry of
              a parameter file, for example ``("EffectiveTE", 0)``.
            - dict: the key per vendor, for example
              ``{"bruker": "PVM_RepetitionTime", "agilent": "tr"}``. Vendor
              names are lowercase. A scan of a vendor that is not listed uses
              the entry ``"default"`` or, without it, the first entry. The
              chosen key is only searched in the parameter files.
            - callable: a function that takes the scan and returns the value.
              If it raises, the scan gets ``"MISSING"``.

        paramfile : str, list of str or dict, optional
            Name of the vendor handler attribute that holds the parameter
            file, for example ``"method"`` or ``"acqp"`` for Bruker. A list is
            searched in order. A dict gives the name per vendor; a vendor that
            is not listed uses the entry ``"default"`` or ``"method"``.
        column_name : str, optional
            Name of the new column. By default the ``keyword`` string, the
            entry ``"column_name"`` of a ``keyword`` dict, or ``"Parameter"``.
            An existing column of this name is replaced.

        Raises
        ------
        ValueError
            If the value cannot exist in any scan of the study: ``keyword``
            is not an attribute of any scan or handler and no scan has one of
            the requested parameter files, for example because ``paramfile``
            is misspelled or belongs to another vendor. The message lists the
            parameter files of the first scan. The table is left unchanged.

        Notes
        -----
        The table is modified in place. Parameter files are read from
        ``scan.handler``; no raw data are loaded and nothing is reconstructed
        unless ``keyword`` names a scan property that does so.

        Examples
        --------
        >>> study.add_column("repetition_time_ms")  # doctest: +SKIP
        >>> study.add_column("PVM_EncOrder", paramfile="method")  # doctest: +SKIP
        >>> study.add_column(("EffectiveTE", 0), column_name="TE 1")  # doctest: +SKIP
        >>> study.add_column("NR", paramfile=["method", "acqp"])  # doctest: +SKIP
        >>> study.add_column(
        ...     {"bruker": "PVM_RepetitionTime", "agilent": "tr"},
        ...     paramfile={"bruker": "method", "agilent": "procpar"},
        ...     column_name="TR",
        ... )  # doctest: +SKIP
        >>> study.add_column(
        ...     lambda scan: scan.handler.method["PVM_NAverages"],
        ...     column_name="Averages",
        ... )  # doctest: +SKIP
        """
        if column_name is None:
            if isinstance(keyword, str):
                column_name = keyword
            elif isinstance(keyword, dict):
                column_name = keyword.get("column_name", "Parameter")
            else:
                column_name = "Parameter"

        scans = list(self.__scans["ScanObject"])
        values = []
        # True once any scan could hold the value at all.
        value_can_exist = callable(keyword) or not scans
        for scan in scans:
            if callable(keyword):
                try:
                    values.append(keyword(scan))
                except Exception:
                    values.append("MISSING")
                continue

            vendor = _vendor_name(scan)
            if isinstance(keyword, dict):
                key = keyword.get(
                    vendor, keyword.get("default", next(iter(keyword.values())))
                )
            else:
                key = keyword
            if isinstance(paramfile, dict):
                file_names = [paramfile.get(vendor, paramfile.get("default", "method"))]
            elif isinstance(paramfile, list):
                file_names = paramfile
            else:
                file_names = [paramfile]

            value = _NOT_FOUND
            if isinstance(keyword, str):
                value = _scan_or_handler_value(scan, keyword)
                value_can_exist = value_can_exist or value is not _NOT_FOUND
            if value is _NOT_FOUND or value is None:
                value = _NOT_FOUND
                for file_name in file_names:
                    parameters = _scan_or_handler_value(scan, file_name)
                    if parameters is _NOT_FOUND or parameters is None:
                        continue
                    value_can_exist = True
                    value = _value_in_parameter_file(parameters, key)
                    if value is not _NOT_FOUND:
                        break
            values.append("MISSING" if value is _NOT_FOUND else value)

        if not value_can_exist:
            first = scans[0]
            raise ValueError(
                f"Study.add_column: cannot fill the column {column_name!r}. "
                f"{key!r} is not an attribute of any scan, and no scan has a "
                f"parameter file named {file_names} (option 'paramfile'). The "
                f"first scan ({_vendor_name(first)}) has the parameter files "
                f"{_parameter_file_names(first)}. Pass one of them as "
                "'paramfile', or pass a function of the scan as 'keyword'."
            )

        self.__scans[column_name] = values

    def add_columns(self, keyword_list, paramfile="method", column_names=None):
        """Add one column per keyword to the study table.

        Each keyword is passed to :meth:`add_column` in list order.

        Parameters
        ----------
        keyword_list : list
            Keywords in any form that :meth:`add_column` accepts (string,
            tuple, dict or callable).
        paramfile : str, list of str or dict, optional
            Parameter file or files searched for every keyword, as in
            :meth:`add_column`.
        column_names : list of str, optional
            Column name per keyword, by position. Keywords beyond the end of
            this list, or all of them when it is ``None``, get the default
            name of :meth:`add_column`.

        Raises
        ------
        StudyError
            If ``keyword_list`` is not a list.
        ValueError
            If :meth:`add_column` cannot fill a column; the columns of the
            earlier keywords have then already been added.

        Examples
        --------
        >>> study.add_columns(
        ...     ["repetition_time_ms", ("EffectiveTE", 0)],
        ...     column_names=["TR [ms]", "TE 1 [ms]"],
        ... )  # doctest: +SKIP
        """
        if not isinstance(keyword_list, list):
            raise StudyError("keyword_list must be of type 'list'.")

        for i, keyword in enumerate(keyword_list):
            column_name = None
            if column_names and i < len(column_names):
                column_name = column_names[i]
            self.add_column(keyword, paramfile, column_name)

    def _get_subject_name(self):
        """Retrieve the SubjectName of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with _open_jcamp_text(subject_fname) as f:
                for line in f:
                    if line.startswith("##$SUBJECT_name_string"):
                        # Read the next line
                        subject_name = next(f).strip()[1:-1]
                        return subject_name
        return None

    def _get_subject_dob(self):
        """Retrieve the Subject's date of birth our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with _open_jcamp_text(subject_fname) as f:
                for line in f:
                    if line.startswith("##$SUBJECT_dbirth"):
                        # Read the next line
                        subject_dob = next(f).strip()[1:-1]
                        return subject_dob
        return None

    def _get_subject_weight(self):
        """Retrieve the Subject's weight of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with _open_jcamp_text(subject_fname) as f:
                for line in f:
                    if line.startswith("##$SUBJECT_weight"):
                        # Read the next line
                        subject_weight = line.split("=")[1].strip()
                        # convert to gram:
                        subject_weight = 1000 * float(subject_weight)
                        return subject_weight
        return None

    def _get_subject_date(self):
        """Retrieve the SubjectDate of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with _open_jcamp_text(subject_fname) as f:
                for line in f:
                    if line.startswith("##$SUBJECT_date"):
                        date = line.split("=")[1].strip()
            return datetime.strptime(date, "<%Y-%m-%dT%H:%M:%S,%f%z>")
        return None

    def _get_subject_type(self):
        """Retrieve the Subject's date of birth of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with _open_jcamp_text(subject_fname) as f:
                for line in f:
                    if line.startswith("##$SUBJECT_type"):
                        # Read the next line
                        subject_weight = line.split("=")[1].strip()
                        return subject_weight
        return None

    def _get_used_coils(self):
        """Aggregate unique receive/transmit coils from all scans."""
        unique_coils = set()

        if self.__scans is None or self.__scans.empty:
            return None

        for scan in self.__scans["ScanObject"]:
            # Get receive coils
            rx = getattr(scan, "coil_receive", None)
            if rx:
                if isinstance(rx, (list, np.ndarray)):
                    for c in rx:
                        unique_coils.add(str(c))
                else:
                    unique_coils.add(str(rx))

            # Get transmit coils
            tx = getattr(scan, "coil_transmit", None)
            if tx:
                if isinstance(tx, (list, np.ndarray)):
                    for c in tx:
                        unique_coils.add(str(c))
                else:
                    unique_coils.add(str(tx))

        # Filter out empty strings or None-like strings
        unique_coils = {c for c in unique_coils if c and c != "None" and c != "MISSING"}

        if not unique_coils:
            return None

        return ", ".join(sorted(list(unique_coils)))

    def _read_in_measurements(
        self,
        load_data=True,
        db=False,
        *,
        smart_load=False,
    ) -> pd.DataFrame:
        """Discover scans and load each one through the selected public path.

        Parameters
        ----------
        load_data : bool or str
            Loading mode passed to the selected loader.
        db : bool
            Log each loaded handler for debugging.
        smart_load : bool
            Resolve registered sequence classes. When false, return generic
            ``MRIExperiment`` objects through the same loading path.

        Returns
        -------
        out: pandas.DataFrame
            Returns a DataFrame containing all scans. The scans are saved in the
            row 'ScanObject' with their corresponding E-Number as index.

        Notes
        -----
        A scan whose loading raises is left out and its exception is stored
        in :attr:`load_errors` under its path.
        """
        # Find child paths that can represent vendor scans. Keep a vendor found
        # during discovery so the same path is not detected a second time.
        candidate_entries = []
        if self.path.is_file():
            vendor_name = detect_vendor(self.path)
            if vendor_name:
                candidate_entries = [(self.path, vendor_name)]
        elif self.path.is_dir():
            direct_vendor = detect_vendor(self.path)
            if direct_vendor:
                candidate_entries = [(self.path, direct_vendor)]
            else:
                for child_path in self.path.iterdir():
                    if child_path.is_dir():
                        candidate_entries.append((child_path, None))
                    elif child_path.is_file():
                        vendor_name = detect_vendor(child_path)
                        if vendor_name:
                            candidate_entries.append((child_path, vendor_name))
        registry = get_vendor_registry()
        valid_scan_entries = []

        for scan_path, detected_vendor in candidate_entries:
            vendor_name = detected_vendor or detect_vendor(scan_path)
            if not vendor_name:
                logger.debug("Skipping %s: no vendor detected", scan_path)
                continue

            handler_info = registry.get_vendor_info(vendor_name)
            handler_class = handler_info.get("handler_class") if handler_info else None

            if handler_class and hasattr(handler_class, "is_valid_scan_dir"):
                try:
                    if not handler_class.is_valid_scan_dir(scan_path):
                        logger.debug(
                            "Skipping %s: %s handler rejected directory",
                            scan_path,
                            vendor_name,
                        )
                        continue
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning(
                        "Skipping %s: %s validation raised %s",
                        scan_path,
                        vendor_name,
                        exc,
                    )
                    continue

            valid_scan_entries.append((scan_path, vendor_name))

        if not valid_scan_entries:
            logger.warning(
                "No valid scans detected in %s. Check directory structure or vendor support.",
                self.path,
            )
            return pd.DataFrame(columns=["ScanObject"])

        # Load every validated entry. Progress belongs to the batch operation,
        # while ``load_scan`` remains a reusable single-scan function.
        all_scans = []
        entries = valid_scan_entries
        if smart_load:
            entries = tqdm(entries, leave=False, desc="Loading scans")

        for scan_path, vendor_name in entries:
            try:
                experiment = load_scan(
                    scan_path,
                    mode=load_data,
                    force_vendor=vendor_name,
                    log_mode=getattr(self, "log_mode", "error"),
                    resolve_sequence=smart_load,
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialise experiment for %s (%s): %s",
                    scan_path,
                    vendor_name,
                    exc,
                )
                self.load_errors[str(scan_path)] = exc
                continue

            all_scans.append(experiment)

        # use the scan experiment number as index/key for our scans. In case
        # none was found, we assign a unique negative number.
        # Additionally, we check wether the key is already assigned (e.g. when
        # the scan folder has been duplicated). In that case we assign a unique
        # negative number, too.
        if not all_scans:
            logger.warning(
                "No valid experiments could be initialised from %s", self.path
            )
            return pd.DataFrame(columns=["ScanObject"])

        keys = []
        key_types = set()
        cntr = -1
        for scan in all_scans:
            if db:
                logger.info(f"-> Scan.handler: {scan.handler}")

            raw_id = getattr(scan, "id", None)
            normalized_key = None

            if isinstance(raw_id, (int, float)) and raw_id == raw_id:
                try:
                    normalized_int = int(raw_id)
                except (TypeError, ValueError):
                    normalized_int = None
                if normalized_int is not None and normalized_int > 0:
                    normalized_key = normalized_int
            elif isinstance(raw_id, str):
                stripped = raw_id.strip()
                if stripped.isdigit():
                    normalized_key = int(stripped)
                elif stripped:
                    normalized_key = stripped

            if normalized_key is None or normalized_key in keys:
                if normalized_key in keys:
                    msg = f"Scan '{scan.path}' has duplicate key {normalized_key}."
                else:
                    msg = f"Scan '{scan.path}' has no valid key."
                logger.warning(msg + f" -> Assign {cntr} as key instead.")
                normalized_key = cntr
                cntr -= 1

            keys.append(normalized_key)
            key_types.add(type(normalized_key))

        # create a DataFrame with 'keys' as index and scan objects as first entry
        df = pd.DataFrame(
            all_scans, index=pd.Index(keys, name="Key"), columns=["ScanObject"]
        )

        # Sort only when pandas can compare all index values safely
        if len(key_types) == 1:
            try:
                df = df.sort_index()
            except TypeError:
                logger.debug(
                    "Index contains values that cannot be compared reliably; skipping sort."
                )
        else:
            logger.debug(
                "Detected heterogeneous index types %s; keeping insertion order.",
                key_types,
            )

        if smart_load:
            df["SequenceClass"] = [scan.__class__.__name__ for scan in df["ScanObject"]]
            df["Sequence Type"] = [
                scan.type.replace("<", "").replace(">", "")
                if isinstance(getattr(scan, "type", None), str)
                else getattr(scan, "type", None)
                for scan in df["ScanObject"]
            ]

        return df
