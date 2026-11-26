"""Static Matplotlib colorbars, colormaps and subplot layout helpers.

These operations modify figures only when explicitly called. Slice-coordinate
matching is owned by hypermri.utils.image.slices and re-exported here solely
for compatibility with the former combined plot-utility namespace.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from hypermri.utils.image.slices import get_overlapping_slices


def _add_inset_axes_outline(ax, **kwargs):
    """Draw an axes outline fully inside the axes bounds."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    line_kwargs = {
        "color": "black",
        "linewidth": plt.rcParams["axes.linewidth"],
        "solid_capstyle": "butt",
        "zorder": 10,
    }
    line_kwargs.update(kwargs)
    linewidth = line_kwargs.get("linewidth", line_kwargs.get("lw"))
    half_linewidth_inches = float(linewidth) / 2 / 72
    points_to_display = ax.figure.dpi_scale_trans

    segments = (
        ([0, 0], [0, 1], half_linewidth_inches, 0),
        ([1, 1], [0, 1], -half_linewidth_inches, 0),
        ([0, 1], [0, 0], 0, half_linewidth_inches),
        ([0, 1], [1, 1], 0, -half_linewidth_inches),
    )
    lines = []
    for x, y, dx, dy in segments:
        transform = ax.transAxes + ScaledTranslation(dx, dy, points_to_display)
        line = Line2D(x, y, transform=transform, clip_on=False, **line_kwargs)
        ax.add_artist(line)
        lines.append(line)
    return lines


def _hyphen_minus_scalar_formatter():
    """Return a linear formatter that writes negative numbers with a hyphen.

    Matplotlib's default ``ScalarFormatter`` replaces ``-`` by the Unicode minus
    sign U+2212, which many fonts draw much wider than a hyphen.
    """
    from matplotlib.ticker import ScalarFormatter

    class _HyphenMinusScalarFormatter(ScalarFormatter):
        @staticmethod
        def fix_minus(s):
            return s

    return _HyphenMinusScalarFormatter()


def _tick_label_decimals(
    decimal_places: int | None,
    significant_digits: int | None = None,
) -> int:
    """Return the number of digits after the decimal point of tick labels.

    An explicit ``decimal_places`` wins. Otherwise a mantissa between 1 and 10
    with ``significant_digits`` significant digits needs
    ``significant_digits - 1`` decimals.
    """
    if decimal_places is not None:
        if decimal_places < 0:
            msg = f"decimal_places must be at least 0, got {decimal_places}"
            raise ValueError(msg)
        return decimal_places
    if significant_digits is None or significant_digits < 1:
        msg = f"scientific_digits must be at least 1, got {significant_digits}"
        raise ValueError(msg)
    return significant_digits - 1


def _zero_tick_label(decimal_places: int | None) -> str:
    """Return the label of a zero tick.

    Zero stays ``"0"`` unless ``decimal_places`` is given explicitly; then it is
    padded like the other labels, for example ``"0.00"``.
    """
    if decimal_places is None:
        return "0"
    return f"{0.0:.{decimal_places}f}"


def _fixed_decimal_tick_formatter(decimal_places: int):
    """Return a linear formatter with a fixed number of decimals.

    Negative values keep an ASCII hyphen, like
    ``_hyphen_minus_scalar_formatter``. Values that round to zero are written
    without a sign, for example ``0.0`` instead of ``-0.0``.
    """
    from matplotlib.ticker import FuncFormatter

    decimals = _tick_label_decimals(decimal_places)

    def _format_tick(value, _position):
        rounded_value = round(float(value), decimals)
        if rounded_value == 0.0:
            rounded_value = 0.0
        return f"{rounded_value:.{decimals}f}"

    return FuncFormatter(_format_tick)


def _scientific_colorbar_tick_formatter(
    significant_digits: int = 2,
    decimal_places: int | None = None,
    use_latex: bool = True,
):
    """Return a formatter for ticks in scientific notation.

    With ``use_latex=True`` the labels are Matplotlib mathtext such as
    ``$1.3\\times 10^{-2}$``. With ``use_latex=False`` they are plain text such
    as ``1.3×10^-2``, which avoids mathtext entirely. ``decimal_places``, when
    given, fixes the mantissa decimals and overrides ``significant_digits``.
    """
    from matplotlib.ticker import FuncFormatter

    decimals = _tick_label_decimals(decimal_places, significant_digits)
    zero_label = _zero_tick_label(decimal_places)

    def _format_tick(value, _position):
        if np.isclose(value, 0.0):
            return zero_label

        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        rounded_mantissa = round(mantissa, decimals)

        # Rounding can push a value like 9.99 to 10.0. Renormalize in that case.
        if np.isclose(abs(rounded_mantissa), 10.0):
            rounded_mantissa /= 10.0
            exponent += 1

        if use_latex:
            return rf"${rounded_mantissa:.{decimals}f}\times 10^{{{exponent}}}$"
        return f"{rounded_mantissa:.{decimals}f}×10^{exponent}"

    return FuncFormatter(_format_tick)


def _scaled_colorbar_tick_formatter(
    exponent: int,
    significant_digits: int = 2,
    decimal_places: int | None = None,
):
    """Return a formatter for mantissas with one shared scientific exponent.

    ``decimal_places``, when given, fixes the mantissa decimals and overrides
    ``significant_digits``.
    """
    from matplotlib.ticker import FuncFormatter

    decimals = _tick_label_decimals(decimal_places, significant_digits)
    zero_label = _zero_tick_label(decimal_places)
    scale = 10**exponent

    def _format_tick(value, _position):
        if np.isclose(value, 0.0):
            return zero_label

        scaled_value = value / scale
        rounded_value = round(scaled_value, decimals)

        if np.isclose(rounded_value, 0.0):
            rounded_value = 0.0

        return f"{rounded_value:.{decimals}f}"

    return FuncFormatter(_format_tick)


def _scientific_exponent_from_mappable(mappable) -> int:
    """Return a shared exponent based on the color scale limits."""
    lo, hi = mappable.get_clim()
    max_abs = max(abs(lo), abs(hi))
    if np.isclose(max_abs, 0.0):
        return 0
    return int(np.floor(np.log10(max_abs)))


def _set_colorbar_exponent_text(cbar, exponent: int, use_latex: bool = True):
    """Draw a shared scientific exponent label near the top-right of the colorbar."""
    existing_text = getattr(cbar.ax, "_hypermri_exponent_text", None)
    if existing_text is not None:
        existing_text.remove()
        cbar.ax._hypermri_exponent_text = None

    if exponent == 0:
        return

    if use_latex:
        label = rf"$\times 10^{{{exponent}}}$"
    else:
        label = f"×10^{exponent}"

    if cbar.orientation == "vertical":
        text = cbar.ax.text(
            1.02,
            1.04,
            label,
            transform=cbar.ax.transAxes,
            ha="left",
            va="bottom",
            clip_on=False,
        )
    else:
        text = cbar.ax.text(
            1.0,
            1.05,
            label,
            transform=cbar.ax.transAxes,
            ha="right",
            va="bottom",
            clip_on=False,
        )

    cbar.ax._hypermri_exponent_text = text


def add_colorbar(
    im,
    aspect: float = 20,
    pad_fraction: float = 0.5,
    colorbar_width=None,
    highlight=None,
    highlight_kwargs=None,
    ticks_in: bool = True,
    show_ticklabels: bool = True,
    minor_ticks: bool = True,
    scientific_notation: bool | None = None,
    scientific_digits: int = 2,
    decimal_places: int | None = None,
    scientific_shared_exponent: bool = False,
    auto_select_notation: bool = False,
    show_outline: bool = True,
    clip_outline: bool = False,
    image_outline: bool = False,
    image_outline_kwargs=None,
    fit_figure_height: bool = False,
    use_latex: bool = True,
    **kwargs,
):
    """
    Add a colorbar to an image, with custom styling and placement.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The image or mappable object to attach the colorbar to.
    aspect : float, optional
        The aspect ratio of the colorbar (height/width). Default is 20.
        Ignored when ``colorbar_width`` is specified.
    pad_fraction : float, optional
        Fraction of the colorbar width to use as padding. Default is 0.5.
    colorbar_width : float, optional
        Fixed physical width of the colorbar in inches. If None, the colorbar
        width is calculated from ``aspect`` as before. Default is None.
    highlight : float, optional
        Value to highlight with a dashed line on the colorbar. Default is None.
    highlight_kwargs : dict, optional
        Keyword arguments for the highlight line.
    ticks_in : bool, optional
        If True, set colorbar ticks to point inward. Default is True.
    show_ticklabels : bool, optional
        If False, hide the colorbar tick labels while keeping the tick marks.
    minor_ticks : bool, optional
        If True, add minor ticks to the colorbar.
    scientific_notation : bool, optional
        If True, render major tick labels in scientific notation. None
        (default) and False keep linear tick labels.
    scientific_digits : int, optional
        Number of significant mantissa digits in scientific notation, at
        least 1. Ignored when ``decimal_places`` is given. Default is 2.
    decimal_places : int, optional
        Fixed number of digits after the decimal point, at least 0. With
        scientific notation it sets the mantissa decimals and overrides
        ``scientific_digits``. Without it, linear tick labels are written with
        exactly this many decimals instead of Matplotlib's automatic
        precision; this also replaces a non-linear formatter such as a
        logarithmic one. A zero tick is then padded as well, for example
        ``0.00``. None (default) keeps the automatic precision and writes zero
        as ``0``.
    scientific_shared_exponent : bool, optional
        If True, show one shared exponent for the colorbar.
    auto_select_notation : bool, optional
        Automatically disable scientific notation for moderate exponents.
    show_outline : bool, optional
        If False, hide the colorbar outline.
    clip_outline : bool, optional
        If True, clip the colorbar outline to the colorbar axes.
    image_outline : bool, optional
        If True, draw an outline fully inside the image axes.
    image_outline_kwargs : dict, optional
        Styling passed to the image-outline lines.
    fit_figure_height : bool, optional
        Resize the figure height to the final image-axes height.
    use_latex : bool, optional
        If True (default), scientific tick labels and the shared exponent use
        Matplotlib mathtext (``$\\times 10^{n}$``). If False, they are plain
        text such as ``2.5×10^-3``. Linear tick labels always use an ASCII
        hyphen for negative values instead of Matplotlib's wider Unicode minus
        sign U+2212. Mathtext is not TeX; no TeX installation is needed for
        either setting. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to ``figure.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        Colorbar attached to the input's figure. Its tick values retain the
        mappable's units and normalization; only formatting/layout is changed.

    Raises
    ------
    ValueError
        If ``decimal_places`` is negative, if ``scientific_digits`` is smaller
        than 1 while scientific notation is used without ``decimal_places``,
        or if Matplotlib rejects the requested layout, normalization or
        styling.
    TypeError
        If forwarded colorbar arguments are unsupported.

    Notes
    -----
    Adds an axes to the existing figure and can resize it, change outline/tick
    styling and add a highlight. Does not alter the mappable's data, show or
    save a figure, or change numerical scientific settings. The caller owns
    figure cleanup. Existing Matplotlib layout/formatting behavior is retained
    except for the minus sign of linear tick labels, the label text selected
    by ``use_latex`` and the label precision set by ``decimal_places``.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> image = ax.imshow(np.arange(4).reshape(2, 2))
    >>> colorbar = add_colorbar(image, scientific_notation=False, use_latex=False)
    >>> plt.close(fig)

    Linear labels with exactly one decimal, for example ``0.0`` and ``1.5``:

    >>> fig, ax = plt.subplots()
    >>> image = ax.imshow(np.arange(4).reshape(2, 2))
    >>> colorbar = add_colorbar(image, decimal_places=1)
    >>> plt.close(fig)
    """
    from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
    from mpl_toolkits import axes_grid1

    divider = axes_grid1.make_axes_locatable(im.axes)

    if colorbar_width is None:
        # Previous behavior: width depends on the image height.
        width = axes_grid1.axes_size.AxesY(
            im.axes,
            aspect=1.0 / aspect,
        )
    else:
        # Fixed physical width in inches.
        width = axes_grid1.axes_size.Fixed(colorbar_width)

    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)

    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)

    cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)

    cbar.outline.set_visible(show_outline)
    if clip_outline:
        cbar.outline.set_clip_path(cbar.ax.patch)

    cbar.ax.set_axisbelow(False)

    exponent = _scientific_exponent_from_mappable(getattr(cbar, "mappable", im))
    if auto_select_notation and abs(exponent) < 3:
        scientific_notation = False

    if scientific_notation:
        if scientific_shared_exponent:
            cbar.formatter = _scaled_colorbar_tick_formatter(
                exponent=exponent,
                significant_digits=scientific_digits,
                decimal_places=decimal_places,
            )
            _set_colorbar_exponent_text(cbar, exponent, use_latex=use_latex)
        else:
            cbar.formatter = _scientific_colorbar_tick_formatter(
                significant_digits=scientific_digits,
                decimal_places=decimal_places,
                use_latex=use_latex,
            )

        cbar.update_ticks()
    elif decimal_places is not None:
        cbar.formatter = _fixed_decimal_tick_formatter(decimal_places)
        cbar.update_ticks()
    elif isinstance(cbar.formatter, ScalarFormatter):
        cbar.formatter = _hyphen_minus_scalar_formatter()
        cbar.update_ticks()

    if not show_ticklabels:
        cbar.ax.tick_params(
            labelleft=False,
            labelright=False,
            labelbottom=False,
            labeltop=False,
        )
        cbar.ax.xaxis.get_offset_text().set_visible(False)
        cbar.ax.yaxis.get_offset_text().set_visible(False)

        exponent_text = getattr(
            cbar.ax,
            "_hypermri_exponent_text",
            None,
        )
        if exponent_text is not None:
            exponent_text.set_visible(False)

    cbar.ax.tick_params(pad=2)

    if ticks_in:
        cbar.ax.tick_params(direction="in")

    if minor_ticks:
        if cbar.orientation == "vertical":
            cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            cbar.ax.xaxis.set_minor_locator(AutoMinorLocator())

        if ticks_in:
            cbar.ax.tick_params(
                which="minor",
                direction="in",
                length=1,
                width=0.5,
            )

    if highlight is not None:
        mappable = getattr(cbar, "mappable", im)
        lo, hi = mappable.get_clim()

        if lo <= highlight <= hi:
            line_kw = {
                "color": "k",
                "linewidth": 1.8,
                "linestyle": "--",
            }

            if highlight_kwargs:
                line_kw.update(highlight_kwargs)

            if cbar.orientation == "vertical":
                cbar.ax.axhline(highlight, **line_kw)
            else:
                cbar.ax.axvline(highlight, **line_kw)

    if fit_figure_height:
        figure = im.axes.figure
        figure.canvas.draw()

        fitted_height = im.axes.get_window_extent().height / figure.dpi

        figure.set_size_inches(
            figure.get_figwidth(),
            fitted_height,
            forward=True,
        )

    if image_outline:
        outline_kwargs = image_outline_kwargs or {}

        im.axes._hypermri_image_outline = _add_inset_axes_outline(
            im.axes,
            **outline_kwargs,
        )

    return cbar


def plot_3d(data, **scatter_kwargs):
    """Plot an existing three-dimensional array as colored index-space points.

    Parameters
    ----------
    data : numpy.ndarray
        Three-dimensional real values. Coordinates are integer array indices;
        this legacy helper labels the colorbar Frequency [Hz] and gives points
        with values below 1e5 a marker size of 10, otherwise zero. Those
        historical choices are preserved and are not inferred from metadata.
    **scatter_kwargs : object
        Existing Matplotlib scatter options forwarded unchanged. Keys cannot
        duplicate the explicitly supplied color, size and style arguments.

    Returns
    -------
    matplotlib.axes.Axes
        New three-dimensional axes with its scatter and colorbar. Does not
        mutate data, show, save or close the figure; the caller owns cleanup.

    Raises
    ------
    ValueError
        If data is not three-dimensional or Matplotlib rejects its values.
    TypeError
        If scatter arguments are duplicated or unsupported.

    Examples
    --------
    >>> ax = plot_3d(np.ones((2, 2, 2)))
    >>> plt.close(ax.figure)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data < 1e5
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    p = ax.scatter(
        x,
        y,
        z,
        c=data.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.5,
        marker="o",
        linewidth=0,
        **scatter_kwargs,
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = fig.colorbar(p, shrink=0.5, label="Frequency [Hz]")
    cbar.solids.set(alpha=1)
    fig.tight_layout()
    return ax


def subplot_auto_size(num_subplots: int = 0):
    """Return the existing near-square row/column layout for subplots.

    Parameters
    ----------
    num_subplots : int
        Nonnegative number of panels. Zero produces a zero-by-zero layout.

    Returns
    -------
    tuple of int
        Number of rows and columns, using the existing ceiling-square-root
        calculation. No figure is constructed or global state changed.

    Raises
    ------
    ValueError
        If a negative count cannot be converted to an integer grid size.

    Examples
    --------
    >>> subplot_auto_size(10)
    (3, 4)
    """
    # Calculate subplot grid dimensions
    if num_subplots == 0:
        return 0, 0
    grid_size = int(np.ceil(np.sqrt(num_subplots)))
    grid_rows = (
        grid_size - 1 if grid_size * (grid_size - 1) >= num_subplots else grid_size
    )
    grid_cols = grid_size
    return grid_rows, grid_cols


def remove_empty_subplots(fig, ax):
    """Remove axes without lines, images or collections from an existing figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose empty axes are removed in place and whose layout is
        tightened when any axes remain.
    ax : matplotlib.axes.Axes or array_like of Axes
        Single axes or one-/two-dimensional collection from that figure.
        A single axes is returned unchanged, even if empty.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same figure after axes removal.
    ax : matplotlib.axes.Axes or numpy.ndarray
        Historical return shape: a single axes or flattened original axes
        collection. Entries for deleted figure axes remain in this collection;
        callers can inspect fig.axes for the surviving axes.

    Notes
    -----
    Does not show, save or close the figure, and does not edit plotted values.
    """
    if isinstance(ax, plt.Axes):
        return fig, ax

    ax = np.array(ax)  # Ensure ax is a numpy array

    if ax.ndim == 1:
        ax = ax.reshape(1, -1)  # Convert 1D to 2D array

    for i in range(ax.shape[0] - 1, -1, -1):
        for j in range(ax.shape[1] - 1, -1, -1):
            if (
                not ax[i, j].get_lines()
                and not ax[i, j].get_images()
                and not ax[i, j].collections
            ):
                fig.delaxes(ax[i, j])

    # Remove empty rows and columns
    ax = ax[~np.all(np.equal(ax, None), axis=1)]
    ax = ax[:, ~np.all(np.equal(ax, None), axis=0)]

    if fig.get_axes():
        fig.tight_layout()

    return fig, ax.flatten() if ax.size > 1 else ax[0, 0]


def format_y_tick_labels(ax, width=15, font_family="monospace"):
    """Format existing y ticks using the legacy scientific-label rule.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose tick labels are replaced in place. Tick positions and
        plotted samples are unchanged.
    width : int, optional
        Retained fallback width for non-scientific formatting; ordinary
        numerical ticks follow the existing one-decimal exponent formatting.
    font_family : str, optional
        Font family passed to the replacement labels.

    Returns
    -------
    None
        Only changes the axes' labels; does not show, save or close its figure.
    """
    ticks = ax.get_yticks()
    formatted_labels = []
    for tick in ticks:
        formatted = f"{tick:.1e}" if "e" in f"{tick:.1e}" else f"{tick:>{width}.4g}"
        formatted_labels.append(formatted)

    ax.set_yticklabels(formatted_labels, family=font_family)


def create_colormap(
    hex_colors,
    name: str = "custom_cmap",
    steps_for_interpolation: int = 256,
    discrete: bool = False,
    register: bool = True,
    reverse: bool = False,
    show: bool = False,
    show_labels: bool = True,
    nan_color: str = "black",
):
    """
    Create and optionally preview a Matplotlib colormap from a list of hex colors.

    Handles Matplotlib <3.6 and >=3.6. If a colormap with the same `name`
    is already registered, registration is skipped (to avoid ValueError).

    Parameters
    ----------
    hex_colors : list of str
        Hex color strings, e.g. ['#00429d', '#96ffea', '#ffffff'].
    name : str, optional
        Colormap name to (try to) register under.
    steps_for_interpolation : int, optional
        Steps for continuous colormap interpolation (default 256).
    discrete : bool, optional
        If True use ListedColormap (discrete); else LinearSegmentedColormap.
    register : bool, optional
        If True, attempt to register with Matplotlib (default True).
    reverse : bool, optional
        If True, reverse the color order before building the cmap.
    show : bool, optional
        If True, show a small preview of the colormap.
    show_labels : bool, optional
        If True, annotate min/max under the preview.
    nan_color : str or tuple, optional
        Color for NaN values; set to None to skip (default 'black').

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        New color mapping, without transforming or reading scientific data.

    Raises
    ------
    ValueError
        If Matplotlib rejects colors or interpolation settings.

    Notes
    -----
    register=True can modify Matplotlib's process-wide colormap registry;
    existing-name registration failures retain the historical skip behavior.
    show=True creates and shows a preview figure that the caller must close.
    Neither the input color sequence nor acquired data is modified.

    Examples
    --------
    >>> cmap = create_colormap(["#000000", "#ffffff"], register=False)
    >>> cmap.N
    256
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    if reverse:
        hex_colors = list(reversed(hex_colors))

    # Build cmap
    if discrete:
        cmap = ListedColormap(hex_colors, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(
            name, hex_colors, N=steps_for_interpolation
        )

    if nan_color is not None:
        cmap.set_bad(color=nan_color)

    # Try to register safely across Matplotlib versions
    if register:
        import matplotlib as mpl

        registry = mpl.colormaps
        try:
            # Newer Matplotlib supports override_builtin
            registry.register(cmap, name=name, override_builtin=True)
        except TypeError:
            # Older Matplotlib: no override_builtin kw
            if name in registry:
                # Name taken: skip to avoid ValueError
                pass
            else:
                registry.register(cmap, name=name)
        except ValueError:
            # Name already registered (or builtin) -> skip
            # You can choose a new `name` or set register=False if you want to reuse the object
            pass

    if show:
        fig, ax = plt.subplots(figsize=(6, 1))
        gradient = [range(cmap.N)] if not discrete else [range(len(hex_colors))]
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_axis_off()
        if show_labels:
            ax.text(0, 1.5, "min", fontsize=8, ha="left", va="bottom", color="gray")
            ax.text(
                (cmap.N if not discrete else len(hex_colors)) - 1,
                1.5,
                "max",
                fontsize=8,
                ha="right",
                va="bottom",
                color="gray",
            )
        plt.title(name, fontsize=9)
        plt.show()

    return cmap


common_cmaps = [
    "bone",  # Good general-purpose grayscale
    "gray",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "viridis",
    "jet",
    "hsv",
    "cool",
    "seismic",
    "twilight",
    "tab10",
    "prism",
    "gist_rainbow",
    "nipy_spectral",  # Perceptually uniform rainbow
    "cmc.lipari",  # Crameri's Lipari colormap
    "cmc.navia",  # Crameri's Navia colormap
]
