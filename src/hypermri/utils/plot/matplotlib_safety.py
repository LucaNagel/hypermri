"""Matplotlib runtime safety helpers."""

from __future__ import annotations

import warnings
from functools import wraps

import matplotlib as mpl
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

_FALLBACK_INSTALLED = False
_RETRYING_TEX_RENDER = False


def _is_probable_tex_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "tex",
            "latex",
            "dvi",
            "dvipng",
            ".tfm",
            "font metric",
            "type1cm",
            "sfmath",
        )
    )


def _disable_usetex_after_error(exc: Exception, *, stacklevel: int = 2) -> None:
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["text.latex.preamble"] = ""
    warnings.warn(
        "Matplotlib text.usetex was disabled because TeX rendering failed. "
        "The plot will use Matplotlib's built-in text renderer instead. "
        f"Original error: {exc}",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def _disable_usetex_on_existing_text(figure: Figure) -> None:
    for artist in figure.findobj(match=Text):
        artist.set_usetex(False)


def _should_retry_without_usetex(exc: Exception) -> bool:
    return bool(mpl.rcParams.get("text.usetex", False)) and _is_probable_tex_error(exc)


def disable_usetex_if_unavailable() -> bool:
    """Disable Matplotlib TeX rendering if the current TeX setup cannot draw.

    Returns
    -------
    bool
        True if ``text.usetex`` was disabled, False if no change was needed.
    """
    if not mpl.rcParams.get("text.usetex", False):
        return False

    try:
        figure = Figure(figsize=(0.2, 0.2))
        FigureCanvasAgg(figure)
        figure.text(0.5, 0.5, "TeX probe")
        figure.canvas.draw()
    except Exception as exc:
        _disable_usetex_after_error(exc, stacklevel=2)
        return True

    return False


def install_matplotlib_usetex_fallback() -> bool:
    """Install a global retry hook for Matplotlib TeX render failures.

    The hook leaves normal Matplotlib behavior untouched. If a render fails while
    ``text.usetex`` is enabled and the error looks TeX-related, it disables
    ``text.usetex`` and retries the same render once.

    Returns
    -------
    bool
        True if the fallback was installed by this call, False if it was already
        installed.
    """
    global _FALLBACK_INSTALLED

    if _FALLBACK_INSTALLED:
        return False

    original_print_figure = FigureCanvasBase.print_figure
    original_figure_draw = Figure.draw

    @wraps(original_print_figure)
    def print_figure_with_usetex_fallback(self, *args, **kwargs):
        global _RETRYING_TEX_RENDER

        try:
            return original_print_figure(self, *args, **kwargs)
        except Exception as exc:
            if _RETRYING_TEX_RENDER or not _should_retry_without_usetex(exc):
                raise
            _disable_usetex_after_error(exc, stacklevel=3)
            _disable_usetex_on_existing_text(self.figure)
            _RETRYING_TEX_RENDER = True
            try:
                return original_print_figure(self, *args, **kwargs)
            finally:
                _RETRYING_TEX_RENDER = False

    @wraps(original_figure_draw)
    def figure_draw_with_usetex_fallback(self, renderer):
        global _RETRYING_TEX_RENDER

        try:
            return original_figure_draw(self, renderer)
        except Exception as exc:
            if _RETRYING_TEX_RENDER or not _should_retry_without_usetex(exc):
                raise
            _disable_usetex_after_error(exc, stacklevel=3)
            _disable_usetex_on_existing_text(self)
            _RETRYING_TEX_RENDER = True
            try:
                return original_figure_draw(self, renderer)
            finally:
                _RETRYING_TEX_RENDER = False

    FigureCanvasBase.print_figure = print_figure_with_usetex_fallback
    Figure.draw = figure_draw_with_usetex_fallback
    _FALLBACK_INSTALLED = True
    return True
