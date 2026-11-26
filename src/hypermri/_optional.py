"""Report missing optional feature dependencies at their point of use."""

from importlib import import_module


def _import_optional(module: str, extra: str):
    """Import a feature dependency without hiding unrelated import failures."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0] and exc.name != module:
            raise
        raise ImportError(
            f"This operation requires {module}; install it with "
            f"`pip install 'hypermri[{extra}]'`."
        ) from exc


def _notebook_alias(name: str):
    """Resolve historical notebook aliases without initializing UI on import.

    Parameters
    ----------
    name : str
        ``widgets`` or ``display``, a retained notebook dependency alias.

    Returns
    -------
    object
        Original ipywidgets module or IPython display function. No widget or
        scan is constructed, and missing dependencies are not cached.

    Raises
    ------
    AttributeError
        If the requested name is not a retained notebook alias.
    ImportError
        If ``hypermri[notebook]`` is unavailable.
    """
    if name == "widgets":
        return _import_optional("ipywidgets", "notebook")
    if name == "display":
        return _import_optional("IPython.display", "notebook").display
    raise AttributeError(f"Unknown notebook alias {name!r}")
