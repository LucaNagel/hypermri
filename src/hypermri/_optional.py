"""Import viewer, notebook and vendor dependencies at their point of use.

These packages are regular HyperMRI dependencies, but importing them lazily
keeps loading and numerical analysis free of widget and plotting start-up.
"""

from importlib import import_module


def _import_optional(module: str):
    """Import a lazily used dependency without hiding unrelated import failures."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0] and exc.name != module:
            raise
        raise ImportError(
            f"This operation requires {module}, a HyperMRI dependency that is "
            "missing from this environment; reinstall HyperMRI with "
            "`pip install hypermri`."
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
        If ipywidgets or IPython is missing from the environment.
    """
    if name == "widgets":
        return _import_optional("ipywidgets")
    if name == "display":
        return _import_optional("IPython.display").display
    raise AttributeError(f"Unknown notebook alias {name!r}")
