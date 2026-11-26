"""Default colour maps of the viewer and the list it offers."""

from __future__ import annotations

from typing import Iterable

# Default colour maps of map, anatomy and overlay.
DEFAULT_SPATIAL_CMAP_NON_SPECTRO = "bone"
DEFAULT_SPATIAL_CMAP_SPECTRO = "turbo"
# An image shown over an anatomy (an image under an image) uses a colour map
# that stands out from the grey anatomy.
DEFAULT_SPATIAL_CMAP_OVER_ANATOMY = "turbo"
DEFAULT_OVERLAY_CMAP = "inferno"
DEFAULT_ANATOMY_CMAP = "bone"

# Maps short display names to unambiguous canonical names accepted by ``cmap``.
CMAP_CANONICAL = {
    "rainbow": "gnuplot:rainbow",
    "RdBu_r": "colorbrewer:RdBu_r",
}

# Colour maps offered by the viewer controls.
COMMON_CMAPS = (
    "bone",
    "turbo",
    "inferno",
    "magma",
    "viridis",
    "plasma",
    "gray",
    "jet",
    "rainbow",
    "RdBu_r",
    "lipari",
    "navia",
)


def canonical_cmap_name(name: str) -> str:
    """Resolve a supported short colormap alias.

    Parameters
    ----------
    name : str
        Display name or already qualified colormap identifier.

    Returns
    -------
    str
        Canonical alias value, or the input unchanged. No colormap is loaded.
    """
    return CMAP_CANONICAL.get(name, name)


def is_cmap_available(name: str) -> bool:
    """Check whether the optional colormap provider accepts a name.

    Parameters
    ----------
    name : str
        Display alias or canonical colormap identifier.

    Returns
    -------
    bool
        True when cmap can instantiate the resolved identifier; False for a missing
        dependency, unknown name or provider failure. No widgets are constructed.
    """
    try:
        from cmap import Colormap

        Colormap(canonical_cmap_name(name))
        return True
    except Exception:
        return False


def build_cmap_options(
    candidates: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Build alphabetically sorted display-name and colormap pairs.

    Parameters
    ----------
    candidates : iterable of str, optional
        Names to check eagerly; None uses COMMON_CMAPS. Duplicate inputs remain.

    Returns
    -------
    list of tuple
        Display labels and canonical names for available colormaps. Provider
        failures omit candidates. Inputs and shared defaults are not mutated.
    """
    names = tuple(candidates) if candidates is not None else COMMON_CMAPS
    options = [
        (name, canonical_cmap_name(name)) for name in names if is_cmap_available(name)
    ]
    return sorted(options, key=lambda item: item[0].lower())
