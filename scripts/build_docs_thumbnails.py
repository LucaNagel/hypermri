#!/usr/bin/env python3
"""Write the preview images of the tutorial overview page.

Each card on ``docs/source/tutorials/index.md`` shows a small PNG from
``_static/thumbnails`` of the built documentation. The images are cut from
results the documentation already contains, so they show what the tutorial
really produces:

* a viewer page (a snapshot of ``_static/snapshots`` or an HTML file a
  tutorial writes to ``_static/generated``) is opened in a headless Chromium
  browser through Playwright and photographed;
* a Matplotlib figure is taken from the built HTML page of a tutorial.

The documentation build writes the card images itself, after the HTML pages
(see ``docs/source/conf.py``), so they are not part of the repository. Without
Playwright or a browser the cards show only their icon and text. To write them
by hand into an existing build::

    sphinx-build -b html docs/source docs/_build/html
    python scripts/build_docs_thumbnails.py

Playwright needs ``pip install playwright``. It uses its own Chromium headless
shell (``python -m playwright install chromium-headless-shell``) and falls
back to an installed Google Chrome. The viewer pages load Plotly.js from
jsDelivr, so taking their pictures needs network access.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "docs" / "_build" / "html"

THUMBNAIL_SIZE = (640, 400)
"""Width and height of a tutorial preview image in pixels (16:10)."""

VIEWER_WINDOW = (800, 480)
"""Browser window in pixels in which a viewer page is photographed."""


@dataclass(frozen=True)
class ViewerSource:
    """A viewer HTML page and the part of its picture to keep.

    Attributes
    ----------
    page : str
        Path of the HTML page relative to ``docs/source/_static``.
    crop : tuple of int
        ``(left, top, right, bottom)`` in CSS pixels of the
        :data:`VIEWER_WINDOW`; it leaves out the viewer controls.
    size : tuple of int
        Width and height of the written image in pixels.
    pixel_ratio : int
        Image pixels per CSS pixel of the photographed page; 2 gives a sharp
        picture on high-resolution screens.
    """

    page: str
    crop: tuple[int, int, int, int]
    size: tuple[int, int] = THUMBNAIL_SIZE
    pixel_ratio: int = 1


@dataclass(frozen=True)
class FigureSource:
    """A Matplotlib figure in the built HTML page of a tutorial.

    Attributes
    ----------
    tutorial : str
        Tutorial name, the file name of its page without ``.md``.
    index : int
        Position of the figure among the images of the built page,
        counted from 0.
    """

    tutorial: str
    index: int


THUMBNAILS: dict[str, ViewerSource | FigureSource] = {
    "loading": ViewerSource("snapshots/rare.html", (125, 60, 705, 440)),
    "reconstruction": ViewerSource("snapshots/flash_raw.html", (125, 85, 705, 465)),
    "spectroscopy_fitting": FigureSource("spectroscopy_fitting", 0),
    "viewer": ViewerSource("snapshots/csi_map.html", (0, 50, 800, 455)),
}
"""Image of each tutorial card, keyed by the file name without ``.png``."""

PAGE_TIMEOUT_MS = 30_000
"""Longest wait for a viewer page to load Plotly.js and draw, in milliseconds."""


def _launch_browser(playwright):
    """Start Playwright's Chromium, or an installed Google Chrome without it."""
    try:
        return playwright.chromium.launch()
    except Exception as error:  # Playwright raises its own Error type.
        try:
            return playwright.chromium.launch(channel="chrome")
        except Exception:
            raise RuntimeError(
                "No browser for the thumbnails: run 'python -m playwright install "
                "chromium-headless-shell' or install Google Chrome."
            ) from error


def _photograph(browser, page: Path, pixel_ratio: int):
    """Return a PIL image of ``page`` once its Plotly figures are drawn."""
    from PIL import Image

    width, height = VIEWER_WINDOW
    context = browser.new_context(
        viewport={"width": width, "height": height}, device_scale_factor=pixel_ratio
    )
    try:
        tab = context.new_page()
        tab.goto(page.as_uri(), wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
        tab.wait_for_selector(".js-plotly-plot .main-svg", timeout=PAGE_TIMEOUT_MS)
        # Plotly draws its traces just after the figure element appears.
        tab.wait_for_timeout(500)
        return Image.open(BytesIO(tab.screenshot()))
    finally:
        context.close()


def _built_figure(source: FigureSource, build_dir: Path) -> Path:
    """Return the image file of a figure in a built tutorial page."""
    page = build_dir / "tutorials" / f"{source.tutorial}.html"
    if not page.is_file():
        raise FileNotFoundError(
            f"Built tutorial page {page} is missing; build the documentation "
            "with sphinx-build before writing the thumbnails."
        )
    images = re.findall(r'<img[^>]*src="\.\./(_images/[^"]+\.png)"', page.read_text())
    if source.index >= len(images):
        raise ValueError(
            f"{page.name} contains {len(images)} figures, so figure {source.index} "
            f"does not exist; change the index of {source.tutorial!r} in THUMBNAILS."
        )
    return build_dir / images[source.index]


def _write_images(
    sources: dict[str, ViewerSource | FigureSource],
    static_dir: Path,
    build_dir: Path,
    output: Path,
) -> list[Path]:
    """Photograph or cut out each source and save it as ``<name>.png``."""
    from PIL import Image, ImageOps
    from playwright.sync_api import sync_playwright

    output.mkdir(parents=True, exist_ok=True)
    written = []
    with sync_playwright() as playwright:
        browser = None
        try:
            for name, source in sources.items():
                if isinstance(source, ViewerSource):
                    page = static_dir / source.page
                    if not page.is_file():
                        raise FileNotFoundError(
                            f"Viewer page {page} is missing; pages in "
                            "_static/generated are written by the documentation build."
                        )
                    browser = browser or _launch_browser(playwright)
                    box = tuple(edge * source.pixel_ratio for edge in source.crop)
                    image = _photograph(browser, page, source.pixel_ratio).crop(box)
                    size = source.size
                else:
                    image = Image.open(_built_figure(source, build_dir))
                    size = THUMBNAIL_SIZE
                # Pad instead of cut, so a wide figure keeps its axes and legend.
                image = ImageOps.pad(
                    image.convert("RGB"), size, Image.LANCZOS, color="white"
                )
                path = output / f"{name}.png"
                image.save(path, optimize=True)
                written.append(path)
        finally:
            if browser is not None:
                browser.close()
    return written


def build_thumbnails(build_dir: Path, names: list[str] | None = None) -> list[Path]:
    """Write the tutorial card images into a built HTML documentation.

    Parameters
    ----------
    build_dir : pathlib.Path
        Output folder of the Sphinx HTML builder. The viewer pages are read
        from its ``_static`` folder, the figures from its tutorial pages, and
        the images are written to ``_static/thumbnails``.
    names : list of str or None
        Keys of :data:`THUMBNAILS` to write; None writes all of them.

    Returns
    -------
    list of pathlib.Path
        Written files, one ``<name>.png`` per key.

    Raises
    ------
    ImportError
        If Playwright is not installed.
    RuntimeError
        If neither Playwright's Chromium nor Google Chrome can be started.
    FileNotFoundError
        If a viewer page or a built tutorial page is missing.
    ValueError
        If a built tutorial page has fewer figures than its entry expects.

    Notes
    -----
    Each viewer page is loaded over the network (Plotly.js from jsDelivr);
    an existing image of the same name is overwritten.
    """
    selected = sorted(THUMBNAILS) if names is None else names
    static_dir = build_dir / "_static"
    return _write_images(
        {name: THUMBNAILS[name] for name in selected},
        static_dir,
        build_dir,
        static_dir / "thumbnails",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and write the preview images.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments without the program name; None reads
        ``sys.argv``.

    Returns
    -------
    int
        Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(THUMBNAILS),
        default=sorted(THUMBNAILS),
        help="write only these preview images",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=BUILD_DIR,
        help="built HTML documentation to add the card images to",
    )
    args = parser.parse_args(argv)
    for path in build_thumbnails(args.build_dir, args.only):
        print(f"{path}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
