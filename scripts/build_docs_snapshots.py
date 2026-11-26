#!/usr/bin/env python3
"""Write the viewer snapshots embedded in the documentation.

The tutorial pages show interactive viewers of the public Lego phantom
(BrukerAPI test data set, Zenodo DOI 10.5281/zenodo.4048253, CC-BY-4.0). The
documentation build does not download that 780 MB dataset, so the views are
saved once with :meth:`hypermri.visualization.scan_view.ScanView.to_html` and
committed to ``docs/source/_static/snapshots``. Run this script again after a
change to the viewer or to the reconstruction of these scans::

    python scripts/build_docs_snapshots.py /path/to/20200612_094625_lego_phantom_3_1_2

Each snapshot loads Plotly.js from jsDelivr (``plotly_js="cdn"``), so a page
holds only the viewer code and the data of the shown plane.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "source" / "_static" / "snapshots"


def _rare(study):
    """RARE anatomy (scan 8): the first view of the getting-started page."""
    return study[8].plot()


def _csi_map(study):
    """CSI (scan 30) on the RARE anatomy (scan 8): landing and viewer pages."""
    return study[30].plot(anatomy=study[8])


def _nspect(study):
    """Unlocalized spectra over repetitions (scan 33): spectra without a map."""
    return study[33].plot()


def _flash_raw(study):
    """FLASH (scan 3) reconstructed from raw k-space by HyperMRI."""
    from hypermri.visualization.scan_view import plot_scan

    scan = study[3]
    return plot_scan(scan.reconstruct("raw"), scan=scan, name="raw")


def _flash_scanner(study):
    """FLASH (scan 3) as reconstructed by ParaVision (2dseq)."""
    from hypermri.visualization.scan_view import plot_scan

    scan = study[3]
    return plot_scan(scan.reconstruct("seq2d"), scan=scan, name="2dseq")


SNAPSHOTS: dict[str, Callable] = {
    "rare": _rare,
    "csi_map": _csi_map,
    "nspect": _nspect,
    "flash_raw": _flash_raw,
    "flash_scanner": _flash_scanner,
}


def build_snapshots(phantom: Path, output: Path, names: list[str]) -> list[Path]:
    """Write the selected snapshots and return their paths.

    Parameters
    ----------
    phantom : pathlib.Path
        Unpacked Lego phantom study folder (it contains the scan folders
        ``3`` to ``48``).
    output : pathlib.Path
        Directory for the HTML files; created when missing.
    names : list of str
        Keys of :data:`SNAPSHOTS` to write.

    Returns
    -------
    list of pathlib.Path
        Written files, one ``<name>.html`` per key.

    Raises
    ------
    FileNotFoundError
        If ``phantom`` is not a directory.
    """
    from hypermri import Study

    if not phantom.is_dir():
        raise FileNotFoundError(
            f"Lego phantom folder {phantom} does not exist; download it from "
            "https://doi.org/10.5281/zenodo.4048253 and unpack the zip file."
        )
    study = Study(phantom, verbose=False)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        path = output / f"{name}.html"
        SNAPSHOTS[name](study).to_html(path, plotly_js="cdn")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and write the snapshots.

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
    parser.add_argument("phantom", type=Path, help="Lego phantom study folder")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(SNAPSHOTS),
        default=sorted(SNAPSHOTS),
        help="write only these snapshots",
    )
    parser.add_argument("--output", type=Path, default=SNAPSHOT_DIR)
    args = parser.parse_args(argv)
    for path in build_snapshots(args.phantom, args.output, args.only):
        print(f"{path.relative_to(PROJECT_ROOT)}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
