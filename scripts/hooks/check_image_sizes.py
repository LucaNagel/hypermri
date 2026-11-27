#!/usr/bin/env python3
"""Refuse images larger than 500 kB in a commit.

pre-commit passes the staged image files (new or changed) as arguments. The
repository keeps images out of Git where it can (the documentation build
writes its own), so a large image is usually a mistake: compress it, for
example with ``pngquant`` or an online optimiser, or leave it out. The
generic ``check-added-large-files`` hook only looks at new files; this one
also catches an existing image that grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT_BYTES = 500 * 1024
"""Largest allowed image size in bytes (500 kB)."""


def oversized(paths: list[str], limit: int = LIMIT_BYTES) -> list[tuple[str, int]]:
    """Return the files above ``limit`` with their sizes in bytes.

    Parameters
    ----------
    paths : list of str
        Image files to check; paths that do not exist (deleted files) are
        skipped.
    limit : int
        Largest allowed size in bytes.

    Returns
    -------
    list of tuple of (str, int)
        Path and size of each file that is too large, in input order.
    """
    found = []
    for name in paths:
        path = Path(name)
        if path.is_file() and path.stat().st_size > limit:
            found.append((name, path.stat().st_size))
    return found


def main(argv: list[str] | None = None) -> int:
    """Print each oversized image and return 1 if there is one.

    Parameters
    ----------
    argv : list of str or None
        Image paths; None reads ``sys.argv``.

    Returns
    -------
    int
        0 if all images are small enough, otherwise 1.
    """
    found = oversized(sys.argv[1:] if argv is None else argv)
    for name, size in found:
        print(f"{name}: {size / 1024:.0f} kB, more than {LIMIT_BYTES // 1024} kB")
    if found:
        print(
            "Compress these images or leave them out of the commit. To commit "
            "one anyway, run: SKIP=check-image-sizes git commit ..."
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
