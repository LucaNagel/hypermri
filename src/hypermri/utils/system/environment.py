"""Read repository provenance without requiring a Python Git library."""

import subprocess
from pathlib import Path


def get_git_commit_hash(
    path: str | Path | None = None, missing: str = "unknown"
) -> str:
    """Return the full HEAD commit hash from a repository directory.

    Parameters
    ----------
    path : str or pathlib.Path or None, optional
        Directory from which Git searches for its repository. None uses the
        current working directory, preserving the original no-argument API.
    missing : str, optional
        Explicit provenance label returned when Git is unavailable or the
        directory has no resolvable HEAD. Defaults to ``"unknown"``.

    Returns
    -------
    str
        Full commit SHA, or ``missing``. One read-only Git subprocess runs per
        call; no acquired data is read and no result is cached. A Git executable
        is optional. GitPython is no longer needed.

    Notes
    -----
    Vendor wrappers supply their original missing-repository label and their
    module directory. Successful provenance values remain unchanged.

    Examples
    --------
    >>> get_git_commit_hash(path="/a/directory/without/a/repository")
    'unknown'
    """
    command = ["git"]
    if path is not None:
        command.extend(["-C", str(path)])
    command.extend(["rev-parse", "--verify", "HEAD"])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return missing
    return result.stdout.strip() if result.returncode == 0 else missing
