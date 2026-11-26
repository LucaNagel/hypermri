#!/usr/bin/env python3
"""Validate that runtime imports are declared in ``pyproject.toml``."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


IMPORT_TO_DISTRIBUTION_OVERRIDES = {
    "cv2": "opencv-python",
    "git": "GitPython",
    "IPython": "IPython",
    "matplotlib_scalebar": "matplotlib-scalebar",
    "mpl_toolkits": "matplotlib",
    "nrrd": "pynrrd",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}

IGNORED_IMPORT_ROOTS = {"__future__"}
DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class ImportReference:
    """A top-level import found in source code."""

    root: str
    path: Path
    line: int


def normalize_distribution_name(name: str) -> str:
    """Normalize a distribution name according to PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_name(requirement: str) -> str:
    """Extract the distribution name from a dependency requirement string."""
    requirement = requirement.split(";", 1)[0].strip()
    if "@" in requirement:
        requirement = requirement.split("@", 1)[0].strip()
    match = DEPENDENCY_NAME_RE.match(requirement)
    if match is None:
        raise ValueError(f"Could not parse dependency requirement: {requirement!r}")
    return match.group(1)


def declared_dependencies(pyproject_path: Path) -> set[str]:
    """Return normalized dependencies declared in main and optional groups."""
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject.get("project", {})
    dependencies = set(project.get("dependencies", []))

    for optional_group in project.get("optional-dependencies", {}).values():
        dependencies.update(optional_group)

    return {
        normalize_distribution_name(dependency_name(requirement))
        for requirement in dependencies
    }


def source_files(source_root: Path) -> list[Path]:
    """Return Python source files under ``source_root``."""
    return sorted(
        path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts
    )


def imports_in_file(path: Path) -> list[ImportReference]:
    """Collect top-level absolute imports from a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[ImportReference] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                imports.append(ImportReference(root=root, path=path, line=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            imports.append(ImportReference(root=root, path=path, line=node.lineno))

    return imports


def import_distribution_name(import_root: str) -> str:
    """Map an import root to its expected distribution name."""
    return IMPORT_TO_DISTRIBUTION_OVERRIDES.get(import_root, import_root)


def is_stdlib_import(import_root: str) -> bool:
    """Return whether ``import_root`` is part of the Python standard library."""
    return import_root in sys.stdlib_module_names or import_root.startswith("_")


def is_first_party_import(import_root: str, first_party_roots: set[str]) -> bool:
    """Return whether ``import_root`` belongs to this repository."""
    return import_root in first_party_roots


def external_imports(
    source_root: Path,
    first_party_roots: set[str],
) -> dict[str, list[ImportReference]]:
    """Return external import references keyed by normalized distribution name."""
    references: dict[str, list[ImportReference]] = {}

    for path in source_files(source_root):
        for import_ref in imports_in_file(path):
            root = import_ref.root
            if root in IGNORED_IMPORT_ROOTS:
                continue
            if is_stdlib_import(root):
                continue
            if is_first_party_import(root, first_party_roots):
                continue

            distribution = normalize_distribution_name(import_distribution_name(root))
            references.setdefault(distribution, []).append(import_ref)

    return references


def missing_dependency_report(project_root: Path) -> dict[str, list[ImportReference]]:
    """Return external imports that are not declared in ``pyproject.toml``."""
    source_root = project_root / "src" / "hypermri"
    pyproject_path = project_root / "pyproject.toml"

    declared = declared_dependencies(pyproject_path)
    external = external_imports(source_root, first_party_roots={"hypermri"})

    return {
        distribution: refs
        for distribution, refs in sorted(external.items())
        if distribution not in declared
    }


def format_missing_report(
    missing: dict[str, list[ImportReference]],
    project_root: Path,
) -> str:
    """Format missing dependencies for CLI and pytest output."""
    lines = [
        "Missing third-party dependencies in pyproject.toml:",
    ]
    for distribution, refs in missing.items():
        first_ref = refs[0]
        location = first_ref.path.relative_to(project_root)
        lines.append(
            f"- {distribution}: imported as {first_ref.root!r} "
            f"at {location}:{first_ref.line}"
        )
    return "\n".join(lines)


def check_dependencies(project_root: Path) -> int:
    """Run the dependency check and return a process exit code."""
    missing = missing_dependency_report(project_root)
    if missing:
        print(format_missing_report(missing, project_root))
        return 1

    print("All third-party imports are declared in pyproject.toml.")
    return 0


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raise SystemExit(check_dependencies(project_root))


if __name__ == "__main__":
    main()
