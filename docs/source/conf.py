# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# add path of source:
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # for the source code
sys.path.insert(0, os.path.abspath("external_docs"))  # for git submodule docs

# Ensure Matplotlib writes caches to a writable location during docs builds.
_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_docs_dir, "_build", "mplconfig"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HyperMRI"
copyright = "2026, HyperMRI contributors"
author = "HyperMRI contributors"
try:
    from importlib.metadata import version as package_version

    release = package_version("hypermri")
except Exception:
    release = "unknown"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy-style docstrings
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # Type hint rendering in signatures
    "sphinx.ext.mathjax",  # For equations to be displayed nicely
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# sphinx_autodoc_typehints inserts ``:rtype: None`` after the last field of a
# docstring, which lands inside a following "Examples" or "Notes" block and
# breaks its markup. A None return carries no information for the reader.
typehints_document_rtype_none = False

if os.environ.get("HYPERMRI_ENABLE_INTERSPHINX", "").lower() in {"1", "true", "yes"}:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "xarray": ("https://docs.xarray.dev/en/stable/", None),
    }
else:
    intersphinx_mapping = {}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

myst_heading_anchors = 3

# Allow warning-as-error documentation builds in offline environments. The
# external inventories are useful when available, but missing network access
# should not make local docs validation fail.
suppress_warnings = [
    # Avoid failing local warning-as-error builds on external references pulled
    # in from third-party autodoc docstrings.
    "ref.ref",
]

# Tutorials are MyST text notebooks (``.md`` with a ``file_format: mystnb``
# header) that myst-nb executes during the build. Results are cached in
# ``docs/_build/.jupyter_cache`` and reused until a page's code changes.
nb_execution_mode = os.environ.get("HYPERMRI_DOCS_EXECUTE", "cache")
nb_execution_timeout = int(os.environ.get("HYPERMRI_DOCS_TIMEOUT", "900"))
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# Pages that read the public Lego phantom (Zenodo 10.5281/zenodo.4048253) run
# only where that folder exists; elsewhere they show their code without
# output. Their viewer snapshots are committed files in ``_static/snapshots``
# (``scripts/build_docs_snapshots.py``), so they stay interactive either way.
PHANTOM_FOLDER = "20200612_094625_lego_phantom_3_1_2"
PHANTOM_PAGES = [
    "getting_started/first_look.md",
    "tutorials/loading.md",
    "tutorials/reconstruction.md",
    "tutorials/viewer.md",
]


def _find_phantom() -> str | None:
    """Return the local Lego phantom folder, or None when it is not present."""
    repository = os.path.abspath(os.path.join(_docs_dir, ".."))
    candidates = [os.environ.get("HYPERMRI_DOCS_PHANTOM", "")]
    for root in (
        os.environ.get("HYPERMRI_TEST_DATA", ""),
        os.path.join(repository, "tests", "data"),
        os.path.join(repository, "examples"),
    ):
        if root:
            candidates.append(os.path.join(root, PHANTOM_FOLDER))
    return next((path for path in candidates if path and os.path.isdir(path)), None)


_phantom = _find_phantom()
if _phantom is None:
    nb_execution_excludepatterns = list(PHANTOM_PAGES)
else:
    # The executed pages read this variable in a hidden cell.
    os.environ["HYPERMRI_DOCS_PHANTOM"] = _phantom
    nb_execution_excludepatterns = []

templates_path = ["_templates"]
exclude_patterns = [
    "api/generated/**",
    "tutorials/to dos.md",  # Untracked local notes, not published documentation.
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
# html_theme = 'press'
html_theme = "pydata_sphinx_theme"

html_title = "HyperMRI documentation"

html_static_path = ["_static"]

html_css_files = [
    # "css/numpy-theme.css",
    "css/custom.css",  # This will override specific properties
]


html_theme_options = {
    "logo": {
        "text": "hypermri",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "use_edit_page_button": False,
    "collapse_navigation": False,
    "header_links_before_dropdown": 4,
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "index": [],
    },
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LucaNagel/hypermri",
            "icon": "fa-brands fa-github",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}


autodoc_member_order = "groupwise"

# Use a light syntax theme for the default light page background and
# a separate dark theme when the user switches the site to dark mode.
pygments_style = "sphinx"
pygments_dark_style = "github-dark"


# Do not show "Created using Sphinx" footer
html_show_sphinx = False


def setup(app):
    def _qualify_builtin_type_references(app, doctree):
        """Disambiguate builtin class references from handler type properties."""
        from sphinx import addnodes

        for node in doctree.findall(addnodes.pending_xref):
            if (
                node.get("refdomain") == "py"
                and node.get("reftype") == "class"
                and node.get("reftarget") == "type"
            ):
                node["reftarget"] = "builtins.type"

    app.connect("doctree-read", _qualify_builtin_type_references)
