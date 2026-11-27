Release and public-snapshot checklist
=====================================

The private development repository is the archival repository. Its
branches, tags, reflogs and commit metadata are **not** approved for
publication.

.. danger::

   Never add GitHub as a remote of the private repository and never push its
   ``main`` branch or an old tag. The first GitHub commit must be created in a
   separate, history-free directory by following the procedure below.

Release prerequisites
---------------------

Before exporting or publishing a release:

1. Review :doc:`../support` and :doc:`../api/public`. Every stable claim must
   have corresponding tests, and every deprecation must name its replacement
   and removal release.
2. Confirm that incompatible ``0.x`` changes occur only in a minor release and
   are announced. From ``1.0.0`` onward, stable API breaks require a major
   release.
3. Confirm that the approved PyPI distribution name ``hypermri`` is used
   consistently in ``pyproject.toml``, workflow tests, and installation
   documentation, and that Luca Nagel still has release access to the PyPI
   project.
4. Confirm that the PyPI project owners and maintainers are correct.
5. Confirm that every public URL in ``pyproject.toml`` resolves, including the
   Read the Docs documentation. After the release tag is pushed, check that the
   ``stable`` documentation version shows the new release. Set ``version`` and
   ``date-released`` in ``CITATION.cff`` to the release before tagging it.
6. Confirm that the license in ``pyproject.toml``, ``LICENSE``, ``CITATION.cff``,
   GitHub, and PyPI matches the completed rights review.
7. Review the historical roster in ``CONTRIBUTORS.md`` for completeness and
   spelling. Use only explicitly approved public names and optional public
   GitHub, ORCID, or contact identifiers; never infer contact details from
   private commit metadata. Set ``contributors_attribution_complete`` only
   after the listed contributors and the release approver have accepted the
   public attribution.
8. Confirm the stable Bruker raw-reconstruction allowlist and its numerical
   tests against the pinned BrukerAPI ParaVision 6.0.1 Zenodo reference
   dataset. Record the dataset citation, license, checksum, and privacy review.
9. Obtain Luca Nagel's explicit code, scientific, privacy, public-snapshot, and
   publication approval. He is the sole release approver for ``0.1.0rc1`` and
   ``0.1.0``.
10. Record these decisions in ``.github/release-readiness.json``. Run
    ``python scripts/check_release_readiness.py --allow-pending`` to verify that
    recorded decisions are internally consistent. The four approval fields
    ``rights_review_complete``, ``contributors_attribution_complete``,
    ``reference_dataset_validation_complete`` and ``public_snapshot_approved``
    require JSON booleans: ``true`` records approval and ``false`` leaves the
    decision pending. Missing fields also remain pending. Other types, including
    the strings ``"true"`` and ``"false"``, are inconsistent and fail even with
    ``--allow-pending``.
    Text decisions require non-blank JSON strings; the stable Bruker sequence
    decision requires a list of unique, non-blank strings. Missing, null or empty
    text/list decisions remain pending; other incorrect types fail both modes.
    ``schema_version`` must be the JSON integer ``1``. Duplicate keys in any
    loaded JSON object are rejected, including nested and escaped-key aliases,
    so a later value cannot silently replace an earlier decision.
    Project metadata is read once with Python's standard TOML parser. Equivalent
    TOML spellings of ``project.name`` and ``project.license`` have the same
    meaning. Invalid TOML anywhere in ``pyproject.toml``, including duplicate
    keys or tables, and a missing or non-table ``project`` value block both CLI
    modes with a ``project-file`` finding.
11. Keep ``ENABLE_TESTPYPI_PUBLISHING`` and ``ENABLE_PYPI_PUBLISHING`` unset or
    ``false`` until the strict release-readiness check passes.

Create the first public snapshot
--------------------------------

Run the exporter from the reviewed private working tree. The destination must
be a new path outside the private repository:

.. code-block:: bash

   python scripts/export_public_snapshot.py /path/to/hypermri-public

The exporter copies tracked and non-ignored new files from the working tree,
omits both staged deletions and missing tracked files reported by Git,
reapplies the current ignore rules to historically tracked local files, rejects
symbolic links and unresolved Git LFS pointers (the project stores no files
in Git LFS), and does not copy ``.git``. It withholds ``AGENTS.md``, whose
agent instructions refer to the private development plan; the snapshot audit
rejects that file. It refuses to overwrite an existing destination.
It does not stage changes or otherwise modify the source Git index. Remaining
candidates that disappear after Git selection still cause an error. Keep the
source worktree unchanged during export; collection and copying are not atomic.

The first export may be used as a provisional review candidate while
``public_snapshot_approved`` is still ``false``. After Luca Nagel approves its
contents, record that approval in the private source tree and create a new final
export in another empty destination. This avoids adding an approval commit on
top of the public root commit.

In the exported directory, create and inspect a new repository:

.. code-block:: bash

   cd /path/to/hypermri-public
   git init --initial-branch=main
   git add --all
   python scripts/audit_public_snapshot.py
   python scripts/check_release_readiness.py
   git diff --cached --check
   git status --short

The staged file list must not contain ``AGENTS.md``, ``.vscode/``, ``guides/``, ``reports/``,
private test data, or any other path not approved for the public release. Check
the complete list, not only the summary:

.. code-block:: bash

   git ls-files

Create one neutral initial commit without a historical contributor identity:

.. code-block:: bash

   git -c user.name="HyperMRI contributors" \
       -c user.email="noreply@hypermri.invalid" \
       commit -m "Initial public snapshot"
   git rev-list --count HEAD
   git log --format=fuller --max-count=1

``git rev-list`` must print ``1``. Review the author and committer metadata,
then repeat the snapshot audit from the committed tree. Only this new
repository may receive the public GitHub remote:

.. code-block:: bash

   python scripts/audit_public_snapshot.py
   python scripts/check_release_readiness.py
   git remote add origin git@github.com:LucaNagel/hypermri.git
   git push --set-upstream origin main

Do not copy tags from the private repository. Create all public release tags in
the new repository.

Validate release artifacts
--------------------------

Run these checks in the public snapshot:

.. code-block:: bash

   python scripts/clean_build_artifacts.py --apply
   python -m build
   python -m twine check dist/*
   python scripts/validate_release_artifacts.py \
     --channel pypi --tag v0.1.0 --ref-type tag --dist-dir dist
   python -m pip install dist/*.whl
   python -m pip check
   python -c "import hypermri; print(hypermri.__version__)"

For a release candidate, use ``--channel testpypi --tag v0.1.0rc1`` instead.
The validator requires exactly one wheel and one source distribution and checks
their filenames and embedded metadata against the approved release target.
Each artifact's embedded metadata must contain exactly one nonblank
``Metadata-Version``, ``Name`` and ``Version`` header. Header names are
case-insensitive; repeated headers are rejected even when their values agree.
Description text after the header separator is not treated as metadata. These
checks preserve the required single-value identity fields in the `Python core
metadata specification
<https://packaging.python.org/en/latest/specifications/core-metadata/>`_.
Both artifacts must also contain exactly one nonblank ``License-Expression``
header whose value matches ``license_identifier`` in the release policy exactly.
Missing, blank, repeated or different expressions block either publishing
channel. Header names are case-insensitive and description text is ignored.
The legacy ``License`` header must be absent: the `core metadata licence rules
<https://packaging.python.org/en/latest/specifications/core-metadata/#license>`_
make it mutually exclusive with ``License-Expression``. The validator requires
a nonblank string policy value and reports the affected archive on metadata
failure. These consistency checks do not grant or replace the outstanding
rights approval.
The artifact validator reads its release policy as a UTF-8 JSON object and
rejects duplicate decoded keys in every object, including nested objects and
objects inside arrays. Escaped spellings of the same key count as duplicates;
the same key in distinct objects remains valid. Unreadable files, invalid UTF-8
or malformed JSON produce a controlled error naming the policy path and a
nonzero CLI exit code. Approval decisions remain owned by the release-readiness
gate.
It rejects duplicate archive paths after normalization, including aliases such
as ``hypermri/./__init__.py``, before reading distribution metadata.
The built version must also be newer than any version already uploaded under
the ``hypermri`` PyPI distribution name.

TestPyPI release candidate
--------------------------

After the snapshot is public and all prerequisites are approved, set
``ENABLE_TESTPYPI_PUBLISHING=true``, create the release-candidate tag in the
new repository, and push it:

.. code-block:: bash

   git tag v0.1.0rc1
   git push origin v0.1.0rc1

Install the result from TestPyPI in a fresh environment:

.. code-block:: bash

   python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     hypermri==0.1.0rc1

Final PyPI release
------------------

If the TestPyPI installation and scientific smoke checks pass, disable
TestPyPI publishing, set ``ENABLE_PYPI_PUBLISHING=true``, then create and push
the final tag:

.. code-block:: bash

   git tag v0.1.0
   git push origin v0.1.0

The final tag triggers the PyPI workflow. Pre-release tags such as
``v0.1.0rc1`` are intentionally routed only to TestPyPI.

Distribution contents
---------------------

The source distribution contains runtime Python source, the four packaged
configuration/registry files, the scripts and stylesheets of the browser viewer
(``visualization/browser/*.js`` and ``*.css``) with the fit-monitor page
``visualization/fit_monitor_page.html``, license and public policy files, and
this release procedure. It excludes scanner data, notebooks, tutorial media, tests, developer
scripts, local plans and repository/CI configuration. Use a repository checkout
for contributor tests and documentation builds. The wheel contains only the
runtime package and generated distribution metadata. Building a wheel from the
source distribution must retain the same runtime paths and byte-for-byte
identical Python source and packaged configuration. The validator compares all
runtime files, including Python modules beyond the required entry point and
configuration list. It rejects equal path sets with different contents and
names each differing runtime path. Matching empty Python modules are allowed.

Both artifacts retain ``LICENSE``, ``NOTICE``,
``LICENSES/brukerMRI-Apache-2.0.txt`` and ``LICENSES/brukerapi-MIT.txt``.
The upstream licences and attribution cover the adapted BrukerMRI parameter
readers and the brukerapi-based FID block-padding adapter. ``NOTICE`` records
their verified sources and HyperMRI modifications. The source archive
keeps these paths at its root. The wheel keeps them below its single
``.dist-info/licenses/`` directory, including the nested ``LICENSES/`` path.
Each artifact's metadata must declare each path exactly once using
``License-File``. The validator rejects missing or empty files, unexpected or
duplicate licence declarations and unequal licence/notice bytes between the
two artifacts. These retention checks do not approve the remaining rights
review or change the project's ``AGPL-3.0-only`` declaration.

The release artifact validator checks these contents as well as tag/version
identity. Both publishing workflows require the same built wheel to pass its
Python 3.12 tests on Linux, macOS and Windows before upload. These jobs install
the wheel's ``test`` extra. Fresh-process runtime tests derive ``PYTHONPATH``
from the package imported by the parent test, so installed-wheel checks retain
the installed package instead of importing the checkout's ``src`` directory.
A regression compares the parent and child package locations. This also keeps
local source tests on their selected checkout. Static repository-policy checks
and the developer commit hook continue to inspect the checkout explicitly.
The two Bash hook tests model POSIX executable lookup and run only on POSIX
systems. A separate portable Python-runner regression checks checkout selection,
inherited pytest options, failure/success exit codes and stale bytecode without
that shell fixture. Cleanup tests that require symbolic links skip only when
link creation reports an unavailable capability or insufficient permission;
other cleanup tests and unexpected link errors remain mandatory. These local
controls do not establish that hosted Windows, Linux or macOS jobs pass.
The FIFO preservation test additionally requires Python's
`os.mkfifo <https://docs.python.org/3.12/library/os.html#os.mkfifo>`_ capability;
only that case skips when the function is unavailable.
The separate installation job exercises loading and numerical analysis with
core dependencies alone. Neither packaging nor installation downloads MRI
reference data.

Both publishing workflows wait for the single build before calling the shared
release-quality workflow. Its reference job downloads that same distribution
artifact, installs the wheel's ``test`` extra and verifies that HyperMRI imports
from the interpreter's installation directory. It prepares the pinned
BrukerAPI reference archive and runs integration tests with
``python -I -m pytest -m integration --require-real-data``. Python's isolated mode
ignores inherited ``PYTHONPATH`` and the current working directory when selecting
the package. The synthetic notebook kernel receives the parent test's active
package path and checks that both processes import the same package file.
The separate source-quality job retains source lint, snapshot/dependency checks
and the documentation build against the checkout.
Missing, disabled, unreleased or unverified manifest-backed data fail those
tests instead of being skipped. This mode preserves the registry's approval
requirements and does not enable separate external-data tests.

Action dependency maintenance
-----------------------------

All repository-owned workflows pin external GitHub Actions to full commit
identifiers, including the action that receives the publishing job's OIDC
token. Inline comments identify the original version reference. The local
workflow regression rejects mutable tags and branches; the release jobs retain
their single-build, required-quality and three-platform wheel-test contracts.

``.github/dependabot.yml`` configures weekly update proposals for GitHub Actions
in the repository root. The configuration follows GitHub's `Action update
instructions
<https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/auto-update-actions>`_.
Review proposed upstream changes and the resolved commit before accepting an
update. Changes require ordinary review; no automatic merging is configured.
Hosted update proposals and workflow execution still need verification after
an authorized repository transfer and push.

The pins cover direct Action references. The owner reviewed the official PyPI
publisher's transitive runtime inputs on 12 October 2026 and chose to retain it
with the existing pins and reviewed update proposals. Its pinned
`Dockerfile
<https://github.com/pypa/gh-action-pypi-publish/blob/dc37677b2e1c63e2034f94d8a5b11f265b73ba33/Dockerfile>`_
uses the mutable ``python:3.13-slim`` container base and downloads
version-constrained runtime packages without artifact hash locks. The optional
host-Python setup uses ``3.x`` only when no ``python3`` executable is present.
This choice accepts variation in the publisher runtime. Review these inputs
when reviewing Action updates; a project-owned
fork or frozen replacement is not required by the current release contract.
This choice does not grant any outstanding release or publication approval.
