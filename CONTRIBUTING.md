# Contributing to HyperMRI

Contributions are welcome through GitHub issues, discussions, and pull
requests. HyperMRI is currently maintained as a hobby research project, so no
response or review time is guaranteed.

## Before opening a pull request

1. Open an issue or discussion first for large API, architecture, dependency,
   or scientific-method changes.
2. Install the development environment with `pip install -e ".[dev,docs]"`.
3. Add focused tests for changed behaviour and update the relevant
   documentation.
4. Run Ruff and the affected tests. For broad changes, run the complete unit
   suite and documentation build listed in `docs/source/development/setup.rst`.
5. Do not include patient data, identifying metadata, credentials, private
   contact details, or research data that is not explicitly cleared for public
   release.

Changes to numerical methods should identify the scientific reference, expected
result, and tolerance used for validation. A result must not be presented as a
new scientific reference value until the maintainer has reviewed it.

## Commit and CI test gate

After installing the development environment, enable the versioned hooks in
each clone:

```sh
python -m pre_commit install
git config --local hypermri.testPython "$(python -c 'import sys; print(sys.executable)')"
```

The second command records the active environment's absolute Python path in
this clone's local Git configuration. GUI clients such as Fork can then run
tests without inheriting an activated terminal environment. Repeat it after
moving or recreating the environment; this path is not committed or shared.

The `unit-tests` pre-commit hook runs `bash scripts/hooks/run_tests.sh` on
every branch and every commit, including documentation-only commits. Test
failures, collection errors or an unavailable test environment stop the commit.
The launcher uses `hypermri.testPython` when configured, otherwise `python`
from `PATH`. An invalid configured path fails instead of selecting another
environment. The interpreter must use Python 3.12 and have `.[dev]` installed.
The runner imports `src` from this checkout, uses a fresh bytecode cache, and
returns pytest's exit code.
Pre-commit temporarily hides unstaged tracked changes while testing the staged
revision. Include all required new source/test files in the staging area.

Run the configured gate manually with `bash scripts/hooks/run_tests.sh`, or
use `python scripts/hooks/run_tests.py` with an explicitly active environment.
To exercise the staged hook, run `python -m pre_commit run unit-tests`.

The gate collects `tests/unit` only. Tests that import the validation and
exploration code in `scripts` belong in `tests/validation` and are opt-in; run them after changing that code with
`PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests/validation -o addopts= -q`.

Local Git hooks can be bypassed and do not provide server enforcement.
The GitHub workflow `.github/workflows/test.yml` runs the unit gate, lint and
the existing 46% coverage gate under Python 3.12 for pushes and pull
requests. It also checks installed dependencies and runs the synthetic
notebook-kernel test. To enforce it, a maintainer must require this check
and protect the target branch in the repository settings on GitHub; see
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

The scientific evidence for the spectroscopy fitter is recorded in
[CSI fitter validation](docs/source/development/csi_fitter_validation.rst).

## Review and acceptance

Pull requests target `main`. They must pass the required automated checks and
be approved by the maintainer named in `MAINTAINERS.md` before they are merged.
Registry presence alone does not promote an implementation to the stable public
API; support-level changes must also update the public API and support matrix.

## Rights and licensing

Contributors retain copyright in their contributions. By submitting a
contribution, you confirm that you have the right to submit it and agree that it
may be distributed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`) stated in `LICENSE`. HyperMRI does not currently require a
Contributor License Agreement, copyright assignment, or Developer Certificate
of Origin.
