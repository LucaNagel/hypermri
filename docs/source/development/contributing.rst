Contributing
============

1. Fork the repository and install development dependencies with ``pip install -e ".[dev,docs]"``.
2. Run ``pre-commit run --all-files`` before submitting pull requests.
3. Run the focused unit tests for the code you changed. For broad changes, run ``pytest tests/unit --tb=short -q``.
4. Add or update documentation alongside code changes, especially when registries, sequence loading, reconstruction strategies, or vendor handlers are modified.

Pull requests must pass the required automated checks and be approved by the
maintainer before they are merged into ``main``. Contributors retain copyright
in their work. By submitting a contribution, you confirm that you have the
right to do so and agree that it may be distributed under the project license
in ``LICENSE`` (``AGPL-3.0-only``). HyperMRI does not currently require a
Contributor License Agreement, copyright assignment, or Developer Certificate
of Origin.

The complete policy, including research-data safety and scientific validation,
is in the repository-level ``CONTRIBUTING.md``.

Sequence classes should not trigger automatic reconstruction in ``__init__``.
Use ``BaseSequence.run_default_reconstructions`` and class-level
``RECONSTRUCTION_POLICY`` overrides instead. New reconstruction code should live
under ``hypermri.reconstruction.strategies``.

Refer to :doc:`setup` for the environment, Git hooks and required checks,
and to :doc:`testing` for the test suite layout.
