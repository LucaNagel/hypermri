# Testing Guide

This guide covers everything you need to know about testing the hypermri package, from running existing tests to writing new ones.

## Table of Contents

- [Quick Start](#quick-start)
- [Spectroscopy Fitting Baseline](#spectroscopy-fitting-baseline)
- [Test Types](#test-types)
- [Setting Up Test Data](#setting-up-test-data)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install development dependencies used by CI
pip install -e ".[dev]"

# Run the CI unit-test subset
pytest tests/unit --tb=short -q

# Run only fast unit tests
pytest tests/unit

# Run with coverage report
pytest --cov=hypermri --cov-report=html
```

## Spectroscopy Fitting Baseline

The former tutorial scripts have been retired. Synthetic workflow tests now
exercise CSI fitting, T1 fitting and overlays, metabolite detection, noise
estimation and uncertainty without depending on documentation files:

```bash
python -m pytest tests/unit/test_synthetic_workflows.py tests/unit/analysis/test_synthetic_spectroscopy_uncertainty.py
```

Their fixtures in `tests/fixtures` record the invented signals, seeds, units
and tolerances. They require no acquired data or downloads.

Before simplifying spectroscopy fitting, run its small numerical baseline:

```bash
python -m pytest tests/unit/analysis/fitting/test_spectroscopy_numerical_reference.py
```

The numerical reference tests exercise the real fitter with synthetic signals
based on
`08_si_fit_quality_estimation_model_comparison.ipynb`: 256 FID samples at
0.4 ms, five metabolites with T2* = 25 ms, and bicarbonate amplitudes of 0
and 0.05. One Gaussian noise realization is paired across these amplitudes;
SNR = 50 means pyruvate FID amplitude divided by noise sigma in each complex
quadrature. The fixture records the seed, rounded frequencies and acquisition
parameters, so the tests require neither the notebook nor measured data.

The checks cover recovery from shifted initial parameters, reconstruction,
result units and metadata, agreement after transposing the spectral axis
in serial execution (with and without progress) and two-worker execution, and
identical results through the generic API, the old `spectroscopic_fitting`
import and the model-module compatibility function. A further comparison uses
the notebook's Hz parameter spelling without any frequency calibration axes.
Numerical tolerances are stated in the tests. A missing metabolite has no
identifiable frequency or T2*, and fitting noise can give it a nonzero amplitude.
This baseline does not estimate detection power or false-positive rates.

The existing execution-mode comparison in `test_spectroscopy_pipeline.py` also
uses voxel-dependent parameter maps whose dimension order differs from the
input. A small solver stub checks alignment, batch ordering, identical result
metadata and unchanged input data/parameters without additional optimizer runs.

The workflow lives in `hypermri.analysis.fitting.spectroscopy`.
`test_migration_characterization.py` checks that the legacy module exports the
same functions. Tests that replace the workflow's single-spectrum solver must
patch the canonical module, where the execution functions resolve that name.

`test_spectroscopy_parameter_preparation.py` covers finite scalar/per-metabolite
settings, ordered T2* bounds, aliases and safe coordinate alignment when reusing
fit results. Most cases only prepare parameters. One eight-sample fit verifies
that zero search widths keep frequencies and T2* fixed; the noisy numerical
baseline covers normal search behavior.

`test_spectroscopy_grid_search.py` adds four small, 64-sample checks for the
frequency and T2* searches. They verify both diagnostic SSE columns against the
closed-form amplitude of a single basis vector, selection by the requested
error measure, equivalence of FID/spectrum input, and unchanged input arrays.
The cases cover pinv and ridge and include non-positive T2* candidates. This
keeps direct search coverage small while exercising the real numerical code.

`test_spectroscopy_amplitude_solver.py` adds five amplitude-only cases for the
shared solver selection. A 32-sample, noisy two-peak signal checks aliases,
defaults, explicit-solver precedence and FID/spectrum equivalence. Pinv and
ridge are compared with independent least-squares references; normal_eq uses
values recorded before consolidation to retain its plain-transpose behavior.
Two identical components exercise the singular ridge fallback at zero lambda.
No extra frequency/T2* optimization is needed for these checks.

`test_spectroscopy_single_spectrum.py` checks the outer workflow with four small
cases: all-zero/all-NaN shortcuts, fitting the same prefix from FID/spectrum
input, and retaining accepted parameters when later scores tie or worsen.
Controlled proposals check shrinking widths and diagnostic order without a
large optimizer run. The existing amplitude-threshold test in
`test_spectroscopy_pipeline.py` also checks equality at the threshold, exclusion
of all components, and subset refits using original widths with filtering off.

`test_spectroscopy_skip_fitting.py` uses three 64-sample voxels to compare masked
fits with explicit fits of the active subsets. Its five cases cover serial,
serial-with-progress and process execution, non-contiguous active metabolites,
mask/parameter dimension alignment, a skipped peak closer than the minimum
separation, and amplitude filtering within each active subset. They also check
zero reconstruction for fully masked voxels, no numerical work for scalar
`skip_fitting=True`, unchanged inputs and rejection of NaNs in edited masks.

Two API details matter when reproducing the fixture:

- Use `metabs_freqs_hz` or the accepted alias `metabs_freqs_Hz`. Explicit Hz
  values work without a ppm axis; unavailable ppm metadata stays `None`.
  The small `test_spectroscopy_frequency_params.py` checks normalization,
  conflicting aliases, invalid values and missing ppm calibration without
  running the optimizer.
- `reconstruct_fitted_signal(fit_results, input_data)` returns the total fitted
  signal on the input coordinates, allowing direct residual calculations.
  `test_reconstruct_fitted_signal.py` checks FIDs, spectra, mismatched grids,
  invalid component values and compatibility with the existing component-wise
  reconstruction accessor. These checks use known parameters and do not run
  the optimizer.

## Test Types

The hypermri package uses three tiers of tests:

### Unit Tests
- **Purpose**: Test individual functions/classes in isolation
- **Data**: Use synthetic/mock data
- **Speed**: Fast (< 1 second per test)
- **Coverage**: Focus on logic, edge cases, error handling
- **When to run**: Every commit, during development

```bash
# Run only unit tests
pytest tests/unit
# or exclude integration and validation tests
pytest -m "not integration and not validation"
```

### Integration Tests
- **Purpose**: Test end-to-end workflows with real data
- **Data**: Use the checksum-pinned public Zenodo reference corpus
- **Speed**: Slow and storage-heavy; the compressed download is 783.7 MB
- **Coverage**: Test file loading, stable Bruker raw reconstruction, dimensions,
  coordinates, and numerical reference contracts
- **When to run**: Before releases and after reconstruction or Bruker loading
  changes

```bash
# Run only integration tests
pytest -m integration

# Run the numerical reference contract
pytest tests/integration/test_brukerapi_pv601_reference.py -m integration -v
```

### Validation Tests
- **Purpose**: Check the validation and exploration code in `scripts`, which
  is not part of the installed package, such as the 13C spectroscopy
  simulation and the notebook fitting explorer
- **Data**: Small synthetic data generated by the tested code
- **Location**: `tests/validation`; every test there carries the `validation`
  marker
- **When to run**: After changing these scripts, or the
  spectroscopy fitting code they call. They are not part of the unit gate:
  `scripts/hooks/run_tests.py` and the CI unit jobs collect `tests/unit` only

```bash
# Run the validation tests from the repository root
PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests/validation -o addopts= -q
```

## Setting Up Test Data

### Option 1: Unit Tests Only (Recommended for Development)

No setup needed! Unit tests use synthetic data and run immediately:

```bash
pytest tests/unit
```

### Numerical End-to-End Reference Data

The stable Bruker raw-reconstruction allowlist is validated against Tomáš
Pšorn's public **BrukerAPI test data set (Bruker ParaVision v6.0.1)**, version
1.1.0, DOI [`10.5281/zenodo.4048253`](https://doi.org/10.5281/zenodo.4048253),
licensed under CC BY 4.0. The corpus contains Lego-phantom acquisitions for all
ten stable sequence IDs and includes `fid.npz`/`2dseq.npz` reference arrays.

The pinned archive is 783,667,029 bytes compressed and approximately 2.2 GB
after extraction. It is therefore an opt-in integration-test dependency, not a
unit-test fixture. The archive remains on Zenodo and is never committed to this
repository or included in package artifacts.

The executable numerical contract is deliberately explicit:

- Bruker stock CSI must match the published `fid.npz[70:]` array exactly: the
  first 70 digital-filter/transient points are removed once from every FID;
- ISIS, NSPECT, PRESS, Singlepulse, and STEAM must match the published FID
  after HyperMRI's documented cutoff and FFT;
- FLASH, RARE, FAIR_RARE, and RAREVTR are compared frame by frame with the
  scanner-reconstructed `2dseq.npz` magnitude images. Each frame is
  independently L2-normalised to exclude scanner display scaling, while
  requiring an L2 distance below `1e-3` and Pearson correlation above
  `0.99999`. No compensating flips or translations are permitted;
- FAIR_RARE must preserve all six frames as three inversion times by two FAIR
  modes, and RAREVTR must preserve all three variable-TR frames.

For the pinned stock-CSI reference scan, `fid.npz` contains 2048 samples per
FID and `raw-reco` therefore contains 1978 samples. The handler-level `fid`
remains the complete 2048-sample acquisition so reconstruction is repeatable
and an explicit no-cutoff workflow remains possible. HyperMRI rebuilds the
processed time and frequency axes for 1978 samples, with processed FID time
starting at zero; it does not slice a precomputed 2048-point frequency axis.
Scanner-reconstructed `2dseq`/`scanner-reco` data is not cut a second time.

These checks are a regression oracle for HyperMRI's ParaVision import and raw
reconstruction. They do not turn the BrukerAPI corpus into independent
scientific ground truth for every acquisition protocol or quantitative MRI
claim.

### Option 2: Full Testing with Public Reference Data

Keep downloaded data outside Git history. The local `tests/data/` directory is
ignored except for its README and must not be copied into a public snapshot.
For a reusable external cache, run:

```bash
export HYPERMRI_TEST_DATA="/path/to/an/external/cache"
python scripts/download_test_data.py --dataset brukerapi_pv601_lego_phantom
pytest tests/integration -m integration --require-real-data -v
```

The manifest pins SHA-256
`fc273fe4d498f99cd75c4ae39cb64ba0522317c4f6abee1ad40206a2a70fc203`.
The downloader verifies the checksum and validates every ZIP member before
extraction. Repeated runs reuse the cached archive unless `--no-cache` is
passed. After successful extraction it writes
`.hypermri-test-data.json` beside the study. The test fixture accepts an
existing directory only when this marker still matches the dataset ID,
relative path, archive name, byte size, and checksum in the current manifest.
By default, unavailable or unverified reference data skip the affected tests.
Pass `--require-real-data` to fail those tests instead, including disabled or
unreleased reference datasets. This option never downloads or approves data.

Although this is a phantom dataset rather than a human or animal-subject
acquisition, its public scanner metadata contains the creator/operator name,
local workstation paths, exact timestamps, UIDs, and hardware identifiers.
Do not copy or quote those metadata into test logs, fixtures, documentation, or
issues. HyperMRI references and downloads the original CC BY 4.0 record instead
of redistributing a modified archive.

Normal pull-request CI does not download 783.7 MB. The dedicated real-data job
runs only when the GitHub repository variable
`ENABLE_APPROVED_REAL_DATA_TESTS=true`; its archive is cached by checksum.
That job and the release-quality workflow use `--require-real-data` so missing
or unverified manifest-backed data cannot produce a successful skipped run.
Separate external-data tests retain their explicit opt-in requirements.
For release candidates and final releases, the publishing workflows test the
same built wheel on Linux, macOS and Windows with Python 3.12. All three jobs
are required; source-installation checks also require all three platforms.

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test file
pytest tests/unit/core/test_experiment.py

# Run specific test method
pytest tests/unit/core/test_experiment.py::TestMRIExperimentInitialization::test_init_basic
```

### Test Selection

```bash
# By markers
pytest tests/unit                 # Only unit tests
pytest -m integration             # Only integration tests
pytest -m validation              # Only validation tests
pytest -m "not slow"             # Exclude slow tests

# By keyword matching
pytest -k "test_bruker"          # Tests with "bruker" in name
pytest -k "not siemens"          # Exclude tests with "siemens"

# By test path
pytest tests/unit/               # Only unit tests directory
pytest tests/integration/       # Only integration tests
pytest tests/validation/        # Only validation tests
```

### Coverage Reports

```bash
# Terminal coverage report
pytest --cov=hypermri

# HTML coverage report (opens in browser)
pytest --cov=hypermri --cov-report=html
open htmlcov/index.html

# Coverage with missing line numbers
pytest --cov=hypermri --cov-report=term-missing
```

### Parallel Testing (Faster)

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto                  # Auto-detect CPU cores
pytest -n 4                     # Use 4 processes
```

## Writing Tests

### Test Organization

```
tests/
├── unit/                      # Unit tests (fast, mock data)
│   ├── core/                 # Core functionality tests
│   ├── vendors/              # Vendor-specific tests
│   ├── utils/                # Utility function tests
│   └── sequences/            # Sequence-specific tests
├── integration/              # Integration tests (real data)
├── validation/               # Opt-in tests of validation scripts
├── conftest.py               # Shared pytest fixtures
├── real_data_manifest.json   # Real-data dataset locations and checksums
└── data/                    # Small test files (if any)
```

### Writing Unit Tests

Use mocks and synthetic data for fast, isolated tests:

```python
"""Example unit test"""
import pytest
from unittest.mock import Mock, patch
from hypermri.loading.experiment import MRIExperiment

@pytest.mark.unit
class TestMyFunction:
    """Test my specific function."""

    def test_basic_functionality(self, temp_experiment_dir):
        """Test basic case with mock data."""
        with patch.object(MRIExperiment, '_get_vendor_handler') as mock_handler:
            mock_handler.return_value = Mock()
            mock_handler.return_value.id = "test_001"

            exp = MRIExperiment(temp_experiment_dir)

            assert exp.handler.id == "test_001"
            mock_handler.assert_called_once()

    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError, match="Invalid path"):
            MRIExperiment("/nonexistent/path")

    @pytest.mark.parametrize("vendor", ["bruker", "siemens", "agilent"])
    def test_multiple_vendors(self, vendor, temp_experiment_dir):
        """Test with different vendors."""
        with patch.object(MRIExperiment, '_get_vendor_handler'):
            exp = MRIExperiment(temp_experiment_dir, force_vendor=vendor)
            assert exp._forced_vendor == vendor
```

### Writing Integration Tests

Use the manifest-backed ``real_dataset_path`` fixture. It skips cleanly when
the optional corpus is absent and rejects unverified directories without the
provenance marker written by ``scripts/download_test_data.py``. With
`--require-real-data`, these conditions fail instead of skipping. Unknown
dataset IDs always fail:

```python
from pathlib import Path

import pytest
from hypermri import create_experiment


@pytest.mark.integration
def test_reference_scan_detects_bruker(real_dataset_path):
    study = Path(real_dataset_path("brukerapi_pv601_lego_phantom"))
    experiment = create_experiment(study / "3", load_data=False)

    assert experiment.vendor == "bruker"
    assert experiment.path == (study / "3").resolve()
```

Use the existing
``tests/integration/test_brukerapi_pv601_reference.py`` parametrization when a
new assertion belongs to the stable sequence contract. Do not add locally
owned or unpublished scanner data to the public integration suite.

### Test Fixtures

Create reusable fixtures for common test needs:

```python
"""Custom fixtures example"""
import pytest
import numpy as np
import xarray as xr

@pytest.fixture
def synthetic_spectrum():
    """Create synthetic spectroscopy data for testing."""
    n_points = 1024
    freq_axis = np.linspace(12, -12, n_points)

    # Create realistic peaks
    signal = np.zeros(n_points, dtype=complex)
    # Water peak at 4.7 ppm
    water_idx = np.argmin(np.abs(freq_axis - 4.7))
    signal[water_idx-5:water_idx+5] = 100 * np.exp(-0.5 * ((np.arange(-5, 5) / 2)**2))

    return xr.DataArray(
        signal,
        dims=["frequency"],
        coords={"frequency": freq_axis},
        attrs={"sequence": "PRESS", "te_ms": 30}
    )

@pytest.fixture(params=["bruker", "siemens", "agilent"])
def vendor_name(request):
    """Parametrized fixture for testing multiple vendors."""
    return request.param
```

### Testing Best Practices

#### 1. Test Naming
```python
# Good: Descriptive test names
def test_bruker_experiment_loads_method_parameters():
    pass

def test_spectroscopy_fitting_handles_noisy_data():
    pass

def test_coordinate_transformation_preserves_spatial_relationships():
    pass

# Bad: Vague test names
def test_data():
    pass

def test_function():
    pass
```

#### 2. Test Structure (Arrange-Act-Assert)
```python
def test_experiment_initialization():
    # Arrange
    experiment_dir = create_test_experiment()
    expected_vendor = "bruker"

    # Act
    exp = MRIExperiment(experiment_dir, force_vendor=expected_vendor)

    # Assert
    assert exp._forced_vendor == expected_vendor
    assert exp.path == experiment_dir
```

#### 3. Error Testing
```python
import pytest
from hypermri import load_scan
from hypermri.exceptions import DataLoadingError


def test_invalid_experiment_path_raises_error(tmp_path):
    """Test that invalid paths raise appropriate errors."""
    with pytest.raises(DataLoadingError, match="does not exist"):
        load_scan(tmp_path / "missing_scan", mode="lazy")


def test_invalid_load_mode_raises_error(tmp_path):
    """Reject an unsupported mode before vendor detection or data loading."""
    with pytest.raises(ValueError, match="mode"):
        load_scan(tmp_path, mode="unsupported")
```

#### 4. Mock External Dependencies
```python
@patch('hypermri.external.some_slow_function')
@patch('hypermri.external.network_request')
def test_processing_workflow(mock_network, mock_slow_func):
    """Test workflow without external dependencies."""
    mock_slow_func.return_value = expected_result
    mock_network.return_value = mock_response

    result = process_experiment(test_data)

    assert result == expected_result
    mock_slow_func.assert_called_once()
```

## Continuous Integration

### Current GitHub Actions

`.github/workflows/test.yml` is the source of truth; do not duplicate its YAML
configuration in this guide. It runs linting, dependency validation, coverage,
and unit tests on Python 3.12. It also checks installed dependencies and runs
`tests/integration/test_notebook_kernel.py`, which uses synthetic data and a
real local kernel without downloading reference data. The real-data job requires
`ENABLE_APPROVED_REAL_DATA_TESTS=true`, and the manifest independently blocks
unreleased data.

### Pre-commit Hooks

The active pre-commit configuration runs file hygiene checks, YAML validation,
Ruff linting, and Ruff formatting. It does not currently run pytest, so run the
focused test suite yourself before committing.

## Troubleshooting

### Common Issues

#### 1. Tests are being skipped
```bash
# Verify/download the pinned integration corpus into an external cache
export HYPERMRI_TEST_DATA="/path/to/an/external/cache"
python scripts/download_test_data.py --dataset brukerapi_pv601_lego_phantom

# Unit tests do not need real data
pytest tests/unit
```

Real-data tests skip when the corpus is missing, incomplete, disabled, or lacks
the provenance marker written after archive-size and SHA-256 verification. The
`--require-real-data` option makes those conditions fail in mandatory runs. The
packaged tutorial-data registry and `download_example_data.py` are separate
from this integration-test corpus.

#### 2. Import errors
```bash
# Make sure package is installed in development mode
pip install -e .

# Check Python path
python -c "import hypermri; print(hypermri.__file__)"

# Test specific import
python -c "from hypermri.loading.experiment import MRIExperiment"
```

#### 3. Permission errors with test data
```bash
# Fix permissions
chmod -R 755 $HYPERMRI_TEST_DATA

# Check ownership
ls -la $HYPERMRI_TEST_DATA
```

#### 4. Slow test performance
```bash
# Run only fast tests
pytest -m "not slow"

# Use parallel execution
pip install pytest-xdist
pytest -n auto

# Profile slow tests
pytest --durations=10
```

#### 5. Coverage issues
```bash
# Generate detailed coverage report
pytest --cov=hypermri --cov-report=html --cov-report=term-missing

# Check what's not covered
open htmlcov/index.html
```

### Debugging Tests

#### 1. Print debugging in tests
```python
def test_my_function(capsys):
    """Test with output capture."""
    result = my_function(test_input)
    print(f"Debug: result = {result}")  # This will be captured

    captured = capsys.readouterr()
    assert "expected" in captured.out
```

#### 2. Interactive debugging
```python
def test_with_debugger():
    """Test with debugger breakpoint."""
    import pdb; pdb.set_trace()  # Debugger will stop here
    result = function_under_test()
    assert result == expected
```

#### 3. Temporary test isolation
```bash
# Run single test with maximum verbosity
pytest tests/unit/core/test_experiment.py::test_init_basic -vvv -s

# Don't capture output
pytest --capture=no -v
```

### Getting Help

- **Documentation**: Check existing tests in `tests/` for examples
- **Issues**: Create GitHub issue with test output and environment info
- **Discussions**: Use GitHub Discussions for testing questions
- **Code Review**: Add tests to pull requests for review

## Contributing Tests

When contributing new features or fixing bugs:

1. **Always add tests** for new functionality
2. **Update existing tests** if changing behavior
3. **Run full test suite** before submitting PR
4. **Include both unit and integration tests** when appropriate
5. **Document any new test data requirements**

### PR Checklist
- [ ] Added unit tests for new functionality
- [ ] Added integration tests if applicable
- [ ] All existing tests pass
- [ ] Coverage does not regress meaningfully
- [ ] Tests are documented and well-named
- [ ] No test data committed to repository
