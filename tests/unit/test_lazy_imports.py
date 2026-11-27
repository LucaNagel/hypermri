"""Isolated import and missing-feature contracts for the supported public APIs."""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _run(source, tmp_path, package_subprocess_env):
    env = package_subprocess_env.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_BLOCKER = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('invented missing ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
"""


def test_fresh_process_imports_the_active_package(tmp_path, package_subprocess_env):
    """Release wheel checks must not switch back to the source checkout."""
    import hypermri

    expected = str(Path(hypermri.__file__).resolve())
    _run(
        "from pathlib import Path\nimport hypermri\n"
        f"assert str(Path(hypermri.__file__).resolve()) == {expected!r}, "
        "hypermri.__file__\n",
        tmp_path,
        package_subprocess_env,
    )


def test_root_import_needs_only_the_standard_library(tmp_path, package_subprocess_env):
    """Importing and discovering exports must not initialize numerical or UI code."""
    _run(
        "blocked = {'numpy', 'matplotlib', 'IPython', 'ipywidgets', 'plotly', 'git'}\n"
        + _BLOCKER
        + """
import hypermri
assert 'Study' in dir(hypermri)
assert 'BrukerHandler' not in dir(hypermri)
assert 'hypermri.vendors' not in sys.modules
assert 'hypermri.utils.plot' not in sys.modules
assert 'matplotlib' not in sys.modules
try:
    hypermri.invented_missing_name
except AttributeError:
    pass
else:
    raise AssertionError('unknown attributes must fail')
""",
        tmp_path,
        package_subprocess_env,
    )


def test_loading_and_numerical_analysis_do_not_need_notebook_packages(
    tmp_path, package_subprocess_env
):
    """Retain actual registry classes and numerical operations without any UI."""
    _run(
        "blocked = {'IPython', 'ipywidgets', 'plotly', 'anywidget', 'sympy', 'git', 'twixtools'}\n"
        + _BLOCKER
        + """
import numpy as np
from hypermri import Study, MRIExperiment, load_scan
from hypermri.vendors.registry import get_vendor_registry
from hypermri.sequences.registry import get_sequence_registry
from hypermri.analysis.spectroscopy import lorentzian, fit_spectrum
from hypermri.visualization import PlotOptions, plot
assert get_vendor_registry().get('bruker')['handler_class'].__name__ == 'BrukerHandler'
assert get_sequence_registry().get('csi')['sequence_class'].__name__ == 'CSI'
np.testing.assert_allclose(lorentzian(np.array([-1.,0.,1.]), 2., 0., 1.), [.5,1.,.5])
assert PlotOptions().plot_height is None
assert not any(name.split('.')[0] in blocked for name in sys.modules)
from hypermri.utils.plot import matplotlib_safety
assert matplotlib_safety._FALLBACK_INSTALLED is False
""",
        tmp_path,
        package_subprocess_env,
    )


def test_domain_vendor_imports_do_not_use_compatibility_modules(
    tmp_path, package_subprocess_env
):
    """Canonical vendors and registry stay usable without notebook dependencies."""
    _run(
        "blocked = {'IPython', 'ipywidgets', 'plotly', 'anywidget', 'git', 'twixtools'}\n"
        + _BLOCKER
        + """
from hypermri.vendors import BrukerTopSpinHandler, VendorHandler, get_vendor_registry
assert issubclass(BrukerTopSpinHandler, VendorHandler)
assert get_vendor_registry().get('bruker_topspin')['handler_class'] is BrukerTopSpinHandler
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""",
        tmp_path,
        package_subprocess_env,
    )


def test_named_fitting_support_runs_without_notebook_or_legacy_modules(
    tmp_path, package_subprocess_env
):
    """Numerical fitting support imports canonical owners with no UI dependency."""
    _run(
        "blocked = {'IPython', 'ipywidgets', 'plotly', 'anywidget', 'git', 'twixtools'}\n"
        + _BLOCKER
        + """
import numpy as np
import xarray as xr
from hypermri.analysis.dce.conversions import signal_to_concentration
from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor
from hypermri.analysis.fitting.result_metadata import dataset_reconstruction_values
from hypermri.analysis.quantitative import ADCFitOptions, fit_adc
assert ExecutionConfig(False, 1).number_of_cpu_cores == 1
assert VoxelwiseExecutor().__dict__ == {}
np.testing.assert_allclose(signal_to_concentration(
    np.array([100., 120.]), 100., mode='relative_signal_enhancement'), [0., .2])
b_values = np.array([0., 250., 700., 1200.])
signal = xr.DataArray(150. * np.exp(-b_values * .0011), dims='bvalue_idx',
                      coords={'bvalue_idx': b_values})
result = fit_adc(signal, ADCFitOptions(execution=ExecutionConfig(False, 1)))
np.testing.assert_allclose(result.adc, .0011, rtol=1e-10)
np.testing.assert_array_equal(dataset_reconstruction_values(result, 'b_values'), b_values)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""",
        tmp_path,
        package_subprocess_env,
    )


def test_slice_matching_imports_no_matplotlib_or_notebook_runtime(
    tmp_path, package_subprocess_env
):
    """Image-coordinate matching no longer imports its former plotting owner."""
    _run(
        "blocked = {'matplotlib', 'IPython', 'ipywidgets', 'plotly', 'anywidget'}\n"
        + _BLOCKER
        + """
import xarray as xr
from hypermri.utils.image.slices import get_overlapping_slices
first = xr.DataArray([0, 0], dims='slice_idx', coords={'slice_idx': [0, 1]})
first.slice_idx.attrs['axis_mm'] = [0., 5.]
assert get_overlapping_slices(first, first) == ([0, 1], [0, 1])
assert 'hypermri.utils.plot' not in sys.modules
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""",
        tmp_path,
        package_subprocess_env,
    )


def test_frequency_axes_and_calibration_need_no_plotting_or_legacy_core(
    tmp_path, package_subprocess_env
):
    """Canonical and package frequency paths avoid plotting and legacy processing."""
    _run(
        "blocked = {'matplotlib', 'IPython', 'ipywidgets', 'plotly', 'anywidget'}\n"
        + _BLOCKER
        + """
from types import SimpleNamespace
import numpy as np
from hypermri.analysis.spectroscopy import get_freq_axis
from hypermri.analysis.spectroscopy.frequency import (
    get_freq_axis as direct, freq_hz_to_ppm, freq_ppm_to_hz, _ppm_delta_to_hz,
)
assert get_freq_axis is direct
metadata = {'dwell_time_s': .001, 'center_ppm': 170.}
context = SimpleNamespace(bandwidth_hz=1000., bandwidth_ppm=10.,
                          n_acquisition=4, get=metadata.get)
hz = get_freq_axis(context, unit='Hz')
ppm = get_freq_axis(context, unit='ppm')
np.testing.assert_allclose(hz, [-500., -250., 0., 250.])
np.testing.assert_allclose(freq_hz_to_ppm(50., hz, ppm), 170.5)
np.testing.assert_allclose(freq_ppm_to_hz(170.5, hz, ppm), 50.)
np.testing.assert_allclose(_ppm_delta_to_hz(1., hz_axis=hz, ppm_axis=ppm), 100.)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""",
        tmp_path,
        package_subprocess_env,
    )


@pytest.mark.parametrize("missing", ["plotly", "ipywidgets", "anywidget", "cmap"])
def test_plot_missing_dependency_asks_to_reinstall_hypermri(
    tmp_path, missing, package_subprocess_env
):
    """A plot request in a broken environment fails before viewer creation."""
    _run(
        f"blocked = {{{missing!r}}}\n"
        + _BLOCKER
        + """
import numpy as np
import xarray as xr
from hypermri.visualization import plot
try:
    plot(xr.DataArray(np.ones((2, 3)), dims=('read_idx','phase_idx')))
except ImportError as exc:
    assert 'pip install hypermri' in str(exc), str(exc)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('missing optional dependency must fail')
assert 'hypermri.visualization.scan_view.widget' not in sys.modules
""",
        tmp_path,
        package_subprocess_env,
    )


def test_optional_import_does_not_relabel_a_broken_dependency(monkeypatch):
    """Keep failures inside an installed package visible with their original cause."""
    from hypermri import _optional

    error = ModuleNotFoundError("invented internal failure", name="unrelated_internal")

    def fail(_name):
        raise error

    monkeypatch.setattr(_optional, "import_module", fail)
    with pytest.raises(ModuleNotFoundError) as caught:
        _optional._import_optional("plotly")
    assert caught.value is error


def test_csi_fitting_runs_without_any_optional_runtime(
    tmp_path, package_subprocess_env
):
    """Execute the CSI numerical contract with notebook/vendor extras absent."""
    fixture = _ROOT / "tests/fixtures/synthetic_csi.py"
    _run(
        "blocked = {'IPython', 'ipywidgets', 'plotly', 'anywidget', 'sympy', 'git', 'twixtools'}\n"
        + _BLOCKER
        + f"""
import runpy
namespace = runpy.run_path({str(fixture)!r})
assert namespace['signal_rmse'] < 0.002
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""",
        tmp_path,
        package_subprocess_env,
    )


def test_styled_study_display_reports_a_missing_ipython(
    tmp_path, package_subprocess_env
):
    """Fail at the display request before accessing scan metadata or creating UI."""
    _run(
        "blocked = {'IPython'}\n"
        + _BLOCKER
        + """
from hypermri import Study
try:
    Study.__new__(Study).display()
except ImportError as exc:
    assert 'pip install hypermri' in str(exc)
else:
    raise AssertionError('styled display must require its optional runtime')
""",
        tmp_path,
        package_subprocess_env,
    )


def test_sequence_vendor_and_reconstruction_modules_load_no_plotting(
    tmp_path, package_subprocess_env
):
    """Data modules sit below visualization, so importing them loads no plotting.

    Every module of ``hypermri.sequences``, ``hypermri.vendors`` and
    ``hypermri.reconstruction`` is imported; plotting libraries and
    ``hypermri.visualization`` load only when a plotting method runs.
    """
    _run(
        """
import importlib
import pkgutil
import sys
for package_name in (
    'hypermri.sequences', 'hypermri.vendors', 'hypermri.reconstruction'
):
    package = importlib.import_module(package_name)
    for module in pkgutil.walk_packages(package.__path__, package_name + '.'):
        importlib.import_module(module.name)
loaded = sorted(
    name for name in sys.modules
    if name.split('.')[0] in {'matplotlib', 'plotly', 'ipywidgets'}
    or name == 'hypermri.visualization'
    or name.startswith('hypermri.visualization.')
)
assert not loaded, loaded
""",
        tmp_path,
        package_subprocess_env,
    )


def test_vendor_and_reconstruction_modules_load_no_analysis(
    tmp_path, package_subprocess_env
):
    """Vendors and reconstruction sit below analysis and never import it.

    Every module of ``hypermri.vendors`` and ``hypermri.reconstruction`` is
    imported in a fresh interpreter. Shared helpers such as the spectral
    frequency axis live in ``hypermri.reconstruction.axes``, so no module of
    ``hypermri.analysis`` may be loaded.
    """
    _run(
        """
import importlib
import pkgutil
import sys
for package_name in ('hypermri.vendors', 'hypermri.reconstruction'):
    package = importlib.import_module(package_name)
    for module in pkgutil.walk_packages(package.__path__, package_name + '.'):
        importlib.import_module(module.name)
loaded = sorted(
    name for name in sys.modules
    if name == 'hypermri.analysis' or name.startswith('hypermri.analysis.')
)
assert not loaded, loaded
""",
        tmp_path,
        package_subprocess_env,
    )
