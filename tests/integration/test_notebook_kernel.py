"""Exercise synthetic HyperMRI workflows in a real notebook kernel.

These opt-in tests require local kernel sockets and launch the current Python
interpreter with the package selected by the parent test. They use no acquired
data and do not modify existing notebooks or kernels. Frontend rendering in
VS Code still requires a manual smoke check.
"""

from __future__ import annotations

import sys
import time
from importlib.metadata import version
from pathlib import Path

import pytest
from jupyter_client import KernelManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_T1_FIXTURE = PROJECT_ROOT / "tests/fixtures/synthetic_t1.py"

pytestmark = pytest.mark.integration


@pytest.fixture
def notebook_client(tmp_path, package_subprocess_env):
    """Start an isolated kernel using the parent test's active package.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary working directory for the connection file and kernel caches.
    package_subprocess_env : dict of str to str
        Independent child environment selecting the package imported by pytest,
        either the source checkout or an installed wheel.

    Yields
    ------
    jupyter_client.BlockingKernelClient
        Ready client using the current Python interpreter and local sockets.
        Channels and the kernel are closed when the fixture finishes, including
        when setup or a test fails. No acquired data or existing kernel is used.
    """
    manager = KernelManager(
        kernel_name="python3", connection_file=str(tmp_path / "kernel.json")
    )
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    environment = package_subprocess_env.copy()
    environment.update(
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        MPLBACKEND="Agg",
        JUPYTER_RUNTIME_DIR=str(tmp_path),
        IPYTHONDIR=str(tmp_path / "ipython"),
    )
    client = None
    try:
        manager.start_kernel(cwd=str(tmp_path), env=environment)
        client = manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=30)
        yield client
    finally:
        if client is not None:
            client.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def _execute(client, source):
    reply = client.execute_interactive(
        source, timeout=90, output_hook=lambda _message: None
    )
    assert reply["content"]["status"] == "ok", "\n".join(
        reply["content"].get("traceback", [])
    )


def test_fit_viewer_widgets_and_queued_cells_in_notebook_kernel(notebook_client):
    """Fit the public synthetic phantom, update widgets, and finish queued cells."""
    import hypermri

    client = notebook_client
    _execute(
        client,
        "from importlib.metadata import version\n"
        "from pathlib import Path\n"
        "import hypermri\n"
        f"assert version('ipykernel') == {version('ipykernel')!r}\n"
        "assert Path(hypermri.__file__).resolve() == "
        f"Path({str(Path(hypermri.__file__).resolve())!r}), hypermri.__file__",
    )

    _execute(
        client,
        "import runpy\n"
        f"globals().update(runpy.run_path({str(SYNTHETIC_T1_FIXTURE)!r}))",
    )
    _execute(client, "assert t1_mae_ms < 50.0\nassert signal_rmse < 10.0")

    _execute(
        client,
        """import asyncio
import ipywidgets as widgets
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from types import SimpleNamespace
from hypermri.visualization import PlotOptions, plot

synthetic_experiment = SimpleNamespace(
    primary_data=measured, is_spectroscopic=False, type="synthetic:T1_IR"
)

figure = go.FigureWidget(data=[go.Scatter(y=[0.0, 1.0])])
slider = widgets.IntSlider(min=0, max=4)
def update(change):
    figure.data[0].y = [0.0, float(change['new'])]
slider.observe(update, names='value')
for value in range(1, 5):
    slider.value = value
assert tuple(figure.data[0].y) == (0.0, 4.0)
slider.unobserve(update, names='value')
slider.close()
figure.close()

view = plot(synthetic_experiment, PlotOptions(plot_height=300))
view.plot_height = 360
assert view.plot_height == 360
await asyncio.sleep(0.3)
view.close()

plt.close('all')
get_ipython().run_line_magic('matplotlib', 'widget')
from ipympl.backend_nbagg import Canvas
interactive_figure, interactive_axes = plt.subplots()
interactive_axes.plot([0.0, 1.0], [0.0, 1.0])
interactive_figure.canvas.draw()
assert isinstance(interactive_figure.canvas, Canvas)
plt.close('all')
get_ipython().run_line_magic('matplotlib', 'inline')
queued_counter = 0
""",
    )

    pending = {client.execute("queued_counter += 1", silent=True) for _ in range(40)}
    deadline = time.monotonic() + 30
    while pending:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"{len(pending)} notebook cells did not finish"
        reply = client.get_shell_msg(timeout=remaining)
        message_id = reply["parent_header"].get("msg_id")
        if message_id in pending:
            assert reply["content"]["status"] == "ok", reply["content"]
            pending.remove(message_id)
    _execute(client, "assert queued_counter == 40")
