import numpy as np
import pytest


@pytest.mark.parametrize(
    ("specs", "labels"),
    [
        ([np.array([1.0, 0.5])], ["only"]),
        (
            [np.array([1.0, 0.5]), np.array([0.25, 1.0])],
            ["first", "second"],
        ),
    ],
)
def test_speck_simulator_plots_selected_phase_series(monkeypatch, specs, labels):
    from hypermri.visualization import spectrum_simulator as specsimulator

    simulator = specsimulator.SpeckSimulator(
        specs=specs,
        xaxis_hz=np.array([-10.0, 10.0]),
        labels=labels,
    )
    simulator.simulate(np.array([0.0, 0.01, 0.02]))

    plotted = []

    class FakeAxes:
        def plot(self, x, y, **kwargs):
            plotted.append((np.asarray(x), np.asarray(y), kwargs))

        def set_xlabel(self, _label):
            pass

        def set_ylabel(self, _label):
            pass

        def set_title(self, _title):
            pass

        def legend(self):
            pass

    class FakeFigure:
        def tight_layout(self):
            pass

    monkeypatch.setattr(
        specsimulator.plt, "subplots", lambda: (FakeFigure(), FakeAxes())
    )

    simulator.plot_selection_phase()

    assert [call[2]["label"] for call in plotted] == labels
    phase_matrix = np.asarray(simulator.phases).reshape(
        len(simulator.timepoints), len(specs)
    )
    for spectrum_idx, (x_values, y_values, _kwargs) in enumerate(plotted):
        np.testing.assert_allclose(x_values, simulator.timepoints * 1e3)
        np.testing.assert_allclose(y_values, phase_matrix[:, spectrum_idx])
