"""Checks for the notebook-only spectroscopy-fitting explorer."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from scripts.interactive_spectroscopy_fitting import (  # noqa: E402
    _OMPD_PRESET,
    SpectroscopyFittingExplorer,
)


def test_renal_ompd_preset_updates_the_existing_figure_and_fits_current_widgets():
    """Keep artists while updating and fitting the configured Z-OMPD proxy."""
    explorer = SpectroscopyFittingExplorer()
    try:
        figure = explorer.figure
        spectrum_line = explorer._spectrum_data_line
        initial_fid = explorer.data.values.copy()
        history_count = int(explorer.fit_results.fit_history_count.item())
        history_info = explorer.fit_results.fit_history_info.isel(
            history_step=slice(history_count)
        ).values
        returned_rows = np.flatnonzero(history_info[:, 0] == 6)

        assert explorer._current_history_row() == int(returned_rows[-1])
        np.testing.assert_allclose(
            explorer.fit_results.fit_history_freqs.isel(history_step=returned_rows[-1]),
            explorer.fit_results.fit_freqs,
        )
        assert explorer._spectrum_model_line.get_xdata().size

        explorer.event_filter.value = "Frequenz-Sweep"
        assert explorer.step_slider.max > 0
        assert "frequency_candidate" in explorer._parameter_text.get_text()
        explorer.history_metabolite.value = 1
        selected_info = history_info[explorer._history_indices]
        assert np.all(selected_info[:, 0] == 1)
        assert np.all(selected_info[:, 4] == 1)

        explorer._component_widgets[0].amplitude.value = 0.9

        assert explorer.fit_results is None
        assert not np.allclose(explorer.data.values, initial_fid)
        assert not len(explorer._spectrum_model_line.get_xdata())
        assert not len(explorer._fid_model_line.get_xdata())
        assert not len(explorer._fid_residual_line.get_xdata())
        assert not any(line.get_visible() for line in explorer._fit_frequency_lines)
        assert all(line.get_visible() for line in explorer._true_frequency_lines)
        assert explorer.history_metabolite.value is None

        explorer.grid_points.value = 5
        explorer.iterations.value = 1
        explorer.repetitions.value = 1
        explorer._run_fit()

        refreshed_history_count = int(explorer.fit_results.fit_history_count.item())
        refreshed_info = explorer.fit_results.fit_history_info.isel(
            history_step=slice(refreshed_history_count)
        ).values
        assert explorer.event_filter.value == "Alle Schritte"
        assert explorer.history_metabolite.value is None
        assert refreshed_info[explorer._current_history_row(), 0] == 6
        assert explorer._spectrum_model_line.get_xdata().size

        explorer.preset.value = _OMPD_PRESET

        assert explorer.data.attrs["signal_domain"] == "temporal"
        assert explorer.fit_results is None
        assert not np.allclose(explorer.data.values, initial_fid)
        assert explorer.figure is figure
        assert explorer._spectrum_data_line is spectrum_line
        assert explorer.fit_params.metabolite.values.tolist() == [
            "zompd_c1",
            "zompd_c5_ph_6.4",
            "zompd_c5_ph_6.9",
            "zompd_c5_ph_7.4",
        ]
        np.testing.assert_allclose(
            [row.amplitude.value for row in explorer._component_widgets],
            [1.0, 0.07, 0.40, 0.50],
        )
        np.testing.assert_allclose(
            [row.t2_ms.value for row in explorer._component_widgets],
            [35.0, 28.0, 22.0, 21.5],
        )
        np.testing.assert_allclose(
            [row.frequency_hz.value for row in explorer._component_widgets],
            [-197.0, 263.0, 340.0, 367.0],
        )
        np.testing.assert_allclose(
            [row.fit_frequency_hz.value for row in explorer._component_widgets],
            [-200.0, 260.0, 340.0, 370.0],
        )
        np.testing.assert_allclose(
            [row.fit_t2_ms.value for row in explorer._component_widgets],
            [30.0, 25.0, 25.0, 25.0],
        )
        assert explorer.noise_std.value == 0.04
        assert explorer.frequency_range_hz.value == 40.0
        assert explorer.t2_range_ms.value == 25.0
        assert explorer.grid_points.value == 9
        assert explorer.iterations.value == 2
        assert explorer.repetitions.value == 3
        np.testing.assert_allclose(
            explorer.data.sampling_time_axis_s.values,
            np.arange(explorer.n_samples.value) * explorer.dwell_time_us.value * 1e-6,
        )
        assert explorer.fit_params.attrs["store_optimization_history"] is True

        explorer._run_fit()

        assert explorer.fit_results.fit_status.item() == "success"
        np.testing.assert_allclose(
            explorer.fit_results.fit_freqs.values,
            [-197.0, 263.0, 340.0, 367.0],
            atol=5.0,
        )
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in explorer._fit_frequency_lines],
            explorer.fit_results.fit_freqs.values,
        )
        assert explorer._spectrum_model_line.get_xdata().size
    finally:
        explorer.close()
