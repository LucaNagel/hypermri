#!/usr/bin/env python3
"""Explore simulated 13C FIDs and pseudo-inverse spectroscopy fitting.

This module is intended for a Jupyter notebook with the ``ipympl`` backend,
started from the repository root.  It is not part of the installed package::

    %matplotlib widget
    from scripts.interactive_spectroscopy_fitting import (
        launch_spectroscopy_fitting_explorer,
    )

    explorer = launch_spectroscopy_fitting_explorer()

The component rows control the simulated FID.  The second set of frequency and
T2* values supplies the initial values passed to ``def_fit_params``.  Press
``Fit aktualisieren`` after changing fit settings.  The optimisation controls
then expose individual frequency- and T2*-grid candidates recorded by the
production fitter.  The figure and its artists are created once; every widget
change only updates their data.

The renal Z-OMPD preset is a calibration-model proxy with the C1 reference and
C5 pH states 6.4, 6.9 and 7.4 at 7 T.  It is not a measured renal spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from matplotlib.lines import Line2D

from hypermri.analysis.fitting.models.spectroscopy.diagnostics import (
    HISTORY_EVENTS,
)
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    fid_basefunc,
)
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm
from hypermri.analysis.spectroscopy.signals import generate_fid
from hypermri.utils.physics import get_gmr

_PYRUVATE_PRESET = "Pyruvat / Lactat / Alanin / Bicarbonat (7 T)"
_OMPD_PRESET = "Niere: Z-OMPD C5 (pH-Proxy, 7 T)"
_EVENT_NAMES = {value: key for key, value in HISTORY_EVENTS.items()}
_EVENT_LABELS = {
    "Alle Schritte": None,
    "Frequenz-Sweep": "frequency_candidate",
    "T2*-Sweep": "t2_candidate",
    "Zustandswechsel": "state_change",
}
_STATE_EVENTS = frozenset(
    {"initial", "frequency_step", "t2_step", "verified", "returned"}
)
_COMPONENT_COLORS = ("C0", "C1", "C2", "C3")


@dataclass(frozen=True)
class _PresetComponent:
    """One simulated component and its initial fit state."""

    name: str
    amplitude: float
    t2_ms: float
    frequency_hz: float
    fit_frequency_hz: float
    fit_t2_ms: float


@dataclass
class _ComponentWidgets:
    """Keep the widgets belonging to one component together."""

    enabled: widgets.Checkbox
    name: widgets.Text
    amplitude: widgets.FloatSlider
    t2_ms: widgets.FloatSlider
    frequency_hz: widgets.FloatSlider
    fit_frequency_hz: widgets.FloatSlider
    fit_t2_ms: widgets.FloatSlider

    def row(self) -> widgets.HBox:
        """Return the component controls in their display order.

        Returns
        -------
        ipywidgets.HBox
            One horizontal component-control row.
        """
        return widgets.HBox(
            [
                self.enabled,
                self.name,
                self.amplitude,
                self.t2_ms,
                self.frequency_hz,
                self.fit_frequency_hz,
                self.fit_t2_ms,
            ]
        )


def _pyruvate_preset() -> tuple[_PresetComponent, ...]:
    """Return a four-peak 7 T hyperpolarized-pyruvate teaching preset."""
    larmor_mhz = 7.0 * get_gmr(nucleus="13c", unit="MHz_T")
    pyruvate_ppm = get_metab_cs_ppm("pyruvate")
    names = ("pyruvate", "lactate", "alanine", "bicarbonate")
    frequencies = [
        larmor_mhz * (get_metab_cs_ppm(name) - pyruvate_ppm) for name in names
    ]
    return tuple(
        _PresetComponent(
            name=name,
            amplitude=amplitude,
            t2_ms=t2_ms,
            frequency_hz=frequency,
            fit_frequency_hz=frequency + 15.0,
            fit_t2_ms=25.0,
        )
        for name, amplitude, t2_ms, frequency in zip(
            names,
            (1.0, 0.35, 0.15, 0.08),
            (35.0, 25.0, 22.0, 15.0),
            frequencies,
            strict=True,
        )
    )


def _ompd_preset() -> tuple[_PresetComponent, ...]:
    """Return the configured renal Z-OMPD C1/C5 teaching proxy at 7 T.

    The frequencies, amplitudes and T2* defaults reproduce the teaching
    configuration exposed by the explorer.  They are intentionally explicit:
    the preset is an illustrative relative-frequency model rather than a
    measured spectrum or an on-the-fly pH calibration.
    """
    return (
        _PresetComponent("zompd_c1", 1.0, 35.0, -197.0, -200.0, 30.0),
        _PresetComponent("zompd_c5_ph_6.4", 0.07, 28.0, 263.0, 260.0, 25.0),
        _PresetComponent("zompd_c5_ph_6.9", 0.40, 22.0, 340.0, 340.0, 25.0),
        _PresetComponent("zompd_c5_ph_7.4", 0.50, 21.5, 367.0, 370.0, 25.0),
    )


class SpectroscopyFittingExplorer:
    """Interactive simulator and step-by-step viewer for one 13C FID fit.

    The explorer is a notebook object.  It generates one temporal
    ``xarray.DataArray`` with :func:`generate_fid`, translates the editable fit
    settings through :func:`def_fit_params`, and calls
    :func:`fit_data_pseudo_inv` only when the user requests a fit.  Set a
    notebook's backend to ``ipympl`` (``%matplotlib widget``) before displaying
    it so changes are visible in the existing figure.

    Attributes
    ----------
    data : xarray.DataArray or None
        Current simulated complex temporal FID.  It is ``None`` only after an
        invalid widget configuration.
    fit_params : xarray.Dataset or None
        Last parameter dataset successfully created by ``def_fit_params``.
    fit_results : xarray.Dataset or None
        Results from the last requested fit, including optimisation history.
    figure : matplotlib.figure.Figure
        The single reusable Matplotlib figure updated by all controls.
    controls : ipywidgets.VBox
        Widget container to display in a notebook.
    """

    def __init__(self) -> None:
        """Create the controls, one reusable figure, and the initial fit."""
        self.data: xr.DataArray | None = None
        self.fit_params: xr.Dataset | None = None
        self.fit_results: xr.Dataset | None = None
        self._history_indices = np.array([], dtype=int)
        self._callback_suspended = False
        self._component_widgets = self._make_component_widgets()
        self._make_global_widgets()
        # Suppress pyplot's automatic notebook output: the launcher displays
        # this canvas explicitly, so automatic output would duplicate it.
        with plt.ioff():
            self.figure, self._axes = plt.subplots(
                2, 2, figsize=(13, 8), layout="tight"
            )
        self._create_plot_artists()
        self.controls = self._build_controls()
        self._connect_callbacks()
        self._apply_preset(self.preset.value)
        self._run_fit()

    def close(self) -> None:
        """Close the reusable figure when the notebook explorer is discarded.

        Returns
        -------
        None
            The figure is closed in place.
        """
        plt.close(self.figure)

    def _make_component_widgets(self) -> list[_ComponentWidgets]:
        """Create the fixed four editable component rows."""
        component_widgets = []
        for index in range(4):
            component_widgets.append(
                _ComponentWidgets(
                    enabled=widgets.Checkbox(
                        value=True, indent=False, layout=widgets.Layout(width="42px")
                    ),
                    name=widgets.Text(
                        value=f"peak_{index + 1}", layout=widgets.Layout(width="145px")
                    ),
                    amplitude=widgets.FloatSlider(
                        value=1.0,
                        min=0.0,
                        max=2.0,
                        step=0.01,
                        readout_format=".2f",
                        layout=widgets.Layout(width="175px"),
                    ),
                    t2_ms=widgets.FloatSlider(
                        value=25.0,
                        min=2.0,
                        max=100.0,
                        step=0.5,
                        readout_format=".1f",
                        layout=widgets.Layout(width="175px"),
                    ),
                    frequency_hz=widgets.FloatSlider(
                        value=0.0,
                        min=-1_600.0,
                        max=1_600.0,
                        step=1.0,
                        readout_format=".0f",
                        layout=widgets.Layout(width="190px"),
                    ),
                    fit_frequency_hz=widgets.FloatSlider(
                        value=0.0,
                        min=-1_600.0,
                        max=1_600.0,
                        step=1.0,
                        readout_format=".0f",
                        layout=widgets.Layout(width="190px"),
                    ),
                    fit_t2_ms=widgets.FloatSlider(
                        value=25.0,
                        min=2.0,
                        max=100.0,
                        step=0.5,
                        readout_format=".1f",
                        layout=widgets.Layout(width="175px"),
                    ),
                )
            )
        return component_widgets

    def _make_global_widgets(self) -> None:
        """Create simulation, fitting and history-navigation controls."""
        self.preset = widgets.Dropdown(
            options=(_PYRUVATE_PRESET, _OMPD_PRESET),
            value=_PYRUVATE_PRESET,
            description="Preset:",
            layout=widgets.Layout(width="430px"),
        )
        self.noise_std = widgets.FloatSlider(
            value=0.02,
            min=0.0,
            max=0.20,
            step=0.005,
            readout_format=".3f",
            description="Noise σ:",
        )
        self.noise_seed = widgets.BoundedIntText(
            value=42, min=0, max=2**31 - 1, description="Noise seed:"
        )
        self.dwell_time_us = widgets.BoundedFloatText(
            value=400.0, min=100.0, max=2_000.0, step=25.0, description="Dwell [µs]:"
        )
        self.n_samples = widgets.Dropdown(
            options=(128, 256, 512), value=256, description="Samples:"
        )
        self.frequency_range_hz = widgets.BoundedFloatText(
            value=100.0,
            min=0.0,
            max=400.0,
            step=5.0,
            description="f-Suche [Hz]:",
        )
        self.t2_range_ms = widgets.BoundedFloatText(
            value=30.0,
            min=0.0,
            max=100.0,
            step=1.0,
            description="T2*-Suche [ms]:",
        )
        self.min_t2_ms = widgets.BoundedFloatText(
            value=5.0, min=1.0, max=100.0, step=1.0, description="min. T2* [ms]:"
        )
        self.max_t2_ms = widgets.BoundedFloatText(
            value=80.0, min=2.0, max=200.0, step=1.0, description="max. T2* [ms]:"
        )
        self.grid_points = widgets.SelectionSlider(
            options=(5, 7, 9, 11, 13), value=9, description="Gitterpunkte:"
        )
        self.iterations = widgets.IntSlider(
            value=2, min=1, max=4, step=1, description="Zoom-Iterationen:"
        )
        self.repetitions = widgets.IntSlider(
            value=3, min=1, max=8, step=1, description="f/T2*-Pässe:"
        )
        self.run_button = widgets.Button(
            description="Fit aktualisieren", button_style="primary", icon="play"
        )
        self.event_filter = widgets.ToggleButtons(
            options=tuple(_EVENT_LABELS),
            value="Alle Schritte",
            description="Parameter:",
        )
        self.history_metabolite = widgets.Dropdown(
            options=(("Alle Metaboliten", None),),
            value=None,
            description="Metabolit:",
        )
        self.step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="Schritt:",
            continuous_update=False,
        )
        self.status = widgets.HTML()

    def _build_controls(self) -> widgets.VBox:
        """Compose the control panel without creating a second figure."""
        header = widgets.HBox(
            [
                widgets.Label("Aktiv", layout=widgets.Layout(width="42px")),
                widgets.Label("Name", layout=widgets.Layout(width="145px")),
                widgets.Label("Amplitude [a.u.]", layout=widgets.Layout(width="175px")),
                widgets.Label("Signal T2* [ms]", layout=widgets.Layout(width="175px")),
                widgets.Label("Signal f [Hz]", layout=widgets.Layout(width="190px")),
                widgets.Label("Start f [Hz]", layout=widgets.Layout(width="190px")),
                widgets.Label("Start T2* [ms]", layout=widgets.Layout(width="175px")),
            ]
        )
        return widgets.VBox(
            [
                self.preset,
                widgets.HTML("<b>Simulierte Komponenten</b>"),
                header,
                *(component.row() for component in self._component_widgets),
                widgets.HBox(
                    [
                        self.noise_std,
                        self.noise_seed,
                        self.dwell_time_us,
                        self.n_samples,
                    ]
                ),
                widgets.HTML("<b>Parameter für def_fit_params</b>"),
                widgets.HBox(
                    [
                        self.frequency_range_hz,
                        self.t2_range_ms,
                        self.min_t2_ms,
                        self.max_t2_ms,
                    ]
                ),
                widgets.HBox(
                    [
                        self.grid_points,
                        self.iterations,
                        self.repetitions,
                        self.run_button,
                    ]
                ),
                self.status,
                widgets.HTML("<b>Gespeicherte Optimierungshistorie</b>"),
                widgets.HBox(
                    [self.event_filter, self.history_metabolite, self.step_slider]
                ),
            ]
        )

    def _create_plot_artists(self) -> None:
        """Create all artists once; callbacks subsequently only use set_data."""
        spectrum_ax, fid_ax, objective_ax, parameters_ax = self._axes.ravel()
        (self._spectrum_data_line,) = spectrum_ax.plot(
            [], [], color="black", label="Messung |FFT(FID)|"
        )
        (self._spectrum_model_line,) = spectrum_ax.plot(
            [], [], color="C3", label="Modell"
        )
        (self._fid_data_line,) = fid_ax.plot([], [], color="black", label="Messung Re")
        (self._fid_model_line,) = fid_ax.plot([], [], color="C3", label="Modell Re")
        (self._fid_residual_line,) = fid_ax.plot(
            [], [], color="0.5", label="Residuum Re"
        )
        (self._objective_line,) = objective_ax.plot([], [], "o-", color="C0", ms=4)
        (self._objective_marker,) = objective_ax.plot([], [], "o", color="C3", ms=8)
        self._true_frequency_lines = [
            spectrum_ax.axvline(color=color, ls="--", lw=0.65, alpha=0.55)
            for color in _COMPONENT_COLORS
        ]
        self._fit_frequency_lines = [
            spectrum_ax.axvline(color=color, ls="-", lw=0.85, alpha=0.8)
            for color in _COMPONENT_COLORS
        ]
        self._parameter_text = parameters_ax.text(
            0.02,
            0.98,
            "",
            va="top",
            family="monospace",
            transform=parameters_ax.transAxes,
        )
        spectrum_ax.set(
            xlabel="Frequenz [Hz]", ylabel="Magnitude [a.u.]", title="Spektrum"
        )
        fid_ax.set(xlabel="Zeit [ms]", ylabel="Realteil [a.u.]", title="FID und Modell")
        objective_ax.set(
            xlabel="Kandidat", ylabel="Zielfunktion [a.u.²]", title="Optimierung"
        )
        parameters_ax.set_axis_off()
        spectrum_ax.legend(
            handles=[
                self._spectrum_data_line,
                self._spectrum_model_line,
                Line2D(
                    [],
                    [],
                    color="0.25",
                    ls="--",
                    lw=0.65,
                    label="Simulierte Peakposition",
                ),
                Line2D(
                    [],
                    [],
                    color="0.25",
                    lw=0.85,
                    label="Aktuelle Fitposition",
                ),
            ],
            loc="upper left",
        )
        fid_ax.legend(loc="upper right")

    def _connect_callbacks(self) -> None:
        """Connect controls while keeping fitting explicitly button-triggered."""
        self.preset.observe(self._on_preset_change, names="value")
        self.run_button.on_click(self._on_run_fit)
        self.event_filter.observe(self._on_history_filter_change, names="value")
        self.history_metabolite.observe(self._on_history_filter_change, names="value")
        self.step_slider.observe(self._on_history_step_change, names="value")
        for widget in (
            self.noise_std,
            self.noise_seed,
            self.dwell_time_us,
            self.n_samples,
            self.frequency_range_hz,
            self.t2_range_ms,
            self.min_t2_ms,
            self.max_t2_ms,
            self.grid_points,
            self.iterations,
            self.repetitions,
        ):
            widget.observe(self._on_configuration_change, names="value")
        for component in self._component_widgets:
            for widget in (
                component.enabled,
                component.name,
                component.amplitude,
                component.t2_ms,
                component.frequency_hz,
                component.fit_frequency_hz,
                component.fit_t2_ms,
            ):
                widget.observe(self._on_configuration_change, names="value")

    def _on_preset_change(self, change: dict[str, Any]) -> None:
        """Load a preset's signal and fit starts after a user selection."""
        if not self._callback_suspended:
            self._apply_preset(change["new"])
            self._refresh_input_and_params()

    def _on_configuration_change(self, _change: dict[str, Any]) -> None:
        """Regenerate only data and parameter maps, leaving fitting explicit."""
        if not self._callback_suspended:
            self._refresh_input_and_params()

    def _on_run_fit(self, _button: widgets.Button) -> None:
        """Fit the current FID and make its recorded history navigable."""
        self._run_fit()

    def _on_history_filter_change(self, _change: dict[str, Any]) -> None:
        """Reset the displayed history index after changing either filter."""
        if not self._callback_suspended:
            self._set_history_indices(reset_step=True)
            self._update_plot()

    def _on_history_step_change(self, _change: dict[str, Any]) -> None:
        """Display the selected recorded candidate or accepted state."""
        if not self._callback_suspended:
            self._update_plot()

    def _apply_preset(self, preset_name: str) -> None:
        """Copy one fixed preset into the editable component widgets."""
        if preset_name == _PYRUVATE_PRESET:
            components = _pyruvate_preset()
        elif preset_name == _OMPD_PRESET:
            components = _ompd_preset()
        else:  # Defensive guard for programmatic widget changes.
            raise ValueError(f"Unknown spectroscopy preset: {preset_name!r}")

        self._callback_suspended = True
        try:
            for widgets_row, component in zip(
                self._component_widgets, components, strict=True
            ):
                widgets_row.enabled.value = True
                widgets_row.name.value = component.name
                widgets_row.amplitude.value = component.amplitude
                widgets_row.t2_ms.value = component.t2_ms
                widgets_row.frequency_hz.value = component.frequency_hz
                widgets_row.fit_frequency_hz.value = component.fit_frequency_hz
                widgets_row.fit_t2_ms.value = component.fit_t2_ms
            if preset_name == _OMPD_PRESET:
                self.noise_std.value = 0.04
                self.noise_seed.value = 42
                self.dwell_time_us.value = 400.0
                self.n_samples.value = 256
                self.frequency_range_hz.value = 40.0
                self.t2_range_ms.value = 25.0
            else:
                self.noise_std.value = 0.02
                self.noise_seed.value = 42
                self.dwell_time_us.value = 400.0
                self.n_samples.value = 256
                self.frequency_range_hz.value = 100.0
                self.t2_range_ms.value = 30.0
            self.min_t2_ms.value = 5.0
            self.max_t2_ms.value = 80.0
            self.grid_points.value = 9
            self.iterations.value = 2
            self.repetitions.value = 3
        finally:
            self._callback_suspended = False

    def _active_components(self) -> list[_ComponentWidgets]:
        """Return active rows and validate names before creating data."""
        active = [row for row in self._component_widgets if row.enabled.value]
        if not active:
            raise ValueError("Mindestens eine Komponente muss aktiv sein.")
        names = [row.name.value.strip() for row in active]
        if any(not name for name in names):
            raise ValueError("Jede aktive Komponente benötigt einen Namen.")
        if len(set(names)) != len(names):
            raise ValueError("Namen aktiver Komponenten müssen eindeutig sein.")
        return active

    def _generate_input_data(self) -> xr.DataArray:
        """Generate one labelled noisy FID with the canonical ``generate_fid``.

        Noise is added once after summing the component FIDs, so ``noise_std``
        is the standard deviation of each real and imaginary quadrature of the
        displayed total FID rather than a per-component value.
        """
        active = self._active_components()
        n_samples = int(self.n_samples.value)
        dwell_time_s = float(self.dwell_time_us.value) * 1e-6
        time_axis_s = np.arange(n_samples, dtype=float) * dwell_time_s
        clean_fid = generate_fid(
            time_axis_s=time_axis_s,
            amplitude=[row.amplitude.value for row in active],
            t2_s=[row.t2_ms.value * 1e-3 for row in active],
            freq_hz=[row.frequency_hz.value for row in active],
            noise_std=0.0,
            sum_fids=True,
        )
        rng = np.random.default_rng(int(self.noise_seed.value))
        noise_std = float(self.noise_std.value)
        noise = noise_std * (
            rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)
        )
        return xr.DataArray(
            np.asarray(clean_fid + noise, dtype=np.complex128),
            dims=("acquisition_idx",),
            coords={
                "acquisition_idx": np.arange(n_samples),
                "sampling_time_axis_s": ("acquisition_idx", time_axis_s),
            },
            attrs={"signal_domain": "temporal", "unit": "a.u."},
            name="simulated_fid",
        )

    def _build_fit_params(self) -> xr.Dataset:
        """Prepare the current editable fit settings through ``def_fit_params``."""
        if self.data is None:
            raise ValueError("Kein simuliertes FID verfügbar.")
        active = self._active_components()
        nyquist_hz = 0.5 / (float(self.dwell_time_us.value) * 1e-6)
        initial_frequencies_hz = np.array(
            [row.fit_frequency_hz.value for row in active], dtype=float
        )
        search_width_hz = float(self.frequency_range_hz.value)
        if np.any(np.abs(initial_frequencies_hz) + search_width_hz / 2 >= nyquist_hz):
            raise ValueError(
                "Die Frequenzsuche überschreitet Nyquist. Dwell-Zeit verkleinern "
                "oder Startfrequenzen/Fenster anpassen."
            )
        return def_fit_params(
            data=self.data,
            fit_params={
                "metabs": [row.name.value.strip() for row in active],
                "metabs_freqs_hz": initial_frequencies_hz,
                "initial_t2s_s": np.array(
                    [row.fit_t2_ms.value for row in active], dtype=float
                )
                * 1e-3,
                "metabs_freq_ranges": search_width_hz,
                "metabs_t2_ranges": float(self.t2_range_ms.value) * 1e-3,
                "min_t2_s": float(self.min_t2_ms.value) * 1e-3,
                "max_t2_s": float(self.max_t2_ms.value) * 1e-3,
                "npoints": int(self.grid_points.value),
                "niter": int(self.iterations.value),
                "rep_fitting": int(self.repetitions.value),
                "amp_solver": "pinv",
                "min_peak_separation_hz": 5.0,
                "store_optimization_history": True,
                "show_tqdm": False,
                "signal_domain": "temporal",
            },
        )

    def _refresh_input_and_params(self) -> None:
        """Regenerate FID and maps without doing an expensive fit."""
        self.fit_results = None
        self._history_indices = np.array([], dtype=int)
        try:
            self.data = self._generate_input_data()
            self.fit_params = self._build_fit_params()
        except (TypeError, ValueError) as exc:
            self.data = None
            self.fit_params = None
            self._show_error(str(exc))
        else:
            self.status.value = (
                "<span style='color:#7f6000'>Parameter vorbereitet — "
                "für eine neue Optimierung <b>Fit aktualisieren</b> drücken.</span>"
            )
        self._set_history_indices(reset_step=True)
        self._update_plot()

    def _run_fit(self) -> None:
        """Fit a freshly generated FID and expose its optimisation history.

        Rebuilding the FID and fit-parameter maps here makes the button an
        explicit synchronization point.  Thus a click always fits precisely
        the simulation and start values currently visible in the widgets,
        including updates which arrived immediately before the click.
        """
        self._refresh_input_and_params()
        if self.data is None or self.fit_params is None:
            self._show_error(
                "Ungültige Simulation oder Fitparameter; Fit nicht gestartet."
            )
            return
        self.run_button.disabled = True
        self.status.value = "<span style='color:#1f4e79'>Fitting läuft …</span>"
        try:
            self.fit_results = fit_data_pseudo_inv(
                data=self.data,
                fit_params=self.fit_params,
                n_jobs=1,
                show_tqdm=False,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self.fit_results = None
            self._show_error(f"Fit fehlgeschlagen: {exc}")
        else:
            status = str(self.fit_results.fit_status.item())
            objective = float(self.fit_results.fit_objective.item())
            self.status.value = (
                "<span style='color:#38761d'>"
                f"Fitstatus: <b>{escape(status)}</b>; Zielfunktion: {objective:.4g}."
                "</span>"
            )
        finally:
            self.run_button.disabled = False
        # A newly completed fit always opens on its returned state.  Keeping a
        # previous candidate-only filter here would otherwise make the display
        # look stale even though the numerical fit has completed.
        self._callback_suspended = True
        try:
            self.event_filter.value = "Alle Schritte"
            self.history_metabolite.value = None
        finally:
            self._callback_suspended = False
        self._set_history_indices(reset_step=True, select_final_state=True)
        self._update_plot()

    def _show_error(self, message: str) -> None:
        """Show a widget error without replacing the control panel."""
        self.status.value = (
            f"<span style='color:#990000'><b>Fehler:</b> {escape(message)}</span>"
        )

    def _set_history_indices(
        self, *, reset_step: bool, select_final_state: bool = False
    ) -> None:
        """Map the visible slider positions to stored history rows.

        Parameters
        ----------
        reset_step : bool
            Whether to reset the slider to the first visible row.
        select_final_state : bool, default=False
            Select the returned fit state instead of the first history row.
            This is used immediately after fitting so the plot shows the
            completed fit, not its deliberately offset starting state.
        """
        all_indices = np.array([], dtype=int)
        if self.fit_results is not None and "fit_history_count" in self.fit_results:
            count = int(self.fit_results.fit_history_count.item())
            if count:
                info = self.fit_results.fit_history_info.isel(
                    history_step=slice(count)
                ).values
                events = np.array(
                    [_EVENT_NAMES.get(code, "unknown") for code in info[:, 0]]
                )
                selected = _EVENT_LABELS[self.event_filter.value]
                if selected is None:
                    visible = events != "unknown"
                elif selected == "state_change":
                    visible = np.isin(events, tuple(_STATE_EVENTS))
                else:
                    visible = events == selected
                selected_metabolite = self.history_metabolite.value
                if selected_metabolite is not None:
                    visible &= (info[:, 4] == selected_metabolite) | (info[:, 4] == -1)
                all_indices = np.flatnonzero(visible)
        self._history_indices = all_indices
        self._callback_suspended = True
        try:
            self._update_history_metabolite_options()
            self.step_slider.max = max(0, len(all_indices) - 1)
            if select_final_state and len(all_indices):
                self.step_slider.value = len(all_indices) - 1
            elif reset_step:
                self.step_slider.value = 0
            else:
                self.step_slider.value = min(
                    self.step_slider.value, self.step_slider.max
                )
            self.step_slider.disabled = not len(all_indices)
        finally:
            self._callback_suspended = False

    def _update_history_metabolite_options(self) -> None:
        """Synchronize the metabolite filter with the currently fitted data."""
        options: tuple[tuple[str, int | None], ...] = (("Alle Metaboliten", None),)
        if self.fit_results is not None:
            options += tuple(
                (str(name), index)
                for index, name in enumerate(self.fit_results.metabolite.values)
            )
        previous = self.history_metabolite.value
        self.history_metabolite.options = options
        valid_values = {value for _, value in options}
        self.history_metabolite.value = previous if previous in valid_values else None

    def _current_history_row(self) -> int | None:
        """Return the selected absolute history row, if one is available."""
        if not len(self._history_indices):
            return None
        return int(self._history_indices[self.step_slider.value])

    def _display_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
        """Return parameters for the selected history state or raw simulation."""
        active = self._active_components()
        row = self._current_history_row()
        if row is not None and self.fit_results is not None:
            names = self.fit_results.metabolite.values.tolist()
            event = _EVENT_NAMES.get(
                int(self.fit_results.fit_history_info.isel(history_step=row).values[0]),
                "unknown",
            )
            if event == "returned":
                # The returned dataset is the fitter's canonical final state.
                # Do not let a history-recording detail move the final model or
                # its frequency markers away from the actual fit result.
                return (
                    self.fit_results.fit_amps.values,
                    self.fit_results.fit_freqs.values,
                    self.fit_results.fit_t2s.values,
                    names,
                    f"Fit-Endergebnis: {self._history_label(row)}",
                )
            return (
                self.fit_results.fit_history_amps.isel(history_step=row).values,
                self.fit_results.fit_history_freqs.isel(history_step=row).values,
                self.fit_results.fit_history_t2s.isel(history_step=row).values,
                names,
                self._history_label(row),
            )
        return (
            np.asarray(
                [component.amplitude.value for component in active], dtype=complex
            ),
            np.asarray([component.frequency_hz.value for component in active]),
            np.asarray([component.t2_ms.value for component in active]) * 1e-3,
            [component.name.value.strip() for component in active],
            "Simuliertes Signal (noch nicht gefittet)",
        )

    def _history_label(self, row: int) -> str:
        """Format metadata for one stored optimisation event."""
        if self.fit_results is None:
            return ""
        info = self.fit_results.fit_history_info.isel(history_step=row).values
        event = _EVENT_NAMES.get(int(info[0]), "unknown")
        accepted = bool(info[6])
        return (
            f"{event}; Pass {int(info[2])}; Iteration {int(info[3])}; "
            f"Metabolit {int(info[4])}; Kandidat {int(info[5])}; akzeptiert={accepted}"
        )

    def _update_plot(self) -> None:
        """Update data of the existing artists for the current configuration."""
        if self.data is None:
            self._clear_plot("Keine gültige Simulation")
            return
        try:
            amplitudes, frequencies, t2s_s, names, label = self._display_state()
        except ValueError as exc:
            self._clear_plot(str(exc))
            return
        time_s = self.data.sampling_time_axis_s.values
        measured_fid = self.data.values
        show_fit = (
            self.fit_results is not None and self._current_history_row() is not None
        )
        frequency_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(time_s.size, d=float(self.dwell_time_us.value) * 1e-6)
        )
        if show_fit and (
            np.all(np.isfinite(amplitudes))
            and np.all(np.isfinite(frequencies))
            and np.all(np.isfinite(t2s_s))
        ):
            model_fid = np.sum(
                amplitudes[:, np.newaxis] * fid_basefunc(frequencies, t2s_s, time_s),
                axis=0,
            )
        else:
            model_fid = None
        measured_spectrum = np.abs(np.fft.fftshift(np.fft.fft(measured_fid)))
        model_spectrum = (
            np.abs(np.fft.fftshift(np.fft.fft(model_fid)))
            if model_fid is not None
            else None
        )
        self._spectrum_data_line.set_data(frequency_axis_hz, measured_spectrum)
        self._spectrum_model_line.set_data(
            frequency_axis_hz if model_spectrum is not None else [],
            model_spectrum if model_spectrum is not None else [],
        )
        self._fid_data_line.set_data(time_s * 1e3, measured_fid.real)
        self._fid_model_line.set_data(
            time_s * 1e3 if model_fid is not None else [],
            model_fid.real if model_fid is not None else [],
        )
        self._fid_residual_line.set_data(
            time_s * 1e3 if model_fid is not None else [],
            (measured_fid - model_fid).real if model_fid is not None else [],
        )
        self._update_frequency_lines(frequencies, show_fit=show_fit)
        self._update_objective_plot()
        self._update_parameter_text(names, amplitudes, frequencies, t2s_s, label)
        self._rescale_axes(
            frequency_axis_hz,
            measured_spectrum,
            model_spectrum,
            time_s,
            measured_fid,
            model_fid,
        )
        self.figure.canvas.draw_idle()

    def _update_frequency_lines(
        self, frequencies: np.ndarray, *, show_fit: bool
    ) -> None:
        """Move component truth and current-fit frequency markers in place."""
        active = self._active_components()
        for index, (true_line, fit_line) in enumerate(
            zip(self._true_frequency_lines, self._fit_frequency_lines, strict=True)
        ):
            visible = index < len(active)
            true_line.set_visible(visible)
            fit_line.set_visible(show_fit and visible)
            if visible:
                true_frequency = active[index].frequency_hz.value
                true_line.set_xdata((true_frequency, true_frequency))
                if show_fit:
                    fit_line.set_xdata((frequencies[index], frequencies[index]))

    def _update_objective_plot(self) -> None:
        """Show every candidate of the selected frequency or T2* sweep."""
        row = self._current_history_row()
        objective_ax = self._axes.ravel()[2]
        if row is None or self.fit_results is None:
            self._objective_line.set_data([], [])
            self._objective_marker.set_data([], [])
            objective_ax.set(
                title="Optimierung (nach dem Fit verfügbar)", xlabel="Kandidat"
            )
            return
        info = self.fit_results.fit_history_info.values
        event = _EVENT_NAMES.get(int(info[row, 0]), "unknown")
        if event not in {"frequency_candidate", "t2_candidate"}:
            self._objective_line.set_data([], [])
            self._objective_marker.set_data([], [])
            objective_ax.set(
                title="Optimierung: kein Kandidaten-Sweep gewählt", xlabel="Kandidat"
            )
            return
        metabolite_index, repetition, iteration = (
            int(info[row, index]) for index in (4, 2, 3)
        )
        matching = np.flatnonzero(
            (info[:, 0] == HISTORY_EVENTS[event])
            & (info[:, 4] == metabolite_index)
            & (info[:, 2] == repetition)
            & (info[:, 3] == iteration)
        )
        candidate_numbers = info[matching, 5]
        order = np.argsort(candidate_numbers)
        matching = matching[order]
        candidate_numbers = candidate_numbers[order]
        objective = self.fit_results.fit_history_objective.values[matching]
        finite = np.isfinite(objective)
        self._objective_line.set_data(candidate_numbers[finite], objective[finite])
        current_objective = float(
            self.fit_results.fit_history_objective.isel(history_step=row)
        )
        current_candidate = int(info[row, 5])
        if np.isfinite(current_objective):
            self._objective_marker.set_data([current_candidate], [current_objective])
        else:
            self._objective_marker.set_data([], [])
        unit = "Hz" if event == "frequency_candidate" else "s"
        objective_ax.set(
            title=f"{event}: {self.fit_results.metabolite.values[metabolite_index]}",
            xlabel=f"Kandidatindex ({unit})",
            ylabel="Zielfunktion [a.u.²]",
        )

    def _update_parameter_text(
        self,
        names: list[str],
        amplitudes: np.ndarray,
        frequencies: np.ndarray,
        t2s_s: np.ndarray,
        label: str,
    ) -> None:
        """Replace only the text content of the parameter-summary axis."""
        lines = [label, "", "Komponente              |A|      f [Hz]   T2* [ms]"]
        for name, amplitude, frequency, t2_s in zip(
            names, amplitudes, frequencies, t2s_s, strict=True
        ):
            lines.append(
                f"{name:20.20s} {abs(amplitude):7.3f} {frequency:10.2f} {t2_s * 1e3:9.2f}"
            )
        self._parameter_text.set_text("\n".join(lines))

    def _rescale_axes(
        self,
        frequency_axis_hz: np.ndarray,
        measured_spectrum: np.ndarray,
        model_spectrum: np.ndarray | None,
        time_s: np.ndarray,
        measured_fid: np.ndarray,
        model_fid: np.ndarray | None,
    ) -> None:
        """Set limits from finite values while retaining the same axes."""
        spectrum_ax, fid_ax, objective_ax, _ = self._axes.ravel()
        spectrum_ax.set_xlim(float(frequency_axis_hz[0]), float(frequency_axis_hz[-1]))
        spectrum_values = (
            np.concatenate((measured_spectrum, model_spectrum))
            if model_spectrum is not None
            else measured_spectrum
        )
        self._set_y_limits(spectrum_ax, spectrum_values, lower_zero=True)
        fid_values = (
            np.concatenate(
                (measured_fid.real, model_fid.real, (measured_fid - model_fid).real)
            )
            if model_fid is not None
            else measured_fid.real
        )
        self._set_y_limits(fid_ax, fid_values)
        objective_values = self._objective_line.get_ydata()
        if len(objective_values):
            self._set_y_limits(
                objective_ax, np.asarray(objective_values), lower_zero=True
            )
        objective_ax.relim()
        objective_ax.autoscale_view(scalex=True, scaley=False)
        fid_ax.set_xlim(float(time_s[0] * 1e3), float(time_s[-1] * 1e3))

    @staticmethod
    def _set_y_limits(
        axis: Any, values: np.ndarray, *, lower_zero: bool = False
    ) -> None:
        """Set readable axis limits from finite values, including flat signals."""
        finite = np.asarray(values)[np.isfinite(values)]
        if not finite.size:
            return
        lower, upper = float(np.min(finite)), float(np.max(finite))
        padding = (
            0.05 * (upper - lower) if upper > lower else max(0.05, abs(upper) * 0.1)
        )
        lower = max(0.0, lower - padding) if lower_zero else lower - padding
        axis.set_ylim(lower, upper + padding)

    def _clear_plot(self, reason: str) -> None:
        """Clear existing artist data when the widgets are invalid."""
        for line in (
            self._spectrum_data_line,
            self._spectrum_model_line,
            self._fid_data_line,
            self._fid_model_line,
            self._fid_residual_line,
            self._objective_line,
            self._objective_marker,
        ):
            line.set_data([], [])
        self._parameter_text.set_text(reason)
        self.figure.canvas.draw_idle()


def launch_spectroscopy_fitting_explorer() -> SpectroscopyFittingExplorer:
    """Create and display the interactive FID-fitting explorer in a notebook.

    Returns
    -------
    SpectroscopyFittingExplorer
        Explorer retaining the widget state, reusable figure, generated FID,
        effective fit parameters and latest fit result.  Keep this object in a
        notebook variable for further inspection, for example
        ``explorer.fit_results``.

    Notes
    -----
    Run ``%matplotlib widget`` first to use the ``ipympl`` backend.  With a
    static inline Matplotlib backend, widgets still update their state but the
    figure cannot update in place.
    """
    explorer = SpectroscopyFittingExplorer()
    display(explorer.controls)
    display(explorer.figure.canvas)
    return explorer


if __name__ == "__main__":  # pragma: no cover - a notebook is the UI host.
    raise SystemExit(
        "Dieses Skript ist für Jupyter gedacht. In einem Notebook zuerst "
        "'%matplotlib widget' ausführen und dann "
        "launch_spectroscopy_fitting_explorer() aufrufen."
    )
