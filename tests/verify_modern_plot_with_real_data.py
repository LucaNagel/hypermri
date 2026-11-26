"""Manual verification helper for plot fit overlays with real CSI data."""

from __future__ import annotations

import os
from pathlib import Path

import hypermri
from hypermri.utils.io.files import load_data
from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState


def resolve_real_csi_path() -> Path | None:
    """Resolve an explicitly supplied, locally authorised CSI study."""
    configured_path = os.environ.get("HYPERMRI_REAL_CSI_STUDY", "").strip()
    if not configured_path:
        print(
            "No real CSI study configured. Set HYPERMRI_REAL_CSI_STUDY to an "
            "approved local study path."
        )
        return None
    path = Path(configured_path).expanduser()
    if path.exists():
        return path.resolve()
    print(
        f"Real CSI dataset not found at {path}. "
        "HYPERMRI_REAL_CSI_STUDY must point to an approved local study."
    )
    return None


def verify_with_provided_data():
    study_path = resolve_real_csi_path()
    if study_path is None:
        return

    fit_results_path = study_path / "csi_fit_results.pkl"

    print(f"Loading real study from: {study_path}")
    try:
        scans = hypermri.Study(study_path, log_mode="error", load_data=True)
        csi = scans[22]
        data = csi.primary_data
    except Exception as e:
        print(f"Error loading study or CSI: {e}")
        return

    print(f"Loading fit results from: {fit_results_path}")
    try:
        fit_results = load_data(full_name=fit_results_path)
    except Exception as e:
        print(f"Error loading fit results: {e}")
        return

    model = PlotDataModel(csi)
    state = PlotState()

    state.spectral.show_fit = True
    state.spatial.slice_indices = {
        "read_idx": data.sizes["read_idx"] // 2,
        "phase_idx": data.sizes["phase_idx"] // 2,
    }

    print("\n--- Testing with real fit_results ---")
    payload = model.get_render_payload(data, state, overlays={"fit_res": fit_results})
    metabolites_found = []

    if payload.spectral_error:
        print(f"Spectral Error: {payload.spectral_error}")
    else:
        print(f"Available Overlays: {list(payload.overlays.keys())}")
        if "Fit" in payload.overlays:
            print(
                f"Successfully reconstructed 'Fit' overlay. Shape: "
                f"{payload.overlays['Fit'].shape}"
            )

        metabolites_found = [m for m in payload.overlays.keys() if m != "Fit"]
        print(f"Individual Metabolites in Overlays: {metabolites_found}")

    if metabolites_found:
        print(f"\nTesting selection of only: {metabolites_found[0]}")
        state.spectral.fit_metabolites = [metabolites_found[0]]
        payload_sel = model.get_render_payload(
            data, state, overlays={"fit_res": fit_results}
        )
        print(f"Selected Overlays: {list(payload_sel.overlays.keys())}")


if __name__ == "__main__":
    verify_with_provided_data()
