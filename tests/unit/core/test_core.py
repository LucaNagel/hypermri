from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


def _xyz_dataarray(values, *, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    values = np.asarray(values, dtype=np.float64)
    dims = ("read_idx", "phase_idx", "slice_idx")
    read = origin[0] + np.arange(values.shape[0]) * spacing[0]
    phase = origin[1] + np.arange(values.shape[1]) * spacing[1]
    slice_ = origin[2] + np.arange(values.shape[2]) * spacing[2]
    x, y, z = np.meshgrid(read, phase, slice_, indexing="ij")
    return xr.DataArray(
        values,
        dims=dims,
        coords={
            "read_idx": np.arange(values.shape[0]),
            "phase_idx": np.arange(values.shape[1]),
            "slice_idx": np.arange(values.shape[2]),
            "x": (dims, x),
            "y": (dims, y),
            "z": (dims, z),
        },
    )


def test_primary_data_prefers_data_over_scanner_data():
    from hypermri.mixins import StandardizedDataArrayMixin

    data_obj = StandardizedDataArrayMixin()
    scanner = xr.DataArray(np.ones((2, 2)), dims=("read_idx", "phase_idx"))
    reco = xr.DataArray(np.zeros((2, 2)), dims=("read_idx", "phase_idx"))

    data_obj.scanner_data = scanner
    data_obj.data = reco

    assert data_obj.primary_data is reco
    assert data_obj.primary_data is reco


def test_primary_data_falls_back_to_scanner_data():
    from hypermri.mixins import StandardizedDataArrayMixin

    data_obj = StandardizedDataArrayMixin()
    scanner = xr.DataArray(np.ones((2, 2)), dims=("read_idx", "phase_idx"))

    data_obj.scanner_data = scanner

    assert data_obj.primary_data is scanner
    assert data_obj.primary_data is scanner


def test_standardized_data_returns_none_without_loaded_data():
    from hypermri.mixins import StandardizedDataArrayMixin

    data_obj = StandardizedDataArrayMixin()

    assert data_obj.primary_data is None
    assert data_obj.primary_data is None


def test_dummy_core():
    assert True


def test_vendor_type_covers_registered_vendor_names():
    from hypermri.loading.contracts import VendorType
    from hypermri.vendors import get_vendor_registry

    registered_vendors = get_vendor_registry().list_vendors()

    assert {VendorType(vendor) for vendor in registered_vendors}


def test_unknown_vendor_metadata_is_not_reported_as_bruker():
    from hypermri.loading.contracts import VendorType
    from hypermri.loading.experiment import MRIExperiment

    class ResearchScannerHandler:
        vendor = "research_scanner"
        id = "scan-1"
        type = "custom-sequence"
        is_spectroscopic = False

        @staticmethod
        def get_parameters():
            return {"custom": True}

    experiment = MRIExperiment.__new__(MRIExperiment)
    experiment.handler = ResearchScannerHandler()
    experiment._forced_vendor = None

    metadata = experiment._create_metadata()

    assert metadata.vendor is VendorType.UNKNOWN
    assert metadata.custom_fields["vendor_name"] == "research_scanner"


def test_get_finalized_masks_returns_modern_mask_dataset_copy():
    from hypermri.loading.experiment import MRIExperiment

    experiment = MRIExperiment.__new__(MRIExperiment)
    masks = xr.Dataset(
        {
            "roi": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("read_idx", "phase_idx"),
            )
        }
    )

    experiment._initialize_masks(masks)
    returned = experiment.get_finalized_masks()
    assert returned.equals(masks)

    returned["roi"].values[0, 0] = False

    assert bool(experiment.masks["roi"].values[0, 0]) is True


def test_apply_mask_to_projects_mask_from_experiment_to_target_object():
    from hypermri.loading.experiment import MRIExperiment

    experiment = MRIExperiment.__new__(MRIExperiment)
    source = _xyz_dataarray(np.zeros((4, 4, 4)))
    mask = xr.zeros_like(source)
    mask.loc[dict(read_idx=[0])] = 1.0
    experiment._initialize_masks(xr.Dataset({"roi": mask}))
    experiment.data = source
    experiment.scanner_data = None
    experiment._data_loaded = True
    experiment._scanner_data_loaded = True

    target_data = _xyz_dataarray(
        np.full((2, 2, 2), 10.0),
        origin=(0.5, 0.5, 0.5),
        spacing=(2.0, 2.0, 2.0),
    )
    target = SimpleNamespace(primary_data=target_data, masks=xr.Dataset())

    masked, projected = experiment.apply_mask_to(
        target,
        mask_name="roi",
        coverage_samples=2,
        return_mask=True,
    )

    np.testing.assert_allclose(projected.values[0, :, :], 0.5)
    np.testing.assert_allclose(masked.values[0, :, :], 5.0)
    np.testing.assert_allclose(masked.values[1, :, :], 0.0)


def test_paper_summary_returns_styled_parameter_table():
    from hypermri.loading.experiment import MRIExperiment

    experiment = MRIExperiment.__new__(MRIExperiment)
    experiment.handler = None
    experiment.path = "/tmp/example_scan"
    experiment.params = {}
    experiment.experiment_number = 7
    experiment.sequence_name = "CSI"
    experiment.vendor = "bruker"
    experiment.nucleus = "13C"
    experiment.matrix = np.array([16, 16, 4])
    experiment.fov_mm = np.array([32.0, 32.0, 8.0])
    experiment.resolution_mm = np.array([2.0, 2.0, 2.0])
    experiment.repetition_time_ms = 250.0
    experiment.echo_time_ms = 1.2
    experiment.read_bandwidth_hz = 50000.0
    experiment.rfpulse_bandwidth_hz = 5400.0
    experiment.rfpulse_shape = "gauss"
    experiment._data = xr.DataArray(
        np.zeros((16, 16, 4)),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    experiment._scanner_data = None

    styler = experiment.paper_summary(display_output=False)
    table = styler.data

    assert list(table.columns) == ["Section", "Parameter", "Value", "Attribute"]
    assert table.loc[table["Attribute"] == "matrix", "Value"].item() == "16 x 16 x 4"
    assert table.loc[table["Attribute"] == "fov_mm", "Value"].item() == "32 x 32 x 8 mm"
    assert (
        table.loc[table["Attribute"] == "resolution_mm", "Value"].item()
        == "2 x 2 x 2 mm"
    )
    assert (
        table.loc[table["Attribute"] == "read_bandwidth_hz", "Value"].item()
        == "5e+04 Hz"
    )
    assert (
        table.loc[table["Attribute"] == "rfpulse_bandwidth_hz", "Value"].item()
        == "5400 Hz"
    )
    assert (
        table.loc[table["Parameter"] == "Default data", "Value"].item()
        == "16 x 16 x 4 (read_idx x phase_idx x slice_idx)"
    )
    rendered_html = styler.to_html()
    assert "background-color: #eef4ff" in rendered_html
    assert "background-color: #eef8f2" in rendered_html
    assert "background-color: #fff0f0" in rendered_html
