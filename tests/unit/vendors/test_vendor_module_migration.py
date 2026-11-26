"""Characterize invented TopSpin data before moving the vendor modules."""

import importlib
import pickle

import numpy as np
import pytest
import xarray as xr


def _dataset(tmp_path, endian="<", floating=False):
    scan = tmp_path / "10"
    scan.mkdir()
    (scan / "acqus").write_text(
        "##$PULPROG= <invented>\n##$NUC1= <13C>\n"
        "##$BF1= 100.0\n##$SFO1= 100.001\n##$SW_h= 2000.0\n##$SW= 20.0\n"
        "##$TD= 160\n##$O1= 1000.0\n##$DATE= 1700000000\n"
        f"##$DTYPA= {2 if floating else 0}\n"
        f"##$BYTORDA= {0 if endian == '<' else 1}\n",
        encoding="utf-8",
    )
    # Eighty complex samples leave ten after the established 70-point cutoff.
    raw = np.arange(160, dtype=np.float64).astype(endian + ("f8" if floating else "i4"))
    (scan / "fid").write_bytes(raw.tobytes())
    expected = np.arange(140, 160, 2) + 1j * np.arange(141, 160, 2)
    return scan, expected


def _handler_class():
    return importlib.import_module("hypermri.vendors.topspin").BrukerTopSpinHandler


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("floating", [False, True])
def test_topspin_binary_values_labels_and_repeatability(tmp_path, endian, floating):
    """Protect the byte interpretation, cutoff, units and original acquired file."""
    scan, expected = _dataset(tmp_path, endian, floating)
    payload = (scan / "fid").read_bytes()
    handler = _handler_class()(scan)
    first = handler.load_data()
    snapshot = first.copy(deep=True)
    second = handler.load_data()

    assert first.dtype == np.complex128
    assert first.dims == ("acquisition_idx",)
    np.testing.assert_array_equal(first.values, expected)
    np.testing.assert_array_equal(first.acquisition_idx, np.arange(10))
    np.testing.assert_allclose(
        first.acquisition_time_s, np.arange(10) / 2000, atol=1e-15
    )
    np.testing.assert_allclose(
        first.freq_axis_hz, np.arange(-1000, 1000, 200), atol=1e-12
    )
    np.testing.assert_allclose(first.freq_axis_ppm, handler.freq_axis_ppm, atol=1e-12)
    assert first.attrs == {
        "unit": "a.u.",
        "domain": "time",
        "nucleus": "13C",
        "sequence_name": "invented",
        "bandwidth_hz": 2000.0,
        "bandwidth_ppm": 20.0,
        "center_ppm": 10.0,
        "cut_off": 70,
        "reference_frequency_mhz": 100.0,
        "source_path": str(scan.resolve()),
    }
    xr.testing.assert_identical(first, snapshot)
    xr.testing.assert_identical(second, snapshot)
    assert (scan / "fid").read_bytes() == payload


def test_topspin_metadata_constructor_does_not_read_binary(tmp_path, monkeypatch):
    scan, _ = _dataset(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata-only construction read the acquired binary")

    monkeypatch.setattr(np, "fromfile", forbidden)
    handler = _handler_class()(scan, load_data=False)
    assert handler.id == 10
    assert handler.type == "invented"
    assert handler.n_acquisition == 80
    assert handler.dwell_time_s == 0.0005
    assert handler.summary()["vendor"] == "bruker_topspin"


def test_topspin_eager_and_explicit_loading_are_identical(tmp_path):
    scan, _ = _dataset(tmp_path)
    eager = _handler_class()(scan, load_data=True)
    lazy = _handler_class()(scan)
    xr.testing.assert_identical(eager.scanner_data, lazy.load_data())


def test_topspin_rejects_paravision_and_reports_missing_fid(tmp_path):
    scan, _ = _dataset(tmp_path)
    cls = _handler_class()
    handler = cls(scan)
    (scan / "method").touch()
    assert not cls.is_valid_scan_dir(scan)
    with pytest.raises(FileNotFoundError, match="valid TopSpin"):
        cls(scan)
    (scan / "fid").unlink()
    with pytest.raises(FileNotFoundError, match="FID file not found"):
        handler.load_data()


def test_vendor_parameter_snapshot_and_legacy_factory_dispatch(tmp_path, monkeypatch):
    """Retain the original base interface and legacy dispatch without real scanners."""
    from hypermri.vendors import bruker, siemens
    from hypermri.vendors.base import VendorFactory, VendorHandler

    class Dummy(VendorHandler):
        id = "invented"

        def _parse_parameters(self, source):
            return {"source": source}

        def load_data(self, source):
            return xr.DataArray([1.0], dims=("acquisition_idx",))

    handler = Dummy()
    assert handler.get_parameters() == {}
    assert handler.load_parameters("invented") == {"source": "invented"}
    snapshot = handler.summary()
    snapshot["source"] = "changed"
    assert handler.get_parameters() == {"source": "invented"}
    (tmp_path / "fid").touch()
    monkeypatch.setattr(bruker, "BrukerHandler", lambda source: ("bruker", source))
    monkeypatch.setattr(siemens, "SiemensHandler", lambda source: ("siemens", source))
    assert VendorFactory.get_handler(str(tmp_path)) == ("bruker", str(tmp_path))
    assert VendorFactory.get_handler("invented.dat") == ("siemens", "invented.dat")
    with pytest.raises(ValueError, match="Unsupported data source"):
        VendorFactory.get_handler("invented.unsupported")


def test_canonical_vendor_exports_and_instance_pickle(tmp_path):
    """Use the domain exports and retain the complete numerical/metadata instance."""
    from hypermri.vendors import BrukerTopSpinHandler, VendorHandler
    from hypermri.vendors.base import VendorHandler as CanonicalBase
    from hypermri.vendors.topspin import BrukerTopSpinHandler as CanonicalTopSpin

    assert VendorHandler is CanonicalBase
    assert BrukerTopSpinHandler is CanonicalTopSpin
    scan, _ = _dataset(tmp_path)
    handler = BrukerTopSpinHandler(scan, load_data=True)
    restored = pickle.loads(pickle.dumps(handler))
    assert type(restored) is BrukerTopSpinHandler
    xr.testing.assert_identical(restored.scanner_data, handler.scanner_data)
    assert restored.id == handler.id
    assert restored.path == handler.path
