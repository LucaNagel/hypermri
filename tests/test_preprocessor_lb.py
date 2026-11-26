import numpy as np
import xarray as xr

from hypermri.utils.plot.data_preprocessor import DataPreprocessor


def test_preprocessor_lb():
    # Create dummy FID data
    npoints = 100
    t = np.linspace(0, 0.1, npoints)
    fid = np.exp(-t * 10) * np.exp(1j * 2 * np.pi * 50 * t)

    da = xr.DataArray(fid, dims=["acquisition_idx"], coords={"acquisition_idx": t})
    da.attrs["signal_domain"] = "temporal"

    class DummyObj:
        def __init__(self):
            self.sampling_time_axis_s = t
            self.n_acquisition = npoints
            self.bandwidth_hz = 1000
            self.cut_off = 0

        def get(self, key, default=None):
            return getattr(self, key, default)

    obj = DummyObj()
    preprocessor = DataPreprocessor(obj)

    # 1. No LB
    kwargs = {"ds0_linebroadening": 0.0, "ds0_view_toggle": True}
    out1 = preprocessor.apply_preprocessing(da, kwargs, "ds0")

    # 2. With LB
    kwargs = {"ds0_linebroadening": 50.0, "ds0_view_toggle": True}
    out2 = preprocessor.apply_preprocessing(da, kwargs, "ds0")

    print(f"Max out1: {np.max(np.abs(out1.values))}")
    print(f"Max out2: {np.max(np.abs(out2.values))}")

    diff = np.abs(out1.values - out2.values).sum()
    print(f"Total diff: {diff}")

    if diff > 0:
        print("SUCCESS: Line broadening changed the data.")
    else:
        print("FAILURE: Line broadening DID NOT change the data.")


if __name__ == "__main__":
    test_preprocessor_lb()
