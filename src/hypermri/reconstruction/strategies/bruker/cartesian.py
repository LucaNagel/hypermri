"""Shared helpers for specialized Bruker Cartesian raw reconstructions."""

from __future__ import annotations

from typing import ClassVar, Iterable

import numpy as np
import xarray as xr

from hypermri.reconstruction.centred_dft import centered_ifft
from hypermri.reconstruction.precision import convert_for_storage
from hypermri.utils.coordinates.image_shifts import shift_image
from hypermri.utils.system.logging import init_default_logger

from ._encoding import _encoding_step_positions
from .geometry import _attach_image_geometry
from .raw import BrukerRawRecoStrategy

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BrukerCartesianRawRecoStrategy(BrukerRawRecoStrategy):
    """Base class for raw formats whose loop order is sequence-specific."""

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "data_precision",
            "shift_xarray",
            "fov_shift_sign",
            "append_parameters",
        }
    )

    @staticmethod
    def _integer_values(value, parameter_name: str) -> np.ndarray:
        try:
            numeric = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected integer values."
            ) from exc
        if numeric.size == 0 or not np.all(np.isfinite(numeric)):
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected integer values."
            )
        integers = numeric.astype(int)
        if not np.array_equal(numeric, integers):
            raise ValueError(
                f"Invalid Bruker {parameter_name}: expected integer values."
            )
        return integers

    @classmethod
    def _phase_positions(
        cls, steps, matrix_size: int, parameter_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return acquisition-to-sorted order and centered matrix positions."""
        positions = _encoding_step_positions(steps, matrix_size, parameter_name)
        if np.unique(positions).size != positions.size:
            raise ValueError(f"Invalid Bruker {parameter_name}: duplicate steps.")
        order = np.argsort(positions, kind="stable")
        return order, positions[order]

    @staticmethod
    def _object_order(handler, count: int) -> np.ndarray:
        """Return the physical object index for each acquired Bruker object."""
        value = (getattr(handler, "acqp", None) or {}).get("ACQ_obj_order")
        try:
            order = np.asarray(value, dtype=int).reshape(-1)
        except (TypeError, ValueError):
            order = np.arange(count)
        if order.size == 1 and count == 1:
            return np.zeros(1, dtype=int)
        if order.size != count or not np.array_equal(np.sort(order), np.arange(count)):
            raise ValueError(
                "Invalid Bruker ACQ_obj_order: expected a permutation of "
                f"0..{count - 1}, got {order.tolist()}."
            )
        return order

    @staticmethod
    def _coords(handler, dims: list[str], shape: tuple[int, ...]) -> dict:
        coords: dict = {}
        for dim, size in zip(dims, shape, strict=True):
            values = np.arange(size)
            unit = "index"
            if dim == "bvalue_idx":
                candidate = np.asarray(getattr(handler, "b_values", [])).reshape(-1)
                if candidate.size >= size:
                    values = candidate[:size]
                    unit = "s/mm2"
            elif dim == "inversion_idx":
                candidate = np.asarray(
                    getattr(handler, "inversion_time_ms", [])
                ).reshape(-1)
                if candidate.size >= size:
                    values = candidate[:size]
                    unit = "ms"
            elif dim == "echo_idx":
                candidate = np.asarray(
                    (getattr(handler, "method", None) or {}).get("EffectiveTE", [])
                ).reshape(-1)
                if candidate.size >= size:
                    values = candidate[:size] / 1000.0
                    unit = "s"
            elif dim == "acquisition_idx":
                candidate = np.asarray(
                    getattr(handler, "sampling_time_axis_s", [])
                ).reshape(-1)
                if candidate.size >= size:
                    values = candidate[:size]
                    unit = "s"
            coords[dim] = (dim, values, {"unit": unit})
        return coords

    @staticmethod
    def _base_attrs(handler) -> dict:
        return {
            "patient_pos": getattr(handler, "patient_pos", None),
            "slice_orientation": getattr(handler, "slice_orientation", None),
            "read_orientation": getattr(handler, "read_orientation", None),
            "num_image_fov_shift": 0,
            "num_rearrange": 0,
            "signal_domain": "temporal",
            "domain": "kspace",
            "spatial_dimension": getattr(handler, "spatial_dimension", None),
            "git_commit_hash": getattr(handler, "git_commit_hash", None),
        }

    def _finish_reconstruction(
        self,
        handler,
        kspace: np.ndarray,
        dims: list[str],
        *,
        spatial_dims: tuple[str, ...],
        shift_xarray: bool = True,
        fov_shift_sign: float = 1.0,
        data_precision=np.float32,
        append_parameters: bool = True,
        description: str,
        extra_attrs: dict | None = None,
    ) -> xr.DataArray:
        """Attach metadata, apply FOV shifts, and transform k-space to image space."""
        if kspace.ndim != len(dims):
            raise ValueError(
                f"Raw reconstruction produced shape {kspace.shape} for dims {dims}."
            )

        kspace = convert_for_storage(kspace, data_precision)
        coords = self._coords(handler, dims, kspace.shape)
        kspace_xarray = xr.DataArray(
            kspace,
            dims=dims,
            coords=coords,
            attrs=self._base_attrs(handler),
            name="kspace",
        )
        if extra_attrs:
            kspace_xarray.attrs.update(extra_attrs)

        if shift_xarray:
            kspace_xarray = shift_image(
                data=kspace_xarray,
                shift_method="phase",
                scan=handler,
                domain="kspace",
                fov_shift_sign=fov_shift_sign,
            )

        axes = tuple(kspace_xarray.get_axis_num(dim) for dim in spatial_dims)
        image = centered_ifft(kspace_xarray.to_numpy(), axes)
        reco_data = xr.DataArray(
            convert_for_storage(image, data_precision),
            dims=kspace_xarray.dims,
            coords=kspace_xarray.coords,
            attrs=dict(kspace_xarray.attrs),
            name="raw_reconstruction",
        )
        reco_data.attrs.update(
            {
                "domain": "spatial",
                "description": description,
                "raw_reconstruction": True,
            }
        )

        # Bruker ``fid`` and ``rawdata.job0`` are already expressed in the
        # scanner's read/phase/slice convention used by ``2dseq``. Applying
        # the generic patient-position flip here would flip these data twice.
        reco_data = _attach_image_geometry(reco_data, handler)
        if append_parameters:
            reco_data = self._append_parameters(array=reco_data, handler=handler)
        return reco_data
