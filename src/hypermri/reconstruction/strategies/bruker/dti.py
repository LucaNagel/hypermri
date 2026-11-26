"""Raw reconstruction for Cartesian Bruker ``DtiStandard`` acquisitions."""

from pathlib import Path

import numpy as np
from brukerapi.dataset import Dataset

from hypermri.reconstruction.precision import parse_data_precision

from .cartesian import BrukerCartesianRawRecoStrategy


class BrukerDtiRawRecoStrategy(BrukerCartesianRawRecoStrategy):
    """Reconstruct 2D Cartesian DTI data with explicit diffusion-image loops."""

    def reconstruct(self, handler, **reco_options):
        """Reconstruct Cartesian 2D DtiStandard diffusion acquisitions.

        The Bruker reader decodes padding and binary format and sorts phase
        steps and acquisition objects once. Slices vary fastest inside diffusion images, followed by repetitions
        and receivers.
        Only read and phase axes are inverse transformed; missing phase rows
        are zero filled. The acquired arrays are never modified.

        Parameters
        ----------
        handler : BrukerHandler
            Scan metadata and a nonempty ``fid`` file. Matrix, encoding steps,
            object counts and physical geometry must describe the acquisition.
        **reco_options : dict
            ``no_reconstruction`` returns None. ``data_precision`` chooses
            float16/float32 (complex64) or float64 (complex128) storage.
            ``shift_xarray`` enables FOV-offset correction (default True);
            ``fov_shift_sign`` is its sign (default 1). ``append_parameters``
            attaches scanner metadata (default True). Strategy defaults are
            resolved by the experiment API before this call.

        Returns
        -------
        xarray.DataArray or None
            Complex image intensities in scanner units. Dimensions follow
            read, phase and the documented acquisition loops, with index and
            spatial coordinates in mm provided by the Cartesian finisher.
            Reconstruction is eager, deterministic and does not cache raw data.

        Raises
        ------
        ValueError
            If geometry, encoding, object order or raw layout is inconsistent.
        OSError
            If the acquisition file cannot be read.
        brukerapi.exceptions.InvalidDataset
            If the declared storage layout does not match the file.

        Examples
        --------
        >>> image = scan.reconstruct("raw", data_precision="float64")  # doctest: +SKIP
        """
        if reco_options.get("no_reconstruction", False):
            return None
        precision = parse_data_precision(reco_options.get("data_precision", np.float32))

        if str(getattr(handler, "spatial_dimension", "")).upper() != "2D":
            raise ValueError(
                "Bruker DtiStandard raw reconstruction currently supports 2D "
                "Cartesian acquisitions only."
            )

        fid_path = Path(handler.path) / "fid"
        if not fid_path.is_file() or fid_path.stat().st_size <= 0:
            raise ValueError("DTI raw reconstruction requires a non-empty fid file.")

        dataset = Dataset(fid_path)
        raw = np.asarray(dataset.data)
        if raw.ndim != 5:
            raise ValueError(
                "Unsupported Bruker DTI fid layout: expected five axes "
                f"(read, phase, object, repetition, channel), got {raw.shape}."
            )

        matrix = self._integer_values(handler.matrix, "PVM_Matrix")
        if matrix.size < 2:
            raise ValueError(f"Invalid Bruker PVM_Matrix for DTI: {matrix}.")
        read_size, phase_size = map(int, matrix[:2])
        if raw.shape[0] != read_size:
            raise ValueError(
                "DTI raw readout length does not match PVM_Matrix: "
                f"{raw.shape[0]} != {read_size}."
            )

        method = getattr(handler, "method", None) or {}
        phase_order, phase_positions = self._phase_positions(
            method.get("PVM_EncSteps1"), phase_size, "PVM_EncSteps1"
        )
        if phase_order.size != raw.shape[1]:
            raise ValueError(
                "DTI PVM_EncSteps1 length does not match the decoded fid: "
                f"{phase_order.size} != {raw.shape[1]}."
            )

        n_slices = self._positive_integer_parameter(
            getattr(handler, "n_slices", 1), "PVM_SPackArrNSlices"
        )
        n_diffusion_images = self._positive_integer_parameter(
            method.get("PVM_DwNDiffExp"), "PVM_DwNDiffExp"
        )
        object_count = int(raw.shape[2])
        expected_objects = n_slices * n_diffusion_images
        if object_count != expected_objects:
            raise ValueError(
                "DTI object count does not match slice/diffusion metadata: "
                f"{object_count} != {n_slices} * {n_diffusion_images}."
            )

        # Dataset has already applied ACQ_obj_order. The physical slice loop
        # is inside the diffusion-image loop; only split those semantic loops.
        self._object_order(handler, object_count)
        canonical = raw
        grouped = canonical.reshape(
            read_size,
            raw.shape[1],
            n_diffusion_images,
            n_slices,
            raw.shape[3],
            raw.shape[4],
        ).transpose(0, 1, 3, 2, 4, 5)

        kspace = np.zeros(
            (
                read_size,
                phase_size,
                n_slices,
                n_diffusion_images,
                raw.shape[3],
                raw.shape[4],
            ),
            dtype=grouped.dtype,
        )
        kspace[:, phase_positions, :, :, :, :] = grouped

        return self._finish_reconstruction(
            handler,
            kspace,
            [
                "read_idx",
                "phase_idx",
                "slice_idx",
                "bvalue_idx",
                "repetition_idx",
                "channel_idx",
            ],
            spatial_dims=("read_idx", "phase_idx"),
            shift_xarray=reco_options.get("shift_xarray", True),
            fov_shift_sign=float(reco_options.get("fov_shift_sign", 1.0)),
            data_precision=precision,
            append_parameters=reco_options.get("append_parameters", True),
            description="Bruker DtiStandard reconstruction from fid raw data",
            extra_attrs={
                "raw_source": "fid",
                "phase_encode_partial_fourier": raw.shape[1] != phase_size,
            },
        )
