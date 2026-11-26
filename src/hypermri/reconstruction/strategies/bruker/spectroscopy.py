"""Bruker spectroscopy raw-data reconstruction strategies."""

import warnings
from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.reconstruction.precision import (
    convert_for_calculation,
    convert_for_storage,
    parse_data_precision,
)
from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft

from ...axes import (
    _label_cpmg_echo_train,
    apply_processed_axis_metadata,
    processed_sampling_axis,
    resolve_cut_off,
)
from .._common import _bruker_reco_provenance, logger
from .raw import BrukerRawRecoStrategy


def _build_spectroscopy_kspace(kspace, sampling_axis, sampling_unit):
    """Label sorted spectroscopy FIDs without imposing spatial geometry.

    Input shape is acquisition/repetition/slice/phase/channel/read. Acquisition
    coordinates use sampling_axis and sampling_unit; other coordinates are
    indices. Samples retain calculation precision and are not modified.
    """
    # Create simple dimensions for spectroscopy data
    dims = [
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    ]
    logger.debug(f"kspace.shape: {kspace.shape}")
    # Create simple index-based coordinates instead of spatial grids
    coords = {
        "acquisition_idx": (
            "acquisition_idx",
            sampling_axis,
            {"unit": sampling_unit},
        ),
        "repetition_idx": (
            "repetition_idx",
            np.arange(kspace.shape[1]),
            {"unit": "index"},
        ),
        "slice_idx": (
            "slice_idx",
            np.arange(kspace.shape[2]),
            {"unit": "index"},
        ),
        "phase_idx": (
            "phase_idx",
            np.arange(kspace.shape[3]),
            {"unit": "index"},
        ),
        "channel_idx": (
            "channel_idx",
            np.arange(kspace.shape[4]),
            {"unit": "index"},
        ),
        "read_idx": (
            "read_idx",
            np.arange(kspace.shape[5]),
            {"unit": "index"},
        ),
    }

    # Create the kspace xarray with simple coordinates
    kspace_xarray = xr.DataArray(
        kspace,
        dims=dims,
        coords=coords,
        attrs={"domain": "kspace", "signal_domain": "temporal"},
    )

    return kspace_xarray


def _fourier_transform_spectroscopy(reco_data):
    """Apply the spectral FFT and fftshift along acquisition_idx.

    Input is a temporal labelled FID. The array is updated in place with
    NumPy FFT precision and signal_domain=spectral; dimensions, coordinates,
    physical units and the existing FFT normalization are retained.
    """
    axes_idx = reco_data.get_axis_num("acquisition_idx")
    reco_data.data = spectral_fft(reco_data.data, axis=axes_idx)
    reco_data.attrs["signal_domain"] = "spectral"

    return reco_data


class BrukerSpectroRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruction strategy for Bruker Spectroscopy data."""

    accepted_options: ClassVar[frozenset[str]] = frozenset(
        {
            "no_reconstruction",
            "postprocess_fid",
            "reshape_fid",
            "set_kspace_xarray",
            "shift_xarray",
            "set_data",
            "db",
            "data_precision",
            "append_parameters",
            "cut_off",
        }
    )

    def _prepare_spectroscopy_fid(self, handler, DATA_PRECISION, requested_cut_off):
        """Load, reorder and trim a calculation copy of the spectroscopy FID.

        Sorted shape is acquisition/repetition/slice/phase/channel/read. Returns
        cutoff and processed sampling metadata in seconds or indices. The
        acquired FID and receiver metadata are preserved; all intermediate
        arrays and effective counts are returned directly.
        """
        if not hasattr(handler, "fid") or handler.fid is None:
            handler.load_data(source=None)

        # Reorder and reshape a converted copy; handler.fid keeps the acquired data.
        acquired_fid = convert_for_calculation(handler.fid, DATA_PRECISION)
        fid_pp = self.reorder_reshape_fid(handler, acquired_fid=acquired_fid)
        raw_n_acquisition = int(fid_pp.shape[0])

        # Remove Bruker's leading digital-filter/transient points exactly once.
        cut_off = resolve_cut_off(
            handler,
            requested_cut_off,
            raw_n_acquisition,
            context="Spectroscopy",
        )
        fid_pp = self.remove_leading_fid_points(handler, cut_off=cut_off, values=fid_pp)
        sampling_axis, sampling_unit = processed_sampling_axis(
            handler,
            cut_off,
            fid_pp.shape[0],
        )

        return (fid_pp, cut_off, raw_n_acquisition, sampling_axis, sampling_unit)

    def reconstruct(self, handler, **reco_options) -> xr.DataArray | None:
        """Reconstruct a Bruker single-voxel or non-localized spectroscopy scan.

        The FID is sorted into the axes (``acquisition_idx``,
        ``repetition_idx``, ``slice_idx``, ``phase_idx``, ``channel_idx``,
        ``read_idx``), the leading points of the digital filter are removed
        once, and the result is passed to :meth:`reco_rawdata`.

        Parameters
        ----------
        handler : BrukerHandler
            Handler of the scan.  A missing FID is read first.
        **reco_options
            The names in :attr:`accepted_options`: ``no_reconstruction``,
            ``postprocess_fid``, ``reshape_fid``, ``shift_xarray``,
            ``set_data``, ``db`` and ``data_precision`` as for
            :class:`BrukerRawRecoStrategy`, with ``postprocess_fid`` and
            ``reshape_fid`` defaulting to False and ``data_precision`` to
            ``float16``, plus ``append_parameters``
            (bool, default True) and ``cut_off`` (number of leading FID
            points to remove; default from the scan parameters). Temporal
            labels are always built from the current processed input; the
            legacy ``set_kspace_xarray`` option no longer controls a handler
            cache. FFT processing remains eager and deterministic.

        Returns
        -------
        xarray.DataArray or None
            Complex spectrum in acquisition/repetition/slice/phase/channel/read
            order with the acquisition axis in shifted FFT frequency order.
            Frequency coordinates use Hz/ppm and sampling times use seconds.
            The sampling-time axis counts from the first kept sample: after
            a cut-off it starts at 0 s, not at the delay since excitation.
            ``patient_pos`` retains the acquisition's patient position when
            available, independently of ``append_parameters``.
            ``bruker_reco_parameter_source``, ``bruker_reco_time`` and
            ``bruker_reco_objects_per_repetition`` snapshot the selected
            handler parameters when available, also without full attachment.
            Amplitudes retain arbitrary scanner units and the established
            unnormalised FFT scale. None means ``no_reconstruction=True``.

        Raises
        ------
        ValueError
            If ``data_precision`` is not supported or the k-space array
            cannot be built.
        RuntimeError
            If the FID size does not match the scan parameters.

        Notes
        -----
        The FID is converted to the calculation precision as a copy: for the
        default ``float16`` the sorted FID uses complex64 while the acquired
        FID retains its dtype. Final spectral storage follows data_precision:
        float16/float32 use complex64 and float64 uses complex128. The handler
        keeps acquired samples and metadata; intermediate/final arrays are returned.
        """
        # check if no reconstruction is requested
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerSpectroRecoStrategy] No reconstruction performed as per settings."
            )
            return None
        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_data = reco_options.get("set_data", True)
        db = reco_options.get("db", True)
        DATA_PRECISION = parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        append_parameters = reco_options.get("append_parameters", True)
        requested_cut_off = reco_options.get("cut_off", None)

        fid_pp, cut_off, raw_n_acquisition, sampling_axis, sampling_unit = (
            self._prepare_spectroscopy_fid(handler, DATA_PRECISION, requested_cut_off)
        )
        # For spectroscopy data, create kspace_xarray manually with simple coordinates
        # instead of using spatial grids that cause coordinate validation errors
        kspace_xarray = _build_spectroscopy_kspace(
            kspace=fid_pp,
            sampling_axis=sampling_axis,
            sampling_unit=sampling_unit,
        )
        patient_pos = getattr(handler, "patient_pos", None)
        if patient_pos is not None:
            kspace_xarray.attrs["patient_pos"] = patient_pos
        kspace_xarray = apply_processed_axis_metadata(
            data=kspace_xarray,
            handler=handler,
            cut_off=cut_off,
            sampling_axis=sampling_axis,
            sampling_unit=sampling_unit,
            raw_n_acquisition=raw_n_acquisition,
        )
        reco_data = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=False,  # Disabled since we created it above
            shift_xarray=shift_xarray,
            set_data=set_data,
            db=db,
            data_precision=DATA_PRECISION,
            values=fid_pp,
            kspace_xarray=kspace_xarray,
        )
        if append_parameters:
            reco_data = self._append_parameters(array=reco_data, handler=handler)
        reco_data = apply_processed_axis_metadata(
            data=reco_data,
            handler=handler,
            cut_off=cut_off,
            sampling_axis=sampling_axis,
            sampling_unit=sampling_unit,
            raw_n_acquisition=raw_n_acquisition,
        )

        reco_data = _fourier_transform_spectroscopy(reco_data=reco_data)
        reco_data.data = convert_for_storage(reco_data.data, DATA_PRECISION)
        reco_data.attrs.update(_bruker_reco_provenance(handler))
        return _label_cpmg_echo_train(reco_data, handler)

    def remove_leading_fid_points(self, handler, cut_off=None, *, values):
        """Return a temporal FID with its leading transient samples removed.

        Parameters
        ----------
        handler : BrukerHandler
            Source of the default cutoff in acquisition samples.
        cut_off : int, optional
            Nonnegative number of leading samples to remove. Defaults to the
            handler's digital-filter setting; must leave at least one sample.
        values : numpy.ndarray
            Sorted complex FID with acquisition samples first and arbitrary
            remaining dimensions, for example the result of
            :meth:`reorder_reshape_fid`. Neither the array nor handler metadata
            are modified.

        Returns
        -------
        numpy.ndarray
            View of the remaining samples, retaining dtype and dimension order.

        Raises
        ------
        ValueError
            If the cutoff is invalid or removes every acquisition sample.
        """
        resolved = resolve_cut_off(
            handler,
            cut_off,
            values.shape[0],
            context="Spectroscopy",
        )
        return values[resolved:, ...]

    def reorder_reshape_fid(self, handler, acquired_fid=None):
        """Sort the acquired spectroscopy FID into its labelled axis order.

        Repetitions that ParaVision stored unaveraged are averaged here.

        Parameters
        ----------
        handler : VendorHandler
            Bruker handler of the scan.  Its ``n_repetitions``,
            ``n_averages``, ``n_slices``, ``n_receivers`` and
            ``n_acquisition`` and the ``method`` and ``reco`` parameters
            define the shape.
        acquired_fid : numpy.ndarray, optional
            The acquired complex FID of any shape whose size matches these
            parameters.  Defaults to ``handler.fid``.  It is not modified.

        Returns
        -------
        numpy.ndarray
            The FID with the axes (acquisition point, repetition, slice,
            echo or object, receiver, 1).

        Raises
        ------
        RuntimeError
            If the parameters give no valid shape or the FID size does not
            match them.

        Notes
        -----
        If the size only fits one receiver or a different acquisition length,
        effective counts are used locally. Acquired metadata are not modified.
        The returned array shape records these effective counts.
        The currently selected ``handler.reco`` owns
        ``RecoObjectsPerRepetition`` when no method echo count is available.
        :meth:`reconstruct` records that selection in result provenance.
        """
        if acquired_fid is None:
            acquired_fid = handler.fid
        logger.debug(f"acquired FID size: {acquired_fid.size}")
        logger.debug(
            f"handler.n_repetitions: {handler.n_repetitions}, handler.n_averages: {handler.n_averages}, handler.n_slices: {handler.n_slices}, handler.n_receivers: {handler.n_receivers}, handler.n_acquisition: {handler.n_acquisition}"
        )
        n_repetitions = int(handler.n_repetitions)
        n_averages = int(getattr(handler, "n_averages", 1) or 1)
        n_slices = int(handler.n_slices)
        n_receivers = int(handler.n_receivers)
        n_acquisition = int(handler.n_acquisition)
        method = getattr(handler, "method", None) or {}
        reco = getattr(handler, "reco", None) or {}
        n_phase_like = int(method.get("NEchoes", 1) or 1)
        if n_phase_like <= 1:
            n_phase_like = int(reco.get("RecoObjectsPerRepetition", 1) or 1)
        base_size = (
            n_repetitions * n_slices * max(n_phase_like, 1) * max(n_receivers, 1)
        )

        if base_size <= 0:
            raise RuntimeError("Invalid spectroscopy dimensions for FID reshaping")

        original_n_receivers = n_receivers
        fid_size = int(acquired_fid.size)
        averaged_size = base_size * n_acquisition
        unaveraged_size = averaged_size * max(n_averages, 1)

        try:
            if fid_size == averaged_size:
                # Many Bruker default spectroscopy scans store averaged FIDs even
                # though PVM_NAverages remains > 1 in the method file.
                fid = np.reshape(
                    acquired_fid,
                    (
                        n_repetitions,
                        n_slices,
                        n_phase_like,
                        1,
                        n_receivers,
                        n_acquisition,
                    ),
                )
            elif fid_size == unaveraged_size:
                fid = np.reshape(
                    acquired_fid,
                    (
                        n_repetitions,
                        n_averages,
                        n_slices,
                        n_phase_like,
                        1,
                        n_receivers,
                        n_acquisition,
                    ),
                )
                fid = np.mean(fid, axis=1)
                fid = np.reshape(
                    fid,
                    (
                        n_repetitions,
                        n_slices,
                        n_phase_like,
                        1,
                        n_receivers,
                        n_acquisition,
                    ),
                )
            else:
                if fid_size % base_size == 0:
                    actual_n_acquisition = fid_size // base_size
                elif n_receivers != 1 and fid_size % (n_repetitions * n_slices) == 0:
                    n_receivers = 1
                    base_size = n_repetitions * n_slices
                    actual_n_acquisition = fid_size // base_size
                else:
                    raise ValueError(
                        "FID size is incompatible with repetition/slice/channel dimensions"
                    )

                if abs(actual_n_acquisition - n_acquisition) > 1:
                    logger.warning(
                        "Adjusting n_acquisition from %s to %s based on actual FID size",
                        n_acquisition,
                        actual_n_acquisition,
                    )
                    n_acquisition = actual_n_acquisition

                fid = np.reshape(
                    acquired_fid,
                    (
                        n_repetitions,
                        n_slices,
                        n_phase_like,
                        1,
                        n_receivers,
                        n_acquisition,
                    ),
                )
        except ValueError as exc:
            raise RuntimeError(
                "Spectroscopy reconstruction of raw data failed.\n"
                "Please check the input data or the reconstruction strategy.\n"
                f"FID size: {acquired_fid.size}, "
                f"Expected averaged shape: ({n_repetitions}, {n_slices}, {n_phase_like}, 1, {original_n_receivers}, {handler.n_acquisition}), "
                f"Expected unaveraged shape: ({n_repetitions}, {n_averages}, {n_slices}, {n_phase_like}, 1, {original_n_receivers}, {handler.n_acquisition})\n"
                f"Error: {exc}"
            ) from exc

        fid = np.transpose(fid, (5, 0, 1, 2, 4, 3))
        return fid


__all__ = ["BrukerSpectroRecoStrategy"]
