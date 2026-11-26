"""Trim ParaVision storage blocks from already decoded quadrature samples.

The block-byte alignment and acquisition-prefix trimming are adapted from
brukerapi 0.4.6, ``brukerapi/config/properties_fid_core.json`` (``block_size``)
and ``brukerapi/schemas.py`` (``SchemaFid._acquisition_trim``).
Copyright (c) 2020 isi-nmr; MIT licence retained in
``LICENSES/brukerapi-MIT.txt`` with source details in ``NOTICE``.
The published source archive has SHA-256
``24616ea608169cdc7b67f935da403cfd5475824d8637b7f8625350c25887fd52``.

This adapter works on complex sample pairs in row-major acquisition order.
It neither rereads the FID nor repeats decoding, receiver reordering or any
sequence-specific reconstruction. Input samples and metadata are never mutated.
"""

import numpy as np


def _strip_block_padding(complex_file, acqp, num_selected_receivers):
    """Return acquired line prefixes, preserving the decoded sample dtype.

    Parameters
    ----------
    complex_file : numpy.ndarray
        Already decoded complex samples in unscaled scanner units, including
        trailing storage padding on each line. Flattening follows C order.
    acqp : dict
        Parsed acquisition metadata. ``ACQ_size`` is a sequence of scalar
        sample counts, with readout first. ``NI`` and ``NR`` multiply the
        remaining acquisition dimensions to give the number of stored lines.
        ``GO_block_size=Standard_KBlock_Format`` aligns each receiver-combined
        line to 1024 bytes; other block modes retain the acquired line length.
        ``GO_raw_data_format`` determines the stored scalar width independently
        of the decoded array's calculation precision.
    num_selected_receivers : int
        Number of receiver readouts stored together in each acquisition line.

    Returns
    -------
    numpy.ndarray
        Shape ``(lines, ACQ_size[0] * num_selected_receivers // 2)`` in
        acquisition order, with the input dtype and no arithmetic on samples.
        The result shares input storage when reshaping permits a view.

    Raises
    ------
    KeyError
        If required metadata are absent.
    TypeError
        If acquisition dimensions or counts cannot describe the stored layout.
    ValueError
        If the acquired sample count disagrees with the complete storage layout.

    Notes
    -----
    This eager, deterministic adapter retains HyperMRI's four-byte fallback
    for formats other than ``GO_16BIT_SGN_INT``. Actual file decoding validates
    supported formats separately. Nonzero trailing padding is discarded without
    a warning, preserving the existing adapter's behavior. Scalar ``ACQ_size``
    without a dimension sequence remains unsupported.
    """
    scalar_bytes = 2 if acqp["GO_raw_data_format"] == "GO_16BIT_SGN_INT" else 4
    acquisition_shape = acqp["ACQ_size"]
    acquired_scalars = int(acquisition_shape[0] * num_selected_receivers)
    acquired_complex = acquired_scalars // 2
    stored_complex = acquired_complex
    if acqp["GO_block_size"] == "Standard_KBlock_Format":
        acquired_bytes = acquired_scalars * scalar_bytes
        stored_bytes = ((acquired_bytes + 1023) // 1024) * 1024
        stored_complex = stored_bytes // (2 * scalar_bytes)

    lines = int(np.prod(acquisition_shape[1:]) * acqp["NI"] * acqp["NR"])
    expected_samples = lines * stored_complex
    actual_samples = np.size(complex_file)
    if actual_samples != expected_samples:
        raise ValueError(
            "Bruker block-padding removal cannot reshape the raw FID: "
            f"got {actual_samples} complex samples, expected {expected_samples} "
            f"for ACQ_size={acquisition_shape!r}, NI={acqp['NI']!r}, "
            f"NR={acqp['NR']!r}, receivers={num_selected_receivers!r}, "
            f"GO_block_size={acqp['GO_block_size']!r}, "
            f"GO_raw_data_format={acqp['GO_raw_data_format']!r}. Check that the "
            "FID and its acquisition parameter files describe the same "
            "acquisition and that the FID is complete."
        )

    stored_lines = np.reshape(complex_file, (lines, stored_complex))
    if acquired_complex != stored_complex:
        return stored_lines[:, :acquired_complex]
    return stored_lines
