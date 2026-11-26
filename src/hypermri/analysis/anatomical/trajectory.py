"""Slice-selection trajectories estimated from acquired phase differences."""

import matplotlib.pyplot as plt
import numpy as np


def calc_k_space_trajectory_from_measurement(scan=None, db=False):
    """Estimate the slice-selection trajectory from phase-encoded FIDs.

    Parameters
    ----------
    scan : MRIExperiment or vendor handler
        Complex acquired FID in repetition/slice/channel/read order. The read
        length is matrix[0]*matrix[1]. Requires at least two slices, n_slices,
        n_repetitions, n_receivers and a nonzero slice_gap_mm. Vendor samples
        are read from scan.handler.fid when a scan is supplied.
    db : bool, optional
        Whether to create intermediate real-FID and trajectory plots.

    Returns
    -------
    numpy.ndarray
        One trajectory sample per read point, in reciprocal metres. The phase
        difference of slices 0 and 1 is divided by their spacing in metres.

    Raises
    ------
    AttributeError
        If the required samples or acquisition metadata are missing.
    ValueError, IndexError
        If FID size does not match the stated axes or fewer than two slices exist.

    Notes
    -----
    Repetitions and receivers are averaged before unwrapping the slice phases.
    The acquired FID is not mutated. Calculation follows NumPy input precision;
    no storage conversion or reconstruction is performed.
    """
    handler = getattr(scan, "handler", scan)
    fid = handler.fid

    # length of trajetory:
    traj_length = scan.matrix[0] * scan.matrix[1]

    # number of slices
    nslices = scan.n_slices

    # number of repetitions:
    nr = scan.n_repetitions

    # number of repetitions:
    nchans = scan.n_receivers

    # reshape trajectory
    fid = np.reshape(fid, (nr, nslices, nchans, traj_length))

    if db is True:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        for r in range(nr):
            for c in range(nchans):
                ax[0].plot(np.real(fid[r, 0, c, :]), label=f"Rep {r}, Chan {c}")
                ax[1].plot(np.real(fid[r, 1, c, :]), label=f"Rep {r}, Chan {c}")
            ax[0].plot(
                np.mean(np.real(fid[r, 0, :, :]), axis=-2),
                color="k",
                label=f"Rep {r}, Mean",
            )
            ax[1].plot(
                np.mean(np.real(fid[r, 1, :, :]), axis=-2),
                color="k",
                label=f"Rep {r}, Mean",
            )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title("Slice 0")
        ax[1].set_title("Slice 1")

    # average along the repetitions:
    fid = np.squeeze(np.mean(fid, axis=(0, -2)))

    # define slice distance:
    dx = scan.slice_gap_mm * 1e-3  # in metres

    traj = (np.unwrap(np.angle(fid[0, :])) - np.unwrap(np.angle(fid[1, :]))) / (
        2 * np.pi * dx
    )
    if db is True:
        ax[2].plot(traj)
        ax[2].set_title("Trajectory")
        plt.show()

    return traj
