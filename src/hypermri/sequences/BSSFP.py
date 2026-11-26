"""Balanced SSFP anatomical sequence implementation for HyperMRI."""

import os
import traceback
from copy import deepcopy

import numpy as np

from hypermri._optional import _notebook_alias as __getattr__
from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.reconstruction.strategies.bruker.bssfp import MULTI_ECHO_METHODS

from ..utils.io.files import load_plot_params
from ..utils.mathematics.calculations import get_histogram
from ..utils.system.logging import init_default_logger
from ..utils.system.objects import add_counter, get_counter
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class BSSFP(BaseAnatomy):
    """Balanced steady-state free precession anatomy scan with coil-aware reconstruction helpers."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the BSSFP sequence wrapper and trigger reconstructions.

        Parameters
        ----------
        handler : VendorHandlerProtocol
            Bruker vendor handler of the scan.
        **kwargs
            ``BaseSequence`` construction options such as ``load_data``.

        Raises
        ------
        KeyError
            If the ``method`` file has neither ``PVM_NEchoImages`` nor
            ``PVM_NMovieFrames``; the message names the parameter and the
            scan.
        """
        super().__init__(handler, **kwargs)

        self.n_echoes = self.get_nechoes()
        self.Reconstructed_data = None

    def get_nechoes(self) -> int:
        """Return the number of echo images or movie frames of the scan.

        Returns
        -------
        int
            ``PVM_NEchoImages`` when the ``method`` file has it; for the
            multi-echo method ``<User:nmrsuFISP_H2_Fb>`` the number of echo
            times in ``DelayArrayForTE``; otherwise ``PVM_NMovieFrames``,
            unchanged.

        Raises
        ------
        KeyError
            If the ``method`` file has neither parameter; the message names
            ``PVM_NMovieFrames`` and the scan.
        """
        if "PVM_NEchoImages" in self.handler.method:
            return self.handler.method_parameter("PVM_NEchoImages")
        if self.handler.method.get("Method") in MULTI_ECHO_METHODS:
            return int(np.size(self.handler.method_parameter("DelayArrayForTE")))
        return self.handler.method_parameter("PVM_NMovieFrames")

    def reco_scrambled_phase(self, values=None, path_to_coord=None, plot_db=False):
        """Sort the legacy scrambled bSSFP phase encodings and apply its FFT.

        No maintained reconstruction implements this scrambled encoding. The legacy
        ``bssfp_fft`` extension is required and checked before loading or mutation.
        Supported Cartesian acquisitions use ``reconstruct("raw")`` instead.

        Parameters
        ----------
        values : numpy.ndarray, optional
            Complex acquired signal in arbitrary scanner units, with
            ``n_repetitions * matrix[2] * matrix[1] * 4 * matrix[0]`` samples.
            None uses the locally sorted FID or the handler FID. A copy is sorted.
        path_to_coord : str or pathlib.Path, optional
            Required space-separated table of x/y encoding coordinates. Coordinates
            are binned in their supplied units into the phase/slice matrix.
        plot_db : bool, optional
            Display the coordinate acquisition order when True.

        Returns
        -------
        numpy.ndarray
            Sorted complex k-space with shape
            ``(1, matrix[0], matrix[1], matrix[2], n_repetitions, 1)``,
            ordered as echo/read/phase/slice/repetition/channel. The supplied FFT
            stores its spatial image on ``self.image``.

        Raises
        ------
        NotImplementedError
            If the legacy ``bssfp_fft`` implementation is unavailable.
        Exception
            If the coordinate path is absent or its table cannot be loaded.
        ValueError
            If sample counts, coordinate bins or reshaping are inconsistent.

        Notes
        -----
        This eager legacy route stores ``self.fid`` on its first sorting operation
        and updates the unscramble counter. It preserves the acquired handler FID;
        FFT precision and normalization are determined by the injected extension.
        """
        import matplotlib.pyplot as plt

        missing = [name for name in ["bssfp_fft"] if not hasattr(self, name)]
        if missing:
            raise NotImplementedError(
                "BSSFP.reco_scrambled_phase requires unavailable legacy inputs: "
                + ", ".join(missing)
                + ". Use reconstruct('raw') for supported Cartesian acquisitions; scrambled encoding has no maintained reconstruction."
            )
        try:
            from matplotlib import colormaps

            get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
        except ImportError:
            from matplotlib.cm import get_cmap  # Matplotlib <3.10
        if values is None:
            # ``unscramble`` stores its sorted FID on the scan; prefer it.
            values = vars(self)["fid"] if "fid" in vars(self) else self.handler.fid
        else:
            pass

        if path_to_coord is None:
            raise Exception("path_to_coord parameter is empty")
        else:
            pass

        # load path_to_coord:
        import pandas as pd

        # Load the coordinates
        try:
            df = pd.read_csv(path_to_coord, sep=" ", header=None, names=["x", "y"])
        except BaseException:
            raise Exception("Coordinates could not be loaded!")

        # parameters:
        mat = self.matrix
        n_repetitions = self.n_repetitions

        data = values.copy()

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        # factor of 4 will be removed later:
        d = np.reshape(data, [n_repetitions, mat[2], mat[1], 4, mat[0]])

        # only use first out of 4 entries (rest should be empty):

        # If nr is 1, add an extra dimension to d
        if n_repetitions == 1:
            d = d[..., np.newaxis]

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        num_bins = [mat[2], mat[1]]
        matrix = np.zeros_like(d)

        if plot_db is True:
            import matplotlib.colors as colors

            # Create a colormap
            cmap = get_cmap("jet")
            norm = colors.Normalize(vmin=0, vmax=len(df) - 1)

            # Create the plot
            fig, ax = plt.subplots()

            # Scatter plot with points connected and colored based on their
            # index
            for i in range(len(df) - 1):
                ax.plot(
                    df.iloc[i : i + 2, 0], df.iloc[i : i + 2, 1], color=cmap(norm(i))
                )

            # Display the plot
            plt.show()

        # Use binned_statistic_2d to sort the 1D data into a 2D grid based on
        # the x and y coordinates
        from scipy.stats import binned_statistic_2d

        if n_repetitions == 1:
            for r in range(n_repetitions):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic[..., np.newaxis]
        else:
            for r in range(n_repetitions):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic

        num_unscramle_counts = get_counter(
            scan=self,
            counter_name="unscramble",
        )

        # if was not saved yet:
        if num_unscramle_counts == 0:
            # save sorted matrix as fid
            matrix = np.reshape(matrix, (mat[1] * mat[2] * n_repetitions, -1))
            matrix = self.handler.postprocess_fid_file(values=matrix)
            self.fid = matrix

            add_counter(
                scan=self,
                counter_name="unscramble",
                n_counts=1,
            )
        else:
            # reshape data
            matrix = np.reshape(matrix, (mat[1] * mat[2] * n_repetitions, -1))
            # remove 0s
            matrix = self.handler.postprocess_fid_file(values=matrix)

        # bring data into [echoes, z, y, x, repetitions, channels] format
        matrix = np.reshape(
            matrix,
            (
                self.n_repetitions,
                self.matrix[1],
                self.matrix[2],
                self.matrix[0],
            ),
        )
        print(matrix.shape)
        matrix = np.expand_dims(matrix, axis=(4, 5))
        print(matrix.shape)
        matrix = np.transpose(matrix, (4, 3, 1, 2, 0, 5))

        self.image = self.bssfp_fft(values=matrix, axes=(1, 2, 3))

        return matrix
