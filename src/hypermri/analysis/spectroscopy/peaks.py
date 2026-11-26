"""Peak selection along labelled spectroscopy sample dimensions."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["find_npeaks", "find_npeaks_nd"]


def find_npeaks(
    values: np.ndarray = None,
    npeaks: int = None,
    freq_range: np.ndarray = None,
    plot: bool = False,
    find_peaks_params: dict = None,
    sort_order: str = None,
):
    """Select the largest local magnitude peaks in one spectrum.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional real or complex spectrum after singleton squeezing.
    npeaks : int, optional
        Maximum number of peaks; None retains all detected peaks.
    freq_range : numpy.ndarray, optional
        Plot positions matching the spectrum, in caller-defined frequency units.
        Missing positions use sample indices; peak selection is independent of them.
    plot : bool, optional
        Display a spectrum and its selected peaks.
    find_peaks_params : dict, optional
        Explicit SciPy find_peaks options, with distances measured in sample indices.
    sort_order : str, optional
        Ascending or descending sample-index order; other values and None retain
        decreasing magnitude order.

    Returns
    -------
    numpy.ndarray
        Integer sample indices, with at most npeaks entries. Missing peaks are not
        padded; output is shorter when fewer local maxima are detected.

    Raises
    ------
    ValueError
        If squeezed data are not one-dimensional or SciPy rejects peak options.
    TypeError
        If unsupported SciPy option names or nonsliceable peak counts are supplied.

    Examples
    --------
    >>> find_npeaks(np.array([0., 2., 0., 3., 0.]), 1).tolist()
    [3]

    Notes
    -----
    Evaluation is eager and deterministic and does not mutate input samples.
    Plotting creates and displays Matplotlib figures only when requested.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    # Compatible colormap getter for different matplotlib versions
    if find_peaks_params is None:
        find_peaks_params = {}
    try:
        from matplotlib import colormaps

        def get_colormap_colors(cmap_name, n):
            cmap = colormaps.get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    except ImportError:
        from matplotlib.cm import get_cmap

        def get_colormap_colors(cmap_name, n):
            cmap = get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    spec = np.abs(np.squeeze(values))

    # Finding peaks
    peaks, properties = find_peaks(spec, **find_peaks_params)

    # Extract the heights of these peaks
    peak_heights = np.squeeze(spec)[peaks]

    # Sort peaks by height in descending order and get the top n peaks
    top_peaks_indices = peaks[np.argsort(-peak_heights)][:npeaks]

    if sort_order == "descending":
        top_peaks_indices = np.sort(top_peaks_indices)[
            ::-1
        ]  # This sorts the array and then reverses it
    elif sort_order == "ascending":
        top_peaks_indices = np.sort(top_peaks_indices)  # This sorts the array
    else:
        pass

    if plot:
        if freq_range is None:
            freq_range = np.linspace(0, spec.shape[0] - 1, spec.shape[0])

        fig, ax = plt.subplots(1)
        ax.plot(freq_range, np.squeeze(np.abs(spec)), label="spec")

        # Get colors from colormap
        colors = get_colormap_colors("jet", len(top_peaks_indices))

        for k, i in enumerate(top_peaks_indices):
            color = colors[k]
            ax.vlines(freq_range[i], 0, spec[i], colors=color, label=f"Peak {k}")

        plt.legend()
        # Show plot (optional, depending on your environment)
        plt.show()

    return top_peaks_indices


def find_npeaks_nd(
    data: np.ndarray | xr.DataArray = None,
    npeaks: int = None,
    freq_range: np.ndarray = None,
    plot: bool = False,
    find_peaks_params: dict = None,
    sort_order: str = None,
):
    """Select local magnitude peaks independently along a sample dimension.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray, optional
        Real or complex spectral samples. Labelled data use acquisition_idx when
        present; otherwise the last dimension is used. Other dimension order is
        retained. Arrays are materialized eagerly without modifying their values.
    npeaks : int, optional
        Positive count of peak slots per spectrum; required for calculation.
    freq_range : numpy.ndarray, optional
        Matching plot positions in caller-defined units. Available dimension
        coordinates are used when no explicit positions are supplied.
    plot : bool, optional
        Display the first spectrum and the detected peaks in a Matplotlib figure.
    find_peaks_params : dict, optional
        Explicit SciPy find_peaks options; distance is measured in samples.
    sort_order : str, optional
        Ascending or descending sample-index order. None and unrecognized strings
        retain decreasing magnitude order.

    Returns
    -------
    numpy.ndarray or None
        Integer indices, with the selected axis replaced by an axis of length
        npeaks in its original position. Missing peaks are -1. Labels and attrs
        are not retained. Invalid inputs or handled calculation failures print
        diagnostics and return None; plotting failures do not discard results.

    Notes
    -----
    The routine prints progress and legacy error diagnostics. Results are
    uncached and deterministic; acquired samples and metadata are not mutated.
    Nonfinite magnitudes are replaced by zero only for peak detection.

    Examples
    --------
    >>> peaks = find_npeaks_nd(np.array([[0., 2., 0.]]), 1)  # doctest: +SKIP
    >>> peaks.shape  # doctest: +SKIP
    (1, 1)
    """
    if find_peaks_params is None:
        find_peaks_params = {}
    try:
        from matplotlib import colormaps

        def get_colormap_colors(cmap_name, n):
            cmap = colormaps.get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    except ImportError:
        from matplotlib.cm import get_cmap

        def get_colormap_colors(cmap_name, n):
            cmap = get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    from scipy.signal import find_peaks

    if data is None or npeaks is None:
        print("Error: data and npeaks must be provided.")
        return None
    if not isinstance(npeaks, int) or npeaks <= 0:
        print("Error: npeaks must be a positive integer.")
        return None

    data_np = None
    target_axis = -1  # Default to LAST axis
    target_dim_name = None
    original_shape = None
    coords = None
    is_xarray_input = False

    # --- Input Type and Dimension Detection ---
    if isinstance(data, xr.DataArray):
        is_xarray_input = True
        original_shape = data.shape
        print("Input is xarray.DataArray. Checking for 'acquisition_idx' dimension.")
        if "acquisition_idx" in data.dims:
            target_dim_name = "acquisition_idx"
            try:
                # Use get_axis_num for robust axis index retrieval
                target_axis = data.get_axis_num(target_dim_name)
                print(f"Found '{target_dim_name}' at axis {target_axis}.")
                # Try to get coordinates automatically if freq_range not
                # provided
                if freq_range is None and target_dim_name in data.coords:
                    coords = data.coords[target_dim_name].values
                    print(
                        f"Using coordinates from '{target_dim_name}' for frequency axis."
                    )
            except (
                Exception
            ) as e:  # Catch potential errors (e.g., dim exists but not in main data)
                print(
                    f"Warning: Dimension '{target_dim_name}' found but failed to get axis index ({e}). Defaulting to last axis."
                )
                target_axis = -1  # Fallback to last axis
        else:
            print("Dimension 'acquisition_idx' not found. Defaulting to last axis.")
            target_axis = -1

        # If defaulted or didn't find coords initially, try getting coords from
        # the chosen axis
        if coords is None:
            # Ensure target_axis is valid before accessing dims
            if target_axis >= len(data.dims) or target_axis < -len(data.dims):
                print(
                    f"Error: Fallback axis {target_axis} is invalid for dims {data.dims}. Cannot proceed."
                )
                return None
            target_dim_name = data.dims[
                target_axis
            ]  # Get name of the axis we ended up with
            if freq_range is None and target_dim_name in data.coords:
                coords = data.coords[target_dim_name].values
                print(
                    f"Using coordinates from dimension '{target_dim_name}' (axis {target_axis}) for frequency axis."
                )

        data_np = data.values  # Work with numpy array underneath
    else:
        # Assume numpy array or array-like
        try:
            data_np = np.asarray(data)  # Convert input to numpy array
            original_shape = data_np.shape
            target_axis = -1  # Default to last axis for numpy
            if is_xarray_input:  # It was xarray but 'acquisition_idx' wasn't found
                print("Defaulting to last axis for peak finding.")
            else:  # Was not xarray input
                print(
                    "Input is NumPy array or array-like. Defaulting to last axis for peak finding."
                )
        except Exception as e:
            print(f"Error: Could not convert data to NumPy array: {e}")
            return None

    # --- Validate Dimensions and Axis ---
    if data_np.ndim == 0:
        print("Error: Input data is scalar.")
        return None
    elif data_np.ndim == 1:
        # If input is 1D, ensure target_axis is 0
        target_axis = 0
        print("Input data is 1D. Finding peaks along the single axis (axis 0).")
    # Check axis validity after potential modification for 1D case
    if target_axis < -data_np.ndim or target_axis >= data_np.ndim:
        print(
            f"Error: Determined target_axis {target_axis} is invalid for data with {data_np.ndim} dimensions."
        )
        return None
    # Ensure target_axis is positive for apply_along_axis internal use and
    # consistent indexing
    target_axis_positive = target_axis % data_np.ndim

    # --- Helper function to find peaks in 1D ---
    def _find_peaks_1d_slice(spec_1d, npeaks, find_peaks_params, sort_order):
        """Internal function: returns integer indices, uses -1 for padding."""
        placeholder = -1  # Use -1 instead of NaN

        if spec_1d is None or spec_1d.size == 0 or np.all(np.isnan(spec_1d)):
            # Return array of placeholders with integer type
            return np.full(npeaks, placeholder, dtype=int)

        spec = np.abs(spec_1d)
        finite_mask = np.isfinite(spec)
        if not np.any(finite_mask):
            return np.full(npeaks, placeholder, dtype=int)
        spec_finite = np.where(finite_mask, spec, 0)

        try:
            peaks, properties = find_peaks(spec_finite, **find_peaks_params)
        except ValueError as e:
            print(
                f"Warning: find_peaks failed for a slice: {e}. Returning placeholders."
            )
            return np.full(npeaks, placeholder, dtype=int)

        if peaks.size == 0:
            # No peaks found, return placeholders
            return np.full(npeaks, placeholder, dtype=int)

        peak_heights = spec[peaks]
        sorted_peak_indices_by_height = np.argsort(-peak_heights)

        num_peaks_found = len(sorted_peak_indices_by_height)
        num_peaks_to_return = min(npeaks, num_peaks_found)

        top_peaks_indices = peaks[sorted_peak_indices_by_height[:num_peaks_to_return]]

        # Apply sorting by index value if requested
        if sort_order == "descending":
            top_peaks_indices = np.sort(top_peaks_indices)[::-1]
        elif sort_order == "ascending":
            top_peaks_indices = np.sort(top_peaks_indices)

        # Pad with placeholder (-1) if fewer than npeaks were found/selected
        # Ensure output array has integer type
        padded_indices = np.full(
            npeaks, placeholder, dtype=int
        )  # Initialize with placeholder
        # Assign the valid integer indices
        padded_indices[:num_peaks_to_return] = top_peaks_indices  # Already int

        return padded_indices

    # --- Apply the peak finding function along the target axis ---
    target_dim_size = data_np.shape[target_axis_positive]
    print(
        f"Applying peak finding along axis {target_axis_positive}(dimension size {target_dim_size}) across input shape {data_np.shape}..."
    )
    try:
        # Suppress potential RuntimeWarnings from NaN slices if desired
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            all_peak_indices = np.apply_along_axis(
                _find_peaks_1d_slice,
                axis=target_axis_positive,
                arr=data_np,
                # Pass additional arguments to the helper function:
                npeaks=npeaks,
                find_peaks_params=find_peaks_params,
                sort_order=sort_order,
            )
    except Exception as e:
        print(f"\nError during np.apply_along_axis: {e}")
        import traceback

        traceback.print_exc()
        return None

    print(f"Peak finding complete. Result shape: {all_peak_indices.shape}")

    # --- Plotting (optional, for the first slice) ---
    if plot:
        print("Plotting results for the first slice...")
        try:
            # Determine the frequency axis data for plotting
            plot_freq_axis = None
            # Default label
            axis_label = f"Index (Axis {target_axis_positive})"
            if coords is not None:
                # Ensure coords match the dimension length
                if len(coords) == target_dim_size:
                    plot_freq_axis = coords
                    axis_label = (
                        target_dim_name
                        if target_dim_name
                        else f"Axis {target_axis_positive} Coords"
                    )
                    print(f"Using coordinates for plot x-axis ({axis_label}).")
                else:
                    print(
                        f"Warning: Auto - detected coordinates length({len(coords)}) does not match target axis length({target_dim_size}). Plotting against indices."
                    )
                    plot_freq_axis = np.arange(target_dim_size)
            elif freq_range is not None:
                # Ensure provided freq_range matches the dimension length
                if len(freq_range) == target_dim_size:
                    plot_freq_axis = np.asarray(
                        freq_range
                    )  # Ensure numpy array for indexing
                    axis_label = "Provided freq_range"
                    print("Using provided freq_range for plot x-axis.")
                else:
                    print(
                        f"Warning: Provided freq_range length({len(freq_range)}) does not match target axis length({target_dim_size}). Plotting against indices."
                    )
                    plot_freq_axis = np.arange(target_dim_size)
            else:
                # Use indices if no frequency info available
                plot_freq_axis = np.arange(target_dim_size)
                print(
                    "Warning: No freq_range provided or found in coordinates. Plotting against indices."
                )

            # Prepare indices to select the first slice's data and peak results
            slice_spec_indices = [0] * data_np.ndim
            slice_peaks_indices = []
            slice_desc = []  # Description for plot title

            if data_np.ndim == 1:
                # Simple case: data is already 1D
                spec_to_plot = np.abs(data_np)
                # Result is already 1D (npeaks,)
                peaks_to_plot_indices_float = all_peak_indices
                slice_desc.append("1D Data")
            else:
                # ND case: Select the first element along all non-target axes
                current_peak_array_dim = 0  # Tracks position in the result array's dims
                for i in range(data_np.ndim):
                    if i == target_axis_positive:
                        slice_spec_indices[i] = slice(
                            None
                        )  # Get all along target axis for spectrum plot
                    else:
                        # Index 0 for the spectrum slice
                        slice_spec_indices[i] = 0
                        # Index 0 for the results slice (corresponding dim)
                        slice_peaks_indices.append(0)
                        # Get dimension name for description
                        dim_name = "N/A"
                        if is_xarray_input:
                            try:
                                dim_name = data.dims[i]
                            except IndexError:
                                pass
                        elif original_shape:  # Use original shape info if available
                            dim_name = f"Dim{i}"

                        slice_desc.append(f"{dim_name}=0")
                        current_peak_array_dim += 1  # Move to next dim in result array

                # Finally, select all npeaks along the last dimension of the
                # result array
                slice_peaks_indices.append(slice(None))

                slice_spec_indices_tuple = tuple(slice_spec_indices)
                slice_peaks_indices_tuple = tuple(slice_peaks_indices)

                # Get the spectrum data for the selected slice
                spec_to_plot = np.abs(data_np[slice_spec_indices_tuple])
                # Get the peak indices for the selected slice
                peaks_to_plot_indices_float = all_peak_indices[
                    slice_peaks_indices_tuple
                ]

            # Filter out NaNs from peak indices and convert to integer for
            # indexing
            valid_peaks_mask = ~np.isnan(peaks_to_plot_indices_float)
            peaks_to_plot_indices = peaks_to_plot_indices_float[
                valid_peaks_mask
            ].astype(int)
            num_peaks_plotted = len(peaks_to_plot_indices)

            # Create the plot
            fig, ax = plt.subplots(1, figsize=(10, 5))  # Wider plot
            plot_label = (
                f"Spectrum ({', '.join(slice_desc)})" if slice_desc else "Spectrum"
            )
            ax.plot(plot_freq_axis, spec_to_plot, label=plot_label)

            plot_title = f"Spectrum and Top {num_peaks_plotted}/{npeaks} Peaks"
            if slice_desc:
                plot_title += f" (Slice: {', '.join(slice_desc)})"

            if num_peaks_plotted > 0:
                # Get colors from colormap
                colors = get_colormap_colors("jet", num_peaks_plotted)
                # Get the heights of the peaks being plotted from the
                # *absolute* spectrum
                peak_heights_plot = spec_to_plot[peaks_to_plot_indices]

                for k, peak_idx in enumerate(peaks_to_plot_indices):
                    # Ensure index is within bounds (should be, but safety
                    # check)
                    if 0 <= peak_idx < len(plot_freq_axis):
                        color = colors[k]
                        freq_val = plot_freq_axis[peak_idx]
                        # Use .item() if freq_val might be 0-d array (e.g.,
                        # single point coord)
                        freq_val_item = (
                            freq_val.item()
                            if isinstance(freq_val, np.ndarray)
                            else freq_val
                        )

                        ax.vlines(
                            freq_val_item,
                            0,
                            peak_heights_plot[k],
                            colors=color,
                            linestyles="--",
                            label=f"Peak @ Idx {peak_idx}",
                        )
                        # Optional: Add text label for index near the peak top
                        ax.text(
                            freq_val_item,
                            peak_heights_plot[k],
                            f" {peak_idx}",
                            color=color,
                            va="bottom",
                            ha="center",
                            fontsize=8,
                            rotation=90,
                        )
                    else:
                        print(
                            f"Warning: Peak index {peak_idx} out of bounds for frequency axis, skipping plot marker."
                        )

                # Improve legend handling for many peaks
                handles, labels = ax.get_legend_handles_labels()
                # Keep only the first label ('Spectrum') and maybe one 'Peak'
                # label summary
                unique_labels = {}
                new_handles = []
                new_labels = []
                peak_label_added = False
                for handle, label in zip(handles, labels, strict=False):
                    is_peak = "Peak @" in label
                    if not is_peak:  # Keep non-peak labels (like 'Spectrum')
                        if label not in unique_labels:
                            unique_labels[label] = handle
                            new_handles.append(handle)
                            new_labels.append(label)
                    elif not peak_label_added:  # Add only one summary label for peaks
                        unique_labels["Peak"] = handle  # Use a generic handle
                        new_handles.append(
                            plt.Line2D([0], [0], linestyle="--", color="gray")
                        )  # Generic peak line
                        new_labels.append(f"Detected Peaks ({num_peaks_plotted})")
                        peak_label_added = True

                ax.legend(new_handles, new_labels)

            else:
                ax.legend()  # Show legend even if only spectrum is plotted (no peaks found)

            ax.set_title(plot_title)
            ax.set_xlabel(axis_label)
            ax.set_ylabel("Absolute Intensity")
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            # Catch potential errors during slicing or plotting
            print(f"\nError during plotting: {e}")
            import traceback

            traceback.print_exc()
            print("Please check data dimensions, freq_range, and matplotlib setup.")

    return all_peak_indices
