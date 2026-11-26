"""Conversion of Bruker real/imaginary component arrays."""


def bruker2complex(arr):
    """Combine Bruker real and imaginary components into complex samples.

    Parameters
    ----------
    arr : numpy.ndarray
        Real component array with final axis of length two: real then imaginary.
        Other dimensions and signal units are arbitrary and preserved.

    Returns
    -------
    numpy.ndarray
        Complex array with the final component axis removed. NumPy determines
        precision from the input dtype. Evaluation is eager and deterministic;
        the acquired input is not mutated.

    Raises
    ------
    AssertionError
        If the final dimension does not have exactly two components.
    """
    assert arr.shape[-1] == 2, "Last dimension must be complex dimension."
    return arr[..., 0] + 1j * arr[..., 1]
