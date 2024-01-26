"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Computing the Wasserstein distance between two distributions is #P-hard.
To obtain a fast approximation, we restrict the problem to empirical
distributions with numbers of samples being multiples of one another.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np


def wasserstein(xi1, xi2):
    """
    Approximates the wasserstein distance between two empirical distributions.
    To simplify, we assume that the number of samples of xi1 is a multiple of
    the number of samples of xi2. If this is not the case, the function will
    delete some samples of xi1 and raise a warning.
    If xi1 has a smaller number of samples than xi2, they will be swapped.

    :param xi1: First empirical distribution. Columns are samples.
    :param xi2: Second empirical distribution. Columns are samples.
    :return: Approximation of the Wasserstein distance between xi1 and xi2.
    """
    # Check that xi1 and xi2 are compatible and fix them otherwise
    if xi1.shape[0] != xi2.shape[0]:
        raise ValueError("The number of rows of xi1 and xi2 must be equal.")

    if xi1.shape[1] < xi2.shape[1]:
        xi1, xi2 = xi2, xi1
    if xi1.shape[1] % xi2.shape[1] != 0:
        print("Warning: xi1 has a number of samples that is not a multiple of "
              "xi2. Some samples will be deleted from xi1.")
        xi1 = xi1[:, :-(xi1.shape[1] % xi2.shape[1])]

    # Short notations
    _n1 = xi1.shape[1]
    _n2 = xi2.shape[1]

    # transport cost
    _c = lambda x, y: np.linalg.norm(x - y, axis=0) ** 2

    # Compute the distances and indices of xi2
    ds = np.array([_c(xi1, xi[:, None]) for xi in xi2.T]).flatten()
    ids = (np.arange(_n2)[:, None] * np.ones((_n2, _n1), dtype=int)).flatten()

    # Sort the all the aggregated distances
    idx = np.argsort(ds)
    ds, ids = ds[idx], ids[idx]

    d = 0
    # take the _n1/_n2 smallest distances for each element in xi2
    for i in range(_n2):
        d += np.sum((ds[ids == i])[:int(_n1/_n2)])

    return d / _n1

