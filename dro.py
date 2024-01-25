"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for distributionally robust optimization cost as defined in
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from utils.data_structures import Polytope


def drinc_cost(support: Polytope, radius: float):
    """
    This function generates the closure that defines the distributionally robust
    infinite horizon quadratic cost $xi^T Q xi$, where xi is a realization
    of the noise, following a distribution with a given support.
    The closure can be used afterward for various different DRO problems over
    the cost matrix and for noise the distributions in a Wasserstein ball around
    various empirical centers.
    :param support: Polytope, the support of the noise distribution.
    :param radius: Radius of the Wasserstein ball.
    :return: tuple of closures with signature (Q, xis) -> cost or cons, where
        Q is the cost matrix, xis are the samples of the empirical distribution
        at the center of the Wasserstein ball (one column per sample), cost the
        distributionally robust risk of the given Q and xis, and cons is a list
        of linear matrix inequality constraints.
    """

    # Check that the radius is positive
    if radius <= 0:
        raise ValueError("The radius must be positive.")

    # Short notations
    _H = support.h
    _h = support.g if len(support.g.shape) == 2 else support.g[:, None]
    _l = cp.Variable()
    mean_si = cp.Variable()

    def mkcost(q, xis):
        return _l * radius + mean_si

    def mkcons(q, xis):
        _n = xis.shape[1]  # Number of samples

        # Optimization variables
        _a = cp.Variable((1, 1))
        _s = cp.Variable((_n, 1))
        _mu = cp.Variable((_H.shape[0], _n))
        _psi = cp.Variable((_H.shape[0], _n))

        # Equality constraints
        cons = [mean_si == cp.sum(_s) / _n]

        # Inequality constraints
        cons += [_l >= 0, _mu >= 0, _psi >= _mu, _a >= 0]

        # Matrix inequality constraints
        _q2 = cp.kron(np.diag([1, 0]), 4 * np.eye(q.shape[0]) * _l) \
            + cp.kron(np.diag([-1, 1]), 4 * q)
        for i, (xii, (mui, psii)) in enumerate(zip(xis.T, zip(_mu.T, _psi.T))):
            # Element [1,1]
            _scalar = _s[[i]] - _h.T @ psii + _l * xii.T @ xii
            # Elements [2:end, 1]
            _vec = cp.hstack([_l * xii * 2 - _H.T @ psii, _H.T @ mui])[:, None]

            # Main LMI
            cons += [cp.bmat([[_scalar, _vec.T], [_vec, _q2]]) >> 0]

            # Orthogonality constraint
            _h_mui = _H.T @ mui[:, None]
            cons += [cp.bmat([[_a, _h_mui.T],
                              [_h_mui, np.eye(q.shape[0]) * _l - q]]) >> 0]

        return cons

    return mkcost, mkcons
