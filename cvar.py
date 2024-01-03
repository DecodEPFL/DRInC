"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for distributionally robust cvar constraints as defined in
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from data_structures import Polytope


def cvar_constraints(feasible_set: Polytope, support: Polytope,
                     radius: float, p_level: float):
    """
    This function generates the closure that defines the distributionally robust
    conditional value-at-risk constraints for a given feasible set and noise
    distribution support. The closure can be used afterward for various
    different optimization problems over the SLS closed loop map phi.

    :param feasible_set: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :return: closure with signature (phi, xis) -> cons, where phi is the SLS
        closed loop map, xis are the samples of the empirical distribution at
         the center of the Wasserstein ball (one column per sample), and cons
         is a list of linear matrix inequality constraints.
    """

    # Check that the radius is positive
    if radius <= 0:
        raise ValueError("The radius must be positive.")

    # Check that the probability level is in [0, 1]
    if p_level < 0 or p_level > 1:
        raise ValueError("The probability level must be in [0, 1].")

    # Short notations
    _H = support.h
    _h = support.g if len(support.g.shape) == 2 else support.g[:, None]
    _G = np.vstack((feasible_set.h, 0*feasible_set.h[[0], :]))
    _j = _G.shape[0]
    _y = p_level

    def mkcons(phi, xis):
        # Optimization variables
        tau = cp.Variable((1, 1))
        rho = cp.Variable()
        zeta = cp.Variable((xis.shape[1], 1))

        # Short notations
        _n = xis.shape[1]  # Number of samples
        _g = cp.vstack((feasible_set.g, -tau))

        # One-of Constraints
        cons = [rho >= 0,
                rho*radius + (_y - 1)/_y * tau + 1/_n * cp.sum(zeta) <= 0]

        # Wasserstein ball constraints
        for i, xii in enumerate(xis.T):
            k_i = cp.Variable((_j, _h.shape[0]))
            cons += [k_i >= 0]

            for j, _gj in enumerate(_G):
                _s = zeta[i] - (_gj @ phi @ xii - _g[j])/_y \
                    - (_H @ xii[:, None] + _h).T @ k_i[j, :]

                _od = phi.T @ _gj[:, None] / _y - _H.T @ k_i[[j], :].T

                _m = cp.bmat([[_s[:, None], _od.T],
                              [_od, 4*rho*_y*_y*np.eye(phi.shape[1])]])
                cons += [_m >> 0]

        return cons

    return mkcons
