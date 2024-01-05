"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for distributionally robust control design problem from
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from utils.data_structures import Polytope, LinearSystem
from achievability import achievability_constraints
from cvar import cvar_constraints
from dro import drinc_cost
import mosek


def synthesize_drinc(sys: LinearSystem, t_fir: int, feasible_set: Polytope,
                     support: Polytope, radius: float, p_level: float,
                     radius_constraints: float = None, regular: float = None,
                     verbose=False):
    """
    This function generates the closure that defines the distributionally robust
    control design problem from "Distributionally Robust Infinite-horizon
    Control" (DRInC) by JS Brouillon et. al., 2023. The closure can be used
    afterward for various different optimization problems over the SLS closed
    loop map phi.

    :param sys: LinearSystem for which the constraints apply.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param feasible_set: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :param radius_constraints: Radius of the Wasserstein ball for the
        constraints, if None, the same radius as for the cost is used.
    :param regular: float > 0, regularization parameter for the optimization
        (multiplies tr(Q)).
    :param verbose: bool, if True, prints the optimization verbose.
    :return: closure with signature (xis, weights) -> phi, where phi is the SLS
        closed loop map, xis are the samples of the empirical distribution at
        the center of the Wasserstein ball (one column per sample), and weights
        is the matrix square root of the weights W of the control cost
        [x_t, u_t]^T W [x_t, u_t].
    """

    # No argument checks, they are performed in daughter functions
    regular = 1e-2 if regular is None else regular
    if radius_constraints is None:
        radius_constraints = radius

    # Short notations
    _t = t_fir
    _n = sys.a.shape[0]
    _m = 0 if sys.b is None else sys.b.shape[1]
    _p = 0 if sys.c is None else sys.c.shape[0]

    # Get achievability, cvar and cost closures
    mkach = achievability_constraints(sys, t_fir)
    mkcvar = cvar_constraints(feasible_set, support,
                              radius_constraints, p_level)
    mkcost, mkcons = drinc_cost(support, radius)

    def mkdrinc(xis, weights=None):
        # Variables
        weights = np.eye(_n + _m) if weights is None else weights
        phi = cp.Variable((_n + _m, (_n + _p) * _t))
        q = cp.Variable(((_n + _p) * _t, (_n + _p) * _t))

        # Generate the constraints
        cons = mkach(phi) + mkcons(q, xis) + mkcvar(phi, xis)

        # Add the link between Q and phi
        cons += [cp.bmat([[q, (weights @ phi).T],
                          [weights @ phi, np.eye(_n + _m)]]) >> 0,
                 q == q.T]

        # Solve the optimization problem
        cp.Problem(cp.Minimize(mkcost(q, xis) + regular*cp.trace(q)),
                   cons).solve(verbose=verbose)

        return phi.value

    return mkdrinc
