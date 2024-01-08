"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for robust control design problem, i.e., w.r.t. the worst
case realization of the noise within its support.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from utils.data_structures import Polytope, LinearSystem
from achievability import achievability_constraints


def synthesize_robust(sys: LinearSystem, t_fir: int, feasible_set: Polytope,
                      support: Polytope, verbose=False):
    """
    This function generates the closure that defines the robust
    control design problem. The closure can be used afterward for
    various different optimization problems over the SLS closed loop map phi.

    This closure solves the following optimization problem:
        min_{phi} E_{xi~P_empirical} (xi^T Phi^T W Phi xi)
        s.t. max_{xi_j} G_j Phi xi_j ≤ g_j, for all xi_j in support, for all j
    where W is the weight of the control cost [x_t, u_t]^T W [x_t, u_t] and
    G_j, g_j are the j-th row of the feasible set {x| Gx ≤ g}.

    The problem is solved using the duality of the maximization problem,
    which gives the following equivalent constraint:
        min_{mu_j} h^T mu_j   s.t. -(G_j Phi)^T + H^T mu_j = 0, mu_j ≥ 0

    This gives the following equivalent optimization problem:
        min_{phi} E_{xi~P_empirical} (xi^T Phi^T W Phi xi)
        s.t. mu h ≤ g, G Phi = mu H, mu ≥ 0

    :param sys: LinearSystem for which the constraints apply.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param feasible_set: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param verbose: bool, if True, prints the optimization verbose.
    :return: closure with signature (xis, weights) -> phi, where phi is the SLS
        closed loop map, xis are the samples of the empirical distribution
        (one column per sample), and weights is the matrix square root of the
        weight W.
    """
    # No argument checks, they are performed in daughter functions

    # Short notations
    _t = t_fir
    _n = sys.a.shape[0]
    _m = 0 if sys.b is None else sys.b.shape[1]
    _p = 0 if sys.c is None else sys.c.shape[0]

    # Optimization variables
    phi = cp.Variable((_n + _m, (_n + _p) * _t))
    mu = cp.Variable((feasible_set.h.shape[0], support.h.shape[0]))

    # Get achievability constraints
    cons = achievability_constraints(sys, t_fir)(phi)

    # Make robust constraints
    cons += [mu @ support.g <= feasible_set.g,
             feasible_set.h @ phi == mu @ support.h,
             mu >= 0]

    def mkrob(xis, weights=None):
        weights = np.eye(_n + _m) if weights is None else weights

        # CVX is annoying and needs some slack variables for DCP check
        phi_xi = cp.Variable((_n + _m, xis.shape[1]))
        c_xi = [phi_xi == phi @ xis]

        # Quadratic cost function
        cost = cp.sum([pxi.T @ weights @ pxi
                       for pxi in phi_xi.T]) / xis.shape[1]

        # Solve the optimization problem
        cp.Problem(cp.Minimize(cost), cons + c_xi).solve(verbose=verbose)

        return phi.value

    return mkrob



