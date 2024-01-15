"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for empirical control design problem, i.e., s.t. all the
samples satisfies the constraints, and where the cost is the empirical mean.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from utils.data_structures import Polytope, LinearSystem
from achievability import achievability_constraints


def synthesize_auglqg(sys: LinearSystem, t_fir: int, feasible_set: Polytope,
                         verbose=False):
    """
    This function generates the closure that defines the empirical
    control design problem. The closure can be used afterward for
    various different optimization problems over the SLS closed loop map phi.

    This closure solves the following optimization problem:
        min_{phi} E_{xi~P_empirical} (xi^T Phi^T W Phi xi)
        s.t. G Phi xi_j ≤ g, for all samples xi_j in P_empirical

    :param sys: LinearSystem for which the constraints apply.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param feasible_set: Polytope, the feasible set of the optimization problem.
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

    # Get achievability constraints
    cons = achievability_constraints(sys, t_fir)(phi)

    def mkaug(xis, weights=None):
        weights = np.eye(_n + _m) if weights is None else weights
        evalues, evectors = np.linalg.eig(np.cov(xis))
        cov_sqrt = evectors * np.sqrt(evalues) @ evectors.T

        # Add constraints for each sample
        c_xi = [feasible_set.h @ phi @ xi[:, None] <= feasible_set.g
                for xi in xis.T]

        # Quadratic cost function
        cost = cp.norm(weights @ phi @ cov_sqrt, 'fro') ** 2

        # Solve the optimization problem
        cp.Problem(cp.Minimize(cost), cons + c_xi).solve(verbose=verbose)

        return phi.value

    return mkaug



