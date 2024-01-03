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


def synthesize_drinc(sys: LinearSystem, t_fir: int, feasible_set: Polytope,
                     support: Polytope, radius: float, p_level: float):
    """
    This function generates the closure that defines the distributionally robust
    control design problem from "Distributionally Robust Infinite-horizon Control"
    (DRInC) by JS Brouillon et. al., 2023. The closure can be used afterward for
    various different optimization problems over the SLS closed loop map phi.

    :param sys: LinearSystem for which the constraints apply.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param feasible_set: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :return: closure with signature (phi, xis) -> cons, where phi is the SLS
        closed loop map, xis are the samples of the empirical distribution at
         the center of the Wasserstein ball (one column per sample), and cons
         is a list of linear matrix inequality constraints.
    """

    # No argument checks, they are performed in daughter functions

    # Short notations
    _t = t_fir
    _n = sys.a.shape[0]
    _m = 0 if sys.b is None else sys.b.shape[1]
    _p = 0 if sys.c is None else sys.c.shape[0]

    # Get achievability, cvar and cost closures
    mkach = achievability_constraints(sys, t_fir)
    mkcvar = cvar_constraints(feasible_set, support, radius, p_level)
    mkcost, mkcons = drinc_cost(support, radius)

    def mkdrinc(xis):
        phi = cp.Variable((_n + _m, (_n + _p) * _t))
        return phi.value

    return mkdrinc
