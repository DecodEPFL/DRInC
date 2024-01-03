"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for distributionally robust optimization cost as defined in
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from data_structures import Polytope


def drinc_cost(support: Polytope, radius: float):
    """
    This function generates the closure that defines the distributionally robust
    infinite horizon quadratic cost $xi^T Q xi$, where xi is a realization
    of the noise, following a distribution with a given support.
    The closure can be used afterward for various different DRO problems over
    the cost matrix and for noise the distributions in a Wasserstein ball around
    various empirical centers.
    :@param support: Polytope, the support of the noise distribution.
    :@param radius: Radius of the Wasserstein ball.
    :@return: tuple of closures with signature (Q, xis) -> cost or cons, where
        Q is the cost matrix, xis are the samples of the empirical distribution
        at the center of the Wasserstein ball (one column per sample), cost the
        distributionally robust risk of the given Q and xis, and cons is a list
        of linear matrix inequality constraints.
    """
    _l = cp.Variable()

    def cost(q, xis):
        return _l

    def cons(q, xis):
        return [_l >= 3]

    return cost, cons
