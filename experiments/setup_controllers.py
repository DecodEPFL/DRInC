"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from drinc import synthesize_drinc
from benchmarks.robust import synthesize_robust
from benchmarks.empirical import synthesize_empirical
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution

# List of controller names, useful to iterate over all controllers
controller_names = ["DRInC", "Robust", "Empirical", "LQG", "DR-LQG"]


def get_controllers(t_fir: int, radius: float, p_level: float,
                    sys: LinearSystem, fset: Polytope, support: Polytope,
                    verbose=False):
    """
    Sets the DRInC, robust, empirical, LQG, and DR-LQG controllers.
    returns closures with signature (xis, weights) -> phi, where phi is the SLS
    closed loop map, xis are the samples of the empirical distribution that
    the controllers are optimized for (one column per sample), and weights
    is the matrix square root of the weights W of the control cost
    [x_t, u_t]^T W [x_t, u_t].

    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :param sys: LinearSystem to control.
    :param fset: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param verbose: bool, if True, prints the optimization verbose.
    :return: list of closures for each controller in controller_names.
    """
    # Obtain drinc closure
    drinc = synthesize_drinc(sys, t_fir, fset, support,
                             radius, p_level, None, None, verbose)

    # Obtain empirical closure, just drinc with very small Wasserstein ball
    emp = synthesize_empirical(sys, t_fir, fset, verbose)

    # Obtain robust closure
    rob = synthesize_robust(sys, t_fir, fset, support, verbose)

    # Gaussian distribution is an alternative center for the Wasserstein ball
    # We approximate it with an empirical distribution with Gaussian samples
    gauss = get_distribution("gaussian")

    # Make lqg closure, the gaussian has the same variance as the samples
    def lqg(xis, weights=None):
        rn = np.hstack([gauss(xis.shape[0]) for i in range(xis.shape[1])])
        return emp(np.std(xis.flatten()) * rn, weights)

    # Make DR-LQG closure, the gaussian has the same variance as the samples
    # This is an approximation as the center is empirical, not exactly Gaussian
    def drlqg(xis, weights=None):
        rn = np.hstack([gauss(xis.shape[0]) for i in range(xis.shape[1])])
        return drinc(np.std(xis.flatten()) * rn, weights)

    return drinc, rob, emp, lqg, drlqg


