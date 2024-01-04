"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from drinc import synthesize_drinc
from robust import synthesize_robust
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution
from utils.simulate import simulate


def get_controllers(sys: LinearSystem, t_fir: int, fset: Polytope,
                    support: Polytope, radius: float, p_level: float,
                    verbose=False):
    """
    Sets the drinc, robust, empirical and h2 controllers.
    returns closures with signature (xis, weights) -> phi, where phi is the SLS
    closed loop map, xis are the samples of the empirical distribution that
    the controllers are optimized for (one column per sample), and weights
    is the matrix square root of the weights W of the control cost
    [x_t, u_t]^T W [x_t, u_t].

    :param sys: LinearSystem to control.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param fset: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :param verbose: bool, if True, prints the optimization verbose.
    """
    # Obtain drinc closure
    drinc = synthesize_drinc(sys, t_fir, fset, support,
                             radius, p_level, verbose)

    # Obtain robust closure
    rob = synthesize_robust(sys, t_fir, fset, support, verbose)

    # Obtain empirical closure, just drinc with very small Wasserstein ball
    emp = synthesize_drinc(sys, t_fir, fset, support, 1e-6, 1e-3, verbose)

    # Make h2 closure, the gaussian has the same variance as the samples
    gauss = get_distribution("gaussian")

    def h2(xis, weights=None):
        return emp(np.std(xis.flatten()) * gauss(xis.shape[1]), weights)

    return drinc, rob, emp, h2


