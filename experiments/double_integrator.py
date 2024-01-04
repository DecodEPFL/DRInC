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
from utils.distributions import get_distribution, implemented
from utils.simulate import simulate


def double_integrator_experiment(radius=0.1, verbose=False):
    """
    This function runs the experiment for the double integrator system.
    It returns the system, the support, the feasible set, the training and
    testing samples.

    :param radius: Radius of the Wasserstein ball
    :param verbose: bool, if True, prints the optimization verbose.
    :return: LinearSystem, Polytope, Polytope, dict, dict. The first three are
        the system, the support and the feasible set. The last two are the
        training and testing samples, in the form of a dictionary with the
        distribution names for keys.
    """
    # Parameters
    _n, _m, _p = 2, 1, 1
    t_fir, noise = 4, 0.2
    fradius, p_level = 10.0, 5e-2
    _ntrain, _ntest = 10, 100

    # System definition
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.array([[1, 1], [0, 1]]), \
        np.array([[0], [1]]), np.array([[1, 0]])

    # Support definition
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = np.random.uniform(0, noise, (2 * (_n + _p) * t_fir, 1))

    # Feasible set definition
    fset = Polytope()
    fset.h = np.vstack((np.eye(_n + _m),
                        -np.eye(_n + _m)))
    fset.g = np.random.uniform(t_fir*noise, fradius, (2 * (_n + _m), 1))

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    for n in implemented:
        pass

    return sys, support, fset, xis_train, xis_test


