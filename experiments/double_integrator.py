"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution, get_random_int, implemented


def double_integrator_experiment(radius=0.1, verbose=False):
    """
    This function runs the experiment for the double integrator system.
    It returns the system, the support, the feasible set, the training and
    testing samples.

    :param radius: Radius of the Wasserstein ball
    :param verbose: bool, if True, prints the optimization verbose.
    :return: LinearSystem, Polytope, Polytope, dict, dict. The first three are
        the parameters t_fir, radius, and p_level. The next three are
        the system, the support and the feasible set. The last two are the
        training and testing samples, in the form of a dictionary with the
        distribution names for keys.
    """
    # Useful to generate random Polytopes
    uni = get_distribution("uniform")

    # System dimensions
    _m, _n, _p = 1, 2, 1
    # Time horizons, problem ill conditionned if t_fir < 5
    t_fir, t_test = 7, 40
    # Feasible set size, cvar probability level, and noise level
    feas_r, p_level, noise = 10.0, 5e-2, 0.2
    # Number of samples
    _ntrain, _ntest = 5, 100

    # System definition
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.array([[1, 1], [0, 1]]), \
        np.array([[0], [1]]), np.array([[1, 0]])

    # Support definition
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = noise * np.ones((2 * (_n + _p) * t_fir, 1)) * 10

    # Feasible set definition
    fset = Polytope()
    fset.h = np.vstack((np.eye(_n + _m), -np.eye(_n + _m)))
    fset.g = feas_r * np.ones((2 * (_n + _m), 1))

    # Define the distributions to experiment with
    ds = dict()
    for n in implemented:
        if n in ['constant', 'sine', 'sawtooth', 'triangle', 'step']:
            p = [0.1] if n != 'step' else [0, 5]
            ds[n] = (get_distribution(n, p), p)
        else:
            p = [0.2] if n == 'bimodal_gaussian' else []
            ds[n] = (get_distribution(n, p), p)

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    for _xis, _t in zip([xis_train, xis_test], [t_fir, t_test]):
        for n, (d, p) in ds.items():
            xi = []
            for i in range(_ntrain):
                p[1] = get_random_int(_p * _t)
                xi.append(np.vstack([d(_n * _t), d(_p * _t)]))
            xi = np.hstack(xi)
            _xis[n] = xi * noise

    return t_test, t_fir, radius, p_level, sys, \
        fset, support, xis_train, xis_test


