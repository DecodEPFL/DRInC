"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution

savepath = "results/double_integrator.npz"


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
    t_fir, t_test = 6, 40
    # Feasible set size, cvar probability level, and noise level
    feas_r, p_level, noise = 60.0, 5e-2, 1.0  # 70
    # Number of samples. The list contains parameters for distributions.
    # Their values are explained in utils/distributions.py
    _ptrain, _ptest = (25, [1.0, 0.95]), (100, [1.0, 1.05])

    # System definition
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.array([[1, 1], [0, 1]]), \
        np.array([[0], [1]]), np.array([[1, 0]])

    # Support definition as a box [-0.4*noise, 1.2*noise]^d
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = noise * np.array(([1.2] * (_n + _p) * t_fir)
                                 + ([0.4] * (_n + _p) * t_fir))[:, None]

    # Feasible set definition
    fset = Polytope()
    fset.h = np.vstack((np.diag([10, 1, 0]), -np.diag([10, 1, 0])))
    fset.g = feas_r * np.ones((2 * (_n + _m), 1))

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    for _ps, (_xis, _t) in zip([_ptrain, _ptest],
                               zip([xis_train, xis_test], [t_fir, t_test])):
        (_ns, p) = _ps
        for n in ['bimodal_gaussian', 'beta', 'step']:
            d = get_distribution(n, p)
            _xis[n] = np.hstack([np.vstack([d(_n * _t), d(_p * _t)])
                                 for i in range(_ns)]) * noise

    return t_test, t_fir, radius, p_level, sys, \
        fset, support, xis_train, xis_test


