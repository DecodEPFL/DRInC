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


def double_integrator_experiment(radius=0.05, verbose=False, dist=1.0):
    """
    This function runs the experiment for the double integrator system.
    It returns the system, the support, the feasible set, the training and
    testing samples.

    :param radius: Radius of the Wasserstein ball. Note that the Wasserstein
        TYPE 2 metric is used, so the radius is the square of the type 1 radius.
    :param verbose: bool, if True, prints the optimization verbose. (optional)
    :param dist: float, second parameter of the testing distribution. (optional)
        See distributions.py
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
    # Time horizons, problem ill conditioned if t_fir < 5
    t_fir, t_test = 10, 50
    # Feasible set size, cvar probability level, and noise level
    feas_r, p_level, noise = 70.0, 5e-2, 1.0  # 70
    # Number of samples. The list contains parameters for distributions.
    # Their values are explained in utils/distributions.py
    # Takes about 30 mins with 50 samples, 1h with 80 samples on 2020 macbook
    _ptrain, _ptest = (100, [0.5, 1.0]), (2000, [0.5, dist])

    # System definition
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.array([[1, 1], [0, 1]]), \
        np.array([[0], [1]]), np.array([[1, 0]])

    # Support definition as a box [-0.2*noise, 1.0*noise]^d
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = noise * np.array(([1.0] * (_n + _p) * t_fir)
                                 + ([0.2] * (_n + _p) * t_fir))[:, None]

    # Feasible set definition 10*x1 <= feas_r, x2 <= feas_r
    fset = Polytope()
    fset.h = np.vstack((np.diag([10, 1, 0]), -np.diag([10, 1, 0])))
    fset.g = feas_r * np.ones((2 * (_n + _m), 1))

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    for _ps, (_xis, _t) in zip([_ptrain, _ptest],
                               zip([xis_train, xis_test], [t_fir, t_test])):
        (_ns, p) = _ps
        for n in ['log_normal', 'bimodal_gaussian', 'beta']:
            d = get_distribution(n, p)
            _xis[n] = np.hstack([np.vstack([d(_n * _t), d(_p * _t)])
                                 for i in range(_ns)]) * noise

    return t_test, t_fir, radius, p_level, sys, \
        fset, support, xis_train, xis_test


