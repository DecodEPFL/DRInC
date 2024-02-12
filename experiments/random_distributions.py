"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from tqdm import tqdm
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution, get_random_empirical
from utils.wasserstein_approx import reshape_samples

savepath = "results/double_integrator.npz"


def double_integrator_experiment(radius=0.05, params=None, verbose=False):
    """
    This function runs the experiment for the double integrator system.
    It returns the system, the support, the feasible set, the training and
    testing samples.

    :param radius: Radius of the Wasserstein ball. Note that the Wasserstein
        TYPE 2 metric is used, so the radius is the square of the type 1 radius.
    :param params: float, second parameter of the testing distribution.
        (optional, default=[1.0]) See distributions.py
    :param verbose: bool, if True, prints the optimization verbose. (optional)
    :return: LinearSystem, Polytope, Polytope, dict, dict. The first three are
        the parameters t_fir, radius, and p_level. The next three are
        the system, the support and the feasible set. The last two are the
        training and testing samples, in the form of a dictionary with the
        distribution names for keys.
    """
    params = [1.0] if params is None else params

    # Useful to generate random Polytopes
    uni = get_distribution("uniform")

    # System dimensions
    _m, _n, _p = 1, 2, 1
    # Time horizons, problem ill conditioned if t_fir < 5
    t_fir, t_test = 10, 10
    # Feasible set size, cvar probability level, and noise level
    feas_r, p_level, noise = 70.0, 5e-2, 1.0  # 70
    # Number of samples. The list contains parameters for distributions.
    # Their values are explained in utils/distributions.py
    # testing distribution's second parameter is in params
    _ptrain, _ptest = (100, [0.5, 0.5]), (10000, None)

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

    # Make sure t_test is a multiple of t_fir
    t_test = int(t_test/t_fir) * t_fir

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    (_ns, p) = _ptrain
    for n in ['bimodal_gaussian', 'beta']:  # 'log_normal'
        d = get_distribution(n, p)
        xis_train[n] = np.hstack([np.vstack([d(_n * t_fir), d(_p * t_fir)])
                                  for i in range(_ns)]) * noise
        xis_test[n] = []
        from utils.wasserstein_approx import wasserstein
        for param in tqdm(params):
            # Make more samples for testing and reshape
            _xi = np.tile(xis_train[n], (1, int(_ptest[0]/_ns)))
            _xi = reshape_samples(xis_train[n], t_fir, _n, _p)
            # Get a random empirical at the right distance and reshape back
            _xi_test = get_random_empirical(_xi, param/t_fir)
            xis_test[n] += [np.block([
                [np.reshape(_xi_test[:_n, :], (_n*t_test, -1), 'F')],
                [np.reshape(_xi_test[_n:, :], (_p*t_test, -1), 'F')]])]

    return t_test, t_fir, radius, p_level, sys, \
        fset, support, xis_train, xis_test
