"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Experiment setup for a double integrator system. The support and feasible sets
are random polytope of the form {x | [I, -I] x <= g}, where g is a uniformly-
distributed random vector.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from tqdm import tqdm
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution
from utils.setup_controllers import get_controllers
from utils.simulate import simulate
from utils.display import print_results
from utils.wasserstein_approx import wasserstein, reshape_samples

savepath = "results/double_integrator.npz"


def double_integrator_experiment(radius=0.1, params=None, verbose=False):
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
    _m, _n, _p = 3, 3, 0
    # Time horizons, problem ill conditioned if t_fir < 5
    t_fir, t_test = 10, 50
    # Feasible set size, cvar probability level, and noise level
    feas_r, p_level, noise = 70.0, 0.1, 1.0  # 70
    # Number of samples. The list contains parameters for distributions.
    # Their values are explained in utils/distributions.py
    # testing distribution's second parameter is in params
    _ptrain, _ptest = (5, [0.5, 1.5]), (10, [0.5, None])

    # System definition
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.diag([1, 1, 0]), \
        np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]), None

    # Support definition as a box [-0.2*noise, 1.0*noise]^d
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = noise * np.array(([1.0] * (_n + _p) * t_fir)
                                 + ([0.2] * (_n + _p) * t_fir))[:, None]

    # Feasible set definition 10*x1 <= feas_r, x2 <= feas_r
    fset = Polytope()
    fset.h = np.vstack((np.diag([1, 1, 0, 0, 0, 0]), -np.diag([1, 1, 0, 0, 0, 0])))
    fset.g = 100*np.array(([96]*2 + [0]*4)*2)[:, None]# feas_r * np.ones((2 * (_n + _m), 1))

    # Make sure t_test is a multiple of t_fir
    t_test = int(t_test/t_fir) * t_fir

    # Generate training and testing samples
    xis_train, xis_test = dict(), dict()
    for n in ['bimodal_gaussian', 'beta']:
        d = get_distribution(n, _ptrain[1])
        xis_train[n] = np.hstack([d(_n * t_fir)
                                  for i in range(_ptrain[0])]) * noise
        xis_test[n] = []
        for param in params:
            d = get_distribution(n, [_ptest[1][0], param])
            xis_test[n] += \
                [np.hstack([d(_n * t_test)
                            for i in range(_ptest[0])]) * noise]

    return t_test, t_fir, radius, p_level, sys, \
        fset, support, xis_train, xis_test


if __name__ == "__main__":
    # Run the experiment
    _w = np.diag([1, 1, 1]*2)
    verbose = False

    # Get experiment parameters, as a list to pass directly to get_controllers
    params = list(double_integrator_experiment(params=[1.0]))
    [xis_train, xis_test] = params[-2:]
    [t_test, t_fir, radius, p_level, sys, fset, support] = params[:-2]
    (_p, _n) = sys.c.shape if sys.c is not None else (0, sys.a.shape[0])

    # Matrices over the whole test horizon
    w_f = np.kron(np.eye(t_test), _w)
    h_f = np.kron(np.eye(t_test), fset.h)
    g_f = np.kron(np.ones((t_test, 1)), fset.g)

    # Get controllers
    controllers = get_controllers(*params[1:-2], verbose=verbose)
    controllers = {"DRInC": controllers["DRInC"]}


    # Simulate all distributions
    c, v, w, phis = dict(), dict(), dict(), dict()
    for d, xis in xis_test.items():
        print(f"Simulating distribution {d}")

        # Simulate the closed loop maps
        c[d], v[d], w[d], phis[d] = dict(), dict(), [], dict()
        for n, ctrl in tqdm(controllers.items()):
            phis[d][n] = None
            c[d][n], v[d][n] = [], []
            if ctrl is None:  # Skip if controller not available
                continue

            # Synthesize controller
            try:
                phis[d][n] = ctrl(xis_train[d], _w)
                print(d, n, phis[d][n])
            except AttributeError:  # Control design problem infeasible
                print(f"Warning: Controller {n} could not be synthesized"
                      f" for distribution {d}.")
                continue

        for i, xi in tqdm(enumerate(xis)):
            for n, ctrl in controllers.items():
                # Simulate the closed loop map
                if phis[d][n] is None:  # Skip if controller not available
                    c[d][n] += [0.0]
                    v[d][n] += [0.0]
                    continue
                x, u, y, _ = simulate(phis[d][n], sys, xi)

                # Reformat x and u to split each time step and remove x0, u0
                t_split = t_test+1 if n in ["LQG", "DR-LQG"] else t_test+t_fir
                xs = np.split(x, t_split, axis=0)[-t_test:]
                us = np.split(u, t_split, axis=0)[-t_test:]
                ux = np.vstack([np.vstack((_x, _u)) for _x, _u in zip(xs, us)])

                # Compute the costs
                c[d][n] += [np.mean([_ux.T @ w_f @ _ux / t_test
                                     for _ux in ux.T])]
                # Compute the expected number of violations per time step
                v[d][n] += [np.mean([np.sum(h_f @ _ux > g_f, axis=0)
                                     for _ux in ux.T]) / t_test * 100]

            # Reshape samples to split time steps and merge w and v
            _xi = {'train': reshape_samples(xis_train[d], t_fir, _n, _p),
                   'test': reshape_samples(xi, t_test, _n, _p, t_fir)}
            # Compute the test/train Wasserstein distance squared
            w[d] += [wasserstein(_xi['train'], _xi['test'])*t_fir]

    # Save controllers separately in Matlab format for analysis
    from utils.simulate import clm_to_dyn_ctrl
    from scipy.io import savemat
    ctrls = {d: {str(n).replace('-', ''): clm_to_dyn_ctrl(phis[d][n], sys)
                 if phis[d][n] is not None else [] for n in phis[d].keys()}
             for d in phis.keys()}
    savemat(savepath.split('.')[0] + "_ctrl.mat", ctrls)

    # Use .npz format
    np.savez(savepath, phi=phis, xi=xis, c=c, v=v, w=w)

    # Print the costs in a table with a given cell width
    print(f"radius for DRInC: {radius}, p_level: {p_level}")
    print_results(savepath, 20, labels=list(controllers.keys()))
    # print(plot_distributions(savepath, 20))


