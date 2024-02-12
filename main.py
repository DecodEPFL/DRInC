"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main file for the project. This file will be used to run the project and
reproduce the experimental results from the paper entitled
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import sys as system
import time
import numpy as np
from tqdm import tqdm
from utils.display import print_results, plot_distributions
from utils.simulate import simulate
from utils.setup_controllers import get_controllers
from utils.wasserstein_approx import wasserstein, reshape_samples
from experiments.random_distributions import savepath
from experiments.random_distributions \
    import double_integrator_experiment as rd_exp
from experiments.given_distributions \
    import double_integrator_experiment as gv_exp


def run(experiment, dist=None, verbose=False, redo_design=True):
    experiment = rd_exp if experiment == "random" else gv_exp
    dist = [1.0] if dist is None else dist
    _w = np.diag([1, 4, 1])

    # Get experiment parameters, as a list to pass directly to get_controllers
    params = list(experiment(params=dist))
    [xis_train, xis_test] = params[-2:]
    [t_test, t_fir, radius, p_level, sys, fset, support] = params[:-2]
    (_p, _n) = sys.c.shape

    # Matrices over the whole test horizon
    w_f = np.kron(np.eye(t_test), _w)
    h_f = np.kron(np.eye(t_test), fset.h)
    g_f = np.kron(np.ones((t_test, 1)), fset.g)

    # Get controllers
    controllers = get_controllers(*params[1:-2], verbose=verbose)
    past_data = None if redo_design else np.load(savepath, allow_pickle=True)

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
                if redo_design:
                    phis[d][n] = ctrl(xis_train[d], _w)
                else:
                    phis[d][n] = (past_data['phi'].item())[d][n]
            except AttributeError:  # Control design problem infeasible
                print(f"Warning: Controller {n} could not be synthesized"
                      f" for distribution {d}.")
                continue

        for i, xi in tqdm(enumerate(xis)):
            for n, ctrl in controllers.items():
                # Simulate the closed loop map
                if phis[d][n] is None:  # Skip if controller not available
                    continue
                x, u, y, _ = simulate(phis[d][n], sys, xi)

                # Reformat x and u to split each time step and remove x0, u0
                t_split = t_test+1 if n in ["LQG", "DR-LQG"] else t_test+t_fir
                xs = np.split(x, t_split, axis=0)[-t_test:]
                us = np.split(u, t_split, axis=0)[-t_test:]
                ux = np.vstack([np.vstack((_x, _u)) for _x, _u in zip(xs, us)])

                # Compute the costs
                c[d][n] += [np.mean([_ux.T @ w_f @ _ux for _ux in ux.T])]
                # Compute the expected number of violations per time step
                v[d][n] += [np.mean([np.sum(h_f @ _ux > g_f, axis=0)
                                     for _ux in ux.T]) / t_split * 100]

            # Reshape samples to split time steps and merge w and v
            _xi = {'train': reshape_samples(xis_train[d], t_fir, _n, _p),
                   'test': reshape_samples(xi, t_test, _n, _p, t_fir)}
            # Compute the test/train Wasserstein distance squared
            w[d] += [wasserstein(_xi['train'], _xi['test'])*t_fir]

    # Use .npz format
    np.savez(savepath, phi=phis, xi=xis, c=c, v=v, w=w)

    # Print the costs in a table with a given cell width
    print(f"radius for DRInC: {radius}, p_level: {p_level}")
    print(f"distribution parameter: {dist}")
    print_results(savepath, 20, labels=list(controllers.keys()))
    # print(plot_distributions(savepath, 20))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Run from arguments
    #print('input: ', system.argv[1:])
    print('experiment: ', system.argv[1])
    exec("parameters = " + system.argv[2])
    print('parameters: ', parameters)
    print('time ', time.strftime("%H:%M:%S", time.localtime()))
    run(system.argv[1], parameters, redo_design=True)  # Put to false for test
