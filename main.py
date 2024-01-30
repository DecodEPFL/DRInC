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
from experiments.double_integrator import double_integrator_experiment, savepath


def run(dist=1.0, verbose=False):
    experiment = double_integrator_experiment
    _w = np.diag([1, 4, 1])

    # Get experiment parameters, as a list to pass directly to get_controllers
    params = list(experiment(dist=dist))
    [xis_train, xis_test] = params[-2:]
    [t_test, t_fir, radius, p_level, sys, fset, support] = params[:-2]

    # Matrices over the whole test horizon
    w_full = np.kron(np.eye(t_test), _w)
    h_full = np.kron(np.eye(t_test), fset.h)
    g_full = np.kron(np.ones((t_test, 1)), fset.g)

    # Get controllers
    controllers = get_controllers(*params[1:-2], verbose=verbose)

    # Simulate all distributions
    c, v, x, u, y = dict(), dict(), dict(), dict(), dict()
    for d, xis in tqdm(xis_test.items()):
        # Simulate the closed loop maps
        c[d], v[d], x[d], u[d], y[d] = dict(), dict(), dict(), dict(), dict()
        for n, ctrl in controllers.items():
            if ctrl is None:  # Skip if controller not available
                continue

            # Simulate the closed loop map
            try:
                x[d][n], u[d][n], y[d][n], _ = \
                    simulate(ctrl(xis_train[d], _w), sys, xis)
            except AttributeError:  # Control design problem infeasible
                print(f"Warning: Controller {n} could not be synthesized"
                      f" for distribution {d}.")
                continue

            # Reformat x and u to split each time step and remove x0, u0
            t_split = t_test+1 if n in ["LQG", "DR-LQG"] else t_test+t_fir
            xs = np.split(x[d][n], t_split, axis=0)[-t_test:]
            us = np.split(u[d][n], t_split, axis=0)[-t_test:]
            ux = np.vstack([np.vstack((_x, _u)) for _x, _u in zip(xs, us)])

            # Compute the costs and the constraint violations
            c[d][n] = np.mean([_ux.T @ w_full @ _ux for _ux in ux.T])
            v[d][n] = np.mean([np.any(h_full @ _ux > g_full) for _ux in ux.T])

    # Save the results
    xis = {'train': dict(), 'test': dict()}
    # Reshape the samples to split the time steps
    # Makes the shape (samples, time steps, states/outputs)
    for k, _v in zip(['train', 'test'], [xis_train, xis_test]):
        # Short notation
        (_p, _n), _t = sys.c.shape, t_fir if k == 'train' else t_test

        # Deal with all distributions
        xis[k] = dict()
        for d, xi in _v.items():
            xis[k][d] = dict()
            _t2 = _t if d in ['sine', 'sawtooth', 'triangle', 'step'] else 1

            xis[k][d]['w'] = xi[:_n*_t, :].reshape((-1, _n*_t2, xi.shape[1]))
            xis[k][d]['v'] = xi[_n*_t:, :].reshape((-1, _p*_t2, xi.shape[1]))
            xis[k][d]['w'] = np.rollaxis(xis[k][d]['w'], -1)
            xis[k][d]['v'] = np.rollaxis(xis[k][d]['v'], -1)

    # Use .npz format
    np.savez(savepath, c=c, v=v, x=x, u=u, y=y, xi=xis)

    # Print the costs in a table with a given cell width
    print(f"radius for DRInC: {radius}, p_level: {p_level}")
    print(f"distribution parameter: {dist}")
    print_results(savepath, 20, labels=list(controllers.keys()))
    # print(plot_distributions(savepath, 20))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Run from arguments
    print('argument list', system.argv[1:])
    print('time ', time.strftime("%H:%M:%S", time.localtime()))
    for p in system.argv[1:]:
        # 0.5 = move 1/2 of right to left or left to right
        # => W = 0.8^2 / 4 ≈ 0.16
        run(float(p))
