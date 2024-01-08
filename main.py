"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main file for the project. This file will be used to run the project and
reproduce the experimental results from the paper entitled
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
# import cvxpy as cp
from tqdm import tqdm
# from utils.distributions import get_distribution
# from utils.data_structures import LinearSystem, Polytope
from utils.simulate import simulate
from experiments.setup_controllers import get_controllers, controller_names
from experiments.double_integrator import double_integrator_experiment


def run():
    experiment = double_integrator_experiment
    _w = np.eye(3)

    # Get experiment parameters
    params = list(experiment())
    [xis_train, xis_test] = params[-2:]
    [t_test, t_fir, radius, p_level, sys, fset, support] = params[:-2]

    # Full weight matrix
    w_full = np.kron(_w, np.eye(t_test))

    # Get controllers
    controllers = list(get_controllers(*params[1:-2]))

    # Simulate all distributions
    c, x, u, y = dict(), dict(), dict(), dict()
    for d, xis in tqdm(xis_test.items()):
        # Simulate the closed loop maps
        c[d], x[d], u[d], y[d] = dict(), dict(), dict(), dict()
        for n, phi in zip(controller_names,
                          [c(xis_train[d], _w) for c in controllers]):
            # Simulate the closed loop map, key access because I'm lazy
            x[d][n], u[d][n], y[d][n] = simulate(phi, sys, xis)

            # Compute the cost
            xs = np.split(x[d][n], t_test, axis=0)
            us = np.split(u[d][n], t_test, axis=0)
            ux = np.vstack([np.vstack((_x, _u)) for _x, _u in zip(xs, us)])
            c[d][n] = np.mean([_ux.T @ w_full @ _ux for _ux in ux.T])

    # Print the costs in a table with a given cell width
    pitch = 16

    # Print header
    s = "".ljust(pitch)
    for d in xis_test.keys():
        s += str(d).ljust(pitch)
    print(s)

    # Print one line per controller
    for n in controller_names:
        s = str(n).ljust(pitch)
        for d in xis_test.keys():
            s += str(np.round(c[d][n], 2)).ljust(pitch)
        print(s)

    # Plot the results

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
