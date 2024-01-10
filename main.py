"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main file for the project. This file will be used to run the project and
reproduce the experimental results from the paper entitled
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from tqdm import tqdm
from utils.display import print_results
from utils.simulate import simulate
from utils.setup_controllers import get_controllers
from experiments.double_integrator import double_integrator_experiment


def run():
    experiment = double_integrator_experiment
    _w = np.eye(3)

    # Get experiment parameters, as a list to pass directly to get_controllers
    params = list(experiment())
    [xis_train, xis_test] = params[-2:]
    [t_test, t_fir, radius, p_level, sys, fset, support] = params[:-2]

    # Matrices over the whole test horizon
    w_full = np.kron(np.eye(t_test), _w)
    h_full = np.kron(np.eye(t_test), fset.h)
    g_full = np.kron(np.ones((t_test, 1)), fset.g)

    # Get controllers
    controllers = list(get_controllers(*params[1:-2], verbose=True))
    controllers = [controllers[0]]
    controller_names = ["DRInC"]

    # Simulate all distributions
    c, v, x, u, y = dict(), dict(), dict(), dict(), dict()
    for d, xis in tqdm(xis_test.items()):
        # Simulate the closed loop maps
        c[d], v[d], x[d], u[d], y[d] = dict(), dict(), dict(), dict(), dict()
        for n, phi in zip(controller_names,
                          [c(xis_train[d], _w) for c in controllers]):
            # Simulate the closed loop map, key access because I'm lazy
            x[d][n], u[d][n], y[d][n] = simulate(phi, sys, xis)

            # Reformat x and u to split each time step
            xs = np.split(x[d][n], t_test, axis=0)
            us = np.split(u[d][n], t_test, axis=0)
            ux = np.vstack([np.vstack((_x, _u)) for _x, _u in zip(xs, us)])

            # Compute the costs and the constraint violations
            c[d][n] = np.mean([_ux.T @ w_full @ _ux for _ux in ux.T])
            v[d][n] = np.mean([np.any(h_full @ _ux > g_full) for _ux in ux.T])

    # Print the costs in a table with a given cell width
    print_results(c, v, 20)

    # Plot the results
    import matplotlib.pyplot as plt
    for d in x.keys():
        plt.figure().suptitle(d)
        print(np.min(fset.g), np.max((x[d][controller_names[0]])))
        for n in [controller_names[0]]:#[d].keys():
            plt.plot(x[d][n][::2, :], x[d][n][1::2, :], label=n)
        plt.legend()
        plt.show()


    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
