"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main file for the project. This file will be used to run the project and
reproduce the experimental results from the paper entitled
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
# import cvxpy as cp
# from utils.distributions import get_distribution
# from utils.data_structures import LinearSystem, Polytope
from utils.simulate import simulate
from experiments.setup_controllers import get_controllers, controller_names
from experiments.double_integrator import double_integrator_experiment


def run():
    experiment = double_integrator_experiment
    # Get experiment parameters
    params = list(experiment())
    [xis_train, xis_test] = params[-2:]
    [t_fir, radius, p_level, sys, fset, support] = params[:-2]

    # Get controllers
    controllers = list(get_controllers(*params[:-2], verbose=False))
    controllers = controllers[0:3]
    controller_names = ["drinc", "rob", "emp"]

    # Simulate all distributions
    for d, xis in xis_test.items():
        # Simulate the closed loop maps
        x, u, y = dict(), dict(), dict()
        for n, phi in zip(controller_names,
                          [c(xis_train[d]) for c in controllers]):
            # Simulate the closed loop map
            x[n], u[n], y[n] = simulate(phi, sys, xis)

            # Compute the cost
            cost = np.mean(np.sum(np.multiply(x[n], x[n]), axis=0)
                           + np.sum(np.multiply(u[n], u[n]), axis=0))

            # Print the cost
            print(f"Cost of {n} for {d} distribution: ", cost)

            # Plot the results

    # Plot the results

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
