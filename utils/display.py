"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Print and plot functions, shows the results of the experiments for all 
distributions and all controllers.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt
from utils.wasserstein_approx import wasserstein


def print_results(save_path, pitch=20, labels=None):
    """
    Print the costs and violations in a table with a given cell width.

    :param save_path: path to npz file containing the results as a dictionary:
        element 'c': nested dictionary, costs[distribution][controller]
        element 'v': nested dictionary containing the fraction of trajectories
            with at least one constrain violation. Access with
            violations[distribution][controller]
    :param pitch: int, width of the cells in the printed table.
    :param labels: list of strings, labels for the rows of the table.
        (default: None, will use the keys of costs)
    :return: None
    """
    # Load data
    data = np.load(save_path, allow_pickle=True)
    costs = data['c'].item()
    violations = data['v'].item()

    # handle optional labels
    if labels is None:
        labels = costs[list(costs.keys())[0]].keys()

    # Print header
    s = "".ljust(pitch)
    for d in costs.keys():
        s += str(d).ljust(pitch)
    print(s)

    # Print one line per controller
    for n in labels:
        s = str(n).ljust(pitch)
        for d in costs.keys():
            if n in costs[d].keys():
                s += (str(np.round(costs[d][n], 2)) + ", "
                      + str(np.round(violations[d][n]*100, 2))+"%").ljust(pitch)
            else:
                s += "N/A".ljust(pitch)
        print(s)
    return


def plot_distributions(save_path, bins=20):
    """
    Plots the training noise distributions and provides the wasserstein distance
    between the training and testing distributions.

    :param save_path: path to npz file containing the results as a dictionary
        with an  element 'xi': a nested dictionary accessed as
        xi['train'/'test'][distribution]['w'/'v']. Each of these elements are
        arrays of shape (samples, time steps, states/outputs).
        For correlated distributions, time steps = 1.
    :param bins: number of bins for the histogram plot of the distribution.
    :return: dictionary with the Wasserstein distance between the training and
        testing distributions for each distribution.
    """
    # Load data
    data = np.load(save_path, allow_pickle=True)
    xis = data['xi'].item()

    wds = dict()
    # compute the Wasserstein distance between the training and testing
    for d, (xi1, xi2) in zip(xis['train'].keys(), zip(xis['train'].values(),
                                                      xis['test'].values())):
        # Number of time steps and states
        n_t, n_s = xi1['w'].shape[1], xi1['w'].shape[2] + xi1['v'].shape[2]
        # Reshape the distributions to be 2D arrays for wasserstein function
        _xi1 = np.reshape(np.block([[[xi1['w'], xi1['v']]]]), (-1, n_s)).T
        _xi2 = np.reshape(np.block([[[xi2['w'][:, :n_t, :],
                                      xi2['v'][:, :n_t, :]]]]), (-1, n_s)).T
        # Wasserstein distance
        wds[d] = wasserstein(_xi1, _xi2)

    # Plot the training distributions
    for d, xi in xis['train'].items():
        plt.figure().suptitle(f"{d}, Wasserstein distance: "
                              f"{np.round(wds[d], 2)}")
        plt.hist(np.reshape(np.block([[[xi['w'], xi['v']]]]),
                            (-1, xi['w'].shape[2] + xi['v'].shape[2])),
                 bins=bins)
        plt.legend(['w' + str(i+1) for i in range(xi['w'].shape[2])]
                   + ['v' + str(i+1) for i in range(xi['v'].shape[2])])
        plt.show()

    return wds


if __name__ == '__main__':
    # Print the results of this experiment
    from experiments.double_integrator import savepath
    print_results('../' + savepath, 20)
    print(plot_distributions('../' + savepath))

