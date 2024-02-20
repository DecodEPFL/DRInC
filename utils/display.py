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
        element 'c': nested dictionary, wasserstein[distribution][controller].
        The distance is between training and testing distributions
        save_path can also be the dictionary itself.
    :param pitch: int, width of the cells in the printed table.
    :param labels: list of strings, labels for the rows of the table.
        (default: None, will use the keys of costs)
    :return: None
    """
    # Load data
    if isinstance(save_path, str):
        data = np.load(save_path, allow_pickle=True)
    elif isinstance(save_path, dict):
        data = save_path
    else:
        raise ValueError("save_path must be a string or a dictionary.")
    costs = data['c'].item()
    violate = data['v'].item()
    distances = data['w'].item()

    # handle optional labels
    if labels is None:
        labels = costs[list(costs.keys())[0]].keys()

    for d in costs.keys():
        print(d)

        # Print header and get number of rows
        s = "".ljust(pitch)
        for n in labels:
            s += str(n).ljust(pitch)
        print(s + "Wasserstein".ljust(pitch))

        # Print one line per parameter value
        for i in range(len(distances[d])):
            s = str(i).ljust(pitch)
            for n in labels:
                if n in costs[d].keys() and len(costs[d][n]) > i:
                    s += (str(np.round(costs[d][n][i], 2)) + ", "
                          + str(np.round(violate[d][n][i], 2))+"%").ljust(pitch)
                else:
                    s += "N/A".ljust(pitch)
            print(s + str(distances[d][i]).ljust(pitch))

        np.savetxt(save_path[:-4] + '_' + d + '.csv',
                   np.vstack([costs[d][n] for n in labels]
                             + [distances[d]]).T, delimiter=',')
        np.savetxt(save_path[:-4] + '_' + d + '_viol' + '.csv',
                   np.vstack([violate[d][n] for n in labels]
                             + [distances[d]]).T, delimiter=',')
    return


def plot_distributions(save_path, bins=20):
    """
    Plots the training noise distributions and provides the wasserstein distance
    between the training and testing distributions.

    :param save_path: path to npz file containing the results as a dictionary
        with an  element 'xi': a nested dictionary accessed as
        xi['train'/'test'][distribution]['w'/'v']. Each of these elements are
        arrays of shape (samples, time steps, states/outputs). For correlated
        distributions, time steps = 1.
        save_path can also be the dictionary itself.
    :param bins: number of bins for the histogram plot of the distribution.
    :return: dictionary with the Wasserstein distance between the training and
        testing distributions for each distribution.
    """
    # Load data
    if isinstance(save_path, str):
        data = np.load(save_path, allow_pickle=True)
    elif isinstance(save_path, dict):
        data = save_path
    else:
        raise ValueError("save_path must be a string or a dictionary.")
    xis = data['xi'].item()

    # Plot the training distributions
    for d, xi in xis['test'].items():
        plt.figure().suptitle(f"{d}")
        plt.hist(np.reshape(np.block([[[xi['w'], xi['v']]]]),
                            (-1, xi['w'].shape[2] + xi['v'].shape[2])),
                 bins=bins)
        plt.legend(['w' + str(i+1) for i in range(xi['w'].shape[2])]
                   + ['v' + str(i+1) for i in range(xi['v'].shape[2])])
        plt.show()

    return


if __name__ == '__main__':
    # Print the results of this experiment
    from experiments.given_distributions import savepath
    print_results('../' + savepath, 20)
    print(plot_distributions('../' + savepath))

