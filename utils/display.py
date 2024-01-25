"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Print and plot functions, shows the results of the experiments for all 
distributions and all controllers.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt


def print_results(savepath, pitch=20, labels=None):
    """
    Print the costs and violations in a table with a given cell width.

    :param savepath: path to npz file containing the results as a dictionary:
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
    data = np.load(savepath, allow_pickle=True)
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


def plot_results(savepath):
    # Load data
    data = np.load(savepath, allow_pickle=True)
    xis = data['xi'].item()

    for d, xi in xis['train'].items():
        plt.figure().suptitle(d)
        plt.hist(np.reshape(xi['w'], (-1, xi['w'].shape[2])), bins=20)
        plt.show()
    return


if __name__ == '__main__':
    # Print the results of this experiment
    from experiments.double_integrator import savepath
    print_results('../' + savepath, 20)
    plot_results('../' + savepath)

