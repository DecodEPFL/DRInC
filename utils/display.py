"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Print and plot functions, shows the results of the experiments for all 
distributions and all controllers.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np


def print_results(costs, violations, pitch=20, labels=None):
    """
    Print the costs and violations in a table with a given cell width.

    :param costs: nested dictionary, costs[distribution][controller]
    :param violations: nested dictionary containing the fraction of trajectories
        with at least one constrain violation. Access with
        violations[distribution][controller]
    :param pitch: int, width of the cells in the printed table.
    :param labels: list of strings, labels for the rows of the table.
        (default: None, will use the keys of costs)
    :return: None
    """
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

