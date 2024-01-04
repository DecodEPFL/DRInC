"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main file for the project. This file will be used to run the project and
reproduce the experimental results from the paper entitled
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cvxpy as cp
from utils.distributions import get_distribution
from utils.data_structures import LinearSystem, Polytope
from utils.simulate import simulate
from drinc import synthesize_drinc


def experiment():
    # Use a breakpoint in the code line below to debug your script.
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiment()
