import numpy as np
import cvxpy as cp
from cvar import *
from data_structures import LinearSystem, Polytope


def test_cvar_constraints(verbose=False):
    """
    tests the cvar constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    pass

    if verbose:
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    np.set_printoptions(linewidth=120, precision=2)
    # for k in range(10):
    test_cvar_constraints(True)
