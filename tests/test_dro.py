import numpy as np
import cvxpy as cp
from utils.data_structures import Polytope
from dro import drinc_cost


def test_drinc_cost(verbose=False):
    """
    tests the drinc cost generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    np.random.seed(123)

    # Parameters
    _d = 2
    _q = np.eye(_d)
    radius = 2.0

    # Unit box support set
    support = Polytope()
    support.h = np.vstack((np.eye(_d),
                           -np.eye(_d)))
    support.g = np.ones((2 * _d, 1))

    # Center distribution with two samples
    # One at zero and one at d=1 from border of support
    xis_train = np.array([[1.0-0.707, 0], [1.0-0.707, 0]])

    # Worst case distribution and risk: move by 1 + sqrt(2)/sqrt(2) = 2
    xis_test = np.array([[1.0, 0.707], [1.0, 0.707]])
    wc_risk = np.mean([xis_test[:, i].T @ _q @ xis_test[:, i]
                       for i in range(_d)])

    # get the cost and optimize
    cost, cons = drinc_cost(support, radius)
    problem = cp.Problem(cp.Minimize(cost(_q, xis_train)),
                         cons(_q, xis_train)).solve(verbose=verbose)
    risk = problem

    if verbose:
        print(f"Theoretical risk: {wc_risk}")
        print(f"Practical risk: {risk}")

    # Compare risk to theoretical worst case
    np.testing.assert_allclose(risk, wc_risk, rtol=0, atol=1e-2)

    if verbose:
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    np.set_printoptions(linewidth=120, precision=2)
    # for k in range(10):
    test_drinc_cost(True)
